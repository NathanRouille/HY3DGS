"""ShapePCAE + UNITE joint tokenization and flow matching."""

from __future__ import annotations

import copy
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .shape_pc_ae import ShapePCAE
from ...unite.adaln_encoder import AdaLNGenerativeEncoder
from ...unite.flow_loss import (
    OFFPATH_MODES,
    compute_flow_loss,
    make_noise_offpath_state,
    make_onestep_offpath_state,
    noising_latents,
)
from ...unite.transport import Sampler, Transport


class ShapePCUnite(ShapePCAE):
    """ShapePCAE with shared AdaLN GE for tokenizer + flow denoising."""

    def __init__(
        self,
        *,
        modulation_recon_timestep_max: float = 0.01,
        noising_t_start: float = 0.9,
        flow_steps_per_recon: int = 8,
        gen_loss_weight: float = 1.0,
        use_lognorm: bool = False,
        lognorm_mu: float = 1.0,
        lognorm_sigma: float = 1.6,
        t0_force_prob: float = 0.0,
        timestep_shift_alpha: float = 0.0,
        use_rope: bool = False,
        weak_context_dropout: float = 0.1,
        flow_loss_type: str = "velocity",
        num_ge_layers: int = 8,
        qk_norm: bool = True,
        sample_renorm_output: bool = False,
        tokenizer_use_weak_context: bool = False,
        adaln_camera_cond: bool = False,
        use_surflo_global_cam: bool = False,
        offpath_mode: str = "none",
        offpath_weight: float = 1.0,
        offpath_noise_std: float = 0.1,
        offpath_step_size: float = 0.05,
        **kwargs,
    ):
        self._unite_num_ge_layers = num_ge_layers
        self._unite_qk_norm = qk_norm
        super().__init__(num_ge_layers=num_ge_layers, qk_norm=qk_norm, **kwargs)
        self.sample_renorm_output = bool(sample_renorm_output)
        self.modulation_recon_timestep_max = float(modulation_recon_timestep_max)
        self.noising_t_start = float(noising_t_start)
        self.flow_steps_per_recon = int(flow_steps_per_recon)
        self.gen_loss_weight = float(gen_loss_weight)
        self.timestep_shift_alpha = float(timestep_shift_alpha)
        self.weak_context_dropout = float(weak_context_dropout)
        self.flow_loss_type = str(flow_loss_type)
        self.tokenizer_use_weak_context = bool(tokenizer_use_weak_context)
        self.adaln_camera_cond = bool(adaln_camera_cond)
        self.use_surflo_global_cam = bool(use_surflo_global_cam)
        if self.adaln_camera_cond and self.use_surflo_global_cam:
            raise ValueError(
                "Pass only one of adaln_camera_cond (legacy) and "
                "use_surflo_global_cam (Surflo global → AdaLN ablation)."
            )
        # Attached via attach_surflo_global_cam() when the CLI flag is set.
        self.surflo_global_cam: Optional[nn.Module] = None
        mode = str(offpath_mode).lower()
        if mode not in OFFPATH_MODES:
            raise ValueError(f"offpath_mode must be one of {OFFPATH_MODES}, got {offpath_mode!r}")
        self.offpath_mode = mode
        self.offpath_weight = float(offpath_weight)
        self.offpath_noise_std = float(offpath_noise_std)
        self.offpath_step_size = float(offpath_step_size)
        # When True, tokenizer decode applies latent noising (UNITE representation
        # phase). Trainers can set False for clean overfits.
        self.representation_noising = True
        # Optional frozen snapshot of the GE used by the tokenizer pass, so that
        # flow-only training cannot drift the reconstruction pathway through the
        # shared weights.
        self.tokenizer_ge: Optional[nn.Module] = None

        ge_ctx = self.num_registers + self.num_latents
        # Replace Pre-LN GE with AdaLN GE (same depth/width).
        self.ge = AdaLNGenerativeEncoder(
            in_channels=self.embed_dim,
            hidden_size=self.width,
            depth=self._unite_num_ge_layers,
            num_heads=self.width // 64,
            num_output_tokens=self.num_registers,
            max_tokens=ge_ctx + 4096,
            use_qknorm=self._unite_qk_norm,
            use_rope=use_rope,
        )

        # Denoiser PE (width-d, added after ge.up_sample). Cloned from the tokenizer
        # PE; unused (and frozen) until --freeze_recon snapshots the tokenizer GE,
        # after which tokenizer keeps ``register_pos_embed`` and the flow trains this.
        self.denoiser_pos_embed = nn.Parameter(
            self.register_pos_embed.detach().clone(), requires_grad=False
        )

        self.transport = Transport(
            use_lognorm=bool(use_lognorm),
            lognorm_mu=float(lognorm_mu),
            lognorm_sigma=float(lognorm_sigma),
            t0_force_prob=float(t0_force_prob),
        )
        self.flow_sampler = Sampler(self.transport)
        self.null_weak_context = nn.Parameter(
            torch.zeros(1, 1, self.width), requires_grad=True
        )
        nn.init.normal_(self.null_weak_context, std=0.02)

        # Camera → AdaLN (DiT-style sum: c = t_emb + cam). Ablation flag; off by
        # default keeps camera as the first sequence token + timestep-only AdaLN.
        # Tokenizer always uses the learned null cam (encode ≠ denoise signal).
        if self.adaln_camera_cond:
            self.cam_cond_proj = nn.Linear(self.width, self.width, bias=True)
            # Zero-init so early training matches timestep-only AdaLN.
            nn.init.zeros_(self.cam_cond_proj.weight)
            nn.init.zeros_(self.cam_cond_proj.bias)
            self.null_cam_cond = nn.Parameter(
                torch.zeros(1, self.width), requires_grad=True
            )
            nn.init.normal_(self.null_cam_cond, std=0.02)
        else:
            self.cam_cond_proj = None
            self.null_cam_cond = None

    def _ge_pos_embed(self, *, use_tokenizer: bool = False) -> torch.Tensor:
        # Joint training (no tokenizer_ge snapshot): one shared PE — matches pre-split
        # behaviour. After --freeze_recon, tokenizer and denoiser use separate PEs.
        if self.tokenizer_ge is None or use_tokenizer:
            return self.register_pos_embed
        return self.denoiser_pos_embed

    def freeze_tokenizer_ge(self, *, replace: bool = False) -> None:
        """Snapshot the GE so the tokenizer pass stops following flow updates.

        ``--freeze_recon`` only froze the encoder/decoder, but the GE is shared,
        so flow training kept rewriting the tokenizer (Exp 3: recon CD drifted
        0.0012 -> 0.03-0.16 while flow decreased).

        If a snapshot already exists (e.g. loaded from a prior ``--freeze_recon``
        ckpt), keep it unless ``replace=True``. Re-snapshotting from the live
        ``ge`` after flow FT would destroy the tokenizer (exp17 bug).
        """
        if self.tokenizer_ge is not None and not replace:
            self.tokenizer_ge.eval()
            for p in self.tokenizer_ge.parameters():
                p.requires_grad = False
            return
        self.tokenizer_ge = copy.deepcopy(self.ge)
        self.tokenizer_ge.eval()
        for p in self.tokenizer_ge.parameters():
            p.requires_grad = False

    def ensure_tokenizer_ge_eval(self) -> None:
        """Keep the frozen tokenizer GE in eval (no attn dropout) after model.train()."""
        if self.tokenizer_ge is not None:
            self.tokenizer_ge.eval()

    def sync_denoiser_pos_embed_from_tokenizer(self) -> None:
        """Copy tokenizer PE → denoiser PE (resume from pre-split checkpoints / FT start)."""
        with torch.no_grad():
            self.denoiser_pos_embed.copy_(self.register_pos_embed)

    def _run_ge(
        self,
        register_slots: torch.Tensor,
        t: torch.Tensor,
        *,
        context_embed: Optional[torch.Tensor] = None,
        context_keep: Optional[torch.Tensor] = None,
        cond_embed: Optional[torch.Tensor] = None,
        use_tokenizer_ge: bool = False,
    ) -> torch.Tensor:
        ge = self.tokenizer_ge if (use_tokenizer_ge and self.tokenizer_ge is not None) else self.ge
        pos = self._ge_pos_embed(use_tokenizer=use_tokenizer_ge).expand(
            register_slots.shape[0], -1, -1
        )
        return ge(
            register_slots,
            t,
            pos_embed=pos,
            context_embed=context_embed,
            context_keep=context_keep,
            cond_embed=cond_embed,
        )

    def null_context(self, batch_size: int, num_tokens: int, *, dtype=None, device=None) -> torch.Tensor:
        """Learned null weak context, expanded to ``num_tokens``."""
        null = self.null_weak_context
        if device is not None:
            null = null.to(device)
        if dtype is not None:
            null = null.to(dtype)
        return null.expand(batch_size, num_tokens, -1)

    def attach_surflo_global_cam(self, branch: nn.Module) -> None:
        """Register a Surflo global-camera branch (projector+compressor+adapter)."""
        self.surflo_global_cam = branch
        self.use_surflo_global_cam = True
        self.add_module("surflo_global_cam", branch)

    def _resolve_cam_cond(
        self,
        cam_token: Optional[torch.Tensor],
        batch_size: int,
        *,
        device=None,
        dtype=None,
        use_null: bool = False,
    ) -> Optional[torch.Tensor]:
        """Map camera embed → AdaLN cond, or return learned null.

        Returns None when ``adaln_camera_cond`` is off (timestep-only AdaLN).
        """
        if not self.adaln_camera_cond:
            return None
        device = device or (
            cam_token.device if cam_token is not None else self.null_cam_cond.device
        )
        dtype = dtype or (
            cam_token.dtype if cam_token is not None else self.null_cam_cond.dtype
        )
        if use_null or cam_token is None:
            return self.null_cam_cond.to(device=device, dtype=dtype).expand(batch_size, -1)
        if cam_token.dim() == 3:
            cam_token = cam_token.squeeze(1)
        return self.cam_cond_proj(cam_token.to(device=device, dtype=dtype))

    def _resolve_surflo_global_cond(
        self,
        raw_camera_tokens: Optional[torch.Tensor],
        batch_size: int,
        *,
        device=None,
        dtype=None,
        use_null: bool = False,
    ) -> Optional[torch.Tensor]:
        """Surflo global-camera residual for AdaLN (zeros on null / CFG)."""
        if not self.use_surflo_global_cam or self.surflo_global_cam is None:
            return None
        device = device or (
            raw_camera_tokens.device
            if raw_camera_tokens is not None
            else next(self.surflo_global_cam.parameters()).device
        )
        dtype = dtype or (
            raw_camera_tokens.dtype
            if raw_camera_tokens is not None
            else next(self.surflo_global_cam.parameters()).dtype
        )
        if use_null or raw_camera_tokens is None:
            return torch.zeros(batch_size, self.width, device=device, dtype=dtype)
        return self.surflo_global_cam(raw_camera_tokens, use_null=False).to(
            device=device, dtype=dtype
        )

    def _combine_adaln_cond(
        self,
        *,
        batch_size: int,
        cam_cond: Optional[torch.Tensor] = None,
        raw_camera_tokens: Optional[torch.Tensor] = None,
        device=None,
        dtype=None,
        use_null: bool = False,
    ) -> Optional[torch.Tensor]:
        """Legacy AdaLN cam and/or Surflo global residual (sum if both)."""
        cond = self._resolve_cam_cond(
            cam_cond,
            batch_size,
            device=device,
            dtype=dtype,
            use_null=use_null or cam_cond is None,
        )
        g = self._resolve_surflo_global_cond(
            raw_camera_tokens,
            batch_size,
            device=device,
            dtype=dtype,
            use_null=use_null,
        )
        if cond is None:
            return g
        if g is None:
            return cond
        return cond + g

    def encode_tokenizer(
        self,
        surface: torch.FloatTensor,
        *,
        register_noise: Optional[torch.Tensor] = None,
        weak_context: Optional[torch.Tensor] = None,
        weak_context_keep: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Strong-context tokenizer pass → normalized latents."""
        pc, feats = self.encoder_inputs(surface)
        locals_tok, pc_infos = self.encoder(pc, feats)
        fps_xyz = pc_infos[0]

        b = surface.shape[0]
        register_noise = self.resolve_register_noise(
            b,
            device=surface.device,
            dtype=surface.dtype,
            register_noise=register_noise,
        )

        # Tokenizer AdaLN time: exact 0 when max<=0 (matches pure-noise start of
        # the flow ODE); otherwise Uniform(0, max) as in UNITE.
        if self.modulation_recon_timestep_max <= 0:
            t = torch.zeros(b, device=surface.device, dtype=surface.dtype)
        else:
            t = (
                torch.rand(b, device=surface.device, dtype=surface.dtype)
                * self.modulation_recon_timestep_max
            )

        # Optional hybrid tokenizer context:
        #   [Hunyuan locals_tok | VGGT weak-context tokens]
        # This keeps the original registers + GT cross-attn tokens intact and only
        # appends VGGT tokens when explicitly enabled.
        tok_ctx = locals_tok
        tok_keep = None
        if self.tokenizer_use_weak_context and weak_context is not None:
            tok_ctx = torch.cat(
                [locals_tok, weak_context.to(device=locals_tok.device, dtype=locals_tok.dtype)],
                dim=1,
            )
            if weak_context_keep is not None:
                b, n_local, _ = locals_tok.shape
                local_keep = torch.ones(b, n_local, dtype=torch.bool, device=locals_tok.device)
                tok_keep = torch.cat(
                    [local_keep, weak_context_keep.to(device=locals_tok.device, dtype=torch.bool)],
                    dim=1,
                )

        # Tokenizer never sees the real camera in AdaLN — learned null doubles as
        # an encode vs denoise mode signal when adaln_camera_cond is on.
        cond = self._resolve_cam_cond(
            None, b, device=surface.device, dtype=surface.dtype, use_null=True
        )

        h = self._run_ge(
            register_noise,
            t,
            context_embed=tok_ctx,
            context_keep=tok_keep,
            cond_embed=cond,
            use_tokenizer_ge=True,
        )
        z = self.latent_norm(h)
        return z, fps_xyz, locals_tok

    def encode(
        self,
        surface,
        *,
        register_noise=None,
        weak_context: Optional[torch.Tensor] = None,
        weak_context_keep: Optional[torch.Tensor] = None,
    ):
        z, fps_xyz, _ = self.encode_tokenizer(
            surface,
            register_noise=register_noise,
            weak_context=weak_context,
            weak_context_keep=weak_context_keep,
        )
        return z, fps_xyz

    def decode(
        self,
        latents: torch.FloatTensor,
        *,
        representation_phase: bool = False,
        return_features: bool = False,
    ):
        z = latents
        if representation_phase and self.training:
            z = noising_latents(
                self.transport,
                z,
                noising_t_start=self.noising_t_start,
            )
        return super().decode(z, return_features=return_features)

    def forward_denoising(
        self,
        z_norm: torch.Tensor,
        weak_context: Optional[torch.Tensor] = None,
        *,
        context_keep: Optional[torch.Tensor] = None,
        cam_cond: Optional[torch.Tensor] = None,
        raw_camera_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        z_for_flow = z_norm.detach()
        flow_loss = None
        t_list = [
            self.transport.sample(
                z_for_flow, timestep_shift=self.timestep_shift_alpha
            )[0]
            for _ in range(self.flow_steps_per_recon)
        ]

        ctx = weak_context
        keep = context_keep
        cond = self._combine_adaln_cond(
            batch_size=z_norm.shape[0],
            cam_cond=cam_cond,
            raw_camera_tokens=raw_camera_tokens,
            device=z_norm.device,
            dtype=z_norm.dtype,
            use_null=False,
        )
        # If Surflo/legacy cam is enabled but inputs missing, force null residual.
        if cond is None and (self.adaln_camera_cond or self.use_surflo_global_cam):
            cond = self._combine_adaln_cond(
                batch_size=z_norm.shape[0],
                cam_cond=None,
                raw_camera_tokens=None,
                device=z_norm.device,
                dtype=z_norm.dtype,
                use_null=True,
            )

        if ctx is not None and self.training and self.weak_context_dropout > 0:
            drop = torch.rand(z_norm.shape[0], device=z_norm.device) < self.weak_context_dropout
            if drop.any():
                null = self.null_context(z_norm.shape[0], ctx.shape[1], dtype=ctx.dtype)
                ctx = torch.where(drop.view(-1, 1, 1), null, ctx)
                if keep is not None:
                    # Null context is fully valid (single learned token expanded).
                    keep = torch.where(
                        drop.view(-1, 1),
                        torch.ones_like(keep),
                        keep,
                    )
                if cond is not None:
                    null_c = self._combine_adaln_cond(
                        batch_size=z_norm.shape[0],
                        cam_cond=None,
                        raw_camera_tokens=None,
                        device=z_norm.device,
                        dtype=z_norm.dtype,
                        use_null=True,
                    )
                    cond = torch.where(drop.view(-1, 1), null_c, cond)

        velocity_mse = 0.0
        x_start_mse = 0.0
        offpath_loss_acc = None
        for t in t_list:
            model_kwargs = dict(
                context_embed=ctx, context_keep=keep, cond_embed=cond
            )
            flow_dict = self.transport.training_losses(
                self._flow_model_fn,
                z_for_flow,
                t=t,
                model_kwargs=model_kwargs,
                timestep_shift=self.timestep_shift_alpha,
            )
            parts = compute_flow_loss(
                flow_dict,
                train_eps=self.transport.train_eps,
                latent_norm=self.latent_norm,
                loss_type=self.flow_loss_type,
            )
            chunk_loss = parts["loss"]
            flow_loss = chunk_loss if flow_loss is None else flow_loss + chunk_loss
            velocity_mse = velocity_mse + parts["velocity_mse"]
            x_start_mse = x_start_mse + parts["x_start_mse"]

            if self.training and self.offpath_mode != "none" and self.offpath_weight != 0.0:
                off_parts = self._offpath_flow_loss(flow_dict, model_kwargs)
                off_chunk = off_parts["loss"]
                offpath_loss_acc = (
                    off_chunk if offpath_loss_acc is None else offpath_loss_acc + off_chunk
                )
                flow_loss = flow_loss + self.offpath_weight * off_chunk

        n = max(self.flow_steps_per_recon, 1)
        out = {
            "flow/flow_loss": flow_loss / n,
            "flow/velocity_mse": velocity_mse / n,
            "flow/x_start_mse": x_start_mse / n,
            "flow/t_mean": torch.stack(t_list).mean(),
        }
        if offpath_loss_acc is not None:
            out["flow/offpath_loss"] = offpath_loss_acc / n
        return out

    def _offpath_flow_loss(
        self,
        flow_dict: Dict[str, torch.Tensor],
        model_kwargs: Dict,
    ) -> Dict[str, torch.Tensor]:
        """Auxiliary exposure-bias loss (noise-on-path or 1-step correction)."""
        train_eps = self.transport.train_eps
        x1 = flow_dict["x1"]
        xt = flow_dict["xt"]
        t = flow_dict["sampled_t"]

        if self.offpath_mode == "noise":
            xt_off = make_noise_offpath_state(xt, noise_std=self.offpath_noise_std)
            t_off = t
        elif self.offpath_mode == "onestep":
            with torch.no_grad():
                x_pred = flow_dict["model_output"]
                if self.latent_norm is not None:
                    x_pred = self.latent_norm(x_pred)
                xt_off, t_off = make_onestep_offpath_state(
                    xt.detach(),
                    t,
                    x_pred.detach(),
                    step_size=self.offpath_step_size,
                    train_eps=train_eps,
                )
        else:
            raise ValueError(f"Unexpected offpath_mode={self.offpath_mode!r}")

        model_output = self._flow_model_fn(xt_off, t_off, **model_kwargs)
        return compute_flow_loss(
            {
                "x1": x1,
                "xt": xt_off,
                "sampled_t": t_off,
                "model_output": model_output,
            },
            train_eps=train_eps,
            latent_norm=self.latent_norm,
            # Correction target is always the straight velocity / x1 pull;
            # keep the same loss_type as the main path for consistency.
            loss_type=self.flow_loss_type,
        )

    def _flow_model_fn(self, xt, t, context_embed=None, context_keep=None, cond_embed=None):
        return self._run_ge(
            xt,
            t,
            context_embed=context_embed,
            context_keep=context_keep,
            cond_embed=cond_embed,
        )

    def forward_tokenizer(
        self,
        surface: torch.Tensor,
        *,
        representation_phase: Optional[bool] = None,
        weak_context: Optional[torch.Tensor] = None,
        weak_context_keep: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Encode + decode surface.

        ``representation_phase`` controls latent noising before decode (UNITE
        representation training). Default: use ``self.representation_noising``
        (True unless the trainer disables it for clean overfits).
        """
        if representation_phase is None:
            representation_phase = bool(getattr(self, "representation_noising", True))
        z, fps_xyz, _locals = self.encode_tokenizer(
            surface,
            weak_context=weak_context,
            weak_context_keep=weak_context_keep,
        )
        xyz, rgb, centers = self.decode(z, representation_phase=representation_phase)
        return z, fps_xyz, xyz, rgb, centers

    def forward(
        self,
        surface: torch.Tensor,
        weak_context: Optional[torch.Tensor] = None,
        context_keep: Optional[torch.Tensor] = None,
        cam_cond: Optional[torch.Tensor] = None,
        raw_camera_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        z, fps_xyz, xyz, rgb, centers = self.forward_tokenizer(
            surface,
            weak_context=weak_context,
            weak_context_keep=context_keep,
        )
        out = {
            "z": z,
            "fps_xyz": fps_xyz,
            "xyz": xyz,
            "rgb": rgb,
            "centers": centers,
        }
        if weak_context is not None or self.training:
            out.update(
                self.forward_denoising(
                    z,
                    weak_context,
                    context_keep=context_keep,
                    cam_cond=cam_cond,
                    raw_camera_tokens=raw_camera_tokens,
                )
            )
        return out

    @torch.no_grad()
    def sample_latents(
        self,
        weak_context: Optional[torch.Tensor],
        *,
        batch_size: int = 1,
        num_steps: int = 50,
        guidance_scale: float = 1.0,
        num_null_tokens: int = 1,
        noise: Optional[torch.Tensor] = None,
        z_init: Optional[torch.Tensor] = None,
        t_start: float = 0.0,
        context_keep: Optional[torch.Tensor] = None,
        cam_cond: Optional[torch.Tensor] = None,
        raw_camera_tokens: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        renorm_output: Optional[bool] = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """Image-only generation: ODE sample register latents from noise.

        Args:
            guidance_scale: classifier-free guidance strength; 1.0 disables it.
            cam_cond: optional [B, width] for legacy ``adaln_camera_cond``.
            raw_camera_tokens: optional [B, S, 2048] for ``use_surflo_global_cam``.
            z_init / t_start: partial-noise oracle start.
            renorm_output: optional second latent_norm on ODE endpoint.
            return_trajectory: if True, return ``(z, traj, t_grid)``.
        """
        if renorm_output is None:
            renorm_output = self.sample_renorm_output
        device = device or self.register_pos_embed.device
        dtype = dtype or self.register_pos_embed.dtype
        if z_init is None:
            if noise is None:
                noise = torch.randn(
                    batch_size, self.num_registers, self.embed_dim, device=device, dtype=dtype
                )
            x_init = noise
        else:
            eps = torch.randn_like(z_init) if noise is None else noise
            x_init = t_start * z_init + (1.0 - t_start) * eps

        sample_fn = self.flow_sampler.sample_ode(
            sampling_method="euler",
            num_steps=num_steps,
            timestep_shift=self.timestep_shift_alpha,
            t0=t_start,
        )
        train_eps = self.transport.train_eps
        n_tokens = weak_context.shape[1] if weak_context is not None else num_null_tokens
        use_cfg = weak_context is not None and abs(guidance_scale - 1.0) > 1e-6

        cond_c = self._combine_adaln_cond(
            batch_size=batch_size,
            cam_cond=cam_cond,
            raw_camera_tokens=raw_camera_tokens,
            device=device,
            dtype=dtype,
            use_null=(
                (self.adaln_camera_cond and cam_cond is None)
                or (self.use_surflo_global_cam and raw_camera_tokens is None)
            ),
        )
        cond_null = self._combine_adaln_cond(
            batch_size=batch_size,
            cam_cond=None,
            raw_camera_tokens=None,
            device=device,
            dtype=dtype,
            use_null=True,
        )

        def predict(x_t, t, ctx, keep=None, cond=None):
            x_pred = self.latent_norm(
                self._run_ge(
                    x_t, t, context_embed=ctx, context_keep=keep, cond_embed=cond
                )
            )
            denom = (1.0 - t.view(-1, 1, 1)).clamp_min(train_eps)
            return (x_pred - x_t) / denom

        def model_fn(x_t, t, **kwargs):
            ctx = kwargs.get("context_embed")
            keep = kwargs.get("context_keep")
            cond = kwargs.get("cond_embed", cond_c)
            if ctx is None:
                ctx = self.null_context(
                    x_t.shape[0], n_tokens, dtype=x_t.dtype, device=x_t.device
                )
                return predict(x_t, t, ctx, None, cond_null)
            v_cond = predict(x_t, t, ctx, keep, cond)
            if not use_cfg:
                return v_cond
            null = self.null_context(
                x_t.shape[0], n_tokens, dtype=ctx.dtype, device=ctx.device
            )
            v_uncond = predict(x_t, t, null, None, cond_null)
            return v_uncond + guidance_scale * (v_cond - v_uncond)

        kwargs = dict(
            context_embed=weak_context,
            context_keep=context_keep,
            cond_embed=cond_c,
        )
        samples = sample_fn(x_init, model_fn, **kwargs)
        z = samples[-1]
        z = self.latent_norm(z) if renorm_output else z
        if return_trajectory:
            t_grid = torch.linspace(
                float(t_start), 1.0, samples.shape[0], device=device, dtype=dtype
            )
            if self.timestep_shift_alpha > 0:
                a = self.timestep_shift_alpha
                t_grid = a * t_grid / (1.0 + (a - 1.0) * t_grid)
            return z, samples, t_grid
        return z
