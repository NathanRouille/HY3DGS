"""ShapePCAE + UNITE joint tokenization and flow matching."""

from __future__ import annotations

import copy
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .shape_pc_ae import ShapePCAE
from ...unite.adaln_encoder import AdaLNGenerativeEncoder
from ...unite.flow_loss import compute_flow_loss, noising_latents
from ...unite.transport import Sampler, Transport


class ShapePCUnite(ShapePCAE):
    """ShapePCAE with shared AdaLN GE for tokenizer + flow denoising."""

    def __init__(
        self,
        *,
        modulation_recon_timestep_max: float = 0.01,
        noising_t_start: float = 0.7,
        flow_steps_per_recon: int = 3,
        gen_loss_weight: float = 1.0,
        lognorm_mu: float = 1.0,
        lognorm_sigma: float = 1.6,
        timestep_shift_alpha: float = 0.0,
        use_rope: bool = False,
        weak_context_dropout: float = 0.1,
        flow_loss_type: str = "velocity",
        num_ge_layers: int = 8,
        qk_norm: bool = True,
        **kwargs,
    ):
        self._unite_num_ge_layers = num_ge_layers
        self._unite_qk_norm = qk_norm
        super().__init__(num_ge_layers=num_ge_layers, qk_norm=qk_norm, **kwargs)
        self.modulation_recon_timestep_max = float(modulation_recon_timestep_max)
        self.noising_t_start = float(noising_t_start)
        self.flow_steps_per_recon = int(flow_steps_per_recon)
        self.gen_loss_weight = float(gen_loss_weight)
        self.timestep_shift_alpha = float(timestep_shift_alpha)
        self.weak_context_dropout = float(weak_context_dropout)
        self.flow_loss_type = str(flow_loss_type)
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

        self.transport = Transport(
            lognorm_mu=lognorm_mu,
            lognorm_sigma=lognorm_sigma,
        )
        self.flow_sampler = Sampler(self.transport)
        self.null_weak_context = nn.Parameter(
            torch.zeros(1, 1, self.width), requires_grad=True
        )
        nn.init.normal_(self.null_weak_context, std=0.02)

    def _ge_pos_embed(self) -> torch.Tensor:
        return self.register_pos_embed

    def freeze_tokenizer_ge(self) -> None:
        """Snapshot the GE so the tokenizer pass stops following flow updates.

        ``--freeze_recon`` only froze the encoder/decoder, but the GE is shared,
        so flow training kept rewriting the tokenizer (Exp 3: recon CD drifted
        0.0012 -> 0.03-0.16 while flow decreased).
        """
        self.tokenizer_ge = copy.deepcopy(self.ge)
        self.tokenizer_ge.eval()
        for p in self.tokenizer_ge.parameters():
            p.requires_grad = False

    def _run_ge(
        self,
        register_slots: torch.Tensor,
        t: torch.Tensor,
        *,
        context_embed: Optional[torch.Tensor] = None,
        context_keep: Optional[torch.Tensor] = None,
        use_tokenizer_ge: bool = False,
    ) -> torch.Tensor:
        ge = self.tokenizer_ge if (use_tokenizer_ge and self.tokenizer_ge is not None) else self.ge
        pos = self._ge_pos_embed().expand(register_slots.shape[0], -1, -1)
        return ge(
            register_slots,
            t,
            pos_embed=pos,
            context_embed=context_embed,
            context_keep=context_keep,
        )

    def null_context(self, batch_size: int, num_tokens: int, *, dtype=None, device=None) -> torch.Tensor:
        """Learned null weak context, expanded to ``num_tokens``."""
        null = self.null_weak_context
        if device is not None:
            null = null.to(device)
        if dtype is not None:
            null = null.to(dtype)
        return null.expand(batch_size, num_tokens, -1)

    def encode_tokenizer(
        self,
        surface: torch.FloatTensor,
        *,
        register_noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Strong-context tokenizer pass → normalized latents."""
        pc = surface[:, :, :3]
        feats = surface[:, :, 3:]
        locals_tok, pc_infos = self.encoder(pc, feats)
        fps_xyz = pc_infos[0]

        b = surface.shape[0]
        if register_noise is None:
            register_noise = torch.randn(
                b,
                self.num_registers,
                self.embed_dim,
                device=surface.device,
                dtype=surface.dtype,
            )

        t = torch.rand(b, device=surface.device, dtype=surface.dtype)
        t = t * self.modulation_recon_timestep_max

        h = self._run_ge(
            register_noise, t, context_embed=locals_tok, use_tokenizer_ge=True
        )
        z = self.latent_norm(h)
        return z, fps_xyz, locals_tok

    def encode(self, surface, *, register_noise=None):
        z, fps_xyz, _ = self.encode_tokenizer(surface, register_noise=register_noise)
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

        velocity_mse = 0.0
        x_start_mse = 0.0
        for t in t_list:
            model_kwargs = dict(context_embed=ctx, context_keep=keep)
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

        n = max(self.flow_steps_per_recon, 1)
        return {
            "flow/flow_loss": flow_loss / n,
            "flow/velocity_mse": velocity_mse / n,
            "flow/x_start_mse": x_start_mse / n,
            "flow/t_mean": torch.stack(t_list).mean(),
        }

    def _flow_model_fn(self, xt, t, context_embed=None, context_keep=None):
        return self._run_ge(
            xt, t, context_embed=context_embed, context_keep=context_keep
        )

    def forward_tokenizer(
        self,
        surface: torch.Tensor,
        *,
        representation_phase: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Encode + decode surface.

        ``representation_phase`` controls latent noising before decode (UNITE
        representation training). Default: use ``self.representation_noising``
        (True unless the trainer disables it for clean overfits).
        """
        if representation_phase is None:
            representation_phase = bool(getattr(self, "representation_noising", True))
        z, fps_xyz, _locals = self.encode_tokenizer(surface)
        xyz, rgb, centers = self.decode(z, representation_phase=representation_phase)
        return z, fps_xyz, xyz, rgb, centers

    def forward(
        self,
        surface: torch.Tensor,
        weak_context: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        z, fps_xyz, xyz, rgb, centers = self.forward_tokenizer(surface)
        out = {
            "z": z,
            "fps_xyz": fps_xyz,
            "xyz": xyz,
            "rgb": rgb,
            "centers": centers,
        }
        if weak_context is not None or self.training:
            out.update(self.forward_denoising(z, weak_context))
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
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Image-only generation: ODE sample register latents from noise.

        Args:
            guidance_scale: classifier-free guidance strength; 1.0 disables it.
                The null embedding is the one trained via ``weak_context_dropout``.
            z_init / t_start: start the ODE from a partially noised latent
                instead of pure noise (used by the partial-noise oracle eval).
        """
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

        def predict(x_t, t, ctx, keep=None):
            x_pred = self.latent_norm(
                self._run_ge(x_t, t, context_embed=ctx, context_keep=keep)
            )
            denom = (1.0 - t.view(-1, 1, 1)).clamp_min(train_eps)
            return (x_pred - x_t) / denom

        def model_fn(x_t, t, **kwargs):
            ctx = kwargs.get("context_embed")
            keep = kwargs.get("context_keep")
            if ctx is None:
                ctx = self.null_context(
                    x_t.shape[0], n_tokens, dtype=x_t.dtype, device=x_t.device
                )
                return predict(x_t, t, ctx, None)
            v_cond = predict(x_t, t, ctx, keep)
            if not use_cfg:
                return v_cond
            null = self.null_context(
                x_t.shape[0], n_tokens, dtype=ctx.dtype, device=ctx.device
            )
            v_uncond = predict(x_t, t, null, None)
            return v_uncond + guidance_scale * (v_cond - v_uncond)

        kwargs = dict(context_embed=weak_context, context_keep=context_keep)
        samples = sample_fn(x_init, model_fn, **kwargs)
        z = samples[-1]
        return self.latent_norm(z)
