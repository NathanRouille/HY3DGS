"""UNITE-style Shape Point-Cloud Autoencoder (ShapePCAE).

FPS cross-attn local tokens + noise registers through a Generative Encoder,
64-d bottleneck (no KL), anchor-aligned colored point decoder.

Parallel to ShapeGSAE — does not modify the 3DGS path.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .attention_blocks import FourierEmbedder, PointCrossAttentionEncoder
from .generative_encoder import (
    GenerativeEncoder,
    SwiGLUTransformer,
    init_sincos_pos_embed,
)
from .model import ShapeVAE
from ...gs_export import surface_rgb_slice
from ...utils import logger


class ShapePCAE(nn.Module):
    """Colored point cloud → register latents → anchor-aligned colored PC.

    Encode:
        surface → FPS CA locals [B,L,W]
        noise [B,R,D] → up → + register PE → concat with locals → GE → keep R
        → down → LayerNorm → z [B,R,D]

    Decode:
        z → up → + decoder PE → Transformer → centers + K*(xyz_delta, rgb)
    """

    def __init__(
        self,
        *,
        num_latents: int = 1024,
        num_registers: Optional[int] = None,
        embed_dim: int = 64,
        width: int = 1024,
        heads: int = 16,
        num_ge_layers: int = 8,
        num_decoder_layers: int = 4,
        pc_size: int = 5120,
        pc_sharpedge_size: int = 5120,
        point_feats: int = 6,
        downsample_ratio: int = 20,
        num_freqs: int = 8,
        include_pi: bool = True,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        drop_path_rate: float = 0.0,
        use_ln_post: bool = True,
        num_points_per_anchor: int = 8,
        deterministic_encoder: bool = True,
        max_anchor_delta: float = 0.1,
        ckpt_path=None,
    ):
        super().__init__()
        self.num_latents = int(num_latents)
        self.num_registers = int(num_registers) if num_registers is not None else self.num_latents
        self.embed_dim = int(embed_dim)
        self.width = int(width)
        self.point_feats = int(point_feats)
        self.num_points_per_anchor = int(num_points_per_anchor)
        self.max_anchor_delta = float(max_anchor_delta)
        self.latent_shape = (self.num_registers, self.embed_dim)

        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)

        # Cross-attn only (layers=0): GE replaces local self-attn.
        self.encoder = PointCrossAttentionEncoder(
            fourier_embedder=self.fourier_embedder,
            num_latents=self.num_latents,
            downsample_ratio=downsample_ratio,
            pc_size=pc_size,
            pc_sharpedge_size=pc_sharpedge_size,
            point_feats=point_feats,
            width=width,
            heads=heads,
            layers=0,
            qkv_bias=qkv_bias,
            use_ln_post=use_ln_post,
            qk_norm=qk_norm,
            deterministic=deterministic_encoder,
        )

        ge_ctx = self.num_registers + self.num_latents
        self.register_up = nn.Linear(self.embed_dim, width)
        self.register_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_registers, width), requires_grad=True
        )
        init_sincos_pos_embed(self.register_pos_embed, self.num_registers)

        self.ge = GenerativeEncoder(
            n_ctx=ge_ctx,
            width=width,
            layers=num_ge_layers,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            drop_path_rate=drop_path_rate,
        )
        self.latent_down = nn.Linear(width, self.embed_dim)
        self.latent_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=True, eps=1e-6)

        self.decode_up = nn.Linear(self.embed_dim, width)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_registers, width), requires_grad=True
        )
        init_sincos_pos_embed(self.decoder_pos_embed, self.num_registers)

        self.decoder = SwiGLUTransformer(
            n_ctx=self.num_registers,
            width=width,
            layers=num_decoder_layers,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            drop_path_rate=drop_path_rate,
        )

        self.anchor_mlp = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 3),
        )
        # Per anchor: K * (xyz_delta 3 + rgb 3)
        self.point_head = nn.Linear(width, self.num_points_per_anchor * 6)
        nn.init.zeros_(self.point_head.weight)
        nn.init.zeros_(self.point_head.bias)

        if ckpt_path is not None:
            self._init_from_ckpt(ckpt_path)

    def _init_from_ckpt(self, path, ignore_keys=()):
        state_dict = torch.load(path, map_location="cpu")
        state_dict = state_dict.get("state_dict", state_dict)
        for k in list(state_dict.keys()):
            if any(k.startswith(ik) for ik in ignore_keys):
                del state_dict[k]
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        logger.info(
            "Restored ShapePCAE from %s — %d missing, %d unexpected",
            path,
            len(missing),
            len(unexpected),
        )

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode(
        self,
        surface: torch.FloatTensor,
        *,
        register_noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode surface to register latents.

        Returns:
            latents: [B, R, embed_dim]
            fps_xyz: [B, L, 3] FPS query positions (for anchor aux loss)
        """
        pc = surface[:, :, :3]
        feats = surface[:, :, 3:]
        locals_tok, pc_infos = self.encoder(pc, feats)
        fps_xyz = pc_infos[0]

        B = surface.shape[0]
        if register_noise is None:
            register_noise = torch.randn(
                B,
                self.num_registers,
                self.embed_dim,
                device=surface.device,
                dtype=surface.dtype,
            )
        h_reg = self.register_up(register_noise)
        h_reg = h_reg + self.register_pos_embed.to(dtype=h_reg.dtype)

        x = torch.cat([h_reg, locals_tok], dim=1)
        x = self.ge(x)
        h = x[:, : self.num_registers]
        z = self.latent_norm(self.latent_down(h))
        return z, fps_xyz

    def decode(
        self,
        latents: torch.FloatTensor,
        *,
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode latents to colored points + anchors.

        Returns:
            xyz: [B, R*K, 3]
            rgb: [B, R*K, 3] in [0, 1]
            centers: [B, R, 3]
        """
        K = self.num_points_per_anchor
        h = self.decode_up(latents)
        h = h + self.decoder_pos_embed.to(dtype=h.dtype)
        h = self.decoder(h)

        centers = self.anchor_mlp(h)
        raw = self.point_head(h).view(h.shape[0], h.shape[1], K, 6)
        delta = self.max_anchor_delta * torch.tanh(raw[..., :3])
        xyz = centers.unsqueeze(2) + delta
        rgb = torch.sigmoid(raw[..., 3:6])

        xyz = xyz.reshape(h.shape[0], h.shape[1] * K, 3)
        rgb = rgb.reshape(h.shape[0], h.shape[1] * K, 3)
        if return_features:
            return xyz, rgb, centers, {"features": h}
        return xyz, rgb, centers

    def forward(
        self,
        surface: torch.FloatTensor,
        *,
        register_noise: Optional[torch.Tensor] = None,
    ):
        latents, fps_xyz = self.encode(surface, register_noise=register_noise)
        xyz, rgb, centers = self.decode(latents)
        return xyz, rgb, centers, fps_xyz, latents

    @staticmethod
    def surface_gt_points(
        surface: torch.FloatTensor,
        *,
        include_sharp_label: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract GT xyz + rgb from surface tensor layout."""
        xyz = surface[:, :, :3]
        rgb_sl = surface_rgb_slice(
            surface.shape[-1], include_sharp_label=include_sharp_label
        )
        rgb = surface[:, :, rgb_sl].clamp(0.0, 1.0)
        return xyz, rgb

    # ------------------------------------------------------------------
    # Pretrained loading (input_proj + cross_attn only)
    # ------------------------------------------------------------------

    @staticmethod
    def _map_input_proj_vae_to_pcae(
        old_w: torch.Tensor,
        new_w: torch.Tensor,
        *,
        old_point_feats: int,
        new_point_feats: int,
        rgb_feat_init: str,
    ) -> torch.Tensor:
        fourier_dim = old_w.shape[1] - old_point_feats
        if new_w.shape[1] - new_point_feats != fourier_dim:
            raise ValueError(
                f"Fourier dim mismatch: old feats={old_point_feats}, new feats={new_point_feats}"
            )
        out = new_w.clone()
        out[:, : fourier_dim + old_point_feats] = old_w
        rgb_cols = new_point_feats - old_point_feats
        rgb_start = fourier_dim + old_point_feats
        rgb_weight = out[:, rgb_start : rgb_start + rgb_cols]
        if rgb_feat_init == "zero":
            nn.init.zeros_(rgb_weight)
        elif rgb_feat_init == "kaiming":
            nn.init.kaiming_uniform_(rgb_weight, a=math.sqrt(5))
        elif rgb_feat_init == "label_scale":
            label_col = old_w[:, fourier_dim + 3 : fourier_dim + 4]
            rgb_weight.copy_(label_col.unsqueeze(1).expand(-1, rgb_cols) * 0.01)
        else:
            raise ValueError(f"Unknown rgb_feat_init {rgb_feat_init!r}")
        return out

    def _load_shapevae_state_dict(
        self,
        ckpt_or_repo: str,
        *,
        subfolder: str,
        use_safetensors: bool = False,
    ) -> dict:
        if os.path.isfile(ckpt_or_repo):
            if ckpt_or_repo.endswith(".safetensors"):
                import safetensors.torch

                raw = safetensors.torch.load_file(ckpt_or_repo, device="cpu")
            else:
                raw = torch.load(ckpt_or_repo, map_location="cpu", weights_only=True)
            return raw.get("state_dict", raw)

        vae = ShapeVAE.from_pretrained(
            ckpt_or_repo,
            subfolder=subfolder,
            device="cpu",
            dtype=torch.float32,
            use_safetensors=use_safetensors,
        )
        state_dict = vae.state_dict()
        del vae
        return state_dict

    def load_shapevae_cross_attn(
        self,
        ckpt_or_repo: str,
        *,
        rgb_feat_init: str = "kaiming",
        subfolder: str = "hunyuan3d-vae-v2-mini-withencoder",
        shapevae_point_feats: int = 4,
        use_safetensors: bool = False,
        strict_geometry: bool = False,
    ) -> dict:
        """Load Hunyuan ``input_proj`` + ``cross_attn`` only (GE stays scratch)."""
        ckpt = self._load_shapevae_state_dict(
            ckpt_or_repo,
            subfolder=subfolder,
            use_safetensors=use_safetensors,
        )
        report: Dict = {
            "source": ckpt_or_repo,
            "rgb_feat_init": rgb_feat_init,
            "loaded_keys": [],
        }

        enc_prefix = "encoder."
        # input_proj
        for key in ("input_proj.weight", "input_proj.bias"):
            full = enc_prefix + key
            if full not in ckpt:
                continue
            if key == "input_proj.weight":
                mapped = self._map_input_proj_vae_to_pcae(
                    ckpt[full],
                    self.encoder.input_proj.weight.data,
                    old_point_feats=shapevae_point_feats,
                    new_point_feats=self.point_feats,
                    rgb_feat_init=rgb_feat_init,
                )
                self.encoder.input_proj.weight.data.copy_(mapped)
            else:
                if self.encoder.input_proj.bias is not None:
                    self.encoder.input_proj.bias.data.copy_(ckpt[full])
            report["loaded_keys"].append(key)

        # cross_attn subtree
        ca_prefix = enc_prefix + "cross_attn."
        ca_ckpt = {
            k[len(ca_prefix) :]: v for k, v in ckpt.items() if k.startswith(ca_prefix)
        }
        if ca_ckpt:
            missing, unexpected = self.encoder.cross_attn.load_state_dict(
                ca_ckpt, strict=False
            )
            report["cross_attn_missing"] = missing
            report["cross_attn_unexpected"] = unexpected
            if (missing or unexpected) and strict_geometry:
                raise RuntimeError(
                    f"cross_attn load mismatch — missing={missing} unexpected={unexpected}"
                )
            if missing or unexpected:
                logger.warning(
                    "cross_attn load: %d missing, %d unexpected",
                    len(missing),
                    len(unexpected),
                )
            else:
                logger.info("Loaded pretrained cross_attn — perfect match")
            report["loaded_keys"].append("cross_attn.*")

        logger.info(
            "ShapePCAE pretrained CA load from %s: keys=%s",
            ckpt_or_repo,
            report["loaded_keys"],
        )
        return report

    def freeze_modules(
        self,
        *,
        encoder: bool = False,
        ge: bool = False,
        decoder: bool = False,
    ) -> None:
        if encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        if ge:
            for mod in (self.register_up, self.ge, self.latent_down, self.latent_norm):
                for p in mod.parameters():
                    p.requires_grad = False
            self.register_pos_embed.requires_grad = False
        if decoder:
            for mod in (
                self.decode_up,
                self.decoder,
                self.anchor_mlp,
                self.point_head,
            ):
                for p in mod.parameters():
                    p.requires_grad = False
            self.decoder_pos_embed.requires_grad = False
