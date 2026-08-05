"""AdaLN Generative Encoder (adapted from UNITE modules/encoder.py)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .encoder_utils import (
    GaussianFourierEmbedding,
    NormAttention,
    RMSNorm,
    SwiGLUFFN,
    VisionRotaryEmbeddingFast,
    modulate,
)


class Block(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=True,
        use_rmsnorm=True,
        wo_shift=False,
        block_norm=True,
    ):
        super().__init__()
        self.block_norm = block_norm
        norm_cls = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.norm1 = norm_cls(hidden_size)
        self.norm2 = norm_cls(hidden_size)
        self.norm3 = norm_cls(hidden_size) if block_norm else None
        self.attn = NormAttention(
            hidden_size,
            num_heads=num_heads,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
        )
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = (
            SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden))
            if use_swiglu
            else nn.Sequential(
                nn.Linear(hidden_size, mlp_hidden),
                nn.GELU(approximate="tanh"),
                nn.Linear(mlp_hidden, hidden_size),
            )
        )
        n_mod = 4 if wo_shift else 6
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, n_mod * hidden_size, bias=True)
        )
        self.wo_shift = wo_shift

    def forward(self, x, c, feat_rope=None, key_padding_mask=None):
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
            shift_msa = shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(c).chunk(6, dim=1)
            )
        attn_out = self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            rope=feat_rope,
            key_padding_mask=key_padding_mask,
        )
        x = x + gate_msa.unsqueeze(1) * attn_out
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        if self.norm3 is not None:
            x = self.norm3(x)
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, use_rmsnorm=True):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size) if use_rmsnorm else nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class AdaLNGenerativeEncoder(nn.Module):
    """Shared tokenizer + flow denoiser (UNITE GE with AdaLN).

    Args:
        in_channels: latent dim per register slot (embed_dim)
        hidden_size: transformer width
        num_output_tokens: R — keep first R tokens after forward
    """

    def __init__(
        self,
        *,
        in_channels: int = 64,
        hidden_size: int = 1024,
        depth: int = 8,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_output_tokens: int = 1024,
        max_tokens: int = 8192,
        use_qknorm: bool = True,
        use_rope: bool = False,
        use_rmsnorm: bool = True,
        block_norm: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_output_tokens = num_output_tokens
        self.use_rope = use_rope

        self.up_sample = nn.Linear(in_channels, hidden_size, bias=True)
        self.t_embedder = GaussianFourierEmbedding(hidden_size)
        if use_rope:
            half = hidden_size // num_heads // 2
            self.feat_rope = VisionRotaryEmbeddingFast(
                dim=half * 2, pt_seq_len=max_tokens
            )
        else:
            self.feat_rope = None

        self.blocks = nn.ModuleList(
            [
                Block(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    use_qknorm=use_qknorm,
                    use_rmsnorm=use_rmsnorm,
                    block_norm=block_norm,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, in_channels, use_rmsnorm=use_rmsnorm)
        self._init_weights()

    def _init_weights(self):
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        pos_embed: Optional[torch.Tensor] = None,
        context_embed: Optional[torch.Tensor] = None,
        context_keep: Optional[torch.Tensor] = None,
        checkpoint_blocks: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, in_channels] register slots (noise or noised latents)
            t: [B] timesteps in [0, 1]
            pos_embed: [B, T, hidden] register positional embedding
            context_embed: optional [B, C, hidden] strong/weak context tokens
            context_keep: optional [B, C] bool — False marks discarded/padded
                weak-context tokens (background patches). Register tokens are
                always kept.
        """
        x = self.up_sample(x)
        if pos_embed is not None:
            x = x + pos_embed.to(dtype=x.dtype)
        n_reg = x.shape[1]
        key_padding = None
        if context_embed is not None:
            x = torch.cat([x, context_embed.to(dtype=x.dtype)], dim=1)
            if context_keep is not None:
                reg_keep = torch.ones(
                    x.shape[0], n_reg, dtype=torch.bool, device=x.device
                )
                key_padding = torch.cat(
                    [reg_keep, context_keep.to(device=x.device, dtype=torch.bool)], dim=1
                )

        c = self.t_embedder(t)
        rope = self.feat_rope
        for block in self.blocks:
            if checkpoint_blocks and self.training:
                # Must use kwargs — positional 4th arg would bind to feat_rope.
                x = checkpoint(
                    lambda _x, _c, _r, _m: block(
                        _x, _c, feat_rope=_r, key_padding_mask=_m
                    ),
                    x,
                    c,
                    rope,
                    key_padding,
                    use_reentrant=False,
                )
            else:
                x = block(x, c, rope, key_padding_mask=key_padding)
        x = x[:, : self.num_output_tokens]
        return self.final_layer(x, c)
