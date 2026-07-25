"""UNITE-style Generative Encoder blocks for ShapePCAE.

Pre-LN LayerNorm + multi-head self-attention + SwiGLU FFN.
No AdaLN / RoPE (AE-only v1).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_blocks import DropPath, QKVMultiheadAttention


def get_1d_sincos_pos_embed(embed_dim: int, num_tokens: int) -> np.ndarray:
    """1D sinusoidal positional embedding, shape ``(num_tokens, embed_dim)``."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = np.arange(num_tokens, dtype=np.float64)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def init_sincos_pos_embed(param: nn.Parameter, num_tokens: int) -> None:
    """In-place init of ``[1, N, D]`` or ``[N, D]`` parameter from 1D sincos."""
    data = param.data
    if data.ndim == 3:
        n, d = data.shape[1], data.shape[2]
    else:
        n, d = data.shape[0], data.shape[1]
    assert n == num_tokens and d == data.shape[-1]
    pe = get_1d_sincos_pos_embed(d, num_tokens)
    pe_t = torch.from_numpy(pe).float()
    if data.ndim == 3:
        param.data.copy_(pe_t.unsqueeze(0))
    else:
        param.data.copy_(pe_t)


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward (UNITE-style)."""

    def __init__(
        self,
        width: int,
        expand_ratio: float = 8 / 3,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        hidden = int(width * expand_ratio)
        # Round to multiple of 64 for efficiency
        hidden = max(64, (hidden + 63) // 64 * 64)
        self.w12 = nn.Linear(width, 2 * hidden, bias=True)
        self.w3 = nn.Linear(hidden, width, bias=True)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.drop_path(self.w3(F.silu(x1) * x2))


class SwiGLUAttentionBlock(nn.Module):
    """Pre-LN residual block: MHA + SwiGLU."""

    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        heads: int,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        drop_path_rate: float = 0.0,
        mlp_expand_ratio: float = 8 / 3,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(width, elementwise_affine=True, eps=1e-6)
        self.c_qkv = nn.Linear(width, width * 3, bias=qkv_bias)
        self.attention = QKVMultiheadAttention(
            heads=heads,
            n_ctx=n_ctx,
            width=width,
            qk_norm=qk_norm,
            norm_layer=nn.LayerNorm,
        )
        self.c_proj = nn.Linear(width, width)
        self.attn_drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.ln_2 = nn.LayerNorm(width, elementwise_affine=True, eps=1e-6)
        self.mlp = SwiGLUFFN(width, expand_ratio=mlp_expand_ratio, drop_path_rate=drop_path_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln_1(x)
        qkv = self.c_qkv(h)
        h = self.attn_drop_path(self.c_proj(self.attention(qkv)))
        x = x + h
        x = x + self.mlp(self.ln_2(x))
        return x


class GenerativeEncoder(nn.Module):
    """Stack of SwiGLU attention blocks (register + context self-attn)."""

    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        drop_path_rate: float = 0.0,
        mlp_expand_ratio: float = 8 / 3,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                SwiGLUAttentionBlock(
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    drop_path_rate=drop_path_rate,
                    mlp_expand_ratio=mlp_expand_ratio,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.resblocks:
            x = block(x)
        return x


class SwiGLUTransformer(nn.Module):
    """Decoder-side transformer with SwiGLU (same blocks as GE)."""

    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        drop_path_rate: float = 0.0,
        mlp_expand_ratio: float = 8 / 3,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                SwiGLUAttentionBlock(
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    drop_path_rate=drop_path_rate,
                    mlp_expand_ratio=mlp_expand_ratio,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.resblocks:
            x = block(x)
        return x
