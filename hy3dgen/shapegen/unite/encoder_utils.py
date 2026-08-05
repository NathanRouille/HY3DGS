"""AdaLN building blocks (adapted from UNITE modules/autoencoding_utils/encoder_utils.py)."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def modulate(x, shift, scale):
    if shift is not None:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return x * (1 + scale.unsqueeze(1))


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(self, dim, pt_seq_len=16, num_cls_token=0):
        super().__init__()
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(pt_seq_len) / pt_seq_len * pt_seq_len
        freqs = torch.einsum("..., f -> ... f", t, freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])
        if num_cls_token > 0:
            cos_pad = torch.ones(num_cls_token, freqs_cos.shape[-1])
            sin_pad = torch.zeros(num_cls_token, freqs_sin.shape[-1])
            freqs_cos = torch.cat([cos_pad, freqs_cos], dim=0)
            freqs_sin = torch.cat([sin_pad, freqs_sin], dim=0)
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def forward(self, t):
        _, _, lt, _ = t.shape
        freqs_cos = self.freqs_cos[:lt]
        freqs_sin = self.freqs_sin[:lt]
        return t * freqs_cos + rotate_half(t) * freqs_sin


class SwiGLUFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight.to(output.dtype)


class NormAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        use_rmsnorm=True,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        norm_layer = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope=None, key_padding_mask=None):
        """
        Args:
            key_padding_mask: optional [B, N] bool, True = keep token.
        """
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            q = rope(q)
            k = rope(k)
        q = q.to(v.dtype)
        k = k.to(v.dtype)
        attn_mask = None
        if key_padding_mask is not None:
            # SDPA additive mask: [B, 1, 1, N] — True keep → 0, False → -inf
            fill = torch.zeros(
                b, 1, 1, n, device=x.device, dtype=q.dtype
            )
            fill = fill.masked_fill(~key_padding_mask[:, None, None, :], torch.finfo(q.dtype).min)
            attn_mask = fill
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop.p if self.training else 0.0
        )
        x = x.transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        return self.proj_drop(x)


class GaussianFourierEmbedding(nn.Module):
    """Timestep embedding for t in [0, 1].

    ``W`` is a fixed random Fourier frequency basis. It **must** be checkpointed
    (``persistent=True``): the flow is trained against these frequencies. If ``W``
    is re-sampled at load time, generation collapses even when MLP weights match.
    """

    def __init__(self, hidden_size, embedding_size=256, scale=1.0):
        super().__init__()
        w = torch.normal(mean=0.0, std=scale, size=(embedding_size,))
        self.register_buffer("W", w, persistent=True)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size * 2, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t):
        if t.dim() == 0:
            t = t[None]
        dev = self.W.device
        t = t.to(device=dev, dtype=torch.float32)
        tn = t[:, None]
        angles = tn * self.W[None, :] * (2.0 * math.pi)
        feats = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.mlp(feats)


class NullContextEmbedder(nn.Module):
    """Learnable null embedding for classifier-free weak-context dropout."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.null = nn.Parameter(torch.zeros(hidden_size))
        nn.init.normal_(self.null, std=0.02)

    def forward(self, batch_size: int, device, dtype):
        return self.null.view(1, 1, -1).expand(batch_size, 1, -1).to(device=device, dtype=dtype)
