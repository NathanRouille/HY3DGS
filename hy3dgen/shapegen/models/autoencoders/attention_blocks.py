# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.


import os
from typing import Optional, Union, List

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from .attention_processors import CrossAttentionProcessor
from ...utils import logger

scaled_dot_product_attention = nn.functional.scaled_dot_product_attention


if os.environ.get('USE_SAGEATTN', '0') == '1':
    try:
        from sageattention import sageattn
    except ImportError:
        raise ImportError('Please install the package "sageattention" to use this USE_SAGEATTN.')
    scaled_dot_product_attention = sageattn


class FourierEmbedder(nn.Module):
    """The sin/cosine positional embedding. Given an input tensor `x` of shape [n_batch, ..., c_dim], it converts
    each feature dimension of `x[..., i]` into:
        [
            sin(x[..., i]),
            sin(f_1*x[..., i]),
            sin(f_2*x[..., i]),
            ...
            sin(f_N * x[..., i]),
            cos(x[..., i]),
            cos(f_1*x[..., i]),
            cos(f_2*x[..., i]),
            ...
            cos(f_N * x[..., i]),
            x[..., i]     # only present if include_input is True.
        ], here f_i is the frequency.

    Denote the space is [0 / num_freqs, 1 / num_freqs, 2 / num_freqs, 3 / num_freqs, ..., (num_freqs - 1) / num_freqs].
    If logspace is True, then the frequency f_i is [2^(0 / num_freqs), ..., 2^(i / num_freqs), ...];
    Otherwise, the frequencies are linearly spaced between [1.0, 2^(num_freqs - 1)].

    Args:
        num_freqs (int): the number of frequencies, default is 6;
        logspace (bool): If logspace is True, then the frequency f_i is [..., 2^(i / num_freqs), ...],
            otherwise, the frequencies are linearly spaced between [1.0, 2^(num_freqs - 1)];
        input_dim (int): the input dimension, default is 3;
        include_input (bool): include the input tensor or not, default is True.

    Attributes:
        frequencies (torch.Tensor): If logspace is True, then the frequency f_i is [..., 2^(i / num_freqs), ...],
                otherwise, the frequencies are linearly spaced between [1.0, 2^(num_freqs - 1);

        out_dim (int): the embedding size, if include_input is True, it is input_dim * (num_freqs * 2 + 1),
            otherwise, it is input_dim * num_freqs * 2.

    """

    def __init__(self,
                 num_freqs: int = 6,
                 logspace: bool = True,
                 input_dim: int = 3,
                 include_input: bool = True,
                 include_pi: bool = True) -> None:

        """The initialization"""

        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                num_freqs,
                dtype=torch.float32
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (num_freqs - 1),
                num_freqs,
                dtype=torch.float32
            )

        if include_pi:
            frequencies *= torch.pi

        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs

        self.out_dim = self.get_dims(input_dim)

    def get_dims(self, input_dim):
        temp = 1 if self.include_input or self.num_freqs == 0 else 0
        out_dim = input_dim * (self.num_freqs * 2 + temp)

        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward process.

        Args:
            x: tensor of shape [..., dim]

        Returns:
            embedding: an embedding of `x` of shape [..., dim * (num_freqs * 2 + temp)]
                where temp is 1 if include_input is True and 0 otherwise.
        """

        if self.num_freqs > 0:
            embed = (x[..., None].contiguous() * self.frequencies).view(*x.shape[:-1], -1)
            if self.include_input:
                return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
            else:
                return torch.cat((embed.sin(), embed.cos()), dim=-1)
        else:
            return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

        This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
        the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
        See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
        changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
        'survival rate' as the argument.

        """
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


class MLP(nn.Module):
    def __init__(
        self, *,
        width: int,
        expand_ratio: int = 4,
        output_width: int = None,
        drop_path_rate: float = 0.0
    ):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * expand_ratio)
        self.c_proj = nn.Linear(width * expand_ratio, output_width if output_width is not None else width)
        self.gelu = nn.GELU()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.c_proj(self.gelu(self.c_fc(x))))


class QKVMultiheadCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        heads: int,
        n_data: Optional[int] = None,
        width=None,
        qk_norm=False,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.heads = heads
        self.n_data = n_data
        self.q_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()

        self.attn_processor = CrossAttentionProcessor()

    def forward(self, q, kv):
        _, n_ctx, _ = q.shape
        bs, n_data, width = kv.shape
        attn_ch = width // self.heads // 2
        q = q.view(bs, n_ctx, self.heads, -1)
        kv = kv.view(bs, n_data, self.heads, -1)
        k, v = torch.split(kv, attn_ch, dim=-1)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k, v = map(lambda t: rearrange(t, 'b n h d -> b h n d', h=self.heads), (q, k, v))
        out = self.attn_processor(self, q, k, v)
        out = out.transpose(1, 2).reshape(bs, n_ctx, -1)
        return out


class MultiheadCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        heads: int,
        qkv_bias: bool = True,
        n_data: Optional[int] = None,
        data_width: Optional[int] = None,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
        kv_cache: bool = False,
    ):
        super().__init__()
        self.n_data = n_data
        self.width = width
        self.heads = heads
        self.data_width = width if data_width is None else data_width
        self.c_q = nn.Linear(width, width, bias=qkv_bias)
        self.c_kv = nn.Linear(self.data_width, width * 2, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadCrossAttention(
            heads=heads,
            n_data=n_data,
            width=width,
            norm_layer=norm_layer,
            qk_norm=qk_norm
        )
        self.kv_cache = kv_cache
        self.data = None

    def forward(self, x, data):
        x = self.c_q(x)
        if self.kv_cache:
            if self.data is None:
                self.data = self.c_kv(data)
                logger.info('Save kv cache,this should be called only once for one mesh')
            data = self.data
        else:
            data = self.c_kv(data)
        x = self.attention(x, data)
        x = self.c_proj(x)
        return x


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        n_data: Optional[int] = None,
        width: int,
        heads: int,
        mlp_expand_ratio: int = 4,
        data_width: Optional[int] = None,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False
    ):
        super().__init__()

        if data_width is None:
            data_width = width

        self.attn = MultiheadCrossAttention(
            n_data=n_data,
            width=width,
            heads=heads,
            data_width=data_width,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            qk_norm=qk_norm
        )
        self.ln_1 = norm_layer(width, elementwise_affine=True, eps=1e-6)
        self.ln_2 = norm_layer(data_width, elementwise_affine=True, eps=1e-6)
        self.ln_3 = norm_layer(width, elementwise_affine=True, eps=1e-6)
        self.mlp = MLP(width=width, expand_ratio=mlp_expand_ratio)

    def forward(self, x: torch.Tensor, data: torch.Tensor):
        x = x + self.attn(self.ln_1(x), self.ln_2(data))
        x = x + self.mlp(self.ln_3(x))
        return x


class QKVMultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        heads: int,
        n_ctx: int,
        width=None,
        qk_norm=False,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.heads = heads
        self.n_ctx = n_ctx
        self.q_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k, v = map(lambda t: rearrange(t, 'b n h d -> b h n d', h=self.heads), (q, k, v))
        out = scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(bs, n_ctx, -1)
        return out


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        heads: int,
        qkv_bias: bool,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
        drop_path_rate: float = 0.0
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadAttention(
            heads=heads,
            n_ctx=n_ctx,
            width=width,
            norm_layer=norm_layer,
            qk_norm=qk_norm
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        x = self.c_qkv(x)
        x = self.attention(x)
        x = self.drop_path(self.c_proj(x))
        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        heads: int,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.attn = MultiheadAttention(
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            qk_norm=qk_norm,
            drop_path_rate=drop_path_rate
        )
        self.ln_1 = norm_layer(width, elementwise_affine=True, eps=1e-6)
        self.mlp = MLP(width=width, drop_path_rate=drop_path_rate)
        self.ln_2 = norm_layer(width, elementwise_affine=True, eps=1e-6)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
        drop_path_rate: float = 0.0
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    qk_norm=qk_norm,
                    drop_path_rate=drop_path_rate
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            x = block(x)
        return x


class CrossAttentionDecoder(nn.Module):

    def __init__(
        self,
        *,
        num_latents: int,
        out_channels: int,
        fourier_embedder: FourierEmbedder,
        width: int,
        heads: int,
        mlp_expand_ratio: int = 4,
        downsample_ratio: int = 1,
        enable_ln_post: bool = True,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        label_type: str = "binary"
    ):
        super().__init__()

        self.enable_ln_post = enable_ln_post
        self.fourier_embedder = fourier_embedder
        self.downsample_ratio = downsample_ratio
        self.query_proj = nn.Linear(self.fourier_embedder.out_dim, width)
        if self.downsample_ratio != 1:
            self.latents_proj = nn.Linear(width * downsample_ratio, width)
        if self.enable_ln_post == False:
            qk_norm = False
        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            n_data=num_latents,
            width=width,
            mlp_expand_ratio=mlp_expand_ratio,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm
        )

        if self.enable_ln_post:
            self.ln_post = nn.LayerNorm(width)
        self.output_proj = nn.Linear(width, out_channels)
        self.label_type = label_type
        self.count = 0

    def set_cross_attention_processor(self, processor):
        self.cross_attn_decoder.attn.attention.attn_processor = processor

    def set_default_cross_attention_processor(self):
        self.cross_attn_decoder.attn.attention.attn_processor = CrossAttentionProcessor

    def forward(self, queries=None, query_embeddings=None, latents=None):
        if query_embeddings is None:
            query_embeddings = self.query_proj(self.fourier_embedder(queries).to(latents.dtype))
        self.count += query_embeddings.shape[1]
        if self.downsample_ratio != 1:
            latents = self.latents_proj(latents)
        x = self.cross_attn_decoder(query_embeddings, latents)
        if self.enable_ln_post:
            x = self.ln_post(x)
        occ = self.output_proj(x)
        return occ


QUERY_SAMPLE_MODES = ("split_fps", "weighted_fps")


def fps(
    src: torch.Tensor,
    batch: Optional[Tensor] = None,
    ratio: Optional[Union[Tensor, float]] = None,
    random_start: bool = True,
    batch_size: Optional[int] = None,
    ptr: Optional[Union[Tensor, List[int]]] = None,
):
    src = src.float()
    from torch_cluster import fps as fps_fn
    output = fps_fn(src, batch, ratio, random_start, batch_size, ptr)
    return output


def knn_density_from_reference(
    candidates: torch.Tensor,
    reference: torch.Tensor,
    *,
    k: int,
    candidates_in_reference: bool = False,
) -> torch.Tensor:
    """Local density ρ_i = 1 / (d_{i,(k)} + ε) via k-NN into ``reference``.

    When ``candidates_in_reference`` is True, the 0-distance self match is
    skipped so ``k`` still means the k-th *other* neighbour.
    """
    if candidates.numel() == 0:
        return candidates.new_zeros(candidates.shape[0])
    if reference.shape[0] == 0:
        return candidates.new_ones(candidates.shape[0])

    k = max(int(k), 1)
    # Need k (+1 if self is in the reference set) neighbours.
    k_fetch = k + 1 if candidates_in_reference else k
    k_fetch = min(k_fetch, reference.shape[0])

    # Chunked cdist keeps peak memory reasonable for ~32k×16k.
    chunk = 4096
    dk_parts = []
    for start in range(0, candidates.shape[0], chunk):
        end = min(start + chunk, candidates.shape[0])
        dists = torch.cdist(candidates[start:end].float(), reference.float())
        vals, _ = dists.topk(k_fetch, largest=False, dim=-1)
        if candidates_in_reference:
            # vals[:, 0] ≈ 0 (self); use the k-th other neighbour when available.
            col = min(k, vals.shape[-1] - 1)
        else:
            col = min(k - 1, vals.shape[-1] - 1)
        dk_parts.append(vals[:, col])
    dk = torch.cat(dk_parts, dim=0).to(dtype=candidates.dtype)
    return 1.0 / (dk + 1e-6)


def percentile_clip(values: torch.Tensor, p_low: float = 5.0, p_high: float = 95.0) -> torch.Tensor:
    """Clamp to the [p_low, p_high] percentiles of ``values``."""
    if values.numel() == 0:
        return values
    lo = torch.quantile(values.float(), float(p_low) / 100.0)
    hi = torch.quantile(values.float(), float(p_high) / 100.0)
    if not torch.isfinite(lo) or not torch.isfinite(hi) or hi <= lo:
        return values
    return values.clamp(lo.to(values.dtype), hi.to(values.dtype))


def weighted_fps_indices(
    xyz: torch.Tensor,
    weights: torch.Tensor,
    num_samples: int,
    *,
    deterministic: bool = True,
) -> torch.Tensor:
    """Greedy weighted FPS: repeatedly pick argmax_i w_i * min_j∈S ||x_i - x_j||.

    Args:
        xyz: [N, 3]
        weights: [N] non-negative
        num_samples: number of indices to return
    Returns:
        LongTensor [num_samples] into the N points.
    """
    n = xyz.shape[0]
    num_samples = int(min(max(num_samples, 0), n))
    if num_samples == 0:
        return xyz.new_zeros((0,), dtype=torch.long)
    if num_samples == n:
        return torch.arange(n, device=xyz.device, dtype=torch.long)

    w = weights.float().clamp_min(0.0)
    if not torch.isfinite(w).all() or float(w.sum()) <= 0:
        w = torch.ones_like(w)

    selected = torch.empty(num_samples, device=xyz.device, dtype=torch.long)
    min_dist = torch.full((n,), float("inf"), device=xyz.device, dtype=xyz.dtype)
    taken = torch.zeros(n, device=xyz.device, dtype=torch.bool)

    if deterministic:
        start = int(torch.argmax(w).item())
    else:
        start = int(torch.multinomial(w / w.sum(), 1).item())

    pts = xyz.float()
    for i in range(num_samples):
        selected[i] = start
        taken[start] = True
        dist = torch.norm(pts - pts[start], dim=-1)
        min_dist = torch.minimum(min_dist, dist.to(dtype=min_dist.dtype))
        score = w * min_dist.float()
        score = score.masked_fill(taken, float("-inf"))
        start = int(torch.argmax(score).item())
    return selected


class PointCrossAttentionEncoder(nn.Module):

    def __init__(
        self, *,
        num_latents: int,
        downsample_ratio: float,
        pc_size: int,
        pc_sharpedge_size: int,
        fourier_embedder: FourierEmbedder,
        point_feats: int,
        width: int,
        heads: int,
        layers: int,
        normal_pe: bool = False,
        qkv_bias: bool = True,
        use_ln_post: bool = False,
        use_checkpoint: bool = False,
        qk_norm: bool = False,
        deterministic: bool = True,
        query_sample_mode: str = "split_fps",
        fps_density_k: int = 16,
        fps_sharp_beta: float = 0.2,
        fps_density_clip_low: float = 5.0,
        fps_density_clip_high: float = 95.0,
    ):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.num_latents = num_latents
        self.downsample_ratio = downsample_ratio
        self.point_feats = point_feats
        self.normal_pe = normal_pe
        # When True: input subset selection and FPS starting seed are deterministic
        # (sequential indexing + FPS random_start=False). Required for overfit so the
        # loss is a deterministic function of the parameters. Flip to False for the
        # original stochastic-augmentation behaviour during generalisation training.
        self.deterministic = deterministic

        mode = str(query_sample_mode).lower()
        if mode not in QUERY_SAMPLE_MODES:
            raise ValueError(
                f"query_sample_mode must be one of {QUERY_SAMPLE_MODES}, got {query_sample_mode!r}"
            )
        self.query_sample_mode = mode
        self.fps_density_k = int(fps_density_k)
        self.fps_sharp_beta = float(fps_sharp_beta)
        self.fps_density_clip_low = float(fps_density_clip_low)
        self.fps_density_clip_high = float(fps_density_clip_high)

        if pc_sharpedge_size == 0:
            logger.info('PointCrossAttentionEncoder: pc_sharpedge_size=0, using pc_size for both splits')
        else:
            logger.info(f'PointCrossAttentionEncoder: pc_size={pc_size}, pc_sharpedge_size={pc_sharpedge_size}')
        if self.query_sample_mode == "weighted_fps":
            logger.info(
                "PointCrossAttentionEncoder: query_sample_mode=weighted_fps "
                f"(k={self.fps_density_k}, beta={self.fps_sharp_beta}, "
                f"clip=[{self.fps_density_clip_low},{self.fps_density_clip_high}])"
            )

        self.pc_size = pc_size
        self.pc_sharpedge_size = pc_sharpedge_size

        self.fourier_embedder = fourier_embedder

        self.input_proj = nn.Linear(self.fourier_embedder.out_dim + point_feats, width)
        self.cross_attn = ResidualCrossAttentionBlock(
            width=width,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm
        )

        self.self_attn = None
        if layers > 0:
            self.self_attn = Transformer(
                n_ctx=num_latents,
                width=width,
                layers=layers,
                heads=heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm
            )

        if use_ln_post:
            self.ln_post = nn.LayerNorm(width)
        else:
            self.ln_post = None

    @staticmethod
    def _select_point_indices(
        pool_size: int,
        n_select: int,
        device: torch.device,
        *,
        deterministic: bool,
        batch_size: int,
    ) -> List[torch.Tensor]:
        """Per-batch-item point indices into a pool of size ``pool_size``."""
        n_select = min(n_select, pool_size)
        if deterministic:
            idx = torch.arange(n_select, device=device)
            return [idx for _ in range(batch_size)]

        return [
            torch.randperm(pool_size, device=device)[:n_select]
            for _ in range(batch_size)
        ]

    @staticmethod
    def _gather_batch_rows(
        tensor: torch.Tensor,
        indices_per_batch: List[torch.Tensor],
    ) -> torch.Tensor:
        """Index ``tensor[b]`` with ``indices_per_batch[b]`` for each batch item."""
        return torch.stack(
            [tensor[b, indices_per_batch[b]] for b in range(tensor.shape[0])],
            dim=0,
        )

    def _build_split_input_pools(
        self,
        pc: torch.FloatTensor,
        feats: Optional[torch.FloatTensor],
        *,
        num_latents: int,
    ):
        """Subsample uniform|sharp pools used as cross-attn KV (and split-FPS candidates)."""
        B, _, D = pc.shape
        num_random_query = self.pc_size / (self.pc_size + self.pc_sharpedge_size) * num_latents
        num_sharpedge_query = num_latents - num_random_query

        random_pc, sharpedge_pc = torch.split(pc, [self.pc_size, self.pc_sharpedge_size], dim=1)
        assert random_pc.shape[1] <= self.pc_size, "Random surface points size must be less than or equal to pc_size"
        assert sharpedge_pc.shape[
                   1] <= self.pc_sharpedge_size, "Sharpedge surface points size must be less than or equal to pc_sharpedge_size"

        input_random_pc_size = min(int(num_random_query * self.downsample_ratio), random_pc.shape[1])
        idx_random_pc_list = self._select_point_indices(
            random_pc.shape[1],
            input_random_pc_size,
            random_pc.device,
            deterministic=self.deterministic,
            batch_size=B,
        )
        input_random_pc = self._gather_batch_rows(random_pc, idx_random_pc_list)

        input_sharpedge_pc_size = int(num_sharpedge_query * self.downsample_ratio)
        if input_sharpedge_pc_size == 0:
            input_sharpedge_pc = torch.zeros(B, 0, D, dtype=input_random_pc.dtype, device=pc.device)
            idx_sharpedge_pc_list = [
                torch.zeros(0, device=pc.device, dtype=torch.long) for _ in range(B)
            ]
        else:
            input_sharpedge_pc_size = min(input_sharpedge_pc_size, sharpedge_pc.shape[1])
            idx_sharpedge_pc_list = self._select_point_indices(
                sharpedge_pc.shape[1],
                input_sharpedge_pc_size,
                sharpedge_pc.device,
                deterministic=self.deterministic,
                batch_size=B,
            )
            input_sharpedge_pc = self._gather_batch_rows(sharpedge_pc, idx_sharpedge_pc_list)

        input_random_feats = None
        input_sharpedge_feats = None
        if self.point_feats != 0 and feats is not None:
            random_feats, sharpedge_feats = torch.split(
                feats, [self.pc_size, self.pc_sharpedge_size], dim=1
            )
            input_random_feats = self._gather_batch_rows(random_feats, idx_random_pc_list)
            if input_sharpedge_pc_size == 0:
                input_sharpedge_feats = torch.zeros(
                    B, 0, self.point_feats, dtype=input_random_feats.dtype, device=pc.device
                )
            else:
                input_sharpedge_feats = self._gather_batch_rows(
                    sharpedge_feats, idx_sharpedge_pc_list
                )

        return {
            "num_random_query": num_random_query,
            "num_sharpedge_query": num_sharpedge_query,
            "input_random_pc_size": input_random_pc_size,
            "input_sharpedge_pc_size": input_sharpedge_pc_size,
            "idx_random_pc_list": idx_random_pc_list,
            "idx_sharpedge_pc_list": idx_sharpedge_pc_list,
            "input_random_pc": input_random_pc,
            "input_sharpedge_pc": input_sharpedge_pc,
            "input_random_feats": input_random_feats,
            "input_sharpedge_feats": input_sharpedge_feats,
        }

    def _split_fps_queries(self, pools: dict, *, num_latents: int, D: int):
        """Legacy Hunyuan: independent FPS on uniform and sharp pools."""
        B = pools["input_random_pc"].shape[0]
        input_random_pc_size = pools["input_random_pc_size"]
        input_sharpedge_pc_size = pools["input_sharpedge_pc_size"]
        num_random_query = pools["num_random_query"]
        num_sharpedge_query = pools["num_sharpedge_query"]
        fps_random_start = not self.deterministic

        flatten_input_random_pc = pools["input_random_pc"].reshape(B * input_random_pc_size, D)
        batch_down = torch.arange(B, device=flatten_input_random_pc.device)
        batch_down = torch.repeat_interleave(batch_down, input_random_pc_size)
        random_query_ratio = num_random_query / input_random_pc_size
        idx_query_random = fps(
            flatten_input_random_pc, batch_down, ratio=random_query_ratio,
            random_start=fps_random_start,
        )
        query_random_pc = flatten_input_random_pc[idx_query_random].view(B, -1, D)

        if input_sharpedge_pc_size == 0:
            query_sharpedge_pc = torch.zeros(
                B, 0, D, dtype=query_random_pc.dtype, device=query_random_pc.device
            )
            idx_query_sharpedge = None
        else:
            flatten_sharp = pools["input_sharpedge_pc"].reshape(B * input_sharpedge_pc_size, D)
            batch_down = torch.arange(B, device=flatten_sharp.device)
            batch_down = torch.repeat_interleave(batch_down, input_sharpedge_pc_size)
            sharpedge_query_ratio = num_sharpedge_query / input_sharpedge_pc_size
            idx_query_sharpedge = fps(
                flatten_sharp, batch_down, ratio=sharpedge_query_ratio,
                random_start=fps_random_start,
            )
            query_sharpedge_pc = flatten_sharp[idx_query_sharpedge].view(B, -1, D)

        query_pc = torch.cat([query_random_pc, query_sharpedge_pc], dim=1)

        query_feats = None
        if self.point_feats != 0 and pools["input_random_feats"] is not None:
            flat_rf = pools["input_random_feats"].reshape(B * input_random_pc_size, -1)
            query_random_feats = flat_rf[idx_query_random].view(B, -1, flat_rf.shape[-1])
            if input_sharpedge_pc_size == 0:
                query_sharpedge_feats = torch.zeros(
                    B, 0, self.point_feats, dtype=query_random_feats.dtype, device=query_random_feats.device
                )
            else:
                flat_sf = pools["input_sharpedge_feats"].reshape(B * input_sharpedge_pc_size, -1)
                query_sharpedge_feats = flat_sf[idx_query_sharpedge].view(B, -1, flat_sf.shape[-1])
            query_feats = torch.cat([query_random_feats, query_sharpedge_feats], dim=1)

        return query_pc, query_random_pc, query_sharpedge_pc, query_feats

    def _weighted_fps_queries(self, pools: dict, *, num_latents: int):
        """Density-weighted FPS on the concatenated [uniform|sharp] input pool.

        Density ρ is estimated with k-NN into the **uniform** subset only (avoids
        replacement-padded sharp duplicates inflating edge density). Optional
        sharp boost: w = clip(ρ) * (1 + β * sharp).
        """
        input_random_pc = pools["input_random_pc"]
        input_sharpedge_pc = pools["input_sharpedge_pc"]
        B, n_u, D = input_random_pc.shape
        n_s = input_sharpedge_pc.shape[1]
        input_pc = torch.cat([input_random_pc, input_sharpedge_pc], dim=1)
        n_all = input_pc.shape[1]
        num_latents = int(min(num_latents, n_all))

        input_feats = None
        if self.point_feats != 0 and pools["input_random_feats"] is not None:
            if n_s == 0:
                input_feats = pools["input_random_feats"]
            else:
                input_feats = torch.cat(
                    [pools["input_random_feats"], pools["input_sharpedge_feats"]], dim=1
                )

        # point_feats=4 (normals|sharp) or 7 (normals|sharp|rgb) → sharp at channel 3.
        use_sharp = (
            input_feats is not None
            and self.point_feats in (4, 7)
            and input_feats.shape[-1] >= 4
            and self.fps_sharp_beta != 0.0
        )

        query_list = []
        query_feat_list = []
        for b in range(B):
            cand = input_pc[b]  # [N, 3]
            ref = input_random_pc[b]  # uniform-only density reference
            rho_u = knn_density_from_reference(
                ref, ref, k=self.fps_density_k, candidates_in_reference=True
            )
            if n_s > 0:
                rho_s = knn_density_from_reference(
                    input_sharpedge_pc[b], ref, k=self.fps_density_k, candidates_in_reference=False
                )
                rho = torch.cat([rho_u, rho_s], dim=0)
            else:
                rho = rho_u
            rho = percentile_clip(
                rho, self.fps_density_clip_low, self.fps_density_clip_high
            )
            w = rho
            if use_sharp:
                sharp = input_feats[b, :, 3].float().clamp(0.0, 1.0)
                w = w * (1.0 + self.fps_sharp_beta * sharp)
            idx = weighted_fps_indices(
                cand, w, num_latents, deterministic=self.deterministic
            )
            query_list.append(cand[idx])
            if input_feats is not None:
                query_feat_list.append(input_feats[b, idx])

        query_pc = torch.stack(query_list, dim=0)
        # Debug splits: treat all weighted queries as "random" bucket (no U/S FPS split).
        query_random_pc = query_pc
        query_sharpedge_pc = torch.zeros(B, 0, D, dtype=query_pc.dtype, device=query_pc.device)
        query_feats = torch.stack(query_feat_list, dim=0) if query_feat_list else None
        return query_pc, query_random_pc, query_sharpedge_pc, query_feats

    def sample_points_and_latents(
        self,
        pc: torch.FloatTensor,
        feats: Optional[torch.FloatTensor] = None,
    ):
        B, N, D = pc.shape
        num_pts = self.num_latents * self.downsample_ratio

        # Compute number of latents
        num_latents = int(num_pts / self.downsample_ratio)

        # Select random surface points and random query points.
        # In deterministic mode use sequential indexing + FPS starting seed so
        # the encoder output is a pure function of the parameters. This matters for
        # overfit, where stochastic per-step anchors otherwise inject noise into the
        # loss that cannot be explained away by depth-sorting (see plan §1).
        pools = self._build_split_input_pools(pc, feats, num_latents=num_latents)
        input_random_pc = pools["input_random_pc"]
        input_sharpedge_pc = pools["input_sharpedge_pc"]
        input_sharpedge_pc_size = pools["input_sharpedge_pc_size"]
        input_pc = torch.cat([input_random_pc, input_sharpedge_pc], dim=1)

        if self.query_sample_mode == "weighted_fps":
            query_pc, query_random_pc, query_sharpedge_pc, query_feats = self._weighted_fps_queries(
                pools, num_latents=num_latents
            )
        else:
            query_pc, query_random_pc, query_sharpedge_pc, query_feats = self._split_fps_queries(
                pools, num_latents=num_latents, D=D
            )

        # PE
        query = self.fourier_embedder(query_pc)
        data = self.fourier_embedder(input_pc)

        # Concat normal if given
        if self.point_feats != 0 and feats is not None:
            if pools["input_random_feats"] is None:
                raise RuntimeError("point_feats>0 but surface feats were not provided")
            if input_sharpedge_pc_size == 0:
                input_feats = pools["input_random_feats"]
            else:
                input_feats = torch.cat(
                    [pools["input_random_feats"], pools["input_sharpedge_feats"]], dim=1
                )
            if query_feats is None:
                raise RuntimeError("point_feats>0 but query feats were not gathered")

            if self.normal_pe:
                query_normal_pe = self.fourier_embedder(query_feats[..., :3])
                input_normal_pe = self.fourier_embedder(input_feats[..., :3])
                query_feats = torch.cat([query_normal_pe, query_feats[..., 3:]], dim=-1)
                input_feats = torch.cat([input_normal_pe, input_feats[..., 3:]], dim=-1)

            query = torch.cat([query, query_feats], dim=-1)
            data = torch.cat([data, input_feats], dim=-1)

        if input_sharpedge_pc_size == 0:
            query_sharpedge_pc = torch.zeros(B, 1, D).to(pc.device)
            input_sharpedge_pc = torch.zeros(B, 1, D).to(pc.device)
        return query.view(B, -1, query.shape[-1]), data.view(B, -1, data.shape[-1]), [query_pc, input_pc,
                                                                                      query_random_pc, input_random_pc,
                                                                                      query_sharpedge_pc,
                                                                                      input_sharpedge_pc]

    def forward(
        self,
        pc,
        feats,
    ):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]

        Returns:

        """

        query, data, pc_infos = self.sample_points_and_latents(pc, feats)

        query = self.input_proj(query)
        query = query
        data = self.input_proj(data)
        data = data

        latents = self.cross_attn(query, data)
        if self.self_attn is not None:
            latents = self.self_attn(latents)

        if self.ln_post is not None:
            latents = self.ln_post(latents)

        return latents, pc_infos
