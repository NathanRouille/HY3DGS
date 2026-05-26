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
from typing import Union, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from .attention_blocks import FourierEmbedder, Transformer, CrossAttentionDecoder, PointCrossAttentionEncoder
from .surface_extractors import MCSurfaceExtractor, SurfaceExtractors
from .volume_decoders import VanillaVolumeDecoder, FlashVDMVolumeDecoding, HierarchicalVolumeDecoding
from ...utils import logger, synchronize_timer, smart_load_model


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: Union[torch.Tensor, List[torch.Tensor]], deterministic=False, feat_dim=1):
        self.feat_dim = feat_dim
        self.parameters = parameters

        if isinstance(parameters, list):
            self.mean = parameters[0]
            self.logvar = parameters[1]
        else:
            self.mean, self.logvar = torch.chunk(parameters, 2, dim=feat_dim)

        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self):
        x = self.mean + self.std * torch.randn_like(self.mean)
        return x

    def kl(self, other=None, dims=(1, 2, 3)):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.mean(torch.pow(self.mean, 2)
                                        + self.var - 1.0 - self.logvar,
                                        dim=dims)
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=dims)

    def nll(self, sample, dims=(1, 2, 3)):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean


class VectsetVAE(nn.Module):

    @classmethod
    @synchronize_timer('VectsetVAE Model Loading')
    def from_single_file(
        cls,
        ckpt_path,
        config_path,
        device='cuda',
        dtype=torch.float16,
        use_safetensors=None,
        **kwargs,
    ):
        # load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # load ckpt
        if use_safetensors:
            ckpt_path = ckpt_path.replace('.ckpt', '.safetensors')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model file {ckpt_path} not found")

        logger.info(f"Loading model from {ckpt_path}")
        if use_safetensors:
            import safetensors.torch
            ckpt = safetensors.torch.load_file(ckpt_path, device='cpu')
        else:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)

        model_kwargs = config['params']
        model_kwargs.update(kwargs)

        model = cls(**model_kwargs)
        model.load_state_dict(ckpt, strict=False)
        model.to(device=device, dtype=dtype)
        return model

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        device='cuda',
        dtype=torch.float16,
        use_safetensors=True,
        variant='fp16',
        subfolder='hunyuan3d-vae-v2-0',
        **kwargs,
    ):
        config_path, ckpt_path = smart_load_model(
            model_path,
            subfolder=subfolder,
            use_safetensors=use_safetensors,
            variant=variant
        )

        return cls.from_single_file(
            ckpt_path,
            config_path,
            device=device,
            dtype=dtype,
            use_safetensors=use_safetensors,
            **kwargs
        )

    def init_from_ckpt(self, path, ignore_keys=()):
        state_dict = torch.load(path, map_location="cpu")
        state_dict = state_dict.get("state_dict", state_dict)
        keys = list(state_dict.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    logger.info(f"Deleting key {k} from state_dict.")
                    del state_dict[k]
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        logger.info(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            logger.warning(f"Missing Keys: {missing}")
            logger.warning(f"Unexpected Keys: {unexpected}")

    def __init__(
        self,
        volume_decoder=None,
        surface_extractor=None
    ):
        super().__init__()
        if volume_decoder is None:
            volume_decoder = VanillaVolumeDecoder()
        if surface_extractor is None:
            surface_extractor = MCSurfaceExtractor()
        self.volume_decoder = volume_decoder
        self.surface_extractor = surface_extractor

    def latents2mesh(self, latents: torch.FloatTensor, **kwargs):
        with synchronize_timer('Volume decoding'):
            grid_logits = self.volume_decoder(latents, self.geo_decoder, **kwargs)
        with synchronize_timer('Surface extraction'):
            outputs = self.surface_extractor(grid_logits, **kwargs)
        return outputs

    def enable_flashvdm_decoder(
        self,
        enabled: bool = True,
        adaptive_kv_selection=True,
        topk_mode='mean',
        mc_algo='dmc',
    ):
        if enabled:
            if adaptive_kv_selection:
                self.volume_decoder = FlashVDMVolumeDecoding(topk_mode)
            else:
                self.volume_decoder = HierarchicalVolumeDecoding()
            if mc_algo not in SurfaceExtractors.keys():
                raise ValueError(f'Unsupported mc_algo {mc_algo}, available: {list(SurfaceExtractors.keys())}')
            self.surface_extractor = SurfaceExtractors[mc_algo]()
        else:
            self.volume_decoder = VanillaVolumeDecoder()
            self.surface_extractor = MCSurfaceExtractor()


class ShapeVAE(VectsetVAE):
    def __init__(
        self,
        *,
        num_latents: int,
        embed_dim: int,
        width: int,
        heads: int,
        num_decoder_layers: int,
        num_encoder_layers: int = 8,
        pc_size: int = 5120,
        pc_sharpedge_size: int = 5120,
        point_feats: int = 3,
        downsample_ratio: int = 20,
        geo_decoder_downsample_ratio: int = 1,
        geo_decoder_mlp_expand_ratio: int = 4,
        geo_decoder_ln_post: bool = True,
        num_freqs: int = 8,
        include_pi: bool = True,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        label_type: str = "binary",
        drop_path_rate: float = 0.0,
        scale_factor: float = 1.0,
        use_ln_post: bool = True,
        ckpt_path=None
    ):
        super().__init__()
        self.geo_decoder_ln_post = geo_decoder_ln_post
        self.downsample_ratio = downsample_ratio

        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)

        self.encoder = PointCrossAttentionEncoder(
            fourier_embedder=self.fourier_embedder,
            num_latents=num_latents,
            downsample_ratio=self.downsample_ratio,
            pc_size=pc_size,
            pc_sharpedge_size=pc_sharpedge_size,
            point_feats=point_feats,
            width=width,
            heads=heads,
            layers=num_encoder_layers,
            qkv_bias=qkv_bias,
            use_ln_post=use_ln_post,
            qk_norm=qk_norm
        )

        self.pre_kl = nn.Linear(width, embed_dim * 2)
        self.post_kl = nn.Linear(embed_dim, width)

        self.transformer = Transformer(
            n_ctx=num_latents,
            width=width,
            layers=num_decoder_layers,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            drop_path_rate=drop_path_rate
        )

        self.geo_decoder = CrossAttentionDecoder(
            fourier_embedder=self.fourier_embedder,
            out_channels=1,
            num_latents=num_latents,
            mlp_expand_ratio=geo_decoder_mlp_expand_ratio,
            downsample_ratio=geo_decoder_downsample_ratio,
            enable_ln_post=self.geo_decoder_ln_post,
            width=width // geo_decoder_downsample_ratio,
            heads=heads // geo_decoder_downsample_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            label_type=label_type,
        )

        self.scale_factor = scale_factor
        self.latent_shape = (num_latents, embed_dim)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def forward(self, latents):
        latents = self.post_kl(latents)
        latents = self.transformer(latents)
        return latents

    def encode(self, surface, sample_posterior=True):
        pc, feats = surface[:, :, :3], surface[:, :, 3:]
        latents, _ = self.encoder(pc, feats)
        moments = self.pre_kl(latents)
        posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)
        if sample_posterior:
            latents = posterior.sample()
        else:
            latents = posterior.mode()
        return latents

    def decode(self, latents):
        latents = self.post_kl(latents)
        latents = self.transformer(latents)
        return latents


# ---------------------------------------------------------------------------
# Point Cloud → 3D Gaussian Splatting Autoencoder
# ---------------------------------------------------------------------------

class ShapeGSAE(nn.Module):
    """Deterministic autoencoder: colored point cloud → 3D Gaussian Splatting.

    Input surface tensor layout: [B, N, 9] = xyz(0:3) | normals(3:6) | rgb(6:9).
    The first pc_size rows are uniform samples; the next pc_sharpedge_size rows
    are sharp-edge samples (same convention as SharpEdgeSurfaceLoader).

    Output: a tuple (means, scales, rotations, opacities, colors) where each
    element is a per-Gaussian parameter tensor of shape [B, num_latents, C].
    """

    @classmethod
    @synchronize_timer('ShapeGSAE Model Loading')
    def from_single_file(
        cls,
        ckpt_path,
        config_path,
        device='cuda',
        dtype=torch.float32,
        use_safetensors=None,
        **kwargs,
    ):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if use_safetensors:
            ckpt_path = ckpt_path.replace('.ckpt', '.safetensors')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model file {ckpt_path} not found")

        logger.info(f"Loading ShapeGSAE from {ckpt_path}")
        if use_safetensors:
            import safetensors.torch
            ckpt = safetensors.torch.load_file(ckpt_path, device='cpu')
        else:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)

        model_kwargs = config.get('params', config)
        model_kwargs.update(kwargs)
        model = cls(**model_kwargs)
        model.load_state_dict(ckpt, strict=False)
        model.to(device=device, dtype=dtype)
        return model

    def __init__(
        self,
        *,
        num_latents: int,
        embed_dim: int,
        width: int,
        heads: int,
        num_decoder_layers: int,
        num_encoder_layers: int = 8,
        pc_size: int = 5120,
        pc_sharpedge_size: int = 5120,
        point_feats: int = 6,       # normals(3) + rgb(3)
        downsample_ratio: int = 20,
        num_freqs: int = 8,
        include_pi: bool = True,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        drop_path_rate: float = 0.0,
        use_ln_post: bool = True,
        scale_factor: float = 1.0,
        num_gs_per_anchor: int = 1,
        ckpt_path=None,
    ):
        super().__init__()

        self.num_latents = num_latents
        self.embed_dim = embed_dim
        self.scale_factor = scale_factor  # kept for config/checkpoint compat; not used in forward
        self.latent_shape = (num_latents, embed_dim)
        self.num_gs_per_anchor = num_gs_per_anchor

        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)

        self.encoder = PointCrossAttentionEncoder(
            fourier_embedder=self.fourier_embedder,
            num_latents=num_latents,
            downsample_ratio=downsample_ratio,
            pc_size=pc_size,
            pc_sharpedge_size=pc_sharpedge_size,
            point_feats=point_feats,
            width=width,
            heads=heads,
            layers=num_encoder_layers,
            qkv_bias=qkv_bias,
            use_ln_post=use_ln_post,
            qk_norm=qk_norm,
        )

        # Deterministic bottleneck (no KL, no sampling)
        self.bottleneck_down = nn.Linear(width, embed_dim)
        self.bottleneck_up = nn.Linear(embed_dim, width)

        self.transformer = Transformer(
            n_ctx=num_latents,
            width=width,
            layers=num_decoder_layers,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            drop_path_rate=drop_path_rate,
        )

        # 3DGS parameter head: K * 14 outputs per latent token.
        # Per Gaussian: 3 (pos delta) + 3 (log-scale) + 4 (quaternion) + 1 (opacity) + 3 (RGB) = 14
        K = self.num_gs_per_anchor
        self.gs_head = nn.Linear(width, K * 14)

        # Initialise GS head so opacities start near 0.6 and scales start small.
        nn.init.zeros_(self.gs_head.weight)
        nn.init.zeros_(self.gs_head.bias)
        for k in range(K):
            off = k * 14
            # Slight positive bias on opacity logit → sigmoid ≈ 0.6 at init
            self.gs_head.bias.data[off + 10] = 0.4
            # Negative bias on log-scale → small Gaussians at init
            self.gs_head.bias.data[off + 3 : off + 6] = -3.0
            # Identity quaternion (w=1, x=y=z=0) to avoid zero-norm rotations at init.
            self.gs_head.bias.data[off + 6] = 1.0

        if ckpt_path is not None:
            self._init_from_ckpt(ckpt_path)

    def _init_from_ckpt(self, path, ignore_keys=()):
        state_dict = torch.load(path, map_location='cpu')
        state_dict = state_dict.get('state_dict', state_dict)
        for k in list(state_dict.keys()):
            if any(k.startswith(ik) for ik in ignore_keys):
                del state_dict[k]
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        logger.info(
            f"Restored from {path} — {len(missing)} missing, {len(unexpected)} unexpected keys"
        )

    # ------------------------------------------------------------------
    # Core forward methods
    # ------------------------------------------------------------------

    def encode(self, surface: torch.FloatTensor):
        """Encode a colored surface point cloud to compact latents.

        Args:
            surface: [B, N, 9]  xyz | normals | rgb

        Returns:
            latents        : [B, num_latents, embed_dim]
            query_positions: [B, num_latents, 3]  FPS anchor XYZ coordinates
        """
        pc = surface[:, :, :3]
        feats = surface[:, :, 3:]           # normals(3) + rgb(3) = 6 channels
        latents, pc_infos = self.encoder(pc, feats)
        query_positions = pc_infos[0]       # concatenated random + sharpedge FPS queries
        latents = self.bottleneck_down(latents)
        return latents, query_positions

    def decode(
        self,
        latents: torch.FloatTensor,
        query_positions: torch.FloatTensor,
    ):
        """Decode compact latents + FPS anchors into 3DGS parameters.

        Args:
            latents        : [B, num_latents, embed_dim]
            query_positions: [B, num_latents, 3]

        Returns:
            means     : [B, num_latents * K, 3]
            scales    : [B, num_latents * K, 3]  (always positive)
            rotations : [B, num_latents * K, 4]  (unit quaternion, wxyz)
            opacities : [B, num_latents * K, 1]  (in [0, 1])
            colors    : [B, num_latents * K, 3]  (RGB in [0, 1])

        where K = num_gs_per_anchor.  For K=1 this is identical to the original.
        """
        K = self.num_gs_per_anchor
        latents = self.bottleneck_up(latents)
        latents = self.transformer(latents)
        raw = self.gs_head(latents)          # (B, L, K*14)

        B, L, _ = raw.shape
        raw = raw.view(B, L * K, 14)        # (B, L*K, 14)

        # Each anchor is repeated K times so every Gaussian is locally anchored.
        anchors = (
            query_positions                  # (B, L, 3)
            .unsqueeze(2)                    # (B, L, 1, 3)
            .expand(B, L, K, 3)             # (B, L, K, 3)
            .reshape(B, L * K, 3)           # (B, L*K, 3)
        )
        return self._parse_gaussians(raw, anchors)

    def _parse_gaussians(self, raw: torch.FloatTensor, query_positions: torch.FloatTensor):
        """Apply per-parameter activations and anchor means to FPS positions."""
        means = query_positions + raw[..., :3]
        # exp(clamp) keeps scales in (e^-5, e^2) ≈ (0.007, 7.4)
        scales = torch.exp(raw[..., 3:6].clamp(-5.0, 2.0))
        quat_raw = raw[..., 6:10]
        quat_norm = quat_raw.norm(dim=-1, keepdim=True)
        quat_identity = torch.zeros_like(quat_raw)
        quat_identity[..., 0] = 1.0
        rotations = torch.where(quat_norm > 1e-8, quat_raw / quat_norm, quat_identity)
        opacities = torch.sigmoid(raw[..., 10:11])
        colors = torch.sigmoid(raw[..., 11:14])
        return means, scales, rotations, opacities, colors

    def forward(self, surface: torch.FloatTensor):
        """Full encode → decode pass.

        Args:
            surface: [B, N, 9]

        Returns:
            (means, scales, rotations, opacities, colors)
            Each has shape [B, num_latents * num_gs_per_anchor, ...].
        """
        latents, query_positions = self.encode(surface)
        return self.decode(latents, query_positions)

    # ------------------------------------------------------------------
    # Warm-start from a ShapeVAE geometry-only checkpoint
    # ------------------------------------------------------------------

    def load_shapevae_encoder(self, shapevae_ckpt_path: str, zero_pad_rgb: bool = True):
        """Partially initialise the encoder from a ShapeVAE checkpoint.

        The geometry encoder (FourierEmbedder, cross-attention, self-attention)
        transfers directly.  The input_proj weight is zero-padded to accommodate
        the extra RGB channels (last 3 columns of the weight matrix set to 0).

        Args:
            shapevae_ckpt_path: path to the .ckpt or .safetensors file.
            zero_pad_rgb      : if True, zero-initialise the new RGB input columns
                                in encoder.input_proj; otherwise skip that weight.
        """
        ckpt = torch.load(shapevae_ckpt_path, map_location='cpu', weights_only=True)
        ckpt = ckpt.get('state_dict', ckpt)

        # Isolate encoder weights
        enc_prefix = 'encoder.'
        enc_ckpt = {k[len(enc_prefix):]: v for k, v in ckpt.items() if k.startswith(enc_prefix)}

        # Handle input_proj weight dimension mismatch (point_feats changed)
        ip_key = 'input_proj.weight'
        if ip_key in enc_ckpt and zero_pad_rgb:
            old_w = enc_ckpt[ip_key]          # (width, fourier_dim + old_point_feats)
            new_w = self.encoder.input_proj.weight.data.clone()
            cols = min(old_w.shape[1], new_w.shape[1])
            new_w[:, :cols] = old_w[:, :cols]
            enc_ckpt[ip_key] = new_w

        missing, unexpected = self.encoder.load_state_dict(enc_ckpt, strict=False)
        logger.info(
            f"Loaded ShapeVAE encoder — {len(missing)} missing, {len(unexpected)} unexpected"
        )
