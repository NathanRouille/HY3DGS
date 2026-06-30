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


import math
import os
from typing import Union, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from .attention_blocks import FourierEmbedder, Transformer, CrossAttentionDecoder, PointCrossAttentionEncoder
from .surface_extractors import MCSurfaceExtractor, SurfaceExtractors
from .volume_decoders import VanillaVolumeDecoder, FlashVDMVolumeDecoding, HierarchicalVolumeDecoding
from ...gs_renderer import pack_sh_coeffs
from ...utils import logger, synchronize_timer, smart_load_model


def _resolve_and_load_checkpoint(ckpt_path: str, use_safetensors: bool):
    """Load a checkpoint file, falling back to ``.ckpt`` when safetensors is absent."""
    if use_safetensors:
        st_path = ckpt_path.replace(".ckpt", ".safetensors")
        if os.path.exists(st_path):
            ckpt_path = st_path
        elif ckpt_path.endswith(".safetensors") and not os.path.exists(ckpt_path):
            ckpt_alt = ckpt_path.replace(".safetensors", ".ckpt")
            if os.path.exists(ckpt_alt):
                logger.info("Safetensors missing, falling back to %s", ckpt_alt)
                ckpt_path = ckpt_alt
                use_safetensors = False

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model file {ckpt_path} not found")

    logger.info(f"Loading model from {ckpt_path}")
    if use_safetensors:
        import safetensors.torch
        ckpt = safetensors.torch.load_file(ckpt_path, device="cpu")
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    return ckpt


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
        ckpt = _resolve_and_load_checkpoint(ckpt_path, use_safetensors)

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

    Output: (means, scales, rotations, opacities, sh_coeffs) where ``sh_coeffs``
    has shape [B, num_latents * K, (sh_degree+1)^2, 3] for gsplat SH rendering.
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
        ckpt = _resolve_and_load_checkpoint(ckpt_path, use_safetensors)

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
        deterministic_encoder: bool = True,
        sh_degree: int = 1,
        max_anchor_delta: Optional[float] = None,
        ckpt_path=None,
    ):
        super().__init__()

        self.num_latents = num_latents
        self.embed_dim = embed_dim
        self.point_feats = point_feats
        self.scale_factor = scale_factor  # kept for config/checkpoint compat; not used in forward
        self.latent_shape = (num_latents, embed_dim)
        self.num_gs_per_anchor = num_gs_per_anchor
        self.max_anchor_delta = max_anchor_delta
        self.sh_degree = int(sh_degree)
        if self.sh_degree < 0:
            raise ValueError(f"sh_degree must be >= 0, got {sh_degree}")
        self.num_sh_bases = (self.sh_degree + 1) ** 2

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
            deterministic=deterministic_encoder,
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

        # 3DGS parameter head: K_g * D outputs per latent token.
        # Per Gaussian: 11 geometry + 3*num_sh_bases appearance (SH coeffs).
        #   geometry: 3 (pos delta) + 3 (log-scale) + 4 (quat) + 1 (opacity)
        #   appearance: 3 DC (sigmoid RGB) + (num_sh_bases-1)*3 higher-order SH (raw, init 0)
        K = self.num_gs_per_anchor
        self._geom_dim = 11
        self._appearance_dim = 3 * self.num_sh_bases
        self._raw_dim = self._geom_dim + self._appearance_dim
        self.gs_head = nn.Linear(width, K * self._raw_dim)

        # Initialise GS head so opacities start near 0.6 and scales start small.
        nn.init.zeros_(self.gs_head.weight)
        nn.init.zeros_(self.gs_head.bias)
        for k in range(K):
            off = k * self._raw_dim
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
        return_features: bool = False,
    ):
        """Decode compact latents + FPS anchors into 3DGS parameters.

        Args:
            latents         : [B, num_latents, embed_dim]
            query_positions : [B, num_latents, 3]
            return_features : if True, also return the post-transformer features
                              (B, num_latents, width) for diagnostics (PCA → RGB).

        Returns:
            means     : [B, num_latents * K, 3]
            scales    : [B, num_latents * K, 3]  (always positive)
            rotations : [B, num_latents * K, 4]  (unit quaternion, wxyz)
            opacities : [B, num_latents * K, 1]  (in [0, 1])
            sh_coeffs : [B, num_latents * K, num_sh_bases, 3]  (SH appearance)
            (optionally) features : [B, num_latents, width] before the GS head

        where K = num_gs_per_anchor.
        """
        K = self.num_gs_per_anchor
        latents = self.bottleneck_up(latents)
        features = self.transformer(latents)         # (B, L, width)
        raw = self.gs_head(features)                 # (B, L, K * raw_dim)

        B, L, _ = raw.shape
        raw = raw.view(B, L * K, self._raw_dim)    # (B, L*K, D)

        # Each anchor is repeated K times so every Gaussian is locally anchored.
        anchors = (
            query_positions                  # (B, L, 3)
            .unsqueeze(2)                    # (B, L, 1, 3)
            .expand(B, L, K, 3)             # (B, L, K, 3)
            .reshape(B, L * K, 3)           # (B, L*K, 3)
        )
        gaussians = self._parse_gaussians(raw, anchors)
        if return_features:
            return (*gaussians, features)
        return gaussians

    def _parse_gaussians(self, raw: torch.FloatTensor, query_positions: torch.FloatTensor):
        """Apply per-parameter activations and anchor means to FPS positions."""
        raw_delta = raw[..., :3]
        if self.max_anchor_delta is not None:
            # AnchorSplat constrains offsets to a small local range (e.g. 10/128).
            pos_delta = torch.tanh(raw_delta) * self.max_anchor_delta
        else:
            pos_delta = raw_delta
        means = query_positions + pos_delta
        # exp(clamp) keeps scales in (e^-5, e^2) ≈ (0.007, 7.4)
        scales = torch.exp(raw[..., 3:6].clamp(-5.0, 2.0))
        quat_raw = raw[..., 6:10]
        quat_norm = quat_raw.norm(dim=-1, keepdim=True)
        quat_identity = torch.zeros_like(quat_raw)
        quat_identity[..., 0] = 1.0
        rotations = torch.where(quat_norm > 1e-8, quat_raw / quat_norm, quat_identity)
        opacities = torch.sigmoid(raw[..., 10:11])
        dc_rgb = torch.sigmoid(raw[..., 11:14])
        if self.sh_degree == 0:
            sh_coeffs = pack_sh_coeffs(dc_rgb, None, sh_degree=0)
        else:
            sh_rest = raw[..., 14 : 14 + (self.num_sh_bases - 1) * 3]
            sh_coeffs = pack_sh_coeffs(dc_rgb, sh_rest, sh_degree=self.sh_degree)
        return means, scales, rotations, opacities, sh_coeffs

    def forward(self, surface: torch.FloatTensor):
        """Full encode → decode pass.

        Args:
            surface: [B, N, 9]

        Returns:
            (means, scales, rotations, opacities, sh_coeffs)
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

    @staticmethod
    def _map_input_proj_vae_to_gsae(
        old_w: torch.Tensor,
        new_w: torch.Tensor,
        *,
        old_point_feats: int,
        new_point_feats: int,
        rgb_feat_init: str,
    ) -> torch.Tensor:
        """Copy Hunyuan geometry+label columns; initialise RGB feature columns."""
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

    def load_shapevae_pretrained(
        self,
        ckpt_or_repo: str,
        *,
        load_encoder: bool = True,
        load_bottleneck: bool = True,
        load_transformer: bool = True,
        rgb_feat_init: str = "kaiming",
        subfolder: str = "hunyuan3d-vae-v2-mini-withencoder",
        shapevae_point_feats: int = 4,
        use_safetensors: bool = False,
    ) -> dict:
        """Load Hunyuan ShapeVAE weights into ShapeGSAE (geometry path).

        ``input_proj``: copy Fourier + normals + label columns; RGB columns are
        initialised per ``rgb_feat_init``.  ``pre_kl`` mean half maps to
        ``bottleneck_down``; ``post_kl`` maps to ``bottleneck_up``.
        """
        ckpt = self._load_shapevae_state_dict(
            ckpt_or_repo,
            subfolder=subfolder,
            use_safetensors=use_safetensors,
        )
        report = {
            "source": ckpt_or_repo,
            "load_encoder": load_encoder,
            "load_bottleneck": load_bottleneck,
            "load_transformer": load_transformer,
            "rgb_feat_init": rgb_feat_init,
            "encoder_missing": 0,
            "encoder_unexpected": 0,
        }

        if load_encoder:
            enc_prefix = "encoder."
            enc_ckpt = {
                k[len(enc_prefix) :]: v for k, v in ckpt.items() if k.startswith(enc_prefix)
            }
            ip_key = "input_proj.weight"
            if ip_key in enc_ckpt:
                enc_ckpt[ip_key] = self._map_input_proj_vae_to_gsae(
                    enc_ckpt[ip_key],
                    self.encoder.input_proj.weight.data,
                    old_point_feats=shapevae_point_feats,
                    new_point_feats=self.point_feats,
                    rgb_feat_init=rgb_feat_init,
                )
            missing, unexpected = self.encoder.load_state_dict(enc_ckpt, strict=False)
            report["encoder_missing"] = len(missing)
            report["encoder_unexpected"] = len(unexpected)
            logger.info(
                "Loaded pretrained encoder — %d missing, %d unexpected keys",
                len(missing),
                len(unexpected),
            )

        if load_bottleneck:
            if "pre_kl.weight" in ckpt:
                self.bottleneck_down.weight.data.copy_(ckpt["pre_kl.weight"][: self.embed_dim])
            if "pre_kl.bias" in ckpt:
                self.bottleneck_down.bias.data.copy_(ckpt["pre_kl.bias"][: self.embed_dim])
            post_keys = {k: v for k, v in ckpt.items() if k.startswith("post_kl.")}
            if post_keys:
                post_ckpt = {k[len("post_kl.") :]: v for k, v in post_keys.items()}
                missing, unexpected = self.bottleneck_up.load_state_dict(post_ckpt, strict=False)
                logger.info(
                    "Loaded pretrained bottleneck — down from pre_kl mean, up from post_kl "
                    "(%d missing, %d unexpected)",
                    len(missing),
                    len(unexpected),
                )

        if load_transformer:
            tr_ckpt = {
                k[len("transformer.") :]: v
                for k, v in ckpt.items()
                if k.startswith("transformer.")
            }
            if tr_ckpt:
                missing, unexpected = self.transformer.load_state_dict(tr_ckpt, strict=False)
                report["transformer_missing"] = len(missing)
                report["transformer_unexpected"] = len(unexpected)
                logger.info(
                    "Loaded pretrained transformer — %d missing, %d unexpected keys",
                    len(missing),
                    len(unexpected),
                )

        return report

    def freeze_modules(
        self,
        *,
        encoder: bool = False,
        bottleneck: bool = False,
        transformer: bool = False,
    ) -> None:
        """Freeze selected modules for Phase-1-style training."""
        if encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if bottleneck:
            for param in self.bottleneck_down.parameters():
                param.requires_grad = False
            for param in self.bottleneck_up.parameters():
                param.requires_grad = False
        if transformer:
            for param in self.transformer.parameters():
                param.requires_grad = False

    def build_optimizer_param_groups(
        self,
        lr: float,
        weight_decay: float,
        *,
        encoder_lr_scale: float = 1.0,
        transformer_lr_scale: float = 1.0,
        bottleneck_lr_scale: Optional[float] = None,
    ) -> List[dict]:
        """Build AdamW param groups with per-module LR scales."""
        if bottleneck_lr_scale is None:
            bottleneck_lr_scale = encoder_lr_scale

        def _collect(module: nn.Module) -> List[nn.Parameter]:
            return [p for p in module.parameters() if p.requires_grad]

        groups: List[dict] = []
        module_specs = (
            (self.gs_head, 1.0),
            (self.encoder, encoder_lr_scale),
            (self.bottleneck_down, bottleneck_lr_scale),
            (self.bottleneck_up, bottleneck_lr_scale),
            (self.transformer, transformer_lr_scale),
        )
        for module, scale in module_specs:
            params = _collect(module)
            if params:
                groups.append(
                    {"params": params, "lr": lr * scale, "weight_decay": weight_decay}
                )
        return groups
