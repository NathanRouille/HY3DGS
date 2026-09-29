"""VGGT weak-context builder: image features + camera-frame 3D PE from VGGT depth."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models.autoencoders.attention_blocks import FourierEmbedder

logger = logging.getLogger(__name__)

VGGT_LAYERS = (4, 11, 17, 23)
VGGT_IMG_SIZE = 518


def load_state_dict_skip_mismatch(
    module: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    *,
    log_prefix: str = "",
) -> Tuple[List[str], List[str], List[str]]:
    """Like load_state_dict(strict=False), but also skip size-mismatched keys."""
    model_sd = module.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []
    for k, v in state_dict.items():
        if k not in model_sd:
            continue
        if model_sd[k].shape != v.shape:
            skipped.append(k)
            continue
        filtered[k] = v
    missing, unexpected = module.load_state_dict(filtered, strict=False)
    if skipped:
        logger.warning(
            "%sSkipping %d size-mismatched keys: %s",
            log_prefix,
            len(skipped),
            skipped,
        )
    return list(missing), list(unexpected), skipped


def world_to_camera(
    points_world: np.ndarray,
    c2w: np.ndarray,
) -> np.ndarray:
    """Map world/object points into the G-Objaverse camera frame.

    Matches the convention verified against Unity ``c2w`` + pinhole unprojection
    (OpenCV-like: x right, y down, z forward). Inverse of
    ``world = cam @ R.T + t``.
    """
    pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    rot = np.asarray(c2w, dtype=np.float64)[:3, :3]
    origin = np.asarray(c2w, dtype=np.float64)[:3, 3]
    cam = (pts - origin[None, :]) @ rot
    return cam.reshape(points_world.shape).astype(np.float32)


def world_to_camera_torch(
    points_world: torch.Tensor,
    c2w: torch.Tensor,
) -> torch.Tensor:
    """Batched world→camera. points [B,N,3] or [N,3], c2w [B,4,4] or [4,4]."""
    if points_world.dim() == 2:
        points_world = points_world.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    if c2w.dim() == 2:
        c2w = c2w.unsqueeze(0)
    rot = c2w[:, :3, :3].to(dtype=points_world.dtype)
    origin = c2w[:, :3, 3].to(dtype=points_world.dtype)
    cam = torch.bmm(points_world - origin[:, None, :], rot)
    return cam.squeeze(0) if squeeze else cam


def depth_map_to_cam_points(
    depth: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Pinhole unprojection (OpenCV): z = depth. Returns [H,W,3]."""
    h, w = depth.shape
    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    z = depth.astype(np.float32)
    x = (uu - cx) * z / max(fx, 1e-6)
    y = (vv - cy) * z / max(fy, 1e-6)
    return np.stack([x, y, z], axis=-1)


def patch_centers_from_depth(
    cam_pts: np.ndarray,
    pixel_valid: np.ndarray,
    *,
    patch_size: int = 14,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    default_z: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-patch 3D position: center pixel if valid, else mean of valid pixels.

    Discarded (fully-background) patches still get a **real** camera-frame
    position for debugging / visualization:
      1) raw centre pixel if ``z > 0``,
      2) else mean of any positive-z pixels in the patch,
      3) else pinhole ray through the patch centre at ``default_z`` (needs K).

    Returns:
        centers: [Np, 3]
        keep: [Np] bool — False means fully-background (discard token)
    """
    h, w, _ = cam_pts.shape
    gh, gw = h // patch_size, w // patch_size
    if gh == 0 or gw == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=bool)

    centers = np.zeros((gh * gw, 3), dtype=np.float32)
    keep = np.zeros((gh * gw), dtype=bool)
    half = patch_size // 2
    idx = 0
    for iy in range(gh):
        for ix in range(gw):
            y0, x0 = iy * patch_size, ix * patch_size
            py, px = y0 + half, x0 + half
            if pixel_valid[py, px]:
                centers[idx] = cam_pts[py, px]
                keep[idx] = True
            else:
                block = cam_pts[y0 : y0 + patch_size, x0 : x0 + patch_size]
                m = pixel_valid[y0 : y0 + patch_size, x0 : x0 + patch_size]
                if m.any():
                    centers[idx] = block[m].mean(axis=0)
                    keep[idx] = True
                else:
                    keep[idx] = False
                    z_c = float(cam_pts[py, px, 2])
                    if np.isfinite(z_c) and z_c > 1e-6 and np.isfinite(
                        cam_pts[py, px]
                    ).all():
                        centers[idx] = cam_pts[py, px]
                    else:
                        zmask = np.isfinite(block[..., 2]) & (block[..., 2] > 1e-6)
                        if zmask.any():
                            centers[idx] = block[zmask].mean(axis=0)
                        elif fx is not None and fy is not None and cx is not None and cy is not None:
                            z = float(default_z)
                            centers[idx, 0] = (float(px) - float(cx)) * z / max(float(fx), 1e-6)
                            centers[idx, 1] = (float(py) - float(cy)) * z / max(float(fy), 1e-6)
                            centers[idx, 2] = z
                        else:
                            centers[idx] = (0.0, 0.0, float(default_z))
            idx += 1
    return centers, keep


def preprocess_rgb_for_vggt(
    rgb: torch.Tensor,
    *,
    target_size: int = VGGT_IMG_SIZE,
) -> Tuple[torch.Tensor, float, float, int, int]:
    """Resize/pad RGB [B,3,H,W] in [0,1] to VGGT input size (divisible by 14)."""
    _, _, h, w = rgb.shape
    if w >= h:
        new_w = target_size
        new_h = max(round(h * (new_w / w) / 14) * 14, 14)
    else:
        new_h = target_size
        new_w = max(round(w * (new_h / h) / 14) * 14, 14)
    scale_y = new_h / float(h)
    scale_x = new_w / float(w)
    images = F.interpolate(rgb, size=(new_h, new_w), mode="bilinear", align_corners=False)
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    if pad_h > 0 or pad_w > 0:
        images = F.pad(images, (pad_left, pad_right, pad_top, pad_bottom), value=1.0)
    return images, scale_y, scale_x, pad_top, pad_left


class Fourier3DPosEmbed(nn.Module):
    """Fourier features on 3D coordinates → Linear to ``out_dim``.

    Pass a shared :class:`FourierEmbedder` (e.g. ShapePCAE's) so surface points
    and VGGT patch centres use the same frequency basis on the same frame.
    The shared embedder is **not** registered as a submodule (would re-parent
    it away from ShapePCAE); only ``proj`` lives in this module's state_dict.
    """

    def __init__(
        self,
        out_dim: int,
        *,
        fourier: Optional[FourierEmbedder] = None,
        num_freqs: int = 8,
        include_pi: bool = True,
    ):
        super().__init__()
        self._shared_fourier = fourier is not None
        if fourier is None:
            self._own_fourier = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)
            object.__setattr__(self, "_fourier_shared", None)
            fourier_dim = self._own_fourier.out_dim
        else:
            # Do NOT assign to self.fourier — that registers a submodule and steals
            # ownership from ShapePCAE.
            self._own_fourier = None
            object.__setattr__(self, "_fourier_shared", fourier)
            fourier_dim = fourier.out_dim
        self.proj = nn.Linear(fourier_dim, out_dim)

    def attach_fourier(self, fourier: FourierEmbedder) -> None:
        """Point PE at the model's FourierEmbedder (same freqs / same frame)."""
        self._shared_fourier = True
        self._own_fourier = None
        object.__setattr__(self, "_fourier_shared", fourier)
        if self.proj.in_features != fourier.out_dim:
            device = self.proj.weight.device
            dtype = self.proj.weight.dtype
            self.proj = nn.Linear(fourier.out_dim, self.proj.out_features).to(
                device=device, dtype=dtype
            )

    def _fourier(self) -> FourierEmbedder:
        if self._shared_fourier:
            return getattr(self, "_fourier_shared")
        assert self._own_fourier is not None
        return self._own_fourier

    @property
    def out_dim(self) -> int:
        return self.proj.out_features

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        flat = xyz.reshape(-1, 3)
        emb = self._fourier()(flat)
        return self.proj(emb).view(*xyz.shape[:2], -1)


class CachedVGGTContextStore:
    """Load/save precomputed weak context tensors."""

    def __init__(self, cache_root: Union[str, Path]):
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def path_for(self, mesh_path: str, view_idx: int) -> Path:
        stem = Path(mesh_path).stem
        return self.cache_root / f"{stem}_view{view_idx:05d}.pt"

    def has(self, mesh_path: str, view_idx: int) -> bool:
        return self.path_for(mesh_path, view_idx).exists()

    def save(self, mesh_path: str, view_idx: int, payload: Dict) -> Path:
        p = self.path_for(mesh_path, view_idx)
        torch.save(payload, p)
        return p

    def load(self, mesh_path: str, view_idx: int) -> Optional[Dict]:
        p = self.path_for(mesh_path, view_idx)
        if not p.exists():
            return None
        return torch.load(p, map_location="cpu", weights_only=False)

    def joint_tag(self, view_indices: Sequence[int]) -> str:
        """Filename tag for a fixed ordered view tuple (view 0 = VGGT reference)."""
        return "_".join(f"{int(v):05d}" for v in view_indices)

    def joint_path_for(self, mesh_path: str, view_indices: Sequence[int]) -> Path:
        stem = Path(mesh_path).stem
        return self.cache_root / f"{stem}_joint_{self.joint_tag(view_indices)}.pt"

    def has_joint(self, mesh_path: str, view_indices: Sequence[int]) -> bool:
        return self.joint_path_for(mesh_path, view_indices).exists()

    def save_joint(
        self, mesh_path: str, view_indices: Sequence[int], payload: Dict
    ) -> Path:
        p = self.joint_path_for(mesh_path, view_indices)
        torch.save(payload, p)
        return p

    def load_joint(
        self, mesh_path: str, view_indices: Sequence[int]
    ) -> Optional[Dict]:
        p = self.joint_path_for(mesh_path, view_indices)
        if not p.exists():
            return None
        return torch.load(p, map_location="cpu", weights_only=False)


class VGGTContextBuilder(nn.Module):
    """Weak context from frozen VGGT (features + camera-frame PE from VGGT depth).

    Pipeline:
      aggregator layers {4,11,17,23} → channel-concat → feat_proj
      + depth_head → unproject to camera xyz → per-patch centre (or mean)
      → drop fully-background patches → Fourier PE (shared with ShapePCAE)

    Background detection uses VGGT ``depth_conf`` (and optional white-bg mask).
    Yes — patches with no confident depth pixels are discarded.
    """

    def __init__(
        self,
        *,
        width: int = 1024,
        vggt_feat_dim: int = 2048,
        patch_size: int = 14,
        layers: Tuple[int, ...] = VGGT_LAYERS,
        img_size: int = VGGT_IMG_SIZE,
        conf_percentile: float = 20.0,
        min_conf: float = 0.05,
        fourier: Optional[FourierEmbedder] = None,
        mask_white_bg: bool = True,
    ):
        super().__init__()
        self.width = width
        self.patch_size = patch_size
        self.layers = layers
        self.num_layer_concat = len(layers)
        self.vggt_feat_dim = vggt_feat_dim
        self.img_size = img_size
        self.pe_frame = "camera"  # fixed: PE always in camera coordinates
        self.conf_percentile = float(conf_percentile)
        self.min_conf = float(min_conf)
        self.mask_white_bg = bool(mask_white_bg)

        self.feat_proj = nn.Linear(vggt_feat_dim * self.num_layer_concat, width)
        self.cam_proj = nn.Linear(vggt_feat_dim, width)
        self.pos_embed = Fourier3DPosEmbed(width, fourier=fourier)
        self._vggt_holder: List[nn.Module] = []

        self.fallback_stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.Conv2d(64, width, kernel_size=1),
        )

    def attach_fourier(self, fourier: FourierEmbedder) -> None:
        self.pos_embed.attach_fourier(fourier)

    @property
    def vggt(self) -> Optional[nn.Module]:
        return self._vggt_holder[0] if self._vggt_holder else None

    def train(self, mode: bool = True):
        super().train(mode)
        vggt = self.vggt
        if vggt is not None:
            vggt.eval()
        return self

    def _load_vggt(self):
        if self.vggt is not None:
            return self.vggt
        try:
            from vggt.models.vggt import VGGT  # type: ignore
        except ImportError:
            logger.warning("VGGT package not found; using fallback patch encoder.")
            return None
        model = VGGT.from_pretrained("facebook/VGGT-1B")
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        self._vggt_holder.append(model)
        logger.info("Loaded frozen VGGT-1B (aggregator + depth/camera/point heads).")
        return model

    def _pixel_valid_mask(
        self,
        depth: np.ndarray,
        conf: np.ndarray,
        rgb_np: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """True where VGGT depth is usable (not background)."""
        valid = depth > 1e-6
        conf_f = conf.astype(np.float64)
        if conf_f.size == 0:
            return valid
        thr = max(
            self.min_conf,
            float(np.percentile(conf_f[valid], self.conf_percentile)) if valid.any() else self.min_conf,
        )
        valid = valid & (conf > thr)
        if self.mask_white_bg and rgb_np is not None:
            # G-Objaverse composites on white; VGGT demo does the same filter.
            white = (
                (rgb_np[..., 0] > 0.97)
                & (rgb_np[..., 1] > 0.97)
                & (rgb_np[..., 2] > 0.97)
            )
            valid = valid & ~white
        return valid

    def _camera_tokens_from_aggregator(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        *,
        num_views: int,
    ) -> torch.Tensor:
        """Camera tokens from the **last** VGGT layer, one per view.

        Returns ``[B, S, C]`` with ``S=num_views`` (view 0 = reference / cam0).
        """
        layer_idx = self.layers[-1]
        cams: List[torch.Tensor] = []
        for s in range(int(num_views)):
            tok = aggregated_tokens_list[layer_idx][:, s].float()
            cams.append(tok[:, :1, :])
        return torch.cat(cams, dim=1)

    def _fallback_raw(self, rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        b = rgb.shape[0]
        feat = self.fallback_stem(rgb).flatten(2).transpose(1, 2)
        cam = feat.mean(dim=1, keepdim=True)
        n = feat.shape[1]
        return {
            "patch_tokens": feat,
            "camera_token": cam,
            "patch_centers": torch.zeros(b, n, 3, device=rgb.device, dtype=rgb.dtype),
            "patch_keep": torch.ones(b, n, device=rgb.device, dtype=torch.bool),
            "vggt_cam_points": None,
            "vggt_depth": None,
            "vggt_depth_conf": None,
        }

    @torch.no_grad()
    def extract_vggt_raw(
        self,
        rgb: torch.Tensor,
        depth: Optional[torch.Tensor] = None,  # unused (kept for API compat)
        intrinsics: Optional[torch.Tensor] = None,
        c2w: Optional[torch.Tensor] = None,
        *,
        return_dense: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """VGGT features + camera-frame patch centres from **VGGT depth**.

        GT depth / c2w are not used for PE (image-only path). Optional dense
        maps are returned when ``return_dense=True`` (debug PLYs).
        """
        del depth, intrinsics, c2w  # PE must not depend on GT geometry
        vggt = self._load_vggt()
        if vggt is None:
            return self._fallback_raw(rgb)

        device = rgb.device
        dtype = rgb.dtype
        vggt = vggt.to(device)
        images, _, _, _, _ = preprocess_rgb_for_vggt(
            rgb.float().clamp(0, 1), target_size=self.img_size
        )
        images = images.to(device=device)
        images_s = images.unsqueeze(1)  # [B,1,3,H,W]
        _, _, _, H, W = images_s.shape

        amp_dtype = torch.bfloat16
        amp_enabled = device.type == "cuda"
        if amp_enabled:
            major = torch.cuda.get_device_capability(device)[0]
            if major < 8:
                amp_dtype = torch.float16

        from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # type: ignore

        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
            aggregated_tokens_list, patch_start_idx = vggt.aggregator(images_s)
            depth_pred, depth_conf = vggt.depth_head(
                aggregated_tokens_list, images=images_s, patch_start_idx=patch_start_idx
            )
            pose_enc_list = vggt.camera_head(aggregated_tokens_list)
            pose_enc = pose_enc_list[-1]
            world_pts = world_conf = None
            if return_dense and vggt.point_head is not None:
                world_pts, world_conf = vggt.point_head(
                    aggregated_tokens_list,
                    images=images_s,
                    patch_start_idx=patch_start_idx,
                )

        # depth_pred: [B,S,H,W,1], depth_conf: [B,S,H,W]
        depth_b = depth_pred[:, 0, ..., 0].float()
        conf_b = depth_conf[:, 0].float()
        extrinsics, intrins = pose_encoding_to_extri_intri(
            pose_enc, image_size_hw=(H, W)
        )
        # intrins: [B,S,3,3]
        K = intrins[:, 0].float()

        patch_layers: List[torch.Tensor] = []
        cam_token = None
        for layer_idx in self.layers:
            tok = aggregated_tokens_list[layer_idx][:, 0].float()
            if cam_token is None:
                cam_token = tok[:, :1, :]
            patch_layers.append(tok[:, patch_start_idx:, :])
        patch_tokens_full = torch.cat(patch_layers, dim=-1)
        assert cam_token is not None

        b = rgb.shape[0]
        rgb_np = images[:, :3].permute(0, 2, 3, 1).float().cpu().numpy()

        kept_tokens: List[torch.Tensor] = []
        kept_centers: List[torch.Tensor] = []
        keep_masks: List[torch.Tensor] = []
        dense_cam: List[Optional[torch.Tensor]] = []

        for i in range(b):
            d = depth_b[i].cpu().numpy()
            c = conf_b[i].cpu().numpy()
            fx, fy = float(K[i, 0, 0]), float(K[i, 1, 1])
            cx, cy = float(K[i, 0, 2]), float(K[i, 1, 2])
            cam_pts = depth_map_to_cam_points(d, fx=fx, fy=fy, cx=cx, cy=cy)
            pix_valid = self._pixel_valid_mask(d, c, rgb_np=rgb_np[i])
            centers, keep = patch_centers_from_depth(
                cam_pts,
                pix_valid,
                patch_size=self.patch_size,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
            )
            n_full = patch_tokens_full.shape[1]
            if centers.shape[0] != n_full:
                # Resolution mismatch — truncate/pad keep mask
                if centers.shape[0] > n_full:
                    centers, keep = centers[:n_full], keep[:n_full]
                else:
                    pad = n_full - centers.shape[0]
                    centers = np.pad(centers, ((0, pad), (0, 0)))
                    keep = np.pad(keep, (0, pad), constant_values=False)

            keep_t = torch.from_numpy(keep)
            cen_t = torch.from_numpy(centers)
            tok_i = patch_tokens_full[i]
            if keep_t.any():
                kept_tokens.append(tok_i[keep_t])
                kept_centers.append(cen_t[keep_t])
            else:
                # Degenerate: keep a single zero token so shapes stay valid
                logger.warning("All VGGT patches discarded as background for sample %d", i)
                kept_tokens.append(tok_i[:1] * 0)
                kept_centers.append(torch.zeros(1, 3))
                keep_t = torch.zeros(n_full, dtype=torch.bool)
                keep_t[0] = True
            keep_masks.append(keep_t)
            if return_dense:
                dense_cam.append(
                    torch.from_numpy(cam_pts.astype(np.float32)).to(device=device)
                )
            else:
                dense_cam.append(None)

        # Pad variable-length kept tokens to max in batch
        max_n = max(t.shape[0] for t in kept_tokens)
        tok_pad = patch_tokens_full.new_zeros(b, max_n, patch_tokens_full.shape[-1])
        cen_pad = patch_tokens_full.new_zeros(b, max_n, 3)
        keep_pad = torch.zeros(b, max_n, dtype=torch.bool, device=device)
        for i, (t, c) in enumerate(zip(kept_tokens, kept_centers)):
            n = t.shape[0]
            tok_pad[i, :n] = t.to(device=device)
            cen_pad[i, :n] = c.to(device=device, dtype=tok_pad.dtype)
            keep_pad[i, :n] = True

        out: Dict[str, torch.Tensor] = {
            "patch_tokens": tok_pad.to(dtype=dtype),
            "camera_token": cam_token.to(dtype=dtype),
            "patch_centers": cen_pad.to(dtype=dtype),
            "patch_keep": keep_pad,
            # Full-grid keep before discard (for diagnostics)
            "patch_keep_full": torch.stack(keep_masks).to(device=device),
        }
        if return_dense:
            out["vggt_cam_points"] = torch.stack(
                [x if x is not None else torch.zeros(H, W, 3, device=device) for x in dense_cam]
            )
            out["vggt_depth"] = depth_b.to(device=device)
            out["vggt_depth_conf"] = conf_b.to(device=device)
            if world_pts is not None:
                out["vggt_world_points"] = world_pts[:, 0].float().to(device=device)
                out["vggt_world_conf"] = world_conf[:, 0].float().to(device=device)
            out["vggt_extrinsics"] = extrinsics[:, 0].float().to(device=device)
            out["vggt_intrinsics"] = K.to(device=device)
        return out

    @torch.no_grad()
    def extract_vggt_sequence_dense(
        self,
        rgb_views: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Multi-view VGGT dense geometry in the **first-camera / world** frame.

        Args:
            rgb_views: ``[S,3,H,W]`` or ``[1,S,3,H,W]`` RGBs (view 0 = reference).

        Returns (no batch dim on sequence unless noted):
            depth ``[S,H,W]``, depth_conf ``[S,H,W]``,
            extrinsics ``[S,3,4]`` (cam←world), intrinsics ``[S,3,3]``,
            world_points_from_depth ``[S,H,W,3]`` (via VGGT unproject),
            plus view-0 ``vggt_cam_points`` / tokens fields matching
            :meth:`extract_vggt_raw` when ``S==1``.
        """
        if rgb_views.dim() == 5:
            assert rgb_views.shape[0] == 1, "batch>1 not supported in sequence helper"
            rgb_views = rgb_views[0]
        assert rgb_views.dim() == 4 and rgb_views.shape[1] == 3, rgb_views.shape
        S = int(rgb_views.shape[0])
        device = rgb_views.device

        # S==1: reuse the exact single-view path (bit-identical dense fields).
        if S == 1:
            raw = self.extract_vggt_raw(rgb_views, return_dense=True)
            d = raw["vggt_depth"][0]
            c = raw["vggt_depth_conf"][0]
            E = raw["vggt_extrinsics"][0]
            K = raw["vggt_intrinsics"][0]
            cam0 = raw["vggt_cam_points"][0]
            # Full-grid centres → kept / discarded (same recipe as multi-view).
            fx0 = float(K[0, 0])
            fy0 = float(K[1, 1])
            cx0 = float(K[0, 2])
            cy0 = float(K[1, 2])
            rgb0 = (
                preprocess_rgb_for_vggt(
                    rgb_views[0:1].float().clamp(0, 1), target_size=self.img_size
                )[0][0]
                .permute(1, 2, 0)
                .float()
                .cpu()
                .numpy()
            )
            pix0 = self._pixel_valid_mask(
                d.cpu().numpy(), c.cpu().numpy(), rgb_np=rgb0
            )
            full_c, full_keep = patch_centers_from_depth(
                cam0.cpu().numpy(),
                pix0,
                patch_size=self.patch_size,
                fx=fx0,
                fy=fy0,
                cx=cx0,
                cy=cy0,
            )
            keep_b = full_keep.astype(bool)
            disc = full_c[~keep_b] if (~keep_b).any() else np.zeros((0, 3), np.float32)
            dense_fg = (
                cam0.cpu().numpy()[pix0].astype(np.float32)
                if pix0.any()
                else np.zeros((0, 3), np.float32)
            )
            return {
                **{k: v for k, v in raw.items()},
                "patch_centers_discarded": torch.from_numpy(disc.astype(np.float32))
                .unsqueeze(0)
                .to(device=device, dtype=rgb_views.dtype),
                "vggt_points_cam0": torch.from_numpy(dense_fg).to(
                    device=device, dtype=rgb_views.dtype
                ),
                "num_views": torch.tensor(1),
                "vggt_depth_seq": d.unsqueeze(0),
                "vggt_depth_conf_seq": c.unsqueeze(0),
                "vggt_extrinsics_seq": E.unsqueeze(0),
                "vggt_intrinsics_seq": K.unsqueeze(0),
                "vggt_world_points_from_depth": cam0.unsqueeze(0),
            }

        vggt = self._load_vggt()
        if vggt is None:
            raise RuntimeError("VGGT not available for multi-view extract")

        from vggt.utils.geometry import unproject_depth_map_to_point_map  # type: ignore
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # type: ignore

        vggt = vggt.to(device)
        # Preprocess each view to the same padded size as single-view.
        imgs = []
        for s in range(S):
            im, _, _, _, _ = preprocess_rgb_for_vggt(
                rgb_views[s : s + 1].float().clamp(0, 1), target_size=self.img_size
            )
            imgs.append(im.to(device=device))
        images_s = torch.stack(imgs, dim=1)  # [1,S,3,H,W]
        _, _, _, H, W = images_s.shape

        amp_dtype = torch.bfloat16
        amp_enabled = device.type == "cuda"
        if amp_enabled:
            major = torch.cuda.get_device_capability(device)[0]
            if major < 8:
                amp_dtype = torch.float16

        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
            aggregated_tokens_list, patch_start_idx = vggt.aggregator(images_s)
            depth_pred, depth_conf = vggt.depth_head(
                aggregated_tokens_list, images=images_s, patch_start_idx=patch_start_idx
            )
            pose_enc = vggt.camera_head(aggregated_tokens_list)[-1]

        depth_b = depth_pred[0, :, ..., 0].float()  # [S,H,W]
        conf_b = depth_conf[0].float()
        extrinsics, intrins = pose_encoding_to_extri_intri(
            pose_enc, image_size_hw=(H, W)
        )
        E = extrinsics[0].float()  # [S,3,4]
        K = intrins[0].float()  # [S,3,3]

        world = unproject_depth_map_to_point_map(
            depth_b[..., None].cpu().numpy(),
            E.cpu().numpy(),
            K.cpu().numpy(),
        )  # [S,H,W,3]
        world_t = torch.from_numpy(world.astype(np.float32)).to(device=device)

        # View-0 camera-frame points (same as single-view unproject with K0).
        fx = float(K[0, 0, 0])
        fy = float(K[0, 1, 1])
        cx = float(K[0, 0, 2])
        cy = float(K[0, 1, 2])
        cam0 = depth_map_to_cam_points(
            depth_b[0].cpu().numpy(), fx=fx, fy=fy, cx=cx, cy=cy
        )
        cam0_t = torch.from_numpy(cam0.astype(np.float32)).to(device=device)

        # Tokens / centres from **all** views; centres live in VGGT cam0.
        # View 0 uses local depth unproject (bit-match single-view). Other views
        # map VGGT world_from_depth → cam0 via E0.
        from hy3dgen.shapegen.cam_align import vggt_world_to_cam0

        E0_np = E[0].cpu().numpy()
        kept_toks: List[torch.Tensor] = []
        kept_cens: List[torch.Tensor] = []
        disc_cens: List[torch.Tensor] = []
        dense_cam0_fg: List[np.ndarray] = []
        keep0_full = None
        for s in range(S):
            patch_layers_s: List[torch.Tensor] = []
            for layer_idx in self.layers:
                tok = aggregated_tokens_list[layer_idx][:, s].float()
                patch_layers_s.append(tok[:, patch_start_idx:, :])
            patch_full_s = torch.cat(patch_layers_s, dim=-1)[0]  # [Np, C]
            fx_s = float(K[s, 0, 0])
            fy_s = float(K[s, 1, 1])
            cx_s = float(K[s, 0, 2])
            cy_s = float(K[s, 1, 2])
            rgb_s = images_s[0, s].permute(1, 2, 0).float().cpu().numpy()
            depth_s = depth_b[s].cpu().numpy()
            conf_s = conf_b[s].cpu().numpy()
            pix_s = self._pixel_valid_mask(depth_s, conf_s, rgb_np=rgb_s)
            if s == 0:
                cen_grid = cam0
            else:
                cen_grid = vggt_world_to_cam0(
                    world[s].reshape(-1, 3), E0_np
                ).reshape(world[s].shape)
            centers_s, keep_s = patch_centers_from_depth(
                cen_grid,
                pix_s,
                patch_size=self.patch_size,
                fx=fx_s,
                fy=fy_s,
                cx=cx_s,
                cy=cy_s,
            )
            n_full = patch_full_s.shape[0]
            if centers_s.shape[0] != n_full:
                if centers_s.shape[0] > n_full:
                    centers_s, keep_s = centers_s[:n_full], keep_s[:n_full]
                else:
                    pad = n_full - centers_s.shape[0]
                    centers_s = np.pad(centers_s, ((0, pad), (0, 0)))
                    keep_s = np.pad(keep_s, (0, pad), constant_values=False)
            keep_t = torch.from_numpy(keep_s.astype(bool))
            cen_t = torch.from_numpy(centers_s.astype(np.float32))
            if s == 0:
                keep0_full = keep_t
            if keep_t.any():
                kept_toks.append(patch_full_s[keep_t])
                kept_cens.append(cen_t[keep_t])
            else:
                kept_toks.append(patch_full_s[:1] * 0)
                kept_cens.append(torch.zeros(1, 3))
            if (~keep_t).any():
                disc_cens.append(cen_t[~keep_t])
            # Dense FG cloud in cam0 (same frame as PE centres).
            if pix_s.any():
                dense_cam0_fg.append(cen_grid[pix_s].astype(np.float32))

        tok_k = torch.cat(kept_toks, dim=0)
        cen_k = torch.cat(kept_cens, dim=0)
        if disc_cens:
            cen_disc = torch.cat(disc_cens, dim=0)
        else:
            cen_disc = torch.zeros(0, 3, dtype=torch.float32)
        if dense_cam0_fg:
            dense_fg = np.concatenate(dense_cam0_fg, axis=0)
        else:
            dense_fg = np.zeros((0, 3), dtype=np.float32)
        if keep0_full is None:
            keep0_full = torch.ones(tok_k.shape[0], dtype=torch.bool)

        # Last aggregator layer (23): one camera token per view, prepended in-sequence.
        cam_token = self._camera_tokens_from_aggregator(
            aggregated_tokens_list, num_views=S
        ).to(dtype=rgb_views.dtype)

        return {
            "patch_tokens": tok_k.unsqueeze(0).to(dtype=rgb_views.dtype),
            "camera_token": cam_token.to(dtype=rgb_views.dtype),
            "patch_centers": cen_k.unsqueeze(0).to(
                device=device, dtype=rgb_views.dtype
            ),
            "patch_centers_discarded": cen_disc.unsqueeze(0).to(
                device=device, dtype=rgb_views.dtype
            ),
            "patch_keep": torch.ones(1, cen_k.shape[0], dtype=torch.bool, device=device),
            "patch_keep_full": keep0_full.unsqueeze(0).to(device=device),
            "vggt_cam_points": cam0_t.unsqueeze(0),
            "vggt_points_cam0": torch.from_numpy(dense_fg).to(
                device=device, dtype=rgb_views.dtype
            ),
            "vggt_depth": depth_b[0:1],
            "vggt_depth_conf": conf_b[0:1],
            "vggt_extrinsics": E[0:1],
            "vggt_intrinsics": K[0:1],
            "num_views": torch.tensor(S),
            "vggt_depth_seq": depth_b,
            "vggt_depth_conf_seq": conf_b,
            "vggt_extrinsics_seq": E,
            "vggt_intrinsics_seq": K,
            "vggt_world_points_from_depth": world_t,
        }

    def build_c_meanrms_from_rgb_views(
        self,
        rgb_views: torch.Tensor,
        *,
        include_camera_in_sequence: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Online multi-view weak context for ``align_mode=c_meanrms``.

        Runs VGGT on ``rgb_views`` ``[S,3,H,W]`` (view 0 = VGGT reference),
        concatenates kept patch tokens with centres in VGGT cam0, applies own
        mean+RMS, then the usual Fourier PE → ``feat_proj`` → sequence (same as
        single-view ``forward_from_features``).
        """
        from hy3dgen.shapegen.cam_align import align_patch_centers

        raw = self.extract_vggt_sequence_dense(rgb_views)
        centers = raw["patch_centers"]
        if centers.dim() == 3:
            centers = centers[0]
        centers = align_patch_centers(centers, {}, mode="c_meanrms").unsqueeze(0)
        return self.forward_from_features(
            raw["patch_tokens"],
            raw["camera_token"],
            centers.to(device=raw["patch_tokens"].device, dtype=raw["patch_tokens"].dtype),
            patch_keep=raw.get("patch_keep"),
            include_camera_in_sequence=include_camera_in_sequence,
        )

    def build_from_cached(
        self,
        payload: Dict,
        device: torch.device,
        *,
        center_scale: float = 1.0,
        align_mode: str = "cross",
        include_camera_in_sequence: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (weak_context, token_keep, camera_embed [1, W]).

        ``align_mode``: ``cross`` (PE=vggtK own bbox), ``fair_gobK`` (PE=gobK
        shared bbox), or ``c_meanrms`` (PE=vggtK own mean+RMS). Requires cache
        payloads from ``cache_vggt_features`` that include ``align_stats``
        (+ ``patch_centers_gobK`` for fair_gobK). For multi-view ``c_meanrms``,
        prefer :meth:`build_c_meanrms_from_rgb_views` instead of cache.

        When ``include_camera_in_sequence`` is False, the sequence is patch tokens
        only; ``camera_embed`` is still returned for AdaLN conditioning.
        """
        del center_scale  # no world_scale in camera-frame design
        if payload.get("pe_frame", "camera") != "camera":
            raise ValueError(
                f"Cached weak context pe_frame={payload.get('pe_frame')!r}; "
                "re-cache with the camera-frame VGGT-depth pipeline."
            )
        if payload.get("geometry_source") != "vggt_depth":
            raise ValueError(
                f"Cached geometry_source={payload.get('geometry_source')!r}; "
                "expected 'vggt_depth'. Re-cache with cache_vggt_features.py --overwrite."
            )
        from hy3dgen.shapegen.cam_align import (
            ALIGN_MODES,
            align_patch_centers,
            select_cached_centers,
        )

        if align_mode not in ALIGN_MODES:
            raise ValueError(f"align_mode must be one of {ALIGN_MODES}, got {align_mode!r}")

        patch = payload["patch_tokens"].to(device=device, dtype=torch.float32)
        cam = payload["camera_token"].to(device=device, dtype=torch.float32)
        centers = select_cached_centers(payload, align_mode).to(
            device=device, dtype=torch.float32
        )
        keep = payload.get("patch_keep")
        if keep is not None:
            keep = keep.to(device=device)
        if patch.dim() == 2:
            patch = patch.unsqueeze(0)
            cam = cam.unsqueeze(0)
            centers = centers.unsqueeze(0)
            if keep is not None and keep.dim() == 1:
                keep = keep.unsqueeze(0)

        align_stats = payload.get("align_stats")
        if align_mode == "c_meanrms" or align_stats is not None:
            centers = align_patch_centers(
                centers[0], align_stats or {}, mode=align_mode
            ).unsqueeze(0)
        else:
            logger.warning(
                "Cache missing align_stats; PE centres used raw (re-cache recommended)."
            )

        return self.forward_from_features(
            patch,
            cam,
            centers,
            patch_keep=keep,
            include_camera_in_sequence=include_camera_in_sequence,
        )

    def forward_from_features(
        self,
        layer_tokens: torch.Tensor,
        camera_token: torch.Tensor,
        patch_centers: torch.Tensor,
        patch_keep: Optional[torch.Tensor] = None,
        *,
        include_camera_in_sequence: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            weak_context: [B, 1+N, width] if camera in sequence, else [B, N, width]
            token_keep: [B, 1+N] or [B, N] — pads False; camera always True if present
            camera_embed: [B, width] projected VGGT camera token (for AdaLN)
        """
        pos = self.pos_embed(patch_centers)
        if layer_tokens.shape[-1] == self.width:
            tokens = layer_tokens + pos
        else:
            tokens = self.feat_proj(layer_tokens) + pos

        batch_size = int(layer_tokens.shape[0])
        cam_raw = camera_token
        if cam_raw.dim() == 2:
            if cam_raw.shape[0] == batch_size:
                cam_raw = cam_raw.unsqueeze(1)
            else:
                cam_raw = cam_raw.unsqueeze(0)
        elif cam_raw.dim() != 3:
            raise ValueError(f"camera_token must be 2D or 3D, got {cam_raw.shape}")

        if cam_raw.shape[-1] == self.width:
            cam = cam_raw
        else:
            b, s, c_in = cam_raw.shape
            flat = cam_raw.reshape(b * s, c_in)
            cam = self.cam_proj(flat).reshape(b, s, self.width)
        camera_embed = cam[:, 0]

        # Zero out padded slots so they don't inject PE noise before masking.
        if patch_keep is not None:
            tokens = tokens * patch_keep.to(tokens.dtype).unsqueeze(-1)

        n_cam = int(cam.shape[1])
        if include_camera_in_sequence:
            ctx = torch.cat([cam, tokens], dim=1)
            b, n_ctx, _ = ctx.shape
            keep = torch.ones(b, n_ctx, dtype=torch.bool, device=ctx.device)
            if patch_keep is not None:
                keep[:, n_cam:] = patch_keep
        else:
            ctx = tokens
            b = ctx.shape[0]
            if patch_keep is not None:
                keep = patch_keep.to(dtype=torch.bool, device=ctx.device)
            else:
                keep = torch.ones(b, ctx.shape[1], dtype=torch.bool, device=ctx.device)
        return ctx, keep, camera_embed

    def forward(
        self,
        rgb: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        c2w: Optional[torch.Tensor] = None,
        *,
        return_dense: bool = False,
        include_camera_in_sequence: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.extract_vggt_raw(
            rgb, depth=depth, intrinsics=intrinsics, c2w=c2w, return_dense=return_dense
        )
        ctx, keep, camera_embed = self.forward_from_features(
            raw["patch_tokens"],
            raw["camera_token"],
            raw["patch_centers"],
            patch_keep=raw.get("patch_keep"),
            include_camera_in_sequence=include_camera_in_sequence,
        )
        if return_dense:
            # Stash dense maps on the module for the debug exporter.
            self._last_dense = {k: v for k, v in raw.items() if k.startswith("vggt_")}
        return ctx, keep, camera_embed


def cache_vggt_contexts(
    mesh_paths: List[str],
    render_source,
    cache_store: CachedVGGTContextStore,
    builder: VGGTContextBuilder,
    *,
    view_idx: int = 0,
    view_indices: Optional[List[int]] = None,
    device: torch.device,
    overwrite: bool = False,
    store_dtype: torch.dtype = torch.float16,
) -> int:
    """Precompute weak-context payloads (VGGT features + dual-K centres + align stats).

    Caches every ``(mesh, view)`` in ``view_indices`` (default: single ``view_idx``).
    """
    views = (
        [int(v) for v in view_indices]
        if view_indices is not None
        else [int(view_idx)]
    )
    builder.eval()
    written = 0
    for mesh_path in mesh_paths:
        for vid in views:
            written += _cache_one_vggt_context(
                mesh_path,
                vid,
                render_source,
                cache_store,
                builder,
                device=device,
                overwrite=overwrite,
                store_dtype=store_dtype,
            )
    return written


def _cache_one_vggt_context(
    mesh_path: str,
    view_idx: int,
    render_source,
    cache_store: CachedVGGTContextStore,
    builder: VGGTContextBuilder,
    *,
    device: torch.device,
    overwrite: bool,
    store_dtype: torch.dtype,
) -> int:
    from hy3dgen.shapegen.cam_align import (
        compute_pe_align_stats,
        gobjaverse_K_for_vggt_resolution,
    )

    if not overwrite and cache_store.has(mesh_path, view_idx):
        logger.info("Cache hit, skipping %s view %d", mesh_path, view_idx)
        return 0
    try:
        view = render_source.load_view(mesh_path, view_idx=view_idx)
    except Exception as e:
        logger.warning("Skip cache %s view %d: %s", mesh_path, view_idx, e)
        return 0

    rgb = view["rgb"].unsqueeze(0).to(device)
    raw = builder.extract_vggt_raw(rgb, return_dense=True)

    depth = raw["vggt_depth"][0].detach().float().cpu().numpy()
    conf = raw["vggt_depth_conf"][0].detach().float().cpu().numpy()
    cam_vggt = raw["vggt_cam_points"][0].detach().float().cpu().numpy()
    K = raw["vggt_intrinsics"][0].detach().float().cpu().numpy()
    vfx, vfy = float(K[0, 0]), float(K[1, 1])
    vcx, vcy = float(K[0, 2]), float(K[1, 2])
    rgb_np = (
        torch.nn.functional.interpolate(
            view["rgb"].unsqueeze(0).float(),
            size=depth.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0]
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    pix_valid = builder._pixel_valid_mask(depth, conf, rgb_np=rgb_np)

    gfx, gfy, gcx, gcy = gobjaverse_K_for_vggt_resolution(
        view["intrinsics"],
        view["rgb"].unsqueeze(0),
        depth_hw=depth.shape[-2:],
        img_size=builder.img_size,
    )
    cam_gob = depth_map_to_cam_points(depth, fx=gfx, fy=gfy, cx=gcx, cy=gcy)
    full_c_g, _ = patch_centers_from_depth(
        cam_gob,
        pix_valid,
        patch_size=builder.patch_size,
        fx=gfx,
        fy=gfy,
        cx=gcx,
        cy=gcy,
    )

    # Extract already filtered+padded vggtK centres / tokens — keep as source of truth
    cen_v_pad = raw["patch_centers"].squeeze(0).float().cpu().clone()
    keep_pad = raw["patch_keep"].squeeze(0).cpu().bool().clone()
    n_keep = int(keep_pad.sum())
    max_n = int(cen_v_pad.shape[0])

    if "patch_keep_full" in raw:
        keep_full = raw["patch_keep_full"][0].cpu().numpy().astype(bool)
    else:
        _, keep_full = patch_centers_from_depth(
            cam_vggt,
            pix_valid,
            patch_size=builder.patch_size,
            fx=vfx,
            fy=vfy,
            cx=vcx,
            cy=vcy,
        )
    n_grid = min(int(keep_full.shape[0]), int(full_c_g.shape[0]))
    cen_g_kept = full_c_g[:n_grid][keep_full[:n_grid]]
    if int(cen_g_kept.shape[0]) != n_keep:
        logger.warning(
            "%s: gobK kept=%d vs extract kept=%d; using min=%d",
            mesh_path,
            cen_g_kept.shape[0],
            n_keep,
            min(int(cen_g_kept.shape[0]), n_keep),
        )
        n_keep = min(int(cen_g_kept.shape[0]), n_keep)
        keep_pad = torch.zeros(max_n, dtype=torch.bool)
        keep_pad[:n_keep] = True
        cen_v_pad[n_keep:] = 0.0
    cen_g_pad = torch.zeros(max_n, 3, dtype=torch.float32)
    if n_keep > 0:
        cen_g_pad[:n_keep] = torch.from_numpy(
            np.asarray(cen_g_kept[:n_keep], dtype=np.float32)
        )

    st_v = compute_pe_align_stats(cam_vggt[pix_valid])
    st_g = compute_pe_align_stats(cam_gob[pix_valid])

    payload = {
        "patch_tokens": raw["patch_tokens"].squeeze(0).to(store_dtype).cpu(),
        "camera_token": raw["camera_token"].squeeze(0).to(store_dtype).cpu(),
        "patch_centers": cen_v_pad,  # backward compat = vggtK
        "patch_centers_vggtK": cen_v_pad,
        "patch_centers_gobK": cen_g_pad,
        "patch_keep": keep_pad,
        "align_stats": {"vggtK": st_v, "gobK": st_g},
        "align_mode_default": "cross",
        "mesh_path": mesh_path,
        "view_idx": view_idx,
        "pe_frame": "camera",
        "geometry_source": "vggt_depth",
    }
    cache_store.save(mesh_path, view_idx, payload)
    logger.info(
        "Cached VGGT context for %s view %d (%d kept patches, align_stats ok)",
        mesh_path,
        view_idx,
        n_keep,
    )
    return 1


def _joint_view_ids(view_indices: Sequence[int]) -> List[int]:
    views = [int(v) for v in view_indices]
    if len(views) < 2:
        raise ValueError(
            f"Joint VGGT cache requires >=2 views in fixed order, got {views!r}"
        )
    return views


def _cache_one_joint_vggt_context(
    mesh_path: str,
    view_indices: Sequence[int],
    render_source,
    cache_store: CachedVGGTContextStore,
    builder: VGGTContextBuilder,
    *,
    device: torch.device,
    overwrite: bool,
    store_dtype: torch.dtype,
) -> int:
    """Joint multi-view VGGT forward for one mesh (view order = VGGT reference order)."""
    views = _joint_view_ids(view_indices)
    if not overwrite and cache_store.has_joint(mesh_path, views):
        logger.info(
            "Joint cache hit, skipping %s views %s", mesh_path, views
        )
        return 0

    try:
        loaded = [
            render_source.load_view(mesh_path, view_idx=int(vid)) for vid in views
        ]
    except Exception as e:
        logger.warning("Skip joint cache %s views %s: %s", mesh_path, views, e)
        return 0

    rgb_views = torch.stack([v["rgb"] for v in loaded], dim=0).to(device)
    builder.eval()
    with torch.no_grad():
        raw = builder.extract_vggt_sequence_dense(rgb_views)

    def _to_store(t: torch.Tensor) -> torch.Tensor:
        return t.detach().to(store_dtype).cpu()

    patch_tokens = raw["patch_tokens"]
    if patch_tokens.dim() == 3:
        patch_tokens = patch_tokens[0]
    camera_token = raw["camera_token"]
    if camera_token.dim() == 3:
        camera_token = camera_token[0]
    patch_centers = raw["patch_centers"]
    if patch_centers.dim() == 3:
        patch_centers = patch_centers[0]
    patch_keep = raw.get("patch_keep")
    if patch_keep is not None and patch_keep.dim() == 2:
        patch_keep = patch_keep[0]

    disc = raw.get("patch_centers_discarded")
    if disc is not None and disc.dim() == 3:
        disc = disc[0]

    dense = raw.get("vggt_points_cam0")
    if dense is not None:
        dense_np = dense.detach().float().cpu().numpy().reshape(-1, 3)
    else:
        dense_np = np.zeros((0, 3), dtype=np.float32)

    payload = {
        "cache_kind": "joint",
        "view_indices": views,
        "num_camera_tokens": int(len(views)),
        "patch_tokens": _to_store(patch_tokens),
        "camera_token": _to_store(camera_token),
        "patch_centers": _to_store(patch_centers),
        "patch_centers_vggtK": _to_store(patch_centers),
        "patch_keep": patch_keep.detach().cpu().bool()
        if patch_keep is not None
        else torch.ones(patch_centers.shape[0], dtype=torch.bool),
        "patch_centers_discarded": _to_store(disc)
        if disc is not None
        else torch.zeros(0, 3, dtype=store_dtype),
        "vggt_points_cam0": dense_np.astype(np.float32),
        "align_mode_default": "c_meanrms",
        "mesh_path": mesh_path,
        "pe_frame": "camera",
        "geometry_source": "vggt_depth",
        "num_views": int(len(views)),
    }
    cache_store.save_joint(mesh_path, views, payload)
    n_kept = int(payload["patch_keep"].sum()) if payload["patch_keep"].numel() else 0
    logger.info(
        "Cached joint VGGT for %s views %s (%d kept patches, %d dense pts)",
        Path(mesh_path).stem,
        views,
        n_kept,
        dense_np.shape[0],
    )
    return 1


def cache_vggt_joint_contexts(
    mesh_paths: List[str],
    render_source,
    cache_store: CachedVGGTContextStore,
    builder: VGGTContextBuilder,
    *,
    view_indices: Sequence[int],
    device: torch.device,
    overwrite: bool = False,
    store_dtype: torch.dtype = torch.float16,
) -> int:
    """Precompute **joint** multi-view VGGT weak context per mesh.

    Runs a single ``extract_vggt_sequence_dense`` forward on all views together
    (view ``view_indices[0]`` = VGGT cam0 reference). Order matters.
    """
    views = _joint_view_ids(view_indices)
    builder.eval()
    written = 0
    for mesh_path in mesh_paths:
        written += _cache_one_joint_vggt_context(
            mesh_path,
            views,
            render_source,
            cache_store,
            builder,
            device=device,
            overwrite=overwrite,
            store_dtype=store_dtype,
        )
    return written


def ordered_view_tuples(
    view_indices: Sequence[int], *, tuple_size: int = 2
) -> List[List[int]]:
    """All ordered tuples of ``tuple_size`` distinct views (order matters)."""
    import itertools

    views = [int(v) for v in view_indices]
    if tuple_size < 2:
        raise ValueError(f"tuple_size must be >= 2, got {tuple_size}")
    if len(views) < tuple_size:
        raise ValueError(
            f"Need >= {tuple_size} views for ordered tuples, got {views!r}"
        )
    # Deduplicate while preserving first-seen order.
    seen: set = set()
    uniq: List[int] = []
    for v in views:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    if len(uniq) < tuple_size:
        raise ValueError(
            f"Need >= {tuple_size} unique views for ordered tuples, got {uniq!r}"
        )
    return [list(t) for t in itertools.permutations(uniq, tuple_size)]


def cache_vggt_joint_ordered_pair_contexts(
    mesh_paths: List[str],
    render_source,
    cache_store: CachedVGGTContextStore,
    builder: VGGTContextBuilder,
    *,
    view_indices: Sequence[int],
    device: torch.device,
    pair_size: int = 2,
    overwrite: bool = False,
    store_dtype: torch.dtype = torch.float16,
) -> int:
    """Cache joint VGGT for **every ordered pair** (or k-tuple) from a view pool.

    For pool ``[4,10,16,22]`` and ``pair_size=2`` this writes P(4,2)=12 joint
    files per mesh (``*_joint_00004_00010.pt``, ``*_joint_00010_00004.pt``, …).
    """
    tuples = ordered_view_tuples(view_indices, tuple_size=pair_size)
    logger.info(
        "Joint ordered-pair cache: %d unique ordered %d-tuples from pool %s",
        len(tuples),
        pair_size,
        [int(v) for v in view_indices],
    )
    builder.eval()
    written = 0
    for pair in tuples:
        logger.info("Caching joint ordered tuple %s", pair)
        for mesh_path in mesh_paths:
            written += _cache_one_joint_vggt_context(
                mesh_path,
                pair,
                render_source,
                cache_store,
                builder,
                device=device,
                overwrite=overwrite,
                store_dtype=store_dtype,
            )
    return written
