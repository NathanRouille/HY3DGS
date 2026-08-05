"""Camera-frame PE/GT alignment recipes (cross vs fair_gobK).

After mesh⊕c2w and VGGT depth unprojection, both clouds are put in a Hunyuan-style
unit box. Two plug-and-play modes:

**cross** (default, train=infer PE)
  PE  = VGGT depth ⊕ vggtK → /mean(z) → own Hunyuan bbox
  GT  = mesh → /mean_z(GT depth⊕gobK) → Hunyuan bbox from VGGT⊕gobK FG

**fair_gobK** (ablation; best shared-frame align, PE OOD at infer)
  PE  = VGGT depth ⊕ gobK → /mean(z) → Hunyuan bbox
  GT  = mesh → /mean_z(GT depth⊕gobK) → **same** (μ,s) as PE
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from hy3dgen.shapegen.vggt_context import (
    depth_map_to_cam_points,
    preprocess_rgb_for_vggt,
)

logger = logging.getLogger(__name__)

ALIGN_MODES = ("cross", "fair_gobK")
AlignMode = str  # "cross" | "fair_gobK"


def _as_xyz_np(xyz) -> np.ndarray:
    if isinstance(xyz, torch.Tensor):
        xyz = xyz.detach().float().cpu().numpy()
    return np.asarray(xyz, dtype=np.float64).reshape(-1, 3)


def mean_z_scale(xyz, *, eps: float = 1e-6) -> float:
    """Isotropic scale = mean of finite positive z."""
    z = _as_xyz_np(xyz)[:, 2]
    z = z[np.isfinite(z) & (z > eps)]
    if z.size == 0:
        return 1.0
    s = float(np.mean(z))
    if not np.isfinite(s) or s <= eps:
        return 1.0
    return s


def hunyuan_bbox_mu_s(
    pts,
    *,
    fill: float = 0.9999,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, float]:
    """Hunyuan ``normalize_mesh`` stats: μ = AABB center, s = max_side/(2*fill)."""
    p = _as_xyz_np(pts)
    if p.shape[0] == 0:
        return np.zeros(3, dtype=np.float64), 1.0
    lo = p.min(axis=0)
    hi = p.max(axis=0)
    mu = 0.5 * (lo + hi)
    side = float((hi - lo).max())
    s = side / max(2.0 * float(fill), eps)
    if not np.isfinite(s) or s < eps:
        s = 1.0
    return mu.astype(np.float64), float(s)


def apply_mean_then_bbox(
    pts,
    mean_z: float,
    mu: np.ndarray,
    s: float,
) -> np.ndarray:
    """``p' = (p / mean_z - μ) / s`` (numpy float32)."""
    p = _as_xyz_np(pts)
    if p.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    out = (p / max(float(mean_z), 1e-6) - np.asarray(mu, dtype=np.float64)[None, :]) / max(
        float(s), 1e-6
    )
    return out.astype(np.float32)


def apply_mean_then_bbox_torch(
    pts: torch.Tensor,
    mean_z: float,
    mu: Union[np.ndarray, torch.Tensor],
    s: float,
) -> torch.Tensor:
    """Torch version of :func:`apply_mean_then_bbox` (preserves device/dtype)."""
    if pts.numel() == 0:
        return pts
    mu_t = torch.as_tensor(mu, device=pts.device, dtype=pts.dtype).view(1, 3)
    return (pts / max(float(mean_z), 1e-6) - mu_t) / max(float(s), 1e-6)


def gobjaverse_K_for_vggt_resolution(
    intrinsics_fxfycxcy: torch.Tensor,
    rgb: torch.Tensor,
    *,
    depth_hw: Tuple[int, int],
    img_size: int = 518,
) -> Tuple[float, float, float, float]:
    """Map G-Objaverse (fx,fy,cx,cy) from native RGB size → VGGT depth grid."""
    fx, fy, cx, cy = [float(x) for x in intrinsics_fxfycxcy.reshape(-1)[:4]]
    _, scale_y, scale_x, pad_top, pad_left = preprocess_rgb_for_vggt(
        rgb.float().clamp(0, 1), target_size=img_size
    )
    return (
        fx * scale_x,
        fy * scale_y,
        cx * scale_x + pad_left,
        cy * scale_y + pad_top,
    )


def fg_mask_from_depth_rgb(
    depth: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    *,
    depth_eps: float = 1e-6,
) -> np.ndarray:
    """Simple FG mask: depth > eps and not near-white RGB."""
    valid = np.isfinite(depth) & (depth > depth_eps)
    if rgb is not None and rgb.shape[:2] == depth.shape[:2]:
        white = (
            (rgb[..., 0] > 0.97) & (rgb[..., 1] > 0.97) & (rgb[..., 2] > 0.97)
        )
        valid = valid & ~white
    return valid


def gt_depth_mean_z(
    depth: Union[np.ndarray, torch.Tensor],
    intrinsics_fxfycxcy: Union[np.ndarray, torch.Tensor],
    rgb: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> float:
    """mean_z of nd.exr ⊕ native gobK on FG pixels."""
    if isinstance(depth, torch.Tensor):
        depth_np = depth.detach().float().cpu().numpy()
    else:
        depth_np = np.asarray(depth, dtype=np.float32)
    if depth_np.ndim == 3:
        depth_np = depth_np.squeeze()
    fx, fy, cx, cy = [float(x) for x in np.asarray(intrinsics_fxfycxcy).reshape(-1)[:4]]
    rgb_np = None
    if rgb is not None:
        if isinstance(rgb, torch.Tensor):
            r = rgb.detach().float().cpu()
            if r.dim() == 3 and r.shape[0] == 3:
                rgb_np = r.permute(1, 2, 0).numpy()
            else:
                rgb_np = r.numpy()
        else:
            rgb_np = np.asarray(rgb)
    cam = depth_map_to_cam_points(depth_np, fx=fx, fy=fy, cx=cx, cy=cy)
    valid = fg_mask_from_depth_rgb(depth_np, rgb_np)
    pts = cam[valid]
    return mean_z_scale(pts)


def compute_pe_align_stats(
    pts_fg: np.ndarray,
    *,
    fill: float = 0.9999,
) -> Dict[str, float]:
    """mean_z + Hunyuan (μ,s) for one FG cloud (after choosing K)."""
    mz = mean_z_scale(pts_fg)
    pts_m = _as_xyz_np(pts_fg) / max(mz, 1e-6)
    mu, s = hunyuan_bbox_mu_s(pts_m, fill=fill)
    return {
        "mean_z": float(mz),
        "mu_x": float(mu[0]),
        "mu_y": float(mu[1]),
        "mu_z": float(mu[2]),
        "s": float(s),
    }


def stats_to_mu_s(stats: Dict) -> Tuple[np.ndarray, float, float]:
    """Unpack align stats → (mu[3], s, mean_z)."""
    mu = np.array(
        [float(stats["mu_x"]), float(stats["mu_y"]), float(stats["mu_z"])],
        dtype=np.float64,
    )
    return mu, float(stats["s"]), float(stats["mean_z"])


def align_patch_centers(
    centers: torch.Tensor,
    align_stats: Dict,
    *,
    mode: AlignMode,
) -> torch.Tensor:
    """Apply mean+bbox to PE centres using cached stats for ``mode``."""
    key = "vggtK" if mode == "cross" else "gobK"
    st = align_stats[key]
    mu, s, mz = stats_to_mu_s(st)
    return apply_mean_then_bbox_torch(centers, mz, mu, s)


def align_gt_xyz(
    xyz: torch.Tensor,
    *,
    mean_z_gt_depth: float,
    align_stats: Dict,
    mode: AlignMode,
) -> torch.Tensor:
    """Apply cross / fair_gobK GT map (always uses gobK bbox stats for GT)."""
    # Both modes: GT bbox from VGGT⊕gobK FG (fair: same as PE; cross: gob ref).
    st = align_stats["gobK"]
    mu, s, _ = stats_to_mu_s(st)
    return apply_mean_then_bbox_torch(xyz, mean_z_gt_depth, mu, s)


def select_cached_centers(payload: Dict, mode: AlignMode) -> torch.Tensor:
    """Pick vggtK or gobK centres from a cache payload."""
    if mode == "fair_gobK":
        if "patch_centers_gobK" in payload:
            return payload["patch_centers_gobK"]
        logger.warning(
            "Cache missing patch_centers_gobK; falling back to patch_centers (vggtK). "
            "Re-cache for a true fair_gobK ablation."
        )
        return payload["patch_centers"]
    # cross
    if "patch_centers_vggtK" in payload:
        return payload["patch_centers_vggtK"]
    return payload["patch_centers"]
