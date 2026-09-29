"""Camera-frame PE/GT alignment recipes (cross vs fair_gobK vs c_meanrms).

After mesh⊕c2w and VGGT depth unprojection, both clouds are put in a Hunyuan-style
unit box (or mean+RMS for ``c_meanrms``). Plug-and-play modes:

**cross** (default, train=infer PE)
  PE  = VGGT depth ⊕ vggtK → own Hunyuan bbox
        (/mean before bbox is algebraically redundant on the same cloud)
  GT  = mesh → /mean_z(GT depth⊕gobK) → Hunyuan bbox from VGGT⊕gobK FG
        (VGGT⊕gobK still uses /mean before its Hunyuan stats)

**fair_gobK** (ablation; best shared-frame align, PE OOD at infer)
  PE  = VGGT depth ⊕ gobK → /mean(z) → Hunyuan bbox
  GT  = mesh → /mean_z(GT depth⊕gobK) → **same** (μ,s) as PE

**c_meanrms** (``C_gt_depth_filter_zrobust_meanrms``)
  PE  = VGGT depth ⊕ vggtK (multi-view ∪ in VGGT cam0) → own mean+RMS
  GT  = mesh in ref cam → mean+RMS from GT-depth ∪ (erode FG 1px, gobK+GT E)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from hy3dgen.shapegen.vggt_context import (
    depth_map_to_cam_points,
    preprocess_rgb_for_vggt,
)

logger = logging.getLogger(__name__)

ALIGN_MODES = ("cross", "fair_gobK", "c_meanrms")
AlignMode = str  # "cross" | "fair_gobK" | "c_meanrms"


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


def mean_rms_mu_s(pts, *, eps: float = 1e-6) -> Tuple[np.ndarray, float]:
    """μ = point mean, s = RMS of centered points (isotropic)."""
    p = _as_xyz_np(pts)
    if p.shape[0] == 0:
        return np.zeros(3, dtype=np.float64), 1.0
    mu = p.mean(axis=0)
    centered = p - mu[None, :]
    s = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    if not np.isfinite(s) or s < eps:
        s = 1.0
    return mu.astype(np.float64), float(s)


def apply_mu_s(pts, mu, s: float) -> np.ndarray:
    """``p' = (p - μ) / s`` (numpy float32)."""
    p = _as_xyz_np(pts)
    if p.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    out = (p - np.asarray(mu, dtype=np.float64)[None, :]) / max(float(s), 1e-6)
    return out.astype(np.float32)


def apply_mu_s_torch(
    pts: torch.Tensor,
    mu: Union[np.ndarray, torch.Tensor],
    s: float,
) -> torch.Tensor:
    if pts.numel() == 0:
        return pts
    mu_t = torch.as_tensor(mu, device=pts.device, dtype=pts.dtype).view(1, 3)
    return (pts - mu_t) / max(float(s), 1e-6)


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
    """Apply PE normalize using cached stats for ``mode``."""
    if mode == "c_meanrms":
        # Own mean+RMS on the PE centres (train=infer); ignore AABB cache.
        if centers.numel() == 0:
            return centers
        c = centers.detach().float()
        if c.dim() == 3:
            c = c[0]
        mu, s = mean_rms_mu_s(c.cpu().numpy())
        return apply_mu_s_torch(centers if centers.dim() == 2 else centers[0], mu, s)
    key = "vggtK" if mode == "cross" else "gobK"
    st = align_stats[key]
    mu, s, mz = stats_to_mu_s(st)
    return apply_mean_then_bbox_torch(centers, mz, mu, s)


def apply_pe_normalize_np(
    pts: np.ndarray,
    *,
    mode: AlignMode,
    pe_centers_raw: np.ndarray,
    align_stats: Optional[Dict] = None,
) -> np.ndarray:
    """Apply the **same** PE normalize used at train time to an arbitrary cloud.

    For ``c_meanrms``, ``(μ,s)`` come from ``pe_centers_raw`` (kept PE centres).
    For ``cross`` / ``fair_gobK``, uses ``align_stats`` from the VGGT cache.
    """
    pts = _as_xyz_np(pts).astype(np.float64)
    if pts.size == 0:
        return pts.astype(np.float32)
    if mode == "c_meanrms":
        mu, s = mean_rms_mu_s(pe_centers_raw)
        return apply_mu_s(pts, mu, s).astype(np.float32)
    if not align_stats:
        logger.warning("apply_pe_normalize_np: missing align_stats for mode=%s", mode)
        return pts.astype(np.float32)
    key = "vggtK" if mode == "cross" else "gobK"
    st = align_stats[key]
    mu, s, mz = stats_to_mu_s(st)
    return apply_mean_then_bbox(pts, mz, mu, s).astype(np.float32)


def align_gt_xyz(
    xyz: torch.Tensor,
    *,
    mean_z_gt_depth: float,
    align_stats: Dict,
    mode: AlignMode,
) -> torch.Tensor:
    """Apply GT map for ``mode``."""
    if mode == "c_meanrms":
        st = align_stats["c_meanrms"]
        mu = np.array(
            [float(st["mu_x"]), float(st["mu_y"]), float(st["mu_z"])],
            dtype=np.float64,
        )
        return apply_mu_s_torch(xyz, mu, float(st["s"]))
    # Both cross / fair_gobK: GT bbox from VGGT⊕gobK FG.
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
    # cross and c_meanrms use vggtK centres
    if "patch_centers_vggtK" in payload:
        return payload["patch_centers_vggtK"]
    return payload["patch_centers"]


def _erode_mask(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    for _ in range(max(int(iters), 0)):
        up = np.zeros_like(m)
        down = np.zeros_like(m)
        left = np.zeros_like(m)
        right = np.zeros_like(m)
        up[1:] = m[:-1]
        down[:-1] = m[1:]
        left[:, 1:] = m[:, :-1]
        right[:, :-1] = m[:, 1:]
        m = m & up & down & left & right
    return m


def compute_c_meanrms_gt_stats(
    depths: List[np.ndarray],
    rgbs: List[np.ndarray],
    intrinsics_list: List,
    c2ws: List[np.ndarray],
    c2w_ref: np.ndarray,
    *,
    erode_iters: int = 1,
) -> Dict[str, float]:
    """mean+RMS stats from GT-depth ∪ (erode 1px) in ``c2w_ref`` camera frame."""
    valids = []
    Ks = []
    for d, rgb, K in zip(depths, rgbs, intrinsics_list):
        d = np.asarray(d, dtype=np.float32)
        if d.ndim == 3:
            d = d.squeeze()
        rgb_np = np.asarray(rgb)
        if rgb_np.ndim == 3 and rgb_np.shape[0] == 3:
            rgb_np = np.transpose(rgb_np, (1, 2, 0))
        valid = fg_mask_from_depth_rgb(d, rgb_np)
        valid = _erode_mask(valid, iters=erode_iters)
        valids.append(valid)
        fx, fy, cx, cy = [float(x) for x in np.asarray(K).reshape(-1)[:4]]
        Ks.append((fx, fy, cx, cy))
    pts, _ = merge_unprojected_to_cam_ref(
        [np.asarray(d, dtype=np.float32).squeeze() for d in depths],
        valids,
        Ks=Ks,
        c2ws=[np.asarray(c, dtype=np.float64) for c in c2ws],
        c2w_ref=np.asarray(c2w_ref, dtype=np.float64),
    )
    mu, s = mean_rms_mu_s(pts)
    return {
        "mu_x": float(mu[0]),
        "mu_y": float(mu[1]),
        "mu_z": float(mu[2]),
        "s": float(s),
        "n_pts": float(pts.shape[0]),
    }


def aligned_centers_from_payload(
    payload: Dict,
    mode: AlignMode = "cross",
):
    """Raw + training-frame centres from a VGGT cache payload.

    Cache stores pre-normalize VGGT-depth centres; training applies
    :func:`align_patch_centers` before Fourier PE. Debug/eval PLY exports must
    use the aligned centres or they look mis-scaled vs GT.

    Returns:
        ``(raw, aligned, patch_keep)`` — tensors ``[N, 3]`` (or ``None``).
    """
    if not isinstance(payload, dict):
        return None, None, None
    keep = payload.get("patch_keep")
    raw = select_cached_centers(payload, mode).float()
    if raw.dim() == 3:
        raw = raw[0]
    aligned = raw
    if payload.get("align_stats") is not None:
        aligned = align_patch_centers(raw, payload["align_stats"], mode=mode)
    else:
        logger.warning("Cache missing align_stats; using raw patch centres")
    return raw, aligned, keep


def camera_to_world(points_cam, c2w) -> np.ndarray:
    """Camera-frame points → world using Unity/OpenCV-style ``c2w`` (R|t)."""
    pts = _as_xyz_np(points_cam).astype(np.float64)
    c2w = np.asarray(c2w, dtype=np.float64)
    R, t = c2w[:3, :3], c2w[:3, 3]
    return (pts @ R.T + t[None, :]).astype(np.float32)


def cam_i_to_cam_ref(points_cam_i, c2w_i, c2w_ref) -> np.ndarray:
    """Rigid map from camera ``i`` into the reference camera frame."""
    from hy3dgen.shapegen.vggt_context import world_to_camera

    world = camera_to_world(points_cam_i, c2w_i)
    return world_to_camera(world, c2w_ref).astype(np.float32)


def vggt_extrinsic_to_c2w(extrinsic_3x4) -> np.ndarray:
    """VGGT OpenCV ``cam←world`` (3×4) → 4×4 ``c2w``."""
    from vggt.utils.geometry import closed_form_inverse_se3  # type: ignore

    E = np.asarray(extrinsic_3x4, dtype=np.float64).reshape(3, 4)
    homog = np.eye(4, dtype=np.float64)
    homog[:3, :4] = E
    return closed_form_inverse_se3(homog[None])[0].astype(np.float64)


def vggt_world_to_cam0(points_world, extrinsic_0_3x4) -> np.ndarray:
    """Map VGGT 'world' points into view-0 camera using ``E0`` (cam←world)."""
    pts = _as_xyz_np(points_world).astype(np.float64)
    E0 = np.asarray(extrinsic_0_3x4, dtype=np.float64).reshape(3, 4)
    R, t = E0[:3, :3], E0[:3, 3]
    return (pts @ R.T + t[None, :]).astype(np.float32)


def unproject_depth_fg(
    depth: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    valid: np.ndarray,
    colors: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Unproject FG pixels → ``(N,3)`` (+ optional colors)."""
    from hy3dgen.shapegen.vggt_context import depth_map_to_cam_points

    cam = depth_map_to_cam_points(depth, fx=fx, fy=fy, cx=cx, cy=cy)
    m = np.asarray(valid, dtype=bool)
    pts = cam[m].astype(np.float32)
    cols = None
    if colors is not None and colors.shape[:2] == depth.shape[:2]:
        cols = np.asarray(colors, dtype=np.float32)[m]
    return pts, cols


def merge_unprojected_to_cam_ref(
    depths: List[np.ndarray],
    valids: List[np.ndarray],
    *,
    Ks: List[Tuple[float, float, float, float]],
    c2ws: List[np.ndarray],
    c2w_ref: np.ndarray,
    colors: Optional[List[Optional[np.ndarray]]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Unproject each view with its ``K``, rigid-map into ``c2w_ref`` camera frame.

    ``Ks[i] = (fx, fy, cx, cy)``. Empty views are skipped.
    """
    chunks: List[np.ndarray] = []
    col_chunks: List[np.ndarray] = []
    want_cols = colors is not None
    for i, (d, m, K) in enumerate(zip(depths, valids, Ks)):
        fx, fy, cx, cy = K
        cols_i = colors[i] if want_cols else None
        pts_i, cols_out = unproject_depth_fg(
            d, fx=fx, fy=fy, cx=cx, cy=cy, valid=m, colors=cols_i
        )
        if pts_i.shape[0] == 0:
            continue
        pts_ref = cam_i_to_cam_ref(pts_i, c2ws[i], c2w_ref)
        chunks.append(pts_ref)
        if want_cols and cols_out is not None:
            col_chunks.append(cols_out)
    if not chunks:
        empty = np.zeros((0, 3), dtype=np.float32)
        return empty, None
    pts = np.concatenate(chunks, axis=0).astype(np.float32)
    cols = np.concatenate(col_chunks, axis=0) if col_chunks else None
    return pts, cols
