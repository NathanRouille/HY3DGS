"""Debug PLY exports for ShapePCAE / ShapePCUnite reconstruction diagnosis.

Typical CloudCompare overlays (same camera frame):
  gt.ply / recon.ply          — already exported by eval/vis
  fps.ply                     — encoder FPS queries (cyan)
  anchors.ply                 — decoder cluster centres (magenta)
  recon_error.ply             — recon coloured by dist→GT (blue=good → red=far)
  gt_uncovered.ply            — GT coloured by dist→recon
  intruders.ply               — recon points with dist→GT > threshold
  locals_by_anchor.ply        — recon coloured by anchor id
  delta_mag.ply               — recon coloured by |Δ| / max_anchor_delta
  patch_centers.ply           — kept VGGT patch centres (optional)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch

from hy3dgen.shapegen.gs_export import export_xyz_pointcloud_ply
from hy3dgen.shapegen.pc_losses import pairwise_dist2

logger = logging.getLogger(__name__)


def _to_np(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _squeeze_batch(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Accept [3], [N,3], or [1,N,3] → [N,3]."""
    arr = _to_np(x)
    if arr.ndim == 3:
        arr = arr[0]
    return np.ascontiguousarray(arr.reshape(-1, arr.shape[-1]))


def scalar_to_rgb(
    values: np.ndarray,
    *,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Map scalars → RGB in [0,1] (blue → cyan → yellow → red)."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    lo = float(np.min(v) if vmin is None else vmin)
    hi = float(np.max(v) if vmax is None else vmax)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        t = np.zeros_like(v)
    else:
        t = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    # piecewise: blue(0,0,1) → cyan(0,1,1) → yellow(1,1,0) → red(1,0,0)
    rgb = np.zeros((v.size, 3), dtype=np.float64)
    m1 = t < 1.0 / 3.0
    m2 = (t >= 1.0 / 3.0) & (t < 2.0 / 3.0)
    m3 = t >= 2.0 / 3.0
    u = t * 3.0
    rgb[m1, 1] = u[m1]
    rgb[m1, 2] = 1.0
    u2 = (t[m2] - 1.0 / 3.0) * 3.0
    rgb[m2, 0] = u2
    rgb[m2, 1] = 1.0
    rgb[m2, 2] = 1.0 - u2
    u3 = (t[m3] - 2.0 / 3.0) * 3.0
    rgb[m3, 0] = 1.0
    rgb[m3, 1] = 1.0 - u3
    return rgb.astype(np.float32)


def anchor_id_colors(num_anchors: int, points_per_anchor: int) -> np.ndarray:
    """Periodic distinct colours for locals grouped by anchor id → [R*K, 3]."""
    if num_anchors <= 0 or points_per_anchor <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    # Golden-angle hues on a simple HSV→RGB wheel (S=0.85, V=0.95).
    ids = np.arange(num_anchors, dtype=np.float64)
    h = np.fmod(ids * 0.61803398875, 1.0)
    s = np.full_like(h, 0.85)
    val = np.full_like(h, 0.95)
    i = np.floor(h * 6.0).astype(np.int64) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p = val * (1.0 - s)
    q = val * (1.0 - f * s)
    t = val * (1.0 - (1.0 - f) * s)
    rgb = np.zeros((num_anchors, 3), dtype=np.float64)
    for idx, (ii, vv, pp, qq, tt) in enumerate(zip(i, val, p, q, t)):
        if ii == 0:
            rgb[idx] = (vv, tt, pp)
        elif ii == 1:
            rgb[idx] = (qq, vv, pp)
        elif ii == 2:
            rgb[idx] = (pp, vv, tt)
        elif ii == 3:
            rgb[idx] = (pp, qq, vv)
        elif ii == 4:
            rgb[idx] = (tt, pp, vv)
        else:
            rgb[idx] = (vv, pp, qq)
    return np.repeat(rgb.astype(np.float32), int(points_per_anchor), axis=0)


def nn_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For each point in ``a``, squared? No — Euclidean dist to nearest in ``b``."""
    if a.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if b.size == 0:
        return np.full((a.shape[0],), np.inf, dtype=np.float32)
    ta = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).unsqueeze(0)
    tb = torch.from_numpy(np.ascontiguousarray(b, dtype=np.float32)).unsqueeze(0)
    d2 = pairwise_dist2(ta, tb)[0]  # [Na, Nb]
    return torch.sqrt(d2.min(dim=1).values.clamp_min(0)).cpu().numpy().astype(np.float32)


def export_recon_debug_plys(
    out_dir: Union[str, Path],
    *,
    gt_xyz: Union[torch.Tensor, np.ndarray],
    recon_xyz: Union[torch.Tensor, np.ndarray],
    fps_xyz: Optional[Union[torch.Tensor, np.ndarray]] = None,
    centers: Optional[Union[torch.Tensor, np.ndarray]] = None,
    num_points_per_anchor: Optional[int] = None,
    max_anchor_delta: Optional[float] = None,
    patch_centers: Optional[Union[torch.Tensor, np.ndarray]] = None,
    patch_keep: Optional[Union[torch.Tensor, np.ndarray]] = None,
    intruder_thresh: float = 0.02,
    error_vmax: Optional[float] = None,
) -> Dict[str, str]:
    """Write AE debug PLYs into ``out_dir``. Returns map of name → path written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}

    gt = _squeeze_batch(gt_xyz)[:, :3]
    recon = _squeeze_batch(recon_xyz)[:, :3]

    # --- FPS ---
    if fps_xyz is not None:
        fps = _squeeze_batch(fps_xyz)[:, :3]
        p = out_dir / "fps.ply"
        export_xyz_pointcloud_ply(fps, p, rgb=(0, 220, 220))
        written["fps"] = str(p)

    # --- Anchors ---
    centers_np: Optional[np.ndarray] = None
    if centers is not None:
        centers_np = _squeeze_batch(centers)[:, :3]
        p = out_dir / "anchors.ply"
        export_xyz_pointcloud_ply(centers_np, p, rgb=(255, 0, 255))
        written["anchors"] = str(p)

    # --- Error maps ---
    d_recon_to_gt = nn_distances(recon, gt)
    d_gt_to_recon = nn_distances(gt, recon)
    vmax = float(error_vmax) if error_vmax is not None else float(
        np.percentile(np.concatenate([d_recon_to_gt, d_gt_to_recon]), 95)
        if (d_recon_to_gt.size + d_gt_to_recon.size) > 0
        else 0.05
    )
    vmax = max(vmax, 1e-6)

    p = out_dir / "recon_error.ply"
    export_xyz_pointcloud_ply(
        recon, p, colors=scalar_to_rgb(d_recon_to_gt, vmin=0.0, vmax=vmax)
    )
    written["recon_error"] = str(p)

    p = out_dir / "gt_uncovered.ply"
    export_xyz_pointcloud_ply(
        gt, p, colors=scalar_to_rgb(d_gt_to_recon, vmin=0.0, vmax=vmax)
    )
    written["gt_uncovered"] = str(p)

    # --- Intruders (recon far from GT) ---
    thr = float(intruder_thresh)
    mask = d_recon_to_gt > thr
    p = out_dir / "intruders.ply"
    if mask.any():
        export_xyz_pointcloud_ply(
            recon[mask],
            p,
            colors=scalar_to_rgb(d_recon_to_gt[mask], vmin=thr, vmax=max(vmax, thr * 2)),
        )
    else:
        # Empty cloud still useful as a "no intruders" marker.
        export_xyz_pointcloud_ply(np.zeros((0, 3), dtype=np.float32), p, rgb=(255, 80, 80))
    written["intruders"] = str(p)

    # --- Locals by anchor + delta magnitude ---
    K = int(num_points_per_anchor) if num_points_per_anchor else 0
    if centers_np is not None and K > 0 and recon.shape[0] == centers_np.shape[0] * K:
        R = centers_np.shape[0]
        p = out_dir / "locals_by_anchor.ply"
        export_xyz_pointcloud_ply(recon, p, colors=anchor_id_colors(R, K))
        written["locals_by_anchor"] = str(p)

        recon_rk = recon.reshape(R, K, 3)
        delta = recon_rk - centers_np[:, None, :]
        delta_norm = np.linalg.norm(delta, axis=-1).reshape(-1)
        denom = float(max_anchor_delta) if max_anchor_delta and max_anchor_delta > 0 else float(
            np.max(delta_norm) if delta_norm.size else 1.0
        )
        denom = max(denom, 1e-8)
        p = out_dir / "delta_mag.ply"
        export_xyz_pointcloud_ply(
            recon, p, colors=scalar_to_rgb(delta_norm / denom, vmin=0.0, vmax=1.0)
        )
        written["delta_mag"] = str(p)

        # FPS vs anchors overlay helper: write a tiny README of colours
        readme = out_dir / "DEBUG_PLY_README.txt"
        readme.write_text(
            "ShapePCUnite recon debug PLYs (same camera frame as gt/recon).\n"
            "\n"
            "fps.ply              cyan     — encoder FPS query points\n"
            "anchors.ply         magenta  — decoder anchor centres\n"
            "recon_error.ply     heatmap  — recon coloured by dist→nearest GT "
            f"(vmax≈{vmax:.4f})\n"
            "gt_uncovered.ply    heatmap  — GT coloured by dist→nearest recon\n"
            f"intruders.ply       heatmap  — recon with dist→GT > {thr:g}\n"
            "locals_by_anchor.ply rainbow — recon coloured by parent anchor id\n"
            "delta_mag.ply       heatmap  — |xyz-center| / max_anchor_delta "
            "(1=saturated tanh)\n"
            "patch_centers.ply   green    — kept VGGT weak-context centres\n"
            "\n"
            "Reading tips:\n"
            "- FPS sparse on thin parts → need denser / sharper surface sampling.\n"
            "- Anchors in voids (vs FPS on surface) → anc / decoder pull off-surface.\n"
            "- Red clumps on recon_error / intruders between legs → local smear.\n"
            "- Red on gt_uncovered at edges → missing thin detail / undersampling.\n"
            "- delta_mag near red → locals pushed to max_anchor_delta limit.\n"
        )
        written["readme"] = str(readme)
    else:
        readme = out_dir / "DEBUG_PLY_README.txt"
        readme.write_text(
            "ShapePCUnite recon debug PLYs.\n"
            "fps=cyan anchors=magenta error heatmaps blue→red.\n"
            f"intruders: dist→GT > {thr:g}.\n"
            "locals_by_anchor / delta_mag skipped "
            "(need centers + matching num_points_per_anchor).\n"
        )
        written["readme"] = str(readme)

    # --- Patch centres (optional) ---
    if patch_centers is not None:
        pc = _squeeze_batch(patch_centers)[:, :3]
        if patch_keep is not None:
            keep = _to_np(patch_keep).reshape(-1).astype(bool)
            if keep.shape[0] == pc.shape[0]:
                pc = pc[keep]
            elif keep.shape[0] < pc.shape[0]:
                pc = pc[: keep.shape[0]][keep]
        # Drop padded zeros sometimes left after keep
        if pc.shape[0] > 0:
            p = out_dir / "patch_centers.ply"
            export_xyz_pointcloud_ply(pc, p, rgb=(32, 200, 64))
            written["patch_centers"] = str(p)

    logger.info(
        "Wrote %d recon-debug PLYs to %s (%s)",
        len(written),
        out_dir,
        ", ".join(sorted(k for k in written if k != "readme")),
    )
    return written
