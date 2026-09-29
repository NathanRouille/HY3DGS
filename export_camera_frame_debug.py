#!/usr/bin/env python3
"""Export camera-frame alignment debug PLYs with depth-scale ablations.

Default depth normalization is ``p' = p / z_min`` (nearest FG depth → 1), applied
independently to each cloud — **except** ``scale_median`` / ``scale_p*``.

For each object writes:

  raw/  raw_gobK/
    Unscaled baselines (no /z_min)

  scale_zmin/
    VGGT-K unproject + /z_min (main baseline)

  scale_median/, scale_p20/, …
    Same but /median or /percentile (explicit ablations)

  scale_zmin_gobK/
    VGGT depth ⊕ G-Objaverse K, then /z_min

  align_sim/
    /z_min then isotropic Umeyama (full GT→NN)

  align_sim_trim/  align_sim_mutual/  align_sim_visible/
    Same after /z_min but trimmed / mutual / camera-visible GT only

  align_sim_raw_full/  align_sim_raw_trim/
    Umeyama on raw clouds (no /z_min) — full vs trimmed

  gt_visible_zmin/
    GT points projecting onto VGGT FG, /z_min (no Umeyama)

  gt_reproj_vggtK/
    FoV warp GT via pixel project/unproject (can shear), then /z_min

  gt_reproj_vggtK_mean/
    Same FoV warp, then /mean(z) instead of /z_min

  gt_fxscale_vggtK/
    Shear-free FoV match: GT xy *= fx_gob@518/fx_vggt, then /z_min

  gt_depth_gobK/
    G-Objaverse nd.exr ⊕ gobK; mesh + depth + VGGT all /mean(z)

  depth_compare_mean/
    Same-view depth ablations (all /mean(z) — avoids noisy z_min):
      gt_depth_gobK, gt_depth_vggtK, vggt_pred_gobK, vggt_pred_vggtK,
      vggt_pred_scaleshift_gobK, gt_mesh

  zmin_then_shared_canon/
    /z_min each, then shared μ,s from VGGT → ~unit ball

  zmin_then_indep_canon/
    /z_min each, then independent μ and s per cloud

  scale_zmin_pointmap/
    VGGT point_head (+ depth cloud); mesh + clouds /z_min

  canon_shared_vggt/
    Camera-frame GT + VGGT; shared ``(p-μ)/s`` with μ,s from VGGT only
    (inference-safe shared PE frame)

  mesh_vs_depth_cam/
    Metric camera frame (NO /z_min or /mean): mesh⊕c2w vs nd.exr⊕gobK
    — sanity check that extrinsics+intrinsics agree on the visible surface

  sol1_fxscale_vggtK_mean/
    Train=infer PE: VGGT⊕vggtK /mean(z); GT mesh shear-free fxscale then /mean(z)

  sol2_iso_vggtK_mean/
    Train=infer PE: VGGT⊕vggtK /mean(z); GT mesh /own mean(z) (no FoV warp)

  sol3_gobK_train/
    /mean(z) depth-match + shared Hunyuan bbox (μ,s from VGGT gobK FG):
      gt_mesh_by_gtdepth_mean, vggt_pred_gobK, vggt_pred_vggtK (infer ablation),
      patch_centers_gobK, discarded_centers_gobK,
      patch_centers_vggtK, discarded_centers_vggtK (infer ablation)

  k_bakeoff/
    Fair K comparison (+ cross OOD probe), Chamfer/bbox metrics:
      fair_gobK/              — PE⊕gobK, shared bbox from that PE (train recipe)
      fair_vggtK/             — PE⊕vggtK, shared bbox from that PE (train=infer)
      cross_gobGT_vggtK_pe/   — GT boxed w/ gobK stats; PE = vggtK own bbox

  gt_norm_ablation/
    GT-box recipes (A/B/D: gt.ply + vggt_pred_vggtK.ply; C also gt_depth_gobK.ply):
      A_cross_train/          — train cross: mesh/mean(GT depth) + Hunyuan from VGGT⊕gobK;
                                VGGT PE = vggtK own Hunyuan (no /mean; ≡ /mean+bbox)
      B_vggt_gobK_direct/     — mesh Hunyuan from raw VGGT⊕gobK (/mean redundant);
                                VGGT = vggtK own Hunyuan
      C_gt_depth_direct/      — mesh Hunyuan from raw GT depth⊕gobK (baseline)
      C_gt_depth_filter/      — erode FG 1px only, then standard Hunyuan AABB
                                (+ gt_depth_used.ply / gt_depth_discarded.ply)
      C_gt_depth_zrobust/     — unfiltered cloud; Hunyuan xy=min/max, z lo=p1 hi=max
                                (near-tail only; + used/discarded GT-depth PLYs)
      C_gt_depth_filter_zrobust/ — erode 1px, then z lo=p0.5 hi=max on remaining
                                (few leftover extrema; + used/discarded PLYs)
      D_mesh_own_indep/       — mesh own Hunyuan vs VGGT⊕vggtK own Hunyuan (naive)
    Each folder also writes patch_centers.ply + discarded_centers.ply (vggtK own bbox).

  mv_frame0/   (when --num_views>=1; primary visual multi-view check)
    All clouds in reference camera (first view). PE = multi-view VGGT union,
    own Hunyuan (no /mean). Cross GT = mesh/mean(union GT depth) + Hunyuan from
    VGGT⊕gobK after /mean, with gobK×{GT,VGGT} extrinsics compared side-by-side.

Example:
  python export_camera_frame_debug.py \\
    --data_dir .../furniture_351/train \\
    --gobjaverse_render_root /export/home/nathan/datasets \\
    --max_items 1 --num_views 2 \\
    --output_dir runs/debug_cam_frame
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from hy3dgen.shapegen.gs_export import export_xyz_pointcloud_ply
from hy3dgen.shapegen.pc_render_dataset import build_surface_render_dataset, collate_surface_render
from hy3dgen.shapegen.vggt_context import (
    VGGTContextBuilder,
    depth_map_to_cam_points,
    patch_centers_from_depth,
    preprocess_rgb_for_vggt,
)
from hy3dgen.shapegen.cam_align import (
    merge_unprojected_to_cam_ref,
    vggt_extrinsic_to_c2w,
    vggt_world_to_cam0,
)
from hy3dgen.shapegen.gobjaverse_gt import parse_view_indices
from train_gs_ae import load_experiment_manifest, resolve_category_ids

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ScaleSpec = Tuple[str, str, object]


def _scale_specs(percentiles: List[float]) -> List[ScaleSpec]:
    # Default / primary: z_min. median + percentiles kept as explicit ablations.
    # (mean removed — use z_min everywhere else.)
    specs: List[ScaleSpec] = [
        ("zmin", "nearest_z_min", lambda z: float(np.min(z))),
        ("median", "median", lambda z: float(np.median(z))),
    ]
    for p in percentiles:
        tag = f"p{int(p)}" if float(p).is_integer() else f"p{p:g}"
        specs.append((tag, f"percentile_{p:g}", lambda z, pp=p: float(np.percentile(z, pp))))
    return specs


def _zmin_fn(z: np.ndarray) -> float:
    return float(np.min(z))


def normalize_by_zmin(xyz) -> Tuple[np.ndarray, float]:
    """Isotropic ``p' = p / z_min`` so nearest FG depth → 1. Returns (pts, z_min)."""
    pts = _as_xyz_np(xyz).astype(np.float32)
    s = _depth_scale(pts[:, 2], _zmin_fn)
    return (pts / s).astype(np.float32), float(s)


def normalize_by_mean(xyz) -> Tuple[np.ndarray, float]:
    """Isotropic ``p' = p / mean(z)``. Returns (pts, mean_z)."""
    pts = _as_xyz_np(xyz).astype(np.float32)
    s = _depth_scale(pts[:, 2], lambda z: float(np.mean(z)))
    return (pts / s).astype(np.float32), float(s)


def independent_canonicalize(
    *clouds: np.ndarray,
    eps: float = 1e-6,
    scale: str = "rms",
    fill: float = 0.9999,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    """Per-cloud ``(p-μ)/s``.

    ``scale``: ``rms``, ``maxabs``, or ``bbox`` (Hunyuan AABB max-side → ~[-fill, fill]).
    """
    outs: List[np.ndarray] = []
    mus: List[np.ndarray] = []
    scales: List[float] = []
    for c in clouds:
        pts = _as_xyz_np(c).astype(np.float64)
        mu, s = _canonicalize_mu_s(pts, scale=scale, fill=fill, eps=eps)
        outs.append(((pts - mu[None, :]) / s).astype(np.float32))
        mus.append(mu)
        scales.append(s)
    return outs, mus, scales


def _canonicalize_mu_s(
    pts: np.ndarray,
    *,
    scale: str = "rms",
    fill: float = 0.9999,
    eps: float = 1e-6,
    z_p_lo: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """Return (μ, s) for isotropic canonicalize. ``bbox`` matches Hunyuan normalize_mesh."""
    pts = _as_xyz_np(pts).astype(np.float64)
    if pts.shape[0] == 0:
        return np.zeros(3, dtype=np.float64), 1.0
    if scale in ("bbox", "bbox_zrobust"):
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        if scale == "bbox_zrobust" and pts.shape[0] >= 8:
            # Near-plane flyers only: raise lo_z to z_p_lo, keep far = max and xy = AABB.
            lo[2] = float(np.percentile(pts[:, 2], float(z_p_lo)))
        mu = 0.5 * (lo + hi)
        side = float((hi - lo).max())
        # Hunyuan: p' = (p - center) * (2 * fill / side)  ⇒  s = side / (2 * fill)
        s = side / max(2.0 * float(fill), eps)
    else:
        mu = pts.mean(axis=0)
        centered = pts - mu[None, :]
        if scale == "maxabs":
            s = float(np.max(np.abs(centered)))
        else:
            s = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    if not np.isfinite(s) or s < eps:
        s = 1.0
    return mu, float(s)


def shared_canonicalize_from_ref(
    pts_ref: np.ndarray,
    *clouds: np.ndarray,
    eps: float = 1e-6,
    scale: str = "rms",
    fill: float = 0.9999,
    z_p_lo: float = 1.0,
) -> Tuple[List[np.ndarray], np.ndarray, float]:
    """Shared ``p' = (p - μ) / s`` with μ,s from ``pts_ref``.

    ``scale``: ``rms``, ``maxabs``, ``bbox``, or ``bbox_zrobust``
    (z lo=``z_p_lo`` percentile, hi=max).
    """
    mu, s = _canonicalize_mu_s(
        pts_ref, scale=scale, fill=fill, eps=eps, z_p_lo=z_p_lo
    )
    out: List[np.ndarray] = []
    for c in clouds:
        pts = _as_xyz_np(c).astype(np.float64)
        out.append(((pts - mu[None, :]) / s).astype(np.float32))
    return out, mu, s


def resize_depth_nearest(depth: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    """Nearest resize depth HxW → hw=(H',W')."""
    h, w = depth.shape[:2]
    ht, wt = hw
    if (h, w) == (ht, wt):
        return depth.astype(np.float32)
    t = torch.from_numpy(depth.astype(np.float32))[None, None]
    out = torch.nn.functional.interpolate(t, size=(ht, wt), mode="nearest")
    return out[0, 0].numpy()


def depth_scale_shift_align(
    pred: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
) -> Tuple[float, float, np.ndarray]:
    """Least-squares ``gt ≈ a * pred + b`` on valid pixels. Returns a, b, aligned_pred."""
    m = valid & np.isfinite(pred) & np.isfinite(gt) & (pred > 1e-6) & (gt > 1e-6)
    if int(m.sum()) < 32:
        return 1.0, 0.0, pred.astype(np.float32)
    p = pred[m].astype(np.float64).reshape(-1)
    g = gt[m].astype(np.float64).reshape(-1)
    A = np.stack([p, np.ones_like(p)], axis=1)
    try:
        x, _, _, _ = np.linalg.lstsq(A, g, rcond=None)
        a, b = float(x[0]), float(x[1])
    except np.linalg.LinAlgError:
        a, b = 1.0, 0.0
    if not np.isfinite(a) or abs(a) < 1e-8:
        a, b = 1.0, 0.0
    aligned = (a * pred.astype(np.float64) + b).astype(np.float32)
    return a, b, aligned


def _fg_mask_from_depth_rgb(
    depth_map: np.ndarray,
    rgb_np: Optional[np.ndarray],
) -> np.ndarray:
    valid = np.isfinite(depth_map) & (depth_map > 1e-6)
    if rgb_np is not None and rgb_np.shape[:2] == depth_map.shape[:2]:
        white = (
            (rgb_np[..., 0] > 0.97)
            & (rgb_np[..., 1] > 0.97)
            & (rgb_np[..., 2] > 0.97)
        )
        valid = valid & ~white
    return valid


def _erode_mask(mask: np.ndarray, iters: int = 2) -> np.ndarray:
    """4-neighbor binary erosion (kills 1-pixel silhouette mixels per iter)."""
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


def _filter_gt_depth_mask(
    valid: np.ndarray,
    *,
    erode_iters: int = 1,
) -> np.ndarray:
    """Simple flyer filter: 4-neighbor erode only (no z-percentile — that clips real near corners)."""
    return _erode_mask(valid, iters=erode_iters)


def _depth_scale(z: np.ndarray, reduce_fn, *, eps: float = 1e-6) -> float:
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    z = z[np.isfinite(z) & (z > eps)]
    if z.size == 0:
        return 1.0
    s = float(reduce_fn(z))
    if not np.isfinite(s) or s <= eps:
        return 1.0
    return s


def _nn_stats(src: torch.Tensor, tgt: torch.Tensor) -> Tuple[float, float]:
    if src.numel() == 0 or tgt.numel() == 0:
        return float("nan"), float("nan")
    d = torch.cdist(src.unsqueeze(0), tgt.unsqueeze(0)).squeeze(0).min(dim=1).values
    return float(d.mean()), float(d.median())


def _nn_dists(
    src: np.ndarray,
    tgt: np.ndarray,
    *,
    max_n: int = 4000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Per-point NN distances from src→tgt (subsample src if needed)."""
    a = _as_xyz_np(src).astype(np.float32)
    b = _as_xyz_np(tgt).astype(np.float32)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if a.shape[0] > max_n:
        rs = rng if rng is not None else np.random.default_rng(0)
        a = a[rs.choice(a.shape[0], max_n, replace=False)]
    if b.shape[0] > 20000:
        rs = rng if rng is not None else np.random.default_rng(0)
        b = b[rs.choice(b.shape[0], 20000, replace=False)]
    d = torch.cdist(
        torch.from_numpy(a).unsqueeze(0),
        torch.from_numpy(b).unsqueeze(0),
    ).squeeze(0).min(dim=1).values
    return d.numpy()


def alignment_metrics(
    pe: np.ndarray,
    gt: np.ndarray,
    *,
    max_n: int = 4000,
    overlap_thresh: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Chamfer / NN / bbox metrics between PE (or centres) and GT mesh."""
    nan = float("nan")
    pe_np = _as_xyz_np(pe).astype(np.float32)
    gt_np = _as_xyz_np(gt).astype(np.float32)
    out: Dict[str, float] = {
        "n_pe": float(pe_np.shape[0]),
        "n_gt": float(gt_np.shape[0]),
        "chamfer_mean": nan,
        "chamfer_med": nan,
        "nn_pe2gt_mean": nan,
        "nn_pe2gt_med": nan,
        "nn_gt2pe_mean": nan,
        "nn_gt2pe_med": nan,
        "hausdorff_p95": nan,
        "overlap_pe_frac": nan,
        "overlap_gt_frac": nan,
        "bbox_side_pe": nan,
        "bbox_side_gt": nan,
        "bbox_side_ratio": nan,
        "bbox_center_dist": nan,
        "xy_span_pe": nan,
        "xy_span_gt": nan,
        "xy_span_ratio": nan,
        "mean_z_pe": nan,
        "mean_z_gt": nan,
        "mean_z_diff": nan,
    }
    if pe_np.shape[0] == 0 or gt_np.shape[0] == 0:
        return out

    d_pg = _nn_dists(pe_np, gt_np, max_n=max_n, rng=rng)
    d_gp = _nn_dists(gt_np, pe_np, max_n=max_n, rng=rng)
    out["nn_pe2gt_mean"] = float(np.mean(d_pg))
    out["nn_pe2gt_med"] = float(np.median(d_pg))
    out["nn_gt2pe_mean"] = float(np.mean(d_gp))
    out["nn_gt2pe_med"] = float(np.median(d_gp))
    out["chamfer_mean"] = 0.5 * (out["nn_pe2gt_mean"] + out["nn_gt2pe_mean"])
    out["chamfer_med"] = 0.5 * (out["nn_pe2gt_med"] + out["nn_gt2pe_med"])
    out["hausdorff_p95"] = float(
        max(np.percentile(d_pg, 95), np.percentile(d_gp, 95))
    )
    out["overlap_pe_frac"] = float(np.mean(d_pg < overlap_thresh))
    out["overlap_gt_frac"] = float(np.mean(d_gp < overlap_thresh))

    st_pe = cloud_stats(pe_np)
    st_gt = cloud_stats(gt_np)
    pe_min, pe_max = pe_np.min(0), pe_np.max(0)
    gt_min, gt_max = gt_np.min(0), gt_np.max(0)
    side_pe = float((pe_max - pe_min).max())
    side_gt = float((gt_max - gt_min).max())
    c_pe = 0.5 * (pe_min + pe_max)
    c_gt = 0.5 * (gt_min + gt_max)
    out["bbox_side_pe"] = side_pe
    out["bbox_side_gt"] = side_gt
    out["bbox_side_ratio"] = side_pe / max(side_gt, 1e-6)
    out["bbox_center_dist"] = float(np.linalg.norm(c_pe - c_gt))
    out["xy_span_pe"] = st_pe["xy_span"]
    out["xy_span_gt"] = st_gt["xy_span"]
    out["xy_span_ratio"] = st_pe["xy_span"] / max(st_gt["xy_span"], 1e-6)
    out["mean_z_pe"] = st_pe["mean_z"]
    out["mean_z_gt"] = st_gt["mean_z"]
    out["mean_z_diff"] = st_pe["mean_z"] - st_gt["mean_z"]
    return out


def _apply_mean_then_bbox(
    pts,
    mean_z: float,
    mu: np.ndarray,
    s: float,
) -> np.ndarray:
    if pts is None or len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    p = _as_xyz_np(pts).astype(np.float64) / max(float(mean_z), 1e-6)
    return ((p - mu[None, :]) / max(float(s), 1e-6)).astype(np.float32)


def _write_k_bakeoff_experiment(
    *,
    stem: str,
    exp_dir: Path,
    tag: str,
    label: str,
    gt_mesh: np.ndarray,
    gt_rgb: Optional[torch.Tensor],
    pe_fg: np.ndarray,
    pe_cols: Optional[np.ndarray],
    cen_keep: np.ndarray,
    cen_disc: np.ndarray,
    max_cloud_points: int,
    rng: np.random.Generator,
    bakeoff_rows: List[Dict],
    extra: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Export one K-bakeoff recipe folder + rich alignment metrics (centres vs mesh)."""
    exp_dir.mkdir(parents=True, exist_ok=True)
    export_xyz_pointcloud_ply(gt_mesh, exp_dir / "gt_mesh.ply", colors=gt_rgb)

    def _dump(name: str, pts, cols=None, rgb=None):
        arr = _as_xyz_np(pts).astype(np.float32)
        if arr.shape[0] == 0:
            return
        pe, ce, _ = _subsample_fg(arr, cols, max_cloud_points, rng)
        kw = {}
        if ce is not None:
            kw["colors"] = ce
        elif rgb is not None:
            kw["rgb"] = rgb
        export_xyz_pointcloud_ply(pe, exp_dir / f"{name}.ply", **kw)

    _dump("vggt_fg", pe_fg, pe_cols)
    _dump("patch_centers", cen_keep, rgb=(32, 200, 64))
    _dump("discarded_centers", cen_disc, rgb=(220, 40, 40))

    m_cen = alignment_metrics(cen_keep, gt_mesh, rng=rng)
    m_fg = alignment_metrics(pe_fg, gt_mesh, rng=rng)
    logger.info(
        "%s %s: centres Chamfer=%.4f (pe2gt=%.4f) overlap@0.05=%.2f "
        "bbox_side_ratio=%.3f",
        stem,
        tag,
        m_cen["chamfer_mean"],
        m_cen["nn_pe2gt_mean"],
        m_cen["overlap_pe_frac"],
        m_cen["bbox_side_ratio"],
    )
    with open(exp_dir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"mesh={stem}\nmethod={tag}\n{label}\n\n")
        f.write("[patch_centers_vs_gt_mesh]\n")
        for k, v in m_cen.items():
            f.write(f"  {k}={v}\n")
        f.write("\n[vggt_fg_vs_gt_mesh]\n")
        for k, v in m_fg.items():
            f.write(f"  {k}={v}\n")
        if extra:
            f.write("\n[extra]\n")
            for k, v in extra.items():
                f.write(f"  {k}={v}\n")

    row = {
        "mesh": stem,
        "method": tag,
        "chamfer_mean": m_cen["chamfer_mean"],
        "chamfer_med": m_cen["chamfer_med"],
        "nn_pe2gt_mean": m_cen["nn_pe2gt_mean"],
        "nn_gt2pe_mean": m_cen["nn_gt2pe_mean"],
        "hausdorff_p95": m_cen["hausdorff_p95"],
        "overlap_pe_frac": m_cen["overlap_pe_frac"],
        "bbox_side_ratio": m_cen["bbox_side_ratio"],
        "bbox_center_dist": m_cen["bbox_center_dist"],
        "xy_span_ratio": m_cen["xy_span_ratio"],
        "mean_z_diff": m_cen["mean_z_diff"],
        "fg_chamfer_mean": m_fg["chamfer_mean"],
    }
    bakeoff_rows.append(row)
    return m_cen


def _as_xyz_np(xyz) -> np.ndarray:
    if isinstance(xyz, torch.Tensor):
        xyz = xyz.detach().float().cpu().numpy()
    return np.asarray(xyz, dtype=np.float64).reshape(-1, 3)


def cloud_stats(xyz) -> Dict[str, float]:
    pts = _as_xyz_np(xyz)
    if pts.size == 0:
        nan = float("nan")
        return {
            "n": 0,
            "mean_x": nan,
            "mean_y": nan,
            "mean_z": nan,
            "median_z": nan,
            "bb_cx": nan,
            "bb_cy": nan,
            "bb_cz": nan,
            "z_min": nan,
            "z_max": nan,
            "xy_span": nan,
        }
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    mean = pts.mean(axis=0)
    return {
        "n": float(pts.shape[0]),
        "mean_x": float(mean[0]),
        "mean_y": float(mean[1]),
        "mean_z": float(mean[2]),
        "median_z": float(np.median(pts[:, 2])),
        "bb_cx": float(0.5 * (mn[0] + mx[0])),
        "bb_cy": float(0.5 * (mn[1] + mx[1])),
        "bb_cz": float(0.5 * (mn[2] + mx[2])),
        "z_min": float(mn[2]),
        "z_max": float(mx[2]),
        "xy_span": float(max(mx[0] - mn[0], mx[1] - mn[1])),
    }


def _log_cloud_stats(prefix: str, name: str, xyz, *, note: str = "") -> Dict[str, float]:
    s = cloud_stats(xyz)
    extra = f"  ({note})" if note else ""
    logger.info(
        "%s [%s] n=%d  mean_xyz=(%.6f, %.6f, %.6f)  median_z=%.6f  "
        "bb_center=(%.6f, %.6f, %.6f)  z=[%.4f, %.4f]  xy_span=%.4f%s",
        prefix,
        name,
        int(s["n"]),
        s["mean_x"],
        s["mean_y"],
        s["mean_z"],
        s["median_z"],
        s["bb_cx"],
        s["bb_cy"],
        s["bb_cz"],
        s["z_min"],
        s["z_max"],
        s["xy_span"],
        extra,
    )
    return s


def gobjaverse_K_for_vggt_resolution(
    intrinsics_fxfycxcy: torch.Tensor,
    rgb: torch.Tensor,
    *,
    depth_hw: Tuple[int, int],
    img_size: int,
) -> Tuple[float, float, float, float]:
    """Map G-Objaverse (fx,fy,cx,cy) from native RGB size → VGGT depth grid."""
    fx, fy, cx, cy = [float(x) for x in intrinsics_fxfycxcy.reshape(-1)[:4]]
    _, scale_y, scale_x, pad_top, pad_left = preprocess_rgb_for_vggt(
        rgb.float().clamp(0, 1), target_size=img_size
    )
    fx2 = fx * scale_x
    fy2 = fy * scale_y
    cx2 = cx * scale_x + pad_left
    cy2 = cy * scale_y + pad_top
    hd, wd = depth_hw
    # Sanity: preprocess pads to img_size×img_size when possible.
    if (hd, wd) != (img_size, img_size):
        logger.warning(
            "VGGT depth is %dx%d but preprocess target is %d; "
            "gobK may be slightly off.",
            hd,
            wd,
            img_size,
        )
    return fx2, fy2, cx2, cy2


def _subsample_fg(
    pts: np.ndarray,
    cols: Optional[np.ndarray],
    max_n: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Returns (pts_export, cols_export, index into full FG)."""
    n = pts.shape[0]
    if n <= max_n:
        return pts, cols, np.arange(n)
    sel = rng.choice(n, max_n, replace=False)
    cols_out = cols[sel] if cols is not None else None
    return pts[sel], cols_out, sel


def umeyama_similarity(
    src: np.ndarray,
    dst: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Similarity ``dst ≈ s * (src @ R.T) + t`` (row vectors). Returns s, R(3x3), t(3,)."""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    n = src.shape[0]
    if n < 3:
        return 1.0, np.eye(3), np.zeros(3)
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    var_s = float((src_c ** 2).sum() / n)
    # Σ = (1/n) Σ (d−μd)(s−μs)^T  (column-vector Umeyama)
    cov = (dst_c.T @ src_c) / n
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1.0
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / max(var_s, 1e-12))
    if not np.isfinite(s) or s <= 1e-8:
        s = 1.0
        R = np.eye(3)
    t = mu_d - s * (R @ mu_s)
    return s, R.astype(np.float64), t.astype(np.float64)


def apply_similarity(
    pts: np.ndarray,
    s: float,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    # p' = s (R p) + t  with column R; row form: s * pts @ R.T + t
    return (s * (pts @ R.T) + t[None, :]).astype(np.float32)


def similarity_matrix(s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = s * R
    M[:3, 3] = t
    return M


def estimate_gt_to_vggt_similarity(
    gt: np.ndarray,
    vggt: np.ndarray,
    *,
    n_corr: int = 4000,
    n_iters: int = 3,
    rng: np.random.Generator,
    mode: str = "full",
    trim_percentile: float = 30.0,
) -> Tuple[float, np.ndarray, np.ndarray, float, int]:
    """NN-correspondence Umeyama (GT→VGGT), iterated.

    modes:
      full    — use all sampled GT→NN pairs
      trim    — keep only pairs with NN dist ≤ trim_percentile of distances
      mutual  — keep pairs that are mutual nearest neighbors

    Returns s, R, t, mean_nn, n_pairs_used.
    """
    gt = np.asarray(gt, dtype=np.float64).reshape(-1, 3)
    vggt = np.asarray(vggt, dtype=np.float64).reshape(-1, 3)
    if gt.shape[0] < 3 or vggt.shape[0] < 3:
        return 1.0, np.eye(3), np.zeros(3), float("nan"), 0

    vggt_t = torch.from_numpy(vggt.astype(np.float32))
    chunk = 8192
    s, R, t = 1.0, np.eye(3), np.zeros(3)
    nn_mean = float("nan")
    n_used = 0

    for _ in range(n_iters):
        if gt.shape[0] > n_corr:
            sel = rng.choice(gt.shape[0], n_corr, replace=False)
        else:
            sel = np.arange(gt.shape[0])
        src0 = gt[sel]
        warped = apply_similarity(src0, s, R, t)
        warped_t = torch.from_numpy(np.asarray(warped, dtype=np.float32))

        nn_idx_parts = []
        nn_dist_parts = []
        for i0 in range(0, warped_t.shape[0], chunk):
            d = torch.cdist(warped_t[i0 : i0 + chunk], vggt_t)
            nn_dist_parts.append(d.min(dim=1).values)
            nn_idx_parts.append(d.argmin(dim=1))
        nn_idx = torch.cat(nn_idx_parts, dim=0).numpy()
        nn_dist = torch.cat(nn_dist_parts, dim=0).numpy()
        dst = vggt[nn_idx]
        keep = np.ones(src0.shape[0], dtype=bool)

        if mode == "trim":
            thr = float(np.percentile(nn_dist, trim_percentile))
            keep = nn_dist <= thr
        elif mode == "mutual":
            # VGGT → warped NN
            mut = np.zeros(src0.shape[0], dtype=bool)
            # For each src, check if that VGGT point's NN among warped is this src
            # Subsample vggt targets that were matched to limit cost
            uniq = np.unique(nn_idx)
            # Build NN from those vggt points back to warped
            back_nn = np.full(vggt.shape[0], -1, dtype=np.int64)
            vg_sub = torch.from_numpy(vggt[uniq].astype(np.float32))
            for i0 in range(0, vg_sub.shape[0], chunk):
                d = torch.cdist(vg_sub[i0 : i0 + chunk], warped_t)
                back_nn[uniq[i0 : i0 + chunk]] = d.argmin(dim=1).numpy()
            for i, j in enumerate(nn_idx):
                mut[i] = back_nn[j] == i
            keep = mut
        elif mode != "full":
            raise ValueError(f"Unknown Umeyama mode: {mode}")

        if int(keep.sum()) < 3:
            # Fall back to full if filter too aggressive
            keep = np.ones(src0.shape[0], dtype=bool)
        src_k = src0[keep]
        dst_k = dst[keep]
        s, R, t = umeyama_similarity(src_k, dst_k)
        err = np.linalg.norm(
            apply_similarity(src_k, s, R, t).astype(np.float64) - dst_k, axis=1
        )
        nn_mean = float(err.mean())
        n_used = int(src_k.shape[0])
    return s, R, t, nn_mean, n_used


def gt_visible_on_vggt_mask(
    gt_cam: np.ndarray,
    *,
    gob_fx: float,
    gob_fy: float,
    gob_cx: float,
    gob_cy: float,
    scale_x: float,
    scale_y: float,
    pad_left: int,
    pad_top: int,
    pix_valid: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """True for GT points that project onto VGGT FG pixels (gobK → VGGT grid)."""
    pts = np.asarray(gt_cam, dtype=np.float64).reshape(-1, 3)
    h, w = pix_valid.shape
    z = np.maximum(pts[:, 2], eps)
    u_n = gob_fx * (pts[:, 0] / z) + gob_cx
    v_n = gob_fy * (pts[:, 1] / z) + gob_cy
    u_v = u_n * scale_x + pad_left
    v_v = v_n * scale_y + pad_top
    ui = np.rint(u_v).astype(np.int64)
    vi = np.rint(v_v).astype(np.int64)
    inb = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    keep = np.zeros(pts.shape[0], dtype=bool)
    if inb.any():
        keep[inb] = pix_valid[vi[inb], ui[inb]]
    return keep


def _export_align_sim_variant(
    *,
    stem: str,
    obj_dir: Path,
    tag: str,
    label: str,
    gt_src: np.ndarray,
    vggt_src: np.ndarray,
    gt_rgb: Optional[torch.Tensor],
    cols_full: np.ndarray,
    mode: str,
    rng: np.random.Generator,
    max_cloud_points: int,
    summary_rows: List[Dict],
    extra_stats: Optional[Dict[str, float]] = None,
    note: str = "",
    trim_percentile: float = 30.0,
) -> None:
    s_sim, R_sim, t_sim, nn_fit, n_used = estimate_gt_to_vggt_similarity(
        gt_src,
        vggt_src,
        n_corr=4000,
        n_iters=3,
        rng=rng,
        mode=mode,
        trim_percentile=trim_percentile,
    )
    gt_sim = apply_similarity(gt_src, s_sim, R_sim, t_sim)
    M = similarity_matrix(s_sim, R_sim, t_sim)
    mat_txt = note
    if mat_txt and not mat_txt.endswith("\n"):
        mat_txt += "\n"
    mat_txt += (
        f"mode={mode}\nn_corr_used={n_used}\n"
        "transform_4x4 (GT→VGGT, row-major; p' = s R p + t):\n"
    )
    for row in M:
        mat_txt += "  " + " ".join(f"{v:.12f}" for v in row) + "\n"
    logger.info(
        "%s %s (%s): s=%.6f t=(%.4f, %.4f, %.4f) fit_nn=%.4f n_pairs=%d det(R)=%.4f",
        stem,
        tag,
        mode,
        s_sim,
        t_sim[0],
        t_sim[1],
        t_sim[2],
        nn_fit,
        n_used,
        float(np.linalg.det(R_sim)),
    )
    stats = {
        "similarity_scale": s_sim,
        "tx": float(t_sim[0]),
        "ty": float(t_sim[1]),
        "tz": float(t_sim[2]),
        "fit_nn_mean": nn_fit,
        "n_corr_used": float(n_used),
    }
    if extra_stats:
        stats.update(extra_stats)
    _write_pair_export(
        stem=stem,
        method_dir=obj_dir / tag,
        tag=tag,
        label=label,
        gt_xyz=gt_sim,
        gt_rgb=gt_rgb,
        other_xyz=vggt_src,
        other_cols=cols_full,
        other_name="vggt_depth_cloud",
        max_cloud_points=max_cloud_points,
        rng=rng,
        summary_rows=summary_rows,
        extra_stats=stats,
        extra_txt=mat_txt,
    )


def fov_warp_gt_to_vggt_k(
    gt_cam: np.ndarray,
    *,
    gob_fx: float,
    gob_fy: float,
    gob_cx: float,
    gob_cy: float,
    scale_x: float,
    scale_y: float,
    pad_left: int,
    pad_top: int,
    vggt_fx: float,
    vggt_fy: float,
    vggt_cx: float,
    vggt_cy: float,
    eps: float = 1e-6,
) -> np.ndarray:
    """Project GT with gobK (native), map pixels to VGGT grid, unproject with VGGT K.

    Keeps z; changes lateral size to VGGT's FoV. Can introduce depth-dependent
    shear if principal points / pads disagree (``x' = αx + βz``).
    """
    pts = np.asarray(gt_cam, dtype=np.float64).reshape(-1, 3)
    z = np.maximum(pts[:, 2], eps)
    u_n = gob_fx * (pts[:, 0] / z) + gob_cx
    v_n = gob_fy * (pts[:, 1] / z) + gob_cy
    u_v = u_n * scale_x + pad_left
    v_v = v_n * scale_y + pad_top
    x = (u_v - vggt_cx) * z / max(vggt_fx, eps)
    y = (v_v - vggt_cy) * z / max(vggt_fy, eps)
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def fx_scale_gt_to_vggt_k(
    gt_cam: np.ndarray,
    *,
    gob_fx: float,
    gob_fy: float,
    scale_x: float,
    scale_y: float,
    vggt_fx: float,
    vggt_fy: float,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, float, float]:
    """Shear-free FoV match: ``x' = x * (fx_gob@518 / fx_vggt)`` (same for y), ``z'=z``.

    No pixel round-trip / principal-point terms — parallel lines stay parallel.
    Returns (pts, sx, sy).
    """
    pts = np.asarray(gt_cam, dtype=np.float64).reshape(-1, 3).copy()
    sx = (gob_fx * scale_x) / max(vggt_fx, eps)
    sy = (gob_fy * scale_y) / max(vggt_fy, eps)
    pts[:, 0] *= sx
    pts[:, 1] *= sy
    return pts.astype(np.float32), float(sx), float(sy)


def _write_pair_export(
    *,
    stem: str,
    method_dir: Path,
    tag: str,
    label: str,
    gt_xyz,
    gt_rgb: Optional[torch.Tensor],
    other_xyz,
    other_cols: Optional[np.ndarray],
    other_name: str,
    max_cloud_points: int,
    rng: np.random.Generator,
    summary_rows: List[Dict],
    extra_stats: Optional[Dict[str, float]] = None,
    extra_txt: str = "",
    cen_keep: Optional[np.ndarray] = None,
    cen_disc: Optional[np.ndarray] = None,
) -> None:
    method_dir.mkdir(parents=True, exist_ok=True)
    gt_np = _as_xyz_np(gt_xyz).astype(np.float32)
    oth_np = _as_xyz_np(other_xyz).astype(np.float32)
    oth_exp, cols_exp, _ = _subsample_fg(oth_np, other_cols, max_cloud_points, rng)

    export_xyz_pointcloud_ply(gt_np, method_dir / "gt.ply", colors=gt_rgb)
    ply_kwargs = {}
    if cols_exp is not None:
        ply_kwargs["colors"] = cols_exp
    export_xyz_pointcloud_ply(oth_exp, method_dir / f"{other_name}.ply", **ply_kwargs)

    st_cen = None
    st_disc = None
    nn_cen, nn_cen_med = float("nan"), float("nan")
    if cen_keep is not None:
        ck = _as_xyz_np(cen_keep).astype(np.float32)
        if ck.shape[0] > 0:
            export_xyz_pointcloud_ply(
                ck, method_dir / "patch_centers.ply", rgb=(32, 200, 64)
            )
            st_cen = _log_cloud_stats(stem, f"{tag}/patch_centers", ck)
            nn_cen, nn_cen_med = _nn_stats(
                torch.from_numpy(ck[: min(4000, ck.shape[0])]),
                torch.from_numpy(gt_np),
            )
    if cen_disc is not None:
        cd = _as_xyz_np(cen_disc).astype(np.float32)
        if cd.shape[0] > 0:
            export_xyz_pointcloud_ply(
                cd, method_dir / "discarded_centers.ply", rgb=(220, 40, 40)
            )
            st_disc = _log_cloud_stats(stem, f"{tag}/discarded_centers", cd)

    nn_mean, nn_med = _nn_stats(
        torch.from_numpy(oth_exp[: min(4000, oth_exp.shape[0])]),
        torch.from_numpy(gt_np),
    )
    nn_gt_mean, nn_gt_med = _nn_stats(
        torch.from_numpy(gt_np[: min(4000, gt_np.shape[0])]),
        torch.from_numpy(oth_np[: min(20000, oth_np.shape[0])]),
    )
    logger.info(
        "%s %s (%s): other→GT NN mean=%.4f med=%.4f | GT→other mean=%.4f med=%.4f"
        "%s",
        stem,
        tag,
        label,
        nn_mean,
        nn_med,
        nn_gt_mean,
        nn_gt_med,
        (
            f" | centres→GT mean={nn_cen:.4f}"
            if st_cen is not None
            else ""
        ),
    )
    st_gt = _log_cloud_stats(stem, f"{tag}/gt", gt_np)
    st_oth = _log_cloud_stats(stem, f"{tag}/{other_name}", oth_exp)

    with open(method_dir / "stats.txt", "w", encoding="utf-8") as f:
        f.write(f"mesh={stem}\nmethod={tag} ({label})\n")
        f.write(f"nn_other_to_gt_mean={nn_mean:.6f}\nn_other_to_gt_med={nn_med:.6f}\n")
        f.write(f"nn_gt_to_other_mean={nn_gt_mean:.6f}\nn_gt_to_other_med={nn_gt_med:.6f}\n")
        if st_cen is not None:
            f.write(f"nn_centres_to_gt_mean={nn_cen:.6f}\n")
            f.write(f"nn_centres_to_gt_med={nn_cen_med:.6f}\n")
        if extra_stats:
            for k, v in extra_stats.items():
                f.write(f"{k}={v}\n")
        if extra_txt:
            f.write(extra_txt)
            if not extra_txt.endswith("\n"):
                f.write("\n")
        for name, st in (("gt", st_gt), (other_name, st_oth)):
            f.write(f"\n[{name}]\n")
            for k, v in st.items():
                f.write(f"  {k}={v}\n")
        if st_cen is not None:
            f.write("\n[patch_centers]\n")
            for k, v in st_cen.items():
                f.write(f"  {k}={v}\n")
        if st_disc is not None:
            f.write("\n[discarded_centers]\n")
            for k, v in st_disc.items():
                f.write(f"  {k}={v}\n")

    summary_rows.append(
        {
            "mesh": stem,
            "method": tag,
            "nn_mean": nn_gt_mean,
            "nn_med": nn_gt_med,
            "gt_xy_span": st_gt["xy_span"],
            "vggt_xy_span": st_oth["xy_span"],
        }
    )


def _apply_mu_s(pts, mu: np.ndarray, s: float) -> np.ndarray:
    pts = _as_xyz_np(pts).astype(np.float64)
    if pts.shape[0] == 0:
        return pts.astype(np.float32)
    return ((pts - np.asarray(mu, dtype=np.float64)[None, :]) / float(s)).astype(
        np.float32
    )


def _append_gt_depth_ply(
    *,
    stem: str,
    tag: str,
    method_dir: Path,
    gtd_xyz,
    gtd_cols: Optional[np.ndarray] = None,
    max_cloud_points: int,
    rng: np.random.Generator,
    name: str = "gt_depth_gobK",
    rgb: Optional[Tuple[int, int, int]] = None,
) -> None:
    """Write a named GT-depth PLY and append its stats section."""
    gtd = _as_xyz_np(gtd_xyz).astype(np.float32)
    if gtd.shape[0] == 0:
        with open(method_dir / "stats.txt", "a", encoding="utf-8") as f:
            f.write(f"\n[{name}]\n  n=0\n")
        return
    gtd_exp, cols_exp, _ = _subsample_fg(gtd, gtd_cols, max_cloud_points, rng)
    kw: Dict[str, object] = {}
    if rgb is not None:
        kw["rgb"] = rgb
    elif cols_exp is not None:
        kw["colors"] = cols_exp
    export_xyz_pointcloud_ply(gtd_exp, method_dir / f"{name}.ply", **kw)
    st = _log_cloud_stats(stem, f"{tag}/{name}", gtd)
    with open(method_dir / "stats.txt", "a", encoding="utf-8") as f:
        f.write(f"\n[{name}]\n")
        for k, v in st.items():
            f.write(f"  {k}={v}\n")


def export_scaled_method(
    *,
    stem: str,
    method_dir: Path,
    tag: str,
    label: str,
    gt_cam: torch.Tensor,
    gt_rgb: Optional[torch.Tensor],
    pts_full: np.ndarray,
    cols_full: np.ndarray,
    centers_full: np.ndarray,
    centers_keep: np.ndarray,
    reduce_fn,
    max_cloud_points: int,
    rng: np.random.Generator,
    summary_rows: List[Dict],
    sanity: str = "",
) -> None:
    """Write scale_* PLYs + stats for one (cloud, z_stat) setup."""
    method_dir.mkdir(parents=True, exist_ok=True)
    z_v = pts_full[:, 2]
    z_g = gt_cam.numpy()[:, 2]
    s_v = _depth_scale(z_v, reduce_fn)
    s_g = _depth_scale(z_g, reduce_fn)

    gt_s = gt_cam / s_g
    pts_full_s = pts_full / s_v
    cen_all = centers_full / s_v
    cen_s = cen_all[centers_keep] if centers_keep.any() else cen_all[:0]
    cen_disc = cen_all[~centers_keep] if (~centers_keep).any() else cen_all[:0]

    pts_exp, cols_exp, _ = _subsample_fg(pts_full_s, cols_full, max_cloud_points, rng)

    export_xyz_pointcloud_ply(gt_s, method_dir / "gt.ply", colors=gt_rgb)
    export_xyz_pointcloud_ply(pts_exp, method_dir / "vggt_depth_cloud.ply", colors=cols_exp)
    if cen_s.shape[0] > 0:
        export_xyz_pointcloud_ply(
            cen_s, method_dir / "patch_centers.ply", rgb=(32, 200, 64)
        )
    if cen_disc.shape[0] > 0:
        export_xyz_pointcloud_ply(
            cen_disc, method_dir / "discarded_centers.ply", rgb=(220, 40, 40)
        )

    cen_t = torch.from_numpy(np.asarray(cen_s, dtype=np.float32))
    nn_mean, nn_med = _nn_stats(cen_t, gt_s.float())
    logger.info(
        "%s %s (%s): scale_z_vggt=%.6f scale_z_gt=%.6f  centre→GT NN mean=%.4f med=%.4f",
        stem,
        tag,
        label,
        s_v,
        s_g,
        nn_mean,
        nn_med,
    )
    st_gt = _log_cloud_stats(stem, f"{tag}/gt", gt_s)
    st_ex = _log_cloud_stats(
        stem,
        f"{tag}/vggt_export",
        pts_exp,
        note="subsampled PLY" if pts_exp.shape[0] < pts_full_s.shape[0] else "full FG",
    )
    st_full = _log_cloud_stats(
        stem, f"{tag}/vggt_fg_full", pts_full_s, note="same points as scale_z_vggt"
    )
    if cen_s.shape[0] > 0:
        _log_cloud_stats(stem, f"{tag}/patch_centers", cen_s)

    if sanity == "mean":
        logger.info(
            "%s %s sanity: gt.mean_z=%.8f vggt_full.mean_z=%.8f (expect 1)  "
            "xy_span gt=%.4f vggt=%.4f  bb_cz gt=%.4f vggt=%.4f",
            stem,
            tag,
            st_gt["mean_z"],
            st_full["mean_z"],
            st_gt["xy_span"],
            st_full["xy_span"],
            st_gt["bb_cz"],
            st_ex["bb_cz"],
        )
    elif sanity == "zmin":
        logger.info(
            "%s %s sanity: gt.z_min=%.8f vggt_full.z_min=%.8f (expect 1)",
            stem,
            tag,
            st_gt["z_min"],
            st_full["z_min"],
        )
    elif sanity == "median":
        logger.info(
            "%s %s sanity: gt.median_z=%.8f vggt_full.median_z=%.8f (expect 1)",
            stem,
            tag,
            st_gt["median_z"],
            st_full["median_z"],
        )

    with open(method_dir / "stats.txt", "w", encoding="utf-8") as f:
        f.write(f"mesh={stem}\nmethod={tag} ({label})\n")
        f.write(f"scale_z_vggt={s_v:.8f}\nscale_z_gt={s_g:.8f}\n")
        f.write(
            "note: CloudCompare 'Create cloud from entities centers' "
            "uses bb_center, not mean_xyz.\n"
        )
        for name, st in (("gt", st_gt), ("vggt_export", st_ex), ("vggt_fg_full", st_full)):
            f.write(f"\n[{name}]\n")
            for k, v in st.items():
                f.write(f"  {k}={v}\n")

    summary_rows.append(
        {
            "mesh": stem,
            "method": tag,
            "z_vggt": s_v,
            "z_gt": s_g,
            "nn_mean": nn_mean,
            "nn_med": nn_med,
            "gt_xy_span": st_gt["xy_span"],
            "vggt_xy_span": st_full["xy_span"],
        }
    )


def _export_mv_frame0(
    *,
    stem: str,
    obj_dir: Path,
    view_payloads: List[Dict],
    raw: Dict,
    raw_mv: Optional[Dict],
    builder: VGGTContextBuilder,
    gt_cam: torch.Tensor,
    gt_rgb: Optional[torch.Tensor],
    pts_full: np.ndarray,
    cols_full: Optional[np.ndarray],
    pix_valid: np.ndarray,
    vggt_rgb: np.ndarray,
    depth: np.ndarray,
    gfx: float,
    gfy: float,
    gcx: float,
    gcy: float,
    pts_gtd: np.ndarray,
    s_gtd_raw: float,
    pts_vggt_gob_raw: np.ndarray,
    s_vggt_gob: float,
    mu_g: np.ndarray,
    s_g: float,
    mesh_gob_f: np.ndarray,
    vggt_vk_c: np.ndarray,
    cen_vk_keep_c: np.ndarray,
    cen_vk_disc_c: np.ndarray,
    max_cloud_points: int,
    rng: np.random.Generator,
    summary_rows: List[Dict],
) -> None:
    """Export ``mv_frame0/``: multi-view unions in the reference camera frame.

    When ``len(view_payloads)==1``, PE / GT / gobK-GT-E match the single-view
    cross recipes (PE = own Hunyuan; GT = /mean(GT depth)+gobK bbox).
    For ``S>1``, ``raw_mv`` must be the multi-view dense extract.
    """
    S = len(view_payloads)
    mv = obj_dir / "mv_frame0"
    mv.mkdir(parents=True, exist_ok=True)

    c2w_ref = np.asarray(view_payloads[0]["c2w"], dtype=np.float64)
    hd, wd = depth.shape[-2:]

    # Build sequence tensors: S==1 from single-view raw; S>1 from raw_mv.
    if S == 1:
        depth_seq = raw["vggt_depth"].detach().float().cpu().numpy()
        conf_seq = raw["vggt_depth_conf"].detach().float().cpu().numpy()
        E_seq = raw["vggt_extrinsics"].detach().float().cpu().numpy()
        K_seq = raw["vggt_intrinsics"].detach().float().cpu().numpy()
        world_seq = raw["vggt_cam_points"].detach().float().cpu().numpy()
    else:
        assert raw_mv is not None, "raw_mv required when num_views > 1"
        depth_seq = raw_mv["vggt_depth_seq"].detach().float().cpu().numpy()
        conf_seq = raw_mv["vggt_depth_conf_seq"].detach().float().cpu().numpy()
        E_seq = raw_mv["vggt_extrinsics_seq"].detach().float().cpu().numpy()
        K_seq = raw_mv["vggt_intrinsics_seq"].detach().float().cpu().numpy()
        world_seq = raw_mv["vggt_world_points_from_depth"].detach().float().cpu().numpy()

    vggt_c2ws = [vggt_extrinsic_to_c2w(E_seq[s]) for s in range(S)]
    vggt_c2w_ref = vggt_c2ws[0]

    # FG masks + RGB on VGGT grid per view
    vggt_rgbs: List[np.ndarray] = []
    vggt_valids: List[np.ndarray] = []
    for s in range(S):
        if s == 0 and S == 1:
            vggt_rgbs.append(vggt_rgb)
            vggt_valids.append(pix_valid)
            continue
        rgb_s = (
            torch.nn.functional.interpolate(
                view_payloads[s]["rgb"].float().unsqueeze(0),
                size=(hd, wd),
                mode="bilinear",
                align_corners=False,
            )[0]
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        vggt_rgbs.append(rgb_s)
        vggt_valids.append(
            builder._pixel_valid_mask(depth_seq[s], conf_seq[s], rgb_np=rgb_s)
        )

    # PE union (vggtK + VGGT poses): world → cam0
    pe_chunks: List[np.ndarray] = []
    pe_cols: List[np.ndarray] = []
    for s in range(S):
        m = vggt_valids[s]
        w = world_seq[s][m]
        if w.shape[0] == 0:
            continue
        pe_chunks.append(vggt_world_to_cam0(w, E_seq[0]))
        pe_cols.append(vggt_rgbs[s][m])
    if pe_chunks:
        pe_union = np.concatenate(pe_chunks, axis=0).astype(np.float32)
        pe_union_cols = np.concatenate(pe_cols, axis=0)
    else:
        pe_union = pts_full.astype(np.float32)
        pe_union_cols = cols_full

    # For S==1, world_from_depth is cam0 → identical to pts_full (same mask).
    if S == 1:
        pe_union = pts_full.astype(np.float32)
        pe_union_cols = cols_full

    (pe_c,), mu_pe, s_pe = shared_canonicalize_from_ref(
        pe_union, pe_union, scale="bbox", fill=0.9999
    )

    # --- GT depth union (native gobK + GT extrinsics) → ref cam ---
    gt_depths = []
    gt_valids = []
    gt_Ks = []
    gt_c2ws = []
    gt_cols_list: List[Optional[np.ndarray]] = []
    for vp in view_payloads:
        d = vp["depth"].numpy() if torch.is_tensor(vp["depth"]) else np.asarray(vp["depth"])
        rgb = vp["rgb"].permute(1, 2, 0).numpy()
        fx, fy, cx, cy = [float(x) for x in vp["intrinsics"].reshape(-1)[:4]]
        valid = _fg_mask_from_depth_rgb(d, rgb)
        gt_depths.append(d)
        gt_valids.append(valid)
        gt_Ks.append((fx, fy, cx, cy))
        gt_c2ws.append(np.asarray(vp["c2w"], dtype=np.float64))
        gt_cols_list.append(rgb)

    gtd_union, gtd_cols = merge_unprojected_to_cam_ref(
        gt_depths,
        gt_valids,
        Ks=gt_Ks,
        c2ws=gt_c2ws,
        c2w_ref=c2w_ref,
        colors=gt_cols_list,
    )
    if gtd_union.shape[0] == 0:
        gtd_union = pts_gtd.astype(np.float32)
    if S == 1:
        # Exact legacy: native-res GT depth ⊕ gobK in cam0 (no rigid round-trip).
        gtd_union = pts_gtd.astype(np.float32)
    s_gtd_mv = float(np.mean(gtd_union[:, 2])) if gtd_union.shape[0] else float(s_gtd_raw)
    if not np.isfinite(s_gtd_mv) or s_gtd_mv < 1e-6:
        s_gtd_mv = float(s_gtd_raw)

    # --- VGGT⊕gobK@vggt union with GT vs VGGT extrinsics ---
    gob_Ks_vggt: List[Tuple[float, float, float, float]] = []
    for s, vp in enumerate(view_payloads):
        if s == 0:
            gob_Ks_vggt.append((gfx, gfy, gcx, gcy))
        else:
            gob_Ks_vggt.append(
                gobjaverse_K_for_vggt_resolution(
                    vp["intrinsics"],
                    vp["rgb"].unsqueeze(0),
                    depth_hw=(hd, wd),
                    img_size=builder.img_size,
                )
            )

    vggt_gob_gtE, vggt_gob_cols = merge_unprojected_to_cam_ref(
        [depth_seq[s] for s in range(S)],
        vggt_valids,
        Ks=gob_Ks_vggt,
        c2ws=gt_c2ws,
        c2w_ref=c2w_ref,
        colors=vggt_rgbs,
    )
    vggt_gob_vgE, _ = merge_unprojected_to_cam_ref(
        [depth_seq[s] for s in range(S)],
        vggt_valids,
        Ks=gob_Ks_vggt,
        c2ws=vggt_c2ws,
        c2w_ref=vggt_c2w_ref,
        colors=vggt_rgbs,
    )
    if S == 1:
        # Exact legacy gobK unproject (no extrinsic map).
        vggt_gob_gtE = pts_vggt_gob_raw.astype(np.float32)
        vggt_gob_vgE = pts_vggt_gob_raw.astype(np.float32)
        vggt_gob_cols = cols_full

    def _cross_from_gob(pts_gob_raw: np.ndarray, tag: str):
        gob_m, s_gob = normalize_by_mean(pts_gob_raw)
        (pe_gob_c, gt_c), mu, s = shared_canonicalize_from_ref(
            gob_m,
            gob_m,
            (_as_xyz_np(gt_cam) / max(s_gtd_mv, 1e-6)).astype(np.float32),
            scale="bbox",
            fill=0.9999,
        )
        return pe_gob_c, gt_c, mu, s, s_gob, tag

    # Preferred: gobK + GT extrinsics (multi-view cross)
    pe_gob_gtE, gt_gtE, mu_gtE, s_gtE, s_gob_gtE, _ = _cross_from_gob(
        vggt_gob_gtE, "gobK_GT_E"
    )
    pe_gob_vgE, gt_vgE, mu_vgE, s_vgE, s_gob_vgE, _ = _cross_from_gob(
        vggt_gob_vgE, "gobK_VGGT_E"
    )

    # N=1: GT boxed with legacy single-view gobK stats matches A_cross_train
    if S == 1:
        gt_gtE = mesh_gob_f
        pe_gob_gtE = _apply_mean_then_bbox(pts_vggt_gob_raw, s_vggt_gob, mu_g, s_g)
        mu_gtE, s_gtE, s_gob_gtE = mu_g, s_g, s_vggt_gob
        # VGGT-E coincides with GT-E for S=1
        gt_vgE = gt_gtE
        pe_gob_vgE = pe_gob_gtE
        mu_vgE, s_vgE, s_gob_vgE = mu_gtE, s_gtE, s_gob_gtE
        pe_c = vggt_vk_c

    with open(mv / "README.txt", "w", encoding="utf-8") as f:
        f.write(
            f"mv_frame0 — {S} view(s) { [vp['view_idx'] for vp in view_payloads] }\n"
            "All clouds in reference camera (first view).\n\n"
            "A_cross_gobK_GT_E/   [recommended compare]\n"
            "  VGGT depth ⊕ gobK, rigid map with **GT** extrinsics → /mean + Hunyuan\n"
            "  GT mesh / mean_z(union GT depth) + same (μ,s)\n"
            "  PE = multi-view VGGT⊕vggtK own Hunyuan (no /mean)\n\n"
            "A_cross_gobK_VGGT_E/\n"
            "  Same but rigid map with **VGGT** extrinsics\n\n"
            "C_gt_depth_direct/\n"
            "  GT  = mesh Hunyuan from GT-depth union (gobK + GT extrinsics)\n"
            "  PE  = VGGT⊕vggtK union own Hunyuan (same as pe_vggtK_own)\n\n"
            "C_gt_depth_filter_zrobust/\n"
            "  Same as C but per-view erode FG 1px, then bbox z lo=p0.5\n\n"
            "A_cross_gobK_VGGT_E_meanrms/\n"
            "  Same as A_cross_gobK_VGGT_E but μ=mean, s=RMS (not AABB)\n\n"
            "C_gt_depth_filter_zrobust_meanrms/\n"
            "  Same partial cloud as C_filter_zrobust (erode1px∪) but μ=mean, s=RMS\n\n"
            "indep_meanrms/\n"
            "  Baseline: PE=vggtK∪ own mean+RMS; GT=full mesh own mean+RMS\n"
            "  (no GT-depth stats — independent normalize of both clouds)\n\n"
            "pe_vggtK_own/  — PE alone (own Hunyuan)\n"
            "gt_depth_union_GT_E/ — GT depth union (metric, no Hunyuan)\n"
            "For N=1, A_cross_gobK_GT_E matches gt_norm_ablation/A_cross_train.\n"
        )

    def _dump_pair(subdir: str, gt_xyz, pe_xyz, pe_cols_arr, label: str, extra: Dict):
        d = mv / subdir
        d.mkdir(parents=True, exist_ok=True)
        _write_pair_export(
            stem=stem,
            method_dir=d,
            tag=f"mv_frame0_{subdir}",
            label=label,
            gt_xyz=gt_xyz,
            gt_rgb=gt_rgb,
            other_xyz=pe_xyz,
            other_cols=pe_cols_arr,
            other_name="vggt_pred_vggtK",
            max_cloud_points=max_cloud_points,
            rng=rng,
            summary_rows=summary_rows,
            extra_stats=extra,
            cen_keep=cen_vk_keep_c if S == 1 else None,
            cen_disc=cen_vk_disc_c if S == 1 else None,
        )
        # Also dump gobK FG used for GT box (after /mean+bbox)
        return d

    _dump_pair(
        "A_cross_gobK_GT_E",
        gt_gtE,
        pe_c,
        pe_union_cols,
        (
            f"Cross N={S}: GT=/mean(GT-depth∪)+Hunyuan(VGGT⊕gobK, GT extrinsics); "
            "PE=vggtK∪ own Hunyuan"
        ),
        {
            "num_views": float(S),
            "mean_z_gt_depth": float(s_gtd_mv),
            "mean_z_vggt_gobK": float(s_gob_gtE),
            "gt_bbox_s": float(s_gtE),
            "pe_bbox_s": float(s_pe),
            "extrinsics": 0.0,  # 0=GT
        },
    )
    # Overlay gobK cloud (same box as GT) for visual FOV check
    gob_exp, gob_ce, _ = _subsample_fg(
        pe_gob_gtE, vggt_gob_cols, max_cloud_points, rng
    )
    export_xyz_pointcloud_ply(
        gob_exp,
        mv / "A_cross_gobK_GT_E" / "vggt_pred_gobK.ply",
        colors=gob_ce if gob_ce is not None else None,
    )

    _dump_pair(
        "A_cross_gobK_VGGT_E",
        gt_vgE,
        pe_c,
        pe_union_cols,
        (
            f"Cross N={S}: GT=/mean(GT-depth∪)+Hunyuan(VGGT⊕gobK, VGGT extrinsics); "
            "PE=vggtK∪ own Hunyuan"
        ),
        {
            "num_views": float(S),
            "mean_z_gt_depth": float(s_gtd_mv),
            "mean_z_vggt_gobK": float(s_gob_vgE),
            "gt_bbox_s": float(s_vgE),
            "pe_bbox_s": float(s_pe),
            "extrinsics": 1.0,  # 1=VGGT
        },
    )
    gob_exp, gob_ce, _ = _subsample_fg(
        pe_gob_vgE, vggt_gob_cols, max_cloud_points, rng
    )
    export_xyz_pointcloud_ply(
        gob_exp,
        mv / "A_cross_gobK_VGGT_E" / "vggt_pred_gobK.ply",
        colors=gob_ce if gob_ce is not None else None,
    )

    # C: independent Hunyuan — GT from GT-depth∪ (GT K+E); PE = vggtK∪ own bbox
    (gt_c_mv, gtd_c_mv), mu_c_mv, s_c_mv = shared_canonicalize_from_ref(
        gtd_union,
        _as_xyz_np(gt_cam),
        gtd_union,
        scale="bbox",
        fill=0.9999,
    )
    _dump_pair(
        "C_gt_depth_direct",
        gt_c_mv,
        pe_c,
        pe_union_cols,
        (
            f"C N={S}: GT=Hunyuan from GT-depth∪ (gobK+GT E); "
            "PE=vggtK∪ own Hunyuan"
        ),
        {
            "num_views": float(S),
            "gt_bbox_s": float(s_c_mv),
            "pe_bbox_s": float(s_pe),
        },
    )
    gtd_exp, gtd_ce, _ = _subsample_fg(gtd_c_mv, gtd_cols, max_cloud_points, rng)
    export_xyz_pointcloud_ply(
        gtd_exp,
        mv / "C_gt_depth_direct" / "gt_depth_gobK.ply",
        colors=gtd_ce if gtd_ce is not None else None,
    )

    # C_filter_zrobust: erode each view 1px, merge, Hunyuan with z lo=p0.5
    z_p_lo_s = 0.5
    gt_valids_f = [_filter_gt_depth_mask(v, erode_iters=1) for v in gt_valids]
    gtd_union_f, gtd_cols_f = merge_unprojected_to_cam_ref(
        gt_depths,
        gt_valids_f,
        Ks=gt_Ks,
        c2ws=gt_c2ws,
        c2w_ref=c2w_ref,
        colors=gt_cols_list,
    )
    if gtd_union_f.shape[0] == 0:
        gtd_union_f, gtd_cols_f = gtd_union, gtd_cols
    (gt_cs_mv, gtd_cs_mv), mu_cs_mv, s_cs_mv = shared_canonicalize_from_ref(
        gtd_union_f,
        _as_xyz_np(gt_cam),
        gtd_union_f,
        scale="bbox_zrobust",
        fill=0.9999,
        z_p_lo=z_p_lo_s,
    )
    _dump_pair(
        "C_gt_depth_filter_zrobust",
        gt_cs_mv,
        pe_c,
        pe_union_cols,
        (
            f"C_filter_zrobust N={S}: erode1px∪ + bbox z lo=p{z_p_lo_s}; "
            "PE=vggtK∪ own Hunyuan"
        ),
        {
            "num_views": float(S),
            "gt_bbox_s": float(s_cs_mv),
            "pe_bbox_s": float(s_pe),
            "z_p_lo": float(z_p_lo_s),
            "n_gtd_raw": float(len(gtd_union)),
            "n_gtd_eroded": float(len(gtd_union_f)),
        },
    )
    gtd_exp_f, gtd_ce_f, _ = _subsample_fg(
        gtd_cs_mv, gtd_cols_f, max_cloud_points, rng
    )
    export_xyz_pointcloud_ply(
        gtd_exp_f,
        mv / "C_gt_depth_filter_zrobust" / "gt_depth_gobK.ply",
        colors=gtd_ce_f if gtd_ce_f is not None else None,
    )

    # Mean+RMS variants of the 2 best recipes (same partial clouds, not AABB)
    gob_m_vg, _ = normalize_by_mean(vggt_gob_vgE)
    (pe_gob_rms, gt_vgE_rms), mu_a_rms, s_a_rms = shared_canonicalize_from_ref(
        gob_m_vg,
        gob_m_vg,
        (_as_xyz_np(gt_cam) / max(s_gtd_mv, 1e-6)).astype(np.float32),
        scale="rms",
    )
    (pe_c_rms,), mu_pe_rms, s_pe_rms = shared_canonicalize_from_ref(
        pe_union, pe_union, scale="rms"
    )
    _dump_pair(
        "A_cross_gobK_VGGT_E_meanrms",
        gt_vgE_rms,
        pe_c_rms,
        pe_union_cols,
        (
            f"Cross N={S} mean+RMS: GT=/mean(GT-depth∪)+RMS(VGGT⊕gobK, VGGT E); "
            "PE=vggtK∪ own mean+RMS"
        ),
        {
            "num_views": float(S),
            "mean_z_gt_depth": float(s_gtd_mv),
            "gt_rms_s": float(s_a_rms),
            "pe_rms_s": float(s_pe_rms),
            "extrinsics": 1.0,
        },
    )
    gob_exp_r, gob_ce_r, _ = _subsample_fg(
        pe_gob_rms, vggt_gob_cols, max_cloud_points, rng
    )
    export_xyz_pointcloud_ply(
        gob_exp_r,
        mv / "A_cross_gobK_VGGT_E_meanrms" / "vggt_pred_gobK.ply",
        colors=gob_ce_r if gob_ce_r is not None else None,
    )

    (gt_cs_rms, gtd_cs_rms), mu_c_rms, s_c_rms = shared_canonicalize_from_ref(
        gtd_union_f,
        _as_xyz_np(gt_cam),
        gtd_union_f,
        scale="rms",
    )
    _dump_pair(
        "C_gt_depth_filter_zrobust_meanrms",
        gt_cs_rms,
        pe_c_rms,
        pe_union_cols,
        (
            f"C_filter mean+RMS N={S}: erode1px∪ mean+RMS; "
            "PE=vggtK∪ own mean+RMS"
        ),
        {
            "num_views": float(S),
            "gt_rms_s": float(s_c_rms),
            "pe_rms_s": float(s_pe_rms),
            "n_gtd_eroded": float(len(gtd_union_f)),
        },
    )
    gtd_exp_r, gtd_ce_r, _ = _subsample_fg(
        gtd_cs_rms, gtd_cols_f, max_cloud_points, rng
    )
    export_xyz_pointcloud_ply(
        gtd_exp_r,
        mv / "C_gt_depth_filter_zrobust_meanrms" / "gt_depth_gobK.ply",
        colors=gtd_ce_r if gtd_ce_r is not None else None,
    )

    # Independent mean+RMS baseline: full GT mesh own (μ,s); PE = vggtK∪ own.
    # No GT-depth stats — useful control vs C_gt_depth_*_meanrms.
    (gt_indep_rms,), mu_gt_indep, s_gt_indep = shared_canonicalize_from_ref(
        _as_xyz_np(gt_cam),
        _as_xyz_np(gt_cam),
        scale="rms",
    )
    _dump_pair(
        "indep_meanrms",
        gt_indep_rms,
        pe_c_rms,
        pe_union_cols,
        (
            f"Indep mean+RMS N={S}: GT=full mesh own mean+RMS; "
            "PE=vggtK∪ own mean+RMS (no GT-depth)"
        ),
        {
            "num_views": float(S),
            "gt_rms_s": float(s_gt_indep),
            "pe_rms_s": float(s_pe_rms),
            "n_gt_mesh": float(_as_xyz_np(gt_cam).shape[0]),
        },
    )

    # Metrics for multi-view compare methods (PE FG vs GT mesh)
    for mname, gt_xyz, pe_xyz in (
        ("A_cross_gobK_VGGT_E", gt_vgE, pe_c),
        ("C_gt_depth_direct", gt_c_mv, pe_c),
        ("C_gt_depth_filter_zrobust", gt_cs_mv, pe_c),
        ("A_cross_gobK_VGGT_E_meanrms", gt_vgE_rms, pe_c_rms),
        ("C_gt_depth_filter_zrobust_meanrms", gt_cs_rms, pe_c_rms),
        ("indep_meanrms", gt_indep_rms, pe_c_rms),
    ):
        mdir = mv / mname
        m_fg = alignment_metrics(pe_xyz, gt_xyz, rng=rng)
        with open(mdir / "metrics.txt", "w", encoding="utf-8") as f:
            f.write(f"mesh={stem}\nmethod=mv_frame0_{mname}\n")
            f.write("vggt_pred_vggtK vs gt.ply (no patch centres for S>1)\n\n")
            f.write("[vggt_fg_vs_gt_mesh]\n")
            for k, v in m_fg.items():
                f.write(f"  {k}={v}\n")
        summary_rows.append(
            {
                "mesh": stem,
                "method": f"mv_frame0_{mname}",
                "nn_mean": m_fg["nn_pe2gt_mean"],
                "nn_med": m_fg["nn_pe2gt_med"],
                "gt_xy_span": float("nan"),
                "vggt_xy_span": float("nan"),
            }
        )

    # Raw metric unions (no Hunyuan) for sanity
    raw_u = mv / "raw_unions"
    raw_u.mkdir(parents=True, exist_ok=True)
    gtd_e, gtd_c, _ = _subsample_fg(gtd_union, gtd_cols, max_cloud_points, rng)
    export_xyz_pointcloud_ply(
        gtd_e, raw_u / "gt_depth_union_GT_E.ply", colors=gtd_c if gtd_c is not None else None
    )
    gob_e, gob_c, _ = _subsample_fg(vggt_gob_gtE, vggt_gob_cols, max_cloud_points, rng)
    export_xyz_pointcloud_ply(
        gob_e, raw_u / "vggt_gobK_union_GT_E.ply", colors=gob_c if gob_c is not None else None
    )
    gob_e2, gob_c2, _ = _subsample_fg(vggt_gob_vgE, vggt_gob_cols, max_cloud_points, rng)
    export_xyz_pointcloud_ply(
        gob_e2,
        raw_u / "vggt_gobK_union_VGGT_E.ply",
        colors=gob_c2 if gob_c2 is not None else None,
    )
    pe_e, pe_c_rgb, _ = _subsample_fg(pe_union, pe_union_cols, max_cloud_points, rng)
    export_xyz_pointcloud_ply(
        pe_e, raw_u / "vggt_vggtK_union.ply", colors=pe_c_rgb if pe_c_rgb is not None else None
    )
    export_xyz_pointcloud_ply(gt_cam, raw_u / "gt_mesh_ref.ply", colors=gt_rgb)

    pe_own = mv / "pe_vggtK_own"
    pe_own.mkdir(parents=True, exist_ok=True)
    pe_ce, pe_cc, _ = _subsample_fg(pe_c, pe_union_cols, max_cloud_points, rng)
    export_xyz_pointcloud_ply(
        pe_ce, pe_own / "vggt_pred_vggtK.ply", colors=pe_cc if pe_cc is not None else None
    )
    if S == 1 and len(cen_vk_keep_c):
        export_xyz_pointcloud_ply(
            cen_vk_keep_c, pe_own / "patch_centers.ply", rgb=(32, 200, 64)
        )

    with open(mv / "stats.txt", "w", encoding="utf-8") as f:
        f.write(
            f"mesh={stem}\nnum_views={S}\n"
            f"view_indices={[vp['view_idx'] for vp in view_payloads]}\n"
            f"mean_z_gt_depth_union={s_gtd_mv}\n"
            f"mean_z_vggt_gobK_GT_E={s_gob_gtE}\n"
            f"mean_z_vggt_gobK_VGGT_E={s_gob_vgE}\n"
            f"bbox_s_GT_E={s_gtE}\nbbox_s_VGGT_E={s_vgE}\n"
            f"pe_bbox_s={s_pe}\n"
            f"n_pe_union={len(pe_union)}\n"
            f"n_gtd_union={len(gtd_union)}\n"
            f"n_vggt_gob_GT_E={len(vggt_gob_gtE)}\n"
            f"n_vggt_gob_VGGT_E={len(vggt_gob_vgE)}\n"
        )
    logger.info(
        "%s mv_frame0: S=%d pe=%d gtd=%d gob_GT_E=%d gob_VGGT_E=%d",
        stem,
        S,
        len(pe_union),
        len(gtd_union),
        len(vggt_gob_gtE),
        len(vggt_gob_vgE),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--gobjaverse_render_root", default=None)
    p.add_argument("--output_dir", default="runs/debug_cam_frame")
    p.add_argument("--max_items", type=int, default=1)
    p.add_argument("--view_idx", type=int, default=0)
    p.add_argument(
        "--num_views",
        type=int,
        default=None,
        help="Use views 0..N-1 with view 0 as reference frame (1 = legacy single-view).",
    )
    p.add_argument(
        "--view_indices",
        type=str,
        default=None,
        help='Explicit views, e.g. "0,10". First entry is the reference camera.',
    )
    p.add_argument("--categories", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--conf_percentile", type=float, default=20.0)
    p.add_argument("--max_cloud_points", type=int, default=80000)
    p.add_argument(
        "--scale_percentiles",
        type=float,
        nargs="*",
        default=[20.0, 30.0, 40.0],
        help="Depth percentiles for scale ablations (in addition to mean/median/zmin).",
    )
    p.add_argument("--no_experiment_manifest", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    specs = _scale_specs(list(args.scale_percentiles))
    rng = np.random.default_rng(0)

    data_path = Path(args.data_dir).resolve()
    manifest = (
        None if args.no_experiment_manifest else load_experiment_manifest(str(data_path))
    )
    view_list = parse_view_indices(
        view_idx=args.view_idx,
        num_views=args.num_views,
        view_indices=args.view_indices,
    )
    ref_view = int(view_list[0])
    logger.info("Views (ref first): %s", view_list)
    # Dataset stays one sample per mesh at the reference view (mesh⊕c2w_ref).
    dataset = build_surface_render_dataset(
        str(data_path),
        max_items=args.max_items,
        categories=resolve_category_ids(args.categories),
        use_experiment_manifest=not args.no_experiment_manifest,
        manifest=manifest,
        render_root=args.gobjaverse_render_root,
        view_idx=ref_view,
        surface_in_camera_frame=True,
    )

    builder = VGGTContextBuilder(
        width=1024, conf_percentile=args.conf_percentile
    ).to(device)
    builder.eval()
    render_loader = getattr(dataset, "render_loader", None)

    summary_rows: List[Dict] = []
    bakeoff_rows: List[Dict] = []

    for i in range(len(dataset)):
        batch = collate_surface_render([dataset[i]])
        stem = Path(batch["mesh_path"][0]).stem
        mesh_path = batch["mesh_path"][0]
        obj_dir = out_root / f"{i:04d}_{stem[:16]}"
        raw_dir = obj_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Load all requested views (ref first). N=1 reuses the batch tensors.
        view_payloads: List[Dict] = []
        for vid in view_list:
            if int(vid) == ref_view and "rgb" in batch:
                view_payloads.append(
                    {
                        "rgb": batch["rgb"][0].detach().cpu(),
                        "depth": batch["depth"][0].detach().cpu()
                        if "depth" in batch
                        else None,
                        "intrinsics": batch["intrinsics"][0].detach().cpu()
                        if "intrinsics" in batch
                        else None,
                        "c2w": batch["c2w"][0].detach().cpu()
                        if "c2w" in batch
                        else None,
                        "view_idx": int(vid),
                    }
                )
            else:
                if render_loader is None:
                    raise RuntimeError(
                        "Multi-view debug needs a G-Objaverse render loader "
                        "(pass --gobjaverse_render_root)."
                    )
                v = render_loader.load_view(mesh_path, view_idx=int(vid))
                view_payloads.append(
                    {
                        "rgb": v["rgb"].detach().cpu(),
                        "depth": v["depth"].detach().cpu(),
                        "intrinsics": v["intrinsics"].detach().cpu(),
                        "c2w": v["c2w"].detach().cpu(),
                        "view_idx": int(vid),
                    }
                )

        # RGB collage / first view for this object
        if view_payloads[0]["rgb"] is not None:
            rgb_u8 = (
                (view_payloads[0]["rgb"].float().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0)
                .round()
                .astype(np.uint8)
            )
            try:
                import imageio.v2 as imageio  # type: ignore

                imageio.imwrite(obj_dir / "view_rgb.png", rgb_u8)
            except Exception:
                try:
                    from PIL import Image

                    Image.fromarray(rgb_u8).save(obj_dir / "view_rgb.png")
                except Exception as e:
                    logger.warning("%s: failed to write view_rgb.png: %s", stem, e)
            # Optional: save each view RGB for multi-view checks
            if len(view_payloads) > 1:
                for vp in view_payloads:
                    try:
                        import imageio.v2 as imageio  # type: ignore

                        arr = (
                            (vp["rgb"].float().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0)
                            .round()
                            .astype(np.uint8)
                        )
                        imageio.imwrite(obj_dir / f"view_rgb_{vp['view_idx']:02d}.png", arr)
                    except Exception:
                        pass

        surface = batch["surface"]
        gt_cam = surface[0, :, :3].float()
        gt_rgb = surface[0, :, 6:9] if surface.shape[-1] >= 9 else None
        export_xyz_pointcloud_ply(gt_cam, raw_dir / "gt_camera.ply", colors=gt_rgb)

        # Legacy ablations always use **ref-view-only** VGGT (N=1 bit-identical).
        # Multi-view dense geometry is extracted later for mv_frame0 only.
        rgb = view_payloads[0]["rgb"].float().unsqueeze(0).to(device)
        raw = builder.extract_vggt_raw(rgb, return_dense=True)
        centers = raw["patch_centers"][0].cpu().float().numpy()
        keep = raw["patch_keep"][0].cpu().numpy().astype(bool)
        # Cached/extracted centres already filtered+padded; rebuild full-grid keep for gobK.
        export_xyz_pointcloud_ply(
            centers[keep], raw_dir / "patch_centers.ply", rgb=(32, 200, 64)
        )

        cam_pts = raw["vggt_cam_points"][0].cpu().numpy()
        depth = raw["vggt_depth"][0].cpu().numpy()
        conf = raw["vggt_depth_conf"][0].cpu().numpy()
        vggt_K = None
        if "vggt_intrinsics" in raw:
            K = raw["vggt_intrinsics"][0].cpu().numpy()
            vggt_K = (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))

        # Ref-view RGB resampled to VGGT depth grid (same as legacy).
        vggt_rgb = (
            torch.nn.functional.interpolate(
                view_payloads[0]["rgb"].float().unsqueeze(0),
                size=depth.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[0]
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        pix_valid = builder._pixel_valid_mask(depth, conf, rgb_np=vggt_rgb)

        raw_mv = None
        if len(view_payloads) > 1:
            rgb_seq = torch.stack(
                [vp["rgb"].float() for vp in view_payloads], dim=0
            ).to(device)
            raw_mv = builder.extract_vggt_sequence_dense(rgb_seq)

        full_c, full_keep = patch_centers_from_depth(
            cam_pts,
            pix_valid,
            patch_size=builder.patch_size,
            fx=vggt_K[0] if vggt_K else None,
            fy=vggt_K[1] if vggt_K else None,
            cx=vggt_K[2] if vggt_K else None,
            cy=vggt_K[3] if vggt_K else None,
        )
        discarded = full_c[~full_keep]
        if discarded.shape[0] > 0:
            export_xyz_pointcloud_ply(
                discarded, raw_dir / "discarded_centers.ply", rgb=(220, 40, 40)
            )

        pts_full = cam_pts[pix_valid]
        cols_full = vggt_rgb[pix_valid]
        pts_exp, cols_exp, _ = _subsample_fg(
            pts_full, cols_full, args.max_cloud_points, rng
        )
        export_xyz_pointcloud_ply(pts_exp, raw_dir / "vggt_depth_cloud.ply", colors=cols_exp)

        logger.info(
            "%s raw: kept_patches=%d discarded=%d  vggt_fg=%d  "
            "vggt_K=%s",
            stem,
            int(full_keep.sum()),
            int((~full_keep).sum()),
            pts_full.shape[0],
            vggt_K,
        )
        logger.info(
            "%s --- cloud stats (mean_xyz = centroid; bb_center ≈ CloudCompare) ---",
            stem,
        )
        _log_cloud_stats(stem, "raw/gt", gt_cam)
        _log_cloud_stats(stem, "raw/vggt_fg_full", pts_full, note="VGGT K unproject")
        if keep.any():
            _log_cloud_stats(stem, "raw/patch_centers", centers[keep])

        summary_rows.append(
            {
                "mesh": stem,
                "method": "raw",
                "nn_mean": _nn_stats(
                    torch.from_numpy(centers[keep]) if keep.any() else torch.zeros(0, 3),
                    gt_cam,
                )[0],
            }
        )

        # --- Standard ablations: VGGT-K unprojection ---
        for tag, label, reduce_fn in specs:
            sanity = tag if tag in ("median", "zmin") else ""
            export_scaled_method(
                stem=stem,
                method_dir=obj_dir / f"scale_{tag}",
                tag=f"scale_{tag}",
                label=f"{label}, VGGT K",
                gt_cam=gt_cam,
                gt_rgb=gt_rgb,
                pts_full=pts_full,
                cols_full=cols_full,
                centers_full=full_c,
                centers_keep=full_keep,
                reduce_fn=reduce_fn,
                max_cloud_points=args.max_cloud_points,
                rng=rng,
                summary_rows=summary_rows,
                sanity=sanity,
            )

        # Precompute /z_min clouds (used by all non-median/percentile exports below)
        gt_z, s_gt_z = normalize_by_zmin(gt_cam)
        pts_z, s_v_z = normalize_by_zmin(pts_full)

        # --- Shared VGGT-canonical (p-μ)/s — μ,s from VGGT only ---
        cen_raw = full_c[full_keep] if full_keep.any() else full_c[:0]
        (gt_can, pts_can, cen_can), mu_v, s_v = shared_canonicalize_from_ref(
            pts_full, gt_cam.numpy(), pts_full, cen_raw
        )
        logger.info(
            "%s canon_shared_vggt: μ=(%.4f, %.4f, %.4f) s_rms=%.6f (from VGGT FG)",
            stem,
            mu_v[0],
            mu_v[1],
            mu_v[2],
            s_v,
        )
        _write_pair_export(
            stem=stem,
            method_dir=obj_dir / "canon_shared_vggt",
            tag="canon_shared_vggt",
            label="shared (p-μ)/s with μ,s from VGGT FG (camera-oriented PE frame)",
            gt_xyz=gt_can,
            gt_rgb=gt_rgb,
            other_xyz=pts_can,
            other_cols=cols_full,
            other_name="vggt_depth_cloud",
            max_cloud_points=args.max_cloud_points,
            rng=rng,
            summary_rows=summary_rows,
            extra_stats={
                "mu_x": float(mu_v[0]),
                "mu_y": float(mu_v[1]),
                "mu_z": float(mu_v[2]),
                "s_rms_vggt": float(s_v),
            },
            extra_txt=(
                "note: GT already in camera frame via c2w; then SAME μ,s from VGGT "
                "applied to GT + VGGT + patch centres. Inference-safe shared PE frame.\n"
            ),
        )
        if cen_can.shape[0] > 0:
            export_xyz_pointcloud_ply(
                cen_can,
                obj_dir / "canon_shared_vggt" / "patch_centers.ply",
                rgb=(32, 200, 64),
            )

        # Contrast: independent (p-μ)/s per cloud
        (gt_ind, pts_ind), mus_ind, scales_ind = independent_canonicalize(
            gt_cam.numpy(), pts_full
        )
        _write_pair_export(
            stem=stem,
            method_dir=obj_dir / "canon_indep",
            tag="canon_indep",
            label="independent (p-μ)/s per cloud (NOT shared PE frame)",
            gt_xyz=gt_ind,
            gt_rgb=gt_rgb,
            other_xyz=pts_ind,
            other_cols=cols_full,
            other_name="vggt_depth_cloud",
            max_cloud_points=args.max_cloud_points,
            rng=rng,
            summary_rows=summary_rows,
            extra_stats={
                "mu_gt_x": float(mus_ind[0][0]),
                "mu_gt_y": float(mus_ind[0][1]),
                "mu_gt_z": float(mus_ind[0][2]),
                "s_rms_gt": float(scales_ind[0]),
                "mu_vggt_x": float(mus_ind[1][0]),
                "mu_vggt_y": float(mus_ind[1][1]),
                "mu_vggt_z": float(mus_ind[1][2]),
                "s_rms_vggt": float(scales_ind[1]),
            },
            extra_txt=(
                "note: each cloud uses its own μ,s — contrast with canon_shared_vggt.\n"
            ),
        )

        # --- /z_min then canonicalize (shared vs independent μ+s) ---
        # Shared: keep relative /z_min front align; μ,s from VGGT; maxabs → ~[-1,1]
        (gt_zs, pts_zs), mu_zs, s_zs = shared_canonicalize_from_ref(
            pts_z, gt_z, pts_z, scale="maxabs"
        )
        logger.info(
            "%s zmin_then_shared_canon: after /z_min, shared μ=(%.4f,%.4f,%.4f) "
            "s_maxabs=%.6f",
            stem,
            mu_zs[0],
            mu_zs[1],
            mu_zs[2],
            s_zs,
        )
        _write_pair_export(
            stem=stem,
            method_dir=obj_dir / "zmin_then_shared_canon",
            tag="zmin_then_shared_canon",
            label="/z_min then shared (p-μ)/s_maxabs from VGGT",
            gt_xyz=gt_zs,
            gt_rgb=gt_rgb,
            other_xyz=pts_zs,
            other_cols=cols_full,
            other_name="vggt_depth_cloud",
            max_cloud_points=args.max_cloud_points,
            rng=rng,
            summary_rows=summary_rows,
            extra_stats={
                "scale_z_gt": s_gt_z,
                "scale_z_vggt": s_v_z,
                "mu_x": float(mu_zs[0]),
                "mu_y": float(mu_zs[1]),
                "mu_z": float(mu_zs[2]),
                "s_maxabs_vggt": float(s_zs),
            },
            extra_txt=(
                "note: /z_min independently first (front≈1), then SAME μ,s from "
                "VGGT (max-abs) so relative front align is preserved and clouds "
                "sit roughly in [-1,1].\n"
            ),
        )

        # Independent: /z_min then each cloud its own μ and s (may undo front align)
        (gt_zi, pts_zi), mus_zi, scales_zi = independent_canonicalize(
            gt_z, pts_z, scale="maxabs"
        )
        _write_pair_export(
            stem=stem,
            method_dir=obj_dir / "zmin_then_indep_canon",
            tag="zmin_then_indep_canon",
            label="/z_min then independent μ and s_maxabs per cloud",
            gt_xyz=gt_zi,
            gt_rgb=gt_rgb,
            other_xyz=pts_zi,
            other_cols=cols_full,
            other_name="vggt_depth_cloud",
            max_cloud_points=args.max_cloud_points,
            rng=rng,
            summary_rows=summary_rows,
            extra_stats={
                "scale_z_gt": s_gt_z,
                "scale_z_vggt": s_v_z,
                "mu_gt_x": float(mus_zi[0][0]),
                "mu_gt_y": float(mus_zi[0][1]),
                "mu_gt_z": float(mus_zi[0][2]),
                "s_maxabs_gt": float(scales_zi[0]),
                "mu_vggt_x": float(mus_zi[1][0]),
                "mu_vggt_y": float(mus_zi[1][1]),
                "mu_vggt_z": float(mus_zi[1][2]),
                "s_maxabs_vggt": float(scales_zi[1]),
            },
            extra_txt=(
                "note: /z_min first, then EACH cloud (p-μ)/s independently. "
                "Independent centering often undoes front-plane /z_min alignment.\n"
            ),
        )

        # --- z_min + G-Objaverse K unprojection ---
        if "intrinsics" not in batch:
            logger.warning("%s: no G-Objaverse intrinsics in batch; skip scale_zmin_gobK", stem)
        else:
            fx, fy, cx, cy = gobjaverse_K_for_vggt_resolution(
                batch["intrinsics"][0],
                batch["rgb"] if "rgb" in batch else view_payloads[0]["rgb"].unsqueeze(0),
                depth_hw=depth.shape[-2:],
                img_size=builder.img_size,
            )
            logger.info(
                "%s gobK@vggt_res: fx=%.2f fy=%.2f cx=%.2f cy=%.2f  (native %s)",
                stem,
                fx,
                fy,
                cx,
                cy,
                batch["intrinsics"][0].tolist(),
            )
            cam_pts_gob = depth_map_to_cam_points(depth, fx=fx, fy=fy, cx=cx, cy=cy)
            pts_gob = cam_pts_gob[pix_valid]
            full_c_gob, full_keep_gob = patch_centers_from_depth(
                cam_pts_gob,
                pix_valid,
                patch_size=builder.patch_size,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
            )
            gob_raw = obj_dir / "raw_gobK"
            gob_raw.mkdir(parents=True, exist_ok=True)
            pts_gob_exp, cols_gob_exp, _ = _subsample_fg(
                pts_gob, cols_full, args.max_cloud_points, rng
            )
            export_xyz_pointcloud_ply(
                pts_gob_exp, gob_raw / "vggt_depth_cloud.ply", colors=cols_gob_exp
            )
            export_xyz_pointcloud_ply(gt_cam, gob_raw / "gt_camera.ply", colors=gt_rgb)
            if (~full_keep_gob).any():
                export_xyz_pointcloud_ply(
                    full_c_gob[~full_keep_gob],
                    gob_raw / "discarded_centers.ply",
                    rgb=(220, 40, 40),
                )
            if full_keep_gob.any():
                export_xyz_pointcloud_ply(
                    full_c_gob[full_keep_gob],
                    gob_raw / "patch_centers.ply",
                    rgb=(32, 200, 64),
                )
            _log_cloud_stats(stem, "raw_gobK/vggt_fg_full", pts_gob, note="G-Objaverse K")

            export_scaled_method(
                stem=stem,
                method_dir=obj_dir / "scale_zmin_gobK",
                tag="scale_zmin_gobK",
                label="z_min, G-Objaverse K unproject",
                gt_cam=gt_cam,
                gt_rgb=gt_rgb,
                pts_full=pts_gob,
                cols_full=cols_full,
                centers_full=full_c_gob,
                centers_keep=full_keep_gob,
                reduce_fn=_zmin_fn,
                max_cloud_points=args.max_cloud_points,
                rng=rng,
                summary_rows=summary_rows,
                sanity="zmin",
            )

        # --- Umeyama similarity variants (isotropic) ---
        zmin_note = (
            f"note: both clouds /z_min first (scale_z_gt={s_gt_z:.8f}, "
            f"scale_z_vggt={s_v_z:.8f}), then isotropic Umeyama.\n"
        )
        _export_align_sim_variant(
            stem=stem,
            obj_dir=obj_dir,
            tag="align_sim",
            label="/z_min then Umeyama full GT→NN(VGGT)",
            gt_src=gt_z,
            vggt_src=pts_z,
            gt_rgb=gt_rgb,
            cols_full=cols_full,
            mode="full",
            rng=rng,
            max_cloud_points=args.max_cloud_points,
            summary_rows=summary_rows,
            extra_stats={"scale_z_gt": s_gt_z, "scale_z_vggt": s_v_z},
            note=zmin_note,
        )
        _export_align_sim_variant(
            stem=stem,
            obj_dir=obj_dir,
            tag="align_sim_trim",
            label="/z_min then Umeyama trimmed NN (p30 closest)",
            gt_src=gt_z,
            vggt_src=pts_z,
            gt_rgb=gt_rgb,
            cols_full=cols_full,
            mode="trim",
            rng=rng,
            max_cloud_points=args.max_cloud_points,
            summary_rows=summary_rows,
            extra_stats={
                "scale_z_gt": s_gt_z,
                "scale_z_vggt": s_v_z,
                "trim_percentile": 30.0,
            },
            note=zmin_note + "trim: keep NN dist ≤ 30th percentile.\n",
            trim_percentile=30.0,
        )
        _export_align_sim_variant(
            stem=stem,
            obj_dir=obj_dir,
            tag="align_sim_mutual",
            label="/z_min then Umeyama mutual NN only",
            gt_src=gt_z,
            vggt_src=pts_z,
            gt_rgb=gt_rgb,
            cols_full=cols_full,
            mode="mutual",
            rng=rng,
            max_cloud_points=args.max_cloud_points,
            summary_rows=summary_rows,
            extra_stats={"scale_z_gt": s_gt_z, "scale_z_vggt": s_v_z},
            note=zmin_note + "mutual: keep only mutual nearest neighbors.\n",
        )

        # Visible-only GT (project onto VGGT FG), then /z_min + Umeyama full on that subset
        if "intrinsics" in batch:
            gob_fx, gob_fy, gob_cx, gob_cy = [
                float(x) for x in batch["intrinsics"][0].reshape(-1)[:4]
            ]
            _, scale_y, scale_x, pad_top, pad_left = preprocess_rgb_for_vggt(
                batch["rgb"].float().clamp(0, 1), target_size=builder.img_size
            )
            vis = gt_visible_on_vggt_mask(
                gt_cam.numpy(),
                gob_fx=gob_fx,
                gob_fy=gob_fy,
                gob_cx=gob_cx,
                gob_cy=gob_cy,
                scale_x=scale_x,
                scale_y=scale_y,
                pad_left=pad_left,
                pad_top=pad_top,
                pix_valid=pix_valid,
            )
            gt_vis_raw = gt_cam.numpy()[vis]
            if gt_vis_raw.shape[0] >= 3:
                gt_vis_z, s_vis = normalize_by_zmin(gt_vis_raw)
                # Also export the visible subset alone (no Umeyama) for inspection
                vis_dir = obj_dir / "gt_visible_zmin"
                vis_dir.mkdir(parents=True, exist_ok=True)
                export_xyz_pointcloud_ply(
                    gt_vis_z,
                    vis_dir / "gt.ply",
                    colors=gt_rgb[vis] if gt_rgb is not None else None,
                )
                pts_v_exp, cols_v_exp, _ = _subsample_fg(
                    pts_z, cols_full, args.max_cloud_points, rng
                )
                export_xyz_pointcloud_ply(
                    pts_v_exp, vis_dir / "vggt_depth_cloud.ply", colors=cols_v_exp
                )
                with open(vis_dir / "stats.txt", "w", encoding="utf-8") as f:
                    f.write(
                        f"mesh={stem}\nmethod=gt_visible_zmin "
                        "(GT points projecting onto VGGT FG, /z_min)\n"
                        f"n_gt_full={gt_cam.shape[0]}\nn_gt_visible={gt_vis_raw.shape[0]}\n"
                        f"scale_z_gt_visible={s_vis}\nscale_z_vggt={s_v_z}\n"
                    )
                logger.info(
                    "%s gt_visible_zmin: %d / %d GT points on VGGT FG",
                    stem,
                    gt_vis_raw.shape[0],
                    gt_cam.shape[0],
                )
                _export_align_sim_variant(
                    stem=stem,
                    obj_dir=obj_dir,
                    tag="align_sim_visible",
                    label="/z_min on visible GT only, then Umeyama full",
                    gt_src=gt_vis_z,
                    vggt_src=pts_z,
                    gt_rgb=gt_rgb[vis] if gt_rgb is not None else None,
                    cols_full=cols_full,
                    mode="full",
                    rng=rng,
                    max_cloud_points=args.max_cloud_points,
                    summary_rows=summary_rows,
                    extra_stats={
                        "scale_z_gt_visible": s_vis,
                        "scale_z_vggt": s_v_z,
                        "n_gt_visible": float(gt_vis_raw.shape[0]),
                    },
                    note=(
                        "note: GT filtered to pixels on VGGT FG, /z_min, then Umeyama.\n"
                        f"n_gt_visible={gt_vis_raw.shape[0]}\n"
                    ),
                )
            else:
                logger.warning("%s: too few visible GT points; skip align_sim_visible", stem)
        else:
            logger.warning("%s: skip align_sim_visible / gt_visible_zmin (no intrinsics)", stem)

        # Raw (no /z_min) Umeyama — full vs trim (shows why z_min-first helps)
        _export_align_sim_variant(
            stem=stem,
            obj_dir=obj_dir,
            tag="align_sim_raw_full",
            label="Umeyama on RAW clouds (no /z_min), full NN",
            gt_src=_as_xyz_np(gt_cam).astype(np.float32),
            vggt_src=pts_full.astype(np.float32),
            gt_rgb=gt_rgb,
            cols_full=cols_full,
            mode="full",
            rng=rng,
            max_cloud_points=args.max_cloud_points,
            summary_rows=summary_rows,
            note="note: NO /z_min — Umeyama on raw GT vs raw VGGT.\n",
        )
        _export_align_sim_variant(
            stem=stem,
            obj_dir=obj_dir,
            tag="align_sim_raw_trim",
            label="Umeyama on RAW clouds (no /z_min), trimmed NN",
            gt_src=_as_xyz_np(gt_cam).astype(np.float32),
            vggt_src=pts_full.astype(np.float32),
            gt_rgb=gt_rgb,
            cols_full=cols_full,
            mode="trim",
            rng=rng,
            max_cloud_points=args.max_cloud_points,
            summary_rows=summary_rows,
            extra_stats={"trim_percentile": 30.0},
            note="note: NO /z_min — trimmed Umeyama on raw clouds.\n",
            trim_percentile=30.0,
        )

        # --- gt_reproj_vggtK: FoV warp then /z_min ---
        if "intrinsics" in batch and vggt_K is not None:
            gob_fx, gob_fy, gob_cx, gob_cy = [
                float(x) for x in batch["intrinsics"][0].reshape(-1)[:4]
            ]
            _, scale_y, scale_x, pad_top, pad_left = preprocess_rgb_for_vggt(
                batch["rgb"].float().clamp(0, 1), target_size=builder.img_size
            )
            vfx, vfy, vcx, vcy = vggt_K
            gt_warp = fov_warp_gt_to_vggt_k(
                gt_cam.numpy(),
                gob_fx=gob_fx,
                gob_fy=gob_fy,
                gob_cx=gob_cx,
                gob_cy=gob_cy,
                scale_x=scale_x,
                scale_y=scale_y,
                pad_left=pad_left,
                pad_top=pad_top,
                vggt_fx=vfx,
                vggt_fy=vfy,
                vggt_cx=vcx,
                vggt_cy=vcy,
            )
            gt_warp_z, s_warp = normalize_by_zmin(gt_warp)
            _write_pair_export(
                stem=stem,
                method_dir=obj_dir / "gt_reproj_vggtK",
                tag="gt_reproj_vggtK",
                label="FoV warp via pixels (may shear) then /z_min both",
                gt_xyz=gt_warp_z,
                gt_rgb=gt_rgb,
                other_xyz=pts_z,
                other_cols=cols_full,
                other_name="vggt_depth_cloud",
                max_cloud_points=args.max_cloud_points,
                rng=rng,
                summary_rows=summary_rows,
                extra_stats={
                    "scale_z_gt_warped": s_warp,
                    "scale_z_vggt": s_v_z,
                    "gob_fx": gob_fx,
                    "vggt_fx": vfx,
                    "fx_ratio_vggt_over_gob_at_native_ish": vfx
                    / max(gob_fx * scale_x, 1e-6),
                },
            )

            # Same FoV warp but /mean(z) instead of /z_min
            gt_warp_m, s_warp_m = normalize_by_mean(gt_warp)
            pts_m, s_v_m = normalize_by_mean(pts_full)
            _write_pair_export(
                stem=stem,
                method_dir=obj_dir / "gt_reproj_vggtK_mean",
                tag="gt_reproj_vggtK_mean",
                label="FoV warp via pixels then /mean(z) both",
                gt_xyz=gt_warp_m,
                gt_rgb=gt_rgb,
                other_xyz=pts_m,
                other_cols=cols_full,
                other_name="vggt_depth_cloud",
                max_cloud_points=args.max_cloud_points,
                rng=rng,
                summary_rows=summary_rows,
                extra_stats={
                    "scale_mean_gt_warped": s_warp_m,
                    "scale_mean_vggt": s_v_m,
                    "gob_fx": gob_fx,
                    "vggt_fx": vfx,
                },
            )

            # Shear-free FoV: pure fx/fy lateral scale (no cx/cy / pad terms)
            gt_fx, sx, sy = fx_scale_gt_to_vggt_k(
                gt_cam.numpy(),
                gob_fx=gob_fx,
                gob_fy=gob_fy,
                scale_x=scale_x,
                scale_y=scale_y,
                vggt_fx=vfx,
                vggt_fy=vfy,
            )
            gt_fx_z, s_fx = normalize_by_zmin(gt_fx)
            logger.info(
                "%s gt_fxscale_vggtK: sx=%.6f sy=%.6f (gob_f@518 / vggt_f)",
                stem,
                sx,
                sy,
            )
            _write_pair_export(
                stem=stem,
                method_dir=obj_dir / "gt_fxscale_vggtK",
                tag="gt_fxscale_vggtK",
                label="shear-free xy *= fx_gob@518/fx_vggt then /z_min",
                gt_xyz=gt_fx_z,
                gt_rgb=gt_rgb,
                other_xyz=pts_z,
                other_cols=cols_full,
                other_name="vggt_depth_cloud",
                max_cloud_points=args.max_cloud_points,
                rng=rng,
                summary_rows=summary_rows,
                extra_stats={
                    "scale_z_gt": s_fx,
                    "scale_z_vggt": s_v_z,
                    "xy_scale_x": sx,
                    "xy_scale_y": sy,
                    "gob_fx_at_518": gob_fx * scale_x,
                    "gob_fy_at_518": gob_fy * scale_y,
                    "vggt_fx": vfx,
                    "vggt_fy": vfy,
                },
            )
        else:
            logger.warning(
                "%s: skip gt_reproj_* / gt_fxscale_vggtK (need intrinsics + vggt_K)",
                stem,
            )

        # --- G-Objaverse rendered depth vs VGGT pred (gobK / VGGT K) ---
        if "depth" in batch and "intrinsics" in batch:
            depth_gt = batch["depth"][0].numpy()
            gob_fx, gob_fy, gob_cx, gob_cy = [
                float(x) for x in batch["intrinsics"][0].reshape(-1)[:4]
            ]
            rgb_np = batch["rgb"][0].permute(1, 2, 0).numpy()
            valid_gt = _fg_mask_from_depth_rgb(depth_gt, rgb_np)
            cam_gt_gob = depth_map_to_cam_points(
                depth_gt, fx=gob_fx, fy=gob_fy, cx=gob_cx, cy=gob_cy
            )
            pts_gtd = cam_gt_gob[valid_gt]
            cols_gtd = rgb_np[valid_gt] if rgb_np.shape[:2] == depth_gt.shape else None

            # Metric camera-frame sanity: mesh⊕c2w vs depth⊕gobK (NO /mean or /z_min)
            mvd = obj_dir / "mesh_vs_depth_cam"
            mvd.mkdir(parents=True, exist_ok=True)
            pts_gtd_exp, cols_gtd_exp, _ = _subsample_fg(
                pts_gtd.astype(np.float32), cols_gtd, args.max_cloud_points, rng
            )
            export_xyz_pointcloud_ply(
                gt_cam, mvd / "gt_mesh_camera.ply", colors=gt_rgb
            )
            export_xyz_pointcloud_ply(
                pts_gtd_exp,
                mvd / "gt_depth_gobK.ply",
                colors=cols_gtd_exp if cols_gtd_exp is not None else None,
            )
            # Visible-only mesh: project with GT K onto FG depth mask
            gt_np = gt_cam.numpy()
            z = np.maximum(gt_np[:, 2], 1e-6)
            ui = np.rint(gob_fx * (gt_np[:, 0] / z) + gob_cx).astype(np.int64)
            vi = np.rint(gob_fy * (gt_np[:, 1] / z) + gob_cy).astype(np.int64)
            h, w = valid_gt.shape
            inb = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
            vis_m = np.zeros(gt_np.shape[0], dtype=bool)
            if inb.any():
                vis_m[inb] = valid_gt[vi[inb], ui[inb]]
            if int(vis_m.sum()) >= 3:
                export_xyz_pointcloud_ply(
                    gt_cam[vis_m],
                    mvd / "gt_mesh_visible.ply",
                    colors=gt_rgb[vis_m] if gt_rgb is not None else None,
                )
            nn_d2m, nn_d2m_med = _nn_stats(
                torch.from_numpy(
                    pts_gtd_exp[: min(4000, len(pts_gtd_exp))].astype(np.float32)
                ),
                gt_cam.float(),
            )
            st_mesh = _log_cloud_stats(stem, "mesh_vs_depth_cam/mesh", gt_cam)
            st_dep = _log_cloud_stats(stem, "mesh_vs_depth_cam/depth", pts_gtd)
            logger.info(
                "%s mesh_vs_depth_cam (metric): depth→mesh NN mean=%.4f med=%.4f  "
                "mean_z mesh=%.4f depth=%.4f",
                stem,
                nn_d2m,
                nn_d2m_med,
                st_mesh["mean_z"],
                st_dep["mean_z"],
            )
            with open(mvd / "stats.txt", "w", encoding="utf-8") as f:
                f.write(
                    f"mesh={stem}\nmethod=mesh_vs_depth_cam\n"
                    "Metric camera frame: gt_mesh via c2w extrinsics; "
                    "gt_depth via nd.exr ⊕ GT intrinsics. NO /z_min or /mean.\n"
                    f"gob_fx={gob_fx} gob_fy={gob_fy} gob_cx={gob_cx} gob_cy={gob_cy}\n"
                    f"nn_depth_to_mesh_mean={nn_d2m}\n"
                    f"nn_depth_to_mesh_med={nn_d2m_med}\n"
                    f"n_mesh={int(st_mesh['n'])} n_depth_fg={int(st_dep['n'])}\n"
                    f"n_mesh_visible={int(vis_m.sum())}\n"
                )
                for name, st in (("gt_mesh_camera", st_mesh), ("gt_depth_gobK", st_dep)):
                    f.write(f"\n[{name}]\n")
                    for k, v in st.items():
                        f.write(f"  {k}={v}\n")
            summary_rows.append(
                {
                    "mesh": stem,
                    "method": "mesh_vs_depth_cam",
                    "nn_mean": nn_d2m,
                    "nn_med": nn_d2m_med,
                    "gt_xy_span": st_mesh["xy_span"],
                    "vggt_xy_span": st_dep["xy_span"],
                }
            )

            # /mean(z) on both GT depth and mesh/VGGT — avoids noisy nearest-pixel z_min
            pts_gtd_m, s_gtd = normalize_by_mean(pts_gtd)
            gt_mesh_m, s_mesh_m = normalize_by_mean(gt_cam)
            pts_v_m, s_v_m = normalize_by_mean(pts_full)

            # Legacy folder (mesh + gt depth gobK + vggt) — now /mean
            gtd_dir = obj_dir / "gt_depth_gobK"
            gtd_dir.mkdir(parents=True, exist_ok=True)
            pts_gtd_exp, cols_gtd_exp, _ = _subsample_fg(
                pts_gtd_m, cols_gtd, args.max_cloud_points, rng
            )
            pts_v_exp, cols_v_exp, _ = _subsample_fg(
                pts_v_m, cols_full, args.max_cloud_points, rng
            )
            export_xyz_pointcloud_ply(gt_mesh_m, gtd_dir / "gt_mesh.ply", colors=gt_rgb)
            export_xyz_pointcloud_ply(
                pts_gtd_exp,
                gtd_dir / "gt_depth_cloud.ply",
                colors=cols_gtd_exp if cols_gtd_exp is not None else None,
            )
            export_xyz_pointcloud_ply(
                pts_v_exp, gtd_dir / "vggt_depth_cloud.ply", colors=cols_v_exp
            )
            st_mesh = _log_cloud_stats(stem, "gt_depth_gobK/gt_mesh", gt_mesh_m)
            st_gtd = _log_cloud_stats(stem, "gt_depth_gobK/gt_depth", pts_gtd_m)
            st_v = _log_cloud_stats(stem, "gt_depth_gobK/vggt", pts_v_m)
            nn_m, nn_med = _nn_stats(
                torch.from_numpy(
                    pts_gtd_exp[: min(4000, len(pts_gtd_exp))].astype(np.float32)
                ),
                torch.from_numpy(gt_mesh_m),
            )
            with open(gtd_dir / "stats.txt", "w", encoding="utf-8") as f:
                f.write(
                    f"mesh={stem}\nmethod=gt_depth_gobK "
                    "(nd.exr ⊕ gobK, all /mean(z))\n"
                    f"gob_fx={gob_fx}\nscale_mean_mesh={s_mesh_m}\n"
                    f"scale_mean_gt_depth={s_gtd}\nscale_mean_vggt={s_v_m}\n"
                    f"nn_depth_to_mesh_mean={nn_m}\nnn_depth_to_mesh_med={nn_med}\n"
                    "note: mean_z≈1 expected on each cloud after /mean(z).\n"
                )
            summary_rows.append(
                {
                    "mesh": stem,
                    "method": "gt_depth_gobK",
                    "nn_mean": nn_m,
                    "nn_med": nn_med,
                    "gt_xy_span": st_mesh["xy_span"],
                    "vggt_xy_span": st_gtd["xy_span"],
                }
            )

            # Full depth K ablations at VGGT resolution — /mean(z) each
            if vggt_K is not None:
                vfx, vfy, vcx, vcy = vggt_K
                hd, wd = depth.shape[-2:]
                depth_gt_r = resize_depth_nearest(depth_gt, (hd, wd))
                rgb_r = (
                    torch.nn.functional.interpolate(
                        batch["rgb"].float(),
                        size=(hd, wd),
                        mode="bilinear",
                        align_corners=False,
                    )[0]
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                )
                valid_gt_r = _fg_mask_from_depth_rgb(depth_gt_r, rgb_r)
                gfx, gfy, gcx, gcy = gobjaverse_K_for_vggt_resolution(
                    batch["intrinsics"][0],
                    batch["rgb"] if "rgb" in batch else view_payloads[0]["rgb"].unsqueeze(0),
                    depth_hw=(hd, wd),
                    img_size=builder.img_size,
                )

                def _mean_cloud(dmap, fx, fy, cx, cy, valid, cols_src):
                    cam = depth_map_to_cam_points(dmap, fx=fx, fy=fy, cx=cx, cy=cy)
                    pts = cam[valid]
                    cols = cols_src[valid] if cols_src is not None else None
                    pts_n, s = normalize_by_mean(pts)
                    return pts_n, s, cols

                gt_gob_m, s1, c1 = _mean_cloud(
                    depth_gt_r, gfx, gfy, gcx, gcy, valid_gt_r, rgb_r
                )
                gt_vgk_m, s2, c2 = _mean_cloud(
                    depth_gt_r, vfx, vfy, vcx, vcy, valid_gt_r, rgb_r
                )
                vp_gob_m, s3, c3 = _mean_cloud(
                    depth, gfx, gfy, gcx, gcy, pix_valid, vggt_rgb
                )
                vp_vgk_m, s4, c4 = _mean_cloud(
                    depth, vfx, vfy, vcx, vcy, pix_valid, vggt_rgb
                )

                overlap = valid_gt_r & pix_valid
                a_ss, b_ss, depth_vggt_ss = depth_scale_shift_align(
                    depth, depth_gt_r, overlap
                )
                vp_ss_m, s5, c5 = _mean_cloud(
                    depth_vggt_ss, gfx, gfy, gcx, gcy, overlap, vggt_rgb
                )

                dc_dir = obj_dir / "depth_compare_mean"
                dc_dir.mkdir(parents=True, exist_ok=True)
                export_xyz_pointcloud_ply(
                    gt_mesh_m, dc_dir / "gt_mesh.ply", colors=gt_rgb
                )

                def _dump(name, pts, cols):
                    pe, ce, _ = _subsample_fg(pts, cols, args.max_cloud_points, rng)
                    export_xyz_pointcloud_ply(
                        pe, dc_dir / f"{name}.ply", colors=ce if ce is not None else None
                    )
                    return _log_cloud_stats(stem, f"depth_compare_mean/{name}", pts)

                st = {
                    "gt_depth_gobK": _dump("gt_depth_gobK", gt_gob_m, c1),
                    "gt_depth_vggtK": _dump("gt_depth_vggtK", gt_vgk_m, c2),
                    "vggt_pred_gobK": _dump("vggt_pred_gobK", vp_gob_m, c3),
                    "vggt_pred_vggtK": _dump("vggt_pred_vggtK", vp_vgk_m, c4),
                    "vggt_pred_scaleshift_gobK": _dump(
                        "vggt_pred_scaleshift_gobK", vp_ss_m, c5
                    ),
                }
                nn_raw, nn_raw_med = _nn_stats(
                    torch.from_numpy(vp_vgk_m[: min(4000, len(vp_vgk_m))]),
                    torch.from_numpy(gt_gob_m),
                )
                nn_ss, nn_ss_med = _nn_stats(
                    torch.from_numpy(vp_ss_m[: min(4000, len(vp_ss_m))]),
                    torch.from_numpy(gt_gob_m),
                )
                logger.info(
                    "%s depth_compare_mean: vggtK→gt_gobK NN mean=%.4f | "
                    "scaleshift_gobK→gt_gobK NN mean=%.4f  (a=%.4f b=%.4f)  "
                    "mean_z gt_gob=%.4f vp_gob=%.4f (expect ~1)",
                    stem,
                    nn_raw,
                    nn_ss,
                    a_ss,
                    b_ss,
                    st["gt_depth_gobK"]["mean_z"],
                    st["vggt_pred_gobK"]["mean_z"],
                )
                with open(dc_dir / "stats.txt", "w", encoding="utf-8") as f:
                    f.write(
                        f"mesh={stem}\nmethod=depth_compare_mean\n"
                        "All clouds independently /mean(z) so mean_z→1 "
                        "(not /z_min — avoids nearest-pixel noise).\n"
                        "  gt_depth_gobK     = nd.exr ⊕ gobK@vggt_res\n"
                        "  gt_depth_vggtK    = nd.exr ⊕ VGGT K (FoV-only on GT depth)\n"
                        "  vggt_pred_gobK    = VGGT depth ⊕ gobK\n"
                        "  vggt_pred_vggtK   = VGGT depth ⊕ VGGT K (PE path)\n"
                        "  vggt_pred_scaleshift_gobK = scale-shift align depth→GT "
                        "then ⊕ gobK, then /mean(z)\n"
                    )
                    f.write(
                        f"gobK@vggt=({gfx},{gfy},{gcx},{gcy})\n"
                        f"vggtK=({vfx},{vfy},{vcx},{vcy})\n"
                        f"depth_scaleshift_a={a_ss}\ndepth_scaleshift_b={b_ss}\n"
                        f"nn_vggt_pred_vggtK_to_gt_gobK_mean={nn_raw}\n"
                        f"nn_vggt_pred_vggtK_to_gt_gobK_med={nn_raw_med}\n"
                        f"nn_scaleshift_gobK_to_gt_gobK_mean={nn_ss}\n"
                        f"nn_scaleshift_gobK_to_gt_gobK_med={nn_ss_med}\n"
                        f"scale_mean: gt_gob={s1} gt_vgk={s2} "
                        f"vp_gob={s3} vp_vgk={s4} vp_ss={s5}\n"
                    )
                    for name, st_i in st.items():
                        f.write(f"\n[{name}]\n")
                        for k, v in st_i.items():
                            f.write(f"  {k}={v}\n")
                summary_rows.append(
                    {
                        "mesh": stem,
                        "method": "depth_compare_mean",
                        "nn_mean": nn_raw,
                        "nn_med": nn_raw_med,
                        "gt_xy_span": st["gt_depth_gobK"]["xy_span"],
                        "vggt_xy_span": st["vggt_pred_vggtK"]["xy_span"],
                    }
                )
                summary_rows.append(
                    {
                        "mesh": stem,
                        "method": "depth_compare_scaleshift_mean",
                        "nn_mean": nn_ss,
                        "nn_med": nn_ss_med,
                        "gt_xy_span": st["gt_depth_gobK"]["xy_span"],
                        "vggt_xy_span": st["vggt_pred_scaleshift_gobK"]["xy_span"],
                    }
                )

                # --- sol1 / sol2 / sol3 training-frame recipes ---
                _, scale_y, scale_x, _pad_top, _pad_left = preprocess_rgb_for_vggt(
                    view_payloads[0]["rgb"].float().unsqueeze(0).clamp(0, 1),
                    target_size=builder.img_size,
                )

                # Shared PE cloud for sol1/sol2: VGGT depth ⊕ vggtK / mean(z)
                # (kept for sol1/sol2 / fair_vggtK shared-bbox algebra)
                vggt_vk_m, s_vggt_vk = normalize_by_mean(pts_full)

                # sol1: shear-free FoV on GT, then /mean; PE = vggtK /mean
                gt_fx_raw, sx_fx, sy_fx = fx_scale_gt_to_vggt_k(
                    gt_cam.numpy(),
                    gob_fx=gob_fx,
                    gob_fy=gob_fy,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    vggt_fx=vfx,
                    vggt_fy=vfy,
                )
                gt_fx_m, s_gt_fx = normalize_by_mean(gt_fx_raw)
                _write_pair_export(
                    stem=stem,
                    method_dir=obj_dir / "sol1_fxscale_vggtK_mean",
                    tag="sol1_fxscale_vggtK_mean",
                    label=(
                        "PE=VGGT⊕vggtK /mean; GT=fxscale(xy*=f_gob@518/f_vggt) "
                        "then /mean — train=infer PE, FoV-matched GT"
                    ),
                    gt_xyz=gt_fx_m,
                    gt_rgb=gt_rgb,
                    other_xyz=vggt_vk_m,
                    other_cols=cols_full,
                    other_name="vggt_pred_vggtK",
                    max_cloud_points=args.max_cloud_points,
                    rng=rng,
                    summary_rows=summary_rows,
                    extra_stats={
                        "xy_scale_x": sx_fx,
                        "xy_scale_y": sy_fx,
                        "scale_mean_gt_fxscaled": s_gt_fx,
                        "scale_mean_vggt_vggtK": s_vggt_vk,
                        "gob_fx_at_518": gob_fx * scale_x,
                        "vggt_fx": vfx,
                    },
                )

                # sol2: isotropic only — both /own mean(z); PE = vggtK
                gt_own_m, s_gt_own = normalize_by_mean(gt_cam)
                _write_pair_export(
                    stem=stem,
                    method_dir=obj_dir / "sol2_iso_vggtK_mean",
                    tag="sol2_iso_vggtK_mean",
                    label=(
                        "PE=VGGT⊕vggtK /mean; GT=mesh /own mean — train=infer PE, "
                        "no FoV warp (residual xy from K mismatch)"
                    ),
                    gt_xyz=gt_own_m,
                    gt_rgb=gt_rgb,
                    other_xyz=vggt_vk_m,
                    other_cols=cols_full,
                    other_name="vggt_pred_vggtK",
                    max_cloud_points=args.max_cloud_points,
                    rng=rng,
                    summary_rows=summary_rows,
                    extra_stats={
                        "scale_mean_gt_own": s_gt_own,
                        "scale_mean_vggt_vggtK": s_vggt_vk,
                    },
                )

                # sol3: gobK train PE + mesh / GT-depth mean, shared Hunyuan bbox;
                # also transform patch centres (kept + discarded) — real PE path.
                cam_vggt_gob = depth_map_to_cam_points(
                    depth, fx=gfx, fy=gfy, cx=gcx, cy=gcy
                )
                pts_vggt_gob_raw = cam_vggt_gob[pix_valid]
                vggt_gob_m, s_vggt_gob = normalize_by_mean(pts_vggt_gob_raw)
                _, s_gtd_raw = normalize_by_mean(pts_gtd)
                gt_by_gtd = (_as_xyz_np(gt_cam) / max(s_gtd_raw, 1e-6)).astype(
                    np.float32
                )

                full_c_gob, full_keep_gob = patch_centers_from_depth(
                    cam_vggt_gob,
                    pix_valid,
                    patch_size=builder.patch_size,
                    fx=gfx,
                    fy=gfy,
                    cx=gcx,
                    cy=gcy,
                )
                cen_gob_keep_raw = (
                    full_c_gob[full_keep_gob] if full_keep_gob.any() else full_c_gob[:0]
                )
                cen_gob_disc_raw = (
                    full_c_gob[~full_keep_gob]
                    if (~full_keep_gob).any()
                    else full_c_gob[:0]
                )
                cen_vk_keep_raw = (
                    full_c[full_keep] if full_keep.any() else full_c[:0]
                )
                cen_vk_disc_raw = (
                    full_c[~full_keep] if (~full_keep).any() else full_c[:0]
                )

                (vggt_gob_c, gt_by_gtd_c), mu_bbox, s_bbox = shared_canonicalize_from_ref(
                    vggt_gob_m,
                    vggt_gob_m,
                    gt_by_gtd,
                    scale="bbox",
                    fill=0.9999,
                )
                cen_gob_keep_c = _apply_mean_then_bbox(
                    cen_gob_keep_raw, s_vggt_gob, mu_bbox, s_bbox
                )
                cen_gob_disc_c = _apply_mean_then_bbox(
                    cen_gob_disc_raw, s_vggt_gob, mu_bbox, s_bbox
                )

                # Inference ablation: vggtK own Hunyuan (no /mean; ≡ /mean+bbox)
                (vggt_vk_c,), mu_vk, s_vk = shared_canonicalize_from_ref(
                    pts_full, pts_full, scale="bbox", fill=0.9999
                )
                cen_vk_keep_c = _apply_mean_then_bbox(
                    cen_vk_keep_raw, 1.0, mu_vk, s_vk
                )
                cen_vk_disc_c = _apply_mean_then_bbox(
                    cen_vk_disc_raw, 1.0, mu_vk, s_vk
                )

                sol3 = obj_dir / "sol3_gobK_train"
                sol3.mkdir(parents=True, exist_ok=True)

                def _sol3_dump(name, pts, cols=None, *, rgb=None):
                    arr = _as_xyz_np(pts).astype(np.float32)
                    if arr.shape[0] == 0:
                        return _log_cloud_stats(stem, f"sol3_gobK_train/{name}", arr)
                    pe, ce, _ = _subsample_fg(
                        arr, cols, args.max_cloud_points, rng
                    )
                    kw = {}
                    if ce is not None:
                        kw["colors"] = ce
                    elif rgb is not None:
                        kw["rgb"] = rgb
                    export_xyz_pointcloud_ply(pe, sol3 / f"{name}.ply", **kw)
                    return _log_cloud_stats(stem, f"sol3_gobK_train/{name}", arr)

                st3 = {
                    "vggt_pred_gobK": _sol3_dump(
                        "vggt_pred_gobK", vggt_gob_c, cols_full
                    ),
                    "vggt_pred_vggtK": _sol3_dump(
                        "vggt_pred_vggtK", vggt_vk_c, cols_full
                    ),
                    "patch_centers_gobK": _sol3_dump(
                        "patch_centers_gobK",
                        cen_gob_keep_c,
                        rgb=(32, 200, 64),
                    ),
                    "discarded_centers_gobK": _sol3_dump(
                        "discarded_centers_gobK",
                        cen_gob_disc_c,
                        rgb=(220, 40, 40),
                    ),
                    "patch_centers_vggtK": _sol3_dump(
                        "patch_centers_vggtK",
                        cen_vk_keep_c,
                        rgb=(32, 200, 64),
                    ),
                    "discarded_centers_vggtK": _sol3_dump(
                        "discarded_centers_vggtK",
                        cen_vk_disc_c,
                        rgb=(220, 40, 40),
                    ),
                }
                export_xyz_pointcloud_ply(
                    gt_by_gtd_c,
                    sol3 / "gt_mesh_by_gtdepth_mean.ply",
                    colors=gt_rgb,
                )
                st3["gt_mesh_by_gtdepth_mean"] = _log_cloud_stats(
                    stem, "sol3_gobK_train/gt_mesh_by_gtdepth_mean", gt_by_gtd_c
                )

                nn_gob_mesh, nn_gob_mesh_med = _nn_stats(
                    torch.from_numpy(vggt_gob_c[: min(4000, len(vggt_gob_c))]),
                    torch.from_numpy(gt_by_gtd_c),
                )
                nn_vk_mesh, nn_vk_mesh_med = _nn_stats(
                    torch.from_numpy(vggt_vk_c[: min(4000, len(vggt_vk_c))]),
                    torch.from_numpy(gt_by_gtd_c),
                )
                nn_cen_gob, nn_cen_gob_med = (
                    _nn_stats(
                        torch.from_numpy(
                            cen_gob_keep_c[: min(4000, len(cen_gob_keep_c))]
                        ),
                        torch.from_numpy(gt_by_gtd_c),
                    )
                    if len(cen_gob_keep_c) > 0
                    else (float("nan"), float("nan"))
                )
                nn_cen_vk, nn_cen_vk_med = (
                    _nn_stats(
                        torch.from_numpy(
                            cen_vk_keep_c[: min(4000, len(cen_vk_keep_c))]
                        ),
                        torch.from_numpy(gt_by_gtd_c),
                    )
                    if len(cen_vk_keep_c) > 0
                    else (float("nan"), float("nan"))
                )
                logger.info(
                    "%s sol3_gobK_train: bbox s=%.4f | "
                    "vggt_gob→mesh NN=%.4f | centres_gob→mesh NN=%.4f | "
                    "centres_vggtK→mesh NN=%.4f",
                    stem,
                    s_bbox,
                    nn_gob_mesh,
                    nn_cen_gob,
                    nn_cen_vk,
                )
                with open(sol3 / "stats.txt", "w", encoding="utf-8") as f:
                    f.write(
                        f"mesh={stem}\nmethod=sol3_gobK_train\n"
                        "1) /mean(z): VGGT FG ⊕ gobK; mesh / mean_z(GT depth⊕gobK).\n"
                        "2) Shared Hunyuan bbox from VGGT gobK FG: "
                        "p'=(p-μ)/s, μ=bbox_center, s=max_side/(2*0.9999).\n"
                        "Patch centres use the same mean_z + (μ,s) as the FG cloud.\n"
                        "  gt_mesh_by_gtdepth_mean = train GT\n"
                        "  vggt_pred_gobK          = dense FG (debug)\n"
                        "  patch_centers_gobK      = kept PE centres (train)\n"
                        "  discarded_centers_gobK  = dropped centres (train)\n"
                        "  vggt_pred_vggtK         = infer FoV dense FG "
                        "(own Hunyuan, no /mean)\n"
                        "  patch_centers_vggtK     = kept PE centres (infer ablation)\n"
                        "  discarded_centers_vggtK = dropped centres (infer)\n"
                    )
                    f.write(
                        f"gobK@vggt=({gfx},{gfy},{gcx},{gcy})\n"
                        f"gobK_native=({gob_fx},{gob_fy},{gob_cx},{gob_cy})\n"
                        f"vggtK=({vfx},{vfy},{vcx},{vcy})\n"
                        f"scale_mean_vggt_gobK={s_vggt_gob}\n"
                        f"scale_mean_gt_depth_gobK={s_gtd_raw}\n"
                        f"scale_mean_vggt_vggtK={s_vggt_vk}\n"
                        f"bbox_mu_x={mu_bbox[0]}\nbbox_mu_y={mu_bbox[1]}\n"
                        f"bbox_mu_z={mu_bbox[2]}\nbbox_s={s_bbox}\n"
                        f"infer_bbox_mu_x={mu_vk[0]}\ninfer_bbox_mu_y={mu_vk[1]}\n"
                        f"infer_bbox_mu_z={mu_vk[2]}\ninfer_bbox_s={s_vk}\n"
                        f"n_patch_keep_gobK={len(cen_gob_keep_c)}\n"
                        f"n_patch_disc_gobK={len(cen_gob_disc_c)}\n"
                        f"n_patch_keep_vggtK={len(cen_vk_keep_c)}\n"
                        f"n_patch_disc_vggtK={len(cen_vk_disc_c)}\n"
                        f"nn_vggt_gob_to_mesh_mean={nn_gob_mesh}\n"
                        f"nn_vggt_gob_to_mesh_med={nn_gob_mesh_med}\n"
                        f"nn_vggt_vggtK_to_mesh_mean={nn_vk_mesh}\n"
                        f"nn_vggt_vggtK_to_mesh_med={nn_vk_mesh_med}\n"
                        f"nn_centres_gob_to_mesh_mean={nn_cen_gob}\n"
                        f"nn_centres_gob_to_mesh_med={nn_cen_gob_med}\n"
                        f"nn_centres_vggtK_to_mesh_mean={nn_cen_vk}\n"
                        f"nn_centres_vggtK_to_mesh_med={nn_cen_vk_med}\n"
                    )
                    for name, st_i in st3.items():
                        f.write(f"\n[{name}]\n")
                        for k, v in st_i.items():
                            f.write(f"  {k}={v}\n")
                summary_rows.append(
                    {
                        "mesh": stem,
                        "method": "sol3_gobK_train",
                        "nn_mean": nn_cen_gob,
                        "nn_med": nn_cen_gob_med,
                        "gt_xy_span": st3["gt_mesh_by_gtdepth_mean"]["xy_span"],
                        "vggt_xy_span": st3["patch_centers_gobK"]["xy_span"],
                    }
                )
                summary_rows.append(
                    {
                        "mesh": stem,
                        "method": "sol3_infer_ablation_vggtK",
                        "nn_mean": nn_cen_vk,
                        "nn_med": nn_cen_vk_med,
                        "gt_xy_span": st3["gt_mesh_by_gtdepth_mean"]["xy_span"],
                        "vggt_xy_span": st3["patch_centers_vggtK"]["xy_span"],
                    }
                )

                # --- k_bakeoff: fair gobK / fair vggtK / cross ---
                kb = obj_dir / "k_bakeoff"
                kb.mkdir(parents=True, exist_ok=True)
                with open(kb / "README.txt", "w", encoding="utf-8") as f:
                    f.write(
                        "K bakeoff (mesh always / mean_z(GT depth⊕gobK) first).\n"
                        "  fair_gobK/            PE⊕gobK + shared Hunyuan bbox from that PE\n"
                        "  fair_vggtK/           PE⊕vggtK + shared Hunyuan bbox from that PE\n"
                        "  cross_gobGT_vggtK_pe/ GT uses gobK (μ,s); PE uses vggtK own bbox\n"
                        "Primary metrics: patch_centers vs gt_mesh (see metrics.txt).\n"
                    )

                # fair_gobK (same framing as sol3 train)
                (pe_gob_f, mesh_gob_f), mu_g, s_g = shared_canonicalize_from_ref(
                    vggt_gob_m,
                    vggt_gob_m,
                    gt_by_gtd,
                    scale="bbox",
                    fill=0.9999,
                )
                _write_k_bakeoff_experiment(
                    stem=stem,
                    exp_dir=kb / "fair_gobK",
                    tag="fair_gobK",
                    label=(
                        "Fair train recipe: PE=VGGT⊕gobK /mean + shared bbox; "
                        "GT=mesh/m_gtd + same (μ,s)."
                    ),
                    gt_mesh=mesh_gob_f,
                    gt_rgb=gt_rgb,
                    pe_fg=pe_gob_f,
                    pe_cols=cols_full,
                    cen_keep=_apply_mean_then_bbox(
                        cen_gob_keep_raw, s_vggt_gob, mu_g, s_g
                    ),
                    cen_disc=_apply_mean_then_bbox(
                        cen_gob_disc_raw, s_vggt_gob, mu_g, s_g
                    ),
                    max_cloud_points=args.max_cloud_points,
                    rng=rng,
                    bakeoff_rows=bakeoff_rows,
                    extra={
                        "mean_z_pe": s_vggt_gob,
                        "mean_z_gt_depth": s_gtd_raw,
                        "bbox_mu_x": float(mu_g[0]),
                        "bbox_mu_y": float(mu_g[1]),
                        "bbox_mu_z": float(mu_g[2]),
                        "bbox_s": float(s_g),
                        "pe_K": 0.0,  # 0=gobK
                    },
                )

                # fair_vggtK (train = infer)
                (pe_vk_f, mesh_vk_f), mu_f, s_f = shared_canonicalize_from_ref(
                    vggt_vk_m,
                    vggt_vk_m,
                    gt_by_gtd,
                    scale="bbox",
                    fill=0.9999,
                )
                _write_k_bakeoff_experiment(
                    stem=stem,
                    exp_dir=kb / "fair_vggtK",
                    tag="fair_vggtK",
                    label=(
                        "Fair train=infer recipe: PE=VGGT⊕vggtK /mean + shared bbox; "
                        "GT=mesh/m_gtd + same (μ,s)."
                    ),
                    gt_mesh=mesh_vk_f,
                    gt_rgb=gt_rgb,
                    pe_fg=pe_vk_f,
                    pe_cols=cols_full,
                    cen_keep=_apply_mean_then_bbox(
                        cen_vk_keep_raw, s_vggt_vk, mu_f, s_f
                    ),
                    cen_disc=_apply_mean_then_bbox(
                        cen_vk_disc_raw, s_vggt_vk, mu_f, s_f
                    ),
                    max_cloud_points=args.max_cloud_points,
                    rng=rng,
                    bakeoff_rows=bakeoff_rows,
                    extra={
                        "mean_z_pe": s_vggt_vk,
                        "mean_z_gt_depth": s_gtd_raw,
                        "bbox_mu_x": float(mu_f[0]),
                        "bbox_mu_y": float(mu_f[1]),
                        "bbox_mu_z": float(mu_f[2]),
                        "bbox_s": float(s_f),
                        "pe_K": 1.0,  # 1=vggtK
                    },
                )

                # cross: GT boxed with gobK; PE = vggtK own box
                _write_k_bakeoff_experiment(
                    stem=stem,
                    exp_dir=kb / "cross_gobGT_vggtK_pe",
                    tag="cross_gobGT_vggtK_pe",
                    label=(
                        "Cross/OOD probe: GT uses gobK (μ,s); PE=vggtK own Hunyuan "
                        "(no /mean; ≈ gobK-train → vggtK-infer)."
                    ),
                    gt_mesh=mesh_gob_f,
                    gt_rgb=gt_rgb,
                    pe_fg=vggt_vk_c,
                    pe_cols=cols_full,
                    cen_keep=cen_vk_keep_c,
                    cen_disc=cen_vk_disc_c,
                    max_cloud_points=args.max_cloud_points,
                    rng=rng,
                    bakeoff_rows=bakeoff_rows,
                    extra={
                        "mean_z_pe": s_vggt_vk,
                        "mean_z_gt_depth": s_gtd_raw,
                        "gt_bbox_s": float(s_g),
                        "pe_bbox_s": float(s_vk),
                        "pe_K": 1.0,
                    },
                )

                # --- gt_norm_ablation: cross vs direct VGGT-gobK vs direct GT-depth ---
                # B: /mean then Hunyuan from the *same* cloud is algebraically identical to
                #    Hunyuan on raw points; A is NOT (m_d from GT depth ≠ m_v from VGGT).
                gna = obj_dir / "gt_norm_ablation"
                gna.mkdir(parents=True, exist_ok=True)
                pe_centres = dict(
                    cen_keep=cen_vk_keep_c,
                    cen_disc=cen_vk_disc_c,
                )
                with open(gna / "README.txt", "w", encoding="utf-8") as f:
                    f.write(
                        "GT normalization ablations (mesh always starts as mesh⊕c2w).\n"
                        "PE cloud A/B/C*/D is the SAME: VGGT⊕vggtK own Hunyuan "
                        "(no /mean; ≡ /mean+bbox on same cloud),\n"
                        "plus patch_centers.ply (green) and discarded_centers.ply (red).\n"
                        "\n"
                        "A_cross_train/  [train cross]\n"
                        "  GT  = mesh / mean_z(GT depth⊕gobK) + Hunyuan from VGGT⊕gobK after /mean\n"
                        "\n"
                        "B_vggt_gobK_direct/\n"
                        "  GT  = Hunyuan from raw VGGT⊕gobK (/mean redundant)\n"
                        "\n"
                        "C_gt_depth_direct/  [baseline]\n"
                        "  GT  = Hunyuan from raw GT depth⊕gobK (min/max AABB, no filter)\n"
                        "\n"
                        "C_gt_depth_filter/  [filter only — not stacked with zrobust]\n"
                        "  erode FG 1px only (no z-percentile), then standard Hunyuan AABB\n"
                        "  gt_depth_used.ply = kept FG; gt_depth_discarded.ply = eroded ring (red)\n"
                        "\n"
                        "C_gt_depth_zrobust/  [stats only — unfiltered cloud]\n"
                        "  Hunyuan xy=min/max, z lo=p1 hi=max (near-plane flyers only)\n"
                        "  gt_depth_used.ply = z>=p1; gt_depth_discarded.ply = near tail (red)\n"
                        "\n"
                        "C_gt_depth_filter_zrobust/  [stack: filter then few-extrema stats]\n"
                        "  erode FG 1px, then Hunyuan xy=min/max, z lo=p0.5 hi=max\n"
                        "  (ignore leftover near flyers without a heavy percentile cut)\n"
                        "  discarded = erode ring + z<p0.5 of remaining (red)\n"
                        "\n"
                        "D_mesh_own_indep/  [naive / supervisor]\n"
                        "  GT  = mesh own Hunyuan AABB\n"
                    )

                # A: identical to k_bakeoff/cross (train recipe)
                _write_pair_export(
                    stem=stem,
                    method_dir=gna / "A_cross_train",
                    tag="gt_norm_A_cross_train",
                    label=(
                        "Train cross: GT=mesh/mean(GT depth)+Hunyuan(VGGT⊕gobK); "
                        "VGGT=vggtK own Hunyuan (no /mean)"
                    ),
                    gt_xyz=mesh_gob_f,
                    gt_rgb=gt_rgb,
                    other_xyz=vggt_vk_c,
                    other_cols=cols_full,
                    other_name="vggt_pred_vggtK",
                    max_cloud_points=args.max_cloud_points,
                    rng=rng,
                    summary_rows=summary_rows,
                    extra_stats={
                        "mean_z_gt_depth": float(s_gtd_raw),
                        "mean_z_vggt_gobK": float(s_vggt_gob),
                        "mean_z_vggt_vggtK": float(s_vggt_vk),
                        "gt_bbox_s": float(s_g),
                        "pe_bbox_s": float(s_vk),
                        "gt_bbox_mu_x": float(mu_g[0]),
                        "gt_bbox_mu_y": float(mu_g[1]),
                        "gt_bbox_mu_z": float(mu_g[2]),
                    },
                    extra_txt=(
                        "note: /mean uses GT depth; Hunyuan (μ,s) from VGGT⊕gobK "
                        "(not mesh AABB). PE uses vggtK own bbox.\n"
                    ),
                    **pe_centres,
                )

                # B: Hunyuan mesh from raw VGGT⊕gobK; PE = same vggtK own bbox as A/C/D
                (gt_b,), mu_b, s_b = shared_canonicalize_from_ref(
                    pts_vggt_gob_raw,
                    _as_xyz_np(gt_cam),
                    scale="bbox",
                    fill=0.9999,
                )
                # Sanity: /mean_v then bbox from VGGT⊕gobK == direct Hunyuan on mesh
                gt_b_via_mean = _apply_mean_then_bbox(
                    gt_cam, s_vggt_gob, mu_g, s_g
                )
                nn_equiv, _ = _nn_stats(
                    torch.from_numpy(gt_b[: min(4000, len(gt_b))]),
                    torch.from_numpy(gt_b_via_mean),
                )
                _write_pair_export(
                    stem=stem,
                    method_dir=gna / "B_vggt_gobK_direct",
                    tag="gt_norm_B_vggt_gobK_direct",
                    label=(
                        "Mesh Hunyuan from raw VGGT⊕gobK (/mean redundant); "
                        "VGGT=vggtK/mean+own bbox (same PE as A)"
                    ),
                    gt_xyz=gt_b,
                    gt_rgb=gt_rgb,
                    other_xyz=vggt_vk_c,
                    other_cols=cols_full,
                    other_name="vggt_pred_vggtK",
                    max_cloud_points=args.max_cloud_points,
                    rng=rng,
                    summary_rows=summary_rows,
                    extra_stats={
                        "mean_z_vggt_gobK": float(s_vggt_gob),
                        "bbox_s_raw": float(s_b),
                        "bbox_mu_x": float(mu_b[0]),
                        "bbox_mu_y": float(mu_b[1]),
                        "bbox_mu_z": float(mu_b[2]),
                        "nn_direct_vs_mean_then_bbox": float(nn_equiv),
                    },
                    extra_txt=(
                        "note: nn_direct_vs_mean_then_bbox ≈ 0: Hunyuan(raw VGGT⊕gobK) "
                        "≡ mesh/mean_v + bbox after /mean. Differs from A (A uses m_d).\n"
                    ),
                    **pe_centres,
                )

                # C: Hunyuan mesh from raw GT depth⊕gobK; same (μ,s) on GT depth; PE = vggtK
                (gt_c, gtd_c), mu_c, s_c = shared_canonicalize_from_ref(
                    pts_gtd,
                    _as_xyz_np(gt_cam),
                    pts_gtd,
                    scale="bbox",
                    fill=0.9999,
                )
                nn_gtd, nn_gtd_med = _nn_stats(
                    torch.from_numpy(gtd_c[: min(4000, len(gtd_c))]),
                    torch.from_numpy(gt_c),
                )
                _write_pair_export(
                    stem=stem,
                    method_dir=gna / "C_gt_depth_direct",
                    tag="gt_norm_C_gt_depth_direct",
                    label=(
                        "Mesh Hunyuan from raw GT depth⊕gobK; "
                        "gt_depth_gobK uses the same (μ,s); "
                        "VGGT=vggtK/mean+own bbox"
                    ),
                    gt_xyz=gt_c,
                    gt_rgb=gt_rgb,
                    other_xyz=vggt_vk_c,
                    other_cols=cols_full,
                    other_name="vggt_pred_vggtK",
                    max_cloud_points=args.max_cloud_points,
                    rng=rng,
                    summary_rows=summary_rows,
                    extra_stats={
                        "mean_z_gt_depth": float(s_gtd_raw),
                        "bbox_s_raw": float(s_c),
                        "bbox_mu_x": float(mu_c[0]),
                        "bbox_mu_y": float(mu_c[1]),
                        "bbox_mu_z": float(mu_c[2]),
                        "pe_bbox_s": float(s_vk),
                        "nn_gtdepth_to_mesh_mean": float(nn_gtd),
                        "nn_gtdepth_to_mesh_med": float(nn_gtd_med),
                    },
                    extra_txt=(
                        "note: gt_depth_gobK.ply uses the same (μ,s) as gt.ply "
                        "(mesh↔GT-depth should overlay). PE is vggtK own bbox.\n"
                    ),
                    **pe_centres,
                )
                _append_gt_depth_ply(
                    stem=stem,
                    tag="gt_norm_C_gt_depth_direct",
                    method_dir=gna / "C_gt_depth_direct",
                    gtd_xyz=gtd_c,
                    gtd_cols=cols_gtd,
                    max_cloud_points=args.max_cloud_points,
                    rng=rng,
                )

                # C_filter: erode 1px only, then standard Hunyuan (NOT stacked with zrobust)
                valid_f = _filter_gt_depth_mask(valid_gt, erode_iters=1)
                valid_disc_f = valid_gt & ~valid_f
                pts_gtd_f = cam_gt_gob[valid_f]
                cols_gtd_f = (
                    rgb_np[valid_f] if rgb_np.shape[:2] == depth_gt.shape else None
                )
                pts_gtd_disc_f = cam_gt_gob[valid_disc_f]
                logger.info(
                    "%s C_gt_depth_filter: n_gtd %d → %d (erode=1, no z-percentile)",
                    stem,
                    int(len(pts_gtd)),
                    int(len(pts_gtd_f)),
                )
                (gt_cf, gtd_cf), mu_cf, s_cf = shared_canonicalize_from_ref(
                    pts_gtd_f,
                    _as_xyz_np(gt_cam),
                    pts_gtd_f,
                    scale="bbox",
                    fill=0.9999,
                )
                gtd_disc_cf = _apply_mu_s(pts_gtd_disc_f, mu_cf, s_cf)
                if len(gtd_cf) > 0:
                    nn_gtd_f, nn_gtd_f_med = _nn_stats(
                        torch.from_numpy(gtd_cf[: min(4000, len(gtd_cf))]),
                        torch.from_numpy(gt_cf),
                    )
                else:
                    nn_gtd_f, nn_gtd_f_med = float("nan"), float("nan")
                _write_pair_export(
                    stem=stem,
                    method_dir=gna / "C_gt_depth_filter",
                    tag="gt_norm_C_gt_depth_filter",
                    label=(
                        "Filter-only: erode FG 1px (no z-percentile), then Hunyuan AABB; "
                        "VGGT=vggtK own bbox"
                    ),
                    gt_xyz=gt_cf,
                    gt_rgb=gt_rgb,
                    other_xyz=vggt_vk_c,
                    other_cols=cols_full,
                    other_name="vggt_pred_vggtK",
                    max_cloud_points=args.max_cloud_points,
                    rng=rng,
                    summary_rows=summary_rows,
                    extra_stats={
                        "n_gtd_raw": float(len(pts_gtd)),
                        "n_gtd_used": float(len(pts_gtd_f)),
                        "n_gtd_discarded": float(len(pts_gtd_disc_f)),
                        "bbox_s": float(s_cf),
                        "bbox_mu_x": float(mu_cf[0]),
                        "bbox_mu_y": float(mu_cf[1]),
                        "bbox_mu_z": float(mu_cf[2]),
                        "pe_bbox_s": float(s_vk),
                        "nn_gtdepth_to_mesh_mean": float(nn_gtd_f),
                        "nn_gtdepth_to_mesh_med": float(nn_gtd_f_med),
                    },
                    extra_txt=(
                        "note: NOT stacked with zrobust. gt_depth_used.ply = eroded FG; "
                        "gt_depth_discarded.ply = 1px silhouette ring (red).\n"
                    ),
                    **pe_centres,
                )
                _c_filt_dir = gna / "C_gt_depth_filter"
                _append_gt_depth_ply(
                    stem=stem,
                    tag="gt_norm_C_gt_depth_filter",
                    method_dir=_c_filt_dir,
                    gtd_xyz=gtd_cf,
                    gtd_cols=cols_gtd_f,
                    max_cloud_points=args.max_cloud_points,
                    rng=rng,
                )
                _append_gt_depth_ply(
                    stem=stem,
                    tag="gt_norm_C_gt_depth_filter",
                    method_dir=_c_filt_dir,
                    gtd_xyz=gtd_cf,
                    gtd_cols=cols_gtd_f,
                    max_cloud_points=args.max_cloud_points,
                    rng=rng,
                    name="gt_depth_used",
                )
                _append_gt_depth_ply(
                    stem=stem,
                    tag="gt_norm_C_gt_depth_filter",
                    method_dir=_c_filt_dir,
                    gtd_xyz=gtd_disc_cf,
                    max_cloud_points=args.max_cloud_points,
                    rng=rng,
                    name="gt_depth_discarded",
                    rgb=(220, 40, 40),
                )

                # C_zrobust: unfiltered GT depth; xy min/max, z lo=p1 hi=max (near tail only)
                z_gtd = _as_xyz_np(pts_gtd)[:, 2]
                z_p1 = (
                    float(np.percentile(z_gtd, 1.0)) if len(z_gtd) else float("nan")
                )
                used_z = z_gtd >= z_p1 if len(z_gtd) else np.zeros(0, dtype=bool)
                (gt_cz, gtd_cz), mu_cz, s_cz = shared_canonicalize_from_ref(
                    pts_gtd,
                    _as_xyz_np(gt_cam),
                    pts_gtd,
                    scale="bbox_zrobust",
                    fill=0.9999,
                )
                gtd_used_cz = _apply_mu_s(pts_gtd[used_z], mu_cz, s_cz)
                gtd_disc_cz = _apply_mu_s(pts_gtd[~used_z], mu_cz, s_cz)
                cols_used_z = (
                    cols_gtd[used_z] if cols_gtd is not None and len(pts_gtd) else None
                )
                nn_gtd_z, nn_gtd_z_med = _nn_stats(
                    torch.from_numpy(gtd_cz[: min(4000, len(gtd_cz))]),
                    torch.from_numpy(gt_cz),
                )
                _write_pair_export(
                    stem=stem,
                    method_dir=gna / "C_gt_depth_zrobust",
                    tag="gt_norm_C_gt_depth_zrobust",
                    label=(
                        "Stats-only: unfiltered GT depth, Hunyuan xy=min/max z lo=p1 hi=max; "
                        "VGGT=vggtK own bbox"
                    ),
                    gt_xyz=gt_cz,
                    gt_rgb=gt_rgb,
                    other_xyz=vggt_vk_c,
                    other_cols=cols_full,
                    other_name="vggt_pred_vggtK",
                    max_cloud_points=args.max_cloud_points,
                    rng=rng,
                    summary_rows=summary_rows,
                    extra_stats={
                        "bbox_s": float(s_cz),
                        "bbox_mu_x": float(mu_cz[0]),
                        "bbox_mu_y": float(mu_cz[1]),
                        "bbox_mu_z": float(mu_cz[2]),
                        "z_p1": float(z_p1),
                        "n_gtd_used": float(int(used_z.sum())) if len(z_gtd) else 0.0,
                        "n_gtd_discarded": (
                            float(int((~used_z).sum())) if len(z_gtd) else 0.0
                        ),
                        "pe_bbox_s": float(s_vk),
                        "nn_gtdepth_to_mesh_mean": float(nn_gtd_z),
                        "nn_gtdepth_to_mesh_med": float(nn_gtd_z_med),
                    },
                    extra_txt=(
                        "note: NOT stacked on filter. gt_depth_gobK.ply = all points; "
                        "gt_depth_used.ply = z>=p1 (defines lo_z); "
                        "gt_depth_discarded.ply = near tail (red).\n"
                    ),
                    **pe_centres,
                )
                _c_zr_dir = gna / "C_gt_depth_zrobust"
                _append_gt_depth_ply(
                    stem=stem,
                    tag="gt_norm_C_gt_depth_zrobust",
                    method_dir=_c_zr_dir,
                    gtd_xyz=gtd_cz,
                    gtd_cols=cols_gtd,
                    max_cloud_points=args.max_cloud_points,
                    rng=rng,
                )
                _append_gt_depth_ply(
                    stem=stem,
                    tag="gt_norm_C_gt_depth_zrobust",
                    method_dir=_c_zr_dir,
                    gtd_xyz=gtd_used_cz,
                    gtd_cols=cols_used_z,
                    max_cloud_points=args.max_cloud_points,
                    rng=rng,
                    name="gt_depth_used",
                )
                _append_gt_depth_ply(
                    stem=stem,
                    tag="gt_norm_C_gt_depth_zrobust",
                    method_dir=_c_zr_dir,
                    gtd_xyz=gtd_disc_cz,
                    max_cloud_points=args.max_cloud_points,
                    rng=rng,
                    name="gt_depth_discarded",
                    rgb=(220, 40, 40),
                )

                # C_filter_zrobust: erode 1px, then ignore a tiny near-z tail for Hunyuan
                z_p_lo_s = 0.5
                z_f = _as_xyz_np(pts_gtd_f)[:, 2]
                z_thr_s = (
                    float(np.percentile(z_f, z_p_lo_s)) if len(z_f) else float("nan")
                )
                used_s = z_f >= z_thr_s if len(z_f) else np.zeros(0, dtype=bool)
                (gt_cs, gtd_cs), mu_cs, s_cs = shared_canonicalize_from_ref(
                    pts_gtd_f,
                    _as_xyz_np(gt_cam),
                    pts_gtd_f,
                    scale="bbox_zrobust",
                    fill=0.9999,
                    z_p_lo=z_p_lo_s,
                )
                gtd_used_cs = _apply_mu_s(pts_gtd_f[used_s], mu_cs, s_cs)
                cols_used_s = (
                    cols_gtd_f[used_s]
                    if cols_gtd_f is not None and len(pts_gtd_f)
                    else None
                )
                _disc_parts = []
                if len(pts_gtd_disc_f):
                    _disc_parts.append(_as_xyz_np(pts_gtd_disc_f))
                if len(z_f) and np.any(~used_s):
                    _disc_parts.append(_as_xyz_np(pts_gtd_f[~used_s]))
                gtd_disc_cs = _apply_mu_s(
                    np.concatenate(_disc_parts, axis=0)
                    if _disc_parts
                    else np.zeros((0, 3), dtype=np.float32),
                    mu_cs,
                    s_cs,
                )
                logger.info(
                    "%s C_gt_depth_filter_zrobust: erode %d → %d, then z<p%.1f drops %d",
                    stem,
                    int(len(pts_gtd)),
                    int(len(pts_gtd_f)),
                    z_p_lo_s,
                    int((~used_s).sum()) if len(z_f) else 0,
                )
                if len(gtd_used_cs) > 0:
                    nn_gtd_s, nn_gtd_s_med = _nn_stats(
                        torch.from_numpy(gtd_used_cs[: min(4000, len(gtd_used_cs))]),
                        torch.from_numpy(gt_cs),
                    )
                else:
                    nn_gtd_s, nn_gtd_s_med = float("nan"), float("nan")
                _write_pair_export(
                    stem=stem,
                    method_dir=gna / "C_gt_depth_filter_zrobust",
                    tag="gt_norm_C_gt_depth_filter_zrobust",
                    label=(
                        "Stack: erode FG 1px, then Hunyuan xy=min/max z lo=p0.5 hi=max; "
                        "VGGT=vggtK own bbox"
                    ),
                    gt_xyz=gt_cs,
                    gt_rgb=gt_rgb,
                    other_xyz=vggt_vk_c,
                    other_cols=cols_full,
                    other_name="vggt_pred_vggtK",
                    max_cloud_points=args.max_cloud_points,
                    rng=rng,
                    summary_rows=summary_rows,
                    extra_stats={
                        "n_gtd_raw": float(len(pts_gtd)),
                        "n_gtd_eroded": float(len(pts_gtd_f)),
                        "n_gtd_used": float(int(used_s.sum())) if len(z_f) else 0.0,
                        "n_gtd_z_tail": (
                            float(int((~used_s).sum())) if len(z_f) else 0.0
                        ),
                        "n_gtd_erode_ring": float(len(pts_gtd_disc_f)),
                        "z_p_lo": float(z_p_lo_s),
                        "z_thr": float(z_thr_s),
                        "bbox_s": float(s_cs),
                        "bbox_mu_x": float(mu_cs[0]),
                        "bbox_mu_y": float(mu_cs[1]),
                        "bbox_mu_z": float(mu_cs[2]),
                        "pe_bbox_s": float(s_vk),
                        "nn_gtdepth_to_mesh_mean": float(nn_gtd_s),
                        "nn_gtdepth_to_mesh_med": float(nn_gtd_s_med),
                    },
                    extra_txt=(
                        "note: stacked. gt_depth_gobK.ply = eroded FG (leftover flyers "
                        "still visible); gt_depth_used.ply = eroded & z>=p0.5; "
                        "gt_depth_discarded.ply = 1px ring + near tail (red).\n"
                    ),
                    **pe_centres,
                )
                _c_st_dir = gna / "C_gt_depth_filter_zrobust"
                _append_gt_depth_ply(
                    stem=stem,
                    tag="gt_norm_C_gt_depth_filter_zrobust",
                    method_dir=_c_st_dir,
                    gtd_xyz=gtd_cs,
                    gtd_cols=cols_gtd_f,
                    max_cloud_points=args.max_cloud_points,
                    rng=rng,
                )
                _append_gt_depth_ply(
                    stem=stem,
                    tag="gt_norm_C_gt_depth_filter_zrobust",
                    method_dir=_c_st_dir,
                    gtd_xyz=gtd_used_cs,
                    gtd_cols=cols_used_s,
                    max_cloud_points=args.max_cloud_points,
                    rng=rng,
                    name="gt_depth_used",
                )
                _append_gt_depth_ply(
                    stem=stem,
                    tag="gt_norm_C_gt_depth_filter_zrobust",
                    method_dir=_c_st_dir,
                    gtd_xyz=gtd_disc_cs,
                    max_cloud_points=args.max_cloud_points,
                    rng=rng,
                    name="gt_depth_discarded",
                    rgb=(220, 40, 40),
                )

                # D: naive independent Hunyuan (full mesh own AABB vs vggtK own AABB)
                (gt_d,), mu_d, s_d = shared_canonicalize_from_ref(
                    _as_xyz_np(gt_cam),
                    _as_xyz_np(gt_cam),
                    scale="bbox",
                    fill=0.9999,
                )
                _write_pair_export(
                    stem=stem,
                    method_dir=gna / "D_mesh_own_indep",
                    tag="gt_norm_D_mesh_own_indep",
                    label=(
                        "Naive: mesh own Hunyuan AABB vs VGGT⊕vggtK own bbox "
                        "(independent stats; full vs partial)"
                    ),
                    gt_xyz=gt_d,
                    gt_rgb=gt_rgb,
                    other_xyz=vggt_vk_c,
                    other_cols=cols_full,
                    other_name="vggt_pred_vggtK",
                    max_cloud_points=args.max_cloud_points,
                    rng=rng,
                    summary_rows=summary_rows,
                    extra_stats={
                        "mesh_bbox_s": float(s_d),
                        "mesh_bbox_mu_x": float(mu_d[0]),
                        "mesh_bbox_mu_y": float(mu_d[1]),
                        "mesh_bbox_mu_z": float(mu_d[2]),
                        "pe_bbox_s": float(s_vk),
                    },
                    extra_txt=(
                        "note: simplest independent normalize. Full-mesh AABB vs "
                        "partial vggtK PE AABB — expected mismatch (supervisor demo).\n"
                    ),
                    **pe_centres,
                )

                # --- mv_frame0: multi-view union in ref camera (N=1 ≡ single-view) ---
                _export_mv_frame0(
                    stem=stem,
                    obj_dir=obj_dir,
                    view_payloads=view_payloads,
                    raw=raw,
                    raw_mv=raw_mv,
                    builder=builder,
                    gt_cam=gt_cam,
                    gt_rgb=gt_rgb,
                    pts_full=pts_full,
                    cols_full=cols_full,
                    pix_valid=pix_valid,
                    vggt_rgb=vggt_rgb,
                    depth=depth,
                    gfx=gfx,
                    gfy=gfy,
                    gcx=gcx,
                    gcy=gcy,
                    pts_gtd=pts_gtd,
                    s_gtd_raw=s_gtd_raw,
                    pts_vggt_gob_raw=pts_vggt_gob_raw,
                    s_vggt_gob=s_vggt_gob,
                    mu_g=mu_g,
                    s_g=s_g,
                    mesh_gob_f=mesh_gob_f,
                    vggt_vk_c=vggt_vk_c,
                    cen_vk_keep_c=cen_vk_keep_c,
                    cen_vk_disc_c=cen_vk_disc_c,
                    max_cloud_points=args.max_cloud_points,
                    rng=rng,
                    summary_rows=summary_rows,
                )
            else:
                logger.warning(
                    "%s: skip depth_compare_mean / sol1-3 / k_bakeoff / "
                    "gt_norm_ablation (no vggt_K)",
                    stem,
                )
        else:
            logger.warning(
                "%s: skip gt_depth_* / depth_compare / sol1-3 / k_bakeoff / "
                "gt_norm_ablation (no depth/intrinsics)",
                stem,
            )

        # --- scale_zmin_pointmap: VGGT point_head, /z_min ---
        if "vggt_world_points" in raw:
            wpts = raw["vggt_world_points"][0].cpu().numpy()
            if wpts.ndim == 4 and wpts.shape[-1] == 3:
                wpts = wpts[0]
            wconf = None
            if "vggt_world_conf" in raw:
                wconf = raw["vggt_world_conf"][0].cpu().numpy()
                if wconf.ndim == 3:
                    wconf = wconf[..., 0] if wconf.shape[-1] == 1 else wconf[0]
            if wconf is not None and wconf.shape[:2] == wpts.shape[:2]:
                thr = float(np.percentile(wconf[np.isfinite(wconf)], args.conf_percentile))
                pvalid = np.isfinite(wpts).all(axis=-1) & (wconf > thr)
            else:
                pvalid = pix_valid
                if pvalid.shape != wpts.shape[:2]:
                    pvalid_t = torch.from_numpy(pix_valid.astype(np.float32))[None, None]
                    pvalid = (
                        torch.nn.functional.interpolate(
                            pvalid_t, size=wpts.shape[:2], mode="nearest"
                        )[0, 0]
                        .numpy()
                        > 0.5
                    )
            if vggt_rgb.shape[:2] == wpts.shape[:2]:
                white = (
                    (vggt_rgb[..., 0] > 0.97)
                    & (vggt_rgb[..., 1] > 0.97)
                    & (vggt_rgb[..., 2] > 0.97)
                )
                pvalid = pvalid & ~white
                cols_pm = vggt_rgb[pvalid]
            else:
                cols_pm = None
            pts_pm = wpts[pvalid]
            pts_pm_z, s_pm = normalize_by_zmin(pts_pm)
            pm_dir = obj_dir / "scale_zmin_pointmap"
            _write_pair_export(
                stem=stem,
                method_dir=pm_dir,
                tag="scale_zmin_pointmap",
                label="VGGT point_head + /z_min (not depth-unproject)",
                gt_xyz=gt_z,
                gt_rgb=gt_rgb,
                other_xyz=pts_pm_z,
                other_cols=cols_pm,
                other_name="vggt_pointmap",
                max_cloud_points=args.max_cloud_points,
                rng=rng,
                summary_rows=summary_rows,
                extra_stats={
                    "n_pointmap_fg": float(pts_pm.shape[0]),
                    "scale_z_gt": s_gt_z,
                    "scale_z_pointmap": s_pm,
                    "scale_z_vggt_depth": s_v_z,
                },
            )
            pts_d_exp, cols_d_exp, _ = _subsample_fg(
                pts_z, cols_full, args.max_cloud_points, rng
            )
            export_xyz_pointcloud_ply(
                pts_d_exp, pm_dir / "vggt_depth_cloud.ply", colors=cols_d_exp
            )
        else:
            logger.warning(
                "%s: skip scale_zmin_pointmap (vggt_world_points missing)",
                stem,
            )

        logger.info("Wrote %s", obj_dir)

    by_method: Dict[str, List[float]] = {}
    for row in summary_rows:
        by_method.setdefault(row["method"], []).append(row["nn_mean"])
    logger.info("=== Mean centre→GT NN over objects ===")
    for method, vals in sorted(by_method.items(), key=lambda kv: np.nanmean(kv[1])):
        logger.info("  %-28s  mean_nn=%.4f", method, float(np.nanmean(vals)))

    if bakeoff_rows:
        import csv

        csv_path = out_root / "k_bakeoff_summary.csv"
        keys = list(bakeoff_rows[0].keys())
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(bakeoff_rows)
        logger.info("Wrote %s (%d rows)", csv_path, len(bakeoff_rows))
        logger.info("=== k_bakeoff: mean Chamfer (patch centres ↔ GT) ===")
        by_b: Dict[str, List[float]] = {}
        for row in bakeoff_rows:
            by_b.setdefault(row["method"], []).append(float(row["chamfer_mean"]))
        for method, vals in sorted(by_b.items(), key=lambda kv: np.nanmean(kv[1])):
            logger.info(
                "  %-28s  chamfer=%.4f  (n=%d)",
                method,
                float(np.nanmean(vals)),
                len(vals),
            )
        # Win counts: fair_gobK vs fair_vggtK
        gob = {
            r["mesh"]: r for r in bakeoff_rows if r["method"] == "fair_gobK"
        }
        vk = {
            r["mesh"]: r for r in bakeoff_rows if r["method"] == "fair_vggtK"
        }
        common = sorted(set(gob) & set(vk))
        if common:
            wins_gob = sum(
                1
                for m in common
                if gob[m]["chamfer_mean"] < vk[m]["chamfer_mean"]
            )
            logger.info(
                "fair_gobK lower Chamfer than fair_vggtK on %d / %d objects",
                wins_gob,
                len(common),
            )


if __name__ == "__main__":
    main()
