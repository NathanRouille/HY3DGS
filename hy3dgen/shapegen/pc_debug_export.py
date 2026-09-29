"""Debug PLY exports for ShapePCAE / ShapePCUnite reconstruction diagnosis.

Typical CloudCompare overlays (same camera frame):
  gt.ply / recon.ply          — already exported by eval/vis
  fps.ply                     — encoder FPS queries (cyan)
  anchors.ply                 — decoder cluster centres (magenta)
  recon_error.ply             — recon coloured by dist→GT (blue=good → red=far)
  gt_uncovered.ply            — GT coloured by dist→recon
  rgb_error.ply               — recon coloured by L1 RGB vs NN GT
  gt_rgb_error.ply            — GT coloured by L1 RGB vs NN recon
  intruders.ply               — recon points with dist→GT > threshold
  locals_by_anchor.ply        — recon coloured by anchor id
  delta_mag.ply               — recon coloured by |Δ| (vs p95 or hard cap scale)
  patch_centers.ply           — kept VGGT patch centres (green)
  discarded_centers.ply       — background-discarded patch centres (red)
  vggt_points.ply             — VGGT depth FG cloud in train PE frame (gray)
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


def _subsample_xyz(xyz: np.ndarray, max_points: int) -> np.ndarray:
    if xyz.shape[0] <= max_points or max_points <= 0:
        return xyz
    idx = np.linspace(0, xyz.shape[0] - 1, num=max_points, dtype=np.int64)
    return xyz[idx]


@torch.no_grad()
def collect_vggt_debug_clouds(
    builder,
    batch: Dict,
    *,
    align_mode: str,
    device: torch.device,
    sample_index: int = 0,
    max_dense_points: int = 80000,
) -> Dict[str, np.ndarray]:
    """VGGT kept / discarded centres + dense FG cloud in the **train PE frame**.

    Matches training: multi-view centres live in VGGT cam0 (view-1→view-0 via
    VGGT extrinsics); normalize with the same PE recipe as Fourier centres
    (``c_meanrms`` = own mean+RMS of kept centres; ``cross`` = cache Hunyuan bbox).
    """
    from hy3dgen.shapegen.cam_align import apply_pe_normalize_np, select_cached_centers

    if "vggt_cache" in batch and batch["vggt_cache"]:
        payload = batch["vggt_cache"][sample_index]
        if isinstance(payload, dict) and payload.get("cache_kind") == "joint":
            kept_raw = _squeeze_batch(select_cached_centers(payload, align_mode))
            keep = payload.get("patch_keep")
            if keep is not None:
                km = _to_np(keep).reshape(-1).astype(bool)
                if km.shape[0] == kept_raw.shape[0]:
                    kept = kept_raw[km]
                else:
                    kept = kept_raw
            else:
                kept = kept_raw
            disc = _squeeze_batch(payload.get("patch_centers_discarded", []))
            dense = np.asarray(
                payload.get("vggt_points_cam0", np.zeros((0, 3))),
                dtype=np.float32,
            ).reshape(-1, 3)
            dense = _subsample_xyz(dense, int(max_dense_points))
            kept_a = apply_pe_normalize_np(
                kept, mode=align_mode, pe_centers_raw=kept, align_stats=None
            )
            disc_a = apply_pe_normalize_np(
                disc, mode=align_mode, pe_centers_raw=kept, align_stats=None
            )
            dense_a = apply_pe_normalize_np(
                dense, mode=align_mode, pe_centers_raw=kept, align_stats=None
            )
            return {
                "patch_centers": kept_a,
                "discarded_centers": disc_a,
                "vggt_points": dense_a,
            }

    rgb_views = None
    if "rgb_views" in batch:
        rv = batch["rgb_views"]
        rgb_views = rv[sample_index].to(device)  # [S,3,H,W]
    elif "rgb" in batch:
        rgb_views = batch["rgb"][sample_index : sample_index + 1].to(device)

    if rgb_views is None:
        return {}

    raw = builder.extract_vggt_sequence_dense(rgb_views)
    kept = _squeeze_batch(raw["patch_centers"])
    if "patch_keep" in raw:
        keep = _to_np(raw["patch_keep"]).reshape(-1).astype(bool)
        if keep.shape[0] == kept.shape[0]:
            kept = kept[keep]
    disc = (
        _squeeze_batch(raw["patch_centers_discarded"])
        if "patch_centers_discarded" in raw
        else np.zeros((0, 3), np.float32)
    )
    if "vggt_points_cam0" in raw:
        dense = _squeeze_batch(raw["vggt_points_cam0"])
    elif "vggt_cam_points" in raw:
        grid = _to_np(raw["vggt_cam_points"])
        if grid.ndim == 4:
            grid = grid[0]
        if grid.ndim == 3 and grid.shape[-1] == 3:
            depth = _to_np(raw["vggt_depth"])
            conf = _to_np(raw["vggt_depth_conf"])
            while depth.ndim > 2:
                depth = depth[0]
            while conf.ndim > 2:
                conf = conf[0]
            pix = builder._pixel_valid_mask(
                depth,
                conf,
                rgb_np=np.ones((*grid.shape[:2], 3), dtype=np.float32),
            )
            dense = grid[pix].astype(np.float32)
        else:
            dense = grid.reshape(-1, 3).astype(np.float32)
    else:
        dense = np.zeros((0, 3), np.float32)

    dense = _subsample_xyz(dense.astype(np.float32), int(max_dense_points))

    align_stats = None
    if "vggt_cache" in batch and batch["vggt_cache"]:
        payload = batch["vggt_cache"][sample_index]
        if isinstance(payload, dict):
            align_stats = payload.get("align_stats")

    kept_a = apply_pe_normalize_np(
        kept, mode=align_mode, pe_centers_raw=kept, align_stats=align_stats
    )
    disc_a = apply_pe_normalize_np(
        disc, mode=align_mode, pe_centers_raw=kept, align_stats=align_stats
    )
    dense_a = apply_pe_normalize_np(
        dense, mode=align_mode, pe_centers_raw=kept, align_stats=align_stats
    )
    return {
        "patch_centers": kept_a,
        "discarded_centers": disc_a,
        "vggt_points": dense_a,
    }


def export_vggt_debug_plys(
    out_dir: Union[str, Path],
    clouds: Dict[str, np.ndarray],
) -> Dict[str, str]:
    """Write patch_centers / discarded_centers / vggt_points PLYs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}
    specs = (
        ("patch_centers", (32, 200, 64)),
        ("discarded_centers", (220, 40, 40)),
        ("vggt_points", (160, 160, 160)),
    )
    for name, rgb in specs:
        pts = clouds.get(name)
        if pts is None:
            continue
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        p = out_dir / f"{name}.ply"
        export_xyz_pointcloud_ply(pts, p, rgb=rgb)
        written[name] = str(p)
    readme = out_dir / "VGGT_DEBUG_README.txt"
    readme.write_text(
        "VGGT conditioning debug (same PE frame as training Fourier centres).\n"
        "\n"
        "patch_centers.ply      green  — kept patch centres (weak-context PE)\n"
        "discarded_centers.ply  red    — background-discarded patch centres\n"
        "vggt_points.ply        gray   — VGGT depth FG cloud (all views → cam0)\n"
        "\n"
        "Multi-view: non-ref views are mapped into view-0 via VGGT extrinsics,\n"
        "then the same normalize as PE (c_meanrms mean+RMS of kept centres, or\n"
        "cross Hunyuan bbox from cache align_stats).\n"
        "Overlay with gt.ply to check alignment.\n"
    )
    written["readme"] = str(readme)
    return written


def collect_gt_align_debug_cloud(
    batch: Dict,
    *,
    sample_index: int = 0,
) -> np.ndarray:
    """GT surface xyz already aligned/normalized by the dataset (c_meanrms etc.)."""
    surf = batch["surface"]
    if isinstance(surf, torch.Tensor):
        xyz = surf[sample_index, :, :3].detach().float().cpu().numpy()
    else:
        xyz = np.asarray(surf[sample_index][:, :3], dtype=np.float32)
    return np.ascontiguousarray(xyz.reshape(-1, 3), dtype=np.float32)


def export_joint_align_debug_plys(
    out_dir: Union[str, Path],
    *,
    gt_xyz: np.ndarray,
    vggt_clouds: Dict[str, np.ndarray],
    view_rgbs: Optional[Dict[int, np.ndarray]] = None,
    view_indices: Optional[list] = None,
) -> Dict[str, str]:
    """Export GT + joint VGGT clouds in the train PE frame for alignment QC."""
    from hy3dgen.shapegen.gs_export import export_xyz_pointcloud_ply

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}

    gt = np.asarray(gt_xyz, dtype=np.float32).reshape(-1, 3)
    gt_path = out_dir / "gt_c_meanrms.ply"
    export_xyz_pointcloud_ply(gt, gt_path, rgb=(40, 120, 255))
    written["gt_c_meanrms"] = str(gt_path)

    vggt_specs = (
        ("vggt_merged_c_meanrms", "vggt_points", (160, 160, 160)),
        ("patch_centers_c_meanrms", "patch_centers", (32, 200, 64)),
        ("discarded_centers_c_meanrms", "discarded_centers", (220, 40, 40)),
    )
    for out_name, key, rgb in vggt_specs:
        pts = vggt_clouds.get(key)
        if pts is None:
            continue
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        p = out_dir / f"{out_name}.ply"
        export_xyz_pointcloud_ply(pts, p, rgb=rgb)
        written[out_name] = str(p)

    if view_rgbs:
        try:
            import imageio.v2 as imageio  # type: ignore

            for vid, arr in view_rgbs.items():
                p = out_dir / f"view_rgb_{int(vid):02d}.png"
                imageio.imwrite(p, arr)
                written[f"view_rgb_{int(vid):02d}"] = str(p)
        except Exception as e:
            logger.warning("Failed to write view RGB PNGs: %s", e)

    readme = out_dir / "JOINT_ALIGN_README.txt"
    vtxt = view_indices if view_indices is not None else "?"
    readme.write_text(
        "Joint VGGT + c_meanrms alignment QC (same frames as training).\n"
        "\n"
        f"View order (VGGT ref = first): {vtxt}\n"
        "\n"
        "gt_c_meanrms.ply              blue   — GT mesh in ref camera, c_meanrms norm\n"
        "vggt_merged_c_meanrms.ply     gray   — both views' VGGT FG depth → cam0, norm\n"
        "patch_centers_c_meanrms.ply   green  — kept patch centres (weak-context PE)\n"
        "discarded_centers_c_meanrms.ply red — discarded patch centres\n"
        "\n"
        "Overlay gt_c_meanrms.ply with vggt_merged_c_meanrms.ply in CloudCompare.\n"
    )
    written["readme"] = str(readme)
    return written


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
    """For each point in ``a``, Euclidean dist to nearest in ``b``."""
    if a.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if b.size == 0:
        return np.full((a.shape[0],), np.inf, dtype=np.float32)
    ta = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).unsqueeze(0)
    tb = torch.from_numpy(np.ascontiguousarray(b, dtype=np.float32)).unsqueeze(0)
    d2 = pairwise_dist2(ta, tb)[0]  # [Na, Nb]
    return torch.sqrt(d2.min(dim=1).values.clamp_min(0)).cpu().numpy().astype(np.float32)


def nn_indices(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For each point in ``a``, index of nearest neighbour in ``b``."""
    if a.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if b.size == 0:
        return np.zeros((a.shape[0],), dtype=np.int64)
    ta = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).unsqueeze(0)
    tb = torch.from_numpy(np.ascontiguousarray(b, dtype=np.float32)).unsqueeze(0)
    d2 = pairwise_dist2(ta, tb)[0]
    return d2.min(dim=1).indices.cpu().numpy().astype(np.int64)


def rgb_l1_errors(
    rgb_src: np.ndarray,
    rgb_ref: np.ndarray,
    idx_src_to_ref: np.ndarray,
) -> np.ndarray:
    """Per-point mean |rgb_src - rgb_ref[nn]| over channels → [N]."""
    if rgb_src.size == 0:
        return np.zeros((0,), dtype=np.float32)
    matched = rgb_ref[idx_src_to_ref]
    err = np.abs(rgb_src.astype(np.float64) - matched.astype(np.float64)).mean(axis=-1)
    return err.astype(np.float32)


def export_recon_debug_plys(
    out_dir: Union[str, Path],
    *,
    gt_xyz: Union[torch.Tensor, np.ndarray],
    recon_xyz: Union[torch.Tensor, np.ndarray],
    fps_xyz: Optional[Union[torch.Tensor, np.ndarray]] = None,
    centers: Optional[Union[torch.Tensor, np.ndarray]] = None,
    num_points_per_anchor: Optional[int] = None,
    max_anchor_delta: Optional[float] = None,
    gt_rgb: Optional[Union[torch.Tensor, np.ndarray]] = None,
    recon_rgb: Optional[Union[torch.Tensor, np.ndarray]] = None,
    patch_centers: Optional[Union[torch.Tensor, np.ndarray]] = None,
    patch_keep: Optional[Union[torch.Tensor, np.ndarray]] = None,
    intruder_thresh: float = 0.02,
    error_vmax: Optional[float] = None,
    rgb_error_vmax: Optional[float] = None,
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

    # --- Geometric error maps ---
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

    # --- RGB residual maps (high-freq colour failures invisible on recon_error) ---
    rgb_lines = ""
    if gt_rgb is not None and recon_rgb is not None and gt.shape[0] > 0 and recon.shape[0] > 0:
        g_rgb = _squeeze_batch(gt_rgb)[:, :3]
        r_rgb = _squeeze_batch(recon_rgb)[:, :3]
        if g_rgb.shape[0] == gt.shape[0] and r_rgb.shape[0] == recon.shape[0]:
            idx_r2g = nn_indices(recon, gt)
            idx_g2r = nn_indices(gt, recon)
            err_r = rgb_l1_errors(r_rgb, g_rgb, idx_r2g)
            err_g = rgb_l1_errors(g_rgb, r_rgb, idx_g2r)
            if rgb_error_vmax is not None:
                rvmax = float(rgb_error_vmax)
            else:
                cat = np.concatenate([err_r, err_g]) if err_r.size + err_g.size else np.array([1.0])
                rvmax = float(np.percentile(cat, 95)) if cat.size else 1.0
            rvmax = max(rvmax, 1e-6)

            p = out_dir / "rgb_error.ply"
            export_xyz_pointcloud_ply(
                recon, p, colors=scalar_to_rgb(err_r, vmin=0.0, vmax=rvmax)
            )
            written["rgb_error"] = str(p)

            p = out_dir / "gt_rgb_error.ply"
            export_xyz_pointcloud_ply(
                gt, p, colors=scalar_to_rgb(err_g, vmin=0.0, vmax=rvmax)
            )
            written["gt_rgb_error"] = str(p)
            rgb_lines = (
                f"rgb_error.ply       heatmap  — recon |rgb−GT_nn| L1 mean "
                f"(vmax≈{rvmax:.4f}; grid frets show up here)\n"
                "gt_rgb_error.ply    heatmap  — GT |rgb−recon_nn| L1 mean\n"
            )

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
        if max_anchor_delta is not None and float(max_anchor_delta) > 0:
            denom = float(max_anchor_delta)
            delta_note = f"|xyz-center| / max_anchor_delta={denom:g} (1=old hard cap)"
        else:
            denom = float(np.percentile(delta_norm, 95) if delta_norm.size else 1.0)
            denom = max(denom, 1e-8)
            delta_note = f"|xyz-center| / p95≈{denom:.4f} (unbound locals)"
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
            f"(vmax≈{vmax:.4f}; GEOMETRY only — coplanar wrong colour stays blue)\n"
            "gt_uncovered.ply    heatmap  — GT coloured by dist→nearest recon\n"
            f"{rgb_lines}"
            f"intruders.ply       heatmap  — recon with dist→GT > {thr:g}\n"
            "locals_by_anchor.ply rainbow — recon coloured by parent anchor id\n"
            f"delta_mag.ply       heatmap  — {delta_note}\n"
            "patch_centers.ply   green    — kept VGGT weak-context centres\n"
            "\n"
            "Reading tips:\n"
            "- FPS sparse on thin parts → need denser / sharper surface sampling.\n"
            "- Anchors in voids (vs FPS on surface) → anc / decoder pull off-surface.\n"
            "- Red clumps on recon_error / intruders between legs → local smear.\n"
            "- Red on gt_uncovered at edges → missing thin detail / undersampling.\n"
            "- Missing lattice frets show in rgb_error / gt_rgb_error, not recon_error.\n"
            "- delta_mag red → locals far from centres (soft λ_delta should limit this).\n"
        )
        written["readme"] = str(readme)
    else:
        readme = out_dir / "DEBUG_PLY_README.txt"
        readme.write_text(
            "ShapePCUnite recon debug PLYs.\n"
            "fps=cyan anchors=magenta error heatmaps blue→red.\n"
            f"intruders: dist→GT > {thr:g}.\n"
            f"{rgb_lines}"
            "locals_by_anchor / delta_mag skipped "
            "(need centers + matching num_points_per_anchor).\n"
            "Note: recon_error is geometric only; use rgb_error for colour frets.\n"
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
