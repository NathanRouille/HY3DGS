#!/usr/bin/env python3
"""Lean alignment showcase export: raw / naive / C_meanrms (GT + patch centres only).

For each (object, view) writes:

  <out>/<iiii>_<stem>_v<view>/
    view_rgb.png
    raw/{gt.ply, patch_centers.ply, metrics.txt}
    indep_meanrms/{gt.ply, patch_centers.ply, metrics.txt}
    C_gt_depth_filter_zrobust_meanrms/{gt.ply, patch_centers.ply, metrics.txt}

Plus a root ``summary.csv`` and ``ranking.csv`` sorted by showcase gap
(indep worse − C better) so you can pick slides quickly.

Recipes (camera frame of the chosen view):
  raw          — GT mesh⊕c2w and VGGT patch centres, no normalize
  indep_meanrms — GT = full mesh own mean+RMS; centres = own mean+RMS
  C_…_meanrms  — GT = mesh with μ,s from eroded GT-depth FG (train c_meanrms);
                 centres = own mean+RMS (same PE as training)

Default: 100 train objects × 10 spread views
  ``0,5,10,15,20,25,26,27,33,37``.

Example:

  python export_align_showcase_debug.py \\
    --data_dir /export/home/nathan/datasets/gobjaverse_experiments/furniture_351/train \\
    --gobjaverse_render_root /export/home/nathan/datasets \\
    --max_items 100 \\
    --view_indices 0,5,10,15,20,25,26,27,33,37 \\
    --output_dir runs/debug_align_showcase \\
    --device cuda
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from export_camera_frame_debug import alignment_metrics, shared_canonicalize_from_ref
from hy3dgen.shapegen.cam_align import (
    apply_mu_s,
    compute_c_meanrms_gt_stats,
    mean_rms_mu_s,
)
from hy3dgen.shapegen.gobjaverse_gt import (
    gobjaverse_eval_view_indices,
    parse_view_indices,
)
from hy3dgen.shapegen.gs_export import export_xyz_pointcloud_ply
from hy3dgen.shapegen.pc_render_dataset import build_surface_render_dataset
from hy3dgen.shapegen.vggt_context import (
    VGGTContextBuilder,
    depth_map_to_cam_points,
    patch_centers_from_depth,
    world_to_camera_torch,
)
from train_gs_ae import load_experiment_manifest, resolve_category_ids

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

METHODS = (
    "raw",
    "indep_meanrms",
    "C_gt_depth_filter_zrobust_meanrms",
)

METRIC_KEYS = (
    "nn_pe2gt_mean",
    "nn_pe2gt_med",
    "nn_gt2pe_mean",
    "chamfer_mean",
    "overlap_pe_frac",
    "overlap_gt_frac",
    "hausdorff_p95",
    "bbox_side_ratio",
    "bbox_center_dist",
    "xy_span_ratio",
    "mean_z_diff",
    "n_pe",
    "n_gt",
)


def _to_np_xyz(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32).reshape(-1, 3)


def _rgb_u8(rgb: torch.Tensor) -> np.ndarray:
    return (
        (rgb.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0)
        .round()
        .astype(np.uint8)
    )


def _save_png(path: Path, arr: np.ndarray) -> None:
    try:
        import imageio.v2 as imageio  # type: ignore

        imageio.imwrite(path, arr)
    except Exception:
        from PIL import Image

        Image.fromarray(arr).save(path)


def _surface_to_camera(surface: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    xyz = surface[:, :3]
    xyz_cam = world_to_camera_torch(xyz, c2w)
    out = surface.clone()
    out[:, :3] = xyz_cam
    if out.shape[-1] >= 6:
        nrm = out[:, 3:6]
        R = c2w[:3, :3]
        out[:, 3:6] = nrm @ R
    return out


def _export_pair(
    out_dir: Path,
    *,
    gt: np.ndarray,
    centers: np.ndarray,
    gt_rgb: Optional[torch.Tensor],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    export_xyz_pointcloud_ply(gt, out_dir / "gt.ply", colors=gt_rgb)
    export_xyz_pointcloud_ply(
        centers.astype(np.float32), out_dir / "patch_centers.ply", rgb=(32, 200, 64)
    )


def _write_metrics(path: Path, metrics: Dict[str, float], header: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(header.rstrip() + "\n")
        for k in METRIC_KEYS:
            if k in metrics:
                f.write(f"{k}={metrics[k]}\n")


def process_one(
    *,
    stem: str,
    view_idx: int,
    surface: torch.Tensor,
    view: Dict,
    builder: VGGTContextBuilder,
    device: torch.device,
    out_dir: Path,
    rng: np.random.Generator,
    max_gt_points: int,
) -> List[Dict]:
    """Run VGGT + three alignments for one (mesh, view). Returns metric rows."""
    c2w = view["c2w"]
    surf_cam = _surface_to_camera(surface, c2w)
    gt_cam = _to_np_xyz(surf_cam[:, :3])
    gt_rgb = surf_cam[:, 6:9] if surf_cam.shape[-1] >= 9 else None
    if gt_cam.shape[0] > max_gt_points > 0:
        idx = np.linspace(0, gt_cam.shape[0] - 1, num=max_gt_points, dtype=np.int64)
        gt_cam = gt_cam[idx]
        if gt_rgb is not None:
            gt_rgb = gt_rgb[idx]

    out_dir.mkdir(parents=True, exist_ok=True)
    _save_png(out_dir / "view_rgb.png", _rgb_u8(view["rgb"]))

    rgb = view["rgb"].unsqueeze(0).to(device)
    with torch.no_grad():
        raw = builder.extract_vggt_raw(rgb, return_dense=True)

    depth = raw["vggt_depth"][0].detach().float().cpu().numpy()
    conf = raw["vggt_depth_conf"][0].detach().float().cpu().numpy()
    K = raw["vggt_intrinsics"][0].detach().float().cpu().numpy()
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    cam_pts = depth_map_to_cam_points(depth, fx=fx, fy=fy, cx=cx, cy=cy)

    vggt_rgb = (
        torch.nn.functional.interpolate(
            view["rgb"].float().unsqueeze(0),
            size=depth.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0]
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    pix_valid = builder._pixel_valid_mask(depth, conf, rgb_np=vggt_rgb)
    full_c, full_keep = patch_centers_from_depth(
        cam_pts,
        pix_valid,
        patch_size=builder.patch_size,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )
    centers_raw = full_c[full_keep.astype(bool)].astype(np.float32)
    if centers_raw.shape[0] == 0:
        logger.warning("%s view %d: no kept patch centres — skipping", stem, view_idx)
        return []

    # --- raw ---
    pairs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
        "raw": (gt_cam, centers_raw),
    }

    # --- indep_meanrms: each cloud own mean+RMS ---
    mu_gt_i, s_gt_i = mean_rms_mu_s(gt_cam)
    mu_pe_i, s_pe_i = mean_rms_mu_s(centers_raw)
    pairs["indep_meanrms"] = (
        apply_mu_s(gt_cam, mu_gt_i, s_gt_i).astype(np.float32),
        apply_mu_s(centers_raw, mu_pe_i, s_pe_i).astype(np.float32),
    )

    # --- C_gt_depth_filter_zrobust_meanrms (= train c_meanrms) ---
    depth_np = view["depth"].numpy() if torch.is_tensor(view["depth"]) else np.asarray(view["depth"])
    rgb_np = view["rgb"].permute(1, 2, 0).numpy()
    st = compute_c_meanrms_gt_stats(
        [depth_np],
        [rgb_np],
        [view["intrinsics"]],
        [c2w.numpy() if torch.is_tensor(c2w) else np.asarray(c2w)],
        c2w.numpy() if torch.is_tensor(c2w) else np.asarray(c2w),
        erode_iters=1,
    )
    mu_c = np.array([st["mu_x"], st["mu_y"], st["mu_z"]], dtype=np.float64)
    s_c = float(st["s"])
    # PE: own mean+RMS of centres (same as training align_patch_centers c_meanrms)
    (centers_c,), _, _ = shared_canonicalize_from_ref(
        centers_raw, centers_raw, scale="rms"
    )
    pairs["C_gt_depth_filter_zrobust_meanrms"] = (
        apply_mu_s(gt_cam, mu_c, s_c).astype(np.float32),
        centers_c.astype(np.float32),
    )

    rows: List[Dict] = []
    for method, (gt_xyz, cen_xyz) in pairs.items():
        mdir = out_dir / method
        _export_pair(mdir, gt=gt_xyz, centers=cen_xyz, gt_rgb=gt_rgb)
        metrics = alignment_metrics(cen_xyz, gt_xyz, rng=rng)
        _write_metrics(
            mdir / "metrics.txt",
            metrics,
            header=(
                f"mesh={stem}\nview_idx={view_idx}\nmethod={method}\n"
                "scored: patch_centers vs gt\n"
            ),
        )
        row = {
            "mesh": stem,
            "view_idx": int(view_idx),
            "method": method,
            "sample_dir": str(out_dir),
            **{k: metrics[k] for k in METRIC_KEYS},
        }
        rows.append(row)
    return rows


def _write_summary(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    fields = ["mesh", "view_idx", "method", "sample_dir", *METRIC_KEYS]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _write_ranking(path: Path, rows: List[Dict]) -> None:
    """One row per (mesh, view) comparing C vs indep for showcase picking."""
    by: Dict[Tuple[str, int], Dict[str, Dict]] = {}
    for r in rows:
        key = (r["mesh"], int(r["view_idx"]))
        by.setdefault(key, {})[r["method"]] = r

    ranked: List[Dict] = []
    for (mesh, view_idx), methods in by.items():
        c = methods.get("C_gt_depth_filter_zrobust_meanrms")
        n = methods.get("indep_meanrms")
        raw = methods.get("raw")
        if c is None or n is None:
            continue
        gap_nn = float(n["nn_pe2gt_mean"]) - float(c["nn_pe2gt_mean"])
        gap_overlap = float(c["overlap_pe_frac"]) - float(n["overlap_pe_frac"])
        ranked.append(
            {
                "mesh": mesh,
                "view_idx": view_idx,
                "sample_dir": c["sample_dir"],
                "C_nn_pe2gt_mean": c["nn_pe2gt_mean"],
                "indep_nn_pe2gt_mean": n["nn_pe2gt_mean"],
                "gap_nn_indep_minus_C": gap_nn,
                "C_overlap_pe_frac": c["overlap_pe_frac"],
                "indep_overlap_pe_frac": n["overlap_pe_frac"],
                "gap_overlap_C_minus_indep": gap_overlap,
                "C_chamfer_mean": c["chamfer_mean"],
                "indep_chamfer_mean": n["chamfer_mean"],
                "raw_nn_pe2gt_mean": raw["nn_pe2gt_mean"] if raw else float("nan"),
                "showcase_score": gap_nn + 0.5 * gap_overlap,  # prefer big gap + C overlap
            }
        )
    ranked.sort(key=lambda r: float(r["showcase_score"]), reverse=True)

    if not ranked:
        return
    fields = list(ranked[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(ranked)

    # Human-readable top-20
    txt = path.with_suffix(".txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write(
            "Top showcase samples (high = C good & indep bad).\n"
            "Open sample_dir/{C_gt_depth_filter_zrobust_meanrms,indep_meanrms}/\n\n"
        )
        for i, r in enumerate(ranked[:20]):
            f.write(
                f"{i+1:2d}. {r['mesh'][:16]} view={r['view_idx']:02d}  "
                f"score={r['showcase_score']:.4f}  "
                f"C_nn={r['C_nn_pe2gt_mean']:.4f} indep_nn={r['indep_nn_pe2gt_mean']:.4f}  "
                f"gap_nn={r['gap_nn_indep_minus_C']:.4f}  "
                f"C_ov={r['C_overlap_pe_frac']:.3f} indep_ov={r['indep_overlap_pe_frac']:.3f}\n"
                f"    {r['sample_dir']}\n"
            )
    logger.info("Wrote ranking → %s (+ %s)", path, txt.name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--gobjaverse_render_root", default=None)
    p.add_argument("--output_dir", default="runs/debug_align_showcase")
    p.add_argument("--max_items", type=int, default=100)
    p.add_argument(
        "--view_indices",
        type=str,
        default=None,
        help='Views to sweep, e.g. "0,5,10,15,20,25,26,27,33,37". '
        "Default: gobjaverse_eval_view_indices(40) (10 spread views).",
    )
    p.add_argument("--categories", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--conf_percentile", type=float, default=20.0)
    p.add_argument("--max_gt_points", type=int, default=20000)
    p.add_argument("--pc_size", type=int, default=5120)
    p.add_argument("--pc_sharpedge_size", type=int, default=5120)
    p.add_argument("--no_experiment_manifest", action="store_true")
    args = p.parse_args()

    if args.view_indices:
        views = parse_view_indices(view_indices=args.view_indices)
    else:
        views = gobjaverse_eval_view_indices(40)
    if not views:
        raise SystemExit("Empty view list")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    data_path = Path(args.data_dir).resolve()
    manifest = (
        None if args.no_experiment_manifest else load_experiment_manifest(str(data_path))
    )
    # One dataset sample per mesh at a placeholder view; we load each view manually.
    dataset = build_surface_render_dataset(
        str(data_path),
        max_items=args.max_items,
        categories=resolve_category_ids(args.categories),
        use_experiment_manifest=not args.no_experiment_manifest,
        manifest=manifest,
        render_root=args.gobjaverse_render_root,
        view_idx=int(views[0]),
        pc_size=args.pc_size,
        pc_sharpedge_size=args.pc_sharpedge_size,
        surface_in_camera_frame=False,  # we transform per-view ourselves
        gobjaverse_normalization=True,
        filter_missing_views=True,
    )
    if dataset.render_loader is None:
        raise SystemExit("Need G-Objaverse render source (--gobjaverse_render_root)")

    builder = VGGTContextBuilder(
        width=1024, conf_percentile=args.conf_percentile
    ).to(device)
    builder.eval()

    logger.info(
        "Align showcase: %d meshes × %d views → %s",
        len(dataset.mesh_paths),
        len(views),
        out_root,
    )
    logger.info("Views: %s", views)

    all_rows: List[Dict] = []
    obj_i = 0
    for mesh_path in dataset.mesh_paths:
        stem = Path(mesh_path).stem
        try:
            surface = dataset.surface_loader(
                mesh_path,
                gobjaverse_meta=dataset._surface_meta(mesh_path),
            ).squeeze(0)
        except Exception as e:
            logger.warning("Skip mesh %s: %s", stem, e)
            continue

        for view_idx in views:
            try:
                view = dataset.render_loader.load_view(mesh_path, view_idx=int(view_idx))
            except Exception as e:
                logger.warning("Skip %s view %d: %s", stem, view_idx, e)
                continue

            sample_dir = out_root / f"{obj_i:04d}_{stem[:16]}_v{int(view_idx):02d}"
            try:
                rows = process_one(
                    stem=stem,
                    view_idx=int(view_idx),
                    surface=surface,
                    view=view,
                    builder=builder,
                    device=device,
                    out_dir=sample_dir,
                    rng=rng,
                    max_gt_points=args.max_gt_points,
                )
            except Exception as e:
                logger.exception("Failed %s view %d: %s", stem, view_idx, e)
                continue
            all_rows.extend(rows)
            if rows:
                c_nn = next(
                    (
                        r["nn_pe2gt_mean"]
                        for r in rows
                        if r["method"] == "C_gt_depth_filter_zrobust_meanrms"
                    ),
                    float("nan"),
                )
                n_nn = next(
                    (r["nn_pe2gt_mean"] for r in rows if r["method"] == "indep_meanrms"),
                    float("nan"),
                )
                logger.info(
                    "[%d] %s v%02d  C_nn=%.4f  indep_nn=%.4f  → %s",
                    obj_i,
                    stem[:16],
                    view_idx,
                    c_nn,
                    n_nn,
                    sample_dir.name,
                )
        obj_i += 1

    _write_summary(out_root / "summary.csv", all_rows)
    _write_ranking(out_root / "ranking.csv", all_rows)
    logger.info(
        "Done. %d metric rows → %s/summary.csv + ranking.csv",
        len(all_rows),
        out_root,
    )


if __name__ == "__main__":
    main()
