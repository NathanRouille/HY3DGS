#!/usr/bin/env python3
"""Nearest-surface color audit: compare input point-cloud RGB to mesh surface color.

For every point in the training surface tensor, finds the closest point on the
normalized mesh (no camera, no lighting, no viewer) and compares the stored
point RGB to the per-part UV / vertex color at that surface location.

Also exports normalized mesh (OBJ) and training point cloud (PLY) per object.

Usage:
    python assess_pointcloud_colors.py \\
        --checkpoint runs/train/<run>/ckpt_020000.pt \\
        --data_dir /path/to/furniture_351/train \\
        --max_items 10 \\
        --view_idx 0 \\
        --out_dir pointcloud_color_audit_v2
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from assess_pipeline import build_args_from_ckpt, splat_pointcloud
from evaluate_gs_ae import export_normalized_mesh_obj
from hy3dgen.shapegen.gobjaverse_gt import (
    intrinsics_from_meta,
    read_gobjaverse_view_meta,
    unity_c2w_from_meta,
)
from hy3dgen.shapegen.gs_export import export_input_surface_ply, surface_rgb_slice
from hy3dgen.shapegen.pretrained_profiles import resolve_include_sharp_label
from hy3dgen.shapegen.surface_loaders import (
    _colors_at_surface_points_parts,
    extract_mesh_texture,
    normalize_parts_gobjaverse,
    parts_to_trimesh,
    scene_to_parts,
)
from train_gs_ae import GTRGBDRenderer, MeshDataset, load_experiment_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def l1_rgb(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-point mean L1 over RGB channels, shape (N,)."""
    return np.abs(a.astype(np.float64) - b.astype(np.float64)).mean(axis=1)


def error_to_rgb_u8(err: np.ndarray, vmax: float = 0.25) -> np.ndarray:
    """Map scalar error to jet colormap RGB uint8."""
    t = np.clip(err / max(vmax, 1e-8), 0.0, 1.0)
    cmap = plt.get_cmap("jet")
    return (cmap(t)[..., :3] * 255.0).round().astype(np.uint8)


def _load_normalized_parts(mesh_path: str, meta: dict) -> List[trimesh.Trimesh]:
    raw_mesh = trimesh.load(mesh_path, process=False)
    return normalize_parts_gobjaverse(scene_to_parts(raw_mesh), meta)


def assess_pointcloud_colors_one(
    dataset: MeshDataset,
    idx: int,
    include_sharp_label: bool,
    view_idx: int,
    out_dir: Path,
    export_dir: Optional[Path],
    H: int,
    W: int,
) -> Dict:
    sample = dataset[idx]
    mesh_path = sample["mesh_path"]
    mesh_stem = Path(mesh_path).stem

    render_dir = dataset.gobjaverse_gt.render_dir_for_mesh(mesh_path)
    view_json = render_dir / f"{view_idx:05d}" / f"{view_idx:05d}.json"
    meta = read_gobjaverse_view_meta(view_json)
    c2w_unity = unity_c2w_from_meta(meta)
    fx, fy, cx, cy = intrinsics_from_meta(meta, H, W)

    parts = _load_normalized_parts(mesh_path, meta)
    unified, face_part_ids = parts_to_trimesh(parts)
    has_uv_texture = any(extract_mesh_texture(p)[0] is not None for p in parts)

    surf_np = sample["surface"].numpy()
    rgb_slice = surface_rgb_slice(surf_np.shape[-1], include_sharp_label=include_sharp_label)
    pc_xyz = surf_np[:, :3].astype(np.float64)
    pc_rgb = surf_np[:, rgb_slice].astype(np.float32)

    closest_pts, distances, tri_ids = trimesh.proximity.closest_point(unified, pc_xyz)
    mesh_rgb = _colors_at_surface_points_parts(
        parts, closest_pts, tri_ids, face_part_ids,
    ).astype(np.float32)

    err = l1_rgb(pc_rgb, mesh_rgb)
    metrics = {
        "mesh_stem": mesh_stem,
        "n_points": int(len(pc_xyz)),
        "n_parts": len(parts),
        "has_uv_texture": has_uv_texture,
        "mean_l1": float(err.mean()),
        "median_l1": float(np.median(err)),
        "p95_l1": float(np.percentile(err, 95)),
        "max_l1": float(err.max()),
        "mean_dist_to_surface": float(distances.mean()),
        "max_dist_to_surface": float(distances.max()),
    }

    if export_dir is not None:
        mesh_out = export_dir / "meshes" / f"{mesh_stem}.obj"
        ply_out = export_dir / "pointclouds" / f"{mesh_stem}.ply"
        export_normalized_mesh_obj(
            mesh_path, mesh_out, gobjaverse_meta=meta,
        )
        export_input_surface_ply(
            surf_np,
            ply_out,
            include_sharp_label=include_sharp_label,
        )
        logger.info("Exported mesh %s and point cloud %s", mesh_out, ply_out)

    pc_stored_u8 = splat_pointcloud(pc_xyz, pc_rgb, c2w_unity, fx, fy, cx, cy, H, W)
    pc_meshref_u8 = splat_pointcloud(pc_xyz, mesh_rgb, c2w_unity, fx, fy, cx, cy, H, W)
    err_rgb = error_to_rgb_u8(err)
    pc_err_u8 = splat_pointcloud(
        pc_xyz, err_rgb.astype(np.float32) / 255.0, c2w_unity, fx, fy, cx, cy, H, W,
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"{mesh_stem} | nearest-surface color diff (view {view_idx:02d})",
        fontsize=11,
    )

    axes[0, 0].hist(err, bins=50, color="steelblue", edgecolor="white")
    axes[0, 0].set_title("Per-point L1 RGB error")
    axes[0, 0].set_xlabel("L1 error")
    axes[0, 0].axvline(
        metrics["mean_l1"], color="red", ls="--",
        label=f"mean={metrics['mean_l1']:.4f}",
    )
    axes[0, 0].legend(fontsize=8)

    for ax, img, title in [
        (axes[0, 1], pc_stored_u8, "Point cloud (stored RGB)"),
        (axes[1, 0], pc_meshref_u8, "Same points (per-part UV at closest surface)"),
        (axes[1, 1], pc_err_u8, "Error heatmap (jet, vmax=0.25)"),
    ]:
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    stats = (
        f"mean L1={metrics['mean_l1']:.4f}  median={metrics['median_l1']:.4f}  "
        f"p95={metrics['p95_l1']:.4f}  max={metrics['max_l1']:.4f}\n"
        f"dist to surface: mean={metrics['mean_dist_to_surface']:.2e}  "
        f"max={metrics['max_dist_to_surface']:.2e}  parts={metrics['n_parts']}  "
        f"uv={metrics['has_uv_texture']}"
    )
    fig.text(0.5, 0.01, stats, ha="center", fontsize=8)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = out_dir / f"{mesh_stem}_color_diff.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info(
        "Saved %s  mean_l1=%.4f  p95=%.4f  max=%.4f",
        out_path, metrics["mean_l1"], metrics["p95_l1"], metrics["max_l1"],
    )
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="Optional checkpoint; if set, dataset args match training run.")
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--mesh_stems", type=str, nargs="*", default=None)
    ap.add_argument("--max_items", type=int, default=None)
    ap.add_argument("--view_idx", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="pointcloud_color_audit")
    ap.add_argument("--no_export", action="store_true",
                    help="Skip mesh OBJ / point cloud PLY export.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    export_dir = None if args.no_export else out_dir

    train_args: Dict = {}
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        train_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
        if hasattr(train_args, "__dict__"):
            train_args = vars(train_args)
    ns = build_args_from_ckpt(train_args)
    include_sharp_label = resolve_include_sharp_label(ns)

    manifest = load_experiment_manifest(args.data_dir)
    gt_source = (train_args.get("gt_source") or (manifest or {}).get("gt_source") or "gobjaverse")

    dataset = MeshDataset(
        data_dir=args.data_dir,
        pc_size=ns.pc_size,
        pc_sharpedge_size=ns.pc_sharpedge_size,
        render_height=ns.render_height,
        render_width=ns.render_width,
        num_views=ns.gobjaverse_num_views or train_args.get("num_views") or 40,
        camera_distance=ns.camera_distance,
        elevation_deg=ns.elevation_deg,
        max_items=args.max_items,
        mesh_blacklist=ns.mesh_blacklist,
        categories=None,
        precache_full_views=False,
        require_cached_gt=True,
        seed=ns.seed,
        include_sharp_label=include_sharp_label,
        gt_source=gt_source,
        gobjaverse_render_root=ns.gobjaverse_render_root,
        gobjaverse_num_views=ns.gobjaverse_num_views,
    )
    logger.info("Dataset has %d meshes", len(dataset))

    if args.mesh_stems:
        indices = []
        for stem in args.mesh_stems:
            matches = [i for i, p in enumerate(dataset.mesh_paths) if stem in p]
            if not matches:
                logger.warning("No mesh matches stem %r", stem)
            indices.extend(matches)
    else:
        indices = list(range(len(dataset)))

    all_metrics: List[Dict] = []
    for idx in indices:
        try:
            m = assess_pointcloud_colors_one(
                dataset, idx, include_sharp_label, args.view_idx, out_dir,
                export_dir, ns.render_height, ns.render_width,
            )
            all_metrics.append(m)
        except Exception:
            logger.exception("Failed on index %d (%s)", idx, dataset.mesh_paths[idx])

    if not all_metrics:
        logger.warning("No objects assessed.")
        return

    csv_path = out_dir / "summary.csv"
    fieldnames = [
        "mesh_stem", "n_points", "n_parts", "has_uv_texture",
        "mean_l1", "median_l1", "p95_l1", "max_l1",
        "mean_dist_to_surface", "max_dist_to_surface",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for m in sorted(all_metrics, key=lambda x: x["mean_l1"], reverse=True):
            w.writerow({k: m[k] for k in fieldnames})
    with open(out_dir / "summary.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info("\n%s", "=" * 78)
    logger.info("SUMMARY (%d objects) — sorted by mean L1 error (worst first)", len(all_metrics))
    logger.info("%s", "=" * 78)
    for m in sorted(all_metrics, key=lambda x: x["mean_l1"], reverse=True):
        logger.info(
            "%-40s  mean_l1=%.4f  p95=%.4f  max=%.4f  parts=%d  uv=%s",
            m["mesh_stem"], m["mean_l1"], m["p95_l1"], m["max_l1"],
            m["n_parts"], m["has_uv_texture"],
        )
    logger.info("Wrote %s and per-object panels to %s", csv_path, out_dir)
    if export_dir is not None:
        logger.info("Exports: %s/meshes/*.obj  %s/pointclouds/*.ply", export_dir, export_dir)


if __name__ == "__main__":
    main()
