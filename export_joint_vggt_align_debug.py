#!/usr/bin/env python3
"""Export c_meanrms alignment QC for joint 2-view VGGT cache.

Writes per-object PLYs in the **same normalized frames as training**:
  - gt_c_meanrms.ply           — GT surface (ref camera + c_meanrms GT stats)
  - vggt_merged_c_meanrms.ply  — joint VGGT dense FG (both views → cam0, PE norm)
  - patch_centers_c_meanrms.ply / discarded_centers_c_meanrms.ply

Uses joint cache when present; otherwise runs online joint VGGT on rgb_views.

Example (views 16=ref, 29=second — order matters):

    python export_joint_vggt_align_debug.py \\
      --data_dir /export/home/nathan/datasets/gobjaverse_experiments/furniture_351/train \\
      --gobjaverse_render_root /export/home/nathan/datasets \\
      --view_indices "16,29" \\
      --vggt_cache_root runs/vggt_cache/furn4_joint_v16_v29 \\
      --vggt_joint_cache \\
      --max_items 20 \\
      --output_dir runs/debug_joint_v16_v29
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from hy3dgen.shapegen.gobjaverse_gt import parse_view_indices
from hy3dgen.shapegen.pc_debug_export import (
    collect_gt_align_debug_cloud,
    collect_vggt_debug_clouds,
    export_joint_align_debug_plys,
)
from hy3dgen.shapegen.pc_render_dataset import (
    build_surface_render_dataset,
    collate_surface_render,
)
from hy3dgen.shapegen.vggt_context import VGGTContextBuilder
from train_gs_ae import load_experiment_manifest, resolve_category_ids

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _rgb_to_u8(rgb: torch.Tensor) -> np.ndarray:
    return (
        (rgb.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0)
        .round()
        .astype(np.uint8)
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--gobjaverse_render_root", default=None)
    p.add_argument("--view_indices", required=True, help='Ordered views, e.g. "16,29"')
    p.add_argument("--vggt_cache_root", default=None)
    p.add_argument(
        "--vggt_joint_cache",
        action="store_true",
        help="Require joint cache files (faster; no online VGGT).",
    )
    p.add_argument("--max_items", type=int, default=10)
    p.add_argument("--output_dir", default="runs/debug_joint_vggt_align")
    p.add_argument("--align_mode", default="c_meanrms", choices=("c_meanrms",))
    p.add_argument("--device", default="cuda")
    p.add_argument("--no_experiment_manifest", action="store_true")
    p.add_argument("--categories", default=None)
    args = p.parse_args()

    views = parse_view_indices(view_indices=args.view_indices)
    if len(views) < 2:
        raise SystemExit("Need >=2 views in --view_indices (order = VGGT reference first).")
    if args.vggt_joint_cache and not args.vggt_cache_root:
        raise SystemExit("--vggt_joint_cache requires --vggt_cache_root")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_path = Path(args.data_dir).resolve()
    manifest = (
        load_experiment_manifest(str(data_path))
        if not args.no_experiment_manifest
        else None
    )
    dataset = build_surface_render_dataset(
        str(data_path),
        max_items=args.max_items,
        categories=resolve_category_ids(args.categories),
        use_experiment_manifest=not args.no_experiment_manifest,
        manifest=manifest,
        render_root=args.gobjaverse_render_root,
        view_indices=views,
        views_per_sample=len(views),
        view_sample_mode="first",
        vggt_cache_root=args.vggt_cache_root,
        use_joint_vggt_cache=args.vggt_joint_cache,
        align_mode=args.align_mode,
    )
    builder = VGGTContextBuilder().to(device)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Exporting joint align debug for %d objects, views=%s, joint_cache=%s",
        len(dataset),
        views,
        args.vggt_joint_cache,
    )

    for i in range(len(dataset)):
        batch = collate_surface_render([dataset[i]])
        stem = Path(batch["mesh_path"][0]).stem
        obj_dir = out_root / f"{i:04d}_{stem[:16]}"
        obj_dir.mkdir(parents=True, exist_ok=True)

        gt_xyz = collect_gt_align_debug_cloud(batch, sample_index=0)
        vggt_clouds = collect_vggt_debug_clouds(
            builder,
            batch,
            align_mode=args.align_mode,
            device=device,
            sample_index=0,
        )
        if not vggt_clouds:
            logger.warning("[%d] %s: no VGGT clouds (missing cache / rgb_views)", i, stem)
            continue

        view_rgbs = {}
        if "rgb_views" in batch:
            rv = batch["rgb_views"][0]
            vids = (
                batch["view_indices"][0].tolist()
                if "view_indices" in batch
                else views
            )
            for si, vid in enumerate(vids):
                view_rgbs[int(vid)] = _rgb_to_u8(rv[si])

        written = export_joint_align_debug_plys(
            obj_dir,
            gt_xyz=gt_xyz,
            vggt_clouds=vggt_clouds,
            view_rgbs=view_rgbs or None,
            view_indices=views,
        )
        logger.info("[%d] %s → %s", i, stem, obj_dir)
        for k, v in written.items():
            if k != "readme":
                logger.info("  %s", v)

    logger.info("Done. Output: %s", out_root)


if __name__ == "__main__":
    main()
