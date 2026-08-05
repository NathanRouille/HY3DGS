#!/usr/bin/env python3
"""Precompute frozen-VGGT weak-context features for a mesh set.

The VGGT forward dominates ShapePCUnite step time (~1.4 s/step at batch 1).
Features are frozen, so they only need computing once per (mesh, view):

    # Single view (legacy):
    python cache_vggt_features.py \
      --data_dir .../furniture_351/train \
      --gobjaverse_render_root /export/home/nathan/datasets \
      --cache_root runs/vggt_cache/furn4_view0_cam_cross \
      --view_idx 0 --overwrite

    # All 40 G-Objaverse views (Design A multi-view train):
    python cache_vggt_features.py \
      --data_dir .../furniture_351/train \
      --gobjaverse_render_root /export/home/nathan/datasets \
      --cache_root runs/vggt_cache/furn4_mv40_cam_cross \
      --num_views 40 --overwrite

Payload (camera-frame, VGGT-depth centres; background patches already dropped):
  patch_tokens, camera_token, patch_centers, patch_keep,
  pe_frame='camera', geometry_source='vggt_depth'

Stored pre-projection (``feat_proj`` still trains) in fp16.
**Do not reuse** older caches from the GT-depth / object-frame era.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from hy3dgen.shapegen.gobjaverse_gt import parse_view_indices
from hy3dgen.shapegen.pc_render_dataset import build_surface_render_dataset
from hy3dgen.shapegen.vggt_context import (
    CachedVGGTContextStore,
    VGGTContextBuilder,
    cache_vggt_contexts,
)
from train_gs_ae import load_experiment_manifest, resolve_category_ids

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--cache_root", required=True)
    p.add_argument("--gobjaverse_render_root", default=None)
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--categories", default=None)
    p.add_argument("--view_idx", type=int, default=0)
    p.add_argument(
        "--num_views",
        type=int,
        default=None,
        help="Cache views 0..N-1 (overrides --view_idx when set).",
    )
    p.add_argument(
        "--view_indices",
        type=str,
        default=None,
        help='Explicit views, e.g. "0-39" or "0,5,10,20". Overrides --num_views.',
    )
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no_experiment_manifest", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--fp32", action="store_true", help="Store fp32 instead of fp16.")
    args = p.parse_args()

    views = parse_view_indices(
        view_idx=args.view_idx,
        num_views=args.num_views,
        view_indices=args.view_indices,
    )
    logger.info("Caching %d view(s): %s", len(views), views if len(views) <= 12 else f"{views[:6]}…{views[-3:]}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_path = Path(args.data_dir).resolve()
    manifest = (
        load_experiment_manifest(str(data_path)) if not args.no_experiment_manifest else None
    )
    # Dataset here is only used for mesh_paths + render_loader; single-view list is fine.
    dataset = build_surface_render_dataset(
        str(data_path),
        max_items=args.max_items,
        categories=resolve_category_ids(args.categories),
        use_experiment_manifest=not args.no_experiment_manifest,
        manifest=manifest,
        render_root=args.gobjaverse_render_root,
        view_indices=views,
        surface_in_camera_frame=True,
        filter_missing_views=True,
    )
    if dataset.render_loader is None:
        raise SystemExit("No G-Objaverse render source resolved; check --gobjaverse_render_root")

    builder = VGGTContextBuilder(width=args.width).to(device)
    store = CachedVGGTContextStore(args.cache_root)

    written = cache_vggt_contexts(
        list(dataset.mesh_paths),
        dataset.render_loader,
        store,
        builder,
        view_indices=views,
        device=device,
        overwrite=args.overwrite,
        store_dtype=torch.float32 if args.fp32 else torch.float16,
    )
    total = sum(f.stat().st_size for f in Path(args.cache_root).glob("*.pt"))
    logger.info(
        "Cached %d new views into %s (%d files, %.2f GB total)",
        written,
        args.cache_root,
        len(list(Path(args.cache_root).glob("*.pt"))),
        total / 1e9,
    )


if __name__ == "__main__":
    main()
