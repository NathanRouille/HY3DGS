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

    # Per-view cache (independent VGGT forwards — NOT for joint MV train):
    python cache_vggt_features.py \
      --view_indices "16,29" ...

    # Joint 2-view cache (single ordered sequence, order matters):
    python cache_vggt_features.py \
      --data_dir .../furniture_351/train \
      --gobjaverse_render_root /export/home/nathan/datasets \
      --cache_root runs/vggt_cache/furn4_joint_v16_v29 \
      --view_indices "16,29" \
      --joint \
      --max_items 100

    # All ordered pairs from a view pool (for random-pair joint train):
    python cache_vggt_features.py \
      --data_dir .../furniture_351/train \
      --gobjaverse_render_root /export/home/nathan/datasets \
      --cache_root runs/vggt_cache/furn100_joint_pool_4_10_16_22 \
      --view_indices "4,10,16,22" \
      --joint_pairs \
      --max_items 100

    # InternScenes room (mask_white_bg=False):
    python cache_vggt_features.py \
      --dataset internscenes \
      --data_dir ~/datasets/internscenes_bathroom_130 \
      --internscenes_room_ids gen__bathroom__5571 \
      --cache_root runs/vggt_cache/internscenes_5571_joint_1_3_7_9 \
      --view_indices "1,3,7,9" \
      --joint_pairs

Payload (camera-frame, VGGT-depth centres; background patches already dropped):
  patch_tokens, camera_token, patch_centers, patch_keep,
  pe_frame='camera', geometry_source='vggt_depth'

Joint payloads add cache_kind='joint', view_indices, vggt_points_cam0 (merged dense FG).

Stored pre-projection (``feat_proj`` still trains) in fp16.
**Do not reuse** older caches from the GT-depth / object-frame era.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from hy3dgen.shapegen.gobjaverse_gt import parse_view_indices
from hy3dgen.shapegen.pc_render_dataset import (
    build_internscenes_render_dataset,
    build_surface_render_dataset,
)
from hy3dgen.shapegen.vggt_context import (
    CachedVGGTContextStore,
    VGGTContextBuilder,
    cache_vggt_contexts,
    cache_vggt_joint_contexts,
    cache_vggt_joint_ordered_pair_contexts,
    ordered_view_tuples,
)
from train_gs_ae import load_experiment_manifest, resolve_category_ids

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", required=True)
    p.add_argument(
        "--dataset",
        type=str,
        default="gobjaverse",
        choices=("gobjaverse", "internscenes"),
    )
    p.add_argument("--internscenes_split", type=str, default="train")
    p.add_argument("--internscenes_room_ids", type=str, default=None)
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
        help='Explicit views, e.g. "4,10,16,22" (order = pool order). Overrides --num_views.',
    )
    p.add_argument(
        "--joint",
        action="store_true",
        help=(
            "Joint multi-view cache: one VGGT forward per mesh on the full "
            "--view_indices list (requires >=2 views; first view is cam0)."
        ),
    )
    p.add_argument(
        "--joint_pairs",
        action="store_true",
        help=(
            "Cache joint VGGT for every ordered pair from --view_indices "
            "(P(n,2) files per mesh). Use with train "
            "--view_indices pool --views_per_sample 2 --vggt_joint_cache."
        ),
    )
    p.add_argument(
        "--pair_size",
        type=int,
        default=2,
        help="With --joint_pairs: tuple size (default 2 = ordered pairs).",
    )
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no_experiment_manifest", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--fp32", action="store_true", help="Store fp32 instead of fp16.")
    args = p.parse_args()

    if args.joint and args.joint_pairs:
        raise SystemExit("Pass only one of --joint or --joint_pairs")

    views = parse_view_indices(
        view_idx=args.view_idx,
        num_views=args.num_views,
        view_indices=args.view_indices,
    )
    if args.joint and len(views) < 2:
        raise SystemExit("--joint requires >=2 views in --view_indices (order matters).")
    if args.joint_pairs:
        if len(views) < int(args.pair_size):
            raise SystemExit(
                f"--joint_pairs requires >= {args.pair_size} views in --view_indices"
            )
        n_tuples = len(ordered_view_tuples(views, tuple_size=int(args.pair_size)))
        mode = f"joint_pairs({n_tuples} ordered {args.pair_size}-tuples)"
    elif args.joint:
        mode = "joint"
    else:
        mode = "per-view"

    logger.info(
        "Caching %s %d view(s): %s",
        mode,
        len(views),
        views if len(views) <= 12 else f"{views[:6]}…{views[-3:]}",
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_path = Path(args.data_dir).resolve()
    manifest = (
        load_experiment_manifest(str(data_path)) if not args.no_experiment_manifest else None
    )
    # Dataset only needs the view pool for mesh discovery / render filtering.
    if args.dataset == "internscenes":
        room_ids = None
        if args.internscenes_room_ids:
            room_ids = [
                x.strip()
                for x in str(args.internscenes_room_ids).split(",")
                if x.strip()
            ]
        dataset = build_internscenes_render_dataset(
            str(data_path),
            split=str(args.internscenes_split),
            room_ids=room_ids,
            max_items=args.max_items,
            view_indices=views,
            surface_in_camera_frame=True,
            filter_missing_views=True,
            align_mode="c_meanrms",
        )
    else:
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
        raise SystemExit("No render GT source resolved; check paths / --gobjaverse_render_root")

    mask_white_bg = args.dataset != "internscenes"
    builder = VGGTContextBuilder(width=args.width, mask_white_bg=mask_white_bg).to(
        device
    )
    store = CachedVGGTContextStore(args.cache_root)
    store_dtype = torch.float32 if args.fp32 else torch.float16

    if args.joint_pairs:
        written = cache_vggt_joint_ordered_pair_contexts(
            list(dataset.mesh_paths),
            dataset.render_loader,
            store,
            builder,
            view_indices=views,
            pair_size=int(args.pair_size),
            device=device,
            overwrite=args.overwrite,
            store_dtype=store_dtype,
        )
    elif args.joint:
        written = cache_vggt_joint_contexts(
            list(dataset.mesh_paths),
            dataset.render_loader,
            store,
            builder,
            view_indices=views,
            device=device,
            overwrite=args.overwrite,
            store_dtype=store_dtype,
        )
    else:
        written = cache_vggt_contexts(
            list(dataset.mesh_paths),
            dataset.render_loader,
            store,
            builder,
            view_indices=views,
            device=device,
            overwrite=args.overwrite,
            store_dtype=store_dtype,
        )

    total = sum(f.stat().st_size for f in Path(args.cache_root).glob("*.pt"))
    logger.info(
        "Cached %d new %s entries into %s (%d files, %.2f GB total)",
        written,
        mode,
        args.cache_root,
        len(list(Path(args.cache_root).glob("*.pt"))),
        total / 1e9,
    )


if __name__ == "__main__":
    main()
