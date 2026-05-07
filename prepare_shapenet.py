#!/usr/bin/env python3
"""ShapeNet Core v2 dataset preparation for ShapeGSAE training.

This script:
  1. Scans ShapeNetCore.v2 for all valid OBJ models in selected categories.
  2. Validates each mesh (loadable, has geometry, has texture/color).
  3. Creates a reproducible 90/10 train/val split.
  4. Writes train_list.json and val_list.json index files.
  5. Optionally pre-caches GT RGBD renders (identical to --precache_only mode).

Usage
-----
Step 1 — install ShapeNet Core v2
    Register at https://shapenet.org/ and download ShapeNetCore.v2.zip (~25 GB).
    Extract to /path/to/ShapeNetCore.v2/

Step 2 — prepare index files
    python prepare_shapenet.py \\
        --shapenet_dir /path/to/ShapeNetCore.v2 \\
        --output_dir   /path/to/data/shapenet \\
        --categories   03001627,04379243,02958343 \\
        --val_fraction 0.1

Step 3 — pre-cache GT RGBD (CPU-only, run before training to avoid EGL issues)
    CUDA_VISIBLE_DEVICES="" python prepare_shapenet.py \\
        --shapenet_dir /path/to/ShapeNetCore.v2 \\
        --output_dir   /path/to/data/shapenet \\
        --categories   03001627,04379243,02958343 \\
        --precache \\
        --render_height 256 --render_width 256 \\
        --num_views 4 --camera_azimuths "0,90,180,270"

Step 4 — train
    python train_gs_ae.py \\
        --data_dir  /path/to/data/shapenet/train \\
        --val_dir   /path/to/data/shapenet/val \\
        --output_dir runs/shapenet_baseline \\
        --num_views 4 --camera_azimuths "0,90,180,270"

Category IDs (common subsets)
-----------------------------
    02691156  airplane    (~4045 models,  ~2.5 GB)
    02828884  bench       (~1813 models,  ~1.0 GB)
    02933112  cabinet     (~1571 models,  ~1.5 GB)
    02958343  car         (~3533 models,  ~4.5 GB) ← good textures
    03001627  chair       (~6778 models,  ~3.5 GB) ← start here
    03636649  lamp        (~2318 models,  ~2.0 GB)
    04256520  sofa        (~3173 models,  ~2.5 GB)
    04379243  table       (~8509 models,  ~5.0 GB)
    04530566  watercraft  (~1939 models,  ~1.5 GB)

Total for all 9 categories: ~37k models, ~24 GB (well within 50 GB budget).
For a quick start use only chairs (03001627, ~3.5 GB).

Notes on textures
-----------------
ShapeNet v2 models use UV-mapped textures (OBJ + MTL + PNG/JPG).
trimesh.load() handles these correctly and to_color() bakes textures to
per-vertex colors. pyrender renders UV textures directly, producing
higher-quality GT images. Both paths are supported.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Default categories: chairs + tables + cars — good diversity, textures available
DEFAULT_CATEGORIES = "03001627,04379243,02958343"

CATEGORY_NAMES = {
    "02691156": "airplane",
    "02828884": "bench",
    "02933112": "cabinet",
    "02958343": "car",
    "03001627": "chair",
    "03636649": "lamp",
    "04256520": "sofa",
    "04379243": "table",
    "04530566": "watercraft",
}


# ---------------------------------------------------------------------------
# Mesh discovery
# ---------------------------------------------------------------------------

def discover_shapenet_models(
    shapenet_dir: str,
    categories: List[str],
) -> List[Dict]:
    """Find all model_normalized.obj files in the given categories.

    Returns a list of dicts: {'category': str, 'model_id': str, 'obj_path': str}
    """
    root = Path(shapenet_dir)
    models = []
    for cat in categories:
        cat_dir = root / cat
        if not cat_dir.exists():
            logger.warning(f"Category {cat} ({CATEGORY_NAMES.get(cat, '?')}) not found at {cat_dir}")
            continue
        for model_dir in sorted(cat_dir.iterdir()):
            obj_path = model_dir / "models" / "model_normalized.obj"
            if obj_path.exists():
                models.append({
                    "category": cat,
                    "model_id": model_dir.name,
                    "obj_path": str(obj_path),
                })
    logger.info(f"Discovered {len(models)} models across {len(categories)} categories")
    return models


# ---------------------------------------------------------------------------
# Mesh validation
# ---------------------------------------------------------------------------

def validate_mesh(obj_path: str) -> Tuple[bool, str]:
    """Return (is_valid, reason). Loads mesh and checks basic sanity."""
    try:
        import trimesh
        mesh = trimesh.load(obj_path, process=False)
        if isinstance(mesh, trimesh.scene.Scene):
            geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not geoms:
                return False, "scene with no trimesh geometry"
            mesh = trimesh.util.concatenate(geoms)
        if not isinstance(mesh, trimesh.Trimesh):
            return False, f"unexpected type {type(mesh)}"
        if len(mesh.faces) < 10:
            return False, f"too few faces ({len(mesh.faces)})"
        if len(mesh.vertices) < 10:
            return False, f"too few vertices ({len(mesh.vertices)})"
        bbox = mesh.bounds
        if bbox is None or (bbox[1] - bbox[0]).max() < 1e-6:
            return False, "degenerate bounding box"
        return True, "ok"
    except Exception as e:
        return False, str(e)


def check_has_color(obj_path: str) -> bool:
    """Return True if the mesh appears to have non-grey color information."""
    try:
        import trimesh
        import numpy as np
        mesh = trimesh.load(obj_path, process=False)
        if isinstance(mesh, trimesh.scene.Scene):
            geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not geoms:
                return False
            mesh = trimesh.util.concatenate(geoms)
        # Check for UV texture (most ShapeNet models use this)
        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            return True  # UV-mapped → has texture
        # Check vertex colors
        try:
            vc = mesh.visual.to_color().vertex_colors[:, :3].astype(np.float32)
            # Compute per-channel std to see if it's a uniform grey
            std = vc.std(axis=0).mean()
            return float(std) > 0.05  # meaningful color variation
        except Exception:
            return False
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Split creation
# ---------------------------------------------------------------------------

def create_train_val_split(
    models: List[Dict],
    val_fraction: float = 0.1,
    seed: int = 42,
    color_only: bool = False,
) -> Tuple[List[Dict], List[Dict]]:
    """Create reproducible 90/10 (or custom) train/val split.

    Stratifies by category so each category has the same train/val ratio.
    If color_only is True, keeps only models with non-grey color data.
    """
    rng = random.Random(seed)

    if color_only:
        logger.info("Filtering to colored meshes (this may take a few minutes)...")
        colored = []
        for i, m in enumerate(models):
            if check_has_color(m['obj_path']):
                colored.append(m)
            if (i + 1) % 500 == 0:
                logger.info(f"  Color check: {i+1}/{len(models)} done, {len(colored)} colored so far")
        logger.info(f"Color filter: {len(colored)}/{len(models)} models have color data")
        models = colored

    # Group by category
    by_cat: Dict[str, List[Dict]] = {}
    for m in models:
        by_cat.setdefault(m['category'], []).append(m)

    train_list, val_list = [], []
    for cat, cat_models in sorted(by_cat.items()):
        rng.shuffle(cat_models)
        n_val = max(1, int(len(cat_models) * val_fraction))
        val_list.extend(cat_models[:n_val])
        train_list.extend(cat_models[n_val:])
        logger.info(
            f"  {cat} ({CATEGORY_NAMES.get(cat,'?')}): "
            f"{len(cat_models) - n_val} train + {n_val} val"
        )

    return train_list, val_list


# ---------------------------------------------------------------------------
# Symlink / directory structure
# ---------------------------------------------------------------------------

def create_dataset_dirs(
    output_dir: str,
    train_list: List[Dict],
    val_list: List[Dict],
    use_symlinks: bool = True,
) -> Tuple[str, str]:
    """Create train/ and val/ subdirectories with symlinks to OBJ files.

    Returns (train_dir, val_dir) absolute paths.
    """
    root = Path(output_dir)
    train_dir = root / "train"
    val_dir = root / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    def link_models(models: List[Dict], target_dir: Path):
        for m in models:
            src = Path(m['obj_path']).parent     # .../models/
            # Use category_modelid as unique name to avoid collisions
            link_name = f"{m['category']}_{m['model_id']}"
            dst = target_dir / link_name
            if dst.exists() or dst.is_symlink():
                continue
            if use_symlinks:
                try:
                    dst.symlink_to(src.resolve())
                except Exception as e:
                    logger.warning(f"Symlink failed for {link_name}: {e}. Skipping.")
            else:
                # Hard-copy the models/ directory
                import shutil
                shutil.copytree(src, dst, ignore_errors=True)

    logger.info(f"Creating train symlinks in {train_dir} ...")
    link_models(train_list, train_dir)
    logger.info(f"Creating val symlinks in {val_dir} ...")
    link_models(val_list, val_dir)

    return str(train_dir), str(val_dir)


# ---------------------------------------------------------------------------
# Pre-caching helper
# ---------------------------------------------------------------------------

def precache_gt(
    model_list: List[Dict],
    render_height: int,
    render_width: int,
    num_views: int,
    camera_azimuths: Optional[str],
    elevation_deg: float,
    camera_distance: float,
    split_name: str,
):
    """Pre-render and cache GT RGBD for all models in the list."""
    # Import training helpers
    repo = Path(__file__).resolve().parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from train_gs_ae import GTRGBDRenderer

    azimuths_deg = None
    if camera_azimuths:
        azimuths_deg = [float(a.strip()) for a in camera_azimuths.split(",")]
        if len(azimuths_deg) != num_views:
            raise ValueError(f"camera_azimuths has {len(azimuths_deg)} values but num_views={num_views}")

    renderer = GTRGBDRenderer(
        height=render_height,
        width=render_width,
        num_views=num_views,
        camera_distance=camera_distance,
        elevation_deg=elevation_deg,
        azimuths_deg=azimuths_deg,
    )

    n = len(model_list)
    ok, failed = 0, []
    logger.info(f"Pre-caching GT RGBD for {n} {split_name} models (tag: {renderer._tag}) ...")
    for i, m in enumerate(model_list):
        # The OBJ path is inside a models/ directory; we pass the full OBJ path
        obj_path = m["obj_path"]
        try:
            renderer.get_or_render(obj_path)
            ok += 1
        except Exception as e:
            failed.append(obj_path)
            logger.warning(f"  [{i+1}/{n}] FAILED {obj_path}: {e}")
        if (i + 1) % 50 == 0 or (i + 1) == n:
            logger.info(f"  [{i+1}/{n}] {ok} ok, {len(failed)} failed")

    logger.info(f"Pre-cache {split_name}: {ok}/{n} succeeded, {len(failed)} failed.")
    if failed:
        logger.warning("Failed models:\n" + "\n".join(f"  {p}" for p in failed[:20]))


# ---------------------------------------------------------------------------
# Statistics reporting
# ---------------------------------------------------------------------------

def report_stats(train_list: List[Dict], val_list: List[Dict]):
    """Print dataset statistics."""
    all_cats = sorted(set(m['category'] for m in train_list + val_list))
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"{'Category':<40} {'Train':>8} {'Val':>6} {'Total':>8}")
    print("-" * 60)
    for cat in all_cats:
        n_train = sum(1 for m in train_list if m['category'] == cat)
        n_val = sum(1 for m in val_list if m['category'] == cat)
        name = CATEGORY_NAMES.get(cat, "unknown")
        print(f"  {cat} ({name:<20}) {n_train:>8} {n_val:>6} {n_train+n_val:>8}")
    print("-" * 60)
    print(f"  {'TOTAL':<38} {len(train_list):>8} {len(val_list):>6} {len(train_list)+len(val_list):>8}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Prepare ShapeNet Core v2 for ShapeGSAE")

    # Paths
    p.add_argument("--shapenet_dir", required=True,
                   help="Root of extracted ShapeNetCore.v2 (contains category ID subdirs).")
    p.add_argument("--output_dir", required=True,
                   help="Where to write train/, val/ symlinks and index JSON files.")

    # Category selection
    p.add_argument("--categories", type=str, default=DEFAULT_CATEGORIES,
                   help=f"Comma-separated ShapeNet category IDs. Default: {DEFAULT_CATEGORIES} "
                        f"(chair, table, car). Use 'all' for all 55 categories.")

    # Split
    p.add_argument("--val_fraction", type=float, default=0.1,
                   help="Fraction of each category to use for validation (default 0.1 = 10%%).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducible splits.")
    p.add_argument("--color_only", action="store_true",
                   help="Only keep models that have non-grey color data. "
                        "Slower (loads every mesh) but produces a cleaner training set.")

    # Validation
    p.add_argument("--validate_meshes", action="store_true",
                   help="Load every mesh with trimesh to validate geometry (slower but catches bad models).")

    # Pre-caching
    p.add_argument("--precache", action="store_true",
                   help="After creating splits, pre-render and cache GT RGBD for all models.")
    p.add_argument("--render_height", type=int, default=256)
    p.add_argument("--render_width", type=int, default=256)
    p.add_argument("--num_views", type=int, default=4)
    p.add_argument("--camera_azimuths", type=str, default="0,90,180,270",
                   help="Comma-separated azimuth angles. Must match num_views.")
    p.add_argument("--elevation_deg", type=float, default=20.0)
    p.add_argument("--camera_distance", type=float, default=2.5)
    p.add_argument("--precache_train_only", action="store_true",
                   help="Only pre-cache training split (skip val).")

    # Output format
    p.add_argument("--no_symlinks", action="store_true",
                   help="Copy model directories instead of symlinking (for remote filesystems).")

    return p.parse_args()


def main():
    args = parse_args()

    # Parse categories
    if args.categories.lower() == "all":
        shapenet_root = Path(args.shapenet_dir)
        categories = sorted([d.name for d in shapenet_root.iterdir() if d.is_dir()])
        logger.info(f"Using all {len(categories)} categories found in {args.shapenet_dir}")
    else:
        categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    logger.info(f"Selected categories: {categories}")
    for cat in categories:
        logger.info(f"  {cat} → {CATEGORY_NAMES.get(cat, 'unknown')}")

    # Discover models
    models = discover_shapenet_models(args.shapenet_dir, categories)
    if not models:
        logger.error("No models found. Check --shapenet_dir and --categories.")
        sys.exit(1)

    # Optional mesh validation
    if args.validate_meshes:
        logger.info("Validating meshes (this may take several minutes)...")
        valid_models = []
        failed_models = []
        for i, m in enumerate(models):
            is_valid, reason = validate_mesh(m["obj_path"])
            if is_valid:
                valid_models.append(m)
            else:
                failed_models.append((m["obj_path"], reason))
            if (i + 1) % 500 == 0:
                logger.info(f"  Validated {i+1}/{len(models)}: {len(valid_models)} ok, {len(failed_models)} failed")
        logger.info(f"Validation complete: {len(valid_models)}/{len(models)} valid")
        if failed_models:
            bad_path = Path(args.output_dir) / "bad_meshes.txt"
            bad_path.parent.mkdir(parents=True, exist_ok=True)
            with open(bad_path, "w") as f:
                for path, reason in failed_models:
                    f.write(f"{path}\t{reason}\n")
            logger.info(f"Bad mesh list written to {bad_path}")
        models = valid_models

    # Create train/val split
    logger.info("Creating train/val split...")
    train_list, val_list = create_train_val_split(
        models,
        val_fraction=args.val_fraction,
        seed=args.seed,
        color_only=args.color_only,
    )

    # Report statistics
    report_stats(train_list, val_list)

    # Save index files
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    train_json = output_root / "train_list.json"
    val_json = output_root / "val_list.json"
    with open(train_json, "w") as f:
        json.dump(train_list, f, indent=2)
    with open(val_json, "w") as f:
        json.dump(val_list, f, indent=2)
    logger.info(f"Saved {train_json} ({len(train_list)} models)")
    logger.info(f"Saved {val_json} ({len(val_list)} models)")

    # Create symlinked directory structure
    logger.info("Creating train/ and val/ directory structure...")
    train_dir, val_dir = create_dataset_dirs(
        args.output_dir, train_list, val_list,
        use_symlinks=not args.no_symlinks,
    )
    logger.info(f"Train dir: {train_dir}")
    logger.info(f"Val dir:   {val_dir}")

    # Optional pre-caching
    if args.precache:
        logger.info("\n--- Pre-caching GT RGBD ---")
        precache_gt(
            train_list,
            render_height=args.render_height,
            render_width=args.render_width,
            num_views=args.num_views,
            camera_azimuths=args.camera_azimuths,
            elevation_deg=args.elevation_deg,
            camera_distance=args.camera_distance,
            split_name="train",
        )
        if not args.precache_train_only:
            precache_gt(
                val_list,
                render_height=args.render_height,
                render_width=args.render_width,
                num_views=args.num_views,
                camera_azimuths=args.camera_azimuths,
                elevation_deg=args.elevation_deg,
                camera_distance=args.camera_distance,
                split_name="val",
            )

    print("\n✓ Preparation complete.")
    print(f"  Train: {train_dir}  ({len(train_list)} models)")
    print(f"  Val:   {val_dir}  ({len(val_list)} models)")
    print("\nNext steps:")
    print("  1. If not pre-caching now, run pre-cache separately:")
    print(f"     CUDA_VISIBLE_DEVICES=\"\" python prepare_shapenet.py \\")
    print(f"         --shapenet_dir {args.shapenet_dir} \\")
    print(f"         --output_dir {args.output_dir} \\")
    print(f"         --categories {args.categories} \\")
    print(f"         --precache \\")
    print(f"         --render_height {args.render_height} --render_width {args.render_width} \\")
    print(f"         --num_views {args.num_views} --camera_azimuths \"{args.camera_azimuths}\"")
    print()
    print("  2. Start training:")
    print(f"     python train_gs_ae.py \\")
    print(f"         --data_dir {train_dir} \\")
    print(f"         --val_dir  {val_dir} \\")
    print(f"         --output_dir runs/shapenet_baseline \\")
    print(f"         --num_views {args.num_views} \\")
    print(f"         --camera_azimuths \"{args.camera_azimuths}\" \\")
    print(f"         --use_wandb")


if __name__ == "__main__":
    main()
