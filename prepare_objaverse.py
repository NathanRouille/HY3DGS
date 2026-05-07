#!/usr/bin/env python3
"""Objaverse dataset preparation for ShapeGSAE training.

This script mirrors prepare_shapenet.py behavior for Objaverse-style folders:
  1. Recursively scans for mesh files.
  2. Optionally validates each mesh (loadable and non-degenerate).
  3. Optionally keeps only colored/textured meshes.
  4. Creates a reproducible train/val split.
  5. Writes train_list.json and val_list.json.
  6. Builds train/ and val/ directory trees (symlink or copy).
  7. Optionally pre-caches GT RGBD for faster training.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_EXTENSIONS = ".glb,.gltf,.obj,.ply,.stl,.off"


def discover_objaverse_models(
    objaverse_dir: str,
    extensions: List[str],
) -> List[Dict]:
    """Recursively find mesh files under objaverse_dir."""
    root = Path(objaverse_dir)
    if not root.exists() or not root.is_dir():
        logger.error(f"Objaverse root not found or not a directory: {root}")
        return []

    ext_set = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    models: List[Dict] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in ext_set:
            continue
        rel = p.relative_to(root)
        # pseudo category = first path component (for stratified split, if available)
        pseudo_category = rel.parts[0] if len(rel.parts) > 1 else "root"
        models.append(
            {
                "category": pseudo_category,
                "model_id": p.stem,
                "mesh_path": str(p),
            }
        )

    logger.info(f"Discovered {len(models)} mesh files under {objaverse_dir}")
    return models


def validate_mesh(mesh_path: str) -> Tuple[bool, str]:
    """Return (is_valid, reason). Loads mesh and checks basic sanity."""
    try:
        import trimesh

        mesh = trimesh.load(mesh_path, process=False)
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


def check_has_color(mesh_path: str) -> bool:
    """Return True if mesh has meaningful texture/color."""
    try:
        import numpy as np
        import trimesh

        mesh = trimesh.load(mesh_path, process=False)
        if isinstance(mesh, trimesh.scene.Scene):
            geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not geoms:
                return False
            mesh = trimesh.util.concatenate(geoms)

        if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
            return True

        try:
            vc = mesh.visual.to_color().vertex_colors[:, :3].astype(np.float32)
            std = vc.std(axis=0).mean()
            return float(std) > 0.05
        except Exception:
            return False
    except Exception:
        return False


def create_train_val_split(
    models: List[Dict],
    val_fraction: float = 0.1,
    seed: int = 42,
    color_only: bool = False,
) -> Tuple[List[Dict], List[Dict]]:
    """Create reproducible train/val split, stratified by pseudo-category."""
    rng = random.Random(seed)

    if color_only:
        logger.info("Filtering to colored meshes (this may take a while)...")
        colored = []
        for i, m in enumerate(models):
            if check_has_color(m["mesh_path"]):
                colored.append(m)
            if (i + 1) % 500 == 0:
                logger.info(f"  Color check: {i+1}/{len(models)} done, {len(colored)} colored so far")
        logger.info(f"Color filter: {len(colored)}/{len(models)} models have color data")
        models = colored

    by_cat: Dict[str, List[Dict]] = {}
    for m in models:
        by_cat.setdefault(m["category"], []).append(m)

    train_list, val_list = [], []
    for cat, cat_models in sorted(by_cat.items()):
        rng.shuffle(cat_models)
        n_val = max(1, int(len(cat_models) * val_fraction))
        val_list.extend(cat_models[:n_val])
        train_list.extend(cat_models[n_val:])
        logger.info(f"  {cat}: {len(cat_models) - n_val} train + {n_val} val")

    return train_list, val_list


def create_dataset_dirs(
    output_dir: str,
    train_list: List[Dict],
    val_list: List[Dict],
    use_symlinks: bool = True,
) -> Tuple[str, str]:
    """Create train/ and val/ subdirectories with links/copies to mesh files."""
    root = Path(output_dir)
    train_dir = root / "train"
    val_dir = root / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    def link_models(models: List[Dict], target_dir: Path):
        for m in models:
            src = Path(m["mesh_path"])
            safe_name = f"{m['category']}_{m['model_id']}{src.suffix.lower()}"
            dst = target_dir / safe_name
            if dst.exists() or dst.is_symlink():
                continue
            if use_symlinks:
                try:
                    dst.symlink_to(src.resolve())
                except Exception as e:
                    logger.warning(f"Symlink failed for {safe_name}: {e}. Skipping.")
            else:
                import shutil

                shutil.copy2(src, dst)

    logger.info(f"Creating train links in {train_dir} ...")
    link_models(train_list, train_dir)
    logger.info(f"Creating val links in {val_dir} ...")
    link_models(val_list, val_dir)
    return str(train_dir), str(val_dir)


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
    """Pre-render and cache GT RGBD for all meshes in the list."""
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
    logger.info(f"Pre-caching GT RGBD for {n} {split_name} meshes (tag: {renderer._tag}) ...")
    for i, m in enumerate(model_list):
        mesh_path = m["mesh_path"]
        try:
            renderer.get_or_render(mesh_path)
            ok += 1
        except Exception as e:
            failed.append(mesh_path)
            logger.warning(f"  [{i+1}/{n}] FAILED {mesh_path}: {e}")
        if (i + 1) % 50 == 0 or (i + 1) == n:
            logger.info(f"  [{i+1}/{n}] {ok} ok, {len(failed)} failed")

    logger.info(f"Pre-cache {split_name}: {ok}/{n} succeeded, {len(failed)} failed.")
    if failed:
        logger.warning("Failed meshes:\n" + "\n".join(f"  {p}" for p in failed[:20]))


def report_stats(train_list: List[Dict], val_list: List[Dict]):
    all_cats = sorted(set(m["category"] for m in train_list + val_list))
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"{'Category':<35} {'Train':>8} {'Val':>6} {'Total':>8}")
    print("-" * 60)
    for cat in all_cats:
        n_train = sum(1 for m in train_list if m["category"] == cat)
        n_val = sum(1 for m in val_list if m["category"] == cat)
        print(f"  {cat:<35} {n_train:>8} {n_val:>6} {n_train+n_val:>8}")
    print("-" * 60)
    print(f"  {'TOTAL':<35} {len(train_list):>8} {len(val_list):>6} {len(train_list)+len(val_list):>8}")
    print("=" * 60 + "\n")


def parse_args():
    p = argparse.ArgumentParser(description="Prepare Objaverse meshes for ShapeGSAE")
    p.add_argument("--objaverse_dir", required=True, help="Root directory containing Objaverse mesh files.")
    p.add_argument("--output_dir", required=True, help="Output folder for train/val links and JSON lists.")
    p.add_argument(
        "--extensions",
        type=str,
        default=DEFAULT_EXTENSIONS,
        help=f"Comma-separated mesh extensions to include. Default: {DEFAULT_EXTENSIONS}",
    )

    p.add_argument("--val_fraction", type=float, default=0.1, help="Validation split fraction (default 0.1).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducible split.")
    p.add_argument("--max_items", type=int, default=None, help="Optional cap on discovered meshes.")
    p.add_argument(
        "--color_only",
        action="store_true",
        help="Keep only meshes with color/texture signal (useful to avoid gray GT).",
    )
    p.add_argument(
        "--validate_meshes",
        action="store_true",
        help="Load every mesh with trimesh and keep only valid geometry (slower).",
    )

    p.add_argument("--precache", action="store_true", help="Pre-render/cache GT RGBD for train/val.")
    p.add_argument("--render_height", type=int, default=256)
    p.add_argument("--render_width", type=int, default=256)
    p.add_argument("--num_views", type=int, default=4)
    p.add_argument("--camera_azimuths", type=str, default="0,90,180,270")
    p.add_argument("--elevation_deg", type=float, default=20.0)
    p.add_argument("--camera_distance", type=float, default=2.5)
    p.add_argument("--precache_train_only", action="store_true", help="Only pre-cache training split.")

    p.add_argument("--no_symlinks", action="store_true", help="Copy files instead of symlinking.")
    return p.parse_args()


def main():
    args = parse_args()
    extensions = [e.strip() for e in args.extensions.split(",") if e.strip()]

    models = discover_objaverse_models(args.objaverse_dir, extensions)
    if args.max_items is not None:
        models = models[: args.max_items]
        logger.info(f"Capped dataset to {len(models)} meshes (--max_items).")
    if not models:
        logger.error("No meshes found. Check --objaverse_dir and --extensions.")
        sys.exit(1)

    if args.validate_meshes:
        logger.info("Validating meshes (this may take several minutes)...")
        valid_models, failed_models = [], []
        for i, m in enumerate(models):
            ok, reason = validate_mesh(m["mesh_path"])
            if ok:
                valid_models.append(m)
            else:
                failed_models.append((m["mesh_path"], reason))
            if (i + 1) % 500 == 0:
                logger.info(f"  Validated {i+1}/{len(models)}: {len(valid_models)} ok, {len(failed_models)} failed")
        logger.info(f"Validation complete: {len(valid_models)}/{len(models)} valid")
        if failed_models:
            bad_path = Path(args.output_dir) / "bad_meshes.txt"
            bad_path.parent.mkdir(parents=True, exist_ok=True)
            with open(bad_path, "w", encoding="utf-8") as f:
                for path, reason in failed_models:
                    f.write(f"{path}\t{reason}\n")
            logger.info(f"Bad mesh list written to {bad_path}")
        models = valid_models

    logger.info("Creating train/val split...")
    train_list, val_list = create_train_val_split(
        models,
        val_fraction=args.val_fraction,
        seed=args.seed,
        color_only=args.color_only,
    )

    report_stats(train_list, val_list)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    train_json = output_root / "train_list.json"
    val_json = output_root / "val_list.json"
    with open(train_json, "w", encoding="utf-8") as f:
        json.dump(train_list, f, indent=2)
    with open(val_json, "w", encoding="utf-8") as f:
        json.dump(val_list, f, indent=2)
    logger.info(f"Saved {train_json} ({len(train_list)} meshes)")
    logger.info(f"Saved {val_json} ({len(val_list)} meshes)")

    logger.info("Creating train/ and val/ directory structure...")
    train_dir, val_dir = create_dataset_dirs(
        args.output_dir,
        train_list,
        val_list,
        use_symlinks=not args.no_symlinks,
    )
    logger.info(f"Train dir: {train_dir}")
    logger.info(f"Val dir:   {val_dir}")

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
    print(f"  Train: {train_dir}  ({len(train_list)} meshes)")
    print(f"  Val:   {val_dir}  ({len(val_list)} meshes)")
    print("\nNext steps:")
    print("  1. (Optional) pre-cache GT:")
    print(f"     CUDA_VISIBLE_DEVICES=\"\" python prepare_objaverse.py \\")
    print(f"         --objaverse_dir {args.objaverse_dir} \\")
    print(f"         --output_dir {args.output_dir} \\")
    print(f"         --extensions {args.extensions} \\")
    print(f"         --precache \\")
    print(f"         --render_height {args.render_height} --render_width {args.render_width} \\")
    print(f"         --num_views {args.num_views} --camera_azimuths \"{args.camera_azimuths}\"")
    print()
    print("  2. Start training:")
    print(f"     python train_gs_ae.py \\")
    print(f"         --data_dir {train_dir} \\")
    print(f"         --val_dir  {val_dir} \\")
    print(f"         --output_dir runs/objaverse_baseline \\")
    print(f"         --num_views {args.num_views} \\")
    print(f"         --camera_azimuths \"{args.camera_azimuths}\" \\")
    print(f"         --use_wandb")


if __name__ == "__main__":
    main()
