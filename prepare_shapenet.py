#!/usr/bin/env python3
"""ShapeNet Core v2 dataset preparation for ShapeGSAE training.

Creates **experiment** folders under ``--output_dir`` (one per ``--experiment_name``).
Each experiment contains:

  * ``train/``, ``val/`` — symlinks to ShapeNet ``models/`` dirs (subset only)
  * ``manifest.json`` — mesh paths, GT tag, and which canonical OBJs have GT caches
  * GT ``.pt`` files live under **ShapeNetCore** (canonical path), shared across experiments

Pre-cache GT only via this script (not ``train_gs_ae.py``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from hy3dgen.shapegen.gt_cache_util import (
    canonical_obj_path,
    experiment_manifest_path,
    is_usable_gt_cache_file,
    model_folder_name,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

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

CATEGORY_NAME_TO_ID: Dict[str, str] = {name: sid for sid, name in CATEGORY_NAMES.items()}

PrecachePerCategoryLimits = Union[int, Dict[str, int]]


def resolve_category_token(tok: str) -> str:
    tok = tok.strip()
    if not tok:
        raise ValueError("Empty category token")
    if tok in CATEGORY_NAME_TO_ID:
        return CATEGORY_NAME_TO_ID[tok]
    if tok.isdigit() and len(tok) == 8:
        return tok
    known = ", ".join(sorted(CATEGORY_NAME_TO_ID))
    raise ValueError(
        f"Unknown category '{tok}'. Use a known name ({known}) or an 8-digit synset ID."
    )


def parse_precache_per_category(limits_str: str) -> PrecachePerCategoryLimits:
    s = limits_str.strip()
    if not s:
        raise ValueError("Per-category limit string must not be empty")
    if ":" not in s:
        n = int(s)
        if n <= 0:
            raise ValueError(f"Per-category limit must be positive; got {n}")
        return n
    per_cat: Dict[str, int] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid segment {part!r}; expected 'category:limit'")
        key, val = part.split(":", 1)
        cat_id = resolve_category_token(key)
        limit = int(val.strip())
        if limit <= 0:
            raise ValueError(f"Per-category limit must be positive; got {key}={limit}")
        if cat_id in per_cat:
            raise ValueError(f"Duplicate category limit for {cat_id}")
        per_cat[cat_id] = limit
    if not per_cat:
        raise ValueError(f"No valid category limits parsed from {limits_str!r}")
    return per_cat


def limit_models_per_category(
    model_list: List[Dict],
    limits: PrecachePerCategoryLimits,
) -> List[Dict]:
    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for m in model_list:
        buckets[m["category"]].append(m)
    for cat in buckets:
        buckets[cat].sort(key=lambda m: (m.get("model_id", ""), m["obj_path"]))

    out: List[Dict] = []
    parts = []
    for cat in sorted(buckets):
        items = buckets[cat]
        if isinstance(limits, int):
            cap = limits
        else:
            cap = limits.get(cat)
        kept = items if cap is None else items[:cap]
        name = CATEGORY_NAMES.get(cat, "?")
        parts.append(f"{cat}({name})={len(kept)}/{len(items)}")
        out.extend(kept)
    logger.info(
        "Per-category cap: %s models (from %s). %s",
        len(out),
        len(model_list),
        "; ".join(parts),
    )
    return out


def discover_shapenet_models(shapenet_dir: str, categories: List[str]) -> List[Dict]:
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
                    "obj_path": str(obj_path.resolve()),
                })
    logger.info(f"Discovered {len(models)} models across {len(categories)} categories")
    return models


def validate_mesh(obj_path: str) -> Tuple[bool, str]:
    try:
        import trimesh
        mesh = trimesh.load(obj_path, process=False)
        if isinstance(mesh, trimesh.scene.Scene):
            geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not geoms:
                return False, "empty scene"
            mesh = trimesh.util.concatenate(geoms)
        if len(mesh.vertices) < 3 or len(mesh.faces) < 1:
            return False, "degenerate geometry"
        return True, "ok"
    except Exception as e:
        return False, str(e)


def check_has_color(obj_path: str) -> bool:
    try:
        import trimesh
        import numpy as np
        mesh = trimesh.load(obj_path, process=False)
        if isinstance(mesh, trimesh.scene.Scene):
            geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not geoms:
                return False
            mesh = trimesh.util.concatenate(geoms)
        if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
            return True
        try:
            vc = mesh.visual.to_color().vertex_colors[:, :3].astype(np.float32)
            return float(vc.std(axis=0).mean()) > 0.05
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
    rng = random.Random(seed)
    if color_only:
        logger.info("Filtering to colored meshes (this may take a few minutes)...")
        colored = []
        for i, m in enumerate(models):
            if check_has_color(m["obj_path"]):
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
        logger.info(
            f"  {cat} ({CATEGORY_NAMES.get(cat, '?')}): "
            f"{len(cat_models) - n_val} train + {n_val} val"
        )
    return train_list, val_list


def _reset_split_dir(split_dir: Path) -> None:
    if split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)


def create_experiment_symlinks(
    experiment_dir: Path,
    train_models: List[Dict],
    val_models: List[Dict],
    use_symlinks: bool = True,
) -> Tuple[List[str], List[str]]:
    """Symlink only ``train_models`` / ``val_models`` into the experiment tree.

    Returns lists of absolute paths to ``model_normalized.obj`` inside the experiment.
    """
    train_dir = experiment_dir / "train"
    val_dir = experiment_dir / "val"
    _reset_split_dir(train_dir)
    _reset_split_dir(val_dir)

    def link_models(models: List[Dict], target_dir: Path) -> List[str]:
        paths: List[str] = []
        for m in models:
            src = Path(m["obj_path"]).parent
            dst = target_dir / model_folder_name(m)
            if use_symlinks:
                dst.symlink_to(src.resolve(), target_is_directory=True)
            else:
                shutil.copytree(src, dst, dirs_exist_ok=True)
            obj = (dst / "model_normalized.obj").resolve()
            paths.append(str(obj))
        return paths

    logger.info("Creating train symlinks (%d models) ...", len(train_models))
    train_paths = link_models(train_models, train_dir)
    logger.info("Creating val symlinks (%d models) ...", len(val_models))
    val_paths = link_models(val_models, val_dir)
    return train_paths, val_paths


def precache_gt_for_models(
    model_list: List[Dict],
    renderer,
    split_name: str,
) -> Tuple[List[Dict], List[str]]:
    """Pre-render GT on canonical ShapeNet paths; skip if a valid cache already exists."""
    n = len(model_list)
    ok_models: List[Dict] = []
    failed: List[str] = []
    skipped = 0
    rendered = 0
    logger.info("Pre-caching GT for %d %s models (tag: %s) ...", n, split_name, renderer._tag)

    for i, m in enumerate(model_list):
        obj_path = m["obj_path"]
        cache_path = renderer.cache_path(obj_path)
        try:
            if is_usable_gt_cache_file(cache_path):
                skipped += 1
                ok_models.append(m)
            else:
                if os.path.isfile(cache_path):
                    try:
                        os.remove(cache_path)
                    except OSError:
                        pass
                renderer.get_or_render(obj_path, allow_render=True)
                rendered += 1
                ok_models.append(m)
        except Exception as e:
            failed.append(obj_path)
            logger.warning("  [%d/%d] FAILED %s: %s", i + 1, n, obj_path, e)
        if (i + 1) % 50 == 0 or (i + 1) == n:
            logger.info(
                "  [%d/%d] ok=%d skipped_existing=%d rendered=%d failed=%d",
                i + 1, n, len(ok_models), skipped, rendered, len(failed),
            )

    logger.info(
        "Pre-cache %s: %d/%d with usable GT (%d skipped existing, %d newly rendered, %d failed).",
        split_name,
        len(ok_models),
        n,
        skipped,
        rendered,
        len(failed),
    )
    return ok_models, failed


def build_renderer(
    render_height: int,
    render_width: int,
    num_views: int,
    camera_azimuths: Optional[str],
    elevation_deg: float,
    camera_distance: float,
    gt_view_layout: str,
):
    from train_gs_ae import GTRGBDRenderer

    layout = (gt_view_layout or "legacy").lower()
    if layout == "v46":
        return GTRGBDRenderer(
            height=render_height,
            width=render_width,
            num_views=num_views,
            camera_distance=camera_distance,
            elevation_deg=elevation_deg,
            azimuths_deg=None,
            view_layout="v46",
            train_view_indices=None,
        )
    azimuths_deg = None
    if camera_azimuths:
        azimuths_deg = [float(a.strip()) for a in camera_azimuths.split(",")]
        if len(azimuths_deg) != num_views:
            raise ValueError(
                f"camera_azimuths has {len(azimuths_deg)} values but num_views={num_views}"
            )
    return GTRGBDRenderer(
        height=render_height,
        width=render_width,
        num_views=num_views,
        camera_distance=camera_distance,
        elevation_deg=elevation_deg,
        azimuths_deg=azimuths_deg,
        view_layout="legacy",
    )


def models_with_usable_gt(
    model_list: List[Dict],
    renderer,
) -> List[Dict]:
    """Keep models whose canonical GT cache already exists on disk."""
    ok = []
    for m in model_list:
        if is_usable_gt_cache_file(renderer.cache_path(m["obj_path"])):
            ok.append(m)
    return ok


def write_experiment_manifest(
    experiment_dir: Path,
    renderer,
    train_models: List[Dict],
    val_models: List[Dict],
    train_mesh_paths: List[str],
    val_mesh_paths: List[str],
    render_height: int,
    render_width: int,
    gt_view_layout: str,
    camera_distance: float,
    categories: List[str],
    val_fraction: float,
    seed: int,
) -> None:
    cached_canonical = sorted({
        canonical_obj_path(m["obj_path"])
        for m in train_models + val_models
        if is_usable_gt_cache_file(renderer.cache_path(m["obj_path"]))
    })
    manifest = {
        "version": 1,
        "gt_tag": renderer._tag,
        "gt_view_layout": gt_view_layout,
        "render_height": render_height,
        "render_width": render_width,
        "camera_distance": camera_distance,
        "categories": categories,
        "val_fraction": val_fraction,
        "seed": seed,
        "train_mesh_paths": train_mesh_paths,
        "val_mesh_paths": val_mesh_paths,
        "cached_canonical_objs": cached_canonical,
        "train_models": train_models,
        "val_models": val_models,
    }
    path = experiment_manifest_path(experiment_dir)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote %s (%d train, %d val paths with GT)", path, len(train_mesh_paths), len(val_mesh_paths))


def report_stats(train_list: List[Dict], val_list: List[Dict]):
    all_cats = sorted(set(m["category"] for m in train_list + val_list))
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"{'Category':<40} {'Train':>8} {'Val':>6} {'Total':>8}")
    print("-" * 60)
    for cat in all_cats:
        n_train = sum(1 for m in train_list if m["category"] == cat)
        n_val = sum(1 for m in val_list if m["category"] == cat)
        name = CATEGORY_NAMES.get(cat, "unknown")
        print(f"  {cat} ({name:<20}) {n_train:>8} {n_val:>6} {n_train+n_val:>8}")
    print("-" * 60)
    print(f"  {'TOTAL':<38} {len(train_list):>8} {len(val_list):>6} {len(train_list)+len(val_list):>8}")
    print("=" * 60 + "\n")


def parse_args():
    p = argparse.ArgumentParser(description="Prepare ShapeNet Core v2 for ShapeGSAE")

    p.add_argument("--shapenet_dir", required=True,
                   help="Root of ShapeNetCore (category ID subdirs).")
    p.add_argument("--output_dir", required=True,
                   help="Parent directory for experiments (each run uses --experiment_name).")
    p.add_argument("--experiment_name", type=str, default="default",
                   help="Subfolder name under output_dir for this split + precache run.")

    p.add_argument("--categories", type=str, default=DEFAULT_CATEGORIES,
                   help="Comma-separated synset IDs or 'all'.")

    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--color_only", action="store_true")
    p.add_argument("--validate_meshes", action="store_true")

    p.add_argument("--precache", action="store_true",
                   help="Pre-render GT under ShapeNetCore (skips existing valid caches).")
    p.add_argument("--render_height", type=int, default=512)
    p.add_argument("--render_width", type=int, default=512)
    p.add_argument("--num_views", type=int, default=4)
    p.add_argument("--camera_azimuths", type=str, default="0,90,180,270")
    p.add_argument("--elevation_deg", type=float, default=20.0)
    p.add_argument("--camera_distance", type=float, default=3.5)
    p.add_argument("--precache_train_only", action="store_true")
    p.add_argument("--gt_view_layout", type=str, default="v46", choices=("legacy", "v46"))

    p.add_argument(
        "--precache_per_category_train",
        type=str,
        default=None,
        help="Cap train models per synset before precache (int or 'chair:500,...').",
    )
    p.add_argument(
        "--precache_per_category_val",
        type=str,
        default=None,
        help="Cap val models per synset before precache (int or mapping). Default: same as train cap.",
    )

    p.add_argument("--no_symlinks", action="store_true",
                   help="Copy model dirs instead of symlinking.")

    return p.parse_args()


def main():
    args = parse_args()

    if args.categories.lower() == "all":
        shapenet_root = Path(args.shapenet_dir)
        categories = sorted([d.name for d in shapenet_root.iterdir() if d.is_dir()])
    else:
        raw_cats = [c.strip() for c in args.categories.split(",") if c.strip()]
        categories = [resolve_category_token(c) for c in raw_cats]

    experiment_dir = Path(args.output_dir).resolve() / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Experiment directory: %s", experiment_dir)

    models = discover_shapenet_models(args.shapenet_dir, categories)
    if not models:
        logger.error("No models found.")
        sys.exit(1)

    if args.validate_meshes:
        valid_models = []
        for i, m in enumerate(models):
            ok, _ = validate_mesh(m["obj_path"])
            if ok:
                valid_models.append(m)
            if (i + 1) % 500 == 0:
                logger.info(f"  Validated {i+1}/{len(models)}")
        models = valid_models

    train_list, val_list = create_train_val_split(
        models, val_fraction=args.val_fraction, seed=args.seed, color_only=args.color_only,
    )

    train_limits: Optional[PrecachePerCategoryLimits] = None
    val_limits: Optional[PrecachePerCategoryLimits] = None
    if args.precache_per_category_train:
        train_limits = parse_precache_per_category(args.precache_per_category_train)
    if args.precache_per_category_val:
        val_limits = parse_precache_per_category(args.precache_per_category_val)
    elif train_limits is not None:
        val_limits = train_limits

    train_subset = limit_models_per_category(train_list, train_limits) if train_limits else train_list
    val_subset = limit_models_per_category(val_list, val_limits) if val_limits else val_list

    report_stats(train_subset, val_subset)

    renderer = build_renderer(
        args.render_height,
        args.render_width,
        args.num_views,
        args.camera_azimuths,
        args.elevation_deg,
        args.camera_distance,
        args.gt_view_layout,
    )

    train_ok, val_ok = train_subset, val_subset
    if args.precache:
        logger.info("\n--- Pre-caching GT RGBD (canonical ShapeNetCore paths) ---")
        precache_models = list(train_subset)
        if not args.precache_train_only:
            precache_models.extend(val_subset)
        by_canon: Dict[str, Dict] = {}
        for m in precache_models:
            by_canon[canonical_obj_path(m["obj_path"])] = m
        union_list = list(by_canon.values())
        logger.info("Union precache list: %d unique canonical meshes", len(union_list))

        ok_union, _ = precache_gt_for_models(union_list, renderer, "train+val")
        ok_canon = {canonical_obj_path(m["obj_path"]) for m in ok_union}
        train_ok = [m for m in train_subset if canonical_obj_path(m["obj_path"]) in ok_canon]
        val_ok = (
            []
            if args.precache_train_only
            else [m for m in val_subset if canonical_obj_path(m["obj_path"]) in ok_canon]
        )
    else:
        train_ok = models_with_usable_gt(train_subset, renderer)
        val_ok = (
            []
            if args.precache_train_only
            else models_with_usable_gt(val_subset, renderer)
        )
        logger.info(
            "No --precache: symlinking models with existing GT only (%d train, %d val)",
            len(train_ok),
            len(val_ok),
        )

    train_paths, val_paths = create_experiment_symlinks(
        experiment_dir,
        train_ok,
        val_ok,
        use_symlinks=not args.no_symlinks,
    )

    write_experiment_manifest(
        experiment_dir,
        renderer,
        train_ok,
        val_ok,
        train_paths,
        val_paths,
        args.render_height,
        args.render_width,
        args.gt_view_layout,
        args.camera_distance,
        categories,
        args.val_fraction,
        args.seed,
    )

    print("\n✓ Experiment ready.")
    print(f"  Directory: {experiment_dir}")
    print(f"  Train:     {experiment_dir / 'train'}  ({len(train_paths)} symlinks, GT required)")
    if not args.precache_train_only:
        print(f"  Val:       {experiment_dir / 'val'}  ({len(val_paths)} symlinks)")
    print("\nTraining:")
    print(f"  python train_gs_ae.py \\")
    print(f"    --data_dir {experiment_dir / 'train'} \\")
    if not args.precache_train_only and val_paths:
        print(f"    --val_dir {experiment_dir / 'val'} \\")
    print(f"    --gt_view_layout {args.gt_view_layout} \\")
    print(f"    --render_height {args.render_height} --render_width {args.render_width} \\")
    print(f"    --num_views <N> --output_dir runs/<run_name> ...")


if __name__ == "__main__":
    main()
