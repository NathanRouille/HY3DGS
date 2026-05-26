"""GT RGBD on-disk cache paths and validation (shared by prepare + train)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

# Reject empty/truncated caches; full v46 @256 fp16 is ~20–50 MB.
MIN_GT_CACHE_BYTES: int = 1_000_000


def canonical_obj_path(mesh_path: str) -> str:
    """Resolved absolute path to ``model_normalized.obj`` (ShapeNetCore location)."""
    return os.path.realpath(mesh_path)


def gt_cache_file_path(obj_path: str, gt_tag: str) -> str:
    """Path to the ``.pt`` cache next to the canonical OBJ (under ShapeNetCore)."""
    return f"{canonical_obj_path(obj_path)}.gt_rgbd_{gt_tag}.pt"


def model_folder_name(model: Dict) -> str:
    return f"{model['category']}_{model['model_id']}"


def is_usable_gt_cache_file(
    cache_path: str,
    min_bytes: int = MIN_GT_CACHE_BYTES,
) -> bool:
    if not os.path.isfile(cache_path):
        return False
    try:
        if os.path.getsize(cache_path) < min_bytes:
            return False
    except OSError:
        return False
    return True


def experiment_manifest_path(experiment_dir: Path) -> Path:
    return experiment_dir / "manifest.json"


def symlink_obj_path(experiment_dir: Path, split: str, model: Dict) -> Path:
    """Path to ``model_normalized.obj`` inside the experiment symlink tree."""
    return experiment_dir / split / model_folder_name(model) / "model_normalized.obj"
