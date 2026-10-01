#!/usr/bin/env python3
"""Rewrite InternScenes surface.npz xyz from unit-box -> world (invert normalize_parts).

Does NOT re-sample. Reads GLB only to recover bbox center/scale_.
Idempotent: skips rooms whose meta.json has ``frame: world``.

Example (full gen set)::

  conda activate hy3dgs
  cd ~/Documents/research/HY3DGS
  export PYTHONPATH="$PWD:$PYTHONPATH"
  python unnormalize_internscenes_npz_world.py \\
    --rooms_root ~/Documents/research/internscenes_gen_pc_v0/rooms \\
    --mesh_root /mnt/hdd2/ismail/InternScenes/processed/glb_output/sample_scenes \\
    --workers 8

Then the bathroom pack::

  python unnormalize_internscenes_npz_world.py \\
    --rooms_root ~/Documents/research/internscenes_bathroom_130/rooms \\
    --workers 4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

logger = logging.getLogger("unnormalize_world")

# Must match hy3dgen.shapegen.surface_loaders.normalize_parts default ``scale``.
NORM_EPS = 0.9999


def bbox_center_scale(mesh_path: Path) -> Tuple[np.ndarray, float]:
    """Same center / scale_ as surface_loaders.normalize_parts."""
    import trimesh
    from hy3dgen.shapegen.surface_loaders import parts_to_trimesh, scene_to_parts

    mesh = trimesh.load(str(mesh_path), force=None)
    parts = scene_to_parts(mesh)
    if not parts:
        raise RuntimeError(f"no parts: {mesh_path}")
    unified, _ = parts_to_trimesh(parts)
    bbox = unified.bounds  # (2, 3)
    center = ((bbox[1] + bbox[0]) / 2.0).astype(np.float64)
    scale_ = float((bbox[1] - bbox[0]).max())
    if scale_ <= 0:
        raise RuntimeError(f"degenerate bbox: {mesh_path}")
    return center, scale_


def unnormalize_xyz(xyz: np.ndarray, center: np.ndarray, scale_: float) -> np.ndarray:
    """Invert normalize_parts: p_norm = (p_world - center) * (2*NORM_EPS/scale_)."""
    factor = scale_ / (2.0 * NORM_EPS)
    out = xyz.astype(np.float64) * factor + center.reshape(1, 3)
    return out.astype(np.float32)


def process_one(
    room_dir: Path,
    mesh_root: Path,
    dry_run: bool,
    overwrite_world: bool,
) -> Tuple[str, str]:
    scene_id = room_dir.name
    npz_path = room_dir / "surface.npz"
    meta_path = room_dir / "meta.json"
    if not npz_path.is_file():
        return scene_id, "SKIP_no_npz"

    meta: Dict[str, Any] = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("frame") == "world" and not overwrite_world:
        return scene_id, "SKIP_already_world"

    mesh_path = mesh_root / scene_id / "scene_split.glb"
    if not mesh_path.is_file():
        alt = meta.get("mesh")
        if alt and Path(alt).is_file():
            mesh_path = Path(alt)
        else:
            return scene_id, f"FAIL_no_mesh:{mesh_path}"

    try:
        center, scale_ = bbox_center_scale(mesh_path)
    except Exception as exc:
        return scene_id, f"FAIL_bbox:{exc}"

    try:
        data = dict(np.load(npz_path, allow_pickle=False))
        if "xyz" not in data:
            return scene_id, "FAIL_no_xyz"
        xyz_old = np.asarray(data["xyz"])
        xyz_new = unnormalize_xyz(xyz_old, center, scale_)
        data["xyz"] = xyz_new

        if dry_run:
            return scene_id, (
                f"DRY center={center.tolist()} scale_={scale_:.6f} "
                f"xyz_absmax {float(np.abs(xyz_old).max()):.4f} -> "
                f"{float(np.abs(xyz_new).max()):.4f}"
            )

        # Must end with ".npz" or numpy appends another ".npz" (→ *.tmp.npz orphans).
        tmp = npz_path.with_name(npz_path.stem + ".tmp.npz")
        np.savez_compressed(tmp, **data)
        os.replace(tmp, npz_path)

        meta.update(
            {
                "frame": "world",
                "normalize_parts_inverted": True,
                "normalize_center": center.astype(float).tolist(),
                "normalize_scale_": float(scale_),
                "normalize_eps": float(NORM_EPS),
                "mesh_for_unnormalize": str(mesh_path.resolve()),
            }
        )
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        return scene_id, "OK"
    except PermissionError as exc:
        return scene_id, f"FAIL_perm:{exc}"
    except Exception as exc:
        return scene_id, f"FAIL:{type(exc).__name__}:{exc}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--rooms_root", type=Path, required=True)
    p.add_argument(
        "--mesh_root",
        type=Path,
        default=Path(
            "/mnt/hdd2/ismail/InternScenes/processed/glb_output/sample_scenes"
        ),
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--limit", type=int, default=0, help="Process only first N rooms (0=all)"
    )
    p.add_argument("--dry_run", action="store_true")
    p.add_argument(
        "--overwrite_world",
        action="store_true",
        help="Redo even if meta frame=world",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    rooms_root = args.rooms_root.expanduser().resolve()
    mesh_root = args.mesh_root.expanduser().resolve()
    if not rooms_root.is_dir():
        logger.error("rooms_root missing: %s", rooms_root)
        return 2

    room_dirs = sorted(d for d in rooms_root.iterdir() if d.is_dir())
    if args.limit > 0:
        room_dirs = room_dirs[: args.limit]
    logger.info(
        "rooms=%d workers=%d dry_run=%s mesh_root=%s",
        len(room_dirs),
        args.workers,
        args.dry_run,
        mesh_root,
    )

    ok = skip = fail = 0

    def _handle(sid: str, msg: str) -> None:
        nonlocal ok, skip, fail
        if msg.startswith("OK") or msg.startswith("DRY"):
            ok += 1
            if args.verbose or msg.startswith("DRY"):
                logger.info("%s %s", sid, msg)
            elif ok % 500 == 0:
                logger.info("progress ok=%d skip=%d fail=%d", ok, skip, fail)
        elif msg.startswith("SKIP"):
            skip += 1
        else:
            fail += 1
            logger.error("%s %s", sid, msg)

    if args.workers <= 1:
        for rd in room_dirs:
            _handle(*process_one(rd, mesh_root, args.dry_run, args.overwrite_world))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {
                ex.submit(
                    process_one, rd, mesh_root, args.dry_run, args.overwrite_world
                ): rd
                for rd in room_dirs
            }
            for fut in as_completed(futs):
                _handle(*fut.result())

    logger.info("DONE ok=%d skip=%d fail=%d", ok, skip, fail)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
