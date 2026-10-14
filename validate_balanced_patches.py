#!/usr/bin/env python3
"""Cheap immutable data preflight before allocating a full training run."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from hy3dgen.shapegen.pc_patches import validate_partition
from hy3dgen.shapegen.room_view_pools import load_room_view_pools


def validate(root, view_map, *, total=32768, size=16):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    rooms = load_room_view_pools(view_map)
    if set(manifest["rooms"]) != set(rooms) or manifest["total"] != total or manifest["patch_size"] != size:
        raise ValueError("Prepared manifest differs from configured room/point budget")
    for room in rooms:
        path = root / room / "surface.npz"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != manifest["rooms"][room]["surface_sha256"]:
            raise ValueError(f"Prepared surface was modified: {room}")
        with np.load(path) as data:
            xyz = data["xyz"]
            if xyz.dtype != np.float32 or xyz.shape != (total, 3) or len(np.unique(xyz, axis=0)) != total:
                raise ValueError(f"Invalid distinct FP32 geometry: {room}")
            if not all(np.isfinite(data[k]).all() for k in ("xyz", "normals", "rgb", "sharp")):
                raise ValueError(f"Nonfinite prepared inputs: {room}")
        for kind in ("geometry", "features"):
            if "patch_sha256" in manifest["rooms"][room]:
                digest = hashlib.sha256((root / room / f"patches_{kind}.npz").read_bytes()).hexdigest()
                if digest != manifest["rooms"][room]["patch_sha256"][kind]:
                    raise ValueError(f"Prepared patch membership was modified: {room} {kind}")
            with np.load(root / room / f"patches_{kind}.npz") as data:
                validate_partition(data["indices"], data["queries"], total)
                if data["indices"].shape != (total // size, size):
                    raise ValueError("Patch size mismatch")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--prepared_surface_root", required=True)
    p.add_argument("--room_view_pools_json", default="configs/internscenes_bath10_selected_views.json")
    args = p.parse_args()
    result = validate(args.prepared_surface_root, args.room_view_pools_json)
    print(f"Validated {len(result['rooms'])} distinct surfaces, both disjoint patch variants; sampling={result['sampling']}")
