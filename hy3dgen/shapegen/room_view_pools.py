"""Validated per-room image pools shared by caching, training and evaluation."""

import json
from pathlib import Path
from typing import Dict, List


def normalize_room_view_pools(data: dict) -> Dict[str, List[int]]:
    if not isinstance(data, dict) or not data:
        raise ValueError("Room view pools must be a non-empty JSON object")
    result = {}
    for key, values in data.items():
        room_id = str(key)
        if room_id.isdecimal():
            room_id = f"gen__bathroom__{room_id}"
        if Path(room_id).name != room_id or room_id in (".", ".."):
            raise ValueError(f"Invalid room id: {key!r}")
        if room_id in result:
            raise ValueError(f"Duplicate room id: {room_id}")
        if not isinstance(values, list) or not values:
            raise ValueError(f"{room_id}: expected a non-empty list of view IDs")
        views = []
        for value in values:
            if isinstance(value, bool) or not (
                isinstance(value, int)
                or (isinstance(value, str) and value.isdecimal())
            ):
                raise ValueError(f"{room_id}: invalid view ID {value!r}")
            view = int(value)
            if view < 0 or view in views:
                raise ValueError(f"{room_id}: negative or repeated view ID {value!r}")
            views.append(view)
        result[room_id] = views
    return result


def load_room_view_pools(path: str) -> Dict[str, List[int]]:
    return normalize_room_view_pools(json.loads(Path(path).read_text()))


def pooled_view_indices(pools: Dict[str, List[int]]) -> List[int]:
    """Union for legacy CLI plumbing; sampling always uses each room's pool."""
    return list(dict.fromkeys(v for views in pools.values() for v in views))


def validate_room_view_files(pack_root: str, pools: Dict[str, List[int]]) -> None:
    for room_id, views in pools.items():
        room = Path(pack_root) / "rooms" / room_id
        required = [room / "surface.npz", room / "camera_poses.json"]
        for view in views:
            required.extend([
                room / "rgb" / f"view_{view:03d}.png",
                room / "depth_float32" / f"frame_{view:04d}_depth.npz",
            ])
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{room_id}: missing files {missing}")
        entry = next(iter(json.loads((room / "camera_poses.json").read_text()).values()))
        poses = {int(pose["pose_id"]) for pose in entry["poses"]}
        if not set(views).issubset(poses):
            raise ValueError(f"{room_id}: camera poses missing for {sorted(set(views) - poses)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("view_pools_json")
    parser.add_argument("pack_root")
    args = parser.parse_args()
    pools = load_room_view_pools(args.view_pools_json)
    validate_room_view_files(args.pack_root, pools)
    for room_id, views in pools.items():
        print(f"{room_id}: {views} ({len(views) * (len(views) - 1)} ordered pairs)")
