"""InternScenes room pack I/O (surface.npz + traj RGB/depth/cameras).

Frame conventions match ``export_align_showcase_internscenes.py``:
  - ``surface.npz`` xyz in GLB Y-up world (after unnormalize)
  - Trajectory / cameras in Z-up; remap (x, y, z) -> (x, -z, y)
  - ``cam_matrix_opencv`` is OpenCV cam-to-world
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import torch
from PIL import Image

from hy3dgen.shapegen.surface_loaders import stable_mesh_seed

logger = logging.getLogger(__name__)


def glb_yup_to_traj_zup(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    return np.stack([xyz[:, 0], -xyz[:, 2], xyz[:, 1]], axis=1).astype(np.float32)


def glb_yup_to_traj_zup_normals(nrm: np.ndarray) -> np.ndarray:
    nrm = np.asarray(nrm, dtype=np.float64)
    out = np.stack([nrm[:, 0], -nrm[:, 2], nrm[:, 1]], axis=1)
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-8
    return out.astype(np.float32)


class InternScenesGTSource:
    """GT renders for one room directory under ``pack/rooms/<scene_id>/``."""

    KIND = "internscenes"

    def __init__(self, pack_root: Optional[str] = None):
        self.pack_root = Path(pack_root).resolve() if pack_root else None

    def has_gt(self, room_path: str) -> bool:
        return (Path(room_path) / "surface.npz").is_file()

    def room_dir(self, room_path: str) -> Path:
        return Path(room_path).resolve()

    def list_available_views(
        self, room_path: str, pool: List[int]
    ) -> Set[int]:
        rd = self.room_dir(room_path)
        out: Set[int] = set()
        for vid in pool:
            rgb = rd / "rgb" / f"view_{int(vid):03d}.png"
            depth = rd / "depth_float32" / f"frame_{int(vid):04d}_depth.npz"
            if rgb.is_file() and depth.is_file():
                out.add(int(vid))
        return out


def discover_internscenes_room_paths(
    pack_root: str,
    *,
    split: str = "train",
    room_ids: Optional[List[str]] = None,
    max_items: Optional[int] = None,
) -> List[str]:
    """Return absolute paths to ``pack/rooms/<id>/``."""
    root = Path(pack_root).resolve()
    rooms_root = root / "rooms"
    if not rooms_root.is_dir():
        raise FileNotFoundError(f"Missing rooms/ under {root}")

    ids: List[str] = []
    if room_ids:
        ids = [str(x).strip() for x in room_ids if str(x).strip()]
    else:
        split_path = root / "splits" / f"{split}.txt"
        if not split_path.is_file():
            raise FileNotFoundError(split_path)
        ids = [
            ln.strip()
            for ln in split_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]

    paths: List[str] = []
    for sid in ids:
        rd = rooms_root / sid
        if not (rd / "surface.npz").is_file():
            logger.warning("Skip %s: no surface.npz", sid)
            continue
        paths.append(str(rd))
        if max_items is not None and len(paths) >= int(max_items):
            break
    if not paths:
        raise FileNotFoundError(f"No InternScenes rooms under {rooms_root}")
    return paths


def _load_c2w_opencv(poses_entry: dict, pose_id: int) -> torch.Tensor:
    pose = next(
        p for p in poses_entry["poses"] if int(p["pose_id"]) == int(pose_id)
    )
    m = np.eye(4, dtype=np.float64)
    m[:3, :] = np.asarray(pose["cam_matrix_opencv"], dtype=np.float64)
    return torch.from_numpy(m.astype(np.float32))


def load_internscenes_view(room_path: str, view_idx: int) -> Dict:
    rd = Path(room_path)
    cam_json = json.loads((rd / "camera_poses.json").read_text(encoding="utf-8"))
    entry = next(iter(cam_json.values()))
    intr = entry["intrinsics"]
    fx, fy, cx, cy = (
        float(intr["fx"]),
        float(intr["fy"]),
        float(intr["cx"]),
        float(intr["cy"]),
    )
    K = torch.tensor([fx, fy, cx, cy], dtype=torch.float32)

    rgb_path = rd / "rgb" / f"view_{int(view_idx):03d}.png"
    if not rgb_path.is_file():
        raise FileNotFoundError(rgb_path)
    rgb_u8 = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    rgb = torch.from_numpy(rgb_u8.astype(np.float32) / 255.0).permute(2, 0, 1)

    depth_path = rd / "depth_float32" / f"frame_{int(view_idx):04d}_depth.npz"
    if not depth_path.is_file():
        raise FileNotFoundError(depth_path)
    dpack = np.load(depth_path)
    depth = np.asarray(dpack["depth"], dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    mask = np.asarray(dpack["mask"], dtype=bool)
    if mask.ndim == 3:
        mask = mask[..., 0]

    c2w = _load_c2w_opencv(entry, int(view_idx))
    return {
        "rgb": rgb,
        "depth": torch.from_numpy(depth),
        "depth_mask": torch.from_numpy(mask.astype(np.bool_)),
        "intrinsics": K,
        "c2w": c2w,
        "view_idx": int(view_idx),
    }


class InternScenesNPSurfaceLoader:
    """Subsample fixed-size surface from ``surface.npz`` (traj Z-up world)."""

    def __init__(
        self,
        num_uniform_points: int = 5120,
        num_sharp_points: int = 5120,
        *,
        seed: Optional[int] = None,
        include_sharp_label: bool = False,
    ):
        self.num_uniform_points = int(num_uniform_points)
        self.num_sharp_points = int(num_sharp_points)
        self.seed = seed
        self.include_sharp_label = bool(include_sharp_label)

    def __call__(
        self,
        room_path: str,
        num_uniform_points=None,
        num_sharp_points=None,
        gobjaverse_meta: Optional[dict] = None,
    ) -> torch.Tensor:
        del gobjaverse_meta
        nu = self.num_uniform_points if num_uniform_points is None else num_uniform_points
        ns = self.num_sharp_points if num_sharp_points is None else num_sharp_points
        rd = Path(room_path)
        data = np.load(rd / "surface.npz")
        xyz = glb_yup_to_traj_zup(data["xyz"])
        nrm = glb_yup_to_traj_zup_normals(data["normals"])
        sharp = np.asarray(data["sharp"], dtype=np.float32).reshape(-1)
        rgb = np.asarray(data["rgb"], dtype=np.float32)
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
        if rgb.ndim == 1:
            rgb = rgb.reshape(-1, 3)

        subsample_seed: Optional[int] = None
        if self.seed is not None:
            subsample_seed = stable_mesh_seed(self.seed, room_path)
        rng = np.random.default_rng(subsample_seed)

        sharp_idx = np.flatnonzero(sharp > 0.5)
        uniform_idx = np.flatnonzero(sharp <= 0.5)
        if sharp_idx.size == 0:
            sharp_idx = np.arange(xyz.shape[0])
        if uniform_idx.size == 0:
            uniform_idx = np.arange(xyz.shape[0])

        def _pick(idxs: np.ndarray, n: int) -> np.ndarray:
            if idxs.size >= n:
                return rng.choice(idxs, size=n, replace=False)
            return rng.choice(idxs, size=n, replace=True)

        si = _pick(sharp_idx, int(ns))
        ui = _pick(uniform_idx, int(nu))
        # Hunyuan / PointCrossAttentionEncoder layout: first pc_size = uniform,
        # next pc_sharpedge_size = sharp. Do not shuffle — FPS splits by row range.
        pick = np.concatenate([ui, si], axis=0)

        xyz_p = xyz[pick]
        nrm_p = nrm[pick]
        sharp_p = sharp[pick].reshape(-1, 1)
        rgb_p = rgb[pick]

        if self.include_sharp_label:
            surf = np.concatenate(
                [xyz_p, nrm_p, sharp_p, rgb_p.astype(np.float32)], axis=1
            )
        else:
            surf = np.concatenate([xyz_p, nrm_p, rgb_p.astype(np.float32)], axis=1)
        return torch.from_numpy(surf.astype(np.float32)).unsqueeze(0)
