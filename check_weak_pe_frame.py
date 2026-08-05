#!/usr/bin/env python3
"""Verify the depth-unprojection convention used for weak-context 3D PE.

Unprojects each render's depth map with several sign/axis conventions, maps the
points into the render world frame with the view's ``c2w``, and reports the mean
distance to the mesh surface point cloud. The convention that matches the
surface (distance ~ 0) is the correct one for ``pe_frame='object'``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from hy3dgen.shapegen.pc_render_dataset import build_surface_render_dataset
from hy3dgen.shapegen.vggt_context import ray_distance_to_z, unproject_depth_to_camera
from train_gs_ae import load_experiment_manifest, resolve_category_ids

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONVENTIONS = {
    "xyz": (1.0, 1.0, 1.0),
    "x-yz": (1.0, -1.0, 1.0),
    "x-y-z": (1.0, -1.0, -1.0),
    "xy-z": (1.0, 1.0, -1.0),
    "-xyz": (-1.0, 1.0, 1.0),
    "-x-yz": (-1.0, -1.0, 1.0),
}


def one_sided_distance(pts: np.ndarray, surface: np.ndarray, max_pts: int = 4096) -> float:
    """Mean nearest-neighbour distance from ``pts`` to ``surface``."""
    if pts.shape[0] == 0:
        return float("nan")
    rng = np.random.default_rng(0)
    if pts.shape[0] > max_pts:
        pts = pts[rng.choice(pts.shape[0], max_pts, replace=False)]
    a = torch.from_numpy(pts.astype(np.float32))
    b = torch.from_numpy(surface.astype(np.float32))
    d = torch.cdist(a.unsqueeze(0), b.unsqueeze(0)).squeeze(0)
    return float(d.min(dim=1).values.mean())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--gobjaverse_render_root", default=None)
    p.add_argument("--max_items", type=int, default=2)
    p.add_argument("--view_idx", type=int, default=0)
    p.add_argument("--categories", default=None)
    args = p.parse_args()

    data_path = Path(args.data_dir).resolve()
    manifest = load_experiment_manifest(str(data_path))
    dataset = build_surface_render_dataset(
        str(data_path),
        max_items=args.max_items,
        categories=resolve_category_ids(args.categories),
        manifest=manifest,
        render_root=args.gobjaverse_render_root,
        view_idx=args.view_idx,
    )

    results = {k: [] for k in CONVENTIONS}
    results_raydist = {k: [] for k in CONVENTIONS}
    for i in range(len(dataset)):
        item = dataset[i]
        if "depth" not in item:
            logger.warning("No render for item %d", i)
            continue
        surface = item["surface"][:, :3].numpy()
        depth = item["depth"].numpy()
        fx, fy, cx, cy = item["intrinsics"].tolist()
        c2w = item["c2w"].numpy().astype(np.float64)
        rot, origin = c2w[:3, :3], c2w[:3, 3]
        mask = depth > 0

        logger.info(
            "item %d: surface bbox=[%.2f, %.2f], depth range=[%.2f, %.2f], cam dist=%.2f",
            i,
            surface.min(),
            surface.max(),
            depth[mask].min(),
            depth[mask].max(),
            float(np.linalg.norm(origin)),
        )

        for use_ray in (True, False):
            cam = unproject_depth_to_camera(
                depth, fx=fx, fy=fy, cx=cx, cy=cy, depth_is_ray_distance=use_ray
            )
            pts_cam = cam[mask]
            for name, signs in CONVENTIONS.items():
                q = pts_cam * np.asarray(signs, dtype=np.float32)
                world = q.astype(np.float64) @ rot.T + origin[None, :]
                d = one_sided_distance(world, surface)
                (results if use_ray else results_raydist)[name].append(d)

    print("\nmean surface distance (lower is better; ~0.01 means aligned)")
    print(f"{'convention':<10} {'ray_dist->z':>14} {'raw as z':>12}")
    for name in CONVENTIONS:
        a = float(np.mean(results[name])) if results[name] else float("nan")
        b = float(np.mean(results_raydist[name])) if results_raydist[name] else float("nan")
        print(f"{name:<10} {a:>14.4f} {b:>12.4f}")

    best = min(
        ((np.mean(v), k, "ray_dist->z") for k, v in results.items() if v),
        default=None,
    )
    best_raw = min(
        ((np.mean(v), k, "raw as z") for k, v in results_raydist.items() if v),
        default=None,
    )
    for cand in (best, best_raw):
        if cand is not None:
            print(f"best ({cand[2]}): {cand[1]} at {cand[0]:.4f}")


if __name__ == "__main__":
    main()
