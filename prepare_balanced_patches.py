#!/usr/bin/env python3
"""Prepare immutable 32k distinct surfaces and disjoint patches, never edit a pack.

Preferred: --sampling mesh_fp32 --mesh_root /path/to/sample_scenes.
Explicit fallback: --sampling distinct_source. This preserves available detail
samples, but fills a sharp deficit from uniform samples and records the change.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from hy3dgen.shapegen.pc_patches import balanced_partition, validate_partition
from hy3dgen.shapegen.room_view_pools import load_room_view_pools


def distinct_rows(surface):
    _, first = np.unique(surface[:, :3], axis=0, return_index=True)
    return surface[np.sort(first)]


def choose_surface(surface, total, sharp_target, rng, *, allow_deficit):
    # Prefer sharp attributes when a coordinate appears in both pools.
    sharp = distinct_rows(surface[surface[:, 6] > .5])
    uniform = distinct_rows(surface[surface[:, 6] <= .5])
    take_s = min(sharp_target, len(sharp))
    if take_s != sharp_target and not allow_deficit:
        raise ValueError(f"Only {len(sharp)} distinct sharp samples; need {sharp_target}")
    sharp = sharp[rng.choice(len(sharp), take_s, replace=False)]
    # Exclude cross-pool duplicates without approximate rounding.
    combined = np.concatenate([sharp[:, :3], uniform[:, :3]])
    _, first = np.unique(combined, axis=0, return_index=True)
    keep_u = first[first >= len(sharp)] - len(sharp)
    uniform = uniform[np.sort(keep_u)]
    take_u = total - take_s
    if len(uniform) < take_u:
        raise ValueError(f"Only {len(uniform)} distinct uniform samples; need {take_u}")
    uniform = uniform[rng.choice(len(uniform), take_u, replace=False)]
    selected = np.concatenate([uniform, sharp]).astype(np.float32)
    if len(np.unique(selected[:, :3], axis=0)) != total or not np.isfinite(selected).all():
        raise ValueError("Prepared geometry must be finite and distinct in FP32")
    return selected, {"n_uniform": take_u, "n_sharp": take_s,
                      "sharp_target": sharp_target, "sharp_deficit": sharp_target - take_s}


def mesh_surface(mesh_path, total, sharp_target, seed):
    import trimesh
    from hy3dgen.shapegen.surface_loaders import load_surface_sharpedge_rgb, scene_to_parts
    mesh = trimesh.load(str(mesh_path), process=False)
    parts = scene_to_parts(mesh)
    low = np.min([p.bounds[0] for p in parts], 0)
    high = np.max([p.bounds[1] for p in parts], 0)
    center, extent = (low + high) / 2, float((high - low).max())
    if extent <= 0:
        raise ValueError("Degenerate mesh bounds")
    bank = []
    for attempt in range(4):
        sampled, _ = load_surface_sharpedge_rgb(mesh, num_points=total,
            num_sharp_points=max(total, sharp_target), seed=(seed + attempt) % 2**32,
            include_sharp_label=True, sample_dtype=np.float32)
        points = sampled[0].numpy()
        # Undo precisely the common normalize_parts unit-box mapping, keeping
        # original GLB world axes. Dataset applies the existing GLB→trajectory map.
        points[:, :3] = points[:, :3].astype(np.float64) * extent / (2 * .9999) + center
        bank.append(points)
        try:
            return choose_surface(np.concatenate(bank), total, sharp_target,
                                  np.random.default_rng(seed), allow_deficit=False)
        except ValueError:
            if attempt == 3:
                raise


def write_preview(path, xyz, rgb):
    from hy3dgen.shapegen.gs_export import export_xyz_pointcloud_ply
    import torch
    export_xyz_pointcloud_ply(torch.from_numpy(xyz.astype(np.float32)), path,
                             colors=torch.from_numpy(rgb.astype(np.float32)))


def prepare(args):
    pools = load_room_view_pools(args.room_view_pools_json)
    root = Path(args.output_dir)
    manifest = {"version": 1, "sampling": args.sampling, "pack": str(Path(args.data_dir).resolve()),
                "seed": args.seed, "total": args.total, "sharp_target": args.sharp_target,
                "patch_size": args.patch_size, "rooms": {}}
    for room in pools:
        dest = root / room
        if dest.exists():
            raise FileExistsError(f"Immutable preparation exists: {dest}; choose a fresh output_dir")
        source = Path(args.data_dir) / "rooms" / room
        seed = int(hashlib.sha256(f"{args.seed}:{room}".encode()).hexdigest()[:8], 16)
        if args.sampling == "mesh_fp32":
            if not args.mesh_root:
                raise ValueError("mesh_fp32 requires --mesh_root")
            mesh = Path(args.mesh_root) / room / "scene_split.glb"
            if not mesh.is_file():
                raise FileNotFoundError(mesh)
            surface, sampling = mesh_surface(mesh, args.total, args.sharp_target, seed)
            sampling["mesh"] = str(mesh)
        else:
            if (source / "meta.json").is_file():
                meta = json.loads((source / "meta.json").read_text())
                if meta.get("frame") != "world":
                    raise ValueError(f"{room}: source must be restored to world coordinates before preparation")
            with np.load(source / "surface.npz") as pack:
                rgb = pack["rgb"].astype(np.float32)
                if rgb.max() > 1.5:
                    rgb /= 255
                raw = np.concatenate([pack["xyz"], pack["normals"],
                    pack["sharp"].reshape(-1, 1), rgb], 1).astype(np.float32)
            surface, sampling = choose_surface(raw, args.total, args.sharp_target,
                np.random.default_rng(seed), allow_deficit=True)
            sampling["source_distinct_xyz"] = len(np.unique(raw[:, :3], axis=0))
            sampling["notice"] = "Existing quantization is not undone; sharp deficit filled with uniform"
        geometry_indices, geometry_queries, geometry_stats = balanced_partition(
            surface[:, :3], surface[:, 7:10], surface[:, 3:6], size=args.patch_size,
            passes=args.passes, color_weight=0, normal_weight=0)
        feature_indices, feature_queries, feature_stats = balanced_partition(
            surface[:, :3], surface[:, 7:10], surface[:, 3:6], size=args.patch_size,
            passes=args.passes, color_weight=args.color_weight, normal_weight=args.normal_weight)
        dest.mkdir(parents=True)
        np.savez_compressed(dest / "surface.npz", xyz=surface[:, :3], normals=surface[:, 3:6],
                            sharp=surface[:, 6], rgb=surface[:, 7:10])
        patch_hashes = {}
        for kind, indices, queries, stats in (
                ("geometry", geometry_indices, geometry_queries, geometry_stats),
                ("features", feature_indices, feature_queries, feature_stats)):
            validate_partition(indices, queries, args.total)
            np.savez_compressed(dest / f"patches_{kind}.npz", indices=indices, queries=queries)
            patch_hashes[kind] = hashlib.sha256((dest / f"patches_{kind}.npz").read_bytes()).hexdigest()
            colors = np.random.default_rng(seed).random((len(indices), 3)).astype(np.float32)
            point_colors = np.empty((args.total, 3), dtype=np.float32)
            point_colors[indices] = colors[:, None]
            write_preview(dest / f"patches_{kind}.ply", surface[:, :3], point_colors)
            write_preview(dest / f"queries_{kind}.ply", surface[queries, :3], surface[queries, 7:10])
            (dest / f"stats_{kind}.json").write_text(json.dumps(stats, indent=2))
        write_preview(dest / "gt.ply", surface[:, :3], surface[:, 7:10])
        digest = hashlib.sha256((dest / "surface.npz").read_bytes()).hexdigest()
        manifest["rooms"][room] = {**sampling, "surface_sha256": digest, "patch_sha256": patch_hashes,
            "geometry": geometry_stats, "features": feature_stats}
        print(json.dumps({"room": room, **sampling}), flush=True)
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Prepared {len(pools)} rooms: {root}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--room_view_pools_json", default="configs/internscenes_bath10_selected_views.json")
    p.add_argument("--sampling", choices=("mesh_fp32", "distinct_source"), required=True)
    p.add_argument("--mesh_root")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total", type=int, default=32768)
    p.add_argument("--sharp_target", type=int, default=16384)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--passes", type=int, default=4)
    p.add_argument("--color_weight", type=float, default=.05)
    p.add_argument("--normal_weight", type=float, default=.05)
    prepare(p.parse_args())
