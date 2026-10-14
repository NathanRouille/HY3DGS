#!/usr/bin/env python3
"""Re-evaluate a joint checkpoint over all selected ordered image pairs."""
import argparse

import torch

from evaluate_pc_unite import load_unite_model
from train_pc_unite import _build_weak_context
from hy3dgen.shapegen.pc_joint_eval import evaluate_joint_pairs
from hy3dgen.shapegen.pc_recon_runtime import PooledSurfaceDataset
from hy3dgen.shapegen.pc_render_dataset import build_internscenes_render_dataset
from hy3dgen.shapegen.room_view_pools import load_room_view_pools


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    for name in ("ckpt", "data_dir", "room_view_pools_json", "vggt_cache_root", "output_dir"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--prepared_surface_root", default=None)
    p.add_argument("--patch_variant", choices=["geometry", "features"], default=None)
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--gen_seeds", nargs="+", type=int, default=[0, 1, 2])
    args = p.parse_args()
    model, builder, saved = load_unite_model(args.ckpt, torch.device(args.device))
    if model.tokenizer_use_weak_context or model.use_surflo_global_cam:
        raise ValueError("Joint evaluator requires an unconditioned tokenizer and no Surflo camera")
    source = build_internscenes_render_dataset(args.data_dir,
        room_view_pools=load_room_view_pools(args.room_view_pools_json),
        views_per_sample=2, view_sample_mode="random", seed=saved.get("seed", 0),
        align_mode="c_meanrms", strict_load=True, include_sharp_label=True,
        pc_size=saved["pc_size"], pc_sharpedge_size=saved["pc_sharpedge_size"],
        prepared_surface_root=args.prepared_surface_root or saved.get("prepared_surface_root"),
        patch_variant=args.patch_variant or saved.get("patch_variant", "features"),
        vggt_cache_root=args.vggt_cache_root, use_joint_vggt_cache=True)
    pooled = PooledSurfaceDataset(source, keep_context=True)
    evaluate_joint_pairs(model, builder, pooled.items, _build_weak_context,
        output_dir=args.output_dir, step=torch.load(args.ckpt, map_location="cpu", weights_only=False, mmap=True)["step"],
        sample_steps=args.sample_steps, gen_seeds=args.gen_seeds, export_ply=True)
