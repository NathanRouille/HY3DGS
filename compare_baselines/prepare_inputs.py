#!/usr/bin/env python3
"""Prepare shared comparison inputs: letterboxed views + GT surface PLY.

Run inside the ``hy3dgs`` conda env from the HY3DGS repo root:

    python compare_baselines/prepare_inputs.py \\
      --split val --output_dir runs/compare_exp14_baselines/val
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# HY3DGS repo root on path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from compare_baselines.letterbox import letterbox_rgb, make_view_sheet
from compare_baselines.ply_io import write_ply
from hy3dgen.shapegen.pc_render_dataset import build_surface_render_dataset
from train_gs_ae import load_experiment_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prepare_inputs")


def _tensor_chw_to_pil(rgb: torch.Tensor) -> Image.Image:
    arr = rgb.detach().cpu().float().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255.0).astype(np.uint8), mode="RGB")


def _uid_from_mesh(mesh_path: str) -> str:
    return Path(mesh_path).stem


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=("val", "train"), default="val")
    p.add_argument(
        "--data_root",
        default="/export/home/nathan/datasets/gobjaverse_experiments/furniture_351",
    )
    p.add_argument(
        "--gobjaverse_render_root",
        default="/export/home/nathan/datasets",
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument("--view_indices", default="4,16")
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--align_mode", default="c_meanrms")
    p.add_argument("--pc_size", type=int, default=5120)
    p.add_argument("--pc_sharpedge_size", type=int, default=5120)
    p.add_argument("--nova_w", type=int, default=518)
    p.add_argument("--nova_h", type=int, default=392)
    p.add_argument("--surflo_w", type=int, default=518)
    p.add_argument("--surflo_h", type=int, default=392)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    views = [int(x) for x in args.view_indices.split(",") if x.strip()]
    if len(views) < 2:
        raise SystemExit("Need at least 2 view_indices (e.g. 4,16)")

    data_dir = Path(args.data_root) / args.split
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = load_experiment_manifest(str(data_dir))
    dataset = build_surface_render_dataset(
        str(data_dir),
        max_items=args.max_items,
        use_experiment_manifest=True,
        manifest=manifest,
        render_root=args.gobjaverse_render_root,
        view_indices=views,
        views_per_sample=len(views),
        view_sample_mode="first",
        pc_size=args.pc_size,
        pc_sharpedge_size=args.pc_sharpedge_size,
        gobjaverse_normalization=True,
        surface_in_camera_frame=True,
        align_mode=args.align_mode,
        filter_missing_views=True,
        seed=args.seed,
    )

    index = []
    for i in range(len(dataset)):
        sample = dataset[i]
        mesh = sample["mesh_path"]
        uid = _uid_from_mesh(mesh)
        obj_dir = out_root / "objects" / f"{i:04d}_{uid}"
        views_dir = obj_dir / "views"
        views_dir.mkdir(parents=True, exist_ok=True)

        view_ids = [int(v) for v in sample["view_indices"].tolist()]
        rgb_stack = sample["rgb_views"]  # [S,3,H,W]
        originals = []
        letter_nova = []
        letter_surflo = []
        view_meta = []

        for s, vid in enumerate(view_ids):
            pil = _tensor_chw_to_pil(rgb_stack[s])
            orig_path = views_dir / f"original_{vid:02d}.png"
            pil.save(orig_path)

            nova = letterbox_rgb(pil, target_w=args.nova_w, target_h=args.nova_h)
            surf = letterbox_rgb(pil, target_w=args.surflo_w, target_h=args.surflo_h)
            nova_path = views_dir / f"letterbox_nova3r_{vid:02d}.png"
            surf_path = views_dir / f"letterbox_surflo_{vid:02d}.png"
            nova.save(nova_path)
            surf.save(surf_path)

            originals.append(pil)
            letter_nova.append(nova)
            letter_surflo.append(surf)
            view_meta.append(
                {
                    "view_idx": vid,
                    "original": str(orig_path.relative_to(out_root)),
                    "letterbox_nova3r": str(nova_path.relative_to(out_root)),
                    "letterbox_surflo": str(surf_path.relative_to(out_root)),
                    "original_size": list(pil.size),
                    "nova3r_size": list(nova.size),
                    "surflo_size": list(surf.size),
                }
            )

        sheet = make_view_sheet(
            originals,
            letter_nova,
            [f"view {v}" for v in view_ids],
        )
        sheet_path = views_dir / "qa_original_vs_letterbox_nova3r.png"
        sheet.save(sheet_path)

        # GT surface (camera frame of view 0 in the tuple = views[0])
        surface = sample["surface"]
        gt_xyz = surface[:, :3].detach().cpu().numpy()
        gt_rgb = None
        if surface.shape[-1] >= 6:
            gt_rgb = surface[:, 3:6].detach().cpu().numpy()
        gt_path = obj_dir / "gt.ply"
        write_ply(gt_path, gt_xyz, colors=gt_rgb, rgb=(40, 120, 255))

        # Folders for baselines to read
        nova_img_dir = obj_dir / "inputs_nova3r"
        surf_img_dir = obj_dir / "inputs_surflo"
        nova_img_dir.mkdir(exist_ok=True)
        surf_img_dir.mkdir(exist_ok=True)
        for s, vid in enumerate(view_ids):
            # Stable sorted names 00_, 01_ so Surflo folder order matches view order
            letter_nova[s].save(nova_img_dir / f"{s:02d}_view_{vid:02d}.png")
            letter_surflo[s].save(surf_img_dir / f"{s:02d}_view_{vid:02d}.png")

        entry = {
            "index": i,
            "uid": uid,
            "mesh_path": mesh,
            "obj_dir": str(obj_dir.relative_to(out_root)),
            "view_indices": view_ids,
            "views": view_meta,
            "gt_ply": str(gt_path.relative_to(out_root)),
            "n_gt_points": int(gt_xyz.shape[0]),
            "inputs_nova3r": str(nova_img_dir.relative_to(out_root)),
            "inputs_surflo": str(surf_img_dir.relative_to(out_root)),
            "qa_sheet": str(sheet_path.relative_to(out_root)),
        }
        index.append(entry)
        with open(obj_dir / "meta.json", "w") as f:
            json.dump(entry, f, indent=2)
        logger.info("[%d/%d] prepared %s", i + 1, len(dataset), uid)

    manifest_out = {
        "split": args.split,
        "data_dir": str(data_dir),
        "view_indices": views,
        "align_mode": args.align_mode,
        "nova3r_resolution": [args.nova_w, args.nova_h],
        "surflo_resolution": [args.surflo_w, args.surflo_h],
        "num_objects": len(index),
        "objects": index,
    }
    with open(out_root / "manifest.json", "w") as f:
        json.dump(manifest_out, f, indent=2)
    logger.info("Wrote %s (%d objects)", out_root / "manifest.json", len(index))


if __name__ == "__main__":
    main()
