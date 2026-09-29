#!/usr/bin/env python3
"""Run exp14 ShapePCUnite generation on prepared comparison objects.

Must run in ``hy3dgs`` from HY3DGS repo root.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from compare_baselines.ply_io import write_ply
from evaluate_pc_unite import load_unite_model
from hy3dgen.shapegen.pc_render_dataset import (
    build_surface_render_dataset,
    collate_surface_render,
)
from train_gs_ae import load_experiment_manifest
from train_pc_unite import _build_weak_context

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("infer_exp14")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compare_dir", required=True, help="prepare_inputs output dir")
    p.add_argument(
        "--ckpt",
        default="runs/exp14_n100_mv2_joint_pool_4_10_16_22/ckpt_0030000.pt",
    )
    p.add_argument(
        "--data_dir",
        default=None,
        help="Override data_dir (default: from manifest split path)",
    )
    p.add_argument(
        "--gobjaverse_render_root",
        default="/export/home/nathan/datasets",
    )
    p.add_argument(
        "--vggt_cache_root",
        default="runs/vggt_cache/furn100_joint_pool_4_10_16_22",
    )
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    compare_dir = Path(args.compare_dir).resolve()
    manifest = json.loads((compare_dir / "manifest.json").read_text())
    data_dir = args.data_dir or manifest["data_dir"]
    views = manifest["view_indices"]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    model, vggt_builder, train_args = load_unite_model(args.ckpt, device)
    model.eval()
    vggt_builder.eval()

    ds_manifest = load_experiment_manifest(str(data_dir))
    dataset = build_surface_render_dataset(
        str(data_dir),
        max_items=None,
        use_experiment_manifest=True,
        manifest=ds_manifest,
        render_root=args.gobjaverse_render_root,
        view_indices=views,
        views_per_sample=len(views),
        view_sample_mode="first",
        vggt_cache_root=args.vggt_cache_root,
        use_joint_vggt_cache=True,
        pc_size=int(train_args.get("pc_size", 5120)),
        pc_sharpedge_size=int(train_args.get("pc_sharpedge_size", 5120)),
        gobjaverse_normalization=True,
        surface_in_camera_frame=True,
        align_mode=manifest.get("align_mode", "c_meanrms"),
        filter_missing_views=True,
        seed=args.seed,
    )
    # Map mesh stem -> obj dir from prepare manifest
    by_uid = {o["uid"]: o for o in manifest["objects"]}

    # Only evaluate objects present in the compare manifest (subsets / smoke runs).
    kept_meshes = [p for p in dataset.mesh_paths if Path(p).stem in by_uid]
    if not kept_meshes:
        raise SystemExit("No overlap between compare manifest and dataset meshes")
    dataset.mesh_paths = kept_meshes
    if dataset.views_per_sample > 1:
        dataset.samples = [(p, -1) for p in kept_meshes]
    else:
        dataset.samples = [(p, int(dataset.view_indices[0])) for p in kept_meshes]
    logger.info("exp14 will run on %d compare objects", len(kept_meshes))

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_surface_render,
    )

    with torch.no_grad():
        for batch in loader:
            mesh = batch["mesh_path"][0]
            uid = Path(mesh).stem
            entry = by_uid[uid]
            obj_dir = compare_dir / entry["obj_dir"]
            raw_dir = obj_dir / "pred_raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            out_ply = raw_dir / "exp14.ply"
            if out_ply.exists():
                logger.info("exists, skip %s", out_ply)
                continue

            weak, _, _ = _build_weak_context(
                batch,
                vggt_builder,
                device,
                null_weak_context=None,
                use_null=False,
                align_mode=manifest.get("align_mode", "c_meanrms"),
            )
            z = model.sample_latents(
                weak,
                batch_size=1,
                num_steps=args.sample_steps,
                guidance_scale=3.0,
            )
            xyz, rgb, _ = model.decode(z, representation_phase=False)
            xyz_np = xyz[0].detach().cpu().numpy()
            rgb_np = None if rgb is None else rgb[0].detach().cpu().numpy()
            write_ply(out_ply, xyz_np, colors=rgb_np, rgb=(220, 64, 64))
            logger.info("Wrote %s (%d pts)", out_ply, xyz_np.shape[0])

    logger.info("exp14 inference done")


if __name__ == "__main__":
    main()
