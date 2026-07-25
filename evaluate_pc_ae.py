#!/usr/bin/env python3
"""Evaluate ShapePCAE checkpoints: Chamfer / RGB metrics + PLY exports."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch

from hy3dgen.shapegen.gs_export import export_xyz_pointcloud_ply
from hy3dgen.shapegen.models.autoencoders.shape_pc_ae import ShapePCAE
from hy3dgen.shapegen.pc_losses import chamfer_distance, rgb_l1_on_nn, sinkhorn_matching_loss
from hy3dgen.shapegen.pretrained_profiles import resolve_include_sharp_label
from train_pc_ae import SurfaceOnlyDataset
from train_gs_ae import resolve_category_ids

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_model(ckpt_path: str, device: torch.device) -> tuple[ShapePCAE, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_args = ckpt.get("args", {})
    num_latents = int(train_args.get("num_latents", 1024))
    num_registers = train_args.get("num_registers")
    if num_registers is None:
        num_registers = num_latents
    include_sharp = bool(train_args.get("include_sharp_label") or False)
    point_feats = int(train_args.get("point_feats", 7 if include_sharp else 6))

    model = ShapePCAE(
        num_latents=num_latents,
        num_registers=int(num_registers),
        embed_dim=int(train_args.get("embed_dim", 64)),
        width=int(train_args.get("width", 1024)),
        heads=int(train_args.get("heads", 16)),
        num_ge_layers=int(train_args.get("num_ge_layers", 8)),
        num_decoder_layers=int(train_args.get("num_decoder_layers", 4)),
        pc_size=int(train_args.get("pc_size", 5120)),
        pc_sharpedge_size=int(train_args.get("pc_sharpedge_size", 5120)),
        point_feats=point_feats,
        downsample_ratio=int(train_args.get("downsample_ratio", 20)),
        num_points_per_anchor=int(train_args.get("num_points_per_anchor", 8)),
        deterministic_encoder=bool(train_args.get("deterministic_encoder", True)),
        max_anchor_delta=float(train_args.get("max_anchor_delta", 0.1)),
        qk_norm=bool(train_args.get("qk_norm", True)),
        qkv_bias=bool(train_args.get("qkv_bias", True)),
        include_pi=bool(train_args.get("include_pi", True)),
    )
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        logger.warning("Missing keys: %s", missing[:20])
    if unexpected:
        logger.warning("Unexpected keys: %s", unexpected[:20])
    model.to(device).eval()
    return model, train_args


@torch.no_grad()
def evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, train_args = load_model(args.ckpt, device)

    include_sharp_label = train_args.get("include_sharp_label")
    if include_sharp_label is None:
        # fall back via a tiny namespace-like object
        class _A:
            pass
        a = _A()
        a.include_sharp_label = None
        a.pretrained_profile = train_args.get("pretrained_profile", "none")
        include_sharp_label = resolve_include_sharp_label(a)
    include_sharp_label = bool(include_sharp_label)

    categories = resolve_category_ids(args.categories)
    dataset = SurfaceOnlyDataset(
        data_dir=args.data_dir,
        pc_size=int(train_args.get("pc_size", 5120)),
        pc_sharpedge_size=int(train_args.get("pc_sharpedge_size", 5120)),
        max_items=args.max_items,
        categories=categories,
        seed=int(train_args.get("seed", 0)) if train_args.get("seed") is not None else 0,
        include_sharp_label=include_sharp_label,
        use_experiment_manifest=not args.no_experiment_manifest,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    export_dir = out_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    cd_sum = 0.0
    rgb_sum = 0.0
    anc_sum = 0.0
    n = 0

    for i in range(len(dataset)):
        sample = dataset[i]
        surface = sample["surface"].unsqueeze(0).to(device)
        mesh_path = sample["mesh_path"]
        stem = Path(mesh_path).stem
        if stem == "model_normalized":
            stem = Path(mesh_path).parent.name

        xyz, rgb, centers, fps_xyz, _ = model(surface)
        gt_xyz, gt_rgb = ShapePCAE.surface_gt_points(
            surface, include_sharp_label=include_sharp_label
        )
        cd, idx_p2t, idx_t2p = chamfer_distance(xyz, gt_xyz)
        rgb_loss = rgb_l1_on_nn(rgb, gt_rgb, idx_p2t, bidirectional=True, idx_tgt_to_pred=idx_t2p)
        anc = sinkhorn_matching_loss(
            centers,
            fps_xyz,
            epsilon=float(train_args.get("sinkhorn_eps", 0.02)),
            n_iters=int(train_args.get("sinkhorn_iters", 50)),
        )

        cd_sum += float(cd)
        rgb_sum += float(rgb_loss)
        anc_sum += float(anc)
        n += 1
        rows.append(
            {
                "mesh": mesh_path,
                "cd": float(cd),
                "rgb_l1": float(rgb_loss),
                "anc_cd": float(anc),
            }
        )
        logger.info("[%d/%d] %s cd=%.5f rgb=%.5f anc=%.5f", i + 1, len(dataset), stem, cd, rgb_loss, anc)

        if args.export_ply:
            sample_dir = export_dir / f"{i:03d}_{stem}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            export_xyz_pointcloud_ply(xyz[0].cpu(), sample_dir / "pred.ply", colors=rgb[0].cpu())
            export_xyz_pointcloud_ply(gt_xyz[0].cpu(), sample_dir / "gt.ply", colors=gt_rgb[0].cpu())
            export_xyz_pointcloud_ply(centers[0].cpu(), sample_dir / "anchors.ply")
            export_xyz_pointcloud_ply(fps_xyz[0].cpu(), sample_dir / "fps.ply")

        if args.max_eval_items is not None and n >= args.max_eval_items:
            break

    summary = {
        "num_meshes": n,
        "mean_cd": cd_sum / max(n, 1),
        "mean_rgb_l1": rgb_sum / max(n, 1),
        "mean_anc_cd": anc_sum / max(n, 1),
        "ckpt": args.ckpt,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump({"summary": summary, "per_mesh": rows}, f, indent=2)
    logger.info("Summary: %s", summary)
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate ShapePCAE")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="runs/pc_ae_eval")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_items", type=int, default=None, help="Dataset discovery cap")
    p.add_argument("--max_eval_items", type=int, default=None, help="Eval loop cap")
    p.add_argument("--categories", type=str, default=None)
    p.add_argument("--no_experiment_manifest", action="store_true")
    p.add_argument("--export_ply", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
