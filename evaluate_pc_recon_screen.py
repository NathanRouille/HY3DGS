#!/usr/bin/env python3
"""Fast reconstruction-only evaluation: no VGGT, flow or dense CD matrices."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from evaluate_pc_unite import load_unite_model
from hy3dgen.shapegen.gs_export import export_xyz_pointcloud_ply
from hy3dgen.shapegen.models.autoencoders.shape_pc_ae import ShapePCAE
from hy3dgen.shapegen.pc_render_dataset import build_internscenes_render_dataset
from hy3dgen.shapegen.room_view_pools import load_room_view_pools


@torch.no_grad()
def evaluate(args):
    device = torch.device(args.device)
    model, builder, saved = load_unite_model(args.ckpt, device)
    if model.tokenizer_use_weak_context:
        raise ValueError("This evaluator requires tokenizer_use_weak_context=False")
    del builder
    model.encoder.query_cache_max_items = 3
    pools = load_room_view_pools(args.room_view_pools_json)
    dataset = build_internscenes_render_dataset(
        args.data_dir, room_view_pools=pools, views_per_sample=2,
        view_sample_mode="first", seed=0, include_sharp_label=True,
        pc_size=int(saved["pc_size"]), pc_sharpedge_size=int(saved["pc_sharpedge_size"]),
        align_mode="c_meanrms", strict_load=True,
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(len(dataset)):
        item = dataset[index]
        torch.manual_seed(0)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(0)
        surface = item["surface"][None].to(device)
        _, fps, xyz, rgb, anchors = model.forward_tokenizer(surface)
        gt, gt_rgb = ShapePCAE.surface_gt_points(surface, include_sharp_label=True)
        p, q = xyz[0].cpu().numpy(), gt[0].cpu().numpy()
        dp, ip = cKDTree(q).query(p)
        dq, iq = cKDTree(p).query(q)
        record = {"room": Path(item["mesh_path"]).name,
                  "views": item["view_indices"].tolist(),
                  "cd_pred2gt": float(np.mean(dp**2)), "cd_gt2pred": float(np.mean(dq**2)),
                  "cd": float(np.mean(dp**2) + np.mean(dq**2)),
                  "pred_distance_p95": float(np.quantile(dp, .95)),
                  "gt_distance_p95": float(np.quantile(dq, .95))}
        # Normalized thresholds are labelled explicitly; room scale varies by views.
        for threshold in (.002, .005, .01):
            record[f"precision_norm_{threshold}"] = float(np.mean(dp < threshold))
            record[f"coverage_norm_{threshold}"] = float(np.mean(dq < threshold))
        if rgb is not None and gt_rgb is not None:
            color, color_gt = rgb[0].cpu().numpy(), gt_rgb[0].cpu().numpy()
            record["rgb_l1"] = float(.5 * (np.abs(color - color_gt[ip]).mean()
                                            + np.abs(color_gt - color[iq]).mean()))
        obj = out / record["room"]
        obj.mkdir(exist_ok=True)
        export_xyz_pointcloud_ply(gt[0], obj / "gt.ply", colors=gt_rgb[0] if gt_rgb is not None else None)
        export_xyz_pointcloud_ply(xyz[0], obj / "recon.ply", colors=rgb[0] if rgb is not None else None)
        export_xyz_pointcloud_ply(fps[0], obj / "fps.ply", rgb=(0, 255, 255))
        export_xyz_pointcloud_ply(anchors[0], obj / "anchors.ply", rgb=(255, 0, 255))
        records.append(record)
    metrics = {"checkpoint": str(args.ckpt), "evaluation_precision": "fp32",
               "coordinate_frame": "normalized first-camera frame", "rooms": records,
               "mean_cd": float(np.mean([r["cd"] for r in records]))}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--room_view_pools_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    evaluate(parser.parse_args())
