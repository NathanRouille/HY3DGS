#!/usr/bin/env python3
"""Fixed-GT reconstruction evaluation, including holes and density diagnostics."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_pc_unite import load_unite_model
from hy3dgen.shapegen.gs_export import export_xyz_pointcloud_ply
from hy3dgen.shapegen.models.autoencoders.shape_pc_ae import ShapePCAE
from hy3dgen.shapegen.pc_quality_metrics import reconstruction_metrics
from hy3dgen.shapegen.pc_render_dataset import build_internscenes_render_dataset
from hy3dgen.shapegen.room_view_pools import load_room_view_pools


def error_colors(distance, threshold=.02):
    fraction = np.clip(distance / threshold, 0, 1)
    return np.column_stack([fraction, 1-fraction, np.zeros_like(fraction)])


@torch.no_grad()
def evaluate(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but unavailable")
    model, builder, saved = load_unite_model(args.ckpt, device)
    del builder
    if model.tokenizer_use_weak_context or model.geometry_only:
        raise ValueError("Sanity evaluator requires an unconditioned RGB tokenizer")
    if args.register_noise_mode != "checkpoint":
        model.register_noise_mode = args.register_noise_mode
    if args.modulation_recon_timestep_max is not None:
        model.modulation_recon_timestep_max = args.modulation_recon_timestep_max
    model.eval()
    model.encoder.query_cache_max_items = 32
    pools = load_room_view_pools(args.room_view_pools_json)
    if any(len(pool) < 4 for pool in pools.values()) and args.pair_mode == "paired":
        raise ValueError("Paired evaluation requires at least four selected views per room")
    out = Path(args.output_dir)
    if (out / "metrics.json").exists():
        raise ValueError(f"Evaluation already exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for pair_index in range(1 if args.pair_mode == "primary" else 2):
        pairs = {room: pool[2*pair_index:2*pair_index+2] for room,pool in pools.items()}
        dataset = build_internscenes_render_dataset(args.data_dir, room_view_pools=pairs,
            views_per_sample=2, view_sample_mode="first", seed=0, include_sharp_label=True,
            pc_size=int(saved["pc_size"]), pc_sharpedge_size=int(saved["pc_sharpedge_size"]),
            align_mode="c_meanrms", strict_load=True)
        for index in range(len(dataset)):
            item = dataset[index]
            torch.manual_seed(0)
            surface = item["surface"][None].to(device)
            _, fps, xyz, rgb, anchors = model.forward_tokenizer(surface, representation_phase=False)
            gt, gt_rgb = ShapePCAE.surface_gt_points(surface, include_sharp_label=True)
            metrics, dp, dq = reconstruction_metrics(xyz[0].cpu().numpy(), gt[0].cpu().numpy(),
                rgb[0].cpu().numpy(), gt_rgb[0].cpu().numpy(), cell_size=args.cell_size)
            room = Path(item["mesh_path"]).name
            records.append({"room":room,"pair_index":pair_index,"views":item["view_indices"].tolist(),**metrics})
            if pair_index == 0:
                folder = out / room
                folder.mkdir(exist_ok=True)
                export_xyz_pointcloud_ply(gt[0], folder / "gt.ply", colors=gt_rgb[0])
                export_xyz_pointcloud_ply(xyz[0], folder / "recon.ply", colors=rgb[0])
                export_xyz_pointcloud_ply(fps[0], folder / "fps.ply", rgb=(0,255,255))
                export_xyz_pointcloud_ply(anchors[0], folder / "anchors.ply", rgb=(255,0,255))
                export_xyz_pointcloud_ply(gt[0], folder / "gt_uncovered.ply", colors=error_colors(dq))
                export_xyz_pointcloud_ply(xyz[0], folder / "recon_error.ply", colors=error_colors(dp))
            print(f"Evaluated {room} pair={pair_index} CD={metrics['cd']:.6f}",flush=True)
    summaries = {}
    for pair_index in range(1 if args.pair_mode == "primary" else 2):
        rows = [r for r in records if r["pair_index"] == pair_index]
        summaries["primary" if pair_index == 0 else "alternate"] = {
            k:float(np.mean([r[k] for r in rows])) for k in rows[0] if isinstance(rows[0][k], (int,float)) and k != "pair_index"}
    result = {"checkpoint":str(Path(args.ckpt).resolve()),"precision":"fp32","surface_seed":0,
        "register_noise_mode":model.register_noise_mode,"modulation_recon_timestep_max":model.modulation_recon_timestep_max,
        "coordinate_frame":"normalized first-camera; thresholds/cell_size are normalized units",
        "cell_size":args.cell_size,"error_ply_saturation_distance":.02,"pair_mode":args.pair_mode,
        "training_view_sample_mode":saved.get("view_sample_mode"),
        "summary":summaries,"rooms":records}
    (out / "metrics.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(summaries,indent=2),flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    for key in ("ckpt","data_dir","room_view_pools_json","output_dir"):
        p.add_argument("--"+key,required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--pair_mode",choices=("primary","paired"),default="paired")
    p.add_argument("--register_noise_mode",choices=("checkpoint","random","fixed","zeros"),default="checkpoint")
    p.add_argument("--modulation_recon_timestep_max",type=float)
    p.add_argument("--cell_size",type=float,default=.1)
    evaluate(p.parse_args())
