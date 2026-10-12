#!/usr/bin/env python3
"""Matched ten-room evaluation with exact CPU NN metrics and compact PLY output."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from evaluate_pc_unite import load_unite_model
from train_pc_unite import _build_weak_context
from hy3dgen.shapegen.gs_export import export_xyz_pointcloud_ply
from hy3dgen.shapegen.models.autoencoders.shape_pc_ae import ShapePCAE
from hy3dgen.shapegen.pc_render_dataset import build_internscenes_render_dataset, collate_surface_render
from hy3dgen.shapegen.room_view_pools import load_room_view_pools


def point_metrics(xyz, rgb, gt, gt_rgb):
    p, q = xyz[0].cpu().numpy(), gt[0].cpu().numpy()
    dp, ip = cKDTree(q).query(p)
    dq, iq = cKDTree(p).query(q)
    result = {"cd_pred2gt": float(np.mean(dp**2)), "cd_gt2pred": float(np.mean(dq**2)),
              "cd": float(np.mean(dp**2) + np.mean(dq**2)),
              "pred_distance_p95": float(np.quantile(dp, .95)),
              "gt_distance_p95": float(np.quantile(dq, .95))}
    for threshold in (.002, .005, .01):
        precision, coverage = float(np.mean(dp < threshold)), float(np.mean(dq < threshold))
        result[f"precision_norm_{threshold}"] = precision
        result[f"coverage_norm_{threshold}"] = coverage
        result[f"fscore_norm_{threshold}"] = 2 * precision * coverage / max(precision + coverage, 1e-12)
    if rgb is not None and gt_rgb is not None:
        color, target = rgb[0].cpu().numpy(), gt_rgb[0].cpu().numpy()
        result["rgb_l1"] = float(.5 * (np.abs(color - target[ip]).mean() + np.abs(target - color[iq]).mean()))
    return result


def verify_frozen_reconstruction(model, checkpoint):
    """Check all teacher/decoder weights, including the GE handoff snapshot."""
    reference = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    prefixes = ("encoder.", "register_up.", "latent_down.", "latent_norm.", "decode_up.",
                "decoder.", "anchor_mlp.", "point_head.", "ge.")
    singles = {"register_pos_embed", "decoder_pos_embed"}
    current = model.state_dict()
    checked = 0
    for key, value in reference.items():
        if key in singles or key.startswith(prefixes):
            target_key = "tokenizer_ge." + key[3:] if key.startswith("ge.") else key
            if target_key not in current or not torch.equal(value, current[target_key].cpu()):
                raise RuntimeError(f"Frozen reconstruction changed during flow: {key}")
            checked += 1
    if checked == 0 or model.tokenizer_ge is None:
        raise RuntimeError("Missing frozen tokenizer snapshot")
    return checked


@torch.no_grad()
def evaluate(args):
    device = torch.device(args.device)
    model, builder, saved = load_unite_model(args.ckpt, device)
    if model.tokenizer_use_weak_context or model.use_surflo_global_cam:
        raise ValueError("This evaluator requires an unconditioned tokenizer and no Surflo camera")
    model.eval()
    model.encoder.query_cache_max_items = 32
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frozen_count = None
    if args.reference_metrics:
        reference = json.loads(Path(args.reference_metrics).read_text())
        frozen_count = verify_frozen_reconstruction(model, reference["checkpoint"])
    pools = load_room_view_pools(args.room_view_pools_json)
    records = []
    # Two disjoint selected pairs, identical across every variant and both stages.
    for pair_index in (0, 1):
        pair_pools = {room: pool[2 * pair_index:2 * pair_index + 2] for room, pool in pools.items()}
        dataset = build_internscenes_render_dataset(
            args.data_dir, room_view_pools=pair_pools, views_per_sample=2,
            view_sample_mode="first", seed=0, include_sharp_label=True,
            pc_size=int(saved["pc_size"]), pc_sharpedge_size=int(saved["pc_sharpedge_size"]),
            align_mode="c_meanrms", strict_load=True,
            vggt_cache_root=None if args.recon_only else args.vggt_cache_root,
            use_joint_vggt_cache=not args.recon_only)
        for index in range(len(dataset)):
            item = dataset[index]
            torch.manual_seed(0)
            surface = item["surface"][None].to(device)
            z, fps, xyz, rgb, anchors = model.forward_tokenizer(surface, representation_phase=False)
            gt, gt_rgb = ShapePCAE.surface_gt_points(surface, include_sharp_label=True)
            room = Path(item["mesh_path"]).name
            record = {"room": room, "pair_index": pair_index, "views": item["view_indices"].tolist(),
                      "recon": point_metrics(xyz, rgb, gt, gt_rgb), "generation": []}
            # All five clouds exported only for the primary pair/seed.
            obj = out / room
            if pair_index == 0:
                obj.mkdir(exist_ok=True)
                export_xyz_pointcloud_ply(gt[0], obj / "gt.ply", colors=gt_rgb[0])
                export_xyz_pointcloud_ply(xyz[0], obj / "recon.ply", colors=rgb[0])
                export_xyz_pointcloud_ply(fps[0], obj / "fps.ply", rgb=(0, 255, 255))
                export_xyz_pointcloud_ply(anchors[0], obj / "anchors.ply", rgb=(255, 0, 255))
            if not args.recon_only:
                weak, keep, cam = _build_weak_context(collate_surface_render([item]), builder, device,
                    align_mode="c_meanrms", include_camera_in_sequence=not model.adaln_camera_cond)
                if weak is None:
                    raise RuntimeError(f"Missing image conditioning for {room}")
                for seed in args.gen_seeds:
                    generator = torch.Generator(device=device).manual_seed(seed)
                    noise = torch.randn(z.shape, generator=generator, device=device, dtype=z.dtype)
                    generated = model.sample_latents(weak, batch_size=1, num_steps=args.sample_steps,
                        guidance_scale=1, noise=noise, context_keep=keep, cam_cond=cam,
                        device=device, dtype=z.dtype, renorm_output=False)
                    gx, gc, _ = model.decode(generated, representation_phase=False)
                    record["generation"].append({"seed": seed, **point_metrics(gx, gc, gt, gt_rgb)})
                    if pair_index == 0 and seed == args.gen_seeds[0]:
                        export_xyz_pointcloud_ply(gx[0], obj / "gen_cfg1.ply", colors=gc[0])
            records.append(record)
            print(f"Evaluated {room} pair={pair_index}", flush=True)
    metrics = {"checkpoint": str(Path(args.ckpt).resolve()), "evaluation_precision": "fp32",
               "coordinate_frame": "normalized first-camera frame; thresholds are normalized units",
               "sample_steps": args.sample_steps, "guidance_scale": 1, "gen_seeds": args.gen_seeds,
               "frozen_reconstruction_tensors_verified": frozen_count, "rooms": records,
               "mean_recon_cd": float(np.mean([r["recon"]["cd"] for r in records]))}
    generations = [g for r in records for g in r["generation"]]
    if generations:
        metrics["mean_gen_cd"] = float(np.mean([g["cd"] for g in generations]))
        metrics["mean_gen_cd_by_seed"] = {str(seed): float(np.mean([g["cd"] for g in generations if g["seed"] == seed])) for seed in args.gen_seeds}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "rooms"}, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    for flag in ("ckpt", "data_dir", "room_view_pools_json", "output_dir"):
        p.add_argument("--" + flag, required=True)
    p.add_argument("--vggt_cache_root")
    p.add_argument("--reference_metrics")
    p.add_argument("--recon_only", action="store_true")
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--gen_seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--device", default="cuda")
    evaluate(p.parse_args())
