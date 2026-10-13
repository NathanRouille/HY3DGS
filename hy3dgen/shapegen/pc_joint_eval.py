"""Evaluate every cached ordered pair without altering the training RNG stream."""
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .gs_export import export_xyz_pointcloud_ply
from .models.autoencoders.shape_pc_ae import ShapePCAE
from .pc_quality_metrics import reconstruction_metrics
from .pc_render_dataset import collate_surface_render


def _metrics(xyz, rgb, gt, gt_rgb):
    return reconstruction_metrics(xyz[0].cpu().numpy(), gt[0].cpu().numpy(),
        None if rgb is None else rgb[0].cpu().numpy(),
        None if gt_rgb is None else gt_rgb[0].cpu().numpy())


@torch.no_grad()
def evaluate_joint_pairs(model, builder, items, build_context, *, output_dir,
                         step, sample_steps=50, gen_seeds=(0,), export_ply=False):
    if not items or sample_steps < 2 or not gen_seeds:
        raise ValueError("Evaluation requires pairs, seeds, and at least two sampling states")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists():
        raise FileExistsError(f"Evaluation already exists: {out}")
    device = next(model.parameters()).device
    training, builder_training = model.training, builder.training
    py_rng, np_rng = random.getstate(), np.random.get_state()
    started = time.monotonic()
    records, exported = [], set()
    try:
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
            model.eval()
            builder.eval()
            for (_, pair), item in items.items():
                surface = item["surface"][None].to(device)
                z, fps, xyz, rgb, anchors = model.forward_tokenizer(surface, representation_phase=False)
                gt, gt_rgb = ShapePCAE.surface_gt_points(surface, include_sharp_label=True)
                room = Path(item["mesh_path"]).name
                score, dp, dq = _metrics(xyz, rgb, gt, gt_rgb)
                record = {"room": room, "views": list(pair), "recon": score, "generation": []}
                primary = export_ply and room not in exported
                obj = out / room
                if primary:
                    obj.mkdir(exist_ok=True)
                    for name, points, colors in (("gt", gt, gt_rgb), ("recon", xyz, rgb)):
                        export_xyz_pointcloud_ply(points[0], obj / f"{name}.ply", colors=colors[0])
                    export_xyz_pointcloud_ply(fps[0], obj / "fps.ply", rgb=(0, 255, 255))
                    export_xyz_pointcloud_ply(anchors[0], obj / "anchors.ply", rgb=(255, 0, 255))
                    err = np.clip(dp / .02, 0, 1)
                    colors = torch.as_tensor(np.column_stack([err, 1 - err, np.zeros_like(err)]), dtype=torch.float32)
                    export_xyz_pointcloud_ply(xyz[0], obj / "recon_error.ply", colors=colors)
                    missing = torch.as_tensor(dq >= .02, device=device)
                    export_xyz_pointcloud_ply(gt[0][missing], obj / "gt_uncovered.ply", rgb=(255, 0, 0))
                weak, keep, cam = build_context(collate_surface_render([item]), builder, device,
                    align_mode="c_meanrms", include_camera_in_sequence=not model.adaln_camera_cond)
                if weak is None:
                    raise RuntimeError(f"Missing weak context for {room} {pair}")
                for seed in gen_seeds:
                    noise = torch.randn(z.shape, device=device, dtype=z.dtype,
                        generator=torch.Generator(device=device).manual_seed(seed))
                    generated = model.sample_latents(weak, batch_size=1, num_steps=sample_steps,
                        guidance_scale=1, noise=noise, context_keep=keep, cam_cond=cam,
                        device=device, dtype=z.dtype, renorm_output=False)
                    gx, gc, _ = model.decode(generated, representation_phase=False)
                    record["generation"].append({"seed": seed, **_metrics(gx, gc, gt, gt_rgb)[0]})
                    if primary and seed == gen_seeds[0]:
                        export_xyz_pointcloud_ply(gx[0], obj / "gen_cfg1.ply", colors=gc[0])
                records.append(record)
                exported.add(room)
                if len(records) % 12 == 0:
                    print(f"Joint eval step={step}: {len(records)}/{len(items)} pairs", flush=True)
    finally:
        random.setstate(py_rng)
        np.random.set_state(np_rng)
        model.train(training)
        model.ensure_tokenizer_ge_eval()
        builder.train(builder_training)
    generations = [g for r in records for g in r["generation"]]
    summary = {"step": step, "pair_count": len(records), "room_count": len(exported),
        "gen_seeds": list(gen_seeds), "sample_steps": sample_steps, "guidance_scale": 1,
        "coordinate_frame": "normalized first-camera frame; thresholds are not physical metres",
        "evaluation_precision": "fp32", "seconds": time.monotonic() - started}
    for kind, values in (("recon", [r["recon"] for r in records]), ("gen", generations)):
        for key in values[0]:
            if key != "seed":
                summary[f"mean_{kind}/{key}"] = float(np.mean([v[key] for v in values]))
    (out / "metrics.json").write_text(json.dumps({**summary, "rooms": records}, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k in (
        "step", "pair_count", "gen_seeds", "seconds", "mean_recon/cd", "mean_gen/cd",
        "mean_recon/fscore_norm_0.02", "mean_gen/fscore_norm_0.02",
        "mean_recon/gt_distance_p99", "mean_gen/gt_distance_p99")}), flush=True)
    return summary
