#!/usr/bin/env python3
"""A/B: vis-style stale weak_ctx vs eval-style rebuilt weak_ctx after one train step.

Hypothesis: training vis reuses pre-optimizer-step ``weak_ctx`` activations while
eval recomputes context with updated ``vggt_builder`` weights — same ckpt GE,
different conditioning tensor → vis looks great, eval looks mediocre.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from evaluate_pc_unite import load_unite_model
from hy3dgen.shapegen.gs_export import export_xyz_pointcloud_ply
from hy3dgen.shapegen.models.autoencoders.shape_pc_ae import ShapePCAE
from hy3dgen.shapegen.pc_losses import PointCloudAELoss, chamfer_distance
from hy3dgen.shapegen.pc_render_dataset import (
    build_surface_render_dataset,
    collate_surface_render,
)
from train_gs_ae import load_experiment_manifest
from train_pc_unite import _build_weak_context

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _cd(pred: torch.Tensor, gt: torch.Tensor) -> float:
    return float(chamfer_distance(pred, gt)[0])


def _sample(model, ctx, keep, noise, *, steps, gs, device, dtype):
    z = model.sample_latents(
        ctx,
        batch_size=1,
        num_steps=steps,
        guidance_scale=gs,
        noise=noise,
        context_keep=keep,
        device=device,
        dtype=dtype,
    )
    xyz, rgb, _ = model.decode(z, representation_phase=False)
    return xyz, rgb


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/exp4d_cam_cross_velocity/ckpt_final.pt")
    p.add_argument(
        "--data_dir",
        default="/export/home/nathan/datasets/gobjaverse_experiments/furniture_351/train",
    )
    p.add_argument("--gobjaverse_render_root", default="/export/home/nathan/datasets")
    p.add_argument(
        "--vggt_cache_root", default="runs/vggt_cache/furn4_view0_cam_cross"
    )
    p.add_argument("--align_mode", default="cross")
    p.add_argument("--obj_idx", type=int, default=9)
    p.add_argument("--max_items", type=int, default=32)
    p.add_argument("--sample_steps", type=int, default=20)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--output_dir",
        default="runs/exp4d_cam_cross_velocity/diag_stale_ctx_ab",
    )
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model, builder, train_args = load_unite_model(args.ckpt, device)
    include_sharp = bool(train_args.get("include_sharp_label") or False)
    align_mode = args.align_mode
    lr = float(train_args.get("lr", args.lr))

    data_path = Path(args.data_dir).resolve()
    manifest = load_experiment_manifest(str(data_path))
    dataset = build_surface_render_dataset(
        str(data_path),
        max_items=args.max_items,
        use_experiment_manifest=True,
        manifest=manifest,
        render_root=args.gobjaverse_render_root,
        view_idx=int(train_args.get("view_idx", 0)),
        vggt_cache_root=args.vggt_cache_root,
        pc_size=int(train_args.get("pc_size", 5120)),
        pc_sharpedge_size=int(train_args.get("pc_sharpedge_size", 5120)),
        gobjaverse_normalization=not train_args.get(
            "no_gobjaverse_normalization", False
        ),
        surface_in_camera_frame=not train_args.get("no_surface_camera_frame", False),
        align_mode=align_mode,
        include_sharp_label=include_sharp,
    )

    batch = collate_surface_render([dataset[args.obj_idx]])
    mesh = batch["mesh_path"][0]
    stem = Path(mesh).stem
    logger.info("Object %d: %s", args.obj_idx, stem)

    surface = batch["surface"].to(device)
    criterion = PointCloudAELoss(
        lambda_rgb=float(train_args.get("lambda_rgb", 1.0)),
        lambda_anc=float(train_args.get("lambda_anc", 0.3)),
        sinkhorn_eps=float(train_args.get("sinkhorn_eps", 0.02)),
        sinkhorn_iters=int(train_args.get("sinkhorn_iters", 50)),
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- A0: pure eval from ckpt (no train step) -----
    model.eval()
    builder.eval()
    with torch.no_grad():
        gt_xyz, gt_rgb = ShapePCAE.surface_gt_points(
            surface, include_sharp_label=include_sharp
        )
        weak0, keep0 = _build_weak_context(
            batch, builder, device, align_mode=align_mode
        )
        z0, _ = model.encode(surface)
        xyz_r0, rgb_r0, _ = model.decode(z0, representation_phase=False)
        torch.manual_seed(args.seed + 12345)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + 12345)
        noise = torch.randn_like(z0)

        xyz_a0, rgb_a0 = _sample(
            model,
            weak0,
            keep0,
            noise,
            steps=args.sample_steps,
            gs=args.guidance_scale,
            device=device,
            dtype=z0.dtype,
        )
        null0 = model.null_context(1, weak0.shape[1], dtype=z0.dtype, device=device)
        xyz_n0, rgb_n0 = _sample(
            model,
            null0,
            None,
            noise,
            steps=args.sample_steps,
            gs=1.0,
            device=device,
            dtype=z0.dtype,
        )

    metrics = {
        "mesh": mesh,
        "obj_idx": args.obj_idx,
        "sample_steps": args.sample_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "lr": lr,
        "A0_ckpt_eval_no_step": {
            "recon_cd": _cd(xyz_r0, gt_xyz),
            "gen_cd": _cd(xyz_a0, gt_xyz),
            "null_cd": _cd(xyz_n0, gt_xyz),
            "gen_to_recon_cd": _cd(xyz_a0, xyz_r0),
            "ctx_abs_mean": float(weak0.detach().abs().mean()),
        },
    }
    export_xyz_pointcloud_ply(gt_xyz[0].cpu(), out_dir / "gt.ply", colors=gt_rgb[0].cpu())
    export_xyz_pointcloud_ply(
        xyz_r0[0].cpu(), out_dir / "A0_recon.ply", colors=rgb_r0[0].cpu()
    )
    export_xyz_pointcloud_ply(
        xyz_a0[0].cpu(), out_dir / "A0_gen_fresh_ctx.ply", colors=rgb_a0[0].cpu()
    )
    export_xyz_pointcloud_ply(
        xyz_n0[0].cpu(), out_dir / "A0_null.ply", colors=rgb_n0[0].cpu()
    )
    logger.info("A0 (ckpt eval): %s", metrics["A0_ckpt_eval_no_step"])

    # Snapshot builder+model params before the simulated train step (already at ckpt).
    # ----- One train step like training loop -----
    model.train()
    builder.train()
    opt = torch.optim.AdamW(
        [p for p in list(model.parameters()) + list(builder.parameters()) if p.requires_grad],
        lr=lr,
        weight_decay=float(train_args.get("weight_decay", 0.01)),
    )

    weak_pre, keep_pre = _build_weak_context(
        batch, builder, device, align_mode=align_mode
    )
    # Keep a detached copy of pre-step context values for comparison stats.
    weak_pre_values = weak_pre.detach().clone()

    z, fps_xyz, xyz, rgb, centers = model.forward_tokenizer(surface)
    gt_xyz_t, gt_rgb_t = ShapePCAE.surface_gt_points(
        surface, include_sharp_label=include_sharp
    )
    recon_loss, extras = criterion(
        xyz, rgb, gt_xyz_t, gt_rgb_t, centers=centers, fps_xyz=fps_xyz.detach()
    )
    flow_out = model.forward_denoising(z, weak_pre, context_keep=keep_pre)
    flow_loss = flow_out["flow/flow_loss"]
    loss = float(train_args.get("lambda_recon", 1.0)) * recon_loss + float(
        train_args.get("lambda_flow", 1.0)
    ) * flow_loss

    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(model.parameters()) + list(builder.parameters()),
        float(train_args.get("max_grad_norm", 1.0)),
    )
    opt.step()
    logger.info(
        "train step done: loss=%.4f recon=%.4f flow=%.4f cd=%.4f",
        float(loss),
        float(recon_loss),
        float(flow_loss),
        float(extras["loss_cd"]),
    )

    # Context tensor still holds pre-step activations (vis behavior).
    stale_ctx = weak_pre.detach()
    stale_keep = keep_pre.detach() if keep_pre is not None else None
    ctx_change_from_build = float(
        (stale_ctx - weak_pre_values).abs().mean()
    )  # should be ~0 (same tensor values)

    model.eval()
    builder.eval()
    with torch.no_grad():
        # Rebuild context with POST-step builder weights (eval behavior).
        weak_post, keep_post = _build_weak_context(
            batch, builder, device, align_mode=align_mode
        )
        ctx_l1 = float((weak_post - stale_ctx).abs().mean())
        ctx_rel = ctx_l1 / max(float(stale_ctx.abs().mean()), 1e-8)
        ctx_cos = float(
            torch.nn.functional.cosine_similarity(
                weak_post.flatten().float(),
                stale_ctx.flatten().float(),
                dim=0,
            )
        )

        z_post, _ = model.encode(surface)
        xyz_r_post, rgb_r_post, _ = model.decode(z_post, representation_phase=False)

        # Same ODE noise for fair A/B (reuse noise from A0).
        xyz_stale, rgb_stale = _sample(
            model,
            stale_ctx,
            stale_keep,
            noise,
            steps=args.sample_steps,
            gs=args.guidance_scale,
            device=device,
            dtype=z_post.dtype,
        )
        xyz_fresh, rgb_fresh = _sample(
            model,
            weak_post,
            keep_post,
            noise,
            steps=args.sample_steps,
            gs=args.guidance_scale,
            device=device,
            dtype=z_post.dtype,
        )
        null = model.null_context(
            1, weak_post.shape[1], dtype=z_post.dtype, device=device
        )
        xyz_null, rgb_null = _sample(
            model,
            null,
            None,
            noise,
            steps=args.sample_steps,
            gs=1.0,
            device=device,
            dtype=z_post.dtype,
        )

    metrics["ctx_tensor_changed_inplace_during_step"] = ctx_change_from_build
    metrics["ctx_l1_stale_vs_rebuilt"] = ctx_l1
    metrics["ctx_rel_l1_stale_vs_rebuilt"] = ctx_rel
    metrics["ctx_cosine_stale_vs_rebuilt"] = ctx_cos
    metrics["B_after_one_train_step"] = {
        "recon_cd": _cd(xyz_r_post, gt_xyz_t),
        "gen_stale_ctx_cd": _cd(xyz_stale, gt_xyz_t),
        "gen_fresh_ctx_cd": _cd(xyz_fresh, gt_xyz_t),
        "null_cd": _cd(xyz_null, gt_xyz_t),
        "stale_to_recon_cd": _cd(xyz_stale, xyz_r_post),
        "fresh_to_recon_cd": _cd(xyz_fresh, xyz_r_post),
        "stale_vs_fresh_cd": _cd(xyz_stale, xyz_fresh),
        "train_loss": float(loss),
        "train_recon": float(recon_loss),
        "train_flow": float(flow_loss),
    }

    export_xyz_pointcloud_ply(
        xyz_r_post[0].cpu(), out_dir / "B_recon_after_step.ply", colors=rgb_r_post[0].cpu()
    )
    export_xyz_pointcloud_ply(
        xyz_stale[0].cpu(),
        out_dir / "B_gen_STALE_ctx_vis_style.ply",
        colors=rgb_stale[0].cpu(),
    )
    export_xyz_pointcloud_ply(
        xyz_fresh[0].cpu(),
        out_dir / "B_gen_FRESH_ctx_eval_style.ply",
        colors=rgb_fresh[0].cpu(),
    )
    export_xyz_pointcloud_ply(
        xyz_null[0].cpu(), out_dir / "B_null_after_step.ply", colors=rgb_null[0].cpu()
    )

    # Also: sample with stale ctx but BEFORE comparing — already done.
    # Extra: fresh ctx with a few seeds after step (sanity).
    seed_rows = []
    for i in range(8):
        s = args.seed + 1000 + i
        torch.manual_seed(s)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(s)
        with torch.no_grad():
            n2 = torch.randn_like(z_post)
            g_s, _ = _sample(
                model,
                stale_ctx,
                stale_keep,
                n2,
                steps=args.sample_steps,
                gs=args.guidance_scale,
                device=device,
                dtype=z_post.dtype,
            )
            g_f, _ = _sample(
                model,
                weak_post,
                keep_post,
                n2,
                steps=args.sample_steps,
                gs=args.guidance_scale,
                device=device,
                dtype=z_post.dtype,
            )
        seed_rows.append(
            {
                "seed": s,
                "stale_cd": _cd(g_s, gt_xyz_t),
                "fresh_cd": _cd(g_f, gt_xyz_t),
                "delta_fresh_minus_stale": _cd(g_f, gt_xyz_t) - _cd(g_s, gt_xyz_t),
            }
        )
    metrics["seed_ab_after_step"] = seed_rows

    b = metrics["B_after_one_train_step"]
    logger.info(
        "B after step: stale_gen=%.5f fresh_gen=%.5f null=%.5f recon=%.5f "
        "ctx_l1=%.6f ctx_cos=%.6f",
        b["gen_stale_ctx_cd"],
        b["gen_fresh_ctx_cd"],
        b["null_cd"],
        b["recon_cd"],
        ctx_l1,
        ctx_cos,
    )
    for r in seed_rows:
        logger.info(
            "  seed %d: stale=%.5f fresh=%.5f (fresh-stale=%+.5f)",
            r["seed"],
            r["stale_cd"],
            r["fresh_cd"],
            r["delta_fresh_minus_stale"],
        )

    verdict = {
        "hypothesis_supported": bool(
            b["gen_stale_ctx_cd"] + 0.02 < b["gen_fresh_ctx_cd"]
        ),
        "note": (
            "Supported if stale (vis-style) gen CD is clearly better than "
            "fresh (eval-style) gen CD after one train step, with same ODE noise."
        ),
    }
    metrics["verdict"] = verdict

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Wrote %s", out_dir)
    logger.info("verdict: %s", verdict)


if __name__ == "__main__":
    main()
