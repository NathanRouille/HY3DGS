#!/usr/bin/env python3
"""Train ShapePCUnite: joint point-cloud AE + flow matching with VGGT weak context."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:
    wandb = None

import numpy as np

from hy3dgen.shapegen.gs_export import export_xyz_pointcloud_ply
from hy3dgen.shapegen.models.autoencoders.shape_pc_ae import (
    REGISTER_NOISE_MODES,
    ShapePCAE,
)
from hy3dgen.shapegen.models.autoencoders.shape_pc_unite import ShapePCUnite
from hy3dgen.shapegen.pc_debug_export import export_recon_debug_plys
from hy3dgen.shapegen.pc_losses import PointCloudAELoss, chamfer_distance
from hy3dgen.shapegen.pc_render_dataset import (
    build_surface_render_dataset,
    collate_surface_render,
)
from hy3dgen.shapegen.pc_training_utils import (
    anchor_diagnostics,
    compute_pc_loss_grad_norms,
    global_grad_norm,
    latent_statistics,
    loss_balance_ratios,
)
from hy3dgen.shapegen.pretrained_profiles import resolve_include_sharp_label
from hy3dgen.shapegen.surflo_global_camera import (
    SurfloGlobalCameraBranch,
    extract_raw_vggt_camera_tokens,
)
from hy3dgen.shapegen.vggt_context import VGGTContextBuilder, load_state_dict_skip_mismatch
from train_gs_ae import load_experiment_manifest, resolve_category_ids
from train_pc_ae import _apply_ca_profile_flags

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _pad_weak_contexts(
    ctxs: List[torch.Tensor],
    keeps: List[torch.Tensor],
    cams: List[torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length weak contexts to a batch tensor."""
    max_n = max(c.shape[1] for c in ctxs)
    width = ctxs[0].shape[-1]
    b = len(ctxs)
    out = ctxs[0].new_zeros(b, max_n, width)
    keep = torch.zeros(b, max_n, dtype=torch.bool, device=device)
    cam_out = cams[0].new_zeros(b, width)
    for i, (c, k, cam) in enumerate(zip(ctxs, keeps, cams)):
        n = c.shape[1]
        out[i, :n] = c[0]
        keep[i, :n] = k[0]
        cam_out[i] = cam[0]
    return out, keep, cam_out


def _build_weak_context(
    batch: Dict,
    builder: VGGTContextBuilder,
    device: torch.device,
    *,
    null_weak_context: Optional[torch.nn.Parameter] = None,
    use_null: bool = False,
    rgb_source: Optional[torch.Tensor] = None,
    align_mode: str = "cross",
    include_camera_in_sequence: bool = True,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Weak context [B, N, width] + keep mask [B, N] + camera embed [B, width].

    ``camera_embed`` is the projected VGGT camera token (for AdaLN). It is
    returned even when ``include_camera_in_sequence`` is False (patches-only
    sequence). Null / missing RGB paths return ``camera_embed=None``.

    Multi-view ``c_meanrms`` uses ``batch['rgb_views']`` ``[B,S,3,H,W]`` (online
    VGGT, view 0 = reference) unless ``batch['vggt_cache']`` holds a joint cache
    payload (``cache_kind='joint'``), in which case the cached joint forward is
    used instead. Single-view paths keep the cached / RGB builders.
    """
    if use_null and null_weak_context is not None:
        b = batch["surface"].shape[0]
        n = int(null_weak_context.shape[1]) if null_weak_context.dim() == 3 else 1
        ctx = null_weak_context.expand(b, n, -1).to(device)
        keep = torch.ones(b, n, dtype=torch.bool, device=device)
        return ctx, keep, None

    if "vggt_cache" in batch and batch["vggt_cache"] and rgb_source is None:
        ctxs = []
        keeps = []
        cams = []
        for payload in batch["vggt_cache"]:
            c, k, cam = builder.build_from_cached(
                payload,
                device,
                align_mode=align_mode,
                include_camera_in_sequence=include_camera_in_sequence,
            )
            ctxs.append(c)
            keeps.append(k)
            cams.append(cam)
        return _pad_weak_contexts(ctxs, keeps, cams, device)

    # Online multi-view (or single-view) c_meanrms from stacked RGBs.
    if "rgb_views" in batch and rgb_source is None and align_mode == "c_meanrms":
        rgb_views = batch["rgb_views"].to(device)  # [B,S,3,H,W]
        ctxs, keeps, cams = [], [], []
        for i in range(rgb_views.shape[0]):
            c, k, cam = builder.build_c_meanrms_from_rgb_views(
                rgb_views[i],
                include_camera_in_sequence=include_camera_in_sequence,
            )
            ctxs.append(c)
            keeps.append(k)
            cams.append(cam)
        return _pad_weak_contexts(ctxs, keeps, cams, device)

    if "rgb" not in batch and rgb_source is None:
        return None, None, None
    rgb = (batch["rgb"] if rgb_source is None else rgb_source).to(device)
    if align_mode == "c_meanrms":
        # Single RGB → treat as S=1 online c_meanrms (own mean+RMS on centres).
        ctxs, keeps, cams = [], [], []
        for i in range(rgb.shape[0]):
            c, k, cam = builder.build_c_meanrms_from_rgb_views(
                rgb[i : i + 1],
                include_camera_in_sequence=include_camera_in_sequence,
            )
            ctxs.append(c)
            keeps.append(k)
            cams.append(cam)
        return _pad_weak_contexts(ctxs, keeps, cams, device)
    return builder(rgb, include_camera_in_sequence=include_camera_in_sequence)


def _set_requires_grad(module: torch.nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def _apply_recon_freeze(
    model: ShapePCUnite,
    *,
    preserve_tokenizer_ge: bool = False,
    preserve_denoiser_pe: bool = False,
) -> None:
    """Freeze the whole reconstruction pathway; train the flow denoiser only.

    The GE is shared, so freezing only encoder/decoder let flow training drift
    the tokenizer. We snapshot the GE for the tokenizer pass instead.

    Tokenizer PE (``register_pos_embed``) is frozen; denoiser PE is a trainable
    copy seeded from the tokenizer PE. ``latent_norm`` stays shared and frozen.

    When resuming a prior ``--freeze_recon`` run, pass ``preserve_*=True`` so we
    keep the ckpt's tokenizer GE / denoiser PE instead of re-cloning from the
    already flow-tuned live ``ge`` (that bug wiped recon in exp17).
    """
    frozen = (
        model.encoder,
        model.register_up,
        model.latent_down,
        model.latent_norm,
        model.decode_up,
        model.decoder,
        model.anchor_mlp,
        model.point_head,
    )
    for mod in frozen:
        _set_requires_grad(mod, False)
    if not preserve_denoiser_pe:
        model.sync_denoiser_pos_embed_from_tokenizer()
    model.register_pos_embed.requires_grad = False
    model.denoiser_pos_embed.requires_grad = True
    model.decoder_pos_embed.requires_grad = False
    if preserve_tokenizer_ge and model.tokenizer_ge is not None:
        model.freeze_tokenizer_ge(replace=False)
        logger.info("Preserving tokenizer_ge from checkpoint (chained freeze_recon)")
    else:
        model.freeze_tokenizer_ge(replace=True)
        logger.info("Snapshotted tokenizer_ge from live ge")
    model.ensure_tokenizer_ge_eval()


def _save_ckpt(
    path: Path,
    *,
    step: int,
    model: ShapePCUnite,
    vggt_builder: VGGTContextBuilder,
    optimizer: torch.optim.Optimizer,
    scheduler,
    args,
    include_optimizer: bool = True,
) -> None:
    """Save a checkpoint (VGGT itself is intentionally not part of the state)."""
    payload = {
        "step": step,
        "model": model.state_dict(),
        "vggt_builder": vggt_builder.state_dict(),
        "scheduler": scheduler.state_dict(),
        "args": vars(args),
    }
    if include_optimizer:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)
    size_gb = path.stat().st_size / 1e9
    logger.info("Saved %s (%.2f GB)", path, size_gb)


def _first(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """First batch element, passing ``None`` through (geometry-only has no RGB)."""
    return None if t is None else t[0]


def _first_cpu(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if t is None else t[0].cpu()


def _as_object3d(xyz: torch.Tensor, rgb: Optional[torch.Tensor] = None):
    """wandb point cloud: [N,3] xyz or [N,6] xyz+rgb(0-255)."""
    if wandb is None:
        return None
    pts = xyz.detach().float().cpu().numpy()
    if rgb is not None:
        col = rgb.detach().float().clamp(0, 1).cpu().numpy() * 255.0
        pts = np.concatenate([pts, col], axis=1)
    return wandb.Object3D(pts)


@torch.no_grad()
def _log_visuals(
    model: ShapePCUnite,
    batch: Dict,
    weak_ctx: Optional[torch.Tensor],
    *,
    gt_xyz: torch.Tensor,
    gt_rgb: Optional[torch.Tensor] = None,
    xyz_recon: torch.Tensor,
    rgb_recon: torch.Tensor,
    fps_xyz: Optional[torch.Tensor] = None,
    centers: Optional[torch.Tensor] = None,
    output_dir: Path,
    step: int,
    sample_steps: int = 20,
    guidance_scale: float = 1.0,
    export_ply: bool = True,
    export_recon_debug: bool = True,
    intruder_thresh: float = 0.02,
    context_keep: Optional[torch.Tensor] = None,
    cam_cond: Optional[torch.Tensor] = None,
    raw_camera_tokens: Optional[torch.Tensor] = None,
) -> Dict:
    """Sample image-conditioned + unconditional clouds for qualitative checks."""
    was_training = model.training
    model.eval()
    metrics: Dict[str, float] = {}
    objects: Dict[str, object] = {}
    ply_dir = None
    try:
        b = gt_xyz.shape[0]
        n_tokens = weak_ctx.shape[1] if weak_ctx is not None else 1
        noise = torch.randn(
            b, model.num_registers, model.embed_dim, device=gt_xyz.device, dtype=gt_xyz.dtype
        )
        variants: Dict[str, Optional[torch.Tensor]] = {"gen": weak_ctx}
        variants["gen_null"] = model.null_context(
            b, n_tokens, dtype=gt_xyz.dtype, device=gt_xyz.device
        )

        outputs: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for name, ctx in variants.items():
            if ctx is None:
                continue
            z = model.sample_latents(
                ctx,
                batch_size=b,
                num_steps=sample_steps,
                guidance_scale=guidance_scale if name == "gen" else 1.0,
                noise=noise,
                context_keep=context_keep if name == "gen" else None,
                cam_cond=cam_cond if name == "gen" else None,
                raw_camera_tokens=raw_camera_tokens if name == "gen" else None,
                device=gt_xyz.device,
                dtype=gt_xyz.dtype,
            )
            xyz_g, rgb_g, _ = model.decode(z, representation_phase=False)
            cd, _, _ = chamfer_distance(xyz_g, gt_xyz)
            metrics[f"vis/{name}_cd"] = float(cd)
            outputs[name] = (xyz_g, rgb_g)

        if "vis/gen_cd" in metrics and "vis/gen_null_cd" in metrics:
            metrics["vis/cond_gain"] = metrics["vis/gen_null_cd"] - metrics["vis/gen_cd"]

        objects["vis/gt"] = _as_object3d(gt_xyz[0])
        objects["vis/recon"] = _as_object3d(xyz_recon[0], _first(rgb_recon))
        for name, (xyz_g, rgb_g) in outputs.items():
            objects[f"vis/{name}"] = _as_object3d(xyz_g[0], _first(rgb_g))
        objects = {k: v for k, v in objects.items() if v is not None}

        if export_ply:
            ply_dir = output_dir / "vis" / f"step_{step:07d}"
            ply_dir.mkdir(parents=True, exist_ok=True)
            export_xyz_pointcloud_ply(
                gt_xyz[0].cpu(),
                ply_dir / "gt.ply",
                colors=gt_rgb[0].cpu() if gt_rgb is not None else None,
            )
            export_xyz_pointcloud_ply(
                xyz_recon[0].cpu(),
                ply_dir / "recon.ply",
                colors=_first_cpu(rgb_recon),
            )
            for name, (xyz_g, rgb_g) in outputs.items():
                export_xyz_pointcloud_ply(
                    xyz_g[0].cpu(), ply_dir / f"{name}.ply", colors=_first_cpu(rgb_g)
                )
            if export_recon_debug:
                patch_centers = patch_keep = None
                if "vggt_cache" in batch and batch["vggt_cache"]:
                    payload = batch["vggt_cache"][0]
                    if isinstance(payload, dict):
                        patch_centers = payload.get("patch_centers")
                        patch_keep = payload.get("patch_keep")
                export_recon_debug_plys(
                    ply_dir,
                    gt_xyz=gt_xyz[0],
                    recon_xyz=xyz_recon[0],
                    gt_rgb=_first(gt_rgb),
                    recon_rgb=_first(rgb_recon),
                    fps_xyz=fps_xyz[0] if fps_xyz is not None else None,
                    centers=centers[0] if centers is not None else None,
                    num_points_per_anchor=int(model.num_points_per_anchor),
                    max_anchor_delta=model.max_anchor_delta,
                    patch_centers=patch_centers,
                    patch_keep=patch_keep,
                    intruder_thresh=intruder_thresh,
                )
    finally:
        if was_training:
            model.train()
            model.ensure_tokenizer_ge_eval()
    return {"metrics": metrics, "objects": objects, "ply_dir": str(ply_dir) if ply_dir else None}


def train(args):
    _apply_ca_profile_flags(args)
    include_sharp_label = resolve_include_sharp_label(args)
    point_feats = int(getattr(args, "point_feats", 6 if not include_sharp_label else 7))
    if args.geometry_only:
        # Geometry-only consumes exactly Hunyuan3D's surface layout
        # (xyz | normals | sharp), so input_proj loads verbatim and no RGB
        # column is ever read, decoded, or scored.
        include_sharp_label = True
        args.include_sharp_label = True
        point_feats = 4
        args.point_feats = 4
        args.lambda_rgb = 0.0
        logger.info("geometry_only=True → include_sharp_label, point_feats=4, no RGB head/loss")
    elif include_sharp_label and point_feats < 7:
        point_feats = 7
        args.point_feats = 7
        logger.info("include_sharp_label=True → point_feats forced to 7 (normals|sharp|rgb)")
    categories = resolve_category_ids(args.categories)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    num_registers = (
        int(args.num_registers) if args.num_registers is not None else int(args.num_latents)
    )

    model = ShapePCUnite(
        num_latents=args.num_latents,
        num_registers=num_registers,
        embed_dim=args.embed_dim,
        width=args.width,
        heads=args.heads,
        num_ge_layers=args.num_ge_layers,
        num_decoder_layers=args.num_decoder_layers,
        pc_size=args.pc_size,
        pc_sharpedge_size=args.pc_sharpedge_size,
        point_feats=point_feats,
        downsample_ratio=args.downsample_ratio,
        num_points_per_anchor=args.num_points_per_anchor,
        deterministic_encoder=args.deterministic_encoder,
        register_noise_mode=args.register_noise_mode,
        geometry_only=args.geometry_only,
        sample_renorm_output=args.sample_renorm_output,
        tokenizer_use_weak_context=args.tokenizer_use_weak_context,
        adaln_camera_cond=args.adaln_camera_cond,
        use_surflo_global_cam=bool(args.surflo_global_cam),
        max_anchor_delta=getattr(args, "max_anchor_delta", None),
        qk_norm=bool(getattr(args, "qk_norm", True)),
        flow_steps_per_recon=args.flow_steps_per_recon,
        gen_loss_weight=args.lambda_flow,
        modulation_recon_timestep_max=args.modulation_recon_timestep_max,
        noising_t_start=args.noising_t_start,
        weak_context_dropout=args.weak_context_dropout,
        flow_loss_type=args.flow_loss_type,
        use_rope=args.use_rope,
        use_lognorm=bool(args.use_lognorm),
        lognorm_mu=float(args.lognorm_mu),
        lognorm_sigma=float(args.lognorm_sigma),
        t0_force_prob=float(args.t0_force_prob),
        timestep_shift_alpha=float(getattr(args, "timestep_shift_alpha", 0.0)),
        offpath_mode=str(getattr(args, "offpath_mode", "none")),
        offpath_weight=float(getattr(args, "offpath_weight", 1.0)),
        offpath_noise_std=float(getattr(args, "offpath_noise_std", 0.1)),
        offpath_step_size=float(getattr(args, "offpath_step_size", 0.05)),
    ).to(device)
    if args.surflo_global_cam:
        if args.adaln_camera_cond:
            raise ValueError(
                "Do not combine --surflo_global_cam with legacy --adaln_camera_cond"
            )
        branch = SurfloGlobalCameraBranch.from_surflo_checkpoint(
            args.surflo_ckpt,
            width=args.width,
            surflo_root=args.surflo_root,
            freeze_backbone=True,
        ).to(device)
        model.attach_surflo_global_cam(branch)
        logger.info(
            "Surflo global-cam AdaLN enabled (frozen projector+compressor, "
            "trainable zero-init adapter); sequence camera tokens retained"
        )
    # Persist what eval needs to rebuild the exact same architecture.
    args.point_feats = point_feats
    args.include_sharp_label = include_sharp_label
    model.representation_noising = bool(args.representation_noising)
    logger.info(
        "include_sharp_label=%s point_feats=%d geometry_only=%s "
        "representation_noising=%s (t_start=%.2f) register_noise=%s "
        "sample_renorm_output=%s tokenizer_use_weak_context=%s adaln_camera_cond=%s "
        "surflo_global_cam=%s "
        "use_lognorm=%s lognorm=(%.2f,%.2f) t0_force_prob=%.2f "
        "modulation_recon_timestep_max=%.4f "
        "offpath_mode=%s weight=%.3f noise_std=%.4f step_size=%.4f",
        include_sharp_label,
        point_feats,
        args.geometry_only,
        model.representation_noising,
        args.noising_t_start,
        args.register_noise_mode,
        args.sample_renorm_output,
        args.tokenizer_use_weak_context,
        args.adaln_camera_cond,
        bool(args.surflo_global_cam),
        args.use_lognorm,
        args.lognorm_mu,
        args.lognorm_sigma,
        args.t0_force_prob,
        args.modulation_recon_timestep_max,
        model.offpath_mode,
        model.offpath_weight,
        model.offpath_noise_std,
        model.offpath_step_size,
    )
    if args.pretrained_load == "cross_attn" and not args.resume_ckpt:
        model.load_shapevae_cross_attn(
            args.pretrained_repo,
            rgb_feat_init=args.rgb_feat_init,
            subfolder=args.pretrained_subfolder or "hunyuan3d-vae-v2-mini-withencoder",
        )

    vggt_builder = VGGTContextBuilder(width=args.width).to(device)
    # Same Fourier basis as the surface encoder (camera-frame xyz → same freqs).
    vggt_builder.attach_fourier(model.fourier_embedder)
    if args.freeze_vggt_builder:
        _set_requires_grad(vggt_builder, False)

    criterion = PointCloudAELoss(
        lambda_rgb=args.lambda_rgb,
        lambda_anc=args.lambda_anc,
        lambda_anc_cd=args.lambda_anc_cd,
        lambda_delta=float(getattr(args, "lambda_delta", 0.0)),
        sinkhorn_eps=args.sinkhorn_eps,
        sinkhorn_iters=args.sinkhorn_iters,
        rgb_topk_frac=float(getattr(args, "rgb_topk_frac", 0.0)),
        rgb_topk_beta=float(getattr(args, "rgb_topk_beta", 0.0)),
        geometry_only=args.geometry_only,
    )

    data_path = Path(args.data_dir).resolve()
    manifest = load_experiment_manifest(str(data_path)) if not args.no_experiment_manifest else None
    from hy3dgen.shapegen.gobjaverse_gt import parse_view_indices

    train_views = parse_view_indices(
        view_idx=args.view_idx,
        num_views=args.num_views,
        view_indices=args.view_indices,
    )
    views_per_sample = int(getattr(args, "views_per_sample", 1) or 1)
    if views_per_sample > 1 and args.align_mode != "c_meanrms":
        raise ValueError(
            "views_per_sample>1 requires --align_mode c_meanrms "
            f"(got align_mode={args.align_mode!r})"
        )
    use_joint_cache = bool(getattr(args, "vggt_joint_cache", False))
    if use_joint_cache:
        if not args.vggt_cache_root:
            raise ValueError("--vggt_joint_cache requires --vggt_cache_root")
        if views_per_sample < 2:
            raise ValueError("--vggt_joint_cache requires --views_per_sample >= 2")
        if len(train_views) < views_per_sample:
            raise ValueError(
                "--vggt_joint_cache requires len(--view_indices) >= "
                f"--views_per_sample ({len(train_views)} vs {views_per_sample})"
            )
        if len(train_views) == views_per_sample:
            logger.info(
                "Joint VGGT cache (fixed ordered tuple): views=%s "
                "(view 0 = VGGT ref / GT camera)",
                train_views,
            )
        else:
            from hy3dgen.shapegen.vggt_context import ordered_view_tuples

            n_pairs = len(
                ordered_view_tuples(train_views, tuple_size=views_per_sample)
            )
            logger.info(
                "Joint VGGT cache (random ordered %d-tuples from pool %s): "
                "%d possible ordered tuples — cache with "
                "cache_vggt_features.py --joint_pairs --view_indices \"%s\"",
                views_per_sample,
                train_views,
                n_pairs,
                ",".join(str(v) for v in train_views),
            )
    dataset = build_surface_render_dataset(
        str(data_path),
        max_items=args.max_items,
        categories=categories,
        include_sharp_label=include_sharp_label,
        use_experiment_manifest=not args.no_experiment_manifest,
        manifest=manifest,
        render_root=args.gobjaverse_render_root,
        view_indices=train_views,
        views_per_sample=views_per_sample,
        view_sample_mode="random",
        vggt_cache_root=args.vggt_cache_root,
        use_joint_vggt_cache=use_joint_cache,
        pc_size=args.pc_size,
        pc_sharpedge_size=args.pc_sharpedge_size,
        seed=args.seed,
        gobjaverse_normalization=not args.no_gobjaverse_normalization,
        surface_in_camera_frame=not args.no_surface_camera_frame,
        align_mode=args.align_mode,
    )
    if use_joint_cache and len(dataset) == 0:
        raise RuntimeError(
            "No training samples with joint VGGT cache — run "
            "cache_vggt_features.py --joint_pairs "
            f"--view_indices \"{','.join(str(v) for v in train_views)}\" "
            f"--cache_root {args.vggt_cache_root}"
            if len(train_views) > views_per_sample
            else (
                "No training samples with joint VGGT cache — run "
                "cache_vggt_features.py --joint "
                f"--view_indices \"{','.join(str(v) for v in train_views)}\" "
                f"--cache_root {args.vggt_cache_root}"
            )
        )
    if views_per_sample > 1:
        logger.info(
            "Train dataset: %d samples (%d meshes, pool=%d views, "
            "views_per_sample=%d random, align_mode=%s, joint_vggt_cache=%s)",
            len(dataset),
            len(dataset.mesh_paths),
            len(train_views),
            views_per_sample,
            args.align_mode,
            use_joint_cache,
        )
    else:
        logger.info(
            "Train dataset: %d samples (%d meshes × %d views)",
            len(dataset),
            len(dataset.mesh_paths),
            len(train_views),
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_surface_render,
        pin_memory=(device.type == "cuda"),
        drop_last=len(dataset) >= args.batch_size,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    # Resume *before* building the optimizer so that (a) the frozen tokenizer GE
    # snapshot is taken from trained weights and (b) the optimizer only ever sees
    # the parameters we actually train.
    start_step = 0
    ckpt = None
    ckpt_has_tokenizer_ge = False
    ckpt_has_denoiser_pe = False
    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location=device, weights_only=False)
        ckpt_has_tokenizer_ge = any(
            k.startswith("tokenizer_ge.") for k in ckpt["model"]
        )
        ckpt_has_denoiser_pe = "denoiser_pos_embed" in ckpt["model"]
        # Allocate tokenizer_ge before load so its keys are not dropped as unexpected.
        if ckpt_has_tokenizer_ge and model.tokenizer_ge is None:
            model.freeze_tokenizer_ge(replace=True)
        model.load_state_dict(ckpt["model"], strict=False)
        if not ckpt_has_denoiser_pe:
            model.sync_denoiser_pos_embed_from_tokenizer()
            logger.info(
                "ckpt missing denoiser_pos_embed — initialized from register_pos_embed"
            )
        if "vggt_builder" in ckpt:
            load_state_dict_skip_mismatch(
                vggt_builder, ckpt["vggt_builder"], log_prefix="vggt_builder: "
            )
        ckpt_args = ckpt.get("args") or {}
        if isinstance(ckpt_args, dict) and "flow_steps_per_recon" in ckpt_args:
            start_step = int(ckpt.get("step", 0))
        elif not args.reset_step_on_resume:
            start_step = int(ckpt.get("step", 0))
        else:
            logger.info("Resetting step to 0 (resume from non-UNITE checkpoint)")

    if args.freeze_recon:
        _apply_recon_freeze(
            model,
            preserve_tokenizer_ge=ckpt_has_tokenizer_ge,
            preserve_denoiser_pe=ckpt_has_denoiser_pe and ckpt_has_tokenizer_ge,
        )
        if model.surflo_global_cam is not None:
            model.surflo_global_cam.freeze_backbone()
            model.surflo_global_cam.unfreeze_adapter()
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            "freeze_recon=True: decode skipped except every %d steps; "
            "trainable params=%s",
            args.recon_log_interval,
            f"{n_train:,}",
        )

    optimizer = torch.optim.AdamW(
        [p for p in list(model.parameters()) + list(vggt_builder.parameters()) if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = (
        start_step + int(args.additional_steps)
        if args.additional_steps is not None
        else args.num_steps
    )
    if args.additional_steps is not None:
        logger.info(
            "Training until step %d (%d additional steps)", total_steps, args.additional_steps
        )
    # Build the schedule only once total_steps is final; T_max used to come from
    # --num_steps even when --additional_steps shortened the run.
    remaining = max(total_steps - start_step, 1)
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=remaining, eta_min=args.lr * 0.01
        )
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=0
        )

    if ckpt is not None:
        if "optimizer" in ckpt and not args.reset_optimizer_on_resume:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except (ValueError, KeyError):
                logger.warning("Skipping optimizer state (incompatible checkpoint)")
        if "scheduler" in ckpt and start_step > 0 and args.lr_schedule == "cosine":
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except (ValueError, KeyError):
                logger.warning("Skipping scheduler state (incompatible checkpoint)")

    if args.wandb and wandb is not None:
        wandb.init(project=args.wandb_project, name=args.wandb_name or output_dir.name, config=vars(args))

    model.train()
    vggt_builder.train()
    model.ensure_tokenizer_ge_eval()
    step = start_step
    data_iter = iter(loader)
    t0 = time.time()
    zero = torch.zeros((), device=device)
    empty_extras = {
        "loss_cd": zero,
        "loss_rgb": zero,
        "loss_rgb_mean": zero,
        "loss_rgb_topk": zero,
        "loss_anc": zero,
        "loss_anc_cd": zero,
        "loss_delta": zero,
    }

    while step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        surface = batch["surface"].to(device, non_blocking=True)
        # Surflo global-cam keeps per-view cameras in the sequence; only the
        # legacy --adaln_camera_cond path removes them.
        include_cams_in_seq = not args.adaln_camera_cond
        weak_ctx, ctx_keep, cam_cond = _build_weak_context(
            batch,
            vggt_builder,
            device,
            null_weak_context=model.null_weak_context,
            use_null=args.null_weak_context,
            align_mode=args.align_mode,
            include_camera_in_sequence=include_cams_in_seq,
        )
        raw_cams = (
            extract_raw_vggt_camera_tokens(batch, device)
            if args.surflo_global_cam
            else None
        )

        next_step = step + 1
        log_recon = (
            (not args.freeze_recon)
            or step == start_step
            or (next_step % int(args.recon_log_interval) == 0)
            or (args.vis_interval > 0 and next_step % int(args.vis_interval) == 0)
        )

        xyz = rgb = centers = gt_xyz = gt_rgb = None
        if args.freeze_recon:
            with torch.no_grad():
                if log_recon:
                    z, fps_xyz, xyz, rgb, centers = model.forward_tokenizer(
                        surface,
                        weak_context=weak_ctx,
                        weak_context_keep=ctx_keep,
                    )
                    gt_xyz, gt_rgb = ShapePCAE.surface_gt_points(
                        surface,
                        include_sharp_label=include_sharp_label,
                        include_rgb=not args.geometry_only,
                    )
                    recon_loss, extras = criterion(
                        xyz,
                        rgb,
                        gt_xyz,
                        gt_rgb,
                        centers=centers,
                        fps_xyz=fps_xyz.detach(),
                    )
                    recon_loss = recon_loss.detach()
                    extras = {
                        k: (v.detach() if torch.is_tensor(v) else v)
                        for k, v in extras.items()
                    }
                else:
                    z, fps_xyz, _ = model.encode_tokenizer(
                        surface,
                        weak_context=weak_ctx,
                        weak_context_keep=ctx_keep,
                    )
                    recon_loss = zero
                    extras = empty_extras
        else:
            z, fps_xyz, xyz, rgb, centers = model.forward_tokenizer(
                surface,
                weak_context=weak_ctx,
                weak_context_keep=ctx_keep,
            )
            gt_xyz, gt_rgb = ShapePCAE.surface_gt_points(
                surface,
                include_sharp_label=include_sharp_label,
                include_rgb=not args.geometry_only,
            )
            recon_loss, extras = criterion(
                xyz, rgb, gt_xyz, gt_rgb, centers=centers, fps_xyz=fps_xyz.detach()
            )

        flow_out = model.forward_denoising(
            z,
            weak_ctx,
            context_keep=ctx_keep,
            cam_cond=cam_cond,
            raw_camera_tokens=raw_cams,
        )
        flow_loss = flow_out["flow/flow_loss"]
        # Under --freeze_recon recon is monitoring-only (detached); do not fold it
        # into the optimised loss so train/loss stays pure flow.
        if args.freeze_recon:
            loss = args.lambda_flow * flow_loss
        else:
            loss = args.lambda_recon * recon_loss + args.lambda_flow * flow_loss

        will_log = (step + 1) % args.log_interval == 0 or step == 0
        grad_norms = None
        # Per-term grad norms need a live graph; under --freeze_recon the recon
        # terms come from a no_grad block, so skip them instead of crashing.
        if (
            will_log
            and args.wandb
            and wandb is not None
            and args.log_grad_norms
            and not args.freeze_recon
        ):
            grad_norms = compute_pc_loss_grad_norms(model, criterion, extras=extras)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.max_grad_norm > 0:
            grad_norm_preclip = torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(vggt_builder.parameters()),
                args.max_grad_norm,
            )
        else:
            grad_norm_preclip = global_grad_norm(model)
        optimizer.step()
        scheduler.step()
        step += 1

        if will_log:
            keep_msg = ""
            if ctx_keep is not None:
                n_keep = ctx_keep.float().sum(dim=-1)
                keep_msg = (
                    f" keep=[{float(n_keep.min()):.0f},"
                    f"{float(n_keep.mean()):.0f},"
                    f"{float(n_keep.max()):.0f}]"
                    f" frac={float(ctx_keep.float().mean()):.2f}"
                )
            cd_val = float(extras["loss_cd"])
            logger.info(
                "step %d/%d loss=%.4f recon=%.4f flow=%.4f cd=%s%s (%.1fs)",
                step,
                total_steps,
                float(loss),
                float(recon_loss),
                float(flow_loss),
                f"{cd_val:.4f}" if log_recon or not args.freeze_recon else "n/a",
                keep_msg,
                time.time() - t0,
            )
            if args.wandb and wandb is not None:
                log_dict = {
                    "train/loss": float(loss),
                    "train/recon": float(recon_loss),
                    "flow/loss": float(flow_loss),
                    "flow/t_mean": float(flow_out["flow/t_mean"]),
                    "flow/velocity_mse": float(flow_out["flow/velocity_mse"]),
                    "flow/x_start_mse": float(flow_out["flow/x_start_mse"]),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "context/num_weak_tokens": (
                        float(weak_ctx.shape[1]) if weak_ctx is not None else 0.0
                    ),
                    "step": step,
                }
                if "flow/offpath_loss" in flow_out:
                    log_dict["flow/offpath_loss"] = float(flow_out["flow/offpath_loss"])
                if log_recon:
                    log_dict.update(
                        {
                            "train/cd": float(extras["loss_cd"]),
                            "train/rgb": float(extras["loss_rgb"]),
                            "train/rgb_mean": float(
                                extras.get("loss_rgb_mean", extras["loss_rgb"])
                            ),
                            "train/rgb_topk": float(extras.get("loss_rgb_topk", 0.0)),
                            "train/anc": float(extras["loss_anc"]),
                            "train/anc_cd": float(extras.get("loss_anc_cd", 0.0)),
                            "train/delta": float(extras.get("loss_delta", 0.0)),
                        }
                    )
                    log_dict.update(
                        loss_balance_ratios(
                            loss_cd=float(extras["loss_cd"]),
                            loss_rgb=float(extras["loss_rgb"]),
                            loss_anc=float(extras["loss_anc"]),
                            lambda_rgb=args.lambda_rgb,
                            lambda_anc=args.lambda_anc,
                            loss_anc_cd=float(extras.get("loss_anc_cd", 0.0)),
                            lambda_anc_cd=args.lambda_anc_cd,
                            loss_delta=float(extras.get("loss_delta", 0.0)),
                            lambda_delta=float(getattr(args, "lambda_delta", 0.0)),
                            loss_rgb_mean=float(extras.get("loss_rgb_mean", 0.0)),
                            loss_rgb_topk=float(extras.get("loss_rgb_topk", 0.0)),
                            rgb_topk_beta=float(getattr(args, "rgb_topk_beta", 0.0)),
                        )
                    )
                    if centers is not None:
                        log_dict.update(anchor_diagnostics(centers, fps_xyz))
                if ctx_keep is not None:
                    # Sanity: variable-length VGGT pad/keep (camera always kept).
                    n_keep = ctx_keep.float().sum(dim=-1)
                    log_dict["context/n_keep_mean"] = float(n_keep.mean())
                    log_dict["context/n_keep_min"] = float(n_keep.min())
                    log_dict["context/n_keep_max"] = float(n_keep.max())
                    log_dict["context/keep_frac"] = float(ctx_keep.float().mean())
                log_dict.update(latent_statistics(z))
                if grad_norms:
                    log_dict["grad_norm/total"] = float(
                        grad_norm_preclip.detach().item()
                        if torch.is_tensor(grad_norm_preclip)
                        else grad_norm_preclip
                    )
                    for k, v in grad_norms.items():
                        log_dict[f"grad_norm/{k}"] = v
                wandb.log(log_dict, step=step)

        if args.vis_interval > 0 and step % args.vis_interval == 0:
            vis = _log_visuals(
                model,
                batch,
                weak_ctx,
                gt_xyz=gt_xyz,
                gt_rgb=gt_rgb,
                xyz_recon=xyz,
                rgb_recon=rgb,
                fps_xyz=fps_xyz,
                centers=centers,
                output_dir=output_dir,
                step=step,
                sample_steps=args.vis_sample_steps,
                guidance_scale=args.vis_guidance_scale,
                export_ply=args.vis_export_ply,
                export_recon_debug=args.vis_export_recon_debug,
                intruder_thresh=args.intruder_thresh,
                context_keep=ctx_keep,
                cam_cond=cam_cond,
                raw_camera_tokens=raw_cams,
            )
            logger.info(
                "step %d visuals: gen_cd=%.4f null_cd=%.4f gain=%.4f → %s",
                step,
                vis["metrics"].get("vis/gen_cd", float("nan")),
                vis["metrics"].get("vis/gen_null_cd", float("nan")),
                vis["metrics"].get("vis/cond_gain", float("nan")),
                vis["ply_dir"] or "wandb only",
            )
            if args.wandb and wandb is not None:
                wandb.log({**vis["metrics"], **vis["objects"]}, step=step)

        if args.ckpt_interval > 0 and step % args.ckpt_interval == 0:
            _save_ckpt(
                output_dir / f"ckpt_{step:07d}.pt",
                step=step,
                model=model,
                vggt_builder=vggt_builder,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                include_optimizer=args.save_optimizer,
            )

    _save_ckpt(
        output_dir / "ckpt_final.pt",
        step=step,
        model=model,
        vggt_builder=vggt_builder,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args,
        include_optimizer=args.save_optimizer,
    )
    logger.info("Training done: %s", output_dir / "ckpt_final.pt")


def parse_args():
    p = argparse.ArgumentParser(description="Train ShapePCUnite")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="runs/pc_unite")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--categories", type=str, default=None)
    p.add_argument("--no_experiment_manifest", action="store_true")
    p.add_argument("--gobjaverse_render_root", type=str, default=None)
    p.add_argument("--vggt_cache_root", type=str, default=None)
    p.add_argument(
        "--vggt_joint_cache",
        action="store_true",
        help=(
            "Load joint multi-view VGGT cache from --vggt_cache_root. "
            "If len(view_indices)==views_per_sample, use that fixed ordered "
            "tuple. If the pool is larger, randomly sample an ordered "
            "views_per_sample-tuple each step (cache with --joint_pairs)."
        ),
    )
    p.add_argument("--view_idx", type=int, default=0, help="Single view (ignored if --num_views / --view_indices set).")
    p.add_argument(
        "--num_views",
        type=int,
        default=None,
        help="Expand dataset to (mesh, view) for views 0..N-1 (Design A multi-view).",
    )
    p.add_argument(
        "--view_indices",
        type=str,
        default=None,
        help='Explicit train views, e.g. "0-39" or "0,5,10". Overrides --num_views.',
    )
    p.add_argument(
        "--views_per_sample",
        type=int,
        default=1,
        help=(
            "How many views to use per object sample. 1 = Design A (mesh,view) pairs. "
            ">1 = one sample per mesh; randomly pick this many views from the pool "
            "(requires --align_mode c_meanrms; online multi-view VGGT PE unless "
            "--vggt_joint_cache)."
        ),
    )
    p.add_argument(
        "--no_gobjaverse_normalization",
        action="store_true",
        help="Normalize surfaces to a [-1,1] bbox instead of the render frame.",
    )
    p.add_argument(
        "--no_surface_camera_frame",
        action="store_true",
        help="Keep surfaces in object/world frame (default: transform to camera).",
    )
    p.add_argument(
        "--world_scale",
        type=float,
        default=1.0,
        help="Deprecated/ignored. Kept for CLI compatibility.",
    )
    p.add_argument(
        "--pe_frame",
        type=str,
        default="camera",
        choices=("camera",),
        help="Always camera-frame PE from VGGT depth (object-frame removed).",
    )
    p.add_argument(
        "--align_mode",
        type=str,
        default="c_meanrms",
        choices=("cross", "fair_gobK", "c_meanrms"),
        help=(
            "PE/GT align recipe. cross: PE=vggtK own bbox, GT=/mean(GT depth)+gobK bbox. "
            "fair_gobK: PE=gobK + shared gobK bbox with GT (ablation; PE OOD at infer). "
            "c_meanrms: C_gt_depth_filter_zrobust_meanrms (PE=vggtK∪ own mean+RMS; "
            "GT=erode1px GT-depth∪ mean+RMS); use with --views_per_sample>1."
        ),
    )

    p.add_argument("--num_latents", type=int, default=1024)
    p.add_argument("--num_registers", type=int, default=None)
    p.add_argument("--embed_dim", type=int, default=64)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--num_ge_layers", type=int, default=8)
    p.add_argument("--num_decoder_layers", type=int, default=4)
    p.add_argument("--num_points_per_anchor", type=int, default=8)
    p.add_argument("--pc_size", type=int, default=5120)
    p.add_argument("--pc_sharpedge_size", type=int, default=5120)
    p.add_argument("--downsample_ratio", type=int, default=20)
    p.add_argument(
        "--max_anchor_delta",
        type=float,
        default=None,
        help=(
            "Legacy hard clamp: |delta| ≤ max_anchor_delta via tanh. "
            "Default: unbound (prefer --lambda_delta soft reg)."
        ),
    )
    p.add_argument("--deterministic_encoder", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--register_noise_mode",
        type=str,
        default="random",
        choices=REGISTER_NOISE_MODES,
        help=(
            "Tokenizer register slots. 'random' (UNITE) redraws every call, which "
            "makes the flow's regression target a random variable — measured at "
            "~0.69 relative spread, with the cloud mean decoding ~10x worse than "
            "any sample. 'fixed' (frozen persistent draw) or 'zeros' make the "
            "latent a deterministic function of the surface."
        ),
    )
    p.add_argument(
        "--geometry_only",
        action="store_true",
        help=(
            "Drop colour end to end: encoder consumes Hunyuan3D's exact "
            "xyz|normals|sharp layout (point_feats=4, input_proj loads verbatim), "
            "the decoder emits xyz only, and the RGB loss is not computed."
        ),
    )
    p.add_argument(
        "--sample_renorm_output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply latent_norm to the ODE endpoint. The flow already targets "
            "normalized latents, so this is a second application; leave off "
            "(default) to match UNITE. Enable with --sample_renorm_output."
        ),
    )
    p.add_argument(
        "--tokenizer_use_weak_context",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Append VGGT weak-context tokens to tokenizer GE context in addition "
            "to Hunyuan local cross-attention tokens. Keeps the original tokenizer "
            "inputs intact and only adds weak tokens on top."
        ),
    )
    p.add_argument(
        "--adaln_camera_cond",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Ablation: move VGGT camera token from the attention sequence into "
            "AdaLN as c = t_emb + cam_emb (DiT/UNITE sum). Sequence becomes "
            "patches only. Tokenizer AdaLN uses a learned null camera (encode vs "
            "denoise mode). Default off = camera in sequence + timestep-only AdaLN."
        ),
    )
    p.add_argument(
        "--surflo_global_cam",
        action="store_true",
        help=(
            "Ablation: add Surflo frozen camera projector+compressor → zero-init "
            "512→width adapter into AdaLN (c = t_emb + adapter(global_cam)). "
            "Keeps per-view camera tokens in the weak-context sequence. "
            "Do not combine with --adaln_camera_cond."
        ),
    )
    p.add_argument(
        "--surflo_ckpt",
        type=str,
        default="/export/home/nathan/Surflo/checkpoints/surflo_v0.pt",
        help="Surflo checkpoint (ema_state) for camera projector+compressor weights.",
    )
    p.add_argument(
        "--surflo_root",
        type=str,
        default="/export/home/nathan/Surflo",
        help="Surflo repo root (used to import Compressor without full package deps).",
    )
    p.add_argument("--use_rope", action="store_true")
    p.add_argument("--point_feats", type=int, default=7)
    p.add_argument(
        "--include_sharp_label",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include Hunyuan sharp-edge label in surface feats (10ch layout → "
            "point_feats=7: normals|sharp|rgb). Needed for correct pretrained "
            "CA load: sharp stays sharp, all 3 RGB cols get rgb_feat_init. "
            "Default on; disable with --no-include_sharp_label."
        ),
    )

    p.add_argument("--pretrained_profile", type=str, default="none")
    p.add_argument(
        "--pretrained_load",
        type=str,
        default="cross_attn",
        choices=("none", "cross_attn"),
    )
    p.add_argument(
        "--pretrained_repo",
        type=str,
        default="tencent/Hunyuan3D-2mini",
    )
    p.add_argument(
        "--pretrained_subfolder",
        type=str,
        default="hunyuan3d-vae-v2-mini-withencoder",
    )
    p.add_argument("--rgb_feat_init", type=str, default="kaiming")

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--num_steps", type=int, default=5000)
    p.add_argument(
        "--additional_steps",
        type=int,
        default=None,
        help="When resuming, train this many steps beyond the checkpoint step (overrides --num_steps).",
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument(
        "--lr_schedule",
        type=str,
        default="cosine",
        choices=("cosine", "constant"),
        help="Use 'constant' for short continuations; a fresh cosine restart at "
        "peak LR is what blew up Exp 0.",
    )
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--lambda_rgb", type=float, default=10.0)
    p.add_argument(
        "--rgb_topk_frac",
        type=float,
        default=0.0,
        help=(
            "Fraction of worst per-point RGB L1 (Chamfer NN) averaged as a tail "
            "term. 0 disables. Typical frets experiment: 0.1."
        ),
    )
    p.add_argument(
        "--rgb_topk_beta",
        type=float,
        default=1.0,
        help="Weight of top-k RGB residual: L_rgb = mean + beta * topk_mean.",
    )
    p.add_argument("--lambda_anc", type=float, default=0.1)
    p.add_argument(
        "--lambda_anc_cd",
        type=float,
        default=10.0,
        help=(
            "Weight for bidirectional Chamfer between decoder anchors and FPS "
            "(hard NN geometric pin; complements Sinkhorn λ_anc)."
        ),
    )
    p.add_argument(
        "--lambda_delta",
        type=float,
        default=0.05,
        help=(
            "Soft mean ||local-center||^2 regularizer (replaces hard max_anchor_delta)."
        ),
    )
    p.add_argument("--lambda_flow", type=float, default=1.0)
    p.add_argument("--sinkhorn_eps", type=float, default=0.02)
    p.add_argument("--sinkhorn_iters", type=int, default=50)
    p.add_argument("--flow_steps_per_recon", type=int, default=8)
    p.add_argument(
        "--flow_loss_type",
        type=str,
        default="velocity",
        choices=("velocity", "x_start"),
        help="velocity = UNITE form (1/(1-t)^2 weighting, spiky); x_start = "
        "unweighted MSE on the clean latent.",
    )
    p.add_argument(
        "--offpath_mode",
        type=str,
        default="none",
        choices=("none", "noise", "onestep"),
        help=(
            "Exposure-bias / path-drift robustness (aux loss; main FM unchanged). "
            "none: disabled (default). "
            "noise: mode A — jitter interpolant xt, regress velocity to x1. "
            "onestep: mode B — one stop-grad Euler step, then regress from x' at t'."
        ),
    )
    p.add_argument(
        "--offpath_weight",
        type=float,
        default=1.0,
        help="Weight of the auxiliary off-path loss relative to the main flow loss.",
    )
    p.add_argument(
        "--offpath_noise_std",
        type=float,
        default=0.1,
        help="Mode A: std of isotropic Gaussian noise added to xt.",
    )
    p.add_argument(
        "--offpath_step_size",
        type=float,
        default=0.05,
        help="Mode B: Euler dt (also clamped by remaining 1-t-train_eps).",
    )
    p.add_argument(
        "--use_lognorm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Sample flow timesteps from a logit-normal (PointDiT / SD3 style) "
        "instead of Uniform(0,1).",
    )
    p.add_argument(
        "--lognorm_mu",
        type=float,
        default=-0.8,
        help="Logit-normal location (PointDiT uses -0.8).",
    )
    p.add_argument(
        "--lognorm_sigma",
        type=float,
        default=0.8,
        help="Logit-normal scale (PointDiT uses 0.8).",
    )
    p.add_argument(
        "--t0_force_prob",
        type=float,
        default=0.0,
        help="With this probability override the sampled flow t to exact 0 "
        "(PointDiT uses 0.1). At t=0 velocity and x_start losses coincide.",
    )
    p.add_argument(
        "--timestep_shift_alpha",
        type=float,
        default=0.0,
        help="Optional transport timestep shift (0 disables). a<1 biases early-t.",
    )
    p.add_argument(
        "--modulation_recon_timestep_max",
        type=float,
        default=0.01,
        help="Tokenizer AdaLN time ~ Uniform(0, max). Set 0 for exact t=0.",
    )
    p.add_argument("--noising_t_start", type=float, default=0.9)
    p.add_argument(
        "--representation_noising",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Noise latents before tokenizer decode (UNITE representation phase). "
            "Disable with --no-representation_noising for cleaner overfits."
        ),
    )
    p.add_argument("--weak_context_dropout", type=float, default=0.1)
    p.add_argument("--null_weak_context", action="store_true")
    p.add_argument("--freeze_recon", action="store_true")
    p.add_argument(
        "--recon_log_interval",
        type=int,
        default=1000,
        help=(
            "Under --freeze_recon, run full tokenizer decode + PC loss only every "
            "N steps (and on the first / vis steps) for monitoring. Encode-only "
            "otherwise. Ignored when recon is trained."
        ),
    )
    p.add_argument("--freeze_vggt_builder", action="store_true")
    p.add_argument("--lambda_recon", type=float, default=1.0)

    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--ckpt_interval", type=int, default=1000)
    p.add_argument("--vis_interval", type=int, default=500)
    p.add_argument("--vis_sample_steps", type=int, default=20)
    p.add_argument("--vis_guidance_scale", type=float, default=1.0)
    p.add_argument("--vis_export_ply", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--vis_export_recon_debug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write fps/anchors/error/intruder debug PLYs under vis/step_*.",
    )
    p.add_argument(
        "--intruder_thresh",
        type=float,
        default=0.02,
        help="Recon points farther than this from GT go into intruders.ply.",
    )
    p.add_argument("--log_grad_norms", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--save_optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Optimizer state roughly triples checkpoint size; disable for sweeps.",
    )
    p.add_argument("--resume_ckpt", type=str, default=None)
    p.add_argument(
        "--reset_optimizer_on_resume",
        action="store_true",
        help="Ignore optimizer state in the checkpoint.",
    )
    p.add_argument(
        "--reset_step_on_resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When resuming from a ShapePCAE (non-UNITE) checkpoint, start step counter at 0.",
    )
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="shapepcunite")
    p.add_argument("--wandb_name", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    train(args)
