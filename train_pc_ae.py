#!/usr/bin/env python3
"""Train UNITE-style ShapePCAE (point-cloud AE).

Does not require GT RGBD renders — only mesh surfaces.

Usage:
    python train_pc_ae.py --data_dir /path/to/meshes --output_dir runs/pc_ae \\
        --max_items 1 --num_steps 2000 --deterministic_encoder

Defaults match the ShapePCAE plan: L=R=1024, embed_dim=64, GE=8, decoder=4, K=8.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import torch
from torch.utils.data import DataLoader, Dataset

try:
    import wandb
except ImportError:
    wandb = None

from hy3dgen.shapegen.gs_export import export_xyz_pointcloud_ply
from hy3dgen.shapegen.models.autoencoders.shape_pc_ae import ShapePCAE
from hy3dgen.shapegen.pc_losses import PointCloudAELoss
from hy3dgen.shapegen.pretrained_profiles import (
    apply_pretrained_profile,
    get_pretrained_profile,
    resolve_include_sharp_label,
)
from hy3dgen.shapegen.surface_loaders import RGBSharpEdgeSurfaceLoader
from train_gs_ae import (
    load_experiment_manifest,
    resolve_category_ids,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Surface-only dataset (no RGBD GT required)
# ---------------------------------------------------------------------------

def discover_mesh_paths(
    data_dir: str,
    categories: Optional[Set[str]] = None,
    max_items: Optional[int] = None,
    use_experiment_manifest: bool = True,
) -> List[str]:
    data_path = Path(data_dir).resolve()
    manifest = load_experiment_manifest(str(data_path)) if use_experiment_manifest else None
    if manifest is not None:
        split_key = (
            "train_mesh_paths"
            if data_path.name == "train"
            else ("val_mesh_paths" if data_path.name == "val" else None)
        )
        if split_key and manifest.get(split_key):
            paths = list(manifest[split_key])
            if max_items is not None:
                paths = paths[:max_items]
            return paths

    mesh_paths: List[str] = []
    for folder in sorted(data_path.iterdir()):
        if not folder.is_dir():
            continue
        if categories is not None:
            cat_id = folder.name.split("_", 1)[0]
            if cat_id not in categories:
                continue
        obj_path = folder / "model_normalized.obj"
        if obj_path.exists():
            mesh_paths.append(str(obj_path.resolve()))
            continue
        # Also accept loose mesh files in the folder
        for ext in ("*.glb", "*.obj", "*.ply"):
            for p in sorted(folder.glob(ext)):
                mesh_paths.append(str(p.resolve()))
    # Loose files directly under data_dir
    for ext in ("*.glb", "*.obj", "*.ply"):
        for p in sorted(data_path.glob(ext)):
            mesh_paths.append(str(p.resolve()))

    if max_items is not None:
        mesh_paths = mesh_paths[:max_items]
    return mesh_paths


class SurfaceOnlyDataset(Dataset):
    """Mesh → colored surface point cloud (xyz|normals|rgb[|sharp])."""

    def __init__(
        self,
        data_dir: str,
        pc_size: int = 5120,
        pc_sharpedge_size: int = 5120,
        max_items: Optional[int] = None,
        categories: Optional[Set[str]] = None,
        seed: Optional[int] = None,
        include_sharp_label: bool = False,
        use_experiment_manifest: bool = True,
    ):
        self.loader = RGBSharpEdgeSurfaceLoader(
            num_uniform_points=pc_size,
            num_sharp_points=pc_sharpedge_size,
            seed=seed,
            include_sharp_label=include_sharp_label,
        )
        self.include_sharp_label = include_sharp_label
        self.mesh_paths = discover_mesh_paths(
            data_dir,
            categories=categories,
            max_items=max_items,
            use_experiment_manifest=use_experiment_manifest,
        )
        if not self.mesh_paths:
            raise FileNotFoundError(f"No meshes found under {data_dir}")
        logger.info("SurfaceOnlyDataset: %d mesh(es) in %s", len(self.mesh_paths), data_dir)
        self._ram_cache: Dict[int, Dict] = {}

    def __len__(self) -> int:
        return len(self.mesh_paths)

    def __getitem__(self, idx: int) -> Dict:
        if idx in self._ram_cache:
            sample = self._ram_cache[idx]
            return {
                "surface": sample["surface"].clone(),
                "mesh_path": sample["mesh_path"],
            }

        # Iterate forward (non-recursively) to find a working sample — same
        # pattern as MeshDataset in train_gs_ae (skip meshes with too few
        # sharp edges, corrupt GLBs, etc.).
        for attempt in range(len(self.mesh_paths)):
            path = self.mesh_paths[(idx + attempt) % len(self.mesh_paths)]
            try:
                surface = self.loader(path).squeeze(0)
                out = {"surface": surface, "mesh_path": path}
                self._ram_cache[idx] = out
                if len(self._ram_cache) == len(self.mesh_paths):
                    logger.info(
                        "SurfaceOnlyDataset RAM cache full (%d samples)",
                        len(self._ram_cache),
                    )
                elif len(self._ram_cache) == 1:
                    logger.info(
                        "SurfaceOnlyDataset RAM cache: loading samples on first access"
                    )
                return {"surface": surface.clone(), "mesh_path": path}
            except Exception as e:
                logger.warning("Skipping %s: %s", path, e)
        raise RuntimeError(f"All {len(self.mesh_paths)} meshes failed to load")


def collate_surface(batch):
    return {
        "surface": torch.stack([b["surface"] for b in batch], dim=0),
        "mesh_path": [b["mesh_path"] for b in batch],
    }


def _apply_ca_profile_flags(args) -> None:
    """Apply Hunyuan profile flags needed for CA weight load without forcing L/R."""
    profile_name = getattr(args, "pretrained_profile", "none") or "none"
    if profile_name == "none":
        return
    profile = get_pretrained_profile(profile_name)
    for key in (
        "qk_norm",
        "qkv_bias",
        "include_pi",
        "point_feats",
        "pc_size",
        "pc_sharpedge_size",
        "downsample_ratio",
        "shapevae_point_feats",
        "use_safetensors",
    ):
        if key in profile:
            setattr(args, key, profile[key])
    if not getattr(args, "pretrained_repo", None):
        args.pretrained_repo = profile["pretrained_repo"]
    if not getattr(args, "pretrained_subfolder", None):
        args.pretrained_subfolder = profile["pretrained_subfolder"]
    if getattr(args, "use_profile_latents", False):
        apply_pretrained_profile(args)


def train(args):
    _apply_ca_profile_flags(args)
    include_sharp_label = resolve_include_sharp_label(args)
    point_feats = int(getattr(args, "point_feats", 6 if not include_sharp_label else 7))
    if include_sharp_label and point_feats < 7:
        point_feats = 7
        args.point_feats = 7

    categories = resolve_category_ids(args.categories)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Training ShapePCAE on %s", device)

    num_registers = (
        int(args.num_registers) if args.num_registers is not None else int(args.num_latents)
    )

    model = ShapePCAE(
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
        max_anchor_delta=args.max_anchor_delta,
        qk_norm=bool(getattr(args, "qk_norm", True)),
        qkv_bias=bool(getattr(args, "qkv_bias", True)),
        include_pi=bool(getattr(args, "include_pi", True)),
    ).to(device)

    if args.pretrained_load == "cross_attn" and not args.resume_ckpt:
        repo = args.pretrained_repo
        if not repo:
            raise ValueError("--pretrained_load=cross_attn requires --pretrained_repo or --pretrained_profile")
        model.load_shapevae_cross_attn(
            repo,
            rgb_feat_init=args.rgb_feat_init,
            subfolder=args.pretrained_subfolder or "hunyuan3d-vae-v2-mini-withencoder",
            shapevae_point_feats=int(getattr(args, "shapevae_point_feats", 4)),
            use_safetensors=bool(getattr(args, "use_safetensors", False)),
            strict_geometry=False,
        )

    if args.freeze_encoder:
        model.freeze_modules(encoder=True)
        logger.info("Frozen encoder (CA + input_proj)")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "ShapePCAE trainable: %.3fM (L=%d R=%d K=%d GE=%d Dec=%d dim=%d)",
        n_params / 1e6,
        args.num_latents,
        num_registers,
        args.num_points_per_anchor,
        args.num_ge_layers,
        args.num_decoder_layers,
        args.embed_dim,
    )

    criterion = PointCloudAELoss(
        lambda_rgb=args.lambda_rgb,
        lambda_anc=args.lambda_anc,
        bidirectional_rgb=True,
        sinkhorn_eps=args.sinkhorn_eps,
        sinkhorn_iters=args.sinkhorn_iters,
    )

    dataset = SurfaceOnlyDataset(
        data_dir=args.data_dir,
        pc_size=args.pc_size,
        pc_sharpedge_size=args.pc_sharpedge_size,
        max_items=args.max_items,
        categories=categories,
        seed=args.seed,
        include_sharp_label=include_sharp_label,
        use_experiment_manifest=not args.no_experiment_manifest,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_surface,
        pin_memory=(device.type == "cuda"),
        drop_last=len(dataset) >= args.batch_size,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = args.num_steps
    warmup_steps = args.warmup_steps
    if warmup_steps > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=args.lr * 0.01
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=args.lr * 0.01
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    start_step = 0
    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        logger.info("Resumed from %s at step %d", args.resume_ckpt, start_step)

    if args.wandb and wandb is not None:
        wandb.init(project=args.wandb_project, name=args.wandb_name or output_dir.name, config=vars(args))

    model.train()
    step = start_step
    data_iter = iter(loader)
    t0 = time.time()
    while step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        surface = batch["surface"].to(device, non_blocking=True)
        xyz, rgb, centers, fps_xyz, _latents = model(surface)
        gt_xyz, gt_rgb = ShapePCAE.surface_gt_points(
            surface, include_sharp_label=include_sharp_label
        )
        loss, extras = criterion(
            xyz, rgb, gt_xyz, gt_rgb, centers=centers, fps_xyz=fps_xyz.detach()
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        step += 1

        if step % args.log_interval == 0 or step == 1:
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "step %d/%d loss=%.4f cd=%.4f rgb=%.4f anc=%.4f lr=%.2e (%.1fs)",
                step,
                total_steps,
                float(loss),
                float(extras["loss_cd"]),
                float(extras["loss_rgb"]),
                float(extras["loss_anc"]),
                lr,
                time.time() - t0,
            )
            if args.wandb and wandb is not None:
                wandb.log(
                    {
                        "train/loss": float(loss),
                        "train/cd": float(extras["loss_cd"]),
                        "train/rgb": float(extras["loss_rgb"]),
                        "train/anc": float(extras["loss_anc"]),
                        "train/lr": lr,
                        "step": step,
                    },
                    step=step,
                )

        if args.vis_interval > 0 and step % args.vis_interval == 0:
            vis_dir = output_dir / "vis" / f"step_{step:07d}"
            vis_dir.mkdir(parents=True, exist_ok=True)
            with torch.no_grad():
                export_xyz_pointcloud_ply(
                    xyz[0].detach().cpu(),
                    vis_dir / "pred.ply",
                    colors=rgb[0].detach().cpu(),
                )
                export_xyz_pointcloud_ply(
                    gt_xyz[0].detach().cpu(),
                    vis_dir / "gt.ply",
                    colors=gt_rgb[0].detach().cpu(),
                )
                export_xyz_pointcloud_ply(
                    centers[0].detach().cpu(),
                    vis_dir / "anchors.ply",
                )
                export_xyz_pointcloud_ply(
                    fps_xyz[0].detach().cpu(),
                    vis_dir / "fps.ply",
                )

        if args.ckpt_interval > 0 and step % args.ckpt_interval == 0:
            ckpt_path = output_dir / f"ckpt_{step:07d}.pt"
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                },
                ckpt_path,
            )
            logger.info("Saved %s", ckpt_path)

    final_path = output_dir / "ckpt_final.pt"
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        },
        final_path,
    )
    logger.info("Training done. Final checkpoint: %s", final_path)


def parse_args():
    p = argparse.ArgumentParser(description="Train ShapePCAE (UNITE-style PC AE)")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="runs/pc_ae")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--categories", type=str, default=None)
    p.add_argument("--no_experiment_manifest", action="store_true")

    # Architecture (plan defaults)
    p.add_argument("--num_latents", type=int, default=1024, help="FPS local query count L")
    p.add_argument(
        "--num_registers",
        type=int,
        default=None,
        help="Register count R (default: same as --num_latents)",
    )
    p.add_argument("--embed_dim", type=int, default=64)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--num_ge_layers", type=int, default=8)
    p.add_argument("--num_decoder_layers", type=int, default=4)
    p.add_argument("--num_points_per_anchor", type=int, default=8)
    p.add_argument("--pc_size", type=int, default=5120)
    p.add_argument("--pc_sharpedge_size", type=int, default=5120)
    p.add_argument("--downsample_ratio", type=int, default=20)
    p.add_argument("--max_anchor_delta", type=float, default=0.1)
    p.add_argument(
        "--deterministic_encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--qk_norm", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--qkv_bias", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include_pi", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--point_feats", type=int, default=6)
    p.add_argument(
        "--include_sharp_label",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    # Pretrained CA
    p.add_argument(
        "--pretrained_profile",
        type=str,
        default="none",
        choices=("none", "hunyuan_mini", "hunyuan_full"),
    )
    p.add_argument(
        "--use_profile_latents",
        action="store_true",
        help="Also adopt profile num_latents (default: keep CLI L/R).",
    )
    p.add_argument(
        "--pretrained_load",
        type=str,
        default="none",
        choices=("none", "cross_attn"),
    )
    p.add_argument("--pretrained_repo", type=str, default=None)
    p.add_argument("--pretrained_subfolder", type=str, default=None)
    p.add_argument("--rgb_feat_init", type=str, default="kaiming",
                   choices=("kaiming", "zero", "label_scale"))
    p.add_argument("--shapevae_point_feats", type=int, default=4)
    p.add_argument("--use_safetensors", action="store_true")
    p.add_argument("--freeze_encoder", action="store_true")

    # Optim / loop
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--num_steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--lambda_rgb", type=float, default=1.0)
    p.add_argument("--lambda_anc", type=float, default=0.1)
    p.add_argument(
        "--sinkhorn_eps",
        type=float,
        default=0.02,
        help="Entropic regularization for Sinkhorn anchor matching loss",
    )
    p.add_argument(
        "--sinkhorn_iters",
        type=int,
        default=50,
        help="Sinkhorn iterations for anchor matching loss",
    )
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--ckpt_interval", type=int, default=1000)
    p.add_argument("--vis_interval", type=int, default=500)
    p.add_argument("--resume_ckpt", type=str, default=None)

    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="shapepcae")
    p.add_argument("--wandb_name", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    train(args)
