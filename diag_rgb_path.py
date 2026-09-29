#!/usr/bin/env python3
"""Train-free diagnostics: does ShapePCUnite recon use surface RGB for frets?

Focuses on a single mesh (default: grid door 33b463c971e9). Ablations:
  baseline      — true surface RGB
  zero_rgb      — RGB feats set to 0
  grey_rgb      — RGB feats set to 0.5
  shuffle_rgb   — permute RGB across points (geometry fixed)
  invert_rgb    — 1 - RGB
  pink_frets    — frets painted neon pink (high-grad / high chrominance)

Also: FPS colour export, frets vs panel RGB loss mass, input_proj RGB col norms,
encoder CA-token linear probe for frets on FPS colours.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from hy3dgen.shapegen.gs_export import export_xyz_pointcloud_ply, surface_rgb_slice
from hy3dgen.shapegen.models.autoencoders.shape_pc_ae import ShapePCAE
from hy3dgen.shapegen.pc_debug_export import export_recon_debug_plys, nn_distances, nn_indices, scalar_to_rgb
from hy3dgen.shapegen.pc_losses import chamfer_distance, rgb_l1_on_nn
from hy3dgen.shapegen.pc_render_dataset import build_surface_render_dataset, collate_surface_render
from evaluate_pc_unite import load_unite_model
from train_gs_ae import load_experiment_manifest, resolve_category_ids

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _rgb_slice(surface: torch.Tensor, include_sharp: bool) -> slice:
    return surface_rgb_slice(surface.shape[-1], include_sharp_label=include_sharp)


def modify_surface_rgb(
    surface: torch.Tensor,
    *,
    mode: str,
    include_sharp: bool,
    fret_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return a copy of surface with RGB channels modified (other feats fixed)."""
    out = surface.clone()
    sl = _rgb_slice(out, include_sharp)
    rgb = out[..., sl]
    if mode == "baseline":
        pass
    elif mode == "zero_rgb":
        rgb.zero_()
    elif mode == "grey_rgb":
        rgb.fill_(0.5)
    elif mode == "shuffle_rgb":
        B, N, _ = rgb.shape
        for b in range(B):
            perm = torch.randperm(N, device=rgb.device)
            rgb[b] = rgb[b, perm]
    elif mode == "invert_rgb":
        rgb.copy_(1.0 - rgb)
    elif mode == "pink_frets":
        if fret_mask is None:
            raise ValueError("pink_frets needs fret_mask")
        # Neon pink on frets only; leave panels as-is.
        pink = rgb.new_tensor([1.0, 0.05, 0.85])
        m = fret_mask.to(device=rgb.device, dtype=rgb.dtype).reshape(1, -1, 1)
        rgb.copy_(rgb * (1.0 - m) + pink * m)
    else:
        raise ValueError(f"Unknown mode {mode}")
    return out


def fret_mask_from_rgb(
    rgb: torch.Tensor,
    *,
    chroma_thresh: float = 0.08,
    brown_bias: bool = True,
    mode: str = "chroma",
) -> torch.Tensor:
    """Heuristic frets: non-neutral colour (or brownish vs pale panels).

    ``rgb``: [B,N,3] or [N,3] in [0,1].
    mode:
      chroma — global chroma / warm (can over-tag wood)
      panel_outlier — far from robust panel colour (median of low-chroma pts)
      edge — local NN colour jump (thin frets stand out)
    """
    if rgb.ndim == 2:
        rgb = rgb.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    r, g, b = rgb.unbind(-1)
    chroma = rgb.max(dim=-1).values - rgb.min(dim=-1).values
    value = rgb.mean(dim=-1)
    if mode == "chroma":
        mask = chroma > chroma_thresh
        if brown_bias:
            warm = (r > b + 0.02) & (r > g - 0.05)
            pale = (value > 0.75) & (chroma < chroma_thresh * 1.5)
            mask = (mask | warm) & ~pale
    elif mode == "panel_outlier":
        # Panel = low-chroma points; frets = high L1 distance to panel median colour.
        pale = chroma[0] < max(chroma_thresh, float(chroma[0].quantile(0.35)))
        if pale.sum() < 10:
            panel_med = rgb[0].median(dim=0).values
        else:
            panel_med = rgb[0, pale].median(dim=0).values
        dist = (rgb[0] - panel_med).abs().mean(dim=-1)
        thr = float(dist.quantile(0.70))  # top 30% as candidate frets
        thr = max(thr, 0.04)
        mask = (dist > thr).unsqueeze(0)
    elif mode == "edge":
        # Build on xyz not available here — caller should use edge_mask_xyz_rgb.
        mask = chroma > chroma_thresh
    else:
        raise ValueError(mode)
    return mask[0] if squeeze else mask


def edge_mask_xyz_rgb(
    xyz: torch.Tensor,
    rgb: torch.Tensor,
    *,
    k: int = 16,
    q: float = 0.85,
) -> torch.Tensor:
    """Mark points whose mean colour difference to k NN exceeds high quantile.

    Thin frets have higher local colour contrast than smooth panels.
    ``xyz``/``rgb``: [N,3].
    """
    x = xyz.float()
    c = rgb.float()
    # subsample for knn if huge — still N=10240 is ok for cdist on GPU
    d2 = torch.cdist(x.unsqueeze(0), x.unsqueeze(0), p=2)[0]
    # exclude self
    d2.fill_diagonal_(float("inf"))
    knn_idx = d2.topk(k, largest=False, dim=1).indices  # [N,k]
    neigh = c[knn_idx]  # [N,k,3]
    jump = (neigh - c.unsqueeze(1)).abs().mean(dim=-1).mean(dim=-1)  # [N]
    thr = float(jump.quantile(q))
    return jump > thr


def per_point_rgb_l1_gt_to_recon(
    gt_xyz: torch.Tensor,
    gt_rgb: torch.Tensor,
    recon_xyz: torch.Tensor,
    recon_rgb: torch.Tensor,
) -> torch.Tensor:
    """[N_gt] mean |rgb - recon_nn| for each GT point."""
    idx = nn_indices(
        gt_xyz[0].detach().cpu().numpy(),
        recon_xyz[0].detach().cpu().numpy(),
    )
    g = gt_rgb[0].detach().cpu().numpy()
    r = recon_rgb[0].detach().cpu().numpy()
    err = np.abs(g - r[idx]).mean(axis=-1)
    return torch.from_numpy(err.astype(np.float32))


def fps_colours_from_surface(
    surface: torch.Tensor,
    fps_xyz: torch.Tensor,
    include_sharp: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """NN-match FPS positions onto surface points → rgb [B,L,3] and indices."""
    pc = surface[:, :, :3]
    sl = _rgb_slice(surface, include_sharp)
    srgb = surface[:, :, sl]
    # [B,L,N]
    d2 = torch.cdist(fps_xyz.float(), pc.float(), p=2).pow(2)
    idx = d2.min(dim=2).indices  # [B,L]
    B, L = idx.shape
    C = srgb.shape[-1]
    gather_idx = idx.unsqueeze(-1).expand(B, L, C)
    fps_rgb = torch.gather(srgb, 1, gather_idx)
    return fps_rgb, idx


def linear_probe_binary(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    n_iters: int = 400,
    lr: float = 0.05,
) -> Dict[str, float]:
    """Ridge-free logistic probe (single linear layer, no train of AE).

    ``features`` [N,D], ``labels`` [N] float {0,1}.
    """
    with torch.enable_grad():
        x = features.detach().float()
        y = labels.detach().float().view(-1, 1)
        if y.sum() < 3 or (1 - y).sum() < 3:
            return {
                "accuracy": float("nan"),
                "auroc_proxy": float("nan"),
                "n_pos": float(y.sum()),
            }
        x = (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + 1e-6)
        D = x.shape[1]
        w = torch.zeros(D, 1, device=x.device, requires_grad=True)
        b = torch.zeros(1, 1, device=x.device, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=lr)
        loss = None
        for _ in range(n_iters):
            logit = x @ w + b
            loss = F.binary_cross_entropy_with_logits(logit, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            logit = x @ w + b
            pred = (logit.sigmoid() > 0.5).float()
            acc = float((pred == y).float().mean())
            pos = logit[y.view(-1) > 0.5].view(-1)
            neg = logit[y.view(-1) < 0.5].view(-1)
            if pos.numel() and neg.numel():
                n = min(4000, max(pos.numel() * 4, 100))
                pi = torch.randint(0, pos.numel(), (n,), device=x.device)
                ni = torch.randint(0, neg.numel(), (n,), device=x.device)
                auroc = float((pos[pi] > neg[ni]).float().mean())
            else:
                auroc = float("nan")
    return {
        "accuracy": acc,
        "auroc_proxy": auroc,
        "n_pos": float(y.sum()),
        "n_neg": float((1 - y).sum()),
        "final_bce": float(loss.detach()) if loss is not None else float("nan"),
    }


def input_proj_rgb_norms(model, include_sharp: bool) -> Dict[str, float]:
    w = model.encoder.input_proj.weight.detach().float()  # [width, fourier+feats]
    # feats layout: normals(3) [| sharp(1)] | rgb(3)
    point_feats = int(model.point_feats)
    fourier_dim = w.shape[1] - point_feats
    if include_sharp and point_feats >= 7:
        rgb_start = fourier_dim + 4
    else:
        rgb_start = fourier_dim + 3
    norms = {
        "fourier_col_mean_norm": float(w[:, :fourier_dim].norm(dim=0).mean()),
        "normal_col_mean_norm": float(w[:, fourier_dim : fourier_dim + 3].norm(dim=0).mean()),
        "rgb_col_mean_norm": float(w[:, rgb_start : rgb_start + 3].norm(dim=0).mean()),
    }
    if include_sharp and point_feats >= 7:
        norms["sharp_col_norm"] = float(w[:, fourier_dim + 3].norm())
    # relative
    norms["rgb_over_normal"] = norms["rgb_col_mean_norm"] / max(norms["normal_col_mean_norm"], 1e-8)
    return norms


@torch.no_grad()
def run(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, _vggt, train_args = load_unite_model(args.ckpt, device)
    model.representation_noising = False
    include_sharp = bool(train_args.get("include_sharp_label") or False)

    data_path = Path(args.data_dir or train_args["data_dir"]).resolve()
    manifest = load_experiment_manifest(str(data_path))
    categories = resolve_category_ids(None)
    dataset = build_surface_render_dataset(
        str(data_path),
        max_items=int(train_args.get("max_items") or args.max_items or 8),
        categories=categories,
        include_sharp_label=include_sharp,
        use_experiment_manifest=True,
        manifest=manifest,
        render_root=args.gobjaverse_render_root or train_args.get("gobjaverse_render_root"),
        view_idx=int(args.view_idx),
        vggt_cache_root=args.vggt_cache_root or train_args.get("vggt_cache_root"),
        pc_size=int(train_args.get("pc_size", 5120)),
        pc_sharpedge_size=int(train_args.get("pc_sharpedge_size", 5120)),
        gobjaverse_normalization=not train_args.get("no_gobjaverse_normalization", False),
        surface_in_camera_frame=not train_args.get("no_surface_camera_frame", False),
        align_mode=train_args.get("align_mode", "cross"),
    )

    # Locate target mesh
    target = args.mesh_substr
    idx = None
    mesh_path = None
    for i in range(len(dataset)):
        # dataset items: resolve mesh path
        try:
            p = dataset.samples[i] if hasattr(dataset, "samples") else None
        except Exception:
            p = None
        # collate one-by-one
        batch = collate_surface_render([dataset[i]])
        mp = batch.get("mesh_path", [None])[0]
        if mp is None and "mesh" in batch:
            mp = batch["mesh"][0]
        stem = Path(str(mp)).stem if mp else ""
        if target in stem or target in str(mp):
            idx = i
            mesh_path = str(mp)
            break
    if idx is None:
        # fallback: dump stems
        stems = []
        for i in range(len(dataset)):
            batch = collate_surface_render([dataset[i]])
            mp = batch.get("mesh_path", ["?"])[0]
            stems.append(str(mp))
        raise SystemExit(f"Mesh containing {target!r} not found. Stems: {stems}")

    logger.info("Target idx=%d mesh=%s", idx, mesh_path)
    batch = collate_surface_render([dataset[idx]])
    surface0 = batch["surface"].to(device)
    gt_xyz, gt_rgb = ShapePCAE.surface_gt_points(surface0, include_sharp_label=include_sharp)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fixed register noise for all ablations (fair A/B)
    torch.manual_seed(0)
    register_noise = torch.randn(
        1, model.num_registers, model.embed_dim, device=device, dtype=surface0.dtype
    )

    frets_gt = edge_mask_xyz_rgb(gt_xyz[0], gt_rgb[0], k=16, q=0.88)
    frets_chroma = fret_mask_from_rgb(gt_rgb[0], chroma_thresh=args.chroma_thresh, mode="panel_outlier")
    # Prefer edge mask for frets; report both
    n_fret = int(frets_gt.sum())
    n_tot = int(frets_gt.numel())
    logger.info(
        "GT frets edge-mask: %d / %d (%.2f%%); panel_outlier: %d (%.2f%%)",
        n_fret,
        n_tot,
        100.0 * n_fret / max(n_tot, 1),
        int(frets_chroma.sum()),
        100.0 * float(frets_chroma.float().mean()),
    )
    export_xyz_pointcloud_ply(
        gt_xyz[0].cpu(),
        out_dir / "gt_frets_edge_mask.ply",
        colors=torch.stack(
            [
                frets_gt.float().cpu(),
                torch.zeros(n_tot),
                (~frets_gt).float().cpu(),
            ],
            dim=-1,
        ),
    )

    report: Dict = {
        "ckpt": str(args.ckpt),
        "mesh": mesh_path,
        "mesh_substr": target,
        "include_sharp_label": include_sharp,
        "n_gt_points": n_tot,
        "n_fret_points": n_fret,
        "fret_frac": n_fret / max(n_tot, 1),
        "n_fret_panel_outlier": int(frets_chroma.sum()),
        "fret_mask": "edge_xyz_rgb_q0.88",
        "chroma_thresh": args.chroma_thresh,
        "ablations": {},
        "input_proj_norms": input_proj_rgb_norms(model, include_sharp),
    }

    modes = ["baseline", "zero_rgb", "grey_rgb", "shuffle_rgb", "invert_rgb", "pink_frets"]
    baseline_out = None

    for mode in modes:
        if mode == "pink_frets":
            # frets mask over full surface points (same layout as gt)
            surface = modify_surface_rgb(
                surface0,
                mode=mode,
                include_sharp=include_sharp,
                fret_mask=frets_gt.to(device),
            )
        else:
            surface = modify_surface_rgb(
                surface0, mode=mode, include_sharp=include_sharp
            )
        # encode/decode with requires_grad off
        with torch.no_grad():
            z, fps_xyz, _ = model.encode_tokenizer(surface, register_noise=register_noise)
            xyz, rgb, centers = model.decode(z, representation_phase=False)
            cd, idx_p2t, idx_t2p = chamfer_distance(xyz, gt_xyz)
            # Score against TRUE gt colours (always) — measures "would frets appear?"
            rgb_vs_true, _ = rgb_l1_on_nn(rgb, gt_rgb, idx_p2t, idx_tgt_to_pred=idx_t2p)
            # Score against the *input* surface colours (was colour path followed?)
            _, in_rgb = ShapePCAE.surface_gt_points(surface, include_sharp_label=include_sharp)
            rgb_vs_input, _ = rgb_l1_on_nn(rgb, in_rgb, idx_p2t, idx_tgt_to_pred=idx_t2p)

        err_gt = per_point_rgb_l1_gt_to_recon(gt_xyz, gt_rgb, xyz, rgb)
        frets = frets_gt.cpu()
        panel = ~frets
        mean_fret_err = float(err_gt[frets].mean()) if frets.any() else float("nan")
        mean_panel_err = float(err_gt[panel].mean()) if panel.any() else float("nan")
        # loss mass = mean * count / N (fraction of total mean L1 attributed to frets)
        total_sum = float(err_gt.sum()) + 1e-12
        frets_mass = float(err_gt[frets].sum()) / total_sum if frets.any() else 0.0

        entry = {
            "cd": float(cd),
            "rgb_l1_vs_true_gt": float(rgb_vs_true),
            "rgb_l1_vs_input_surface": float(rgb_vs_input),
            "mean_rgb_err_on_frets": mean_fret_err,
            "mean_rgb_err_on_panels": mean_panel_err,
            "frets_error_mass_frac": frets_mass,
            "panel_error_mass_frac": 1.0 - frets_mass,
        }
        report["ablations"][mode] = entry
        logger.info(
            "[%s] cd=%.5f rgb_vs_true=%.4f rgb_vs_input=%.4f fret_err=%.4f panel_err=%.4f frets_mass=%.3f",
            mode,
            entry["cd"],
            entry["rgb_l1_vs_true_gt"],
            entry["rgb_l1_vs_input_surface"],
            mean_fret_err,
            mean_panel_err,
            frets_mass,
        )

        sub = out_dir / mode
        sub.mkdir(parents=True, exist_ok=True)
        export_xyz_pointcloud_ply(gt_xyz[0].cpu(), sub / "gt.ply", colors=gt_rgb[0].cpu())
        export_xyz_pointcloud_ply(xyz[0].cpu(), sub / "recon.ply", colors=rgb[0].cpu())
        export_xyz_pointcloud_ply(
            surface[0, :, :3].cpu(), sub / "input_surface.ply", colors=in_rgb[0].cpu()
        )
        export_recon_debug_plys(
            sub,
            gt_xyz=gt_xyz,
            recon_xyz=xyz,
            gt_rgb=gt_rgb,
            recon_rgb=rgb,
            fps_xyz=fps_xyz,
            centers=centers,
            num_points_per_anchor=int(model.num_points_per_anchor),
            max_anchor_delta=model.max_anchor_delta,
        )
        if mode == "baseline":
            baseline_out = {
                "xyz": xyz,
                "rgb": rgb,
                "centers": centers,
                "z": z,
                "fps_xyz": fps_xyz,
                "locals_tok": None,
            }

    # Re-run baseline encode with locals_tok + CA features for probes
    with torch.no_grad():
        z_b, fps_xyz, locals_tok = model.encode_tokenizer(
            surface0, register_noise=register_noise
        )
        xyz_b, rgb_b, centers_b = model.decode(z_b, representation_phase=False)

    fps_rgb, fps_idx = fps_colours_from_surface(surface0, fps_xyz, include_sharp)
    # Propagate GT frets mask to FPS via nearest surface index
    frets_surf = frets_gt.to(device)
    frets_fps = frets_surf[fps_idx[0]]
    # recon colour nearest to each FPS
    idx_f2r = nn_indices(
        fps_xyz[0].detach().cpu().numpy(),
        xyz_b[0].detach().cpu().numpy(),
    )
    recon_at_fps = rgb_b[0, torch.from_numpy(idx_f2r).to(device)]
    fps_rgb0 = fps_rgb[0]
    err_fps = (fps_rgb0 - recon_at_fps).abs().mean(dim=-1)
    frets_fps_err = float(err_fps[frets_fps].mean()) if frets_fps.any() else float("nan")
    panel_fps_err = float(err_fps[~frets_fps].mean()) if (~frets_fps).any() else float("nan")

    # Export FPS with true colour + error
    export_xyz_pointcloud_ply(
        fps_xyz[0].cpu(), out_dir / "fps_true_rgb.ply", colors=fps_rgb0.cpu()
    )
    export_xyz_pointcloud_ply(
        fps_xyz[0].cpu(),
        out_dir / "fps_recon_nn_rgb.ply",
        colors=recon_at_fps.cpu(),
    )
    export_xyz_pointcloud_ply(
        fps_xyz[0].cpu(),
        out_dir / "fps_rgb_err.ply",
        colors=scalar_to_rgb(
            err_fps.detach().cpu().numpy(),
            vmin=0.0,
            vmax=float(np.percentile(err_fps.detach().cpu().numpy(), 95) + 1e-6),
        ),
    )
    # frets-only FPS subset
    if frets_fps.any():
        export_xyz_pointcloud_ply(
            fps_xyz[0, frets_fps].cpu(),
            out_dir / "fps_on_frets_true_rgb.ply",
            colors=fps_rgb0[frets_fps].cpu(),
        )
        export_xyz_pointcloud_ply(
            fps_xyz[0, frets_fps].cpu(),
            out_dir / "fps_on_frets_recon_nn_rgb.ply",
            colors=recon_at_fps[frets_fps].cpu(),
        )

    # Linear probes: CA tokens + latents → frets on FPS
    probe_ca = linear_probe_binary(locals_tok[0], frets_fps.float().to(device))
    probe_z = linear_probe_binary(z_b[0], frets_fps.float().to(device))
    # Can CA tokens predict continuous RGB of FPS? linear MSE probe
    with torch.enable_grad():
        x = locals_tok[0].detach().float()
        y = fps_rgb0.detach().float()
        x = (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + 1e-6)
        W = torch.zeros(x.shape[1], 3, device=device, requires_grad=True)
        b = torch.zeros(3, device=device, requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=0.05)
        for _ in range(500):
            pred = x @ W + b
            loss = F.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = (x @ W + b).clamp(0, 1)
            mse = float(F.mse_loss(pred, y))
            l1 = float((pred - y).abs().mean())
            # frets only
            if frets_fps.any():
                l1_f = float((pred[frets_fps] - y[frets_fps]).abs().mean())
                l1_p = float((pred[~frets_fps] - y[~frets_fps]).abs().mean())
            else:
                l1_f = l1_p = float("nan")
    rgb_probe = {
        "mse": mse,
        "l1": l1,
        "l1_on_frets": l1_f,
        "l1_on_panels": l1_p,
    }

    # Delta: ablation vs baseline (rgb_vs_true)
    base_rgb = report["ablations"]["baseline"]["rgb_l1_vs_true_gt"]
    for mode, e in report["ablations"].items():
        e["delta_rgb_vs_true_vs_baseline"] = e["rgb_l1_vs_true_gt"] - base_rgb
        e["relative_rgb_vs_true"] = e["rgb_l1_vs_true_gt"] / max(base_rgb, 1e-8)

    report["fps"] = {
        "n_fps": int(fps_xyz.shape[1]),
        "n_fps_on_frets": int(frets_fps.sum()),
        "frac_fps_on_frets": float(frets_fps.float().mean()),
        "mean_rgb_err_fps_on_frets": frets_fps_err,
        "mean_rgb_err_fps_on_panels": panel_fps_err,
        "mean_fps_true_chroma": float(
            (fps_rgb0.max(-1).values - fps_rgb0.min(-1).values).mean()
        ),
        "mean_recon_at_fps_chroma": float(
            (recon_at_fps.max(-1).values - recon_at_fps.min(-1).values).mean()
        ),
        "mean_fps_true_chroma_on_frets": float(
            (fps_rgb0[frets_fps].max(-1).values - fps_rgb0[frets_fps].min(-1).values).mean()
        )
        if frets_fps.any()
        else float("nan"),
        "mean_recon_chroma_on_fret_fps": float(
            (
                recon_at_fps[frets_fps].max(-1).values
                - recon_at_fps[frets_fps].min(-1).values
            ).mean()
        )
        if frets_fps.any()
        else float("nan"),
    }
    report["linear_probe_fret_from_ca_token"] = probe_ca
    report["linear_probe_fret_from_latent_z"] = probe_z
    report["linear_probe_rgb_from_ca_token"] = rgb_probe

    # Summary verdicts
    zero_rel = report["ablations"]["zero_rgb"]["relative_rgb_vs_true"]
    shuf_rel = report["ablations"]["shuffle_rgb"]["relative_rgb_vs_true"]
    pink_fret_err = report["ablations"]["pink_frets"]["mean_rgb_err_on_frets"]
    base_fret_err = report["ablations"]["baseline"]["mean_rgb_err_on_frets"]

    verdicts = []
    if zero_rel < 1.15 and shuf_rel < 1.15:
        verdicts.append(
            "RGB_SURFACE_WEAKLY_USED: zero/shuffle barely hurts rgb_l1 vs true GT "
            f"(zero×{zero_rel:.2f}, shuffle×{shuf_rel:.2f}) — recon colour is largely "
            "memorized / geometry side-channel, not surface RGB frets."
        )
    elif zero_rel > 1.5 or shuf_rel > 1.5:
        verdicts.append(
            "RGB_SURFACE_USED: zero/shuffle clearly degrades colour — encoder RGB path "
            f"matters for bulk colour (zero×{zero_rel:.2f}, shuffle×{shuf_rel:.2f})."
        )
    else:
        verdicts.append(
            f"RGB_SURFACE_PARTIAL: moderate effect (zero×{zero_rel:.2f}, shuffle×{shuf_rel:.2f})."
        )

    if report["fps"]["n_fps_on_frets"] >= 10 and frets_fps_err > 0.08:
        verdicts.append(
            f"FPS_ON_FRETS_BUT_RECON_WASHED: {report['fps']['n_fps_on_frets']} FPS on frets "
            f"with mean |fps_rgb−recon_nn|={frets_fps_err:.3f} (panels {panel_fps_err:.3f}) "
            "— frets present at query/FPS colour but recon does not paint them."
        )

    if probe_ca["accuracy"] == probe_ca["accuracy"] and probe_ca["accuracy"] > 0.75:
        verdicts.append(
            f"CA_TOKENS_ENCODE_FRETS: linear probe acc={probe_ca['accuracy']:.2f} "
            f"auroc≈{probe_ca['auroc_proxy']:.2f} — frets live in CA tokens; decoder/loss drop them."
        )
    elif probe_ca["accuracy"] == probe_ca["accuracy"] and probe_ca["accuracy"] < 0.6:
        verdicts.append(
            f"CA_TOKENS_WEAK_ON_FRETS: probe acc={probe_ca['accuracy']:.2f} — frets poorly "
            "linearly readable from CA locals (washed by CA / weak RGB proj)."
        )

    if rgb_probe["l1_on_frets"] == rgb_probe["l1_on_frets"] and rgb_probe["l1_on_frets"] > 0.1:
        verdicts.append(
            f"CA_LINEAR_RGB_POOR_ON_FRETS: linear RGB decode L1 frets={rgb_probe['l1_on_frets']:.3f} "
            f"vs panels={rgb_probe['l1_on_panels']:.3f}."
        )

    frets_mass = report["ablations"]["baseline"]["frets_error_mass_frac"]
    verdicts.append(
        f"LOSS_MASS: frets own {frets_mass*100:.1f}% of per-point |rgb| residual sum "
        f"(area frac {report['fret_frac']*100:.1f}%); mean err frets={base_fret_err:.3f} "
        f"vs panels={report['ablations']['baseline']['mean_rgb_err_on_panels']:.3f}."
    )

    report["verdicts"] = verdicts
    report_path = out_dir / "analysis.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    # Human summary
    summary = out_dir / "SUMMARY.txt"
    lines = [
        f"RGB path diagnostics for {target}",
        f"ckpt: {args.ckpt}",
        f"mesh: {mesh_path}",
        f"GT frets: {n_fret}/{n_tot} ({100*report['fret_frac']:.2f}%)",
        "",
        "=== Ablations (rgb_l1 vs TRUE GT colour; lower better) ===",
    ]
    for mode, e in report["ablations"].items():
        lines.append(
            f"  {mode:12s}  cd={e['cd']:.5f}  rgb_true={e['rgb_l1_vs_true_gt']:.4f} "
            f"(×{e['relative_rgb_vs_true']:.2f} base)  "
            f"fret_err={e['mean_rgb_err_on_frets']:.4f} panel={e['mean_rgb_err_on_panels']:.4f} "
            f"frets_mass={e['frets_error_mass_frac']:.3f}"
        )
    lines += [
        "",
        "=== FPS ===",
        json.dumps(report["fps"], indent=2),
        "",
        "=== input_proj col norms ===",
        json.dumps(report["input_proj_norms"], indent=2),
        "",
        "=== linear probes ===",
        f"fret@CA: {probe_ca}",
        f"fret@z:  {probe_z}",
        f"rgb@CA:  {rgb_probe}",
        "",
        "=== VERDICTS ===",
    ]
    lines += [f"- {v}" for v in verdicts]
    summary.write_text("\n".join(lines) + "\n")
    logger.info("Wrote %s and %s", report_path, summary)
    print(summary.read_text())
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt",
        default="runs/exp4d_cam_cross_mv10_n8_f8_anccd5_rgb3_softdelta/ckpt_final.pt",
    )
    p.add_argument(
        "--output_dir",
        default="runs/diag_rgb_path/softdelta_obj_0001_33b463c971e9",
    )
    p.add_argument("--mesh_substr", default="33b463c971e9")
    p.add_argument("--data_dir", default=None)
    p.add_argument("--gobjaverse_render_root", default=None)
    p.add_argument("--vggt_cache_root", default=None)
    p.add_argument("--view_idx", type=int, default=0)
    p.add_argument("--max_items", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--chroma_thresh", type=float, default=0.08)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
