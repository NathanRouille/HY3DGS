#!/usr/bin/env python3
"""InternScenes align showcase — same recipes as export_align_showcase_debug.py.

Locked scene frame (verified on gen__bathroom__5658):
  - surface.npz is GLB Y-up world after unnormalize
  - remap to trajectory Z-up: (x, y, z) -> (x, -z, y)
  - cam_matrix_opencv is cam-to-world (OpenCV axes)
  - Xc = (X_traj - t) @ R   (HY3DGS world_to_camera)

Methods (identical names to the object showcase):
  raw
  indep_meanrms
  C_gt_depth_filter_zrobust_meanrms

VGGT: mask_white_bg=False (white tiles are geometry); keep conf percentile filter.

Example:
  conda activate hy3dgs
  cd ~/Documents/research/HY3DGS
  export PYTHONPATH="$PWD:$PYTHONPATH"
  python export_align_showcase_internscenes.py \\
    --room_dir ~/Documents/research/internscenes_bathroom_130/rooms/gen__bathroom__5658 \\
    --view_indices 50,100,150 \\
    --output_dir runs/debug_align_showcase_internscenes_5658 \\
    --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from export_align_showcase_debug import (
    METRIC_KEYS,
    METHODS,
    _export_pair,
    _save_png,
    _to_np_xyz,
    _write_metrics,
    _write_ranking,
    _write_summary,
)
from export_camera_frame_debug import alignment_metrics, shared_canonicalize_from_ref
from hy3dgen.shapegen.cam_align import (
    _erode_mask,
    apply_mu_s,
    mean_rms_mu_s,
    merge_unprojected_to_cam_ref,
)
from hy3dgen.shapegen.vggt_context import (
    VGGTContextBuilder,
    depth_map_to_cam_points,
    patch_centers_from_depth,
    world_to_camera_torch,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("align_showcase_internscenes")

# Extra metrics beyond object METRIC_KEYS
F1_KEYS = ("f1_at_0p05", "precision_at_0p05", "recall_at_0p05")


def glb_yup_to_traj_zup(xyz: np.ndarray) -> np.ndarray:
    """GLB / surface.npz Y-up → InternScenes trajectory Z-up."""
    xyz = np.asarray(xyz, dtype=np.float64)
    return np.stack([xyz[:, 0], -xyz[:, 2], xyz[:, 1]], axis=1).astype(np.float32)


def glb_yup_to_traj_zup_normals(nrm: np.ndarray) -> np.ndarray:
    nrm = np.asarray(nrm, dtype=np.float64)
    out = np.stack([nrm[:, 0], -nrm[:, 2], nrm[:, 1]], axis=1)
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-8
    return out.astype(np.float32)


def load_surface_traj(room_dir: Path) -> torch.Tensor:
    """Return surface [N,10] float: xyz|nrm|sharp|rgb in traj Z-up world."""
    data = np.load(room_dir / "surface.npz")
    xyz = glb_yup_to_traj_zup(data["xyz"])
    nrm = glb_yup_to_traj_zup_normals(data["normals"])
    sharp = np.asarray(data["sharp"], dtype=np.float32).reshape(-1, 1)
    rgb = np.asarray(data["rgb"], dtype=np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    if rgb.ndim == 1:
        rgb = rgb.reshape(-1, 3)
    surf = np.concatenate([xyz, nrm, sharp, rgb.astype(np.float32)], axis=1)
    return torch.from_numpy(surf.astype(np.float32))


def load_c2w_opencv(poses_entry: dict, pose_id: int) -> torch.Tensor:
    """Build 4x4 c2w from cam_matrix_opencv (3x4)."""
    pose = poses_entry["poses"][pose_id]
    if int(pose["pose_id"]) != int(pose_id):
        # fall back to search
        pose = next(p for p in poses_entry["poses"] if int(p["pose_id"]) == int(pose_id))
    m = np.eye(4, dtype=np.float64)
    m[:3, :] = np.asarray(pose["cam_matrix_opencv"], dtype=np.float64)
    return torch.from_numpy(m.astype(np.float32))


def load_view(room_dir: Path, view_idx: int) -> Dict:
    cam_json = json.loads((room_dir / "camera_poses.json").read_text(encoding="utf-8"))
    entry = next(iter(cam_json.values()))
    intr = entry["intrinsics"]
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["cx"])
    cy = float(intr["cy"])
    K = torch.tensor([fx, fy, cx, cy], dtype=torch.float32)

    rgb_path = room_dir / "rgb" / f"view_{view_idx:03d}.png"
    if not rgb_path.is_file():
        raise FileNotFoundError(rgb_path)
    rgb_u8 = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    rgb = torch.from_numpy(rgb_u8.astype(np.float32) / 255.0).permute(2, 0, 1)

    depth_path = room_dir / "depth_float32" / f"frame_{view_idx:04d}_depth.npz"
    if not depth_path.is_file():
        raise FileNotFoundError(depth_path)
    dpack = np.load(depth_path)
    depth = np.asarray(dpack["depth"], dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    mask = np.asarray(dpack["mask"], dtype=bool)
    if mask.ndim == 3:
        mask = mask[..., 0]

    c2w = load_c2w_opencv(entry, view_idx)
    return {
        "rgb": rgb,
        "depth": torch.from_numpy(depth),
        "depth_mask": torch.from_numpy(mask.astype(np.bool_)),
        "intrinsics": K,
        "c2w": c2w,
        "view_idx": int(view_idx),
    }


def surface_to_camera(surface: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    xyz = surface[:, :3]
    xyz_cam = world_to_camera_torch(xyz, c2w)
    out = surface.clone()
    out[:, :3] = xyz_cam
    if out.shape[-1] >= 6:
        nrm = out[:, 3:6]
        R = c2w[:3, :3]
        out[:, 3:6] = nrm @ R
    return out


def c_meanrms_stats_from_depth_mask(
    depth: np.ndarray,
    mask: np.ndarray,
    K_fxfycxcy: Sequence[float],
    c2w: np.ndarray,
    *,
    erode_iters: int = 1,
) -> Dict[str, float]:
    """Like compute_c_meanrms_gt_stats but uses traj depth mask (no white-bg filter)."""
    d = np.asarray(depth, dtype=np.float32).squeeze()
    valid = np.asarray(mask, dtype=bool) & np.isfinite(d) & (d > 1e-6)
    valid = _erode_mask(valid, iters=erode_iters)
    fx, fy, cx, cy = [float(x) for x in K_fxfycxcy]
    pts, _ = merge_unprojected_to_cam_ref(
        [d],
        [valid],
        Ks=[(fx, fy, cx, cy)],
        c2ws=[np.asarray(c2w, dtype=np.float64)],
        c2w_ref=np.asarray(c2w, dtype=np.float64),
    )
    if pts.shape[0] == 0:
        raise RuntimeError("c_meanrms: no valid GT-depth points after mask/erode")
    mu, s = mean_rms_mu_s(pts)
    return {
        "mu_x": float(mu[0]),
        "mu_y": float(mu[1]),
        "mu_z": float(mu[2]),
        "s": float(s),
        "n_pts": float(pts.shape[0]),
    }


def f1_at_thresh(
    pe: np.ndarray, gt: np.ndarray, *, thresh: float, max_n: int, rng: np.random.Generator
) -> Tuple[float, float, float]:
    """Precision/recall/F1: fraction of points with NN < thresh (symmetric average style)."""
    from export_camera_frame_debug import _nn_dists

    pe = np.asarray(pe, dtype=np.float32).reshape(-1, 3)
    gt = np.asarray(gt, dtype=np.float32).reshape(-1, 3)
    if pe.shape[0] == 0 or gt.shape[0] == 0:
        return float("nan"), float("nan"), float("nan")
    d_pg = _nn_dists(pe, gt, max_n=max_n, rng=rng)
    d_gp = _nn_dists(gt, pe, max_n=max_n, rng=rng)
    precision = float(np.mean(d_pg < thresh))
    recall = float(np.mean(d_gp < thresh))
    if precision + recall < 1e-12:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return f1, precision, recall


def process_one(
    *,
    stem: str,
    view_idx: int,
    surface_world: torch.Tensor,
    view: Dict,
    builder: VGGTContextBuilder,
    device: torch.device,
    out_dir: Path,
    rng: np.random.Generator,
    max_gt_points: int,
    f1_thresh: float,
) -> List[Dict]:
    c2w = view["c2w"]
    surf_cam = surface_to_camera(surface_world, c2w)
    gt_cam = _to_np_xyz(surf_cam[:, :3])
    # Layout xyz|nrm|sharp|rgb → rgb at columns 7:10
    gt_rgb = surf_cam[:, 7:10] if surf_cam.shape[-1] >= 10 else None
    if gt_cam.shape[0] > max_gt_points > 0:
        idx = np.linspace(0, gt_cam.shape[0] - 1, num=max_gt_points, dtype=np.int64)
        gt_cam = gt_cam[idx]
        if gt_rgb is not None:
            gt_rgb = gt_rgb[idx]

    out_dir.mkdir(parents=True, exist_ok=True)
    _save_png(
        out_dir / "view_rgb.png",
        (view["rgb"].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8),
    )

    rgb = view["rgb"].unsqueeze(0).to(device)
    with torch.no_grad():
        raw = builder.extract_vggt_raw(rgb, return_dense=True)

    depth = raw["vggt_depth"][0].detach().float().cpu().numpy()
    conf = raw["vggt_depth_conf"][0].detach().float().cpu().numpy()
    K = raw["vggt_intrinsics"][0].detach().float().cpu().numpy()
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    cam_pts = depth_map_to_cam_points(depth, fx=fx, fy=fy, cx=cx, cy=cy)

    vggt_rgb = (
        torch.nn.functional.interpolate(
            view["rgb"].float().unsqueeze(0),
            size=depth.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0]
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    pix_valid = builder._pixel_valid_mask(depth, conf, rgb_np=vggt_rgb)
    full_c, full_keep = patch_centers_from_depth(
        cam_pts,
        pix_valid,
        patch_size=builder.patch_size,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )
    centers_raw = full_c[full_keep.astype(bool)].astype(np.float32)
    if centers_raw.shape[0] == 0:
        logger.warning("%s view %d: no kept patch centres — skipping", stem, view_idx)
        return []

    pairs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
        "raw": (gt_cam, centers_raw),
    }

    mu_gt_i, s_gt_i = mean_rms_mu_s(gt_cam)
    mu_pe_i, s_pe_i = mean_rms_mu_s(centers_raw)
    pairs["indep_meanrms"] = (
        apply_mu_s(gt_cam, mu_gt_i, s_gt_i).astype(np.float32),
        apply_mu_s(centers_raw, mu_pe_i, s_pe_i).astype(np.float32),
    )

    depth_np = view["depth"].numpy() if torch.is_tensor(view["depth"]) else np.asarray(view["depth"])
    mask_np = (
        view["depth_mask"].numpy()
        if torch.is_tensor(view["depth_mask"])
        else np.asarray(view["depth_mask"])
    )
    st = c_meanrms_stats_from_depth_mask(
        depth_np,
        mask_np,
        view["intrinsics"].numpy().tolist(),
        c2w.numpy() if torch.is_tensor(c2w) else np.asarray(c2w),
        erode_iters=1,
    )
    mu_c = np.array([st["mu_x"], st["mu_y"], st["mu_z"]], dtype=np.float64)
    s_c = float(st["s"])
    (centers_c,), _, _ = shared_canonicalize_from_ref(
        centers_raw, centers_raw, scale="rms"
    )
    pairs["C_gt_depth_filter_zrobust_meanrms"] = (
        apply_mu_s(gt_cam, mu_c, s_c).astype(np.float32),
        centers_c.astype(np.float32),
    )

    rows: List[Dict] = []
    for method, (gt_xyz, cen_xyz) in pairs.items():
        mdir = out_dir / method
        _export_pair(mdir, gt=gt_xyz, centers=cen_xyz, gt_rgb=gt_rgb)
        metrics = alignment_metrics(cen_xyz, gt_xyz, rng=rng)
        f1, prec, rec = f1_at_thresh(
            cen_xyz, gt_xyz, thresh=f1_thresh, max_n=4000, rng=rng
        )
        metrics["f1_at_0p05"] = f1
        metrics["precision_at_0p05"] = prec
        metrics["recall_at_0p05"] = rec
        # keep key name stable even if thresh changes via CLI (documented in header)
        _write_metrics(
            mdir / "metrics.txt",
            metrics,
            header=(
                f"mesh={stem}\nview_idx={view_idx}\nmethod={method}\n"
                f"dataset=internscenes\nmask_white_bg=False\n"
                f"f1_thresh={f1_thresh}\n"
                "frame=traj_zup_from_glb_yup\n"
                "scored: patch_centers vs gt\n"
            ),
        )
        # also write f1 keys into metrics.txt via extended keys
        with open(mdir / "metrics.txt", "a", encoding="utf-8") as f:
            for k in F1_KEYS:
                f.write(f"{k}={metrics[k]}\n")

        row = {
            "mesh": stem,
            "view_idx": int(view_idx),
            "method": method,
            "sample_dir": str(out_dir),
            **{k: metrics[k] for k in METRIC_KEYS},
            **{k: metrics[k] for k in F1_KEYS},
        }
        rows.append(row)
    return rows


def parse_view_indices(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--room_dir",
        type=Path,
        required=True,
        help=".../rooms/<scene_id> with surface.npz, rgb/, depth_float32/, camera_poses.json",
    )
    p.add_argument(
        "--view_indices",
        type=str,
        default="50,100,150",
        help="Comma-separated pose/view ids (match view_XXX.png / frame_XXXX_depth.npz)",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("runs/debug_align_showcase_internscenes"),
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--conf_percentile", type=float, default=20.0)
    p.add_argument("--min_conf", type=float, default=0.05)
    p.add_argument("--max_gt_points", type=int, default=20000)
    p.add_argument(
        "--f1_thresh",
        type=float,
        default=0.05,
        help="NN threshold for F1/precision/recall (same space as clouds after each method)",
    )
    args = p.parse_args()

    room_dir = args.room_dir.expanduser().resolve()
    if not (room_dir / "surface.npz").is_file():
        logger.error("Missing surface.npz in %s", room_dir)
        return 2
    stem = room_dir.name
    views = parse_view_indices(args.view_indices)
    if not views:
        logger.error("Empty --view_indices")
        return 2

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_root = args.output_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    logger.info("Loading surface (GLB Y-up → traj Z-up) from %s", room_dir)
    surface = load_surface_traj(room_dir)
    meta = {}
    meta_path = room_dir / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    logger.info(
        "surface N=%d frame_meta=%s methods=%s views=%s",
        surface.shape[0],
        meta.get("frame"),
        METHODS,
        views,
    )

    builder = VGGTContextBuilder(
        width=1024,
        conf_percentile=args.conf_percentile,
        min_conf=args.min_conf,
        mask_white_bg=False,
    ).to(device)
    builder.eval()

    all_rows: List[Dict] = []
    for view_idx in views:
        try:
            view = load_view(room_dir, int(view_idx))
        except Exception as e:
            logger.warning("Skip %s view %d: %s", stem, view_idx, e)
            continue

        sample_dir = out_root / f"0000_{stem[:24]}_v{int(view_idx):03d}"
        try:
            rows = process_one(
                stem=stem,
                view_idx=int(view_idx),
                surface_world=surface,
                view=view,
                builder=builder,
                device=device,
                out_dir=sample_dir,
                rng=rng,
                max_gt_points=args.max_gt_points,
                f1_thresh=float(args.f1_thresh),
            )
        except Exception:
            logger.exception("Failed %s view %d", stem, view_idx)
            continue

        all_rows.extend(rows)
        if rows:
            c_nn = next(
                (
                    r["nn_pe2gt_mean"]
                    for r in rows
                    if r["method"] == "C_gt_depth_filter_zrobust_meanrms"
                ),
                float("nan"),
            )
            n_nn = next(
                (r["nn_pe2gt_mean"] for r in rows if r["method"] == "indep_meanrms"),
                float("nan"),
            )
            c_f1 = next(
                (
                    r["f1_at_0p05"]
                    for r in rows
                    if r["method"] == "C_gt_depth_filter_zrobust_meanrms"
                ),
                float("nan"),
            )
            logger.info(
                "%s v%03d  C_nn=%.4f  indep_nn=%.4f  C_f1@%.3f=%.4f  → %s",
                stem[:24],
                view_idx,
                c_nn,
                n_nn,
                args.f1_thresh,
                c_f1,
                sample_dir.name,
            )

    # summary with F1 columns
    if all_rows:
        fields = ["mesh", "view_idx", "method", "sample_dir", *METRIC_KEYS, *F1_KEYS]
        with open(out_root / "summary.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_rows)
        _write_ranking(out_root / "ranking.csv", all_rows)

    (out_root / "provenance.json").write_text(
        json.dumps(
            {
                "room_dir": str(room_dir),
                "views": views,
                "mask_white_bg": False,
                "conf_percentile": args.conf_percentile,
                "min_conf": args.min_conf,
                "f1_thresh": args.f1_thresh,
                "frame": "traj_zup = (x, -z, y) from GLB Y-up surface.npz",
                "c2w": "camera_poses.json poses[i].cam_matrix_opencv",
                "methods": list(METHODS),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("DONE rows=%d → %s", len(all_rows), out_root)
    return 0 if all_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
