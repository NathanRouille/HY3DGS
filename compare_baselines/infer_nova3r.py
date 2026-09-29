#!/usr/bin/env python3
"""Run NOVA3R on prepared letterboxed views.

Must run in the ``nova3r`` conda env. Does not modify the nova3r package;
imports ``demo_nova3r`` from the cloned repo.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("infer_nova3r")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compare_dir", required=True)
    p.add_argument(
        "--nova3r_root",
        default="/export/home/nathan/nova3r",
    )
    p.add_argument(
        "--ckpt",
        default="/export/home/nathan/nova3r/checkpoints/scene_n2/checkpoint-last.pth",
    )
    p.add_argument("--resolution", type=int, nargs=2, default=[518, 392])
    p.add_argument("--num_queries", type=int, default=50000)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of objects (debug).",
    )
    args = p.parse_args()

    nova_root = Path(args.nova3r_root).resolve()
    compare_dir = Path(args.compare_dir).resolve()
    if not (compare_dir / "manifest.json").is_file():
        raise SystemExit(f"Missing manifest: {compare_dir / 'manifest.json'}")

    if str(nova_root) not in sys.path:
        sys.path.insert(0, str(nova_root))
    os.chdir(nova_root)

    # Import after chdir/path setup
    from demo_nova3r import load_model, inference_nova3r  # type: ignore
    from dust3r.image_pairs import make_pairs  # type: ignore
    from omegaconf import OmegaConf
    import PIL.Image
    import torch
    import torchvision.transforms as transforms

    manifest = json.loads((compare_dir / "manifest.json").read_text())
    objects = manifest["objects"]
    if args.limit is not None:
        objects = objects[: args.limit]

    device = args.device
    model, cfg = load_model(args.ckpt, device)
    OmegaConf.set_struct(cfg, False)
    if "fm_step_size" not in cfg:
        cfg.fm_step_size = 0.04
    if "fm_sampling" not in cfg:
        cfg.fm_sampling = "euler"

    target_W, target_H = int(args.resolution[0]), int(args.resolution[1])
    img_norm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    for i, entry in enumerate(objects):
        obj_dir = compare_dir / entry["obj_dir"]
        raw_dir = obj_dir / "pred_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        out_ply = raw_dir / "nova3r.ply"
        if out_ply.exists():
            logger.info("[%d/%d] skip existing %s", i + 1, len(objects), out_ply)
            continue

        img_dir = compare_dir / entry["inputs_nova3r"]
        img_paths = sorted(img_dir.glob("*.png"))
        if len(img_paths) < 2:
            logger.error("Need 2 images in %s, found %d", img_dir, len(img_paths))
            continue
        img_paths = img_paths[:2]

        images = []
        for j, path in enumerate(img_paths):
            img = PIL.Image.open(path).convert("RGB")
            w0, h0 = img.size
            # Already letterboxed to target; resize is identity if sizes match
            img = img.resize((target_W, target_H), PIL.Image.LANCZOS)
            logger.info(
                "  %s %dx%d -> %dx%d", path.name, w0, h0, target_W, target_H
            )
            images.append(
                dict(
                    img=img_norm(img)[None],
                    true_shape=np.int32([target_H, target_W]),
                    idx=j,
                    instance=str(j),
                    view_label=f"input_{j}",
                )
            )

        pairs = make_pairs(
            images, scene_graph="complete", prefilter=None, symmetrize=False
        )
        with torch.no_grad():
            output = inference_nova3r(
                cfg,
                pairs,
                model,
                device,
                batch_size=1,
                num_queries=args.num_queries,
                method=cfg.get("fm_sampling", "euler"),
            )
        pts = output["pred"]["pts3d_xyz"][0].detach().cpu().numpy()

        # Write ascii PLY locally (avoid depending on open3d write path)
        with open(out_ply, "w", encoding="ascii") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {pts.shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for x, y, z in pts:
                f.write(f"{x:.6f} {y:.6f} {z:.6f} 80 200 80\n")
        logger.info(
            "[%d/%d] Wrote %s (%d pts)", i + 1, len(objects), out_ply, pts.shape[0]
        )

    logger.info("nova3r inference done")


if __name__ == "__main__":
    main()
