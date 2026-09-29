#!/usr/bin/env python3
"""Run Surflo (plain mode) on prepared letterboxed views.

Must run in the ``surflo-cu124`` conda env. Does not modify Surflo sources.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("infer_surflo")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compare_dir", required=True)
    p.add_argument(
        "--surflo_root",
        default="/export/home/nathan/Surflo",
    )
    p.add_argument(
        "--ckpt",
        default="/export/home/nathan/Surflo/checkpoints/surflo_v0.pt",
    )
    p.add_argument("--num_query_points", type=int, default=50000)
    p.add_argument("--num_steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n_images", type=int, default=2)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    compare_dir = Path(args.compare_dir).resolve()
    surflo_root = Path(args.surflo_root).resolve()
    if str(surflo_root) not in sys.path:
        sys.path.insert(0, str(surflo_root))

    ckpt = Path(args.ckpt)
    if not ckpt.is_file():
        # Allow relative ckpt paths resolved from surflo_root
        alt = surflo_root / args.ckpt
        if alt.is_file():
            ckpt = alt
    if not ckpt.is_file():
        logger.info("Checkpoint missing at %s — downloading from HuggingFace…", ckpt)
        ckpt = Path(args.ckpt)
        if not ckpt.is_absolute():
            ckpt = surflo_root / "checkpoints" / "surflo_v0.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            "AntoineGuedon/Surflo-v0",
            "surflo_v0.pt",
            local_dir=str(ckpt.parent),
        )
        ckpt = Path(downloaded)
        logger.info("Downloaded to %s", ckpt)

    import torch
    from surflo import Surflo, save_ply, set_global_seeds  # type: ignore

    manifest = json.loads((compare_dir / "manifest.json").read_text())
    objects = manifest["objects"]
    if args.limit is not None:
        objects = objects[: args.limit]

    set_global_seeds(args.seed)
    surflo = Surflo.from_checkpoint(str(ckpt), device=args.device)

    for i, entry in enumerate(objects):
        obj_dir = compare_dir / entry["obj_dir"]
        raw_dir = obj_dir / "pred_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        out_ply = raw_dir / "surflo.ply"
        if out_ply.exists():
            logger.info("[%d/%d] skip existing %s", i + 1, len(objects), out_ply)
            continue

        img_dir = compare_dir / entry["inputs_surflo"]
        with torch.no_grad():
            scene = surflo.encode(
                str(img_dir),
                n_images=args.n_images,
                cull_radius=10.0,
            )
            plain = scene.reconstruct(
                mode="plain",
                num_steps=args.num_steps,
                num_query_points=args.num_query_points,
                seed=args.seed,
                return_source=False,
            )
            # save_ply writes oriented points; copy/symlink path we want
            tmp = raw_dir / "surflo_plain_points.ply"
            n = save_ply(plain, tmp)
            # Normalize name to surflo.ply
            if tmp.resolve() != out_ply.resolve():
                if out_ply.exists():
                    out_ply.unlink()
                tmp.replace(out_ply)
        logger.info(
            "[%d/%d] Wrote %s (%s pts)", i + 1, len(objects), out_ply, n
        )

    logger.info("surflo inference done")


if __name__ == "__main__":
    main()
