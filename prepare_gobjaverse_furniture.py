#!/usr/bin/env python3
"""Build experiment manifest for G-Objaverse furniture subset (351 GLBs).

Writes manifest.json with pre-rendered GT metadata — no mesh re-rendering.
Requires:
  - furniture_glb_paths.json (uid -> glb path)
  - gobjaverse_280k_index_to_objaverse.json
  - gobjaverse_280k_Furnitures.json
  - ~/datasets/{partition}/{index}/ pre-rendered views
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from hy3dgen.shapegen.gobjaverse_gt import (
    GOBJAVERSE_GT_TAG,
    GOBJAVERSE_NUM_VIEWS,
    GOBJAVERSE_VIEW_LAYOUT,
    GObjaverseGTSource,
)


def parse_args():
    p = argparse.ArgumentParser(description="Prepare G-Objaverse furniture experiment manifest")
    p.add_argument(
        "--render_root",
        type=str,
        default=str(Path.home() / "datasets"),
        help="G-Objaverse render root (partition/index/view folders)",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=str(Path.home() / "datasets/gobjaverse_experiments/furniture_351"),
        help="Experiment directory (writes manifest.json)",
    )
    p.add_argument(
        "--glb_map",
        type=str,
        default="furniture_glb_paths.json",
        help="JSON map uid -> local GLB path",
    )
    p.add_argument(
        "--index_to_objaverse",
        type=str,
        default="gobjaverse_280k_index_to_objaverse.json",
    )
    p.add_argument(
        "--furniture_list",
        type=str,
        default="gobjaverse_280k_Furnitures.json",
    )
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--render_height", type=int, default=512)
    p.add_argument("--render_width", type=int, default=512)
    return p.parse_args()


def main():
    args = parse_args()
    render_root = Path(args.render_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    glb_map = json.load(open(args.glb_map))
    idx2obj = json.load(open(args.index_to_objaverse))
    furn = json.load(open(args.furniture_list))

    entries = []
    for gobjaverse_id in furn:
        if gobjaverse_id not in idx2obj:
            continue
        uid = Path(idx2obj[gobjaverse_id]).stem
        if uid not in glb_map:
            continue
        mesh_path = os.path.realpath(glb_map[uid])
        if not (render_root / gobjaverse_id).is_dir():
            continue
        entries.append(
            {
                "gobjaverse_id": gobjaverse_id,
                "uid": uid,
                "mesh_path": mesh_path,
            }
        )

    random.seed(args.seed)
    random.shuffle(entries)
    n_val = max(1, int(args.val_ratio * len(entries)))
    val_entries = entries[:n_val]
    train_entries = entries[n_val:]

    mesh_to_gobjaverse_id = {
        e["mesh_path"]: e["gobjaverse_id"] for e in entries
    }

    gt_source = GObjaverseGTSource(
        render_root=str(render_root),
        mesh_to_gobjaverse_id=mesh_to_gobjaverse_id,
        height=args.render_height,
        width=args.render_width,
    )
    train_paths = [e["mesh_path"] for e in train_entries if gt_source.has_gt(e["mesh_path"])]
    val_paths = [e["mesh_path"] for e in val_entries if gt_source.has_gt(e["mesh_path"])]

    manifest = {
        "version": 2,
        "gt_source": "gobjaverse",
        "gt_tag": GOBJAVERSE_GT_TAG,
        "gt_view_layout": GOBJAVERSE_VIEW_LAYOUT,
        "render_root": str(render_root),
        "render_height": args.render_height,
        "render_width": args.render_width,
        "num_views": GOBJAVERSE_NUM_VIEWS,
        "mesh_to_gobjaverse_id": mesh_to_gobjaverse_id,
        "train_mesh_paths": train_paths,
        "val_mesh_paths": val_paths,
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    (output_dir / "train").mkdir(exist_ok=True)
    (output_dir / "val").mkdir(exist_ok=True)
    print(f"Wrote {manifest_path}")
    print(f"  train: {len(train_paths)} meshes")
    print(f"  val:   {len(val_paths)} meshes")
    print(f"  render_root: {render_root}")
    print("\nTraining:")
    print(f"  python train_gs_ae.py \\")
    print(f"    --data_dir {output_dir / 'train'} \\")
    print(f"    --val_dir {output_dir / 'val'} \\")
    print(f"    --render_height {args.render_height} --render_width {args.render_width} \\")
    print(f"    --num_views {GOBJAVERSE_NUM_VIEWS} --gt_source gobjaverse ...")


if __name__ == "__main__":
    main()
