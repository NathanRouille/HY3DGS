import json, random, os
from pathlib import Path

glb_map = json.load(open("furniture_glb_paths.json"))          # uid -> glb path
idx2obj = json.load(open("gobjaverse_280k_index_to_objaverse.json"))
furn = json.load(open("gobjaverse_280k_Furnitures.json"))

# keep only objects you actually have meshes + renders for
render_root = Path.home() / "datasets"
entries = []
for e in furn:
    if e not in idx2obj:
        continue
    uid = Path(idx2obj[e]).stem
    if uid not in glb_map:
        continue
    if not (render_root / e).is_dir():
        continue
    entries.append({"gobjaverse_id": e, "uid": uid, "mesh_path": glb_map[uid]})

random.seed(42)
random.shuffle(entries)
n_val = max(1, int(0.1 * len(entries)))   # ~35 val, ~316 train
val_entries = entries[:n_val]
train_entries = entries[n_val:]

exp = Path("/export/home/nathan/datasets/gobjaverse_experiments/furniture_351_v46")
exp.mkdir(parents=True, exist_ok=True)
manifest = {
    "version": 1,
    "gt_tag": "gt_rgbd_h512w512_v46_fp16_norm",   # must match GTRGBDRenderer tag
    "gt_view_layout": "v46",
    "train_mesh_paths": [e["mesh_path"] for e in train_entries],
    "val_mesh_paths": [e["mesh_path"] for e in val_entries],
}
(exp / "manifest.json").write_text(json.dumps(manifest, indent=2))