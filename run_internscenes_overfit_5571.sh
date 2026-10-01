#!/usr/bin/env bash
# 1-scene InternScenes overfit smoke: gen__bathroom__5571, view pool 1,3,7,9, MV2.
# Run on gaas (CUDA). Object pipeline unchanged — pass --dataset internscenes only here.
set -euo pipefail

PACK="${PACK:-$HOME/datasets/internscenes_bathroom_130}"
# Or group: PACK=/projects_vol/gp_chuanxia.zheng/nathan/datasets/internscenes_bathroom_130

ROOM=gen__bathroom__5571
VIEWS="1,3,7,9"
CACHE="${CACHE:-runs/vggt_cache/internscenes_${ROOM}_joint_${VIEWS//,/}}"
OUT="${OUT:-runs/internscenes_overfit_${ROOM}}"

cd "$(dirname "$0")"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

echo "=== VGGT joint cache (12 ordered pairs, mask_white_bg=False) ==="
python cache_vggt_features.py \
  --dataset internscenes \
  --data_dir "$PACK" \
  --internscenes_room_ids "$ROOM" \
  --cache_root "$CACHE" \
  --view_indices "$VIEWS" \
  --joint_pairs \
  --overwrite

echo "=== Overfit train (same arch as object exp15+: L=1024, 10k GT subsample) ==="
python train_pc_unite.py \
  --dataset internscenes \
  --data_dir "$PACK" \
  --internscenes_room_ids "$ROOM" \
  --max_items 1 \
  --no_experiment_manifest \
  --view_indices "$VIEWS" \
  --views_per_sample 2 \
  --align_mode c_meanrms \
  --vggt_cache_root "$CACHE" \
  --vggt_joint_cache \
  --output_dir "$OUT" \
  --batch_size 1 \
  --num_steps 5000 \
  --log_interval 50 \
  --ckpt_interval 1000 \
  --vis_interval 500

echo "=== Eval (fixed first pair in pool: views 1,3) ==="
python evaluate_pc_unite.py \
  --dataset internscenes \
  --data_dir "$PACK" \
  --internscenes_room_ids "$ROOM" \
  --ckpt "$OUT/ckpt_final.pt" \
  --output_dir "$OUT/eval" \
  --view_indices "$VIEWS" \
  --views_per_sample 2 \
  --vggt_cache_root "$CACHE" \
  --vggt_joint_cache \
  --max_items 1 \
  --no_experiment_manifest \
  --export_ply
