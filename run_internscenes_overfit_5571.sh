#!/usr/bin/env bash
# 1-scene InternScenes overfit: gen__bathroom__5571, view pool 1,3,7,9, MV2.
# Recipe mirrors exp15b (joint cache + c_meanrms + sharp/7 feats), except:
#   - InternScenes data / views
#   - max_items=1, batch_size=1, num_steps=20000 (overfit scale)
#   - sample_renorm_output OFF (current correct default; exp15b inherited old True)
#   - no offpath / no Surflo global cam / no adaln_camera_cond
# Capacity (vs first overfit R=L=1024, K=8, 5k+5k, lambda_delta=0.05):
#   - R=L=2048, K=16 → 32768 out
#   - pc 10923 + 21845 = 32768 in (≈1/3 uniform, 2/3 sharp FPS)
#   - lambda_delta=0
set -euo pipefail

PACK="${PACK:-$HOME/datasets/internscenes_bathroom_130}"
# Or group: PACK=/projects_vol/gp_chuanxia.zheng/nathan/datasets/internscenes_bathroom_130

ROOM=gen__bathroom__5571
VIEWS="1,3,7,9"
CACHE="${CACHE:-runs/vggt_cache/internscenes_${ROOM}_joint_${VIEWS//,/}}"
OUT="${OUT:-runs/internscenes_overfit_${ROOM}_r2048_k16}"

cd "$(dirname "$0")"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

# Re-enable if views / room / mask_white_bg change (cache already built for 1,3,7,9).
# echo "=== VGGT joint cache (12 ordered pairs, mask_white_bg=False) ==="
# python cache_vggt_features.py \
#   --dataset internscenes \
#   --data_dir "$PACK" \
#   --internscenes_room_ids "$ROOM" \
#   --cache_root "$CACHE" \
#   --view_indices "$VIEWS" \
#   --joint_pairs \
#   --overwrite

echo "=== Overfit train (exp15b knobs; R=L=2048 K=16; sample_renorm OFF; lambda_delta=0) ==="
echo "=== Reusing VGGT cache: $CACHE ==="
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
  --device cuda \
  --seed 0 \
  --batch_size 1 \
  --num_workers 0 \
  --num_steps 20000 \
  --lr 1e-4 \
  --lr_schedule cosine \
  --weight_decay 0.01 \
  --num_latents 2048 \
  --num_registers 2048 \
  --num_points_per_anchor 16 \
  --pc_size 10923 \
  --pc_sharpedge_size 21845 \
  --pretrained_load cross_attn \
  --pretrained_repo tencent/Hunyuan3D-2mini \
  --pretrained_subfolder hunyuan3d-vae-v2-mini-withencoder \
  --include_sharp_label \
  --point_feats 7 \
  --register_noise_mode random \
  --representation_noising \
  --noising_t_start 0.9 \
  --flow_loss_type velocity \
  --flow_steps_per_recon 8 \
  --lambda_flow 1.0 \
  --lambda_recon 1.0 \
  --lambda_rgb 10.0 \
  --lambda_anc 0.1 \
  --lambda_anc_cd 10.0 \
  --lambda_delta 0 \
  --weak_context_dropout 0.1 \
  --no-sample_renorm_output \
  --offpath_mode none \
  --no-adaln_camera_cond \
  --no-tokenizer_use_weak_context \
  --log_interval 100 \
  --ckpt_interval 1000 \
  --vis_interval 500 \
  --wandb \
  --wandb_project shapepcunite \
  --wandb_name "internscenes_overfit_${ROOM}_r2048_k16" \
  --output_dir "$OUT"

echo "=== Eval (view_sample_mode=first → fixed pair 1,3) ==="
python evaluate_pc_unite.py \
  --dataset internscenes \
  --data_dir "$PACK" \
  --internscenes_room_ids "$ROOM" \
  --ckpt "$OUT/ckpt_final.pt" \
  --output_dir "$OUT/eval" \
  --view_indices "$VIEWS" \
  --views_per_sample 2 \
  --align_mode c_meanrms \
  --vggt_cache_root "$CACHE" \
  --vggt_joint_cache \
  --max_items 1 \
  --no_experiment_manifest \
  --no-sample_renorm_output \
  --export_ply
