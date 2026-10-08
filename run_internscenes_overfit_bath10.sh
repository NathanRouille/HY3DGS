#!/usr/bin/env bash
# 10-room InternScenes overfit: first 10 ids in splits/train.txt, views 1,3,7,9, MV2.
# Same capacity/recipe as the 5571 wfps overfit, except:
#   - max_items=10, batch_size=2, num_steps=20000
#   - VGGT joint cache built automatically at start (skips existing .pt unless --overwrite)
set -euo pipefail

PACK="${PACK:-$HOME/datasets/internscenes_bathroom_130}"
# Or group: PACK=/projects_vol/gp_chuanxia.zheng/nathan/datasets/internscenes_bathroom_130

VIEWS="1,3,7,9"
CACHE="${CACHE:-runs/vggt_cache/internscenes_bath10_joint_${VIEWS//,/}}"
OUT="${OUT:-runs/internscenes_overfit_bath10_r2048_k16_5050_cdgt2_wfps_k16_b0.2}"

cd "$(dirname "$0")"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

echo "=== VGGT joint cache (first 10 train rooms, 12 ordered pairs, mask_white_bg=False) ==="
echo "=== Cache root: $CACHE ==="
python cache_vggt_features.py \
  --dataset internscenes \
  --data_dir "$PACK" \
  --internscenes_split train \
  --max_items 10 \
  --cache_root "$CACHE" \
  --view_indices "$VIEWS" \
  --joint_pairs

echo "=== Overfit train (10 rooms; R=L=2048 K=16; batch=2; pc 50/50; wfps k=16 β=0.2; cd_gt2pred=2) ==="
python train_pc_unite.py \
  --dataset internscenes \
  --data_dir "$PACK" \
  --internscenes_split train \
  --max_items 10 \
  --no_experiment_manifest \
  --view_indices "$VIEWS" \
  --views_per_sample 2 \
  --align_mode c_meanrms \
  --vggt_cache_root "$CACHE" \
  --vggt_joint_cache \
  --device cuda \
  --seed 0 \
  --batch_size 2 \
  --num_workers 0 \
  --num_steps 20000 \
  --lr 1e-4 \
  --lr_schedule cosine \
  --weight_decay 0.01 \
  --num_latents 2048 \
  --num_registers 2048 \
  --num_points_per_anchor 16 \
  --pc_size 16384 \
  --pc_sharpedge_size 16384 \
  --query_sample_mode weighted_fps \
  --fps_density_k 16 \
  --fps_sharp_beta 0.2 \
  --fps_density_clip_low 5 \
  --fps_density_clip_high 95 \
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
  --cd_pred2gt 1.0 \
  --cd_gt2pred 2.0 \
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
  --wandb_name "internscenes_overfit_bath10_r2048_k16_5050_cdgt2_wfps_k16_b0.2" \
  --output_dir "$OUT"

echo "=== Eval (view_sample_mode=first → fixed pair 1,3; 10 rooms) ==="
python evaluate_pc_unite.py \
  --dataset internscenes \
  --data_dir "$PACK" \
  --internscenes_split train \
  --ckpt "$OUT/ckpt_final.pt" \
  --output_dir "$OUT/eval" \
  --view_indices "$VIEWS" \
  --views_per_sample 2 \
  --align_mode c_meanrms \
  --vggt_cache_root "$CACHE" \
  --vggt_joint_cache \
  --max_items 10 \
  --no_experiment_manifest \
  --no-sample_renorm_output \
  --export_ply
