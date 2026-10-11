#!/usr/bin/env bash
# Ten-room overfit with individually selected view pools.
# Run matched controls with DELTA_WEIGHT=0 and DELTA_WEIGHT=0.001.
# New cache/output paths preserve the previous experiment.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

PACK="${PACK:-/projects_vol/gp_chuanxia.zheng/nathan/datasets/internscenes_bathroom_130}"

VIEW_MAP="${VIEW_MAP:-$SCRIPT_DIR/configs/internscenes_bath10_selected_views.json}"
if [[ "$VIEW_MAP" != /* ]]; then VIEW_MAP="$PWD/$VIEW_MAP"; fi
DELTA_WEIGHT="${DELTA_WEIGHT:-0.001}"
NUM_STEPS="${NUM_STEPS:-20000}"
CACHE="${CACHE:-runs/vggt_cache/internscenes_bath10_joint_selected_v2}"
OUT="${OUT:-runs/internscenes_overfit_bath10_selected_v2_delta${DELTA_WEIGHT//./p}}"

cd "$SCRIPT_DIR"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

python -m hy3dgen.shapegen.room_view_pools "$VIEW_MAP" "$PACK"

echo "=== VGGT joint cache (selected per-room pools; 12 ordered pairs per room) ==="
echo "=== Cache root: $CACHE ==="
python cache_vggt_features.py \
  --dataset internscenes \
  --data_dir "$PACK" \
  --internscenes_split train \
  --max_items 10 \
  --cache_root "$CACHE" \
  --room_view_pools_json "$VIEW_MAP" \
  --joint_pairs

echo "=== Selected-view train: batch=2; steps=$NUM_STEPS; delta=$DELTA_WEIGHT ==="
python train_pc_unite.py \
  --dataset internscenes \
  --data_dir "$PACK" \
  --internscenes_split train \
  --max_items 10 \
  --no_experiment_manifest \
  --room_view_pools_json "$VIEW_MAP" \
  --views_per_sample 2 \
  --align_mode c_meanrms \
  --vggt_cache_root "$CACHE" \
  --vggt_joint_cache \
  --device cuda \
  --seed 0 \
  --batch_size 2 \
  --num_workers 0 \
  --num_steps "$NUM_STEPS" \
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
  --lambda_delta "$DELTA_WEIGHT" \
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
  --wandb_name "$(basename "$OUT")" \
  --output_dir "$OUT"

echo "=== Eval (fixed first ordered pair from each room pool) ==="
python evaluate_pc_unite.py \
  --dataset internscenes \
  --data_dir "$PACK" \
  --internscenes_split train \
  --ckpt "$OUT/ckpt_final.pt" \
  --output_dir "$OUT/eval" \
  --room_view_pools_json "$VIEW_MAP" \
  --views_per_sample 2 \
  --align_mode c_meanrms \
  --vggt_cache_root "$CACHE" \
  --vggt_joint_cache \
  --max_items 10 \
  --no_experiment_manifest \
  --no-sample_renorm_output \
  --export_ply
