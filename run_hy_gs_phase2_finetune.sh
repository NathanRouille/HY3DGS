#!/bin/bash
# Phase 2: Finetune encoder RGB path (+ gs_head) from Phase 1 best checkpoint.
# Unfrozen encoder at reduced LR; transformer optional at lower scale.

set -euo pipefail

: "${EXPERIMENTS:=/export/home/nathan/datasets/shapenet_experiments}"
: "${DATASET:=chair_800}"
: "${CATEGORIES:=chair}"
: "${N_TRAIN:=800}"
: "${PHASE1_CKPT:=runs/train/chair_800_hy_mini_p1/ckpt_best.pt}"

DATA_DIR="${EXPERIMENTS}/${DATASET}/train"
VAL_DIR="${EXPERIMENTS}/${DATASET}/val"
OUTPUT_DIR="runs/train/${DATASET}_hy_mini_p2"

python train_gs_ae.py \
  --data_dir "${DATA_DIR}" \
  --val_dir "${VAL_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --categories "${CATEGORIES}" \
  --max_items "${N_TRAIN}" \
  --pretrained_profile hunyuan_mini \
  --resume_ckpt "${PHASE1_CKPT}" \
  --encoder_lr_scale 0.1 \
  --transformer_lr_scale 0.05 \
  --deterministic_encoder \
  --num_gs_per_anchor 10 \
  --sh_degree 0 \
  --num_views 22 \
  --views_per_step 6 \
  --batch_size 4 \
  --num_workers 4 \
  --max_steps 20000 \
  --lr 1e-4 \
  --warmup_steps 500 \
  --weight_decay 0.01 \
  --rgb_loss_type l1 \
  --lambda_rgb 1.0 \
  --lambda_ssim 0.2 \
  --lambda_lpips 0.1 \
  --lambda_d 0.2 \
  --lambda_alpha 0.1 \
  --alpha_bg_weight 5.0 \
  --lambda_scale 50.0 \
  --lambda_opa 0.001 \
  --lambda_delta 0.05 \
  --val_every 2000 \
  --num_val_samples 20 \
  --save_every 5000 \
  --log_every 50 \
  --seed 42
