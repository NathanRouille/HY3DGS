#!/usr/bin/env bash
# Exp21-B: matched control — same as Exp21-A but WITHOUT Surflo global-cam.
# E19-A@110k → +20k Mode A, optimizer reset.
#
# Usage:
#   bash train_exp21b_continue_no_cam.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
set +u
conda activate hy3dgs
set -u

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DEVICE="${DEVICE:-cuda}"
PARENT="${PARENT:-runs/exp19a_offpath_continue_from_exp18/ckpt_0110000.pt}"
OUT="${OUT:-runs/exp21b_continue_no_cam_from_exp19a}"
DATA_DIR="${DATA_DIR:-/export/home/nathan/datasets/gobjaverse_experiments/furniture_351/train}"
RENDER_ROOT="${RENDER_ROOT:-/export/home/nathan/datasets}"
VGGT_CACHE="${VGGT_CACHE:-runs/vggt_cache/furn100_joint_pool_4_10_16_22}"
ADDITIONAL_STEPS="${ADDITIONAL_STEPS:-20000}"
CKPT_INTERVAL="${CKPT_INTERVAL:-5000}"

if [[ ! -f "$PARENT" ]]; then
  echo "ERROR: missing parent ckpt $PARENT" >&2
  exit 1
fi

echo "=== Exp21-B no-cam control  parent=$PARENT  out=$OUT  +${ADDITIONAL_STEPS} ==="
python train_pc_unite.py \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUT" \
  --device "$DEVICE" \
  --seed 0 \
  --gobjaverse_render_root "$RENDER_ROOT" \
  --vggt_cache_root "$VGGT_CACHE" \
  --vggt_joint_cache \
  --view_indices 4,10,16,22 \
  --views_per_sample 2 \
  --align_mode c_meanrms \
  --point_feats 7 \
  --register_noise_mode random \
  --flow_loss_type velocity \
  --flow_steps_per_recon 4 \
  --offpath_mode noise \
  --offpath_weight 1.0 \
  --offpath_noise_std 0.1 \
  --batch_size 5 \
  --lr 5e-5 \
  --lr_schedule constant \
  --freeze_recon \
  --freeze_vggt_builder \
  --recon_log_interval 1000 \
  --ckpt_interval "$CKPT_INTERVAL" \
  --vis_interval 5000 \
  --save_optimizer \
  --no-sample_renorm_output \
  --resume_ckpt "$PARENT" \
  --reset_optimizer_on_resume \
  --additional_steps "$ADDITIONAL_STEPS" \
  --wandb \
  --wandb_project shapepcunite \
  --wandb_name "$(basename "$OUT")"

echo "DONE. Eval: bash eval_exp21b_continue_no_cam.sh"
