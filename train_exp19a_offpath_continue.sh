#!/usr/bin/env bash
# E19-A: Exp18@100k continuation with Mode A off-path noise unchanged (+10k).
# Parent: runs/exp18_offpath_noise_ft_from_exp16/ckpt_0100000.pt
#
# Usage:
#   bash train_exp19a_offpath_continue.sh
#   DEVICE=cuda:0 bash train_exp19a_offpath_continue.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
set +u
conda activate hy3dgs
set -u

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DEVICE="${DEVICE:-cuda}"
PARENT="${PARENT:-runs/exp18_offpath_noise_ft_from_exp16/ckpt_0100000.pt}"
OUT="${OUT:-runs/exp19a_offpath_continue_from_exp18}"
DATA_DIR="${DATA_DIR:-/export/home/nathan/datasets/gobjaverse_experiments/furniture_351/train}"
RENDER_ROOT="${RENDER_ROOT:-/export/home/nathan/datasets}"
VGGT_CACHE="${VGGT_CACHE:-runs/vggt_cache/furn100_joint_pool_4_10_16_22}"
ADDITIONAL_STEPS="${ADDITIONAL_STEPS:-10000}"
CKPT_INTERVAL="${CKPT_INTERVAL:-5000}"

if [[ ! -f "$PARENT" ]]; then
  echo "ERROR: missing parent ckpt $PARENT" >&2
  exit 1
fi

echo "=== E19-A train  parent=$PARENT  out=$OUT  +${ADDITIONAL_STEPS} steps ==="
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
  --additional_steps "$ADDITIONAL_STEPS" \
  --wandb \
  --wandb_project shapepcunite \
  --wandb_name "$(basename "$OUT")"

echo "DONE. Final ckpt expected under $OUT (100k+${ADDITIONAL_STEPS})."
echo "Eval with: bash eval_exp19a_offpath_continue.sh"
