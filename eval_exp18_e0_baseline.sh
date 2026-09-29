#!/usr/bin/env bash
# Re-score Exp18@100k under the E0 primary deterministic protocol.
# Use these unique/* numbers as the baseline before comparing E19-A/B.
#
# Usage:
#   bash eval_exp18_e0_baseline.sh
#   GEN_SEEDS="0 1 2" bash eval_exp18_e0_baseline.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
set +u
conda activate hy3dgs
set -u

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_DIR="${RUN_DIR:-runs/exp18_offpath_noise_ft_from_exp16}"
CKPT="${CKPT:-$RUN_DIR/ckpt_0100000.pt}"
DEVICE="${DEVICE:-cuda}"
VAL_DIR="${VAL_DIR:-/export/home/nathan/datasets/gobjaverse_experiments/furniture_351/val}"
TRAIN_DIR="${TRAIN_DIR:-/export/home/nathan/datasets/gobjaverse_experiments/furniture_351/train}"
RENDER_ROOT="${RENDER_ROOT:-/export/home/nathan/datasets}"
VGGT_CACHE="${VGGT_CACHE:-runs/vggt_cache/furn100_joint_pool_4_10_16_22}"
VIEW_INDICES="${VIEW_INDICES:-4,16}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
TRAIN_MAX_ITEMS="${TRAIN_MAX_ITEMS:-100}"
SEED="${SEED:-0}"
GEN_SEEDS_STR="${GEN_SEEDS:-0}"
# shellcheck disable=SC2206
GEN_SEEDS=($GEN_SEEDS_STR)

if [[ -z "${GUIDANCE_SCALES+x}" ]]; then
  GUIDANCE_SCALES=(1.0 1.5 2.0 3.0)
fi
if [[ -z "${TRAIN_GUIDANCE_SCALES+x}" ]]; then
  TRAIN_GUIDANCE_SCALES=(1.0 2.0 3.0)
fi

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: missing ckpt $CKPT" >&2
  exit 1
fi

VAL_OUT="$RUN_DIR/eval_val_ckpt100k_v4_16_e0"
TRAIN_OUT="$RUN_DIR/eval_train100_ckpt100k_v4_16_e0"

echo "=== Exp18 E0 VAL  ckpt=$CKPT  out=$VAL_OUT ==="
python evaluate_pc_unite.py \
  --ckpt "$CKPT" \
  --output_dir "$VAL_OUT" \
  --data_dir "$VAL_DIR" \
  --gobjaverse_render_root "$RENDER_ROOT" \
  --vggt_cache_root "$VGGT_CACHE" \
  --vggt_joint_cache \
  --view_indices "$VIEW_INDICES" \
  --views_per_sample 2 \
  --sample_steps "$SAMPLE_STEPS" \
  --guidance_scales "${GUIDANCE_SCALES[@]}" \
  --seed "$SEED" \
  --gen_seeds "${GEN_SEEDS[@]}" \
  --strict_load \
  --no-sample_renorm_output \
  --diagnostics \
  --diag_path \
  --diag_oracle \
  --diag_velocity \
  --device "$DEVICE"

echo "=== Exp18 E0 TRAIN100  out=$TRAIN_OUT ==="
python evaluate_pc_unite.py \
  --ckpt "$CKPT" \
  --output_dir "$TRAIN_OUT" \
  --data_dir "$TRAIN_DIR" \
  --max_items "$TRAIN_MAX_ITEMS" \
  --gobjaverse_render_root "$RENDER_ROOT" \
  --vggt_cache_root "$VGGT_CACHE" \
  --vggt_joint_cache \
  --view_indices "$VIEW_INDICES" \
  --views_per_sample 2 \
  --sample_steps "$SAMPLE_STEPS" \
  --guidance_scales "${TRAIN_GUIDANCE_SCALES[@]}" \
  --seed "$SEED" \
  --gen_seeds "${GEN_SEEDS[@]}" \
  --strict_load \
  --no-sample_renorm_output \
  --diagnostics \
  --diag_path \
  --diag_oracle \
  --diag_velocity \
  --device "$DEVICE"

echo "DONE."
echo "  $VAL_OUT/results.json   (use summary['unique/gen_cd_cfg2'] etc.)"
echo "  $TRAIN_OUT/results.json"
echo "  $VAL_OUT/eval_metadata.json"
