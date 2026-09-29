#!/usr/bin/env bash
# Eval exp17b (x_start FT from exp16@70k): val @ 75k/80k/85k with CFG sweep +
# train100 @ final. Same protocol as the exp16/exp17 diagnosis runs.
#
# Usage:
#   bash eval_exp17b_xstart.sh
#   RUN_DIR=runs/exp17b_xstart_ft_from_exp16 bash eval_exp17b_xstart.sh
#   DEVICE=cuda:0 bash eval_exp17b_xstart.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Conda hooks reference unset vars (e.g. CUDAARCHS_BACKUP); don't let `set -u` abort.
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
set +u
conda activate hy3dgs
set -u

RUN_DIR="${RUN_DIR:-runs/exp17b_xstart_ft_from_exp16}"
DEVICE="${DEVICE:-cuda}"
VAL_DIR="${VAL_DIR:-/export/home/nathan/datasets/gobjaverse_experiments/furniture_351/val}"
TRAIN_DIR="${TRAIN_DIR:-/export/home/nathan/datasets/gobjaverse_experiments/furniture_351/train}"
RENDER_ROOT="${RENDER_ROOT:-/export/home/nathan/datasets}"
VGGT_CACHE="${VGGT_CACHE:-runs/vggt_cache/furn100_joint_pool_4_10_16_22}"
VIEW_INDICES="${VIEW_INDICES:-4,16}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
TRAIN_MAX_ITEMS="${TRAIN_MAX_ITEMS:-100}"

# Default CFG sweeps (override by exporting a bash array before invoking, or edit here).
if [[ -z "${GUIDANCE_SCALES+x}" ]]; then
  GUIDANCE_SCALES=(1.0 1.5 2.0 3.0)
fi
if [[ -z "${TRAIN_GUIDANCE_SCALES+x}" ]]; then
  TRAIN_GUIDANCE_SCALES=(1.0 2.0 3.0)
fi

eval_val() {
  local step_tag="$1"
  local ckpt="$2"
  local out="$RUN_DIR/eval_val_ckpt${step_tag}_v4_16"
  if [[ ! -f "$ckpt" ]]; then
    echo "ERROR: missing ckpt $ckpt" >&2
    exit 1
  fi
  echo "=== VAL @ ${step_tag}  ckpt=$ckpt  out=$out ==="
  python evaluate_pc_unite.py \
    --ckpt "$ckpt" \
    --output_dir "$out" \
    --data_dir "$VAL_DIR" \
    --gobjaverse_render_root "$RENDER_ROOT" \
    --vggt_cache_root "$VGGT_CACHE" \
    --vggt_joint_cache \
    --view_indices "$VIEW_INDICES" \
    --views_per_sample 2 \
    --sample_steps "$SAMPLE_STEPS" \
    --guidance_scales "${GUIDANCE_SCALES[@]}" \
    --device "$DEVICE"
}

echo "RUN_DIR=$RUN_DIR  DEVICE=$DEVICE  views=$VIEW_INDICES  steps=$SAMPLE_STEPS"
echo "CFG val: ${GUIDANCE_SCALES[*]}  CFG train: ${TRAIN_GUIDANCE_SCALES[*]}"

# Convergence curve (+5k / +10k / +15k FT from exp16@70k)
eval_val 75k "$RUN_DIR/ckpt_0075000.pt"
eval_val 80k "$RUN_DIR/ckpt_0080000.pt"
eval_val 85k "$RUN_DIR/ckpt_final.pt"

FINAL_CKPT="$RUN_DIR/ckpt_final.pt"
TRAIN_OUT="$RUN_DIR/eval_train100_ckpt85k_v4_16"
echo "=== TRAIN100 @ 85k  ckpt=$FINAL_CKPT  out=$TRAIN_OUT ==="
python evaluate_pc_unite.py \
  --ckpt "$FINAL_CKPT" \
  --data_dir "$TRAIN_DIR" \
  --max_items "$TRAIN_MAX_ITEMS" \
  --output_dir "$TRAIN_OUT" \
  --gobjaverse_render_root "$RENDER_ROOT" \
  --vggt_cache_root "$VGGT_CACHE" \
  --vggt_joint_cache \
  --view_indices "$VIEW_INDICES" \
  --views_per_sample 2 \
  --sample_steps "$SAMPLE_STEPS" \
  --guidance_scales "${TRAIN_GUIDANCE_SCALES[@]}" \
  --device "$DEVICE"

echo "DONE. Results under:"
echo "  $RUN_DIR/eval_val_ckpt75k_v4_16/results.json"
echo "  $RUN_DIR/eval_val_ckpt80k_v4_16/results.json"
echo "  $RUN_DIR/eval_val_ckpt85k_v4_16/results.json"
echo "  $RUN_DIR/eval_train100_ckpt85k_v4_16/results.json"
