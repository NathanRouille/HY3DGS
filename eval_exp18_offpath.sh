#!/usr/bin/env bash
# Eval exp18 (off-path noise FT from exp16@70k): val @ every 5k ckpt with CFG
# sweep + train100 @ final with CFG sweep. Same protocol as exp16/exp17b.
#
# Usage:
#   bash eval_exp18_offpath.sh
#   RUN_DIR=runs/exp18_offpath_noise_ft_from_exp16 bash eval_exp18_offpath.sh
#   DEVICE=cuda:0 bash eval_exp18_offpath.sh
#   # subset of val ckpts:
#   VAL_STEPS="90k 95k 100k" bash eval_exp18_offpath.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Conda hooks reference unset vars (e.g. CUDAARCHS_BACKUP); don't let `set -u` abort.
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
set +u
conda activate hy3dgs
set -u

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_DIR="${RUN_DIR:-runs/exp18_offpath_noise_ft_from_exp16}"
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

# Val checkpoints: FT from exp16@70k → 100k (every 5k). Override with e.g.
#   VAL_STEPS="90k 95k 100k"
VAL_STEPS_STR="${VAL_STEPS:-75k 80k 85k 90k 95k 100k}"
# shellcheck disable=SC2206
VAL_STEPS=($VAL_STEPS_STR)

step_to_ckpt() {
  local tag="$1"
  case "$tag" in
    75k)  echo "$RUN_DIR/ckpt_0075000.pt" ;;
    80k)  echo "$RUN_DIR/ckpt_0080000.pt" ;;
    85k)  echo "$RUN_DIR/ckpt_0085000.pt" ;;
    90k)  echo "$RUN_DIR/ckpt_0090000.pt" ;;
    95k)  echo "$RUN_DIR/ckpt_0095000.pt" ;;
    100k) echo "$RUN_DIR/ckpt_0100000.pt" ;;
    final) echo "$RUN_DIR/ckpt_final.pt" ;;
    *)
      echo "ERROR: unknown step tag '$tag'" >&2
      exit 1
      ;;
  esac
}

eval_val() {
  local step_tag="$1"
  local ckpt
  ckpt="$(step_to_ckpt "$step_tag")"
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
    --no-sample_renorm_output \
    --device "$DEVICE"
}

echo "RUN_DIR=$RUN_DIR  DEVICE=$DEVICE  views=$VIEW_INDICES  steps=$SAMPLE_STEPS"
echo "VAL_STEPS: ${VAL_STEPS[*]}"
echo "CFG val: ${GUIDANCE_SCALES[*]}  CFG train: ${TRAIN_GUIDANCE_SCALES[*]}"

for step_tag in "${VAL_STEPS[@]}"; do
  eval_val "$step_tag"
done

FINAL_CKPT="$(step_to_ckpt 100k)"
if [[ ! -f "$FINAL_CKPT" ]]; then
  FINAL_CKPT="$RUN_DIR/ckpt_final.pt"
fi
TRAIN_OUT="$RUN_DIR/eval_train100_ckpt100k_v4_16"
echo "=== TRAIN100 @ 100k  ckpt=$FINAL_CKPT  out=$TRAIN_OUT ==="
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
  --no-sample_renorm_output \
  --device "$DEVICE"

echo "DONE. Results under:"
for step_tag in "${VAL_STEPS[@]}"; do
  echo "  $RUN_DIR/eval_val_ckpt${step_tag}_v4_16/results.json"
done
echo "  $TRAIN_OUT/results.json"
