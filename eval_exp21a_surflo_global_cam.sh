#!/usr/bin/env bash
# E0 eval for Exp21-A (Surflo global-cam). Val every 5k from 115k→130k; train100 @ final.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
set +u
conda activate hy3dgs
set -u

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_DIR="${RUN_DIR:-runs/exp21a_surflo_global_cam_from_exp19a}"
DEVICE="${DEVICE:-cuda}"
VAL_DIR="${VAL_DIR:-/export/home/nathan/datasets/gobjaverse_experiments/furniture_351/val}"
TRAIN_DIR="${TRAIN_DIR:-/export/home/nathan/datasets/gobjaverse_experiments/furniture_351/train}"
RENDER_ROOT="${RENDER_ROOT:-/export/home/nathan/datasets}"
VGGT_CACHE="${VGGT_CACHE:-runs/vggt_cache/furn100_joint_pool_4_10_16_22}"
VIEW_INDICES="${VIEW_INDICES:-4,16}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
TRAIN_MAX_ITEMS="${TRAIN_MAX_ITEMS:-100}"
SEED="${SEED:-0}"
GEN_SEEDS_STR="${GEN_SEEDS:-0 1 2}"
# shellcheck disable=SC2206
GEN_SEEDS=($GEN_SEEDS_STR)

if [[ -z "${GUIDANCE_SCALES+x}" ]]; then
  GUIDANCE_SCALES=(1.0 1.5 2.0 3.0)
fi
if [[ -z "${TRAIN_GUIDANCE_SCALES+x}" ]]; then
  TRAIN_GUIDANCE_SCALES=(1.0 2.0 3.0)
fi

VAL_STEPS_STR="${VAL_STEPS:-115k 120k 125k 130k}"
# shellcheck disable=SC2206
VAL_STEPS=($VAL_STEPS_STR)

step_to_ckpt() {
  local tag="$1"
  case "$tag" in
    115k) echo "$RUN_DIR/ckpt_0115000.pt" ;;
    120k) echo "$RUN_DIR/ckpt_0120000.pt" ;;
    125k) echo "$RUN_DIR/ckpt_0125000.pt" ;;
    130k) echo "$RUN_DIR/ckpt_0130000.pt" ;;
    final) echo "$RUN_DIR/ckpt_final.pt" ;;
    *) echo "ERROR: unknown step tag '$tag'" >&2; exit 1 ;;
  esac
}

run_eval() {
  local split="$1"; local step_tag="$2"; local ckpt="$3"; local data_dir="$4"; local out="$5"
  shift 5
  if [[ ! -f "$ckpt" ]]; then echo "ERROR: missing ckpt $ckpt" >&2; exit 1; fi
  echo "=== ${split} @ ${step_tag}  ckpt=$ckpt  out=$out ==="
  python evaluate_pc_unite.py \
    --ckpt "$ckpt" \
    --output_dir "$out" \
    --data_dir "$data_dir" \
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
    --diagnostics --diag_path --diag_oracle --diag_velocity \
    --device "$DEVICE" \
    "$@"
}

echo "RUN_DIR=$RUN_DIR  gen_seeds=${GEN_SEEDS[*]}"
for step_tag in "${VAL_STEPS[@]}"; do
  run_eval val "$step_tag" "$(step_to_ckpt "$step_tag")" "$VAL_DIR" \
    "$RUN_DIR/eval_val_ckpt${step_tag}_v4_16_e0"
done

FINAL_TAG="${FINAL_TAG:-130k}"
FINAL_CKPT="$(step_to_ckpt "$FINAL_TAG")"
[[ -f "$FINAL_CKPT" ]] || FINAL_CKPT="$RUN_DIR/ckpt_final.pt"
GUIDANCE_SCALES=("${TRAIN_GUIDANCE_SCALES[@]}")
run_eval train100 "$FINAL_TAG" "$FINAL_CKPT" "$TRAIN_DIR" \
  "$RUN_DIR/eval_train100_ckpt${FINAL_TAG}_v4_16_e0" \
  --max_items "$TRAIN_MAX_ITEMS"

echo "DONE. Prefer summary['unique/*'] vs Exp21-B and E19-A@110k."
