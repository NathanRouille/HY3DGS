#!/usr/bin/env bash
# Evaluate ShapePCUnite experiments 8–12 with the current diagnostics suite.
# Runs both train and val for each experiment. Exp12 uses 2-view c_meanrms.
#
# Usage (from HY3DGS, with hy3dgs env active):
#   bash eval_exp8_to_12.sh
#   MAX_ITEMS_TRAIN=20 MAX_ITEMS_VAL=10 bash eval_exp8_to_12.sh   # quick smoke
#   ONLY=10b,12 bash eval_exp8_to_12.sh                          # subset of exps
#   SPLITS=train bash eval_exp8_to_12.sh                         # train only
#   SPLITS=val bash eval_exp8_to_12.sh                           # val only
#   SKIP_PLY=1 bash eval_exp8_to_12.sh                           # CD+diag only

set -euo pipefail
cd "$(dirname "$0")"

DATA_ROOT="${DATA_ROOT:-/export/home/nathan/datasets/gobjaverse_experiments/furniture_351}"
RENDER="${RENDER:-/export/home/nathan/datasets}"
CACHE="${CACHE:-runs/vggt_cache/furn4_mv40_cam_cross}"
MAX_ITEMS_TRAIN="${MAX_ITEMS_TRAIN:-100}"
MAX_ITEMS_VAL="${MAX_ITEMS_VAL:-35}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
ONLY="${ONLY:-}"
SPLITS="${SPLITS:-train,val}"
SKIP_PLY="${SKIP_PLY:-0}"

PLY_ARGS=(--export_ply --export_recon_debug)
if [[ "$SKIP_PLY" == "1" ]]; then
  PLY_ARGS=(--no-export_ply --no-export_recon_debug)
fi

should_run() {
  local tag="$1"
  [[ -z "$ONLY" ]] && return 0
  [[ ",$ONLY," == *",$tag,"* ]]
}

should_split() {
  local split="$1"
  [[ ",$SPLITS," == *",$split,"* ]]
}

run_one() {
  local tag="$1" ckpt="$2" out="$3" data_dir="$4" max_items="$5" mode="$6"
  if ! should_run "$tag"; then
    echo "=== skip $tag ==="
    return 0
  fi
  if [[ ! -f "$ckpt" ]]; then
    echo "=== MISSING ckpt for $tag: $ckpt ===" >&2
    return 1
  fi
  echo "=== eval $tag [$mode] max_items=$max_items → $out ==="
  local extra=()
  if [[ "$mode" == "cross" ]]; then
    extra=(--vggt_cache_root "$CACHE")
  else
    extra=(--views_per_sample 2 --align_mode c_meanrms)
  fi
  python evaluate_pc_unite.py \
    --data_dir "$data_dir" \
    --gobjaverse_render_root "$RENDER" \
    --max_items "$max_items" \
    --sample_steps "$SAMPLE_STEPS" \
    --view_idx 0 \
    --no-sample_renorm_output \
    --diagnostics \
    "${PLY_ARGS[@]}" \
    --ckpt "$ckpt" \
    --output_dir "$out" \
    "${extra[@]}"
}

run_exp() {
  local tag="$1" ckpt="$2" out_train="$3" out_val="$4" mode="$5"
  if should_split train; then
    run_one "$tag" "$ckpt" "$out_train" \
      "$DATA_ROOT/train" "$MAX_ITEMS_TRAIN" "$mode"
  fi
  if should_split val; then
    run_one "$tag" "$ckpt" "$out_val" \
      "$DATA_ROOT/val" "$MAX_ITEMS_VAL" "$mode"
  fi
}

run_exp 8 \
  runs/exp8_n100_randomreg_repnoise_uniform_t/ckpt_0020000.pt \
  runs/exp8_n100_randomreg_repnoise_uniform_t/eval_ckpt20k_newdiag \
  runs/exp8_n100_randomreg_repnoise_uniform_t/eval_val_ckpt20k_newdiag \
  cross

run_exp 9 \
  runs/exp9_n100_randomreg_repnoise_uniform_t_adaln_cam/ckpt_0020000.pt \
  runs/exp9_n100_randomreg_repnoise_uniform_t_adaln_cam/eval_ckpt20k_newdiag \
  runs/exp9_n100_randomreg_repnoise_uniform_t_adaln_cam/eval_val_ckpt20k_newdiag \
  cross

run_exp 10b \
  runs/exp10b_flowft_xstart_from_exp8/ckpt_0030000.pt \
  runs/exp10b_flowft_xstart_from_exp8/eval_ckpt30k_newdiag \
  runs/exp10b_flowft_xstart_from_exp8/eval_val_ckpt30k_newdiag \
  cross

run_exp 11 \
  runs/exp11_n100_pointdit_t0_tok0/ckpt_0020000.pt \
  runs/exp11_n100_pointdit_t0_tok0/eval_ckpt20k_newdiag \
  runs/exp11_n100_pointdit_t0_tok0/eval_val_ckpt20k_newdiag \
  cross

run_exp 12 \
  runs/exp12_n100_mv2_c_meanrms/ckpt_0030000.pt \
  runs/exp12_n100_mv2_c_meanrms/eval_ckpt30k_mv2_newdiag \
  runs/exp12_n100_mv2_c_meanrms/eval_val_ckpt30k_mv2_newdiag \
  mv2

echo "=== all requested evals finished ==="
