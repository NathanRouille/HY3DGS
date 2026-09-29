#!/usr/bin/env bash
# Run full ShapePCUnite diagnostics (path_from_noise, velocity_along_ode,
# oracle_endpoints, pca_trajectories) for experiments that are still missing them.
#
# As of 2026-09-01 only exp6 and exp7 need re-eval; exp8–exp15 already have the
# newdiag suite on disk. This script skips any output dir that already contains
# diagnostics/summary.json unless FORCE=1.
#
# Usage (from HY3DGS, hy3dgs env active):
#   bash eval_missing_newdiag.sh
#
# Options (env vars):
#   ONLY=6,7              subset of experiment tags (default: all registered)
#   SPLITS=train,val      which splits to run (default: both)
#   SKIP_COMPLETE=1       skip dirs that already have diagnostics (default: 1)
#   FORCE=1               re-run even when diagnostics exist
#   MAX_ITEMS_TRAIN=100   train split cap (exp15 uses 100 for comparability)
#   MAX_ITEMS_VAL=35      val split cap
#   SKIP_PLY=1            CD + diagnostics only (no PLY export)
#   SAMPLE_STEPS=50       ODE steps (match prior evals)
#
# Quick smoke:
#   MAX_ITEMS_TRAIN=4 MAX_ITEMS_VAL=4 SKIP_PLY=1 bash eval_missing_newdiag.sh

set -euo pipefail
cd "$(dirname "$0")"

DATA_ROOT="${DATA_ROOT:-/export/home/nathan/datasets/gobjaverse_experiments/furniture_351}"
RENDER="${RENDER:-/export/home/nathan/datasets}"
CACHE_CROSS="${CACHE_CROSS:-runs/vggt_cache/furn4_mv40_cam_cross}"
MAX_ITEMS_TRAIN="${MAX_ITEMS_TRAIN:-100}"
MAX_ITEMS_VAL="${MAX_ITEMS_VAL:-35}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
ONLY="${ONLY:-}"
SPLITS="${SPLITS:-train,val}"
SKIP_COMPLETE="${SKIP_COMPLETE:-1}"
FORCE="${FORCE:-0}"
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

has_newdiag() {
  local out="$1"
  [[ "$FORCE" == "1" ]] && return 1
  [[ "$SKIP_COMPLETE" == "1" && -f "$out/diagnostics/summary.json" ]]
}

run_eval() {
  local tag="$1" ckpt="$2" out="$3" data_dir="$4" max_items="$5"
  shift 5
  local -a extra=("$@")

  if ! should_run "$tag"; then
    echo "=== skip $tag (ONLY filter) ==="
    return 0
  fi
  if [[ ! -f "$ckpt" ]]; then
    echo "=== MISSING ckpt for $tag: $ckpt ===" >&2
    return 1
  fi
  if has_newdiag "$out"; then
    echo "=== skip $tag → $out (diagnostics already present; set FORCE=1 to redo) ==="
    return 0
  fi

  echo "=== eval $tag max_items=$max_items → $out ==="
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

run_cross() {
  local tag="$1" ckpt="$2" out_train="$3" out_val="$4"
  local -a extra=(--vggt_cache_root "$CACHE_CROSS")
  if should_split train; then
    run_eval "$tag" "$ckpt" "$out_train" "$DATA_ROOT/train" "$MAX_ITEMS_TRAIN" "${extra[@]}"
  fi
  if should_split val; then
    run_eval "$tag" "$ckpt" "$out_val" "$DATA_ROOT/val" "$MAX_ITEMS_VAL" "${extra[@]}"
  fi
}

run_mv2_online() {
  local tag="$1" ckpt="$2" out_train="$3" out_val="$4"
  local -a extra=(--views_per_sample 2 --align_mode c_meanrms)
  if should_split train; then
    run_eval "$tag" "$ckpt" "$out_train" "$DATA_ROOT/train" "$MAX_ITEMS_TRAIN" "${extra[@]}"
  fi
  if should_split val; then
    run_eval "$tag" "$ckpt" "$out_val" "$DATA_ROOT/val" "$MAX_ITEMS_VAL" "${extra[@]}"
  fi
}

run_joint() {
  local tag="$1" ckpt="$2" cache="$3" view_indices="$4" out_train="$5" out_val="$6"
  local -a extra=(
    --vggt_cache_root "$cache"
    --vggt_joint_cache
    --views_per_sample 2
    --align_mode c_meanrms
    --view_indices "$view_indices"
  )
  if should_split train; then
    run_eval "$tag" "$ckpt" "$out_train" "$DATA_ROOT/train" "$MAX_ITEMS_TRAIN" "${extra[@]}"
  fi
  if should_split val; then
    run_eval "$tag" "$ckpt" "$out_val" "$DATA_ROOT/val" "$MAX_ITEMS_VAL" "${extra[@]}"
  fi
}

echo "=== eval_missing_newdiag.sh ==="
echo "SKIP_COMPLETE=$SKIP_COMPLETE  FORCE=$FORCE  SPLITS=$SPLITS  ONLY=${ONLY:-all}"

# --- Missing (old eval lacks diagnostics) -----------------------------------
run_cross 6 \
  runs/exp6_n100_randomreg_repnoise/ckpt_0017500.pt \
  runs/exp6_n100_randomreg_repnoise/eval_ckpt17k_newdiag \
  runs/exp6_n100_randomreg_repnoise/eval_val_ckpt17k_newdiag

run_cross 7 \
  runs/exp7_n100_randomreg_repnoise_tokvggt_v2/ckpt_0020000.pt \
  runs/exp7_n100_randomreg_repnoise_tokvggt_v2/eval_ckpt20k_newdiag \
  runs/exp7_n100_randomreg_repnoise_tokvggt_v2/eval_val_ckpt20k_newdiag

# --- Already complete (skipped unless FORCE=1) --------------------------------
run_cross 8 \
  runs/exp8_n100_randomreg_repnoise_uniform_t/ckpt_0020000.pt \
  runs/exp8_n100_randomreg_repnoise_uniform_t/eval_ckpt20k_newdiag \
  runs/exp8_n100_randomreg_repnoise_uniform_t/eval_val_ckpt20k_newdiag

run_cross 9 \
  runs/exp9_n100_randomreg_repnoise_uniform_t_adaln_cam/ckpt_0020000.pt \
  runs/exp9_n100_randomreg_repnoise_uniform_t_adaln_cam/eval_ckpt20k_newdiag \
  runs/exp9_n100_randomreg_repnoise_uniform_t_adaln_cam/eval_val_ckpt20k_newdiag

run_cross 10b \
  runs/exp10b_flowft_xstart_from_exp8/ckpt_0030000.pt \
  runs/exp10b_flowft_xstart_from_exp8/eval_ckpt30k_newdiag \
  runs/exp10b_flowft_xstart_from_exp8/eval_val_ckpt30k_newdiag

run_cross 11 \
  runs/exp11_n100_pointdit_t0_tok0/ckpt_0020000.pt \
  runs/exp11_n100_pointdit_t0_tok0/eval_ckpt20k_newdiag \
  runs/exp11_n100_pointdit_t0_tok0/eval_val_ckpt20k_newdiag

run_mv2_online 12 \
  runs/exp12_n100_mv2_c_meanrms/ckpt_0030000.pt \
  runs/exp12_n100_mv2_c_meanrms/eval_ckpt30k_mv2_newdiag \
  runs/exp12_n100_mv2_c_meanrms/eval_val_ckpt30k_mv2_newdiag

run_joint 13 \
  runs/exp13_n100_mv2_joint_v16_v29/ckpt_0020000.pt \
  runs/vggt_cache/furn4_joint_v16_v29 \
  "16,29" \
  runs/exp13_n100_mv2_joint_v16_v29/eval_ckpt20k_newdiag \
  runs/exp13_n100_mv2_joint_v16_v29/eval_val_ckpt20k_newdiag

run_joint 14 \
  runs/exp14_n100_mv2_joint_pool_4_10_16_22/ckpt_0030000.pt \
  runs/vggt_cache/furn100_joint_pool_4_10_16_22 \
  "4,16" \
  runs/exp14_n100_mv2_joint_pool_4_10_16_22/eval_ckpt30k_v4_16_newdiag \
  runs/exp14_n100_mv2_joint_pool_4_10_16_22/eval_val_ckpt30k_v4_16_newdiag

run_joint 14@20k \
  runs/exp14_n100_mv2_joint_pool_4_10_16_22/ckpt_0020000.pt \
  runs/vggt_cache/furn100_joint_pool_4_10_16_22 \
  "4,16" \
  runs/exp14_n100_mv2_joint_pool_4_10_16_22/eval_ckpt20k_v4_16_newdiag \
  runs/exp14_n100_mv2_joint_pool_4_10_16_22/eval_val_ckpt20k_v4_16_newdiag

# exp15: train eval uses first-100 subset; outputs already exist (no _newdiag suffix)
run_joint 15@30k \
  runs/exp15_nfull_mv2_joint_pool_4_10_16_22/ckpt_0030000.pt \
  runs/vggt_cache/furn100_joint_pool_4_10_16_22 \
  "4,16" \
  runs/exp15_nfull_mv2_joint_pool_4_10_16_22/eval_train100_ckpt30k_v4_16 \
  runs/exp15_nfull_mv2_joint_pool_4_10_16_22/eval_val_ckpt30k_v4_16

run_joint 15@20k \
  runs/exp15_nfull_mv2_joint_pool_4_10_16_22/ckpt_0020000.pt \
  runs/vggt_cache/furn100_joint_pool_4_10_16_22 \
  "4,16" \
  runs/exp15_nfull_mv2_joint_pool_4_10_16_22/eval_train100_ckpt20k_v4_16 \
  runs/exp15_nfull_mv2_joint_pool_4_10_16_22/eval_val_ckpt20k_v4_16

echo "=== all requested evals finished ==="
echo ""
echo "Outputs: <run_dir>/eval_*_newdiag/diagnostics/{path_from_noise,velocity_along_ode,oracle_endpoints,pca_trajectories}/"
echo "Tip: ONLY=6,7 bash eval_missing_newdiag.sh   # just the missing exps"
echo "Tip: FORCE=1 ONLY=8 bash eval_missing_newdiag.sh   # re-run one exp"
