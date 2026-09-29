#!/usr/bin/env bash
# One-shot comparison: exp14 vs NOVA3R vs Surflo on furniture_351.
#
# Each model runs in its own conda env via subprocess; no edits to those repos.
#
# Usage (from HY3DGS repo root):
#   bash compare_baselines/run_compare.sh val
#   bash compare_baselines/run_compare.sh train --max_items 100
#   bash compare_baselines/run_compare.sh val --skip_prepare --skip_exp14
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SPLIT="${1:-val}"
shift || true

# Defaults (override via env)
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
HY3DGS_ENV="${HY3DGS_ENV:-hy3dgs}"
NOVA3R_ENV="${NOVA3R_ENV:-nova3r}"
SURFLO_ENV="${SURFLO_ENV:-surflo-cu124}"
DATA_ROOT="${DATA_ROOT:-/export/home/nathan/datasets/gobjaverse_experiments/furniture_351}"
RENDER_ROOT="${RENDER_ROOT:-/export/home/nathan/datasets}"
EXP14_CKPT="${EXP14_CKPT:-runs/exp14_n100_mv2_joint_pool_4_10_16_22/ckpt_0030000.pt}"
VGGT_CACHE="${VGGT_CACHE:-runs/vggt_cache/furn100_joint_pool_4_10_16_22}"
NOVA3R_ROOT="${NOVA3R_ROOT:-/export/home/nathan/nova3r}"
NOVA3R_CKPT="${NOVA3R_CKPT:-/export/home/nathan/nova3r/checkpoints/scene_n2/checkpoint-last.pth}"
SURFLO_ROOT="${SURFLO_ROOT:-/export/home/nathan/Surflo}"
SURFLO_CKPT="${SURFLO_CKPT:-/export/home/nathan/Surflo/checkpoints/surflo_v0.pt}"
OUT_ROOT="${OUT_ROOT:-runs/compare_exp14_baselines/${SPLIT}}"
# Absolute paths so subprocesses that chdir (nova3r) still find the compare dir
if [[ "$OUT_ROOT" != /* ]]; then
  OUT_ROOT="$ROOT/$OUT_ROOT"
fi
if [[ "$EXP14_CKPT" != /* ]]; then
  EXP14_CKPT="$ROOT/$EXP14_CKPT"
fi
if [[ "$VGGT_CACHE" != /* ]]; then
  VGGT_CACHE="$ROOT/$VGGT_CACHE"
fi

SKIP_PREPARE=0
SKIP_EXP14=0
SKIP_NOVA=0
SKIP_SURFLO=0
SKIP_SCORE=0
MAX_ITEMS=""
LIMIT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir) OUT_ROOT="$2"; shift 2 ;;
    --max_items) MAX_ITEMS="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --skip_prepare) SKIP_PREPARE=1; shift ;;
    --skip_exp14) SKIP_EXP14=1; shift ;;
    --skip_nova3r) SKIP_NOVA=1; shift ;;
    --skip_surflo) SKIP_SURFLO=1; shift ;;
    --skip_score) SKIP_SCORE=1; shift ;;
    --exp14_ckpt) EXP14_CKPT="$2"; shift 2 ;;
    --surflo_ckpt) SURFLO_CKPT="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

# shellcheck disable=SC1090
source "$CONDA_SH"

run_hy3dgs() {
  conda run --no-capture-output -n "$HY3DGS_ENV" "$@"
}
run_nova() {
  conda run --no-capture-output -n "$NOVA3R_ENV" "$@"
}
run_surflo() {
  conda run --no-capture-output -n "$SURFLO_ENV" "$@"
}

echo "============================================================"
echo " Compare baselines on furniture_351 / $SPLIT"
echo " Output: $OUT_ROOT"
echo "============================================================"

mkdir -p "$OUT_ROOT"

if [[ "$SKIP_PREPARE" -eq 0 ]]; then
  echo ""
  echo "=== [1/5] prepare_inputs ($HY3DGS_ENV) ==="
  PREP=(python compare_baselines/prepare_inputs.py
    --split "$SPLIT"
    --data_root "$DATA_ROOT"
    --gobjaverse_render_root "$RENDER_ROOT"
    --output_dir "$OUT_ROOT"
    --view_indices "4,16"
    --align_mode c_meanrms
  )
  if [[ -n "$MAX_ITEMS" ]]; then
    PREP+=(--max_items "$MAX_ITEMS")
  fi
  run_hy3dgs "${PREP[@]}"
else
  echo "=== [1/5] prepare_inputs SKIPPED ==="
fi

if [[ ! -f "$OUT_ROOT/manifest.json" ]]; then
  echo "ERROR: missing $OUT_ROOT/manifest.json — run prepare first." >&2
  exit 1
fi

LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

if [[ "$SKIP_EXP14" -eq 0 ]]; then
  echo ""
  echo "=== [2/5] infer_exp14 ($HY3DGS_ENV) ==="
  run_hy3dgs python compare_baselines/infer_exp14.py \
    --compare_dir "$OUT_ROOT" \
    --ckpt "$EXP14_CKPT" \
    --gobjaverse_render_root "$RENDER_ROOT" \
    --vggt_cache_root "$VGGT_CACHE"
else
  echo "=== [2/5] infer_exp14 SKIPPED ==="
fi

if [[ "$SKIP_NOVA" -eq 0 ]]; then
  echo ""
  echo "=== [3/5] infer_nova3r ($NOVA3R_ENV) ==="
  run_nova python "$ROOT/compare_baselines/infer_nova3r.py" \
    --compare_dir "$OUT_ROOT" \
    --nova3r_root "$NOVA3R_ROOT" \
    --ckpt "$NOVA3R_CKPT" \
    --resolution 518 392 \
    "${LIMIT_ARGS[@]}"
else
  echo "=== [3/5] infer_nova3r SKIPPED ==="
fi

if [[ "$SKIP_SURFLO" -eq 0 ]]; then
  echo ""
  echo "=== [4/5] infer_surflo ($SURFLO_ENV) ==="
  run_surflo python "$ROOT/compare_baselines/infer_surflo.py" \
    --compare_dir "$OUT_ROOT" \
    --surflo_root "$SURFLO_ROOT" \
    --ckpt "$SURFLO_CKPT" \
    "${LIMIT_ARGS[@]}"
else
  echo "=== [4/5] infer_surflo SKIPPED ==="
fi

if [[ "$SKIP_SCORE" -eq 0 ]]; then
  echo ""
  echo "=== [5/5] align_and_score ($HY3DGS_ENV) — multi-align exports ==="
  run_hy3dgs python compare_baselines/align_and_score.py \
    --compare_dir "$OUT_ROOT"
else
  echo "=== [5/5] align_and_score SKIPPED ==="
fi

echo ""
echo "Done. See:"
echo "  $OUT_ROOT/results_table_by_align.txt"
echo "  $OUT_ROOT/results_by_align.json"
echo "  $OUT_ROOT/objects/*/pred_aligned/primary/{gt,exp14,nova3r,surflo}.ply"
echo "Align: gt_anchored + robust_filter + pred_aligned/primary/ (per-model recipe)"
