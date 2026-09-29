#!/usr/bin/env bash
# Full post-exp8 diagnosis suite (no retrain).
#
# Stages:
#   data_check         — aligned GT/patches, normals, view_rgb.png
#   diffusion_curve    — oracle CD + velocity MSE vs t
#   encode_vggt        — tok ± VGGT (sanity; flag should be off on exp8)
#   latent_align       — gen vs tok + isotropic excess + mean-latent CD
#   pca_trajectories   — linear bridge vs ODE waypoints in 2D PCA (plots)
#   cfg_sweep          — gen CD at CFG ∈ {1,1.5,2,3}
#   multiview_eval     — same meshes at several views
#
# Usage:
#   bash run_debug_exp8.sh
#   bash run_debug_exp8.sh --quick
#   STAGES="pca_trajectories latent_align" bash run_debug_exp8.sh
#   DATA=.../val EVAL=.../eval_val_ckpt20k/results.json bash run_debug_exp8.sh
#
# Outputs:
#   runs/debug_exp8/<timestamp>/
#     summary.json, README.md, run.log
#     data_check/  diffusion_curve/  encode_vggt/  latent_align/
#     pca_trajectories/plots/*.png
#     cfg_sweep/  multiview_eval/
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="${PY:-/export/home/nathan/miniconda3/envs/hy3dgs/bin/python}"
DATA="${DATA:-/export/home/nathan/datasets/gobjaverse_experiments/furniture_351/train}"
RENDER_ROOT="${RENDER_ROOT:-/export/home/nathan/datasets}"
VGGT_CACHE="${VGGT_CACHE:-runs/vggt_cache/furn4_mv40_cam_cross}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
VIEW_IDX="${VIEW_IDX:-0}"
ALIGN_MODE="${ALIGN_MODE:-cross}"
MAX_ITEMS="${MAX_ITEMS:-100}"

CKPT="${CKPT:-runs/exp8_n100_randomreg_repnoise_uniform_t/ckpt_0020000.pt}"
EVAL="${EVAL:-runs/exp8_n100_randomreg_repnoise_uniform_t/eval_ckpt20k/results.json}"

STAGES="${STAGES:-data_check diffusion_curve encode_vggt latent_align pca_trajectories cfg_sweep multiview_eval}"

QUICK=0
for arg in "$@"; do
  case "$arg" in
    --quick) QUICK=1 ;;
    --help|-h)
      sed -n '2,30p' "$0"
      exit 0
      ;;
  esac
done

if [[ "$QUICK" == "1" ]]; then
  N_WORST=2; N_BEST=2; N_RANDOM=2
  SAMPLE_STEPS=20
  NUM_DRAWS=3
  NUM_LATENT_DRAWS=1
  NUM_NOISE_DRAWS=1
  PCA_MAX_OBJECTS=4
  WAYPOINT_STRIDE=4
  EVAL_VIEWS=(0 5 10)
  echo "[quick] reduced objects / ODE steps / PCA draws"
else
  N_WORST=4; N_BEST=4; N_RANDOM=4
  SAMPLE_STEPS=50
  NUM_DRAWS=6
  NUM_LATENT_DRAWS=2
  NUM_NOISE_DRAWS=2
  PCA_MAX_OBJECTS=8
  WAYPOINT_STRIDE=5
  EVAL_VIEWS=(0 5 10 15 20)
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-runs/debug_exp8/${TS}}"
mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/run.log"

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: missing ckpt $CKPT"
  exit 1
fi

exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "exp8 diagnosis suite"
echo "  ckpt:   $CKPT"
echo "  eval:   $EVAL"
echo "  data:   $DATA"
echo "  out:    $OUT_ROOT"
echo "  stages: $STAGES"
echo "============================================================"

# shellcheck disable=SC2086
"$PY" debug_pc_unite.py \
  --ckpt "$CKPT" \
  --eval_results "$EVAL" \
  --data_dir "$DATA" \
  --gobjaverse_render_root "$RENDER_ROOT" \
  --vggt_cache_root "$VGGT_CACHE" \
  --align_mode "$ALIGN_MODE" \
  --view_idx "$VIEW_IDX" \
  --max_items "$MAX_ITEMS" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --n_worst "$N_WORST" \
  --n_best "$N_BEST" \
  --n_random "$N_RANDOM" \
  --sample_steps "$SAMPLE_STEPS" \
  --num_draws "$NUM_DRAWS" \
  --num_latent_draws "$NUM_LATENT_DRAWS" \
  --num_noise_draws "$NUM_NOISE_DRAWS" \
  --pca_max_objects "$PCA_MAX_OBJECTS" \
  --waypoint_stride "$WAYPOINT_STRIDE" \
  --t_grid 0.0 0.1 0.25 0.5 0.75 0.9 0.99 \
  --cfg_scales 1.0 1.5 2.0 3.0 \
  --eval_views "${EVAL_VIEWS[@]}" \
  --stages $STAGES \
  --output_dir "$OUT_ROOT"

# ---- README ----
"$PY" - <<'PY' "$OUT_ROOT" "$CKPT"
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
ckpt = sys.argv[2]
s = json.load(open(root / "summary.json")) if (root / "summary.json").exists() else {}
stages = s.get("stages", {})

lines = [
    f"# exp8 diagnosis\n",
    f"- ckpt: `{ckpt}`\n",
    f"- wall_s: {s.get('wall_seconds', '?')}\n\n",
    "## How to read\n",
    "- `data_check/`: overlay `gt.ply` + `patch_centers.ply` (aligned) + `view_rgb.png`\n",
    "- `diffusion_curve/oracle_curve.csv`: early-t cliff?\n",
    "- `latent_align/`: `excess_vs_isotropic`, `cd_of_mean_latent`\n",
    "- `pca_trajectories/plots/*.png`: linear bridge vs ODE paths (same noise)\n",
    "- `pca_trajectories/endpoints.csv`: rel_to_z / cos_to_mean vs t_start\n",
    "- `cfg_sweep/`: does CFG>1 help hard objects?\n",
    "- `multiview_eval/`: `frac_view_sensitive` — view-specific failures?\n\n",
    "## Summaries\n",
]
for name, st in stages.items():
    if not isinstance(st, dict):
        continue
    lines.append(f"### {name}\n")
    for k, v in st.items():
        if k in ("objects", "cases", "per_object", "oracle_curve", "velocity_curve"):
            continue
        if isinstance(v, float):
            lines.append(f"- `{k}`: {v:.5f}\n")
        elif isinstance(v, (int, bool, str)):
            lines.append(f"- `{k}`: {v}\n")
    lines.append("\n")

(root / "README.md").write_text("".join(lines))
print("Wrote", root / "README.md")
PY

echo ""
echo "============================================================"
echo "Done."
echo "  Log:    $LOG"
echo "  README: $OUT_ROOT/README.md"
echo "  PCA:    $OUT_ROOT/pca_trajectories/plots/"
echo "============================================================"
