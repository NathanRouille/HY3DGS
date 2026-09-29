#!/usr/bin/env bash
# Run all ShapePCUnite debug stages (no retraining) on baseline + tok-VGGT ckpts.
#
# Stages per checkpoint:
#   1. data_check      — aligned GT+patch_centers PLYs, view_rgb.png, normals nx/ny/nz
#   2. diffusion_curve — oracle CD + velocity error vs t ∈ {0,0.1,...,0.99}
#   3. encode_vggt     — encode with/without VGGT (tokenizer dependence)
#   4. latent_align    — gen vs tok latent distance + isotropic excess test
#
# Re-export visuals only after fixing alignment:
#   STAGES=data_check ONLY=exp4 bash run_debug_pc_unite.sh --quick
#
# Usage:
#   bash run_debug_pc_unite.sh
#   bash run_debug_pc_unite.sh --quick          # fewer objects / fewer ODE steps
#   STAGES="diffusion_curve encode_vggt" bash run_debug_pc_unite.sh
#   ONLY=exp7 bash run_debug_pc_unite.sh
#
# Outputs:
#   runs/debug_pc_unite/<timestamp>/
#     exp4_ckpt20k/ ...
#     exp7_ckpt20k/ ...
#     COMPARISON.md
#     run.log
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

# Default: all stages. Override e.g. STAGES="diffusion_curve encode_vggt"
STAGES="${STAGES:-data_check diffusion_curve encode_vggt latent_align}"
ONLY="${ONLY:-all}"   # all | exp4 | exp7

QUICK=0
for arg in "$@"; do
  case "$arg" in
    --quick) QUICK=1 ;;
    --help|-h)
      sed -n '2,25p' "$0"
      exit 0
      ;;
  esac
done

if [[ "$QUICK" == "1" ]]; then
  N_WORST=3; N_BEST=3; N_RANDOM=4
  SAMPLE_STEPS=20
  NUM_DRAWS=4
  echo "[quick] n_worst=$N_WORST n_best=$N_BEST n_random=$N_RANDOM sample_steps=$SAMPLE_STEPS"
else
  N_WORST=5; N_BEST=5; N_RANDOM=10
  SAMPLE_STEPS=50
  NUM_DRAWS=6
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-runs/debug_pc_unite/${TS}}"
mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/run.log"

EXP4_CKPT="${EXP4_CKPT:-runs/exp4d_cam_cross_mv10_n100_f8_rgb10_anc01_delta005_stoch/ckpt_0020000.pt}"
EXP4_EVAL="${EXP4_EVAL:-runs/exp4d_cam_cross_mv10_n100_f8_rgb10_anc01_delta005_stoch/eval_n100_ckpt20k_noRenorm/results.json}"
EXP7_CKPT="${EXP7_CKPT:-runs/exp7_n100_randomreg_repnoise_tokvggt_v2/ckpt_0020000.pt}"
EXP7_EVAL="${EXP7_EVAL:-runs/exp7_n100_randomreg_repnoise_tokvggt_v2/eval_ckpt20k/results.json}"

exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "ShapePCUnite debug suite"
echo "  out:    $OUT_ROOT"
echo "  py:     $PY"
echo "  stages: $STAGES"
echo "  only:   $ONLY"
echo "  device: $DEVICE"
echo "============================================================"

run_one() {
  local name="$1"
  local ckpt="$2"
  local eval_json="$3"
  local out_dir="$OUT_ROOT/$name"

  if [[ ! -f "$ckpt" ]]; then
    echo "ERROR: missing ckpt $ckpt — skip $name"
    return 1
  fi
  mkdir -p "$out_dir"
  echo ""
  echo "---------- $name ----------"
  echo "ckpt: $ckpt"
  echo "eval: $eval_json"
  echo "out:  $out_dir"

  # shellcheck disable=SC2086
  "$PY" debug_pc_unite.py \
    --ckpt "$ckpt" \
    --eval_results "$eval_json" \
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
    --t_grid 0.0 0.1 0.25 0.5 0.75 0.9 0.99 \
    --stages $STAGES \
    --output_dir "$out_dir"

  echo "DONE $name → $out_dir/summary.json"
}

FAILED=0
if [[ "$ONLY" == "all" || "$ONLY" == "exp4" ]]; then
  run_one "exp4_ckpt20k" "$EXP4_CKPT" "$EXP4_EVAL" || FAILED=1
fi
if [[ "$ONLY" == "all" || "$ONLY" == "exp7" ]]; then
  run_one "exp7_ckpt20k" "$EXP7_CKPT" "$EXP7_EVAL" || FAILED=1
fi

# ---- Write a short comparison markdown from the JSON summaries ----
"$PY" - <<'PY' "$OUT_ROOT"
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
lines = ["# Debug comparison\n", f"Root: `{root}`\n"]

def load(name):
    p = root / name / "summary.json"
    if not p.exists():
        return None
    return json.load(open(p))

def stage(s, key):
    return (s or {}).get("stages", {}).get(key) or {}

exp4 = load("exp4_ckpt20k")
exp7 = load("exp7_ckpt20k")

lines.append("## Encode ± VGGT\n")
lines.append("| run | tok_vggt flag | cd_with | cd_without | rel(with,wo) | cd_mean_z |\n|---|---|---:|---:|---:|---:|\n")
for name, s in [("exp4", exp4), ("exp7", exp7)]:
    e = stage(s, "encode_vggt")
    if not e:
        lines.append(f"| {name} | — | — | — | — | — |\n")
        continue
    lines.append(
        f"| {name} | {e.get('tokenizer_use_weak_context')} | "
        f"{e.get('cd_with_vggt', float('nan')):.5f} | "
        f"{e.get('cd_without_vggt', float('nan')):.5f} | "
        f"{e.get('rel_with_vs_without', float('nan')):.3f} | "
        f"{e.get('cd_of_mean_latent', float('nan')):.5f} |\n"
    )

lines.append("\n## Latent alignment (gen vs tok)\n")
lines.append("| run | recon | gen | rel_err | excess_iso | mean_z_cd |\n|---|---:|---:|---:|---:|---:|\n")
for name, s in [("exp4", exp4), ("exp7", exp7)]:
    a = stage(s, "latent_align")
    if not a:
        lines.append(f"| {name} | — | — | — | — | — |\n")
        continue
    lines.append(
        f"| {name} | {a.get('recon_cd', float('nan')):.5f} | "
        f"{a.get('gen_cd', float('nan')):.5f} | "
        f"{a.get('rel_latent_err', float('nan')):.3f} | "
        f"{a.get('excess_vs_isotropic', float('nan')):.2f}x | "
        f"{a.get('cd_of_mean_latent', float('nan')):.5f} |\n"
    )

lines.append("\n## Diffusion oracle curve (mean CD vs t_start)\n")
lines.append("| t | exp4 oracle CD | exp7 oracle CD | exp4 vel_mse | exp7 vel_mse |\n|---:|---:|---:|---:|---:|\n")
c4 = stage(exp4, "diffusion_curve")
c7 = stage(exp7, "diffusion_curve")
oc4 = {r["t_start"]: r for r in (c4.get("oracle_curve") or [])}
oc7 = {r["t_start"]: r for r in (c7.get("oracle_curve") or [])}
vc4 = {r["t"]: r for r in (c4.get("velocity_curve") or [])}
vc7 = {r["t"]: r for r in (c7.get("velocity_curve") or [])}
ts = sorted(set(oc4) | set(oc7))
for t in ts:
    lines.append(
        f"| {t:.2f} | "
        f"{oc4.get(t, {}).get('cd', float('nan')):.5f} | "
        f"{oc7.get(t, {}).get('cd', float('nan')):.5f} | "
        f"{vc4.get(t, {}).get('velocity_mse', float('nan')):.5f} | "
        f"{vc7.get(t, {}).get('velocity_mse', float('nan')):.5f} |\n"
    )

lines.append("\n## Where to look\n")
lines.append("- Per-run JSON: `*/summary.json`, `*/diffusion_curve/`, `*/encode_vggt/`, `*/latent_align/`\n")
lines.append("- Camera-frame PLYs: `*/data_check/` (gt, gt_normals_rgb, patch_centers, fps, anchors, recon, gen)\n")
lines.append("- CSV curves: `*/diffusion_curve/oracle_curve.csv`, `velocity_curve.csv`\n")
lines.append("- Full log: `run.log`\n")

(root / "COMPARISON.md").write_text("".join(lines))
print("Wrote", root / "COMPARISON.md")
PY

echo ""
echo "============================================================"
echo "All debug runs finished (failed=$FAILED)"
echo "  Log:        $LOG"
echo "  Comparison: $OUT_ROOT/COMPARISON.md"
echo "  PLYs:       $OUT_ROOT/*/data_check/"
echo "============================================================"
exit "$FAILED"
