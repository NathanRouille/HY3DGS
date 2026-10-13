#!/usr/bin/env bash
# Reconstruction first: four independent controls from the original joint AE.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
EXPERIMENT="${EXPERIMENT:-fixed_global}"
LOCAL_LOSS=none
VIEW_MODE=first
GT_WEIGHT=2
CACHE_FLAG=--cache_fixed_surfaces
case "$EXPERIMENT" in
  fixed_global) ;;
  fixed_ot) LOCAL_LOSS=ot ;;
  random_global) VIEW_MODE=random; CACHE_FLAG=--cache_pooled_surfaces ;;
  fixed_coverage) GT_WEIGHT=4 ;;
  *) echo "Unknown EXPERIMENT=$EXPERIMENT (fixed_global/fixed_ot/random_global/fixed_coverage)" >&2; exit 2 ;;
esac
PACK="${PACK:-/projects_vol/gp_chuanxia.zheng/nathan/datasets/internscenes_bathroom_130}"
SOURCE_CKPT="${SOURCE_CKPT:-$SCRIPT_DIR/runs/internscenes_overfit_bath10_selected_v2_delta0p001/ckpt_final.pt}"
VIEW_MAP="$SCRIPT_DIR/configs/internscenes_bath10_selected_views.json"
OUT="${OUT:-$SCRIPT_DIR/runs/recon_sanity/${EXPERIMENT}_${PBS_JOBID:-manual}}"
STEPS="${STEPS:-8000}"
[[ "$STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "STEPS must be a positive integer" >&2; exit 2; }
[[ -f "$SOURCE_CKPT" ]] || { echo "Missing source checkpoint: $SOURCE_CKPT" >&2; exit 2; }
[[ ! -e "$OUT/train/args.json" && ! -e "$OUT/train/ckpt_final.pt" && ! -e "$OUT/baseline_native/metrics.json" ]] || {
  echo "Experiment already exists: $OUT; select a fresh OUT" >&2; exit 2;
}
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
TRAIN=(python train_pc_unite.py
  --dataset internscenes --data_dir "$PACK" --internscenes_split train
  --no_experiment_manifest --room_view_pools_json "$VIEW_MAP" --views_per_sample 2
  --view_sample_mode "$VIEW_MODE" --align_mode c_meanrms "$CACHE_FLAG"
  --recon_only --device cuda --seed 0 --batch_size 2 --num_workers 0
  --resume_ckpt "$SOURCE_CKPT" --reset_optimizer_on_resume --additional_steps "$STEPS"
  --lr 1e-5 --lr_schedule constant --weight_decay 0.01
  --num_latents 2048 --num_registers 2048 --embed_dim 64 --width 1024 --heads 16
  --num_ge_layers 8 --num_decoder_layers 4 --num_points_per_anchor 16
  --pc_size 16384 --pc_sharpedge_size 16384 --query_sample_mode weighted_fps
  --fps_density_k 16 --fps_sharp_beta 0.2 --fps_density_clip_low 5 --fps_density_clip_high 95
  --fps_query_cache_items 128 --local_target_cache_items 128 --pretrained_load none
  --include_sharp_label --point_feats 7 --register_noise_mode fixed
  --modulation_recon_timestep_max 0 --no-representation_noising
  --no-tokenizer_use_weak_context --no-adaln_camera_cond --no-sample_renorm_output
  --cd_pred2gt 1 --cd_gt2pred "$GT_WEIGHT" --lambda_rgb 10 --lambda_anc 0.01
  --lambda_anc_cd 10 --lambda_delta 0.001 --lambda_recon 1
  --chamfer_backend chunked --chamfer_chunk_size 1024
  --local_loss "$LOCAL_LOSS" --lambda_local 0.1 --local_patch_count 64 --local_radius_scale 0.5
  --precision fp32 --log_interval 100 --log_grad_norms --grad_log_interval 500
  --ckpt_interval 1000 --no-save_optimizer --vis_interval 0 --output_dir "$OUT/train")
EVAL=(python evaluate_pc_recon_sanity.py --data_dir "$PACK" --room_view_pools_json "$VIEW_MAP")
NATIVE=("${EVAL[@]}" --ckpt "$SOURCE_CKPT" --pair_mode primary --output_dir "$OUT/baseline_native")
INITIAL=("${EVAL[@]}" --ckpt "$SOURCE_CKPT" --pair_mode primary
  --register_noise_mode fixed --modulation_recon_timestep_max 0 --output_dir "$OUT/baseline_fixed")
FINAL=("${EVAL[@]}" --ckpt "$OUT/train/ckpt_final.pt" --pair_mode paired --output_dir "$OUT/eval")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  for stage in NATIVE INITIAL TRAIN FINAL; do
    declare -n command="$stage"
    printf '%q ' "${command[@]}"; printf '\n'
  done
  exit 0
fi
mkdir -p "$OUT"
# Validate data and the teacher checkpoint before allocating GPU training state.
python - "$PACK" "$VIEW_MAP" "$SOURCE_CKPT" <<'PY'
import sys, torch
from hy3dgen.shapegen.pc_render_dataset import build_internscenes_render_dataset
from hy3dgen.shapegen.room_view_pools import load_room_view_pools
pools = load_room_view_pools(sys.argv[2])
data = build_internscenes_render_dataset(sys.argv[1], room_view_pools=pools,
    views_per_sample=2, view_sample_mode="random", seed=0, align_mode="c_meanrms", strict_load=True)
if len(data) != 10 or len(pools) != 10 or any(len(v) != 4 for v in pools.values()):
    raise SystemExit("Expected exactly ten rooms with four selected views each")
checkpoint = torch.load(sys.argv[3], map_location="cpu", weights_only=False, mmap=True)
if any(k.startswith("tokenizer_ge.") for k in checkpoint["model"]):
    raise SystemExit("Use the original joint/reconstruction checkpoint, not a frozen-flow checkpoint")
expected = {"num_latents":2048,"num_registers":2048,"embed_dim":64,"width":1024,
    "num_ge_layers":8,"num_decoder_layers":4,"num_points_per_anchor":16,
    "pc_size":16384,"pc_sharpedge_size":16384,"point_feats":7,
    "geometry_only":False,"tokenizer_use_weak_context":False,"adaln_camera_cond":False}
saved=checkpoint.get("args") or {}
for key,value in expected.items():
    if key in saved and saved[key] != value:
        raise SystemExit(f"Unexpected checkpoint {key}: {saved[key]} (expected {value})")
print(f"Validated ten rooms; source checkpoint step={checkpoint.get('step')}", flush=True)
PY
DRY_RUN=1 bash "$SCRIPT_DIR/run_internscenes_recon_sanity.sh" > "$OUT/commands.txt"
"${NATIVE[@]}"
"${INITIAL[@]}"
"${TRAIN[@]}"
# Export intermediate learned states after training, without interrupting Adam
# or introducing optimizer resets between 1k segments.
shopt -s nullglob
for checkpoint in "$OUT"/train/ckpt_[0-9]*.pt; do
  stage="$(basename "$checkpoint" .pt)"
  "${EVAL[@]}" --ckpt "$checkpoint" --pair_mode primary --output_dir "$OUT/eval_$stage"
done
"${FINAL[@]}"
python - "$OUT" <<'PY'
import csv, json, sys
from pathlib import Path
root=Path(sys.argv[1])
folders=[root/'baseline_native', root/'baseline_fixed', *sorted(root.glob('eval_ckpt_*')), root/'eval']
rows=[]
for folder in folders:
    metrics=json.loads((folder/'metrics.json').read_text())
    for pair, summary in metrics['summary'].items():
        rows.append({'evaluation':folder.name, 'pair':pair, 'checkpoint':metrics['checkpoint'],
            'register_noise_mode':metrics['register_noise_mode'],
            'modulation_time_max':metrics['modulation_recon_timestep_max'], **summary})
keys=list(dict.fromkeys(k for row in rows for k in row))
with (root/'quality_summary.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader(); writer.writerows(rows)
PY
touch "$OUT/COMPLETE"
echo "Finished $EXPERIMENT: $OUT"
