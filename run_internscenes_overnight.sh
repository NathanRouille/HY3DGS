#!/usr/bin/env bash
# Matched ten-room reconstruction -> frozen-tokenizer flow pipelines.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
EXPERIMENT="${EXPERIMENT:-control}"
LOCAL_LOSS=ot
LAMBDA_ANC=0.1
DENSITY_K=16
case "$EXPERIMENT" in
  control) LOCAL_LOSS=none ;;
  local_ot) ;;
  weak_anchor) LAMBDA_ANC=0.01 ;;
  density_k8) LAMBDA_ANC=0.01; DENSITY_K=8 ;;
  *) echo "Unknown EXPERIMENT=$EXPERIMENT (control/local_ot/weak_anchor/density_k8)" >&2; exit 2 ;;
esac
PACK="${PACK:-/projects_vol/gp_chuanxia.zheng/nathan/datasets/internscenes_bathroom_130}"
SOURCE_CKPT="${SOURCE_CKPT:-$SCRIPT_DIR/runs/internscenes_overfit_bath10_selected_v2_delta0p001/ckpt_final.pt}"
VGGT_CACHE="${VGGT_CACHE:-$SCRIPT_DIR/runs/vggt_cache/internscenes_bath10_joint_selected_v2}"
VIEW_MAP="$SCRIPT_DIR/configs/internscenes_bath10_selected_views.json"
OUT="${OUT:-$SCRIPT_DIR/runs/overnight/${EXPERIMENT}_${PBS_JOBID:-manual}}"
RECON_STEPS="${RECON_STEPS:-8000}"
FLOW_STEPS="${FLOW_STEPS:-12000}"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONDONTWRITEBYTECODE=1
[[ "$RECON_STEPS" =~ ^[1-9][0-9]*$ && "$FLOW_STEPS" =~ ^[1-9][0-9]*$ ]] || {
  echo "Step budgets must be positive integers" >&2; exit 2;
}
[[ -f "$SOURCE_CKPT" ]] || { echo "Missing checkpoint: $SOURCE_CKPT" >&2; exit 2; }
[[ ! -e "$OUT/recon/args.json" && ! -e "$OUT/flow/args.json" &&
   ! -e "$OUT/recon/ckpt_final.pt" && ! -e "$OUT/flow/ckpt_final.pt" ]] || {
  echo "Experiment already started in $OUT; choose a fresh OUT" >&2; exit 2;
}
COMMON=(--dataset internscenes --data_dir "$PACK" --internscenes_split train
  --no_experiment_manifest --room_view_pools_json "$VIEW_MAP" --views_per_sample 2
  --view_sample_mode random --align_mode c_meanrms --cache_pooled_surfaces
  --device cuda --seed 0 --batch_size 2 --num_workers 0
  --reset_optimizer_on_resume --lr 1e-5 --lr_schedule constant --weight_decay 0.01
  --num_latents 2048 --num_registers 2048 --embed_dim 64 --width 1024 --heads 16
  --num_ge_layers 8 --num_decoder_layers 4 --num_points_per_anchor 16
  --pc_size 16384 --pc_sharpedge_size 16384 --query_sample_mode weighted_fps
  --fps_density_k "$DENSITY_K" --fps_sharp_beta 0.2
  --fps_density_clip_low 5 --fps_density_clip_high 95 --fps_query_cache_items 128
  --local_target_cache_items 128 --pretrained_load none --include_sharp_label --point_feats 7
  --register_noise_mode random --no-representation_noising --no-tokenizer_use_weak_context
  --no-adaln_camera_cond --no-sample_renorm_output
  --cd_pred2gt 1 --cd_gt2pred 2 --lambda_rgb 10 --lambda_anc "$LAMBDA_ANC"
  --lambda_anc_cd 10 --lambda_delta 0.001 --lambda_recon 1
  --chamfer_backend chunked --chamfer_chunk_size 1024
  --local_loss "$LOCAL_LOSS" --lambda_local 0.1 --local_patch_count 64 --local_radius_scale 0.5
  --precision fp32 --log_interval 100 --ckpt_interval 2000 --no-save_optimizer --vis_interval 0)
RECON=(python train_pc_unite.py "${COMMON[@]}" --recon_only
  --resume_ckpt "$SOURCE_CKPT" --additional_steps "$RECON_STEPS"
  --log_grad_norms --grad_log_interval 500 --output_dir "$OUT/recon")
FLOW=(python train_pc_unite.py "${COMMON[@]}" --freeze_recon
  --resume_ckpt "$OUT/recon/ckpt_final.pt" --additional_steps "$FLOW_STEPS"
  --vggt_cache_root "$VGGT_CACHE" --vggt_joint_cache
  --flow_steps_per_recon 8 --flow_loss_type velocity --offpath_mode none
  --lambda_flow 1 --weak_context_dropout 0.1 --recon_log_interval 2000
  --output_dir "$OUT/flow")
EVAL_COMMON=(--data_dir "$PACK" --room_view_pools_json "$VIEW_MAP" --vggt_cache_root "$VGGT_CACHE"
  --sample_steps 50 --gen_seeds 0 1 2)
EVAL_RECON=(python evaluate_pc_overnight.py "${EVAL_COMMON[@]}" --recon_only
  --ckpt "$OUT/recon/ckpt_final.pt" --output_dir "$OUT/recon_eval")
EVAL_FLOW=(python evaluate_pc_overnight.py "${EVAL_COMMON[@]}"
  --ckpt "$OUT/flow/ckpt_final.pt" --reference_metrics "$OUT/recon_eval/metrics.json"
  --output_dir "$OUT/eval")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  for stage in RECON EVAL_RECON FLOW EVAL_FLOW; do
    declare -n command="$stage"
    printf '%q ' "${command[@]}"; printf '\n'
  done
  exit 0
fi
mkdir -p "$OUT"
# Validate all ten rooms and all 120 joint pairs BEFORE the long AE stage.
python - "$PACK" "$VIEW_MAP" "$VGGT_CACHE" <<'PY'
import sys
from hy3dgen.shapegen.pc_render_dataset import build_internscenes_render_dataset
from hy3dgen.shapegen.room_view_pools import load_room_view_pools
pools = load_room_view_pools(sys.argv[2])
d = build_internscenes_render_dataset(sys.argv[1], room_view_pools=pools,
    views_per_sample=2, view_sample_mode="random", seed=0, align_mode="c_meanrms",
    vggt_cache_root=sys.argv[3], use_joint_vggt_cache=True, strict_load=True)
if len(d) != 10 or any(len(pool) != 4 for pool in pools.values()):
    raise SystemExit("Expected exactly ten rooms with four selected views each")
print("Validated ten rooms and all ordered joint VGGT cache pairs", flush=True)
PY
DRY_RUN=1 bash "$SCRIPT_DIR/run_internscenes_overnight.sh" > "$OUT/commands.txt"
"${RECON[@]}"
"${EVAL_RECON[@]}"
"${FLOW[@]}"
"${EVAL_FLOW[@]}"
touch "$OUT/COMPLETE"
echo "Completed $EXPERIMENT: $OUT"
