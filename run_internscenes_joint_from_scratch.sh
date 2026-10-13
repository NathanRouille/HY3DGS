#!/usr/bin/env bash
# Two matched random-view, full joint cold starts; encoder initialization only.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
EXPERIMENT="${EXPERIMENT:-coverage4}"
case "$EXPERIMENT" in
  coverage4) COVERAGE_WEIGHT=4 ;;
  coverage2) COVERAGE_WEIGHT=2 ;;
  *) echo "Unknown EXPERIMENT=$EXPERIMENT (coverage4/coverage2)" >&2; exit 2 ;;
esac
PACK="${PACK:-/projects_vol/gp_chuanxia.zheng/nathan/datasets/internscenes_bathroom_130}"
VGGT_CACHE="${VGGT_CACHE:-$SCRIPT_DIR/runs/vggt_cache/internscenes_bath10_joint_selected_v2}"
VIEW_MAP="$SCRIPT_DIR/configs/internscenes_bath10_selected_views.json"
PRETRAINED_FILE="${PRETRAINED_FILE:-$SCRIPT_DIR/runs/model_cache/hunyuan_encoder_cross_attn.pt}"
OUT="${OUT:-$SCRIPT_DIR/runs/joint_from_scratch/${EXPERIMENT}_${PBS_JOBID:-manual}}"
NUM_STEPS="${NUM_STEPS:-40000}"
[[ "$NUM_STEPS" =~ ^[1-9][0-9]*$ && "$NUM_STEPS" -gt 1000 ]] || {
  echo "NUM_STEPS must exceed the 1000-step warmup" >&2; exit 2;
}
[[ ! -e "$OUT/args.json" && ! -e "$OUT/ckpt_final.pt" ]] || {
  echo "Experiment already started in $OUT; choose a fresh OUT" >&2; exit 2;
}
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONDONTWRITEBYTECODE=1
TRAIN=(python train_pc_unite.py
  --dataset internscenes --data_dir "$PACK" --internscenes_split train
  --no_experiment_manifest --room_view_pools_json "$VIEW_MAP" --views_per_sample 2
  --view_sample_mode random --align_mode c_meanrms --cache_pooled_surfaces
  --vggt_cache_root "$VGGT_CACHE" --vggt_joint_cache
  --device cuda --seed 0 --batch_size 2 --num_workers 0 --num_steps "$NUM_STEPS"
  --lr 1e-4 --lr_schedule cosine --lr_warmup_steps 1000 --lr_min_ratio 0.1 --weight_decay 0.01
  --num_latents 2048 --num_registers 2048 --embed_dim 64 --width 1024 --heads 16
  --num_ge_layers 8 --num_decoder_layers 4 --num_points_per_anchor 16
  --pc_size 16384 --pc_sharpedge_size 16384 --query_sample_mode weighted_fps
  --fps_density_k 8 --fps_sharp_beta 0.2 --fps_density_clip_low 5 --fps_density_clip_high 95
  --fps_query_cache_items 128 --local_target_cache_items 128
  --pretrained_load cross_attn --pretrained_repo "$PRETRAINED_FILE" --include_sharp_label --point_feats 7
  --register_noise_mode fixed --modulation_recon_timestep_max 0 --no-representation_noising
  --no-tokenizer_use_weak_context --no-adaln_camera_cond --no-sample_renorm_output
  --lambda_recon 1 --lambda_flow 1 --flow_steps_per_recon 4 --flow_loss_type velocity
  --weak_context_dropout 0.1 --offpath_mode none
  --cd_pred2gt 1 --cd_gt2pred "$COVERAGE_WEIGHT" --lambda_rgb 10
  --lambda_anc 0.01 --lambda_anc_cd 10 --lambda_delta 0.001
  --chamfer_backend chunked --chamfer_chunk_size 1024
  --local_loss ot --lambda_local 0.1 --local_patch_count 64 --local_radius_scale 0.5
  --precision fp32 --max_grad_norm 1 --log_interval 100
  --log_grad_norms --grad_log_interval 1000 --ckpt_interval 2000
  --no-save_optimizer --save_latest_optimizer --vis_interval 0
  --joint_eval_interval 2000 --joint_eval_sample_steps 50 --output_dir "$OUT")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${TRAIN[@]}"; printf '\n'; exit 0
fi
[[ -f "$PRETRAINED_FILE" ]] || {
  echo "Missing persistent encoder weights. Run python prepare_joint_pretrained.py before qsub." >&2; exit 2;
}
mkdir -p "$OUT"
python - "$PACK" "$VIEW_MAP" "$VGGT_CACHE" "$PRETRAINED_FILE" <<'PY'
import sys
from hy3dgen.shapegen.pc_render_dataset import build_internscenes_render_dataset
from hy3dgen.shapegen.room_view_pools import load_room_view_pools
from prepare_joint_pretrained import prepare
pools = load_room_view_pools(sys.argv[2])
if len(pools) != 10 or any(len(pool) != 4 for pool in pools.values()):
    raise SystemExit("Expected ten rooms with four selected views each")
dataset = build_internscenes_render_dataset(sys.argv[1], room_view_pools=pools,
    views_per_sample=2, view_sample_mode="random", seed=0, align_mode="c_meanrms",
    vggt_cache_root=sys.argv[3], use_joint_vggt_cache=True, strict_load=True)
if len(dataset) != 10:
    raise SystemExit("Expected all ten rooms")
prepare(sys.argv[4])
print("Validated all ten rooms, 120 ordered joint cache pairs, and persistent pretrained weights", flush=True)
PY
DRY_RUN=1 bash "$SCRIPT_DIR/run_internscenes_joint_from_scratch.sh" > "$OUT/commands.txt"
"${TRAIN[@]}"
touch "$OUT/COMPLETE"
echo "Completed full joint $EXPERIMENT: $OUT"
