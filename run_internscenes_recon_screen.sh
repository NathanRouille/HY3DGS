#!/usr/bin/env bash
# Three matched, short AE fine-tunes from the SAME selected-v2 checkpoint.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
EXPERIMENT="${EXPERIMENT:-global}"
case "$EXPERIMENT" in
  global) LOCAL_LOSS=none ;;
  local_cd) LOCAL_LOSS=cd ;;
  local_ot) LOCAL_LOSS=ot ;;
  *) echo "Unknown EXPERIMENT=$EXPERIMENT (global/local_cd/local_ot)" >&2; exit 2 ;;
esac
PACK="${PACK:-/projects_vol/gp_chuanxia.zheng/nathan/datasets/internscenes_bathroom_130}"
SOURCE_CKPT="${SOURCE_CKPT:-$SCRIPT_DIR/runs/internscenes_overfit_bath10_selected_v2_delta0p001/ckpt_final.pt}"
VIEW_MAP="$SCRIPT_DIR/configs/internscenes_recon_screen_train.json"
EXTRA_VIEW_MAP="$SCRIPT_DIR/configs/internscenes_recon_screen_extra_views.json"
OUT="${OUT:-$SCRIPT_DIR/runs/recon_screen/${EXPERIMENT}_${PBS_JOBID:-manual}}"
STEPS="${STEPS:-2000}"
PRECISION="${PRECISION:-bf16}"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONDONTWRITEBYTECODE=1
[[ -f "$SOURCE_CKPT" ]] || { echo "Missing source checkpoint: $SOURCE_CKPT" >&2; exit 2; }
# Never silently restart/overwrite another experiment.
[[ ! -e "$OUT/args.json" && ! -e "$OUT/ckpt_final.pt" ]] || {
  echo "Output already contains an experiment: $OUT. Choose a new OUT." >&2; exit 2;
}
TRAIN=(python train_pc_unite.py
  --dataset internscenes --data_dir "$PACK" --internscenes_split train --max_items 3
  --no_experiment_manifest --room_view_pools_json "$VIEW_MAP" --views_per_sample 2
  --view_sample_mode first --align_mode c_meanrms --cache_fixed_surfaces
  --recon_only --device cuda --seed 0 --batch_size 2 --num_workers 0
  --resume_ckpt "$SOURCE_CKPT" --reset_optimizer_on_resume --additional_steps "$STEPS"
  --lr 1e-5 --lr_schedule constant --weight_decay 0.01
  --num_latents 2048 --num_registers 2048 --num_points_per_anchor 16
  --pc_size 16384 --pc_sharpedge_size 16384 --query_sample_mode weighted_fps
  --fps_density_k 16 --fps_sharp_beta 0.2 --fps_density_clip_low 5 --fps_density_clip_high 95
  --fps_query_cache_items 3 --pretrained_load none --include_sharp_label --point_feats 7
  --register_noise_mode random --no-representation_noising
  --no-tokenizer_use_weak_context --no-adaln_camera_cond --no-sample_renorm_output
  --cd_pred2gt 1 --cd_gt2pred 2 --lambda_rgb 10 --lambda_anc 0.1
  --lambda_anc_cd 10 --lambda_delta 0.001 --lambda_recon 1
  --chamfer_backend chunked --chamfer_chunk_size 1024
  --local_loss "$LOCAL_LOSS" --lambda_local 0.1 --local_patch_count 64 --local_radius_scale 0.5
  --precision "$PRECISION" --precision_preflight
  --log_interval 100 --log_grad_norms --grad_log_interval 500
  --ckpt_interval 1000 --no-save_optimizer --vis_interval 0 --output_dir "$OUT")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${TRAIN[@]}"
  printf '\n'
  exit 0
fi
mkdir -p "$OUT"
python -m hy3dgen.shapegen.room_view_pools "$VIEW_MAP" "$PACK"
python -m hy3dgen.shapegen.room_view_pools "$EXTRA_VIEW_MAP" "$PACK"
python -m unittest discover -s tests -p 'test_pc_recon_screen.py' -q
python evaluate_pc_recon_screen.py --ckpt "$SOURCE_CKPT" --data_dir "$PACK" \
  --room_view_pools_json "$VIEW_MAP" --output_dir "$OUT/baseline_eval"
"${TRAIN[@]}"
python evaluate_pc_recon_screen.py --ckpt "$OUT/ckpt_final.pt" --data_dir "$PACK" \
  --room_view_pools_json "$VIEW_MAP" --output_dir "$OUT/eval"
python evaluate_pc_recon_screen.py --ckpt "$OUT/ckpt_final.pt" --data_dir "$PACK" \
  --room_view_pools_json "$EXTRA_VIEW_MAP" --output_dir "$OUT/eval_extra_views"
echo "Finished $EXPERIMENT: $OUT"
