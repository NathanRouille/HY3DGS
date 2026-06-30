#!/bin/bash
# Phase 0: Hunyuan3D-2mini ShapeVAE mesh reconstruction sanity check on ShapeNet val.
#
# Defaults to posterior sampling (matches minimal_vae_demo.py).
# Set SAMPLE_POSTERIOR=false to compare deterministic posterior mode (mean).

set -euo pipefail

: "${EXPERIMENTS:=/export/home/nathan/datasets/shapenet_experiments}"
: "${DATASET:=chair_800train_40val_v46}"
: "${CATEGORIES:=chair}"
: "${NUM_SAMPLES:=15}"

DATA_DIR="${EXPERIMENTS}/${DATASET}/val"
MODE_TAG="mode"
if [[ "${SAMPLE_POSTERIOR:-true}" == "true" ]]; then
  MODE_TAG="sample"
fi
OUTPUT_DIR="runs/sanity/hy_mini_vae_val_${MODE_TAG}"

python evaluate_shapevae.py \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_samples "${NUM_SAMPLES}" \
  --categories "${CATEGORIES}" \
  $([[ "${SAMPLE_POSTERIOR:-true}" == "true" ]] && echo --sample_posterior || echo --no-sample_posterior)
