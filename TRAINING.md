# ShapeGSAE Training Guide

## Quick-reference: past failure root causes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `valid_views_fraction < 1.0` | Azimuth=180° produces an empty back-face view | Use `--camera_azimuths "0,90,180,270"` (all cardinal) or `"30,120,210,300"` (offset) |
| `time_data_s > 5s` / corrupt GT | Multiple DataLoader workers race for EGL context | Run `--precache_only` first (CPU-only), then train with `--num_workers 0` |
| `alpha_mean → 0` within 100 steps | Opacity collapse. Often caused by above two, or absent alpha supervision | Fix camera/GT; add `--lambda_alpha 0.05` |
| Steps hang at 7, 22, 62 … | DataLoader worker deadlock (trimesh + OMP threads) | `--num_workers 0` |
| GT renders are grey | Mesh has no vertex colors **and** no UV texture (Objaverse fallback) | Use ShapeNet (UV-textured) or filter Objaverse to textured models |

---

## Dataset setup: ShapeNet Core v2

ShapeNet Core v2 (~25 GB for all 9 recommended categories) gives reliable RGB
textures for all objects. Download requires free registration at https://shapenet.org/.

### Step 1 — Prepare index files and symlinks

```bash
cd /home/nathan/Documents/research/HY3DGS

python prepare_shapenet.py \
    --shapenet_dir /path/to/ShapeNetCore.v2 \
    --output_dir   data/shapenet \
    --categories   03001627,04379243,02958343 \
    --val_fraction 0.1
```

This creates `data/shapenet/train/` and `data/shapenet/val/` symlinks plus
`train_list.json` and `val_list.json`.

For a quick start with chairs only (3.5 GB, 6778 models):
```bash
python prepare_shapenet.py \
    --shapenet_dir /path/to/ShapeNetCore.v2 \
    --output_dir   data/shapenet_chairs \
    --categories   03001627
```

### Step 2 — Pre-cache GT RGBD (run once, CPU-only)

```bash
CUDA_VISIBLE_DEVICES="" PYOPENGL_PLATFORM=egl \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python train_gs_ae.py \
    --data_dir data/shapenet/train \
    --num_views 4 --camera_azimuths "0,90,180,270" \
    --render_height 256 --render_width 256 \
    --precache_only \
    2>&1 | tee runs/precache_train.log

# Val set
CUDA_VISIBLE_DEVICES="" PYOPENGL_PLATFORM=egl \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python train_gs_ae.py \
    --data_dir data/shapenet/val \
    --num_views 4 --camera_azimuths "0,90,180,270" \
    --render_height 256 --render_width 256 \
    --precache_only \
    2>&1 | tee runs/precache_val.log
```

Cache files are named `<obj_path>.gt_rgbd_h256w256az0_90_180_270_normv1.pt`.

---

## Experimental roadmap

### Phase 1 — Smoke overfit and baseline validation

**Goal:** confirm the pipeline works end-to-end with the new dataset.

#### 1a. Smoke overfit (5 objects, 1 view, ~5 min)

```bash
CUDA_VISIBLE_DEVICES=0 PYOPENGL_PLATFORM=egl \
python train_gs_ae.py \
    --data_dir data/shapenet_chairs/train \
    --output_dir runs/p1_smoke \
    --max_items 5 --max_steps 1000 \
    --smoke_overfit \
    --batch_size 2 --num_workers 0 \
    --use_wandb --wandb_run_name p1_smoke
```

**Pass criteria:** `alpha_sup < 0.05` by step 200, `l1_fg < 0.10` by step 500.

#### 1b. Baseline — single view, 256×256, chairs (20k steps)

This is your primary reference point. All future experiments compare to this.

```bash
CUDA_VISIBLE_DEVICES=0 PYOPENGL_PLATFORM=egl \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python train_gs_ae.py \
    --data_dir  data/shapenet_chairs/train \
    --val_dir   data/shapenet_chairs/val \
    --output_dir runs/p1_baseline_1v \
    --num_views 1 \
    --render_height 256 --render_width 256 \
    --lr 1e-4 --warmup_steps 500 \
    --batch_size 4 --num_workers 0 \
    --lambda_ssim 0.2 \
    --lambda_d 1.0 \
    --lambda_alpha 0.05 \
    --lambda_scale 0.01 \
    --lambda_opa   0.01 \
    --fg_weight 0.75 \
    --max_steps 20000 \
    --val_every 2000 --num_val_samples 30 \
    --save_every 5000 \
    --use_wandb --wandb_run_name p1_baseline_1v
```

**Target after 20k steps:** PSNR > 22 dB, SSIM > 0.80, alpha_fg > 0.7.

#### Evaluate baseline

```bash
python evaluate_gs_ae.py \
    --checkpoints runs/p1_baseline_1v/ckpt_020000.pt \
    --data_dir    data/shapenet_chairs/val \
    --output_dir  eval/p1_baseline \
    --num_samples 50 \
    --num_views 1 --camera_azimuths "0"
```

---

### Phase 2 — Multi-view + depth tuning

**Goal:** improve 3D consistency by adding more views.

#### 2a. Two-view baseline (0° + 180°, ~30k steps)

```bash
CUDA_VISIBLE_DEVICES=0 PYOPENGL_PLATFORM=egl \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python train_gs_ae.py \
    --data_dir  data/shapenet_chairs/train \
    --val_dir   data/shapenet_chairs/val \
    --output_dir runs/p2_2v \
    --num_views 2 --camera_azimuths "0,180" \
    --render_height 256 --render_width 256 \
    --lr 1e-4 --warmup_steps 500 \
    --batch_size 4 --num_workers 0 \
    --lambda_ssim 0.2 \
    --lambda_d 1.0 \
    --lambda_alpha 0.05 \
    --lambda_scale 0.01 \
    --lambda_opa   0.01 \
    --fg_weight 0.75 \
    --max_steps 30000 \
    --val_every 2000 --num_val_samples 30 \
    --use_wandb --wandb_run_name p2_2v
```

#### 2b. Four-view baseline (0°+90°+180°+270°, ~50k steps)

Requires pre-cached GT at this azimuth set (already done in dataset setup).

```bash
CUDA_VISIBLE_DEVICES=0 PYOPENGL_PLATFORM=egl \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python train_gs_ae.py \
    --data_dir  data/shapenet_chairs/train \
    --val_dir   data/shapenet_chairs/val \
    --output_dir runs/p2_4v \
    --num_views 4 --camera_azimuths "0,90,180,270" \
    --render_height 256 --render_width 256 \
    --lr 1e-4 --warmup_steps 500 \
    --batch_size 4 --num_workers 0 \
    --lambda_ssim 0.2 \
    --lambda_d 1.5 \
    --lambda_alpha 0.05 \
    --lambda_scale 0.01 \
    --lambda_opa   0.01 \
    --fg_weight 0.75 \
    --max_steps 50000 \
    --val_every 2000 --num_val_samples 30 \
    --use_wandb --wandb_run_name p2_4v
```

Note: with 4 views `lambda_d 1.5` is slightly stronger — depth constraint is more
over-determined (4 cameras agree on 3D positions), so more weight helps convergence.

#### 2c. Compare 1v vs 2v vs 4v at step 20k/30k/50k

```bash
python evaluate_gs_ae.py \
    --checkpoints \
        runs/p1_baseline_1v/ckpt_020000.pt \
        runs/p2_2v/ckpt_020000.pt \
        runs/p2_4v/ckpt_020000.pt \
    --data_dir   data/shapenet_chairs/val \
    --output_dir eval/p2_view_comparison \
    --num_samples 50 \
    --num_views 4 --camera_azimuths "0,90,180,270"
```

**Decision rule:** proceed with whichever achieves highest PSNR at equal compute.

#### 2d. Loss weight ablation (optional, run in parallel)

Try these variants on top of p2_4v:

```bash
# Higher depth weight
python train_gs_ae.py ... --lambda_d 2.0 \
    --output_dir runs/p2_lambda_d2 --wandb_run_name p2_lambda_d2

# Stronger alpha (opacity-focused)
python train_gs_ae.py ... --lambda_alpha 0.1 \
    --output_dir runs/p2_alpha01 --wandb_run_name p2_alpha01

# Higher FG weight
python train_gs_ae.py ... --fg_weight 0.85 \
    --output_dir runs/p2_fg085 --wandb_run_name p2_fg085
```

---

### Phase 3 — Scale up: more data, larger model, architecture upgrades

**Goal:** push toward best reconstruction quality.

#### 3a. Expand to 3 categories (chair+table+car, ~18k models)

Re-use the best view configuration from Phase 2.

```bash
CUDA_VISIBLE_DEVICES=0 PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python train_gs_ae.py \
    --data_dir  data/shapenet/train \
    --val_dir   data/shapenet/val \
    --output_dir runs/p3_multi_cat \
    --num_views 4 --camera_azimuths "0,90,180,270" \
    --render_height 256 --render_width 256 \
    --lr 1e-4 --warmup_steps 1000 \
    --batch_size 4 --num_workers 0 \
    --lambda_ssim 0.2 --lambda_d 1.5 \
    --lambda_alpha 0.05 --fg_weight 0.75 \
    --lambda_scale 0.01 --lambda_opa 0.01 \
    --max_steps 100000 \
    --val_every 5000 --num_val_samples 50 \
    --save_every 10000 \
    --use_wandb --wandb_run_name p3_multi_cat
```

#### 3b. Larger model — more Gaussians and deeper decoder

When the baseline reaches PSNR saturation (< 0.5 dB improvement per 10k steps),
try a larger configuration:

```bash
CUDA_VISIBLE_DEVICES=0 ... python train_gs_ae.py \
    ...  \
    --num_latents 4096 \
    --embed_dim 128 \
    --width 1024 \
    --num_decoder_layers 12 \
    --num_encoder_layers 8 \
    --output_dir runs/p3_large_model \
    --wandb_run_name p3_large_model
```

Note: 4096 latents doubles memory. Use `--batch_size 2` if OOM.

#### 3c. QK normalization — add when to add it

**QK normalization** (`--qk_norm` in model, exposed via `ShapeGSAE(qk_norm=True)`)
stabilizes attention weight magnitudes in deep transformers. It is currently
*not exposed as a CLI flag* in `train_gs_ae.py`. To enable it, add
`qk_norm=True` in the `ShapeGSAE(...)` constructor call inside `train_gs_ae.py`.

**Recommendation: add in Phase 3 only** — specifically when:
- You have a decoder with ≥ 12 layers, OR
- You are training the larger model (4096 latents), AND
- Training is unstable (loss oscillates, grad_norm spikes > 10).

It has minimal benefit for the default 8-layer decoder and may slow convergence
if added prematurely. Do not add in Phase 1 or Phase 2.

To expose it as a CLI flag, add this line in `parse_args()` in `train_gs_ae.py`:
```python
p.add_argument('--qk_norm', action='store_true', default=False,
               help='QK normalization in attention. Add for ≥12-layer decoders or large models.')
```
And pass `qk_norm=args.qk_norm` in the `ShapeGSAE(...)` constructor.

#### 3d. DropPath (stochastic depth) — when to add

**DropPath** (`drop_path_rate`) is a regularization technique. Add it only if:
- Val PSNR plateaus while train PSNR keeps improving (gap > 2 dB), OR
- Dataset has fewer than 5k models and model is large.

Start with `drop_path_rate=0.05`, increase to `0.10` if overfitting persists.

To expose as CLI flag:
```python
p.add_argument('--drop_path_rate', type=float, default=0.0,
               help='Stochastic depth rate. 0.05-0.10 if train/val gap > 2 dB PSNR.')
```
And pass `drop_path_rate=args.drop_path_rate` in the `ShapeGSAE(...)` constructor.

**Summary decision table:**

| Condition | QK norm | DropPath |
|-----------|---------|----------|
| Phase 1–2, ≤8 decoder layers | ✗ | ✗ |
| Phase 3, 12+ decoder layers, stable | ✓ | ✗ |
| Phase 3, any, val/train gap > 2 dB | ✗ | ✓ (0.05) |
| Phase 3, large model, both issues | ✓ | ✓ (0.05) |

---

## Tunable parameters and expected impact

| Parameter | Default | Range | Expected impact |
|-----------|---------|-------|-----------------|
| `lr` | 1e-4 | 5e-5 – 3e-4 | Increasing → faster early convergence but risk collapse. Lower for large models. |
| `warmup_steps` | 500 | 0 – 2000 | Prevents gradient explosions at step 0. Increase with large LR. |
| `lambda_d` | 1.0 | 0.5 – 2.0 | **Most impactful for depth quality.** Increase first if depth predictions are poor. |
| `lambda_alpha` | 0.05 | 0.01 – 0.15 | Prevents opacity collapse. Increase if alpha_fg_mean stays < 0.4. |
| `lambda_ssim` | 0.2 | 0.0 – 0.5 | Improves texture sharpness. Disable during first 2k steps if causing instability. |
| `fg_weight` | 0.75 | 0.5 – 0.9 | Higher = network focuses more on object; lower = more balanced. |
| `lambda_scale` | 0.01 | 0.0 – 0.05 | Prevents Gaussian explosion. Increase if Gaussians go to extreme sizes. |
| `lambda_opa` | 0.01 | 0.0 – 0.05 | Regularizes opacity toward 0.5. Can disable once stable. |
| `target_log_scale` | -3.0 | -4.0 – -1.5 | -3 → scale≈0.05. Increase to -2 for coarser, larger Gaussians on big structures. |
| `num_latents` | 2048 | 512 – 8192 | More Gaussians = finer detail but slower training. 2048 is good for chairs/tables. |
| `num_views` | 4 | 1 – 8 | More views = better 3D consistency at cost of compute per step. |
| `num_decoder_layers` | 8 | 4 – 16 | Deeper decoder = more expressive but slower + risk of overfitting. |
| `embed_dim` | 64 | 32 – 256 | Larger bottleneck = more latent capacity. |

---

## Metric interpretation guide

### Critical health indicators

| Metric | Healthy | Collapsing | Action |
|--------|---------|------------|--------|
| `train/mean_view_pred_alpha_mean` | > 0.5 | → 0 | See collapse section |
| `train/valid_views_fraction` | = 1.0 | < 1.0 | Fix camera angles, pre-cache GT |
| `train/time_data_s` | < 1s | > 5s | Pre-cache GT before training |
| `val/psnr` | > 20 dB at 10k steps | < 15 dB | Lower LR, check GT quality |
| `val/ssim` | > 0.75 at 20k steps | < 0.60 | Increase lambda_ssim, more views |
| `val/alpha_fg_mean` | > 0.6 | < 0.3 | Increase lambda_alpha |

### Loss components

| Metric | Meaning | Expected trend |
|--------|---------|----------------|
| `train/l1_fg` | L1 error on foreground pixels | Decreases from ~0.3 to < 0.08 |
| `train/l1_bg` | L1 error on background pixels | Should stay near 0 (background is white) |
| `train/alpha_sup` | `(1-pred_α)²` on foreground pixels | Drops from ~0.5 to < 0.02 by step 1k |
| `train/depth` | L1 depth error on valid pixels | Decreases monotonically; most informative 3D signal |
| `train/ssim` | `1 - SSIM` (foreground only) | Decreases alongside l1_fg |
| `val/psnr` | Peak signal-to-noise ratio on fg | Increases; plateau signals capacity limit |
| `val/ssim` | Structural similarity on fg | Increases with PSNR; penalizes blur more |

### What "working" looks like in wandb at 2k steps

- `valid_views_fraction = 1.0` (constant)
- `mean_view_pred_alpha_mean`: rises from ~0.3 to > 0.6
- `alpha_sup`: drops from ~0.5 to < 0.05
- `l1_fg`: drops from ~0.3 to < 0.15
- `time_data_s`: flat < 0.5s (fast GT loading from cache)
- `lr`: rises linearly for first 500 steps then decays (warmup visible)

---

## Troubleshooting

### Collapse: `alpha_mean → 0` within first 100 steps

1. Check `valid_views_fraction`: if < 1.0, fix camera angles and re-cache GT.
2. Check `time_data_s`: if > 5s, GT is being rendered on-the-fly → pre-cache first.
3. If both are fine: LR is too high → halve it.
4. Add `--lambda_alpha 0.1` for a stronger alpha signal.

### Loss does not decrease (flat `total_loss`)

- Verify `alpha_mean > 0`; if not, see collapse section.
- If alpha is fine but `l1_fg` is flat: increase `--fg_weight 0.85` and `--lambda_d 1.5`.
- If depth is flat: check `min_view_valid_depth_ratio > 0.1`.

### Steps hang or are very slow

- **Hang at step N**: DataLoader worker deadlock → `--num_workers 0`.
- **Slow steps (>30s)**: GT rendering on-the-fly → run `--precache_only` first.

### CUDA out of memory

- Reduce `--batch_size 2`.
- Reduce `--render_height 128 --render_width 128`.
- Reduce `--num_latents 1024`.

### Grey GT images (Objaverse)

This indicates the mesh has no vertex colors and no UV texture. Solutions:
1. Switch to ShapeNet (always has UV textures).
2. For Objaverse: filter to GLB files that have embedded textures.

---

## Recommended training workflow (condensed)

```text
Phase 0  (done): Bug fixes, ShapeNet dataset preparation
Phase 1a (smoke): 5 objects, 1 view, 1k steps → verify pipeline
Phase 1b (baseline): chairs, 1 view, 20k steps → establish reference PSNR
Phase 2a: chairs, 2 views, 30k steps → test multi-view benefit
Phase 2b: chairs, 4 views, 50k steps → best 3D consistency
Phase 3a: 3 categories, 4 views, 100k steps → more data
Phase 3b: larger model (4096 latents), if Phase 3a plateaus
Phase 3c: add QK norm (only for 12+ layers); add DropPath (only if overfitting)
```

After each phase, run `evaluate_gs_ae.py` and compare PSNR/SSIM in the summary CSV.
The model with the highest `mean_psnr_fg` on the val set is the winner.
