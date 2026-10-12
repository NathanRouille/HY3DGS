# Fast reconstruction screen

Three independent 2,000-step fine-tunes from `runs/internscenes_overfit_bath10_selected_v2_delta0p001/ckpt_final.pt`:

| EXPERIMENT | Objective |
|---|---|
| `global` | Existing reconstruction objective |
| `local_cd` | Existing objective + 0.1 × local symmetric CD |
| `local_ot` | Existing objective + 0.1 × local quadratic OT |

Rooms: 12147, 5658 and 6996. Training uses each room's first selected ordered pair; final evaluation also checks the other selected pair. These additional pairs are not an unseen-room test: the shared starting checkpoint already trained on the full selected pools.

All runs use the same parent checkpoint, surface seed, model/data RNG seed, batch size two, reset AdamW optimizer, constant learning rate 1e-5, global CD weights 1/2, delta weight 0.001 and FPS density k=16. Representation noising is disabled for the common reconstruction diagnostic. The baseline controls for this protocol change; comparing any candidate directly to the old joint run does not isolate the local term.

## Launch from the persistent checkout

Commit and push the changes from the home checkout, then pull them here before submitting:

```bash
cd /projects_vol/gp_chuanxia.zheng/nathan/HY3DGS
git pull --ff-only
qsub -N recon_global -v EXPERIMENT=global train_recon_screen.pbs
qsub -N recon_local_cd -v EXPERIMENT=local_cd train_recon_screen.pbs
qsub -N recon_local_ot -v EXPERIMENT=local_ot train_recon_screen.pbs
qstat -u n2506007l
```

Each job requests one GPU. PBS assigns different devices to simultaneous jobs, even if two jobs share a physical node. Do not set `CUDA_VISIBLE_DEVICES` or choose a GPU index manually; PBS/cgroups handles isolation. Avoid host pinning, which can prolong queueing. The supplied newer snapshot has all `gpu_as` nodes' GPUs allocated, so the jobs may queue. GPU counts outside this queue do not imply access through your project.

To force a common FP32 policy, add `PRECISION=fp32` to **all three** submissions, e.g. `qsub -N recon_global -v EXPERIMENT=global,PRECISION=fp32 train_recon_screen.pbs`. To change the common step budget, pass `STEPS=3000` consistently to all three. No jobs are automatically submitted by the scripts.

## Runtime changes

`--recon_only` bypasses the entire flow forward, and avoids VGGT/cache conditioning when the tokenizer does not use it. The original joint path remains the default. Fixed normalized surfaces are loaded once and render tensors discarded. Deterministic weighted-FPS indices are cached in memory; full candidate coordinates, features and all query settings are checked before reuse. Caches are bounded and are not checkpoint state. Learned encoder/decoder features are recomputed every step and remain trainable.

Exact chunked Chamfer searches all target points, preserves first-index ties, and differentiates the selected squared residuals. It does not subsample the point clouds or use approximate neighbours. Its objective and gradients agree with dense Chamfer up to floating-point roundoff. Chunking is primarily a memory improvement; GPU throughput must be measured before assigning it a speedup factor.

The existing anchor Sinkhorn already detaches its transport plan. Its iterations now run without building an unused autograd graph; the original plan values, count scaling and coordinate gradients are preserved and checked against the historical objective.

The launchers request BF16 for neural-network computation, with FP32 decoder coordinate/color heads, geometry losses and transport. Before any optimizer update, each job probes the three diagnostic rooms using the same parent checkpoint and the same baseline criterion in FP32 and BF16, checking finite gradients, coordinate deviation, loss difference and gradient norm ratio. BF16 failures fall back to FP32. RNG is restored after the probe. Inspect `precision_preflight.json` to confirm all three used the same precision before comparing them. The thresholds are normalized-coordinate p95 <=0.002 and p99 <=0.004 in every room, maximum deviation <=0.01, baseline total-loss relative difference <=3%, Chamfer relative difference <=5%, and gradient norm ratio in [0.8,1.2]. This is an initial numerical guard, not proof that training trajectories are identical. The login environment has no usable GPU, so no full-size GPU timing or BF16 validation was performed before submission.

The output heads remain FP32 under autocast to avoid BF16 coordinate quantization. Final reconstruction evaluation uses FP32 for every candidate. BF16 is opt-in for the trainer and is currently restricted to the reconstruction-only path.

## Local targets

Each FPS seed gets the nearest 16 **distinct** GT coordinates in the same scene frame. A patch is supported only if its farthest target is within half the eighth-other-FPS distance. Targets can overlap spatially; this is overlapping patch shape supervision, not a partition or global density constraint. Unsupported patches are skipped rather than filled with duplicates or remote points.

FPS and decoder anchors are matched by a detached mutual nearest-neighbour correspondence, giving a partial one-to-one spatial matching without assuming register index identity. Matches farther than the support radius are rejected. Up to 64 valid patches per sample are used, cycling deterministically through the supported matches without consuming the model RNG. The GT targets are cached but current anchor matching and prediction gradients are recomputed each step.

Local CD and OT share this target construction, support gate and schedule. OT uses SciPy Hungarian assignment on 16×16 patches, an exact equal-mass quadratic EMD-style objective with no entropic shrinkage or new dependency. Its mean assignment cost is multiplied by two to match the two unit-weight directions of symmetric local CD. This normalization differs from the historical anchor Sinkhorn's N multiplier; that existing anchor term is preserved.

On the completed checkpoint's exported clouds, these gates yielded 823, 506 and 584 valid correspondences for rooms 12147, 5658 and 6996, respectively, enough for 64-patch supervision. Initial runs abort if fewer than eight supported local patches per sample are available, and logs include supported/used counts and weighted loss gradients.

## Results and checks

All runtime files are under `/projects_vol/gp_chuanxia.zheng/nathan/HY3DGS/runs`:

- `logs/recon_<experiment>_<PBS_JOBID>.log`
- `recon_screen/<experiment>_<PBS_JOBID>/args.json`
- `recon_screen/<experiment>_<PBS_JOBID>/precision_preflight.json`
- `recon_screen/<experiment>_<PBS_JOBID>/metrics.jsonl`
- `recon_screen/<experiment>_<PBS_JOBID>/ckpt_0021000.pt` (midpoint, model only)
- `recon_screen/<experiment>_<PBS_JOBID>/ckpt_final.pt`
- `recon_screen/<experiment>_<PBS_JOBID>/baseline_eval/`, `eval/`, `eval_extra_views/`

Each evaluation exports `gt.ply`, `recon.ply`, `fps.ply`, `anchors.ply` per room and writes `metrics.json`. Coordinates are normalized in the first-camera frame, like prior eval outputs. Metrics explicitly label normalized thresholds; do not interpret these as metres. No generation PLY is exported during these AE diagnostics because the flow model has not been adapted to the updated latent representation.

Training logs record global CD directions, RGB, delta, local loss, valid patch counts, weighted parameter-gradient norms, preclip gradient norm, clipping factor, averaged seconds per step and peak PyTorch allocated GPU memory. Gradient diagnostics run initially and every 500 steps. Intermediate generation visualization is disabled and checkpoints omit optimizer state. These runs intentionally reset the optimizer; do not interpret model-only checkpoints as exact optimizer continuation snapshots.

The PBS files request four CPUs, one GPU, 48 GB host memory and eight hours. Output paths include the PBS job ID and refuse to overwrite an existing run. The wrapper checks that CUDA can initialize and sees exactly one assigned GPU, and records the device and Git revision. On the allocated GPU it runs the CPU regression tests, exports baseline reconstruction, performs the precision preflight, trains and exports both final view-pair evaluations. Training memory peaks are reset after the preflight to measure the actual training batch rather than the larger precision probe.

## Deadline-driven follow-up

Run these three candidates together, then review small-object coverage, surface noise and room scale alongside per-room CD/RGB and the extra views. If a candidate is clearly better, continue that recipe on all ten rooms and adapt flow once. Repeat the winner/control or test a single CD/delta change only when the outcome is close or reveals a specific remaining problem. This is a quick controlled screen, not the original 18-phase campaign or a three-seed robustness claim. Changing batch size or point count is intentionally excluded from these comparisons.
