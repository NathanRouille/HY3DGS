# Reconstruction sanity experiments

These four independent continuations address the reported clumps, holes and
noisy objects before spending more time on generation or dataset expansion.
They use the original 20k selected-v2 joint checkpoint, not an OT-finetuned
checkpoint or a frozen-flow checkpoint.

| EXPERIMENT | Training pairs | Local loss | GT-to-pred CD weight |
| --- | --- | --- | --- |
| fixed_global | First selected pair per room | none | 2 |
| fixed_ot | Same fixed pair | OT, weight 0.1 | 2 |
| random_global | All 120 ordered selected pairs, randomly sampled | none | 2 |
| fixed_coverage | Same fixed pair | none | 4 |

Compare every variant with fixed_global. Fixed_ot tests whether the existing
nearest-16 local target construction causes compact patches and gaps.
Random_global tests the added difficulty of changing camera frames and pair
normalization. Fixed_coverage tests filling missing GT regions; precision and
outlier tails must also be checked because a coverage gain can have a cost.
The latter two are supplementary controls, not a claim they will improve quality.

Shared settings: all ten rooms, 8,000 additional AE updates, FP32, seed 0,
batch size 2, reset optimizer, constant LR 1e-5, weight decay 0.01, fixed register
noise from the checkpoint's persistent buffer, tokenizer modulation time zero,
representation noising off, no tokenizer image context. Pred-to-GT CD weight 1,
RGB weight 10, anchor Sinkhorn 0.01, anchor CD 10, delta 0.001. All use
fps_density_k=16, 2,048 anchors/registers and 16 output points per anchor.
Fixed runs cache ten normalized surfaces. Random_global caches all 120 surfaces.
Encoder/local target cache capacities are 128. No VGGT, weak context, flow
forward or generation evaluation occurs in any job.

The first pair is the first two entries in each room's selected-view pool.
Those views remain the user's curated views. Random_global changes only pair
selection and its necessary cache wrapper. Neither fixed run changes camera
alignment or checks normalization using another recipe.

## Evaluation and outputs

Each job evaluates the original source checkpoint with its saved stochastic
tokenizer (baseline_native), then with fixed register noise/time zero
(baseline_fixed). Both use the same fixed GT samples and primary pair as the
training/evaluation protocol. This distinguishes the original checkpoint's
quality from the immediate effect of changing tokenizer inputs.

The train process runs once without intermediate optimizer resets. Model-only
checkpoints are saved every 1,000 steps, then evaluated after training completes.
Intermediate and baseline evaluations use the primary pair only. The final
evaluation reports primary and alternate pairs separately; it exports PLYs for
the primary pair. The alternate pair is not fitted by the fixed runs and should
not be averaged into their overfitting pass/fail decision. It is in the training
pool for random_global.

Metrics include both squared CD directions, precision/coverage/F1 at normalized
0.002/0.005/0.01/0.02/0.05, distance medians/p95/p99, RGB error, GT-defined occupied
cell coverage, macro cell recall and worst-decile cell recall. Cells have size
0.1 in the pair's normalized coordinates; macro/tail summaries require at least
four GT samples per cell. These are spatial diagnostics, not semantic object
scores. Nearest-neighbor spacing statistics and a near-duplicate fraction describe
density/clumping relative to median distinct-GT spacing. They should not be
optimized blindly: GT sampling itself is nonuniform and includes sharp samples.
All thresholds are normalized units, not centimeters.

Each primary room exports gt.ply, recon.ply, fps.ply and anchors.ply, plus
gt_uncovered.ply and recon_error.ply. The diagnostic clouds contain all points;
green means zero distance error, red saturates at normalized distance 0.02.
Use gt_uncovered to locate holes and recon_error to locate outliers, with the
ordinary RGB clouds beside them. These colors are diagnostics, not RGB predictions.

```text
/projects_vol/gp_chuanxia.zheng/nathan/HY3DGS/runs/recon_sanity/<experiment>_<jobid>/
  commands.txt, git_commit.txt, gpu.json
  baseline_native/, baseline_fixed/
  train/args.json, metrics.jsonl, ckpt_*.pt, ckpt_final.pt
  eval_ckpt_0021000/, ... , eval_ckpt_0027000/
  eval/metrics.json, <room>/*.ply
  quality_summary.csv
  COMPLETE
```

Baseline and intermediate directories also contain per-room metrics and PLYs.
Absolute checkpoint names above assume the default 20k source and 8k budget.
The final checkpoint is at step 28,000. Baselines, seven intermediate states and
the final state provide matched learning curves and PLYs across the entire run.
Checkpoints are model-only; exact optimizer-state continuation is unavailable.
Outputs, model caches, temporary files and logs all use the persistent group
repository through the PBS launcher. Logs are runs/logs/recon_sanity_*.log.

Based on the last runs, training should take approximately 64 minutes for 8k
updates. Baseline, intermediate and final evaluation add time; plan roughly
1.5–2 hours per job, not a verified timing guarantee. PBS allows three hours
for startup/I/O variation. Each independent job requests one GPU. Jobs can start
together when resources are available; PBS controls GPU placement.

```bash
cd /projects_vol/gp_chuanxia.zheng/nathan/HY3DGS
qsub -N sanity_global -v EXPERIMENT=fixed_global train_recon_sanity.pbs
qsub -N sanity_ot -v EXPERIMENT=fixed_ot train_recon_sanity.pbs
qsub -N sanity_views -v EXPERIMENT=random_global train_recon_sanity.pbs
qsub -N sanity_cover -v EXPERIMENT=fixed_coverage train_recon_sanity.pbs
```

Inspect primary-pair reconstruction first: all ten scenes should have recognizable
small objects, continuous large surfaces and no obvious clusters or outlier clouds.
Compare hole tails, cell recall and precision alongside CD, rather than ranking
by CD alone. The budget is a short diagnostic continuation, not proof that a
failure after 8k updates means insufficient capacity. If curves and PLYs improve,
we can extend the useful run. If local OT hurts coverage, inspect its target
construction before increasing its weight. Fit flow only after a reconstruction
setup passes this visual and numerical check.
