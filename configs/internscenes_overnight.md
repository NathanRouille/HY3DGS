# Ten-room overnight experiments

Each PBS job independently runs reconstruction fine-tuning, evaluates that
checkpoint, adapts the flow with the entire reconstruction pathway frozen, then
evaluates reconstruction and image-only generation. These are overfitting
experiments on ten rooms, not held-out generalization experiments.

| EXPERIMENT | Local loss | Anchor Sinkhorn weight | FPS density k |
| --- | --- | --- | --- |
| control | none | 0.1 | 16 |
| local_ot | exact 16-point assignment, weight 0.1 | 0.1 | 16 |
| weak_anchor | same local OT | 0.01 | 16 |
| density_k8 | same local OT | 0.01 | 8 |

Compare control vs local_ot, then local_ot vs weak_anchor, then weak_anchor vs
density_k8. Each adjacent comparison changes one factor. All start from the
original selected-v2, delta=0.001, 20,000-step ten-room checkpoint, rather than
the short three-room screening checkpoints.

Shared settings: 8,000 additional reconstruction steps, 12,000 additional flow
steps, batch size 2, seed 0, FP32, fresh optimizer in each phase, constant LR
1e-5, representation noising off. Global CD direction weights stay 1/2,
RGB weight 10, anchor CD weight 10, delta weight 0.001. Flow uses velocity loss,
eight sampled timesteps per optimizer update, context dropout 0.1, no off-path
loss, and trains the live GE and context projections. It snapshots the trained
AE GE for the frozen tokenizer and initializes denoiser positions from the AE.
Latent targets are recomputed because tokenizer register noise and modulation
time are stochastic; caching them would change the training objective.

All four selected views per room participate through random ordered pairs.
Every normalized surface is preloaded on CPU; flow also preloads the existing
120 joint VGGT payloads (~5.1 GB on disk). Room sampling and random pair choices
remain unchanged. Learned context projections still receive gradients. Runtime
FPS and local-target caches hold all 120 pairs; no online VGGT forwards or image
reads occur in the training loop. The global chunked NN search is exact, not
an approximate loss. The earlier BF16 gate failed, so these jobs use FP32.

The step budget is identical across jobs; there is no per-variant timed stop.
The 14-hour PBS limit leaves margin for startup and evaluation. Ten hours is
an aim, not a verified duration: pure-flow throughput has not been measured
for this new pipeline. Timing and peak GPU memory appear in each phase's
metrics.jsonl. If changing budgets, pass the same RECON_STEPS/FLOW_STEPS to all
four submissions. Each job requests one GPU; PBS determines placement and may
queue jobs if four GPUs are not available.

From the persistent repository after pulling the changes:

```bash
cd /projects_vol/gp_chuanxia.zheng/nathan/HY3DGS
qsub -N bath10_control -v EXPERIMENT=control train_overnight.pbs
qsub -N bath10_ot -v EXPERIMENT=local_ot train_overnight.pbs
qsub -N bath10_anchor -v EXPERIMENT=weak_anchor train_overnight.pbs
qsub -N bath10_k8 -v EXPERIMENT=density_k8 train_overnight.pbs
```

All outputs use the persistent group repository:

```text
runs/overnight/<experiment>_<PBS_JOBID>/
  git_commit.txt, gpu.json, commands.txt
  recon/ckpt_final.pt, args.json, metrics.jsonl, ckpt_*.pt
  recon_eval/metrics.json, <room>/{gt,recon,fps,anchors}.ply
  flow/ckpt_final.pt, args.json, metrics.jsonl, ckpt_*.pt
  eval/metrics.json, <room>/{gt,recon,fps,anchors,gen_cfg1}.ply
  COMPLETE
runs/logs/overnight_<experiment>_<PBS_JOBID>.log
```

Checkpoints are model-only, with intermediate saves every 2,000 steps. Resume
requires a fresh optimizer; exact optimizer-state continuation is unavailable.
Flow monitoring runs full reconstruction only every 2,000 steps and initially.
Before the expensive reconstruction phase, each launcher validates all ten
rooms and all ordered cache pairs. A COMPLETE marker is written only after
both phases and both evaluations succeed.

Evaluation uses the first two selected views and the last two selected views
of every room, three common generation seeds (0/1/2), CFG=1, 50 ODE states and
no endpoint renormalization. These are two training-pool pairs, not held-out
views. Only the primary pair and seed are exported to PLY. Metrics use exact
CPU nearest neighbors and report both CD directions, outlier p95, coverage,
precision, F-score and RGB error. Thresholds are in normalized units; pair
normalization changes physical scale. Compare variants on identical room/pair
records, and inspect objects visually alongside the room averages.

The final evaluator checks every frozen reconstruction tensor against the AE
checkpoint, including equality of the tokenizer GE snapshot with the trained
AE GE. It aborts if flow training changed those weights. Compare AE quality
first, then generation quality across pairs and seeds. Global CD alone can
favor walls over small objects.
