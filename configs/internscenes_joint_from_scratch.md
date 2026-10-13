# Two random-view joint cold starts

These replace the staged/fixed-pair proposal. `coverage4` is the main candidate;
`coverage2` is the matched control. **Only `cd_gt2pred` differs** (4 versus 2).
The combination of local OT and coverage weight 4 has not previously been tested
in a cold-start joint run; it is a candidate, not a validated optimum.

Both use all four curated views per room, sampling random ordered pairs: twelve
pairs per room and 120 in total. There is no reconstruction-only stage, frozen
tokenizer, scene checkpoint resume, or reduced flow weight. GE, encoder, decoder,
and learned VGGT projections train jointly from update one. Original Hunyuan
point encoder input projection and cross-attention initialize the encoder; the
rest starts fresh, as in the original project experiment.
The original encoder architecture has Q/KV biases absent from the Hunyuan
checkpoint; these retain their fresh initialization, as before. Any other
cross-attention key mismatch or missing encoder subtree aborts initialization.

| Shared setting | Value |
|---|---|
| Updates / batch / seed | 40,000 / 2 / 0 |
| Flow loss samples per update | 4, averaged (does not change lambda_flow) |
| Reconstruction / flow weights | 1 / 1 |
| FPS density k / sharp beta / percentile clip | 8 / 0.2 / 5–95 |
| Register noise / AE modulation time / representation noising | fixed / 0 / disabled |
| Flow initial noise / context dropout | random Gaussian / 0.1 |
| Global Chamfer pred→GT | 1 |
| Local OT | 0.1, 64 supported patches per sample, radius scale 0.5 |
| Anchor Sinkhorn / anchor CD / delta / RGB | 0.01 / 10 / 0.001 / 10 |
| Precision / Chamfer | FP32 / exact chunked, chunk size 1024 |
| LR | 1,000-update linear warmup to 1e-4, cosine decay to 1e-5 |
| Gradient clip | 1; abort on nonfinite loss or gradient |

The deterministic AE settings apply to other dataset sizes; generated latents
retain random initial noise. Geometry and raw frozen VGGT features preload
once for every pair. Learned features/latent targets are recomputed each update.
FPS indices and local GT targets cache up to 128 entries. A separate seeded
DataLoader generator keeps room order independent of flow RNG consumption.

Evaluation runs in process, reusing the preloaded inputs and restoring Python,
NumPy, CPU/CUDA RNG and train/eval mode. Every 2,000 updates it scores recon and
CFG=1 generation on **all 120 pairs**, with seed 0 and 50 sampling states. Final
evaluation uses seeds 0,1,2 on every pair. It reports directional CD, precision,
recall/F1 at normalized thresholds, distance tails, GT-cell coverage/worst-cell
recall, point spacing/duplicates, and RGB. These are normalized distances, not
metres, and whole-room metrics do not replace object-level visual inspection.
Compare candidates on identical room/pair GT. Intermediate and final seed-0
metrics are directly comparable; the final three-seed mean has a different scope.

`eval/final/<room>/` exports the first curated ordered pair: gt, fps, anchors,
recon, gen_cfg1 (seed 0), recon_error and gt_uncovered. Other pairs/seeds have
metrics. To export a promising earlier checkpoint, run `evaluate_pc_joint.py`
with the same pack, view map and VGGT cache and a new output directory.

Model-only snapshots save every 2,000 steps. `ckpt_last_resume.pt` atomically
replaces one full checkpoint with Adam/scheduler state. Final checkpoint includes
optimizer state. Resume is suitable for recovery, but exact random-stream replay
after interruption is not promised. About 20 model snapshots plus two full
checkpoints per run are retained. Completion marker requires both training and
final evaluation to succeed. `metrics.jsonl`, `evaluation_summary.jsonl`,
`runtime_summary.json`, exact command, git revision and GPU identity are saved.
Shared-GE weighted reconstruction/flow gradient norms, their ratio and cosine
are logged every 1,000 steps, before total-gradient clipping.

## Submission

Commit/push the changes from the home checkout, then pull in the persistent
checkout. Prepare the original encoder weights **once before qsub**:

```bash
cd /projects_vol/gp_chuanxia.zheng/nathan/HY3DGS
git pull --ff-only
/projects_vol/gp_chuanxia.zheng/nathan/env_hy3dgs/bin/python prepare_joint_pretrained.py
qsub -N joint_cov4 -v EXPERIMENT=coverage4 train_joint_from_scratch.pbs
qsub -N joint_cov2 -v EXPERIMENT=coverage2 train_joint_from_scratch.pbs
qstat -u n2506007l
```

Preparation uses the already downloaded original Hunyuan weights offline and
copies only encoder tensors into `runs/model_cache/hunyuan_encoder_cross_attn.pt`.
It does not load any room-trained checkpoint. Once prepared, jobs have no home
cache dependency. If the original cache is elsewhere, pass `--source /path/to/
original/model.fp16.ckpt` to preparation.

PBS requests one isolated GPU, 12 CPUs, 250 GB host memory and a 24-hour maximum.
Concurrent jobs receive separate scheduler-assigned GPUs when resources permit;
there is no manual GPU index. All outputs, logs, checkpoints and temporary files
are rooted at `/projects_vol/gp_chuanxia.zheng/nathan/HY3DGS/runs/`.
Runs are `runs/joint_from_scratch/coverage{4,2}_<PBS_JOBID>/` and logs are
`runs/logs/joint_from_scratch_coverage{4,2}_<PBS_JOBID>.log`.
Do not reuse an existing output directory.

Full-size GPU throughput and evaluation cost have not been measured on the
login host. Four flow samples are more expensive than the prior two-sample
proposal. Use training seconds/step and the first all-pair evaluation duration
to estimate completion; the 24-hour request is a ceiling, not an ETA.

Choose using both recon and gen, across all pairs: F1/precision, uncovered GT
tails and cells, duplicate clusters, RGB, and visual small-object clarity. A lower
mean CD alone cannot establish that the ten-room overfit works or justify scaling
to 100 rooms. Both candidates need evaluation; no optimum is guaranteed.
