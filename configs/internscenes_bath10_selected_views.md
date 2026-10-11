# Selected-view ten-room overfit comparison

The JSON records the user-provided room IDs and view order. Room 12237 was confirmed as the correction to 12137. All ten rooms and their selected RGB/depth/pose files have been validated. Leading zeroes are display formatting; integer IDs preserve their meaning and listed order.

From the HY3DGS checkout with its usual environment:

```bash
DELTA_WEIGHT=0 bash run_internscenes_overfit_bath10_selected_views.sh
DELTA_WEIGHT=0.001 bash run_internscenes_overfit_bath10_selected_views.sh
```

Each command runs one experiment: validates image/pose files, creates/reuses 120 ordered-pair cache files (12 per room), trains, then evaluates the first ordered pair from each room's listed pool. Run inside the existing GPU job setup, not on a login node. These are fresh runs with the same initialization and Python/Torch seeds. Both use batch 2, 20000 steps, the existing normalization/FPS/model/loss recipe, and separate output directories. Their difference is lambda_delta. PACK, VIEW_MAP, CACHE, OUT, DELTA_WEIGHT, and NUM_STEPS are configurable environment variables. Set NUM_STEPS identically for both experiments; default 20000 keeps the initial comparison budget fixed, not proof of converged ten-room overfit.

The cache directory is `runs/vggt_cache/internscenes_bath10_joint_selected_v2`. Output directories are `runs/internscenes_overfit_bath10_selected_v2_delta0` and `runs/internscenes_overfit_bath10_selected_v2_delta0p001`. The earlier script/run/cache are preserved. Selected pools are embedded in checkpoints so evaluation can use them without the original JSON path. An explicit evaluation --view_indices overrides checkpoint pools globally; --room_view_pools_json overrides them per room.

Training randomly samples ordered pairs from each room's four-image pool. Evaluation uses its first two listed views in that order. All 12 ordered pairs must be cached for each four-image pool; missing views/cache tuples fail rather than silently dropping rooms. Evaluation on a different pair can use a separate per-room JSON mapping. Keep the same pairs/noise seeds between comparisons. New images change normalization and metric scale, so comparison against the previous [1,3] evaluation is not a strict same-input comparison.

For 5658 the new first pair [23,29] has visible-depth RMS 2.581 m and full GT normalized RMS about 2.914, versus 0.712 m and 8.72 with [1,3]. Across the new six unordered pairs, full GT normalized RMS ranges about 1.55–4.65. Keep watching difficult pairs, small-structure GT-to-pred coverage, parent-offset percentiles, and plane/clump density. Delta reduction alone is not proof of recovered detail. Batch 2 is retained because the prior run used approximately 88% of device memory.

No GPU cache build, training, or model evaluation was launched when preparing this comparison.

## PBS submission from persistent project storage

`train_selected_views.pbs` is version-controlled in this repository. Commit and push from the home checkout, then pull in `/projects_vol/gp_chuanxia.zheng/nathan/HY3DGS` before submitting:

```bash
cd /projects_vol/gp_chuanxia.zheng/nathan/HY3DGS
git pull --ff-only
qsub train_selected_views.pbs
```

The job uses group paths for source code, selected-view JSON, the dataset and Conda environment. It preserves the previous project, queue, CPU/GPU request and 24-hour walltime. The default experiment uses delta 0.001, batch 2, and 20000 steps.

Persistent runtime paths:
- Results/checkpoints/evaluation: `runs/internscenes_overfit_bath10_selected_v2_delta0p001`.
- Joint VGGT cache: `runs/vggt_cache/internscenes_bath10_joint_selected_v2`.
- Job logs: `runs/logs/bath10_selected_views_<PBS_JOBID>.log`.
- Hugging Face / Torch / XDG caches: `runs/model_cache`.
- W&B run records: under the experiment output; W&B cache: `runs/wandb_cache`.

These paths are all relative to the persistent group checkout. Existing default caches are not moved; model weights may download into the new persistent cache on the first run. W&B online authentication still uses the account's existing login or WANDB_API_KEY. No credentials were copied. No job was submitted during preparation.
