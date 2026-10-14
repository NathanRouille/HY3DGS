# Balanced-patch joint cold starts

All four runs start GE, decoder, flow and learned VGGT projections fresh. Only
the original Hunyuan input projection/cross-attention initialize the encoder.
There is no scene checkpoint resume, reconstruction-only stage or frozen
tokenizer. All ten rooms use random ordered pairs from their four curated views.

| Experiment | Queries / target patches | Query descriptors | Anchor / local supervision |
|---|---|---|---|
| `clean_fps` | Legacy density FPS k8, repaired GT | None | Previous Sinkhorn + anchor CD, gated 16-NN OT, delta .001 |
| `patch_geometry` | Disjoint 16-point patches, spatial refinement | None | Hungarian anchor loss + complete owned-patch OT, delta 0 |
| `patch_features` | Same construction with weak RGB/normal preferences | None | Same as patch_geometry |
| `patch_full` | Same patches as patch_features | Log radius + normalized covariance | Same as patch_geometry |

The matched comparisons are geometry→features (partition features), and
features→full (query descriptors). Control→geometry tests the new ownership and
loss package; it does not isolate each individual change. `clean_fps` differs
from 85935 by its cleaned target cloud and less frequent evaluation/checkpointing.
Its training loss, FPS k8, model, optimizer schedule and update budget match.

All use the exact same prepared surface rows: 32,768 distinct FP32 world-space
positions, R=2048, K=16. Query descriptors are derived after the existing camera
rotation and depth-based normalization. They are never fed into image-only
generation. Free predicted registers match target patches spatially, without
an assumed index identity. Every target patch is eligible; 256 target patches
per room presentation cycle across all view pairs, completing a cycle in eight
presentations. No sparse-radius gate, target reuse, hard delta clamp or delta
penalty exists in the three patch candidates.

Shared settings: seed0, batch2, FP32, 40k updates, full λrecon=λflow=1, four flow
noise/time samples per update, LR warmup1000→1e-4 cosine→1e-5, exact chunked global
CD weights1/2, RGB10, local squared EMD0.1 with the previous factor-two convention.
Patch target losses use absolute positions (anchor+delta), so anchor displacement
is included. Radius normalization is not applied to the geometry loss.

The matched anchor loss is a **mean squared Euclidean distance**. On the first
batch, its weight calibrates to the old .01 Sinkhorn+10 anchor-CD gradient norm
on an isolated, deterministic near-target perturbation probe. This avoids using
unusually large many-to-one NN coverage gradients at cold start to set a
permanent matched-loss coefficient. This calibrates magnitude,
not gradient direction or exact loss equivalence. The resolved weight is logged
and saved in args/checkpoints, then reused on resume. The reference losses are
not evaluated on subsequent steps. Per-term weighted model gradients and shared
GE reconstruction/flow gradients remain logged every1000 updates.

## Prepare data once, outside GPU jobs

Preferred, when original GLBs are accessible:

```bash
cd /projects_vol/gp_chuanxia.zheng/nathan/HY3DGS
python prepare_balanced_patches.py \
  --data_dir /projects_vol/gp_chuanxia.zheng/nathan/datasets/internscenes_bathroom_130 \
  --mesh_root /path/to/InternScenes/processed/glb_output/sample_scenes \
  --sampling mesh_fp32 --output_dir runs/patch_data/bath10_distinct_v1
```

This retains the existing surface/edge sampler, area quotas and 16k/16k intended
allocation, but removes both float16 casts and checks distinct positions after
FP32 world restoration. Additional draws fill rare collisions; inability to
obtain the quota aborts preparation. It does not modify the original pack.

The original room meshes were unavailable in this workspace. An explicit
fallback can prepare usable targets from the packed points:

```bash
python prepare_balanced_patches.py \
  --data_dir /projects_vol/gp_chuanxia.zheng/nathan/datasets/internscenes_bathroom_130 \
  --sampling distinct_source --output_dir runs/patch_data/bath10_distinct_v1
```

**The fallback cannot undo quantization or preserve 16k distinct sharp points.**
It keeps available distinct sharp points (5658:941), fills the budget with
distinct uniform points, and records actual counts/deficits. This changes the
target sampling distribution. Compare all candidates with the clean_fps control
on this common GT; previous raw-GT CD/F1 are not directly comparable. Do not mix
preparation modes between the four runs. Choose a fresh output directory for
another preparation; existing room artifacts are protected from overwriting.

Preparation creates geometry/features partitions using PCA median splits and
objective-reducing exchanges among neighboring groups. RGB/normal penalties
are .05 times the initial median patch spatial variance. This is a local
optimization, not a guarantee of optimal compactness or semantic purity.
Mesh component costs are optional in the underlying function, but these first
experiments use the already available XYZ/RGB/normals without mesh-ID requirements.

Each room has gt.ply, colored patches_geometry/features.ply, query previews and
radius/extent statistics. These previews share original GLB world coordinates;
training/evaluation PLYs use the normalized first-camera coordinates as before.
Review outlying extents or obvious mixing in the previews before allocating an
overnight run. The manifest records source mode, counts and surface/patch hashes.
All GPU launchers validate distinctness, ownership, quotas and hashes before
training, and check the ten-room / 120 ordered VGGT-cache-pair coverage.

## Optional GPU assignment

SciPy is the reference and requires no new dependencies. An optional GPU package
can be installed **outside the shared conda environment**, without upgrading
Torch or its dependencies:

```bash
python -m pip install --no-deps --target runs/assignment_deps 'torch-linear-assignment==0.1.0rc0'
```

The existing Torch/Triton must satisfy the package requirements. PBS adds this
directory to PYTHONPATH. `auto` checks the first actual cost matrix against
SciPy's optimal cost and a complete bijection, then measures warm execution.
GPU is selected only when it passes and is at least10% faster. This happens
separately for anchor and local-patch workloads; cold compilation is excluded
from warm timing. Missing/incompatible packages or validation failures fall back
to SciPy and log the reason. No download/install is attempted inside a job.
For a strictly SciPy run, add ASSIGNMENT_BACKEND=scipy to qsub's variable list.
GPU execution was not available for verification on this login host.

Reference: https://github.com/ivan-chai/torch-linear-assignment

## Submit

Push the home changes and pull the persistent checkout, activate the existing
env, then prepare data once. Persistent pretrained encoder weights already exist
under runs/model_cache; if missing, run python prepare_joint_pretrained.py.

```bash
qsub -N patch_full -v EXPERIMENT=patch_full train_balanced_joint.pbs
qsub -N clean_fps -v EXPERIMENT=clean_fps train_balanced_joint.pbs
qsub -N patch_feat -v EXPERIMENT=patch_features train_balanced_joint.pbs
qsub -N patch_geom -v EXPERIMENT=patch_geometry train_balanced_joint.pbs
qstat -u n2506007l
```

Each requests one scheduler-isolated GPU; different concurrent jobs receive
different allocated GPUs. Do not set a physical CUDA device number manually.
Host memory is64GB rather than250GB: the previous job used about10GB resident
host RAM. This permits better node packing while leaving a substantial margin;
GPU memory and batch size are unchanged.
All training outputs/checkpoints, logs, data, temporary files and model/Triton
caches remain under `/projects_vol/gp_chuanxia.zheng/nathan/HY3DGS/runs`.

Runs: runs/balanced_joint/<experiment>_<PBS_JOBID>
Logs: runs/logs/balanced_joint_<experiment>_<PBS_JOBID>.log
Snapshots/evaluation every4000 updates; full optimizer recovery snapshot retained.
All120 pairs receive intermediate recon + generation seed0 evaluation. Final
evaluation uses generation seeds0/1/2. Final primary-pair exports retain the
familiar gt/fps/anchors/recon/gen_cfg1.ply names; fps.ply represents patch queries
in the candidates. Each primary room also gets owned_patch_metrics.json with
full-patch EMD, anchor error, centroid error and dense/sparse spread ratios.
These diagnostic ratios use the run's selected target partition; global CD/F1
and shared GT remain the primary cross-run metrics.

40k previously took about20h including evaluation; the new assignment throughput
is not yet measured on a GPU. PBS requests24h; first runtime logs determine the
ETA, and the recovery checkpoint protects against a walltime interruption.
To use a shorter matched budget, specify NUM_STEPS=24000 on **all four jobs**;
this changes their cosine schedule and requires comparison to the matched control.
Do not declare a successful 10-room generation overfit from recon metrics alone.
