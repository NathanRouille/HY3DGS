# Compare exp14 vs NOVA3R vs Surflo (furniture_351)

Self-contained harness under `compare_baselines/`. **Does not modify** the
`nova3r` or `Surflo` repositories — each model is invoked in its own conda env.

## One-shot (recommended)

From the HY3DGS repo root:

```bash
# Full validation set (35 objects)
bash compare_baselines/run_compare.sh val

# Train subset (e.g. first 100)
bash compare_baselines/run_compare.sh train --max_items 100
```

This runs, in order:

1. **`hy3dgs`** — prepare letterboxed views + GT PLY  
2. **`hy3dgs`** — exp14 generation (`ckpt_0030000.pt`)  
3. **`nova3r`** — scene_n2 on letterboxed 518×392 views  
4. **`surflo-cu124`** — plain Surflo (downloads `surflo_v0.pt` if missing)  
5. **`hy3dgs`** — align export + Chamfer / accuracy / completeness  

### Useful flags

```bash
bash compare_baselines/run_compare.sh val --limit 2          # debug few objects (nova/surflo)
bash compare_baselines/run_compare.sh val --skip_prepare     # reuse existing OUT_ROOT
bash compare_baselines/run_compare.sh val --skip_exp14 --skip_nova3r --skip_surflo  # re-score only
```

## Re-score alignments only (no re-inference)

```bash
conda activate hy3dgs
cd /export/home/nathan/HY3DGS

python compare_baselines/align_and_score.py \
  --compare_dir runs/compare_exp14_baselines/val \
  --identity_exp14

# Single object debug:
python compare_baselines/align_and_score.py \
  --compare_dir runs/compare_exp14_baselines/val \
  --identity_exp14 --index 0

python compare_baselines/align_and_score.py --list_align_methods
```

## Outputs (per object)

```
objects/0000_<uid>/
  pred_raw/{exp14,nova3r,surflo}.ply
  pred_aligned/
    gt_anchored/          # Surflo-friendly: free-scale ICP on raw cloud
    robust_filter/        # NOVA-friendly: densify → normalize → orient ICP
    primary/              # thesis default: exp14 identity, Surflo gt_anchored, NOVA robust_rot_search
      {model}.ply         # full aligned cloud
      {model}_core.ply    # robust ICP subset (NOVA only)
      {model}_filtered.ply
```

### Alignment methods (default: `gt_anchored` + `robust_filter`)

| Name | Use for | Idea |
|---|---|---|
| `gt_anchored` | **Surflo** | Free-scale ICP on raw cloud; strict export filter (5% GT diag) |
| `robust_filter` | debug / NOVA without rot search | Densify → normalize-both → orient ICP (identity only) |
| `robust_rot_search` | **NOVA** (in `primary`) | Densify → multi-orient (incl. 90° Z) → core + **full** ICP; best by **full CD** |
| `primary/` | **Compare** | exp14 identity · Surflo `gt_anchored` · NOVA `robust_rot_search` |

**NOVA exports:** `*_filtered` uses a looser crop (prefer thin geometry over tight shell). `*_core` is densified-only (no post-align GT trim). Full `*.ply` keeps all raw points.

Global: `results_by_align.json`, `results.json` (primary folder stats). Subset runs write `results_*_debug_*.json`.

## Metrics

- **CD_L2** / **CD_L2sq**, **accuracy** (pred→GT), **completeness** (GT→pred)
- F-scores @ 0.05 / 0.02 / 0.01
- Default scoring on `*_filtered.ply` (`--no-score_on_filtered` for full cloud)

## Notes

- Exp14 uses the **30k** checkpoint by default (`--identity_exp14`).
- Images are **letterboxed** to 518×392 (no anisotropic stretch).
- Transfer comparison: scene models on object-centric white-bg data.
