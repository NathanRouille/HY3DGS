#!/usr/bin/env bash
# One-shot overnight preprocess: all InternScenes gen__* rooms → surface.npz
#
# Usage (inside tmux on the workstation):
#   conda activate hy3dgs
#   cd ~/Documents/research/HY3DGS
#   bash run_internscenes_gen_preprocess.sh
#
# Resume-safe: re-run the same command; existing surface.npz are skipped.
# Does NOT copy GLBs. Does NOT write preview.ply. Read-only on Ismail.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

# ---------- knobs (12h target on 48-thread / 500GB box) ----------
OUT="${OUT:-$HOME/Documents/research/internscenes_gen_pc_v0}"
MESH_ROOT="${MESH_ROOT:-/mnt/hdd2/ismail/InternScenes/processed/glb_output/sample_scenes}"
IS_BOTH="${IS_BOTH:-$HOME/Documents/research/pilot_scene_pc_v0/lists/is_both.txt}"
N_LIGHT="${N_LIGHT:-20}"
N_KITCHEN="${N_KITCHEN:-6}"
N_TOTAL="${N_TOTAL:-65536}"
N_SHARP="${N_SHARP:-32768}"
SEED="${SEED:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SMOKE="${SMOKE:-0}"   # set SMOKE=1 to process only 2 rooms then exit

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1

mkdir -p "$OUT"/{lists,splits,rooms,logs,shards}

echo "============================================================"
echo "InternScenes gen PC preprocess"
echo "  OUT         = $OUT"
echo "  MESH_ROOT   = $MESH_ROOT"
echo "  workers     = light x$N_LIGHT + kitchen x$N_KITCHEN"
echo "  n_total     = $N_TOTAL  sharp_target=$N_SHARP  seed=$SEED"
echo "  started     = $(date -Is)"
echo "============================================================"

if [[ ! -d "$MESH_ROOT" ]]; then
  echo "ERROR: mesh_root missing: $MESH_ROOT" >&2
  exit 2
fi
if [[ ! -f "$IS_BOTH" ]]; then
  echo "ERROR: is_both list missing: $IS_BOTH" >&2
  echo "  (build pilot lists first, or set IS_BOTH=...)" >&2
  exit 2
fi

# ---------- lists ----------
grep '^gen__' "$IS_BOTH" | sort -u > "$OUT/lists/gen_all.txt"
grep '^gen__kitchen__' "$OUT/lists/gen_all.txt" > "$OUT/lists/gen_kitchen.txt"
grep -v '^gen__kitchen__' "$OUT/lists/gen_all.txt" > "$OUT/lists/gen_light.txt"
wc -l "$OUT/lists/gen_all.txt" "$OUT/lists/gen_light.txt" "$OUT/lists/gen_kitchen.txt"

if [[ "$SMOKE" == "1" ]]; then
  head -1 "$OUT/lists/gen_light.txt" > "$OUT/lists/smoke.txt"
  head -1 "$OUT/lists/gen_kitchen.txt" >> "$OUT/lists/smoke.txt"
  echo "SMOKE mode: only $(wc -l < "$OUT/lists/smoke.txt") rooms"
  "$PYTHON_BIN" "$ROOT/sample_internscenes_pilot.py" \
    --mesh_root "$MESH_ROOT" \
    --out_root "$OUT" \
    --list_file "$OUT/lists/smoke.txt" \
    --n_total "$N_TOTAL" \
    --n_sharp_target "$N_SHARP" \
    --seed "$SEED" \
    --no_ply
  echo "SMOKE done. Inspect $OUT/rooms/ then re-run without SMOKE=1"
  exit 0
fi

# ---------- provenance ----------
cat > "$OUT/provenance.json" <<EOF
{
  "dataset": "internscenes_gen",
  "mesh_root": "$MESH_ROOT",
  "is_both": "$IS_BOTH",
  "n_total": $N_TOTAL,
  "n_sharp_target": $N_SHARP,
  "seed": $SEED,
  "no_ply": true,
  "workers_light": $N_LIGHT,
  "workers_kitchen": $N_KITCHEN,
  "thread_env": "OMP/MKL/OPENBLAS/NUMEXPR/TORCH_NUM_THREADS=1",
  "script": "$ROOT/sample_internscenes_pilot.py",
  "started": "$(date -Is)"
}
EOF

# ---------- shards (refresh each run; resume is via existing npz) ----------
rm -f "$OUT/shards"/light_shard_* "$OUT/shards"/kitchen_shard_*
split -n "l/$N_LIGHT" -d -a 2 "$OUT/lists/gen_light.txt" "$OUT/shards/light_shard_"
split -n "l/$N_KITCHEN" -d -a 2 "$OUT/lists/gen_kitchen.txt" "$OUT/shards/kitchen_shard_"

echo "Shards:"
ls "$OUT/shards" | wc -l

run_shard () {
  local list=$1
  local tag=$2
  local logfile="$OUT/logs/${tag}.log"
  echo "[$(date -Is)] START $tag ($(wc -l < "$list") ids) -> $logfile"
  "$PYTHON_BIN" "$ROOT/sample_internscenes_pilot.py" \
    --mesh_root "$MESH_ROOT" \
    --out_root "$OUT" \
    --list_file "$list" \
    --n_total "$N_TOTAL" \
    --n_sharp_target "$N_SHARP" \
    --seed "$SEED" \
    --no_ply \
    > "$logfile" 2>&1
  local rc=$?
  echo "[$(date -Is)] END $tag rc=$rc" | tee -a "$logfile"
  return $rc
}

PIDS=()
TAGS=()
FAIL_SHARDS=0

for f in "$OUT"/shards/light_shard_* "$OUT"/shards/kitchen_shard_*; do
  [[ -f "$f" ]] || continue
  tag=$(basename "$f")
  run_shard "$f" "$tag" &
  PIDS+=($!)
  TAGS+=("$tag")
done

echo "Launched ${#PIDS[@]} workers. Waiting..."
echo "  monitor:  watch -n 60 'find $OUT/rooms -name surface.npz | wc -l'"
echo "  logs:     tail -f $OUT/logs/light_shard_00.log"

for i in "${!PIDS[@]}"; do
  pid=${PIDS[$i]}
  tag=${TAGS[$i]}
  if wait "$pid"; then
    echo "OK worker $tag"
  else
    echo "FAIL worker $tag (see $OUT/logs/${tag}.log)" >&2
    FAIL_SHARDS=$((FAIL_SHARDS + 1))
  fi
done

# ---------- summarize ----------
DONE_LIST="$OUT/lists/gen_done.txt"
find "$OUT/rooms" -mindepth 2 -maxdepth 2 -name surface.npz -printf '%h\n' \
  | sed 's|.*/||' | sort -u > "$DONE_LIST"

ALL_N=$(wc -l < "$OUT/lists/gen_all.txt")
DONE_N=$(wc -l < "$DONE_LIST")
MISS_LIST="$OUT/lists/gen_missing.txt"
comm -23 "$OUT/lists/gen_all.txt" "$DONE_LIST" > "$MISS_LIST"
MISS_N=$(wc -l < "$MISS_LIST")

echo "------------------------------------------------------------"
echo "Finished sampling: done=$DONE_N / $ALL_N   missing=$MISS_N   failed_shards=$FAIL_SHARDS"
echo "  done list:    $DONE_LIST"
echo "  missing list: $MISS_LIST"
du -sh "$OUT" || true
df -h "$OUT" | tail -1 || true

# ---------- stratified splits 90/5/5 ----------
export OUT
"$PYTHON_BIN" - <<'PY'
import hashlib
from collections import defaultdict
from pathlib import Path
import os

out = Path(os.environ["OUT"])
done = [ln.strip() for ln in (out / "lists" / "gen_done.txt").read_text().splitlines() if ln.strip()]

def room_type(sid: str) -> str:
    parts = sid.split("__")
    return parts[1] if len(parts) >= 2 else "unknown"

by_type = defaultdict(list)
for sid in done:
    by_type[room_type(sid)].append(sid)

train, val, test = [], [], []
for t, ids in sorted(by_type.items()):
    ids = sorted(ids, key=lambda s: hashlib.md5(f"split0:{s}".encode()).hexdigest())
    n = len(ids)
    n_test = max(1, int(round(n * 0.05))) if n >= 20 else max(0, n // 20)
    n_val = max(1, int(round(n * 0.05))) if n >= 20 else max(0, n // 20)
    if n_test + n_val >= n:
        n_test = min(1, n // 10)
        n_val = min(1, max(0, n // 10))
    test.extend(ids[:n_test])
    val.extend(ids[n_test : n_test + n_val])
    train.extend(ids[n_test + n_val :])

splits = out / "splits"
splits.mkdir(parents=True, exist_ok=True)
for name, rows in ("train", train), ("val", val), ("test", test):
    p = splits / f"{name}.txt"
    p.write_text("\n".join(sorted(rows)) + ("\n" if rows else ""))
    print(f"split {name}: {len(rows)} -> {p}")
PY

# write finished stamp into provenance
"$PYTHON_BIN" - <<PY
import json
from pathlib import Path
from datetime import datetime, timezone
p = Path("$OUT") / "provenance.json"
data = json.loads(p.read_text())
data["finished"] = datetime.now(timezone.utc).astimezone().isoformat()
data["n_done"] = int(Path("$DONE_LIST").read_text().count("\n"))
data["n_missing"] = int(Path("$MISS_LIST").read_text().count("\n") if Path("$MISS_LIST").stat().st_size else 0)
data["failed_shards"] = int("$FAIL_SHARDS")
p.write_text(json.dumps(data, indent=2) + "\n")
print("updated", p)
PY

echo "============================================================"
echo "ALL DONE at $(date -Is)"
echo "  rooms:   $OUT/rooms/<id>/surface.npz"
echo "  splits:  $OUT/splits/{train,val,test}.txt"
echo "  missing: $MISS_LIST ($MISS_N)"
if [[ "$MISS_N" -gt 0 || "$FAIL_SHARDS" -gt 0 ]]; then
  echo "  WARNING: some rooms/shards failed — re-run this script to retry missing only."
  exit 1
fi
exit 0
