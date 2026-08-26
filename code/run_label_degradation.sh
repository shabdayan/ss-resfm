#!/bin/bash
# Label-degradation curve: run the per-scene COLMAP-label-vs-GT-truth measurement
# across the contamination axis and collect JSONOUT lines. COLMAP is CPU-only, so
# scenes run sequentially in one job. Caps cams at 120 so high-cam scenes stay fast.
# RESUMABLE: skips scenes already in results.jsonl (the queue requeues this job
# freely, so every restart must not redo finished scenes). Per-scene logs avoid
# the global tail-1 grep picking up a previous scene's line on timeout.
set -u
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO"
PY="$REPO/../.venv38-resfm/bin/python"
OUT="$REPO/logs_labeldeg"; mkdir -p "$OUT"
RES="$OUT/results.jsonl"; touch "$RES"        # append-only across restarts

# dataset scene  (chosen to span contamination: strecha/bmvs low, megadepth mid, 1dsfm/hard high)
SCENES=(
 "strecha entry-P10" "strecha fountain-P11" "strecha Herz-Jesu-P25" "strecha Herz-Jesu-P8"
 "blendedmvs 58c4bb4f4a69c55606122be4" "blendedmvs 5be883a4f98cee15019d5b83"
 "blendedmvs 5b6e716d67b396324c2d77cb" "blendedmvs 5bfd0f32ec61ca1dd69dc77b"
 "blendedmvs 5b950c71608de421b1e7318f"
 "megadepth 0007" "megadepth 0047" "megadepth 0041" "megadepth 0060"
 "1dsfm Alamo" "1dsfm Ellis_Island" "1dsfm Madrid_Metropolis" "1dsfm Piazza_del_Popolo"
 "1dsfm_hard_300 Gendarmenmarkt"
)
for pair in "${SCENES[@]}"; do
  set -- $pair; ds="$1"; sc="$2"
  grep -q "\"dataset\": \"$ds\", \"scene\": \"$sc\"" "$RES" && { echo "SKIP $ds/$sc (done)" >> "$OUT/run.log"; continue; }
  echo "=== $(date '+%T') $ds/$sc ===" >> "$OUT/run.log"
  slog="$OUT/scene_${ds}_${sc}.log"
  timeout 2400 "$PY" measure_label_degradation.py "$ds" "$sc" --max_cams 120 \
      > "$slog" 2>&1
  tail -20 "$slog" >> "$OUT/run.log"
  grep "^JSONOUT:" "$slog" | tail -1 | sed 's/^JSONOUT://' >> "$RES"
done
echo "DONE $(date '+%T')" >> "$OUT/run.log"
