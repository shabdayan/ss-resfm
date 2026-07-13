#!/bin/bash
# Submit the three track builders as LSF jobs (one per dataset).
# Usage: ./run_build_tracks.sh [strecha|blendedmvs|1dsfm ...]  (default: all)
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
PY="$DIR/../../.venv/bin/python"
LOGS="$DIR/../lsf_output/build_tracks"
mkdir -p "$LOGS"
QUEUE=waic-risk
GPU='num=1:j_exclusive=yes:gmem=80G'

for ds in "${@:-strecha blendedmvs 1dsfm}"; do
  for d in $ds; do
    bsub -q "$QUEUE" -gpu "$GPU" -R 'rusage[mem=64000]' \
         -J "tracks_$d" -o "$LOGS/$d.%J.out" -e "$LOGS/$d.%J.err" \
         "$PY $DIR/build_$d.py"
  done
done
