#!/bin/bash
# Launch the ESFM two-stage (report Sec 2.2.1: MAD/STD/Huber) across all Olsson
# scenes, seed 0, for BOTH fair-budget designs:
#   B (full)  : reuse esfm_rc 100k baseline as Stage 1, Stage 2 = 100k  -> label "full"
#   A (eqbud) : fresh Stage 1 = 50k, Stage 2 = 50k (total 100k == ESFM)  -> label "eqbud"
# One bsub job per (scene, design); each job runs all 3 methods at the report's
# reported-best deltas (mad=2, std=3, huber=1) and, for design A, the shared 50k
# Stage 1 once. Usage:  ./launch_esfm_twostage.sh {A|B|both}
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
LOGDIR=/tmp/claude-53387/-home-projects-bagon-ortalda-MVG-final-project-u-esfm/91da1084-3c11-4cd8-a3f6-dbc5d23634a9/scratchpad/esfm_ts_logs
mkdir -p "$LOGDIR"

which=${1:-both}
mapfile -t SLUGS < <($PY -c "from run_single_scene_sweep import OLSSON_SCENES, slug; print('\n'.join(slug(s) for s in OLSSON_SCENES))")

submit() {  # $1=scene_slug  $2=design
  local s=$1 d=$2 args jname
  if [ "$d" = "B" ]; then
    args="--stage1-source esfm_rc --stage2-epochs 100000 --label full"
    jname="ets_B_${s}"
  else
    args="--stage1-source fresh --stage1-epochs 50000 --stage2-epochs 50000 --label eqbud"
    jname="ets_A_${s}"
  fi
  bsub -q waic-risk -gpu "num=1:j_exclusive=yes:gmem=80G" \
       -R "select[hname!='lgn15']" -J "$jname" -o "$LOGDIR/${jname}.out" \
       $PY esfm_twostage.py --scene "$s" --seeds 0 --methods mad,std,huber $args
}

for s in "${SLUGS[@]}"; do
  [ "$which" = "B" ] || [ "$which" = "both" ] && submit "$s" B || true
  [ "$which" = "A" ] || [ "$which" = "both" ] && submit "$s" A || true
done
echo "submitted design=$which for ${#SLUGS[@]} scenes (logs in $LOGDIR)"
