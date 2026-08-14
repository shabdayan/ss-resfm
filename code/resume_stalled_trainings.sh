#!/bin/bash
# Self-heal for the long from-scratch trainings that keep falling out of the
# waic-risk queue (heavy cross-project contention + babysitter retry cap).
# For each training: if it is NOT complete (no models_all/Model_Ep19999.pt) AND
# has no job in the queue, resume it (resume=true continues from its latest
# checkpoint) with a fresh babysitter. Idempotent per cycle.
set -u
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO"
PY="$REPO/../.venv38-resfm/bin/python"
QNAMES=$(bjobs -w 2>/dev/null | awk '{print $7}')

# exp | conf | resdir
JOBS=(
 "resfm_shallow_ms27|confs/multiscene_resfm_shallow.conf|resfm_shallow_27scenes"
 "uesfm_shallow_adaptive_p1090|confs/multiscene_uesfm_shallow_adaptive_p1090.conf|uesfm_27scenes_shallow_adaptive_p1090"
 "uesfm_shallow_adaptive_p4060|confs/multiscene_uesfm_shallow_adaptive_p4060.conf|uesfm_27scenes_shallow_adaptive_p4060"
)
for row in "${JOBS[@]}"; do
  IFS='|' read -r exp conf resdir <<< "$row"
  [ -e "results/multiscene/$resdir/models_all/Model_Ep19999.pt" ] && continue   # done
  echo "$QNAMES" | grep -qx "$exp" && continue                                    # already queued/running
  SUB=$(bsub -q waic-risk -J "$exp" -oo "lsf_output/multiscene/${exp}_%J.out" -eo "lsf_output/multiscene/${exp}_%J.err" \
    -gpu "num=1:j_exclusive=yes:gmem=80G" -R "rusage[mem=64000]" \
    "cd $REPO; TORCHDYNAMO_DISABLE=1 $PY multiple_scenes_learning.py --conf $conf --phase TRAINING --exp_version $exp --wandb 1")
  JID=$(echo "$SUB" | grep -oE "Job <[0-9]+>" | grep -oE "[0-9]+")
  [ -z "$JID" ] && { echo "resume $exp: FAILED to submit"; continue; }
  bsub -q waic-risk -gpu "num=1:j_exclusive=yes:gmem=80G" -J "${exp}_trigger" -w "ended(${JID})" \
    -oo "lsf_output/multiscene/${exp}_trigger_%J.out" -eo "lsf_output/multiscene/${exp}_trigger_%J.err" \
    "bash $REPO/trigger_ablation_resume.sh $conf $exp results/multiscene/$resdir 0" >/dev/null
  echo "resume $exp -> $JID (+fresh babysitter)"
done
