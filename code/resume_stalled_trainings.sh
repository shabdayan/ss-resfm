#!/bin/bash
# Self-heal for the long from-scratch trainings that keep falling out of the
# waic-risk queue (heavy cross-project contention + babysitter retry cap).
# For each training: if it is NOT complete (no models_all/Model_Ep19999.pt) AND
# has no job in the queue, resume it (resume=true continues from its latest
# checkpoint) with a fresh babysitter. Idempotent per cycle.
set -u
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO"
PY="$REPO/../.venv38-resfm/bin/python"
QUEUE="waic-medium"   # was waic-risk; medium won the 2026-08-26 dual-queue race 94-0
QNAMES=$(bjobs -w 2>/dev/null | awk '{print $7}')
# Congestion guard: under heavy load bjobs can return nothing even though jobs
# exist; treating that as "empty queue" caused mass duplicate resubmission
# (2026-08-26). If we list zero jobs but LSF also can't confirm the queue is
# reachable, skip this cycle instead of resubmitting everything.
if [ -z "$QNAMES" ]; then
  bqueues "$QUEUE" >/dev/null 2>&1 || { echo "resume: bjobs empty and LSF unreachable -- skipping cycle"; exit 0; }
fi

# exp | conf | resdir
JOBS=(
 "resfm_shallow_ms27|confs/multiscene_resfm_shallow.conf|resfm_shallow_27scenes"
 "uesfm_shallow_adaptive_p1090|confs/multiscene_uesfm_shallow_adaptive_p1090.conf|uesfm_27scenes_shallow_adaptive_p1090"
 "uesfm_shallow_adaptive_p4060|confs/multiscene_uesfm_shallow_adaptive_p4060.conf|uesfm_27scenes_shallow_adaptive_p4060"
 "uesfm_adaptive_reportfaithful|confs/multiscene_uesfm_adaptive_reportfaithful.conf|uesfm_27scenes_adaptive_reportfaithful"
 "uesfm_shallow_adaptive_reportfaithful|confs/multiscene_uesfm_shallow_adaptive_reportfaithful.conf|uesfm_27scenes_shallow_adaptive_reportfaithful"
 "uesfm_sa_rf_p1090|confs/multiscene_uesfm_shallow_adaptive_reportfaithful_p1090.conf|uesfm_27scenes_shallow_adaptive_reportfaithful_p1090"
 "uesfm_sa_rf_p1585|confs/multiscene_uesfm_shallow_adaptive_reportfaithful_p1585.conf|uesfm_27scenes_shallow_adaptive_reportfaithful_p1585"
 "uesfm_sa_rf_p2575|confs/multiscene_uesfm_shallow_adaptive_reportfaithful_p2575.conf|uesfm_27scenes_shallow_adaptive_reportfaithful_p2575"
 "uesfm_sa_rf_p3070|confs/multiscene_uesfm_shallow_adaptive_reportfaithful_p3070.conf|uesfm_27scenes_shallow_adaptive_reportfaithful_p3070"
 "resfm_shallow_finelr|confs/multiscene_resfm_shallow_finelr.conf|resfm_shallow_27scenes_finelr"
 "uesfm_sa_rf_finelr|confs/multiscene_uesfm_shallow_adaptive_reportfaithful_finelr.conf|uesfm_27scenes_shallow_adaptive_reportfaithful_finelr"
 "uesfm_sa_rf_finelr_wd1e4|confs/multiscene_uesfm_sa_rf_finelr_wd1e4.conf|uesfm_27scenes_shallow_adaptive_reportfaithful_finelr_wd1e4"
 "uesfm_sa_rf_finelr_wd1e3|confs/multiscene_uesfm_sa_rf_finelr_wd1e3.conf|uesfm_27scenes_shallow_adaptive_reportfaithful_finelr_wd1e3"
 "uesfm_sa_rf_s21|confs/multiscene_uesfm_shallow_adaptive_reportfaithful_s21.conf|uesfm_27scenes_shallow_adaptive_reportfaithful_s21"
 "uesfm_sa_rf_s22|confs/multiscene_uesfm_shallow_adaptive_reportfaithful_s22.conf|uesfm_27scenes_shallow_adaptive_reportfaithful_s22"
 "uesfm_sa_rf_finelr_s21|confs/multiscene_uesfm_shallow_adaptive_reportfaithful_finelr_s21.conf|uesfm_27scenes_shallow_adaptive_reportfaithful_finelr_s21"
 "uesfm_sa_rf_finelr_s22|confs/multiscene_uesfm_shallow_adaptive_reportfaithful_finelr_s22.conf|uesfm_27scenes_shallow_adaptive_reportfaithful_finelr_s22"
 "uesfm_sa_rf_multids|confs/multiscene_uesfm_sa_rf_multids.conf|uesfm_multids_shallow_adaptive_reportfaithful"
 "resfm_shallow_multids|confs/multiscene_resfm_shallow_multids.conf|resfm_shallow_multids"
 "uesfm_sa_rf_multids_v2|confs/multiscene_uesfm_sa_rf_multids_v2.conf|uesfm_multids_v2_shallow_adaptive_reportfaithful"
 "resfm_shallow_multids_v2|confs/multiscene_resfm_shallow_multids_v2.conf|resfm_shallow_multids_v2"
 "uesfm_deep_multids_v2|confs/multiscene_uesfm_deep_multids_v2.conf|uesfm_multids_v2_deep_adaptive_reportfaithful"
 "resfm_deep_multids_v2|confs/multiscene_resfm_deep_multids_v2.conf|resfm_deep_multids_v2"
 "uesfm_sa_rf_100k|confs/multiscene_uesfm_sa_rf_100k.conf|uesfm_27scenes_sa_rf_100k|99999"
 "resfm_shallow_100k|confs/multiscene_resfm_shallow_100k.conf|resfm_shallow_27scenes_100k|99999"
 "uesfm_multids_v2_100k|confs/multiscene_uesfm_sa_rf_multids_v2_100k.conf|uesfm_multids_v2_sa_rf_100k|99999"
 "uesfm_multids_v2_deep_100k|confs/multiscene_uesfm_deep_multids_v2_100k.conf|uesfm_multids_v2_deep_100k|99999"
)
for row in "${JOBS[@]}"; do
  IFS='|' read -r exp conf resdir final <<< "$row"
  final=${final:-19999}                                                          # per-job final epoch (default 20k runs)
  [ -e "results/multiscene/$resdir/models_all/Model_Ep${final}.pt" ] && continue # done
  echo "$QNAMES" | grep -qx "$exp" && continue                                    # already queued/running
  SUB=$(bsub -q "$QUEUE" -J "$exp" -oo "lsf_output/multiscene/${exp}_%J.out" -eo "lsf_output/multiscene/${exp}_%J.err" \
    -gpu "num=1:j_exclusive=yes:gmem=80G" -R "rusage[mem=64000]" \
    "cd $REPO; TORCHDYNAMO_DISABLE=1 $PY multiple_scenes_learning.py --conf $conf --phase TRAINING --exp_version $exp --wandb 1")
  JID=$(echo "$SUB" | grep -oE "Job <[0-9]+>" | grep -oE "[0-9]+")
  [ -z "$JID" ] && { echo "resume $exp: FAILED to submit"; continue; }
  bsub -q "$QUEUE" -gpu "num=1:j_exclusive=yes:gmem=80G" -J "${exp}_trigger" -w "ended(${JID})" \
    -oo "lsf_output/multiscene/${exp}_trigger_%J.out" -eo "lsf_output/multiscene/${exp}_trigger_%J.err" \
    "bash $REPO/trigger_ablation_resume.sh $conf $exp results/multiscene/$resdir 0 $JID $final" >/dev/null
  echo "resume $exp -> $JID (+fresh babysitter)"
done
