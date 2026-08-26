#!/bin/bash
# Generic resubmit-on-preemption babysitter for the R11 ablation training arms
# (modeled on trigger_uesfm_eval.sh, WITHOUT the eval fan-out: the shallow arms
# need a shallow eval conf template, built when the grid is evaluated).
#
# Usage (armed via bsub -w "ended(<train_job>)"):
#   trigger_ablation_resume.sh <conf_rel_path> <exp_version> <results_dir_rel> [retry]
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
CONF="$1"; EXP="$2"; RESULTS_REL="$3"; RETRY_COUNT=${4:-0}; PREV_JID=${5:-}
# Queue policy (2026-08-26): default resubmission goes to waic-medium, but a job
# that died on the medium queue's 4h RUNLIMIT is relaunched on waic-risk (no
# run limit) so long stretches can complete there when risk has slots.
RESUB_QUEUE="waic-medium"
if [ -n "$PREV_JID" ] && bhist -l "$PREV_JID" 2>/dev/null | grep -q "TERM_RUNLIMIT"; then
    echo "TRIGGER[${EXP}]: previous job ${PREV_JID} hit RUNLIMIT -> resubmitting on waic-risk."
    RESUB_QUEUE="waic-risk"
fi
MODELS="${REPO_ROOT}/${RESULTS_REL}/models"
# True completion marker: the final-epoch checkpoint. The best-val checkpoint in
# models/ can be EARLY (validation plateaus before epoch 19000), which otherwise
# makes the epoch check below fail forever -> infinite resubmit loop. Check the
# final marker first and stop unconditionally if training actually reached 20k.
if [ -e "${REPO_ROOT}/${RESULTS_REL}/models_all/Model_Ep19999.pt" ]; then
    echo "TRIGGER[${EXP}]: training complete (models_all/Model_Ep19999.pt present). Not resubmitting."
    exit 0
fi
TARGET_EPOCH=19000  # 20000-epoch run; require the best checkpoint to be late-stage
MAX_RETRIES=10
PY=/home/projects/bagon/ortalda/MVG/final-project/u-esfm/.venv38-resfm/bin/python

CKPT=$(ls -1v "${MODELS}"/Model_Ep*.pt 2>/dev/null | tail -1)
EPOCH=0
[ -n "$CKPT" ] && EPOCH=$(basename "$CKPT" | sed -E 's/Model_Ep([0-9]+)\.pt/\1/')

if [ "$EPOCH" -ge "$TARGET_EPOCH" ]; then
    echo "TRIGGER[${EXP}]: training complete (best checkpoint ${CKPT}). Evaluate with a matching eval conf (see RUN_MULTISCENE.md, R11)."
    exit 0
fi

if [ "$RETRY_COUNT" -ge "$MAX_RETRIES" ]; then
    echo "TRIGGER[${EXP}]: still incomplete (epoch ${EPOCH} < ${TARGET_EPOCH}) after ${MAX_RETRIES} resubmissions — giving up. Investigate and re-arm manually."
    exit 1
fi

echo "TRIGGER[${EXP}]: incomplete (epoch ${EPOCH} < ${TARGET_EPOCH}); resubmitting (retry $((RETRY_COUNT+1))/${MAX_RETRIES})."
SUB=$(bsub -q "${RESUB_QUEUE}" -J "${EXP}" \
    -oo "${REPO_ROOT}/lsf_output/multiscene/${EXP}_resume_%J.out" \
    -eo "${REPO_ROOT}/lsf_output/multiscene/${EXP}_resume_%J.err" \
    -gpu "num=1:j_exclusive=yes:gmem=80G" -R "rusage[mem=64000]" \
    "cd ${REPO_ROOT}; ${PY} multiple_scenes_learning.py --conf ${CONF} --phase TRAINING --exp_version ${EXP} --wandb 1")
echo "$SUB"
NEWID=$(echo "$SUB" | grep -oE "Job <[0-9]+>" | grep -oE "[0-9]+")
if [ -z "$NEWID" ]; then
    echo "TRIGGER[${EXP}]: failed to resubmit training — giving up."
    exit 1
fi
bsub -q waic-medium -gpu "num=1:j_exclusive=yes:gmem=80G" -J "${EXP}_trigger" -w "ended(${NEWID})" \
    -oo "${REPO_ROOT}/lsf_output/multiscene/${EXP}_trigger_%J.out" \
    -eo "${REPO_ROOT}/lsf_output/multiscene/${EXP}_trigger_%J.err" \
    "bash ${REPO_ROOT}/trigger_ablation_resume.sh ${CONF} ${EXP} ${RESULTS_REL} $((RETRY_COUNT+1)) ${NEWID}"
echo "TRIGGER[${EXP}]: re-armed on training job ${NEWID}."
