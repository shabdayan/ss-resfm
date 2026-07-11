#!/bin/bash
# Generic resubmit-on-preemption babysitter for the R11 ablation training arms
# (modeled on trigger_uesfm_eval.sh, WITHOUT the eval fan-out: the shallow arms
# need a shallow eval conf template, built when the grid is evaluated).
#
# Usage (armed via bsub -w "ended(<train_job>)"):
#   trigger_ablation_resume.sh <conf_rel_path> <exp_version> <results_dir_rel> [retry]
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
CONF="$1"; EXP="$2"; RESULTS_REL="$3"; RETRY_COUNT=${4:-0}
MODELS="${REPO_ROOT}/${RESULTS_REL}/models"
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
SUB=$(bsub -q waic-risk -J "${EXP}" \
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
bsub -q waic-risk -gpu "num=1:j_exclusive=yes:gmem=80G" -J "${EXP}_trigger" -w "ended(${NEWID})" \
    -oo "${REPO_ROOT}/lsf_output/multiscene/${EXP}_trigger_%J.out" \
    -eo "${REPO_ROOT}/lsf_output/multiscene/${EXP}_trigger_%J.err" \
    "bash ${REPO_ROOT}/trigger_ablation_resume.sh ${CONF} ${EXP} ${RESULTS_REL} $((RETRY_COUNT+1))"
echo "TRIGGER[${EXP}]: re-armed on training job ${NEWID}."
