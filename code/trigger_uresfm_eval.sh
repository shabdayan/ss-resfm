#!/bin/bash
# Babysitter for the 27-scene U-RESfM training on the preemptible waic-risk queue.
# Submitted with `bsub -w "ended(<train_job>)" ... trigger_uresfm_eval.sh <retry>`:
# fires whenever the training job leaves the queue, then either
#   - training complete (best checkpoint epoch >= TARGET_EPOCH): fan out the
#     36-scene per-scene fine-tune evaluation for seeds 20-24, or
#   - training incomplete (preempted/crashed): resubmit training (the conf has
#     resume = true, so it continues from the latest models_all checkpoint) and
#     re-arm this trigger on the new job, up to MAX_RETRIES times.
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
MODELS="${REPO_ROOT}/results/multiscene/uresfm_27scenes_lr1e4/models"
TARGET_EPOCH=19000  # 20000-epoch run; require the best checkpoint to be late-stage
RETRY_COUNT=${1:-0}
MAX_RETRIES=10
PY=/home/projects/bagon/ortalda/MVG/final-project/u-resfm/.venv38-resfm/bin/python

CKPT=$(ls -1v "${MODELS}"/Model_Ep*.pt 2>/dev/null | tail -1)
EPOCH=0
[ -n "$CKPT" ] && EPOCH=$(basename "$CKPT" | sed -E 's/Model_Ep([0-9]+)\.pt/\1/')

if [ "$EPOCH" -ge "$TARGET_EPOCH" ]; then
    echo "TRIGGER: training complete (best checkpoint ${CKPT}); fanning out 36-scene evaluation."
    # Seed 20 = paper-protocol run; seeds 21-24 complete the 5-seed median protocol.
    for SEED in 20 21 22 23 24; do
        EVAL_PYTHON="$PY" "${REPO_ROOT}/run_multiscene_eval.sh" --queue waic-short --seed "$SEED"
    done
    exit 0
fi

if [ "$RETRY_COUNT" -ge "$MAX_RETRIES" ]; then
    echo "TRIGGER: training still incomplete (epoch ${EPOCH} < ${TARGET_EPOCH}) after ${MAX_RETRIES} resubmissions — giving up. Investigate and re-arm manually."
    exit 1
fi

echo "TRIGGER: training incomplete (epoch ${EPOCH} < ${TARGET_EPOCH}); resubmitting (retry $((RETRY_COUNT+1))/${MAX_RETRIES})."
SUB=$(bsub -q waic-risk -J uresfm_ms27_lr1e4 \
    -oo "${REPO_ROOT}/lsf_output/multiscene/uresfm27_resume_%J.out" \
    -eo "${REPO_ROOT}/lsf_output/multiscene/uresfm27_resume_%J.err" \
    -gpu "num=1:j_exclusive=yes:gmem=80G" -R "rusage[mem=64000]" \
    "cd ${REPO_ROOT}; ${PY} multiple_scenes_learning.py --conf confs/multiscene_uresfm_lr1e4.conf --phase TRAINING --exp_version multiscene_uresfm_lr1e4 --wandb 1")
echo "$SUB"
NEWID=$(echo "$SUB" | grep -oE "Job <[0-9]+>" | grep -oE "[0-9]+")
if [ -z "$NEWID" ]; then
    echo "TRIGGER: failed to resubmit training — giving up."
    exit 1
fi
bsub -q waic-short -J uresfm_eval_trigger -w "ended(${NEWID})" \
    -oo "${REPO_ROOT}/lsf_output/multiscene/eval_trigger_%J.out" \
    -eo "${REPO_ROOT}/lsf_output/multiscene/eval_trigger_%J.err" \
    "bash ${REPO_ROOT}/trigger_uresfm_eval.sh $((RETRY_COUNT+1))"
echo "TRIGGER: re-armed on training job ${NEWID}."
