#!/bin/bash
# Submitted with `bsub -w "ended(<train_job>)"`: fires when the 27-scene U-RESfM
# training leaves the queue, verifies it actually finished (waic-risk jobs can be
# preempted), then fans out the 36-scene per-scene fine-tune evaluation in the
# upstream-matched environment.
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
MODELS="${REPO_ROOT}/results/multiscene/uresfm_27scenes/models"
TARGET_EPOCH=19000  # 20000-epoch run; require the best checkpoint to be late-stage

CKPT=$(ls -1v "${MODELS}"/Model_Ep*.pt 2>/dev/null | tail -1)
if [ -z "$CKPT" ]; then
    echo "TRIGGER: no checkpoint found under ${MODELS} — training produced nothing; not fanning out."
    exit 1
fi
EPOCH=$(basename "$CKPT" | sed -E 's/Model_Ep([0-9]+)\.pt/\1/')
if [ "$EPOCH" -lt "$TARGET_EPOCH" ]; then
    echo "TRIGGER: latest best checkpoint is epoch ${EPOCH} (< ${TARGET_EPOCH}) — training looks incomplete (preempted?); not fanning out."
    echo "Resume training or rerun manually: EVAL_PYTHON=.../.venv38-resfm/bin/python ./run_multiscene_eval.sh"
    exit 1
fi

echo "TRIGGER: training complete (best checkpoint ${CKPT}); fanning out 36-scene evaluation."
# Seed 20 = paper-protocol run; seeds 21-24 complete the 5-seed median protocol.
for SEED in 20 21 22 23 24; do
    EVAL_PYTHON=/home/projects/bagon/ortalda/MVG/final-project/u-resfm/.venv38-resfm/bin/python \
        "${REPO_ROOT}/run_multiscene_eval.sh" --queue waic-short --seed "$SEED"
done
