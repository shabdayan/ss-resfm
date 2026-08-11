#!/bin/bash
# Resubmit-on-preemption babysitter for the resfm_deep 1DSfM straggler cells:
# Notre_Dame + Vienna_Cathedral (all 5 seeds) — the two largest 1DSfM scenes,
# which keep getting preempted on waic-risk before their classifier-prune +
# 1K fine-tune + BA finishes (eval jobs don't checkpoint-resume, so a preempt
# just means restart-from-scratch next time).
#
# One pass: for each target cell, resubmit iff (a) its Results_FINE_TUNE xlsx is
# missing AND (b) no ueval job for that scene+seed is currently PEND/RUN. Then
# re-arm itself ~30 min later via `bsub -b <begin-time>`, unless all cells are
# done (exit 0) or MAX_RETRIES is hit (exit 1).
#
# Name-based in-progress check is safe here: esfm_vanilla 1DSfM is already 50/50,
# so any live ueval_s<seed>_{Notre_Dame,Vienna_Cathedral} is necessarily resfm_deep.
#
# Usage (armed automatically; manual first pass): bash babysit_resfm_deep_stragglers.sh [retry]
set -u
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"
PY38="$REPO_ROOT/../.venv38-resfm/bin/python"
RETRY=${1:-0}
MAX_RETRIES=40                         # ~20h at 30-min cadence
SCENES="Notre_Dame Vienna_Cathedral"
SEEDS="20 21 22 23 24"
TEMPLATE="confs/crossdataset_resfm_deep_1dsfm.conf.template"
TRAIN="results/multiscene/resfm_deep_27scenes"
EVAL_ROOT="results/crossdataset/resfm_deep_1dsfm_eval"

# scene+seed of every ueval job currently PEND or RUN (bjobs shows both)
RUNNING=$(bjobs -w 2>/dev/null | grep ueval | awk '{print $7}' | sed 's/ueval_s//')

missing=0; resub=0
for seed in $SEEDS; do
    base="$EVAL_ROOT"; [ "$seed" -ne 20 ] && base="${base}_seed${seed}"
    for sc in $SCENES; do
        if ls "$base/${sc}_ba/Results_FINE_TUNE"*.xlsx >/dev/null 2>&1; then
            continue                                   # done
        fi
        missing=$((missing+1))
        if echo "$RUNNING" | grep -qx "${seed}_${sc}"; then
            echo "in-progress ${seed}_${sc}"; continue # already queued/running
        fi
        EVAL_PYTHON="$PY38" ./run_multiscene_eval.sh \
            --template "$TEMPLATE" --train_results "$TRAIN" \
            --eval_root "$EVAL_ROOT" --arch esfm_outliers_deep \
            --scans "$sc" --seed "$seed" >/dev/null 2>&1 \
            && { resub=$((resub+1)); echo "resubmitted ${seed}_${sc}"; }
    done
done
echo "pass ${RETRY}: missing=${missing} resubmitted=${resub}"

if [ "$missing" -eq 0 ]; then
    echo "ALL STRAGGLERS COMPLETE — babysitter exiting."
    exit 0
fi
if [ "$RETRY" -ge "$MAX_RETRIES" ]; then
    echo "gave up after ${MAX_RETRIES} passes; ${missing} cell(s) still missing."
    exit 1
fi

NEXT=$(date -d '+30 minutes' +%Y:%m:%d:%H:%M)
bsub -q waic-risk -J resfm_deep_straggler_bs -b "$NEXT" \
    -gpu "num=1:j_exclusive=yes:gmem=80G" \
    -oo "${REPO_ROOT}/lsf_output/multiscene_eval/straggler_bs_%J.out" \
    -eo "${REPO_ROOT}/lsf_output/multiscene_eval/straggler_bs_%J.err" \
    "bash ${REPO_ROOT}/babysit_resfm_deep_stragglers.sh $((RETRY+1))"
echo "re-armed pass $((RETRY+1)) at ${NEXT}"
