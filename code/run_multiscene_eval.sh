#!/bin/bash
# Per-test-scene evaluation of the U-RESfM multi-scene model (RESfM protocol:
# per-scene ~1K-epoch fine-tune + BA evaluation) over the 36 reconstructed
# MegaDepth test scenes (see RUN_MULTISCENE.md).
#
# For each scene this generates a conf from
# confs/multiscene_uresfm_eval.conf.template (own results_path per scene — a
# shared one would mix per-scene fine-tune checkpoints) and submits one LSF job
# running single_scene_optimization.py --phase FINE_TUNE. The multi-scene best
# checkpoint is injected via the conf's pretrainedPath.
#
# Usage:
#   ./run_multiscene_eval.sh [--queue waic-short] [--scans "0238,0060"] [--dry_run]
#
# ⚠️ With output_mode=3 in the template, the FINE_TUNE data loader expects
# predicted outliers saved by a prior TEST evaluation — not produced yet (see
# RUN_MULTISCENE.md "TTT compatibility"). Run this only after that gap is fixed,
# or set output_mode=1 in the template to fine-tune without outlier pruning.

set -e
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
TRAIN_RESULTS="${REPO_ROOT}/results/multiscene/uresfm_27scenes"
EVAL_ROOT="${REPO_ROOT}/results/multiscene/uresfm_27scenes_eval"
TEMPLATE="${REPO_ROOT}/confs/multiscene_uresfm_eval.conf.template"
CONF_DIR="${REPO_ROOT}/confs/multiscene_eval_generated"
PY="${REPO_ROOT}/../.venv/bin/python"

QUEUE="waic-short"
DRY_RUN=false
SCENES="0238 0060 0197 0094 0265 0083 0076 0185 0048 0024 0223 5016 0046 0099 1001 0231 0411 0377 0102 0147 0148 0446 0022 0327 0015 0455 0496 1589 0012 0104 0019 0063 0130 0080 0240 0007"

while [[ $# -gt 0 ]]; do
    case $1 in
        --queue) QUEUE="$2"; shift 2;;
        --scans) SCENES="${2//,/ }"; shift 2;;
        --dry_run) DRY_RUN=true; shift;;
        *) echo "Unknown option $1"; exit 1;;
    esac
done

# Latest best checkpoint from the multi-scene training run
CKPT=$(ls -1v "${TRAIN_RESULTS}/models"/Model_Ep*.pt 2>/dev/null | tail -1)
if [ -z "$CKPT" ]; then
    echo "ERROR: no multi-scene checkpoint found under ${TRAIN_RESULTS}/models/"
    echo "Run the 27-scene training first (see RUN_MULTISCENE.md)."
    exit 1
fi
echo "Using multi-scene checkpoint: ${CKPT}"

mkdir -p "${CONF_DIR}" "${EVAL_ROOT}" "${REPO_ROOT}/lsf_output/multiscene_eval"

for SCAN in $SCENES; do
    CONF="${CONF_DIR}/${SCAN}.conf"
    RESULTS_PATH="${EVAL_ROOT}/${SCAN}_ba"
    sed -e "s|__SCAN__|${SCAN}|g" \
        -e "s|__RESULTS_PATH__|${RESULTS_PATH}|g" \
        -e "s|__PRETRAINED__|${CKPT}|g" \
        "${TEMPLATE}" > "${CONF}"
    echo "Generated ${CONF}"

    CMD="cd ${REPO_ROOT}; ${PY} single_scene_optimization.py \
        --conf ${CONF} \
        --scan ${SCAN} \
        --stage 1 \
        --architecture_type esfm_outliers_deep \
        --phase FINE_TUNE \
        --exp_version multiscene_eval \
        --results_aggregation_file ${EVAL_ROOT}/Aggregated_eval_results.xlsx \
        --wandb 0"

    if [ "$DRY_RUN" = true ]; then
        echo "DRY RUN: ${CMD}"
    else
        bsub -q "${QUEUE}" \
            -J "ueval_${SCAN}" \
            -oo "${REPO_ROOT}/lsf_output/multiscene_eval/${SCAN}_%J.out" \
            -eo "${REPO_ROOT}/lsf_output/multiscene_eval/${SCAN}_%J.err" \
            -gpu "num=1:j_exclusive=yes:gmem=40G" \
            -R "rusage[mem=50000]" \
            "${CMD}"
    fi
done

echo ""
echo "Submitted $(echo $SCENES | wc -w) evaluation jobs to ${QUEUE}."
echo "Results land under: ${EVAL_ROOT}/<scan>_ba/"
echo "Aggregated table:   ${EVAL_ROOT}/Aggregated_eval_results.xlsx"
