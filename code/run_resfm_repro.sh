#!/bin/bash
# Reproduce RESfM paper Table 1 with the OFFICIAL pretrained checkpoint:
# per test scene, TEST-eval (saves predicted outliers) + ~1K-epoch fine-tune on
# pruned tracks + robust-BA evaluation, via single_scene_optimization.py
# --phase FINE_TUNE. See confs/resfm_repro_eval.conf.template.
#
# Usage:
#   ./run_resfm_repro.sh [--queue waic-short] [--scans "0238,5016"] [--dry_run]

set -e
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
# Override with REPRO_EVAL_ROOT / REPRO_PYTHON to run in a different environment
# (e.g. .venv38-resfm, which matches upstream RESfM's environment.yaml pins).
EVAL_ROOT="${REPRO_EVAL_ROOT:-${REPO_ROOT}/results/multiscene/resfm_repro}"
TEMPLATE="${REPO_ROOT}/confs/resfm_repro_eval.conf.template"
CONF_DIR="${REPO_ROOT}/confs/resfm_repro_generated"
PY="${REPRO_PYTHON:-${REPO_ROOT}/../.venv/bin/python}"
CKPT="${REPO_ROOT}/pretrained/pretrained_model.pt"

QUEUE="waic-short"
DRY_RUN=false
# The 36 Table-1 test scenes (13 Group-1 + 23 Group-2 subsamples; RUN_MULTISCENE.md)
SCENES="0238 0060 0197 0094 0265 0083 0076 0185 0048 0024 0223 5016 0046 0099 1001 0231 0411 0377 0102 0147 0148 0446 0022 0327 0015 0455 0496 1589 0012 0104 0019 0063 0130 0080 0240 0007"

while [[ $# -gt 0 ]]; do
    case $1 in
        --queue) QUEUE="$2"; shift 2;;
        --scans) SCENES="${2//,/ }"; shift 2;;
        --dry_run) DRY_RUN=true; shift;;
        *) echo "Unknown option $1"; exit 1;;
    esac
done

if [ ! -f "$CKPT" ]; then
    echo "ERROR: official checkpoint not found at ${CKPT}"
    exit 1
fi

mkdir -p "${CONF_DIR}" "${EVAL_ROOT}" "${REPO_ROOT}/lsf_output/resfm_repro"

for SCAN in $SCENES; do
    CONF="${CONF_DIR}/${SCAN}.conf"
    RESULTS_PATH="${EVAL_ROOT}/${SCAN}_ba"
    sed -e "s|__SCAN__|${SCAN}|g" \
        -e "s|__RESULTS_PATH__|${RESULTS_PATH}|g" \
        "${TEMPLATE}" > "${CONF}"

    # TORCHDYNAMO_DISABLE=1 makes u-resfm's torch.compile a no-op: upstream RESfM
    # ran eager, and torch 2.0.1's inductor crashes on this model (index_put assert).
    CMD="cd ${REPO_ROOT}; TORCHDYNAMO_DISABLE=1 ${PY} single_scene_optimization.py \
        --conf ${CONF} \
        --scan ${SCAN} \
        --stage 1 \
        --architecture_type esfm_outliers \
        --phase FINE_TUNE \
        --exp_version resfm_repro \
        --results_aggregation_file ${EVAL_ROOT}/Aggregated_repro_results.xlsx \
        --wandb 0"

    if [ "$DRY_RUN" = true ]; then
        echo "DRY RUN: ${CMD}"
    else
        bsub -q "${QUEUE}" \
            -J "repro_${SCAN}" \
            -oo "${REPO_ROOT}/lsf_output/resfm_repro/${SCAN}_%J.out" \
            -eo "${REPO_ROOT}/lsf_output/resfm_repro/${SCAN}_%J.err" \
            -gpu "num=1:j_exclusive=yes:gmem=40G" \
            -R "rusage[mem=50000]" \
            "${CMD}"
    fi
done

echo ""
echo "Submitted $(echo $SCENES | wc -w) reproduction jobs to ${QUEUE}."
echo "Per-scene results: ${EVAL_ROOT}/<scan>_ba/  (Results_FINE_TUNE_*.xlsx)"
echo "Compare to the paper with: python compare_repro_to_paper.py"
