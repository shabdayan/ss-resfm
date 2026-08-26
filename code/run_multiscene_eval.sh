#!/bin/bash
# Per-test-scene evaluation of the U-ESFM multi-scene (Stage-1) model (RESfM protocol:
# per-scene ~1K-epoch fine-tune + BA evaluation) over the 36 reconstructed
# MegaDepth test scenes (see RUN_MULTISCENE.md).
#
# For each scene this generates a conf from
# confs/uesfm_eval.conf.template (own results_path per scene — a
# shared one would mix per-scene fine-tune checkpoints) and submits one LSF job
# running single_scene_optimization.py --phase FINE_TUNE. The multi-scene best
# checkpoint is injected via the conf's pretrainedPath.
#
# Usage:
#   ./run_multiscene_eval.sh [--queue waic-medium] [--scans "0238,0060"] [--dry_run]
#
# Per-arm evaluation (completed multi-scene arms; see RUN_MULTISCENE.md):
#   --template <conf.template>   eval conf template (model block must match the arm)
#   --train_results <dir>        arm's training results dir (best checkpoint source)
#   --eval_root <dir>            where per-scene eval results land
#   --arch <name>                architecture_type tag for the Results_*.xlsx naming
#
# The FINE_TUNE flow follows upstream RESfM: a TEST evaluation with the loaded
# checkpoint first saves predicted outliers, which the fine-tune data load prunes
# (restored in train.py / single_scene_optimization.py).

set -e
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
TRAIN_RESULTS="${REPO_ROOT}/results/multiscene/uesfm_27scenes_stage1"
# Override with EVAL_ROOT_OVERRIDE / EVAL_PYTHON to run in a different environment
# (e.g. .venv38-resfm, the upstream-matched env used for training).
EVAL_ROOT="${EVAL_ROOT_OVERRIDE:-${REPO_ROOT}/results/multiscene/uesfm_stage1_eval}"
TEMPLATE="${REPO_ROOT}/confs/uesfm_eval.conf.template"
CONF_DIR="${REPO_ROOT}/confs/multiscene_eval_generated"
PY="${EVAL_PYTHON:-${REPO_ROOT}/../.venv/bin/python}"

QUEUE="waic-medium"
DRY_RUN=false
SEED=20  # paper/RESfM-code default; use --seed N for the multi-seed median protocol
JOBPREFIX="ueval"  # LSF job-name prefix; override with --jobprefix so distinct fleets
                   # sharing scene+seed don't collide in the heal RUNNING-dedup check
ARCH="esfm_outliers_deep"
SCENES="0238 0060 0197 0094 0265 0083 0076 0185 0048 0024 0223 5016 0046 0099 1001 0231 0411 0377 0102 0147 0148 0446 0022 0327 0015 0455 0496 1589 0012 0104 0019 0063 0130 0080 0240 0007"

while [[ $# -gt 0 ]]; do
    case $1 in
        --queue) QUEUE="$2"; shift 2;;
        --scans) SCENES="${2//,/ }"; shift 2;;
        --seed) SEED="$2"; shift 2;;
        --template) TEMPLATE="$2"; shift 2;;
        --train_results) TRAIN_RESULTS="$2"; shift 2;;
        --eval_root) EVAL_ROOT="$2"; shift 2;;
        --arch) ARCH="$2"; shift 2;;
        --jobprefix) JOBPREFIX="$2"; shift 2;;
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

if [ "$SEED" != "20" ]; then EVAL_ROOT="${EVAL_ROOT}_seed${SEED}"; fi

mkdir -p "${CONF_DIR}" "${EVAL_ROOT}" "${REPO_ROOT}/lsf_output/multiscene_eval"

for SCAN in $SCENES; do
    # Conf name carries the eval root so concurrent per-arm fan-outs don't collide
    CONF="${CONF_DIR}/$(basename ${EVAL_ROOT})_${SCAN}_seed${SEED}.conf"
    RESULTS_PATH="${EVAL_ROOT}/${SCAN}_ba"
    sed -e "s|__SCAN__|${SCAN}|g" \
        -e "s|__RESULTS_PATH__|${RESULTS_PATH}|g" \
        -e "s|__PRETRAINED__|${CKPT}|g" \
        -e "s|random_seed = 20|random_seed = ${SEED}|" \
        "${TEMPLATE}" > "${CONF}"
    echo "Generated ${CONF}"

    # TORCHDYNAMO_DISABLE=1: torch 2.0.1's inductor crashes on this model's compile
    CMD="cd ${REPO_ROOT}; TORCHDYNAMO_DISABLE=1 ${PY} single_scene_optimization.py \
        --conf ${CONF} \
        --scan ${SCAN} \
        --stage 1 \
        --architecture_type ${ARCH} \
        --phase FINE_TUNE \
        --exp_version multiscene_eval \
        --results_aggregation_file ${EVAL_ROOT}/Aggregated_eval_results.xlsx \
        --wandb 0"

    if [ "$DRY_RUN" = true ]; then
        echo "DRY RUN: ${CMD}"
    else
        bsub -q "${QUEUE}" \
            -J "${JOBPREFIX}_s${SEED}_${SCAN}" \
            -oo "${REPO_ROOT}/lsf_output/multiscene_eval/${SCAN}_s${SEED}_%J.out" \
            -eo "${REPO_ROOT}/lsf_output/multiscene_eval/${SCAN}_s${SEED}_%J.err" \
            -gpu "num=1:j_exclusive=yes:gmem=80G" \
            -R "rusage[mem=50000]" \
            "${CMD}"
    fi
done

echo ""
echo "Submitted $(echo $SCENES | wc -w) evaluation jobs to ${QUEUE}."
echo "Results land under: ${EVAL_ROOT}/<scan>_ba/"
echo "Aggregated table:   ${EVAL_ROOT}/Aggregated_eval_results.xlsx"
