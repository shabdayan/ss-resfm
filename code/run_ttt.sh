#!/bin/bash
# TTT runner (SPEC_cvpr_experiments R1-R2): per-test-scene fine-tune from a
# multi-scene checkpoint with conf-selectable loss variant and step-count
# snapshots at {0,10,100,1000} (0 = frozen inference; snapshots log pre-BA
# metrics, full robust-BA eval at first+last step, collapse-check stats, timing).
#
# Variants:
#   --ttt_loss comb         full unsupervised loss (CombinedLoss), outlier head active
#   --ttt_loss reproj_only  RESfM-style control (ESFMLoss, cameras/points only)
# The "frozen" Table-2 column comes from step-0 rows (identical across variants).
#
# Usage:
#   ./run_ttt.sh --ttt_loss comb [--scans "0223,5016"] [--checkpoint /path/Model_Ep*.pt]
#                [--steps "0,10,100,1000"] [--seed 20] [--queue waic-risk] [--dry_run]
#
# Per-arm TTT (sweep arms; model block must match the checkpoint):
#   --template <conf.template>   eval conf template (default: stage-1 Deep-2x3 MAD template)
#   --tag <name>                 results subdir tag: results/multiscene/ttt/<tag>_<loss>_seed<S>
#                                (also prefixes generated conf names — avoids arm collisions)

set -e
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="${REPO_ROOT}/confs/uesfm_eval.conf.template"
CONF_DIR="${REPO_ROOT}/confs/ttt_generated"
PY="${REPO_ROOT}/../.venv38-resfm/bin/python"

QUEUE="waic-risk"
DRY_RUN=false
SEED=20
TTT_LOSS=""
STEPS="0,10,100,1000"
CHECKPOINT=""
OUTLIER_SOURCE="mad"
TAG=""
SCENES="0238 0060 0197 0094 0265 0083 0076 0185 0048 0024 0223 5016 0046 0099 1001 0231 0411 0377 0102 0147 0148 0446 0022 0327 0015 0455 0496 1589 0012 0104 0019 0063 0130 0080 0240 0007"

while [[ $# -gt 0 ]]; do
    case $1 in
        --ttt_loss) TTT_LOSS="$2"; shift 2;;
        --scans) SCENES="${2//,/ }"; shift 2;;
        --checkpoint) CHECKPOINT="$2"; shift 2;;
        --steps) STEPS="$2"; shift 2;;
        --seed) SEED="$2"; shift 2;;
        --queue) QUEUE="$2"; shift 2;;
        --outlier_source) OUTLIER_SOURCE="$2"; shift 2;;
        --template) TEMPLATE="$2"; shift 2;;
        --tag) TAG="$2"; shift 2;;
        --dry_run) DRY_RUN=true; shift;;
        *) echo "Unknown option $1"; exit 1;;
    esac
done

case "$TTT_LOSS" in
    comb)        FUNC_TUNING="CombinedLoss"; FT_MODE=3;;
    reproj_only) FUNC_TUNING="ESFMLoss";     FT_MODE=1;;
    *) echo "ERROR: --ttt_loss must be comb or reproj_only"; exit 1;;
esac

if [ -z "$CHECKPOINT" ]; then
    CHECKPOINT=$(ls -1v "${REPO_ROOT}/results/multiscene/uesfm_27scenes_stage1/models"/Model_Ep*.pt 2>/dev/null | tail -1)
fi
if [ ! -f "$CHECKPOINT" ]; then echo "ERROR: no checkpoint (${CHECKPOINT})"; exit 1; fi
echo "TTT from checkpoint: ${CHECKPOINT}  variant: ${TTT_LOSS}  seed: ${SEED}"

MAXSTEP=$(echo "$STEPS" | tr ',' '\n' | sort -n | tail -1)
NUM_EPOCHS=$((MAXSTEP + 1))
PREFIX="${TAG:+${TAG}_}"
EVAL_ROOT="${REPO_ROOT}/results/multiscene/ttt/${PREFIX}${TTT_LOSS}_seed${SEED}"
mkdir -p "${CONF_DIR}" "${EVAL_ROOT}" "${REPO_ROOT}/lsf_output/ttt"

for SCAN in $SCENES; do
    CONF="${CONF_DIR}/${PREFIX}${SCAN}_${TTT_LOSS}_seed${SEED}.conf"
    sed -e "s|__SCAN__|${SCAN}|g" \
        -e "s|__RESULTS_PATH__|${EVAL_ROOT}/${SCAN}_ba|g" \
        -e "s|random_seed = 20|random_seed = ${SEED}|" \
        -e "s|func_tuning = ESFMLoss.*|func_tuning = ${FUNC_TUNING}|" \
        -e "s|    num_epochs = 1001|    num_epochs = ${NUM_EPOCHS}|" \
        -e "s|outlier_source = \"mad\"|outlier_source = \"${OUTLIER_SOURCE}\"|" \
        -e "s|pretrainedPath = \".*\"|pretrainedPath = \"${CHECKPOINT}\"|" \
        -e "/validation_metric_fine_tuning/a\\    snapshot_epochs = [${STEPS}]\\n    fine_tune_output_mode = ${FT_MODE}" \
        "${TEMPLATE}" > "${CONF}"

    CMD="cd ${REPO_ROOT}; TORCHDYNAMO_DISABLE=1 ${PY} single_scene_optimization.py \
        --conf ${CONF} --scan ${SCAN} --stage 1 --architecture_type esfm_outliers_deep \
        --phase FINE_TUNE --exp_version ttt_${TTT_LOSS} \
        --results_aggregation_file ${EVAL_ROOT}/Aggregated_ttt.xlsx --wandb 0"

    if [ "$DRY_RUN" = true ]; then echo "DRY RUN: ${CMD}"; else
        bsub -q "${QUEUE}" -J "ttt_${PREFIX}${TTT_LOSS}_s${SEED}_${SCAN}" \
            -oo "${REPO_ROOT}/lsf_output/ttt/${PREFIX}${SCAN}_${TTT_LOSS}_s${SEED}_%J.out" \
            -eo "${REPO_ROOT}/lsf_output/ttt/${PREFIX}${SCAN}_${TTT_LOSS}_s${SEED}_%J.err" \
            -gpu "num=1:j_exclusive=yes:gmem=40G" -R "rusage[mem=50000]" "${CMD}"
    fi
done
echo "Submitted $(echo $SCENES | wc -w) TTT jobs (${TTT_LOSS}); results: ${EVAL_ROOT}"
