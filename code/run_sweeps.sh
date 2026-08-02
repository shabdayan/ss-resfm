#!/bin/bash
# R12 loss-component sweep runner (SPEC_uesfm_combined §2.C): per-test-scene
# fine-tune from the multi-scene checkpoint, one LSF job per (sweep value, scene),
# results under results/multiscene/sweeps/<sweep>/<tag>/<scan>_ba.
# Aggregate with: python sweep_aggregate.py --sweep <name>
#
# Sweeps (one per invocation):
#   percentiles        loss.inlier/outlier_percentile {(10,90),(20,80),(30,70)}  [CombinedLoss fine-tune]
#   warmup             loss.warmup_epochs {0,10,50}                              [CombinedLoss fine-tune]
#   beta               loss.classification_loss_weight {0,0.5,1.0}               [CombinedLoss fine-tune;
#                      beta=0 doubles as the confidence-weighting-off control]
#   removal_threshold  test.outliers_threshold {0.4,0.5,0.6,0.7,0.8} (RESfM App. B grid).
#                      Switches outlier_source to classifier scores — requires a
#                      checkpoint with a TRAINED outlier head (adaptive arm), not
#                      the stage-1 output_mode=1 checkpoint.
#   mad_alpha          test.mad_alpha {1.0,1.5,2.0,2.5,3.0} — the active MAD
#                      removal path's analogous knob (works with the stage-1 model).
#
# Usage:
#   ./run_sweeps.sh --sweep percentiles [--scans "0223,5016,0015"]
#                   [--checkpoint /path/Model_Ep*.pt] [--seed 20] [--queue waic-risk] [--dry_run]

set -e
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="${REPO_ROOT}/confs/uesfm_eval.conf.template"
PY="${REPO_ROOT}/../.venv38-resfm/bin/python"

QUEUE="waic-risk"
DRY_RUN=false
SEED=20
SWEEP=""
CHECKPOINT=""
# Small default subset (sweeps are sensitivity curves, not the full table):
# 0223 (Group 1, 17% outliers), 5016 (Group 1, 0.2%), 0015 (Group 2, 20.6%).
SCENES="0223 5016 0015"

while [[ $# -gt 0 ]]; do
    case $1 in
        --sweep) SWEEP="$2"; shift 2;;
        --scans) SCENES="${2//,/ }"; shift 2;;
        --checkpoint) CHECKPOINT="$2"; shift 2;;
        --seed) SEED="$2"; shift 2;;
        --queue) QUEUE="$2"; shift 2;;
        --dry_run) DRY_RUN=true; shift;;
        *) echo "Unknown option $1"; exit 1;;
    esac
done

LOSS_SWEEP=false
case "$SWEEP" in
    percentiles)       VALUES="10:90 20:80 30:70"; LOSS_SWEEP=true;;
    warmup)            VALUES="0 10 50"; LOSS_SWEEP=true;;
    beta)              VALUES="0 0.5 1.0"; LOSS_SWEEP=true;;
    removal_threshold) VALUES="0.4 0.5 0.6 0.7 0.8";;
    mad_alpha)         VALUES="1.0 1.5 2.0 2.5 3.0";;
    *) echo "ERROR: --sweep must be percentiles|warmup|beta|removal_threshold|mad_alpha"; exit 1;;
esac

if [ -z "$CHECKPOINT" ]; then
    CHECKPOINT=$(ls -1v "${REPO_ROOT}/results/multiscene/uesfm_27scenes_stage1/models"/Model_Ep*.pt 2>/dev/null | tail -1)
fi
if [ ! -f "$CHECKPOINT" ]; then echo "ERROR: no checkpoint (${CHECKPOINT})"; exit 1; fi
echo "Sweep ${SWEEP} from checkpoint: ${CHECKPOINT}  seed: ${SEED}"

CONF_DIR="${REPO_ROOT}/confs/sweeps_generated/${SWEEP}"
SWEEP_ROOT="${REPO_ROOT}/results/multiscene/sweeps/${SWEEP}"
mkdir -p "${CONF_DIR}" "${SWEEP_ROOT}" "${REPO_ROOT}/lsf_output/sweeps"

for VALUE in $VALUES; do
    TAG="${SWEEP}_${VALUE/:/_}"
    # Per-sweep conf substitutions on top of the protocol template.
    SED_ARGS=()
    case "$SWEEP" in
        percentiles)
            LO="${VALUE%%:*}"; HI="${VALUE##*:}"
            SED_ARGS+=(-e "s|inlier_percentile = 20.0|inlier_percentile = ${LO}.0|"
                       -e "s|outlier_percentile = 80.0|outlier_percentile = ${HI}.0|");;
        warmup)
            SED_ARGS+=(-e "s|warmup_epochs = 10|warmup_epochs = ${VALUE}|");;
        beta)
            SED_ARGS+=(-e "s|classification_loss_weight = 0.3|classification_loss_weight = ${VALUE}|");;
        removal_threshold)
            SED_ARGS+=(-e "s|outliers_threshold = 0.6|outliers_threshold = ${VALUE}|"
                       -e "s|outlier_source = \"mad\"|outlier_source = \"\"|");;
        mad_alpha)
            SED_ARGS+=(-e "s|mad_alpha = 2.0.*|mad_alpha = ${VALUE}|");;
    esac
    if [ "$LOSS_SWEEP" = true ]; then
        # The swept parameters live in CombinedLoss: fine-tune with the full loss
        # and the outlier head active (same wiring as run_ttt.sh --ttt_loss comb).
        SED_ARGS+=(-e "s|func_tuning = ESFMLoss.*|func_tuning = CombinedLoss|"
                   -e "/validation_metric_fine_tuning/a\\    fine_tune_output_mode = 3")
    fi

    for SCAN in $SCENES; do
        CONF="${CONF_DIR}/${SCAN}_${TAG}_seed${SEED}.conf"
        sed -e "s|__SCAN__|${SCAN}|g" \
            -e "s|__RESULTS_PATH__|${SWEEP_ROOT}/${TAG}/${SCAN}_ba|g" \
            -e "s|random_seed = 20|random_seed = ${SEED}|" \
            -e "s|pretrainedPath = \".*\"|pretrainedPath = \"${CHECKPOINT}\"|" \
            "${SED_ARGS[@]}" \
            "${TEMPLATE}" > "${CONF}"

        CMD="cd ${REPO_ROOT}; TORCHDYNAMO_DISABLE=1 ${PY} single_scene_optimization.py \
            --conf ${CONF} --scan ${SCAN} --stage 1 --architecture_type esfm_outliers_deep \
            --phase FINE_TUNE --exp_version sweep_${TAG} \
            --results_aggregation_file ${SWEEP_ROOT}/${TAG}/Aggregated_sweep.xlsx --wandb 0"

        if [ "$DRY_RUN" = true ]; then echo "DRY RUN: ${CMD}"; else
            bsub -q "${QUEUE}" -J "sw_${TAG}_s${SEED}_${SCAN}" \
                -oo "${REPO_ROOT}/lsf_output/sweeps/${SCAN}_${TAG}_s${SEED}_%J.out" \
                -eo "${REPO_ROOT}/lsf_output/sweeps/${SCAN}_${TAG}_s${SEED}_%J.err" \
                -gpu "num=1:j_exclusive=yes:gmem=40G" -R "rusage[mem=50000]" "${CMD}"
        fi
    done
done
echo "Submitted sweep '${SWEEP}': $(echo $VALUES | wc -w) values x $(echo $SCENES | wc -w) scenes; results: ${SWEEP_ROOT}"
