#!/bin/bash
# Cross-dataset readiness SMOKE tests (TASK_crossdataset_readiness Part 3).
# 5 configs x 3 datasets, ONE small scene each, reduced fine-tune budget
# (50 epochs) — sanity/plumbing checks, NOT results runs.
#
# All fine-tuning configs run through the SAME pipeline
# (single_scene_optimization.py --phase FINE_TUNE) with conf-selected
# model/loss/protocol, per parent-spec R6/R13. Removal threshold follows
# RESfM's per-dataset convention (0.6 MegaDepth/1DSfM, 0.8 Strecha/BlendedMVS)
# and is recorded in every generated conf (kept alongside the results).
#
# Usage: ./run_crossdataset_smoke.sh [--dry_run] [--configs "1,2,3,4,5"] [--queue waic-risk]

set -e
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="${REPO_ROOT}/confs/sos_adaptive_eval.conf.template"
CONF_DIR="${REPO_ROOT}/confs/crossdataset_smoke_generated"
SMOKE_ROOT="${REPO_ROOT}/results/crossdataset_smoke"
PY="${REPO_ROOT}/../.venv38-resfm/bin/python"
QUEUE="waic-risk"; DRY_RUN=false; CONFIGS="1 2 3 4 5"

while [[ $# -gt 0 ]]; do case $1 in
    --dry_run) DRY_RUN=true; shift;;
    --configs) CONFIGS="${2//,/ }"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    *) echo "Unknown option $1"; exit 1;;
esac; done

# Checkpoints (Part 2 inventory; provenance-clean, best-validation models)
CKPT_ESFM="${REPO_ROOT}/results/multiscene/uesfm_27scenes_shallow_reproj/models/Model_Ep19999.pt"
CKPT_RESFM="${REPO_ROOT}/pretrained/pretrained_model.pt"
CKPT_UESFM="${REPO_ROOT}/results/multiscene/uesfm_27scenes_sos_adaptive_1gpu_any80g/models/Model_Ep16500.pt"

# dataset  smoke-scene                         removal-threshold
DATASETS="1dsfm:Ellis_Island:0.6 strecha:entry-P10:0.8 blendedmvs:5a48ba95c7dab83a7d7b44ed:0.8"

mkdir -p "${CONF_DIR}" "${SMOKE_ROOT}" "${REPO_ROOT}/lsf_output/crossdataset_smoke"

for DSPEC in $DATASETS; do
  DS="${DSPEC%%:*}"; REST="${DSPEC#*:}"; SCAN="${REST%%:*}"; THR="${REST##*:}"
  for CFG in $CONFIGS; do
    case $CFG in
      # cfg  checkpoint       model-class swap  out_mode ft_mode func_tuning  epochs snapshots
      1) CKPT=$CKPT_ESFM;  SWAP=deep; OM=1; FTM=1; FT=ESFMLoss;     NE=51; SNAP="";;
      2) CKPT=$CKPT_RESFM; SWAP=sos;  OM=3; FTM=1; FT=ESFMLoss;     NE=51; SNAP="";;
      3) CKPT=$CKPT_UESFM; SWAP=sos;  OM=1; FTM=1; FT=ESFMLoss;     NE=1;  SNAP="[0]";;
      4) CKPT=$CKPT_UESFM; SWAP=sos;  OM=3; FTM=1; FT=ESFMLoss;     NE=51; SNAP="";;
      5) CKPT=$CKPT_UESFM; SWAP=sos;  OM=1; FTM=3; FT=CombinedLoss; NE=51; SNAP="[0,10,50]";;
    esac
    TAG="cfg${CFG}_${DS}"
    CONF="${CONF_DIR}/${TAG}.conf"
    SED_ARGS=(-e "s|__SCAN__|${SCAN}|g"
              -e "s|__RESULTS_PATH__|${SMOKE_ROOT}/${TAG}|g"
              -e "s|pretrainedPath = \".*\"|pretrainedPath = \"${CKPT}\"|"
              -e "s|    dataset = \"megadepth\"|    dataset = \"${DS}\"|"
              -e "s|    outliers_threshold = 0.6|    outliers_threshold = ${THR}  # RESfM per-dataset convention (0.6 MegaDepth/1DSfM, 0.8 low-outlier)|"
              -e "s|    num_epochs = 1001|    num_epochs = ${NE}|"
              -e "s|    eval_intervals = 250|    eval_intervals = 25|"
              -e "s|    output_mode = 3|    output_mode = ${OM}|"
              -e "s|func_tuning = ESFMLoss|func_tuning = ${FT}|")
    if [ "$SWAP" = "deep" ]; then
      # config 1 (ESFM row) uses OUR trained baseline: DeepSetOfSetOutliersNet at 1x3
      SED_ARGS+=(-e 's|type = "SetOfSet.SetOfSetOutliersNet"|type = "SetOfSet.DeepSetOfSetOutliersNet"|')
      ARCH="esfm_outliers_deep"
    else
      ARCH="esfm_outliers"
    fi
    [ -n "$SNAP" ] && SED_ARGS+=(-e "/validation_metric_fine_tuning/a\\    snapshot_epochs = ${SNAP}")
    [ "$FTM" = "3" ] && SED_ARGS+=(-e "/validation_metric_fine_tuning/a\\    fine_tune_output_mode = 3")
    sed "${SED_ARGS[@]}" "${TEMPLATE}" > "${CONF}"

    CMD="cd ${REPO_ROOT}; TORCHDYNAMO_DISABLE=1 ${PY} single_scene_optimization.py \
        --conf ${CONF} --scan ${SCAN} --stage 1 --architecture_type ${ARCH} \
        --phase FINE_TUNE --exp_version smoke_${TAG} \
        --results_aggregation_file ${SMOKE_ROOT}/Aggregated_smoke.xlsx --wandb 0"
    if [ "$DRY_RUN" = true ]; then echo "DRY RUN: ${TAG}"; else
      bsub -q "${QUEUE}" -J "xsmoke_${TAG}" \
        -oo "${REPO_ROOT}/lsf_output/crossdataset_smoke/${TAG}_%J.out" \
        -eo "${REPO_ROOT}/lsf_output/crossdataset_smoke/${TAG}_%J.err" \
        -gpu "num=1:j_exclusive=yes:gmem=80G" -R "rusage[mem=50000]" "${CMD}"
    fi
  done
done
echo "Smoke matrix submitted: configs {${CONFIGS}} x {1dsfm/Ellis_Island, strecha/entry-P10, blendedmvs/5a48ba95}"
