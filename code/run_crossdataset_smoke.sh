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
#        --full : FULL-BUDGET runs (1001-epoch fine-tune, eval every 250) over ALL
#                 scenes (1DSfM 10, Strecha 4, BlendedMVS the 4 scenes matching the
#                 other workstream's repro selection); results under
#                 results/crossdataset_full/. Configs 1-2 are already covered at
#                 full budget by results/crossdataset/{esfm_baseline,resfm_repro}_*.

set -e
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="${REPO_ROOT}/confs/sos_adaptive_eval.conf.template"
CONF_DIR="${REPO_ROOT}/confs/crossdataset_smoke_generated"
SMOKE_ROOT="${REPO_ROOT}/results/crossdataset_smoke"
PY="${REPO_ROOT}/../.venv38-resfm/bin/python"
QUEUE="waic-risk"; DRY_RUN=false; CONFIGS="1 2 3 4 5"; FULL=false

while [[ $# -gt 0 ]]; do case $1 in
    --dry_run) DRY_RUN=true; shift;;
    --configs) CONFIGS="${2//,/ }"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --full) FULL=true; shift;;
    *) echo "Unknown option $1"; exit 1;;
esac; done

# Checkpoints (Part 2 inventory; provenance-clean, best-validation models)
CKPT_ESFM="${REPO_ROOT}/results/multiscene/uesfm_27scenes_shallow_reproj/models/Model_Ep19999.pt"
CKPT_RESFM="${REPO_ROOT}/pretrained/pretrained_model.pt"
CKPT_UESFM="${REPO_ROOT}/results/multiscene/uesfm_27scenes_sos_adaptive_1gpu_any80g/models/Model_Ep16500.pt"

# dataset : removal-threshold (RESfM convention)
DATASETS="1dsfm:0.6 strecha:0.8 blendedmvs:0.8"
scenes_for() {
  case $1 in
    1dsfm) if $FULL; then echo "Alamo Ellis_Island Madrid_Metropolis Montreal_Notre_Dame Notre_Dame NYC_Library Piazza_del_Popolo Tower_of_London Vienna_Cathedral Yorkminster"; else echo "Ellis_Island"; fi;;
    strecha) if $FULL; then echo "entry-P10 fountain-P11 Herz-Jesu-P8 Herz-Jesu-P25"; else echo "entry-P10"; fi;;
    blendedmvs) if $FULL; then echo "58c4bb4f4a69c55606122be4 5a48ba95c7dab83a7d7b44ed 5acf8ca0f3d8a750097e4b15 5b950c71608de421b1e7318f"; else echo "5a48ba95c7dab83a7d7b44ed"; fi;;
  esac
}
if $FULL; then FT_EPOCHS=1001; EVAL_INT=250; SMOKE_ROOT="${REPO_ROOT}/results/crossdataset_full"; else FT_EPOCHS=51; EVAL_INT=25; fi

mkdir -p "${CONF_DIR}" "${SMOKE_ROOT}" "${REPO_ROOT}/lsf_output/crossdataset_smoke"

for DSPEC in $DATASETS; do
  DS="${DSPEC%%:*}"; THR="${DSPEC##*:}"
  for SCAN in $(scenes_for $DS); do
  for CFG in $CONFIGS; do
    case $CFG in
      # cfg  checkpoint       model-class swap  out_mode ft_mode func_tuning  epochs snapshots
      1) CKPT=$CKPT_ESFM;  SWAP=deep; OM=1; FTM=1; FT=ESFMLoss;     NE=$FT_EPOCHS; SNAP="";;
      2) CKPT=$CKPT_RESFM; SWAP=sos;  OM=3; FTM=1; FT=ESFMLoss;     NE=$FT_EPOCHS; SNAP="";;
      3) CKPT=$CKPT_UESFM; SWAP=sos;  OM=1; FTM=1; FT=ESFMLoss;     NE=1;  SNAP="[0]";;
      4) CKPT=$CKPT_UESFM; SWAP=sos;  OM=3; FTM=1; FT=ESFMLoss;     NE=$FT_EPOCHS; SNAP="";;
      5) CKPT=$CKPT_UESFM; SWAP=sos;  OM=1; FTM=3; FT=CombinedLoss; NE=$FT_EPOCHS; SNAP="[0,10,$((FT_EPOCHS-1))]";;
    esac
    TAG="cfg${CFG}_${DS}_${SCAN}"
    CONF="${CONF_DIR}/${TAG}.conf"
    SED_ARGS=(-e "s|__SCAN__|${SCAN}|g"
              -e "s|__RESULTS_PATH__|${SMOKE_ROOT}/${TAG}|g"
              -e "s|pretrainedPath = \".*\"|pretrainedPath = \"${CKPT}\"|"
              -e "s|    dataset = \"megadepth\"|    dataset = \"${DS}\"|"
              -e "s|    outliers_threshold = 0.6|    outliers_threshold = ${THR}  # RESfM per-dataset convention (0.6 MegaDepth/1DSfM, 0.8 low-outlier)|"
              -e "s|    num_epochs = 1001|    num_epochs = ${NE}|"
              -e "s|    eval_intervals = 250|    eval_intervals = ${EVAL_INT}|"
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
done
echo "Submitted: configs {${CONFIGS}}, full=${FULL}, root=${SMOKE_ROOT}"
