#!/bin/bash
# Cross-dataset 5-seed median protocol fan-out (seeds 21-24; seed 20 already
# run). Submits every arm of CROSSDATASET_RESULTS.md per seed:
#   7 plain arms x 18 scenes + 2 TTT variants x 18 scenes = 162 jobs/seed.
# (6 U-ESFM/ESFM arms via run_multiscene_eval.sh, RESfM repro via
# run_resfm_repro.sh, TTT via run_ttt.sh.)
# Usage: ./run_crossdataset_seeds.sh [seeds...]   (default: 21 22 23 24)
set -u
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"
PY38="$REPO_ROOT/../.venv38-resfm/bin/python"
SHAL_CKPT="$REPO_ROOT/results/multiscene/uesfm_27scenes_shallow_adaptive/models/Model_Ep17000.pt"

DS_LIST=(
  "strecha:entry-P10,fountain-P11,Herz-Jesu-P8,Herz-Jesu-P25"
  "blendedmvs:58c4bb4f4a69c55606122be4,5acf8ca0f3d8a750097e4b15,5a48ba95c7dab83a7d7b44ed,5b950c71608de421b1e7318f"
  "1dsfm:Alamo,Ellis_Island,Madrid_Metropolis,Montreal_Notre_Dame,Notre_Dame,NYC_Library,Piazza_del_Popolo,Tower_of_London,Vienna_Cathedral,Yorkminster"
)

# template-prefix : template-suffix : train-results-dir : eval-root-prefix : arch-tag
# template file = confs/<prefix>_<dataset><suffix>.conf.template
ARMS=(
  "crossdataset_eval::uesfm_27scenes_stage1:uesfm_stage1:esfm_outliers_deep"
  "crossdataset_eval:_ftlr1e4:uesfm_27scenes_stage1:uesfm_stage1_ftlr1e4:esfm_outliers_deep"
  "crossdataset_esfm_baseline::uesfm_27scenes_stage1:esfm_baseline:esfm_baseline"
  "crossdataset_esfm_baseline:_ftlr1e4:uesfm_27scenes_stage1:esfm_baseline_ftlr1e4:esfm_baseline"
  "crossdataset_shallow_adaptive::uesfm_27scenes_shallow_adaptive:uesfm_shallow_adaptive_ep17k:esfm_outliers_shallow"
  "crossdataset_shallow_adaptive:_ftlr1e4:uesfm_27scenes_shallow_adaptive:uesfm_shallow_adaptive_ep17k_ftlr1e4:esfm_outliers_shallow"
)

# Preflight: every template must exist before anything is submitted.
for arm in "${ARMS[@]}"; do
  IFS=: read -r tpl suf _ _ _ <<< "$arm"
  for ds in "${DS_LIST[@]}"; do
    n=${ds%%:*}
    tfile="confs/${tpl}_${n}${suf}.conf.template"
    [ -f "$tfile" ] || { echo "MISSING TEMPLATE $tfile"; exit 1; }
  done
done

for S in "${@:-21 22 23 24}"; do
  echo "=== seed $S ==="
  for arm in "${ARMS[@]}"; do
    IFS=: read -r tpl suf train root arch <<< "$arm"
    for ds in "${DS_LIST[@]}"; do
      n=${ds%%:*}; scans=${ds##*:}
      cnt=$(EVAL_PYTHON="$PY38" ./run_multiscene_eval.sh \
        --template "confs/${tpl}_${n}${suf}.conf.template" \
        --train_results "results/multiscene/$train" \
        --eval_root "results/crossdataset/${root}_${n}_eval" \
        --arch "$arch" --scans "$scans" --seed "$S" 2>/dev/null \
        | grep -c "^Generated" || true)
      echo "  ${root}_${n}: $cnt jobs"
    done
  done
  for ds in "${DS_LIST[@]}"; do
    n=${ds%%:*}; scans=${ds##*:}
    if REPRO_PYTHON="$PY38" ./run_resfm_repro.sh \
        --template "confs/crossdataset_resfm_repro_${n}.conf.template" \
        --eval_root "results/crossdataset/resfm_repro_${n}" \
        --scans "$scans" --seed "$S" >/dev/null 2>&1; then
      echo "  resfm_repro_${n}: submitted"
    else
      echo "  resfm_repro_${n}: SUBMIT FAILED (seed $S)"
    fi
  done
  for loss in comb reproj_only; do
    for ds in "${DS_LIST[@]}"; do
      n=${ds%%:*}; scans=${ds##*:}
      ./run_ttt.sh --ttt_loss "$loss" --checkpoint "$SHAL_CKPT" \
        --template "confs/crossdataset_shallow_adaptive_${n}.conf.template" \
        --scans "$scans" --tag "xd_${n}_shallow_adaptive_ep17k" --seed "$S" 2>/dev/null \
        | grep -oE "Submitted [0-9]+ TTT jobs" || echo "  ttt_${n}_${loss}: SUBMIT FAILED (seed $S)"
    done
  done
done
