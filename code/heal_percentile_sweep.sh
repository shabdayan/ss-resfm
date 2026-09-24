#!/bin/bash
# Launch + self-heal the adaptive-loss percentile sweep for the shallow report-
# faithful checkpoint: for each percentile retrain {p1090,p1585,p2575,p3070}, once
# its 20k checkpoint exists, evaluate the 4 mechanisms {weight, weight_ttt, hybrid,
# hybrid_ttt} on all 4 datasets (1DSfM/Strecha/BlendedMVS/MegaDepth), seed 20.
# Reuses the crossdataset_rf_* / megadepth_rf_* shallow eval templates, pointing
# --train_results at each percentile checkpoint. (20/80 is the existing baseline.)
set -u
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO"
PY38="$REPO/../.venv38-resfm/bin/python"
S1DS="Alamo Ellis_Island Madrid_Metropolis Montreal_Notre_Dame Notre_Dame NYC_Library Piazza_del_Popolo Tower_of_London Vienna_Cathedral Yorkminster"
SSTR="entry-P10 fountain-P11 Herz-Jesu-P8 Herz-Jesu-P25"
SBMV="58c4bb4f4a69c55606122be4 5acf8ca0f3d8a750097e4b15 5a48ba95c7dab83a7d7b44ed 5b950c71608de421b1e7318f"
MDSCENES="0238 0060 0197 0094 0265 0083 0076 0185 0048 0024 0223 5016 0046 0099 1001 0231 0411 0377 0102 0147 0148 0446 0022 0327 0015 0455 0496 1589 0012 0104 0019 0063 0130 0080 0240 0007"
scenes_for(){ case "$1" in 1dsfm) echo "$S1DS";; strecha) echo "$SSTR";; blendedmvs) echo "$SBMV";; megadepth) echo "$MDSCENES";; esac; }
ALLRUN=$(bjobs -w 2>/dev/null)
total=0
for tag in p1090 p1585 p2575 p3070; do
  train="uesfm_27scenes_shallow_adaptive_reportfaithful_${tag}"
  [ -e "results/multiscene/$train/models_all/Model_Ep19999.pt" ] || { echo "  $tag checkpoint not ready (skip)"; continue; }
  for mech in weight weight_ttt hybrid hybrid_ttt; do
    for ds in 1dsfm strecha blendedmvs megadepth; do
      if [ "$ds" = megadepth ]; then
        tmpl="confs/megadepth_rf_${mech}_shallow.conf.template"
        root="results/multiscene/uesfm_shallow_${tag}_rf_${mech}_megadepth_eval"
      else
        tmpl="confs/crossdataset_rf_${mech}_shallow_${ds}.conf.template"
        root="results/crossdataset/uesfm_shallow_${tag}_rf_${mech}_${ds}_eval"
      fi
      [ -f "$tmpl" ] || { echo "  missing template $tmpl"; continue; }
      jp="sw_${tag}_${mech}_${ds}"
      donelist=$(ls "$root"/*_ba/Results_FINE_TUNE*.xlsx 2>/dev/null)
      runlist=$(echo "$ALLRUN" | grep -F "$jp" | awk '{print $7}')
      todo=""
      for sc in $(scenes_for "$ds"); do
        echo "$donelist" | grep -q "/${sc}_ba/" && continue
        echo "$runlist" | grep -q "_${sc}$" && continue
        todo="$todo,$sc"
      done
      todo="${todo#,}"; [ -z "$todo" ] && continue
      EVAL_PYTHON="$PY38" ./run_multiscene_eval.sh --template "$tmpl" \
        --train_results "results/multiscene/$train" --eval_root "$root" \
        --arch esfm_outliers_shallow --scans "$todo" --seed 20 --jobprefix "$jp" >/dev/null 2>&1 \
        && { c=$(echo "$todo" | tr ',' '\n' | wc -l); total=$((total+c)); echo "  $jp: $todo"; }
    done
  done
done
echo "heal_percentile_sweep: $total cell(s)"
