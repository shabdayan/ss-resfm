#!/bin/bash
# Launch + self-heal the RESfM-paper-style evaluation of the two REPORT-FAITHFUL
# arms (deep + shallow, reproj_weighting=weighted, beta=1). For each arm, gated on
# its 20k checkpoint (models_all/Model_Ep19999.pt):
#   - OOD cross-datasets: 1dsfm(10) / strecha(4) / blendedmvs(4), seeds 20-24
#   - MegaDepth in-distribution: 36 RESfM Table-1 scenes, seed 20
# Same conservative per-cell logic as heal_eval_fleets.sh: resubmit a cell iff its
# Results_FINE_TUNE is missing AND no ueval job for that scene+seed is PEND/RUN.
# Standard MAD-based eval protocol (matches how the other adaptive arms were
# evaluated), so the numbers slot directly into the existing comparison tables.
#
# Usage: bash heal_reportfaithful_evals.sh   (run by the /loop watcher after 20k)
set -u
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO"
PY38="$REPO/../.venv38-resfm/bin/python"
S1DS="Alamo Ellis_Island Madrid_Metropolis Montreal_Notre_Dame Notre_Dame NYC_Library Piazza_del_Popolo Tower_of_London Vienna_Cathedral Yorkminster"
SSTR="entry-P10 fountain-P11 Herz-Jesu-P8 Herz-Jesu-P25"
SBMV="58c4bb4f4a69c55606122be4 5acf8ca0f3d8a750097e4b15 5a48ba95c7dab83a7d7b44ed 5b950c71608de421b1e7318f"
MDSCENES="0238 0060 0197 0094 0265 0083 0076 0185 0048 0024 0223 5016 0046 0099 1001 0231 0411 0377 0102 0147 0148 0446 0022 0327 0015 0455 0496 1589 0012 0104 0019 0063 0130 0080 0240 0007"
scenes_for(){ case "$1" in 1dsfm) echo "$S1DS";; strecha) echo "$SSTR";; blendedmvs) echo "$SBMV";; esac; }
RUNNING=$(bjobs -w 2>/dev/null | grep ueval | awk '{print $7}' | sed 's/ueval_s//')

# arm rows: train | rp | arch | tpl1dsfm | tplstrecha | tplblendedmvs | tplmega
ARMS=(
 "uesfm_27scenes_adaptive_reportfaithful|uesfm_deep_reportfaithful|esfm_outliers_deep|crossdataset_eval_1dsfm|crossdataset_eval_strecha|crossdataset_eval_blendedmvs|megadepth_deep_adaptive"
 "uesfm_27scenes_shallow_adaptive_reportfaithful|uesfm_shallow_reportfaithful|esfm_outliers_shallow|crossdataset_shallow_adaptive_1dsfm|crossdataset_shallow_adaptive_strecha_ftlr1e4|crossdataset_shallow_adaptive_blendedmvs|megadepth_shallow_adaptive"
)
total=0
for row in "${ARMS[@]}"; do
  IFS='|' read -r train rp arch t1 tstr tbmv tmega <<< "$row"
  [ -e "results/multiscene/$train/models_all/Model_Ep19999.pt" ] || { echo "  $rp: checkpoint not ready (skip)"; continue; }
  # --- OOD cross-datasets, seeds 20-24 ---
  for ds in 1dsfm strecha blendedmvs; do
    case "$ds" in 1dsfm) tmpl="confs/${t1}.conf.template";; strecha) tmpl="confs/${tstr}.conf.template";; blendedmvs) tmpl="confs/${tbmv}.conf.template";; esac
    [ -f "$tmpl" ] || { echo "  missing template $tmpl"; continue; }
    for seed in 20 21 22 23 24; do
      base="results/crossdataset/${rp}_${ds}_eval"; [ "$seed" != 20 ] && base="${base}_seed${seed}"
      donelist=$(ls "$base"/*_ba/Results_FINE_TUNE*.xlsx 2>/dev/null)
      todo=""
      for sc in $(scenes_for $ds); do
        echo "$donelist" | grep -q "/${sc}_ba/" && continue
        echo "$RUNNING" | grep -qx "${seed}_${sc}" && continue
        todo="$todo,$sc"
      done
      todo="${todo#,}"; [ -z "$todo" ] && continue
      EVAL_PYTHON="$PY38" ./run_multiscene_eval.sh --template "$tmpl" \
        --train_results "results/multiscene/$train" --eval_root "results/crossdataset/${rp}_${ds}_eval" \
        --arch "$arch" --scans "$todo" --seed "$seed" >/dev/null 2>&1 \
        && { c=$(echo "$todo" | tr ',' '\n' | wc -l); total=$((total+c)); echo "  heal ${rp}_${ds} s${seed}: $todo"; }
    done
  done
  # --- MegaDepth in-distribution, seed 20 ---
  mdbase="results/multiscene/${rp}_megadepth_eval"
  mddone=$(ls "$mdbase"/*_ba/Results_FINE_TUNE*.xlsx 2>/dev/null)
  mdtodo=""
  for sc in $MDSCENES; do
    echo "$mddone" | grep -q "/${sc}_ba/" && continue
    echo "$RUNNING" | grep -qx "20_${sc}" && continue
    mdtodo="$mdtodo,$sc"
  done
  mdtodo="${mdtodo#,}"
  if [ -n "$mdtodo" ]; then
    EVAL_PYTHON="$PY38" ./run_multiscene_eval.sh --template "confs/${tmega}.conf.template" \
      --train_results "results/multiscene/$train" --eval_root "$mdbase" \
      --arch "$arch" --scans "$mdtodo" --seed 20 >/dev/null 2>&1 \
      && { c=$(echo "$mdtodo"|tr ',' '\n'|wc -l); total=$((total+c)); echo "  heal ${rp}_megadepth s20: $mdtodo"; }
  fi
done
echo "heal_reportfaithful_evals: resubmitted $total cell(s)"
