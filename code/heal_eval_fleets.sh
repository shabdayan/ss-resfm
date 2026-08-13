#!/bin/bash
# Conservative self-heal for all active cross-dataset eval fleets. For each cell
# (fleet x dataset x seed x scene): resubmit iff its Results_FINE_TUNE is missing
# AND no ueval job for that scene+seed is currently PEND/RUN. The queue check is
# fleet-agnostic (job name is ueval_s<seed>_<scene>), so this UNDER-resubmits when
# two fleets share a scene+seed — safe (a skipped cell just retries next cycle;
# it never double-submits into the same results dir, which would corrupt it).
#
# Usage: bash heal_eval_fleets.sh   (run by the /loop watcher; prints resubmits)
set -u
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO"
PY38="$REPO/../.venv38-resfm/bin/python"
S1DS="Alamo Ellis_Island Madrid_Metropolis Montreal_Notre_Dame Notre_Dame NYC_Library Piazza_del_Popolo Tower_of_London Vienna_Cathedral Yorkminster"
SSTR="entry-P10 fountain-P11 Herz-Jesu-P8 Herz-Jesu-P25"
SBMV="58c4bb4f4a69c55606122be4 5acf8ca0f3d8a750097e4b15 5a48ba95c7dab83a7d7b44ed 5b950c71608de421b1e7318f"
scenes_for(){ case "$1" in 1dsfm) echo "$S1DS";; strecha) echo "$SSTR";; blendedmvs) echo "$SBMV";; esac; }
RUNNING=$(bjobs -w 2>/dev/null | grep ueval | awk '{print $7}' | sed 's/ueval_s//')

# fleet rows: tplPrefix | strechaTpl | train | rootPrefix | arch | seeds
FLEETS=(
 "crossdataset_advhead|crossdataset_advhead_strecha|uesfm_27scenes_adaptive|uesfm_deep_advhead|esfm_outliers_deep|20"
 "crossdataset_advheadttt|crossdataset_advheadttt_strecha|uesfm_27scenes_adaptive|uesfm_deep_advheadttt|esfm_outliers_deep|20"
 "crossdataset_advweight|crossdataset_advweight_strecha|uesfm_27scenes_adaptive|uesfm_deep_advweight|esfm_outliers_deep|20"
 "crossdataset_eval|crossdataset_eval_strecha|uesfm_27scenes_adaptive_p3070|uesfm_deep_adaptive_p3070|esfm_outliers_deep|20 21 22 23 24"
 "crossdataset_shallow_adaptive|crossdataset_shallow_adaptive_strecha_ftlr1e4|uesfm_27scenes_shallow_adaptive_p3070|uesfm_shallow_adaptive_p3070|esfm_outliers_shallow|20 21 22 23 24"
 "crossdataset_shallow_adaptive|crossdataset_shallow_adaptive_strecha_ftlr1e4|uesfm_27scenes_shallow_adaptive_madlink|uesfm_shallow_adaptive_madlink|esfm_outliers_shallow|20 21 22 23 24"
 "crossdataset_eval|crossdataset_eval_strecha|uesfm_27scenes_adaptive_madlink|uesfm_deep_adaptive_madlink|esfm_outliers_deep|20 21 22 23 24"
 # These three auto-launch ONLY once their training reaches 20k (gate field 7 =
 # models_all/Model_Ep19999.pt); heal then keeps them healed to completion.
 "crossdataset_resfm_shallow|crossdataset_resfm_shallow_strecha|resfm_shallow_27scenes|resfm_shallow|esfm_outliers_shallow|20 21 22 23 24|models_all/Model_Ep19999.pt"
 "crossdataset_shallow_adaptive|crossdataset_shallow_adaptive_strecha_ftlr1e4|uesfm_27scenes_shallow_adaptive_p1090|uesfm_shallow_adaptive_p1090|esfm_outliers_shallow|20 21 22 23 24|models_all/Model_Ep19999.pt"
 "crossdataset_shallow_adaptive|crossdataset_shallow_adaptive_strecha_ftlr1e4|uesfm_27scenes_shallow_adaptive_p4060|uesfm_shallow_adaptive_p4060|esfm_outliers_shallow|20 21 22 23 24|models_all/Model_Ep19999.pt"
)
total=0
for row in "${FLEETS[@]}"; do
  IFS='|' read -r tp stp train rp arch seeds gate <<< "$row"
  gate="${gate:-models}"                                   # default gate = models dir
  [ -e "results/multiscene/$train/$gate" ] || continue     # training/checkpoint not ready yet
  for ds in 1dsfm strecha blendedmvs; do
    tmpl="confs/${tp}_${ds}.conf.template"; [ "$ds" = strecha ] && tmpl="confs/${stp}.conf.template"
    [ -f "$tmpl" ] || continue
    for seed in $seeds; do
      base="results/crossdataset/${rp}_${ds}_eval"; [ "$seed" != 20 ] && base="${base}_seed${seed}"
      todo=""
      for sc in $(scenes_for $ds); do
        ls "$base/${sc}_ba/Results_FINE_TUNE"*.xlsx >/dev/null 2>&1 && continue
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
done
echo "heal_eval_fleets: resubmitted $total cell(s)"
