#!/bin/bash
# Head-classification-threshold sweep for the REMOVE mechanism (RESfM analogue).
# Sweeps test.outliers_threshold in {0.4,0.5,0.6,0.7,0.8} on two checkpoints:
#   - 20/80 report-faithful  (principled anchor + Path B threshold-vs-contamination scout)
#   - p3070 (30/70)          (MegaDepth-favorable head; best shot at the MegaDepth gap)
# across all 4 datasets, seed 20. Test-time only (no retraining).
set -u
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO"
PY38="$REPO/../.venv38-resfm/bin/python"
TSWEEP="confs/threshsweep"; mkdir -p "$TSWEEP"
S1DS="Alamo,Ellis_Island,Madrid_Metropolis,Montreal_Notre_Dame,Notre_Dame,NYC_Library,Piazza_del_Popolo,Tower_of_London,Vienna_Cathedral,Yorkminster"
SSTR="entry-P10,fountain-P11,Herz-Jesu-P8,Herz-Jesu-P25"
SBMV="58c4bb4f4a69c55606122be4,5acf8ca0f3d8a750097e4b15,5a48ba95c7dab83a7d7b44ed,5b950c71608de421b1e7318f"
MD="0238,0060,0197,0094,0265,0083,0076,0185,0048,0024,0223,5016,0046,0099,1001,0231,0411,0377,0102,0147,0148,0446,0022,0327,0015,0455,0496,1589,0012,0104,0019,0063,0130,0080,0240,0007"
scenes_for(){ case "$1" in 1dsfm) echo "$S1DS";; strecha) echo "$SSTR";; blendedmvs) echo "$SBMV";; megadepth) echo "$MD";; esac; }
tmpl_for(){ if [ "$1" = megadepth ]; then echo "confs/megadepth_rf_remove_shallow.conf.template"; else echo "confs/crossdataset_rf_remove_shallow_$1.conf.template"; fi; }
# ckpttag | train_results
CKPTS=(
 "2080|results/multiscene/uesfm_27scenes_shallow_adaptive_reportfaithful"
 "3070|results/multiscene/uesfm_27scenes_shallow_adaptive_reportfaithful_p3070"
)
total=0
ALLRUN=$(bjobs -w 2>/dev/null)
for row in "${CKPTS[@]}"; do
  IFS='|' read -r ck train <<< "$row"
  [ -e "$train/models_all/Model_Ep19999.pt" ] || { echo "  $ck checkpoint not ready (skip)"; continue; }
  for thr in 0.4 0.5 0.6 0.7 0.8; do
    thrtag="${thr/./}"   # 0.4 -> 04
    for ds in 1dsfm strecha blendedmvs megadepth; do
      src="$(tmpl_for $ds)"
      [ -f "$src" ] || { echo "  missing $src"; continue; }
      tmp="$TSWEEP/$(basename "$src" .conf.template)_${ck}_thr${thrtag}.conf.template"
      sed -E "s|outliers_threshold = .*|outliers_threshold = ${thr}|" "$src" > "$tmp"
      if [ "$ds" = megadepth ]; then
        root="results/multiscene/uesfm_shallow_${ck}_rmthr${thrtag}_${ds}_eval"
      else
        root="results/crossdataset/uesfm_shallow_${ck}_rmthr${thrtag}_${ds}_eval"
      fi
      jp="rt_${ck}_${thrtag}_${ds}"
      # dedup: skip scenes already done or currently running under this jobprefix
      donelist=$(ls "$root"/*_ba/Results_FINE_TUNE*.xlsx 2>/dev/null)
      runlist=$(echo "$ALLRUN" | grep -F "$jp" | awk '{print $7}')
      todo=""
      for sc in $(echo "$(scenes_for $ds)" | tr ',' ' '); do
        echo "$donelist" | grep -q "/${sc}_ba/" && continue
        echo "$runlist" | grep -q "_${sc}$" && continue
        todo="$todo,$sc"
      done
      todo="${todo#,}"; [ -z "$todo" ] && continue
      EVAL_PYTHON="$PY38" ./run_multiscene_eval.sh --template "$tmp" \
        --train_results "$train" --eval_root "$root" \
        --arch esfm_outliers_shallow --scans "$todo" --seed 20 --jobprefix "$jp" >/dev/null 2>&1 \
        && { c=$(echo "$todo" | tr ',' '\n' | wc -l); total=$((total+c)); echo "  $jp: $todo"; }
    done
  done
done
echo "thresh_sweep launched: $total scene-cell(s)"
