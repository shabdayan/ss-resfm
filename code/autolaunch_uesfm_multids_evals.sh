#!/bin/bash
# Fire the U-ESFM multids eval fleets the moment each training completes.
# Arms mirror the paper's complementarity protocol: madweight (high-contam story)
# + weight+TTT (clean/in-dist story) across the five test sets.
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO"
PY="$REPO/../.venv38-resfm/bin/python"
MD="0238 0060 0197 0094 0265 0083 0076 0185 0048 0024 0223 5016 0046 0099 1001 0231 0411 0377 0102 0147 0148 0446 0022 0327 0015 0455 0496 1589 0012 0104 0019 0063 0130 0080 0240 0007"
DS="Alamo Ellis_Island Madrid_Metropolis Montreal_Notre_Dame Notre_Dame NYC_Library Piazza_del_Popolo Tower_of_London Vienna_Cathedral Yorkminster"
ST="entry-P10 fountain-P11 Herz-Jesu-P25 Herz-Jesu-P8"
BM=$(ls datasets/blendedmvs/*.npz | xargs -n1 basename | sed 's/.npz//' | tr '\n' ' ')
HD="Gendarmenmarkt Piccadilly Roman_Forum Trafalgar Union_Square"
launch(){ EVAL_PYTHON="$PY" ./run_multiscene_eval.sh --template "$1" --train_results "results/multiscene/$2" --eval_root "$3" --arch esfm_outliers_shallow --scans "$4" --seed 20 --jobprefix "$5" >/dev/null 2>&1; }
fire(){ # $1=train resdir  $2=prefix
  d="$1"; p="$2"
  launch confs/crossdataset_rf_madweight_shallow_1dsfm.conf.template $d results/crossdataset/${d}_madweight_1dsfm_eval "$DS" ${p}m1
  launch confs/crossdataset_rf_madweight_shallow_1dsfmhard.conf.template $d results/crossdataset/${d}_madweight_1dsfmhard_eval "$HD" ${p}mh
  launch confs/crossdataset_rf_madweight_shallow_strecha.conf.template $d results/crossdataset/${d}_madweight_strecha_eval "$ST" ${p}ms
  launch confs/crossdataset_rf_madweight_shallow_blendedmvs.conf.template $d results/crossdataset/${d}_madweight_blendedmvs_eval "$BM" ${p}mb
  launch confs/megadepth_rf_weight_ttt_shallow.conf.template $d results/multiscene/${d}_wttt_megadepth_eval "$MD" ${p}w0
  launch confs/crossdataset_rf_weight_ttt_shallow_strecha.conf.template $d results/crossdataset/${d}_wttt_strecha_eval "$ST" ${p}ws
  launch confs/crossdataset_rf_weight_ttt_shallow_blendedmvs.conf.template $d results/crossdataset/${d}_wttt_blendedmvs_eval "$BM" ${p}wb
  echo "$(date '+%T') fleets launched for $d"
}
declare -A fired
while true; do
  for pair in "uesfm_multids_v2_shallow_adaptive_reportfaithful uv2" "uesfm_multids_shallow_adaptive_reportfaithful um1"; do
    set -- $pair; d=$1; p=$2
    if [ -z "${fired[$d]:-}" ] && [ -e "results/multiscene/$d/models_all/Model_Ep19999.pt" ]; then
      fired[$d]=1; fire "$d" "$p"
    fi
  done
  [ "${#fired[@]}" -ge 2 ] && { echo "ALL_FIRED"; exit 0; }
  sleep 300
done
