#!/bin/bash
# B00 (SIFT control) per-observation feature extraction on 1DSfM images.
# 12 std scenes (Notre_Dame excluded: no public images) + 5 hard scenes.
CODE=/home/projects/bagon/ortalda/MVG/final-project/u-esfm/code
PY=$CODE/../.venv38-resfm/bin/python
mkdir -p $CODE/lsf_output/b00
sub() {
  ds=$1; scene=$2
  bsub -q waic-risk -J "b00_${ds}_${scene}" \
    -oo "$CODE/lsf_output/b00/${ds}_${scene}_%J.out" \
    -eo "$CODE/lsf_output/b00/${ds}_${scene}_%J.err" \
    -R "rusage[mem=32000]" \
    "cd $CODE && $PY extract_obs_features.py $scene --dataset $ds --source sift"
}
for S in Alamo Ellis_Island Gendarmenmarkt Madrid_Metropolis Montreal_Notre_Dame \
         NYC_Library Piazza_del_Popolo Roman_Forum Tower_of_London Trafalgar \
         Vienna_Cathedral Yorkminster; do
  sub 1dsfm $S
done
for S in Gendarmenmarkt Piccadilly Roman_Forum Trafalgar Union_Square; do
  sub 1dsfm_hard_300 $S
done
