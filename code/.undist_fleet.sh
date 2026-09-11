#!/bin/bash
CODE=/home/projects/bagon/ortalda/MVG/final-project/u-esfm/code
PY=$CODE/../.venv38-resfm/bin/python
for f in $CODE/datasets/megadepth/*.npz; do
  S=$(basename $f .npz)
  [ "$S" = "0007" ] && continue
  [ -d $CODE/datasets/raw_megadepth/MegaDepth_SfM/$S ] || { echo "no raw: $S"; continue; }
  [ -d $CODE/datasets/raw_megadepth/undistorted/$S ] && { echo "done: $S"; continue; }
  bsub -q waic-risk -J "undist_$S" \
    -oo $CODE/lsf_output/b00/undist_${S}_%J.out -eo $CODE/lsf_output/b00/undist_${S}_%J.err \
    -R "rusage[mem=32000]" "cd $CODE && $PY undistort_md_scene.py $S"
done
