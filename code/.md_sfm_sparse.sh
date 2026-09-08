#!/bin/bash
# Second pass over MegaDepth_SfM_v1.tar.xz: extract per-scene EVERYTHING EXCEPT
# images (sparse COLMAP models + metadata) for the 68 scenes. Needed to map
# between raw-Flickr and undistorted (npz) coordinate frames.
cd /home/projects/bagon/ortalda/MVG/final-project/u-esfm/code
TAR=datasets/raw_megadepth/md_sfm_v1.tar.xz
EXPECTED=666570101304
URL=https://www.cs.cornell.edu/projects/megadepth/dataset/MegaDepth_SfM/MegaDepth_SfM_v1.tar.xz
SCENES="0007 0012 0015 0019 0022 0024 0041 0046 0047 0048 0058 0060 0062 0063 0076 0080 0083 0094 0099 0102 0104 0107 0130 0147 0148 0156 0175 0176 0181 0185 0186 0197 0204 0205 0214 0217 0223 0229 0231 0238 0240 0265 0271 0275 0285 0286 0290 0299 0303 0327 0349 0360 0377 0411 0412 0446 0455 0474 0476 0478 0496 0733 0768 0860 1001 1589 5015 5016"
if [ "$(stat -c%s $TAR 2>/dev/null)" != "$EXPECTED" ]; then
  for i in $(seq 1 400); do
    curl -sL -C - --retry 20 --retry-delay 15 -o $TAR "$URL" && break
    sleep 30
  done
fi
SZ=$(stat -c%s $TAR 2>/dev/null); echo "DOWNLOAD size=$SZ expected=$EXPECTED"
[ "$SZ" != "$EXPECTED" ] && { echo "DOWNLOAD INCOMPLETE"; exit 1; }
if [ ! -f datasets/raw_megadepth/.sfm_sparse_done ]; then
  ARGS=""
  for s in $SCENES; do ARGS="$ARGS --wildcards */$s/*"; done
  mkdir -p datasets/raw_megadepth/sfm_meta
  tar -xJf $TAR -C datasets/raw_megadepth/sfm_meta --exclude='*/images/*' $ARGS 2> .md_sfm_sparse.err
  echo "EXTRACT rc=$?"
  touch datasets/raw_megadepth/.sfm_sparse_done
fi
find datasets/raw_megadepth/sfm_meta -maxdepth 4 -type d | head -20
du -sh datasets/raw_megadepth/sfm_meta
rm -f $TAR
echo SFM_SPARSE_DONE
