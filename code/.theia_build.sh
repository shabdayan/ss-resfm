#!/bin/bash
# Build TheiaSfM for the classical-baseline parity study.
set -x
ROOT=/home/projects/bagon/ortalda/MVG/final-project/u-esfm/tools/theia
mkdir -p $ROOT && cd $ROOT
# 1) micromamba bootstrap (static binary)
if [ ! -x $ROOT/micromamba ]; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba
  mv bin/micromamba $ROOT/micromamba; rmdir bin 2>/dev/null
fi
export MAMBA_ROOT_PREFIX=$ROOT/mm
# 2) deps env
if [ ! -d $ROOT/mm/envs/theia ]; then
  $ROOT/micromamba create -y -n theia -c conda-forge cmake=3.28 eigen=3.4 ceres-solver=2.1 glog gflags openimageio=2.4 rapidjson rocksdb suitesparse make
fi
ENV=$ROOT/mm/envs/theia
$ROOT/micromamba install -y -n theia -c conda-forge rocksdb suitesparse 2>&1 | tail -1
# 3) source
if [ ! -d $ROOT/TheiaSfM ]; then
  git clone --depth 1 https://github.com/sweeneychris/TheiaSfM.git
fi
cd TheiaSfM && mkdir -p build && cd build
$ENV/bin/cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
  -DCMAKE_PREFIX_PATH=$ENV -DCMAKE_CXX_STANDARD=14 \
  -DROCKSDB_INCLUDE_DIR=$ENV/include -DROCKSDB_LIBRARIES=$ENV/lib/librocksdb.so \
  -DCMAKE_CXX_FLAGS="-Wno-error -fpermissive" 2>&1 | tail -20
$ENV/bin/make -j8 build_reconstruction 2>&1 | tail -30
ls -la bin/ 2>/dev/null | head
echo BUILD_SCRIPT_DONE
