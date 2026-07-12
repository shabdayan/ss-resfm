#!/bin/bash
# LSF launcher for the single-scene ESFM vs U-ESFM comparison
# (SPEC_single_scene_experiments.md). One bsub job per SCENE: inside each job,
# all (seed x method) runs for that scene execute strictly sequentially on the
# job's single GPU, so each esfm/u-esfm comparison shares identical hardware.
#
# Usage:
#   ./run_single_scene_bench.sh --smoke                # 1 scene, 100 epochs, seed 0
#   ./run_single_scene_bench.sh                        # full: all scenes, seeds 0,1,2
#   ./run_single_scene_bench.sh --scenes "Gustav_Vasa,Door_Lund" --seeds 0
#   ./run_single_scene_bench.sh --dry-run              # print bsub commands only
set -u

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${REPO_ROOT}/../.venv/bin/python"
QUEUE="waic-risk"
SEEDS="0,1,2"
EPOCHS=100000
EVAL_INTERVALS=5000
SCENES=""            # empty = all 36 Olsson scenes
DRY_RUN=false
SMOKE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke) SMOKE=true; shift ;;
        --scenes) SCENES="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --eval-intervals) EVAL_INTERVALS="$2"; shift 2 ;;
        --queue) QUEUE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ "$SMOKE" = true ]; then
    SCENES="${SCENES:-Drinking_Fountain_Somewhere_In_Zurich}"
    SEEDS="0"
    EPOCHS=100
    EVAL_INTERVALS=20
fi

if [ -z "$SCENES" ]; then
    # all 36 Olsson scene slugs, from run_single_scene_sweep.py
    SCENES=$("$PY" - <<'EOF'
import run_single_scene_sweep as s
print(','.join(s.slug(x) for x in s.OLSSON_SCENES))
EOF
    )
fi

mkdir -p "${REPO_ROOT}/lsf_output/single_scene"

IFS=',' read -ra SCENE_ARR <<< "$SCENES"
for SCENE in "${SCENE_ARR[@]}"; do
    CMD="cd ${REPO_ROOT} && ${PY} run_single_scene_sweep.py \
        --scenes '${SCENE}' --seeds ${SEEDS} \
        --epochs ${EPOCHS} --eval-intervals ${EVAL_INTERVALS}"
    if [ "$DRY_RUN" = true ]; then
        echo "DRY RUN: bsub -q ${QUEUE} -J ss_${SCENE} ... \"${CMD}\""
        continue
    fi
    bsub -q "${QUEUE}" \
        -J "ss_${SCENE}" \
        -oo "${REPO_ROOT}/lsf_output/single_scene/${SCENE}_%J.out" \
        -eo "${REPO_ROOT}/lsf_output/single_scene/${SCENE}_%J.err" \
        -gpu "num=1:j_exclusive=yes:gmem=80G" \
        -R "rusage[mem=50000]" \
        "${CMD}"
done

if [ "$DRY_RUN" = false ]; then
    echo ""
    echo "Submitted ${#SCENE_ARR[@]} scene job(s) to ${QUEUE} (seeds: ${SEEDS}, epochs: ${EPOCHS})."
    echo "Runs are resumable: resubmitting after preemption skips completed (method,scene,seed)."
    echo "Then: ${PY} evaluate_single_scene.py [--ba]  &&  ${PY} aggregate_single_scene.py"
fi
