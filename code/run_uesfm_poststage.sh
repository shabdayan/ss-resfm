#!/bin/bash
# LSF launcher for the U-ESFM single-scene post-stage (uesfm_poststage.py):
# one bsub job per scene, running 3 seeds x {ft_learned, ft_mad, ttt} x {1k, 5k}
# sequentially on the job's GPU (18 short warm-start runs per job). Resumable.
#
#   ./run_uesfm_poststage.sh --smoke     # 1 scene, seed 0, all variants, 1k only
#   ./run_uesfm_poststage.sh             # all 36 scenes
set -u
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${REPO_ROOT}/../.venv/bin/python"
QUEUE="waic-risk"
SEEDS="0,1,2"
VARIANTS="ft_learned,ft_mad,ttt"
BUDGETS="1000,5000"
SCENES=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke) SCENES="Gustav_Vasa"; SEEDS="0"; BUDGETS="1000"; shift ;;
        --scenes) SCENES="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --variants) VARIANTS="$2"; shift 2 ;;
        --budgets) BUDGETS="$2"; shift 2 ;;
        --queue) QUEUE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$SCENES" ]; then
    SCENES=$("$PY" - <<'EOF'
import run_single_scene_sweep as s
print(','.join(s.slug(x) for x in s.OLSSON_SCENES))
EOF
    )
fi

mkdir -p "${REPO_ROOT}/lsf_output/single_scene_post"
IFS=',' read -ra SCENE_ARR <<< "$SCENES"
for SCENE in "${SCENE_ARR[@]}"; do
    CMD="cd ${REPO_ROOT} && ${PY} uesfm_poststage.py --scene '${SCENE}' \
        --seeds ${SEEDS} --variants ${VARIANTS} --budgets ${BUDGETS}"
    if [ "$DRY_RUN" = true ]; then echo "DRY RUN: ${CMD}"; continue; fi
    bsub -q "${QUEUE}" \
        -J "ssp_${SCENE}" \
        -oo "${REPO_ROOT}/lsf_output/single_scene_post/${SCENE}_%J.out" \
        -eo "${REPO_ROOT}/lsf_output/single_scene_post/${SCENE}_%J.err" \
        -gpu "num=1:j_exclusive=yes:gmem=80G" \
        -R "rusage[mem=50000]" \
        "${CMD}"
done
[ "$DRY_RUN" = false ] && echo "Submitted ${#SCENE_ARR[@]} post-stage job(s) to ${QUEUE}."
