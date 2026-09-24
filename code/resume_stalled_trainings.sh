#!/bin/bash
# Self-heal for the long from-scratch trainings that keep falling out of the
# waic-risk queue (heavy cross-project contention + babysitter retry cap).
# For each training: if it is NOT complete (no models_all/Model_Ep19999.pt) AND
# has no job in the queue, resume it (resume=true continues from its latest
# checkpoint) with a fresh babysitter. Idempotent per cycle.
set -u
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO"
PY="$REPO/../.venv38-resfm/bin/python"
QUEUE="waic-risk"   # policy 2026-09-06: ALL jobs -> waic-risk (user directive; supersedes the Sep-5 medium episodes)
QNAMES=$(bjobs -w 2>/dev/null | awk '{print $7}')
# Congestion guard: under heavy load bjobs can return nothing even though jobs
# exist; treating that as "empty queue" caused mass duplicate resubmission
# (2026-08-26). If we list zero jobs but LSF also can't confirm the queue is
# reachable, skip this cycle instead of resubmitting everything.
if [ -z "$QNAMES" ]; then
  bqueues "$QUEUE" >/dev/null 2>&1 || { echo "resume: bjobs empty and LSF unreachable -- skipping cycle"; exit 0; }
fi

# exp | conf | resdir
JOBS=(
 "resfm_shallow_ms27|confs/multiscene_resfm_shallow.conf|resfm_shallow_27scenes"
 "uesfm_shallow_adaptive_p1090|confs/multiscene_uesfm_shallow_adaptive_p1090.conf|uesfm_27scenes_shallow_adaptive_p1090"
 "uesfm_shallow_adaptive_p4060|confs/multiscene_uesfm_shallow_adaptive_p4060.conf|uesfm_27scenes_shallow_adaptive_p4060"
 "uesfm_adaptive_reportfaithful|confs/multiscene_uesfm_adaptive_reportfaithful.conf|uesfm_27scenes_adaptive_reportfaithful"
 "uesfm_shallow_adaptive_reportfaithful|confs/multiscene_uesfm_shallow_adaptive_reportfaithful.conf|uesfm_27scenes_shallow_adaptive_reportfaithful"
 "uesfm_sa_rf_p1090|confs/multiscene_uesfm_shallow_adaptive_reportfaithful_p1090.conf|uesfm_27scenes_shallow_adaptive_reportfaithful_p1090"
 "uesfm_sa_rf_p1585|confs/multiscene_uesfm_shallow_adaptive_reportfaithful_p1585.conf|uesfm_27scenes_shallow_adaptive_reportfaithful_p1585"
 "uesfm_sa_rf_p2575|confs/multiscene_uesfm_shallow_adaptive_reportfaithful_p2575.conf|uesfm_27scenes_shallow_adaptive_reportfaithful_p2575"
 "uesfm_sa_rf_p3070|confs/multiscene_uesfm_shallow_adaptive_reportfaithful_p3070.conf|uesfm_27scenes_shallow_adaptive_reportfaithful_p3070"
 "resfm_shallow_finelr|confs/multiscene_resfm_shallow_finelr.conf|resfm_shallow_27scenes_finelr"
 "uesfm_sa_rf_finelr|confs/multiscene_uesfm_shallow_adaptive_reportfaithful_finelr.conf|uesfm_27scenes_shallow_adaptive_reportfaithful_finelr"
 "uesfm_sa_rf_finelr_wd1e4|confs/multiscene_uesfm_sa_rf_finelr_wd1e4.conf|uesfm_27scenes_shallow_adaptive_reportfaithful_finelr_wd1e4"
 "uesfm_sa_rf_finelr_wd1e3|confs/multiscene_uesfm_sa_rf_finelr_wd1e3.conf|uesfm_27scenes_shallow_adaptive_reportfaithful_finelr_wd1e3"
 "uesfm_sa_rf_s21|confs/multiscene_uesfm_shallow_adaptive_reportfaithful_s21.conf|uesfm_27scenes_shallow_adaptive_reportfaithful_s21"
 "uesfm_sa_rf_s22|confs/multiscene_uesfm_shallow_adaptive_reportfaithful_s22.conf|uesfm_27scenes_shallow_adaptive_reportfaithful_s22"
 "uesfm_sa_rf_finelr_s21|confs/multiscene_uesfm_shallow_adaptive_reportfaithful_finelr_s21.conf|uesfm_27scenes_shallow_adaptive_reportfaithful_finelr_s21"
 "uesfm_sa_rf_finelr_s22|confs/multiscene_uesfm_shallow_adaptive_reportfaithful_finelr_s22.conf|uesfm_27scenes_shallow_adaptive_reportfaithful_finelr_s22"
 "uesfm_sa_rf_multids|confs/multiscene_uesfm_sa_rf_multids.conf|uesfm_multids_shallow_adaptive_reportfaithful"
 "resfm_shallow_multids|confs/multiscene_resfm_shallow_multids.conf|resfm_shallow_multids"
 "uesfm_sa_rf_multids_v2|confs/multiscene_uesfm_sa_rf_multids_v2.conf|uesfm_multids_v2_shallow_adaptive_reportfaithful"
 "resfm_shallow_multids_v2|confs/multiscene_resfm_shallow_multids_v2.conf|resfm_shallow_multids_v2"
 "uesfm_deep_multids_v2|confs/multiscene_uesfm_deep_multids_v2.conf|uesfm_multids_v2_deep_adaptive_reportfaithful"
 "resfm_deep_multids_v2|confs/multiscene_resfm_deep_multids_v2.conf|resfm_deep_multids_v2"
 "resfm_shallow_faithful_lr1e4|confs/multiscene_resfm_shallow_faithful_lr1e4.conf|resfm_shallow_27scenes_faithful_lr1e4"
 "resfm_finelr_lr1e4|confs/multiscene_resfm_shallow_finelr_lr1e4.conf|resfm_shallow_27scenes_finelr_lr1e4"
 "esfm_star|confs/multiscene_esfm_star.conf|esfm_star_27scenes"
 "resfm_shallow_finelr_s23|confs/multiscene_resfm_shallow_finelr_s23.conf|resfm_shallow_27scenes_finelr_s23"
 "uesfm_sa_rf_finelr_s23|confs/multiscene_uesfm_sa_rf_finelr_s23.conf|uesfm_27scenes_sa_rf_finelr_s23"
 "resfm_shallow_finelr_s24|confs/multiscene_resfm_shallow_finelr_s24.conf|resfm_shallow_27scenes_finelr_s24"
 "uesfm_sa_rf_finelr_s24|confs/multiscene_uesfm_sa_rf_finelr_s24.conf|uesfm_27scenes_sa_rf_finelr_s24"
 "resfm_shallow_finelr_s21|confs/multiscene_resfm_shallow_finelr_s21.conf|resfm_shallow_27scenes_finelr_s21"
 "resfm_shallow_finelr_s22|confs/multiscene_resfm_shallow_finelr_s22.conf|resfm_shallow_27scenes_finelr_s22"
 "resfm_multids_v2_af|confs/multiscene_resfm_multids_v2_af.conf|resfm_multids_v2_af"
 "resfm_multids_v2_af_lr1e4|confs/multiscene_resfm_multids_v2_af_lr1e4.conf|resfm_multids_v2_af_lr1e4"
 "uesfm_multids_v2_af|confs/multiscene_uesfm_multids_v2_af.conf|uesfm_multids_v2_af"
 "resfm_multids_v2_af_s21|confs/multiscene_resfm_multids_v2_af_s21.conf|resfm_multids_v2_af_s21"
 "resfm_multids_v2_af_lr1e4_s21|confs/multiscene_resfm_multids_v2_af_lr1e4_s21.conf|resfm_multids_v2_af_lr1e4_s21"
 "uesfm_multids_v2_af_s21|confs/multiscene_uesfm_multids_v2_af_s21.conf|uesfm_multids_v2_af_s21"
 "resfm_multids_v2_af_s22|confs/multiscene_resfm_multids_v2_af_s22.conf|resfm_multids_v2_af_s22"
 "resfm_multids_v2_af_lr1e4_s22|confs/multiscene_resfm_multids_v2_af_lr1e4_s22.conf|resfm_multids_v2_af_lr1e4_s22"
 "uesfm_multids_v2_af_s22|confs/multiscene_uesfm_multids_v2_af_s22.conf|uesfm_multids_v2_af_s22"
 "resfm_shallow_faithful|confs/multiscene_resfm_shallow_faithful.conf|resfm_shallow_27scenes_faithful"
 "resfm_deep_multids_v2_ln|confs/multiscene_resfm_deep_multids_v2_ln.conf|resfm_deep_multids_v2_ln"
 "resfm_finelr_lr1e4_30k|confs/multiscene_resfm_shallow_finelr_lr1e4_30k.conf|resfm_shallow_27scenes_finelr_lr1e4_30k|29999"
 "uesfm_sa_rf_lr1e3|confs/multiscene_uesfm_sa_rf_lr1e3.conf|uesfm_27scenes_sa_rf_lr1e3"
 "esfm_star_af|confs/multiscene_esfm_star_af.conf|esfm_star_af_27scenes"
 "resfm_faithful_s21|confs/multiscene_resfm_shallow_faithful_s21.conf|resfm_shallow_27scenes_faithful_s21"
 "resfm_faithful_s22|confs/multiscene_resfm_shallow_faithful_s22.conf|resfm_shallow_27scenes_faithful_s22"
 "resfm_faithful_s23|confs/multiscene_resfm_shallow_faithful_s23.conf|resfm_shallow_27scenes_faithful_s23"
 "resfm_faithful_s24|confs/multiscene_resfm_shallow_faithful_s24.conf|resfm_shallow_27scenes_faithful_s24"
 "resfm_faithful_selfclean|confs/multiscene_resfm_shallow_faithful_selfclean.conf|resfm_shallow_27scenes_faithful_selfclean"
 "resfm_faithful_selfclean_s21|confs/multiscene_resfm_shallow_faithful_selfclean_s21.conf|resfm_shallow_27scenes_faithful_selfclean_s21"
 "resfm_faithful_selfclean_s22|confs/multiscene_resfm_shallow_faithful_selfclean_s22.conf|resfm_shallow_27scenes_faithful_selfclean_s22"
 "resfm_faithful_selfclean_s23|confs/multiscene_resfm_shallow_faithful_selfclean_s23.conf|resfm_shallow_27scenes_faithful_selfclean_s23"
 "resfm_faithful_selfclean_s24|confs/multiscene_resfm_shallow_faithful_selfclean_s24.conf|resfm_shallow_27scenes_faithful_selfclean_s24"
 "resfm_deep_selfclean|confs/multiscene_resfm_deep_selfclean.conf|resfm_deep_27scenes_selfclean"
 "resfm_multids_selfclean|confs/multiscene_resfm_multids_selfclean.conf|resfm_shallow_multids_v2_selfclean"
 "resfm_multids_deep_selfclean|confs/multiscene_resfm_multids_deep_selfclean.conf|resfm_deep_multids_v2_ln_selfclean"
 "rf10k_s20|confs/multiscene_resfm_faithful_10k_s20.conf|resfm_shallow_27scenes_faithful_10k_s20|9999"
 "rf10k_s21|confs/multiscene_resfm_faithful_10k_s21.conf|resfm_shallow_27scenes_faithful_10k_s21|9999"
 "rf10k_s22|confs/multiscene_resfm_faithful_10k_s22.conf|resfm_shallow_27scenes_faithful_10k_s22|9999"
 "rf10k_s23|confs/multiscene_resfm_faithful_10k_s23.conf|resfm_shallow_27scenes_faithful_10k_s23|9999"
 "rf10k_s24|confs/multiscene_resfm_faithful_10k_s24.conf|resfm_shallow_27scenes_faithful_10k_s24|9999"
 "sc10k_s20|confs/multiscene_resfm_selfclean_10k_s20.conf|resfm_shallow_27scenes_selfclean_10k_s20|9999"
 "sc10k_s21|confs/multiscene_resfm_selfclean_10k_s21.conf|resfm_shallow_27scenes_selfclean_10k_s21|9999"
 "sc10k_s22|confs/multiscene_resfm_selfclean_10k_s22.conf|resfm_shallow_27scenes_selfclean_10k_s22|9999"
 "sc10k_s23|confs/multiscene_resfm_selfclean_10k_s23.conf|resfm_shallow_27scenes_selfclean_10k_s23|9999"
 "sc10k_s24|confs/multiscene_resfm_selfclean_10k_s24.conf|resfm_shallow_27scenes_selfclean_10k_s24|9999"
 "uesfm_1dsfmid_s20|confs/multiscene_uesfm_1dsfmid_s20.conf|uesfm_1dsfmid_s20"
 "uesfm_1dsfmid_s21|confs/multiscene_uesfm_1dsfmid_s21.conf|uesfm_1dsfmid_s21"
 "resfm_1dsfmid_s20|confs/multiscene_resfm_1dsfmid_s20.conf|resfm_1dsfmid_s20"
 "resfm_1dsfmid_s21|confs/multiscene_resfm_1dsfmid_s21.conf|resfm_1dsfmid_s21"
 "uesfm_bmvsid_s20|confs/multiscene_uesfm_bmvsid_s20.conf|uesfm_bmvsid_s20"
 "uesfm_strechaid_s20|confs/multiscene_uesfm_strechaid_s20.conf|uesfm_strechaid_s20"
 "uesfm_olssonid_s20|confs/multiscene_uesfm_olssonid_s20.conf|uesfm_olssonid_s20"
 "uesfm_bmvsid_s21|confs/multiscene_uesfm_bmvsid_s21.conf|uesfm_bmvsid_s21"
 "uesfm_strechaid_s21|confs/multiscene_uesfm_strechaid_s21.conf|uesfm_strechaid_s21"
 "uesfm_olssonid_s21|confs/multiscene_uesfm_olssonid_s21.conf|uesfm_olssonid_s21"
 "resfm_bmvsid_s20|confs/multiscene_resfm_bmvsid_s20.conf|resfm_bmvsid_s20"
 "resfm_strechaid_s20|confs/multiscene_resfm_strechaid_s20.conf|resfm_strechaid_s20"
 "resfm_olssonid_s20|confs/multiscene_resfm_olssonid_s20.conf|resfm_olssonid_s20"
 "resfm_bmvsid_s21|confs/multiscene_resfm_bmvsid_s21.conf|resfm_bmvsid_s21"
 "resfm_strechaid_s21|confs/multiscene_resfm_strechaid_s21.conf|resfm_strechaid_s21"
 "resfm_olssonid_s21|confs/multiscene_resfm_olssonid_s21.conf|resfm_olssonid_s21"
 "uesfm_1donly_s20|confs/multiscene_uesfm_1donly_s20.conf|uesfm_1donly_s20"
 "uesfm_hardonly_s20|confs/multiscene_uesfm_hardonly_s20.conf|uesfm_hardonly_s20"
 "uesfm_1donly_s21|confs/multiscene_uesfm_1donly_s21.conf|uesfm_1donly_s21"
 "uesfm_hardonly_s21|confs/multiscene_uesfm_hardonly_s21.conf|uesfm_hardonly_s21"
 "resfm_1donly_s20|confs/multiscene_resfm_1donly_s20.conf|resfm_1donly_s20"
 "resfm_hardonly_s20|confs/multiscene_resfm_hardonly_s20.conf|resfm_hardonly_s20"
 "resfm_1donly_s21|confs/multiscene_resfm_1donly_s21.conf|resfm_1donly_s21"
 "resfm_hardonly_s21|confs/multiscene_resfm_hardonly_s21.conf|resfm_hardonly_s21"
)
for row in "${JOBS[@]}"; do
  IFS='|' read -r exp conf resdir final ngpu pyexe <<< "$row"
  final=${final:-19999}                                                          # per-job final epoch (default 20k runs)
  ngpu=${ngpu:-1}                                                                # per-job GPU count (Fabric DDP shards scenes across ranks)
  pyexe=${pyexe:-$PY}                                                             # per-job python (matched-env runs)
  [ -e "results/multiscene/$resdir/models_all/Model_Ep${final}.pt" ] && continue # done
  echo "$QNAMES" | grep -qx "$exp" && continue                                    # already queued/running
  SUBQ="$QUEUE"
  MEMR=64000; [ "$ngpu" -gt 1 ] && MEMR=60000                              # multi-GPU: ~39-45G per rank (each rank loads the full dataset)
  SUB=$(bsub -q "$SUBQ" -J "$exp" -n "$ngpu" -R "affinity[core(4)]" -oo "lsf_output/multiscene/${exp}_%J.out" -eo "lsf_output/multiscene/${exp}_%J.err" \
    -gpu "num=${ngpu}:j_exclusive=yes:gmem=80G" -R "rusage[mem=${MEMR}]" -R "span[hosts=1]" \
    "cd $REPO; TORCHDYNAMO_DISABLE=1 $pyexe multiple_scenes_learning.py --conf $conf --phase TRAINING --exp_version $exp --wandb 1")
  JID=$(echo "$SUB" | grep -oE "Job <[0-9]+>" | grep -oE "[0-9]+")
  [ -z "$JID" ] && { echo "resume $exp: FAILED to submit"; continue; }
  bsub -q "$QUEUE" -gpu "num=1:j_exclusive=yes:gmem=80G" -J "${exp}_trigger" -w "ended(${JID})" \
    -oo "lsf_output/multiscene/${exp}_trigger_%J.out" -eo "lsf_output/multiscene/${exp}_trigger_%J.err" \
    "bash $REPO/trigger_ablation_resume.sh $conf $exp results/multiscene/$resdir 0 $JID $final $ngpu $pyexe" >/dev/null
  echo "resume $exp -> $JID (+fresh babysitter)"
done
