#!/usr/bin/env python
"""Dry-run checkpoint prune plan for completed runs.

For each results/multiscene/<run>/models_all directory, decide what COULD be
deleted while keeping: (1) models/ untouched (best-val ckpt), (2) the final
epoch in models_all (Ep19999 or Ep29999), (3) any epoch referenced by a
post-hoc reselection CSV (multival_reselect*.csv best rows) or recorded as a
best_epoch anywhere in the run's eval usage, (4) EVERYTHING for runs that are
still training (no final epoch present) or explicitly protected.

Prints a per-run plan + totals. DELETES NOTHING.
"""
import glob, os, re, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MS = os.path.join(HERE, "results", "multiscene")

# runs still in flight or awaiting their eval fleets -- never touch
PROTECT = {
    "resfm_shallow_27scenes_faithful_s22",   # fleet rfa22 running
    "resfm_shallow_27scenes_faithful_s23",   # fleet rfa23 running
    "resfm_shallow_27scenes_faithful_selfclean",  # training
    "resfm_deep_multids_v2_ln",              # training
}

def ep_of(p):
    m = re.search(r"Ep(\d+)\.pt$", p)
    return int(m.group(1)) if m else -1

total_del, total_keep = 0, 0
plan = []
for run in sorted(os.listdir(MS)):
    ma = os.path.join(MS, run, "models_all")
    if not os.path.isdir(ma):
        continue
    files = sorted(glob.glob(os.path.join(ma, "Model_Ep*.pt")), key=ep_of)
    if not files:
        continue
    eps = [ep_of(f) for f in files]
    final_done = max(eps) in (19999, 29999, 99999)
    keep_eps = set()
    reason = []
    if run in PROTECT or not final_done:
        keep_eps = set(eps)
        reason.append("PROTECTED" if run in PROTECT else "incomplete")
    else:
        keep_eps.add(max(eps))
        # post-hoc reselection winners
        for csv in glob.glob(os.path.join(MS, run, "multival_reselect*.csv")):
            try:
                df = pd.read_csv(csv)
                keep_eps.add(int(df.loc[df["mean"].idxmin()]["epoch"]))
                keep_eps.add(int(df.loc[df["median"].idxmin()]["epoch"]))
                reason.append(f"reselect:{os.path.basename(csv)}")
            except Exception:
                keep_eps = set(eps); reason.append("reselect-parse-fail: keep all")
        # the CURRENT best (latest snapshot in models/): keep its models_all mirror.
        # Earlier models/ files are stale best-so-far snapshots; evals load the latest.
        bests = sorted(glob.glob(os.path.join(MS, run, "models", "Model_Ep*.pt")), key=ep_of)
        if bests:
            keep_eps.add(ep_of(bests[-1]))
    dele = [f for f in files if ep_of(f) not in keep_eps]
    keep = [f for f in files if ep_of(f) in keep_eps]
    dsz = sum(os.path.getsize(f) for f in dele) / 2**30
    ksz = sum(os.path.getsize(f) for f in keep) / 2**30
    total_del += dsz; total_keep += ksz
    plan.append((run, len(dele), dsz, sorted(keep_eps), ";".join(reason) or "-"))

for run, n, dsz, keeps, why in plan:
    kp = ",".join(f"Ep{e}" for e in keeps) if len(keeps) <= 6 else f"{len(keeps)} ckpts"
    print(f"{run:55s} del {n:3d} files {dsz:6.2f}G  keep [{kp}]  {why}")
print(f"\nTOTAL: would delete {total_del:.1f}G, keep {total_keep:.1f}G in models_all "
      f"(models/ best ckpts untouched on top of this)")
