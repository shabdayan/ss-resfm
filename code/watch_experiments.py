#!/usr/bin/env python3
"""Watcher for the in-flight U-ESFM experiments. Prints a compact status table,
the DELTA since the last run (newly-finished cells / trainings), collapse checks
for the learned-head PoC, and an ALL-COMPLETE banner. Stateful: persists last
counts to results/crossdataset/.watcher_state.json so each run surfaces only
what changed. Fast: counts files by glob, only reads xlsx for the PoC Nr check.

Usage: python watch_experiments.py
Exit code 42 == everything complete (loop can stop on it).
"""
import glob, json, os, subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
XD = "results/crossdataset"
STATE = os.path.join(XD, ".watcher_state.json")


def n_done(root_glob):
    return len(glob.glob(f"{root_glob}/*_ba/Results_FINE_TUNE*.xlsx"))


def eval_count(base_roots):
    """Count Results across seed-suffixed roots (base for seed20, _seed21.. else)."""
    tot = 0
    for b in base_roots:
        for s in [20, 21, 22, 23, 24]:
            tot += n_done(b if s == 20 else f"{b}_seed{s}")
    return tot


def ckpt_epoch(resdir):
    fs = sorted(glob.glob(f"results/multiscene/{resdir}/models/Model_Ep*.pt"),
                key=lambda p: int(''.join(filter(str.isdigit, os.path.basename(p)))))
    best = int(''.join(filter(str.isdigit, os.path.basename(fs[-1])))) if fs else 0
    done = os.path.exists(f"results/multiscene/{resdir}/models_all/Model_Ep19999.pt")
    return best, done


# ---- experiment definitions ----
EVALS = {
    "shallow-U-ESFM(MAD)": ([XD + "/esfm_vanilla_mad_1dsfm_eval",
                             XD + "/esfm_vanilla_mad_ftlr1e4_strecha_eval",
                             XD + "/esfm_vanilla_mad_blendedmvs_eval"], 90),
    "RESfM-deep 1DSfM": ([XD + "/resfm_deep_1dsfm_eval"], 50),
    "PoC advhead (s20)": ([XD + "/uesfm_deep_advhead_1dsfm_eval",
                           XD + "/uesfm_deep_advhead_strecha_eval",
                           XD + "/uesfm_deep_advhead_blendedmvs_eval"], 18),
    "PoC advhead+TTT (s20)": ([XD + "/uesfm_deep_advheadttt_1dsfm_eval",
                               XD + "/uesfm_deep_advheadttt_strecha_eval",
                               XD + "/uesfm_deep_advheadttt_blendedmvs_eval"], 18),
    "PoC advweight (s20)": ([XD + "/uesfm_deep_advweight_1dsfm_eval",
                             XD + "/uesfm_deep_advweight_strecha_eval",
                             XD + "/uesfm_deep_advweight_blendedmvs_eval"], 18),
    "deep 30/70 eval": ([XD + "/uesfm_deep_adaptive_p3070_%s_eval" % d
                         for d in ("1dsfm", "strecha", "blendedmvs")], 90),
    "shallow 30/70 eval": ([XD + "/uesfm_shallow_adaptive_p3070_%s_eval" % d
                            for d in ("1dsfm", "strecha", "blendedmvs")], 90),
    "shallow MADlink eval": ([XD + "/uesfm_shallow_adaptive_madlink_%s_eval" % d
                              for d in ("1dsfm", "strecha", "blendedmvs")], 90),
    "deep MADlink eval": ([XD + "/uesfm_deep_adaptive_madlink_%s_eval" % d
                           for d in ("1dsfm", "strecha", "blendedmvs")], 90),
    "U-ESFM-SA MegaDepth (s20)": (["results/multiscene/uesfm_shallow_adaptive_megadepth_eval"], 36),
    # auto-launch after their training completes (heal gates on Ep19999)
    "RESfM-shallow eval": ([XD + "/resfm_shallow_%s_eval" % d
                            for d in ("1dsfm", "strecha", "blendedmvs")], 90),
    "shallow 10/90 eval": ([XD + "/uesfm_shallow_adaptive_p1090_%s_eval" % d
                            for d in ("1dsfm", "strecha", "blendedmvs")], 90),
    "shallow 40/60 eval": ([XD + "/uesfm_shallow_adaptive_p4060_%s_eval" % d
                            for d in ("1dsfm", "strecha", "blendedmvs")], 90),
}
# PoC roots for the collapse (Nr) check
POC_ROOTS = {"advhead": [XD + "/uesfm_deep_advhead_%s_eval" % d for d in ("1dsfm", "strecha", "blendedmvs")],
             "advhead+TTT": [XD + "/uesfm_deep_advheadttt_%s_eval" % d for d in ("1dsfm", "strecha", "blendedmvs")]}
TRAININGS = {
    "deep 30/70": "uesfm_27scenes_adaptive_p3070",
    "shallow 30/70": "uesfm_27scenes_shallow_adaptive_p3070",
    "deep MAD-linked": "uesfm_27scenes_adaptive_madlink",
    "shallow MAD-linked": "uesfm_27scenes_shallow_adaptive_madlink",
    "RESfM-shallow (from scratch)": "resfm_shallow_27scenes",
    "shallow 10/90": "uesfm_27scenes_shallow_adaptive_p1090",
    "shallow 40/60": "uesfm_27scenes_shallow_adaptive_p4060",
}

prev = {}
if os.path.exists(STATE):
    try:
        prev = json.load(open(STATE))
    except Exception:
        prev = {}

cur, lines, deltas, all_done = {}, [], [], True
lines.append("=== EVAL FLEETS ===")
for name, (roots, tot) in EVALS.items():
    d = eval_count([r.rsplit("_seed", 1)[0] if False else r for r in roots]) if False else \
        sum(eval_count([r]) for r in roots)
    cur[name] = d
    flag = " DONE" if d >= tot else ""
    if d < tot:
        all_done = False
    lines.append(f"  {name:24s} {d:3d}/{tot}{flag}")
    if name in prev and d > prev[name]:
        deltas.append(f"  +{d - prev[name]} {name}  ({d}/{tot})")

lines.append("=== TRAININGS (best epoch / 20000) ===")
for name, resdir in TRAININGS.items():
    ep, done = ckpt_epoch(resdir)
    cur["TR:" + name] = ep
    st = "COMPLETE (20k) -> needs eval" if done else f"Ep{ep}"
    if not done:
        all_done = False
    lines.append(f"  {name:24s} {st}")
    if done and not prev.get("TRDONE:" + name):
        deltas.append(f"  TRAINING COMPLETE: {name} (launch its eval fleet)")
    cur["TRDONE:" + name] = int(done)

# collapse check for any completed PoC cells
lines.append("=== PoC learned-head collapse check (Nr) ===")
try:
    import pandas as pd
    for label, roots in POC_ROOTS.items():
        nrs = []
        for r in roots:
            for f in glob.glob(f"{r}/*_ba/Results_FINE_TUNE*.xlsx"):
                try:
                    nrs.append(float(pd.read_excel(f).iloc[-1]["#registered_cams_final"]))
                except Exception:
                    pass
        if nrs:
            import statistics
            m = statistics.mean(nrs)
            warn = "  <-- COLLAPSE?" if m < 5 else ""
            lines.append(f"  {label:12s} n={len(nrs):2d}  mean Nr={m:6.1f}{warn}")
        else:
            lines.append(f"  {label:12s} (no results yet)")
except Exception as e:
    lines.append(f"  (Nr check skipped: {e})")

print("\n".join(lines))
if deltas:
    print("\n*** CHANGES SINCE LAST CHECK ***")
    print("\n".join(deltas))
else:
    print("\n(no change since last check)")
if all_done:
    print("\n### ALL EXPERIMENTS COMPLETE ###")

json.dump(cur, open(STATE, "w"))
raise SystemExit(42 if all_done else 0)
