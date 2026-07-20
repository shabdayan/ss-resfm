"""Aggregate the cross-dataset 5-seed protocol: per-scene MEAN (default,
Ortal's reporting preference) or MEDIAN over seeds
of Rot / Trans / Nr for every arm of CROSSDATASET_RESULTS.md, then
per-dataset means of those medians.

Roots follow the launchers' layout: <root> for seed 20, <root>_seed<S> else;
TTT results live under results/multiscene/ttt/<tag>_<loss>_seed<S>. Scenes
are discovered from the union of <root*>/<scene>_ba dirs (no hardcoded
lists). Cells missing a seed are reported in the coverage column, medians
use whatever seeds exist.

Usage: python aggregate_crossdataset_seeds.py [--seeds 20,21,22,23,24]
       [--csv results/crossdataset/all_arms_seed_medians.csv]
"""
import argparse
import glob
import os
import re

import pandas as pd

XD = "results/crossdataset"
TTT = "results/multiscene/ttt"

# arm label -> (root template with {ds}, results-file glob)
ARMS = {
    "ESFM": (XD + "/esfm_baseline_{ds}_eval", "Results_FINE_TUNE*"),
    "ESFM@1e-4": (XD + "/esfm_baseline_ftlr1e4_{ds}_eval", "Results_FINE_TUNE*"),
    "RESfM-off": (XD + "/resfm_repro_{ds}", "Results_FINE_TUNE*"),
    "stage1": (XD + "/uesfm_{ds}_eval", "Results_FINE_TUNE*"),
    "stage1@1e-4": (XD + "/uesfm_stage1_ftlr1e4_{ds}_eval", "Results_FINE_TUNE*"),
    "shal-Ep17k": (XD + "/uesfm_shallow_adaptive_ep17k_{ds}_eval", "Results_FINE_TUNE*"),
    "shal-Ep17k@1e-4": (XD + "/uesfm_shallow_adaptive_ep17k_ftlr1e4_{ds}_eval", "Results_FINE_TUNE*"),
    "TTT-comb": (TTT + "/xd_{ds}_shallow_adaptive_ep17k_comb", "Results_FINE_TUNE*"),
    "TTT-reproj": (TTT + "/xd_{ds}_shallow_adaptive_ep17k_reproj_only", "Results_FINE_TUNE*"),
}
DATASETS = ["1dsfm", "strecha", "blendedmvs"]
# seed-20 stage1 ran under uesfm_{ds}_eval; seeds 21+ use uesfm_stage1_{ds}_eval
ALT_ROOTS = {"stage1": XD + "/uesfm_stage1_{ds}_eval"}
# TTT roots end _seed<S> for ALL seeds incl. 20
TTT_ALWAYS_SUFFIXED = ("TTT-comb", "TTT-reproj")


def seed_root(label, base, seed):
    if label in TTT_ALWAYS_SUFFIXED:
        return f"{base}_seed{seed}"
    return base if seed == 20 else f"{base}_seed{seed}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="20,21,22,23,24")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--stat", choices=["mean", "median"], default="mean")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.csv is None:
        args.csv = XD + f"/all_arms_seed_{args.stat}s.csv"
    agg = (lambda s: s.mean()) if args.stat == "mean" else (lambda s: s.median())

    rows = []
    for label, (base_t, rf_glob) in ARMS.items():
        for ds in DATASETS:
            bases = [base_t.format(ds=ds)]
            if label in ALT_ROOTS:
                bases.append(ALT_ROOTS[label].format(ds=ds))
            scenes = set()
            for b in bases:
                for s in seeds:
                    for d in glob.glob(seed_root(label, b, s) + "/*_ba"):
                        scenes.add(os.path.basename(d)[:-3])
            for scene in sorted(scenes):
                per_seed = {}
                for s in seeds:
                    for b in bases:
                        fs = glob.glob(os.path.join(seed_root(label, b, s),
                                                    f"{scene}_ba", rf_glob + ".xlsx"))
                        if fs:
                            r = pd.read_excel(fs[0]).iloc[-1]
                            per_seed[s] = (float(r["Rs_ba_final_mean"]),
                                           float(r["ts_ba_final_mean"]),
                                           int(r["#registered_cams_final"]))
                            break
                if not per_seed:
                    continue
                df = pd.DataFrame(per_seed.values(), columns=["Rot", "Trans", "Nr"])
                rows.append(dict(arm=label, ds=ds, scene=scene,
                                 n_seeds=len(per_seed),
                                 Rot_agg=agg(df["Rot"]),
                                 Trans_agg=agg(df["Trans"]),
                                 Nr_agg=agg(df["Nr"]),
                                 Rot_std=df["Rot"].std()))

    t = pd.DataFrame(rows)
    t.to_csv(args.csv, index=False)
    order = [a for a in ARMS if a in set(t.arm)]

    print("=== coverage: scenes x seeds present (want n_seeds=%d) ===" % len(seeds))
    cov = t.pivot_table(index="ds", columns="arm", values="n_seeds", aggfunc="mean")
    print(cov.reindex(columns=order).round(2).to_string())
    incomplete = t[t.n_seeds < len(seeds)]
    if len(incomplete):
        print(f"\nincomplete cells ({len(incomplete)}):")
        print(incomplete[["arm", "ds", "scene", "n_seeds"]].to_string(index=False))

    print(f"\n=== per-dataset MEAN of per-scene seed-{args.stat.upper()} Rot (deg) ===")
    print(t.pivot_table(index="ds", columns="arm", values="Rot_agg",
                        aggfunc="mean").reindex(columns=order).round(2).to_string())
    print(f"\n=== per-dataset MEAN of per-scene seed-{args.stat.upper()} Trans ===")
    print(t.pivot_table(index="ds", columns="arm", values="Trans_agg",
                        aggfunc="mean").reindex(columns=order).round(3).to_string())
    print(f"\nper-scene medians -> {args.csv} ({len(t)} rows)")


if __name__ == "__main__":
    main()
