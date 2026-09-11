#!/usr/bin/env python
"""VGPA-style five-lens readout of OUR results matrix (per-scene wins, mean vs
median with blow-up attribution, registration coverage, rotation; RA-AUC is
produced by the separate pose-recovery program).

Reads the per-scene eval xlsx files (ts_ba_final_mean, Rs_ba_final_mean,
#registered_cams_final) for every arm/baseline x dataset x seed. Emits a
markdown report to stdout.
"""
import glob, os, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
M = f"{HERE}/results/multiscene"; C = f"{HERE}/results/crossdataset"
DS = ["megadepth", "1dsfm", "1dsfmhard", "strecha", "blendedmvs", "olsson"]
NC = {"megadepth": 13126, "1dsfm": 4563, "1dsfmhard": 1497, "strecha": 54,
      "blendedmvs": 225, "olsson": 3858}
SEEDS = ["", 21, 22, 23, 24]


def scene_map(root):
    out = {}
    for f in glob.glob(f"{root}/*_ba/Results_FINE_TUNE_stage_1_*.xlsx"):
        r = pd.read_excel(f).iloc[0]
        sc = os.path.basename(os.path.dirname(f))[:-3]
        out[sc] = (float(r["ts_ba_final_mean"]), float(r["Rs_ba_final_mean"]),
                   int(r["#registered_cams_final"]))
    return out


def roots_for(arm, ds, s):
    tag = f"uesfm_finelr{'' if s == '' else f'_s{s}'}"
    if arm == "RESfM":
        return [f"{M}/resfm_finelr{'' if s == '' else f'_s{s}'}_megadepth_eval"] \
            if ds == "megadepth" else [f"{C}/resfm_finelr{'' if s == '' else f'_s{s}'}_{ds}_eval"]
    if arm == "vanilla":
        pre = M if ds == "megadepth" else C
        return [f"{pre}/esfm_vanilla_{ds}_eval"] if s == "" else \
            [f"{pre}/esfm_vanilla_s{s}_{ds}_eval", f"{pre}/esfm_vanilla_{ds}_eval_seed{s}"]
    if arm == "star":
        sd = {"olsson": "olssonstar"}.get(ds, ds)
        pre = M if ds == "megadepth" else C
        return [f"{pre}/esfm_star_af_{sd}_eval"] if s == "" else [f"{pre}/esfm_star_af_s{s}_{sd}_eval"]
    if arm == "hybrid":
        return [f"{M}/{tag}_hybrid_megadepth_eval"] if ds == "megadepth" \
            else [f"{C}/{tag}_rf_hybrid_{ds}_eval"]
    if ds == "megadepth":
        names = [arm] + (["wttt"] if arm == "weight_ttt" else [])
        return [f"{M}/{tag}_{n}_megadepth_eval" for n in names]
    return [f"{C}/{tag}_rf_{arm}_{ds}_eval"]


def load(arm, ds):
    per_seed = {}
    for s in SEEDS:
        for r in roots_for(arm, ds, s):
            if os.path.isdir(r):
                sm = scene_map(r)
                if sm:
                    per_seed[20 if s == "" else s] = sm
                break
    return per_seed


ARMS = ["madweight", "weight", "weight_ttt", "remove", "remove_ttt", "hybrid",
        "RESfM", "vanilla", "star"]
LABEL = {"madweight": "SS madweight", "weight": "SS weight", "weight_ttt": "SS weight+TTT",
         "remove": "SS remove", "remove_ttt": "SS remove+TTT", "hybrid": "SS hybrid",
         "RESfM": "RESfM-scratch (sup.)", "vanilla": "ESFM (no mech.)", "star": "ESFM* (clean tracks)"}

data = {(a, d): load(a, d) for a in ARMS for d in DS}

for ds in DS:
    print(f"\n## {ds}  (Nc={NC[ds]})\n")
    # Lens 2+3+5: means / medians / rot / coverage
    print("| Method | Trans mean±std | Trans med (pooled) | Rot mean | Nr% | seeds |")
    print("|---|---|---|---|---|---|")
    for a in ARMS:
        ps = data[(a, ds)]
        if not ps:
            print(f"| {LABEL[a]} | — | — | — | — | 0 |"); continue
        seed_means = [np.mean([v[0] for v in sm.values()]) for sm in ps.values()]
        pooled = [v[0] for sm in ps.values() for v in sm.values()]
        rots = [np.mean([v[1] for v in sm.values()]) for sm in ps.values()]
        nr = np.mean([100 * sum(v[2] for v in sm.values()) / NC[ds] for sm in ps.values()])
        std = f"±{np.std(seed_means, ddof=1):.2f}" if len(seed_means) > 1 else ""
        print(f"| {LABEL[a]} | {np.mean(seed_means):.3f}{std} | {np.median(pooled):.3f} "
              f"| {np.mean(rots):.2f} | {nr:.0f}% | {len(ps)} |")
    # Lens 1: matched-seed per-scene win counts vs RESfM
    base = data[("RESfM", ds)]
    print("\n**Per-scene wins vs RESfM-scratch (matched seed, scene x seed cells):**\n")
    for a in ARMS:
        if a == "RESfM" or not data[(a, ds)]:
            continue
        tw = rw = tot = 0
        for s, sm in data[(a, ds)].items():
            if s not in base:
                continue
            for sc, v in sm.items():
                if sc in base[s]:
                    tot += 1
                    tw += v[0] < base[s][sc][0]
                    rw += v[1] < base[s][sc][1]
        if tot:
            print(f"- {LABEL[a]}: trans {tw}/{tot} ({100*tw/tot:.0f}%), rot {rw}/{tot} ({100*rw/tot:.0f}%)")
    # Lens 2b: blow-up attribution (seed-20 scene means, top mean-drivers)
    print("\n**Mean-driving scenes (seed-20, top 3 by scene mean):**\n")
    for a in ARMS:
        ps = data[(a, ds)]
        if 20 not in ps:
            continue
        top = sorted(ps[20].items(), key=lambda kv: -kv[1][0])[:3]
        med = np.median([v[0] for v in ps[20].values()])
        print(f"- {LABEL[a]} (med {med:.2f}): " +
              ", ".join(f"{sc} {v[0]:.2f}" for sc, v in top))
print("\n(RA-AUC/AUC: pending the Final_recon pose-recovery program.)")
