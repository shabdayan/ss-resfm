#!/usr/bin/env python
"""Recompute every cell of the ICLR draft's quantitative tables from the saved
eval outputs and print PASS/MISMATCH per cell. Read-only."""
import glob, json, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
M = f"{HERE}/results/multiscene"; C = f"{HERE}/results/crossdataset"
DS = ["megadepth", "1dsfm", "1dsfmhard", "strecha", "blendedmvs", "olsson"]
NC = {"megadepth": 13126, "1dsfm": 4563, "1dsfmhard": 1497, "strecha": 54,
      "blendedmvs": 225, "olsson": 3858}


def scene_vals(root):
    out = []
    for f in glob.glob(f"{root}/*_ba/Results_FINE_TUNE_stage_1_*.xlsx"):
        r = pd.read_excel(f).iloc[0]
        out.append((float(r["ts_ba_final_mean"]), float(r["Rs_ba_final_mean"]),
                    int(r["#registered_cams_final"])))
    return out


def roots(arm, ds):
    rs = []
    for s in ["", 21, 22, 23, 24]:
        tag = f"uesfm_finelr{'' if s == '' else f'_s{s}'}"
        if arm == "RESfM":
            cands = [f"{M}/resfm_finelr{'' if s == '' else f'_s{s}'}_megadepth_eval"] \
                if ds == "megadepth" else [f"{C}/resfm_finelr{'' if s == '' else f'_s{s}'}_{ds}_eval"]
        elif ds == "megadepth":
            names = [arm] + (["wttt"] if arm == "weight_ttt" else [])
            cands = [f"{M}/{tag}_{n}_megadepth_eval" for n in names]
        else:
            cands = [f"{C}/{tag}_rf_{arm}_{ds}_eval"]
        for r in cands:
            if os.path.isdir(r):
                rs.append(r); break
    return rs


def band(arm, ds):
    tm, td, rm, nr = [], [], [], []
    for r in roots(arm, ds):
        v = scene_vals(r)
        if not v: continue
        a = np.array(v)
        tm.append(a[:, 0].mean()); td.append(np.median(a[:, 0]))
        rm.append(a[:, 1].mean()); nr.append(100 * a[:, 2].sum() / NC[ds])
    return (np.mean(tm), np.std(tm, ddof=1) if len(tm) > 1 else 0, np.mean(td),
            np.mean(rm), np.mean(nr), len(tm))


def chk(label, printed, computed, tol):
    ok = abs(printed - computed) <= tol
    print(f"{'PASS' if ok else 'MISMATCH':8s} {label:48s} printed {printed:<8g} data {computed:.3f}")
    return ok


# ---- Table 2 expected values (as printed) ----
T2 = {  # arm: ds -> (tmean, tstd, tmed, rot, nr%)
 "madweight":  {"megadepth": (0.51,0.08,0.16,3.70,81), "1dsfm": (11.0,1.7,4.9,7.74,70),
                "1dsfmhard": (18.9,4.1,13.4,27.7,43), "strecha": (2.65,0.25,1.16,12.80,91),
                "blendedmvs": (0.14,0.02,0.05,9.01,92), "olsson": (3.0,0.3,1.4,14.4,86)},
 "weight":     {"megadepth": (0.51,0.09,0.20,3.28,79), "1dsfm": (14.4,3.0,9.6,8.24,79),
                "1dsfmhard": (22.9,2.8,15.6,28.0,40), "strecha": (1.97,0.09,0.04,7.61,98),
                "blendedmvs": (0.14,0.03,0.02,9.17,94), "olsson": (2.9,0.3,1.1,13.2,90)},
 "weight_ttt": {"megadepth": (0.40,0.11,0.18,2.88,77), "1dsfm": (15.7,2.1,10.8,10.09,81),
                "1dsfmhard": (22.2,3.2,15.3,30.6,41), "strecha": (2.00,0.05,0.12,7.95,99),
                "blendedmvs": (0.14,0.02,0.01,8.30,94), "olsson": (8.0,0.3,0.6,11.6,90)},
 "remove":     {"megadepth": (0.51,0.08,0.13,4.04,72), "1dsfm": (9.1,2.2,1.7,6.94,58),
                "1dsfmhard": (12.4,2.1,10.8,12.3,29), "strecha": (3.20,0.11,2.03,14.39,97),
                "blendedmvs": (0.24,0.06,0.17,16.04,88), "olsson": (3.4,0.6,1.7,13.9,78)},
 "remove_ttt": {"megadepth": (0.56,0.07,0.14,4.22,71), "1dsfm": (8.9,1.2,2.0,6.90,58),
                "1dsfmhard": (15.2,1.0,11.3,16.4,32), "strecha": (3.01,0.07,1.70,13.23,96),
                "blendedmvs": (0.22,0.10,0.18,13.19,90), "olsson": (3.6,0.4,1.9,14.5,78)},
 "RESfM":      {"megadepth": (0.37,0.12,0.07,1.89,85), "1dsfm": (10.5,1.5,3.4,9.62,77),
                "1dsfmhard": (19.2,3.4,16.5,27.3,34), "strecha": (0.39,0.72,0.01,1.56,99),
                "blendedmvs": (0.35,0.04,0.33,31.66,92), "olsson": (7.4,2.6,0.95,12.6,89)},
}
bad = 0
for arm, cells in T2.items():
    for ds, (tm, ts, td, rot, nr) in cells.items():
        ctm, cts, ctd, crm, cnr, n = band(arm, ds)
        for lbl, pv, cv, tol in [("tmean", tm, ctm, 0.06), ("tstd", ts, cts, 0.06),
                                 ("tmed", td, ctd, 0.07), ("rot", rot, crm, 0.11),
                                 ("nr%", nr, cnr, 0.9)]:
            if not chk(f"T2 {arm}/{ds} {lbl}", pv, cv, tol): bad += 1
        print(f"         (n={n} seeds)")

# ---- reference rows ----
def agg1(root, col="ts_ba_final_mean"):
    v = [float(pd.read_excel(f).iloc[0][col])
         for f in glob.glob(f"{root}/*_ba/Results_FINE_TUNE_stage_1_*.xlsx")]
    return (np.mean(v), len(v)) if v else (float("nan"), 0)

FAITH = {"megadepth": 0.496, "1dsfm": 9.75, "1dsfmhard": 17.58, "strecha": 0.144,
         "blendedmvs": 0.367, "olsson": 8.77}
for ds, pv in FAITH.items():
    pre = M if ds == "megadepth" else C
    cv, n = agg1(f"{pre}/resfm_faithful_{ds}_eval")
    if not chk(f"REF faithful/{ds}", pv, cv, 0.03): bad += 1

# GLOMAP rows
GLO = {"megadepth": (3.33, 5.30), "1dsfm": (27.21, 8.78), "1dsfmhard": (35.42, 26.4),
       "strecha": (0.047, 0.25), "blendedmvs": (0.339, 2.15), "olsson": (3.17, 1.29)}
DSMAP = {"megadepth": "megadepth", "1dsfm": "1dsfm", "1dsfmhard": "1dsfm_hard_300",
         "strecha": "strecha", "blendedmvs": "blendedmvs", "olsson": "olsson"}
for ds, (pt, pr) in GLO.items():
    js = glob.glob(f"{HERE}/results/classical/glomap/{DSMAP[ds]}__*.json")
    tv, rv = [], []
    for j in js:
        d = json.load(open(j))
        if d.get("failed") or "trans_mean" not in d: continue
        tv.append(d["trans_mean"]); rv.append(d["rot_mean"])
    if not chk(f"GLOMAP/{ds} trans", pt, np.mean(tv), 0.06): bad += 1
    if not chk(f"GLOMAP/{ds} rot", pr, np.mean(rv), 0.11): bad += 1

print(f"\nDONE: {bad} mismatches")


# ---- tab:indist (in-distribution contamination curve) ----
def band_mean(fmt, seeds):
    ms = []
    for s in seeds:
        v = [float(pd.read_excel(f)["ts_ba_final_mean"].iloc[-1])
             for f in glob.glob(f"{C}/{fmt.format(s=s)}/*_ba/Results_FINE_TUNE_stage_1_*.xlsx")]
        if v: ms.append(np.mean(v))
    return (np.mean(ms), np.std(ms, ddof=1)) if len(ms) > 1 else (float("nan"), float("nan"))

INDIST = {  # (pool, seeds): {arm: (paper_mean, paper_std)}
    ("olssonid", (20, 21, 22, 23, 24)): {"madweight": (1.97, 0.94), "weight": (3.05, 1.39),
        "weight_ttt": (1.60, 0.81), "remove": (2.94, 0.27), "SUP": (2.03, 1.86)},
    ("bmvsid", (20, 21, 22, 23, 24)): {"madweight": (0.093, 0.099), "weight": (0.114, 0.006),
        "weight_ttt": (0.081, 0.046), "remove": (0.228, 0.080), "SUP": (0.260, 0.149)},
    ("1donly", (20, 21, 22, 23, 24)): {"madweight": (6.37, 0.20), "weight": (10.65, 2.49),
        "weight_ttt": (9.91, 2.39), "remove": (6.68, 1.44), "SUP": (4.97, 2.08)},
    ("hardonly", (20, 21, 22, 23, 24)): {"madweight": (12.1, 6.1), "weight": (28.1, 5.9),
        "weight_ttt": (29.3, 8.4), "remove": (19.4, 7.7), "SUP": (23.4, 7.8)},
}
for (pool, seeds), arms in INDIST.items():
    for arm, (pm, ps) in arms.items():
        fmt = f"resfm_{pool}_s{{s}}_ind_eval" if arm == "SUP" \
            else f"uesfm_{pool}_s{{s}}_ind_{arm}_eval"
        cm, cs = band_mean(fmt, seeds)
        if not chk(f"INDIST {pool}/{arm} mean", pm, cm, 0.06): bad += 1
        if not chk(f"INDIST {pool}/{arm} std", ps, cs, 0.12): bad += 1

print(f"\nDONE(indist): {bad} total mismatches")
