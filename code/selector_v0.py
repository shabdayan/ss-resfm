#!/usr/bin/env python
"""Selector v0: can a LEARNED per-scene mechanism selector beat fixed choices?

Uses only material already on disk (finelr U-ESFM generation): per-scene
post-BA outcomes of the three test-time mechanisms {madweight, weight,
weight+TTT} on megadepth(36)/1dsfm(10)/1dsfm_hard(5)/blendedmvs(9)/strecha(4).

Features are strictly LABEL-FREE (computable on a new scene without GT):
  - track-matrix statistics from the npz (sizes, density, cams-per-track and
    track-length distribution quantiles)
  - the model's own pre-BA reprojection error (our_repro from the weight arm =
    plain probe forward; a self-signal, no GT involved)

Protocol: leave-one-dataset-out (LODO) -- the selector never sees any scene of
the dataset it is tested on (mirrors deployment on a new collection).
Baselines: fixed-global-best arm, always-madweight, always-weight(+TTT),
per-dataset oracle-fixed (upper bound for fixed policies; uses test identity),
and the per-scene oracle (upper bound, uses GT outcomes).
Metric: mean post-BA translation over each dataset's scenes when following
each policy; plus selection accuracy on decisive scenes (margin > max(0.05,
15% of best) -- below the measured BA noise floor a "wrong" pick is a coin
flip, so those scenes are excluded from accuracy but included in means).
"""
import glob, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["madweight", "weight", "wttt"]
ROOTS = {
    "megadepth":  {"madweight": "results/multiscene/uesfm_finelr_madweight_megadepth_eval",
                   "weight":    "results/multiscene/uesfm_finelr_weight_megadepth_eval",
                   "wttt":      "results/multiscene/uesfm_finelr_wttt_megadepth_eval"},
    "1dsfm":      {"madweight": "results/crossdataset/uesfm_finelr_rf_madweight_1dsfm_eval",
                   "weight":    "results/crossdataset/uesfm_finelr_rf_weight_1dsfm_eval",
                   "wttt":      "results/crossdataset/uesfm_finelr_rf_weight_ttt_1dsfm_eval"},
    "1dsfmhard":  {"madweight": "results/crossdataset/uesfm_finelr_rf_madweight_1dsfmhard_eval",
                   "weight":    "results/crossdataset/uesfm_finelr_rf_weight_1dsfmhard_eval",
                   "wttt":      "results/crossdataset/uesfm_finelr_rf_weight_ttt_1dsfmhard_eval"},
    "blendedmvs": {"madweight": "results/crossdataset/uesfm_finelr_rf_madweight_blendedmvs_eval",
                   "weight":    "results/crossdataset/uesfm_finelr_rf_weight_blendedmvs_eval",
                   "wttt":      "results/crossdataset/uesfm_finelr_rf_weight_ttt_blendedmvs_eval"},
    "strecha":    {"madweight": "results/crossdataset/uesfm_finelr_rf_madweight_strecha_eval",
                   "weight":    "results/crossdataset/uesfm_finelr_rf_weight_strecha_eval",
                   "wttt":      "results/crossdataset/uesfm_finelr_rf_weight_ttt_strecha_eval"},
}
NPZ_FAM = {"megadepth": "megadepth", "1dsfm": "1dsfm", "1dsfmhard": "1dsfm_hard_300",
           "blendedmvs": "blendedmvs", "strecha": "strecha"}


def read_root(root):
    out = {}
    for f in glob.glob(os.path.join(HERE, root, "*_ba", "Results_FINE_TUNE_stage_1_*.xlsx")):
        df = pd.read_excel(f)
        if df.empty or "ts_ba_final_mean" not in df.columns:
            continue
        sc = os.path.basename(os.path.dirname(f)).replace("_ba", "")
        out[sc] = dict(t=float(df["ts_ba_final_mean"].iloc[0]),
                       our_repro=float(df["our_repro"].iloc[0]))
    return out


def npz_features(fam, scene):
    p = os.path.join(HERE, "datasets", fam, f"{scene}.npz")
    d = np.load(p, allow_pickle=True)
    M = d["M"]; C = M.shape[0] // 2
    X = M.reshape(C, 2, -1); vis = (X[:, 0] != 0) | (X[:, 1] != 0)
    cams_per_track = vis.sum(0).astype(float)
    pts_per_cam = vis.sum(1).astype(float)
    q = lambda a, qq: float(np.quantile(a, qq))
    return dict(
        n_cams=float(C), n_tracks=float(M.shape[1]), n_obs=float(vis.sum()),
        density=float(vis.mean()),
        cpt_mean=float(cams_per_track.mean()), cpt_std=float(cams_per_track.std()),
        cpt_q10=q(cams_per_track, .1), cpt_q50=q(cams_per_track, .5), cpt_q90=q(cams_per_track, .9),
        ppc_mean=float(pts_per_cam.mean()), ppc_std=float(pts_per_cam.std()),
        ppc_min=float(pts_per_cam.min()),
        log_cams=float(np.log(C)), log_tracks=float(np.log(M.shape[1])),
    )


def main():
    rows = []
    for ds, roots in ROOTS.items():
        per_arm = {a: read_root(r) for a, r in roots.items()}
        scenes = set.intersection(*[set(v) for v in per_arm.values()])
        for sc in sorted(scenes):
            feat = npz_features(NPZ_FAM[ds], sc)
            feat["probe_repro"] = per_arm["weight"][sc]["our_repro"]  # plain-forward self-signal
            feat["log_probe_repro"] = float(np.log1p(feat["probe_repro"]))
            outs = {a: per_arm[a][sc]["t"] for a in ARMS}
            rows.append(dict(ds=ds, scene=sc, **feat, **{f"t_{a}": outs[a] for a in ARMS}))
    df = pd.DataFrame(rows)
    print(f"assembled {len(df)} scenes: " + ", ".join(f"{d}={len(g)}" for d, g in df.groupby('ds')))

    T = df[[f"t_{a}" for a in ARMS]].values
    best_idx = T.argmin(1)
    best = T.min(1); second = np.sort(T, 1)[:, 1]
    decisive = (second - best) > np.maximum(0.05, 0.15 * best)
    df["label"] = best_idx; df["decisive"] = decisive
    print(f"decisive scenes: {decisive.sum()}/{len(df)}  | label dist (decisive): "
          + str(np.bincount(best_idx[decisive], minlength=3).tolist()) + f" {ARMS}")

    FEATS = [c for c in df.columns if c not in
             ["ds", "scene", "label", "decisive"] + [f"t_{a}" for a in ARMS]]
    from sklearn.ensemble import RandomForestClassifier

    # LODO
    pred = np.empty(len(df), dtype=int)
    for ds in ROOTS:
        tr = (df["ds"] != ds).values & decisive          # train on decisive scenes of other datasets
        te = (df["ds"] == ds).values
        clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                     class_weight="balanced", random_state=0)
        clf.fit(df.loc[tr, FEATS], df.loc[tr, "label"])
        pred[te] = clf.predict(df.loc[te, FEATS])
    df["pred"] = pred

    # policies -> chosen error per scene
    pol = {
        "selector (LODO)": T[np.arange(len(df)), pred],
        "oracle (per-scene, UB)": best,
    }
    for i, a in enumerate(ARMS):
        pol[f"always-{a}"] = T[:, i]
    # fixed-global best arm chosen on OTHER datasets' mean (honest LODO fixed baseline)
    fg = np.empty(len(df))
    for ds in ROOTS:
        te = (df["ds"] == ds).values
        others = ~te
        arm_means = [T[others, i].mean() for i in range(3)]
        fg[te] = T[te, int(np.argmin(arm_means))]
    pol["fixed-global (LODO)"] = fg

    print("\n=== mean post-BA trans by policy (rows) x dataset (cols) ===")
    hdr = f"{'policy':<24}" + "".join(f"{d:>11}" for d in ROOTS) + f"{'ALL':>11}"
    print(hdr); print("-" * len(hdr))
    order = ["oracle (per-scene, UB)", "selector (LODO)", "fixed-global (LODO)"] + [f"always-{a}" for a in ARMS]
    for name in order:
        v = pol[name]
        cells = "".join(f"{v[(df['ds']==d).values].mean():>11.3f}" for d in ROOTS)
        print(f"{name:<24}{cells}{v.mean():>11.3f}")

    acc = (df["pred"] == df["label"])[decisive].mean()
    print(f"\nselection accuracy on decisive scenes: {acc*100:.0f}%")
    # oracle-margin recovery: how much of (fixed-global - oracle) does the selector close?
    fgm, orm, sem = pol["fixed-global (LODO)"].mean(), best.mean(), pol["selector (LODO)"].mean()
    if fgm > orm:
        print(f"oracle-margin recovered vs fixed-global: {100*(fgm-sem)/(fgm-orm):.0f}%")
    df.to_csv(os.path.join(HERE, "logs_authors_env", "selector_v0_data.csv"), index=False)
    print("saved: logs_authors_env/selector_v0_data.csv")


if __name__ == "__main__":
    main()
