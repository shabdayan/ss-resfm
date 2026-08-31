#!/usr/bin/env python
"""Per-scene mechanism-selector study (Paper-B go/no-go).

Dataset: for each of the 98 finelr seed-20 scenes, unsupervised FIRST-PASS
features (initial-checkpoint TEST outputs of the weight arm: head-score
distribution + reprojection scalars + track stats) and per-mechanism post-BA
translation outcomes for the oracle set {remove, hybrid, madweight, weight,
weight_ttt}. Rules evaluated:
  A. contamination-proxy threshold (frac head>0.6 >= 10% -> madweight else weight)
  B. last-epoch rule: argmin repro_ba_final (retention-confounded; expected fail)
  C. learned first-pass selector (GradientBoosting, leave-one-dataset-out)
  D. oracle (GT argmin) upper bound.
Writes results/selector_study/{dataset.csv,report.txt}.
"""
import glob, os, json, numpy as np, pandas as pd
M="results/multiscene"; C="results/crossdataset"
MECH=["remove","hybrid","madweight","weight","weight_ttt"]
def root(mech, ds):
    mdn={"weight_ttt":"wttt"}.get(mech,mech)
    if ds=="megadepth": return f"{M}/uesfm_finelr_{mdn}_megadepth_eval"
    return f"{C}/uesfm_finelr_rf_{mech}_{ds}_eval"
SCENES={
 "megadepth":"0238 0060 0197 0094 0265 0083 0076 0185 0048 0024 0223 5016 0046 0099 1001 0231 0411 0377 0102 0147 0148 0446 0022 0327 0015 0455 0496 1589 0012 0104 0019 0063 0130 0080 0240 0007".split(),
 "1dsfm":"Alamo Ellis_Island Madrid_Metropolis Montreal_Notre_Dame Notre_Dame NYC_Library Piazza_del_Popolo Tower_of_London Vienna_Cathedral Yorkminster".split(),
 "1dsfmhard":"Gendarmenmarkt Piccadilly Roman_Forum Trafalgar Union_Square".split(),
 "strecha":"entry-P10 fountain-P11 Herz-Jesu-P25 Herz-Jesu-P8".split(),
 "blendedmvs":"58c4bb4f4a69c55606122be4 5a48ba95c7dab83a7d7b44ed 5acf8ca0f3d8a750097e4b15 5b950c71608de421b1e7318f".split(),
}
import subprocess
ol=[d[:-3] for d in os.listdir(f"{C}/uesfm_finelr_rf_madweight_olsson_eval") if d.endswith("_ba")]
SCENES["olsson"]=sorted(ol)
rows=[]
for ds, scenes in SCENES.items():
    wroot=root("weight",ds)
    for s in scenes:
        r={"dataset":ds,"scene":s}
        ok=True
        # outcomes + per-mech last-epoch reproj/Nr
        for mech in MECH:
            f=glob.glob(f"{root(mech,ds)}/{s}_ba/Results_FINE_TUNE_stage_1_*.xlsx")
            if not f: ok=False; break
            x=pd.read_excel(f[0]).iloc[0]
            r[f"t_{mech}"]=float(x["ts_ba_final_mean"])
            r[f"reproba_{mech}"]=float(x["repro_ba_final"])
            r[f"nr_{mech}"]=int(x["#registered_cams_final"])
        if not ok: continue
        # first-pass features from the weight root
        f=glob.glob(f"{wroot}/{s}_ba/Results_FINE_TUNE_stage_1_*.xlsx")
        x=pd.read_excel(f[0]).iloc[0]
        r["our_repro"]=float(x["our_repro"]); r["tri_repro"]=float(x["triangulated_repro"])
        np_path=f"{wroot}/{s}_ba/TEST/{s}/outliers_results/Final_outliers.npz"
        try:
            z=np.load(np_path, allow_pickle=True)
            p=z["outliers_pred"]; Mm=z["M"]
            m=Mm.shape[0]//2
            X2=Mm.reshape(m,2,-1); vis=(X2[:,0]!=0)|(X2[:,1]!=0)
            sc=p[vis]
            r.update(ncams=m, ntracks=Mm.shape[1], nobs=int(vis.sum()),
                     tracklen=float(vis.sum()/Mm.shape[1]), obspercam=float(vis.sum()/m),
                     h_mean=float(sc.mean()), h_std=float(sc.std()),
                     h_q10=float(np.quantile(sc,.1)), h_q25=float(np.quantile(sc,.25)),
                     h_q50=float(np.quantile(sc,.5)), h_q75=float(np.quantile(sc,.75)),
                     h_q90=float(np.quantile(sc,.9)),
                     h_f05=float((sc>0.5).mean()), h_f06=float((sc>0.6).mean()),
                     h_f08=float((sc>0.8).mean()))
        except Exception as e:
            print("feat fail", ds, s, e); continue
        rows.append(r)
        print(f"{ds}/{s} ok", flush=True)
df=pd.DataFrame(rows)
os.makedirs("results/selector_study", exist_ok=True)
df.to_csv("results/selector_study/dataset.csv", index=False)
print("dataset:", df.shape)

# ---- evaluation ----
out=[]
T=df[[f"t_{m}" for m in MECH]].values
def report(name, pick_idx):
    sel=T[np.arange(len(df)), pick_idx]
    lines=[f"== {name}"]
    for ds in SCENES:
        m=df.dataset.values==ds
        if m.sum(): lines.append(f"  {ds:11s} mean {sel[m].mean():7.3f}  (n={m.sum()})")
    lines.append(f"  ALL mean {sel.mean():.3f}")
    out.extend(lines); print("\n".join(lines))
    return sel
# fixed arms + oracle
for i,mech in enumerate(MECH):
    report(f"fixed:{mech}", np.full(len(df), i))
report("ORACLE (GT argmin)", T.argmin(1))
# Rule A: contamination proxy
pick=np.where(df.h_f06.values>=0.10, MECH.index("madweight"), MECH.index("weight"))
report("Rule A: head-frac>=10% -> madweight else weight", pick)
# Rule B: last-epoch retained reproj argmin
R=df[[f"reproba_{m}" for m in MECH]].values
report("Rule B: argmin last-epoch reproj (retention-confounded)", R.argmin(1))
# Rule C: learned LODO
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
FEATS=["our_repro","tri_repro","ncams","ntracks","nobs","tracklen","obspercam",
       "h_mean","h_std","h_q10","h_q25","h_q50","h_q75","h_q90","h_f05","h_f06","h_f08"]
X=df[FEATS].values
y=T.argmin(1)
pickC=np.zeros(len(df),dtype=int)
for ds in SCENES:
    te=df.dataset.values==ds; tr=~te
    if te.sum()==0: continue
    clf=GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=0)
    clf.fit(X[tr], y[tr]); pickC[te]=clf.predict(X[te])
report("Rule C: LEARNED first-pass selector (LODO, classifier)", pickC)
# Rule C2: regression per mechanism (predict log-error, argmin)
pickR=np.zeros(len(df),dtype=int)
P=np.zeros((len(df),len(MECH)))
for ds in SCENES:
    te=df.dataset.values==ds; tr=~te
    if te.sum()==0: continue
    for i,m in enumerate(MECH):
        rg=GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=0)
        rg.fit(X[tr], np.log1p(T[tr,i])); P[te,i]=rg.predict(X[te])
pickR=P.argmin(1)
report("Rule C2: LEARNED (LODO, per-mech regression argmin)", pickR)
open("results/selector_study/report.txt","w").write("\n".join(out))
print("DONE")
