#!/usr/bin/env python
"""Figure 4 (Path A): per-scene complementarity. For every scene we plot its own
outlier rate (x, log) against the signed advantage of removal over soft weighting,
log10(weight_err / min removal_err). y>0 = a removal-family mechanism wins; y<0 =
soft weight wins. The best mechanism flips from weight (low contamination) to
removal (high contamination) at the scene level. Colorblind-safe (Okabe-Ito).
"""
import matplotlib
import matplotlib as _mpl; matplotlib.use("Agg")
_mpl.rcParams.update({"font.size": 15})
import matplotlib.pyplot as plt
import glob, numpy as np, pandas as pd

def contam(ds, s):
    c=glob.glob(f"datasets/{ds}/*{s}*.npz")
    if not c: return None
    d=np.load(c[0],allow_pickle=True)
    if 'outlier_pct' not in d: return None
    v=float(d['outlier_pct']); return v*100 if ds=="megadepth" else v
def cd(ds): return "results/multiscene" if ds=="megadepth" else "results/crossdataset"
def err(ds,m,s):
    root=(f"{cd(ds)}/uesfm_finelr_{m}_megadepth_eval" if ds=="megadepth"
          else f"{cd(ds)}/uesfm_finelr_rf_{m}_{ds}_eval")
    f=glob.glob(f"{root}/{s}_ba/Results_FINE_TUNE*.xlsx")
    if not f: return None
    try: return float(pd.read_excel(f[0]).iloc[0]['ts_ba_final_mean'])
    except: return None
SC={"1dsfm":"Alamo Ellis_Island Madrid_Metropolis Montreal_Notre_Dame Notre_Dame NYC_Library Piazza_del_Popolo Tower_of_London Vienna_Cathedral Yorkminster".split(),
"strecha":"entry-P10 fountain-P11 Herz-Jesu-P8 Herz-Jesu-P25".split(),
"blendedmvs":"58c4bb4f4a69c55606122be4 5acf8ca0f3d8a750097e4b15 5a48ba95c7dab83a7d7b44ed 5b950c71608de421b1e7318f".split(),
"megadepth":"0238 0060 0197 0094 0265 0083 0076 0185 0048 0024 0223 5016 0046 0099 1001 0231 0411 0377 0102 0147 0148 0446 0022 0327 0015 0455 0496 1589 0012 0104 0019 0063 0130 0080 0240 0007".split()}
COL={"1dsfm":"#D55E00","strecha":"#0072B2","blendedmvs":"#009E73","megadepth":"#999999"}
LBL={"1dsfm":"1DSfM (OOD)","strecha":"Strecha (OOD)","blendedmvs":"BlendedMVS (OOD)","megadepth":"MegaDepth (in-dist.)"}
EPS=0.02
pts={ds:[] for ds in SC}
for ds,scenes in SC.items():
    for s in scenes:
        c=contam(ds,s)
        w=err(ds,"weight",s)
        rem=[err(ds,m,s) for m in ("madweight","remove","hybrid")]
        rem=[r for r in rem if r is not None]
        if c is None or w is None or not rem: continue
        adv=np.log10((w+EPS)/(min(rem)+EPS))
        pts[ds].append((c,adv))

INK,MUTED="#1a1a1a","#666666"
fig,ax=plt.subplots(figsize=(6.2,4.3))
ax.set_xscale("log")
ax.axhspan(0,10,color="#fdf3ee",zorder=0)      # removal-wins half
ax.axhspan(-10,0,color="#eef4fb",zorder=0)     # weight-wins half
ax.axvspan(3,29,color="#f0f0f0",zorder=0)      # OOD flip zone (unresolved)
ax.axhline(0,color=MUTED,lw=1.0,zorder=2)
for ds in ["megadepth","strecha","blendedmvs","1dsfm"]:
    if not pts[ds]: continue
    xs=[p[0] for p in pts[ds]]; ys=[p[1] for p in pts[ds]]
    mk="s" if ds=="megadepth" else "o"
    ax.scatter(xs,ys,c=COL[ds],marker=mk,s=46 if ds!="megadepth" else 30,
               edgecolors="white",lw=0.6,label=LBL[ds],
               alpha=0.95 if ds!="megadepth" else 0.7,zorder=3,clip_on=False)
ax.set_ylim(-2.0,2.0)
ax.set_xlim(0.4,65)
ax.set_xlabel("per-scene outlier rate (%, log scale)",fontsize=14,color=INK)
ax.set_ylabel(r"advantage $\log_{10}\!\frac{\mathrm{weight\ err}}{\mathrm{removal\ err}}$",fontsize=14,color=INK)
ax.text(0.5,1.75,"removal wins",fontsize=14,color="#D55E00",style="italic",fontweight="bold")
ax.text(0.5,-1.9,"weight wins",fontsize=14,color="#0072B2",style="italic",fontweight="bold")
ax.text(9,-1.9,"OOD flip zone\n(no OOD scenes here)",ha="center",fontsize=10,color=MUTED)
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
for sp in ["left","bottom"]: ax.spines[sp].set_color("#cccccc")
ax.tick_params(colors=MUTED)
ax.legend(loc="lower right",frameon=False,fontsize=11)
ax.set_title("Per-scene mechanism flip",
             fontsize=15,color=INK,fontweight="bold",loc="left",pad=8)
plt.tight_layout()
out="../claude specs/PATHA_fig4_perscene"
fig.savefig(out+".pdf",bbox_inches="tight"); fig.savefig(out+".png",dpi=200,bbox_inches="tight")
print("wrote",out+".pdf and .png")
print("counts:",{ds:len(pts[ds]) for ds in pts})