"""Emit the cross-dataset results in the exact layout of RESfM Tables 2-4.

Table 2 (1DSfM):      Scene | Nc | Outliers% | per-method Nr, Rot, Trans
Tables 3-4 (Strecha/  Scene | Nc | Out.%     | per-method Nr, Rot, Trans, Time
BlendedMVS):

Methods (our arms, 5-seed means): U-ESFM-DA (deep 2x3 adaptive = our "Ours"),
U-ESFM (deep 2x3 + MAD), ESFM (no removal), RESfM (official released ckpt).
Strecha uses the per-dataset lr policy for ESFM/U-ESFM (lr_tuning=1e-4).
Nc and Outliers% come from the scene npz (our tracks); Time = mean over seeds
of inference + fine-tune convergence + BA seconds from the per-scene xlsx.

Usage: python make_paper_style_tables.py
Writes results/crossdataset/paper_style_tables.md and .xlsx.
"""
import glob
import os
import re

import numpy as np
import pandas as pd

XD = "results/crossdataset"
# method label -> (root template, use per-dataset alternates)
METHODS = {
    "U-ESFM-DA": {"1dsfm": XD + "/uesfm_deep_adaptive_1dsfm_eval",
                  "strecha": XD + "/uesfm_deep_adaptive_strecha_eval",
                  "blendedmvs": XD + "/uesfm_deep_adaptive_blendedmvs_eval"},
    "U-ESFM": {"1dsfm": [XD + "/uesfm_1dsfm_eval", XD + "/uesfm_stage1_1dsfm_eval"],
               "strecha": XD + "/uesfm_stage1_ftlr1e4_strecha_eval",   # lr policy
               "blendedmvs": [XD + "/uesfm_blendedmvs_eval", XD + "/uesfm_stage1_blendedmvs_eval"]},
    "ESFM": {"1dsfm": XD + "/esfm_baseline_1dsfm_eval",
             "strecha": XD + "/esfm_baseline_ftlr1e4_strecha_eval",    # lr policy
             "blendedmvs": XD + "/esfm_baseline_blendedmvs_eval"},
    "RESfM": {"1dsfm": XD + "/resfm_repro_1dsfm",
              "strecha": XD + "/resfm_repro_strecha",
              "blendedmvs": XD + "/resfm_repro_blendedmvs"},
}
SEEDS = [20, 21, 22, 23, 24]
DS_SCENES = {
    "1dsfm": ["Alamo", "Ellis_Island", "Madrid_Metropolis", "Montreal_Notre_Dame",
              "Notre_Dame", "NYC_Library", "Piazza_del_Popolo", "Tower_of_London",
              "Vienna_Cathedral", "Yorkminster"],
    "strecha": ["entry-P10", "fountain-P11", "Herz-Jesu-P8", "Herz-Jesu-P25"],
    "blendedmvs": ["58c4bb4f4a69c55606122be4", "5acf8ca0f3d8a750097e4b15",
                   "5a48ba95c7dab83a7d7b44ed", "5b950c71608de421b1e7318f"],
}
# paper's anonymized names for BlendedMVS
BMVS_NAME = {"58c4bb4f4a69c55606122be4": "scene0", "5acf8ca0f3d8a750097e4b15": "scene1",
             "5a48ba95c7dab83a7d7b44ed": "scene2", "5b950c71608de421b1e7318f": "scene3"}


def scene_meta(ds, scene):
    d = np.load(f"datasets/{ds}/{scene}.npz", allow_pickle=True)
    return d["M"].shape[0] // 2, float(d["outlier_pct"])


def per_seed_rows(bases, scene):
    if isinstance(bases, str):
        bases = [bases]
    out = []
    for s in SEEDS:
        for b in bases:
            root = b if s == 20 else f"{b}_seed{s}"
            fs = glob.glob(os.path.join(root, f"{scene}_ba", "Results_FINE_TUNE*.xlsx"))
            if fs:
                r = pd.read_excel(fs[0]).iloc[-1]
                secs = float(r.get("inference_seconds", 0)) + \
                       float(r.get("Convergence time", 0)) + float(r.get("ba_seconds", 0))
                out.append((int(r["#registered_cams_final"]),
                            float(r["Rs_ba_final_mean"]),
                            float(r["ts_ba_final_mean"]), secs))
                break
    return out


def build(ds, with_time):
    rows = []
    for scene in DS_SCENES[ds]:
        nc, outp = scene_meta(ds, scene)
        row = {"Scene": BMVS_NAME.get(scene, scene.replace("_", " ")),
               "Nc": nc, "Out.%": round(outp, 1)}
        for m, roots in METHODS.items():
            vals = per_seed_rows(roots[ds], scene)
            if not vals:
                continue
            v = pd.DataFrame(vals, columns=["Nr", "Rot", "Trans", "Time"]).mean()
            row[f"{m} Nr"] = int(round(v["Nr"]))
            row[f"{m} Rot"] = round(v["Rot"], 2)
            row[f"{m} Trans"] = round(v["Trans"], 3)
            if with_time:
                row[f"{m} Time"] = int(round(v["Time"]))
        rows.append(row)
    df = pd.DataFrame(rows)
    mean_row = {"Scene": "Mean"}
    for c in df.columns[1:]:
        mean_row[c] = round(df[c].mean(), 3 if "Trans" in c else 2)
    return pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)


def main():
    titles = {"1dsfm": "Table 2-style: 1DSFM experiment (our tracks, 5-seed means)",
              "strecha": "Table 3-style: Strecha experiment (our tracks, 5-seed means; "
                         "ESFM/U-ESFM at the per-dataset lr 1e-4)",
              "blendedmvs": "Table 4-style: BlendedMVS experiment (our tracks, 5-seed means)"}
    md = ["# Cross-dataset results in RESfM Tables 2-4 layout\n",
          "Methods: U-ESFM-DA = deep 2x3 adaptive loss ('Ours'); U-ESFM = deep 2x3 + MAD;",
          "ESFM = no outlier removal; RESfM = official released checkpoint.",
          "Per scene: Nc / Out.% from our tracks; per method: Nr (registered cameras,",
          "5-seed mean), mean rotation (deg), mean translation; Time in seconds",
          "(inference + fine-tune + BA) for Tables 3-4.\n"]
    with pd.ExcelWriter(XD + "/paper_style_tables.xlsx") as xl:
        for ds in ["1dsfm", "strecha", "blendedmvs"]:
            df = build(ds, with_time=(ds != "1dsfm"))
            df.to_excel(xl, sheet_name=ds, index=False)
            md.append(f"\n## {titles[ds]}\n")
            md.append(df.to_markdown(index=False))
            print(titles[ds]); print(df.to_string(index=False)); print()
    open(XD + "/paper_style_tables.md", "w").write("\n".join(md) + "\n")
    print(f"wrote {XD}/paper_style_tables.md and .xlsx")


if __name__ == "__main__":
    main()
