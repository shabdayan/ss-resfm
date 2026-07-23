"""Emit the cross-dataset results in the exact layout of RESfM Tables 2-4.

Table 2 (1DSfM):      Scene | Nc | Outliers% | per-method Nr, Rot, Trans
Tables 3-4 (Strecha/  Scene | Nc | Out.%     | per-method Nr, Rot, Trans, Time
BlendedMVS):

Methods (our arms, 5-seed means): U-ESFM-DA / U-ESFM-SA (deep 2x3 / shallow
1x3 adaptive-loss checkpoints), each with a +TTT column (full unsupervised
CombinedLoss test-time training), U-ESFM (deep 2x3 + MAD), ESFM-deep (the
deep 2x3 stage-1 checkpoint with pruning disarmed), ESFM-van (true vanilla
1x3 ESFM — pending the esfm_vanilla training), RESfM-deep (supervised RESfM
recipe at deep 2x3 — pending resfm_deep_27scenes training), RESfM (official
released ckpt). Pending arms are skipped until their eval dirs exist.
Strecha uses the per-dataset lr policy for ESFM/U-ESFM arms (lr_tuning=1e-4).
Nc and Outliers% come from the scene npz (our tracks); Time = mean over seeds
of inference + fine-tune convergence + BA seconds from the per-scene xlsx.

Usage: python make_paper_style_tables.py
Writes results/crossdataset/paper_style_tables.{md,xlsx,pdf}.
"""
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

XD = "results/crossdataset"
TTT = "results/multiscene/ttt"
# method label -> per-dataset root (or list of alternates); "ttt": True marks
# roots that are ALWAYS seed-suffixed (TTT runner layout, incl. seed 20)
METHODS = {
    "U-ESFM-DA": {"1dsfm": XD + "/uesfm_deep_adaptive_1dsfm_eval",
                  "strecha": XD + "/uesfm_deep_adaptive_strecha_eval",
                  "blendedmvs": XD + "/uesfm_deep_adaptive_blendedmvs_eval"},
    "U-ESFM-DA+TTT": {"1dsfm": TTT + "/xd_1dsfm_deep_adaptive_ep16500_comb",
                      "strecha": TTT + "/xd_strecha_deep_adaptive_ep16500_comb",
                      "blendedmvs": TTT + "/xd_blendedmvs_deep_adaptive_ep16500_comb",
                      "ttt": True},
    "U-ESFM-SA": {"1dsfm": XD + "/uesfm_shallow_adaptive_ep17k_1dsfm_eval",
                  "strecha": XD + "/uesfm_shallow_adaptive_ep17k_ftlr1e4_strecha_eval",  # lr policy
                  "blendedmvs": XD + "/uesfm_shallow_adaptive_ep17k_blendedmvs_eval"},
    "U-ESFM-SA+TTT": {"1dsfm": TTT + "/xd_1dsfm_shallow_adaptive_ep17k_comb",
                      "strecha": TTT + "/xd_strecha_shallow_adaptive_ep17k_comb",
                      "blendedmvs": TTT + "/xd_blendedmvs_shallow_adaptive_ep17k_comb",
                      "ttt": True},
    "U-ESFM": {"1dsfm": [XD + "/uesfm_1dsfm_eval", XD + "/uesfm_stage1_1dsfm_eval"],
               "strecha": XD + "/uesfm_stage1_ftlr1e4_strecha_eval",   # lr policy
               "blendedmvs": [XD + "/uesfm_blendedmvs_eval", XD + "/uesfm_stage1_blendedmvs_eval"]},
    # A1 relabel: these runs load the deep 2x3 stage-1 checkpoint with pruning
    # disarmed (output_mode=1) — "deep ESFM", not vanilla ESFM.
    "ESFM-deep": {"1dsfm": XD + "/esfm_baseline_1dsfm_eval",
                  "strecha": XD + "/esfm_baseline_ftlr1e4_strecha_eval",  # lr policy
                  "blendedmvs": XD + "/esfm_baseline_blendedmvs_eval"},
    # pending: true vanilla ESFM (TASK_esfm_vanilla_baseline, training in flight)
    "ESFM-van": {"1dsfm": XD + "/esfm_vanilla_1dsfm_eval",
                 "strecha": XD + "/esfm_vanilla_ftlr1e4_strecha_eval",   # lr policy
                 "blendedmvs": XD + "/esfm_vanilla_blendedmvs_eval"},
    # pending: RESfM's supervised recipe at matched deep 2x3 (resfm_deep_27scenes)
    "RESfM-deep": {"1dsfm": XD + "/resfm_deep_1dsfm_eval",
                   "strecha": XD + "/resfm_deep_strecha_eval",
                   "blendedmvs": XD + "/resfm_deep_blendedmvs_eval"},
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

# RESfM paper Tables 2-4 "Ours" (THEIR tracks + their GT — reference only;
# extracted from resfm.pdf). (Nr, Rot, Trans) or (Nr, Rot, Trans, Time).
PAPER_VALS = {
    "Alamo": (484, 3.66, 0.515), "Ellis Island": (214, 0.82, 0.122),
    "Madrid Metropolis": (244, 8.42, 0.827), "Montreal Notre Dame": (346, 2.82, 0.352),
    "Notre Dame": (517, 1.20, 0.231), "NYC Library": (224, 3.96, 0.429),
    "Piazza del Popolo": (249, 2.20, 0.186), "Tower of London": (94, 0.67, 0.026),
    "Vienna Cathedral": (479, 1.52, 0.112), "Yorkminster": (331, 14.54, 1.468),
    "entry-P10": (10, 0.024, 0.008, 3), "fountain-P11": (11, 0.028, 0.003, 5),
    "Herz-Jesu-P8": (8, 0.026, 0.004, 3), "Herz-Jesu-P25": (24, 0.030, 0.006, 9),
    "scene0": (75, 0.016, 0.0007, 54), "scene1": (51, 0.011, 0.0021, 32),
    "scene2": (33, 0.009, 0.0006, 21), "scene3": (66, 0.007, 0.0007, 52),
}


def scene_meta(ds, scene):
    d = np.load(f"datasets/{ds}/{scene}.npz", allow_pickle=True)
    return d["M"].shape[0] // 2, float(d["outlier_pct"])


def per_seed_rows(bases, scene, always_suffixed=False):
    if isinstance(bases, str):
        bases = [bases]
    out = []
    for s in SEEDS:
        for b in bases:
            root = f"{b}_seed{s}" if (always_suffixed or s != 20) else b
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
            vals = per_seed_rows(roots[ds], scene, roots.get("ttt", False))
            if not vals:
                continue
            v = pd.DataFrame(vals, columns=["Nr", "Rot", "Trans", "Time"]).mean()
            row[f"{m} Nr"] = int(round(v["Nr"]))
            row[f"{m} Rot"] = round(v["Rot"], 2)
            row[f"{m} Trans"] = round(v["Trans"], 3)
            if with_time:
                row[f"{m} Time"] = int(round(v["Time"]))
        pv = PAPER_VALS.get(row["Scene"])
        if pv:
            row["Paper Nr"], row["Paper Rot"], row["Paper Trans"] = pv[0], pv[1], pv[2]
            if with_time and len(pv) > 3:
                row["Paper Time"] = pv[3]
        rows.append(row)
    df = pd.DataFrame(rows)
    mean_row = {"Scene": "Mean"}
    for c in df.columns[1:]:
        mean_row[c] = round(df[c].mean(), 3 if "Trans" in c else 2)
    return pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)


PROVENANCE = """\
## Column provenance (exact inference protocol and loss configuration)

Every one of our columns runs the same per-scene pipeline on our tracks —
frozen multi-scene checkpoint -> TEST pass -> arm-specific outlier removal ->
1001-epoch per-scene fine-tune -> robust BA (`ba.filter_outliers = 5.0` px) —
i.e. **protocol-mode**; no column is frozen-only. The +TTT columns replace the
plain fine-tune with **full TTT** (unsupervised CombinedLoss, outlier head
active, snapshots {0,10,100,1000}; the tabulated row is the final step).
Fine-tune lr follows the per-dataset policy (`train.lr_tuning`): 1DSfM 5e-3,
Strecha 1e-4, BlendedMVS 5e-3 — for ALL our arms on ALL three datasets.
MAD pruning = `test.outlier_source = "mad"`, alpha 2.0 on the TEST-pass
reprojection errors; the adaptive arms' learned scores are NOT used to prune.
`test.outliers_threshold` (classifier gates): 0.6 on 1DSfM, 0.8 on
Strecha/BlendedMVS.

| Column | Checkpoint (training loss) | Removal | Fine-tune loss | Mode |
|---|---|---|---|---|
| U-ESFM-DA | uesfm_27scenes_adaptive Ep16500, deep 2x3; unsupervised CombinedLoss = reproj (w 1.0) + confident-pseudo-label BCE (w 0.3); fixed percentiles inlier 20 / outlier 80, warmup 10 ep, min_threshold_separation 0, hinge w 1 — PRE-A2.5 (contamination-linked percentile OFF) | MAD | ESFMLoss | protocol |
| U-ESFM-DA+TTT | same Ep16500 checkpoint | MAD | CombinedLoss (same pre-A2.5 config) | full TTT |
| U-ESFM-SA | uesfm_27scenes_shallow_adaptive Ep17000, shallow 1x3; same CombinedLoss config | MAD | ESFMLoss | protocol |
| U-ESFM-SA+TTT | same Ep17000 checkpoint | MAD | CombinedLoss | full TTT |
| U-ESFM | uesfm_27scenes_stage1 Ep19500, deep 2x3; plain ESFMLoss | MAD | ESFMLoss | protocol |
| ESFM-deep | SAME stage1 Ep19500 checkpoint, output_mode=1 | none | ESFMLoss | protocol |
| ESFM-van | esfm_vanilla (1x3, plain ESFMLoss) — PENDING | none | ESFMLoss | protocol |
| RESfM-deep | resfm_deep_27scenes, deep 2x3; supervised CombinedLossSupervised = outlier-weighted reproj (w 1.0) + GT-label BCE (w 0.1) — PENDING | classifier | ESFMLoss | protocol |
| RESfM | official released `pretrained/pretrained_model.pt` | classifier | ESFMLoss | protocol |

Aggregation: every cell is the per-scene mean over seeds 20-24 of the post-BA
`Rs_ba_final_mean` / `ts_ba_final_mean`; the Mean row averages scenes. The
1DSfM U-ESFM-DA mean of 5.73 deg is exactly this aggregation of the
`uesfm_deep_adaptive_1dsfm_eval[_seed*]` runs (single-seed means range
4.16-6.69; seed 20 alone gives 5.35).
"""


def main():
    titles = {"1dsfm": "Table 2-style: 1DSFM experiment (our tracks, 5-seed means)",
              "strecha": "Table 3-style: Strecha experiment (our tracks, 5-seed means; "
                         "ESFM/U-ESFM at the per-dataset lr 1e-4)",
              "blendedmvs": "Table 4-style: BlendedMVS experiment (our tracks, 5-seed means)"}
    md = ["# Cross-dataset results in RESfM Tables 2-4 layout\n",
          "Methods: U-ESFM-DA / U-ESFM-SA = deep 2x3 / shallow 1x3 adaptive-loss",
          "checkpoints (+TTT = full unsupervised CombinedLoss test-time training);",
          "U-ESFM = deep 2x3 + MAD; ESFM-deep = deep 2x3 stage-1 ckpt, pruning",
          "disarmed (A1 relabel of the former 'ESFM' column); ESFM-van = true",
          "vanilla 1x3 ESFM (pending); RESfM-deep = supervised RESfM recipe at",
          "deep 2x3 (pending); RESfM = official released checkpoint.",
          "Per scene: Nc / Out.% from our tracks; per method: Nr (registered cameras,",
          "5-seed mean), mean rotation (deg), mean translation; Time in seconds",
          "(inference + fine-tune + BA) for Tables 3-4.",
          "Paper columns = RESfM Tables 2-4 'Ours' on THEIR tracks + their GT",
          "(reference only; 1DSfM Paper Trans is in their GT scale).\n",
          PROVENANCE]
    with pd.ExcelWriter(XD + "/paper_style_tables.xlsx") as xl, \
         PdfPages(XD + "/paper_style_tables.pdf") as pdf:
        for ds in ["1dsfm", "strecha", "blendedmvs"]:
            df = build(ds, with_time=(ds != "1dsfm"))
            df.to_excel(xl, sheet_name=ds, index=False)
            md.append(f"\n## {titles[ds]}\n")
            md.append(df.to_markdown(index=False))
            print(titles[ds]); print(df.to_string(index=False)); print()

            # Chunk the method column-groups so each landscape page stays legible
            # (base scene columns repeat on every page).
            base_cols = [c for c in df.columns if " " not in c or c == "Out.%"]
            groups = [[c for c in df.columns if c.startswith(m + " ")]
                      for m in list(METHODS) + ["Paper"]]
            groups = [g for g in groups if g]
            per_page = 4
            chunks = [groups[i:i + per_page] for i in range(0, len(groups), per_page)]
            for pi, chunk in enumerate(chunks):
                page_df = df[base_cols + [c for g in chunk for c in g]]
                fig = plt.figure(figsize=(16.5, 5.8))  # wide landscape
                fig.text(0.03, 0.93,
                         f"{titles[ds]} ({pi + 1}/{len(chunks)})",
                         fontsize=12, weight="bold", color="#1a1f24")
                fig.text(0.03, 0.86,
                         "U-ESFM-DA/-SA = deep/shallow adaptive (+TTT = full "
                         "CombinedLoss TTT) | U-ESFM = deep + MAD | ESFM-deep = "
                         "deep ckpt, no removal | ESFM-van = vanilla (pending) | "
                         "RESfM-deep = supervised 2x3 (pending) | RESfM = released "
                         "ckpt | Paper = RESfM Tables 2-4 on THEIR tracks (reference)",
                         fontsize=8, color="#5b6570")
                fig.text(0.03, 0.80, page_df.to_string(index=False), fontsize=6.5,
                         family="monospace", va="top", color="#1a1f24",
                         linespacing=1.6)
                pdf.savefig(fig)
                plt.close(fig)
    open(XD + "/paper_style_tables.md", "w").write("\n".join(md) + "\n")
    print(f"wrote {XD}/paper_style_tables.md, .xlsx and .pdf")


if __name__ == "__main__":
    main()
