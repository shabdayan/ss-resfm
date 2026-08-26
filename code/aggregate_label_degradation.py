#!/usr/bin/env python
"""Aggregate the label-degradation results into the curve: how COLMAP-derived
outlier-label quality (vs robust GT-pose truth) falls as contamination rises.
Reads logs_labeldeg/results.jsonl, prints a per-scene table sorted by
contamination + contamination-binned means, writes results.csv."""
import json, os, csv
import numpy as np

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "logs_labeldeg", "results.jsonl")
rows = []
with open(RES) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))
rows.sort(key=lambda r: r.get("contamination", 0))

def pct(x):
    return f"{100*x:5.1f}" if isinstance(x, (int, float)) and x == x else "  nan"
def f3(x):
    return f"{x:5.3f}" if isinstance(x, (int, float)) and x == x else "  nan"

hdr = f"{'dataset':<16}{'scene':<26}{'cams':>5}{'contam%':>8}{'reg%':>6}{'cov%':>7}{'P':>7}{'R':>7}{'F1':>7}{'poseT':>8}"
print(hdr); print("-" * len(hdr))
for r in rows:
    print(f"{r['dataset']:<16}{r['scene'][:25]:<26}{r['cams']:>5}{pct(r['contamination']):>8}"
          f"{pct(r['colmap_reg_frac']):>6}{pct(r['coverage']):>7}{f3(r['precision']):>7}"
          f"{f3(r['recall']):>7}{f3(r['f1']):>7}{r['colmap_pose_trans_mean']:>8.3f}")

# contamination-binned means (the curve)
bins = [(0, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 0.45), (0.45, 1.01)]
labels = ["0-5%", "5-15%", "15-30%", "30-45%", "45-100%"]
print("\n=== contamination-binned means (label-degradation curve) ===")
print(f"{'contam bin':<12}{'n':>3}{'cov%':>8}{'P':>8}{'R':>8}{'F1':>8}{'poseT':>9}")
for (lo, hi), lab in zip(bins, labels):
    sub = [r for r in rows if lo <= r.get("contamination", -1) < hi]
    if not sub:
        continue
    def m(key):
        v = [r[key] for r in sub if isinstance(r.get(key), (int, float)) and r[key] == r[key]]
        return np.mean(v) if v else float("nan")
    print(f"{lab:<12}{len(sub):>3}{100*m('coverage'):>8.1f}{m('precision'):>8.3f}"
          f"{m('recall'):>8.3f}{m('f1'):>8.3f}{m('colmap_pose_trans_mean'):>9.3f}")

# CSV
csv_path = os.path.join(HERE, "logs_labeldeg", "results.csv")
if rows:
    keys = ["dataset", "scene", "cams", "tracks", "obs", "contamination",
            "colmap_reg_frac", "coverage", "precision", "recall", "f1",
            "colmap_pose_trans_mean"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {csv_path}  ({len(rows)} scenes)")
