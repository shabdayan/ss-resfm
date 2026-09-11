#!/usr/bin/env python
"""Submit lens-4 metric jobs for every existing matrix root lacking vgpa_metrics.json."""
import os, subprocess, importlib.util
spec = importlib.util.spec_from_file_location("a", "analyze_vgpa_style.py")
# analyze module runs analysis at import; reuse only roots_for by exec of its defs
src = open("analyze_vgpa_style.py").read().split("ARMS = ")[0]
ns = {"__file__": os.path.abspath("analyze_vgpa_style.py")}
exec(src, ns)
roots_for = ns["roots_for"]; SEEDS = ns["SEEDS"]; DS = ns["DS"]
NPZ = {"megadepth": "datasets/megadepth", "1dsfm": "datasets/1dsfm",
       "1dsfmhard": "datasets/1dsfm_hard_300", "strecha": "datasets/strecha",
       "blendedmvs": "datasets/blendedmvs", "olsson": "datasets/olsson"}
ARMS = ["madweight", "weight", "weight_ttt", "remove", "remove_ttt", "hybrid",
        "RESfM", "vanilla", "star"]
CODE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(CODE, "..", ".venv38-resfm", "bin", "python")
n = 0
for a in ARMS:
    for d in DS:
        todo = []
        for s in SEEDS:
            for r in roots_for(a, d, s):
                if os.path.isdir(r) and not os.path.exists(os.path.join(r, "vgpa_metrics.json")):
                    npz = NPZ[d]
                    if a == "star":
                        npz = {"olsson": "datasets/olsson_star"}.get(d, f"datasets/{NPZ[d].split('/')[-1]}_star")
                        if not os.path.isdir(os.path.join(CODE, npz)): npz = NPZ[d]
                    todo.append((r, npz)); break
        if not todo: continue
        cmd = " && ".join(f"{PY} {CODE}/compute_vgpa_metrics.py {r} {CODE}/{z}" for r, z in todo)
        subprocess.run(["bsub", "-q", "waic-risk", "-J", f"vgm_{a}_{d}",
                        "-oo", f"{CODE}/lsf_output/b00/vgm_{a}_{d}_%J.out",
                        "-eo", f"{CODE}/lsf_output/b00/vgm_{a}_{d}_%J.err",
                        "-R", "rusage[mem=24000]", f"cd {CODE} && {cmd}"],
                       capture_output=True)
        n += 1
print("submitted", n)
