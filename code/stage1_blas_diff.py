#!/usr/bin/env python
"""Stage 1 of the environment-matching experiment: isolate the BLAS layer.

Run the exact triangulation/DLT step our pipeline uses to initialize robust BA
(geo_utils.n_view_triangulation on the raw M with Ns) for one scene, and dump
the resulting 3D points to an npz keyed by the numpy backend in use. Running
this under BOTH envs (pip/OpenBLAS .venv38-resfm and the authors' conda/MKL
env) and diffing the two dumps tells us whether the BA initialization differs
at all between environments -- the gate for Stage 2.

Usage: python stage1_blas_diff.py <dataset> <scene> [--out DIR]
       python stage1_blas_diff.py --compare <dump1.npz> <dump2.npz>
"""
import argparse, io, contextlib, os, sys
import numpy as np


def backend_tag():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        np.show_config()
    s = buf.getvalue().lower()
    return "mkl" if "mkl" in s else ("openblas" if "openblas" in s else "unknown")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("args", nargs="+")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--out", default="logs_authors_env")
    a = ap.parse_args()

    if a.compare:
        d1, d2 = (np.load(p) for p in a.args[:2])
        X1, X2 = d1["X"], d2["X"]
        ok = np.isfinite(X1) & np.isfinite(X2)
        diff = np.abs(X1 - X2)[ok]
        denom = np.maximum(np.abs(X1[ok]), 1e-12)
        rel = diff / denom
        print(f"backends: {d1['backend']} vs {d2['backend']}  points: {X1.shape}")
        print(f"finite-agree: {ok.mean()*100:.2f}%  (nan-pattern same: {np.array_equal(np.isfinite(X1), np.isfinite(X2))})")
        print(f"abs diff  max {diff.max():.3e}  mean {diff.mean():.3e}")
        print(f"rel diff  max {rel.max():.3e}  mean {rel.mean():.3e}  frac>1e-9: {(rel>1e-9).mean()*100:.2f}%")
        print("VERDICT:", "BIT-IDENTICAL" if diff.max() == 0 else
              ("NEGLIGIBLE (<1e-12 rel)" if rel.max() < 1e-12 else "DIFFERENT -> BLAS layer is live"))
        return

    dataset, scene = a.args[0], a.args[1]
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    # Execute the VERBATIM triangulation functions from geo_utils.py source in a
    # numpy-only namespace (the full module imports torch/cv2/cvxpy/dask, which
    # the minimal MKL env deliberately lacks). Extracting the same source keeps
    # the numerics identical to the pipeline; scenes with tracks seen by <=40
    # cameras never reach the dask branch.
    import re as _re
    src = open(os.path.join(here, "utils", "geo_utils.py")).read()
    ns = {"np": np, "da": None, "torch": None}
    for fn in ["pflat", "M_to_xs", "normalize_points_cams", "dlt_triangulation",
               "n_view_triangulation", "xs_valid_points"]:
        m = _re.search(rf"^def {fn}\(.*?(?=^def |\Z)", src, _re.S | _re.M)
        assert m, fn
        code = m.group(0).replace("\t", "    ")     # geo_utils mixes tabs
        exec(compile(code, f"geo_utils.{fn}", "exec"), ns)
    p = os.path.join(here, "datasets", dataset, f"{scene}.npz")
    d = np.load(p, allow_pickle=True)
    M, Ns = d["M"], d["Ns"]
    Ps = d["Ps_gt"]  # any consistent camera set exercises the same SVD path
    X = ns["n_view_triangulation"](Ps, M, Ns=Ns)  # (4,n) homogeneous DLT output
    tag = backend_tag()
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, f"stage1_{dataset}_{scene}_{tag}.npz")
    np.savez(out, X=X, backend=tag)
    print(f"[{tag}] wrote {out}  X {X.shape}  numpy {np.__version__}")


if __name__ == "__main__":
    main()
