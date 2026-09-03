#!/usr/bin/env python
"""Eval-output prune (dry-run by default; --execute to delete).

Targets the per-scene diagnostic dumps inside evaluation scene dirs across all
results/ families. A "scene dir" is any directory that has eval-machinery
subdirs (models_all/TEST/plots). Policy per scene dir:

  - PRUNE only if its Results_*.xlsx summary exists (result harvested).
  - Delete: plots/, models_all/, models/, wandb/, TEST/ (after rescuing any
    Final_outliers.npz into <scene>/outliers_rescued/ -- the selector study
    reads these), and forFigures/ UNLESS the root is figure-protected.
  - Keep: every xlsx/json/npy at scene-dir top level, colmap_reconstructions/
    (the BA evidence), and anything unrecognized.
  - Figure-protected roots (forFigures kept): ONLY the declared seed-20
    paper rows (option-2 whitelist): uesfm_finelr_<arm>_*, resfm_finelr_*,
    resfm_faithful_* (no seed suffix), resfm_repro_*, resfm_deep_ln_*, and
    the released/menv repro roots. Everything else loses forFigures too.

Dry-run prints per-root: scene dirs, prunable, estimated reclaim (per-root
sample of one scene dir extrapolated) and the grand total.

Usage: prune_evalroots.py [--execute] [--families multiscene,crossdataset,...]
"""
import argparse, glob, os, re, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FAMILIES = ["multiscene", "crossdataset", "single_scene", "esfm_outliers",
            "esfm_outliers_deep", "crossdataset_full", "diagnosis",
            "single_scene_uesfm_kaiming_init_20260714"]
DELETE_ALWAYS = ["plots", "models_all", "models", "wandb", "TEST"]
SEEDVAR = re.compile(r"(pair_|_s2[1-4](_|$)|_seed2[1-4](_|$))")
# option-2 whitelist: only these root-name patterns keep forFigures (paper rows, seed 20)
FIGWHITELIST = re.compile(
    r"^(uesfm_finelr_(madweight|weight|weight_ttt|remove|remove_ttt|hybrid)(_|$)"
    r"|resfm_finelr_"
    r"|resfm_faithful_(?!s2)"
    r"|resfm_repro_"
    r"|resfm_deep_ln_"
    r"|uesfm_finelr_rf_)")


def du_bytes(path):
    total = 0
    for dp, dn, fn in os.walk(path):
        for f in fn:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return total


def scene_dirs(root):
    out = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(d):
            continue
        if any(os.path.isdir(os.path.join(d, s)) for s in ("models_all", "TEST", "plots")):
            out.append(d)
    return out


def prune_scene(d, fig_protected, execute):
    """Returns bytes (deleted or would-delete)."""
    freed = 0
    targets = list(DELETE_ALWAYS) + ([] if fig_protected else ["forFigures"])
    for sub in targets:
        p = os.path.join(d, sub)
        if not os.path.isdir(p):
            continue
        if sub == "TEST":
            rescued = glob.glob(os.path.join(p, "**", "Final_outliers.npz"), recursive=True)
            if rescued and execute:
                rdir = os.path.join(d, "outliers_rescued")
                os.makedirs(rdir, exist_ok=True)
                for i, r in enumerate(rescued):
                    shutil.copy2(r, os.path.join(rdir, f"Final_outliers_{i}.npz" if i else "Final_outliers.npz"))
        freed += du_bytes(p)
        if execute:
            shutil.rmtree(p, ignore_errors=True)
    return freed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--families", default=",".join(FAMILIES))
    args = ap.parse_args()

    grand = 0
    for fam in args.families.split(","):
        base = os.path.join(RES, fam)
        if not os.path.isdir(base):
            continue
        roots = sorted(d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d))
        for root in roots:
            sds = scene_dirs(root)
            if not sds:
                continue
            name = os.path.basename(root)
            fig_prot = bool(FIGWHITELIST.match(name)) and not SEEDVAR.search(name)
            prunable = [d for d in sds if glob.glob(os.path.join(d, "Results_*.xlsx"))]
            skipped = len(sds) - len(prunable)
            if not prunable:
                print(f"{fam}/{name}: 0/{len(sds)} harvested -- SKIP", flush=True)
                continue
            if args.execute:
                freed = sum(prune_scene(d, fig_prot, True) for d in prunable)
            else:
                sample = prune_scene(prunable[0], fig_prot, False)
                freed = sample * len(prunable)
            grand += freed
            tag = "FIGPROT" if fig_prot else ""
            print(f"{fam}/{name}: {len(prunable)}/{len(sds)} scenes, "
                  f"{'freed' if args.execute else 'est'} {freed/2**30:.1f}G "
                  f"{'(skip ' + str(skipped) + ' unharvested)' if skipped else ''} {tag}", flush=True)
    print(f"\nGRAND TOTAL {'FREED' if args.execute else 'ESTIMATE'}: {grand/2**40:.2f}T")


if __name__ == "__main__":
    main()
