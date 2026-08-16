#!/usr/bin/env python3
"""Generate report-faithful HEAD-based eval templates: mechanism {remove,weight,
hybrid} x fine-tune {frozen,ttt} x arch {deep,shallow} x dataset {1dsfm,strecha,
blendedmvs,megadepth}. Derives from existing head bases (advhead for OOD) and the
MegaDepth adaptive templates (MAD stripped -> head-based). Writes to confs/."""
import re, os

CONF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "confs")

# base template per (arch, dataset). OOD deep -> advhead_{ds}; OOD shallow ->
# advhead_sa_{ds}; MegaDepth -> megadepth_{arch}_adaptive (MAD stripped below).
def base_for(arch, ds):
    if ds == "megadepth":
        return f"megadepth_{'deep' if arch=='deep' else 'shallow'}_adaptive"
    suff = "" if arch == "deep" else "_sa"
    return f"crossdataset_advhead{suff}_{ds}"

def transform(text, mechanism, ttt):
    out = []
    for ln in text.splitlines():
        st = ln.strip()
        # make head-based: drop MAD source + any knobs we re-set below
        if st.startswith("outlier_source") or st.startswith("mad_alpha"):
            continue
        if st.startswith("fine_tune_output_mode"):
            continue
        if st.startswith("weight_not_remove") or st.startswith("hybrid_remove_weight"):
            continue
        if re.match(r"func_tuning\s*=", st):
            if ttt:
                out.append("    func_tuning = CombinedLoss")
            elif mechanism in ("weight", "hybrid"):
                out.append("    func_tuning = ESFMLoss_weighted_by_rep_err")
            else:
                out.append("    func_tuning = ESFMLoss")
            continue
        out.append(ln)
    lines = out
    # insert fine_tune_output_mode after the train-block 'output_mode = 3'
    ftom = 3 if ttt else (1 if mechanism in ("weight", "hybrid") else None)
    if ftom is not None:
        for i, ln in enumerate(lines):
            if ln.strip() == "output_mode = 3":
                lines.insert(i + 1, f"    fine_tune_output_mode = {ftom}")
                break
    # insert test flags after 'outliers_threshold = 0.6' (in the test block)
    flags = []
    if mechanism == "weight":
        flags = ["    weight_not_remove = True"]
    elif mechanism == "hybrid":
        flags = ["    hybrid_remove_weight = True", "    hybrid_low_pct = 20.0", "    hybrid_high_pct = 80.0"]
    if flags:
        for i, ln in enumerate(lines):
            if ln.strip().startswith("outliers_threshold"):
                for k, f in enumerate(flags):
                    lines.insert(i + 1 + k, f)
                break
    return "\n".join(lines) + "\n"

made = []
for arch in ("deep", "shallow"):
    for ds in ("1dsfm", "strecha", "blendedmvs", "megadepth"):
        bpath = os.path.join(CONF, base_for(arch, ds) + ".conf.template")
        if not os.path.exists(bpath):
            print("MISSING BASE:", bpath); continue
        btxt = open(bpath).read()
        for mech in ("remove", "weight", "hybrid"):
            for ttt in (False, True):
                tt = "_ttt" if ttt else ""
                if ds == "megadepth":
                    name = f"megadepth_rf_{mech}{tt}_{arch}.conf.template"
                else:
                    name = f"crossdataset_rf_{mech}{tt}_{arch}_{ds}.conf.template"
                open(os.path.join(CONF, name), "w").write(transform(btxt, mech, ttt))
                made.append(name)
print(f"generated {len(made)} templates")
for m in made:
    print(" ", m)
