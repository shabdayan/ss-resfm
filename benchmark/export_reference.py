#!/usr/bin/env python
"""Emit reference_results.json — the five-seed reference grid for RUC-SfM.

Values are the paper's printed numbers, machine-verified against the raw
evaluation outputs by code/verify_iclr_tables.py (322 checks green). Regenerate
after any table change; the verify script is the source of truth.
"""
import json, os

DS = ["megadepth", "olsson", "strecha", "blendedmvs", "1dsfm", "1dsfmhard"]

# trans mean+-std / pooled median / rot mean / Nr% / RA-AUC@30 / AUC@30 / cat>30 / cat>100
REF = {
 "ss_madweight":   {"megadepth": [0.512,0.08,0.151,3.70,81,0.599,0.778,0,0], "olsson": [3.010,0.34,1.169,14.44,86,0.568,0.640,0,0],
                    "strecha": [2.650,0.25,0.938,12.80,91,0.587,0.601,0,0], "blendedmvs": [0.140,0.02,0.004,9.01,92,0.589,0.642,0,0],
                    "1dsfm": [11.041,1.68,3.956,7.74,70,0.374,0.642,4,0], "1dsfmhard": [18.918,4.05,13.552,27.65,43,0.067,0.354,7,0]},
 "ss_weight":      {"megadepth": [0.512,0.09,0.188,3.28,79,0.582,0.777,0,0], "olsson": [2.906,0.33,1.139,13.24,90,0.626,0.670,0,0],
                    "strecha": [1.969,0.09,0.006,7.61,98,0.633,0.648,0,0], "blendedmvs": [0.144,0.03,0.015,9.17,94,0.594,0.635,0,0],
                    "1dsfm": [14.349,2.97,11.102,8.24,79,0.444,0.582,6,0], "1dsfmhard": [22.924,2.79,13.134,27.95,40,0.092,0.408,8,0]},
 "ss_weight_ttt":  {"megadepth": [0.398,0.11,0.175,2.88,77,0.580,0.777,0,0], "olsson": [8.033,0.30,0.433,11.60,90,0.627,0.674,5,5],
                    "strecha": [2.000,0.05,0.006,7.95,99,0.635,0.650,0,0], "blendedmvs": [0.137,0.02,0.002,8.30,94,0.595,0.637,0,0],
                    "1dsfm": [15.719,2.10,11.205,10.09,81,0.445,0.578,7,0], "1dsfmhard": [22.242,3.17,13.305,30.60,41,0.092,0.411,5,0]},
 "ss_remove":      {"megadepth": [0.512,0.08,0.117,4.04,72,0.442,0.755,0,0], "olsson": [3.413,0.57,1.619,13.89,78,0.468,0.584,0,0],
                    "strecha": [3.201,0.11,2.034,14.39,97,0.612,0.636,0,0], "blendedmvs": [0.240,0.06,0.160,16.04,88,0.500,0.577,0,0],
                    "1dsfm": [9.125,2.17,1.615,6.94,58,0.259,0.623,5,0], "1dsfmhard": [12.419,2.05,11.853,12.27,29,0.036,0.411,2,0]},
 "ss_remove_ttt":  {"megadepth": [0.561,0.07,0.143,4.22,71,0.442,0.748,0,0], "olsson": [3.626,0.35,1.913,14.53,78,0.467,0.581,0,0],
                    "strecha": [3.008,0.07,1.796,13.23,96,0.609,0.634,0,0], "blendedmvs": [0.219,0.10,0.211,13.19,90,0.501,0.578,0,0],
                    "1dsfm": [8.925,1.20,1.749,6.90,58,0.262,0.638,5,0], "1dsfmhard": [15.150,0.98,11.299,16.40,32,0.037,0.388,5,0]},
 "ss_hybrid":      {"megadepth": [0.455,0.09,0.179,3.06,77,0.559,0.760,0,0], "olsson": [2.905,0.23,1.559,13.35,88,0.531,0.644,0,0],
                    "strecha": [3.254,0.45,2.704,15.15,71,0.438,0.603,0,0], "blendedmvs": [0.140,0.02,0.004,9.01,93,0.594,0.637,0,0],
                    "1dsfm": [16.232,2.20,10.181,7.89,80,0.442,0.573,9,0], "1dsfmhard": [22.983,4.02,13.150,28.21,41,0.095,0.399,8,0]},
 "resfm_supervised": {"megadepth": [0.365,0.12,0.068,1.89,85,0.605,0.811,0,0], "olsson": [7.374,2.63,1.123,12.62,89,0.620,0.653,4,4],
                    "strecha": [0.388,0.72,0.005,1.56,99,0.725,0.735,0,0], "blendedmvs": [0.353,0.04,0.260,31.66,92,0.600,0.654,0,0],
                    "1dsfm": [10.498,1.45,2.558,9.62,77,0.408,0.623,5,0], "1dsfmhard": [19.158,3.42,13.714,27.30,34,0.044,0.329,6,0]},
 "esfm_no_mechanism": {"megadepth": [0.714,0.11,0.297,6.03,77,0.574,0.756,0,0], "olsson": [5.914,3.32,0.558,13.03,90,0.645,0.688,3,3],
                    "strecha": [2.034,0.14,0.005,7.61,100,0.662,0.677,0,0], "blendedmvs": [6.449,3.75,0.000,12.80,96,0.722,0.744,3,0],
                    "1dsfm": [18.415,0.75,13.409,14.53,84,0.461,0.570,10,0], "1dsfmhard": [21.758,0.80,24.280,32.01,40,0.099,0.409,6,0]},
 "esfm_star_oracle_clean": {"megadepth": [0.594,0.18,0.071,6.18,90,0.691,0.790,0,0], "olsson": [9.363,1.28,1.459,18.95,90,0.570,0.607,5,5],
                    "strecha": [1.139,0.45,0.007,11.13,99,0.517,0.519,0,0], "blendedmvs": [3.795,3.51,0.000,10.62,94,0.652,0.760,1,0],
                    "1dsfm": [13.236,0.52,2.951,13.29,80,0.461,0.669,5,0], "1dsfmhard": [18.121,4.35,15.248,21.55,36,0.035,0.330,2,0]},
 # single-run classical references (trans mean / rot; deterministic or replicate-banded)
 "glomap":  {"megadepth": [3.33,None,None,5.30,96,None,None,None,None], "olsson": [3.17,None,None,1.29,100,None,None,None,None],
             "strecha": [0.047,None,None,0.25,100,None,None,None,None], "blendedmvs": [0.339,None,None,2.15,86,None,None,None,None],
             "1dsfm": [27.21,None,None,8.78,96,None,None,None,None], "1dsfmhard": [35.42,None,None,26.4,80,None,None,None,None]},
 "colmap":  {"megadepth": [0.62,0.03,None,None,None,None,None,None,None], "olsson": [0.20,0.00,None,None,None,None,None,None,None],
             "strecha": [0.03,0.00,None,None,None,None,None,None,None], "blendedmvs": [0.01,0.00,None,None,None,None,None,None,None],
             "1dsfm": [6.29,0.42,None,None,None,None,None,None,None], "1dsfmhard": [20.4,1.1,None,None,None,None,None,None,None]},
 "gasfm":   {"megadepth": [2.25,None,None,None,None,None,None,None,None], "olsson": [4.45,None,None,None,None,None,None,None,None],
             "strecha": [1.14,None,None,None,None,None,None,None,None], "blendedmvs": [11.13,None,None,None,None,None,None,None,None],
             "1dsfm": [25.84,None,None,None,None,None,None,None,None], "1dsfmhard": [36.49,None,None,None,None,None,None,None,None]},
}

FIELDS = ["trans_mean", "trans_std", "trans_median_pooled", "rot_mean", "nr_pct",
          "ra_auc30", "auc30", "cat_cells_gt30", "cat_cells_gt100"]

out = {"benchmark": "RUC-SfM (working name)",
       "protocol": "5 training seeds (20-24); errors over registered cameras + coverage-priced RA-AUC; catastrophic-cell census over all scene x seed cells",
       "provenance": "printed paper values, machine-verified by code/verify_iclr_tables.py (322 checks green); COLMAP note: errors over its registered cameras only",
       "methods": {m: {d: dict(zip(FIELDS, v)) for d, v in cells.items()} for m, cells in REF.items()}}
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_results.json")
json.dump(out, open(path, "w"), indent=1)
print("WROTE", path)
