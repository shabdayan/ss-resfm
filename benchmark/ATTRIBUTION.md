# Attribution

All benchmark suites are *derived* data (point-track tensors + labels) built by
our released scripts from public datasets. The images and ground-truth
reconstructions belong to their authors — cite them alongside this benchmark:

| Suite | Source dataset | Citation |
|---|---|---|
| MegaDepth split | MegaDepth (Li & Snavely, CVPR 2018); track tensors from RESfM (Khatib et al., ICLR 2025) | both |
| 1DSfM, 1DSfM-hard-300 | 1DSfM (Wilson & Snavely, ECCV 2014) | Wilson & Snavely |
| Strecha | Strecha et al., CVPR 2008 | Strecha et al. |
| BlendedMVS | Yao et al., CVPR 2020 | Yao et al. |
| Olsson | Olsson & Enqvist, SCIA 2011 | Olsson & Enqvist |
| ETH3D pool | Schöps et al., CVPR 2017 | Schöps et al. |

Protocol lineage: the sets-of-sets track format follows ESFM (Moran et al.,
ICML 2021) and RESfM; the registration-aware AUC convention follows VGPA
(Khatib, Galun, Basri, ECCV 2026). The contamination axis, labels, splits,
reliability protocol, and reference grid are contributions of this benchmark.
