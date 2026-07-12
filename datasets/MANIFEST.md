# Dataset manifest — RESfM-style cross-dataset SfM evaluation

Generated 2026-07-12 per `claude specs/SPEC_sfm_datasets_setup.md`.
Mode chosen: **Mode A verification** (preprocessed RESfM benchmark, already on disk)
plus **Mode B raw downloads for 1DSfM, Strecha, BlendedMVS** (MegaDepth raw skipped —
official RESfM tracks already cover it).

## Layout

```
datasets/
├── resfm_benchmark/megadepth -> ../../code/datasets/megadepth   (68 npz, 77 GB, official RESfM tracks)
├── pretrained -> ../code/pretrained                             (pretrained_model.pt, 8,800,077 B)
├── raw/1dsfm/<Scene>/          15 scenes (bundler GT + EGs + tracks; no raw images)
├── raw/strecha/<Scene>/        6 scenes (images + GT .camera/.P files)
├── raw/blendedmvs/BlendedMVS/  113 scenes (blended_images + cams GT + rendered_depth_maps)
└── MANIFEST.md
```

## Artifacts

| dataset | mode | source URL | archive | size (bytes) | sha256 | date |
|---|---|---|---|---|---|---|
| RESfM benchmark (MegaDepth tracks) | A | Dropbox link in https://github.com/FadiKhatib/resfm README ("Data and Pretrained Model") | datasets.zip | 487,382,109 | d834d4125632ad2c5af618765d293a0633d11dc00cba1eafccbcc4f609d9c382 | 2026-07-12 |
| RESfM pretrained model | A | Dropbox link in https://github.com/FadiKhatib/resfm README | pretrained_model.zip (kept at code/pretrained/) | 7,619,446 | 16419cbdc3ecd65f4f0c95855a41d63d88a6548690d26a41365b81ed403cf352 | pre-existing, verified 2026-07-12 |
| 1DSfM (all 15 scenes, reconstruction data) | B | http://landmark.cs.cornell.edu/projects/1dsfm/datasets.tar.gz | raw/1dsfm/datasets.tar.gz | 682,117,483 | 2a608789cef64083177f68da9dfb725d791cf56ba51684a4a70304f14f0a7147 | 2026-07-12 |
| BlendedMVS low-res (113 scenes) | B | https://github.com/YoYo000/BlendedMVS/releases/tag/v1.0.0 (split zip BlendedMVS.z01–z15 + BlendedMVS.zip, recombined with `zip -s 0`; split parts deleted after verified extraction) | raw/blendedmvs/BlendedMVS_combined.zip | 29,532,430,181 | see below | 2026-07-12 |
| Strecha fountain-P11 | B | http://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/multiview/denseMVS.html (official EPFL) | fountain_dense_{images,cameras,p,bounding}.tar.gz | — | see per-file table | 2026-07-12 |
| Strecha Herz-Jesu-P8 | B | same | herzjesu_dense_{images,cameras,p,bounding}.tar.gz | — | see per-file table | 2026-07-12 |
| Strecha Herz-Jesu-P25 | B | same | herzjesu_dense_large_{images,cameras,p,bounding}.tar.gz | — | see per-file table | 2026-07-12 |
| Strecha castle-P19 | B | same | castle_dense_{images,cameras,p,bounding}.tar.gz | — | see per-file table | 2026-07-12 |
| Strecha castle-P30 | B | same | castle_dense_large_{images,cameras,p,bounding}.tar.gz | — | see per-file table | 2026-07-12 |
| Strecha entry-P10 | B | same | castle_entry_dense_{images,cameras,p,bounding}.tar.gz | — | see per-file table | 2026-07-12 |
| Olsson/ESfM Euclidean (39 scene tracks, extracted to code/datasets/Euclidean/) | A (preprocessed) | ESfM release (source URL not recorded by the downloading session; wget.log empty) | olsson_download/esfm_datasets.zip | 8,105,428,411 | 799402d5a3d08aad103c1f2fc5cdd8daeaf2eef6edfb60b7ee252d4b738f1719 | 2026-07-12 |

Per-file sha256 for all 39 Euclidean npz: see `MANIFEST_euclidean_sha256.txt`
(downloaded/verified by a parallel session, recorded here for completeness).

BlendedMVS_combined.zip sha256:
`5e57c624895c6e3ebeb2546f34d194ff23a1e4057ce44e93b5f5486f2777cb13`

### Strecha per-file sha256

```
6939a88a5ac421b3c0264aecefe71eec18c0ede2f0daf5a889daa3d641c1e2fc  castle-P19/castle_dense_bounding.tar.gz
352a5386f0e5c69a161c0c6385f7fb4e90bc2596bd7aea5f2babea31d8c17efd  castle-P19/castle_dense_cameras.tar.gz
a12c820637ed67c42197070ad339ce62c236f92111177e810877dce63cc99fb9  castle-P19/castle_dense_images.tar.gz
44582042a3a3b2887ae51d569b1a1960add0e84590cbcaff1e99272a55ce5c9d  castle-P19/castle_dense_p.tar.gz
577deabed70845edd7692304fbfb444a3a0fe19cd9373b5a6557d622f5e2852a  castle-P30/castle_dense_large_bounding.tar.gz
2ad2fd0f174f3e1aa213c5bb959851c1319afd18fcabf750065754d825a34abf  castle-P30/castle_dense_large_cameras.tar.gz
a3ebbe460c2458d59d7b2fff8d5f3299177d820e5fa55594642b666752a2ca3d  castle-P30/castle_dense_large_images.tar.gz
576243cad9a9b0af129b115b9d867951cf1c92ad9d84d02b9654cd4eca06be67  castle-P30/castle_dense_large_p.tar.gz
b10be35b742214b65cf522edc95b11466f5fd7c6302781750db4563e969f96a4  entry-P10/castle_entry_dense_bounding.tar.gz
d965b8f3c285b8810534827d0d97162893aa727aafe228041269005dccac22ce  entry-P10/castle_entry_dense_cameras.tar.gz
b7198d73048ad894d23e917c0df654b89b3b65618cbd40a833f633da0735177b  entry-P10/castle_entry_dense_images.tar.gz
8f9ea9b1caed9a21e7753c516e94fa0759112c346f3e6e9b5757c091b1714940  entry-P10/castle_entry_dense_p.tar.gz
8668e4c652cf45147119cfff35bf5ecd023b4847ebbbfb6c895e2a4b2fd11d92  fountain-P11/fountain_dense_bounding.tar.gz
6d4724b6868832289844149f1a519c23e8504e0cce9beb3356c73273e5217fc5  fountain-P11/fountain_dense_cameras.tar.gz
94df79b5645a0ca07f82cee30e59e1cf9e0cefaefae817a4adb80a0f02e30cd5  fountain-P11/fountain_dense_images.tar.gz
629ea77156cd29126ca064f8ee92fec49020404c955d6fe57f9dfae6468ec5bb  fountain-P11/fountain_dense_p.tar.gz
869ec53b37a64cf12f54490b817cef863fc084a686c88bb5bcbf8efaa39f2635  Herz-Jesu-P25/herzjesu_dense_large_bounding.tar.gz
5be217b0f6bfe05d3df501804d3513caae36d0865a417bced5da20371219bc04  Herz-Jesu-P25/herzjesu_dense_large_cameras.tar.gz
2d45b5e036245a2971ca810213106af3a98304319babc145e32c1830ad450b6e  Herz-Jesu-P25/herzjesu_dense_large_images.tar.gz
eaef967fca179f52882b06729dd0be437e982ff39601fd6b66946559ccb8dccb  Herz-Jesu-P25/herzjesu_dense_large_p.tar.gz
7b11ac6d86c8078b869e4be582c85861ef2cf97edcbf8207f3fbf091e6e2e31e  Herz-Jesu-P8/herzjesu_dense_bounding.tar.gz
6362513b483ca75704f2e5600d7c715c9ba9d306c84c3e691802e7f99488cbcc  Herz-Jesu-P8/herzjesu_dense_cameras.tar.gz
875bc1fb0f928d39f6ebcbc5c62fd8fde379a059ce34be97b5e36f55cb5f07cd  Herz-Jesu-P8/herzjesu_dense_images.tar.gz
40347e26e8169b6de9a02d17725307e97284ea42f48ca42f46c837af98172232  Herz-Jesu-P8/herzjesu_dense_p.tar.gz
```

## Verification summary

### Mode A — RESfM benchmark
- Official `datasets.zip` (re-downloaded 2026-07-12 from the current README link) contains
  **only** `datasets/megadepth/` — 68 npz, nothing else. No 1DSfM/Strecha/BlendedMVS tracks
  have been released (README checklist "Release all the datasets" still unchecked).
- All 68 local npz at `code/datasets/megadepth/` match the official archive **scene-for-scene
  in exact uncompressed byte size** (local names have `" new"`/`"_300"` suffixes stripped).
- Load test: `0007.npz` opens with keys `K_gt, M, Ns, Ps_gt, R_gt, T_gt, connected,
  namesList, outlier_pct, outliers2, two_connected`; `M` shape (580, 61354).
- `pretrained_model.zip` passes `unzip -t`; its single member `pretrained_model.pt`
  (8,800,077 B) is byte-size-identical to the extracted `code/pretrained/pretrained_model.pt`.
- The re-downloaded `datasets.zip` was kept only transiently for verification (session
  scratchpad), not stored under `datasets/`.

### 1DSfM — 3.5 GB, 15 scenes
Each scene has `cc.txt, coords.txt, EGs.txt, gt_bundle.out, list.txt, tracks.txt`.
Images listed per scene: Alamo 2915, Ellis_Island 2587, Gendarmenmarkt 1463,
Madrid_Metropolis 1344, Montreal_Notre_Dame 2298, Notre_Dame 553, NYC_Library 2550,
Piazza_del_Popolo 2251, Piccadilly 7351, Roman_Forum 2364, Tower_of_London 1576,
Trafalgar 15685, Union_Square 5961, Vienna_Cathedral 6288, Yorkminster 3368.
- Raw Flickr images are NOT in this bundle (reconstruction data only). Unlike the spec's
  note, per-scene image tarballs ARE now hosted at
  `http://landmark.cs.cornell.edu/projects/1dsfm/images.<Scene>.tar` if ever needed.

### Strecha — 1.5 GB, 6 scenes (official EPFL server, no unofficial mirror needed)
fountain-P11 (11), Herz-Jesu-P8 (8), Herz-Jesu-P25 (25), castle-P19 (19),
castle-P30 (30), entry-P10 (10) — image counts match scene names exactly; every image has
LIDAR-derived GT `.camera` (intrinsics+pose) and `.P` (projection matrix) files, plus
`.bounding`. The current EPFL lab page no longer hosts these; the classic
`documents.epfl.ch` server does and is official.

### BlendedMVS — 60 GB extracted, 113 scenes (low-res set, 768×576)
Each scene: `blended_images/`, `cams/` (per-view `*_cam.txt` with extrinsic + intrinsic —
the real GT poses), `rendered_depth_maps/`. Downloaded from the GitHub release (stable
alternative to the OneDrive links), passed full `unzip -t` before extraction.

## Failures
None. All four datasets obtained; every archive passed integrity checks.

## Licensing / citation reminder
These datasets are for research use only. Cite:
- MegaDepth: Li & Snavely, CVPR 2018
- 1DSfM: Wilson & Snavely, ECCV 2014
- Strecha: Strecha et al., CVPR 2008 (cite scenes by exact name, e.g. fountain-P11)
- BlendedMVS: Yao et al., CVPR 2020
- RESfM: Khatib et al., ICLR 2025 (when using their preprocessed tracks / pretrained model)

---

## Euclidean (Olsson) — ESFM single-scene benchmark

Added 2026-07-12 per `claude specs/SPEC_single_scene_experiments.md` (prerequisite
dataset setup executed as part of that spec).

| dataset | mode | source URL | archive | size | sha256 | date |
|---|---|---|---|---|---|---|
| Olsson Euclidean point tracks (36 Olsson + 3 DTU scenes) | preprocessed (ESFM release) | https://www.dropbox.com/sh/s2714jqsstwp9uc/AAAhFdqDoyK0naDG7eA6dd3Ra?dl=1 (from drormoran/Equivariant-SFM README) | `olsson_download/esfm_datasets.zip` (Euclidean/ extracted; Projective/ left in the zip) | 2,129,755,970 B | `799402d5a3d08aad103c1f2fc5cdd8daeaf2eef6edfb60b7ee252d4b738f1719` | 2026-07-12 |

- Extracted to `datasets/Euclidean/` (39 npz; keys `M, Ps_gt, Ns, K_gt, R_gt, T_gt, namesList`).
- Shared between BOTH single-scene repos via symlinks (identical inputs by construction):
  - `u-esfm/code/datasets/Euclidean -> u-esfm/datasets/Euclidean`
  - `esfm-baseline/datasets/Euclidean -> u-esfm/datasets/Euclidean`
- Per-file sha256: `datasets/MANIFEST_euclidean_sha256.txt`.
- Research use; cite Olsson & Enqvist (SCIA 2011) and ESFM (Moran et al., ICCV 2021).
