# Track construction for the cross-dataset scenes (RESfM Appendix C)

Builds the npz scene files the pipeline consumes (`code/datasets/<dataset>/<scene>.npz`,
schema: `M [2m,n]` pixel tracks with 0 = unobserved, `Ns [m,3,3]` = K^-1,
`Ps_gt [m,3,4]` = K[R|t] in pixels, `outliers2 [m,n]` GT outlier mask,
`outlier_pct`, `namesList`, `K_gt`) for the three cross-dataset test sets of
RESfM Table 2-4. **Provenance: these are OUR tracks, not RESfM's released
ones** — per the R13 provenance switch, RESfM comparison rows on these
datasets are "RESfM (retrained/evaluated on our tracks)".

## Scene lists (from the paper)

- **1DSfM (Table 2, 10 scenes):** Alamo, Ellis_Island, Madrid_Metropolis,
  Montreal_Notre_Dame, Notre_Dame, NYC_Library, Piazza_del_Popolo,
  Tower_of_London, Vienna_Cathedral, Yorkminster.
- **Strecha (Table 3, 4 scenes):** entry-P10, fountain-P11, Herz-Jesu-P8,
  Herz-Jesu-P25 (the castle scenes are not used).
- **BlendedMVS (Table 4):** anonymized as scene0-3 with Nc = 75/51/33/66 and
  outliers 2.0/1.4/2.2/8.8%. Local scenes matching those camera counts:
  75 -> {58c4bb4f..., 5b6e716d...}, 51 -> {5acf8ca0...}, 66 -> {5b950c71...,
  5bc5f0e8..., 5be883a4..., 5bfd0f32...}, 33 -> {5a48ba95..., 5a572fd9...}.
  We build all 9 candidates and pick per count the scene whose outlier
  fraction is closest to the paper's.

## Appendix C parameters (paper) and our choices for the unspecified ones

| step | paper | ours |
|---|---|---|
| features | SIFT | OpenCV SIFT, capped at 8192/image (COLMAP default; unspecified in paper) |
| matching | exhaustive pairwise RANSAC | Lowe ratio 0.8 + `findFundamentalMat` RANSAC, 3 px, conf 0.999; pairs with <15 inliers dropped |
| chaining | chain two-view matches | union-find over (image, keypoint) nodes |
| track validity | >=3 cameras, no two keypoints in same image | identical |
| outlier labels | COLMAP-track membership -> clean -> triangulate -> 4 px relabel | see per-dataset GT below, 4 px threshold identical |

## Ground truth per dataset

- **Strecha:** released per-image `.P` (used verbatim as `Ps_gt`) and `.camera`
  (first 9 numbers = K -> `Ns`). Labeling: no reference reconstruction exists,
  so instead of COLMAP membership we RANSAC-triangulate each track under the
  GT cameras (best <4 px consensus over view pairs, re-triangulated) and
  apply the 4 px relabel. High-res 3072x2048 images used as released.
- **BlendedMVS:** `cams/<id>_cam.txt` (MVSNet format), `extrinsic` = 4x4
  world-to-cam E, `intrinsic` = K at low-res resolution; `Ps_gt = K @ E[:3,:]`.
  Images: `blended_images/<id>.jpg` (non-masked). Labeling as for Strecha.
- **1DSfM:** no images needed — the release ships the original SIFT tracks
  (`tracks.txt` + `coords.txt`). GT cameras from `gt_bundle.out` (cameras with
  focal > 0). Bundler conventions: camera looks down -z with y up =>
  `P = K @ [D R | D t]`, `D = diag(1,-1,-1)`; K = diag(f,f,1) with principal
  point (px,py) from coords.txt. Labeling uses the linear model: coords.txt
  keypoints are already undistorted (adding bundler's k1,k2 was tested and
  raised Notre_Dame's outlier rate 48.1% -> 51.4%, i.e. it over-corrects).
  Labeling follows Appendix C most literally: initial inlier = the (image,
  keypoint) appears in a gt_bundle point's view list (bundler membership
  standing in for COLMAP membership), clean tracks triangulated under GT,
  then the 4 px relabel over all observed keypoints.
  Indexing quirk (cost a debugging round): gt_bundle camera blocks are
  global (one per list.txt image) but its point view lists index into the
  valid (focal>0) camera list — Notre_Dame masks this (all cameras valid),
  Alamo doesn't (761/2915). The builder remaps and asserts per scene that
  gt_bundle's own points project onto the referenced coords.txt keys
  (< 4 px median; Alamo 0.67 px / 97.9% < 4 px). View-list (x,y) positions
  are unusable for this check — several scenes store them as all zeros.
  Cameras are further restricted to images present in coords.txt, and
  zero-observation camera rows are dropped from the final npz.
  NOTE: RESfM instead ran COLMAP on the (Flickr) images to get 1DSfM GT; our
  GT is the dataset's own gt_bundle reference — a second provenance difference
  on top of R13.

## Running

```bash
.venv/bin/python code/build_tracks/build_strecha.py      # 4 scenes, minutes
.venv/bin/python code/build_tracks/build_blendedmvs.py   # 9 candidates
.venv/bin/python code/build_tracks/build_1dsfm.py        # 10 scenes, no SIFT
.venv/bin/python code/build_tracks/verify_npz.py         # loads + GT sanity
```

Outputs land in `code/datasets/{strecha,blendedmvs,1dsfm}/<scene>.npz`;
`verify_npz.py` checks each file loads through the `Euclidean.get_raw_data`
schema and reports inlier reprojection stats + outlier fractions vs paper.
