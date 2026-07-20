# Reproducibility record — single-scene ESFM vs U-ESFM (Olsson, calibrated)
Generated: 2026-07-17T18:11:30

## Fairness policy (SPEC ground rules)
- Identical inputs: both methods read the same shared `u-esfm/datasets/Euclidean/<scene>.npz` (symlinked into both repos).
- Post-processing: BOTH methods optimized with `ba.run_ba = false`; pre-BA numbers come from each method's raw cameras. Post-BA numbers (when present) come from ONE shared pycolmap BA applied by `evaluate_single_scene.py --ba` to both methods identically. Neither method's own BA was used. Pre- and post-BA are reported separately.
- Identical budgets: same epochs / eval intervals / lr / schedule per run (recorded in each run's `run.conf` and below).
- Same hardware, sequential: for each (scene, seed) the two methods run back-to-back in the same LSF job on the same GPU (see per-run `run_meta.json` host/gpu fields).
- Seeds: per-run `random_seed` in the conf; raw per-seed numbers kept in `summary.csv`.

## Sweep actually run
- methods: ['esfm', 'esfm_rc', 'uesfm']
- scenes (36): ['Alcatraz Courtyard', 'Alcatraz Water Tower', 'Buddah Tooth Relic Temple Singapore', 'Doge Palace Venice', 'Door Lund', 'Drinking Fountain Somewhere In Zurich', 'East Indiaman Goteborg', 'Ecole Superior De Guerre', 'Eglise du dome', 'Folke Filbyter', 'Fort Channing Gate Singapore', 'Golden Statue Somewhere In Hong Kong', 'Gustav Vasa', 'GustavIIAdolf', 'Jonas Ahlstromer', 'Kings College University Of Toronto', 'Lund University Sphinx', 'Nijo Castle Gate', 'Pantheon Paris', 'Park Gate Clermont Ferrand', 'Plaza De Armas Santiago', 'Porta San Donato Bologna', 'Round Church Cambridge', 'Skansen Kronan Gothenburg', 'Smolny Cathedral St Petersburg', 'Some Cathedral In Barcelona', 'Sri Mariamman Singapore', 'Sri Thendayuthapani Singapore', 'Sri Veeramakaliamman Singapore', 'Statue Of Liberty', 'The Pumpkin', 'Thian Hook Keng Temple Singapore', 'Tsar Nikolai I', 'Urban II', 'Vercingetorix', 'Yueh Hai Ching Temple Singapore']
- seeds: [0, 1, 2]

## Budgets / exact commands
### ESFM (official code)
- epochs: 100000  eval_intervals: 5000
- example command: `/home/projects/bagon/ortalda/MVG/final-project/u-esfm/.venv/bin/python single_scene_optimization.py --conf single_scene_generated/ss_esfm_Jonas_Ahlstromer_seed1.conf --exp_version Jonas_Ahlstromer_seed1` (cwd: `/home/projects/bagon/ortalda/MVG/final-project/esfm-baseline/code`)
- full config: `<run_dir>/run.conf` in every run directory
### ESFM (RESfM code)
- epochs: 100000  eval_intervals: 5000
- example command: `/home/projects/bagon/ortalda/MVG/final-project/u-esfm/.venv/bin/python single_scene_optimization.py --conf confs/single_scene_generated/ss_esfm_rc_Jonas_Ahlstromer_seed1.conf --wandb 0 --stage 1 --architecture_type single_scene_bench --results_aggregation_file /home/projects/bagon/ortalda/MVG/final-project/u-esfm/code/results/single_scene/esfm_rc/Jonas_Ahlstromer/seed1/raw/aggregated_results.xlsx` (cwd: `/home/projects/bagon/ortalda/MVG/final-project/u-esfm/code`)
- full config: `<run_dir>/run.conf` in every run directory
### U-ESFM
- epochs: 100000  eval_intervals: 5000
- example command: `/home/projects/bagon/ortalda/MVG/final-project/u-esfm/.venv/bin/python single_scene_optimization.py --conf confs/single_scene_generated/ss_uesfm_Jonas_Ahlstromer_seed1.conf --wandb 0 --stage 1 --architecture_type single_scene_bench --results_aggregation_file /home/projects/bagon/ortalda/MVG/final-project/u-esfm/code/results/single_scene/uesfm/Jonas_Ahlstromer/seed1/raw/aggregated_results.xlsx` (cwd: `/home/projects/bagon/ortalda/MVG/final-project/u-esfm/code`)
- full config: `<run_dir>/run.conf` in every run directory

## Code versions
### u-esfm (U-ESFM)
- path: `/home/projects/bagon/ortalda/MVG/final-project/u-esfm`
- commit: `da78a1e8cf675db4191cc86fda3bd72067f72338`
- **WARNING: working tree dirty.** Diff:
```diff
diff --git a/code/aggregate_single_scene.py b/code/aggregate_single_scene.py
index 6df67c7..f8ee1f1 100755
--- a/code/aggregate_single_scene.py
+++ b/code/aggregate_single_scene.py
@@ -344,9 +344,22 @@ def write_repro(df, results_root, out_path):
 def main():
     ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
     ap.add_argument('--results-root', default=os.path.join(CODE_DIR, 'results', 'single_scene'))
+    ap.add_argument('--exclude-methods', default='uesfm_abl',
+                    help='comma-separated methods to leave out of the tables. Default '
+                         'excludes uesfm_abl: the layer-norm/residual/dropout flags only '
+                         'exist in the Deep models, so at 1x3 that arm is a bit-exact '
+                         'duplicate of esfm_rc (verified per-run); uesfm vs esfm_rc is '
+                         'already the pure loss ablation.')
     args = ap.parse_args()
 
     df, failures = collect(args.results_root)
+    excluded = [m.strip() for m in args.exclude_methods.split(',') if m.strip()]
+    if excluded and not df.empty:
+        n_before = len(df)
+        df = df[~df['method'].isin(excluded)]
+        if len(df) < n_before:
+            print('NOTE: excluded {} runs from methods {} (see --exclude-methods help)'.format(
+                n_before - len(df), excluded))
     if df.empty:
         sys.exit('No evaluated runs found under {} — run the sweep and '
                  'evaluate_single_scene.py first.'.format(args.results_root))
diff --git a/code/results/single_scene/REPRO.md b/code/results/single_scene/REPRO.md
index df58798..45eaef9 100644
--- a/code/results/single_scene/REPRO.md
+++ b/code/results/single_scene/REPRO.md
@@ -1,5 +1,5 @@
 # Reproducibility record — single-scene ESFM vs U-ESFM (Olsson, calibrated)
-Generated: 2026-07-13T20:41:04
+Generated: 2026-07-17T18:10:13
 
 ## Fairness policy (SPEC ground rules)
 - Identical inputs: both methods read the same shared `u-esfm/datasets/Euclidean/<scene>.npz` (symlinked into both repos).
@@ -9,42 +9,4525 @@ Generated: 2026-07-13T20:41:04
 - Seeds: per-run `random_seed` in the conf; raw per-seed numbers kept in `summary.csv`.
 
 ## Sweep actually run
-- methods: ['esfm', 'uesfm']
-- scenes (35): ['Alcatraz Courtyard', 'Alcatraz Water Tower', 'Buddah Tooth Relic Temple Singapore', 'Doge Palace Venice', 'Door Lund', 'Drinking Fountain Somewhere In Zurich', 'East Indiaman Goteborg', 'Ecole Superior De Guerre', 'Eglise du dome', 'Folke Filbyter', 'Fort Channing Gate Singapore', 'Golden Statue Somewhere In Hong Kong', 'Gustav Vasa', 'GustavIIAdolf', 'Jonas Ahlstromer', 'Kings College University Of Toronto', 'Lund University Sphinx', 'Nijo Castle Gate', 'Pantheon Paris', 'Park Gate Clermont Ferrand', 'Plaza De Armas Santiago', 'Porta San Donato Bologna', 'Round Church Cambridge', 'Skansen Kronan Gothenburg', 'Smolny Cathedral St Petersburg', 'Sri Mariamman Singapore', 'Sri Thendayuthapani Singapore', 'Sri Veeramakaliamman Singapore', 'Statue Of Liberty', 'The Pumpkin', 'Thian Hook Keng Temple Singapore', 'Tsar Nikolai I', 'Urban II', 'Vercingetorix', 'Yueh Hai Ching Temple Singapore']
+- methods: ['esfm', 'esfm_rc', 'uesfm', 'uesfm_abl']
+- scenes (36): ['Alcatraz Courtyard', 'Alcatraz Water Tower', 'Buddah Tooth Relic Temple Singapore', 'Doge Palace Venice', 'Door Lund', 'Drinking Fountain Somewhere In Zurich', 'East Indiaman Goteborg', 'Ecole Superior De Guerre', 'Eglise du dome', 'Folke Filbyter', 'Fort Channing Gate Singapore', 'Golden Statue Somewhere In Hong Kong', 'Gustav Vasa', 'GustavIIAdolf', 'Jonas Ahlstromer', 'Kings College University Of Toronto', 'Lund University Sphinx', 'Nijo Castle Gate', 'Pantheon Paris', 'Park Gate Clermont Ferrand', 'Plaza De Armas Santiago', 'Porta San Donato Bologna', 'Round Church Cambridge', 'Skansen Kronan Gothenburg', 'Smolny Cathedral St Petersburg', 'Some Cathedral In Barcelona', 'Sri Mariamman Singapore', 'Sri Thendayuthapani Singapore', 'Sri Veeramakaliamman Singapore', 'Statue Of Liberty', 'The Pumpkin', 'Thian Hook Keng Temple Singapore', 'Tsar Nikolai I', 'Urban II', 'Vercingetorix', 'Yueh Hai Ching Temple Singapore']
 - seeds: [0, 1, 2]
 
 ## Budgets / exact commands
-### ESFM
+### ESFM (official code)
 - epochs: 100000  eval_intervals: 5000
 - example command: `/home/projects/bagon/ortalda/MVG/final-project/u-esfm/.venv/bin/python single_scene_optimization.py --conf single_scene_generated/ss_esfm_Jonas_Ahlstromer_seed1.conf --exp_version Jonas_Ahlstromer_seed1` (cwd: `/home/projects/bagon/ortalda/MVG/final-project/esfm-baseline/code`)
 - full config: `<run_dir>/run.conf` in every run directory
+### ESFM (RESfM code)
+- epochs: 100000  eval_intervals: 5000
+- example command: `/home/projects/bagon/ortalda/MVG/final-project/u-esfm/.venv/bin/python single_scene_optimization.py --conf confs/single_scene_generated/ss_esfm_rc_Jonas_Ahlstromer_seed1.conf --wandb 0 --stage 1 --architecture_type single_scene_bench --results_aggregation_file /home/projects/bagon/ortalda/MVG/final-project/u-esfm/code/results/single_scene/esfm_rc/Jonas_Ahlstromer/seed1/raw/aggregated_results.xlsx` (cwd: `/home/projects/bagon/ortalda/MVG/final-project/u-esfm/code`)
+- full config: `<run_dir>/run.conf` in every run directory
 ### U-ESFM
 - epochs: 100000  eval_intervals: 5000
 - example command: `/home/projects/bagon/ortalda/MVG/final-project/u-esfm/.venv/bin/python single_scene_optimization.py --conf confs/single_scene_generated/ss_uesfm_Jonas_Ahlstromer_seed1.conf --wandb 0 --stage 1 --architecture_type single_scene_bench --results_aggregation_file /home/projects/bagon/ortalda/MVG/final-project/u-esfm/code/results/single_scene/uesfm/Jonas_Ahlstromer/seed1/raw/aggregated_results.xlsx` (cwd: `/home/projects/bagon/ortalda/MVG/final-project/u-esfm/code`)
 - full config: `<run_dir>/run.conf` in every run directory
+### U-ESFM arch + ESFMLoss
+- epochs: 100000  eval_intervals: 5000
+- example command: `/home/projects/bagon/ortalda/MVG/final-project/u-esfm/.venv/bin/python single_scene_optimization.py --conf confs/single_scene_generated/ss_uesfm_abl_Jonas_Ahlstromer_seed1.conf --wandb 0 --stage 1 --architecture_type single_scene_bench --results_aggregation_file /home/projects/bagon/ortalda/MVG/final-project/u-esfm/code/results/single_scene/uesfm_abl/Jonas_Ahlstromer/seed1/raw/aggregated_results.xlsx` (cwd: `/home/projects/bagon/ortalda/MVG/final-project/u-esfm/code`)
+- full config: `<run_dir>/run.conf` in every run directory
 
 ## Code versions
 ### u-esfm (U-ESFM)
 - path: `/home/projects/bagon/ortalda/MVG/final-project/u-esfm`
-- commit: `3f871108d3e843b4698d6af48adb12f7ad835f6d`
+- commit: `da78a1e8cf675db4191cc86fda3bd72067f72338`
 - **WARNING: working tree dirty.** Diff:
 ```diff
-(untracked files only)
+diff --git a/code/results/single_scene/REPRO.md b/code/results/single_scene/REPRO.md
+index df58798..486ce1c 100644
+--- a/code/results/single_scene/REPRO.md
++++ b/code/results/single_scene/REPRO.md
+@@ -1,5 +1,5 @@
+ # Reproducibility record — single-scene ESFM vs U-ESFM (Olsson, calibrated)
+-Generated: 2026-07-13T20:41:04
++Generated: 2026-07-15T19:09:47
+ 
+ ## Fairness policy (SPEC ground rules)
+ - Identical inputs: both methods read the same shared `u-esfm/datasets/Euclidean/<scene>.npz` (symlinked into both repos).
+@@ -10,11 +10,11 @@ Generated: 2026-07-13T20:41:04
+ 
+ ## Sweep actually run
+ - methods: ['esfm', 'uesfm']
+-- scenes (35): ['Alcatraz Courtyard', 'Alcatraz Water Tower', 'Buddah Tooth Relic Temple Singapore', 'Doge Palace Venice', 'Door Lund', 'Drinking Fountain Somewhere In Zurich', 'East Indiaman Goteborg', 'Ecole Superior De Guerre', 'Eglise du dome', 'Folke Filbyter', 'Fort Channing Gate Singapore', 'Golden Statue Somewhere In Hong Kong', 'Gustav Vasa', 'GustavIIAdolf', 'Jonas Ahlstromer', 'Kings College University Of Toronto', 'Lund University Sphinx', 'Nijo Castle Gate', 'Pantheon Paris', 'Park Gate Clermont Ferrand', 'Plaza De Armas Santiago', 'Porta San Donato Bologna', 'Round Church Cambridge', 'Skansen Kronan Gothenburg', 'Smolny Cathedral St Petersburg', 'Sri Mariamman Singapore', 'Sri Thendayuthapani Singapore', 'Sri Veeramakaliamman Singapore', 'Statue Of Liberty', 'The Pumpkin', 'Thian Hook Keng Temple Singapore', 'Tsar Nikolai I', 'Urban II', 'Vercingetorix', 'Yueh Hai Ching Temple Singapore']
++- scenes (36): ['Alcatraz Courtyard', 'Alcatraz Water Tower', 'Buddah Tooth Relic Temple Singapore', 'Doge Palace Venice', 'Door Lund', 'Drinking Fountain Somewhere In Zurich', 'East Indiaman Goteborg', 'Ecole Superior De Guerre', 'Eglise du dome', 'Folke Filbyter', 'Fort Channing Gate Singapore', 'Golden Statue Somewhere In Hong Kong', 'Gustav Vasa', 'GustavIIAdolf', 'Jonas Ahlstromer', 'Kings College University Of Toronto', 'Lund University Sphinx', 'Nijo Castle Gate', 'Pantheon Paris', 'Park Gate Clermont Ferrand', 'Plaza De Armas Santiago', 'Porta San Donato Bologna', 'Round Church Cambridge', 'Skansen Kronan Gothenburg', 'Smolny Cathedral St Petersburg', 'Some Cathedral In Barcelona', 'Sri Mariamman Singapore', 'Sri Thendayuthapani Singapore', 'Sri Veeramakaliamman Singapore', 'Statue Of Liberty', 'The Pumpkin', 'Thian Hook Keng Temple Singapore', 'Tsar Nikolai I', 'Urban II', 'Vercingetorix', 'Yueh Hai Ching Temple Singapore']
+ - seeds: [0, 1, 2]
+ 
+ ## Budgets / exact commands
+-### ESFM
++### ESFM (official code)
+ - epochs: 100000  eval_intervals: 5000
+ - example command: `/home/projects/bagon/ortalda/MVG/final-project/u-esfm/.venv/bin/python single_scene_optimization.py --conf single_scene_generated/ss_esfm_Jonas_Ahlstromer_seed1.conf --exp_version Jonas_Ahlstromer_seed1` (cwd: `/home/projects/bagon/ortalda/MVG/final-project/esfm-baseline/code`)
+ - full config: `<run_dir>/run.conf` in every run directory
+@@ -26,25 +26,3603 @@ Generated: 2026-07-13T20:41:04
+ ## Code versions
+ ### u-esfm (U-ESFM)
+ - path: `/home/projects/bagon/ortalda/MVG/final-project/u-esfm`
+-- commit: `3f871108d3e843b4698d6af48adb12f7ad835f6d`
++- commit: `e08bf89a16b2fb210a96fe230628dd815c95dcab`
+ - **WARNING: working tree dirty.** Diff:
+ ```diff
+-(untracked files only)
++diff --git a/code/results/single_scene/REPRO.md b/code/results/single_scene/REPRO.md
++index df58798..9e660bd 100644
++--- a/code/results/single_scene/REPRO.md
+++++ b/code/results/single_scene/REPRO.md
++@@ -1,5 +1,5 @@
++ # Reproducibility record — single-scene ESFM vs U-ESFM (Olsson, calibrated)
++-Generated: 2026-07-13T20:41:04
+++Generated: 2026-07-14T21:27:13
++ 
++ ## Fairness policy (SPEC ground rules)
++ - Identical inputs: both methods read the same shared `u-esfm/datasets/Euclidean/<scene>.npz` (symlinked into both repos).
++@@ -10,11 +10,11 @@ Generated: 2026-07-13T20:41:04
++ 
++ ## Sweep actually run
++ - methods: ['esfm', 'uesfm']
++-- scenes (35): ['Alcatraz Courtyard', 'Alcatraz Water Tower', 'Buddah Tooth Relic Temple Singapore', 'Doge Palace Venice', 'Door Lund', 'Drinking Fountain Somewhere In Zurich', 'East Indiaman Goteborg', 'Ecole Superior De Guerre', 'Eglise du dome', 'Folke Filbyter', 'Fort Channing Gate Singapore', 'Golden Statue Somewhere In Hong Kong', 'Gustav Vasa', 'GustavIIAdolf', 'Jonas Ahlstromer', 'Kings College University Of Toronto', 'Lund University Sphinx', 'Nijo Castle Gate', 'Pantheon Paris', 'Park Gate Clermont Ferrand', 'Plaza De Armas Santiago', 'Porta San Donato Bologna', 'Round Church Cambridge', 'Skansen Kronan Gothenburg', 'Smolny Cathedral St Petersburg', 'Sri Mariamman Singapore', 'Sri Thendayuthapani Singapore', 'Sri Veeramakaliamman Singapore', 'Statue Of Liberty', 'The Pumpkin', 'Thian Hook Keng Temple Singapore', 'Tsar Nikolai I', 'Urban II', 'Vercingetorix', 'Yueh Hai Ching Temple Singapore']
+++- scenes (36): ['Alcatraz Courtyard', 'Alcatraz Water Tower', 'Buddah Tooth Relic Temple Singapore', 'Doge Palace Venice', 'Door Lund', 'Drinking Fountain Somewhere In Zurich', 'East Indiaman Goteborg', 'Ecole Superior De Guerre', 'Eglise du dome', 'Folke Filbyter', 'Fort Channing Gate Singapore', 'Golden Statue Somewhere In Hong Kong', 'Gustav Vasa', 'GustavIIAdolf', 'Jonas Ahlstromer', 'Kings College University Of Toronto', 'Lund University Sphinx', 'Nijo Castle Gate', 'Pantheon Paris', 'Park Gate Clermont Ferrand', 'Plaza De Armas Santiago', 'Porta San Donato Bologna', 'Round Church Cambridge', 'Skansen Kronan Gothenburg', 'Smolny Cathedral St Petersburg', 'Some Cathedral In Barcelona', 'Sri Mariamman Singapore', 'Sri Thendayuthapani Singapore', 'Sri Veeramakaliamman Singapore', 'Statue Of Liberty', 'The Pumpkin', 'Thian Hook Keng Temple Singapore', 'Tsar Nikolai I', 'Urban II', 'Vercingetorix', 'Yueh Hai Ching Temple Singapore']
++ - seeds: [0, 1, 2]
++ 
++ ## Budgets / exact commands
++-### ESFM
+++### ESFM (official code)
++ - epochs: 100000  eval_intervals: 5000
++ - example command: `/home/projects/bagon/ortalda/MVG/final-project/u-esfm/.venv/bin/python single_scene_optimization.py --conf single_scene_generated/ss_esfm_Jonas_Ahlstromer_seed1.conf --exp_version Jonas_Ahlstromer_seed1` (cwd: `/home/projects/bagon/ortalda/MVG/final-project/esfm-baseline/code`)
++ - full config: `<run_dir>/run.conf` in every run directory
++@@ -26,25 +26,3029 @@ Generated: 2026-07-13T20:41:04
++ ## Code versions
++ ### u-esfm (U-ESFM)
++ - path: `/home/projects/bagon/ortalda/MVG/final-project/u-esfm`
++-- commit: `3f871108d3e843b4698d6af48adb12f7ad835f6d`
+++- commit: `e08bf89a16b2fb210a96fe230628dd815c95dcab`
++ - **WARNING: working tree dirty.** Diff:
++ ```diff
++-(untracked files only)
+++diff --git a/code/results/single_scene/REPRO.md b/code/results/single_scene/REPRO.md
+++index df58798..464f2a3 100644
+++--- a/code/results/single_scene/REPRO.md
++++++ b/code/results/single_scene/REPRO.md
+++@@ -1,5 +1,5 @@
+++ # Reproducibility record — single-scene ESFM vs U-ESFM (Olsson, calibrated)
+++-Generated: 2026-07-13T20:41:04
++++Generated: 2026-07-14T20:48:10
+++ 
+++ ## Fairness policy (SPEC ground rules)
+++ - Identical inputs: both methods read the same shared `u-esfm/datasets/Euclidean/<scene>.npz` (symlinked into both repos).
+++@@ -10,11 +10,11 @@ Generated: 2026-07-13T20:41:04
+++ 
+++ ## Sweep actually run
+++ - methods: ['esfm', 'uesfm']
+++-- scenes (35): ['Alcatraz Courtyard', 'Alcatraz Water Tower', 'Buddah Tooth Relic Temple Singapore', 'Doge Palace Venice', 'Door Lund', 'Drinking Fountain Somewhere In Zurich', 'East Indiaman Goteborg', 'Ecole Superior De Guerre', 'Eglise du dome', 'Folke Filbyter', 'Fort Channing Gate Singapore', 'Golden Statue Somewhere In Hong Kong', 'Gustav Vasa', 'GustavIIAdolf', 'Jonas Ahlstromer', 'Kings College University Of Toronto', 'Lund University Sphinx', 'Nijo Castle Gate', 'Pantheon Paris', 'Park Gate Clermont Ferrand', 'Plaza De Armas Santiago', 'Porta San Donato Bologna', 'Round Church Cambridge', 'Skansen Kronan Gothenburg', 'Smolny Cathedral St Petersburg', 'Sri Mariamman Singapore', 'Sri Thendayuthapani Singapore', 'Sri Veeramakaliamman Singapore', 'Statue Of Liberty', 'The Pumpkin', 'Thian Hook Keng Temple Singapore', 'Tsar Nikolai I', 'Urban II', 'Vercingetorix', 'Yueh Hai Ching Temple Singapore']
++++- scenes (36): ['Alcatraz Courtyard', 'Alcatraz Water Tower', 'Buddah Tooth Relic Temple Singapore', 'Doge Palace Venice', 'Door Lund', 'Drinking Fountain Somewhere In Zurich', 'East Indiaman Goteborg', 'Ecole Superior De Guerre', 'Eglise du dome', 'Folke Filbyter', 'Fort Channing Gate Singapore', 'Golden Statue Somewhere In Hong Kong', 'Gustav Vasa', 'GustavIIAdolf', 'Jonas Ahlstromer', 'Kings College University Of Toronto', 'Lund University Sphinx', 'Nijo Castle Gate', 'Pantheon Paris', 'Park Gate Clermont Ferrand', 'Plaza De Armas Santiago', 'Porta San Donato Bologna', 'Round Church Cambridge', 'Skansen Kronan Gothenburg', 'Smolny Cathedral St Petersburg', 'Some Cathedral In Barcelona', 'Sri Mariamman Singapore', 'Sri Thendayuthapani Singapore', 'Sri Veeramakaliamman Singapore', 'Statue Of Liberty', 'The Pumpkin', 'Thian Hook Keng Temple Singapore', 'Tsar Nikolai I', 'Urban II', 'Vercingetorix', 'Yueh Hai Ching Temple Singapore']
+++ - seeds: [0, 1, 2]
+++ 
+++ ## Budgets / exact commands
+++-### ESFM
++++### ESFM (official code)
+++ - epochs: 100000  eval_intervals: 5000
+++ - example command: `/home/projects/bagon/ortalda/MVG/final-project/u-esfm/.venv/bin/python single_scene_optimization.py --conf single_scene_generated/ss_esfm_Jonas_Ahlstromer_seed1.conf --exp_version Jonas_Ahlstromer_seed1` (cwd: `/home/projects/bagon/ortalda/MVG/final-project/esfm-baseline/code`)
+++ - full config: `<run_dir>/run.conf` in every run directory
+++@@ -26,25 +26,2499 @@ Generated: 2026-07-13T20:41:04
+++ ## Code versions
+++ ### u-esfm (U-ESFM)
+++ - path: `/home/projects/bagon/ortalda/MVG/final-project/u-esfm`
+++-- commit: `3f871108d3e843b4698d6af48adb12f7ad835f6d`
++++- commit: `e08bf89a16b2fb210a96fe230628dd815c95dcab`
+++ - **WARNING: working tree dirty.** Diff:
+++ ```diff
+++-(untracked files only)
++++diff --git a/code/results/single_scene/REPRO.md b/code/results/single_scene/REPRO.md
++++index df58798..4b9954c 100644
++++--- a/code/results/single_scene/REPRO.md
+++++++ b/code/results/single_scene/REPRO.md
++++@@ -1,5 +1,5 @@
++++ # Reproducibility record — single-scene ESFM vs U-ESFM (Olsson, calibrated)
++++-Generated: 2026-07-13T20:41:04
+++++Generated: 2026-07-14T19:57:53
++++ 
++++ ## Fairness policy (SPEC ground rules)
++++ - Identical inputs: both methods read the same shared `u-esfm/datasets/Euclidean/<scene>.npz` (symlinked into both repos).
++++@@ -10,11 +10,11 @@ Generated: 2026-07-13T20:41:04
++++ 
++++ ## Sweep actually run
++++ - methods: ['esfm', 'uesfm']
++++-- scenes (35): ['Alcatraz Courtyard', 'Alcatraz Water Tower', 'Buddah Tooth Relic Temple Singapore', 'Doge Palace Venice', 'Door Lund', 'Drinking Fountain Somewhere In Zurich', 'East Indiaman Goteborg', 'Ecole Superior De Guerre', 'Eglise du dome', 'Folke Filbyter', 'Fort Channing Gate Singapore', 'Golden Statue Somewhere In Hong Kong', 'Gustav Vasa', 'GustavIIAdolf', 'Jonas Ahlstromer', 'Kings College University Of Toronto', 'Lund University Sphinx', 'Nijo Castle Gate', 'Pantheon Paris', 'Park Gate Clermont Ferrand', 'Plaza De Armas Santiago', 'Porta San Donato Bologna', 'Round Church Cambridge', 'Skansen Kronan Gothenburg', 'Smolny Cathedral St Petersburg', 'Sri Mariamman Singapore', 'Sri Thendayuthapani Singapore', 'Sri Veeramakaliamman Singapore', 'Statue Of Liberty', 'The Pumpkin', 'Thian Hook Keng Temple Singapore', 'Tsar Nikolai I', 'Urban II', 'Vercingetorix', 'Yueh Hai Ching Temple Singapore']
+++++- scenes (36): ['Alcatraz Courtyard', 'Alcatraz Water Tower', 'Buddah Tooth Relic Temple Singapore', 'Doge Palace Venice', 'Door Lund', 'Drinking Fountain Somewhere In Zurich', 'East Indiaman Goteborg', 'Ecole Superior De Guerre', 'Eglise du dome', 'Folke Filbyter', 'Fort Channing Gate Singapore', 'Golden Statue Somewhere In Hong Kong', 'Gustav Vasa', 'GustavIIAdolf', 'Jonas Ahlstromer', 'Kings College University Of Toronto', 'Lund University Sphinx', 'Nijo Castle Gate', 'Pantheon Paris', 'Park Gate Clermont Ferrand', 'Plaza De Armas Santiago', 'Porta San Donato Bologna', 'Round Church Cambridge', 'Skansen Kronan Gothenburg', 'Smolny Cathedral St Petersburg', 'Some Cathedral In Barcelona', 'Sri Mariamman Singapore', 'Sri Thendayuthapani Singapore', 'Sri Veeramakaliamman Singapore', 'Statue Of Liberty', 'The Pumpkin', 'Thian Hook Keng Temple Singapore', 'Tsar Nikolai I', 'Urban II', 'Vercingetorix', 'Yueh Hai Ching Temple Singapore']
++++ - seeds: [0, 1, 2]
++++ 
++++ ## Budgets / exact commands
++++-### ESFM
+++++### ESFM (official code)
++++ - epochs: 100000  eval_intervals: 5000
++++ - example command: `/home/projects/bagon/ortalda/MVG/final-project/u-esfm/.venv/bin/python single_scene_optimization.py --conf single_scene_generated/ss_esfm_Jonas_Ahlstromer_seed1.conf --exp_version Jonas_Ahlstromer_seed1` (cwd: `/home/projects/bagon/ortalda/MVG/final-project/esfm-baseline/code`)
++++ - full config: `<run_dir>/run.conf` in every run directory
++++@@ -26,25 +26,1974 @@ Generated: 2026-07-13T20:41:04
++++ ## Code versions
++++ ### u-esfm (U-ESFM)
++++ - path: `/home/projects/bagon/ortalda/MVG/final-project/u-esfm`
++++-- commit: `3f871108d3e843b4698d6af48adb12f7ad835f6d`
+++++- commit: `e08bf89a16b2fb210a96fe230628dd815c95dcab`
++++ - **WARNING: working tree dirty.** Diff:
++++ ```diff
++++-(untracked files only)
+++++diff --git a/code/results/single_scene/REPRO.md b/code/results/single_scene/REPRO.md
+++++index df58798..36a1576 100644
+++++--- a/code/results/single_scene/REPRO.md
++++++++ b/code/results/single_scene/REPRO.md
+++++@@ -1,5 +1,5 @@
+++++ # Reproducibility record — single-scene ESFM vs U-ESFM (Olsson, calibrated)
+++++-Generated: 2026-07-13T20:41:04
++++++Generated: 2026-07-14T19:52:15
+++++ 
+++++ ## Fairness policy (SPEC ground rules)
+++++ - Identical inputs: both methods read the same shared `u-esfm/datasets/Euclidean/<scene>.npz` (symlinked into both repos).
+++++@@ -10,11 +10,11 @@ Generated: 2026-07-13T20:41:04
+++++ 
+++++ ## Sweep actually run
+++++ - methods: ['esfm', 'uesfm']
+++++-- scenes (35): ['Alcatraz Courtyard', 'Alcatraz Water Tower', 'Buddah Tooth Relic Temple Singapore', 'Doge Palace Venice', 'Door Lund', 'Drinking Fountain Somewhere In Zurich', 'East Indiaman Goteborg', 'Ecole Superior De Guerre', 'Eglise du dome', 'Folke Filbyter', 'Fort Channing Gate Singapore', 'Golden Statue Somewhere In Hong Kong', 'Gustav Vasa', 'GustavIIAdolf', 'Jonas Ahlstromer', 'Kings College University Of Toronto', 'Lund University Sphinx', 'Nijo Castle Gate', 'Pantheon Paris', 'Park Gate Clermont Ferrand', 'Plaza De Armas Santiago', 'Porta San Donato Bologna', 'Round Church Cambridge', 'Skansen Kronan Gothenburg', 'Smolny Cathedral St Petersburg', 'Sri Mariamman Singapore', 'Sri Thendayuthapani Singapore', 'Sri Veeramakaliamman Singapore', 'Statue Of Liberty', 'The Pumpkin', 'Thian Hook Keng Temple Singapore', 'Tsar Nikolai I', 'Urban II', 'Vercingetorix', 'Yueh Hai Ching Temple Singapore']
++++++- scenes (36): ['Alcatraz Courtyard', 'Alcatraz Water Tower', 'Buddah Tooth Relic Temple Singapore', 'Doge Palace Venice', 'Door Lund', 'Drinking Fountain Somewhere In Zurich', 'East Indiaman Goteborg', 'Ecole Superior De Guerre', 'Eglise du dome', 'Folke Filbyter', 'Fort Channing Gate Singapore', 'Golden Statue Somewhere In Hong Kong', 'Gustav Vasa', 'GustavIIAdolf', 'Jonas Ahlstromer', 'Kings College University Of Toronto', 'Lund University Sphinx', 'Nijo Castle Gate', 'Pantheon Paris', 'Park Gate Clermont Ferrand', 'Plaza De Armas Santiago', 'Porta San Donato Bologna', 'Round Church Cambridge', 'Skansen Kronan Gothenburg', 'Smolny Cathedral St Petersburg', 'Some Cathedral In Barcelona', 'Sri Mariamman Singapore', 'Sri Thendayuthapani Singapore', 'Sri Veeramakaliamman Singapore', 'Statue Of Liberty', 'The Pumpkin', 'Thian Hook Keng Temple Singapore', 'Tsar Nikolai I', 'Urban II', 'Vercingetorix', 'Yueh Hai Ching Temple Singapore']
+++++ - seeds: [0, 1, 2]
+++++ 
+++++ ## Budgets / exact commands
+++++-### ESFM
++++++### ESFM (official code)
+++++ - epochs: 100000  eval_intervals: 5000
+++++ - example command: `/home/projects/bagon/ortalda/MVG/final-project/u-esfm/.venv/bin/python single_scene_optimization.py --conf single_scene_generated/ss_esfm_Jonas_Ahlstromer_seed1.conf --exp_version Jonas_Ahlstromer_seed1` (cwd: `/home/projects/bagon/ortalda/MVG/final-project/esfm-baseline/code`)
+++++ - full config: `<run_dir>/run.conf` in every run directory
+++++@@ -26,25 +26,1455 @@ Generated: 2026-07-13T20:41:04
+++++ ## Code versions
+++++ ### u-esfm (U-ESFM)
+++++ - path: `/home/projects/bagon/ortalda/MVG/final-project/u-esfm`
+++++-- commit: `3f871108d3e843b4698d6af48adb12f7ad835f6d`
++++++- commit: `e08bf89a16b2fb210a96fe230628dd815c95dcab`
+++++ - **WARNING: working tree dirty.** Diff:
+++++ ```diff
+++++-(untracked files only)
++++++diff --git a/code/results/single_scene/REPRO.md b/code/results/single_scene/REPRO.md
++++++index df58798..0d34158 100644
++++++--- a/code/results/single_scene/REPRO.md
+++++++++ b/code/results/single_scene/REPRO.md
++++++@@ -1,5 +1,5 @@
++++++ # Reproducibility record — single-scene ESFM vs U-ESFM (Olsson, calibrated)
++++++-Generated: 2026-07-13T20:41:04
+++++++Generated: 2026-07-14T18:43:27
++++++ 
++++++ ## Fairness policy (SPEC ground rules)
++++++ - Identical inputs: both methods read the same shared `u-esfm/datasets/Euclidean/<scene>.npz` (symlinked into both repos).
++++++@@ -10,11 +10,11 @@ Generated: 2026-07-13T20:41:04
++++++ 
++++++ ## Sweep actually run
++++++ - methods: ['esfm', 'uesfm']
++++++-- scenes (35): ['Alcatraz Courtyard', 'Alcatraz Water Tower', 'Buddah Tooth Relic Temple Singapore', 'Doge Palace Venice', 'Door Lund', 'Drinking Fountain Somewhere In Zurich', 'East Indiaman Goteborg', 'Ecole Superior De Guerre', 'Eglise du dome', 'Folke Filbyter', 'Fort Channing Gate Singapore', 'Golden Statue Somewhere In Hong Kong', 'Gustav Vasa', 'GustavIIAdolf', 'Jonas Ahlstromer', 'Kings College University Of Toronto', 'Lund University Sphinx', 'Nijo Castle Gate', 'Pantheon Paris', 'Park Gate Clermont Ferrand', 'Plaza De Armas Santiago', 'Porta San Donato Bologna', 'Round Church Cambridge', 'Skansen Kronan Gothenburg', 'Smolny Cathedral St Petersburg', 'Sri Mariamman Singapore', 'Sri Thendayuthapani Singapore', 'Sri Veeramakaliamman Singapore', 'Statue Of Liberty', 'The Pumpkin', 'Thian Hook Keng Temple Singapore', 'Tsar Nikolai I', 'Urban II', 'Vercingetorix', 'Yueh Hai Ching Temple Singapore']
+++++++- scenes (36): ['Alcatraz Courtyard', 'Alcatraz Water Tower', 'Buddah Tooth Relic Temple Singapore', 'Doge Palace Venice', 'Door Lund', 'Drinking Fountain Somewhere In Zurich', 'East Indiaman Goteborg', 'Ecole Superior De Guerre', 'Eglise du dome', 'Folke Filbyter', 'Fort Channing Gate Singapore', 'Golden Statue Somewhere In Hong Kong', 'Gustav Vasa', 'GustavIIAdolf', 'Jonas Ahlstromer', 'Kings College University Of Toronto', 'Lund University Sphinx', 'Nijo Castle Gate', 'Pantheon Paris', 'Park Gate Clermont Ferrand', 'Plaza De Armas Santiago', 'Porta San Donato Bologna', 'Round Church Cambridge', 'Skansen Kronan Gothenburg', 'Smolny Cathedral St Petersburg', 'Some Cathedral In Barcelona', 'Sri Mariamman Singapore', 'Sri Thendayuthapani Singapore', 'Sri Veeramakaliamman Singapore', 'Statue Of Liberty', 'The Pumpkin', 'Thian Hook Keng Temple Singapore', 'Tsar Nikolai I', 'Urban II', 'Vercingetorix', 'Yueh Hai Ching Temple Singapore']
++++++ - seeds: [0, 1, 2]
++++++ 
++++++ ## Budgets / exact commands
++++++-### ESFM
+++++++### ESFM (official code)
++++++ - epochs: 100000  eval_intervals: 5000
++++++ - example command: `/home/projects/bagon/ortalda/MVG/final-project/u-esfm/.venv/bin/python single_scene_optimization.py --conf single_scene_generated/ss_esfm_Jonas_Ahlstromer_seed1.conf --exp_version Jonas_Ahlstromer_seed1` (cwd: `/home/projects/bagon/ortalda/MVG/final-project/esfm-baseline/code`)
++++++ - full config: `<run_dir>/run.conf` in every run directory
++++++@@ -26,25 +26,936 @@ Generated: 2026-07-13T20:41:04
++++++ ## Code versions
++++++ ### u-esfm (U-ESFM)
++++++ - path: `/home/projects/bagon/ortalda/MVG/final-project/u-esfm`
++++++-- commit: `3f871108d3e843b4698d6af48adb12f7ad835f6d`
+++++++- commit: `e08bf89a16b2fb210a96fe230628dd815c95dcab`
++++++ - **WARNING: working tree dirty.** Diff:
++++++ ```diff
++++++-(untracked files only)
+++++++diff --git a/code/results/single_scene/REPRO.md b/code/results/single_scene/REPRO.md
+++++++index df58798..37520f8 100644
+++++++--- a/code/results/single_scene/REPRO.md
++++++++++ b/code/results/single_scene/REPRO.md
+++++++@@ -1,5 +1,5 @@
+++++++ # Reproducibility record — single-scene ESFM vs U-ESFM (Olsson, calibrated)
+++++++-Generated: 2026-07-13T20:41:04
++++++++Generated: 2026-07-14T17:11:15
+++++++ 
+++++++ ## Fairness policy (SPEC ground rules)
+++++++ - Identical inputs: both methods read the same shared `u-esfm/datasets/Euclidean/<scene>.npz` (symlinked into both repos).
+++++++@@ -10,41 +10,447 @@ Generated: 2026-07-13T20:41:04
+++++++ 
+++++++ ## Sweep actually run
+++++++ - methods: ['esfm', 'uesfm']
+++++++-- scenes (35): ['Alcatraz Courtyard', 'Alcatraz Water Tower', 'Buddah Tooth Relic Temple Singapore', 'Doge Palace Venice', 'Door Lund', 'Drinking Fountain Somewhere In Zurich', 'East Indiaman Goteborg', 'Ecole Superior De Guerre', 'Eglise du dome', 'Folke Filbyter', 'Fort Channing Gate Singapore', 'Golden Statue Somewhere In Hong Kong', 'Gustav Vasa', 'GustavIIAdolf', 'Jonas Ahlstromer', 'Kings College University Of Toronto', 'Lund University Sphinx', 'Nijo Castle Gate', 'Pantheon Paris', 'Park Gate Clermont Ferrand', 'Plaza De Armas Santiago', 'Porta San Donato Bologna', 'Round Church Cambridge', 'Skansen Kronan Gothenburg', 'Smolny Cathedral St Petersburg', 'Sri Mariamman Singapore', 'Sri Thendayuthapani Singapore', 'Sri Veeramakaliamman Singapore', 'Statue Of Liberty', 'The Pumpkin', 'Thian Hook Keng Temple Singapore', 'Tsar Nikolai I', 'Urban II', 'Vercingetorix', 'Yueh Hai Ching Temple Singapore']
++++++++- scenes (36): ['Alcatraz Courtyard', 'Alcatraz Water Tower', 'Buddah Tooth Relic Temple Singapore', 'Doge Palace Venice', 'Door Lund', 'Drinking Fountain Somewhere In Zurich', 'East Indiaman Goteborg', 'Ecole Superior De Guerre', 'Eglise du dome', 'Folke Filbyter', 'Fort Channing Gate Singapore', 'Golden Statue Somewhere In Hong Kong', 'Gustav Vasa', 'GustavIIAdolf', 'Jonas Ahlstromer', 'Kings College University Of Toronto', 'Lund University Sphinx', 'Nijo Castle Gate', 'Pantheon Paris', 'Park Gate Clermont Ferrand', 'Plaza De Armas Santiago', 'Porta San Donato Bologna', 'Round Church Cambridge', 'Skansen Kronan Gothenburg', 'Smolny Cathedral St Petersburg', 'Some Cathedral In Barcelona', 'Sri Mariamman Singapore', 'Sri Thendayuthapani Singapore', 'Sri Veeramakaliamman Singapore', 'Statue Of Liberty', 'The Pumpkin', 'Thian Hook Keng Temple Singapore', 'Tsar Nikolai I', 'Urban II', 'Vercingetorix', 'Yueh Hai Ching Temple Singapore']
+++++++ - seeds: [0, 1, 2]
+++++++ 
+++++++ ## Budgets / exact commands
+++++++-### ESFM
++++++++### ESFM (official code)
+++++++ - epochs: 100000  eval_intervals: 5000
+++++++ - example command: `/home/projects/bagon/ortalda/MVG/final-project/u-esfm/.venv/bin/python single_scene_optimization.py --conf single_scene_generated/ss_esfm_Jonas_Ahlstromer_seed1.conf --exp_version Jonas_Ahlstromer_seed1` (cwd: `/home/projects/bagon/ortalda/MVG/final-project/esfm-baseline/code`)
+++++++ - full config: `<run_dir>/run.conf` in every run directory
+++++++ ### U-ESFM
+++++++ - epochs: 100000  eval_intervals: 5000
+++++++-- example command: `/home/projects/bagon/ortalda/MVG/final-project/u-esfm/.venv/bin/python single_scene_optimization.py --conf confs/single_scene_generated/ss_uesfm_Jonas_Ahlstromer_seed1.conf --wandb 0 --stage 1 --architecture_type single_scene_bench --results_aggregation_file /home/projects/bagon/ortalda/MVG/final-project/u-esfm/code/results/single_scene/uesfm/Jonas_Ahlstromer/seed1/raw/aggregated_results.xlsx` (cwd: `/home/projects/bagon/ortalda/MVG/final-project/u-esfm/code`)
++++++++- example command: `/home/projects/bagon/ortalda/MVG/final-project/u-esfm/.venv/bin/python single_scene_optimization.py --conf confs/single_scene_generated/ss_uesfm_Buddah_Tooth_Relic_Temple_Singapore_seed1.conf --wandb 0 --stage 1 --architecture_type single_scene_bench --results_aggregation_file /home/projects/bagon/ortalda/MVG/final-project/u-esfm/code/results/single_scene/uesfm/Buddah_Tooth_Relic_Temple_Singapore/seed1/raw/aggregated_results.xlsx` (cwd: `/home/projects/bagon/ortalda/MVG/final-project/u-esfm/code`)
+++++++ - full config: `<run_dir>/run.conf` in every run directory
+++++++ 
+++++++ ## Code versions
+++++++ ### u-esfm (U-ESFM)
+++++++ - path: `/home/projects/bagon/ortalda/MVG/final-project/u-esfm`
+++++++-- commit: `3f871108d3e843b4698d6af48adb12f7ad835f6d`
++++++++- commit: `4867a66e593c5263d48eabd4a4a4e85a1219792e`
+++++++ - **WARNING: working tree dirty.** Diff:
+++++++ ```diff
+++++++-(untracked files only)
++++++++diff --git a/code/evaluate_single_scene.py b/code/evaluate_single_scene.py
++++++++index fae3780..96b8fe9 100755
++++++++--- a/code/evaluate_single_scene.py
+++++++++++ b/code/evaluate_single_scene.py
++++++++@@ -191,7 +191,16 @@ def main():
++++++++ 
++++++++         rec = {'method': method, 'scene': scene, 'seed': seed,
++++++++                'wall_clock_s': meta.get('wall_clock_s')}
++++++++-        rec.update(harmonized_from_cameras(cams, gt))
+++++++++        # Fail-soft: a run with degenerate final cameras (e.g. NaN rotations from a
+++++++++        # diverged optimization) must not kill the evaluation of every other run.
+++++++++        # Such runs get no harmonized_metrics.json and are reported loudly instead.
+++++++++        try:
+++++++++            rec.update(harmonized_from_cameras(cams, gt))
+++++++++        except Exception as e:
+++++++++            finite = bool(np.isfinite(cams['Rs']).all())
+++++++++            flags.append('{} seed{} {}: HARMONIZED EVAL FAILED ({}); '
+++++++++                         'finite rotations: {}'.format(method, seed, scene, e, finite))
+++++++++            continue
++++++++ 
++++++++         nat = native_metrics(run_dir, method, scene)
++++++++         rec['native'] = nat
++++++++diff --git a/code/results/single_scene/summary.csv b/code/results/single_scene/summary.csv
++++++++index e4f9f5e..29a658f 100644
++++++++--- a/code/results/single_scene/summary.csv
+++++++++++ b/code/results/single_scene/summary.csv
++++++++@@ -2,171 +2,116 @@ method,scene,seed,rot_err_deg,pos_err,reproj_err_px,reproj_med_px,n_cams,runtime
++++++++ esfm,Alcatraz Courtyard,0,0.324288380204305,0.0854554642858637,3.379232694353053,1.9204072326560708,133,4821.4,4792.707528591156,99999.0
++++++++ esfm,Alcatraz Courtyard,1,0.36138678157391557,0.09112912757839253,3.3691035794153503,1.8536696768772378,133,4825.72,4804.594581127167,99999.0
++++++++ esfm,Alcatraz Courtyard,2,0.4015043649148433,0.10235190268069758,3.557242072642648,1.9198007399315369,133,4833.02,4812.242679834366,99999.0
++++++++-uesfm,Alcatraz Courtyard,0,0.6370266175309793,0.1671943803031883,5.956592777085197,3.6638282340102246,133,6478.06,6448.735792636871,99999.0
++++++++-uesfm,Alcatraz Courtyard,1,0.5812669610362707,0.15343994793419002,5.71686544495911,3.550076086474883,133,6483.48,6454.087207078934,99999.0
++++++++-uesfm,Alcatraz Courtyard,2,0.6293740753551684,0.16343948136157807,5.713459273661585,3.459422664018691,133,6483.73,6451.153926610947,99999.0
+++++++++uesfm,Alcatraz Courtyard,0,0.5509063563081962,0.14050635120746804,5.528957815039945,3.569220349534855,133,6556.5,6511.41095662117,99999.0
++++++++ esfm,Alcatraz Water Tower,0,0.6984432687370988,0.38694796813795407,4.741313233832226,2.1058898944784543,172,2461.59,2440.421297073364,99999.0
++++++++ esfm,Alcatraz Water Tower,1,0.5944425806062753,0.3287537445614408,4.5656682619915765,1.9703698309044646,172,2467.28,2454.429310798645,99999.0
++++++++ esfm,Alcatraz Water Tower,2,0.7051984712289041,0.39359395644708545,4.982802041506025,2.1318655747232063,172,2455.14,2441.783402681351,99999.0
++++++++-uesfm,Alcatraz Water Tower,0,18.69941055700858,8.189402666249999,12.38813738647375,6.481578998967327,172,3857.94,3831.41609287262,99999.0
++++++++-uesfm,Alcatraz Water Tower,1,1.3394060108035364,0.7384271683550214,6.577617375342355,3.200105455600137,172,3887.47,3865.343374729156,99999.0
++++++++-uesfm,Alcatraz Water Tower,2,2.200189184784068,1.3839747978399466,6.854463337738994,3.1730434024886014,172,3819.31,3799.390902757645,99999.0
+++++++++uesfm,Alcatraz Water Tower,0,2.016029335283383,1.087082854315953,7.9502009315303575,4.413488326253207,172,3805.19,3775.908159971237,99999.0
+++++++++uesfm,Alcatraz Water Tower,1,1.4259772151091428,0.7645994701417347,7.177871865477131,3.7683369835989504,172,3787.07,3763.27596282959,99999.0
++++++++ esfm,Buddah Tooth Relic Temple Singapore,0,1.3132762212436329,0.31652548993670354,9.903890396062208,4.634952217191846,162,2921.45,2903.059753894806,99999.0
++++++++ esfm,Buddah Tooth Relic Temple Singapore,1,0.9908742070528852,0.2335767172234499,9.836436953133868,4.661767906617449,162,2915.39,2904.387546062469,99999.0
++++++++ esfm,Buddah Tooth Relic Temple Singapore,2,1.089403690069201,0.2896515999941069,10.151103921611115,4.628183275251743,162,2913.67,2902.928613901138,99999.0
++++++++-uesfm,Buddah Tooth Relic Temple Singapore,0,1.9767734026333772,0.5777116480820644,15.584999477939833,8.800407928031486,162,4315.97,4296.739196300507,99999.0
++++++++-uesfm,Buddah Tooth Relic Temple Singapore,1,1.4532821126601407,0.345133635718661,16.753182673783428,9.555989756393366,162,4334.28,4311.474883317947,99999.0
++++++++-uesfm,Buddah Tooth Relic Temple Singapore,2,32.81105779154638,6.0024233739495125,26.564606536203048,13.925597303414058,162,4312.37,4291.042752504349,99999.0
+++++++++uesfm,Buddah Tooth Relic Temple Singapore,0,1.5799948547864517,0.43886498334860274,13.484389374830494,7.719072745032001,162,4294.34,4266.538989067078,99999.0
+++++++++uesfm,Buddah Tooth Relic Temple Singapore,1,1.934129180062467,0.5924991617860504,13.872322562031941,8.04977435062149,162,4303.62,4259.311170101166,99999.0
++++++++ esfm,Doge Palace Venice,0,0.9741701978804636,0.2578214102266912,6.511459231891615,3.6947627756511525,241,28795.18,28746.22650837898,99999.0
++++++++ esfm,Doge Palace Venice,1,0.627860879835449,0.1773446161890865,6.351890452986879,3.537139014829222,241,28742.27,28699.91308617592,99999.0
++++++++-uesfm,Doge Palace Venice,0,0.9110673544913167,0.2904399918743801,10.966183720673449,6.771416746021886,241,21074.95,21017.30448675156,99999.0
++++++++ esfm,Door Lund,0,0.01167051570787235,0.003654424593160691,1.8552512563951393,0.9492901559591532,12,4892.41,4878.351888656616,99999.0
++++++++ esfm,Door Lund,1,0.009685491454411136,0.0030270923396151107,1.6462990657331118,0.9049495114630693,12,4884.51,4877.529450178146,99999.0
++++++++ esfm,Door Lund,2,0.02068285629921104,0.005193423834514985,1.933833706356197,0.9506754982524039,12,4888.17,4880.201772212982,99999.0
++++++++-uesfm,Door Lund,0,14.059454220682019,0.9733778175929079,11.85413335315025,6.517535151323047,12,9185.47,8715.343282699585,95000.0
++++++++-uesfm,Door Lund,1,14.111993883089163,0.9618908892808385,11.853785941610944,6.504290333453439,12,9203.34,8729.614821910858,95000.0
++++++++-uesfm,Door Lund,2,14.344083655704312,0.9193773651816372,12.058851721280973,6.643622749322514,12,9176.97,9162.127201795578,99999.0
++++++++ esfm,Drinking Fountain Somewhere In Zurich,0,25.804179845300204,1.1242096722251314,8.219025541657992,3.3464550934365884,14,888.07,874.4874215126038,99999.0
++++++++ esfm,Drinking Fountain Somewhere In Zurich,1,25.754577171528883,1.1271567319301725,8.245599860088616,3.352895635561939,14,882.33,873.2415659427643,99999.0
++++++++ esfm,Drinking Fountain Somewhere In Zurich,2,0.019604308735801684,0.006780729271172703,1.7090486008047308,0.6375575382268279,14,881.11,873.6504108905792,99999.0
++++++++-uesfm,Drinking Fountain Somewhere In Zurich,0,26.27952541668808,1.102546079947844,8.40971803772888,3.4593948661771203,14,2969.96,2950.480526924133,99999.0
++++++++-uesfm,Drinking Fountain Somewhere In Zurich,1,26.264849988219943,1.1078406736783482,8.547474098662555,3.585334207185317,14,2960.9,2504.42572927475,85000.0
++++++++-uesfm,Drinking Fountain Somewhere In Zurich,2,0.06235835436589852,0.011103715758719332,2.5743283079384534,1.5361135173722633,14,2960.08,2945.673992872238,99999.0
+++++++++uesfm,Drinking Fountain Somewhere In Zurich,0,25.946100100284117,1.1178920745623426,8.471712388928934,3.6241397282821626,14,5817.37,5796.999963760376,99999.0
++++++++ esfm,East Indiaman Goteborg,0,6.033696331509855,1.0773322362193534,7.427433157074123,2.292729458380311,179,3990.25,3970.732353687286,99999.0
++++++++ esfm,East Indiaman Goteborg,1,5.313967690277326,0.9356515362150493,7.350753973659207,2.2747477951684703,179,3986.66,3970.914752960205,99999.0
++++++++ esfm,East Indiaman Goteborg,2,5.357164002763642,0.9254557315236661,7.174538078735992,2.2493605774605823,179,3984.64,3967.810528755188,99999.0
++++++++-uesfm,East Indiaman Goteborg,0,4.992588850190341,0.9093886386013725,9.930437318497303,4.011819888270754,179,5247.16,5221.816437721252,99999.0
++++++++-uesfm,East Indiaman Goteborg,1,5.382092262718057,0.9927766021898647,10.035611563702533,3.912004668740913,179,5245.65,5210.653891563416,99999.0
++++++++-uesfm,East Indiaman Goteborg,2,5.554715231405653,0.9752252406138763,9.5551521706325,3.7417116053580193,179,5231.48,5207.815300703049,99999.0
+++++++++uesfm,East Indiaman Goteborg,0,5.365772057886591,0.9552115949640507,9.655123223073948,4.174486084237091,179,5203.6,5178.753549814224,99999.0
++++++++ esfm,Ecole Superior De Guerre,0,0.4863901164687255,0.1239377922325492,3.750666445958877,2.0528113108131016,35,2385.33,2371.708672523499,99999.0
++++++++ esfm,Ecole Superior De Guerre,1,0.3031794677170201,0.07921628898965839,3.2456935905644406,1.730123845260616,35,2379.34,2371.191816806793,99999.0
++++++++ esfm,Ecole Superior De Guerre,2,0.2580439805296837,0.06772089648290676,3.257951412507941,1.687291979251782,35,2376.49,2368.935952663422,99999.0
++++++++-uesfm,Ecole Superior De Guerre,0,0.28243957678843373,0.0778675008788964,5.291358032458062,3.601773151002547,35,4647.51,4622.850021600723,99999.0
++++++++-uesfm,Ecole Superior De Guerre,1,0.25507313740145815,0.06695766472508183,5.170495431958795,3.5179325723573998,35,4636.07,4616.890106916428,99999.0
++++++++-uesfm,Ecole Superior De Guerre,2,0.3137499276035537,0.08683967313665406,5.583934687507232,3.7813953498055604,35,4617.92,4602.193422079086,99999.0
++++++++ esfm,Eglise du dome,0,0.3242410065346163,0.0824797769992069,3.906129661328729,1.3512292775673158,85,25054.74,25028.64922428131,99999.0
++++++++ esfm,Eglise du dome,1,0.3771623368290559,0.09432763361451474,3.692490941110161,1.297032357665518,85,25144.16,25116.01952123642,99999.0
++++++++-uesfm,Eglise du dome,0,45.800562123362404,6.451383786272971,7.963674120656965,3.986938865913781,85,23371.73,23333.49408817291,99999.0
++++++++ esfm,Folke Filbyter,0,89.03873163576341,0.12895560134543632,17.986624161769267,7.423093846189277,40,1582.8,1573.360969305038,99999.0
++++++++ esfm,Folke Filbyter,1,84.74740427947457,0.13132264058966686,17.98706393015447,7.385900243615879,40,1579.13,1572.079373121262,99999.0
++++++++ esfm,Folke Filbyter,2,83.70817707245973,0.1261878953205589,21.85200129525089,6.324245998829542,40,1583.76,1576.207699537277,99999.0
++++++++-uesfm,Folke Filbyter,0,80.34590333370599,0.13169898199385566,22.83327162619912,11.711593158296864,40,3701.49,3129.452286243439,85000.0
++++++++-uesfm,Folke Filbyter,1,81.28676564061475,0.1282318052949447,24.480394314414323,9.74969805057316,40,3698.11,3681.891911745071,99999.0
++++++++-uesfm,Folke Filbyter,2,82.23270227083374,0.1312531386108397,22.153947807931974,10.961536903594071,40,3709.96,3695.84988451004,99999.0
+++++++++uesfm,Folke Filbyter,0,84.55526303793802,0.13091275274611416,29.190267154194732,11.160073225547443,40,3747.65,3727.990335702896,99999.0
+++++++++uesfm,Folke Filbyter,1,84.87047537526948,0.13112291273275417,32.36503566471496,13.743964264262598,40,3743.53,3728.497133255005,99999.0
++++++++ esfm,Fort Channing Gate Singapore,0,0.10143646071220622,0.04531774433800129,1.7158776598376824,0.8165178663538488,27,3438.39,3428.288990497589,99999.0
++++++++ esfm,Fort Channing Gate Singapore,1,0.08873951454303304,0.039001318240256114,1.783712095722291,0.8156020254191567,27,3430.88,3424.490573883057,99999.0
++++++++ esfm,Fort Channing Gate Singapore,2,0.09031224636384508,0.03957289173255022,1.850800887931411,0.820658413919779,27,3433.07,3426.547735452652,99999.0
++++++++-uesfm,Fort Channing Gate Singapore,0,33.158343156383296,11.908537566474344,12.69248500308198,5.438758486477886,27,6209.29,6192.107179880142,99999.0
++++++++-uesfm,Fort Channing Gate Singapore,1,0.07319798057764343,0.03541606344760984,3.4819579894798927,2.2085534812612853,27,6159.7,6144.797720909119,99999.0
++++++++-uesfm,Fort Channing Gate Singapore,2,0.15394831153888047,0.0686769122909015,3.77993377770425,2.2995094868650368,27,6147.27,6132.745970487595,99999.0
++++++++ esfm,Golden Statue Somewhere In Hong Kong,0,0.23894829924734365,0.053817571637812096,1.7239352536484445,0.744089758310296,18,4096.74,4087.775670289993,99999.0
++++++++ esfm,Golden Statue Somewhere In Hong Kong,1,0.2233990458438394,0.053061028269618456,1.8936878914364637,0.7746436618607616,18,4096.28,4088.994391202927,99999.0
++++++++ esfm,Golden Statue Somewhere In Hong Kong,2,0.23397914670516012,0.051137498616263405,1.87070255923672,0.7763877145267694,18,4094.54,4087.674698591232,99999.0
++++++++-uesfm,Golden Statue Somewhere In Hong Kong,0,0.36074365888948035,0.06894135246785482,2.939092008443413,1.863811153764833,18,7761.69,7746.569520235062,99999.0
++++++++-uesfm,Golden Statue Somewhere In Hong Kong,1,0.35638930493960375,0.06132592684828702,2.686157398440538,1.735571387461727,18,7759.16,7745.389947891235,99999.0
++++++++-uesfm,Golden Statue Somewhere In Hong Kong,2,0.3955566726941678,0.09197790539367184,2.878341172309852,1.8164738202745854,18,7766.46,7751.458931922913,99999.0
++++++++ esfm,Gustav Vasa,0,3.4881363194229826,0.2273181104934767,1.7369784534725496,0.9521606063342205,18,711.53,703.3091380596161,99999.0
++++++++ esfm,Gustav Vasa,1,3.4307883498221035,0.21432466556133864,1.5556448596237022,0.9232184455756751,18,710.28,703.9906163215637,99999.0
++++++++ esfm,Gustav Vasa,2,3.6089062562451386,0.2196578287395002,1.673803634770729,0.9429755567757121,18,707.79,701.5903799533844,99999.0
++++++++-uesfm,Gustav Vasa,0,5.828154714648938,0.32510821331670736,3.0711042143581646,2.4170166616864357,18,2566.0,2547.931907176971,99999.0
++++++++-uesfm,Gustav Vasa,1,6.515239842974278,0.35941242702736925,3.089677100739632,2.4476428398062584,18,2530.9,1385.089058637619,55000.0
++++++++-uesfm,Gustav Vasa,2,5.954036423569478,0.34942942401850696,3.101296679577594,2.400483554095863,18,2533.76,2520.798699855804,99999.0
+++++++++uesfm,Gustav Vasa,0,36.645536646380805,1.0070193571286277,4.980488290477233,3.197977230260218,18,2562.94,2419.297919034958,95000.0
++++++++ esfm,GustavIIAdolf,0,46.92758389264633,8.570499881283022,11.684017169857798,5.131125459552879,57,1047.82,1038.018162250519,99999.0
++++++++ esfm,GustavIIAdolf,1,47.21999673071185,8.515052769156952,11.695065125063977,5.091617086264161,57,1041.41,1034.5799472332,99999.0
++++++++ esfm,GustavIIAdolf,2,47.2312223396789,8.523261295519099,11.787583212607423,5.2715388868550574,57,1043.28,1036.478415489197,99999.0
++++++++-uesfm,GustavIIAdolf,0,89.37607436538163,13.276440526172165,15.53883714368065,5.864332719302596,57,2904.42,2881.687849521637,99999.0
++++++++-uesfm,GustavIIAdolf,1,70.16367503302509,11.555242253627961,15.162174576474532,6.9709512967490355,57,2884.62,2870.103493452072,99999.0
++++++++-uesfm,GustavIIAdolf,2,72.72996778765942,11.054260851523775,16.85269001998736,9.009102977660282,57,2888.7,2874.096453428268,99999.0
+++++++++uesfm,GustavIIAdolf,0,85.51179144359693,12.792264407383987,17.018024925092185,8.196266635685827,57,2973.35,2956.916826248169,99999.0
+++++++++uesfm,GustavIIAdolf,1,88.9494736685426,13.260723460498589,17.699982689478404,8.888010081099381,57,2995.3,2976.055320978165,99999.0
++++++++ esfm,Jonas Ahlstromer,0,64.46662959289324,11.985605264945464,12.602584968179986,2.9168445332917745,40,614.88,607.3498430252075,99999.0
++++++++ esfm,Jonas Ahlstromer,1,38.42510202464649,8.741159740710629,10.25268858530355,3.7024742418884333,40,618.73,611.5567183494568,99999.0
++++++++ esfm,Jonas Ahlstromer,2,43.32891670826628,8.741053567623224,10.553031866494822,3.031717304343782,40,612.74,606.6802380084991,99999.0
++++++++-uesfm,Jonas Ahlstromer,0,42.30073879858729,10.272232427418913,9.098970988131883,3.992857074878733,40,2339.57,1280.604686737061,55000.0
++++++++-uesfm,Jonas Ahlstromer,1,42.18412804385421,10.310808592826955,8.453808908218651,3.4329852879523464,40,2334.33,2320.362742424011,99999.0
++++++++-uesfm,Jonas Ahlstromer,2,42.25301394187691,10.315519611163806,8.793789077520445,3.8046125347596442,40,2311.04,2297.139800786972,99999.0
++++++++ esfm,Kings College University Of Toronto,0,14.400461796497643,3.2653370282498866,4.064353916685246,1.6753245173381437,77,1074.94,1066.306197404861,99999.0
++++++++ esfm,Kings College University Of Toronto,1,11.956699445712928,2.705452710546247,3.9727144140636783,1.5797087908756111,77,1071.64,1064.129351377487,99999.0
++++++++ esfm,Kings College University Of Toronto,2,0.7191591237413282,0.16178494376153177,2.656807607821862,1.1949677972638288,77,1090.36,1082.987608671188,99999.0
++++++++-uesfm,Kings College University Of Toronto,0,31.932532001930838,2.9775501815475,6.099555590572752,3.8717713150174773,77,2884.23,2866.194714784622,99999.0
++++++++-uesfm,Kings College University Of Toronto,1,2.4241467960672063,0.41577835428574456,4.631696476065898,3.148046400399828,77,2899.51,2884.114916801453,99999.0
++++++++-uesfm,Kings College University Of Toronto,2,30.212552334890066,2.3209884971628156,5.844709739288878,3.754268841443037,77,2874.99,2860.901805400848,99999.0
++++++++ esfm,Lund University Sphinx,0,13.661465370705821,3.5591094267381553,6.242667318159146,2.715620555070548,70,2718.96,2709.5872194767,99999.0
++++++++ esfm,Lund University Sphinx,1,12.524302160109885,3.2130942199657664,6.1182420215295075,2.7468468987195953,70,2722.86,2715.36418390274,99999.0
++++++++ esfm,Lund University Sphinx,2,7.98961938632192,2.1710959765491817,5.807289242508196,2.893370267054225,70,2713.88,2706.478076219559,99999.0
++++++++-uesfm,Lund University Sphinx,0,15.165716414733652,3.5564827651349713,10.803408462752863,6.730422995453065,70,4644.92,4626.135177612305,99999.0
++++++++-uesfm,Lund University Sphinx,1,68.88899086055604,13.548881652837478,14.336626262363456,7.626356837439143,70,4703.31,4687.50173997879,99999.0
++++++++-uesfm,Lund University Sphinx,2,15.014326075576053,3.5607694164340526,11.039432108003597,6.81081192121135,70,4615.6,4600.294319868088,99999.0
++++++++ esfm,Nijo Castle Gate,0,0.6521524412276166,0.13305866403975986,7.404953325705311,3.0629804954987674,19,1097.29,1084.301982402802,99999.0
++++++++ esfm,Nijo Castle Gate,1,0.8564043952718835,0.16950374865879153,7.358076475544799,3.1235223975185864,19,1094.89,1084.640900611877,99999.0
++++++++ esfm,Nijo Castle Gate,2,1.0089926380330574,0.19095470213426863,7.481049991287358,3.1671779658403687,19,1093.35,1083.446812152863,99999.0
++++++++-uesfm,Nijo Castle Gate,0,1.6923297320319277,0.3246009364794658,11.792009654594867,6.945622003724464,19,3310.13,3288.75860619545,99999.0
++++++++-uesfm,Nijo Castle Gate,1,1.6697791743236226,0.32440429939345133,12.704681259536075,7.466128811395402,19,3308.19,3290.209539175034,99999.0
++++++++-uesfm,Nijo Castle Gate,2,1.682403384345942,0.32817821817561266,11.917977420973372,7.102279949089778,19,3301.77,3284.964460611343,99999.0
++++++++-uesfm,Pantheon Paris,0,0.3823220210788846,0.05987691585642601,7.649321203280243,4.797127317249956,179,5506.88,5473.694003582001,99999.0
++++++++-uesfm,Pantheon Paris,1,45.61471478909275,6.671068486991568,16.912359249810606,6.602415648206237,179,5491.31,5459.329566717148,99999.0
++++++++-uesfm,Pantheon Paris,2,12.03796171625132,2.465102828965073,16.125095834055482,6.860965809143959,179,5493.94,5463.555550813675,99999.0
+++++++++uesfm,Pantheon Paris,0,0.5453004964472504,0.06999784414241637,7.790870638815493,5.057509442480535,179,5594.6,5562.967379331589,99999.0
++++++++ esfm,Park Gate Clermont Ferrand,0,19.399216821978534,9.438471236639753,7.521969683469249,3.892136696028138,34,1597.96,1588.07075381279,99999.0
++++++++ esfm,Park Gate Clermont Ferrand,1,19.325402297843297,9.417998214523331,7.465007292717891,3.849088919564891,34,1596.87,1589.703137874603,99999.0
++++++++ esfm,Park Gate Clermont Ferrand,2,25.440611579781365,11.720641869372841,8.15032188787613,4.0871739024940315,34,1595.99,1588.841923236847,99999.0
++++++++-uesfm,Park Gate Clermont Ferrand,0,19.27114964212729,9.381930985797803,7.856513901068924,4.317004881495189,34,3598.69,3580.396904230118,99999.0
++++++++-uesfm,Park Gate Clermont Ferrand,1,19.288145181777256,9.394362801959678,7.762082007624792,4.207851936960509,34,3592.66,3577.494926214218,99999.0
++++++++-uesfm,Park Gate Clermont Ferrand,2,19.345727373971844,9.415066249707772,7.936652645099164,4.425507919979627,34,3600.86,3581.473034620285,99999.0
++++++++ esfm,Plaza De Armas Santiago,0,0.7341628728579594,0.32156050159900956,6.000154308231358,2.679971945741675,240,6375.02,6333.455562829971,99999.0
++++++++ esfm,Plaza De Armas Santiago,1,0.5176165701053533,0.22744923951256435,5.661576462867733,2.533952088248649,240,6373.68,6336.030319213867,99999.0
++++++++ esfm,Plaza De Armas Santiago,2,1.0740077415291467,0.46304622323619105,6.365716833190761,2.733195456670397,240,6369.35,6331.436369419098,99999.0
++++++++-uesfm,Plaza De Armas Santiago,0,6.6772176498607925,2.909933722327364,11.384209379387066,5.623987151554922,240,6540.76,6491.619550704956,99999.0
++++++++-uesfm,Plaza De Armas Santiago,1,1.3364898126442175,0.5686650140764532,7.951759184923103,3.933795015892702,240,6536.61,6482.320575714111,99999.0
++++++++-uesfm,Plaza De Armas Santiago,2,1.9671483940705004,0.8276654865349118,9.054185015150745,4.441179120460681,240,6529.24,6477.723381996155,99999.0
++++++++ esfm,Porta San Donato Bologna,0,0.5746001989174886,0.10608894230214515,5.691906895070585,2.4183460353027986,141,3751.75,3734.281898736954,99999.0
++++++++ esfm,Porta San Donato Bologna,1,0.8978003419853328,0.14474404936935045,5.682379581165418,2.563141107750346,141,3758.51,3742.688964605331,99999.0
++++++++ esfm,Porta San Donato Bologna,2,0.30440272958620346,0.07026546193502164,5.435311727517131,2.1626312981864086,141,3752.23,3738.056202173233,99999.0
++++++++-uesfm,Porta San Donato Bologna,0,1.1269164513732723,0.19275737956525044,10.071225650489282,5.557466375146339,141,5111.57,5083.21807050705,99999.0
++++++++-uesfm,Porta San Donato Bologna,1,60.66366912240625,10.234123128225525,19.593360464637342,9.305574874002819,141,5125.57,5098.085618019104,99999.0
++++++++-uesfm,Porta San Donato Bologna,2,1.2271289404198877,0.19605254084885107,10.066750816092268,5.595831381287904,141,5116.95,5093.473650932312,99999.0
++++++++ esfm,Round Church Cambridge,0,2.1173198662081005,0.9129854085205968,5.854320175654564,1.9683268910531708,92,9788.65,9759.719386577606,99999.0
++++++++ esfm,Round Church Cambridge,1,2.1415279636494953,0.9267927094737857,6.33905951937897,2.0439997148640834,92,9764.61,9741.356333732605,99999.0
++++++++-uesfm,Round Church Cambridge,0,2.5434984075631974,1.0219048587087352,8.914080922081759,4.131734829440289,92,12088.06,12047.02147889137,99999.0
++++++++-uesfm,Round Church Cambridge,1,2.435781054756339,0.9827352719070902,8.178482224933683,3.8400051230290893,92,12034.64,11993.72608184814,99999.0
+++++++++esfm,Round Church Cambridge,2,2.1507951248804917,0.9401429799233081,6.115230559713554,2.0517209910251797,92,9784.55,9763.525354623795,99999.0
++++++++ esfm,Skansen Kronan Gothenburg,0,0.3380139741078169,0.1158931606908392,3.0478862369293624,1.5340801117570733,131,17163.94,17140.15425729752,99999.0
++++++++-uesfm,Skansen Kronan Gothenburg,0,0.37472494768162085,0.1243676312212316,4.798736338080396,3.348916057055834,131,16235.09,16202.21351981163,99999.0
+++++++++esfm,Skansen Kronan Gothenburg,1,0.21516831826827018,0.07458182746108981,2.786705021535272,1.4619994473169777,131,17129.62,17109.18022465706,99999.0
+++++++++esfm,Skansen Kronan Gothenburg,2,0.3508356515622903,0.11667319277987132,2.8836839507901773,1.5216584205042176,131,17167.47,17142.66050887108,99999.0
++++++++ esfm,Smolny Cathedral St Petersburg,0,21.81278753416543,2.110667559831311,11.2315357584775,6.379259553776089,131,10364.67,10320.51852750778,99999.0
++++++++ esfm,Smolny Cathedral St Petersburg,1,21.799828748358873,2.111313741354268,11.225141339870355,6.36605939788532,131,10360.52,10319.01374697685,99999.0
++++++++-uesfm,Smolny Cathedral St Petersburg,0,22.143392894578028,2.0985584569799265,12.52645725687386,7.895265841408903,131,10475.74,10416.62969732285,99999.0
+++++++++esfm,Smolny Cathedral St Petersburg,2,21.71090600320656,2.110307413725059,11.259799763291888,6.4136596726420905,131,10363.17,10320.08415436745,99999.0
+++++++++esfm,Some Cathedral In Barcelona,0,0.6775058368464424,0.24064262464083658,6.20165095815897,3.7253636674062887,177,5002.81,4985.274734258652,99999.0
+++++++++esfm,Some Cathedral In Barcelona,1,0.6722147495861831,0.2423973792150467,6.350786792749083,3.720536755931007,177,5009.15,4991.552013158798,99999.0
+++++++++esfm,Some Cathedral In Barcelona,2,0.6046048207068493,0.21699252142655406,6.261075100347952,3.7258171467923633,177,4993.96,4975.845166683197,99999.0
++++++++ esfm,Sri Mariamman Singapore,0,1.4920493985513708,0.4412389781578158,10.406406306167431,6.369585247957454,222,5987.98,5970.066485881805,99999.0
++++++++ esfm,Sri Mariamman Singapore,1,1.1345564396944785,0.3608906720626337,9.703927906328842,5.885963785879842,222,5993.62,5975.365797519684,99999.0
++++++++ esfm,Sri Mariamman Singapore,2,1.0293707921546553,0.3426709250774071,10.15220066698994,6.067537729054635,222,5975.02,5958.786058187485,99999.0
++++++++-uesfm,Sri Mariamman Singapore,0,16.417883129790784,5.526406595652791,19.962711816743173,12.18830031311293,222,6531.45,6494.750840425491,99999.0
++++++++-uesfm,Sri Mariamman Singapore,1,1.3793173305528281,0.5014534676018023,16.10857603648069,10.651562750176396,222,6467.65,6426.201541185379,99999.0
++++++++-uesfm,Sri Mariamman Singapore,2,1.639001766282598,0.536397596737868,17.490398022274007,11.439337896248928,222,6457.9,6427.717175960541,99999.0
++++++++ esfm,Sri Thendayuthapani Singapore,0,0.7190396606626565,0.17922550285921848,7.12365374281766,2.231392236341273,98,13362.09,13320.09781932831,99999.0
++++++++-uesfm,Sri Thendayuthapani Singapore,0,47.0750455982536,3.8553135654690074,14.133068630999212,7.117179633590087,98,13808.44,13754.94554877281,99999.0
+++++++++esfm,Sri Thendayuthapani Singapore,1,0.5350953449295449,0.1261925950852554,6.4467364012557615,2.04778550672059,98,13347.85,13309.03068971634,99999.0
+++++++++esfm,Sri Thendayuthapani Singapore,2,0.5262093764563488,0.14374074415280041,7.476614790858722,2.2575164196527426,98,13304.56,13262.61834168434,99999.0
++++++++ esfm,Sri Veeramakaliamman Singapore,0,2.333869921178247,0.542847254574005,11.965270419451508,5.472877662905037,157,12392.18,12365.15180635452,99999.0
++++++++-uesfm,Sri Veeramakaliamman Singapore,0,29.40487246624615,3.8676026992817505,28.950500021302087,14.32295224594287,157,12544.69,12504.26410722733,99999.0
+++++++++esfm,Sri Veeramakaliamman Singapore,1,2.4804161216418734,0.5731176371311499,12.150997217479388,5.5309200508775875,157,12387.11,12363.29210400581,99999.0
+++++++++esfm,Sri Veeramakaliamman Singapore,2,2.3974569344813434,0.5599333960859275,12.58650507397198,5.711763361742226,157,12354.49,12330.03034973145,99999.0
++++++++ esfm,Statue Of Liberty,0,52.35559762033152,28.241594953600423,688383.1603142131,683624.423417858,134,102.64,43.32754778862,45000.0
++++++++ esfm,Statue Of Liberty,1,52.36920727661949,28.627801879821945,9277.715116667516,9303.546387753544,134,99.78,59.98358988761902,65000.0
++++++++ esfm,Statue Of Liberty,2,52.3478568435799,28.64052858422677,7449.598084265948,7416.803834848509,134,97.28,3.520912647247314,0.0
++++++++-uesfm,Statue Of Liberty,0,40.92379801166407,19.915794833689585,48.781849500680366,25.904882064330994,134,12927.11,12902.23110127449,99999.0
++++++++-uesfm,Statue Of Liberty,1,38.79484508812604,17.090491683338875,33.27459817133403,17.608726936833005,134,12920.89,12896.32496738434,99999.0
++++++++ esfm,The Pumpkin,0,27.59459354015253,5.94835892276036,27.52772578222832,9.11126349703041,196,4330.9,4316.109169960022,99999.0
++++++++ esfm,The Pumpkin,1,26.54397520858153,5.701234509093555,27.342758332715125,8.907676505669826,196,4327.98,4314.043692827225,99999.0
++++++++ esfm,The Pumpkin,2,23.05395646123413,5.007378502393024,25.75879879971585,7.898438464943201,196,4322.94,4310.356798410416,99999.0
++++++++-uesfm,The Pumpkin,0,25.490608822831348,5.440568114393512,34.198429037653845,15.6024365790899,196,5793.03,5767.982522964478,99999.0
++++++++-uesfm,The Pumpkin,1,27.26556900533122,5.820572144056583,35.6951097814524,16.572200360468877,196,5785.61,5760.521738529205,99999.0
++++++++-uesfm,The Pumpkin,2,26.95722616022698,5.73349875876088,36.207441893083136,17.07678356860021,196,5793.15,5768.243654251099,99999.0
++++++++ esfm,Thian Hook Keng Temple Singapore,0,1.0653505173922067,0.1079565219850111,15.054152885837562,5.81055046268767,138,3299.26,3281.187238931656,99999.0
++++++++ esfm,Thian Hook Keng Temple Singapore,1,0.8390930608109779,0.08272919436528102,14.587041005106803,5.705702663819475,138,3306.93,3283.257929086685,99999.0
++++++++ esfm,Thian Hook Keng Temple Singapore,2,0.8760513596629002,0.08764395630143504,15.102705362055175,5.7567965038255124,138,3289.42,3278.173619031906,99999.0
++++++++-uesfm,Thian Hook Keng Temple Singapore,0,1.146131162829257,0.10799243740327723,20.24888615229727,10.297158058757331,138,4626.48,4595.79723072052,99999.0
++++++++-uesfm,Thian Hook Keng Temple Singapore,1,1.2072956626750737,0.12821580405567637,21.242388560332195,10.655698470690366,138,4600.34,4578.699931621552,99999.0
++++++++-uesfm,Thian Hook Keng Temple Singapore,2,1.251054659622132,0.12561972972485672,20.875018460120653,10.416875106878496,138,4610.3,4581.792747020721,99999.0
++++++++ esfm,Tsar Nikolai I,0,42.507553768489565,8.869914103947062,10.846690028504929,4.007961025902478,98,4932.15,4920.938139438629,99999.0
++++++++ esfm,Tsar Nikolai I,1,43.09835370739786,8.983133733509343,11.402130572143294,4.032588092580996,98,4924.71,4914.929541349411,99999.0
++++++++-uesfm,Tsar Nikolai I,0,75.94880014543322,15.146573953653359,15.307363921859029,6.734065570805788,98,6652.35,6629.384099006653,99999.0
++++++++-uesfm,Tsar Nikolai I,1,80.87807139992418,15.64102228288136,21.553650044939403,9.179513384124702,98,6641.34,6623.797205209732,99999.0
+++++++++esfm,Tsar Nikolai I,2,40.892180545819315,8.342265568650971,11.381967217434166,4.272306098513181,98,4930.74,4920.495186090469,99999.0
++++++++ esfm,Urban II,0,59.0297968076411,11.008489348760156,17.600594623525765,5.400933781867582,96,8889.69,8878.01452589035,99999.0
+++++++++esfm,Urban II,1,58.947927470837534,11.025374954076108,17.465515658589908,5.68378594242137,96,8885.17,8876.618469953537,99999.0
+++++++++esfm,Urban II,2,58.77798771298748,10.982786678935327,17.546556385759054,5.767405629316944,96,8893.99,8885.34243106842,99999.0
++++++++ esfm,Vercingetorix,0,79.07478057258302,9.776081681696928,6.649989325889541,1.7432748253588333,69,1223.75,1212.340968608856,99999.0
++++++++ esfm,Vercingetorix,1,80.09313371434186,9.858626097407651,6.426263444651325,1.6979463697629655,69,1222.8,1213.763489961624,99999.0
++++++++ esfm,Vercingetorix,2,89.68787080902364,10.950615257181747,8.695412468830675,3.360645595149667,69,1225.36,1215.808262825012,99999.0
++++++++-uesfm,Vercingetorix,0,78.95942988866727,10.356244187384963,8.07065603849916,3.372297052368834,69,3105.42,3085.208213567734,99999.0
++++++++-uesfm,Vercingetorix,1,89.99175121301754,11.178484677344343,8.929255781006072,4.302365177465691,69,3105.88,3091.564247369766,99999.0
++++++++-uesfm,Vercingetorix,2,89.31151857112468,11.134780185118505,9.044030033429575,4.351952473195928,69,3105.52,3089.0203332901,99999.0
++++++++ esfm,Yueh Hai Ching Temple Singapore,0,0.6268598783684821,0.08360284283022794,5.56223941139456,2.132543748253723,43,1749.28,1738.747783660889,99999.0
+++++++++esfm,Yueh Hai Ching Temple Singapore,1,0.5064956072004329,0.07235771132992698,5.30294026596974,2.06113770612239,43,1750.4,1739.704204082489,99999.0
+++++++++esfm,Yueh Hai Ching Temple Singapore,2,0.49998151063409024,0.07009863154178562,5.92393490371475,2.2317763685579015,43,1746.32,1738.905562877655,99999.0
++++++++diff --git a/code/results/single_scene/summary_table.md b/code/results/single_scene/summary_table.md
++++++++index bd3f673..005d9ef 100644
++++++++--- a/code/results/single_scene/summary_table.md
+++++++++++ b/code/results/single_scene/summary_table.md
++++++++@@ -2,44 +2,45 @@
++++++++ 
++++++++ Mean over seeds per (scene, method); best value per scene in **bold** (lower is better except Nr).
++++++++ 
++++++++-| Scene | ESFM Rot (deg) | U-ESFM Rot (deg) | ESFM (paper) Rot (deg) | ESFM Trans | U-ESFM Trans | ESFM (paper) Trans | ESFM Reproj (px) | U-ESFM Reproj (px) | ESFM (paper) Reproj (px) | ESFM Nr | U-ESFM Nr | ESFM (paper) Nr | ESFM Time (s) | U-ESFM Time (s) | ESFM (paper) Time (s) | ESFM Rot-BA (deg) | U-ESFM Rot-BA (deg) | ESFM (paper) Rot-BA (deg) | ESFM Trans-BA | U-ESFM Trans-BA | ESFM (paper) Trans-BA | ESFM Reproj-BA (px) | U-ESFM Reproj-BA (px) | ESFM (paper) Reproj-BA (px) |
+++++++++| Scene | ESFM (official code) Rot (deg) | U-ESFM Rot (deg) | ESFM (paper) Rot (deg) | ESFM (official code) Trans | U-ESFM Trans | ESFM (paper) Trans | ESFM (official code) Reproj (px) | U-ESFM Reproj (px) | ESFM (paper) Reproj (px) | ESFM (official code) Nr | U-ESFM Nr | ESFM (paper) Nr | ESFM (official code) Time (s) | U-ESFM Time (s) | ESFM (paper) Time (s) | ESFM (official code) Rot-BA (deg) | U-ESFM Rot-BA (deg) | ESFM (paper) Rot-BA (deg) | ESFM (official code) Trans-BA | U-ESFM Trans-BA | ESFM (paper) Trans-BA | ESFM (official code) Reproj-BA (px) | U-ESFM Reproj-BA (px) | ESFM (paper) Reproj-BA (px) |
++++++++ |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
++++++++-| Alcatraz Courtyard | **0.362** | 0.616 | 0.619 | **0.093** | 0.161 | 0.160 | **3.435** | 5.796 | 1.640 | **133** | 133 | -- | **4826.713** | 6481.757 | -- | -- | -- | 0.049 | -- | -- | 0.015 | -- | -- | 0.810 |
++++++++-| Alcatraz Water Tower | **0.666** | 7.413 | 0.933 | **0.370** | 3.437 | 0.518 | **4.763** | 8.607 | 2.130 | **172** | 172 | -- | **2461.337** | 3854.907 | -- | -- | -- | 0.230 | -- | -- | 0.116 | -- | -- | 0.550 |
++++++++-| Buddah Tooth Relic Temple Singapore | **1.131** | 12.080 | 1.030 | **0.280** | 2.308 | 0.233 | **9.964** | 19.634 | 2.060 | **162** | 162 | -- | **2916.837** | 4320.873 | -- | -- | -- | 0.081 | -- | -- | 0.014 | -- | -- | 0.850 |
++++++++-| Doge Palace Venice | **0.801** | 0.911 | 1.163 | **0.218** | 0.290 | 0.342 | **6.432** | 10.966 | 3.620 | **241** | 241 | -- | 28768.725 | **21074.950** | -- | -- | -- | 0.211 | -- | -- | 0.029 | -- | -- | 1 |
++++++++-| Door Lund | **0.014** | 14.172 | 0.024 | **0.004** | 0.952 | 0.006 | **1.812** | 11.922 | 0.320 | **12** | 12 | -- | **4888.363** | 9188.593 | -- | -- | -- | 0.006 | -- | -- | 0.001 | -- | -- | 0.300 |
++++++++-| Drinking Fountain Somewhere In Zurich | **17.193** | 17.536 | 0.031 | 0.753 | **0.740** | 0.004 | **6.058** | 6.511 | 0.330 | **14** | 14 | -- | **883.837** | 2963.647 | -- | -- | -- | 0.007 | -- | -- | 0.002 | -- | -- | 0.310 |
++++++++-| East Indiaman Goteborg | 5.568 | **5.310** | 3.814 | 0.979 | **0.959** | 0.621 | **7.318** | 9.840 | 4.130 | **179** | 179 | -- | **3987.183** | 5241.430 | -- | -- | -- | 3.117 | -- | -- | 0.509 | -- | -- | 1.850 |
++++++++-| Ecole Superior De Guerre | 0.349 | **0.284** | 0.318 | 0.090 | **0.077** | 0.081 | **3.418** | 5.349 | 0.720 | **35** | 35 | -- | **2380.387** | 4633.833 | -- | -- | -- | 0.024 | -- | -- | 0.005 | -- | -- | 0.340 |
++++++++-| Eglise du dome | **0.351** | 45.801 | 0.808 | **0.088** | 6.451 | 0.205 | **3.799** | 7.964 | 0.910 | **85** | 85 | -- | 25099.450 | **23371.730** | -- | -- | -- | 0.037 | -- | -- | 0.010 | -- | -- | 0.270 |
++++++++-| Folke Filbyter | 85.831 | **81.288** | 74.596 | **0.129** | 0.130 | 0.125 | **19.275** | 23.156 | 10.370 | **40** | 40 | -- | **1581.897** | 3703.187 | -- | -- | -- | 70.157 | -- | -- | 0.118 | -- | -- | 4.290 |
++++++++-| Fort Channing Gate Singapore | **0.093** | 11.128 | 0.207 | **0.041** | 4.004 | 0.093 | **1.783** | 6.651 | 0.520 | **27** | 27 | -- | **3434.113** | 6172.087 | -- | -- | -- | 0.020 | -- | -- | 0.008 | -- | -- | 0.250 |
++++++++-| Golden Statue Somewhere In Hong Kong | **0.232** | 0.371 | 0.292 | **0.053** | 0.074 | 0.073 | **1.829** | 2.835 | 0.400 | **18** | 18 | -- | **4095.853** | 7762.437 | -- | -- | -- | 0.031 | -- | -- | 0.004 | -- | -- | 0.270 |
++++++++-| Gustav Vasa | **3.509** | 6.099 | 34.181 | **0.220** | 0.345 | 1.085 | **1.655** | 3.087 | 3.520 | **18** | 18 | -- | **709.867** | 2543.553 | -- | -- | -- | 32.266 | -- | -- | 1.145 | -- | -- | 3.150 |
++++++++-| GustavIIAdolf | **47.126** | 77.423 | 67.784 | **8.536** | 11.962 | 9.714 | **11.722** | 15.851 | 13.910 | **57** | 57 | -- | **1044.170** | 2892.580 | -- | -- | -- | 58.458 | -- | -- | 8.524 | -- | -- | 11.490 |
++++++++-| Jonas Ahlstromer | 48.740 | **42.246** | 50.190 | **9.823** | 10.300 | 10.888 | 11.136 | **8.782** | 10.820 | **40** | 40 | -- | **615.450** | 2328.313 | -- | -- | -- | 47.117 | -- | -- | 10.451 | -- | -- | 8.410 |
++++++++-| Kings College University Of Toronto | **9.025** | 21.523 | 0.989 | 2.044 | **1.905** | 0.235 | **3.565** | 5.525 | 0.900 | **77** | 77 | -- | **1078.980** | 2886.243 | -- | -- | -- | 0.085 | -- | -- | 0.017 | -- | -- | 0.340 |
++++++++-| Lund University Sphinx | **11.392** | 33.023 | 19.522 | **2.981** | 6.889 | 4.585 | **6.056** | 12.060 | 4.780 | **70** | 70 | -- | **2718.567** | 4654.610 | -- | -- | -- | 8.752 | -- | -- | 2.191 | -- | -- | 1.360 |
++++++++-| Nijo Castle Gate | **0.839** | 1.682 | 1.495 | **0.165** | 0.326 | 0.286 | **7.415** | 12.138 | 1.700 | **19** | 19 | -- | **1095.177** | 3306.697 | -- | -- | -- | 0.069 | -- | -- | 0.012 | -- | -- | 0.730 |
++++++++-| Pantheon Paris | -- | 19.345 | 0.192 | -- | 3.065 | 0.050 | -- | 13.562 | 1.470 | -- | 179 | -- | -- | 5497.377 | -- | -- | -- | 0.040 | -- | -- | 0.005 | -- | -- | 0.490 |
++++++++-| Park Gate Clermont Ferrand | 21.388 | **19.302** | 0.391 | 10.192 | **9.397** | 0.125 | **7.712** | 7.852 | 0.570 | **34** | 34 | -- | **1596.940** | 3597.403 | -- | -- | -- | 0.049 | -- | -- | 0.022 | -- | -- | 0.350 |
++++++++-| Plaza De Armas Santiago | **0.775** | 3.327 | 6.782 | **0.337** | 1.435 | 2.944 | **6.009** | 9.463 | 7.400 | **240** | 240 | -- | **6372.683** | 6535.537 | -- | -- | -- | 2.556 | -- | -- | 1.383 | -- | -- | 4.900 |
++++++++-| Porta San Donato Bologna | **0.592** | 21.006 | 2.153 | **0.107** | 3.541 | 0.388 | **5.603** | 13.244 | 2.280 | **141** | 141 | -- | **3754.163** | 5118.030 | -- | -- | -- | 0.095 | -- | -- | 0.046 | -- | -- | 0.750 |
++++++++-| Round Church Cambridge | **2.129** | 2.490 | 2.451 | **0.920** | 1.002 | 1.003 | **6.097** | 8.546 | 2.660 | **92** | 92 | -- | **9776.630** | 12061.350 | -- | -- | -- | 1.107 | -- | -- | 0.582 | -- | -- | 1.540 |
++++++++-| Skansen Kronan Gothenburg | **0.338** | 0.375 | 0.736 | **0.116** | 0.124 | 0.226 | **3.048** | 4.799 | 1.240 | **131** | 131 | -- | 17163.940 | **16235.090** | -- | -- | -- | 0.026 | -- | -- | 0.008 | -- | -- | 0.670 |
++++++++-| Smolny Cathedral St Petersburg | **21.806** | 22.143 | 0.554 | 2.111 | **2.099** | 0.051 | **11.228** | 12.526 | 1.660 | **131** | 131 | -- | **10362.595** | 10475.740 | -- | -- | -- | 0.033 | -- | -- | 0.006 | -- | -- | 0.810 |
++++++++-| Sri Mariamman Singapore | **1.219** | 6.479 | 2.302 | **0.382** | 2.188 | 0.683 | **10.088** | 17.854 | 4.130 | **222** | 222 | -- | **5985.540** | 6485.667 | -- | -- | -- | 0.077 | -- | -- | 0.023 | -- | -- | 0.910 |
++++++++-| Sri Thendayuthapani Singapore | **0.719** | 47.075 | 46.269 | **0.179** | 3.855 | 3.812 | **7.124** | 14.133 | 23.370 | **98** | 98 | -- | **13362.090** | 13808.440 | -- | -- | -- | 44.170 | -- | -- | 2.870 | -- | -- | 8.440 |
++++++++-| Sri Veeramakaliamman Singapore | **2.334** | 29.405 | 2.559 | **0.543** | 3.868 | 0.597 | **11.965** | 28.951 | 3.470 | **157** | 157 | -- | **12392.180** | 12544.690 | -- | -- | -- | 0.175 | -- | -- | 0.040 | -- | -- | 0.730 |
++++++++-| Statue Of Liberty | 52.358 | **39.859** | 46.887 | 28.503 | **18.503** | 20.012 | 235036.825 | **41.028** | 26.160 | **134** | 134 | -- | **99.900** | 12924 | -- | -- | -- | 9.091 | -- | -- | 4.122 | -- | -- | 6.970 |
++++++++-| The Pumpkin | **25.731** | 26.571 | 94.672 | **5.552** | 5.665 | 14.890 | **26.876** | 35.367 | 33.410 | **196** | 196 | -- | **4327.273** | 5790.597 | -- | -- | -- | 98.862 | -- | -- | 14.952 | -- | -- | 24.850 |
++++++++-| Thian Hook Keng Temple Singapore | **0.927** | 1.201 | 0.832 | **0.093** | 0.121 | 0.082 | **14.915** | 20.789 | 2.750 | **138** | 138 | -- | **3298.537** | 4612.373 | -- | -- | -- | 0.081 | -- | -- | 0.008 | -- | -- | 1.130 |
++++++++-| Tsar Nikolai I | **42.803** | 78.413 | 48.499 | **8.927** | 15.394 | 9.467 | **11.124** | 18.431 | 9.790 | **98** | 98 | -- | **4928.430** | 6646.845 | -- | -- | -- | 36.280 | -- | -- | 7.836 | -- | -- | 6.530 |
++++++++-| Urban II | 59.030 | -- | 47.490 | 11.008 | -- | 9.467 | 17.601 | -- | 9.380 | 96 | -- | -- | 8889.690 | -- | -- | -- | -- | 48.214 | -- | -- | 9.586 | -- | -- | 6.920 |
++++++++-| Vercingetorix | **82.952** | 86.088 | 69.328 | **10.195** | 10.890 | 8.788 | **7.257** | 8.681 | 5.080 | **69** | 69 | -- | **1223.970** | 3105.607 | -- | -- | -- | 17.706 | -- | -- | 3.104 | -- | -- | 1.500 |
++++++++-| Yueh Hai Ching Temple Singapore | 0.627 | -- | 0.720 | 0.084 | -- | 0.098 | 5.562 | -- | 0.940 | 43 | -- | -- | 1749.280 | -- | -- | -- | -- | 0.043 | -- | -- | 0.014 | -- | -- | 0.650 |
++++++++-| **Mean** | **16.146** | 23.697 | 18.023 | **3.121** | 4.014 | 2.912 | 6920.602 | **13.088** | 5.673 | 100.559 | **104.818** | -- | **5819.728** | 7358.187 | -- | -- | -- | 13.695 | -- | -- | 1.937 | -- | -- | 2.992 |
+++++++++| Alcatraz Courtyard | **0.362** | 0.551 | 0.619 | **0.093** | 0.141 | 0.160 | **3.435** | 5.529 | 1.640 | **133** | 133 | -- | **4826.713** | 6556.500 | -- | -- | -- | 0.049 | -- | -- | 0.015 | -- | -- | 0.810 |
+++++++++| Alcatraz Water Tower | **0.666** | 1.721 | 0.933 | **0.370** | 0.926 | 0.518 | **4.763** | 7.564 | 2.130 | **172** | 172 | -- | **2461.337** | 3796.130 | -- | -- | -- | 0.230 | -- | -- | 0.116 | -- | -- | 0.550 |
+++++++++| Buddah Tooth Relic Temple Singapore | **1.131** | 1.757 | 1.030 | **0.280** | 0.516 | 0.233 | **9.964** | 13.678 | 2.060 | **162** | 162 | -- | **2916.837** | 4298.980 | -- | -- | -- | 0.081 | -- | -- | 0.014 | -- | -- | 0.850 |
+++++++++| Doge Palace Venice | 0.801 | -- | 1.163 | 0.218 | -- | 0.342 | 6.432 | -- | 3.620 | 241 | -- | -- | 28768.725 | -- | -- | -- | -- | 0.211 | -- | -- | 0.029 | -- | -- | 1 |
+++++++++| Door Lund | 0.014 | -- | 0.024 | 0.004 | -- | 0.006 | 1.812 | -- | 0.320 | 12 | -- | -- | 4888.363 | -- | -- | -- | -- | 0.006 | -- | -- | 0.001 | -- | -- | 0.300 |
+++++++++| Drinking Fountain Somewhere In Zurich | **17.193** | 25.946 | 0.031 | **0.753** | 1.118 | 0.004 | **6.058** | 8.472 | 0.330 | **14** | 14 | -- | **883.837** | 5817.370 | -- | -- | -- | 0.007 | -- | -- | 0.002 | -- | -- | 0.310 |
+++++++++| East Indiaman Goteborg | 5.568 | **5.366** | 3.814 | 0.979 | **0.955** | 0.621 | **7.318** | 9.655 | 4.130 | **179** | 179 | -- | **3987.183** | 5203.600 | -- | -- | -- | 3.117 | -- | -- | 0.509 | -- | -- | 1.850 |
+++++++++| Ecole Superior De Guerre | 0.349 | -- | 0.318 | 0.090 | -- | 0.081 | 3.418 | -- | 0.720 | 35 | -- | -- | 2380.387 | -- | -- | -- | -- | 0.024 | -- | -- | 0.005 | -- | -- | 0.340 |
+++++++++| Eglise du dome | 0.351 | -- | 0.808 | 0.088 | -- | 0.205 | 3.799 | -- | 0.910 | 85 | -- | -- | 25099.450 | -- | -- | -- | -- | 0.037 | -- | -- | 0.010 | -- | -- | 0.270 |
+++++++++| Folke Filbyter | 85.831 | **84.713** | 74.596 | **0.129** | 0.131 | 0.125 | **19.275** | 30.778 | 10.370 | **40** | 40 | -- | **1581.897** | 3745.590 | -- | -- | -- | 70.157 | -- | -- | 0.118 | -- | -- | 4.290 |
+++++++++| Fort Channing Gate Singapore | 0.093 | -- | 0.207 | 0.041 | -- | 0.093 | 1.783 | -- | 0.520 | 27 | -- | -- | 3434.113 | -- | -- | -- | -- | 0.020 | -- | -- | 0.008 | -- | -- | 0.250 |
+++++++++| Golden Statue Somewhere In Hong Kong | 0.232 | -- | 0.292 | 0.053 | -- | 0.073 | 1.829 | -- | 0.400 | 18 | -- | -- | 4095.853 | -- | -- | -- | -- | 0.031 | -- | -- | 0.004 | -- | -- | 0.270 |
+++++++++| Gustav Vasa | **3.509** | 36.646 | 34.181 | **0.220** | 1.007 | 1.085 | **1.655** | 4.980 | 3.520 | **18** | 18 | -- | **709.867** | 2562.940 | -- | -- | -- | 32.266 | -- | -- | 1.145 | -- | -- | 3.150 |
+++++++++| GustavIIAdolf | **47.126** | 87.231 | 67.784 | **8.536** | 13.026 | 9.714 | **11.722** | 17.359 | 13.910 | **57** | 57 | -- | **1044.170** | 2984.325 | -- | -- | -- | 58.458 | -- | -- | 8.524 | -- | -- | 11.490 |
+++++++++| Jonas Ahlstromer | 48.740 | -- | 50.190 | 9.823 | -- | 10.888 | 11.136 | -- | 10.820 | 40 | -- | -- | 615.450 | -- | -- | -- | -- | 47.117 | -- | -- | 10.451 | -- | -- | 8.410 |
+++++++++| Kings College University Of Toronto | 9.025 | -- | 0.989 | 2.044 | -- | 0.235 | 3.565 | -- | 0.900 | 77 | -- | -- | 1078.980 | -- | -- | -- | -- | 0.085 | -- | -- | 0.017 | -- | -- | 0.340 |
+++++++++| Lund University Sphinx | 11.392 | -- | 19.522 | 2.981 | -- | 4.585 | 6.056 | -- | 4.780 | 70 | -- | -- | 2718.567 | -- | -- | -- | -- | 8.752 | -- | -- | 2.191 | -- | -- | 1.360 |
+++++++++| Nijo Castle Gate | 0.839 | -- | 1.495 | 0.165 | -- | 0.286 | 7.415 | -- | 1.700 | 19 | -- | -- | 1095.177 | -- | -- | -- | -- | 0.069 | -- | -- | 0.012 | -- | -- | 0.730 |
+++++++++| Pantheon Paris | -- | 0.545 | 0.192 | -- | 0.070 | 0.050 | -- | 7.791 | 1.470 | -- | 179 | -- | -- | 5594.600 | -- | -- | -- | 0.040 | -- | -- | 0.005 | -- | -- | 0.490 |
+++++++++| Park Gate Clermont Ferrand | 21.388 | -- | 0.391 | 10.192 | -- | 0.125 | 7.712 | -- | 0.570 | 34 | -- | -- | 1596.940 | -- | -- | -- | -- | 0.049 | -- | -- | 0.022 | -- | -- | 0.350 |
+++++++++| Plaza De Armas Santiago | 0.775 | -- | 6.782 | 0.337 | -- | 2.944 | 6.009 | -- | 7.400 | 240 | -- | -- | 6372.683 | -- | -- | -- | -- | 2.556 | -- | -- | 1.383 | -- | -- | 4.900 |
+++++++++| Porta San Donato Bologna | 0.592 | -- | 2.153 | 0.107 | -- | 0.388 | 5.603 | -- | 2.280 | 141 | -- | -- | 3754.163 | -- | -- | -- | -- | 0.095 | -- | -- | 0.046 | -- | -- | 0.750 |
+++++++++| Round Church Cambridge | 2.137 | -- | 2.451 | 0.927 | -- | 1.003 | 6.103 | -- | 2.660 | 92 | -- | -- | 9779.270 | -- | -- | -- | -- | 1.107 | -- | -- | 0.582 | -- | -- | 1.540 |
+++++++++| Skansen Kronan Gothenburg | 0.301 | -- | 0.736 | 0.102 | -- | 0.226 | 2.906 | -- | 1.240 | 131 | -- | -- | 17153.677 | -- | -- | -- | -- | 0.026 | -- | -- | 0.008 | -- | -- | 0.670 |
+++++++++| Smolny Cathedral St Petersburg | 21.775 | -- | 0.554 | 2.111 | -- | 0.051 | 11.239 | -- | 1.660 | 131 | -- | -- | 10362.787 | -- | -- | -- | -- | 0.033 | -- | -- | 0.006 | -- | -- | 0.810 |
+++++++++| Some Cathedral In Barcelona | 0.651 | -- | 0.880 | 0.233 | -- | 0.315 | 6.271 | -- | 2.870 | 177 | -- | -- | 5001.973 | -- | -- | -- | -- | 0.026 | -- | -- | 0.011 | -- | -- | 0.890 |
+++++++++| Sri Mariamman Singapore | 1.219 | -- | 2.302 | 0.382 | -- | 0.683 | 10.088 | -- | 4.130 | 222 | -- | -- | 5985.540 | -- | -- | -- | -- | 0.077 | -- | -- | 0.023 | -- | -- | 0.910 |
+++++++++| Sri Thendayuthapani Singapore | 0.593 | -- | 46.269 | 0.150 | -- | 3.812 | 7.016 | -- | 23.370 | 98 | -- | -- | 13338.167 | -- | -- | -- | -- | 44.170 | -- | -- | 2.870 | -- | -- | 8.440 |
+++++++++| Sri Veeramakaliamman Singapore | 2.404 | -- | 2.559 | 0.559 | -- | 0.597 | 12.234 | -- | 3.470 | 157 | -- | -- | 12377.927 | -- | -- | -- | -- | 0.175 | -- | -- | 0.040 | -- | -- | 0.730 |
+++++++++| Statue Of Liberty | 52.358 | -- | 46.887 | 28.503 | -- | 20.012 | 235036.825 | -- | 26.160 | 134 | -- | -- | 99.900 | -- | -- | -- | -- | 9.091 | -- | -- | 4.122 | -- | -- | 6.970 |
+++++++++| The Pumpkin | 25.731 | -- | 94.672 | 5.552 | -- | 14.890 | 26.876 | -- | 33.410 | 196 | -- | -- | 4327.273 | -- | -- | -- | -- | 98.862 | -- | -- | 14.952 | -- | -- | 24.850 |
+++++++++| Thian Hook Keng Temple Singapore | 0.927 | -- | 0.832 | 0.093 | -- | 0.082 | 14.915 | -- | 2.750 | 138 | -- | -- | 3298.537 | -- | -- | -- | -- | 0.081 | -- | -- | 0.008 | -- | -- | 1.130 |
+++++++++| Tsar Nikolai I | 42.166 | -- | 48.499 | 8.732 | -- | 9.467 | 11.210 | -- | 9.790 | 98 | -- | -- | 4929.200 | -- | -- | -- | -- | 36.280 | -- | -- | 7.836 | -- | -- | 6.530 |
+++++++++| Urban II | 58.919 | -- | 47.490 | 11.006 | -- | 9.467 | 17.538 | -- | 9.380 | 96 | -- | -- | 8889.617 | -- | -- | -- | -- | 48.214 | -- | -- | 9.586 | -- | -- | 6.920 |
+++++++++| Vercingetorix | 82.952 | -- | 69.328 | 10.195 | -- | 8.788 | 7.257 | -- | 5.080 | 69 | -- | -- | 1223.970 | -- | -- | -- | -- | 17.706 | -- | -- | 3.104 | -- | -- | 1.500 |
+++++++++| Yueh Hai Ching Temple Singapore | 0.544 | -- | 0.720 | 0.075 | -- | 0.098 | 5.596 | -- | 0.940 | 43 | -- | -- | 1748.667 | -- | -- | -- | -- | 0.043 | -- | -- | 0.014 | -- | -- | 0.650 |
+++++++++| **Mean** | **15.676** | 27.164 | 17.547 | 3.032 | **1.988** | 2.840 | 6723.052 | **11.756** | 5.595 | 102.743 | **106** | -- | 5795.063 | **4506.671** | -- | -- | -- | 13.315 | -- | -- | 1.883 | -- | -- | 2.933 |
++++++++ 
++++++++ _Seeds per cell: [0, 1, 2]._
++++++++ 
++++++++diff --git a/code/results/single_scene/summary_table.tex b/code/results/single_scene/summary_table.tex
++++++++index 258e3f2..fc3cdbc 100644
++++++++--- a/code/results/single_scene/summary_table.tex
+++++++++++ b/code/results/single_scene/summary_table.tex
++++++++@@ -7,45 +7,46 @@
++++++++ \begin{tabular}{lcccccccccccccccccccccccc}
++++++++ \toprule
++++++++  & \multicolumn{3}{c}{Rot (deg)} & \multicolumn{3}{c}{Trans} & \multicolumn{3}{c}{Reproj (px)} & \multicolumn{3}{c}{Nr} & \multicolumn{3}{c}{Time (s)} & \multicolumn{3}{c}{Rot-BA (deg)} & \multicolumn{3}{c}{Trans-BA} & \multicolumn{3}{c}{Reproj-BA (px)} \\
++++++++-Scene & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) \\
+++++++++Scene & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) \\
++++++++ \midrule
++++++++-Alcatraz Courtyard & \textbf{0.362} & 0.616 & 0.619 & \textbf{0.093} & 0.161 & 0.160 & \textbf{3.435} & 5.796 & 1.640 & \textbf{133} & 133 & -- & \textbf{4826.713} & 6481.757 & -- & -- & -- & 0.049 & -- & -- & 0.015 & -- & -- & 0.810 \\
++++++++-Alcatraz Water Tower & \textbf{0.666} & 7.413 & 0.933 & \textbf{0.370} & 3.437 & 0.518 & \textbf{4.763} & 8.607 & 2.130 & \textbf{172} & 172 & -- & \textbf{2461.337} & 3854.907 & -- & -- & -- & 0.230 & -- & -- & 0.116 & -- & -- & 0.550 \\
++++++++-Buddah Tooth Relic Temple Singapore & \textbf{1.131} & 12.080 & 1.030 & \textbf{0.280} & 2.308 & 0.233 & \textbf{9.964} & 19.634 & 2.060 & \textbf{162} & 162 & -- & \textbf{2916.837} & 4320.873 & -- & -- & -- & 0.081 & -- & -- & 0.014 & -- & -- & 0.850 \\
++++++++-Doge Palace Venice & \textbf{0.801} & 0.911 & 1.163 & \textbf{0.218} & 0.290 & 0.342 & \textbf{6.432} & 10.966 & 3.620 & \textbf{241} & 241 & -- & 28768.725 & \textbf{21074.950} & -- & -- & -- & 0.211 & -- & -- & 0.029 & -- & -- & 1 \\
++++++++-Door Lund & \textbf{0.014} & 14.172 & 0.024 & \textbf{0.004} & 0.952 & 0.006 & \textbf{1.812} & 11.922 & 0.320 & \textbf{12} & 12 & -- & \textbf{4888.363} & 9188.593 & -- & -- & -- & 0.006 & -- & -- & 0.001 & -- & -- & 0.300 \\
++++++++-Drinking Fountain Somewhere In Zurich & \textbf{17.193} & 17.536 & 0.031 & 0.753 & \textbf{0.740} & 0.004 & \textbf{6.058} & 6.511 & 0.330 & \textbf{14} & 14 & -- & \textbf{883.837} & 2963.647 & -- & -- & -- & 0.007 & -- & -- & 0.002 & -- & -- & 0.310 \\
++++++++-East Indiaman Goteborg & 5.568 & \textbf{5.310} & 3.814 & 0.979 & \textbf{0.959} & 0.621 & \textbf{7.318} & 9.840 & 4.130 & \textbf{179} & 179 & -- & \textbf{3987.183} & 5241.430 & -- & -- & -- & 3.117 & -- & -- & 0.509 & -- & -- & 1.850 \\
++++++++-Ecole Superior De Guerre & 0.349 & \textbf{0.284} & 0.318 & 0.090 & \textbf{0.077} & 0.081 & \textbf{3.418} & 5.349 & 0.720 & \textbf{35} & 35 & -- & \textbf{2380.387} & 4633.833 & -- & -- & -- & 0.024 & -- & -- & 0.005 & -- & -- & 0.340 \\
++++++++-Eglise du dome & \textbf{0.351} & 45.801 & 0.808 & \textbf{0.088} & 6.451 & 0.205 & \textbf{3.799} & 7.964 & 0.910 & \textbf{85} & 85 & -- & 25099.450 & \textbf{23371.730} & -- & -- & -- & 0.037 & -- & -- & 0.010 & -- & -- & 0.270 \\
++++++++-Folke Filbyter & 85.831 & \textbf{81.288} & 74.596 & \textbf{0.129} & 0.130 & 0.125 & \textbf{19.275} & 23.156 & 10.370 & \textbf{40} & 40 & -- & \textbf{1581.897} & 3703.187 & -- & -- & -- & 70.157 & -- & -- & 0.118 & -- & -- & 4.290 \\
++++++++-Fort Channing Gate Singapore & \textbf{0.093} & 11.128 & 0.207 & \textbf{0.041} & 4.004 & 0.093 & \textbf{1.783} & 6.651 & 0.520 & \textbf{27} & 27 & -- & \textbf{3434.113} & 6172.087 & -- & -- & -- & 0.020 & -- & -- & 0.008 & -- & -- & 0.250 \\
++++++++-Golden Statue Somewhere In Hong Kong & \textbf{0.232} & 0.371 & 0.292 & \textbf{0.053} & 0.074 & 0.073 & \textbf{1.829} & 2.835 & 0.400 & \textbf{18} & 18 & -- & \textbf{4095.853} & 7762.437 & -- & -- & -- & 0.031 & -- & -- & 0.004 & -- & -- & 0.270 \\
++++++++-Gustav Vasa & \textbf{3.509} & 6.099 & 34.181 & \textbf{0.220} & 0.345 & 1.085 & \textbf{1.655} & 3.087 & 3.520 & \textbf{18} & 18 & -- & \textbf{709.867} & 2543.553 & -- & -- & -- & 32.266 & -- & -- & 1.145 & -- & -- & 3.150 \\
++++++++-GustavIIAdolf & \textbf{47.126} & 77.423 & 67.784 & \textbf{8.536} & 11.962 & 9.714 & \textbf{11.722} & 15.851 & 13.910 & \textbf{57} & 57 & -- & \textbf{1044.170} & 2892.580 & -- & -- & -- & 58.458 & -- & -- & 8.524 & -- & -- & 11.490 \\
++++++++-Jonas Ahlstromer & 48.740 & \textbf{42.246} & 50.190 & \textbf{9.823} & 10.300 & 10.888 & 11.136 & \textbf{8.782} & 10.820 & \textbf{40} & 40 & -- & \textbf{615.450} & 2328.313 & -- & -- & -- & 47.117 & -- & -- & 10.451 & -- & -- & 8.410 \\
++++++++-Kings College University Of Toronto & \textbf{9.025} & 21.523 & 0.989 & 2.044 & \textbf{1.905} & 0.235 & \textbf{3.565} & 5.525 & 0.900 & \textbf{77} & 77 & -- & \textbf{1078.980} & 2886.243 & -- & -- & -- & 0.085 & -- & -- & 0.017 & -- & -- & 0.340 \\
++++++++-Lund University Sphinx & \textbf{11.392} & 33.023 & 19.522 & \textbf{2.981} & 6.889 & 4.585 & \textbf{6.056} & 12.060 & 4.780 & \textbf{70} & 70 & -- & \textbf{2718.567} & 4654.610 & -- & -- & -- & 8.752 & -- & -- & 2.191 & -- & -- & 1.360 \\
++++++++-Nijo Castle Gate & \textbf{0.839} & 1.682 & 1.495 & \textbf{0.165} & 0.326 & 0.286 & \textbf{7.415} & 12.138 & 1.700 & \textbf{19} & 19 & -- & \textbf{1095.177} & 3306.697 & -- & -- & -- & 0.069 & -- & -- & 0.012 & -- & -- & 0.730 \\
++++++++-Pantheon Paris & -- & 19.345 & 0.192 & -- & 3.065 & 0.050 & -- & 13.562 & 1.470 & -- & 179 & -- & -- & 5497.377 & -- & -- & -- & 0.040 & -- & -- & 0.005 & -- & -- & 0.490 \\
++++++++-Park Gate Clermont Ferrand & 21.388 & \textbf{19.302} & 0.391 & 10.192 & \textbf{9.397} & 0.125 & \textbf{7.712} & 7.852 & 0.570 & \textbf{34} & 34 & -- & \textbf{1596.940} & 3597.403 & -- & -- & -- & 0.049 & -- & -- & 0.022 & -- & -- & 0.350 \\
++++++++-Plaza De Armas Santiago & \textbf{0.775} & 3.327 & 6.782 & \textbf{0.337} & 1.435 & 2.944 & \textbf{6.009} & 9.463 & 7.400 & \textbf{240} & 240 & -- & \textbf{6372.683} & 6535.537 & -- & -- & -- & 2.556 & -- & -- & 1.383 & -- & -- & 4.900 \\
++++++++-Porta San Donato Bologna & \textbf{0.592} & 21.006 & 2.153 & \textbf{0.107} & 3.541 & 0.388 & \textbf{5.603} & 13.244 & 2.280 & \textbf{141} & 141 & -- & \textbf{3754.163} & 5118.030 & -- & -- & -- & 0.095 & -- & -- & 0.046 & -- & -- & 0.750 \\
++++++++-Round Church Cambridge & \textbf{2.129} & 2.490 & 2.451 & \textbf{0.920} & 1.002 & 1.003 & \textbf{6.097} & 8.546 & 2.660 & \textbf{92} & 92 & -- & \textbf{9776.630} & 12061.350 & -- & -- & -- & 1.107 & -- & -- & 0.582 & -- & -- & 1.540 \\
++++++++-Skansen Kronan Gothenburg & \textbf{0.338} & 0.375 & 0.736 & \textbf{0.116} & 0.124 & 0.226 & \textbf{3.048} & 4.799 & 1.240 & \textbf{131} & 131 & -- & 17163.940 & \textbf{16235.090} & -- & -- & -- & 0.026 & -- & -- & 0.008 & -- & -- & 0.670 \\
++++++++-Smolny Cathedral St Petersburg & \textbf{21.806} & 22.143 & 0.554 & 2.111 & \textbf{2.099} & 0.051 & \textbf{11.228} & 12.526 & 1.660 & \textbf{131} & 131 & -- & \textbf{10362.595} & 10475.740 & -- & -- & -- & 0.033 & -- & -- & 0.006 & -- & -- & 0.810 \\
++++++++-Sri Mariamman Singapore & \textbf{1.219} & 6.479 & 2.302 & \textbf{0.382} & 2.188 & 0.683 & \textbf{10.088} & 17.854 & 4.130 & \textbf{222} & 222 & -- & \textbf{5985.540} & 6485.667 & -- & -- & -- & 0.077 & -- & -- & 0.023 & -- & -- & 0.910 \\
++++++++-Sri Thendayuthapani Singapore & \textbf{0.719} & 47.075 & 46.269 & \textbf{0.179} & 3.855 & 3.812 & \textbf{7.124} & 14.133 & 23.370 & \textbf{98} & 98 & -- & \textbf{13362.090} & 13808.440 & -- & -- & -- & 44.170 & -- & -- & 2.870 & -- & -- & 8.440 \\
++++++++-Sri Veeramakaliamman Singapore & \textbf{2.334} & 29.405 & 2.559 & \textbf{0.543} & 3.868 & 0.597 & \textbf{11.965} & 28.951 & 3.470 & \textbf{157} & 157 & -- & \textbf{12392.180} & 12544.690 & -- & -- & -- & 0.175 & -- & -- & 0.040 & -- & -- & 0.730 \\
++++++++-Statue Of Liberty & 52.358 & \textbf{39.859} & 46.887 & 28.503 & \textbf{18.503} & 20.012 & 235036.825 & \textbf{41.028} & 26.160 & \textbf{134} & 134 & -- & \textbf{99.900} & 12924 & -- & -- & -- & 9.091 & -- & -- & 4.122 & -- & -- & 6.970 \\
++++++++-The Pumpkin & \textbf{25.731} & 26.571 & 94.672 & \textbf{5.552} & 5.665 & 14.890 & \textbf{26.876} & 35.367 & 33.410 & \textbf{196} & 196 & -- & \textbf{4327.273} & 5790.597 & -- & -- & -- & 98.862 & -- & -- & 14.952 & -- & -- & 24.850 \\
++++++++-Thian Hook Keng Temple Singapore & \textbf{0.927} & 1.201 & 0.832 & \textbf{0.093} & 0.121 & 0.082 & \textbf{14.915} & 20.789 & 2.750 & \textbf{138} & 138 & -- & \textbf{3298.537} & 4612.373 & -- & -- & -- & 0.081 & -- & -- & 0.008 & -- & -- & 1.130 \\
++++++++-Tsar Nikolai I & \textbf{42.803} & 78.413 & 48.499 & \textbf{8.927} & 15.394 & 9.467 & \textbf{11.124} & 18.431 & 9.790 & \textbf{98} & 98 & -- & \textbf{4928.430} & 6646.845 & -- & -- & -- & 36.280 & -- & -- & 7.836 & -- & -- & 6.530 \\
++++++++-Urban II & 59.030 & -- & 47.490 & 11.008 & -- & 9.467 & 17.601 & -- & 9.380 & 96 & -- & -- & 8889.690 & -- & -- & -- & -- & 48.214 & -- & -- & 9.586 & -- & -- & 6.920 \\
++++++++-Vercingetorix & \textbf{82.952} & 86.088 & 69.328 & \textbf{10.195} & 10.890 & 8.788 & \textbf{7.257} & 8.681 & 5.080 & \textbf{69} & 69 & -- & \textbf{1223.970} & 3105.607 & -- & -- & -- & 17.706 & -- & -- & 3.104 & -- & -- & 1.500 \\
++++++++-Yueh Hai Ching Temple Singapore & 0.627 & -- & 0.720 & 0.084 & -- & 0.098 & 5.562 & -- & 0.940 & 43 & -- & -- & 1749.280 & -- & -- & -- & -- & 0.043 & -- & -- & 0.014 & -- & -- & 0.650 \\
+++++++++Alcatraz Courtyard & \textbf{0.362} & 0.551 & 0.619 & \textbf{0.093} & 0.141 & 0.160 & \textbf{3.435} & 5.529 & 1.640 & \textbf{133} & 133 & -- & \textbf{4826.713} & 6556.500 & -- & -- & -- & 0.049 & -- & -- & 0.015 & -- & -- & 0.810 \\
+++++++++Alcatraz Water Tower & \textbf{0.666} & 1.721 & 0.933 & \textbf{0.370} & 0.926 & 0.518 & \textbf{4.763} & 7.564 & 2.130 & \textbf{172} & 172 & -- & \textbf{2461.337} & 3796.130 & -- & -- & -- & 0.230 & -- & -- & 0.116 & -- & -- & 0.550 \\
+++++++++Buddah Tooth Relic Temple Singapore & \textbf{1.131} & 1.757 & 1.030 & \textbf{0.280} & 0.516 & 0.233 & \textbf{9.964} & 13.678 & 2.060 & \textbf{162} & 162 & -- & \textbf{2916.837} & 4298.980 & -- & -- & -- & 0.081 & -- & -- & 0.014 & -- & -- & 0.850 \\
+++++++++Doge Palace Venice & 0.801 & -- & 1.163 & 0.218 & -- & 0.342 & 6.432 & -- & 3.620 & 241 & -- & -- & 28768.725 & -- & -- & -- & -- & 0.211 & -- & -- & 0.029 & -- & -- & 1 \\
+++++++++Door Lund & 0.014 & -- & 0.024 & 0.004 & -- & 0.006 & 1.812 & -- & 0.320 & 12 & -- & -- & 4888.363 & -- & -- & -- & -- & 0.006 & -- & -- & 0.001 & -- & -- & 0.300 \\
+++++++++Drinking Fountain Somewhere In Zurich & \textbf{17.193} & 25.946 & 0.031 & \textbf{0.753} & 1.118 & 0.004 & \textbf{6.058} & 8.472 & 0.330 & \textbf{14} & 14 & -- & \textbf{883.837} & 5817.370 & -- & -- & -- & 0.007 & -- & -- & 0.002 & -- & -- & 0.310 \\
+++++++++East Indiaman Goteborg & 5.568 & \textbf{5.366} & 3.814 & 0.979 & \textbf{0.955} & 0.621 & \textbf{7.318} & 9.655 & 4.130 & \textbf{179} & 179 & -- & \textbf{3987.183} & 5203.600 & -- & -- & -- & 3.117 & -- & -- & 0.509 & -- & -- & 1.850 \\
+++++++++Ecole Superior De Guerre & 0.349 & -- & 0.318 & 0.090 & -- & 0.081 & 3.418 & -- & 0.720 & 35 & -- & -- & 2380.387 & -- & -- & -- & -- & 0.024 & -- & -- & 0.005 & -- & -- & 0.340 \\
+++++++++Eglise du dome & 0.351 & -- & 0.808 & 0.088 & -- & 0.205 & 3.799 & -- & 0.910 & 85 & -- & -- & 25099.450 & -- & -- & -- & -- & 0.037 & -- & -- & 0.010 & -- & -- & 0.270 \\
+++++++++Folke Filbyter & 85.831 & \textbf{84.713} & 74.596 & \textbf{0.129} & 0.131 & 0.125 & \textbf{19.275} & 30.778 & 10.370 & \textbf{40} & 40 & -- & \textbf{1581.897} & 3745.590 & -- & -- & -- & 70.157 & -- & -- & 0.118 & -- & -- & 4.290 \\
+++++++++Fort Channing Gate Singapore & 0.093 & -- & 0.207 & 0.041 & -- & 0.093 & 1.783 & -- & 0.520 & 27 & -- & -- & 3434.113 & -- & -- & -- & -- & 0.020 & -- & -- & 0.008 & -- & -- & 0.250 \\
+++++++++Golden Statue Somewhere In Hong Kong & 0.232 & -- & 0.292 & 0.053 & -- & 0.073 & 1.829 & -- & 0.400 & 18 & -- & -- & 4095.853 & -- & -- & -- & -- & 0.031 & -- & -- & 0.004 & -- & -- & 0.270 \\
+++++++++Gustav Vasa & \textbf{3.509} & 36.646 & 34.181 & \textbf{0.220} & 1.007 & 1.085 & \textbf{1.655} & 4.980 & 3.520 & \textbf{18} & 18 & -- & \textbf{709.867} & 2562.940 & -- & -- & -- & 32.266 & -- & -- & 1.145 & -- & -- & 3.150 \\
+++++++++GustavIIAdolf & \textbf{47.126} & 87.231 & 67.784 & \textbf{8.536} & 13.026 & 9.714 & \textbf{11.722} & 17.359 & 13.910 & \textbf{57} & 57 & -- & \textbf{1044.170} & 2984.325 & -- & -- & -- & 58.458 & -- & -- & 8.524 & -- & -- & 11.490 \\
+++++++++Jonas Ahlstromer & 48.740 & -- & 50.190 & 9.823 & -- & 10.888 & 11.136 & -- & 10.820 & 40 & -- & -- & 615.450 & -- & -- & -- & -- & 47.117 & -- & -- & 10.451 & -- & -- & 8.410 \\
+++++++++Kings College University Of Toronto & 9.025 & -- & 0.989 & 2.044 & -- & 0.235 & 3.565 & -- & 0.900 & 77 & -- & -- & 1078.980 & -- & -- & -- & -- & 0.085 & -- & -- & 0.017 & -- & -- & 0.340 \\
+++++++++Lund University Sphinx & 11.392 & -- & 19.522 & 2.981 & -- & 4.585 & 6.056 & -- & 4.780 & 70 & -- & -- & 2718.567 & -- & -- & -- & -- & 8.752 & -- & -- & 2.191 & -- & -- & 1.360 \\
+++++++++Nijo Castle Gate & 0.839 & -- & 1.495 & 0.165 & -- & 0.286 & 7.415 & -- & 1.700 & 19 & -- & -- & 1095.177 & -- & -- & -- & -- & 0.069 & -- & -- & 0.012 & -- & -- & 0.730 \\
+++++++++Pantheon Paris & -- & 0.545 & 0.192 & -- & 0.070 & 0.050 & -- & 7.791 & 1.470 & -- & 179 & -- & -- & 5594.600 & -- & -- & -- & 0.040 & -- & -- & 0.005 & -- & -- & 0.490 \\
+++++++++Park Gate Clermont Ferrand & 21.388 & -- & 0.391 & 10.192 & -- & 0.125 & 7.712 & -- & 0.570 & 34 & -- & -- & 1596.940 & -- & -- & -- & -- & 0.049 & -- & -- & 0.022 & -- & -- & 0.350 \\
+++++++++Plaza De Armas Santiago & 0.775 & -- & 6.782 & 0.337 & -- & 2.944 & 6.009 & -- & 7.400 & 240 & -- & -- & 6372.683 & -- & -- & -- & -- & 2.556 & -- & -- & 1.383 & -- & -- & 4.900 \\
+++++++++Porta San Donato Bologna & 0.592 & -- & 2.153 & 0.107 & -- & 0.388 & 5.603 & -- & 2.280 & 141 & -- & -- & 3754.163 & -- & -- & -- & -- & 0.095 & -- & -- & 0.046 & -- & -- & 0.750 \\
+++++++++Round Church Cambridge & 2.137 & -- & 2.451 & 0.927 & -- & 1.003 & 6.103 & -- & 2.660 & 92 & -- & -- & 9779.270 & -- & -- & -- & -- & 1.107 & -- & -- & 0.582 & -- & -- & 1.540 \\
+++++++++Skansen Kronan Gothenburg & 0.301 & -- & 0.736 & 0.102 & -- & 0.226 & 2.906 & -- & 1.240 & 131 & -- & -- & 17153.677 & -- & -- & -- & -- & 0.026 & -- & -- & 0.008 & -- & -- & 0.670 \\
+++++++++Smolny Cathedral St Petersburg & 21.775 & -- & 0.554 & 2.111 & -- & 0.051 & 11.239 & -- & 1.660 & 131 & -- & -- & 10362.787 & -- & -- & -- & -- & 0.033 & -- & -- & 0.006 & -- & -- & 0.810 \\
+++++++++Some Cathedral In Barcelona & 0.651 & -- & 0.880 & 0.233 & -- & 0.315 & 6.271 & -- & 2.870 & 177 & -- & -- & 5001.973 & -- & -- & -- & -- & 0.026 & -- & -- & 0.011 & -- & -- & 0.890 \\
+++++++++Sri Mariamman Singapore & 1.219 & -- & 2.302 & 0.382 & -- & 0.683 & 10.088 & -- & 4.130 & 222 & -- & -- & 5985.540 & -- & -- & -- & -- & 0.077 & -- & -- & 0.023 & -- & -- & 0.910 \\
+++++++++Sri Thendayuthapani Singapore & 0.593 & -- & 46.269 & 0.150 & -- & 3.812 & 7.016 & -- & 23.370 & 98 & -- & -- & 13338.167 & -- & -- & -- & -- & 44.170 & -- & -- & 2.870 & -- & -- & 8.440 \\
+++++++++Sri Veeramakaliamman Singapore & 2.404 & -- & 2.559 & 0.559 & -- & 0.597 & 12.234 & -- & 3.470 & 157 & -- & -- & 12377.927 & -- & -- & -- & -- & 0.175 & -- & -- & 0.040 & -- & -- & 0.730 \\
+++++++++Statue Of Liberty & 52.358 & -- & 46.887 & 28.503 & -- & 20.012 & 235036.825 & -- & 26.160 & 134 & -- & -- & 99.900 & -- & -- & -- & -- & 9.091 & -- & -- & 4.122 & -- & -- & 6.970 \\
+++++++++The Pumpkin & 25.731 & -- & 94.672 & 5.552 & -- & 14.890 & 26.876 & -- & 33.410 & 196 & -- & -- & 4327.273 & -- & -- & -- & -- & 98.862 & -- & -- & 14.952 & -- & -- & 24.850 \\
+++++++++Thian Hook Keng Temple Singapore & 0.927 & -- & 0.832 & 0.093 & -- & 0.082 & 14.915 & -- & 2.750 & 138 & -- & -- & 3298.537 & -- & -- & -- & -- & 0.081 & -- & -- & 0.008 & -- & -- & 1.130 \\
+++++++++Tsar Nikolai I & 42.166 & -- & 48.499 & 8.732 & -- & 9.467 & 11.210 & -- & 9.790 & 98 & -- & -- & 4929.200 & -- & -- & -- & -- & 36.280 & -- & -- & 7.836 & -- & -- & 6.530 \\
+++++++++Urban II & 58.919 & -- & 47.490 & 11.006 & -- & 9.467 & 17.538 & -- & 9.380 & 96 & -- & -- & 8889.617 & -- & -- & -- & -- & 48.214 & -- & -- & 9.586 & -- & -- & 6.920 \\
+++++++++Vercingetorix & 82.952 & -- & 69.328 & 10.195 & -- & 8.788 & 7.257 & -- & 5.080 & 69 & -- & -- & 1223.970 & -- & -- & -- & -- & 17.706 & -- & -- & 3.104 & -- & -- & 1.500 \\
+++++++++Yueh Hai Ching Temple Singapore & 0.544 & -- & 0.720 & 0.075 & -- & 0.098 & 5.596 & -- & 0.940 & 43 & -- & -- & 1748.667 & -- & -- & -- & -- & 0.043 & -- & -- & 0.014 & -- & -- & 0.650 \\
++++++++ \midrule
++++++++-Mean & \textbf{16.146} & 23.697 & 18.023 & \textbf{3.121} & 4.014 & 2.912 & 6920.602 & \textbf{13.088} & 5.673 & 100.559 & \textbf{104.818} & -- & \textbf{5819.728} & 7358.187 & -- & -- & -- & 13.695 & -- & -- & 1.937 & -- & -- & 2.992 \\
+++++++++Mean & \textbf{15.676} & 27.164 & 17.547 & 3.032 & \textbf{1.988} & 2.840 & 6723.052 & \textbf{11.756} & 5.595 & 102.743 & \textbf{106} & -- & 5795.063 & \textbf{4506.671} & -- & -- & -- & 13.315 & -- & -- & 1.883 & -- & -- & 2.933 \\
++++++++ \bottomrule
++++++++ \end{tabular}}
++++++++ \end{table*}
+++++++ ```
+++++++ - untracked/modified files:
+++++++ ```
++++++++M code/evaluate_single_scene.py
++++++++ M code/results/single_scene/summary.csv
++++++++ M code/results/single_scene/summary_table.md
++++++++ M code/results/single_scene/summary_table.tex
+++++++ ?? MVG_Project_Report_Ortal_Dayan.pdf
+++++++ ?? "RESFM- ROBUST DEEP EQUIVARIANT STRUCTURE FROM MOTION.pdf"
+++++++ ?? "claude specs/SPEC_cvpr_experiments.md"
+++++++ ?? "claude specs/SPEC_sfm_datasets_setup.md"
+++++++ ?? "claude specs/SPEC_single_scene_experiments.md"
+++++++ ?? "claude specs/SPEC_uesfm_combined.md"
++++++++?? "claude specs/TASK_crossdataset_readiness.md"
+++++++ ?? code/datasets/Euclidean
+++++++ ?? tmp_ab_check/
+++++++ ```
+++++++ ### esfm-baseline (official ESFM)
+++++++ - path: `/home/projects/bagon/ortalda/MVG/final-project/esfm-baseline`
+++++++-- commit: `2c0079497bf5dda1eeb2a902b2f859c6387f213a`
++++++++- commit: `de666af53423c1dc3dd412e53c4e153c03d70922`
+++++++ - **WARNING: working tree dirty.** Diff:
+++++++ ```diff
+++++++ (untracked files only)
+++++++@@ -58,6 +464,7 @@ Generated: 2026-07-13T20:41:04
+++++++ ?? code/confs/single_scene_generated/ss_esfm_Buddah_Tooth_Relic_Temple_Singapore_seed1.conf
+++++++ ?? code/confs/single_scene_generated/ss_esfm_Buddah_Tooth_Relic_Temple_Singapore_seed2.conf
+++++++ ?? code/confs/single_scene_generated/ss_esfm_Doge_Palace_Venice_seed1.conf
++++++++?? code/confs/single_scene_generated/ss_esfm_Doge_Palace_Venice_seed2.conf
+++++++ ?? code/confs/single_scene_generated/ss_esfm_Door_Lund_seed1.conf
+++++++ ?? code/confs/single_scene_generated/ss_esfm_Door_Lund_seed2.conf
+++++++ ?? code/confs/single_scene_generated/ss_esfm_Drinking_Fountain_Somewhere_In_Zurich_seed1.conf
+++++++@@ -114,16 +521,22 @@ Generated: 2026-07-13T20:41:04
+++++++ ?? code/confs/single_scene_generated/ss_esfm_Round_Church_Cambridge_seed2.conf
+++++++ ?? code/confs/single_scene_generated/ss_esfm_Skansen_Kronan_Gothenburg_seed0.conf
+++++++ ?? code/confs/single_scene_generated/ss_esfm_Skansen_Kronan_Gothenburg_seed1.conf
++++++++?? code/confs/single_scene_generated/ss_esfm_Skansen_Kronan_Gothenburg_seed2.conf
+++++++ ?? code/confs/single_scene_generated/ss_esfm_Smolny_Cathedral_St_Petersburg_seed0.conf
+++++++ ?? code/confs/single_scene_generated/ss_esfm_Smolny_Cathedral_St_Petersburg_seed1.conf
++++++++?? code/confs/single_scene_generated/ss_esfm_Smolny_Cathedral_St_Petersburg_seed2.conf
+++++++ ?? code/confs/single_scene_generated/ss_esfm_Some_Cathedral_In_Barcelona_seed0.conf
++++++++?? code/confs/single_scene_generated/ss_esfm_Some_Cathedral_In_Barcelona_seed1.conf
++++++++?? code/confs/single_scene_generated/ss_esfm_Some_Cathedral_In_Barcelona_seed2.conf
+++++++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Mariamman_Singapore_seed0.conf
+++++++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Mariamman_Singapore_seed1.conf
+++++++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Mariamman_Singapore_seed2.conf
+++++++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Thendayuthapani_Singapore_seed0.conf
+++++++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Thendayuthapani_Singapore_seed1.conf
++++++++?? code/confs/single_scene_generated/ss_esfm_Sri_Thendayuthapani_Singapore_seed2.conf
+++++++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Veeramakaliamman_Singapore_seed0.conf
+++++++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Veeramakaliamman_Singapore_seed1.conf
++++++++?? code/confs/single_scene_generated/ss_esfm_Sri_Veeramakaliamman_Singapore_seed2.conf
+++++++ ?? code/confs/single_scene_generated/ss_esfm_Statue_Of_Liberty_seed0.conf
+++++++ ?? code/confs/single_scene_generated/ss_esfm_Statue_Of_Liberty_seed1.conf
+++++++ ?? code/confs/single_scene_generated/ss_esfm_Statue_Of_Liberty_seed2.conf
+++++++@@ -137,10 +550,14 @@ Generated: 2026-07-13T20:41:04
+++++++ ?? code/confs/single_scene_generated/ss_esfm_Tsar_Nikolai_I_seed1.conf
+++++++ ?? code/confs/single_scene_generated/ss_esfm_Tsar_Nikolai_I_seed2.conf
+++++++ ?? code/confs/single_scene_generated/ss_esfm_Urban_II_seed0.conf
++++++++?? code/confs/single_scene_generated/ss_esfm_Urban_II_seed1.conf
++++++++?? code/confs/single_scene_generated/ss_esfm_Urban_II_seed2.conf
+++++++ ?? code/confs/single_scene_generated/ss_esfm_Vercingetorix_seed0.conf
+++++++ ?? code/confs/single_scene_generated/ss_esfm_Vercingetorix_seed1.conf
+++++++ ?? code/confs/single_scene_generated/ss_esfm_Vercingetorix_seed2.conf
+++++++ ?? code/confs/single_scene_generated/ss_esfm_Yueh_Hai_Ching_Temple_Singapore_seed0.conf
++++++++?? code/confs/single_scene_generated/ss_esfm_Yueh_Hai_Ching_Temple_Singapore_seed1.conf
++++++++?? code/confs/single_scene_generated/ss_esfm_Yueh_Hai_Ching_Temple_Singapore_seed2.conf
+++++++ ```
+++++++ 
+++++++ ## Environment (shared venv used for BOTH methods)
+++++++diff --git a/code/results/single_scene/summary.csv b/code/results/single_scene/summary.csv
+++++++index e4f9f5e..4a35106 100644
+++++++--- a/code/results/single_scene/summary.csv
++++++++++ b/code/results/single_scene/summary.csv
+++++++@@ -2,171 +2,132 @@ method,scene,seed,rot_err_deg,pos_err,reproj_err_px,reproj_med_px,n_cams,runtime
+++++++ esfm,Alcatraz Courtyard,0,0.324288380204305,0.0854554642858637,3.379232694353053,1.9204072326560708,133,4821.4,4792.707528591156,99999.0
+++++++ esfm,Alcatraz Courtyard,1,0.36138678157391557,0.09112912757839253,3.3691035794153503,1.8536696768772378,133,4825.72,4804.594581127167,99999.0
+++++++ esfm,Alcatraz Courtyard,2,0.4015043649148433,0.10235190268069758,3.557242072642648,1.9198007399315369,133,4833.02,4812.242679834366,99999.0
+++++++-uesfm,Alcatraz Courtyard,0,0.6370266175309793,0.1671943803031883,5.956592777085197,3.6638282340102246,133,6478.06,6448.735792636871,99999.0
+++++++-uesfm,Alcatraz Courtyard,1,0.5812669610362707,0.15343994793419002,5.71686544495911,3.550076086474883,133,6483.48,6454.087207078934,99999.0
+++++++-uesfm,Alcatraz Courtyard,2,0.6293740753551684,0.16343948136157807,5.713459273661585,3.459422664018691,133,6483.73,6451.153926610947,99999.0
++++++++uesfm,Alcatraz Courtyard,0,0.5509063563081962,0.14050635120746804,5.528957815039945,3.569220349534855,133,6556.5,6511.41095662117,99999.0
++++++++uesfm,Alcatraz Courtyard,1,0.6823224031499351,0.17277455831381494,5.435931436314427,3.4771386471547348,133,6552.19,6506.419185638428,99999.0
+++++++ esfm,Alcatraz Water Tower,0,0.6984432687370988,0.38694796813795407,4.741313233832226,2.1058898944784543,172,2461.59,2440.421297073364,99999.0
+++++++ esfm,Alcatraz Water Tower,1,0.5944425806062753,0.3287537445614408,4.5656682619915765,1.9703698309044646,172,2467.28,2454.429310798645,99999.0
+++++++ esfm,Alcatraz Water Tower,2,0.7051984712289041,0.39359395644708545,4.982802041506025,2.1318655747232063,172,2455.14,2441.783402681351,99999.0
+++++++-uesfm,Alcatraz Water Tower,0,18.69941055700858,8.189402666249999,12.38813738647375,6.481578998967327,172,3857.94,3831.41609287262,99999.0
+++++++-uesfm,Alcatraz Water Tower,1,1.3394060108035364,0.7384271683550214,6.577617375342355,3.200105455600137,172,3887.47,3865.343374729156,99999.0
+++++++-uesfm,Alcatraz Water Tower,2,2.200189184784068,1.3839747978399466,6.854463337738994,3.1730434024886014,172,3819.31,3799.390902757645,99999.0
++++++++uesfm,Alcatraz Water Tower,0,2.016029335283383,1.087082854315953,7.9502009315303575,4.413488326253207,172,3805.19,3775.908159971237,99999.0
++++++++uesfm,Alcatraz Water Tower,1,1.4259772151091428,0.7645994701417347,7.177871865477131,3.7683369835989504,172,3787.07,3763.27596282959,99999.0
++++++++uesfm,Alcatraz Water Tower,2,0.9982626119141534,0.558403777925872,7.063395425015731,3.814475911180562,172,3795.37,3773.579354286194,99999.0
+++++++ esfm,Buddah Tooth Relic Temple Singapore,0,1.3132762212436329,0.31652548993670354,9.903890396062208,4.634952217191846,162,2921.45,2903.059753894806,99999.0
+++++++ esfm,Buddah Tooth Relic Temple Singapore,1,0.9908742070528852,0.2335767172234499,9.836436953133868,4.661767906617449,162,2915.39,2904.387546062469,99999.0
+++++++ esfm,Buddah Tooth Relic Temple Singapore,2,1.089403690069201,0.2896515999941069,10.151103921611115,4.628183275251743,162,2913.67,2902.928613901138,99999.0
+++++++-uesfm,Buddah Tooth Relic Temple Singapore,0,1.9767734026333772,0.5777116480820644,15.584999477939833,8.800407928031486,162,4315.97,4296.739196300507,99999.0
+++++++-uesfm,Buddah Tooth Relic Temple Singapore,1,1.4532821126601407,0.345133635718661,16.753182673783428,9.555989756393366,162,4334.28,4311.474883317947,99999.0
+++++++-uesfm,Buddah Tooth Relic Temple Singapore,2,32.81105779154638,6.0024233739495125,26.564606536203048,13.925597303414058,162,4312.37,4291.042752504349,99999.0
++++++++uesfm,Buddah Tooth Relic Temple Singapore,0,1.5799948547864517,0.43886498334860274,13.484389374830494,7.719072745032001,162,4294.34,4266.538989067078,99999.0
++++++++uesfm,Buddah Tooth Relic Temple Singapore,1,1.934129180062467,0.5924991617860504,13.872322562031941,8.04977435062149,162,4303.62,4259.311170101166,99999.0
++++++++uesfm,Buddah Tooth Relic Temple Singapore,2,1.8591664615796053,0.5279087705483619,14.731758217446671,8.481044153007865,162,4270.55,4249.088565349579,99999.0
+++++++ esfm,Doge Palace Venice,0,0.9741701978804636,0.2578214102266912,6.511459231891615,3.6947627756511525,241,28795.18,28746.22650837898,99999.0
+++++++ esfm,Doge Palace Venice,1,0.627860879835449,0.1773446161890865,6.351890452986879,3.537139014829222,241,28742.27,28699.91308617592,99999.0
+++++++-uesfm,Doge Palace Venice,0,0.9110673544913167,0.2904399918743801,10.966183720673449,6.771416746021886,241,21074.95,21017.30448675156,99999.0
++++++++esfm,Doge Palace Venice,2,0.6576305681966796,0.19114542381679772,6.214841374149415,3.591638795332466,241,10030.76,9989.16938996315,99999.0
+++++++ esfm,Door Lund,0,0.01167051570787235,0.003654424593160691,1.8552512563951393,0.9492901559591532,12,4892.41,4878.351888656616,99999.0
+++++++ esfm,Door Lund,1,0.009685491454411136,0.0030270923396151107,1.6462990657331118,0.9049495114630693,12,4884.51,4877.529450178146,99999.0
+++++++ esfm,Door Lund,2,0.02068285629921104,0.005193423834514985,1.933833706356197,0.9506754982524039,12,4888.17,4880.201772212982,99999.0
+++++++-uesfm,Door Lund,0,14.059454220682019,0.9733778175929079,11.85413335315025,6.517535151323047,12,9185.47,8715.343282699585,95000.0
+++++++-uesfm,Door Lund,1,14.111993883089163,0.9618908892808385,11.853785941610944,6.504290333453439,12,9203.34,8729.614821910858,95000.0
+++++++-uesfm,Door Lund,2,14.344083655704312,0.9193773651816372,12.058851721280973,6.643622749322514,12,9176.97,9162.127201795578,99999.0
++++++++uesfm,Door Lund,0,0.03084538453455712,0.007522151486969653,2.3711871496098955,1.6703501976219666,12,9262.05,9243.771874427795,99999.0
+++++++ esfm,Drinking Fountain Somewhere In Zurich,0,25.804179845300204,1.1242096722251314,8.219025541657992,3.3464550934365884,14,888.07,874.4874215126038,99999.0
+++++++ esfm,Drinking Fountain Somewhere In Zurich,1,25.754577171528883,1.1271567319301725,8.245599860088616,3.352895635561939,14,882.33,873.2415659427643,99999.0
+++++++ esfm,Drinking Fountain Somewhere In Zurich,2,0.019604308735801684,0.006780729271172703,1.7090486008047308,0.6375575382268279,14,881.11,873.6504108905792,99999.0
+++++++-uesfm,Drinking Fountain Somewhere In Zurich,0,26.27952541668808,1.102546079947844,8.40971803772888,3.4593948661771203,14,2969.96,2950.480526924133,99999.0
+++++++-uesfm,Drinking Fountain Somewhere In Zurich,1,26.264849988219943,1.1078406736783482,8.547474098662555,3.585334207185317,14,2960.9,2504.42572927475,85000.0
+++++++-uesfm,Drinking Fountain Somewhere In Zurich,2,0.06235835436589852,0.011103715758719332,2.5743283079384534,1.5361135173722633,14,2960.08,2945.673992872238,99999.0
++++++++uesfm,Drinking Fountain Somewhere In Zurich,0,25.946100100284117,1.1178920745623426,8.471712388928934,3.6241397282821626,14,5817.37,5796.999963760376,99999.0
++++++++uesfm,Drinking Fountain Somewhere In Zurich,1,25.973070279327214,1.1166462082604052,8.538877295313494,3.6343523276029472,14,5821.94,5806.947921991348,99999.0
+++++++ esfm,East Indiaman Goteborg,0,6.033696331509855,1.0773322362193534,7.427433157074123,2.292729458380311,179,3990.25,3970.732353687286,99999.0
+++++++ esfm,East Indiaman Goteborg,1,5.313967690277326,0.9356515362150493,7.350753973659207,2.2747477951684703,179,3986.66,3970.914752960205,99999.0
+++++++ esfm,East Indiaman Goteborg,2,5.357164002763642,0.9254557315236661,7.174538078735992,2.2493605774605823,179,3984.64,3967.810528755188,99999.0
+++++++-uesfm,East Indiaman Goteborg,0,4.992588850190341,0.9093886386013725,9.930437318497303,4.011819888270754,179,5247.16,5221.816437721252,99999.0
+++++++-uesfm,East Indiaman Goteborg,1,5.382092262718057,0.9927766021898647,10.035611563702533,3.912004668740913,179,5245.65,5210.653891563416,99999.0
+++++++-uesfm,East Indiaman Goteborg,2,5.554715231405653,0.9752252406138763,9.5551521706325,3.7417116053580193,179,5231.48,5207.815300703049,99999.0
++++++++uesfm,East Indiaman Goteborg,0,5.365772057886591,0.9552115949640507,9.655123223073948,4.174486084237091,179,5203.6,5178.753549814224,99999.0
++++++++uesfm,East Indiaman Goteborg,1,10.45533482189932,2.0687908277631517,9.98932850274456,4.110771460953736,179,5195.98,5174.958552598953,99999.0
+++++++ esfm,Ecole Superior De Guerre,0,0.4863901164687255,0.1239377922325492,3.750666445958877,2.0528113108131016,35,2385.33,2371.708672523499,99999.0
+++++++ esfm,Ecole Superior De Guerre,1,0.3031794677170201,0.07921628898965839,3.2456935905644406,1.730123845260616,35,2379.34,2371.191816806793,99999.0
+++++++ esfm,Ecole Superior De Guerre,2,0.2580439805296837,0.06772089648290676,3.257951412507941,1.687291979251782,35,2376.49,2368.935952663422,99999.0
+++++++-uesfm,Ecole Superior De Guerre,0,0.28243957678843373,0.0778675008788964,5.291358032458062,3.601773151002547,35,4647.51,4622.850021600723,99999.0
+++++++-uesfm,Ecole Superior De Guerre,1,0.25507313740145815,0.06695766472508183,5.170495431958795,3.5179325723573998,35,4636.07,4616.890106916428,99999.0
+++++++-uesfm,Ecole Superior De Guerre,2,0.3137499276035537,0.08683967313665406,5.583934687507232,3.7813953498055604,35,4617.92,4602.193422079086,99999.0
++++++++uesfm,Ecole Superior De Guerre,0,51.32681795107141,2.57407526776939,11.751384768771835,6.449293516058521,35,10426.77,10405.3174200058,99999.0
+++++++ esfm,Eglise du dome,0,0.3242410065346163,0.0824797769992069,3.906129661328729,1.3512292775673158,85,25054.74,25028.64922428131,99999.0
+++++++ esfm,Eglise du dome,1,0.3771623368290559,0.09432763361451474,3.692490941110161,1.297032357665518,85,25144.16,25116.01952123642,99999.0
+++++++-uesfm,Eglise du dome,0,45.800562123362404,6.451383786272971,7.963674120656965,3.986938865913781,85,23371.73,23333.49408817291,99999.0
++++++++uesfm,Eglise du dome,0,0.8528700994829498,0.23193079249562099,5.944108200898114,2.9306630821877517,85,9803.94,9764.585114717484,99999.0
+++++++ esfm,Folke Filbyter,0,89.03873163576341,0.12895560134543632,17.986624161769267,7.423093846189277,40,1582.8,1573.360969305038,99999.0
+++++++ esfm,Folke Filbyter,1,84.74740427947457,0.13132264058966686,17.98706393015447,7.385900243615879,40,1579.13,1572.079373121262,99999.0
+++++++ esfm,Folke Filbyter,2,83.70817707245973,0.1261878953205589,21.85200129525089,6.324245998829542,40,1583.76,1576.207699537277,99999.0
+++++++-uesfm,Folke Filbyter,0,80.34590333370599,0.13169898199385566,22.83327162619912,11.711593158296864,40,3701.49,3129.452286243439,85000.0
+++++++-uesfm,Folke Filbyter,1,81.28676564061475,0.1282318052949447,24.480394314414323,9.74969805057316,40,3698.11,3681.891911745071,99999.0
+++++++-uesfm,Folke Filbyter,2,82.23270227083374,0.1312531386108397,22.153947807931974,10.961536903594071,40,3709.96,3695.84988451004,99999.0
++++++++uesfm,Folke Filbyter,0,84.55526303793802,0.13091275274611416,29.190267154194732,11.160073225547443,40,3747.65,3727.990335702896,99999.0
++++++++uesfm,Folke Filbyter,1,84.87047537526948,0.13112291273275417,32.36503566471496,13.743964264262598,40,3743.53,3728.497133255005,99999.0
++++++++uesfm,Folke Filbyter,2,78.54375401873024,0.1316690349103016,23.11854509839131,10.94439865744236,40,3774.4,3007.95537352562,80000.0
+++++++ esfm,Fort Channing Gate Singapore,0,0.10143646071220622,0.04531774433800129,1.7158776598376824,0.8165178663538488,27,3438.39,3428.288990497589,99999.0
+++++++ esfm,Fort Channing Gate Singapore,1,0.08873951454303304,0.039001318240256114,1.783712095722291,0.8156020254191567,27,3430.88,3424.490573883057,99999.0
+++++++ esfm,Fort Channing Gate Singapore,2,0.09031224636384508,0.03957289173255022,1.850800887931411,0.820658413919779,27,3433.07,3426.547735452652,99999.0
+++++++-uesfm,Fort Channing Gate Singapore,0,33.158343156383296,11.908537566474344,12.69248500308198,5.438758486477886,27,6209.29,6192.107179880142,99999.0
+++++++-uesfm,Fort Channing Gate Singapore,1,0.07319798057764343,0.03541606344760984,3.4819579894798927,2.2085534812612853,27,6159.7,6144.797720909119,99999.0
+++++++-uesfm,Fort Channing Gate Singapore,2,0.15394831153888047,0.0686769122909015,3.77993377770425,2.2995094868650368,27,6147.27,6132.745970487595,99999.0
++++++++uesfm,Fort Channing Gate Singapore,0,0.10214634402234797,0.048923384893640356,3.7907267588398517,2.449846070018404,27,6175.88,6156.048281908035,99999.0
+++++++ esfm,Golden Statue Somewhere In Hong Kong,0,0.23894829924734365,0.053817571637812096,1.7239352536484445,0.744089758310296,18,4096.74,4087.775670289993,99999.0
+++++++ esfm,Golden Statue Somewhere In Hong Kong,1,0.2233990458438394,0.053061028269618456,1.8936878914364637,0.7746436618607616,18,4096.28,4088.994391202927,99999.0
+++++++ esfm,Golden Statue Somewhere In Hong Kong,2,0.23397914670516012,0.051137498616263405,1.87070255923672,0.7763877145267694,18,4094.54,4087.674698591232,99999.0
+++++++-uesfm,Golden Statue Somewhere In Hong Kong,0,0.36074365888948035,0.06894135246785482,2.939092008443413,1.863811153764833,18,7761.69,7746.569520235062,99999.0
+++++++-uesfm,Golden Statue Somewhere In Hong Kong,1,0.35638930493960375,0.06132592684828702,2.686157398440538,1.735571387461727,18,7759.16,7745.389947891235,99999.0
+++++++-uesfm,Golden Statue Somewhere In Hong Kong,2,0.3955566726941678,0.09197790539367184,2.878341172309852,1.8164738202745854,18,7766.46,7751.458931922913,99999.0
+++++++ esfm,Gustav Vasa,0,3.4881363194229826,0.2273181104934767,1.7369784534725496,0.9521606063342205,18,711.53,703.3091380596161,99999.0
+++++++ esfm,Gustav Vasa,1,3.4307883498221035,0.21432466556133864,1.5556448596237022,0.9232184455756751,18,710.28,703.9906163215637,99999.0
+++++++ esfm,Gustav Vasa,2,3.6089062562451386,0.2196578287395002,1.673803634770729,0.9429755567757121,18,707.79,701.5903799533844,99999.0
+++++++-uesfm,Gustav Vasa,0,5.828154714648938,0.32510821331670736,3.0711042143581646,2.4170166616864357,18,2566.0,2547.931907176971,99999.0
+++++++-uesfm,Gustav Vasa,1,6.515239842974278,0.35941242702736925,3.089677100739632,2.4476428398062584,18,2530.9,1385.089058637619,55000.0
+++++++-uesfm,Gustav Vasa,2,5.954036423569478,0.34942942401850696,3.101296679577594,2.400483554095863,18,2533.76,2520.798699855804,99999.0
++++++++uesfm,Gustav Vasa,0,36.645536646380805,1.0070193571286277,4.980488290477233,3.197977230260218,18,2562.94,2419.297919034958,95000.0
++++++++uesfm,Gustav Vasa,1,36.4914104514929,1.03254185606701,5.377024289258223,3.4818421684593366,18,4566.81,4543.375423431396,99999.0
++++++++uesfm,Gustav Vasa,2,4.4238740734834625,0.284762443333143,3.8820183248893803,2.8663959682636566,18,4533.21,4516.61039352417,99999.0
+++++++ esfm,GustavIIAdolf,0,46.92758389264633,8.570499881283022,11.684017169857798,5.131125459552879,57,1047.82,1038.018162250519,99999.0
+++++++ esfm,GustavIIAdolf,1,47.21999673071185,8.515052769156952,11.695065125063977,5.091617086264161,57,1041.41,1034.5799472332,99999.0
+++++++ esfm,GustavIIAdolf,2,47.2312223396789,8.523261295519099,11.787583212607423,5.2715388868550574,57,1043.28,1036.478415489197,99999.0
+++++++-uesfm,GustavIIAdolf,0,89.37607436538163,13.276440526172165,15.53883714368065,5.864332719302596,57,2904.42,2881.687849521637,99999.0
+++++++-uesfm,GustavIIAdolf,1,70.16367503302509,11.555242253627961,15.162174576474532,6.9709512967490355,57,2884.62,2870.103493452072,99999.0
+++++++-uesfm,GustavIIAdolf,2,72.72996778765942,11.054260851523775,16.85269001998736,9.009102977660282,57,2888.7,2874.096453428268,99999.0
++++++++uesfm,GustavIIAdolf,0,85.51179144359693,12.792264407383987,17.018024925092185,8.196266635685827,57,2973.35,2956.916826248169,99999.0
++++++++uesfm,GustavIIAdolf,1,88.9494736685426,13.260723460498589,17.699982689478404,8.888010081099381,57,2995.3,2976.055320978165,99999.0
++++++++uesfm,GustavIIAdolf,2,90.12096202318821,13.335169188185864,16.941538974035925,7.772181288890227,57,2974.4,2960.110015630722,99999.0
+++++++ esfm,Jonas Ahlstromer,0,64.46662959289324,11.985605264945464,12.602584968179986,2.9168445332917745,40,614.88,607.3498430252075,99999.0
+++++++ esfm,Jonas Ahlstromer,1,38.42510202464649,8.741159740710629,10.25268858530355,3.7024742418884333,40,618.73,611.5567183494568,99999.0
+++++++ esfm,Jonas Ahlstromer,2,43.32891670826628,8.741053567623224,10.553031866494822,3.031717304343782,40,612.74,606.6802380084991,99999.0
+++++++-uesfm,Jonas Ahlstromer,0,42.30073879858729,10.272232427418913,9.098970988131883,3.992857074878733,40,2339.57,1280.604686737061,55000.0
+++++++-uesfm,Jonas Ahlstromer,1,42.18412804385421,10.310808592826955,8.453808908218651,3.4329852879523464,40,2334.33,2320.362742424011,99999.0
+++++++-uesfm,Jonas Ahlstromer,2,42.25301394187691,10.315519611163806,8.793789077520445,3.8046125347596442,40,2311.04,2297.139800786972,99999.0
++++++++uesfm,Jonas Ahlstromer,0,45.264596178655594,9.403212752141883,11.336617283269852,4.728596140494824,40,2421.11,2401.221394062042,99999.0
+++++++ esfm,Kings College University Of Toronto,0,14.400461796497643,3.2653370282498866,4.064353916685246,1.6753245173381437,77,1074.94,1066.306197404861,99999.0
+++++++ esfm,Kings College University Of Toronto,1,11.956699445712928,2.705452710546247,3.9727144140636783,1.5797087908756111,77,1071.64,1064.129351377487,99999.0
+++++++ esfm,Kings College University Of Toronto,2,0.7191591237413282,0.16178494376153177,2.656807607821862,1.1949677972638288,77,1090.36,1082.987608671188,99999.0
+++++++-uesfm,Kings College University Of Toronto,0,31.932532001930838,2.9775501815475,6.099555590572752,3.8717713150174773,77,2884.23,2866.194714784622,99999.0
+++++++-uesfm,Kings College University Of Toronto,1,2.4241467960672063,0.41577835428574456,4.631696476065898,3.148046400399828,77,2899.51,2884.114916801453,99999.0
+++++++-uesfm,Kings College University Of Toronto,2,30.212552334890066,2.3209884971628156,5.844709739288878,3.754268841443037,77,2874.99,2860.901805400848,99999.0
++++++++uesfm,Kings College University Of Toronto,0,29.200261744017546,2.0893308856114428,7.037455848998898,4.8227457854711675,77,2856.72,2838.923613071442,99999.0
+++++++ esfm,Lund University Sphinx,0,13.661465370705821,3.5591094267381553,6.242667318159146,2.715620555070548,70,2718.96,2709.5872194767,99999.0
+++++++ esfm,Lund University Sphinx,1,12.524302160109885,3.2130942199657664,6.1182420215295075,2.7468468987195953,70,2722.86,2715.36418390274,99999.0
+++++++ esfm,Lund University Sphinx,2,7.98961938632192,2.1710959765491817,5.807289242508196,2.893370267054225,70,2713.88,2706.478076219559,99999.0
+++++++-uesfm,Lund University Sphinx,0,15.165716414733652,3.5564827651349713,10.803408462752863,6.730422995453065,70,4644.92,4626.135177612305,99999.0
+++++++-uesfm,Lund University Sphinx,1,68.88899086055604,13.548881652837478,14.336626262363456,7.626356837439143,70,4703.31,4687.50173997879,99999.0
+++++++-uesfm,Lund University Sphinx,2,15.014326075576053,3.5607694164340526,11.039432108003597,6.81081192121135,70,4615.6,4600.294319868088,99999.0
+++++++ esfm,Nijo Castle Gate,0,0.6521524412276166,0.13305866403975986,7.404953325705311,3.0629804954987674,19,1097.29,1084.301982402802,99999.0
+++++++ esfm,Nijo Castle Gate,1,0.8564043952718835,0.16950374865879153,7.358076475544799,3.1235223975185864,19,1094.89,1084.640900611877,99999.0
+++++++ esfm,Nijo Castle Gate,2,1.0089926380330574,0.19095470213426863,7.481049991287358,3.1671779658403687,19,1093.35,1083.446812152863,99999.0
+++++++-uesfm,Nijo Castle Gate,0,1.6923297320319277,0.3246009364794658,11.792009654594867,6.945622003724464,19,3310.13,3288.75860619545,99999.0
+++++++-uesfm,Nijo Castle Gate,1,1.6697791743236226,0.32440429939345133,12.704681259536075,7.466128811395402,19,3308.19,3290.209539175034,99999.0
+++++++-uesfm,Nijo Castle Gate,2,1.682403384345942,0.32817821817561266,11.917977420973372,7.102279949089778,19,3301.77,3284.964460611343,99999.0
+++++++-uesfm,Pantheon Paris,0,0.3823220210788846,0.05987691585642601,7.649321203280243,4.797127317249956,179,5506.88,5473.694003582001,99999.0
+++++++-uesfm,Pantheon Paris,1,45.61471478909275,6.671068486991568,16.912359249810606,6.602415648206237,179,5491.31,5459.329566717148,99999.0
+++++++-uesfm,Pantheon Paris,2,12.03796171625132,2.465102828965073,16.125095834055482,6.860965809143959,179,5493.94,5463.555550813675,99999.0
++++++++uesfm,Pantheon Paris,0,0.5453004964472504,0.06999784414241637,7.790870638815493,5.057509442480535,179,5594.6,5562.967379331589,99999.0
+++++++ esfm,Park Gate Clermont Ferrand,0,19.399216821978534,9.438471236639753,7.521969683469249,3.892136696028138,34,1597.96,1588.07075381279,99999.0
+++++++ esfm,Park Gate Clermont Ferrand,1,19.325402297843297,9.417998214523331,7.465007292717891,3.849088919564891,34,1596.87,1589.703137874603,99999.0
+++++++ esfm,Park Gate Clermont Ferrand,2,25.440611579781365,11.720641869372841,8.15032188787613,4.0871739024940315,34,1595.99,1588.841923236847,99999.0
+++++++-uesfm,Park Gate Clermont Ferrand,0,19.27114964212729,9.381930985797803,7.856513901068924,4.317004881495189,34,3598.69,3580.396904230118,99999.0
+++++++-uesfm,Park Gate Clermont Ferrand,1,19.288145181777256,9.394362801959678,7.762082007624792,4.207851936960509,34,3592.66,3577.494926214218,99999.0
+++++++-uesfm,Park Gate Clermont Ferrand,2,19.345727373971844,9.415066249707772,7.936652645099164,4.425507919979627,34,3600.86,3581.473034620285,99999.0
+++++++ esfm,Plaza De Armas Santiago,0,0.7341628728579594,0.32156050159900956,6.000154308231358,2.679971945741675,240,6375.02,6333.455562829971,99999.0
+++++++ esfm,Plaza De Armas Santiago,1,0.5176165701053533,0.22744923951256435,5.661576462867733,2.533952088248649,240,6373.68,6336.030319213867,99999.0
+++++++ esfm,Plaza De Armas Santiago,2,1.0740077415291467,0.46304622323619105,6.365716833190761,2.733195456670397,240,6369.35,6331.436369419098,99999.0
+++++++-uesfm,Plaza De Armas Santiago,0,6.6772176498607925,2.909933722327364,11.384209379387066,5.623987151554922,240,6540.76,6491.619550704956,99999.0
+++++++-uesfm,Plaza De Armas Santiago,1,1.3364898126442175,0.5686650140764532,7.951759184923103,3.933795015892702,240,6536.61,6482.320575714111,99999.0
+++++++-uesfm,Plaza De Armas Santiago,2,1.9671483940705004,0.8276654865349118,9.054185015150745,4.441179120460681,240,6529.24,6477.723381996155,99999.0
+++++++ esfm,Porta San Donato Bologna,0,0.5746001989174886,0.10608894230214515,5.691906895070585,2.4183460353027986,141,3751.75,3734.281898736954,99999.0
+++++++ esfm,Porta San Donato Bologna,1,0.8978003419853328,0.14474404936935045,5.682379581165418,2.563141107750346,141,3758.51,3742.688964605331,99999.0
+++++++ esfm,Porta San Donato Bologna,2,0.30440272958620346,0.07026546193502164,5.435311727517131,2.1626312981864086,141,3752.23,3738.056202173233,99999.0
+++++++-uesfm,Porta San Donato Bologna,0,1.1269164513732723,0.19275737956525044,10.071225650489282,5.557466375146339,141,5111.57,5083.21807050705,99999.0
+++++++-uesfm,Porta San Donato Bologna,1,60.66366912240625,10.234123128225525,19.593360464637342,9.305574874002819,141,5125.57,5098.085618019104,99999.0
+++++++-uesfm,Porta San Donato Bologna,2,1.2271289404198877,0.19605254084885107,10.066750816092268,5.595831381287904,141,5116.95,5093.473650932312,99999.0
+++++++ esfm,Round Church Cambridge,0,2.1173198662081005,0.9129854085205968,5.854320175654564,1.9683268910531708,92,9788.65,9759.719386577606,99999.0
+++++++ esfm,Round Church Cambridge,1,2.1415279636494953,0.9267927094737857,6.33905951937897,2.0439997148640834,92,9764.61,9741.356333732605,99999.0
+++++++-uesfm,Round Church Cambridge,0,2.5434984075631974,1.0219048587087352,8.914080922081759,4.131734829440289,92,12088.06,12047.02147889137,99999.0
+++++++-uesfm,Round Church Cambridge,1,2.435781054756339,0.9827352719070902,8.178482224933683,3.8400051230290893,92,12034.64,11993.72608184814,99999.0
++++++++esfm,Round Church Cambridge,2,2.1507951248804917,0.9401429799233081,6.115230559713554,2.0517209910251797,92,9784.55,9763.525354623795,99999.0
+++++++ esfm,Skansen Kronan Gothenburg,0,0.3380139741078169,0.1158931606908392,3.0478862369293624,1.5340801117570733,131,17163.94,17140.15425729752,99999.0
+++++++-uesfm,Skansen Kronan Gothenburg,0,0.37472494768162085,0.1243676312212316,4.798736338080396,3.348916057055834,131,16235.09,16202.21351981163,99999.0
++++++++esfm,Skansen Kronan Gothenburg,1,0.21516831826827018,0.07458182746108981,2.786705021535272,1.4619994473169777,131,17129.62,17109.18022465706,99999.0
++++++++esfm,Skansen Kronan Gothenburg,2,0.3508356515622903,0.11667319277987132,2.8836839507901773,1.5216584205042176,131,17167.47,17142.66050887108,99999.0
+++++++ esfm,Smolny Cathedral St Petersburg,0,21.81278753416543,2.110667559831311,11.2315357584775,6.379259553776089,131,10364.67,10320.51852750778,99999.0
+++++++ esfm,Smolny Cathedral St Petersburg,1,21.799828748358873,2.111313741354268,11.225141339870355,6.36605939788532,131,10360.52,10319.01374697685,99999.0
+++++++-uesfm,Smolny Cathedral St Petersburg,0,22.143392894578028,2.0985584569799265,12.52645725687386,7.895265841408903,131,10475.74,10416.62969732285,99999.0
++++++++esfm,Smolny Cathedral St Petersburg,2,21.71090600320656,2.110307413725059,11.259799763291888,6.4136596726420905,131,10363.17,10320.08415436745,99999.0
++++++++esfm,Some Cathedral In Barcelona,0,0.6775058368464424,0.24064262464083658,6.20165095815897,3.7253636674062887,177,5002.81,4985.274734258652,99999.0
++++++++esfm,Some Cathedral In Barcelona,1,0.6722147495861831,0.2423973792150467,6.350786792749083,3.720536755931007,177,5009.15,4991.552013158798,99999.0
++++++++esfm,Some Cathedral In Barcelona,2,0.6046048207068493,0.21699252142655406,6.261075100347952,3.7258171467923633,177,4993.96,4975.845166683197,99999.0
+++++++ esfm,Sri Mariamman Singapore,0,1.4920493985513708,0.4412389781578158,10.406406306167431,6.369585247957454,222,5987.98,5970.066485881805,99999.0
+++++++ esfm,Sri Mariamman Singapore,1,1.1345564396944785,0.3608906720626337,9.703927906328842,5.885963785879842,222,5993.62,5975.365797519684,99999.0
+++++++ esfm,Sri Mariamman Singapore,2,1.0293707921546553,0.3426709250774071,10.15220066698994,6.067537729054635,222,5975.02,5958.786058187485,99999.0
+++++++-uesfm,Sri Mariamman Singapore,0,16.417883129790784,5.526406595652791,19.962711816743173,12.18830031311293,222,6531.45,6494.750840425491,99999.0
+++++++-uesfm,Sri Mariamman Singapore,1,1.3793173305528281,0.5014534676018023,16.10857603648069,10.651562750176396,222,6467.65,6426.201541185379,99999.0
+++++++-uesfm,Sri Mariamman Singapore,2,1.639001766282598,0.536397596737868,17.490398022274007,11.439337896248928,222,6457.9,6427.717175960541,99999.0
+++++++ esfm,Sri Thendayuthapani Singapore,0,0.7190396606626565,0.17922550285921848,7.12365374281766,2.231392236341273,98,13362.09,13320.09781932831,99999.0
+++++++-uesfm,Sri Thendayuthapani Singapore,0,47.0750455982536,3.8553135654690074,14.133068630999212,7.117179633590087,98,13808.44,13754.94554877281,99999.0
++++++++esfm,Sri Thendayuthapani Singapore,1,0.5350953449295449,0.1261925950852554,6.4467364012557615,2.04778550672059,98,13347.85,13309.03068971634,99999.0
++++++++esfm,Sri Thendayuthapani Singapore,2,0.5262093764563488,0.14374074415280041,7.476614790858722,2.2575164196527426,98,13304.56,13262.61834168434,99999.0
+++++++ esfm,Sri Veeramakaliamman Singapore,0,2.333869921178247,0.542847254574005,11.965270419451508,5.472877662905037,157,12392.18,12365.15180635452,99999.0
+++++++-uesfm,Sri Veeramakaliamman Singapore,0,29.40487246624615,3.8676026992817505,28.950500021302087,14.32295224594287,157,12544.69,12504.26410722733,99999.0
++++++++esfm,Sri Veeramakaliamman Singapore,1,2.4804161216418734,0.5731176371311499,12.150997217479388,5.5309200508775875,157,12387.11,12363.29210400581,99999.0
++++++++esfm,Sri Veeramakaliamman Singapore,2,2.3974569344813434,0.5599333960859275,12.58650507397198,5.711763361742226,157,12354.49,12330.03034973145,99999.0
+++++++ esfm,Statue Of Liberty,0,52.35559762033152,28.241594953600423,688383.1603142131,683624.423417858,134,102.64,43.32754778862,45000.0
+++++++ esfm,Statue Of Liberty,1,52.36920727661949,28.627801879821945,9277.715116667516,9303.546387753544,134,99.78,59.98358988761902,65000.0
+++++++ esfm,Statue Of Liberty,2,52.3478568435799,28.64052858422677,7449.598084265948,7416.803834848509,134,97.28,3.520912647247314,0.0
+++++++-uesfm,Statue Of Liberty,0,40.92379801166407,19.915794833689585,48.781849500680366,25.904882064330994,134,12927.11,12902.23110127449,99999.0
+++++++-uesfm,Statue Of Liberty,1,38.79484508812604,17.090491683338875,33.27459817133403,17.608726936833005,134,12920.89,12896.32496738434,99999.0
+++++++ esfm,The Pumpkin,0,27.59459354015253,5.94835892276036,27.52772578222832,9.11126349703041,196,4330.9,4316.109169960022,99999.0
+++++++ esfm,The Pumpkin,1,26.54397520858153,5.701234509093555,27.342758332715125,8.907676505669826,196,4327.98,4314.043692827225,99999.0
+++++++ esfm,The Pumpkin,2,23.05395646123413,5.007378502393024,25.75879879971585,7.898438464943201,196,4322.94,4310.356798410416,99999.0
+++++++-uesfm,The Pumpkin,0,25.490608822831348,5.440568114393512,34.198429037653845,15.6024365790899,196,5793.03,5767.982522964478,99999.0
+++++++-uesfm,The Pumpkin,1,27.26556900533122,5.820572144056583,35.6951097814524,16.572200360468877,196,5785.61,5760.521738529205,99999.0
+++++++-uesfm,The Pumpkin,2,26.95722616022698,5.73349875876088,36.207441893083136,17.07678356860021,196,5793.15,5768.243654251099,99999.0
+++++++ esfm,Thian Hook Keng Temple Singapore,0,1.0653505173922067,0.1079565219850111,15.054152885837562,5.81055046268767,138,3299.26,3281.187238931656,99999.0
+++++++ esfm,Thian Hook Keng Temple Singapore,1,0.8390930608109779,0.08272919436528102,14.587041005106803,5.705702663819475,138,3306.93,3283.257929086685,99999.0
+++++++ esfm,Thian Hook Keng Temple Singapore,2,0.8760513596629002,0.08764395630143504,15.102705362055175,5.7567965038255124,138,3289.42,3278.173619031906,99999.0
+++++++-uesfm,Thian Hook Keng Temple Singapore,0,1.146131162829257,0.10799243740327723,20.24888615229727,10.297158058757331,138,4626.48,4595.79723072052,99999.0
+++++++-uesfm,Thian Hook Keng Temple Singapore,1,1.2072956626750737,0.12821580405567637,21.242388560332195,10.655698470690366,138,4600.34,4578.699931621552,99999.0
+++++++-uesfm,Thian Hook Keng Temple Singapore,2,1.251054659622132,0.12561972972485672,20.875018460120653,10.416875106878496,138,4610.3,4581.792747020721,99999.0
+++++++ esfm,Tsar Nikolai I,0,42.507553768489565,8.869914103947062,10.846690028504929,4.007961025902478,98,4932.15,4920.938139438629,99999.0
+++++++ esfm,Tsar Nikolai I,1,43.09835370739786,8.983133733509343,11.402130572143294,4.032588092580996,98,4924.71,4914.929541349411,99999.0
+++++++-uesfm,Tsar Nikolai I,0,75.94880014543322,15.146573953653359,15.307363921859029,6.734065570805788,98,6652.35,6629.384099006653,99999.0
+++++++-uesfm,Tsar Nikolai I,1,80.87807139992418,15.64102228288136,21.553650044939403,9.179513384124702,98,6641.34,6623.797205209732,99999.0
++++++++esfm,Tsar Nikolai I,2,40.892180545819315,8.342265568650971,11.381967217434166,4.272306098513181,98,4930.74,4920.495186090469,99999.0
+++++++ esfm,Urban II,0,59.0297968076411,11.008489348760156,17.600594623525765,5.400933781867582,96,8889.69,8878.01452589035,99999.0
++++++++esfm,Urban II,1,58.947927470837534,11.025374954076108,17.465515658589908,5.68378594242137,96,8885.17,8876.618469953537,99999.0
++++++++esfm,Urban II,2,58.77798771298748,10.982786678935327,17.546556385759054,5.767405629316944,96,8893.99,8885.34243106842,99999.0
+++++++ esfm,Vercingetorix,0,79.07478057258302,9.776081681696928,6.649989325889541,1.7432748253588333,69,1223.75,1212.340968608856,99999.0
+++++++ esfm,Vercingetorix,1,80.09313371434186,9.858626097407651,6.426263444651325,1.6979463697629655,69,1222.8,1213.763489961624,99999.0
+++++++ esfm,Vercingetorix,2,89.68787080902364,10.950615257181747,8.695412468830675,3.360645595149667,69,1225.36,1215.808262825012,99999.0
+++++++-uesfm,Vercingetorix,0,78.95942988866727,10.356244187384963,8.07065603849916,3.372297052368834,69,3105.42,3085.208213567734,99999.0
+++++++-uesfm,Vercingetorix,1,89.99175121301754,11.178484677344343,8.929255781006072,4.302365177465691,69,3105.88,3091.564247369766,99999.0
+++++++-uesfm,Vercingetorix,2,89.31151857112468,11.134780185118505,9.044030033429575,4.351952473195928,69,3105.52,3089.0203332901,99999.0
+++++++ esfm,Yueh Hai Ching Temple Singapore,0,0.6268598783684821,0.08360284283022794,5.56223941139456,2.132543748253723,43,1749.28,1738.747783660889,99999.0
++++++++esfm,Yueh Hai Ching Temple Singapore,1,0.5064956072004329,0.07235771132992698,5.30294026596974,2.06113770612239,43,1750.4,1739.704204082489,99999.0
++++++++esfm,Yueh Hai Ching Temple Singapore,2,0.49998151063409024,0.07009863154178562,5.92393490371475,2.2317763685579015,43,1746.32,1738.905562877655,99999.0
+++++++diff --git a/code/results/single_scene/summary_table.md b/code/results/single_scene/summary_table.md
+++++++index bd3f673..565fc23 100644
+++++++--- a/code/results/single_scene/summary_table.md
++++++++++ b/code/results/single_scene/summary_table.md
+++++++@@ -2,44 +2,45 @@
+++++++ 
+++++++ Mean over seeds per (scene, method); best value per scene in **bold** (lower is better except Nr).
+++++++ 
+++++++-| Scene | ESFM Rot (deg) | U-ESFM Rot (deg) | ESFM (paper) Rot (deg) | ESFM Trans | U-ESFM Trans | ESFM (paper) Trans | ESFM Reproj (px) | U-ESFM Reproj (px) | ESFM (paper) Reproj (px) | ESFM Nr | U-ESFM Nr | ESFM (paper) Nr | ESFM Time (s) | U-ESFM Time (s) | ESFM (paper) Time (s) | ESFM Rot-BA (deg) | U-ESFM Rot-BA (deg) | ESFM (paper) Rot-BA (deg) | ESFM Trans-BA | U-ESFM Trans-BA | ESFM (paper) Trans-BA | ESFM Reproj-BA (px) | U-ESFM Reproj-BA (px) | ESFM (paper) Reproj-BA (px) |
++++++++| Scene | ESFM (official code) Rot (deg) | U-ESFM Rot (deg) | ESFM (paper) Rot (deg) | ESFM (official code) Trans | U-ESFM Trans | ESFM (paper) Trans | ESFM (official code) Reproj (px) | U-ESFM Reproj (px) | ESFM (paper) Reproj (px) | ESFM (official code) Nr | U-ESFM Nr | ESFM (paper) Nr | ESFM (official code) Time (s) | U-ESFM Time (s) | ESFM (paper) Time (s) | ESFM (official code) Rot-BA (deg) | U-ESFM Rot-BA (deg) | ESFM (paper) Rot-BA (deg) | ESFM (official code) Trans-BA | U-ESFM Trans-BA | ESFM (paper) Trans-BA | ESFM (official code) Reproj-BA (px) | U-ESFM Reproj-BA (px) | ESFM (paper) Reproj-BA (px) |
+++++++ |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
+++++++-| Alcatraz Courtyard | **0.362** | 0.616 | 0.619 | **0.093** | 0.161 | 0.160 | **3.435** | 5.796 | 1.640 | **133** | 133 | -- | **4826.713** | 6481.757 | -- | -- | -- | 0.049 | -- | -- | 0.015 | -- | -- | 0.810 |
+++++++-| Alcatraz Water Tower | **0.666** | 7.413 | 0.933 | **0.370** | 3.437 | 0.518 | **4.763** | 8.607 | 2.130 | **172** | 172 | -- | **2461.337** | 3854.907 | -- | -- | -- | 0.230 | -- | -- | 0.116 | -- | -- | 0.550 |
+++++++-| Buddah Tooth Relic Temple Singapore | **1.131** | 12.080 | 1.030 | **0.280** | 2.308 | 0.233 | **9.964** | 19.634 | 2.060 | **162** | 162 | -- | **2916.837** | 4320.873 | -- | -- | -- | 0.081 | -- | -- | 0.014 | -- | -- | 0.850 |
+++++++-| Doge Palace Venice | **0.801** | 0.911 | 1.163 | **0.218** | 0.290 | 0.342 | **6.432** | 10.966 | 3.620 | **241** | 241 | -- | 28768.725 | **21074.950** | -- | -- | -- | 0.211 | -- | -- | 0.029 | -- | -- | 1 |
+++++++-| Door Lund | **0.014** | 14.172 | 0.024 | **0.004** | 0.952 | 0.006 | **1.812** | 11.922 | 0.320 | **12** | 12 | -- | **4888.363** | 9188.593 | -- | -- | -- | 0.006 | -- | -- | 0.001 | -- | -- | 0.300 |
+++++++-| Drinking Fountain Somewhere In Zurich | **17.193** | 17.536 | 0.031 | 0.753 | **0.740** | 0.004 | **6.058** | 6.511 | 0.330 | **14** | 14 | -- | **883.837** | 2963.647 | -- | -- | -- | 0.007 | -- | -- | 0.002 | -- | -- | 0.310 |
+++++++-| East Indiaman Goteborg | 5.568 | **5.310** | 3.814 | 0.979 | **0.959** | 0.621 | **7.318** | 9.840 | 4.130 | **179** | 179 | -- | **3987.183** | 5241.430 | -- | -- | -- | 3.117 | -- | -- | 0.509 | -- | -- | 1.850 |
+++++++-| Ecole Superior De Guerre | 0.349 | **0.284** | 0.318 | 0.090 | **0.077** | 0.081 | **3.418** | 5.349 | 0.720 | **35** | 35 | -- | **2380.387** | 4633.833 | -- | -- | -- | 0.024 | -- | -- | 0.005 | -- | -- | 0.340 |
+++++++-| Eglise du dome | **0.351** | 45.801 | 0.808 | **0.088** | 6.451 | 0.205 | **3.799** | 7.964 | 0.910 | **85** | 85 | -- | 25099.450 | **23371.730** | -- | -- | -- | 0.037 | -- | -- | 0.010 | -- | -- | 0.270 |
+++++++-| Folke Filbyter | 85.831 | **81.288** | 74.596 | **0.129** | 0.130 | 0.125 | **19.275** | 23.156 | 10.370 | **40** | 40 | -- | **1581.897** | 3703.187 | -- | -- | -- | 70.157 | -- | -- | 0.118 | -- | -- | 4.290 |
+++++++-| Fort Channing Gate Singapore | **0.093** | 11.128 | 0.207 | **0.041** | 4.004 | 0.093 | **1.783** | 6.651 | 0.520 | **27** | 27 | -- | **3434.113** | 6172.087 | -- | -- | -- | 0.020 | -- | -- | 0.008 | -- | -- | 0.250 |
+++++++-| Golden Statue Somewhere In Hong Kong | **0.232** | 0.371 | 0.292 | **0.053** | 0.074 | 0.073 | **1.829** | 2.835 | 0.400 | **18** | 18 | -- | **4095.853** | 7762.437 | -- | -- | -- | 0.031 | -- | -- | 0.004 | -- | -- | 0.270 |
+++++++-| Gustav Vasa | **3.509** | 6.099 | 34.181 | **0.220** | 0.345 | 1.085 | **1.655** | 3.087 | 3.520 | **18** | 18 | -- | **709.867** | 2543.553 | -- | -- | -- | 32.266 | -- | -- | 1.145 | -- | -- | 3.150 |
+++++++-| GustavIIAdolf | **47.126** | 77.423 | 67.784 | **8.536** | 11.962 | 9.714 | **11.722** | 15.851 | 13.910 | **57** | 57 | -- | **1044.170** | 2892.580 | -- | -- | -- | 58.458 | -- | -- | 8.524 | -- | -- | 11.490 |
+++++++-| Jonas Ahlstromer | 48.740 | **42.246** | 50.190 | **9.823** | 10.300 | 10.888 | 11.136 | **8.782** | 10.820 | **40** | 40 | -- | **615.450** | 2328.313 | -- | -- | -- | 47.117 | -- | -- | 10.451 | -- | -- | 8.410 |
+++++++-| Kings College University Of Toronto | **9.025** | 21.523 | 0.989 | 2.044 | **1.905** | 0.235 | **3.565** | 5.525 | 0.900 | **77** | 77 | -- | **1078.980** | 2886.243 | -- | -- | -- | 0.085 | -- | -- | 0.017 | -- | -- | 0.340 |
+++++++-| Lund University Sphinx | **11.392** | 33.023 | 19.522 | **2.981** | 6.889 | 4.585 | **6.056** | 12.060 | 4.780 | **70** | 70 | -- | **2718.567** | 4654.610 | -- | -- | -- | 8.752 | -- | -- | 2.191 | -- | -- | 1.360 |
+++++++-| Nijo Castle Gate | **0.839** | 1.682 | 1.495 | **0.165** | 0.326 | 0.286 | **7.415** | 12.138 | 1.700 | **19** | 19 | -- | **1095.177** | 3306.697 | -- | -- | -- | 0.069 | -- | -- | 0.012 | -- | -- | 0.730 |
+++++++-| Pantheon Paris | -- | 19.345 | 0.192 | -- | 3.065 | 0.050 | -- | 13.562 | 1.470 | -- | 179 | -- | -- | 5497.377 | -- | -- | -- | 0.040 | -- | -- | 0.005 | -- | -- | 0.490 |
+++++++-| Park Gate Clermont Ferrand | 21.388 | **19.302** | 0.391 | 10.192 | **9.397** | 0.125 | **7.712** | 7.852 | 0.570 | **34** | 34 | -- | **1596.940** | 3597.403 | -- | -- | -- | 0.049 | -- | -- | 0.022 | -- | -- | 0.350 |
+++++++-| Plaza De Armas Santiago | **0.775** | 3.327 | 6.782 | **0.337** | 1.435 | 2.944 | **6.009** | 9.463 | 7.400 | **240** | 240 | -- | **6372.683** | 6535.537 | -- | -- | -- | 2.556 | -- | -- | 1.383 | -- | -- | 4.900 |
+++++++-| Porta San Donato Bologna | **0.592** | 21.006 | 2.153 | **0.107** | 3.541 | 0.388 | **5.603** | 13.244 | 2.280 | **141** | 141 | -- | **3754.163** | 5118.030 | -- | -- | -- | 0.095 | -- | -- | 0.046 | -- | -- | 0.750 |
+++++++-| Round Church Cambridge | **2.129** | 2.490 | 2.451 | **0.920** | 1.002 | 1.003 | **6.097** | 8.546 | 2.660 | **92** | 92 | -- | **9776.630** | 12061.350 | -- | -- | -- | 1.107 | -- | -- | 0.582 | -- | -- | 1.540 |
+++++++-| Skansen Kronan Gothenburg | **0.338** | 0.375 | 0.736 | **0.116** | 0.124 | 0.226 | **3.048** | 4.799 | 1.240 | **131** | 131 | -- | 17163.940 | **16235.090** | -- | -- | -- | 0.026 | -- | -- | 0.008 | -- | -- | 0.670 |
+++++++-| Smolny Cathedral St Petersburg | **21.806** | 22.143 | 0.554 | 2.111 | **2.099** | 0.051 | **11.228** | 12.526 | 1.660 | **131** | 131 | -- | **10362.595** | 10475.740 | -- | -- | -- | 0.033 | -- | -- | 0.006 | -- | -- | 0.810 |
+++++++-| Sri Mariamman Singapore | **1.219** | 6.479 | 2.302 | **0.382** | 2.188 | 0.683 | **10.088** | 17.854 | 4.130 | **222** | 222 | -- | **5985.540** | 6485.667 | -- | -- | -- | 0.077 | -- | -- | 0.023 | -- | -- | 0.910 |
+++++++-| Sri Thendayuthapani Singapore | **0.719** | 47.075 | 46.269 | **0.179** | 3.855 | 3.812 | **7.124** | 14.133 | 23.370 | **98** | 98 | -- | **13362.090** | 13808.440 | -- | -- | -- | 44.170 | -- | -- | 2.870 | -- | -- | 8.440 |
+++++++-| Sri Veeramakaliamman Singapore | **2.334** | 29.405 | 2.559 | **0.543** | 3.868 | 0.597 | **11.965** | 28.951 | 3.470 | **157** | 157 | -- | **12392.180** | 12544.690 | -- | -- | -- | 0.175 | -- | -- | 0.040 | -- | -- | 0.730 |
+++++++-| Statue Of Liberty | 52.358 | **39.859** | 46.887 | 28.503 | **18.503** | 20.012 | 235036.825 | **41.028** | 26.160 | **134** | 134 | -- | **99.900** | 12924 | -- | -- | -- | 9.091 | -- | -- | 4.122 | -- | -- | 6.970 |
+++++++-| The Pumpkin | **25.731** | 26.571 | 94.672 | **5.552** | 5.665 | 14.890 | **26.876** | 35.367 | 33.410 | **196** | 196 | -- | **4327.273** | 5790.597 | -- | -- | -- | 98.862 | -- | -- | 14.952 | -- | -- | 24.850 |
+++++++-| Thian Hook Keng Temple Singapore | **0.927** | 1.201 | 0.832 | **0.093** | 0.121 | 0.082 | **14.915** | 20.789 | 2.750 | **138** | 138 | -- | **3298.537** | 4612.373 | -- | -- | -- | 0.081 | -- | -- | 0.008 | -- | -- | 1.130 |
+++++++-| Tsar Nikolai I | **42.803** | 78.413 | 48.499 | **8.927** | 15.394 | 9.467 | **11.124** | 18.431 | 9.790 | **98** | 98 | -- | **4928.430** | 6646.845 | -- | -- | -- | 36.280 | -- | -- | 7.836 | -- | -- | 6.530 |
+++++++-| Urban II | 59.030 | -- | 47.490 | 11.008 | -- | 9.467 | 17.601 | -- | 9.380 | 96 | -- | -- | 8889.690 | -- | -- | -- | -- | 48.214 | -- | -- | 9.586 | -- | -- | 6.920 |
+++++++-| Vercingetorix | **82.952** | 86.088 | 69.328 | **10.195** | 10.890 | 8.788 | **7.257** | 8.681 | 5.080 | **69** | 69 | -- | **1223.970** | 3105.607 | -- | -- | -- | 17.706 | -- | -- | 3.104 | -- | -- | 1.500 |
+++++++-| Yueh Hai Ching Temple Singapore | 0.627 | -- | 0.720 | 0.084 | -- | 0.098 | 5.562 | -- | 0.940 | 43 | -- | -- | 1749.280 | -- | -- | -- | -- | 0.043 | -- | -- | 0.014 | -- | -- | 0.650 |
+++++++-| **Mean** | **16.146** | 23.697 | 18.023 | **3.121** | 4.014 | 2.912 | 6920.602 | **13.088** | 5.673 | 100.559 | **104.818** | -- | **5819.728** | 7358.187 | -- | -- | -- | 13.695 | -- | -- | 1.937 | -- | -- | 2.992 |
++++++++| Alcatraz Courtyard | **0.362** | 0.617 | 0.619 | **0.093** | 0.157 | 0.160 | **3.435** | 5.482 | 1.640 | **133** | 133 | -- | **4826.713** | 6554.345 | -- | -- | -- | 0.049 | -- | -- | 0.015 | -- | -- | 0.810 |
++++++++| Alcatraz Water Tower | **0.666** | 1.480 | 0.933 | **0.370** | 0.803 | 0.518 | **4.763** | 7.397 | 2.130 | **172** | 172 | -- | **2461.337** | 3795.877 | -- | -- | -- | 0.230 | -- | -- | 0.116 | -- | -- | 0.550 |
++++++++| Buddah Tooth Relic Temple Singapore | **1.131** | 1.791 | 1.030 | **0.280** | 0.520 | 0.233 | **9.964** | 14.029 | 2.060 | **162** | 162 | -- | **2916.837** | 4289.503 | -- | -- | -- | 0.081 | -- | -- | 0.014 | -- | -- | 0.850 |
++++++++| Doge Palace Venice | 0.753 | -- | 1.163 | 0.209 | -- | 0.342 | 6.359 | -- | 3.620 | 241 | -- | -- | 22522.737 | -- | -- | -- | -- | 0.211 | -- | -- | 0.029 | -- | -- | 1 |
++++++++| Door Lund | **0.014** | 0.031 | 0.024 | **0.004** | 0.008 | 0.006 | **1.812** | 2.371 | 0.320 | **12** | 12 | -- | **4888.363** | 9262.050 | -- | -- | -- | 0.006 | -- | -- | 0.001 | -- | -- | 0.300 |
++++++++| Drinking Fountain Somewhere In Zurich | **17.193** | 25.960 | 0.031 | **0.753** | 1.117 | 0.004 | **6.058** | 8.505 | 0.330 | **14** | 14 | -- | **883.837** | 5819.655 | -- | -- | -- | 0.007 | -- | -- | 0.002 | -- | -- | 0.310 |
++++++++| East Indiaman Goteborg | **5.568** | 7.911 | 3.814 | **0.979** | 1.512 | 0.621 | **7.318** | 9.822 | 4.130 | **179** | 179 | -- | **3987.183** | 5199.790 | -- | -- | -- | 3.117 | -- | -- | 0.509 | -- | -- | 1.850 |
++++++++| Ecole Superior De Guerre | **0.349** | 51.327 | 0.318 | **0.090** | 2.574 | 0.081 | **3.418** | 11.751 | 0.720 | **35** | 35 | -- | **2380.387** | 10426.770 | -- | -- | -- | 0.024 | -- | -- | 0.005 | -- | -- | 0.340 |
++++++++| Eglise du dome | **0.351** | 0.853 | 0.808 | **0.088** | 0.232 | 0.205 | **3.799** | 5.944 | 0.910 | **85** | 85 | -- | 25099.450 | **9803.940** | -- | -- | -- | 0.037 | -- | -- | 0.010 | -- | -- | 0.270 |
++++++++| Folke Filbyter | 85.831 | **82.656** | 74.596 | **0.129** | 0.131 | 0.125 | **19.275** | 28.225 | 10.370 | **40** | 40 | -- | **1581.897** | 3755.193 | -- | -- | -- | 70.157 | -- | -- | 0.118 | -- | -- | 4.290 |
++++++++| Fort Channing Gate Singapore | **0.093** | 0.102 | 0.207 | **0.041** | 0.049 | 0.093 | **1.783** | 3.791 | 0.520 | **27** | 27 | -- | **3434.113** | 6175.880 | -- | -- | -- | 0.020 | -- | -- | 0.008 | -- | -- | 0.250 |
++++++++| Golden Statue Somewhere In Hong Kong | 0.232 | -- | 0.292 | 0.053 | -- | 0.073 | 1.829 | -- | 0.400 | 18 | -- | -- | 4095.853 | -- | -- | -- | -- | 0.031 | -- | -- | 0.004 | -- | -- | 0.270 |
++++++++| Gustav Vasa | **3.509** | 25.854 | 34.181 | **0.220** | 0.775 | 1.085 | **1.655** | 4.747 | 3.520 | **18** | 18 | -- | **709.867** | 3887.653 | -- | -- | -- | 32.266 | -- | -- | 1.145 | -- | -- | 3.150 |
++++++++| GustavIIAdolf | **47.126** | 88.194 | 67.784 | **8.536** | 13.129 | 9.714 | **11.722** | 17.220 | 13.910 | **57** | 57 | -- | **1044.170** | 2981.017 | -- | -- | -- | 58.458 | -- | -- | 8.524 | -- | -- | 11.490 |
++++++++| Jonas Ahlstromer | 48.740 | **45.265** | 50.190 | 9.823 | **9.403** | 10.888 | **11.136** | 11.337 | 10.820 | **40** | 40 | -- | **615.450** | 2421.110 | -- | -- | -- | 47.117 | -- | -- | 10.451 | -- | -- | 8.410 |
++++++++| Kings College University Of Toronto | **9.025** | 29.200 | 0.989 | **2.044** | 2.089 | 0.235 | **3.565** | 7.037 | 0.900 | **77** | 77 | -- | **1078.980** | 2856.720 | -- | -- | -- | 0.085 | -- | -- | 0.017 | -- | -- | 0.340 |
++++++++| Lund University Sphinx | 11.392 | -- | 19.522 | 2.981 | -- | 4.585 | 6.056 | -- | 4.780 | 70 | -- | -- | 2718.567 | -- | -- | -- | -- | 8.752 | -- | -- | 2.191 | -- | -- | 1.360 |
++++++++| Nijo Castle Gate | 0.839 | -- | 1.495 | 0.165 | -- | 0.286 | 7.415 | -- | 1.700 | 19 | -- | -- | 1095.177 | -- | -- | -- | -- | 0.069 | -- | -- | 0.012 | -- | -- | 0.730 |
++++++++| Pantheon Paris | -- | 0.545 | 0.192 | -- | 0.070 | 0.050 | -- | 7.791 | 1.470 | -- | 179 | -- | -- | 5594.600 | -- | -- | -- | 0.040 | -- | -- | 0.005 | -- | -- | 0.490 |
++++++++| Park Gate Clermont Ferrand | 21.388 | -- | 0.391 | 10.192 | -- | 0.125 | 7.712 | -- | 0.570 | 34 | -- | -- | 1596.940 | -- | -- | -- | -- | 0.049 | -- | -- | 0.022 | -- | -- | 0.350 |
++++++++| Plaza De Armas Santiago | 0.775 | -- | 6.782 | 0.337 | -- | 2.944 | 6.009 | -- | 7.400 | 240 | -- | -- | 6372.683 | -- | -- | -- | -- | 2.556 | -- | -- | 1.383 | -- | -- | 4.900 |
++++++++| Porta San Donato Bologna | 0.592 | -- | 2.153 | 0.107 | -- | 0.388 | 5.603 | -- | 2.280 | 141 | -- | -- | 3754.163 | -- | -- | -- | -- | 0.095 | -- | -- | 0.046 | -- | -- | 0.750 |
++++++++| Round Church Cambridge | 2.137 | -- | 2.451 | 0.927 | -- | 1.003 | 6.103 | -- | 2.660 | 92 | -- | -- | 9779.270 | -- | -- | -- | -- | 1.107 | -- | -- | 0.582 | -- | -- | 1.540 |
++++++++| Skansen Kronan Gothenburg | 0.301 | -- | 0.736 | 0.102 | -- | 0.226 | 2.906 | -- | 1.240 | 131 | -- | -- | 17153.677 | -- | -- | -- | -- | 0.026 | -- | -- | 0.008 | -- | -- | 0.670 |
++++++++| Smolny Cathedral St Petersburg | 21.775 | -- | 0.554 | 2.111 | -- | 0.051 | 11.239 | -- | 1.660 | 131 | -- | -- | 10362.787 | -- | -- | -- | -- | 0.033 | -- | -- | 0.006 | -- | -- | 0.810 |
++++++++| Some Cathedral In Barcelona | 0.651 | -- | 0.880 | 0.233 | -- | 0.315 | 6.271 | -- | 2.870 | 177 | -- | -- | 5001.973 | -- | -- | -- | -- | 0.026 | -- | -- | 0.011 | -- | -- | 0.890 |
++++++++| Sri Mariamman Singapore | 1.219 | -- | 2.302 | 0.382 | -- | 0.683 | 10.088 | -- | 4.130 | 222 | -- | -- | 5985.540 | -- | -- | -- | -- | 0.077 | -- | -- | 0.023 | -- | -- | 0.910 |
++++++++| Sri Thendayuthapani Singapore | 0.593 | -- | 46.269 | 0.150 | -- | 3.812 | 7.016 | -- | 23.370 | 98 | -- | -- | 13338.167 | -- | -- | -- | -- | 44.170 | -- | -- | 2.870 | -- | -- | 8.440 |
++++++++| Sri Veeramakaliamman Singapore | 2.404 | -- | 2.559 | 0.559 | -- | 0.597 | 12.234 | -- | 3.470 | 157 | -- | -- | 12377.927 | -- | -- | -- | -- | 0.175 | -- | -- | 0.040 | -- | -- | 0.730 |
++++++++| Statue Of Liberty | 52.358 | -- | 46.887 | 28.503 | -- | 20.012 | 235036.825 | -- | 26.160 | 134 | -- | -- | 99.900 | -- | -- | -- | -- | 9.091 | -- | -- | 4.122 | -- | -- | 6.970 |
++++++++| The Pumpkin | 25.731 | -- | 94.672 | 5.552 | -- | 14.890 | 26.876 | -- | 33.410 | 196 | -- | -- | 4327.273 | -- | -- | -- | -- | 98.862 | -- | -- | 14.952 | -- | -- | 24.850 |
++++++++| Thian Hook Keng Temple Singapore | 0.927 | -- | 0.832 | 0.093 | -- | 0.082 | 14.915 | -- | 2.750 | 138 | -- | -- | 3298.537 | -- | -- | -- | -- | 0.081 | -- | -- | 0.008 | -- | -- | 1.130 |
++++++++| Tsar Nikolai I | 42.166 | -- | 48.499 | 8.732 | -- | 9.467 | 11.210 | -- | 9.790 | 98 | -- | -- | 4929.200 | -- | -- | -- | -- | 36.280 | -- | -- | 7.836 | -- | -- | 6.530 |
++++++++| Urban II | 58.919 | -- | 47.490 | 11.006 | -- | 9.467 | 17.538 | -- | 9.380 | 96 | -- | -- | 8889.617 | -- | -- | -- | -- | 48.214 | -- | -- | 9.586 | -- | -- | 6.920 |
++++++++| Vercingetorix | 82.952 | -- | 69.328 | 10.195 | -- | 8.788 | 7.257 | -- | 5.080 | 69 | -- | -- | 1223.970 | -- | -- | -- | -- | 17.706 | -- | -- | 3.104 | -- | -- | 1.500 |
++++++++| Yueh Hai Ching Temple Singapore | 0.544 | -- | 0.720 | 0.075 | -- | 0.098 | 5.596 | -- | 0.940 | 43 | -- | -- | 1748.667 | -- | -- | -- | -- | 0.043 | -- | -- | 0.014 | -- | -- | 0.650 |
++++++++| **Mean** | **15.675** | 24.119 | 17.547 | 3.032 | **2.171** | 2.840 | 6723.050 | **9.697** | 5.595 | **102.743** | 82 | -- | 5616.606 | **5521.607** | -- | -- | -- | 13.315 | -- | -- | 1.883 | -- | -- | 2.933 |
+++++++ 
+++++++ _Seeds per cell: [0, 1, 2]._
+++++++ 
+++++++diff --git a/code/results/single_scene/summary_table.tex b/code/results/single_scene/summary_table.tex
+++++++index 258e3f2..5291012 100644
+++++++--- a/code/results/single_scene/summary_table.tex
++++++++++ b/code/results/single_scene/summary_table.tex
+++++++@@ -7,45 +7,46 @@
+++++++ \begin{tabular}{lcccccccccccccccccccccccc}
+++++++ \toprule
+++++++  & \multicolumn{3}{c}{Rot (deg)} & \multicolumn{3}{c}{Trans} & \multicolumn{3}{c}{Reproj (px)} & \multicolumn{3}{c}{Nr} & \multicolumn{3}{c}{Time (s)} & \multicolumn{3}{c}{Rot-BA (deg)} & \multicolumn{3}{c}{Trans-BA} & \multicolumn{3}{c}{Reproj-BA (px)} \\
+++++++-Scene & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) \\
++++++++Scene & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) \\
+++++++ \midrule
+++++++-Alcatraz Courtyard & \textbf{0.362} & 0.616 & 0.619 & \textbf{0.093} & 0.161 & 0.160 & \textbf{3.435} & 5.796 & 1.640 & \textbf{133} & 133 & -- & \textbf{4826.713} & 6481.757 & -- & -- & -- & 0.049 & -- & -- & 0.015 & -- & -- & 0.810 \\
+++++++-Alcatraz Water Tower & \textbf{0.666} & 7.413 & 0.933 & \textbf{0.370} & 3.437 & 0.518 & \textbf{4.763} & 8.607 & 2.130 & \textbf{172} & 172 & -- & \textbf{2461.337} & 3854.907 & -- & -- & -- & 0.230 & -- & -- & 0.116 & -- & -- & 0.550 \\
+++++++-Buddah Tooth Relic Temple Singapore & \textbf{1.131} & 12.080 & 1.030 & \textbf{0.280} & 2.308 & 0.233 & \textbf{9.964} & 19.634 & 2.060 & \textbf{162} & 162 & -- & \textbf{2916.837} & 4320.873 & -- & -- & -- & 0.081 & -- & -- & 0.014 & -- & -- & 0.850 \\
+++++++-Doge Palace Venice & \textbf{0.801} & 0.911 & 1.163 & \textbf{0.218} & 0.290 & 0.342 & \textbf{6.432} & 10.966 & 3.620 & \textbf{241} & 241 & -- & 28768.725 & \textbf{21074.950} & -- & -- & -- & 0.211 & -- & -- & 0.029 & -- & -- & 1 \\
+++++++-Door Lund & \textbf{0.014} & 14.172 & 0.024 & \textbf{0.004} & 0.952 & 0.006 & \textbf{1.812} & 11.922 & 0.320 & \textbf{12} & 12 & -- & \textbf{4888.363} & 9188.593 & -- & -- & -- & 0.006 & -- & -- & 0.001 & -- & -- & 0.300 \\
+++++++-Drinking Fountain Somewhere In Zurich & \textbf{17.193} & 17.536 & 0.031 & 0.753 & \textbf{0.740} & 0.004 & \textbf{6.058} & 6.511 & 0.330 & \textbf{14} & 14 & -- & \textbf{883.837} & 2963.647 & -- & -- & -- & 0.007 & -- & -- & 0.002 & -- & -- & 0.310 \\
+++++++-East Indiaman Goteborg & 5.568 & \textbf{5.310} & 3.814 & 0.979 & \textbf{0.959} & 0.621 & \textbf{7.318} & 9.840 & 4.130 & \textbf{179} & 179 & -- & \textbf{3987.183} & 5241.430 & -- & -- & -- & 3.117 & -- & -- & 0.509 & -- & -- & 1.850 \\
+++++++-Ecole Superior De Guerre & 0.349 & \textbf{0.284} & 0.318 & 0.090 & \textbf{0.077} & 0.081 & \textbf{3.418} & 5.349 & 0.720 & \textbf{35} & 35 & -- & \textbf{2380.387} & 4633.833 & -- & -- & -- & 0.024 & -- & -- & 0.005 & -- & -- & 0.340 \\
+++++++-Eglise du dome & \textbf{0.351} & 45.801 & 0.808 & \textbf{0.088} & 6.451 & 0.205 & \textbf{3.799} & 7.964 & 0.910 & \textbf{85} & 85 & -- & 25099.450 & \textbf{23371.730} & -- & -- & -- & 0.037 & -- & -- & 0.010 & -- & -- & 0.270 \\
+++++++-Folke Filbyter & 85.831 & \textbf{81.288} & 74.596 & \textbf{0.129} & 0.130 & 0.125 & \textbf{19.275} & 23.156 & 10.370 & \textbf{40} & 40 & -- & \textbf{1581.897} & 3703.187 & -- & -- & -- & 70.157 & -- & -- & 0.118 & -- & -- & 4.290 \\
+++++++-Fort Channing Gate Singapore & \textbf{0.093} & 11.128 & 0.207 & \textbf{0.041} & 4.004 & 0.093 & \textbf{1.783} & 6.651 & 0.520 & \textbf{27} & 27 & -- & \textbf{3434.113} & 6172.087 & -- & -- & -- & 0.020 & -- & -- & 0.008 & -- & -- & 0.250 \\
+++++++-Golden Statue Somewhere In Hong Kong & \textbf{0.232} & 0.371 & 0.292 & \textbf{0.053} & 0.074 & 0.073 & \textbf{1.829} & 2.835 & 0.400 & \textbf{18} & 18 & -- & \textbf{4095.853} & 7762.437 & -- & -- & -- & 0.031 & -- & -- & 0.004 & -- & -- & 0.270 \\
+++++++-Gustav Vasa & \textbf{3.509} & 6.099 & 34.181 & \textbf{0.220} & 0.345 & 1.085 & \textbf{1.655} & 3.087 & 3.520 & \textbf{18} & 18 & -- & \textbf{709.867} & 2543.553 & -- & -- & -- & 32.266 & -- & -- & 1.145 & -- & -- & 3.150 \\
+++++++-GustavIIAdolf & \textbf{47.126} & 77.423 & 67.784 & \textbf{8.536} & 11.962 & 9.714 & \textbf{11.722} & 15.851 & 13.910 & \textbf{57} & 57 & -- & \textbf{1044.170} & 2892.580 & -- & -- & -- & 58.458 & -- & -- & 8.524 & -- & -- & 11.490 \\
+++++++-Jonas Ahlstromer & 48.740 & \textbf{42.246} & 50.190 & \textbf{9.823} & 10.300 & 10.888 & 11.136 & \textbf{8.782} & 10.820 & \textbf{40} & 40 & -- & \textbf{615.450} & 2328.313 & -- & -- & -- & 47.117 & -- & -- & 10.451 & -- & -- & 8.410 \\
+++++++-Kings College University Of Toronto & \textbf{9.025} & 21.523 & 0.989 & 2.044 & \textbf{1.905} & 0.235 & \textbf{3.565} & 5.525 & 0.900 & \textbf{77} & 77 & -- & \textbf{1078.980} & 2886.243 & -- & -- & -- & 0.085 & -- & -- & 0.017 & -- & -- & 0.340 \\
+++++++-Lund University Sphinx & \textbf{11.392} & 33.023 & 19.522 & \textbf{2.981} & 6.889 & 4.585 & \textbf{6.056} & 12.060 & 4.780 & \textbf{70} & 70 & -- & \textbf{2718.567} & 4654.610 & -- & -- & -- & 8.752 & -- & -- & 2.191 & -- & -- & 1.360 \\
+++++++-Nijo Castle Gate & \textbf{0.839} & 1.682 & 1.495 & \textbf{0.165} & 0.326 & 0.286 & \textbf{7.415} & 12.138 & 1.700 & \textbf{19} & 19 & -- & \textbf{1095.177} & 3306.697 & -- & -- & -- & 0.069 & -- & -- & 0.012 & -- & -- & 0.730 \\
+++++++-Pantheon Paris & -- & 19.345 & 0.192 & -- & 3.065 & 0.050 & -- & 13.562 & 1.470 & -- & 179 & -- & -- & 5497.377 & -- & -- & -- & 0.040 & -- & -- & 0.005 & -- & -- & 0.490 \\
+++++++-Park Gate Clermont Ferrand & 21.388 & \textbf{19.302} & 0.391 & 10.192 & \textbf{9.397} & 0.125 & \textbf{7.712} & 7.852 & 0.570 & \textbf{34} & 34 & -- & \textbf{1596.940} & 3597.403 & -- & -- & -- & 0.049 & -- & -- & 0.022 & -- & -- & 0.350 \\
+++++++-Plaza De Armas Santiago & \textbf{0.775} & 3.327 & 6.782 & \textbf{0.337} & 1.435 & 2.944 & \textbf{6.009} & 9.463 & 7.400 & \textbf{240} & 240 & -- & \textbf{6372.683} & 6535.537 & -- & -- & -- & 2.556 & -- & -- & 1.383 & -- & -- & 4.900 \\
+++++++-Porta San Donato Bologna & \textbf{0.592} & 21.006 & 2.153 & \textbf{0.107} & 3.541 & 0.388 & \textbf{5.603} & 13.244 & 2.280 & \textbf{141} & 141 & -- & \textbf{3754.163} & 5118.030 & -- & -- & -- & 0.095 & -- & -- & 0.046 & -- & -- & 0.750 \\
+++++++-Round Church Cambridge & \textbf{2.129} & 2.490 & 2.451 & \textbf{0.920} & 1.002 & 1.003 & \textbf{6.097} & 8.546 & 2.660 & \textbf{92} & 92 & -- & \textbf{9776.630} & 12061.350 & -- & -- & -- & 1.107 & -- & -- & 0.582 & -- & -- & 1.540 \\
+++++++-Skansen Kronan Gothenburg & \textbf{0.338} & 0.375 & 0.736 & \textbf{0.116} & 0.124 & 0.226 & \textbf{3.048} & 4.799 & 1.240 & \textbf{131} & 131 & -- & 17163.940 & \textbf{16235.090} & -- & -- & -- & 0.026 & -- & -- & 0.008 & -- & -- & 0.670 \\
+++++++-Smolny Cathedral St Petersburg & \textbf{21.806} & 22.143 & 0.554 & 2.111 & \textbf{2.099} & 0.051 & \textbf{11.228} & 12.526 & 1.660 & \textbf{131} & 131 & -- & \textbf{10362.595} & 10475.740 & -- & -- & -- & 0.033 & -- & -- & 0.006 & -- & -- & 0.810 \\
+++++++-Sri Mariamman Singapore & \textbf{1.219} & 6.479 & 2.302 & \textbf{0.382} & 2.188 & 0.683 & \textbf{10.088} & 17.854 & 4.130 & \textbf{222} & 222 & -- & \textbf{5985.540} & 6485.667 & -- & -- & -- & 0.077 & -- & -- & 0.023 & -- & -- & 0.910 \\
+++++++-Sri Thendayuthapani Singapore & \textbf{0.719} & 47.075 & 46.269 & \textbf{0.179} & 3.855 & 3.812 & \textbf{7.124} & 14.133 & 23.370 & \textbf{98} & 98 & -- & \textbf{13362.090} & 13808.440 & -- & -- & -- & 44.170 & -- & -- & 2.870 & -- & -- & 8.440 \\
+++++++-Sri Veeramakaliamman Singapore & \textbf{2.334} & 29.405 & 2.559 & \textbf{0.543} & 3.868 & 0.597 & \textbf{11.965} & 28.951 & 3.470 & \textbf{157} & 157 & -- & \textbf{12392.180} & 12544.690 & -- & -- & -- & 0.175 & -- & -- & 0.040 & -- & -- & 0.730 \\
+++++++-Statue Of Liberty & 52.358 & \textbf{39.859} & 46.887 & 28.503 & \textbf{18.503} & 20.012 & 235036.825 & \textbf{41.028} & 26.160 & \textbf{134} & 134 & -- & \textbf{99.900} & 12924 & -- & -- & -- & 9.091 & -- & -- & 4.122 & -- & -- & 6.970 \\
+++++++-The Pumpkin & \textbf{25.731} & 26.571 & 94.672 & \textbf{5.552} & 5.665 & 14.890 & \textbf{26.876} & 35.367 & 33.410 & \textbf{196} & 196 & -- & \textbf{4327.273} & 5790.597 & -- & -- & -- & 98.862 & -- & -- & 14.952 & -- & -- & 24.850 \\
+++++++-Thian Hook Keng Temple Singapore & \textbf{0.927} & 1.201 & 0.832 & \textbf{0.093} & 0.121 & 0.082 & \textbf{14.915} & 20.789 & 2.750 & \textbf{138} & 138 & -- & \textbf{3298.537} & 4612.373 & -- & -- & -- & 0.081 & -- & -- & 0.008 & -- & -- & 1.130 \\
+++++++-Tsar Nikolai I & \textbf{42.803} & 78.413 & 48.499 & \textbf{8.927} & 15.394 & 9.467 & \textbf{11.124} & 18.431 & 9.790 & \textbf{98} & 98 & -- & \textbf{4928.430} & 6646.845 & -- & -- & -- & 36.280 & -- & -- & 7.836 & -- & -- & 6.530 \\
+++++++-Urban II & 59.030 & -- & 47.490 & 11.008 & -- & 9.467 & 17.601 & -- & 9.380 & 96 & -- & -- & 8889.690 & -- & -- & -- & -- & 48.214 & -- & -- & 9.586 & -- & -- & 6.920 \\
+++++++-Vercingetorix & \textbf{82.952} & 86.088 & 69.328 & \textbf{10.195} & 10.890 & 8.788 & \textbf{7.257} & 8.681 & 5.080 & \textbf{69} & 69 & -- & \textbf{1223.970} & 3105.607 & -- & -- & -- & 17.706 & -- & -- & 3.104 & -- & -- & 1.500 \\
+++++++-Yueh Hai Ching Temple Singapore & 0.627 & -- & 0.720 & 0.084 & -- & 0.098 & 5.562 & -- & 0.940 & 43 & -- & -- & 1749.280 & -- & -- & -- & -- & 0.043 & -- & -- & 0.014 & -- & -- & 0.650 \\
++++++++Alcatraz Courtyard & \textbf{0.362} & 0.617 & 0.619 & \textbf{0.093} & 0.157 & 0.160 & \textbf{3.435} & 5.482 & 1.640 & \textbf{133} & 133 & -- & \textbf{4826.713} & 6554.345 & -- & -- & -- & 0.049 & -- & -- & 0.015 & -- & -- & 0.810 \\
++++++++Alcatraz Water Tower & \textbf{0.666} & 1.480 & 0.933 & \textbf{0.370} & 0.803 & 0.518 & \textbf{4.763} & 7.397 & 2.130 & \textbf{172} & 172 & -- & \textbf{2461.337} & 3795.877 & -- & -- & -- & 0.230 & -- & -- & 0.116 & -- & -- & 0.550 \\
++++++++Buddah Tooth Relic Temple Singapore & \textbf{1.131} & 1.791 & 1.030 & \textbf{0.280} & 0.520 & 0.233 & \textbf{9.964} & 14.029 & 2.060 & \textbf{162} & 162 & -- & \textbf{2916.837} & 4289.503 & -- & -- & -- & 0.081 & -- & -- & 0.014 & -- & -- & 0.850 \\
++++++++Doge Palace Venice & 0.753 & -- & 1.163 & 0.209 & -- & 0.342 & 6.359 & -- & 3.620 & 241 & -- & -- & 22522.737 & -- & -- & -- & -- & 0.211 & -- & -- & 0.029 & -- & -- & 1 \\
++++++++Door Lund & \textbf{0.014} & 0.031 & 0.024 & \textbf{0.004} & 0.008 & 0.006 & \textbf{1.812} & 2.371 & 0.320 & \textbf{12} & 12 & -- & \textbf{4888.363} & 9262.050 & -- & -- & -- & 0.006 & -- & -- & 0.001 & -- & -- & 0.300 \\
++++++++Drinking Fountain Somewhere In Zurich & \textbf{17.193} & 25.960 & 0.031 & \textbf{0.753} & 1.117 & 0.004 & \textbf{6.058} & 8.505 & 0.330 & \textbf{14} & 14 & -- & \textbf{883.837} & 5819.655 & -- & -- & -- & 0.007 & -- & -- & 0.002 & -- & -- & 0.310 \\
++++++++East Indiaman Goteborg & \textbf{5.568} & 7.911 & 3.814 & \textbf{0.979} & 1.512 & 0.621 & \textbf{7.318} & 9.822 & 4.130 & \textbf{179} & 179 & -- & \textbf{3987.183} & 5199.790 & -- & -- & -- & 3.117 & -- & -- & 0.509 & -- & -- & 1.850 \\
++++++++Ecole Superior De Guerre & \textbf{0.349} & 51.327 & 0.318 & \textbf{0.090} & 2.574 & 0.081 & \textbf{3.418} & 11.751 & 0.720 & \textbf{35} & 35 & -- & \textbf{2380.387} & 10426.770 & -- & -- & -- & 0.024 & -- & -- & 0.005 & -- & -- & 0.340 \\
++++++++Eglise du dome & \textbf{0.351} & 0.853 & 0.808 & \textbf{0.088} & 0.232 & 0.205 & \textbf{3.799} & 5.944 & 0.910 & \textbf{85} & 85 & -- & 25099.450 & \textbf{9803.940} & -- & -- & -- & 0.037 & -- & -- & 0.010 & -- & -- & 0.270 \\
++++++++Folke Filbyter & 85.831 & \textbf{82.656} & 74.596 & \textbf{0.129} & 0.131 & 0.125 & \textbf{19.275} & 28.225 & 10.370 & \textbf{40} & 40 & -- & \textbf{1581.897} & 3755.193 & -- & -- & -- & 70.157 & -- & -- & 0.118 & -- & -- & 4.290 \\
++++++++Fort Channing Gate Singapore & \textbf{0.093} & 0.102 & 0.207 & \textbf{0.041} & 0.049 & 0.093 & \textbf{1.783} & 3.791 & 0.520 & \textbf{27} & 27 & -- & \textbf{3434.113} & 6175.880 & -- & -- & -- & 0.020 & -- & -- & 0.008 & -- & -- & 0.250 \\
++++++++Golden Statue Somewhere In Hong Kong & 0.232 & -- & 0.292 & 0.053 & -- & 0.073 & 1.829 & -- & 0.400 & 18 & -- & -- & 4095.853 & -- & -- & -- & -- & 0.031 & -- & -- & 0.004 & -- & -- & 0.270 \\
++++++++Gustav Vasa & \textbf{3.509} & 25.854 & 34.181 & \textbf{0.220} & 0.775 & 1.085 & \textbf{1.655} & 4.747 & 3.520 & \textbf{18} & 18 & -- & \textbf{709.867} & 3887.653 & -- & -- & -- & 32.266 & -- & -- & 1.145 & -- & -- & 3.150 \\
++++++++GustavIIAdolf & \textbf{47.126} & 88.194 & 67.784 & \textbf{8.536} & 13.129 & 9.714 & \textbf{11.722} & 17.220 & 13.910 & \textbf{57} & 57 & -- & \textbf{1044.170} & 2981.017 & -- & -- & -- & 58.458 & -- & -- & 8.524 & -- & -- & 11.490 \\
++++++++Jonas Ahlstromer & 48.740 & \textbf{45.265} & 50.190 & 9.823 & \textbf{9.403} & 10.888 & \textbf{11.136} & 11.337 & 10.820 & \textbf{40} & 40 & -- & \textbf{615.450} & 2421.110 & -- & -- & -- & 47.117 & -- & -- & 10.451 & -- & -- & 8.410 \\
++++++++Kings College University Of Toronto & \textbf{9.025} & 29.200 & 0.989 & \textbf{2.044} & 2.089 & 0.235 & \textbf{3.565} & 7.037 & 0.900 & \textbf{77} & 77 & -- & \textbf{1078.980} & 2856.720 & -- & -- & -- & 0.085 & -- & -- & 0.017 & -- & -- & 0.340 \\
++++++++Lund University Sphinx & 11.392 & -- & 19.522 & 2.981 & -- & 4.585 & 6.056 & -- & 4.780 & 70 & -- & -- & 2718.567 & -- & -- & -- & -- & 8.752 & -- & -- & 2.191 & -- & -- & 1.360 \\
++++++++Nijo Castle Gate & 0.839 & -- & 1.495 & 0.165 & -- & 0.286 & 7.415 & -- & 1.700 & 19 & -- & -- & 1095.177 & -- & -- & -- & -- & 0.069 & -- & -- & 0.012 & -- & -- & 0.730 \\
++++++++Pantheon Paris & -- & 0.545 & 0.192 & -- & 0.070 & 0.050 & -- & 7.791 & 1.470 & -- & 179 & -- & -- & 5594.600 & -- & -- & -- & 0.040 & -- & -- & 0.005 & -- & -- & 0.490 \\
++++++++Park Gate Clermont Ferrand & 21.388 & -- & 0.391 & 10.192 & -- & 0.125 & 7.712 & -- & 0.570 & 34 & -- & -- & 1596.940 & -- & -- & -- & -- & 0.049 & -- & -- & 0.022 & -- & -- & 0.350 \\
++++++++Plaza De Armas Santiago & 0.775 & -- & 6.782 & 0.337 & -- & 2.944 & 6.009 & -- & 7.400 & 240 & -- & -- & 6372.683 & -- & -- & -- & -- & 2.556 & -- & -- & 1.383 & -- & -- & 4.900 \\
++++++++Porta San Donato Bologna & 0.592 & -- & 2.153 & 0.107 & -- & 0.388 & 5.603 & -- & 2.280 & 141 & -- & -- & 3754.163 & -- & -- & -- & -- & 0.095 & -- & -- & 0.046 & -- & -- & 0.750 \\
++++++++Round Church Cambridge & 2.137 & -- & 2.451 & 0.927 & -- & 1.003 & 6.103 & -- & 2.660 & 92 & -- & -- & 9779.270 & -- & -- & -- & -- & 1.107 & -- & -- & 0.582 & -- & -- & 1.540 \\
++++++++Skansen Kronan Gothenburg & 0.301 & -- & 0.736 & 0.102 & -- & 0.226 & 2.906 & -- & 1.240 & 131 & -- & -- & 17153.677 & -- & -- & -- & -- & 0.026 & -- & -- & 0.008 & -- & -- & 0.670 \\
++++++++Smolny Cathedral St Petersburg & 21.775 & -- & 0.554 & 2.111 & -- & 0.051 & 11.239 & -- & 1.660 & 131 & -- & -- & 10362.787 & -- & -- & -- & -- & 0.033 & -- & -- & 0.006 & -- & -- & 0.810 \\
++++++++Some Cathedral In Barcelona & 0.651 & -- & 0.880 & 0.233 & -- & 0.315 & 6.271 & -- & 2.870 & 177 & -- & -- & 5001.973 & -- & -- & -- & -- & 0.026 & -- & -- & 0.011 & -- & -- & 0.890 \\
++++++++Sri Mariamman Singapore & 1.219 & -- & 2.302 & 0.382 & -- & 0.683 & 10.088 & -- & 4.130 & 222 & -- & -- & 5985.540 & -- & -- & -- & -- & 0.077 & -- & -- & 0.023 & -- & -- & 0.910 \\
++++++++Sri Thendayuthapani Singapore & 0.593 & -- & 46.269 & 0.150 & -- & 3.812 & 7.016 & -- & 23.370 & 98 & -- & -- & 13338.167 & -- & -- & -- & -- & 44.170 & -- & -- & 2.870 & -- & -- & 8.440 \\
++++++++Sri Veeramakaliamman Singapore & 2.404 & -- & 2.559 & 0.559 & -- & 0.597 & 12.234 & -- & 3.470 & 157 & -- & -- & 12377.927 & -- & -- & -- & -- & 0.175 & -- & -- & 0.040 & -- & -- & 0.730 \\
++++++++Statue Of Liberty & 52.358 & -- & 46.887 & 28.503 & -- & 20.012 & 235036.825 & -- & 26.160 & 134 & -- & -- & 99.900 & -- & -- & -- & -- & 9.091 & -- & -- & 4.122 & -- & -- & 6.970 \\
++++++++The Pumpkin & 25.731 & -- & 94.672 & 5.552 & -- & 14.890 & 26.876 & -- & 33.410 & 196 & -- & -- & 4327.273 & -- & -- & -- & -- & 98.862 & -- & -- & 14.952 & -- & -- & 24.850 \\
++++++++Thian Hook Keng Temple Singapore & 0.927 & -- & 0.832 & 0.093 & -- & 0.082 & 14.915 & -- & 2.750 & 138 & -- & -- & 3298.537 & -- & -- & -- & -- & 0.081 & -- & -- & 0.008 & -- & -- & 1.130 \\
++++++++Tsar Nikolai I & 42.166 & -- & 48.499 & 8.732 & -- & 9.467 & 11.210 & -- & 9.790 & 98 & -- & -- & 4929.200 & -- & -- & -- & -- & 36.280 & -- & -- & 7.836 & -- & -- & 6.530 \\
++++++++Urban II & 58.919 & -- & 47.490 & 11.006 & -- & 9.467 & 17.538 & -- & 9.380 & 96 & -- & -- & 8889.617 & -- & -- & -- & -- & 48.214 & -- & -- & 9.586 & -- & -- & 6.920 \\
++++++++Vercingetorix & 82.952 & -- & 69.328 & 10.195 & -- & 8.788 & 7.257 & -- & 5.080 & 69 & -- & -- & 1223.970 & -- & -- & -- & -- & 17.706 & -- & -- & 3.104 & -- & -- & 1.500 \\
++++++++Yueh Hai Ching Temple Singapore & 0.544 & -- & 0.720 & 0.075 & -- & 0.098 & 5.596 & -- & 0.940 & 43 & -- & -- & 1748.667 & -- & -- & -- & -- & 0.043 & -- & -- & 0.014 & -- & -- & 0.650 \\
+++++++ \midrule
+++++++-Mean & \textbf{16.146} & 23.697 & 18.023 & \textbf{3.121} & 4.014 & 2.912 & 6920.602 & \textbf{13.088} & 5.673 & 100.559 & \textbf{104.818} & -- & \textbf{5819.728} & 7358.187 & -- & -- & -- & 13.695 & -- & -- & 1.937 & -- & -- & 2.992 \\
++++++++Mean & \textbf{15.675} & 24.119 & 17.547 & 3.032 & \textbf{2.171} & 2.840 & 6723.050 & \textbf{9.697} & 5.595 & \textbf{102.743} & 82 & -- & 5616.606 & \textbf{5521.607} & -- & -- & -- & 13.315 & -- & -- & 1.883 & -- & -- & 2.933 \\
+++++++ \bottomrule
+++++++ \end{tabular}}
+++++++ \end{table*}
++++++ ```
++++++ - untracked/modified files:
++++++ ```
+++++++M code/results/single_scene/REPRO.md
+++++++ M code/results/single_scene/summary.csv
+++++++ M code/results/single_scene/summary_table.md
+++++++ M code/results/single_scene/summary_table.tex
++++++ ?? MVG_Project_Report_Ortal_Dayan.pdf
++++++ ?? "RESFM- ROBUST DEEP EQUIVARIANT STRUCTURE FROM MOTION.pdf"
++++++ ?? "claude specs/SPEC_cvpr_experiments.md"
++++++ ?? "claude specs/SPEC_sfm_datasets_setup.md"
++++++ ?? "claude specs/SPEC_single_scene_experiments.md"
++++++ ?? "claude specs/SPEC_uesfm_combined.md"
+++++++?? "claude specs/TASK_crossdataset_readiness.md"
++++++ ?? code/datasets/Euclidean
++++++ ?? tmp_ab_check/
++++++ ```
++++++ ### esfm-baseline (official ESFM)
++++++ - path: `/home/projects/bagon/ortalda/MVG/final-project/esfm-baseline`
++++++-- commit: `2c0079497bf5dda1eeb2a902b2f859c6387f213a`
+++++++- commit: `de666af53423c1dc3dd412e53c4e153c03d70922`
++++++ - **WARNING: working tree dirty.** Diff:
++++++ ```diff
++++++ (untracked files only)
++++++@@ -58,6 +969,7 @@ Generated: 2026-07-13T20:41:04
++++++ ?? code/confs/single_scene_generated/ss_esfm_Buddah_Tooth_Relic_Temple_Singapore_seed1.conf
++++++ ?? code/confs/single_scene_generated/ss_esfm_Buddah_Tooth_Relic_Temple_Singapore_seed2.conf
++++++ ?? code/confs/single_scene_generated/ss_esfm_Doge_Palace_Venice_seed1.conf
+++++++?? code/confs/single_scene_generated/ss_esfm_Doge_Palace_Venice_seed2.conf
++++++ ?? code/confs/single_scene_generated/ss_esfm_Door_Lund_seed1.conf
++++++ ?? code/confs/single_scene_generated/ss_esfm_Door_Lund_seed2.conf
++++++ ?? code/confs/single_scene_generated/ss_esfm_Drinking_Fountain_Somewhere_In_Zurich_seed1.conf
++++++@@ -70,6 +982,7 @@ Generated: 2026-07-13T20:41:04
++++++ ?? code/confs/single_scene_generated/ss_esfm_Ecole_Superior_De_Guerre_seed2.conf
++++++ ?? code/confs/single_scene_generated/ss_esfm_Eglise_du_dome_seed0.conf
++++++ ?? code/confs/single_scene_generated/ss_esfm_Eglise_du_dome_seed1.conf
+++++++?? code/confs/single_scene_generated/ss_esfm_Eglise_du_dome_seed2.conf
++++++ ?? code/confs/single_scene_generated/ss_esfm_Folke_Filbyter_seed0.conf
++++++ ?? code/confs/single_scene_generated/ss_esfm_Folke_Filbyter_seed1.conf
++++++ ?? code/confs/single_scene_generated/ss_esfm_Folke_Filbyter_seed2.conf
++++++@@ -114,16 +1027,22 @@ Generated: 2026-07-13T20:41:04
++++++ ?? code/confs/single_scene_generated/ss_esfm_Round_Church_Cambridge_seed2.conf
++++++ ?? code/confs/single_scene_generated/ss_esfm_Skansen_Kronan_Gothenburg_seed0.conf
++++++ ?? code/confs/single_scene_generated/ss_esfm_Skansen_Kronan_Gothenburg_seed1.conf
+++++++?? code/confs/single_scene_generated/ss_esfm_Skansen_Kronan_Gothenburg_seed2.conf
++++++ ?? code/confs/single_scene_generated/ss_esfm_Smolny_Cathedral_St_Petersburg_seed0.conf
++++++ ?? code/confs/single_scene_generated/ss_esfm_Smolny_Cathedral_St_Petersburg_seed1.conf
+++++++?? code/confs/single_scene_generated/ss_esfm_Smolny_Cathedral_St_Petersburg_seed2.conf
++++++ ?? code/confs/single_scene_generated/ss_esfm_Some_Cathedral_In_Barcelona_seed0.conf
+++++++?? code/confs/single_scene_generated/ss_esfm_Some_Cathedral_In_Barcelona_seed1.conf
+++++++?? code/confs/single_scene_generated/ss_esfm_Some_Cathedral_In_Barcelona_seed2.conf
++++++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Mariamman_Singapore_seed0.conf
++++++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Mariamman_Singapore_seed1.conf
++++++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Mariamman_Singapore_seed2.conf
++++++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Thendayuthapani_Singapore_seed0.conf
++++++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Thendayuthapani_Singapore_seed1.conf
+++++++?? code/confs/single_scene_generated/ss_esfm_Sri_Thendayuthapani_Singapore_seed2.conf
++++++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Veeramakaliamman_Singapore_seed0.conf
++++++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Veeramakaliamman_Singapore_seed1.conf
+++++++?? code/confs/single_scene_generated/ss_esfm_Sri_Veeramakaliamman_Singapore_seed2.conf
++++++ ?? code/confs/single_scene_generated/ss_esfm_Statue_Of_Liberty_seed0.conf
++++++ ?? code/confs/single_scene_generated/ss_esfm_Statue_Of_Liberty_seed1.conf
++++++ ?? code/confs/single_scene_generated/ss_esfm_Statue_Of_Liberty_seed2.conf
++++++@@ -137,10 +1056,14 @@ Generated: 2026-07-13T20:41:04
++++++ ?? code/confs/single_scene_generated/ss_esfm_Tsar_Nikolai_I_seed1.conf
++++++ ?? code/confs/single_scene_generated/ss_esfm_Tsar_Nikolai_I_seed2.conf
++++++ ?? code/confs/single_scene_generated/ss_esfm_Urban_II_seed0.conf
+++++++?? code/confs/single_scene_generated/ss_esfm_Urban_II_seed1.conf
+++++++?? code/confs/single_scene_generated/ss_esfm_Urban_II_seed2.conf
++++++ ?? code/confs/single_scene_generated/ss_esfm_Vercingetorix_seed0.conf
++++++ ?? code/confs/single_scene_generated/ss_esfm_Vercingetorix_seed1.conf
++++++ ?? code/confs/single_scene_generated/ss_esfm_Vercingetorix_seed2.conf
++++++ ?? code/confs/single_scene_generated/ss_esfm_Yueh_Hai_Ching_Temple_Singapore_seed0.conf
+++++++?? code/confs/single_scene_generated/ss_esfm_Yueh_Hai_Ching_Temple_Singapore_seed1.conf
+++++++?? code/confs/single_scene_generated/ss_esfm_Yueh_Hai_Ching_Temple_Singapore_seed2.conf
++++++ ```
++++++ 
++++++ ## Environment (shared venv used for BOTH methods)
++++++diff --git a/code/results/single_scene/summary.csv b/code/results/single_scene/summary.csv
++++++index e4f9f5e..0634f5d 100644
++++++--- a/code/results/single_scene/summary.csv
+++++++++ b/code/results/single_scene/summary.csv
++++++@@ -2,171 +2,143 @@ method,scene,seed,rot_err_deg,pos_err,reproj_err_px,reproj_med_px,n_cams,runtime
++++++ esfm,Alcatraz Courtyard,0,0.324288380204305,0.0854554642858637,3.379232694353053,1.9204072326560708,133,4821.4,4792.707528591156,99999.0
++++++ esfm,Alcatraz Courtyard,1,0.36138678157391557,0.09112912757839253,3.3691035794153503,1.8536696768772378,133,4825.72,4804.594581127167,99999.0
++++++ esfm,Alcatraz Courtyard,2,0.4015043649148433,0.10235190268069758,3.557242072642648,1.9198007399315369,133,4833.02,4812.242679834366,99999.0
++++++-uesfm,Alcatraz Courtyard,0,0.6370266175309793,0.1671943803031883,5.956592777085197,3.6638282340102246,133,6478.06,6448.735792636871,99999.0
++++++-uesfm,Alcatraz Courtyard,1,0.5812669610362707,0.15343994793419002,5.71686544495911,3.550076086474883,133,6483.48,6454.087207078934,99999.0
++++++-uesfm,Alcatraz Courtyard,2,0.6293740753551684,0.16343948136157807,5.713459273661585,3.459422664018691,133,6483.73,6451.153926610947,99999.0
+++++++uesfm,Alcatraz Courtyard,0,0.5509063563081962,0.14050635120746804,5.528957815039945,3.569220349534855,133,6556.5,6511.41095662117,99999.0
+++++++uesfm,Alcatraz Courtyard,1,0.6823224031499351,0.17277455831381494,5.435931436314427,3.4771386471547348,133,6552.19,6506.419185638428,99999.0
++++++ esfm,Alcatraz Water Tower,0,0.6984432687370988,0.38694796813795407,4.741313233832226,2.1058898944784543,172,2461.59,2440.421297073364,99999.0
++++++ esfm,Alcatraz Water Tower,1,0.5944425806062753,0.3287537445614408,4.5656682619915765,1.9703698309044646,172,2467.28,2454.429310798645,99999.0
++++++ esfm,Alcatraz Water Tower,2,0.7051984712289041,0.39359395644708545,4.982802041506025,2.1318655747232063,172,2455.14,2441.783402681351,99999.0
++++++-uesfm,Alcatraz Water Tower,0,18.69941055700858,8.189402666249999,12.38813738647375,6.481578998967327,172,3857.94,3831.41609287262,99999.0
++++++-uesfm,Alcatraz Water Tower,1,1.3394060108035364,0.7384271683550214,6.577617375342355,3.200105455600137,172,3887.47,3865.343374729156,99999.0
++++++-uesfm,Alcatraz Water Tower,2,2.200189184784068,1.3839747978399466,6.854463337738994,3.1730434024886014,172,3819.31,3799.390902757645,99999.0
+++++++uesfm,Alcatraz Water Tower,0,2.016029335283383,1.087082854315953,7.9502009315303575,4.413488326253207,172,3805.19,3775.908159971237,99999.0
+++++++uesfm,Alcatraz Water Tower,1,1.4259772151091428,0.7645994701417347,7.177871865477131,3.7683369835989504,172,3787.07,3763.27596282959,99999.0
+++++++uesfm,Alcatraz Water Tower,2,0.9982626119141534,0.558403777925872,7.063395425015731,3.814475911180562,172,3795.37,3773.579354286194,99999.0
++++++ esfm,Buddah Tooth Relic Temple Singapore,0,1.3132762212436329,0.31652548993670354,9.903890396062208,4.634952217191846,162,2921.45,2903.059753894806,99999.0
++++++ esfm,Buddah Tooth Relic Temple Singapore,1,0.9908742070528852,0.2335767172234499,9.836436953133868,4.661767906617449,162,2915.39,2904.387546062469,99999.0
++++++ esfm,Buddah Tooth Relic Temple Singapore,2,1.089403690069201,0.2896515999941069,10.151103921611115,4.628183275251743,162,2913.67,2902.928613901138,99999.0
++++++-uesfm,Buddah Tooth Relic Temple Singapore,0,1.9767734026333772,0.5777116480820644,15.584999477939833,8.800407928031486,162,4315.97,4296.739196300507,99999.0
++++++-uesfm,Buddah Tooth Relic Temple Singapore,1,1.4532821126601407,0.345133635718661,16.753182673783428,9.555989756393366,162,4334.28,4311.474883317947,99999.0
++++++-uesfm,Buddah Tooth Relic Temple Singapore,2,32.81105779154638,6.0024233739495125,26.564606536203048,13.925597303414058,162,4312.37,4291.042752504349,99999.0
+++++++uesfm,Buddah Tooth Relic Temple Singapore,0,1.5799948547864517,0.43886498334860274,13.484389374830494,7.719072745032001,162,4294.34,4266.538989067078,99999.0
+++++++uesfm,Buddah Tooth Relic Temple Singapore,1,1.934129180062467,0.5924991617860504,13.872322562031941,8.04977435062149,162,4303.62,4259.311170101166,99999.0
+++++++uesfm,Buddah Tooth Relic Temple Singapore,2,1.8591664615796053,0.5279087705483619,14.731758217446671,8.481044153007865,162,4270.55,4249.088565349579,99999.0
++++++ esfm,Doge Palace Venice,0,0.9741701978804636,0.2578214102266912,6.511459231891615,3.6947627756511525,241,28795.18,28746.22650837898,99999.0
++++++ esfm,Doge Palace Venice,1,0.627860879835449,0.1773446161890865,6.351890452986879,3.537139014829222,241,28742.27,28699.91308617592,99999.0
++++++-uesfm,Doge Palace Venice,0,0.9110673544913167,0.2904399918743801,10.966183720673449,6.771416746021886,241,21074.95,21017.30448675156,99999.0
+++++++esfm,Doge Palace Venice,2,0.6576305681966796,0.19114542381679772,6.214841374149415,3.591638795332466,241,10030.76,9989.16938996315,99999.0
++++++ esfm,Door Lund,0,0.01167051570787235,0.003654424593160691,1.8552512563951393,0.9492901559591532,12,4892.41,4878.351888656616,99999.0
++++++ esfm,Door Lund,1,0.009685491454411136,0.0030270923396151107,1.6462990657331118,0.9049495114630693,12,4884.51,4877.529450178146,99999.0
++++++ esfm,Door Lund,2,0.02068285629921104,0.005193423834514985,1.933833706356197,0.9506754982524039,12,4888.17,4880.201772212982,99999.0
++++++-uesfm,Door Lund,0,14.059454220682019,0.9733778175929079,11.85413335315025,6.517535151323047,12,9185.47,8715.343282699585,95000.0
++++++-uesfm,Door Lund,1,14.111993883089163,0.9618908892808385,11.853785941610944,6.504290333453439,12,9203.34,8729.614821910858,95000.0
++++++-uesfm,Door Lund,2,14.344083655704312,0.9193773651816372,12.058851721280973,6.643622749322514,12,9176.97,9162.127201795578,99999.0
+++++++uesfm,Door Lund,0,0.03084538453455712,0.007522151486969653,2.3711871496098955,1.6703501976219666,12,9262.05,9243.771874427795,99999.0
+++++++uesfm,Door Lund,1,13.889154901234134,1.0160235649641791,11.830787694675347,6.5775252925915195,12,9243.42,7848.296544790268,85000.0
++++++ esfm,Drinking Fountain Somewhere In Zurich,0,25.804179845300204,1.1242096722251314,8.219025541657992,3.3464550934365884,14,888.07,874.4874215126038,99999.0
++++++ esfm,Drinking Fountain Somewhere In Zurich,1,25.754577171528883,1.1271567319301725,8.245599860088616,3.352895635561939,14,882.33,873.2415659427643,99999.0
++++++ esfm,Drinking Fountain Somewhere In Zurich,2,0.019604308735801684,0.006780729271172703,1.7090486008047308,0.6375575382268279,14,881.11,873.6504108905792,99999.0
++++++-uesfm,Drinking Fountain Somewhere In Zurich,0,26.27952541668808,1.102546079947844,8.40971803772888,3.4593948661771203,14,2969.96,2950.480526924133,99999.0
++++++-uesfm,Drinking Fountain Somewhere In Zurich,1,26.264849988219943,1.1078406736783482,8.547474098662555,3.585334207185317,14,2960.9,2504.42572927475,85000.0
++++++-uesfm,Drinking Fountain Somewhere In Zurich,2,0.06235835436589852,0.011103715758719332,2.5743283079384534,1.5361135173722633,14,2960.08,2945.673992872238,99999.0
+++++++uesfm,Drinking Fountain Somewhere In Zurich,0,25.946100100284117,1.1178920745623426,8.471712388928934,3.6241397282821626,14,5817.37,5796.999963760376,99999.0
+++++++uesfm,Drinking Fountain Somewhere In Zurich,1,25.973070279327214,1.1166462082604052,8.538877295313494,3.6343523276029472,14,5821.94,5806.947921991348,99999.0
+++++++uesfm,Drinking Fountain Somewhere In Zurich,2,0.06926036271356541,0.01532826351119577,3.015375672575656,1.7451989053599506,14,5806.35,5792.085475206375,99999.0
++++++ esfm,East Indiaman Goteborg,0,6.033696331509855,1.0773322362193534,7.427433157074123,2.292729458380311,179,3990.25,3970.732353687286,99999.0
++++++ esfm,East Indiaman Goteborg,1,5.313967690277326,0.9356515362150493,7.350753973659207,2.2747477951684703,179,3986.66,3970.914752960205,99999.0
++++++ esfm,East Indiaman Goteborg,2,5.357164002763642,0.9254557315236661,7.174538078735992,2.2493605774605823,179,3984.64,3967.810528755188,99999.0
++++++-uesfm,East Indiaman Goteborg,0,4.992588850190341,0.9093886386013725,9.930437318497303,4.011819888270754,179,5247.16,5221.816437721252,99999.0
++++++-uesfm,East Indiaman Goteborg,1,5.382092262718057,0.9927766021898647,10.035611563702533,3.912004668740913,179,5245.65,5210.653891563416,99999.0
++++++-uesfm,East Indiaman Goteborg,2,5.554715231405653,0.9752252406138763,9.5551521706325,3.7417116053580193,179,5231.48,5207.815300703049,99999.0
+++++++uesfm,East Indiaman Goteborg,0,5.365772057886591,0.9552115949640507,9.655123223073948,4.174486084237091,179,5203.6,5178.753549814224,99999.0
+++++++uesfm,East Indiaman Goteborg,1,10.45533482189932,2.0687908277631517,9.98932850274456,4.110771460953736,179,5195.98,5174.958552598953,99999.0
+++++++uesfm,East Indiaman Goteborg,2,5.872735570127935,1.0672842053553182,10.745053220477157,4.480528977857714,179,5194.7,5173.540862560272,99999.0
++++++ esfm,Ecole Superior De Guerre,0,0.4863901164687255,0.1239377922325492,3.750666445958877,2.0528113108131016,35,2385.33,2371.708672523499,99999.0
++++++ esfm,Ecole Superior De Guerre,1,0.3031794677170201,0.07921628898965839,3.2456935905644406,1.730123845260616,35,2379.34,2371.191816806793,99999.0
++++++ esfm,Ecole Superior De Guerre,2,0.2580439805296837,0.06772089648290676,3.257951412507941,1.687291979251782,35,2376.49,2368.935952663422,99999.0
++++++-uesfm,Ecole Superior De Guerre,0,0.28243957678843373,0.0778675008788964,5.291358032458062,3.601773151002547,35,4647.51,4622.850021600723,99999.0
++++++-uesfm,Ecole Superior De Guerre,1,0.25507313740145815,0.06695766472508183,5.170495431958795,3.5179325723573998,35,4636.07,4616.890106916428,99999.0
++++++-uesfm,Ecole Superior De Guerre,2,0.3137499276035537,0.08683967313665406,5.583934687507232,3.7813953498055604,35,4617.92,4602.193422079086,99999.0
+++++++uesfm,Ecole Superior De Guerre,0,51.32681795107141,2.57407526776939,11.751384768771835,6.449293516058521,35,10426.77,10405.3174200058,99999.0
++++++ esfm,Eglise du dome,0,0.3242410065346163,0.0824797769992069,3.906129661328729,1.3512292775673158,85,25054.74,25028.64922428131,99999.0
++++++ esfm,Eglise du dome,1,0.3771623368290559,0.09432763361451474,3.692490941110161,1.297032357665518,85,25144.16,25116.01952123642,99999.0
++++++-uesfm,Eglise du dome,0,45.800562123362404,6.451383786272971,7.963674120656965,3.986938865913781,85,23371.73,23333.49408817291,99999.0
+++++++esfm,Eglise du dome,2,0.45623311880194967,0.11755570327842058,4.363674496823582,1.384257241090686,85,8694.98,8671.502047538757,99999.0
+++++++uesfm,Eglise du dome,0,0.8528700994829498,0.23193079249562099,5.944108200898114,2.9306630821877517,85,9803.94,9764.585114717484,99999.0
++++++ esfm,Folke Filbyter,0,89.03873163576341,0.12895560134543632,17.986624161769267,7.423093846189277,40,1582.8,1573.360969305038,99999.0
++++++ esfm,Folke Filbyter,1,84.74740427947457,0.13132264058966686,17.98706393015447,7.385900243615879,40,1579.13,1572.079373121262,99999.0
++++++ esfm,Folke Filbyter,2,83.70817707245973,0.1261878953205589,21.85200129525089,6.324245998829542,40,1583.76,1576.207699537277,99999.0
++++++-uesfm,Folke Filbyter,0,80.34590333370599,0.13169898199385566,22.83327162619912,11.711593158296864,40,3701.49,3129.452286243439,85000.0
++++++-uesfm,Folke Filbyter,1,81.28676564061475,0.1282318052949447,24.480394314414323,9.74969805057316,40,3698.11,3681.891911745071,99999.0
++++++-uesfm,Folke Filbyter,2,82.23270227083374,0.1312531386108397,22.153947807931974,10.961536903594071,40,3709.96,3695.84988451004,99999.0
+++++++uesfm,Folke Filbyter,0,84.55526303793802,0.13091275274611416,29.190267154194732,11.160073225547443,40,3747.65,3727.990335702896,99999.0
+++++++uesfm,Folke Filbyter,1,84.87047537526948,0.13112291273275417,32.36503566471496,13.743964264262598,40,3743.53,3728.497133255005,99999.0
+++++++uesfm,Folke Filbyter,2,78.54375401873024,0.1316690349103016,23.11854509839131,10.94439865744236,40,3774.4,3007.95537352562,80000.0
++++++ esfm,Fort Channing Gate Singapore,0,0.10143646071220622,0.04531774433800129,1.7158776598376824,0.8165178663538488,27,3438.39,3428.288990497589,99999.0
++++++ esfm,Fort Channing Gate Singapore,1,0.08873951454303304,0.039001318240256114,1.783712095722291,0.8156020254191567,27,3430.88,3424.490573883057,99999.0
++++++ esfm,Fort Channing Gate Singapore,2,0.09031224636384508,0.03957289173255022,1.850800887931411,0.820658413919779,27,3433.07,3426.547735452652,99999.0
++++++-uesfm,Fort Channing Gate Singapore,0,33.158343156383296,11.908537566474344,12.69248500308198,5.438758486477886,27,6209.29,6192.107179880142,99999.0
++++++-uesfm,Fort Channing Gate Singapore,1,0.07319798057764343,0.03541606344760984,3.4819579894798927,2.2085534812612853,27,6159.7,6144.797720909119,99999.0
++++++-uesfm,Fort Channing Gate Singapore,2,0.15394831153888047,0.0686769122909015,3.77993377770425,2.2995094868650368,27,6147.27,6132.745970487595,99999.0
+++++++uesfm,Fort Channing Gate Singapore,0,0.10214634402234797,0.048923384893640356,3.7907267588398517,2.449846070018404,27,6175.88,6156.048281908035,99999.0
+++++++uesfm,Fort Channing Gate Singapore,1,0.14219758622796153,0.06357969294387673,3.5066994916782015,2.1555120267718575,27,6191.97,6176.711992740631,99999.0
++++++ esfm,Golden Statue Somewhere In Hong Kong,0,0.23894829924734365,0.053817571637812096,1.7239352536484445,0.744089758310296,18,4096.74,4087.775670289993,99999.0
++++++ esfm,Golden Statue Somewhere In Hong Kong,1,0.2233990458438394,0.053061028269618456,1.8936878914364637,0.7746436618607616,18,4096.28,4088.994391202927,99999.0
++++++ esfm,Golden Statue Somewhere In Hong Kong,2,0.23397914670516012,0.051137498616263405,1.87070255923672,0.7763877145267694,18,4094.54,4087.674698591232,99999.0
++++++-uesfm,Golden Statue Somewhere In Hong Kong,0,0.36074365888948035,0.06894135246785482,2.939092008443413,1.863811153764833,18,7761.69,7746.569520235062,99999.0
++++++-uesfm,Golden Statue Somewhere In Hong Kong,1,0.35638930493960375,0.06132592684828702,2.686157398440538,1.735571387461727,18,7759.16,7745.389947891235,99999.0
++++++-uesfm,Golden Statue Somewhere In Hong Kong,2,0.3955566726941678,0.09197790539367184,2.878341172309852,1.8164738202745854,18,7766.46,7751.458931922913,99999.0
++++++ esfm,Gustav Vasa,0,3.4881363194229826,0.2273181104934767,1.7369784534725496,0.9521606063342205,18,711.53,703.3091380596161,99999.0
++++++ esfm,Gustav Vasa,1,3.4307883498221035,0.21432466556133864,1.5556448596237022,0.9232184455756751,18,710.28,703.9906163215637,99999.0
++++++ esfm,Gustav Vasa,2,3.6089062562451386,0.2196578287395002,1.673803634770729,0.9429755567757121,18,707.79,701.5903799533844,99999.0
++++++-uesfm,Gustav Vasa,0,5.828154714648938,0.32510821331670736,3.0711042143581646,2.4170166616864357,18,2566.0,2547.931907176971,99999.0
++++++-uesfm,Gustav Vasa,1,6.515239842974278,0.35941242702736925,3.089677100739632,2.4476428398062584,18,2530.9,1385.089058637619,55000.0
++++++-uesfm,Gustav Vasa,2,5.954036423569478,0.34942942401850696,3.101296679577594,2.400483554095863,18,2533.76,2520.798699855804,99999.0
+++++++uesfm,Gustav Vasa,0,36.645536646380805,1.0070193571286277,4.980488290477233,3.197977230260218,18,2562.94,2419.297919034958,95000.0
+++++++uesfm,Gustav Vasa,1,36.4914104514929,1.03254185606701,5.377024289258223,3.4818421684593366,18,4566.81,4543.375423431396,99999.0
+++++++uesfm,Gustav Vasa,2,4.4238740734834625,0.284762443333143,3.8820183248893803,2.8663959682636566,18,4533.21,4516.61039352417,99999.0
++++++ esfm,GustavIIAdolf,0,46.92758389264633,8.570499881283022,11.684017169857798,5.131125459552879,57,1047.82,1038.018162250519,99999.0
++++++ esfm,GustavIIAdolf,1,47.21999673071185,8.515052769156952,11.695065125063977,5.091617086264161,57,1041.41,1034.5799472332,99999.0
++++++ esfm,GustavIIAdolf,2,47.2312223396789,8.523261295519099,11.787583212607423,5.2715388868550574,57,1043.28,1036.478415489197,99999.0
++++++-uesfm,GustavIIAdolf,0,89.37607436538163,13.276440526172165,15.53883714368065,5.864332719302596,57,2904.42,2881.687849521637,99999.0
++++++-uesfm,GustavIIAdolf,1,70.16367503302509,11.555242253627961,15.162174576474532,6.9709512967490355,57,2884.62,2870.103493452072,99999.0
++++++-uesfm,GustavIIAdolf,2,72.72996778765942,11.054260851523775,16.85269001998736,9.009102977660282,57,2888.7,2874.096453428268,99999.0
+++++++uesfm,GustavIIAdolf,0,85.51179144359693,12.792264407383987,17.018024925092185,8.196266635685827,57,2973.35,2956.916826248169,99999.0
+++++++uesfm,GustavIIAdolf,1,88.9494736685426,13.260723460498589,17.699982689478404,8.888010081099381,57,2995.3,2976.055320978165,99999.0
+++++++uesfm,GustavIIAdolf,2,90.12096202318821,13.335169188185864,16.941538974035925,7.772181288890227,57,2974.4,2960.110015630722,99999.0
++++++ esfm,Jonas Ahlstromer,0,64.46662959289324,11.985605264945464,12.602584968179986,2.9168445332917745,40,614.88,607.3498430252075,99999.0
++++++ esfm,Jonas Ahlstromer,1,38.42510202464649,8.741159740710629,10.25268858530355,3.7024742418884333,40,618.73,611.5567183494568,99999.0
++++++ esfm,Jonas Ahlstromer,2,43.32891670826628,8.741053567623224,10.553031866494822,3.031717304343782,40,612.74,606.6802380084991,99999.0
++++++-uesfm,Jonas Ahlstromer,0,42.30073879858729,10.272232427418913,9.098970988131883,3.992857074878733,40,2339.57,1280.604686737061,55000.0
++++++-uesfm,Jonas Ahlstromer,1,42.18412804385421,10.310808592826955,8.453808908218651,3.4329852879523464,40,2334.33,2320.362742424011,99999.0
++++++-uesfm,Jonas Ahlstromer,2,42.25301394187691,10.315519611163806,8.793789077520445,3.8046125347596442,40,2311.04,2297.139800786972,99999.0
+++++++uesfm,Jonas Ahlstromer,0,45.264596178655594,9.403212752141883,11.336617283269852,4.728596140494824,40,2421.11,2401.221394062042,99999.0
+++++++uesfm,Jonas Ahlstromer,1,51.490398556789884,10.860548616493958,12.821407884052217,5.012487046810724,40,2428.25,2415.139223337173,99999.0
+++++++uesfm,Jonas Ahlstromer,2,42.99784237868634,10.470910789899067,9.274184391894982,4.4612757046777345,40,2394.6,2381.585824012756,99999.0
++++++ esfm,Kings College University Of Toronto,0,14.400461796497643,3.2653370282498866,4.064353916685246,1.6753245173381437,77,1074.94,1066.306197404861,99999.0
++++++ esfm,Kings College University Of Toronto,1,11.956699445712928,2.705452710546247,3.9727144140636783,1.5797087908756111,77,1071.64,1064.129351377487,99999.0
++++++ esfm,Kings College University Of Toronto,2,0.7191591237413282,0.16178494376153177,2.656807607821862,1.1949677972638288,77,1090.36,1082.987608671188,99999.0
++++++-uesfm,Kings College University Of Toronto,0,31.932532001930838,2.9775501815475,6.099555590572752,3.8717713150174773,77,2884.23,2866.194714784622,99999.0
++++++-uesfm,Kings College University Of Toronto,1,2.4241467960672063,0.41577835428574456,4.631696476065898,3.148046400399828,77,2899.51,2884.114916801453,99999.0
++++++-uesfm,Kings College University Of Toronto,2,30.212552334890066,2.3209884971628156,5.844709739288878,3.754268841443037,77,2874.99,2860.901805400848,99999.0
+++++++uesfm,Kings College University Of Toronto,0,29.200261744017546,2.0893308856114428,7.037455848998898,4.8227457854711675,77,2856.72,2838.923613071442,99999.0
+++++++uesfm,Kings College University Of Toronto,1,7.107988087620218,1.3235014088907995,5.965363878755208,4.179664786063016,77,2852.71,2839.42994761467,99999.0
++++++ esfm,Lund University Sphinx,0,13.661465370705821,3.5591094267381553,6.242667318159146,2.715620555070548,70,2718.96,2709.5872194767,99999.0
++++++ esfm,Lund University Sphinx,1,12.524302160109885,3.2130942199657664,6.1182420215295075,2.7468468987195953,70,2722.86,2715.36418390274,99999.0
++++++ esfm,Lund University Sphinx,2,7.98961938632192,2.1710959765491817,5.807289242508196,2.893370267054225,70,2713.88,2706.478076219559,99999.0
++++++-uesfm,Lund University Sphinx,0,15.165716414733652,3.5564827651349713,10.803408462752863,6.730422995453065,70,4644.92,4626.135177612305,99999.0
++++++-uesfm,Lund University Sphinx,1,68.88899086055604,13.548881652837478,14.336626262363456,7.626356837439143,70,4703.31,4687.50173997879,99999.0
++++++-uesfm,Lund University Sphinx,2,15.014326075576053,3.5607694164340526,11.039432108003597,6.81081192121135,70,4615.6,4600.294319868088,99999.0
+++++++uesfm,Lund University Sphinx,0,42.95863056778373,8.719684065980621,14.999945982069319,8.361965506089375,70,4705.61,4681.787243127823,99999.0
++++++ esfm,Nijo Castle Gate,0,0.6521524412276166,0.13305866403975986,7.404953325705311,3.0629804954987674,19,1097.29,1084.301982402802,99999.0
++++++ esfm,Nijo Castle Gate,1,0.8564043952718835,0.16950374865879153,7.358076475544799,3.1235223975185864,19,1094.89,1084.640900611877,99999.0
++++++ esfm,Nijo Castle Gate,2,1.0089926380330574,0.19095470213426863,7.481049991287358,3.1671779658403687,19,1093.35,1083.446812152863,99999.0
++++++-uesfm,Nijo Castle Gate,0,1.6923297320319277,0.3246009364794658,11.792009654594867,6.945622003724464,19,3310.13,3288.75860619545,99999.0
++++++-uesfm,Nijo Castle Gate,1,1.6697791743236226,0.32440429939345133,12.704681259536075,7.466128811395402,19,3308.19,3290.209539175034,99999.0
++++++-uesfm,Nijo Castle Gate,2,1.682403384345942,0.32817821817561266,11.917977420973372,7.102279949089778,19,3301.77,3284.964460611343,99999.0
++++++-uesfm,Pantheon Paris,0,0.3823220210788846,0.05987691585642601,7.649321203280243,4.797127317249956,179,5506.88,5473.694003582001,99999.0
++++++-uesfm,Pantheon Paris,1,45.61471478909275,6.671068486991568,16.912359249810606,6.602415648206237,179,5491.31,5459.329566717148,99999.0
++++++-uesfm,Pantheon Paris,2,12.03796171625132,2.465102828965073,16.125095834055482,6.860965809143959,179,5493.94,5463.555550813675,99999.0
+++++++uesfm,Nijo Castle Gate,0,8.641303631787597,2.791258638759238,20.96738948203601,10.297911838853688,19,3241.33,3220.324777603149,99999.0
+++++++uesfm,Pantheon Paris,0,0.5453004964472504,0.06999784414241637,7.790870638815493,5.057509442480535,179,5594.6,5562.967379331589,99999.0
+++++++uesfm,Pantheon Paris,1,0.472097972346089,0.062163803587728925,7.194706475909503,4.639524309087925,179,5536.74,5501.721322774887,99999.0
++++++ esfm,Park Gate Clermont Ferrand,0,19.399216821978534,9.438471236639753,7.521969683469249,3.892136696028138,34,1597.96,1588.07075381279,99999.0
++++++ esfm,Park Gate Clermont Ferrand,1,19.325402297843297,9.417998214523331,7.465007292717891,3.849088919564891,34,1596.87,1589.703137874603,99999.0
++++++ esfm,Park Gate Clermont Ferrand,2,25.440611579781365,11.720641869372841,8.15032188787613,4.0871739024940315,34,1595.99,1588.841923236847,99999.0
++++++-uesfm,Park Gate Clermont Ferrand,0,19.27114964212729,9.381930985797803,7.856513901068924,4.317004881495189,34,3598.69,3580.396904230118,99999.0
++++++-uesfm,Park Gate Clermont Ferrand,1,19.288145181777256,9.394362801959678,7.762082007624792,4.207851936960509,34,3592.66,3577.494926214218,99999.0
++++++-uesfm,Park Gate Clermont Ferrand,2,19.345727373971844,9.415066249707772,7.936652645099164,4.425507919979627,34,3600.86,3581.473034620285,99999.0
++++++ esfm,Plaza De Armas Santiago,0,0.7341628728579594,0.32156050159900956,6.000154308231358,2.679971945741675,240,6375.02,6333.455562829971,99999.0
++++++ esfm,Plaza De Armas Santiago,1,0.5176165701053533,0.22744923951256435,5.661576462867733,2.533952088248649,240,6373.68,6336.030319213867,99999.0
++++++ esfm,Plaza De Armas Santiago,2,1.0740077415291467,0.46304622323619105,6.365716833190761,2.733195456670397,240,6369.35,6331.436369419098,99999.0
++++++-uesfm,Plaza De Armas Santiago,0,6.6772176498607925,2.909933722327364,11.384209379387066,5.623987151554922,240,6540.76,6491.619550704956,99999.0
++++++-uesfm,Plaza De Armas Santiago,1,1.3364898126442175,0.5686650140764532,7.951759184923103,3.933795015892702,240,6536.61,6482.320575714111,99999.0
++++++-uesfm,Plaza De Armas Santiago,2,1.9671483940705004,0.8276654865349118,9.054185015150745,4.441179120460681,240,6529.24,6477.723381996155,99999.0
++++++ esfm,Porta San Donato Bologna,0,0.5746001989174886,0.10608894230214515,5.691906895070585,2.4183460353027986,141,3751.75,3734.281898736954,99999.0
++++++ esfm,Porta San Donato Bologna,1,0.8978003419853328,0.14474404936935045,5.682379581165418,2.563141107750346,141,3758.51,3742.688964605331,99999.0
++++++ esfm,Porta San Donato Bologna,2,0.30440272958620346,0.07026546193502164,5.435311727517131,2.1626312981864086,141,3752.23,3738.056202173233,99999.0
++++++-uesfm,Porta San Donato Bologna,0,1.1269164513732723,0.19275737956525044,10.071225650489282,5.557466375146339,141,5111.57,5083.21807050705,99999.0
++++++-uesfm,Porta San Donato Bologna,1,60.66366912240625,10.234123128225525,19.593360464637342,9.305574874002819,141,5125.57,5098.085618019104,99999.0
++++++-uesfm,Porta San Donato Bologna,2,1.2271289404198877,0.19605254084885107,10.066750816092268,5.595831381287904,141,5116.95,5093.473650932312,99999.0
++++++ esfm,Round Church Cambridge,0,2.1173198662081005,0.9129854085205968,5.854320175654564,1.9683268910531708,92,9788.65,9759.719386577606,99999.0
++++++ esfm,Round Church Cambridge,1,2.1415279636494953,0.9267927094737857,6.33905951937897,2.0439997148640834,92,9764.61,9741.356333732605,99999.0
++++++-uesfm,Round Church Cambridge,0,2.5434984075631974,1.0219048587087352,8.914080922081759,4.131734829440289,92,12088.06,12047.02147889137,99999.0
++++++-uesfm,Round Church Cambridge,1,2.435781054756339,0.9827352719070902,8.178482224933683,3.8400051230290893,92,12034.64,11993.72608184814,99999.0
+++++++esfm,Round Church Cambridge,2,2.1507951248804917,0.9401429799233081,6.115230559713554,2.0517209910251797,92,9784.55,9763.525354623795,99999.0
++++++ esfm,Skansen Kronan Gothenburg,0,0.3380139741078169,0.1158931606908392,3.0478862369293624,1.5340801117570733,131,17163.94,17140.15425729752,99999.0
++++++-uesfm,Skansen Kronan Gothenburg,0,0.37472494768162085,0.1243676312212316,4.798736338080396,3.348916057055834,131,16235.09,16202.21351981163,99999.0
+++++++esfm,Skansen Kronan Gothenburg,1,0.21516831826827018,0.07458182746108981,2.786705021535272,1.4619994473169777,131,17129.62,17109.18022465706,99999.0
+++++++esfm,Skansen Kronan Gothenburg,2,0.3508356515622903,0.11667319277987132,2.8836839507901773,1.5216584205042176,131,17167.47,17142.66050887108,99999.0
++++++ esfm,Smolny Cathedral St Petersburg,0,21.81278753416543,2.110667559831311,11.2315357584775,6.379259553776089,131,10364.67,10320.51852750778,99999.0
++++++ esfm,Smolny Cathedral St Petersburg,1,21.799828748358873,2.111313741354268,11.225141339870355,6.36605939788532,131,10360.52,10319.01374697685,99999.0
++++++-uesfm,Smolny Cathedral St Petersburg,0,22.143392894578028,2.0985584569799265,12.52645725687386,7.895265841408903,131,10475.74,10416.62969732285,99999.0
+++++++esfm,Smolny Cathedral St Petersburg,2,21.71090600320656,2.110307413725059,11.259799763291888,6.4136596726420905,131,10363.17,10320.08415436745,99999.0
+++++++esfm,Some Cathedral In Barcelona,0,0.6775058368464424,0.24064262464083658,6.20165095815897,3.7253636674062887,177,5002.81,4985.274734258652,99999.0
+++++++esfm,Some Cathedral In Barcelona,1,0.6722147495861831,0.2423973792150467,6.350786792749083,3.720536755931007,177,5009.15,4991.552013158798,99999.0
+++++++esfm,Some Cathedral In Barcelona,2,0.6046048207068493,0.21699252142655406,6.261075100347952,3.7258171467923633,177,4993.96,4975.845166683197,99999.0
++++++ esfm,Sri Mariamman Singapore,0,1.4920493985513708,0.4412389781578158,10.406406306167431,6.369585247957454,222,5987.98,5970.066485881805,99999.0
++++++ esfm,Sri Mariamman Singapore,1,1.1345564396944785,0.3608906720626337,9.703927906328842,5.885963785879842,222,5993.62,5975.365797519684,99999.0
++++++ esfm,Sri Mariamman Singapore,2,1.0293707921546553,0.3426709250774071,10.15220066698994,6.067537729054635,222,5975.02,5958.786058187485,99999.0
++++++-uesfm,Sri Mariamman Singapore,0,16.417883129790784,5.526406595652791,19.962711816743173,12.18830031311293,222,6531.45,6494.750840425491,99999.0
++++++-uesfm,Sri Mariamman Singapore,1,1.3793173305528281,0.5014534676018023,16.10857603648069,10.651562750176396,222,6467.65,6426.201541185379,99999.0
++++++-uesfm,Sri Mariamman Singapore,2,1.639001766282598,0.536397596737868,17.490398022274007,11.439337896248928,222,6457.9,6427.717175960541,99999.0
++++++ esfm,Sri Thendayuthapani Singapore,0,0.7190396606626565,0.17922550285921848,7.12365374281766,2.231392236341273,98,13362.09,13320.09781932831,99999.0
++++++-uesfm,Sri Thendayuthapani Singapore,0,47.0750455982536,3.8553135654690074,14.133068630999212,7.117179633590087,98,13808.44,13754.94554877281,99999.0
+++++++esfm,Sri Thendayuthapani Singapore,1,0.5350953449295449,0.1261925950852554,6.4467364012557615,2.04778550672059,98,13347.85,13309.03068971634,99999.0
+++++++esfm,Sri Thendayuthapani Singapore,2,0.5262093764563488,0.14374074415280041,7.476614790858722,2.2575164196527426,98,13304.56,13262.61834168434,99999.0
++++++ esfm,Sri Veeramakaliamman Singapore,0,2.333869921178247,0.542847254574005,11.965270419451508,5.472877662905037,157,12392.18,12365.15180635452,99999.0
++++++-uesfm,Sri Veeramakaliamman Singapore,0,29.40487246624615,3.8676026992817505,28.950500021302087,14.32295224594287,157,12544.69,12504.26410722733,99999.0
+++++++esfm,Sri Veeramakaliamman Singapore,1,2.4804161216418734,0.5731176371311499,12.150997217479388,5.5309200508775875,157,12387.11,12363.29210400581,99999.0
+++++++esfm,Sri Veeramakaliamman Singapore,2,2.3974569344813434,0.5599333960859275,12.58650507397198,5.711763361742226,157,12354.49,12330.03034973145,99999.0
++++++ esfm,Statue Of Liberty,0,52.35559762033152,28.241594953600423,688383.1603142131,683624.423417858,134,102.64,43.32754778862,45000.0
++++++ esfm,Statue Of Liberty,1,52.36920727661949,28.627801879821945,9277.715116667516,9303.546387753544,134,99.78,59.98358988761902,65000.0
++++++ esfm,Statue Of Liberty,2,52.3478568435799,28.64052858422677,7449.598084265948,7416.803834848509,134,97.28,3.520912647247314,0.0
++++++-uesfm,Statue Of Liberty,0,40.92379801166407,19.915794833689585,48.781849500680366,25.904882064330994,134,12927.11,12902.23110127449,99999.0
++++++-uesfm,Statue Of Liberty,1,38.79484508812604,17.090491683338875,33.27459817133403,17.608726936833005,134,12920.89,12896.32496738434,99999.0
++++++ esfm,The Pumpkin,0,27.59459354015253,5.94835892276036,27.52772578222832,9.11126349703041,196,4330.9,4316.109169960022,99999.0
++++++ esfm,The Pumpkin,1,26.54397520858153,5.701234509093555,27.342758332715125,8.907676505669826,196,4327.98,4314.043692827225,99999.0
++++++ esfm,The Pumpkin,2,23.05395646123413,5.007378502393024,25.75879879971585,7.898438464943201,196,4322.94,4310.356798410416,99999.0
++++++-uesfm,The Pumpkin,0,25.490608822831348,5.440568114393512,34.198429037653845,15.6024365790899,196,5793.03,5767.982522964478,99999.0
++++++-uesfm,The Pumpkin,1,27.26556900533122,5.820572144056583,35.6951097814524,16.572200360468877,196,5785.61,5760.521738529205,99999.0
++++++-uesfm,The Pumpkin,2,26.95722616022698,5.73349875876088,36.207441893083136,17.07678356860021,196,5793.15,5768.243654251099,99999.0
++++++ esfm,Thian Hook Keng Temple Singapore,0,1.0653505173922067,0.1079565219850111,15.054152885837562,5.81055046268767,138,3299.26,3281.187238931656,99999.0
++++++ esfm,Thian Hook Keng Temple Singapore,1,0.8390930608109779,0.08272919436528102,14.587041005106803,5.705702663819475,138,3306.93,3283.257929086685,99999.0
++++++ esfm,Thian Hook Keng Temple Singapore,2,0.8760513596629002,0.08764395630143504,15.102705362055175,5.7567965038255124,138,3289.42,3278.173619031906,99999.0
++++++-uesfm,Thian Hook Keng Temple Singapore,0,1.146131162829257,0.10799243740327723,20.24888615229727,10.297158058757331,138,4626.48,4595.79723072052,99999.0
++++++-uesfm,Thian Hook Keng Temple Singapore,1,1.2072956626750737,0.12821580405567637,21.242388560332195,10.655698470690366,138,4600.34,4578.699931621552,99999.0
++++++-uesfm,Thian Hook Keng Temple Singapore,2,1.251054659622132,0.12561972972485672,20.875018460120653,10.416875106878496,138,4610.3,4581.792747020721,99999.0
++++++ esfm,Tsar Nikolai I,0,42.507553768489565,8.869914103947062,10.846690028504929,4.007961025902478,98,4932.15,4920.938139438629,99999.0
++++++ esfm,Tsar Nikolai I,1,43.09835370739786,8.983133733509343,11.402130572143294,4.032588092580996,98,4924.71,4914.929541349411,99999.0
++++++-uesfm,Tsar Nikolai I,0,75.94880014543322,15.146573953653359,15.307363921859029,6.734065570805788,98,6652.35,6629.384099006653,99999.0
++++++-uesfm,Tsar Nikolai I,1,80.87807139992418,15.64102228288136,21.553650044939403,9.179513384124702,98,6641.34,6623.797205209732,99999.0
+++++++esfm,Tsar Nikolai I,2,40.892180545819315,8.342265568650971,11.381967217434166,4.272306098513181,98,4930.74,4920.495186090469,99999.0
++++++ esfm,Urban II,0,59.0297968076411,11.008489348760156,17.600594623525765,5.400933781867582,96,8889.69,8878.01452589035,99999.0
+++++++esfm,Urban II,1,58.947927470837534,11.025374954076108,17.465515658589908,5.68378594242137,96,8885.17,8876.618469953537,99999.0
+++++++esfm,Urban II,2,58.77798771298748,10.982786678935327,17.546556385759054,5.767405629316944,96,8893.99,8885.34243106842,99999.0
++++++ esfm,Vercingetorix,0,79.07478057258302,9.776081681696928,6.649989325889541,1.7432748253588333,69,1223.75,1212.340968608856,99999.0
++++++ esfm,Vercingetorix,1,80.09313371434186,9.858626097407651,6.426263444651325,1.6979463697629655,69,1222.8,1213.763489961624,99999.0
++++++ esfm,Vercingetorix,2,89.68787080902364,10.950615257181747,8.695412468830675,3.360645595149667,69,1225.36,1215.808262825012,99999.0
++++++-uesfm,Vercingetorix,0,78.95942988866727,10.356244187384963,8.07065603849916,3.372297052368834,69,3105.42,3085.208213567734,99999.0
++++++-uesfm,Vercingetorix,1,89.99175121301754,11.178484677344343,8.929255781006072,4.302365177465691,69,3105.88,3091.564247369766,99999.0
++++++-uesfm,Vercingetorix,2,89.31151857112468,11.134780185118505,9.044030033429575,4.351952473195928,69,3105.52,3089.0203332901,99999.0
++++++ esfm,Yueh Hai Ching Temple Singapore,0,0.6268598783684821,0.08360284283022794,5.56223941139456,2.132543748253723,43,1749.28,1738.747783660889,99999.0
+++++++esfm,Yueh Hai Ching Temple Singapore,1,0.5064956072004329,0.07235771132992698,5.30294026596974,2.06113770612239,43,1750.4,1739.704204082489,99999.0
+++++++esfm,Yueh Hai Ching Temple Singapore,2,0.49998151063409024,0.07009863154178562,5.92393490371475,2.2317763685579015,43,1746.32,1738.905562877655,99999.0
++++++diff --git a/code/results/single_scene/summary_table.md b/code/results/single_scene/summary_table.md
++++++index bd3f673..1cac978 100644
++++++--- a/code/results/single_scene/summary_table.md
+++++++++ b/code/results/single_scene/summary_table.md
++++++@@ -2,44 +2,45 @@
++++++ 
++++++ Mean over seeds per (scene, method); best value per scene in **bold** (lower is better except Nr).
++++++ 
++++++-| Scene | ESFM Rot (deg) | U-ESFM Rot (deg) | ESFM (paper) Rot (deg) | ESFM Trans | U-ESFM Trans | ESFM (paper) Trans | ESFM Reproj (px) | U-ESFM Reproj (px) | ESFM (paper) Reproj (px) | ESFM Nr | U-ESFM Nr | ESFM (paper) Nr | ESFM Time (s) | U-ESFM Time (s) | ESFM (paper) Time (s) | ESFM Rot-BA (deg) | U-ESFM Rot-BA (deg) | ESFM (paper) Rot-BA (deg) | ESFM Trans-BA | U-ESFM Trans-BA | ESFM (paper) Trans-BA | ESFM Reproj-BA (px) | U-ESFM Reproj-BA (px) | ESFM (paper) Reproj-BA (px) |
+++++++| Scene | ESFM (official code) Rot (deg) | U-ESFM Rot (deg) | ESFM (paper) Rot (deg) | ESFM (official code) Trans | U-ESFM Trans | ESFM (paper) Trans | ESFM (official code) Reproj (px) | U-ESFM Reproj (px) | ESFM (paper) Reproj (px) | ESFM (official code) Nr | U-ESFM Nr | ESFM (paper) Nr | ESFM (official code) Time (s) | U-ESFM Time (s) | ESFM (paper) Time (s) | ESFM (official code) Rot-BA (deg) | U-ESFM Rot-BA (deg) | ESFM (paper) Rot-BA (deg) | ESFM (official code) Trans-BA | U-ESFM Trans-BA | ESFM (paper) Trans-BA | ESFM (official code) Reproj-BA (px) | U-ESFM Reproj-BA (px) | ESFM (paper) Reproj-BA (px) |
++++++ |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
++++++-| Alcatraz Courtyard | **0.362** | 0.616 | 0.619 | **0.093** | 0.161 | 0.160 | **3.435** | 5.796 | 1.640 | **133** | 133 | -- | **4826.713** | 6481.757 | -- | -- | -- | 0.049 | -- | -- | 0.015 | -- | -- | 0.810 |
++++++-| Alcatraz Water Tower | **0.666** | 7.413 | 0.933 | **0.370** | 3.437 | 0.518 | **4.763** | 8.607 | 2.130 | **172** | 172 | -- | **2461.337** | 3854.907 | -- | -- | -- | 0.230 | -- | -- | 0.116 | -- | -- | 0.550 |
++++++-| Buddah Tooth Relic Temple Singapore | **1.131** | 12.080 | 1.030 | **0.280** | 2.308 | 0.233 | **9.964** | 19.634 | 2.060 | **162** | 162 | -- | **2916.837** | 4320.873 | -- | -- | -- | 0.081 | -- | -- | 0.014 | -- | -- | 0.850 |
++++++-| Doge Palace Venice | **0.801** | 0.911 | 1.163 | **0.218** | 0.290 | 0.342 | **6.432** | 10.966 | 3.620 | **241** | 241 | -- | 28768.725 | **21074.950** | -- | -- | -- | 0.211 | -- | -- | 0.029 | -- | -- | 1 |
++++++-| Door Lund | **0.014** | 14.172 | 0.024 | **0.004** | 0.952 | 0.006 | **1.812** | 11.922 | 0.320 | **12** | 12 | -- | **4888.363** | 9188.593 | -- | -- | -- | 0.006 | -- | -- | 0.001 | -- | -- | 0.300 |
++++++-| Drinking Fountain Somewhere In Zurich | **17.193** | 17.536 | 0.031 | 0.753 | **0.740** | 0.004 | **6.058** | 6.511 | 0.330 | **14** | 14 | -- | **883.837** | 2963.647 | -- | -- | -- | 0.007 | -- | -- | 0.002 | -- | -- | 0.310 |
++++++-| East Indiaman Goteborg | 5.568 | **5.310** | 3.814 | 0.979 | **0.959** | 0.621 | **7.318** | 9.840 | 4.130 | **179** | 179 | -- | **3987.183** | 5241.430 | -- | -- | -- | 3.117 | -- | -- | 0.509 | -- | -- | 1.850 |
++++++-| Ecole Superior De Guerre | 0.349 | **0.284** | 0.318 | 0.090 | **0.077** | 0.081 | **3.418** | 5.349 | 0.720 | **35** | 35 | -- | **2380.387** | 4633.833 | -- | -- | -- | 0.024 | -- | -- | 0.005 | -- | -- | 0.340 |
++++++-| Eglise du dome | **0.351** | 45.801 | 0.808 | **0.088** | 6.451 | 0.205 | **3.799** | 7.964 | 0.910 | **85** | 85 | -- | 25099.450 | **23371.730** | -- | -- | -- | 0.037 | -- | -- | 0.010 | -- | -- | 0.270 |
++++++-| Folke Filbyter | 85.831 | **81.288** | 74.596 | **0.129** | 0.130 | 0.125 | **19.275** | 23.156 | 10.370 | **40** | 40 | -- | **1581.897** | 3703.187 | -- | -- | -- | 70.157 | -- | -- | 0.118 | -- | -- | 4.290 |
++++++-| Fort Channing Gate Singapore | **0.093** | 11.128 | 0.207 | **0.041** | 4.004 | 0.093 | **1.783** | 6.651 | 0.520 | **27** | 27 | -- | **3434.113** | 6172.087 | -- | -- | -- | 0.020 | -- | -- | 0.008 | -- | -- | 0.250 |
++++++-| Golden Statue Somewhere In Hong Kong | **0.232** | 0.371 | 0.292 | **0.053** | 0.074 | 0.073 | **1.829** | 2.835 | 0.400 | **18** | 18 | -- | **4095.853** | 7762.437 | -- | -- | -- | 0.031 | -- | -- | 0.004 | -- | -- | 0.270 |
++++++-| Gustav Vasa | **3.509** | 6.099 | 34.181 | **0.220** | 0.345 | 1.085 | **1.655** | 3.087 | 3.520 | **18** | 18 | -- | **709.867** | 2543.553 | -- | -- | -- | 32.266 | -- | -- | 1.145 | -- | -- | 3.150 |
++++++-| GustavIIAdolf | **47.126** | 77.423 | 67.784 | **8.536** | 11.962 | 9.714 | **11.722** | 15.851 | 13.910 | **57** | 57 | -- | **1044.170** | 2892.580 | -- | -- | -- | 58.458 | -- | -- | 8.524 | -- | -- | 11.490 |
++++++-| Jonas Ahlstromer | 48.740 | **42.246** | 50.190 | **9.823** | 10.300 | 10.888 | 11.136 | **8.782** | 10.820 | **40** | 40 | -- | **615.450** | 2328.313 | -- | -- | -- | 47.117 | -- | -- | 10.451 | -- | -- | 8.410 |
++++++-| Kings College University Of Toronto | **9.025** | 21.523 | 0.989 | 2.044 | **1.905** | 0.235 | **3.565** | 5.525 | 0.900 | **77** | 77 | -- | **1078.980** | 2886.243 | -- | -- | -- | 0.085 | -- | -- | 0.017 | -- | -- | 0.340 |
++++++-| Lund University Sphinx | **11.392** | 33.023 | 19.522 | **2.981** | 6.889 | 4.585 | **6.056** | 12.060 | 4.780 | **70** | 70 | -- | **2718.567** | 4654.610 | -- | -- | -- | 8.752 | -- | -- | 2.191 | -- | -- | 1.360 |
++++++-| Nijo Castle Gate | **0.839** | 1.682 | 1.495 | **0.165** | 0.326 | 0.286 | **7.415** | 12.138 | 1.700 | **19** | 19 | -- | **1095.177** | 3306.697 | -- | -- | -- | 0.069 | -- | -- | 0.012 | -- | -- | 0.730 |
++++++-| Pantheon Paris | -- | 19.345 | 0.192 | -- | 3.065 | 0.050 | -- | 13.562 | 1.470 | -- | 179 | -- | -- | 5497.377 | -- | -- | -- | 0.040 | -- | -- | 0.005 | -- | -- | 0.490 |
++++++-| Park Gate Clermont Ferrand | 21.388 | **19.302** | 0.391 | 10.192 | **9.397** | 0.125 | **7.712** | 7.852 | 0.570 | **34** | 34 | -- | **1596.940** | 3597.403 | -- | -- | -- | 0.049 | -- | -- | 0.022 | -- | -- | 0.350 |
++++++-| Plaza De Armas Santiago | **0.775** | 3.327 | 6.782 | **0.337** | 1.435 | 2.944 | **6.009** | 9.463 | 7.400 | **240** | 240 | -- | **6372.683** | 6535.537 | -- | -- | -- | 2.556 | -- | -- | 1.383 | -- | -- | 4.900 |
++++++-| Porta San Donato Bologna | **0.592** | 21.006 | 2.153 | **0.107** | 3.541 | 0.388 | **5.603** | 13.244 | 2.280 | **141** | 141 | -- | **3754.163** | 5118.030 | -- | -- | -- | 0.095 | -- | -- | 0.046 | -- | -- | 0.750 |
++++++-| Round Church Cambridge | **2.129** | 2.490 | 2.451 | **0.920** | 1.002 | 1.003 | **6.097** | 8.546 | 2.660 | **92** | 92 | -- | **9776.630** | 12061.350 | -- | -- | -- | 1.107 | -- | -- | 0.582 | -- | -- | 1.540 |
++++++-| Skansen Kronan Gothenburg | **0.338** | 0.375 | 0.736 | **0.116** | 0.124 | 0.226 | **3.048** | 4.799 | 1.240 | **131** | 131 | -- | 17163.940 | **16235.090** | -- | -- | -- | 0.026 | -- | -- | 0.008 | -- | -- | 0.670 |
++++++-| Smolny Cathedral St Petersburg | **21.806** | 22.143 | 0.554 | 2.111 | **2.099** | 0.051 | **11.228** | 12.526 | 1.660 | **131** | 131 | -- | **10362.595** | 10475.740 | -- | -- | -- | 0.033 | -- | -- | 0.006 | -- | -- | 0.810 |
++++++-| Sri Mariamman Singapore | **1.219** | 6.479 | 2.302 | **0.382** | 2.188 | 0.683 | **10.088** | 17.854 | 4.130 | **222** | 222 | -- | **5985.540** | 6485.667 | -- | -- | -- | 0.077 | -- | -- | 0.023 | -- | -- | 0.910 |
++++++-| Sri Thendayuthapani Singapore | **0.719** | 47.075 | 46.269 | **0.179** | 3.855 | 3.812 | **7.124** | 14.133 | 23.370 | **98** | 98 | -- | **13362.090** | 13808.440 | -- | -- | -- | 44.170 | -- | -- | 2.870 | -- | -- | 8.440 |
++++++-| Sri Veeramakaliamman Singapore | **2.334** | 29.405 | 2.559 | **0.543** | 3.868 | 0.597 | **11.965** | 28.951 | 3.470 | **157** | 157 | -- | **12392.180** | 12544.690 | -- | -- | -- | 0.175 | -- | -- | 0.040 | -- | -- | 0.730 |
++++++-| Statue Of Liberty | 52.358 | **39.859** | 46.887 | 28.503 | **18.503** | 20.012 | 235036.825 | **41.028** | 26.160 | **134** | 134 | -- | **99.900** | 12924 | -- | -- | -- | 9.091 | -- | -- | 4.122 | -- | -- | 6.970 |
++++++-| The Pumpkin | **25.731** | 26.571 | 94.672 | **5.552** | 5.665 | 14.890 | **26.876** | 35.367 | 33.410 | **196** | 196 | -- | **4327.273** | 5790.597 | -- | -- | -- | 98.862 | -- | -- | 14.952 | -- | -- | 24.850 |
++++++-| Thian Hook Keng Temple Singapore | **0.927** | 1.201 | 0.832 | **0.093** | 0.121 | 0.082 | **14.915** | 20.789 | 2.750 | **138** | 138 | -- | **3298.537** | 4612.373 | -- | -- | -- | 0.081 | -- | -- | 0.008 | -- | -- | 1.130 |
++++++-| Tsar Nikolai I | **42.803** | 78.413 | 48.499 | **8.927** | 15.394 | 9.467 | **11.124** | 18.431 | 9.790 | **98** | 98 | -- | **4928.430** | 6646.845 | -- | -- | -- | 36.280 | -- | -- | 7.836 | -- | -- | 6.530 |
++++++-| Urban II | 59.030 | -- | 47.490 | 11.008 | -- | 9.467 | 17.601 | -- | 9.380 | 96 | -- | -- | 8889.690 | -- | -- | -- | -- | 48.214 | -- | -- | 9.586 | -- | -- | 6.920 |
++++++-| Vercingetorix | **82.952** | 86.088 | 69.328 | **10.195** | 10.890 | 8.788 | **7.257** | 8.681 | 5.080 | **69** | 69 | -- | **1223.970** | 3105.607 | -- | -- | -- | 17.706 | -- | -- | 3.104 | -- | -- | 1.500 |
++++++-| Yueh Hai Ching Temple Singapore | 0.627 | -- | 0.720 | 0.084 | -- | 0.098 | 5.562 | -- | 0.940 | 43 | -- | -- | 1749.280 | -- | -- | -- | -- | 0.043 | -- | -- | 0.014 | -- | -- | 0.650 |
++++++-| **Mean** | **16.146** | 23.697 | 18.023 | **3.121** | 4.014 | 2.912 | 6920.602 | **13.088** | 5.673 | 100.559 | **104.818** | -- | **5819.728** | 7358.187 | -- | -- | -- | 13.695 | -- | -- | 1.937 | -- | -- | 2.992 |
+++++++| Alcatraz Courtyard | **0.362** | 0.617 | 0.619 | **0.093** | 0.157 | 0.160 | **3.435** | 5.482 | 1.640 | **133** | 133 | -- | **4826.713** | 6554.345 | -- | -- | -- | 0.049 | -- | -- | 0.015 | -- | -- | 0.810 |
+++++++| Alcatraz Water Tower | **0.666** | 1.480 | 0.933 | **0.370** | 0.803 | 0.518 | **4.763** | 7.397 | 2.130 | **172** | 172 | -- | **2461.337** | 3795.877 | -- | -- | -- | 0.230 | -- | -- | 0.116 | -- | -- | 0.550 |
+++++++| Buddah Tooth Relic Temple Singapore | **1.131** | 1.791 | 1.030 | **0.280** | 0.520 | 0.233 | **9.964** | 14.029 | 2.060 | **162** | 162 | -- | **2916.837** | 4289.503 | -- | -- | -- | 0.081 | -- | -- | 0.014 | -- | -- | 0.850 |
+++++++| Doge Palace Venice | 0.753 | -- | 1.163 | 0.209 | -- | 0.342 | 6.359 | -- | 3.620 | 241 | -- | -- | 22522.737 | -- | -- | -- | -- | 0.211 | -- | -- | 0.029 | -- | -- | 1 |
+++++++| Door Lund | **0.014** | 6.960 | 0.024 | **0.004** | 0.512 | 0.006 | **1.812** | 7.101 | 0.320 | **12** | 12 | -- | **4888.363** | 9252.735 | -- | -- | -- | 0.006 | -- | -- | 0.001 | -- | -- | 0.300 |
+++++++| Drinking Fountain Somewhere In Zurich | **17.193** | 17.329 | 0.031 | 0.753 | **0.750** | 0.004 | **6.058** | 6.675 | 0.330 | **14** | 14 | -- | **883.837** | 5815.220 | -- | -- | -- | 0.007 | -- | -- | 0.002 | -- | -- | 0.310 |
+++++++| East Indiaman Goteborg | **5.568** | 7.231 | 3.814 | **0.979** | 1.364 | 0.621 | **7.318** | 10.130 | 4.130 | **179** | 179 | -- | **3987.183** | 5198.093 | -- | -- | -- | 3.117 | -- | -- | 0.509 | -- | -- | 1.850 |
+++++++| Ecole Superior De Guerre | **0.349** | 51.327 | 0.318 | **0.090** | 2.574 | 0.081 | **3.418** | 11.751 | 0.720 | **35** | 35 | -- | **2380.387** | 10426.770 | -- | -- | -- | 0.024 | -- | -- | 0.005 | -- | -- | 0.340 |
+++++++| Eglise du dome | **0.386** | 0.853 | 0.808 | **0.098** | 0.232 | 0.205 | **3.987** | 5.944 | 0.910 | **85** | 85 | -- | 19631.293 | **9803.940** | -- | -- | -- | 0.037 | -- | -- | 0.010 | -- | -- | 0.270 |
+++++++| Folke Filbyter | 85.831 | **82.656** | 74.596 | **0.129** | 0.131 | 0.125 | **19.275** | 28.225 | 10.370 | **40** | 40 | -- | **1581.897** | 3755.193 | -- | -- | -- | 70.157 | -- | -- | 0.118 | -- | -- | 4.290 |
+++++++| Fort Channing Gate Singapore | **0.093** | 0.122 | 0.207 | **0.041** | 0.056 | 0.093 | **1.783** | 3.649 | 0.520 | **27** | 27 | -- | **3434.113** | 6183.925 | -- | -- | -- | 0.020 | -- | -- | 0.008 | -- | -- | 0.250 |
+++++++| Golden Statue Somewhere In Hong Kong | 0.232 | -- | 0.292 | 0.053 | -- | 0.073 | 1.829 | -- | 0.400 | 18 | -- | -- | 4095.853 | -- | -- | -- | -- | 0.031 | -- | -- | 0.004 | -- | -- | 0.270 |
+++++++| Gustav Vasa | **3.509** | 25.854 | 34.181 | **0.220** | 0.775 | 1.085 | **1.655** | 4.747 | 3.520 | **18** | 18 | -- | **709.867** | 3887.653 | -- | -- | -- | 32.266 | -- | -- | 1.145 | -- | -- | 3.150 |
+++++++| GustavIIAdolf | **47.126** | 88.194 | 67.784 | **8.536** | 13.129 | 9.714 | **11.722** | 17.220 | 13.910 | **57** | 57 | -- | **1044.170** | 2981.017 | -- | -- | -- | 58.458 | -- | -- | 8.524 | -- | -- | 11.490 |
+++++++| Jonas Ahlstromer | 48.740 | **46.584** | 50.190 | **9.823** | 10.245 | 10.888 | **11.136** | 11.144 | 10.820 | **40** | 40 | -- | **615.450** | 2414.653 | -- | -- | -- | 47.117 | -- | -- | 10.451 | -- | -- | 8.410 |
+++++++| Kings College University Of Toronto | **9.025** | 18.154 | 0.989 | 2.044 | **1.706** | 0.235 | **3.565** | 6.501 | 0.900 | **77** | 77 | -- | **1078.980** | 2854.715 | -- | -- | -- | 0.085 | -- | -- | 0.017 | -- | -- | 0.340 |
+++++++| Lund University Sphinx | **11.392** | 42.959 | 19.522 | **2.981** | 8.720 | 4.585 | **6.056** | 15.000 | 4.780 | **70** | 70 | -- | **2718.567** | 4705.610 | -- | -- | -- | 8.752 | -- | -- | 2.191 | -- | -- | 1.360 |
+++++++| Nijo Castle Gate | **0.839** | 8.641 | 1.495 | **0.165** | 2.791 | 0.286 | **7.415** | 20.967 | 1.700 | **19** | 19 | -- | **1095.177** | 3241.330 | -- | -- | -- | 0.069 | -- | -- | 0.012 | -- | -- | 0.730 |
+++++++| Pantheon Paris | -- | 0.509 | 0.192 | -- | 0.066 | 0.050 | -- | 7.493 | 1.470 | -- | 179 | -- | -- | 5565.670 | -- | -- | -- | 0.040 | -- | -- | 0.005 | -- | -- | 0.490 |
+++++++| Park Gate Clermont Ferrand | 21.388 | -- | 0.391 | 10.192 | -- | 0.125 | 7.712 | -- | 0.570 | 34 | -- | -- | 1596.940 | -- | -- | -- | -- | 0.049 | -- | -- | 0.022 | -- | -- | 0.350 |
+++++++| Plaza De Armas Santiago | 0.775 | -- | 6.782 | 0.337 | -- | 2.944 | 6.009 | -- | 7.400 | 240 | -- | -- | 6372.683 | -- | -- | -- | -- | 2.556 | -- | -- | 1.383 | -- | -- | 4.900 |
+++++++| Porta San Donato Bologna | 0.592 | -- | 2.153 | 0.107 | -- | 0.388 | 5.603 | -- | 2.280 | 141 | -- | -- | 3754.163 | -- | -- | -- | -- | 0.095 | -- | -- | 0.046 | -- | -- | 0.750 |
+++++++| Round Church Cambridge | 2.137 | -- | 2.451 | 0.927 | -- | 1.003 | 6.103 | -- | 2.660 | 92 | -- | -- | 9779.270 | -- | -- | -- | -- | 1.107 | -- | -- | 0.582 | -- | -- | 1.540 |
+++++++| Skansen Kronan Gothenburg | 0.301 | -- | 0.736 | 0.102 | -- | 0.226 | 2.906 | -- | 1.240 | 131 | -- | -- | 17153.677 | -- | -- | -- | -- | 0.026 | -- | -- | 0.008 | -- | -- | 0.670 |
+++++++| Smolny Cathedral St Petersburg | 21.775 | -- | 0.554 | 2.111 | -- | 0.051 | 11.239 | -- | 1.660 | 131 | -- | -- | 10362.787 | -- | -- | -- | -- | 0.033 | -- | -- | 0.006 | -- | -- | 0.810 |
+++++++| Some Cathedral In Barcelona | 0.651 | -- | 0.880 | 0.233 | -- | 0.315 | 6.271 | -- | 2.870 | 177 | -- | -- | 5001.973 | -- | -- | -- | -- | 0.026 | -- | -- | 0.011 | -- | -- | 0.890 |
+++++++| Sri Mariamman Singapore | 1.219 | -- | 2.302 | 0.382 | -- | 0.683 | 10.088 | -- | 4.130 | 222 | -- | -- | 5985.540 | -- | -- | -- | -- | 0.077 | -- | -- | 0.023 | -- | -- | 0.910 |
+++++++| Sri Thendayuthapani Singapore | 0.593 | -- | 46.269 | 0.150 | -- | 3.812 | 7.016 | -- | 23.370 | 98 | -- | -- | 13338.167 | -- | -- | -- | -- | 44.170 | -- | -- | 2.870 | -- | -- | 8.440 |
+++++++| Sri Veeramakaliamman Singapore | 2.404 | -- | 2.559 | 0.559 | -- | 0.597 | 12.234 | -- | 3.470 | 157 | -- | -- | 12377.927 | -- | -- | -- | -- | 0.175 | -- | -- | 0.040 | -- | -- | 0.730 |
+++++++| Statue Of Liberty | 52.358 | -- | 46.887 | 28.503 | -- | 20.012 | 235036.825 | -- | 26.160 | 134 | -- | -- | 99.900 | -- | -- | -- | -- | 9.091 | -- | -- | 4.122 | -- | -- | 6.970 |
+++++++| The Pumpkin | 25.731 | -- | 94.672 | 5.552 | -- | 14.890 | 26.876 | -- | 33.410 | 196 | -- | -- | 4327.273 | -- | -- | -- | -- | 98.862 | -- | -- | 14.952 | -- | -- | 24.850 |
+++++++| Thian Hook Keng Temple Singapore | 0.927 | -- | 0.832 | 0.093 | -- | 0.082 | 14.915 | -- | 2.750 | 138 | -- | -- | 3298.537 | -- | -- | -- | -- | 0.081 | -- | -- | 0.008 | -- | -- | 1.130 |
+++++++| Tsar Nikolai I | 42.166 | -- | 48.499 | 8.732 | -- | 9.467 | 11.210 | -- | 9.790 | 98 | -- | -- | 4929.200 | -- | -- | -- | -- | 36.280 | -- | -- | 7.836 | -- | -- | 6.530 |
+++++++| Urban II | 58.919 | -- | 47.490 | 11.006 | -- | 9.467 | 17.538 | -- | 9.380 | 96 | -- | -- | 8889.617 | -- | -- | -- | -- | 48.214 | -- | -- | 9.586 | -- | -- | 6.920 |
+++++++| Vercingetorix | 82.952 | -- | 69.328 | 10.195 | -- | 8.788 | 7.257 | -- | 5.080 | 69 | -- | -- | 1223.970 | -- | -- | -- | -- | 17.706 | -- | -- | 3.104 | -- | -- | 1.500 |
+++++++| Yueh Hai Ching Temple Singapore | 0.544 | -- | 0.720 | 0.075 | -- | 0.098 | 5.596 | -- | 0.940 | 43 | -- | -- | 1748.667 | -- | -- | -- | -- | 0.043 | -- | -- | 0.014 | -- | -- | 0.650 |
+++++++| **Mean** | **15.676** | 23.604 | 17.547 | 3.032 | **2.619** | 2.840 | 6723.056 | **10.792** | 5.595 | **102.743** | 77.588 | -- | 5460.373 | **5336.838** | -- | -- | -- | 13.315 | -- | -- | 1.883 | -- | -- | 2.933 |
++++++ 
++++++ _Seeds per cell: [0, 1, 2]._
++++++ 
++++++diff --git a/code/results/single_scene/summary_table.tex b/code/results/single_scene/summary_table.tex
++++++index 258e3f2..95defe0 100644
++++++--- a/code/results/single_scene/summary_table.tex
+++++++++ b/code/results/single_scene/summary_table.tex
++++++@@ -7,45 +7,46 @@
++++++ \begin{tabular}{lcccccccccccccccccccccccc}
++++++ \toprule
++++++  & \multicolumn{3}{c}{Rot (deg)} & \multicolumn{3}{c}{Trans} & \multicolumn{3}{c}{Reproj (px)} & \multicolumn{3}{c}{Nr} & \multicolumn{3}{c}{Time (s)} & \multicolumn{3}{c}{Rot-BA (deg)} & \multicolumn{3}{c}{Trans-BA} & \multicolumn{3}{c}{Reproj-BA (px)} \\
++++++-Scene & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) \\
+++++++Scene & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) \\
++++++ \midrule
++++++-Alcatraz Courtyard & \textbf{0.362} & 0.616 & 0.619 & \textbf{0.093} & 0.161 & 0.160 & \textbf{3.435} & 5.796 & 1.640 & \textbf{133} & 133 & -- & \textbf{4826.713} & 6481.757 & -- & -- & -- & 0.049 & -- & -- & 0.015 & -- & -- & 0.810 \\
++++++-Alcatraz Water Tower & \textbf{0.666} & 7.413 & 0.933 & \textbf{0.370} & 3.437 & 0.518 & \textbf{4.763} & 8.607 & 2.130 & \textbf{172} & 172 & -- & \textbf{2461.337} & 3854.907 & -- & -- & -- & 0.230 & -- & -- & 0.116 & -- & -- & 0.550 \\
++++++-Buddah Tooth Relic Temple Singapore & \textbf{1.131} & 12.080 & 1.030 & \textbf{0.280} & 2.308 & 0.233 & \textbf{9.964} & 19.634 & 2.060 & \textbf{162} & 162 & -- & \textbf{2916.837} & 4320.873 & -- & -- & -- & 0.081 & -- & -- & 0.014 & -- & -- & 0.850 \\
++++++-Doge Palace Venice & \textbf{0.801} & 0.911 & 1.163 & \textbf{0.218} & 0.290 & 0.342 & \textbf{6.432} & 10.966 & 3.620 & \textbf{241} & 241 & -- & 28768.725 & \textbf{21074.950} & -- & -- & -- & 0.211 & -- & -- & 0.029 & -- & -- & 1 \\
++++++-Door Lund & \textbf{0.014} & 14.172 & 0.024 & \textbf{0.004} & 0.952 & 0.006 & \textbf{1.812} & 11.922 & 0.320 & \textbf{12} & 12 & -- & \textbf{4888.363} & 9188.593 & -- & -- & -- & 0.006 & -- & -- & 0.001 & -- & -- & 0.300 \\
++++++-Drinking Fountain Somewhere In Zurich & \textbf{17.193} & 17.536 & 0.031 & 0.753 & \textbf{0.740} & 0.004 & \textbf{6.058} & 6.511 & 0.330 & \textbf{14} & 14 & -- & \textbf{883.837} & 2963.647 & -- & -- & -- & 0.007 & -- & -- & 0.002 & -- & -- & 0.310 \\
++++++-East Indiaman Goteborg & 5.568 & \textbf{5.310} & 3.814 & 0.979 & \textbf{0.959} & 0.621 & \textbf{7.318} & 9.840 & 4.130 & \textbf{179} & 179 & -- & \textbf{3987.183} & 5241.430 & -- & -- & -- & 3.117 & -- & -- & 0.509 & -- & -- & 1.850 \\
++++++-Ecole Superior De Guerre & 0.349 & \textbf{0.284} & 0.318 & 0.090 & \textbf{0.077} & 0.081 & \textbf{3.418} & 5.349 & 0.720 & \textbf{35} & 35 & -- & \textbf{2380.387} & 4633.833 & -- & -- & -- & 0.024 & -- & -- & 0.005 & -- & -- & 0.340 \\
++++++-Eglise du dome & \textbf{0.351} & 45.801 & 0.808 & \textbf{0.088} & 6.451 & 0.205 & \textbf{3.799} & 7.964 & 0.910 & \textbf{85} & 85 & -- & 25099.450 & \textbf{23371.730} & -- & -- & -- & 0.037 & -- & -- & 0.010 & -- & -- & 0.270 \\
++++++-Folke Filbyter & 85.831 & \textbf{81.288} & 74.596 & \textbf{0.129} & 0.130 & 0.125 & \textbf{19.275} & 23.156 & 10.370 & \textbf{40} & 40 & -- & \textbf{1581.897} & 3703.187 & -- & -- & -- & 70.157 & -- & -- & 0.118 & -- & -- & 4.290 \\
++++++-Fort Channing Gate Singapore & \textbf{0.093} & 11.128 & 0.207 & \textbf{0.041} & 4.004 & 0.093 & \textbf{1.783} & 6.651 & 0.520 & \textbf{27} & 27 & -- & \textbf{3434.113} & 6172.087 & -- & -- & -- & 0.020 & -- & -- & 0.008 & -- & -- & 0.250 \\
++++++-Golden Statue Somewhere In Hong Kong & \textbf{0.232} & 0.371 & 0.292 & \textbf{0.053} & 0.074 & 0.073 & \textbf{1.829} & 2.835 & 0.400 & \textbf{18} & 18 & -- & \textbf{4095.853} & 7762.437 & -- & -- & -- & 0.031 & -- & -- & 0.004 & -- & -- & 0.270 \\
++++++-Gustav Vasa & \textbf{3.509} & 6.099 & 34.181 & \textbf{0.220} & 0.345 & 1.085 & \textbf{1.655} & 3.087 & 3.520 & \textbf{18} & 18 & -- & \textbf{709.867} & 2543.553 & -- & -- & -- & 32.266 & -- & -- & 1.145 & -- & -- & 3.150 \\
++++++-GustavIIAdolf & \textbf{47.126} & 77.423 & 67.784 & \textbf{8.536} & 11.962 & 9.714 & \textbf{11.722} & 15.851 & 13.910 & \textbf{57} & 57 & -- & \textbf{1044.170} & 2892.580 & -- & -- & -- & 58.458 & -- & -- & 8.524 & -- & -- & 11.490 \\
++++++-Jonas Ahlstromer & 48.740 & \textbf{42.246} & 50.190 & \textbf{9.823} & 10.300 & 10.888 & 11.136 & \textbf{8.782} & 10.820 & \textbf{40} & 40 & -- & \textbf{615.450} & 2328.313 & -- & -- & -- & 47.117 & -- & -- & 10.451 & -- & -- & 8.410 \\
++++++-Kings College University Of Toronto & \textbf{9.025} & 21.523 & 0.989 & 2.044 & \textbf{1.905} & 0.235 & \textbf{3.565} & 5.525 & 0.900 & \textbf{77} & 77 & -- & \textbf{1078.980} & 2886.243 & -- & -- & -- & 0.085 & -- & -- & 0.017 & -- & -- & 0.340 \\
++++++-Lund University Sphinx & \textbf{11.392} & 33.023 & 19.522 & \textbf{2.981} & 6.889 & 4.585 & \textbf{6.056} & 12.060 & 4.780 & \textbf{70} & 70 & -- & \textbf{2718.567} & 4654.610 & -- & -- & -- & 8.752 & -- & -- & 2.191 & -- & -- & 1.360 \\
++++++-Nijo Castle Gate & \textbf{0.839} & 1.682 & 1.495 & \textbf{0.165} & 0.326 & 0.286 & \textbf{7.415} & 12.138 & 1.700 & \textbf{19} & 19 & -- & \textbf{1095.177} & 3306.697 & -- & -- & -- & 0.069 & -- & -- & 0.012 & -- & -- & 0.730 \\
++++++-Pantheon Paris & -- & 19.345 & 0.192 & -- & 3.065 & 0.050 & -- & 13.562 & 1.470 & -- & 179 & -- & -- & 5497.377 & -- & -- & -- & 0.040 & -- & -- & 0.005 & -- & -- & 0.490 \\
++++++-Park Gate Clermont Ferrand & 21.388 & \textbf{19.302} & 0.391 & 10.192 & \textbf{9.397} & 0.125 & \textbf{7.712} & 7.852 & 0.570 & \textbf{34} & 34 & -- & \textbf{1596.940} & 3597.403 & -- & -- & -- & 0.049 & -- & -- & 0.022 & -- & -- & 0.350 \\
++++++-Plaza De Armas Santiago & \textbf{0.775} & 3.327 & 6.782 & \textbf{0.337} & 1.435 & 2.944 & \textbf{6.009} & 9.463 & 7.400 & \textbf{240} & 240 & -- & \textbf{6372.683} & 6535.537 & -- & -- & -- & 2.556 & -- & -- & 1.383 & -- & -- & 4.900 \\
++++++-Porta San Donato Bologna & \textbf{0.592} & 21.006 & 2.153 & \textbf{0.107} & 3.541 & 0.388 & \textbf{5.603} & 13.244 & 2.280 & \textbf{141} & 141 & -- & \textbf{3754.163} & 5118.030 & -- & -- & -- & 0.095 & -- & -- & 0.046 & -- & -- & 0.750 \\
++++++-Round Church Cambridge & \textbf{2.129} & 2.490 & 2.451 & \textbf{0.920} & 1.002 & 1.003 & \textbf{6.097} & 8.546 & 2.660 & \textbf{92} & 92 & -- & \textbf{9776.630} & 12061.350 & -- & -- & -- & 1.107 & -- & -- & 0.582 & -- & -- & 1.540 \\
++++++-Skansen Kronan Gothenburg & \textbf{0.338} & 0.375 & 0.736 & \textbf{0.116} & 0.124 & 0.226 & \textbf{3.048} & 4.799 & 1.240 & \textbf{131} & 131 & -- & 17163.940 & \textbf{16235.090} & -- & -- & -- & 0.026 & -- & -- & 0.008 & -- & -- & 0.670 \\
++++++-Smolny Cathedral St Petersburg & \textbf{21.806} & 22.143 & 0.554 & 2.111 & \textbf{2.099} & 0.051 & \textbf{11.228} & 12.526 & 1.660 & \textbf{131} & 131 & -- & \textbf{10362.595} & 10475.740 & -- & -- & -- & 0.033 & -- & -- & 0.006 & -- & -- & 0.810 \\
++++++-Sri Mariamman Singapore & \textbf{1.219} & 6.479 & 2.302 & \textbf{0.382} & 2.188 & 0.683 & \textbf{10.088} & 17.854 & 4.130 & \textbf{222} & 222 & -- & \textbf{5985.540} & 6485.667 & -- & -- & -- & 0.077 & -- & -- & 0.023 & -- & -- & 0.910 \\
++++++-Sri Thendayuthapani Singapore & \textbf{0.719} & 47.075 & 46.269 & \textbf{0.179} & 3.855 & 3.812 & \textbf{7.124} & 14.133 & 23.370 & \textbf{98} & 98 & -- & \textbf{13362.090} & 13808.440 & -- & -- & -- & 44.170 & -- & -- & 2.870 & -- & -- & 8.440 \\
++++++-Sri Veeramakaliamman Singapore & \textbf{2.334} & 29.405 & 2.559 & \textbf{0.543} & 3.868 & 0.597 & \textbf{11.965} & 28.951 & 3.470 & \textbf{157} & 157 & -- & \textbf{12392.180} & 12544.690 & -- & -- & -- & 0.175 & -- & -- & 0.040 & -- & -- & 0.730 \\
++++++-Statue Of Liberty & 52.358 & \textbf{39.859} & 46.887 & 28.503 & \textbf{18.503} & 20.012 & 235036.825 & \textbf{41.028} & 26.160 & \textbf{134} & 134 & -- & \textbf{99.900} & 12924 & -- & -- & -- & 9.091 & -- & -- & 4.122 & -- & -- & 6.970 \\
++++++-The Pumpkin & \textbf{25.731} & 26.571 & 94.672 & \textbf{5.552} & 5.665 & 14.890 & \textbf{26.876} & 35.367 & 33.410 & \textbf{196} & 196 & -- & \textbf{4327.273} & 5790.597 & -- & -- & -- & 98.862 & -- & -- & 14.952 & -- & -- & 24.850 \\
++++++-Thian Hook Keng Temple Singapore & \textbf{0.927} & 1.201 & 0.832 & \textbf{0.093} & 0.121 & 0.082 & \textbf{14.915} & 20.789 & 2.750 & \textbf{138} & 138 & -- & \textbf{3298.537} & 4612.373 & -- & -- & -- & 0.081 & -- & -- & 0.008 & -- & -- & 1.130 \\
++++++-Tsar Nikolai I & \textbf{42.803} & 78.413 & 48.499 & \textbf{8.927} & 15.394 & 9.467 & \textbf{11.124} & 18.431 & 9.790 & \textbf{98} & 98 & -- & \textbf{4928.430} & 6646.845 & -- & -- & -- & 36.280 & -- & -- & 7.836 & -- & -- & 6.530 \\
++++++-Urban II & 59.030 & -- & 47.490 & 11.008 & -- & 9.467 & 17.601 & -- & 9.380 & 96 & -- & -- & 8889.690 & -- & -- & -- & -- & 48.214 & -- & -- & 9.586 & -- & -- & 6.920 \\
++++++-Vercingetorix & \textbf{82.952} & 86.088 & 69.328 & \textbf{10.195} & 10.890 & 8.788 & \textbf{7.257} & 8.681 & 5.080 & \textbf{69} & 69 & -- & \textbf{1223.970} & 3105.607 & -- & -- & -- & 17.706 & -- & -- & 3.104 & -- & -- & 1.500 \\
++++++-Yueh Hai Ching Temple Singapore & 0.627 & -- & 0.720 & 0.084 & -- & 0.098 & 5.562 & -- & 0.940 & 43 & -- & -- & 1749.280 & -- & -- & -- & -- & 0.043 & -- & -- & 0.014 & -- & -- & 0.650 \\
+++++++Alcatraz Courtyard & \textbf{0.362} & 0.617 & 0.619 & \textbf{0.093} & 0.157 & 0.160 & \textbf{3.435} & 5.482 & 1.640 & \textbf{133} & 133 & -- & \textbf{4826.713} & 6554.345 & -- & -- & -- & 0.049 & -- & -- & 0.015 & -- & -- & 0.810 \\
+++++++Alcatraz Water Tower & \textbf{0.666} & 1.480 & 0.933 & \textbf{0.370} & 0.803 & 0.518 & \textbf{4.763} & 7.397 & 2.130 & \textbf{172} & 172 & -- & \textbf{2461.337} & 3795.877 & -- & -- & -- & 0.230 & -- & -- & 0.116 & -- & -- & 0.550 \\
+++++++Buddah Tooth Relic Temple Singapore & \textbf{1.131} & 1.791 & 1.030 & \textbf{0.280} & 0.520 & 0.233 & \textbf{9.964} & 14.029 & 2.060 & \textbf{162} & 162 & -- & \textbf{2916.837} & 4289.503 & -- & -- & -- & 0.081 & -- & -- & 0.014 & -- & -- & 0.850 \\
+++++++Doge Palace Venice & 0.753 & -- & 1.163 & 0.209 & -- & 0.342 & 6.359 & -- & 3.620 & 241 & -- & -- & 22522.737 & -- & -- & -- & -- & 0.211 & -- & -- & 0.029 & -- & -- & 1 \\
+++++++Door Lund & \textbf{0.014} & 6.960 & 0.024 & \textbf{0.004} & 0.512 & 0.006 & \textbf{1.812} & 7.101 & 0.320 & \textbf{12} & 12 & -- & \textbf{4888.363} & 9252.735 & -- & -- & -- & 0.006 & -- & -- & 0.001 & -- & -- & 0.300 \\
+++++++Drinking Fountain Somewhere In Zurich & \textbf{17.193} & 17.329 & 0.031 & 0.753 & \textbf{0.750} & 0.004 & \textbf{6.058} & 6.675 & 0.330 & \textbf{14} & 14 & -- & \textbf{883.837} & 5815.220 & -- & -- & -- & 0.007 & -- & -- & 0.002 & -- & -- & 0.310 \\
+++++++East Indiaman Goteborg & \textbf{5.568} & 7.231 & 3.814 & \textbf{0.979} & 1.364 & 0.621 & \textbf{7.318} & 10.130 & 4.130 & \textbf{179} & 179 & -- & \textbf{3987.183} & 5198.093 & -- & -- & -- & 3.117 & -- & -- & 0.509 & -- & -- & 1.850 \\
+++++++Ecole Superior De Guerre & \textbf{0.349} & 51.327 & 0.318 & \textbf{0.090} & 2.574 & 0.081 & \textbf{3.418} & 11.751 & 0.720 & \textbf{35} & 35 & -- & \textbf{2380.387} & 10426.770 & -- & -- & -- & 0.024 & -- & -- & 0.005 & -- & -- & 0.340 \\
+++++++Eglise du dome & \textbf{0.386} & 0.853 & 0.808 & \textbf{0.098} & 0.232 & 0.205 & \textbf{3.987} & 5.944 & 0.910 & \textbf{85} & 85 & -- & 19631.293 & \textbf{9803.940} & -- & -- & -- & 0.037 & -- & -- & 0.010 & -- & -- & 0.270 \\
+++++++Folke Filbyter & 85.831 & \textbf{82.656} & 74.596 & \textbf{0.129} & 0.131 & 0.125 & \textbf{19.275} & 28.225 & 10.370 & \textbf{40} & 40 & -- & \textbf{1581.897} & 3755.193 & -- & -- & -- & 70.157 & -- & -- & 0.118 & -- & -- & 4.290 \\
+++++++Fort Channing Gate Singapore & \textbf{0.093} & 0.122 & 0.207 & \textbf{0.041} & 0.056 & 0.093 & \textbf{1.783} & 3.649 & 0.520 & \textbf{27} & 27 & -- & \textbf{3434.113} & 6183.925 & -- & -- & -- & 0.020 & -- & -- & 0.008 & -- & -- & 0.250 \\
+++++++Golden Statue Somewhere In Hong Kong & 0.232 & -- & 0.292 & 0.053 & -- & 0.073 & 1.829 & -- & 0.400 & 18 & -- & -- & 4095.853 & -- & -- & -- & -- & 0.031 & -- & -- & 0.004 & -- & -- & 0.270 \\
+++++++Gustav Vasa & \textbf{3.509} & 25.854 & 34.181 & \textbf{0.220} & 0.775 & 1.085 & \textbf{1.655} & 4.747 & 3.520 & \textbf{18} & 18 & -- & \textbf{709.867} & 3887.653 & -- & -- & -- & 32.266 & -- & -- & 1.145 & -- & -- & 3.150 \\
+++++++GustavIIAdolf & \textbf{47.126} & 88.194 & 67.784 & \textbf{8.536} & 13.129 & 9.714 & \textbf{11.722} & 17.220 & 13.910 & \textbf{57} & 57 & -- & \textbf{1044.170} & 2981.017 & -- & -- & -- & 58.458 & -- & -- & 8.524 & -- & -- & 11.490 \\
+++++++Jonas Ahlstromer & 48.740 & \textbf{46.584} & 50.190 & \textbf{9.823} & 10.245 & 10.888 & \textbf{11.136} & 11.144 & 10.820 & \textbf{40} & 40 & -- & \textbf{615.450} & 2414.653 & -- & -- & -- & 47.117 & -- & -- & 10.451 & -- & -- & 8.410 \\
+++++++Kings College University Of Toronto & \textbf{9.025} & 18.154 & 0.989 & 2.044 & \textbf{1.706} & 0.235 & \textbf{3.565} & 6.501 & 0.900 & \textbf{77} & 77 & -- & \textbf{1078.980} & 2854.715 & -- & -- & -- & 0.085 & -- & -- & 0.017 & -- & -- & 0.340 \\
+++++++Lund University Sphinx & \textbf{11.392} & 42.959 & 19.522 & \textbf{2.981} & 8.720 & 4.585 & \textbf{6.056} & 15.000 & 4.780 & \textbf{70} & 70 & -- & \textbf{2718.567} & 4705.610 & -- & -- & -- & 8.752 & -- & -- & 2.191 & -- & -- & 1.360 \\
+++++++Nijo Castle Gate & \textbf{0.839} & 8.641 & 1.495 & \textbf{0.165} & 2.791 & 0.286 & \textbf{7.415} & 20.967 & 1.700 & \textbf{19} & 19 & -- & \textbf{1095.177} & 3241.330 & -- & -- & -- & 0.069 & -- & -- & 0.012 & -- & -- & 0.730 \\
+++++++Pantheon Paris & -- & 0.509 & 0.192 & -- & 0.066 & 0.050 & -- & 7.493 & 1.470 & -- & 179 & -- & -- & 5565.670 & -- & -- & -- & 0.040 & -- & -- & 0.005 & -- & -- & 0.490 \\
+++++++Park Gate Clermont Ferrand & 21.388 & -- & 0.391 & 10.192 & -- & 0.125 & 7.712 & -- & 0.570 & 34 & -- & -- & 1596.940 & -- & -- & -- & -- & 0.049 & -- & -- & 0.022 & -- & -- & 0.350 \\
+++++++Plaza De Armas Santiago & 0.775 & -- & 6.782 & 0.337 & -- & 2.944 & 6.009 & -- & 7.400 & 240 & -- & -- & 6372.683 & -- & -- & -- & -- & 2.556 & -- & -- & 1.383 & -- & -- & 4.900 \\
+++++++Porta San Donato Bologna & 0.592 & -- & 2.153 & 0.107 & -- & 0.388 & 5.603 & -- & 2.280 & 141 & -- & -- & 3754.163 & -- & -- & -- & -- & 0.095 & -- & -- & 0.046 & -- & -- & 0.750 \\
+++++++Round Church Cambridge & 2.137 & -- & 2.451 & 0.927 & -- & 1.003 & 6.103 & -- & 2.660 & 92 & -- & -- & 9779.270 & -- & -- & -- & -- & 1.107 & -- & -- & 0.582 & -- & -- & 1.540 \\
+++++++Skansen Kronan Gothenburg & 0.301 & -- & 0.736 & 0.102 & -- & 0.226 & 2.906 & -- & 1.240 & 131 & -- & -- & 17153.677 & -- & -- & -- & -- & 0.026 & -- & -- & 0.008 & -- & -- & 0.670 \\
+++++++Smolny Cathedral St Petersburg & 21.775 & -- & 0.554 & 2.111 & -- & 0.051 & 11.239 & -- & 1.660 & 131 & -- & -- & 10362.787 & -- & -- & -- & -- & 0.033 & -- & -- & 0.006 & -- & -- & 0.810 \\
+++++++Some Cathedral In Barcelona & 0.651 & -- & 0.880 & 0.233 & -- & 0.315 & 6.271 & -- & 2.870 & 177 & -- & -- & 5001.973 & -- & -- & -- & -- & 0.026 & -- & -- & 0.011 & -- & -- & 0.890 \\
+++++++Sri Mariamman Singapore & 1.219 & -- & 2.302 & 0.382 & -- & 0.683 & 10.088 & -- & 4.130 & 222 & -- & -- & 5985.540 & -- & -- & -- & -- & 0.077 & -- & -- & 0.023 & -- & -- & 0.910 \\
+++++++Sri Thendayuthapani Singapore & 0.593 & -- & 46.269 & 0.150 & -- & 3.812 & 7.016 & -- & 23.370 & 98 & -- & -- & 13338.167 & -- & -- & -- & -- & 44.170 & -- & -- & 2.870 & -- & -- & 8.440 \\
+++++++Sri Veeramakaliamman Singapore & 2.404 & -- & 2.559 & 0.559 & -- & 0.597 & 12.234 & -- & 3.470 & 157 & -- & -- & 12377.927 & -- & -- & -- & -- & 0.175 & -- & -- & 0.040 & -- & -- & 0.730 \\
+++++++Statue Of Liberty & 52.358 & -- & 46.887 & 28.503 & -- & 20.012 & 235036.825 & -- & 26.160 & 134 & -- & -- & 99.900 & -- & -- & -- & -- & 9.091 & -- & -- & 4.122 & -- & -- & 6.970 \\
+++++++The Pumpkin & 25.731 & -- & 94.672 & 5.552 & -- & 14.890 & 26.876 & -- & 33.410 & 196 & -- & -- & 4327.273 & -- & -- & -- & -- & 98.862 & -- & -- & 14.952 & -- & -- & 24.850 \\
+++++++Thian Hook Keng Temple Singapore & 0.927 & -- & 0.832 & 0.093 & -- & 0.082 & 14.915 & -- & 2.750 & 138 & -- & -- & 3298.537 & -- & -- & -- & -- & 0.081 & -- & -- & 0.008 & -- & -- & 1.130 \\
+++++++Tsar Nikolai I & 42.166 & -- & 48.499 & 8.732 & -- & 9.467 & 11.210 & -- & 9.790 & 98 & -- & -- & 4929.200 & -- & -- & -- & -- & 36.280 & -- & -- & 7.836 & -- & -- & 6.530 \\
+++++++Urban II & 58.919 & -- & 47.490 & 11.006 & -- & 9.467 & 17.538 & -- & 9.380 & 96 & -- & -- & 8889.617 & -- & -- & -- & -- & 48.214 & -- & -- & 9.586 & -- & -- & 6.920 \\
+++++++Vercingetorix & 82.952 & -- & 69.328 & 10.195 & -- & 8.788 & 7.257 & -- & 5.080 & 69 & -- & -- & 1223.970 & -- & -- & -- & -- & 17.706 & -- & -- & 3.104 & -- & -- & 1.500 \\
+++++++Yueh Hai Ching Temple Singapore & 0.544 & -- & 0.720 & 0.075 & -- & 0.098 & 5.596 & -- & 0.940 & 43 & -- & -- & 1748.667 & -- & -- & -- & -- & 0.043 & -- & -- & 0.014 & -- & -- & 0.650 \\
++++++ \midrule
++++++-Mean & \textbf{16.146} & 23.697 & 18.023 & \textbf{3.121} & 4.014 & 2.912 & 6920.602 & \textbf{13.088} & 5.673 & 100.559 & \textbf{104.818} & -- & \textbf{5819.728} & 7358.187 & -- & -- & -- & 13.695 & -- & -- & 1.937 & -- & -- & 2.992 \\
+++++++Mean & \textbf{15.676} & 23.604 & 17.547 & 3.032 & \textbf{2.619} & 2.840 & 6723.056 & \textbf{10.792} & 5.595 & \textbf{102.743} & 77.588 & -- & 5460.373 & \textbf{5336.838} & -- & -- & -- & 13.315 & -- & -- & 1.883 & -- & -- & 2.933 \\
++++++ \bottomrule
++++++ \end{tabular}}
++++++ \end{table*}
+++++ ```
+++++ - untracked/modified files:
+++++ ```
++++++M code/results/single_scene/REPRO.md
++++++ M code/results/single_scene/summary.csv
++++++ M code/results/single_scene/summary_table.md
++++++ M code/results/single_scene/summary_table.tex
+++++ ?? MVG_Project_Report_Ortal_Dayan.pdf
+++++ ?? "RESFM- ROBUST DEEP EQUIVARIANT STRUCTURE FROM MOTION.pdf"
+++++ ?? "claude specs/SPEC_cvpr_experiments.md"
+++++ ?? "claude specs/SPEC_sfm_datasets_setup.md"
+++++ ?? "claude specs/SPEC_single_scene_experiments.md"
+++++ ?? "claude specs/SPEC_uesfm_combined.md"
++++++?? "claude specs/TASK_crossdataset_readiness.md"
+++++ ?? code/datasets/Euclidean
+++++ ?? tmp_ab_check/
+++++ ```
+++++ ### esfm-baseline (official ESFM)
+++++ - path: `/home/projects/bagon/ortalda/MVG/final-project/esfm-baseline`
+++++-- commit: `2c0079497bf5dda1eeb2a902b2f859c6387f213a`
++++++- commit: `de666af53423c1dc3dd412e53c4e153c03d70922`
+++++ - **WARNING: working tree dirty.** Diff:
+++++ ```diff
+++++ (untracked files only)
+++++@@ -58,6 +1488,7 @@ Generated: 2026-07-13T20:41:04
+++++ ?? code/confs/single_scene_generated/ss_esfm_Buddah_Tooth_Relic_Temple_Singapore_seed1.conf
+++++ ?? code/confs/single_scene_generated/ss_esfm_Buddah_Tooth_Relic_Temple_Singapore_seed2.conf
+++++ ?? code/confs/single_scene_generated/ss_esfm_Doge_Palace_Venice_seed1.conf
++++++?? code/confs/single_scene_generated/ss_esfm_Doge_Palace_Venice_seed2.conf
+++++ ?? code/confs/single_scene_generated/ss_esfm_Door_Lund_seed1.conf
+++++ ?? code/confs/single_scene_generated/ss_esfm_Door_Lund_seed2.conf
+++++ ?? code/confs/single_scene_generated/ss_esfm_Drinking_Fountain_Somewhere_In_Zurich_seed1.conf
+++++@@ -70,6 +1501,7 @@ Generated: 2026-07-13T20:41:04
+++++ ?? code/confs/single_scene_generated/ss_esfm_Ecole_Superior_De_Guerre_seed2.conf
+++++ ?? code/confs/single_scene_generated/ss_esfm_Eglise_du_dome_seed0.conf
+++++ ?? code/confs/single_scene_generated/ss_esfm_Eglise_du_dome_seed1.conf
++++++?? code/confs/single_scene_generated/ss_esfm_Eglise_du_dome_seed2.conf
+++++ ?? code/confs/single_scene_generated/ss_esfm_Folke_Filbyter_seed0.conf
+++++ ?? code/confs/single_scene_generated/ss_esfm_Folke_Filbyter_seed1.conf
+++++ ?? code/confs/single_scene_generated/ss_esfm_Folke_Filbyter_seed2.conf
+++++@@ -114,16 +1546,22 @@ Generated: 2026-07-13T20:41:04
+++++ ?? code/confs/single_scene_generated/ss_esfm_Round_Church_Cambridge_seed2.conf
+++++ ?? code/confs/single_scene_generated/ss_esfm_Skansen_Kronan_Gothenburg_seed0.conf
+++++ ?? code/confs/single_scene_generated/ss_esfm_Skansen_Kronan_Gothenburg_seed1.conf
++++++?? code/confs/single_scene_generated/ss_esfm_Skansen_Kronan_Gothenburg_seed2.conf
+++++ ?? code/confs/single_scene_generated/ss_esfm_Smolny_Cathedral_St_Petersburg_seed0.conf
+++++ ?? code/confs/single_scene_generated/ss_esfm_Smolny_Cathedral_St_Petersburg_seed1.conf
++++++?? code/confs/single_scene_generated/ss_esfm_Smolny_Cathedral_St_Petersburg_seed2.conf
+++++ ?? code/confs/single_scene_generated/ss_esfm_Some_Cathedral_In_Barcelona_seed0.conf
++++++?? code/confs/single_scene_generated/ss_esfm_Some_Cathedral_In_Barcelona_seed1.conf
++++++?? code/confs/single_scene_generated/ss_esfm_Some_Cathedral_In_Barcelona_seed2.conf
+++++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Mariamman_Singapore_seed0.conf
+++++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Mariamman_Singapore_seed1.conf
+++++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Mariamman_Singapore_seed2.conf
+++++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Thendayuthapani_Singapore_seed0.conf
+++++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Thendayuthapani_Singapore_seed1.conf
++++++?? code/confs/single_scene_generated/ss_esfm_Sri_Thendayuthapani_Singapore_seed2.conf
+++++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Veeramakaliamman_Singapore_seed0.conf
+++++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Veeramakaliamman_Singapore_seed1.conf
++++++?? code/confs/single_scene_generated/ss_esfm_Sri_Veeramakaliamman_Singapore_seed2.conf
+++++ ?? code/confs/single_scene_generated/ss_esfm_Statue_Of_Liberty_seed0.conf
+++++ ?? code/confs/single_scene_generated/ss_esfm_Statue_Of_Liberty_seed1.conf
+++++ ?? code/confs/single_scene_generated/ss_esfm_Statue_Of_Liberty_seed2.conf
+++++@@ -137,10 +1575,14 @@ Generated: 2026-07-13T20:41:04
+++++ ?? code/confs/single_scene_generated/ss_esfm_Tsar_Nikolai_I_seed1.conf
+++++ ?? code/confs/single_scene_generated/ss_esfm_Tsar_Nikolai_I_seed2.conf
+++++ ?? code/confs/single_scene_generated/ss_esfm_Urban_II_seed0.conf
++++++?? code/confs/single_scene_generated/ss_esfm_Urban_II_seed1.conf
++++++?? code/confs/single_scene_generated/ss_esfm_Urban_II_seed2.conf
+++++ ?? code/confs/single_scene_generated/ss_esfm_Vercingetorix_seed0.conf
+++++ ?? code/confs/single_scene_generated/ss_esfm_Vercingetorix_seed1.conf
+++++ ?? code/confs/single_scene_generated/ss_esfm_Vercingetorix_seed2.conf
+++++ ?? code/confs/single_scene_generated/ss_esfm_Yueh_Hai_Ching_Temple_Singapore_seed0.conf
++++++?? code/confs/single_scene_generated/ss_esfm_Yueh_Hai_Ching_Temple_Singapore_seed1.conf
++++++?? code/confs/single_scene_generated/ss_esfm_Yueh_Hai_Ching_Temple_Singapore_seed2.conf
+++++ ```
+++++ 
+++++ ## Environment (shared venv used for BOTH methods)
+++++diff --git a/code/results/single_scene/summary.csv b/code/results/single_scene/summary.csv
+++++index e4f9f5e..0634f5d 100644
+++++--- a/code/results/single_scene/summary.csv
++++++++ b/code/results/single_scene/summary.csv
+++++@@ -2,171 +2,143 @@ method,scene,seed,rot_err_deg,pos_err,reproj_err_px,reproj_med_px,n_cams,runtime
+++++ esfm,Alcatraz Courtyard,0,0.324288380204305,0.0854554642858637,3.379232694353053,1.9204072326560708,133,4821.4,4792.707528591156,99999.0
+++++ esfm,Alcatraz Courtyard,1,0.36138678157391557,0.09112912757839253,3.3691035794153503,1.8536696768772378,133,4825.72,4804.594581127167,99999.0
+++++ esfm,Alcatraz Courtyard,2,0.4015043649148433,0.10235190268069758,3.557242072642648,1.9198007399315369,133,4833.02,4812.242679834366,99999.0
+++++-uesfm,Alcatraz Courtyard,0,0.6370266175309793,0.1671943803031883,5.956592777085197,3.6638282340102246,133,6478.06,6448.735792636871,99999.0
+++++-uesfm,Alcatraz Courtyard,1,0.5812669610362707,0.15343994793419002,5.71686544495911,3.550076086474883,133,6483.48,6454.087207078934,99999.0
+++++-uesfm,Alcatraz Courtyard,2,0.6293740753551684,0.16343948136157807,5.713459273661585,3.459422664018691,133,6483.73,6451.153926610947,99999.0
++++++uesfm,Alcatraz Courtyard,0,0.5509063563081962,0.14050635120746804,5.528957815039945,3.569220349534855,133,6556.5,6511.41095662117,99999.0
++++++uesfm,Alcatraz Courtyard,1,0.6823224031499351,0.17277455831381494,5.435931436314427,3.4771386471547348,133,6552.19,6506.419185638428,99999.0
+++++ esfm,Alcatraz Water Tower,0,0.6984432687370988,0.38694796813795407,4.741313233832226,2.1058898944784543,172,2461.59,2440.421297073364,99999.0
+++++ esfm,Alcatraz Water Tower,1,0.5944425806062753,0.3287537445614408,4.5656682619915765,1.9703698309044646,172,2467.28,2454.429310798645,99999.0
+++++ esfm,Alcatraz Water Tower,2,0.7051984712289041,0.39359395644708545,4.982802041506025,2.1318655747232063,172,2455.14,2441.783402681351,99999.0
+++++-uesfm,Alcatraz Water Tower,0,18.69941055700858,8.189402666249999,12.38813738647375,6.481578998967327,172,3857.94,3831.41609287262,99999.0
+++++-uesfm,Alcatraz Water Tower,1,1.3394060108035364,0.7384271683550214,6.577617375342355,3.200105455600137,172,3887.47,3865.343374729156,99999.0
+++++-uesfm,Alcatraz Water Tower,2,2.200189184784068,1.3839747978399466,6.854463337738994,3.1730434024886014,172,3819.31,3799.390902757645,99999.0
++++++uesfm,Alcatraz Water Tower,0,2.016029335283383,1.087082854315953,7.9502009315303575,4.413488326253207,172,3805.19,3775.908159971237,99999.0
++++++uesfm,Alcatraz Water Tower,1,1.4259772151091428,0.7645994701417347,7.177871865477131,3.7683369835989504,172,3787.07,3763.27596282959,99999.0
++++++uesfm,Alcatraz Water Tower,2,0.9982626119141534,0.558403777925872,7.063395425015731,3.814475911180562,172,3795.37,3773.579354286194,99999.0
+++++ esfm,Buddah Tooth Relic Temple Singapore,0,1.3132762212436329,0.31652548993670354,9.903890396062208,4.634952217191846,162,2921.45,2903.059753894806,99999.0
+++++ esfm,Buddah Tooth Relic Temple Singapore,1,0.9908742070528852,0.2335767172234499,9.836436953133868,4.661767906617449,162,2915.39,2904.387546062469,99999.0
+++++ esfm,Buddah Tooth Relic Temple Singapore,2,1.089403690069201,0.2896515999941069,10.151103921611115,4.628183275251743,162,2913.67,2902.928613901138,99999.0
+++++-uesfm,Buddah Tooth Relic Temple Singapore,0,1.9767734026333772,0.5777116480820644,15.584999477939833,8.800407928031486,162,4315.97,4296.739196300507,99999.0
+++++-uesfm,Buddah Tooth Relic Temple Singapore,1,1.4532821126601407,0.345133635718661,16.753182673783428,9.555989756393366,162,4334.28,4311.474883317947,99999.0
+++++-uesfm,Buddah Tooth Relic Temple Singapore,2,32.81105779154638,6.0024233739495125,26.564606536203048,13.925597303414058,162,4312.37,4291.042752504349,99999.0
++++++uesfm,Buddah Tooth Relic Temple Singapore,0,1.5799948547864517,0.43886498334860274,13.484389374830494,7.719072745032001,162,4294.34,4266.538989067078,99999.0
++++++uesfm,Buddah Tooth Relic Temple Singapore,1,1.934129180062467,0.5924991617860504,13.872322562031941,8.04977435062149,162,4303.62,4259.311170101166,99999.0
++++++uesfm,Buddah Tooth Relic Temple Singapore,2,1.8591664615796053,0.5279087705483619,14.731758217446671,8.481044153007865,162,4270.55,4249.088565349579,99999.0
+++++ esfm,Doge Palace Venice,0,0.9741701978804636,0.2578214102266912,6.511459231891615,3.6947627756511525,241,28795.18,28746.22650837898,99999.0
+++++ esfm,Doge Palace Venice,1,0.627860879835449,0.1773446161890865,6.351890452986879,3.537139014829222,241,28742.27,28699.91308617592,99999.0
+++++-uesfm,Doge Palace Venice,0,0.9110673544913167,0.2904399918743801,10.966183720673449,6.771416746021886,241,21074.95,21017.30448675156,99999.0
++++++esfm,Doge Palace Venice,2,0.6576305681966796,0.19114542381679772,6.214841374149415,3.591638795332466,241,10030.76,9989.16938996315,99999.0
+++++ esfm,Door Lund,0,0.01167051570787235,0.003654424593160691,1.8552512563951393,0.9492901559591532,12,4892.41,4878.351888656616,99999.0
+++++ esfm,Door Lund,1,0.009685491454411136,0.0030270923396151107,1.6462990657331118,0.9049495114630693,12,4884.51,4877.529450178146,99999.0
+++++ esfm,Door Lund,2,0.02068285629921104,0.005193423834514985,1.933833706356197,0.9506754982524039,12,4888.17,4880.201772212982,99999.0
+++++-uesfm,Door Lund,0,14.059454220682019,0.9733778175929079,11.85413335315025,6.517535151323047,12,9185.47,8715.343282699585,95000.0
+++++-uesfm,Door Lund,1,14.111993883089163,0.9618908892808385,11.853785941610944,6.504290333453439,12,9203.34,8729.614821910858,95000.0
+++++-uesfm,Door Lund,2,14.344083655704312,0.9193773651816372,12.058851721280973,6.643622749322514,12,9176.97,9162.127201795578,99999.0
++++++uesfm,Door Lund,0,0.03084538453455712,0.007522151486969653,2.3711871496098955,1.6703501976219666,12,9262.05,9243.771874427795,99999.0
++++++uesfm,Door Lund,1,13.889154901234134,1.0160235649641791,11.830787694675347,6.5775252925915195,12,9243.42,7848.296544790268,85000.0
+++++ esfm,Drinking Fountain Somewhere In Zurich,0,25.804179845300204,1.1242096722251314,8.219025541657992,3.3464550934365884,14,888.07,874.4874215126038,99999.0
+++++ esfm,Drinking Fountain Somewhere In Zurich,1,25.754577171528883,1.1271567319301725,8.245599860088616,3.352895635561939,14,882.33,873.2415659427643,99999.0
+++++ esfm,Drinking Fountain Somewhere In Zurich,2,0.019604308735801684,0.006780729271172703,1.7090486008047308,0.6375575382268279,14,881.11,873.6504108905792,99999.0
+++++-uesfm,Drinking Fountain Somewhere In Zurich,0,26.27952541668808,1.102546079947844,8.40971803772888,3.4593948661771203,14,2969.96,2950.480526924133,99999.0
+++++-uesfm,Drinking Fountain Somewhere In Zurich,1,26.264849988219943,1.1078406736783482,8.547474098662555,3.585334207185317,14,2960.9,2504.42572927475,85000.0
+++++-uesfm,Drinking Fountain Somewhere In Zurich,2,0.06235835436589852,0.011103715758719332,2.5743283079384534,1.5361135173722633,14,2960.08,2945.673992872238,99999.0
++++++uesfm,Drinking Fountain Somewhere In Zurich,0,25.946100100284117,1.1178920745623426,8.471712388928934,3.6241397282821626,14,5817.37,5796.999963760376,99999.0
++++++uesfm,Drinking Fountain Somewhere In Zurich,1,25.973070279327214,1.1166462082604052,8.538877295313494,3.6343523276029472,14,5821.94,5806.947921991348,99999.0
++++++uesfm,Drinking Fountain Somewhere In Zurich,2,0.06926036271356541,0.01532826351119577,3.015375672575656,1.7451989053599506,14,5806.35,5792.085475206375,99999.0
+++++ esfm,East Indiaman Goteborg,0,6.033696331509855,1.0773322362193534,7.427433157074123,2.292729458380311,179,3990.25,3970.732353687286,99999.0
+++++ esfm,East Indiaman Goteborg,1,5.313967690277326,0.9356515362150493,7.350753973659207,2.2747477951684703,179,3986.66,3970.914752960205,99999.0
+++++ esfm,East Indiaman Goteborg,2,5.357164002763642,0.9254557315236661,7.174538078735992,2.2493605774605823,179,3984.64,3967.810528755188,99999.0
+++++-uesfm,East Indiaman Goteborg,0,4.992588850190341,0.9093886386013725,9.930437318497303,4.011819888270754,179,5247.16,5221.816437721252,99999.0
+++++-uesfm,East Indiaman Goteborg,1,5.382092262718057,0.9927766021898647,10.035611563702533,3.912004668740913,179,5245.65,5210.653891563416,99999.0
+++++-uesfm,East Indiaman Goteborg,2,5.554715231405653,0.9752252406138763,9.5551521706325,3.7417116053580193,179,5231.48,5207.815300703049,99999.0
++++++uesfm,East Indiaman Goteborg,0,5.365772057886591,0.9552115949640507,9.655123223073948,4.174486084237091,179,5203.6,5178.753549814224,99999.0
++++++uesfm,East Indiaman Goteborg,1,10.45533482189932,2.0687908277631517,9.98932850274456,4.110771460953736,179,5195.98,5174.958552598953,99999.0
++++++uesfm,East Indiaman Goteborg,2,5.872735570127935,1.0672842053553182,10.745053220477157,4.480528977857714,179,5194.7,5173.540862560272,99999.0
+++++ esfm,Ecole Superior De Guerre,0,0.4863901164687255,0.1239377922325492,3.750666445958877,2.0528113108131016,35,2385.33,2371.708672523499,99999.0
+++++ esfm,Ecole Superior De Guerre,1,0.3031794677170201,0.07921628898965839,3.2456935905644406,1.730123845260616,35,2379.34,2371.191816806793,99999.0
+++++ esfm,Ecole Superior De Guerre,2,0.2580439805296837,0.06772089648290676,3.257951412507941,1.687291979251782,35,2376.49,2368.935952663422,99999.0
+++++-uesfm,Ecole Superior De Guerre,0,0.28243957678843373,0.0778675008788964,5.291358032458062,3.601773151002547,35,4647.51,4622.850021600723,99999.0
+++++-uesfm,Ecole Superior De Guerre,1,0.25507313740145815,0.06695766472508183,5.170495431958795,3.5179325723573998,35,4636.07,4616.890106916428,99999.0
+++++-uesfm,Ecole Superior De Guerre,2,0.3137499276035537,0.08683967313665406,5.583934687507232,3.7813953498055604,35,4617.92,4602.193422079086,99999.0
++++++uesfm,Ecole Superior De Guerre,0,51.32681795107141,2.57407526776939,11.751384768771835,6.449293516058521,35,10426.77,10405.3174200058,99999.0
+++++ esfm,Eglise du dome,0,0.3242410065346163,0.0824797769992069,3.906129661328729,1.3512292775673158,85,25054.74,25028.64922428131,99999.0
+++++ esfm,Eglise du dome,1,0.3771623368290559,0.09432763361451474,3.692490941110161,1.297032357665518,85,25144.16,25116.01952123642,99999.0
+++++-uesfm,Eglise du dome,0,45.800562123362404,6.451383786272971,7.963674120656965,3.986938865913781,85,23371.73,23333.49408817291,99999.0
++++++esfm,Eglise du dome,2,0.45623311880194967,0.11755570327842058,4.363674496823582,1.384257241090686,85,8694.98,8671.502047538757,99999.0
++++++uesfm,Eglise du dome,0,0.8528700994829498,0.23193079249562099,5.944108200898114,2.9306630821877517,85,9803.94,9764.585114717484,99999.0
+++++ esfm,Folke Filbyter,0,89.03873163576341,0.12895560134543632,17.986624161769267,7.423093846189277,40,1582.8,1573.360969305038,99999.0
+++++ esfm,Folke Filbyter,1,84.74740427947457,0.13132264058966686,17.98706393015447,7.385900243615879,40,1579.13,1572.079373121262,99999.0
+++++ esfm,Folke Filbyter,2,83.70817707245973,0.1261878953205589,21.85200129525089,6.324245998829542,40,1583.76,1576.207699537277,99999.0
+++++-uesfm,Folke Filbyter,0,80.34590333370599,0.13169898199385566,22.83327162619912,11.711593158296864,40,3701.49,3129.452286243439,85000.0
+++++-uesfm,Folke Filbyter,1,81.28676564061475,0.1282318052949447,24.480394314414323,9.74969805057316,40,3698.11,3681.891911745071,99999.0
+++++-uesfm,Folke Filbyter,2,82.23270227083374,0.1312531386108397,22.153947807931974,10.961536903594071,40,3709.96,3695.84988451004,99999.0
++++++uesfm,Folke Filbyter,0,84.55526303793802,0.13091275274611416,29.190267154194732,11.160073225547443,40,3747.65,3727.990335702896,99999.0
++++++uesfm,Folke Filbyter,1,84.87047537526948,0.13112291273275417,32.36503566471496,13.743964264262598,40,3743.53,3728.497133255005,99999.0
++++++uesfm,Folke Filbyter,2,78.54375401873024,0.1316690349103016,23.11854509839131,10.94439865744236,40,3774.4,3007.95537352562,80000.0
+++++ esfm,Fort Channing Gate Singapore,0,0.10143646071220622,0.04531774433800129,1.7158776598376824,0.8165178663538488,27,3438.39,3428.288990497589,99999.0
+++++ esfm,Fort Channing Gate Singapore,1,0.08873951454303304,0.039001318240256114,1.783712095722291,0.8156020254191567,27,3430.88,3424.490573883057,99999.0
+++++ esfm,Fort Channing Gate Singapore,2,0.09031224636384508,0.03957289173255022,1.850800887931411,0.820658413919779,27,3433.07,3426.547735452652,99999.0
+++++-uesfm,Fort Channing Gate Singapore,0,33.158343156383296,11.908537566474344,12.69248500308198,5.438758486477886,27,6209.29,6192.107179880142,99999.0
+++++-uesfm,Fort Channing Gate Singapore,1,0.07319798057764343,0.03541606344760984,3.4819579894798927,2.2085534812612853,27,6159.7,6144.797720909119,99999.0
+++++-uesfm,Fort Channing Gate Singapore,2,0.15394831153888047,0.0686769122909015,3.77993377770425,2.2995094868650368,27,6147.27,6132.745970487595,99999.0
++++++uesfm,Fort Channing Gate Singapore,0,0.10214634402234797,0.048923384893640356,3.7907267588398517,2.449846070018404,27,6175.88,6156.048281908035,99999.0
++++++uesfm,Fort Channing Gate Singapore,1,0.14219758622796153,0.06357969294387673,3.5066994916782015,2.1555120267718575,27,6191.97,6176.711992740631,99999.0
+++++ esfm,Golden Statue Somewhere In Hong Kong,0,0.23894829924734365,0.053817571637812096,1.7239352536484445,0.744089758310296,18,4096.74,4087.775670289993,99999.0
+++++ esfm,Golden Statue Somewhere In Hong Kong,1,0.2233990458438394,0.053061028269618456,1.8936878914364637,0.7746436618607616,18,4096.28,4088.994391202927,99999.0
+++++ esfm,Golden Statue Somewhere In Hong Kong,2,0.23397914670516012,0.051137498616263405,1.87070255923672,0.7763877145267694,18,4094.54,4087.674698591232,99999.0
+++++-uesfm,Golden Statue Somewhere In Hong Kong,0,0.36074365888948035,0.06894135246785482,2.939092008443413,1.863811153764833,18,7761.69,7746.569520235062,99999.0
+++++-uesfm,Golden Statue Somewhere In Hong Kong,1,0.35638930493960375,0.06132592684828702,2.686157398440538,1.735571387461727,18,7759.16,7745.389947891235,99999.0
+++++-uesfm,Golden Statue Somewhere In Hong Kong,2,0.3955566726941678,0.09197790539367184,2.878341172309852,1.8164738202745854,18,7766.46,7751.458931922913,99999.0
+++++ esfm,Gustav Vasa,0,3.4881363194229826,0.2273181104934767,1.7369784534725496,0.9521606063342205,18,711.53,703.3091380596161,99999.0
+++++ esfm,Gustav Vasa,1,3.4307883498221035,0.21432466556133864,1.5556448596237022,0.9232184455756751,18,710.28,703.9906163215637,99999.0
+++++ esfm,Gustav Vasa,2,3.6089062562451386,0.2196578287395002,1.673803634770729,0.9429755567757121,18,707.79,701.5903799533844,99999.0
+++++-uesfm,Gustav Vasa,0,5.828154714648938,0.32510821331670736,3.0711042143581646,2.4170166616864357,18,2566.0,2547.931907176971,99999.0
+++++-uesfm,Gustav Vasa,1,6.515239842974278,0.35941242702736925,3.089677100739632,2.4476428398062584,18,2530.9,1385.089058637619,55000.0
+++++-uesfm,Gustav Vasa,2,5.954036423569478,0.34942942401850696,3.101296679577594,2.400483554095863,18,2533.76,2520.798699855804,99999.0
++++++uesfm,Gustav Vasa,0,36.645536646380805,1.0070193571286277,4.980488290477233,3.197977230260218,18,2562.94,2419.297919034958,95000.0
++++++uesfm,Gustav Vasa,1,36.4914104514929,1.03254185606701,5.377024289258223,3.4818421684593366,18,4566.81,4543.375423431396,99999.0
++++++uesfm,Gustav Vasa,2,4.4238740734834625,0.284762443333143,3.8820183248893803,2.8663959682636566,18,4533.21,4516.61039352417,99999.0
+++++ esfm,GustavIIAdolf,0,46.92758389264633,8.570499881283022,11.684017169857798,5.131125459552879,57,1047.82,1038.018162250519,99999.0
+++++ esfm,GustavIIAdolf,1,47.21999673071185,8.515052769156952,11.695065125063977,5.091617086264161,57,1041.41,1034.5799472332,99999.0
+++++ esfm,GustavIIAdolf,2,47.2312223396789,8.523261295519099,11.787583212607423,5.2715388868550574,57,1043.28,1036.478415489197,99999.0
+++++-uesfm,GustavIIAdolf,0,89.37607436538163,13.276440526172165,15.53883714368065,5.864332719302596,57,2904.42,2881.687849521637,99999.0
+++++-uesfm,GustavIIAdolf,1,70.16367503302509,11.555242253627961,15.162174576474532,6.9709512967490355,57,2884.62,2870.103493452072,99999.0
+++++-uesfm,GustavIIAdolf,2,72.72996778765942,11.054260851523775,16.85269001998736,9.009102977660282,57,2888.7,2874.096453428268,99999.0
++++++uesfm,GustavIIAdolf,0,85.51179144359693,12.792264407383987,17.018024925092185,8.196266635685827,57,2973.35,2956.916826248169,99999.0
++++++uesfm,GustavIIAdolf,1,88.9494736685426,13.260723460498589,17.699982689478404,8.888010081099381,57,2995.3,2976.055320978165,99999.0
++++++uesfm,GustavIIAdolf,2,90.12096202318821,13.335169188185864,16.941538974035925,7.772181288890227,57,2974.4,2960.110015630722,99999.0
+++++ esfm,Jonas Ahlstromer,0,64.46662959289324,11.985605264945464,12.602584968179986,2.9168445332917745,40,614.88,607.3498430252075,99999.0
+++++ esfm,Jonas Ahlstromer,1,38.42510202464649,8.741159740710629,10.25268858530355,3.7024742418884333,40,618.73,611.5567183494568,99999.0
+++++ esfm,Jonas Ahlstromer,2,43.32891670826628,8.741053567623224,10.553031866494822,3.031717304343782,40,612.74,606.6802380084991,99999.0
+++++-uesfm,Jonas Ahlstromer,0,42.30073879858729,10.272232427418913,9.098970988131883,3.992857074878733,40,2339.57,1280.604686737061,55000.0
+++++-uesfm,Jonas Ahlstromer,1,42.18412804385421,10.310808592826955,8.453808908218651,3.4329852879523464,40,2334.33,2320.362742424011,99999.0
+++++-uesfm,Jonas Ahlstromer,2,42.25301394187691,10.315519611163806,8.793789077520445,3.8046125347596442,40,2311.04,2297.139800786972,99999.0
++++++uesfm,Jonas Ahlstromer,0,45.264596178655594,9.403212752141883,11.336617283269852,4.728596140494824,40,2421.11,2401.221394062042,99999.0
++++++uesfm,Jonas Ahlstromer,1,51.490398556789884,10.860548616493958,12.821407884052217,5.012487046810724,40,2428.25,2415.139223337173,99999.0
++++++uesfm,Jonas Ahlstromer,2,42.99784237868634,10.470910789899067,9.274184391894982,4.4612757046777345,40,2394.6,2381.585824012756,99999.0
+++++ esfm,Kings College University Of Toronto,0,14.400461796497643,3.2653370282498866,4.064353916685246,1.6753245173381437,77,1074.94,1066.306197404861,99999.0
+++++ esfm,Kings College University Of Toronto,1,11.956699445712928,2.705452710546247,3.9727144140636783,1.5797087908756111,77,1071.64,1064.129351377487,99999.0
+++++ esfm,Kings College University Of Toronto,2,0.7191591237413282,0.16178494376153177,2.656807607821862,1.1949677972638288,77,1090.36,1082.987608671188,99999.0
+++++-uesfm,Kings College University Of Toronto,0,31.932532001930838,2.9775501815475,6.099555590572752,3.8717713150174773,77,2884.23,2866.194714784622,99999.0
+++++-uesfm,Kings College University Of Toronto,1,2.4241467960672063,0.41577835428574456,4.631696476065898,3.148046400399828,77,2899.51,2884.114916801453,99999.0
+++++-uesfm,Kings College University Of Toronto,2,30.212552334890066,2.3209884971628156,5.844709739288878,3.754268841443037,77,2874.99,2860.901805400848,99999.0
++++++uesfm,Kings College University Of Toronto,0,29.200261744017546,2.0893308856114428,7.037455848998898,4.8227457854711675,77,2856.72,2838.923613071442,99999.0
++++++uesfm,Kings College University Of Toronto,1,7.107988087620218,1.3235014088907995,5.965363878755208,4.179664786063016,77,2852.71,2839.42994761467,99999.0
+++++ esfm,Lund University Sphinx,0,13.661465370705821,3.5591094267381553,6.242667318159146,2.715620555070548,70,2718.96,2709.5872194767,99999.0
+++++ esfm,Lund University Sphinx,1,12.524302160109885,3.2130942199657664,6.1182420215295075,2.7468468987195953,70,2722.86,2715.36418390274,99999.0
+++++ esfm,Lund University Sphinx,2,7.98961938632192,2.1710959765491817,5.807289242508196,2.893370267054225,70,2713.88,2706.478076219559,99999.0
+++++-uesfm,Lund University Sphinx,0,15.165716414733652,3.5564827651349713,10.803408462752863,6.730422995453065,70,4644.92,4626.135177612305,99999.0
+++++-uesfm,Lund University Sphinx,1,68.88899086055604,13.548881652837478,14.336626262363456,7.626356837439143,70,4703.31,4687.50173997879,99999.0
+++++-uesfm,Lund University Sphinx,2,15.014326075576053,3.5607694164340526,11.039432108003597,6.81081192121135,70,4615.6,4600.294319868088,99999.0
++++++uesfm,Lund University Sphinx,0,42.95863056778373,8.719684065980621,14.999945982069319,8.361965506089375,70,4705.61,4681.787243127823,99999.0
+++++ esfm,Nijo Castle Gate,0,0.6521524412276166,0.13305866403975986,7.404953325705311,3.0629804954987674,19,1097.29,1084.301982402802,99999.0
+++++ esfm,Nijo Castle Gate,1,0.8564043952718835,0.16950374865879153,7.358076475544799,3.1235223975185864,19,1094.89,1084.640900611877,99999.0
+++++ esfm,Nijo Castle Gate,2,1.0089926380330574,0.19095470213426863,7.481049991287358,3.1671779658403687,19,1093.35,1083.446812152863,99999.0
+++++-uesfm,Nijo Castle Gate,0,1.6923297320319277,0.3246009364794658,11.792009654594867,6.945622003724464,19,3310.13,3288.75860619545,99999.0
+++++-uesfm,Nijo Castle Gate,1,1.6697791743236226,0.32440429939345133,12.704681259536075,7.466128811395402,19,3308.19,3290.209539175034,99999.0
+++++-uesfm,Nijo Castle Gate,2,1.682403384345942,0.32817821817561266,11.917977420973372,7.102279949089778,19,3301.77,3284.964460611343,99999.0
+++++-uesfm,Pantheon Paris,0,0.3823220210788846,0.05987691585642601,7.649321203280243,4.797127317249956,179,5506.88,5473.694003582001,99999.0
+++++-uesfm,Pantheon Paris,1,45.61471478909275,6.671068486991568,16.912359249810606,6.602415648206237,179,5491.31,5459.329566717148,99999.0
+++++-uesfm,Pantheon Paris,2,12.03796171625132,2.465102828965073,16.125095834055482,6.860965809143959,179,5493.94,5463.555550813675,99999.0
++++++uesfm,Nijo Castle Gate,0,8.641303631787597,2.791258638759238,20.96738948203601,10.297911838853688,19,3241.33,3220.324777603149,99999.0
++++++uesfm,Pantheon Paris,0,0.5453004964472504,0.06999784414241637,7.790870638815493,5.057509442480535,179,5594.6,5562.967379331589,99999.0
++++++uesfm,Pantheon Paris,1,0.472097972346089,0.062163803587728925,7.194706475909503,4.639524309087925,179,5536.74,5501.721322774887,99999.0
+++++ esfm,Park Gate Clermont Ferrand,0,19.399216821978534,9.438471236639753,7.521969683469249,3.892136696028138,34,1597.96,1588.07075381279,99999.0
+++++ esfm,Park Gate Clermont Ferrand,1,19.325402297843297,9.417998214523331,7.465007292717891,3.849088919564891,34,1596.87,1589.703137874603,99999.0
+++++ esfm,Park Gate Clermont Ferrand,2,25.440611579781365,11.720641869372841,8.15032188787613,4.0871739024940315,34,1595.99,1588.841923236847,99999.0
+++++-uesfm,Park Gate Clermont Ferrand,0,19.27114964212729,9.381930985797803,7.856513901068924,4.317004881495189,34,3598.69,3580.396904230118,99999.0
+++++-uesfm,Park Gate Clermont Ferrand,1,19.288145181777256,9.394362801959678,7.762082007624792,4.207851936960509,34,3592.66,3577.494926214218,99999.0
+++++-uesfm,Park Gate Clermont Ferrand,2,19.345727373971844,9.415066249707772,7.936652645099164,4.425507919979627,34,3600.86,3581.473034620285,99999.0
+++++ esfm,Plaza De Armas Santiago,0,0.7341628728579594,0.32156050159900956,6.000154308231358,2.679971945741675,240,6375.02,6333.455562829971,99999.0
+++++ esfm,Plaza De Armas Santiago,1,0.5176165701053533,0.22744923951256435,5.661576462867733,2.533952088248649,240,6373.68,6336.030319213867,99999.0
+++++ esfm,Plaza De Armas Santiago,2,1.0740077415291467,0.46304622323619105,6.365716833190761,2.733195456670397,240,6369.35,6331.436369419098,99999.0
+++++-uesfm,Plaza De Armas Santiago,0,6.6772176498607925,2.909933722327364,11.384209379387066,5.623987151554922,240,6540.76,6491.619550704956,99999.0
+++++-uesfm,Plaza De Armas Santiago,1,1.3364898126442175,0.5686650140764532,7.951759184923103,3.933795015892702,240,6536.61,6482.320575714111,99999.0
+++++-uesfm,Plaza De Armas Santiago,2,1.9671483940705004,0.8276654865349118,9.054185015150745,4.441179120460681,240,6529.24,6477.723381996155,99999.0
+++++ esfm,Porta San Donato Bologna,0,0.5746001989174886,0.10608894230214515,5.691906895070585,2.4183460353027986,141,3751.75,3734.281898736954,99999.0
+++++ esfm,Porta San Donato Bologna,1,0.8978003419853328,0.14474404936935045,5.682379581165418,2.563141107750346,141,3758.51,3742.688964605331,99999.0
+++++ esfm,Porta San Donato Bologna,2,0.30440272958620346,0.07026546193502164,5.435311727517131,2.1626312981864086,141,3752.23,3738.056202173233,99999.0
+++++-uesfm,Porta San Donato Bologna,0,1.1269164513732723,0.19275737956525044,10.071225650489282,5.557466375146339,141,5111.57,5083.21807050705,99999.0
+++++-uesfm,Porta San Donato Bologna,1,60.66366912240625,10.234123128225525,19.593360464637342,9.305574874002819,141,5125.57,5098.085618019104,99999.0
+++++-uesfm,Porta San Donato Bologna,2,1.2271289404198877,0.19605254084885107,10.066750816092268,5.595831381287904,141,5116.95,5093.473650932312,99999.0
+++++ esfm,Round Church Cambridge,0,2.1173198662081005,0.9129854085205968,5.854320175654564,1.9683268910531708,92,9788.65,9759.719386577606,99999.0
+++++ esfm,Round Church Cambridge,1,2.1415279636494953,0.9267927094737857,6.33905951937897,2.0439997148640834,92,9764.61,9741.356333732605,99999.0
+++++-uesfm,Round Church Cambridge,0,2.5434984075631974,1.0219048587087352,8.914080922081759,4.131734829440289,92,12088.06,12047.02147889137,99999.0
+++++-uesfm,Round Church Cambridge,1,2.435781054756339,0.9827352719070902,8.178482224933683,3.8400051230290893,92,12034.64,11993.72608184814,99999.0
++++++esfm,Round Church Cambridge,2,2.1507951248804917,0.9401429799233081,6.115230559713554,2.0517209910251797,92,9784.55,9763.525354623795,99999.0
+++++ esfm,Skansen Kronan Gothenburg,0,0.3380139741078169,0.1158931606908392,3.0478862369293624,1.5340801117570733,131,17163.94,17140.15425729752,99999.0
+++++-uesfm,Skansen Kronan Gothenburg,0,0.37472494768162085,0.1243676312212316,4.798736338080396,3.348916057055834,131,16235.09,16202.21351981163,99999.0
++++++esfm,Skansen Kronan Gothenburg,1,0.21516831826827018,0.07458182746108981,2.786705021535272,1.4619994473169777,131,17129.62,17109.18022465706,99999.0
++++++esfm,Skansen Kronan Gothenburg,2,0.3508356515622903,0.11667319277987132,2.8836839507901773,1.5216584205042176,131,17167.47,17142.66050887108,99999.0
+++++ esfm,Smolny Cathedral St Petersburg,0,21.81278753416543,2.110667559831311,11.2315357584775,6.379259553776089,131,10364.67,10320.51852750778,99999.0
+++++ esfm,Smolny Cathedral St Petersburg,1,21.799828748358873,2.111313741354268,11.225141339870355,6.36605939788532,131,10360.52,10319.01374697685,99999.0
+++++-uesfm,Smolny Cathedral St Petersburg,0,22.143392894578028,2.0985584569799265,12.52645725687386,7.895265841408903,131,10475.74,10416.62969732285,99999.0
++++++esfm,Smolny Cathedral St Petersburg,2,21.71090600320656,2.110307413725059,11.259799763291888,6.4136596726420905,131,10363.17,10320.08415436745,99999.0
++++++esfm,Some Cathedral In Barcelona,0,0.6775058368464424,0.24064262464083658,6.20165095815897,3.7253636674062887,177,5002.81,4985.274734258652,99999.0
++++++esfm,Some Cathedral In Barcelona,1,0.6722147495861831,0.2423973792150467,6.350786792749083,3.720536755931007,177,5009.15,4991.552013158798,99999.0
++++++esfm,Some Cathedral In Barcelona,2,0.6046048207068493,0.21699252142655406,6.261075100347952,3.7258171467923633,177,4993.96,4975.845166683197,99999.0
+++++ esfm,Sri Mariamman Singapore,0,1.4920493985513708,0.4412389781578158,10.406406306167431,6.369585247957454,222,5987.98,5970.066485881805,99999.0
+++++ esfm,Sri Mariamman Singapore,1,1.1345564396944785,0.3608906720626337,9.703927906328842,5.885963785879842,222,5993.62,5975.365797519684,99999.0
+++++ esfm,Sri Mariamman Singapore,2,1.0293707921546553,0.3426709250774071,10.15220066698994,6.067537729054635,222,5975.02,5958.786058187485,99999.0
+++++-uesfm,Sri Mariamman Singapore,0,16.417883129790784,5.526406595652791,19.962711816743173,12.18830031311293,222,6531.45,6494.750840425491,99999.0
+++++-uesfm,Sri Mariamman Singapore,1,1.3793173305528281,0.5014534676018023,16.10857603648069,10.651562750176396,222,6467.65,6426.201541185379,99999.0
+++++-uesfm,Sri Mariamman Singapore,2,1.639001766282598,0.536397596737868,17.490398022274007,11.439337896248928,222,6457.9,6427.717175960541,99999.0
+++++ esfm,Sri Thendayuthapani Singapore,0,0.7190396606626565,0.17922550285921848,7.12365374281766,2.231392236341273,98,13362.09,13320.09781932831,99999.0
+++++-uesfm,Sri Thendayuthapani Singapore,0,47.0750455982536,3.8553135654690074,14.133068630999212,7.117179633590087,98,13808.44,13754.94554877281,99999.0
++++++esfm,Sri Thendayuthapani Singapore,1,0.5350953449295449,0.1261925950852554,6.4467364012557615,2.04778550672059,98,13347.85,13309.03068971634,99999.0
++++++esfm,Sri Thendayuthapani Singapore,2,0.5262093764563488,0.14374074415280041,7.476614790858722,2.2575164196527426,98,13304.56,13262.61834168434,99999.0
+++++ esfm,Sri Veeramakaliamman Singapore,0,2.333869921178247,0.542847254574005,11.965270419451508,5.472877662905037,157,12392.18,12365.15180635452,99999.0
+++++-uesfm,Sri Veeramakaliamman Singapore,0,29.40487246624615,3.8676026992817505,28.950500021302087,14.32295224594287,157,12544.69,12504.26410722733,99999.0
++++++esfm,Sri Veeramakaliamman Singapore,1,2.4804161216418734,0.5731176371311499,12.150997217479388,5.5309200508775875,157,12387.11,12363.29210400581,99999.0
++++++esfm,Sri Veeramakaliamman Singapore,2,2.3974569344813434,0.5599333960859275,12.58650507397198,5.711763361742226,157,12354.49,12330.03034973145,99999.0
+++++ esfm,Statue Of Liberty,0,52.35559762033152,28.241594953600423,688383.1603142131,683624.423417858,134,102.64,43.32754778862,45000.0
+++++ esfm,Statue Of Liberty,1,52.36920727661949,28.627801879821945,9277.715116667516,9303.546387753544,134,99.78,59.98358988761902,65000.0
+++++ esfm,Statue Of Liberty,2,52.3478568435799,28.64052858422677,7449.598084265948,7416.803834848509,134,97.28,3.520912647247314,0.0
+++++-uesfm,Statue Of Liberty,0,40.92379801166407,19.915794833689585,48.781849500680366,25.904882064330994,134,12927.11,12902.23110127449,99999.0
+++++-uesfm,Statue Of Liberty,1,38.79484508812604,17.090491683338875,33.27459817133403,17.608726936833005,134,12920.89,12896.32496738434,99999.0
+++++ esfm,The Pumpkin,0,27.59459354015253,5.94835892276036,27.52772578222832,9.11126349703041,196,4330.9,4316.109169960022,99999.0
+++++ esfm,The Pumpkin,1,26.54397520858153,5.701234509093555,27.342758332715125,8.907676505669826,196,4327.98,4314.043692827225,99999.0
+++++ esfm,The Pumpkin,2,23.05395646123413,5.007378502393024,25.75879879971585,7.898438464943201,196,4322.94,4310.356798410416,99999.0
+++++-uesfm,The Pumpkin,0,25.490608822831348,5.440568114393512,34.198429037653845,15.6024365790899,196,5793.03,5767.982522964478,99999.0
+++++-uesfm,The Pumpkin,1,27.26556900533122,5.820572144056583,35.6951097814524,16.572200360468877,196,5785.61,5760.521738529205,99999.0
+++++-uesfm,The Pumpkin,2,26.95722616022698,5.73349875876088,36.207441893083136,17.07678356860021,196,5793.15,5768.243654251099,99999.0
+++++ esfm,Thian Hook Keng Temple Singapore,0,1.0653505173922067,0.1079565219850111,15.054152885837562,5.81055046268767,138,3299.26,3281.187238931656,99999.0
+++++ esfm,Thian Hook Keng Temple Singapore,1,0.8390930608109779,0.08272919436528102,14.587041005106803,5.705702663819475,138,3306.93,3283.257929086685,99999.0
+++++ esfm,Thian Hook Keng Temple Singapore,2,0.8760513596629002,0.08764395630143504,15.102705362055175,5.7567965038255124,138,3289.42,3278.173619031906,99999.0
+++++-uesfm,Thian Hook Keng Temple Singapore,0,1.146131162829257,0.10799243740327723,20.24888615229727,10.297158058757331,138,4626.48,4595.79723072052,99999.0
+++++-uesfm,Thian Hook Keng Temple Singapore,1,1.2072956626750737,0.12821580405567637,21.242388560332195,10.655698470690366,138,4600.34,4578.699931621552,99999.0
+++++-uesfm,Thian Hook Keng Temple Singapore,2,1.251054659622132,0.12561972972485672,20.875018460120653,10.416875106878496,138,4610.3,4581.792747020721,99999.0
+++++ esfm,Tsar Nikolai I,0,42.507553768489565,8.869914103947062,10.846690028504929,4.007961025902478,98,4932.15,4920.938139438629,99999.0
+++++ esfm,Tsar Nikolai I,1,43.09835370739786,8.983133733509343,11.402130572143294,4.032588092580996,98,4924.71,4914.929541349411,99999.0
+++++-uesfm,Tsar Nikolai I,0,75.94880014543322,15.146573953653359,15.307363921859029,6.734065570805788,98,6652.35,6629.384099006653,99999.0
+++++-uesfm,Tsar Nikolai I,1,80.87807139992418,15.64102228288136,21.553650044939403,9.179513384124702,98,6641.34,6623.797205209732,99999.0
++++++esfm,Tsar Nikolai I,2,40.892180545819315,8.342265568650971,11.381967217434166,4.272306098513181,98,4930.74,4920.495186090469,99999.0
+++++ esfm,Urban II,0,59.0297968076411,11.008489348760156,17.600594623525765,5.400933781867582,96,8889.69,8878.01452589035,99999.0
++++++esfm,Urban II,1,58.947927470837534,11.025374954076108,17.465515658589908,5.68378594242137,96,8885.17,8876.618469953537,99999.0
++++++esfm,Urban II,2,58.77798771298748,10.982786678935327,17.546556385759054,5.767405629316944,96,8893.99,8885.34243106842,99999.0
+++++ esfm,Vercingetorix,0,79.07478057258302,9.776081681696928,6.649989325889541,1.7432748253588333,69,1223.75,1212.340968608856,99999.0
+++++ esfm,Vercingetorix,1,80.09313371434186,9.858626097407651,6.426263444651325,1.6979463697629655,69,1222.8,1213.763489961624,99999.0
+++++ esfm,Vercingetorix,2,89.68787080902364,10.950615257181747,8.695412468830675,3.360645595149667,69,1225.36,1215.808262825012,99999.0
+++++-uesfm,Vercingetorix,0,78.95942988866727,10.356244187384963,8.07065603849916,3.372297052368834,69,3105.42,3085.208213567734,99999.0
+++++-uesfm,Vercingetorix,1,89.99175121301754,11.178484677344343,8.929255781006072,4.302365177465691,69,3105.88,3091.564247369766,99999.0
+++++-uesfm,Vercingetorix,2,89.31151857112468,11.134780185118505,9.044030033429575,4.351952473195928,69,3105.52,3089.0203332901,99999.0
+++++ esfm,Yueh Hai Ching Temple Singapore,0,0.6268598783684821,0.08360284283022794,5.56223941139456,2.132543748253723,43,1749.28,1738.747783660889,99999.0
++++++esfm,Yueh Hai Ching Temple Singapore,1,0.5064956072004329,0.07235771132992698,5.30294026596974,2.06113770612239,43,1750.4,1739.704204082489,99999.0
++++++esfm,Yueh Hai Ching Temple Singapore,2,0.49998151063409024,0.07009863154178562,5.92393490371475,2.2317763685579015,43,1746.32,1738.905562877655,99999.0
+++++diff --git a/code/results/single_scene/summary_table.md b/code/results/single_scene/summary_table.md
+++++index bd3f673..1cac978 100644
+++++--- a/code/results/single_scene/summary_table.md
++++++++ b/code/results/single_scene/summary_table.md
+++++@@ -2,44 +2,45 @@
+++++ 
+++++ Mean over seeds per (scene, method); best value per scene in **bold** (lower is better except Nr).
+++++ 
+++++-| Scene | ESFM Rot (deg) | U-ESFM Rot (deg) | ESFM (paper) Rot (deg) | ESFM Trans | U-ESFM Trans | ESFM (paper) Trans | ESFM Reproj (px) | U-ESFM Reproj (px) | ESFM (paper) Reproj (px) | ESFM Nr | U-ESFM Nr | ESFM (paper) Nr | ESFM Time (s) | U-ESFM Time (s) | ESFM (paper) Time (s) | ESFM Rot-BA (deg) | U-ESFM Rot-BA (deg) | ESFM (paper) Rot-BA (deg) | ESFM Trans-BA | U-ESFM Trans-BA | ESFM (paper) Trans-BA | ESFM Reproj-BA (px) | U-ESFM Reproj-BA (px) | ESFM (paper) Reproj-BA (px) |
++++++| Scene | ESFM (official code) Rot (deg) | U-ESFM Rot (deg) | ESFM (paper) Rot (deg) | ESFM (official code) Trans | U-ESFM Trans | ESFM (paper) Trans | ESFM (official code) Reproj (px) | U-ESFM Reproj (px) | ESFM (paper) Reproj (px) | ESFM (official code) Nr | U-ESFM Nr | ESFM (paper) Nr | ESFM (official code) Time (s) | U-ESFM Time (s) | ESFM (paper) Time (s) | ESFM (official code) Rot-BA (deg) | U-ESFM Rot-BA (deg) | ESFM (paper) Rot-BA (deg) | ESFM (official code) Trans-BA | U-ESFM Trans-BA | ESFM (paper) Trans-BA | ESFM (official code) Reproj-BA (px) | U-ESFM Reproj-BA (px) | ESFM (paper) Reproj-BA (px) |
+++++ |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
+++++-| Alcatraz Courtyard | **0.362** | 0.616 | 0.619 | **0.093** | 0.161 | 0.160 | **3.435** | 5.796 | 1.640 | **133** | 133 | -- | **4826.713** | 6481.757 | -- | -- | -- | 0.049 | -- | -- | 0.015 | -- | -- | 0.810 |
+++++-| Alcatraz Water Tower | **0.666** | 7.413 | 0.933 | **0.370** | 3.437 | 0.518 | **4.763** | 8.607 | 2.130 | **172** | 172 | -- | **2461.337** | 3854.907 | -- | -- | -- | 0.230 | -- | -- | 0.116 | -- | -- | 0.550 |
+++++-| Buddah Tooth Relic Temple Singapore | **1.131** | 12.080 | 1.030 | **0.280** | 2.308 | 0.233 | **9.964** | 19.634 | 2.060 | **162** | 162 | -- | **2916.837** | 4320.873 | -- | -- | -- | 0.081 | -- | -- | 0.014 | -- | -- | 0.850 |
+++++-| Doge Palace Venice | **0.801** | 0.911 | 1.163 | **0.218** | 0.290 | 0.342 | **6.432** | 10.966 | 3.620 | **241** | 241 | -- | 28768.725 | **21074.950** | -- | -- | -- | 0.211 | -- | -- | 0.029 | -- | -- | 1 |
+++++-| Door Lund | **0.014** | 14.172 | 0.024 | **0.004** | 0.952 | 0.006 | **1.812** | 11.922 | 0.320 | **12** | 12 | -- | **4888.363** | 9188.593 | -- | -- | -- | 0.006 | -- | -- | 0.001 | -- | -- | 0.300 |
+++++-| Drinking Fountain Somewhere In Zurich | **17.193** | 17.536 | 0.031 | 0.753 | **0.740** | 0.004 | **6.058** | 6.511 | 0.330 | **14** | 14 | -- | **883.837** | 2963.647 | -- | -- | -- | 0.007 | -- | -- | 0.002 | -- | -- | 0.310 |
+++++-| East Indiaman Goteborg | 5.568 | **5.310** | 3.814 | 0.979 | **0.959** | 0.621 | **7.318** | 9.840 | 4.130 | **179** | 179 | -- | **3987.183** | 5241.430 | -- | -- | -- | 3.117 | -- | -- | 0.509 | -- | -- | 1.850 |
+++++-| Ecole Superior De Guerre | 0.349 | **0.284** | 0.318 | 0.090 | **0.077** | 0.081 | **3.418** | 5.349 | 0.720 | **35** | 35 | -- | **2380.387** | 4633.833 | -- | -- | -- | 0.024 | -- | -- | 0.005 | -- | -- | 0.340 |
+++++-| Eglise du dome | **0.351** | 45.801 | 0.808 | **0.088** | 6.451 | 0.205 | **3.799** | 7.964 | 0.910 | **85** | 85 | -- | 25099.450 | **23371.730** | -- | -- | -- | 0.037 | -- | -- | 0.010 | -- | -- | 0.270 |
+++++-| Folke Filbyter | 85.831 | **81.288** | 74.596 | **0.129** | 0.130 | 0.125 | **19.275** | 23.156 | 10.370 | **40** | 40 | -- | **1581.897** | 3703.187 | -- | -- | -- | 70.157 | -- | -- | 0.118 | -- | -- | 4.290 |
+++++-| Fort Channing Gate Singapore | **0.093** | 11.128 | 0.207 | **0.041** | 4.004 | 0.093 | **1.783** | 6.651 | 0.520 | **27** | 27 | -- | **3434.113** | 6172.087 | -- | -- | -- | 0.020 | -- | -- | 0.008 | -- | -- | 0.250 |
+++++-| Golden Statue Somewhere In Hong Kong | **0.232** | 0.371 | 0.292 | **0.053** | 0.074 | 0.073 | **1.829** | 2.835 | 0.400 | **18** | 18 | -- | **4095.853** | 7762.437 | -- | -- | -- | 0.031 | -- | -- | 0.004 | -- | -- | 0.270 |
+++++-| Gustav Vasa | **3.509** | 6.099 | 34.181 | **0.220** | 0.345 | 1.085 | **1.655** | 3.087 | 3.520 | **18** | 18 | -- | **709.867** | 2543.553 | -- | -- | -- | 32.266 | -- | -- | 1.145 | -- | -- | 3.150 |
+++++-| GustavIIAdolf | **47.126** | 77.423 | 67.784 | **8.536** | 11.962 | 9.714 | **11.722** | 15.851 | 13.910 | **57** | 57 | -- | **1044.170** | 2892.580 | -- | -- | -- | 58.458 | -- | -- | 8.524 | -- | -- | 11.490 |
+++++-| Jonas Ahlstromer | 48.740 | **42.246** | 50.190 | **9.823** | 10.300 | 10.888 | 11.136 | **8.782** | 10.820 | **40** | 40 | -- | **615.450** | 2328.313 | -- | -- | -- | 47.117 | -- | -- | 10.451 | -- | -- | 8.410 |
+++++-| Kings College University Of Toronto | **9.025** | 21.523 | 0.989 | 2.044 | **1.905** | 0.235 | **3.565** | 5.525 | 0.900 | **77** | 77 | -- | **1078.980** | 2886.243 | -- | -- | -- | 0.085 | -- | -- | 0.017 | -- | -- | 0.340 |
+++++-| Lund University Sphinx | **11.392** | 33.023 | 19.522 | **2.981** | 6.889 | 4.585 | **6.056** | 12.060 | 4.780 | **70** | 70 | -- | **2718.567** | 4654.610 | -- | -- | -- | 8.752 | -- | -- | 2.191 | -- | -- | 1.360 |
+++++-| Nijo Castle Gate | **0.839** | 1.682 | 1.495 | **0.165** | 0.326 | 0.286 | **7.415** | 12.138 | 1.700 | **19** | 19 | -- | **1095.177** | 3306.697 | -- | -- | -- | 0.069 | -- | -- | 0.012 | -- | -- | 0.730 |
+++++-| Pantheon Paris | -- | 19.345 | 0.192 | -- | 3.065 | 0.050 | -- | 13.562 | 1.470 | -- | 179 | -- | -- | 5497.377 | -- | -- | -- | 0.040 | -- | -- | 0.005 | -- | -- | 0.490 |
+++++-| Park Gate Clermont Ferrand | 21.388 | **19.302** | 0.391 | 10.192 | **9.397** | 0.125 | **7.712** | 7.852 | 0.570 | **34** | 34 | -- | **1596.940** | 3597.403 | -- | -- | -- | 0.049 | -- | -- | 0.022 | -- | -- | 0.350 |
+++++-| Plaza De Armas Santiago | **0.775** | 3.327 | 6.782 | **0.337** | 1.435 | 2.944 | **6.009** | 9.463 | 7.400 | **240** | 240 | -- | **6372.683** | 6535.537 | -- | -- | -- | 2.556 | -- | -- | 1.383 | -- | -- | 4.900 |
+++++-| Porta San Donato Bologna | **0.592** | 21.006 | 2.153 | **0.107** | 3.541 | 0.388 | **5.603** | 13.244 | 2.280 | **141** | 141 | -- | **3754.163** | 5118.030 | -- | -- | -- | 0.095 | -- | -- | 0.046 | -- | -- | 0.750 |
+++++-| Round Church Cambridge | **2.129** | 2.490 | 2.451 | **0.920** | 1.002 | 1.003 | **6.097** | 8.546 | 2.660 | **92** | 92 | -- | **9776.630** | 12061.350 | -- | -- | -- | 1.107 | -- | -- | 0.582 | -- | -- | 1.540 |
+++++-| Skansen Kronan Gothenburg | **0.338** | 0.375 | 0.736 | **0.116** | 0.124 | 0.226 | **3.048** | 4.799 | 1.240 | **131** | 131 | -- | 17163.940 | **16235.090** | -- | -- | -- | 0.026 | -- | -- | 0.008 | -- | -- | 0.670 |
+++++-| Smolny Cathedral St Petersburg | **21.806** | 22.143 | 0.554 | 2.111 | **2.099** | 0.051 | **11.228** | 12.526 | 1.660 | **131** | 131 | -- | **10362.595** | 10475.740 | -- | -- | -- | 0.033 | -- | -- | 0.006 | -- | -- | 0.810 |
+++++-| Sri Mariamman Singapore | **1.219** | 6.479 | 2.302 | **0.382** | 2.188 | 0.683 | **10.088** | 17.854 | 4.130 | **222** | 222 | -- | **5985.540** | 6485.667 | -- | -- | -- | 0.077 | -- | -- | 0.023 | -- | -- | 0.910 |
+++++-| Sri Thendayuthapani Singapore | **0.719** | 47.075 | 46.269 | **0.179** | 3.855 | 3.812 | **7.124** | 14.133 | 23.370 | **98** | 98 | -- | **13362.090** | 13808.440 | -- | -- | -- | 44.170 | -- | -- | 2.870 | -- | -- | 8.440 |
+++++-| Sri Veeramakaliamman Singapore | **2.334** | 29.405 | 2.559 | **0.543** | 3.868 | 0.597 | **11.965** | 28.951 | 3.470 | **157** | 157 | -- | **12392.180** | 12544.690 | -- | -- | -- | 0.175 | -- | -- | 0.040 | -- | -- | 0.730 |
+++++-| Statue Of Liberty | 52.358 | **39.859** | 46.887 | 28.503 | **18.503** | 20.012 | 235036.825 | **41.028** | 26.160 | **134** | 134 | -- | **99.900** | 12924 | -- | -- | -- | 9.091 | -- | -- | 4.122 | -- | -- | 6.970 |
+++++-| The Pumpkin | **25.731** | 26.571 | 94.672 | **5.552** | 5.665 | 14.890 | **26.876** | 35.367 | 33.410 | **196** | 196 | -- | **4327.273** | 5790.597 | -- | -- | -- | 98.862 | -- | -- | 14.952 | -- | -- | 24.850 |
+++++-| Thian Hook Keng Temple Singapore | **0.927** | 1.201 | 0.832 | **0.093** | 0.121 | 0.082 | **14.915** | 20.789 | 2.750 | **138** | 138 | -- | **3298.537** | 4612.373 | -- | -- | -- | 0.081 | -- | -- | 0.008 | -- | -- | 1.130 |
+++++-| Tsar Nikolai I | **42.803** | 78.413 | 48.499 | **8.927** | 15.394 | 9.467 | **11.124** | 18.431 | 9.790 | **98** | 98 | -- | **4928.430** | 6646.845 | -- | -- | -- | 36.280 | -- | -- | 7.836 | -- | -- | 6.530 |
+++++-| Urban II | 59.030 | -- | 47.490 | 11.008 | -- | 9.467 | 17.601 | -- | 9.380 | 96 | -- | -- | 8889.690 | -- | -- | -- | -- | 48.214 | -- | -- | 9.586 | -- | -- | 6.920 |
+++++-| Vercingetorix | **82.952** | 86.088 | 69.328 | **10.195** | 10.890 | 8.788 | **7.257** | 8.681 | 5.080 | **69** | 69 | -- | **1223.970** | 3105.607 | -- | -- | -- | 17.706 | -- | -- | 3.104 | -- | -- | 1.500 |
+++++-| Yueh Hai Ching Temple Singapore | 0.627 | -- | 0.720 | 0.084 | -- | 0.098 | 5.562 | -- | 0.940 | 43 | -- | -- | 1749.280 | -- | -- | -- | -- | 0.043 | -- | -- | 0.014 | -- | -- | 0.650 |
+++++-| **Mean** | **16.146** | 23.697 | 18.023 | **3.121** | 4.014 | 2.912 | 6920.602 | **13.088** | 5.673 | 100.559 | **104.818** | -- | **5819.728** | 7358.187 | -- | -- | -- | 13.695 | -- | -- | 1.937 | -- | -- | 2.992 |
++++++| Alcatraz Courtyard | **0.362** | 0.617 | 0.619 | **0.093** | 0.157 | 0.160 | **3.435** | 5.482 | 1.640 | **133** | 133 | -- | **4826.713** | 6554.345 | -- | -- | -- | 0.049 | -- | -- | 0.015 | -- | -- | 0.810 |
++++++| Alcatraz Water Tower | **0.666** | 1.480 | 0.933 | **0.370** | 0.803 | 0.518 | **4.763** | 7.397 | 2.130 | **172** | 172 | -- | **2461.337** | 3795.877 | -- | -- | -- | 0.230 | -- | -- | 0.116 | -- | -- | 0.550 |
++++++| Buddah Tooth Relic Temple Singapore | **1.131** | 1.791 | 1.030 | **0.280** | 0.520 | 0.233 | **9.964** | 14.029 | 2.060 | **162** | 162 | -- | **2916.837** | 4289.503 | -- | -- | -- | 0.081 | -- | -- | 0.014 | -- | -- | 0.850 |
++++++| Doge Palace Venice | 0.753 | -- | 1.163 | 0.209 | -- | 0.342 | 6.359 | -- | 3.620 | 241 | -- | -- | 22522.737 | -- | -- | -- | -- | 0.211 | -- | -- | 0.029 | -- | -- | 1 |
++++++| Door Lund | **0.014** | 6.960 | 0.024 | **0.004** | 0.512 | 0.006 | **1.812** | 7.101 | 0.320 | **12** | 12 | -- | **4888.363** | 9252.735 | -- | -- | -- | 0.006 | -- | -- | 0.001 | -- | -- | 0.300 |
++++++| Drinking Fountain Somewhere In Zurich | **17.193** | 17.329 | 0.031 | 0.753 | **0.750** | 0.004 | **6.058** | 6.675 | 0.330 | **14** | 14 | -- | **883.837** | 5815.220 | -- | -- | -- | 0.007 | -- | -- | 0.002 | -- | -- | 0.310 |
++++++| East Indiaman Goteborg | **5.568** | 7.231 | 3.814 | **0.979** | 1.364 | 0.621 | **7.318** | 10.130 | 4.130 | **179** | 179 | -- | **3987.183** | 5198.093 | -- | -- | -- | 3.117 | -- | -- | 0.509 | -- | -- | 1.850 |
++++++| Ecole Superior De Guerre | **0.349** | 51.327 | 0.318 | **0.090** | 2.574 | 0.081 | **3.418** | 11.751 | 0.720 | **35** | 35 | -- | **2380.387** | 10426.770 | -- | -- | -- | 0.024 | -- | -- | 0.005 | -- | -- | 0.340 |
++++++| Eglise du dome | **0.386** | 0.853 | 0.808 | **0.098** | 0.232 | 0.205 | **3.987** | 5.944 | 0.910 | **85** | 85 | -- | 19631.293 | **9803.940** | -- | -- | -- | 0.037 | -- | -- | 0.010 | -- | -- | 0.270 |
++++++| Folke Filbyter | 85.831 | **82.656** | 74.596 | **0.129** | 0.131 | 0.125 | **19.275** | 28.225 | 10.370 | **40** | 40 | -- | **1581.897** | 3755.193 | -- | -- | -- | 70.157 | -- | -- | 0.118 | -- | -- | 4.290 |
++++++| Fort Channing Gate Singapore | **0.093** | 0.122 | 0.207 | **0.041** | 0.056 | 0.093 | **1.783** | 3.649 | 0.520 | **27** | 27 | -- | **3434.113** | 6183.925 | -- | -- | -- | 0.020 | -- | -- | 0.008 | -- | -- | 0.250 |
++++++| Golden Statue Somewhere In Hong Kong | 0.232 | -- | 0.292 | 0.053 | -- | 0.073 | 1.829 | -- | 0.400 | 18 | -- | -- | 4095.853 | -- | -- | -- | -- | 0.031 | -- | -- | 0.004 | -- | -- | 0.270 |
++++++| Gustav Vasa | **3.509** | 25.854 | 34.181 | **0.220** | 0.775 | 1.085 | **1.655** | 4.747 | 3.520 | **18** | 18 | -- | **709.867** | 3887.653 | -- | -- | -- | 32.266 | -- | -- | 1.145 | -- | -- | 3.150 |
++++++| GustavIIAdolf | **47.126** | 88.194 | 67.784 | **8.536** | 13.129 | 9.714 | **11.722** | 17.220 | 13.910 | **57** | 57 | -- | **1044.170** | 2981.017 | -- | -- | -- | 58.458 | -- | -- | 8.524 | -- | -- | 11.490 |
++++++| Jonas Ahlstromer | 48.740 | **46.584** | 50.190 | **9.823** | 10.245 | 10.888 | **11.136** | 11.144 | 10.820 | **40** | 40 | -- | **615.450** | 2414.653 | -- | -- | -- | 47.117 | -- | -- | 10.451 | -- | -- | 8.410 |
++++++| Kings College University Of Toronto | **9.025** | 18.154 | 0.989 | 2.044 | **1.706** | 0.235 | **3.565** | 6.501 | 0.900 | **77** | 77 | -- | **1078.980** | 2854.715 | -- | -- | -- | 0.085 | -- | -- | 0.017 | -- | -- | 0.340 |
++++++| Lund University Sphinx | **11.392** | 42.959 | 19.522 | **2.981** | 8.720 | 4.585 | **6.056** | 15.000 | 4.780 | **70** | 70 | -- | **2718.567** | 4705.610 | -- | -- | -- | 8.752 | -- | -- | 2.191 | -- | -- | 1.360 |
++++++| Nijo Castle Gate | **0.839** | 8.641 | 1.495 | **0.165** | 2.791 | 0.286 | **7.415** | 20.967 | 1.700 | **19** | 19 | -- | **1095.177** | 3241.330 | -- | -- | -- | 0.069 | -- | -- | 0.012 | -- | -- | 0.730 |
++++++| Pantheon Paris | -- | 0.509 | 0.192 | -- | 0.066 | 0.050 | -- | 7.493 | 1.470 | -- | 179 | -- | -- | 5565.670 | -- | -- | -- | 0.040 | -- | -- | 0.005 | -- | -- | 0.490 |
++++++| Park Gate Clermont Ferrand | 21.388 | -- | 0.391 | 10.192 | -- | 0.125 | 7.712 | -- | 0.570 | 34 | -- | -- | 1596.940 | -- | -- | -- | -- | 0.049 | -- | -- | 0.022 | -- | -- | 0.350 |
++++++| Plaza De Armas Santiago | 0.775 | -- | 6.782 | 0.337 | -- | 2.944 | 6.009 | -- | 7.400 | 240 | -- | -- | 6372.683 | -- | -- | -- | -- | 2.556 | -- | -- | 1.383 | -- | -- | 4.900 |
++++++| Porta San Donato Bologna | 0.592 | -- | 2.153 | 0.107 | -- | 0.388 | 5.603 | -- | 2.280 | 141 | -- | -- | 3754.163 | -- | -- | -- | -- | 0.095 | -- | -- | 0.046 | -- | -- | 0.750 |
++++++| Round Church Cambridge | 2.137 | -- | 2.451 | 0.927 | -- | 1.003 | 6.103 | -- | 2.660 | 92 | -- | -- | 9779.270 | -- | -- | -- | -- | 1.107 | -- | -- | 0.582 | -- | -- | 1.540 |
++++++| Skansen Kronan Gothenburg | 0.301 | -- | 0.736 | 0.102 | -- | 0.226 | 2.906 | -- | 1.240 | 131 | -- | -- | 17153.677 | -- | -- | -- | -- | 0.026 | -- | -- | 0.008 | -- | -- | 0.670 |
++++++| Smolny Cathedral St Petersburg | 21.775 | -- | 0.554 | 2.111 | -- | 0.051 | 11.239 | -- | 1.660 | 131 | -- | -- | 10362.787 | -- | -- | -- | -- | 0.033 | -- | -- | 0.006 | -- | -- | 0.810 |
++++++| Some Cathedral In Barcelona | 0.651 | -- | 0.880 | 0.233 | -- | 0.315 | 6.271 | -- | 2.870 | 177 | -- | -- | 5001.973 | -- | -- | -- | -- | 0.026 | -- | -- | 0.011 | -- | -- | 0.890 |
++++++| Sri Mariamman Singapore | 1.219 | -- | 2.302 | 0.382 | -- | 0.683 | 10.088 | -- | 4.130 | 222 | -- | -- | 5985.540 | -- | -- | -- | -- | 0.077 | -- | -- | 0.023 | -- | -- | 0.910 |
++++++| Sri Thendayuthapani Singapore | 0.593 | -- | 46.269 | 0.150 | -- | 3.812 | 7.016 | -- | 23.370 | 98 | -- | -- | 13338.167 | -- | -- | -- | -- | 44.170 | -- | -- | 2.870 | -- | -- | 8.440 |
++++++| Sri Veeramakaliamman Singapore | 2.404 | -- | 2.559 | 0.559 | -- | 0.597 | 12.234 | -- | 3.470 | 157 | -- | -- | 12377.927 | -- | -- | -- | -- | 0.175 | -- | -- | 0.040 | -- | -- | 0.730 |
++++++| Statue Of Liberty | 52.358 | -- | 46.887 | 28.503 | -- | 20.012 | 235036.825 | -- | 26.160 | 134 | -- | -- | 99.900 | -- | -- | -- | -- | 9.091 | -- | -- | 4.122 | -- | -- | 6.970 |
++++++| The Pumpkin | 25.731 | -- | 94.672 | 5.552 | -- | 14.890 | 26.876 | -- | 33.410 | 196 | -- | -- | 4327.273 | -- | -- | -- | -- | 98.862 | -- | -- | 14.952 | -- | -- | 24.850 |
++++++| Thian Hook Keng Temple Singapore | 0.927 | -- | 0.832 | 0.093 | -- | 0.082 | 14.915 | -- | 2.750 | 138 | -- | -- | 3298.537 | -- | -- | -- | -- | 0.081 | -- | -- | 0.008 | -- | -- | 1.130 |
++++++| Tsar Nikolai I | 42.166 | -- | 48.499 | 8.732 | -- | 9.467 | 11.210 | -- | 9.790 | 98 | -- | -- | 4929.200 | -- | -- | -- | -- | 36.280 | -- | -- | 7.836 | -- | -- | 6.530 |
++++++| Urban II | 58.919 | -- | 47.490 | 11.006 | -- | 9.467 | 17.538 | -- | 9.380 | 96 | -- | -- | 8889.617 | -- | -- | -- | -- | 48.214 | -- | -- | 9.586 | -- | -- | 6.920 |
++++++| Vercingetorix | 82.952 | -- | 69.328 | 10.195 | -- | 8.788 | 7.257 | -- | 5.080 | 69 | -- | -- | 1223.970 | -- | -- | -- | -- | 17.706 | -- | -- | 3.104 | -- | -- | 1.500 |
++++++| Yueh Hai Ching Temple Singapore | 0.544 | -- | 0.720 | 0.075 | -- | 0.098 | 5.596 | -- | 0.940 | 43 | -- | -- | 1748.667 | -- | -- | -- | -- | 0.043 | -- | -- | 0.014 | -- | -- | 0.650 |
++++++| **Mean** | **15.676** | 23.604 | 17.547 | 3.032 | **2.619** | 2.840 | 6723.056 | **10.792** | 5.595 | **102.743** | 77.588 | -- | 5460.373 | **5336.838** | -- | -- | -- | 13.315 | -- | -- | 1.883 | -- | -- | 2.933 |
+++++ 
+++++ _Seeds per cell: [0, 1, 2]._
+++++ 
+++++diff --git a/code/results/single_scene/summary_table.tex b/code/results/single_scene/summary_table.tex
+++++index 258e3f2..95defe0 100644
+++++--- a/code/results/single_scene/summary_table.tex
++++++++ b/code/results/single_scene/summary_table.tex
+++++@@ -7,45 +7,46 @@
+++++ \begin{tabular}{lcccccccccccccccccccccccc}
+++++ \toprule
+++++  & \multicolumn{3}{c}{Rot (deg)} & \multicolumn{3}{c}{Trans} & \multicolumn{3}{c}{Reproj (px)} & \multicolumn{3}{c}{Nr} & \multicolumn{3}{c}{Time (s)} & \multicolumn{3}{c}{Rot-BA (deg)} & \multicolumn{3}{c}{Trans-BA} & \multicolumn{3}{c}{Reproj-BA (px)} \\
+++++-Scene & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) \\
++++++Scene & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) \\
+++++ \midrule
+++++-Alcatraz Courtyard & \textbf{0.362} & 0.616 & 0.619 & \textbf{0.093} & 0.161 & 0.160 & \textbf{3.435} & 5.796 & 1.640 & \textbf{133} & 133 & -- & \textbf{4826.713} & 6481.757 & -- & -- & -- & 0.049 & -- & -- & 0.015 & -- & -- & 0.810 \\
+++++-Alcatraz Water Tower & \textbf{0.666} & 7.413 & 0.933 & \textbf{0.370} & 3.437 & 0.518 & \textbf{4.763} & 8.607 & 2.130 & \textbf{172} & 172 & -- & \textbf{2461.337} & 3854.907 & -- & -- & -- & 0.230 & -- & -- & 0.116 & -- & -- & 0.550 \\
+++++-Buddah Tooth Relic Temple Singapore & \textbf{1.131} & 12.080 & 1.030 & \textbf{0.280} & 2.308 & 0.233 & \textbf{9.964} & 19.634 & 2.060 & \textbf{162} & 162 & -- & \textbf{2916.837} & 4320.873 & -- & -- & -- & 0.081 & -- & -- & 0.014 & -- & -- & 0.850 \\
+++++-Doge Palace Venice & \textbf{0.801} & 0.911 & 1.163 & \textbf{0.218} & 0.290 & 0.342 & \textbf{6.432} & 10.966 & 3.620 & \textbf{241} & 241 & -- & 28768.725 & \textbf{21074.950} & -- & -- & -- & 0.211 & -- & -- & 0.029 & -- & -- & 1 \\
+++++-Door Lund & \textbf{0.014} & 14.172 & 0.024 & \textbf{0.004} & 0.952 & 0.006 & \textbf{1.812} & 11.922 & 0.320 & \textbf{12} & 12 & -- & \textbf{4888.363} & 9188.593 & -- & -- & -- & 0.006 & -- & -- & 0.001 & -- & -- & 0.300 \\
+++++-Drinking Fountain Somewhere In Zurich & \textbf{17.193} & 17.536 & 0.031 & 0.753 & \textbf{0.740} & 0.004 & \textbf{6.058} & 6.511 & 0.330 & \textbf{14} & 14 & -- & \textbf{883.837} & 2963.647 & -- & -- & -- & 0.007 & -- & -- & 0.002 & -- & -- & 0.310 \\
+++++-East Indiaman Goteborg & 5.568 & \textbf{5.310} & 3.814 & 0.979 & \textbf{0.959} & 0.621 & \textbf{7.318} & 9.840 & 4.130 & \textbf{179} & 179 & -- & \textbf{3987.183} & 5241.430 & -- & -- & -- & 3.117 & -- & -- & 0.509 & -- & -- & 1.850 \\
+++++-Ecole Superior De Guerre & 0.349 & \textbf{0.284} & 0.318 & 0.090 & \textbf{0.077} & 0.081 & \textbf{3.418} & 5.349 & 0.720 & \textbf{35} & 35 & -- & \textbf{2380.387} & 4633.833 & -- & -- & -- & 0.024 & -- & -- & 0.005 & -- & -- & 0.340 \\
+++++-Eglise du dome & \textbf{0.351} & 45.801 & 0.808 & \textbf{0.088} & 6.451 & 0.205 & \textbf{3.799} & 7.964 & 0.910 & \textbf{85} & 85 & -- & 25099.450 & \textbf{23371.730} & -- & -- & -- & 0.037 & -- & -- & 0.010 & -- & -- & 0.270 \\
+++++-Folke Filbyter & 85.831 & \textbf{81.288} & 74.596 & \textbf{0.129} & 0.130 & 0.125 & \textbf{19.275} & 23.156 & 10.370 & \textbf{40} & 40 & -- & \textbf{1581.897} & 3703.187 & -- & -- & -- & 70.157 & -- & -- & 0.118 & -- & -- & 4.290 \\
+++++-Fort Channing Gate Singapore & \textbf{0.093} & 11.128 & 0.207 & \textbf{0.041} & 4.004 & 0.093 & \textbf{1.783} & 6.651 & 0.520 & \textbf{27} & 27 & -- & \textbf{3434.113} & 6172.087 & -- & -- & -- & 0.020 & -- & -- & 0.008 & -- & -- & 0.250 \\
+++++-Golden Statue Somewhere In Hong Kong & \textbf{0.232} & 0.371 & 0.292 & \textbf{0.053} & 0.074 & 0.073 & \textbf{1.829} & 2.835 & 0.400 & \textbf{18} & 18 & -- & \textbf{4095.853} & 7762.437 & -- & -- & -- & 0.031 & -- & -- & 0.004 & -- & -- & 0.270 \\
+++++-Gustav Vasa & \textbf{3.509} & 6.099 & 34.181 & \textbf{0.220} & 0.345 & 1.085 & \textbf{1.655} & 3.087 & 3.520 & \textbf{18} & 18 & -- & \textbf{709.867} & 2543.553 & -- & -- & -- & 32.266 & -- & -- & 1.145 & -- & -- & 3.150 \\
+++++-GustavIIAdolf & \textbf{47.126} & 77.423 & 67.784 & \textbf{8.536} & 11.962 & 9.714 & \textbf{11.722} & 15.851 & 13.910 & \textbf{57} & 57 & -- & \textbf{1044.170} & 2892.580 & -- & -- & -- & 58.458 & -- & -- & 8.524 & -- & -- & 11.490 \\
+++++-Jonas Ahlstromer & 48.740 & \textbf{42.246} & 50.190 & \textbf{9.823} & 10.300 & 10.888 & 11.136 & \textbf{8.782} & 10.820 & \textbf{40} & 40 & -- & \textbf{615.450} & 2328.313 & -- & -- & -- & 47.117 & -- & -- & 10.451 & -- & -- & 8.410 \\
+++++-Kings College University Of Toronto & \textbf{9.025} & 21.523 & 0.989 & 2.044 & \textbf{1.905} & 0.235 & \textbf{3.565} & 5.525 & 0.900 & \textbf{77} & 77 & -- & \textbf{1078.980} & 2886.243 & -- & -- & -- & 0.085 & -- & -- & 0.017 & -- & -- & 0.340 \\
+++++-Lund University Sphinx & \textbf{11.392} & 33.023 & 19.522 & \textbf{2.981} & 6.889 & 4.585 & \textbf{6.056} & 12.060 & 4.780 & \textbf{70} & 70 & -- & \textbf{2718.567} & 4654.610 & -- & -- & -- & 8.752 & -- & -- & 2.191 & -- & -- & 1.360 \\
+++++-Nijo Castle Gate & \textbf{0.839} & 1.682 & 1.495 & \textbf{0.165} & 0.326 & 0.286 & \textbf{7.415} & 12.138 & 1.700 & \textbf{19} & 19 & -- & \textbf{1095.177} & 3306.697 & -- & -- & -- & 0.069 & -- & -- & 0.012 & -- & -- & 0.730 \\
+++++-Pantheon Paris & -- & 19.345 & 0.192 & -- & 3.065 & 0.050 & -- & 13.562 & 1.470 & -- & 179 & -- & -- & 5497.377 & -- & -- & -- & 0.040 & -- & -- & 0.005 & -- & -- & 0.490 \\
+++++-Park Gate Clermont Ferrand & 21.388 & \textbf{19.302} & 0.391 & 10.192 & \textbf{9.397} & 0.125 & \textbf{7.712} & 7.852 & 0.570 & \textbf{34} & 34 & -- & \textbf{1596.940} & 3597.403 & -- & -- & -- & 0.049 & -- & -- & 0.022 & -- & -- & 0.350 \\
+++++-Plaza De Armas Santiago & \textbf{0.775} & 3.327 & 6.782 & \textbf{0.337} & 1.435 & 2.944 & \textbf{6.009} & 9.463 & 7.400 & \textbf{240} & 240 & -- & \textbf{6372.683} & 6535.537 & -- & -- & -- & 2.556 & -- & -- & 1.383 & -- & -- & 4.900 \\
+++++-Porta San Donato Bologna & \textbf{0.592} & 21.006 & 2.153 & \textbf{0.107} & 3.541 & 0.388 & \textbf{5.603} & 13.244 & 2.280 & \textbf{141} & 141 & -- & \textbf{3754.163} & 5118.030 & -- & -- & -- & 0.095 & -- & -- & 0.046 & -- & -- & 0.750 \\
+++++-Round Church Cambridge & \textbf{2.129} & 2.490 & 2.451 & \textbf{0.920} & 1.002 & 1.003 & \textbf{6.097} & 8.546 & 2.660 & \textbf{92} & 92 & -- & \textbf{9776.630} & 12061.350 & -- & -- & -- & 1.107 & -- & -- & 0.582 & -- & -- & 1.540 \\
+++++-Skansen Kronan Gothenburg & \textbf{0.338} & 0.375 & 0.736 & \textbf{0.116} & 0.124 & 0.226 & \textbf{3.048} & 4.799 & 1.240 & \textbf{131} & 131 & -- & 17163.940 & \textbf{16235.090} & -- & -- & -- & 0.026 & -- & -- & 0.008 & -- & -- & 0.670 \\
+++++-Smolny Cathedral St Petersburg & \textbf{21.806} & 22.143 & 0.554 & 2.111 & \textbf{2.099} & 0.051 & \textbf{11.228} & 12.526 & 1.660 & \textbf{131} & 131 & -- & \textbf{10362.595} & 10475.740 & -- & -- & -- & 0.033 & -- & -- & 0.006 & -- & -- & 0.810 \\
+++++-Sri Mariamman Singapore & \textbf{1.219} & 6.479 & 2.302 & \textbf{0.382} & 2.188 & 0.683 & \textbf{10.088} & 17.854 & 4.130 & \textbf{222} & 222 & -- & \textbf{5985.540} & 6485.667 & -- & -- & -- & 0.077 & -- & -- & 0.023 & -- & -- & 0.910 \\
+++++-Sri Thendayuthapani Singapore & \textbf{0.719} & 47.075 & 46.269 & \textbf{0.179} & 3.855 & 3.812 & \textbf{7.124} & 14.133 & 23.370 & \textbf{98} & 98 & -- & \textbf{13362.090} & 13808.440 & -- & -- & -- & 44.170 & -- & -- & 2.870 & -- & -- & 8.440 \\
+++++-Sri Veeramakaliamman Singapore & \textbf{2.334} & 29.405 & 2.559 & \textbf{0.543} & 3.868 & 0.597 & \textbf{11.965} & 28.951 & 3.470 & \textbf{157} & 157 & -- & \textbf{12392.180} & 12544.690 & -- & -- & -- & 0.175 & -- & -- & 0.040 & -- & -- & 0.730 \\
+++++-Statue Of Liberty & 52.358 & \textbf{39.859} & 46.887 & 28.503 & \textbf{18.503} & 20.012 & 235036.825 & \textbf{41.028} & 26.160 & \textbf{134} & 134 & -- & \textbf{99.900} & 12924 & -- & -- & -- & 9.091 & -- & -- & 4.122 & -- & -- & 6.970 \\
+++++-The Pumpkin & \textbf{25.731} & 26.571 & 94.672 & \textbf{5.552} & 5.665 & 14.890 & \textbf{26.876} & 35.367 & 33.410 & \textbf{196} & 196 & -- & \textbf{4327.273} & 5790.597 & -- & -- & -- & 98.862 & -- & -- & 14.952 & -- & -- & 24.850 \\
+++++-Thian Hook Keng Temple Singapore & \textbf{0.927} & 1.201 & 0.832 & \textbf{0.093} & 0.121 & 0.082 & \textbf{14.915} & 20.789 & 2.750 & \textbf{138} & 138 & -- & \textbf{3298.537} & 4612.373 & -- & -- & -- & 0.081 & -- & -- & 0.008 & -- & -- & 1.130 \\
+++++-Tsar Nikolai I & \textbf{42.803} & 78.413 & 48.499 & \textbf{8.927} & 15.394 & 9.467 & \textbf{11.124} & 18.431 & 9.790 & \textbf{98} & 98 & -- & \textbf{4928.430} & 6646.845 & -- & -- & -- & 36.280 & -- & -- & 7.836 & -- & -- & 6.530 \\
+++++-Urban II & 59.030 & -- & 47.490 & 11.008 & -- & 9.467 & 17.601 & -- & 9.380 & 96 & -- & -- & 8889.690 & -- & -- & -- & -- & 48.214 & -- & -- & 9.586 & -- & -- & 6.920 \\
+++++-Vercingetorix & \textbf{82.952} & 86.088 & 69.328 & \textbf{10.195} & 10.890 & 8.788 & \textbf{7.257} & 8.681 & 5.080 & \textbf{69} & 69 & -- & \textbf{1223.970} & 3105.607 & -- & -- & -- & 17.706 & -- & -- & 3.104 & -- & -- & 1.500 \\
+++++-Yueh Hai Ching Temple Singapore & 0.627 & -- & 0.720 & 0.084 & -- & 0.098 & 5.562 & -- & 0.940 & 43 & -- & -- & 1749.280 & -- & -- & -- & -- & 0.043 & -- & -- & 0.014 & -- & -- & 0.650 \\
++++++Alcatraz Courtyard & \textbf{0.362} & 0.617 & 0.619 & \textbf{0.093} & 0.157 & 0.160 & \textbf{3.435} & 5.482 & 1.640 & \textbf{133} & 133 & -- & \textbf{4826.713} & 6554.345 & -- & -- & -- & 0.049 & -- & -- & 0.015 & -- & -- & 0.810 \\
++++++Alcatraz Water Tower & \textbf{0.666} & 1.480 & 0.933 & \textbf{0.370} & 0.803 & 0.518 & \textbf{4.763} & 7.397 & 2.130 & \textbf{172} & 172 & -- & \textbf{2461.337} & 3795.877 & -- & -- & -- & 0.230 & -- & -- & 0.116 & -- & -- & 0.550 \\
++++++Buddah Tooth Relic Temple Singapore & \textbf{1.131} & 1.791 & 1.030 & \textbf{0.280} & 0.520 & 0.233 & \textbf{9.964} & 14.029 & 2.060 & \textbf{162} & 162 & -- & \textbf{2916.837} & 4289.503 & -- & -- & -- & 0.081 & -- & -- & 0.014 & -- & -- & 0.850 \\
++++++Doge Palace Venice & 0.753 & -- & 1.163 & 0.209 & -- & 0.342 & 6.359 & -- & 3.620 & 241 & -- & -- & 22522.737 & -- & -- & -- & -- & 0.211 & -- & -- & 0.029 & -- & -- & 1 \\
++++++Door Lund & \textbf{0.014} & 6.960 & 0.024 & \textbf{0.004} & 0.512 & 0.006 & \textbf{1.812} & 7.101 & 0.320 & \textbf{12} & 12 & -- & \textbf{4888.363} & 9252.735 & -- & -- & -- & 0.006 & -- & -- & 0.001 & -- & -- & 0.300 \\
++++++Drinking Fountain Somewhere In Zurich & \textbf{17.193} & 17.329 & 0.031 & 0.753 & \textbf{0.750} & 0.004 & \textbf{6.058} & 6.675 & 0.330 & \textbf{14} & 14 & -- & \textbf{883.837} & 5815.220 & -- & -- & -- & 0.007 & -- & -- & 0.002 & -- & -- & 0.310 \\
++++++East Indiaman Goteborg & \textbf{5.568} & 7.231 & 3.814 & \textbf{0.979} & 1.364 & 0.621 & \textbf{7.318} & 10.130 & 4.130 & \textbf{179} & 179 & -- & \textbf{3987.183} & 5198.093 & -- & -- & -- & 3.117 & -- & -- & 0.509 & -- & -- & 1.850 \\
++++++Ecole Superior De Guerre & \textbf{0.349} & 51.327 & 0.318 & \textbf{0.090} & 2.574 & 0.081 & \textbf{3.418} & 11.751 & 0.720 & \textbf{35} & 35 & -- & \textbf{2380.387} & 10426.770 & -- & -- & -- & 0.024 & -- & -- & 0.005 & -- & -- & 0.340 \\
++++++Eglise du dome & \textbf{0.386} & 0.853 & 0.808 & \textbf{0.098} & 0.232 & 0.205 & \textbf{3.987} & 5.944 & 0.910 & \textbf{85} & 85 & -- & 19631.293 & \textbf{9803.940} & -- & -- & -- & 0.037 & -- & -- & 0.010 & -- & -- & 0.270 \\
++++++Folke Filbyter & 85.831 & \textbf{82.656} & 74.596 & \textbf{0.129} & 0.131 & 0.125 & \textbf{19.275} & 28.225 & 10.370 & \textbf{40} & 40 & -- & \textbf{1581.897} & 3755.193 & -- & -- & -- & 70.157 & -- & -- & 0.118 & -- & -- & 4.290 \\
++++++Fort Channing Gate Singapore & \textbf{0.093} & 0.122 & 0.207 & \textbf{0.041} & 0.056 & 0.093 & \textbf{1.783} & 3.649 & 0.520 & \textbf{27} & 27 & -- & \textbf{3434.113} & 6183.925 & -- & -- & -- & 0.020 & -- & -- & 0.008 & -- & -- & 0.250 \\
++++++Golden Statue Somewhere In Hong Kong & 0.232 & -- & 0.292 & 0.053 & -- & 0.073 & 1.829 & -- & 0.400 & 18 & -- & -- & 4095.853 & -- & -- & -- & -- & 0.031 & -- & -- & 0.004 & -- & -- & 0.270 \\
++++++Gustav Vasa & \textbf{3.509} & 25.854 & 34.181 & \textbf{0.220} & 0.775 & 1.085 & \textbf{1.655} & 4.747 & 3.520 & \textbf{18} & 18 & -- & \textbf{709.867} & 3887.653 & -- & -- & -- & 32.266 & -- & -- & 1.145 & -- & -- & 3.150 \\
++++++GustavIIAdolf & \textbf{47.126} & 88.194 & 67.784 & \textbf{8.536} & 13.129 & 9.714 & \textbf{11.722} & 17.220 & 13.910 & \textbf{57} & 57 & -- & \textbf{1044.170} & 2981.017 & -- & -- & -- & 58.458 & -- & -- & 8.524 & -- & -- & 11.490 \\
++++++Jonas Ahlstromer & 48.740 & \textbf{46.584} & 50.190 & \textbf{9.823} & 10.245 & 10.888 & \textbf{11.136} & 11.144 & 10.820 & \textbf{40} & 40 & -- & \textbf{615.450} & 2414.653 & -- & -- & -- & 47.117 & -- & -- & 10.451 & -- & -- & 8.410 \\
++++++Kings College University Of Toronto & \textbf{9.025} & 18.154 & 0.989 & 2.044 & \textbf{1.706} & 0.235 & \textbf{3.565} & 6.501 & 0.900 & \textbf{77} & 77 & -- & \textbf{1078.980} & 2854.715 & -- & -- & -- & 0.085 & -- & -- & 0.017 & -- & -- & 0.340 \\
++++++Lund University Sphinx & \textbf{11.392} & 42.959 & 19.522 & \textbf{2.981} & 8.720 & 4.585 & \textbf{6.056} & 15.000 & 4.780 & \textbf{70} & 70 & -- & \textbf{2718.567} & 4705.610 & -- & -- & -- & 8.752 & -- & -- & 2.191 & -- & -- & 1.360 \\
++++++Nijo Castle Gate & \textbf{0.839} & 8.641 & 1.495 & \textbf{0.165} & 2.791 & 0.286 & \textbf{7.415} & 20.967 & 1.700 & \textbf{19} & 19 & -- & \textbf{1095.177} & 3241.330 & -- & -- & -- & 0.069 & -- & -- & 0.012 & -- & -- & 0.730 \\
++++++Pantheon Paris & -- & 0.509 & 0.192 & -- & 0.066 & 0.050 & -- & 7.493 & 1.470 & -- & 179 & -- & -- & 5565.670 & -- & -- & -- & 0.040 & -- & -- & 0.005 & -- & -- & 0.490 \\
++++++Park Gate Clermont Ferrand & 21.388 & -- & 0.391 & 10.192 & -- & 0.125 & 7.712 & -- & 0.570 & 34 & -- & -- & 1596.940 & -- & -- & -- & -- & 0.049 & -- & -- & 0.022 & -- & -- & 0.350 \\
++++++Plaza De Armas Santiago & 0.775 & -- & 6.782 & 0.337 & -- & 2.944 & 6.009 & -- & 7.400 & 240 & -- & -- & 6372.683 & -- & -- & -- & -- & 2.556 & -- & -- & 1.383 & -- & -- & 4.900 \\
++++++Porta San Donato Bologna & 0.592 & -- & 2.153 & 0.107 & -- & 0.388 & 5.603 & -- & 2.280 & 141 & -- & -- & 3754.163 & -- & -- & -- & -- & 0.095 & -- & -- & 0.046 & -- & -- & 0.750 \\
++++++Round Church Cambridge & 2.137 & -- & 2.451 & 0.927 & -- & 1.003 & 6.103 & -- & 2.660 & 92 & -- & -- & 9779.270 & -- & -- & -- & -- & 1.107 & -- & -- & 0.582 & -- & -- & 1.540 \\
++++++Skansen Kronan Gothenburg & 0.301 & -- & 0.736 & 0.102 & -- & 0.226 & 2.906 & -- & 1.240 & 131 & -- & -- & 17153.677 & -- & -- & -- & -- & 0.026 & -- & -- & 0.008 & -- & -- & 0.670 \\
++++++Smolny Cathedral St Petersburg & 21.775 & -- & 0.554 & 2.111 & -- & 0.051 & 11.239 & -- & 1.660 & 131 & -- & -- & 10362.787 & -- & -- & -- & -- & 0.033 & -- & -- & 0.006 & -- & -- & 0.810 \\
++++++Some Cathedral In Barcelona & 0.651 & -- & 0.880 & 0.233 & -- & 0.315 & 6.271 & -- & 2.870 & 177 & -- & -- & 5001.973 & -- & -- & -- & -- & 0.026 & -- & -- & 0.011 & -- & -- & 0.890 \\
++++++Sri Mariamman Singapore & 1.219 & -- & 2.302 & 0.382 & -- & 0.683 & 10.088 & -- & 4.130 & 222 & -- & -- & 5985.540 & -- & -- & -- & -- & 0.077 & -- & -- & 0.023 & -- & -- & 0.910 \\
++++++Sri Thendayuthapani Singapore & 0.593 & -- & 46.269 & 0.150 & -- & 3.812 & 7.016 & -- & 23.370 & 98 & -- & -- & 13338.167 & -- & -- & -- & -- & 44.170 & -- & -- & 2.870 & -- & -- & 8.440 \\
++++++Sri Veeramakaliamman Singapore & 2.404 & -- & 2.559 & 0.559 & -- & 0.597 & 12.234 & -- & 3.470 & 157 & -- & -- & 12377.927 & -- & -- & -- & -- & 0.175 & -- & -- & 0.040 & -- & -- & 0.730 \\
++++++Statue Of Liberty & 52.358 & -- & 46.887 & 28.503 & -- & 20.012 & 235036.825 & -- & 26.160 & 134 & -- & -- & 99.900 & -- & -- & -- & -- & 9.091 & -- & -- & 4.122 & -- & -- & 6.970 \\
++++++The Pumpkin & 25.731 & -- & 94.672 & 5.552 & -- & 14.890 & 26.876 & -- & 33.410 & 196 & -- & -- & 4327.273 & -- & -- & -- & -- & 98.862 & -- & -- & 14.952 & -- & -- & 24.850 \\
++++++Thian Hook Keng Temple Singapore & 0.927 & -- & 0.832 & 0.093 & -- & 0.082 & 14.915 & -- & 2.750 & 138 & -- & -- & 3298.537 & -- & -- & -- & -- & 0.081 & -- & -- & 0.008 & -- & -- & 1.130 \\
++++++Tsar Nikolai I & 42.166 & -- & 48.499 & 8.732 & -- & 9.467 & 11.210 & -- & 9.790 & 98 & -- & -- & 4929.200 & -- & -- & -- & -- & 36.280 & -- & -- & 7.836 & -- & -- & 6.530 \\
++++++Urban II & 58.919 & -- & 47.490 & 11.006 & -- & 9.467 & 17.538 & -- & 9.380 & 96 & -- & -- & 8889.617 & -- & -- & -- & -- & 48.214 & -- & -- & 9.586 & -- & -- & 6.920 \\
++++++Vercingetorix & 82.952 & -- & 69.328 & 10.195 & -- & 8.788 & 7.257 & -- & 5.080 & 69 & -- & -- & 1223.970 & -- & -- & -- & -- & 17.706 & -- & -- & 3.104 & -- & -- & 1.500 \\
++++++Yueh Hai Ching Temple Singapore & 0.544 & -- & 0.720 & 0.075 & -- & 0.098 & 5.596 & -- & 0.940 & 43 & -- & -- & 1748.667 & -- & -- & -- & -- & 0.043 & -- & -- & 0.014 & -- & -- & 0.650 \\
+++++ \midrule
+++++-Mean & \textbf{16.146} & 23.697 & 18.023 & \textbf{3.121} & 4.014 & 2.912 & 6920.602 & \textbf{13.088} & 5.673 & 100.559 & \textbf{104.818} & -- & \textbf{5819.728} & 7358.187 & -- & -- & -- & 13.695 & -- & -- & 1.937 & -- & -- & 2.992 \\
++++++Mean & \textbf{15.676} & 23.604 & 17.547 & 3.032 & \textbf{2.619} & 2.840 & 6723.056 & \textbf{10.792} & 5.595 & \textbf{102.743} & 77.588 & -- & 5460.373 & \textbf{5336.838} & -- & -- & -- & 13.315 & -- & -- & 1.883 & -- & -- & 2.933 \\
+++++ \bottomrule
+++++ \end{tabular}}
+++++ \end{table*}
++++ ```
++++ - untracked/modified files:
++++ ```
+++++M code/results/single_scene/REPRO.md
+++++ M code/results/single_scene/summary.csv
+++++ M code/results/single_scene/summary_table.md
+++++ M code/results/single_scene/summary_table.tex
++++ ?? MVG_Project_Report_Ortal_Dayan.pdf
++++ ?? "RESFM- ROBUST DEEP EQUIVARIANT STRUCTURE FROM MOTION.pdf"
++++ ?? "claude specs/SPEC_cvpr_experiments.md"
++++ ?? "claude specs/SPEC_sfm_datasets_setup.md"
++++ ?? "claude specs/SPEC_single_scene_experiments.md"
++++ ?? "claude specs/SPEC_uesfm_combined.md"
+++++?? "claude specs/TASK_crossdataset_readiness.md"
++++ ?? code/datasets/Euclidean
++++ ?? tmp_ab_check/
++++ ```
++++ ### esfm-baseline (official ESFM)
++++ - path: `/home/projects/bagon/ortalda/MVG/final-project/esfm-baseline`
++++-- commit: `2c0079497bf5dda1eeb2a902b2f859c6387f213a`
+++++- commit: `de666af53423c1dc3dd412e53c4e153c03d70922`
++++ - **WARNING: working tree dirty.** Diff:
++++ ```diff
++++ (untracked files only)
++++@@ -58,6 +2007,7 @@ Generated: 2026-07-13T20:41:04
++++ ?? code/confs/single_scene_generated/ss_esfm_Buddah_Tooth_Relic_Temple_Singapore_seed1.conf
++++ ?? code/confs/single_scene_generated/ss_esfm_Buddah_Tooth_Relic_Temple_Singapore_seed2.conf
++++ ?? code/confs/single_scene_generated/ss_esfm_Doge_Palace_Venice_seed1.conf
+++++?? code/confs/single_scene_generated/ss_esfm_Doge_Palace_Venice_seed2.conf
++++ ?? code/confs/single_scene_generated/ss_esfm_Door_Lund_seed1.conf
++++ ?? code/confs/single_scene_generated/ss_esfm_Door_Lund_seed2.conf
++++ ?? code/confs/single_scene_generated/ss_esfm_Drinking_Fountain_Somewhere_In_Zurich_seed1.conf
++++@@ -70,6 +2020,7 @@ Generated: 2026-07-13T20:41:04
++++ ?? code/confs/single_scene_generated/ss_esfm_Ecole_Superior_De_Guerre_seed2.conf
++++ ?? code/confs/single_scene_generated/ss_esfm_Eglise_du_dome_seed0.conf
++++ ?? code/confs/single_scene_generated/ss_esfm_Eglise_du_dome_seed1.conf
+++++?? code/confs/single_scene_generated/ss_esfm_Eglise_du_dome_seed2.conf
++++ ?? code/confs/single_scene_generated/ss_esfm_Folke_Filbyter_seed0.conf
++++ ?? code/confs/single_scene_generated/ss_esfm_Folke_Filbyter_seed1.conf
++++ ?? code/confs/single_scene_generated/ss_esfm_Folke_Filbyter_seed2.conf
++++@@ -114,16 +2065,22 @@ Generated: 2026-07-13T20:41:04
++++ ?? code/confs/single_scene_generated/ss_esfm_Round_Church_Cambridge_seed2.conf
++++ ?? code/confs/single_scene_generated/ss_esfm_Skansen_Kronan_Gothenburg_seed0.conf
++++ ?? code/confs/single_scene_generated/ss_esfm_Skansen_Kronan_Gothenburg_seed1.conf
+++++?? code/confs/single_scene_generated/ss_esfm_Skansen_Kronan_Gothenburg_seed2.conf
++++ ?? code/confs/single_scene_generated/ss_esfm_Smolny_Cathedral_St_Petersburg_seed0.conf
++++ ?? code/confs/single_scene_generated/ss_esfm_Smolny_Cathedral_St_Petersburg_seed1.conf
+++++?? code/confs/single_scene_generated/ss_esfm_Smolny_Cathedral_St_Petersburg_seed2.conf
++++ ?? code/confs/single_scene_generated/ss_esfm_Some_Cathedral_In_Barcelona_seed0.conf
+++++?? code/confs/single_scene_generated/ss_esfm_Some_Cathedral_In_Barcelona_seed1.conf
+++++?? code/confs/single_scene_generated/ss_esfm_Some_Cathedral_In_Barcelona_seed2.conf
++++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Mariamman_Singapore_seed0.conf
++++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Mariamman_Singapore_seed1.conf
++++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Mariamman_Singapore_seed2.conf
++++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Thendayuthapani_Singapore_seed0.conf
++++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Thendayuthapani_Singapore_seed1.conf
+++++?? code/confs/single_scene_generated/ss_esfm_Sri_Thendayuthapani_Singapore_seed2.conf
++++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Veeramakaliamman_Singapore_seed0.conf
++++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Veeramakaliamman_Singapore_seed1.conf
+++++?? code/confs/single_scene_generated/ss_esfm_Sri_Veeramakaliamman_Singapore_seed2.conf
++++ ?? code/confs/single_scene_generated/ss_esfm_Statue_Of_Liberty_seed0.conf
++++ ?? code/confs/single_scene_generated/ss_esfm_Statue_Of_Liberty_seed1.conf
++++ ?? code/confs/single_scene_generated/ss_esfm_Statue_Of_Liberty_seed2.conf
++++@@ -137,10 +2094,14 @@ Generated: 2026-07-13T20:41:04
++++ ?? code/confs/single_scene_generated/ss_esfm_Tsar_Nikolai_I_seed1.conf
++++ ?? code/confs/single_scene_generated/ss_esfm_Tsar_Nikolai_I_seed2.conf
++++ ?? code/confs/single_scene_generated/ss_esfm_Urban_II_seed0.conf
+++++?? code/confs/single_scene_generated/ss_esfm_Urban_II_seed1.conf
+++++?? code/confs/single_scene_generated/ss_esfm_Urban_II_seed2.conf
++++ ?? code/confs/single_scene_generated/ss_esfm_Vercingetorix_seed0.conf
++++ ?? code/confs/single_scene_generated/ss_esfm_Vercingetorix_seed1.conf
++++ ?? code/confs/single_scene_generated/ss_esfm_Vercingetorix_seed2.conf
++++ ?? code/confs/single_scene_generated/ss_esfm_Yueh_Hai_Ching_Temple_Singapore_seed0.conf
+++++?? code/confs/single_scene_generated/ss_esfm_Yueh_Hai_Ching_Temple_Singapore_seed1.conf
+++++?? code/confs/single_scene_generated/ss_esfm_Yueh_Hai_Ching_Temple_Singapore_seed2.conf
++++ ```
++++ 
++++ ## Environment (shared venv used for BOTH methods)
++++diff --git a/code/results/single_scene/summary.csv b/code/results/single_scene/summary.csv
++++index e4f9f5e..d859f0c 100644
++++--- a/code/results/single_scene/summary.csv
+++++++ b/code/results/single_scene/summary.csv
++++@@ -2,171 +2,149 @@ method,scene,seed,rot_err_deg,pos_err,reproj_err_px,reproj_med_px,n_cams,runtime
++++ esfm,Alcatraz Courtyard,0,0.324288380204305,0.0854554642858637,3.379232694353053,1.9204072326560708,133,4821.4,4792.707528591156,99999.0
++++ esfm,Alcatraz Courtyard,1,0.36138678157391557,0.09112912757839253,3.3691035794153503,1.8536696768772378,133,4825.72,4804.594581127167,99999.0
++++ esfm,Alcatraz Courtyard,2,0.4015043649148433,0.10235190268069758,3.557242072642648,1.9198007399315369,133,4833.02,4812.242679834366,99999.0
++++-uesfm,Alcatraz Courtyard,0,0.6370266175309793,0.1671943803031883,5.956592777085197,3.6638282340102246,133,6478.06,6448.735792636871,99999.0
++++-uesfm,Alcatraz Courtyard,1,0.5812669610362707,0.15343994793419002,5.71686544495911,3.550076086474883,133,6483.48,6454.087207078934,99999.0
++++-uesfm,Alcatraz Courtyard,2,0.6293740753551684,0.16343948136157807,5.713459273661585,3.459422664018691,133,6483.73,6451.153926610947,99999.0
+++++uesfm,Alcatraz Courtyard,0,0.5509063563081962,0.14050635120746804,5.528957815039945,3.569220349534855,133,6556.5,6511.41095662117,99999.0
+++++uesfm,Alcatraz Courtyard,1,0.6823224031499351,0.17277455831381494,5.435931436314427,3.4771386471547348,133,6552.19,6506.419185638428,99999.0
+++++uesfm,Alcatraz Courtyard,2,29.163200369074893,6.342299350838519,10.849871644564203,6.781661682554276,133,6544.26,6512.667362689972,99999.0
++++ esfm,Alcatraz Water Tower,0,0.6984432687370988,0.38694796813795407,4.741313233832226,2.1058898944784543,172,2461.59,2440.421297073364,99999.0
++++ esfm,Alcatraz Water Tower,1,0.5944425806062753,0.3287537445614408,4.5656682619915765,1.9703698309044646,172,2467.28,2454.429310798645,99999.0
++++ esfm,Alcatraz Water Tower,2,0.7051984712289041,0.39359395644708545,4.982802041506025,2.1318655747232063,172,2455.14,2441.783402681351,99999.0
++++-uesfm,Alcatraz Water Tower,0,18.69941055700858,8.189402666249999,12.38813738647375,6.481578998967327,172,3857.94,3831.41609287262,99999.0
++++-uesfm,Alcatraz Water Tower,1,1.3394060108035364,0.7384271683550214,6.577617375342355,3.200105455600137,172,3887.47,3865.343374729156,99999.0
++++-uesfm,Alcatraz Water Tower,2,2.200189184784068,1.3839747978399466,6.854463337738994,3.1730434024886014,172,3819.31,3799.390902757645,99999.0
+++++uesfm,Alcatraz Water Tower,0,2.016029335283383,1.087082854315953,7.9502009315303575,4.413488326253207,172,3805.19,3775.908159971237,99999.0
+++++uesfm,Alcatraz Water Tower,1,1.4259772151091428,0.7645994701417347,7.177871865477131,3.7683369835989504,172,3787.07,3763.27596282959,99999.0
+++++uesfm,Alcatraz Water Tower,2,0.9982626119141534,0.558403777925872,7.063395425015731,3.814475911180562,172,3795.37,3773.579354286194,99999.0
++++ esfm,Buddah Tooth Relic Temple Singapore,0,1.3132762212436329,0.31652548993670354,9.903890396062208,4.634952217191846,162,2921.45,2903.059753894806,99999.0
++++ esfm,Buddah Tooth Relic Temple Singapore,1,0.9908742070528852,0.2335767172234499,9.836436953133868,4.661767906617449,162,2915.39,2904.387546062469,99999.0
++++ esfm,Buddah Tooth Relic Temple Singapore,2,1.089403690069201,0.2896515999941069,10.151103921611115,4.628183275251743,162,2913.67,2902.928613901138,99999.0
++++-uesfm,Buddah Tooth Relic Temple Singapore,0,1.9767734026333772,0.5777116480820644,15.584999477939833,8.800407928031486,162,4315.97,4296.739196300507,99999.0
++++-uesfm,Buddah Tooth Relic Temple Singapore,1,1.4532821126601407,0.345133635718661,16.753182673783428,9.555989756393366,162,4334.28,4311.474883317947,99999.0
++++-uesfm,Buddah Tooth Relic Temple Singapore,2,32.81105779154638,6.0024233739495125,26.564606536203048,13.925597303414058,162,4312.37,4291.042752504349,99999.0
+++++uesfm,Buddah Tooth Relic Temple Singapore,0,1.5799948547864517,0.43886498334860274,13.484389374830494,7.719072745032001,162,4294.34,4266.538989067078,99999.0
+++++uesfm,Buddah Tooth Relic Temple Singapore,1,1.934129180062467,0.5924991617860504,13.872322562031941,8.04977435062149,162,4303.62,4259.311170101166,99999.0
+++++uesfm,Buddah Tooth Relic Temple Singapore,2,1.8591664615796053,0.5279087705483619,14.731758217446671,8.481044153007865,162,4270.55,4249.088565349579,99999.0
++++ esfm,Doge Palace Venice,0,0.9741701978804636,0.2578214102266912,6.511459231891615,3.6947627756511525,241,28795.18,28746.22650837898,99999.0
++++ esfm,Doge Palace Venice,1,0.627860879835449,0.1773446161890865,6.351890452986879,3.537139014829222,241,28742.27,28699.91308617592,99999.0
++++-uesfm,Doge Palace Venice,0,0.9110673544913167,0.2904399918743801,10.966183720673449,6.771416746021886,241,21074.95,21017.30448675156,99999.0
+++++esfm,Doge Palace Venice,2,0.6576305681966796,0.19114542381679772,6.214841374149415,3.591638795332466,241,10030.76,9989.16938996315,99999.0
+++++uesfm,Doge Palace Venice,0,1.120669822868001,0.3500031006198606,10.496216444068507,6.9730858315176345,241,20767.26,20701.21099281311,99999.0
++++ esfm,Door Lund,0,0.01167051570787235,0.003654424593160691,1.8552512563951393,0.9492901559591532,12,4892.41,4878.351888656616,99999.0
++++ esfm,Door Lund,1,0.009685491454411136,0.0030270923396151107,1.6462990657331118,0.9049495114630693,12,4884.51,4877.529450178146,99999.0
++++ esfm,Door Lund,2,0.02068285629921104,0.005193423834514985,1.933833706356197,0.9506754982524039,12,4888.17,4880.201772212982,99999.0
++++-uesfm,Door Lund,0,14.059454220682019,0.9733778175929079,11.85413335315025,6.517535151323047,12,9185.47,8715.343282699585,95000.0
++++-uesfm,Door Lund,1,14.111993883089163,0.9618908892808385,11.853785941610944,6.504290333453439,12,9203.34,8729.614821910858,95000.0
++++-uesfm,Door Lund,2,14.344083655704312,0.9193773651816372,12.058851721280973,6.643622749322514,12,9176.97,9162.127201795578,99999.0
+++++uesfm,Door Lund,0,0.03084538453455712,0.007522151486969653,2.3711871496098955,1.6703501976219666,12,9262.05,9243.771874427795,99999.0
+++++uesfm,Door Lund,1,13.889154901234134,1.0160235649641791,11.830787694675347,6.5775252925915195,12,9243.42,7848.296544790268,85000.0
++++ esfm,Drinking Fountain Somewhere In Zurich,0,25.804179845300204,1.1242096722251314,8.219025541657992,3.3464550934365884,14,888.07,874.4874215126038,99999.0
++++ esfm,Drinking Fountain Somewhere In Zurich,1,25.754577171528883,1.1271567319301725,8.245599860088616,3.352895635561939,14,882.33,873.2415659427643,99999.0
++++ esfm,Drinking Fountain Somewhere In Zurich,2,0.019604308735801684,0.006780729271172703,1.7090486008047308,0.6375575382268279,14,881.11,873.6504108905792,99999.0
++++-uesfm,Drinking Fountain Somewhere In Zurich,0,26.27952541668808,1.102546079947844,8.40971803772888,3.4593948661771203,14,2969.96,2950.480526924133,99999.0
++++-uesfm,Drinking Fountain Somewhere In Zurich,1,26.264849988219943,1.1078406736783482,8.547474098662555,3.585334207185317,14,2960.9,2504.42572927475,85000.0
++++-uesfm,Drinking Fountain Somewhere In Zurich,2,0.06235835436589852,0.011103715758719332,2.5743283079384534,1.5361135173722633,14,2960.08,2945.673992872238,99999.0
+++++uesfm,Drinking Fountain Somewhere In Zurich,0,25.946100100284117,1.1178920745623426,8.471712388928934,3.6241397282821626,14,5817.37,5796.999963760376,99999.0
+++++uesfm,Drinking Fountain Somewhere In Zurich,1,25.973070279327214,1.1166462082604052,8.538877295313494,3.6343523276029472,14,5821.94,5806.947921991348,99999.0
+++++uesfm,Drinking Fountain Somewhere In Zurich,2,0.06926036271356541,0.01532826351119577,3.015375672575656,1.7451989053599506,14,5806.35,5792.085475206375,99999.0
++++ esfm,East Indiaman Goteborg,0,6.033696331509855,1.0773322362193534,7.427433157074123,2.292729458380311,179,3990.25,3970.732353687286,99999.0
++++ esfm,East Indiaman Goteborg,1,5.313967690277326,0.9356515362150493,7.350753973659207,2.2747477951684703,179,3986.66,3970.914752960205,99999.0
++++ esfm,East Indiaman Goteborg,2,5.357164002763642,0.9254557315236661,7.174538078735992,2.2493605774605823,179,3984.64,3967.810528755188,99999.0
++++-uesfm,East Indiaman Goteborg,0,4.992588850190341,0.9093886386013725,9.930437318497303,4.011819888270754,179,5247.16,5221.816437721252,99999.0
++++-uesfm,East Indiaman Goteborg,1,5.382092262718057,0.9927766021898647,10.035611563702533,3.912004668740913,179,5245.65,5210.653891563416,99999.0
++++-uesfm,East Indiaman Goteborg,2,5.554715231405653,0.9752252406138763,9.5551521706325,3.7417116053580193,179,5231.48,5207.815300703049,99999.0
+++++uesfm,East Indiaman Goteborg,0,5.365772057886591,0.9552115949640507,9.655123223073948,4.174486084237091,179,5203.6,5178.753549814224,99999.0
+++++uesfm,East Indiaman Goteborg,1,10.45533482189932,2.0687908277631517,9.98932850274456,4.110771460953736,179,5195.98,5174.958552598953,99999.0
+++++uesfm,East Indiaman Goteborg,2,5.872735570127935,1.0672842053553182,10.745053220477157,4.480528977857714,179,5194.7,5173.540862560272,99999.0
++++ esfm,Ecole Superior De Guerre,0,0.4863901164687255,0.1239377922325492,3.750666445958877,2.0528113108131016,35,2385.33,2371.708672523499,99999.0
++++ esfm,Ecole Superior De Guerre,1,0.3031794677170201,0.07921628898965839,3.2456935905644406,1.730123845260616,35,2379.34,2371.191816806793,99999.0
++++ esfm,Ecole Superior De Guerre,2,0.2580439805296837,0.06772089648290676,3.257951412507941,1.687291979251782,35,2376.49,2368.935952663422,99999.0
++++-uesfm,Ecole Superior De Guerre,0,0.28243957678843373,0.0778675008788964,5.291358032458062,3.601773151002547,35,4647.51,4622.850021600723,99999.0
++++-uesfm,Ecole Superior De Guerre,1,0.25507313740145815,0.06695766472508183,5.170495431958795,3.5179325723573998,35,4636.07,4616.890106916428,99999.0
++++-uesfm,Ecole Superior De Guerre,2,0.3137499276035537,0.08683967313665406,5.583934687507232,3.7813953498055604,35,4617.92,4602.193422079086,99999.0
+++++uesfm,Ecole Superior De Guerre,0,51.32681795107141,2.57407526776939,11.751384768771835,6.449293516058521,35,10426.77,10405.3174200058,99999.0
+++++uesfm,Ecole Superior De Guerre,1,0.2738850931952059,0.0734625834578241,5.5413290518324265,3.89237755756953,35,10405.39,10384.56669402122,99999.0
++++ esfm,Eglise du dome,0,0.3242410065346163,0.0824797769992069,3.906129661328729,1.3512292775673158,85,25054.74,25028.64922428131,99999.0
++++ esfm,Eglise du dome,1,0.3771623368290559,0.09432763361451474,3.692490941110161,1.297032357665518,85,25144.16,25116.01952123642,99999.0
++++-uesfm,Eglise du dome,0,45.800562123362404,6.451383786272971,7.963674120656965,3.986938865913781,85,23371.73,23333.49408817291,99999.0
+++++esfm,Eglise du dome,2,0.45623311880194967,0.11755570327842058,4.363674496823582,1.384257241090686,85,8694.98,8671.502047538757,99999.0
+++++uesfm,Eglise du dome,0,0.8528700994829498,0.23193079249562099,5.944108200898114,2.9306630821877517,85,9803.94,9764.585114717484,99999.0
+++++uesfm,Eglise du dome,1,0.6936386490465627,0.1932815436676001,6.091509620719386,2.9784748994578076,85,9788.95,9753.742706537247,99999.0
++++ esfm,Folke Filbyter,0,89.03873163576341,0.12895560134543632,17.986624161769267,7.423093846189277,40,1582.8,1573.360969305038,99999.0
++++ esfm,Folke Filbyter,1,84.74740427947457,0.13132264058966686,17.98706393015447,7.385900243615879,40,1579.13,1572.079373121262,99999.0
++++ esfm,Folke Filbyter,2,83.70817707245973,0.1261878953205589,21.85200129525089,6.324245998829542,40,1583.76,1576.207699537277,99999.0
++++-uesfm,Folke Filbyter,0,80.34590333370599,0.13169898199385566,22.83327162619912,11.711593158296864,40,3701.49,3129.452286243439,85000.0
++++-uesfm,Folke Filbyter,1,81.28676564061475,0.1282318052949447,24.480394314414323,9.74969805057316,40,3698.11,3681.891911745071,99999.0
++++-uesfm,Folke Filbyter,2,82.23270227083374,0.1312531386108397,22.153947807931974,10.961536903594071,40,3709.96,3695.84988451004,99999.0
+++++uesfm,Folke Filbyter,0,84.55526303793802,0.13091275274611416,29.190267154194732,11.160073225547443,40,3747.65,3727.990335702896,99999.0
+++++uesfm,Folke Filbyter,1,84.87047537526948,0.13112291273275417,32.36503566471496,13.743964264262598,40,3743.53,3728.497133255005,99999.0
+++++uesfm,Folke Filbyter,2,78.54375401873024,0.1316690349103016,23.11854509839131,10.94439865744236,40,3774.4,3007.95537352562,80000.0
++++ esfm,Fort Channing Gate Singapore,0,0.10143646071220622,0.04531774433800129,1.7158776598376824,0.8165178663538488,27,3438.39,3428.288990497589,99999.0
++++ esfm,Fort Channing Gate Singapore,1,0.08873951454303304,0.039001318240256114,1.783712095722291,0.8156020254191567,27,3430.88,3424.490573883057,99999.0
++++ esfm,Fort Channing Gate Singapore,2,0.09031224636384508,0.03957289173255022,1.850800887931411,0.820658413919779,27,3433.07,3426.547735452652,99999.0
++++-uesfm,Fort Channing Gate Singapore,0,33.158343156383296,11.908537566474344,12.69248500308198,5.438758486477886,27,6209.29,6192.107179880142,99999.0
++++-uesfm,Fort Channing Gate Singapore,1,0.07319798057764343,0.03541606344760984,3.4819579894798927,2.2085534812612853,27,6159.7,6144.797720909119,99999.0
++++-uesfm,Fort Channing Gate Singapore,2,0.15394831153888047,0.0686769122909015,3.77993377770425,2.2995094868650368,27,6147.27,6132.745970487595,99999.0
+++++uesfm,Fort Channing Gate Singapore,0,0.10214634402234797,0.048923384893640356,3.7907267588398517,2.449846070018404,27,6175.88,6156.048281908035,99999.0
+++++uesfm,Fort Channing Gate Singapore,1,0.14219758622796153,0.06357969294387673,3.5066994916782015,2.1555120267718575,27,6191.97,6176.711992740631,99999.0
++++ esfm,Golden Statue Somewhere In Hong Kong,0,0.23894829924734365,0.053817571637812096,1.7239352536484445,0.744089758310296,18,4096.74,4087.775670289993,99999.0
++++ esfm,Golden Statue Somewhere In Hong Kong,1,0.2233990458438394,0.053061028269618456,1.8936878914364637,0.7746436618607616,18,4096.28,4088.994391202927,99999.0
++++ esfm,Golden Statue Somewhere In Hong Kong,2,0.23397914670516012,0.051137498616263405,1.87070255923672,0.7763877145267694,18,4094.54,4087.674698591232,99999.0
++++-uesfm,Golden Statue Somewhere In Hong Kong,0,0.36074365888948035,0.06894135246785482,2.939092008443413,1.863811153764833,18,7761.69,7746.569520235062,99999.0
++++-uesfm,Golden Statue Somewhere In Hong Kong,1,0.35638930493960375,0.06132592684828702,2.686157398440538,1.735571387461727,18,7759.16,7745.389947891235,99999.0
++++-uesfm,Golden Statue Somewhere In Hong Kong,2,0.3955566726941678,0.09197790539367184,2.878341172309852,1.8164738202745854,18,7766.46,7751.458931922913,99999.0
++++ esfm,Gustav Vasa,0,3.4881363194229826,0.2273181104934767,1.7369784534725496,0.9521606063342205,18,711.53,703.3091380596161,99999.0
++++ esfm,Gustav Vasa,1,3.4307883498221035,0.21432466556133864,1.5556448596237022,0.9232184455756751,18,710.28,703.9906163215637,99999.0
++++ esfm,Gustav Vasa,2,3.6089062562451386,0.2196578287395002,1.673803634770729,0.9429755567757121,18,707.79,701.5903799533844,99999.0
++++-uesfm,Gustav Vasa,0,5.828154714648938,0.32510821331670736,3.0711042143581646,2.4170166616864357,18,2566.0,2547.931907176971,99999.0
++++-uesfm,Gustav Vasa,1,6.515239842974278,0.35941242702736925,3.089677100739632,2.4476428398062584,18,2530.9,1385.089058637619,55000.0
++++-uesfm,Gustav Vasa,2,5.954036423569478,0.34942942401850696,3.101296679577594,2.400483554095863,18,2533.76,2520.798699855804,99999.0
+++++uesfm,Gustav Vasa,0,36.645536646380805,1.0070193571286277,4.980488290477233,3.197977230260218,18,2562.94,2419.297919034958,95000.0
+++++uesfm,Gustav Vasa,1,36.4914104514929,1.03254185606701,5.377024289258223,3.4818421684593366,18,4566.81,4543.375423431396,99999.0
+++++uesfm,Gustav Vasa,2,4.4238740734834625,0.284762443333143,3.8820183248893803,2.8663959682636566,18,4533.21,4516.61039352417,99999.0
++++ esfm,GustavIIAdolf,0,46.92758389264633,8.570499881283022,11.684017169857798,5.131125459552879,57,1047.82,1038.018162250519,99999.0
++++ esfm,GustavIIAdolf,1,47.21999673071185,8.515052769156952,11.695065125063977,5.091617086264161,57,1041.41,1034.5799472332,99999.0
++++ esfm,GustavIIAdolf,2,47.2312223396789,8.523261295519099,11.787583212607423,5.2715388868550574,57,1043.28,1036.478415489197,99999.0
++++-uesfm,GustavIIAdolf,0,89.37607436538163,13.276440526172165,15.53883714368065,5.864332719302596,57,2904.42,2881.687849521637,99999.0
++++-uesfm,GustavIIAdolf,1,70.16367503302509,11.555242253627961,15.162174576474532,6.9709512967490355,57,2884.62,2870.103493452072,99999.0
++++-uesfm,GustavIIAdolf,2,72.72996778765942,11.054260851523775,16.85269001998736,9.009102977660282,57,2888.7,2874.096453428268,99999.0
+++++uesfm,GustavIIAdolf,0,85.51179144359693,12.792264407383987,17.018024925092185,8.196266635685827,57,2973.35,2956.916826248169,99999.0
+++++uesfm,GustavIIAdolf,1,88.9494736685426,13.260723460498589,17.699982689478404,8.888010081099381,57,2995.3,2976.055320978165,99999.0
+++++uesfm,GustavIIAdolf,2,90.12096202318821,13.335169188185864,16.941538974035925,7.772181288890227,57,2974.4,2960.110015630722,99999.0
++++ esfm,Jonas Ahlstromer,0,64.46662959289324,11.985605264945464,12.602584968179986,2.9168445332917745,40,614.88,607.3498430252075,99999.0
++++ esfm,Jonas Ahlstromer,1,38.42510202464649,8.741159740710629,10.25268858530355,3.7024742418884333,40,618.73,611.5567183494568,99999.0
++++ esfm,Jonas Ahlstromer,2,43.32891670826628,8.741053567623224,10.553031866494822,3.031717304343782,40,612.74,606.6802380084991,99999.0
++++-uesfm,Jonas Ahlstromer,0,42.30073879858729,10.272232427418913,9.098970988131883,3.992857074878733,40,2339.57,1280.604686737061,55000.0
++++-uesfm,Jonas Ahlstromer,1,42.18412804385421,10.310808592826955,8.453808908218651,3.4329852879523464,40,2334.33,2320.362742424011,99999.0
++++-uesfm,Jonas Ahlstromer,2,42.25301394187691,10.315519611163806,8.793789077520445,3.8046125347596442,40,2311.04,2297.139800786972,99999.0
+++++uesfm,Jonas Ahlstromer,0,45.264596178655594,9.403212752141883,11.336617283269852,4.728596140494824,40,2421.11,2401.221394062042,99999.0
+++++uesfm,Jonas Ahlstromer,1,51.490398556789884,10.860548616493958,12.821407884052217,5.012487046810724,40,2428.25,2415.139223337173,99999.0
+++++uesfm,Jonas Ahlstromer,2,42.99784237868634,10.470910789899067,9.274184391894982,4.4612757046777345,40,2394.6,2381.585824012756,99999.0
++++ esfm,Kings College University Of Toronto,0,14.400461796497643,3.2653370282498866,4.064353916685246,1.6753245173381437,77,1074.94,1066.306197404861,99999.0
++++ esfm,Kings College University Of Toronto,1,11.956699445712928,2.705452710546247,3.9727144140636783,1.5797087908756111,77,1071.64,1064.129351377487,99999.0
++++ esfm,Kings College University Of Toronto,2,0.7191591237413282,0.16178494376153177,2.656807607821862,1.1949677972638288,77,1090.36,1082.987608671188,99999.0
++++-uesfm,Kings College University Of Toronto,0,31.932532001930838,2.9775501815475,6.099555590572752,3.8717713150174773,77,2884.23,2866.194714784622,99999.0
++++-uesfm,Kings College University Of Toronto,1,2.4241467960672063,0.41577835428574456,4.631696476065898,3.148046400399828,77,2899.51,2884.114916801453,99999.0
++++-uesfm,Kings College University Of Toronto,2,30.212552334890066,2.3209884971628156,5.844709739288878,3.754268841443037,77,2874.99,2860.901805400848,99999.0
+++++uesfm,Kings College University Of Toronto,0,29.200261744017546,2.0893308856114428,7.037455848998898,4.8227457854711675,77,2856.72,2838.923613071442,99999.0
+++++uesfm,Kings College University Of Toronto,1,7.107988087620218,1.3235014088907995,5.965363878755208,4.179664786063016,77,2852.71,2839.42994761467,99999.0
+++++uesfm,Kings College University Of Toronto,2,6.543121223644844,1.5041055210367995,6.0501544660211914,4.127143945227024,77,2863.77,2850.061238765717,99999.0
++++ esfm,Lund University Sphinx,0,13.661465370705821,3.5591094267381553,6.242667318159146,2.715620555070548,70,2718.96,2709.5872194767,99999.0
++++ esfm,Lund University Sphinx,1,12.524302160109885,3.2130942199657664,6.1182420215295075,2.7468468987195953,70,2722.86,2715.36418390274,99999.0
++++ esfm,Lund University Sphinx,2,7.98961938632192,2.1710959765491817,5.807289242508196,2.893370267054225,70,2713.88,2706.478076219559,99999.0
++++-uesfm,Lund University Sphinx,0,15.165716414733652,3.5564827651349713,10.803408462752863,6.730422995453065,70,4644.92,4626.135177612305,99999.0
++++-uesfm,Lund University Sphinx,1,68.88899086055604,13.548881652837478,14.336626262363456,7.626356837439143,70,4703.31,4687.50173997879,99999.0
++++-uesfm,Lund University Sphinx,2,15.014326075576053,3.5607694164340526,11.039432108003597,6.81081192121135,70,4615.6,4600.294319868088,99999.0
+++++uesfm,Lund University Sphinx,0,42.95863056778373,8.719684065980621,14.999945982069319,8.361965506089375,70,4705.61,4681.787243127823,99999.0
++++ esfm,Nijo Castle Gate,0,0.6521524412276166,0.13305866403975986,7.404953325705311,3.0629804954987674,19,1097.29,1084.301982402802,99999.0
++++ esfm,Nijo Castle Gate,1,0.8564043952718835,0.16950374865879153,7.358076475544799,3.1235223975185864,19,1094.89,1084.640900611877,99999.0
++++ esfm,Nijo Castle Gate,2,1.0089926380330574,0.19095470213426863,7.481049991287358,3.1671779658403687,19,1093.35,1083.446812152863,99999.0
++++-uesfm,Nijo Castle Gate,0,1.6923297320319277,0.3246009364794658,11.792009654594867,6.945622003724464,19,3310.13,3288.75860619545,99999.0
++++-uesfm,Nijo Castle Gate,1,1.6697791743236226,0.32440429939345133,12.704681259536075,7.466128811395402,19,3308.19,3290.209539175034,99999.0
++++-uesfm,Nijo Castle Gate,2,1.682403384345942,0.32817821817561266,11.917977420973372,7.102279949089778,19,3301.77,3284.964460611343,99999.0
++++-uesfm,Pantheon Paris,0,0.3823220210788846,0.05987691585642601,7.649321203280243,4.797127317249956,179,5506.88,5473.694003582001,99999.0
++++-uesfm,Pantheon Paris,1,45.61471478909275,6.671068486991568,16.912359249810606,6.602415648206237,179,5491.31,5459.329566717148,99999.0
++++-uesfm,Pantheon Paris,2,12.03796171625132,2.465102828965073,16.125095834055482,6.860965809143959,179,5493.94,5463.555550813675,99999.0
+++++uesfm,Nijo Castle Gate,0,8.641303631787597,2.791258638759238,20.96738948203601,10.297911838853688,19,3241.33,3220.324777603149,99999.0
+++++uesfm,Pantheon Paris,0,0.5453004964472504,0.06999784414241637,7.790870638815493,5.057509442480535,179,5594.6,5562.967379331589,99999.0
+++++uesfm,Pantheon Paris,1,0.472097972346089,0.062163803587728925,7.194706475909503,4.639524309087925,179,5536.74,5501.721322774887,99999.0
++++ esfm,Park Gate Clermont Ferrand,0,19.399216821978534,9.438471236639753,7.521969683469249,3.892136696028138,34,1597.96,1588.07075381279,99999.0
++++ esfm,Park Gate Clermont Ferrand,1,19.325402297843297,9.417998214523331,7.465007292717891,3.849088919564891,34,1596.87,1589.703137874603,99999.0
++++ esfm,Park Gate Clermont Ferrand,2,25.440611579781365,11.720641869372841,8.15032188787613,4.0871739024940315,34,1595.99,1588.841923236847,99999.0
++++-uesfm,Park Gate Clermont Ferrand,0,19.27114964212729,9.381930985797803,7.856513901068924,4.317004881495189,34,3598.69,3580.396904230118,99999.0
++++-uesfm,Park Gate Clermont Ferrand,1,19.288145181777256,9.394362801959678,7.762082007624792,4.207851936960509,34,3592.66,3577.494926214218,99999.0
++++-uesfm,Park Gate Clermont Ferrand,2,19.345727373971844,9.415066249707772,7.936652645099164,4.425507919979627,34,3600.86,3581.473034620285,99999.0
+++++uesfm,Park Gate Clermont Ferrand,0,26.45121972215746,12.46688671568752,9.374365736774523,5.572822711026232,34,3659.9,3637.290126085281,99999.0
++++ esfm,Plaza De Armas Santiago,0,0.7341628728579594,0.32156050159900956,6.000154308231358,2.679971945741675,240,6375.02,6333.455562829971,99999.0
++++ esfm,Plaza De Armas Santiago,1,0.5176165701053533,0.22744923951256435,5.661576462867733,2.533952088248649,240,6373.68,6336.030319213867,99999.0
++++ esfm,Plaza De Armas Santiago,2,1.0740077415291467,0.46304622323619105,6.365716833190761,2.733195456670397,240,6369.35,6331.436369419098,99999.0
++++-uesfm,Plaza De Armas Santiago,0,6.6772176498607925,2.909933722327364,11.384209379387066,5.623987151554922,240,6540.76,6491.619550704956,99999.0
++++-uesfm,Plaza De Armas Santiago,1,1.3364898126442175,0.5686650140764532,7.951759184923103,3.933795015892702,240,6536.61,6482.320575714111,99999.0
++++-uesfm,Plaza De Armas Santiago,2,1.9671483940705004,0.8276654865349118,9.054185015150745,4.441179120460681,240,6529.24,6477.723381996155,99999.0
++++ esfm,Porta San Donato Bologna,0,0.5746001989174886,0.10608894230214515,5.691906895070585,2.4183460353027986,141,3751.75,3734.281898736954,99999.0
++++ esfm,Porta San Donato Bologna,1,0.8978003419853328,0.14474404936935045,5.682379581165418,2.563141107750346,141,3758.51,3742.688964605331,99999.0
++++ esfm,Porta San Donato Bologna,2,0.30440272958620346,0.07026546193502164,5.435311727517131,2.1626312981864086,141,3752.23,3738.056202173233,99999.0
++++-uesfm,Porta San Donato Bologna,0,1.1269164513732723,0.19275737956525044,10.071225650489282,5.557466375146339,141,5111.57,5083.21807050705,99999.0
++++-uesfm,Porta San Donato Bologna,1,60.66366912240625,10.234123128225525,19.593360464637342,9.305574874002819,141,5125.57,5098.085618019104,99999.0
++++-uesfm,Porta San Donato Bologna,2,1.2271289404198877,0.19605254084885107,10.066750816092268,5.595831381287904,141,5116.95,5093.473650932312,99999.0
++++ esfm,Round Church Cambridge,0,2.1173198662081005,0.9129854085205968,5.854320175654564,1.9683268910531708,92,9788.65,9759.719386577606,99999.0
++++ esfm,Round Church Cambridge,1,2.1415279636494953,0.9267927094737857,6.33905951937897,2.0439997148640834,92,9764.61,9741.356333732605,99999.0
++++-uesfm,Round Church Cambridge,0,2.5434984075631974,1.0219048587087352,8.914080922081759,4.131734829440289,92,12088.06,12047.02147889137,99999.0
++++-uesfm,Round Church Cambridge,1,2.435781054756339,0.9827352719070902,8.178482224933683,3.8400051230290893,92,12034.64,11993.72608184814,99999.0
+++++esfm,Round Church Cambridge,2,2.1507951248804917,0.9401429799233081,6.115230559713554,2.0517209910251797,92,9784.55,9763.525354623795,99999.0
++++ esfm,Skansen Kronan Gothenburg,0,0.3380139741078169,0.1158931606908392,3.0478862369293624,1.5340801117570733,131,17163.94,17140.15425729752,99999.0
++++-uesfm,Skansen Kronan Gothenburg,0,0.37472494768162085,0.1243676312212316,4.798736338080396,3.348916057055834,131,16235.09,16202.21351981163,99999.0
+++++esfm,Skansen Kronan Gothenburg,1,0.21516831826827018,0.07458182746108981,2.786705021535272,1.4619994473169777,131,17129.62,17109.18022465706,99999.0
+++++esfm,Skansen Kronan Gothenburg,2,0.3508356515622903,0.11667319277987132,2.8836839507901773,1.5216584205042176,131,17167.47,17142.66050887108,99999.0
++++ esfm,Smolny Cathedral St Petersburg,0,21.81278753416543,2.110667559831311,11.2315357584775,6.379259553776089,131,10364.67,10320.51852750778,99999.0
++++ esfm,Smolny Cathedral St Petersburg,1,21.799828748358873,2.111313741354268,11.225141339870355,6.36605939788532,131,10360.52,10319.01374697685,99999.0
++++-uesfm,Smolny Cathedral St Petersburg,0,22.143392894578028,2.0985584569799265,12.52645725687386,7.895265841408903,131,10475.74,10416.62969732285,99999.0
+++++esfm,Smolny Cathedral St Petersburg,2,21.71090600320656,2.110307413725059,11.259799763291888,6.4136596726420905,131,10363.17,10320.08415436745,99999.0
+++++esfm,Some Cathedral In Barcelona,0,0.6775058368464424,0.24064262464083658,6.20165095815897,3.7253636674062887,177,5002.81,4985.274734258652,99999.0
+++++esfm,Some Cathedral In Barcelona,1,0.6722147495861831,0.2423973792150467,6.350786792749083,3.720536755931007,177,5009.15,4991.552013158798,99999.0
+++++esfm,Some Cathedral In Barcelona,2,0.6046048207068493,0.21699252142655406,6.261075100347952,3.7258171467923633,177,4993.96,4975.845166683197,99999.0
++++ esfm,Sri Mariamman Singapore,0,1.4920493985513708,0.4412389781578158,10.406406306167431,6.369585247957454,222,5987.98,5970.066485881805,99999.0
++++ esfm,Sri Mariamman Singapore,1,1.1345564396944785,0.3608906720626337,9.703927906328842,5.885963785879842,222,5993.62,5975.365797519684,99999.0
++++ esfm,Sri Mariamman Singapore,2,1.0293707921546553,0.3426709250774071,10.15220066698994,6.067537729054635,222,5975.02,5958.786058187485,99999.0
++++-uesfm,Sri Mariamman Singapore,0,16.417883129790784,5.526406595652791,19.962711816743173,12.18830031311293,222,6531.45,6494.750840425491,99999.0
++++-uesfm,Sri Mariamman Singapore,1,1.3793173305528281,0.5014534676018023,16.10857603648069,10.651562750176396,222,6467.65,6426.201541185379,99999.0
++++-uesfm,Sri Mariamman Singapore,2,1.639001766282598,0.536397596737868,17.490398022274007,11.439337896248928,222,6457.9,6427.717175960541,99999.0
++++ esfm,Sri Thendayuthapani Singapore,0,0.7190396606626565,0.17922550285921848,7.12365374281766,2.231392236341273,98,13362.09,13320.09781932831,99999.0
++++-uesfm,Sri Thendayuthapani Singapore,0,47.0750455982536,3.8553135654690074,14.133068630999212,7.117179633590087,98,13808.44,13754.94554877281,99999.0
+++++esfm,Sri Thendayuthapani Singapore,1,0.5350953449295449,0.1261925950852554,6.4467364012557615,2.04778550672059,98,13347.85,13309.03068971634,99999.0
+++++esfm,Sri Thendayuthapani Singapore,2,0.5262093764563488,0.14374074415280041,7.476614790858722,2.2575164196527426,98,13304.56,13262.61834168434,99999.0
++++ esfm,Sri Veeramakaliamman Singapore,0,2.333869921178247,0.542847254574005,11.965270419451508,5.472877662905037,157,12392.18,12365.15180635452,99999.0
++++-uesfm,Sri Veeramakaliamman Singapore,0,29.40487246624615,3.8676026992817505,28.950500021302087,14.32295224594287,157,12544.69,12504.26410722733,99999.0
+++++esfm,Sri Veeramakaliamman Singapore,1,2.4804161216418734,0.5731176371311499,12.150997217479388,5.5309200508775875,157,12387.11,12363.29210400581,99999.0
+++++esfm,Sri Veeramakaliamman Singapore,2,2.3974569344813434,0.5599333960859275,12.58650507397198,5.711763361742226,157,12354.49,12330.03034973145,99999.0
++++ esfm,Statue Of Liberty,0,52.35559762033152,28.241594953600423,688383.1603142131,683624.423417858,134,102.64,43.32754778862,45000.0
++++ esfm,Statue Of Liberty,1,52.36920727661949,28.627801879821945,9277.715116667516,9303.546387753544,134,99.78,59.98358988761902,65000.0
++++ esfm,Statue Of Liberty,2,52.3478568435799,28.64052858422677,7449.598084265948,7416.803834848509,134,97.28,3.520912647247314,0.0
++++-uesfm,Statue Of Liberty,0,40.92379801166407,19.915794833689585,48.781849500680366,25.904882064330994,134,12927.11,12902.23110127449,99999.0
++++-uesfm,Statue Of Liberty,1,38.79484508812604,17.090491683338875,33.27459817133403,17.608726936833005,134,12920.89,12896.32496738434,99999.0
++++ esfm,The Pumpkin,0,27.59459354015253,5.94835892276036,27.52772578222832,9.11126349703041,196,4330.9,4316.109169960022,99999.0
++++ esfm,The Pumpkin,1,26.54397520858153,5.701234509093555,27.342758332715125,8.907676505669826,196,4327.98,4314.043692827225,99999.0
++++ esfm,The Pumpkin,2,23.05395646123413,5.007378502393024,25.75879879971585,7.898438464943201,196,4322.94,4310.356798410416,99999.0
++++-uesfm,The Pumpkin,0,25.490608822831348,5.440568114393512,34.198429037653845,15.6024365790899,196,5793.03,5767.982522964478,99999.0
++++-uesfm,The Pumpkin,1,27.26556900533122,5.820572144056583,35.6951097814524,16.572200360468877,196,5785.61,5760.521738529205,99999.0
++++-uesfm,The Pumpkin,2,26.95722616022698,5.73349875876088,36.207441893083136,17.07678356860021,196,5793.15,5768.243654251099,99999.0
++++ esfm,Thian Hook Keng Temple Singapore,0,1.0653505173922067,0.1079565219850111,15.054152885837562,5.81055046268767,138,3299.26,3281.187238931656,99999.0
++++ esfm,Thian Hook Keng Temple Singapore,1,0.8390930608109779,0.08272919436528102,14.587041005106803,5.705702663819475,138,3306.93,3283.257929086685,99999.0
++++ esfm,Thian Hook Keng Temple Singapore,2,0.8760513596629002,0.08764395630143504,15.102705362055175,5.7567965038255124,138,3289.42,3278.173619031906,99999.0
++++-uesfm,Thian Hook Keng Temple Singapore,0,1.146131162829257,0.10799243740327723,20.24888615229727,10.297158058757331,138,4626.48,4595.79723072052,99999.0
++++-uesfm,Thian Hook Keng Temple Singapore,1,1.2072956626750737,0.12821580405567637,21.242388560332195,10.655698470690366,138,4600.34,4578.699931621552,99999.0
++++-uesfm,Thian Hook Keng Temple Singapore,2,1.251054659622132,0.12561972972485672,20.875018460120653,10.416875106878496,138,4610.3,4581.792747020721,99999.0
++++ esfm,Tsar Nikolai I,0,42.507553768489565,8.869914103947062,10.846690028504929,4.007961025902478,98,4932.15,4920.938139438629,99999.0
++++ esfm,Tsar Nikolai I,1,43.09835370739786,8.983133733509343,11.402130572143294,4.032588092580996,98,4924.71,4914.929541349411,99999.0
++++-uesfm,Tsar Nikolai I,0,75.94880014543322,15.146573953653359,15.307363921859029,6.734065570805788,98,6652.35,6629.384099006653,99999.0
++++-uesfm,Tsar Nikolai I,1,80.87807139992418,15.64102228288136,21.553650044939403,9.179513384124702,98,6641.34,6623.797205209732,99999.0
+++++esfm,Tsar Nikolai I,2,40.892180545819315,8.342265568650971,11.381967217434166,4.272306098513181,98,4930.74,4920.495186090469,99999.0
++++ esfm,Urban II,0,59.0297968076411,11.008489348760156,17.600594623525765,5.400933781867582,96,8889.69,8878.01452589035,99999.0
+++++esfm,Urban II,1,58.947927470837534,11.025374954076108,17.465515658589908,5.68378594242137,96,8885.17,8876.618469953537,99999.0
+++++esfm,Urban II,2,58.77798771298748,10.982786678935327,17.546556385759054,5.767405629316944,96,8893.99,8885.34243106842,99999.0
++++ esfm,Vercingetorix,0,79.07478057258302,9.776081681696928,6.649989325889541,1.7432748253588333,69,1223.75,1212.340968608856,99999.0
++++ esfm,Vercingetorix,1,80.09313371434186,9.858626097407651,6.426263444651325,1.6979463697629655,69,1222.8,1213.763489961624,99999.0
++++ esfm,Vercingetorix,2,89.68787080902364,10.950615257181747,8.695412468830675,3.360645595149667,69,1225.36,1215.808262825012,99999.0
++++-uesfm,Vercingetorix,0,78.95942988866727,10.356244187384963,8.07065603849916,3.372297052368834,69,3105.42,3085.208213567734,99999.0
++++-uesfm,Vercingetorix,1,89.99175121301754,11.178484677344343,8.929255781006072,4.302365177465691,69,3105.88,3091.564247369766,99999.0
++++-uesfm,Vercingetorix,2,89.31151857112468,11.134780185118505,9.044030033429575,4.351952473195928,69,3105.52,3089.0203332901,99999.0
++++ esfm,Yueh Hai Ching Temple Singapore,0,0.6268598783684821,0.08360284283022794,5.56223941139456,2.132543748253723,43,1749.28,1738.747783660889,99999.0
+++++esfm,Yueh Hai Ching Temple Singapore,1,0.5064956072004329,0.07235771132992698,5.30294026596974,2.06113770612239,43,1750.4,1739.704204082489,99999.0
+++++esfm,Yueh Hai Ching Temple Singapore,2,0.49998151063409024,0.07009863154178562,5.92393490371475,2.2317763685579015,43,1746.32,1738.905562877655,99999.0
++++diff --git a/code/results/single_scene/summary_table.md b/code/results/single_scene/summary_table.md
++++index bd3f673..f23bbb9 100644
++++--- a/code/results/single_scene/summary_table.md
+++++++ b/code/results/single_scene/summary_table.md
++++@@ -2,44 +2,45 @@
++++ 
++++ Mean over seeds per (scene, method); best value per scene in **bold** (lower is better except Nr).
++++ 
++++-| Scene | ESFM Rot (deg) | U-ESFM Rot (deg) | ESFM (paper) Rot (deg) | ESFM Trans | U-ESFM Trans | ESFM (paper) Trans | ESFM Reproj (px) | U-ESFM Reproj (px) | ESFM (paper) Reproj (px) | ESFM Nr | U-ESFM Nr | ESFM (paper) Nr | ESFM Time (s) | U-ESFM Time (s) | ESFM (paper) Time (s) | ESFM Rot-BA (deg) | U-ESFM Rot-BA (deg) | ESFM (paper) Rot-BA (deg) | ESFM Trans-BA | U-ESFM Trans-BA | ESFM (paper) Trans-BA | ESFM Reproj-BA (px) | U-ESFM Reproj-BA (px) | ESFM (paper) Reproj-BA (px) |
+++++| Scene | ESFM (official code) Rot (deg) | U-ESFM Rot (deg) | ESFM (paper) Rot (deg) | ESFM (official code) Trans | U-ESFM Trans | ESFM (paper) Trans | ESFM (official code) Reproj (px) | U-ESFM Reproj (px) | ESFM (paper) Reproj (px) | ESFM (official code) Nr | U-ESFM Nr | ESFM (paper) Nr | ESFM (official code) Time (s) | U-ESFM Time (s) | ESFM (paper) Time (s) | ESFM (official code) Rot-BA (deg) | U-ESFM Rot-BA (deg) | ESFM (paper) Rot-BA (deg) | ESFM (official code) Trans-BA | U-ESFM Trans-BA | ESFM (paper) Trans-BA | ESFM (official code) Reproj-BA (px) | U-ESFM Reproj-BA (px) | ESFM (paper) Reproj-BA (px) |
++++ |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
++++-| Alcatraz Courtyard | **0.362** | 0.616 | 0.619 | **0.093** | 0.161 | 0.160 | **3.435** | 5.796 | 1.640 | **133** | 133 | -- | **4826.713** | 6481.757 | -- | -- | -- | 0.049 | -- | -- | 0.015 | -- | -- | 0.810 |
++++-| Alcatraz Water Tower | **0.666** | 7.413 | 0.933 | **0.370** | 3.437 | 0.518 | **4.763** | 8.607 | 2.130 | **172** | 172 | -- | **2461.337** | 3854.907 | -- | -- | -- | 0.230 | -- | -- | 0.116 | -- | -- | 0.550 |
++++-| Buddah Tooth Relic Temple Singapore | **1.131** | 12.080 | 1.030 | **0.280** | 2.308 | 0.233 | **9.964** | 19.634 | 2.060 | **162** | 162 | -- | **2916.837** | 4320.873 | -- | -- | -- | 0.081 | -- | -- | 0.014 | -- | -- | 0.850 |
++++-| Doge Palace Venice | **0.801** | 0.911 | 1.163 | **0.218** | 0.290 | 0.342 | **6.432** | 10.966 | 3.620 | **241** | 241 | -- | 28768.725 | **21074.950** | -- | -- | -- | 0.211 | -- | -- | 0.029 | -- | -- | 1 |
++++-| Door Lund | **0.014** | 14.172 | 0.024 | **0.004** | 0.952 | 0.006 | **1.812** | 11.922 | 0.320 | **12** | 12 | -- | **4888.363** | 9188.593 | -- | -- | -- | 0.006 | -- | -- | 0.001 | -- | -- | 0.300 |
++++-| Drinking Fountain Somewhere In Zurich | **17.193** | 17.536 | 0.031 | 0.753 | **0.740** | 0.004 | **6.058** | 6.511 | 0.330 | **14** | 14 | -- | **883.837** | 2963.647 | -- | -- | -- | 0.007 | -- | -- | 0.002 | -- | -- | 0.310 |
++++-| East Indiaman Goteborg | 5.568 | **5.310** | 3.814 | 0.979 | **0.959** | 0.621 | **7.318** | 9.840 | 4.130 | **179** | 179 | -- | **3987.183** | 5241.430 | -- | -- | -- | 3.117 | -- | -- | 0.509 | -- | -- | 1.850 |
++++-| Ecole Superior De Guerre | 0.349 | **0.284** | 0.318 | 0.090 | **0.077** | 0.081 | **3.418** | 5.349 | 0.720 | **35** | 35 | -- | **2380.387** | 4633.833 | -- | -- | -- | 0.024 | -- | -- | 0.005 | -- | -- | 0.340 |
++++-| Eglise du dome | **0.351** | 45.801 | 0.808 | **0.088** | 6.451 | 0.205 | **3.799** | 7.964 | 0.910 | **85** | 85 | -- | 25099.450 | **23371.730** | -- | -- | -- | 0.037 | -- | -- | 0.010 | -- | -- | 0.270 |
++++-| Folke Filbyter | 85.831 | **81.288** | 74.596 | **0.129** | 0.130 | 0.125 | **19.275** | 23.156 | 10.370 | **40** | 40 | -- | **1581.897** | 3703.187 | -- | -- | -- | 70.157 | -- | -- | 0.118 | -- | -- | 4.290 |
++++-| Fort Channing Gate Singapore | **0.093** | 11.128 | 0.207 | **0.041** | 4.004 | 0.093 | **1.783** | 6.651 | 0.520 | **27** | 27 | -- | **3434.113** | 6172.087 | -- | -- | -- | 0.020 | -- | -- | 0.008 | -- | -- | 0.250 |
++++-| Golden Statue Somewhere In Hong Kong | **0.232** | 0.371 | 0.292 | **0.053** | 0.074 | 0.073 | **1.829** | 2.835 | 0.400 | **18** | 18 | -- | **4095.853** | 7762.437 | -- | -- | -- | 0.031 | -- | -- | 0.004 | -- | -- | 0.270 |
++++-| Gustav Vasa | **3.509** | 6.099 | 34.181 | **0.220** | 0.345 | 1.085 | **1.655** | 3.087 | 3.520 | **18** | 18 | -- | **709.867** | 2543.553 | -- | -- | -- | 32.266 | -- | -- | 1.145 | -- | -- | 3.150 |
++++-| GustavIIAdolf | **47.126** | 77.423 | 67.784 | **8.536** | 11.962 | 9.714 | **11.722** | 15.851 | 13.910 | **57** | 57 | -- | **1044.170** | 2892.580 | -- | -- | -- | 58.458 | -- | -- | 8.524 | -- | -- | 11.490 |
++++-| Jonas Ahlstromer | 48.740 | **42.246** | 50.190 | **9.823** | 10.300 | 10.888 | 11.136 | **8.782** | 10.820 | **40** | 40 | -- | **615.450** | 2328.313 | -- | -- | -- | 47.117 | -- | -- | 10.451 | -- | -- | 8.410 |
++++-| Kings College University Of Toronto | **9.025** | 21.523 | 0.989 | 2.044 | **1.905** | 0.235 | **3.565** | 5.525 | 0.900 | **77** | 77 | -- | **1078.980** | 2886.243 | -- | -- | -- | 0.085 | -- | -- | 0.017 | -- | -- | 0.340 |
++++-| Lund University Sphinx | **11.392** | 33.023 | 19.522 | **2.981** | 6.889 | 4.585 | **6.056** | 12.060 | 4.780 | **70** | 70 | -- | **2718.567** | 4654.610 | -- | -- | -- | 8.752 | -- | -- | 2.191 | -- | -- | 1.360 |
++++-| Nijo Castle Gate | **0.839** | 1.682 | 1.495 | **0.165** | 0.326 | 0.286 | **7.415** | 12.138 | 1.700 | **19** | 19 | -- | **1095.177** | 3306.697 | -- | -- | -- | 0.069 | -- | -- | 0.012 | -- | -- | 0.730 |
++++-| Pantheon Paris | -- | 19.345 | 0.192 | -- | 3.065 | 0.050 | -- | 13.562 | 1.470 | -- | 179 | -- | -- | 5497.377 | -- | -- | -- | 0.040 | -- | -- | 0.005 | -- | -- | 0.490 |
++++-| Park Gate Clermont Ferrand | 21.388 | **19.302** | 0.391 | 10.192 | **9.397** | 0.125 | **7.712** | 7.852 | 0.570 | **34** | 34 | -- | **1596.940** | 3597.403 | -- | -- | -- | 0.049 | -- | -- | 0.022 | -- | -- | 0.350 |
++++-| Plaza De Armas Santiago | **0.775** | 3.327 | 6.782 | **0.337** | 1.435 | 2.944 | **6.009** | 9.463 | 7.400 | **240** | 240 | -- | **6372.683** | 6535.537 | -- | -- | -- | 2.556 | -- | -- | 1.383 | -- | -- | 4.900 |
++++-| Porta San Donato Bologna | **0.592** | 21.006 | 2.153 | **0.107** | 3.541 | 0.388 | **5.603** | 13.244 | 2.280 | **141** | 141 | -- | **3754.163** | 5118.030 | -- | -- | -- | 0.095 | -- | -- | 0.046 | -- | -- | 0.750 |
++++-| Round Church Cambridge | **2.129** | 2.490 | 2.451 | **0.920** | 1.002 | 1.003 | **6.097** | 8.546 | 2.660 | **92** | 92 | -- | **9776.630** | 12061.350 | -- | -- | -- | 1.107 | -- | -- | 0.582 | -- | -- | 1.540 |
++++-| Skansen Kronan Gothenburg | **0.338** | 0.375 | 0.736 | **0.116** | 0.124 | 0.226 | **3.048** | 4.799 | 1.240 | **131** | 131 | -- | 17163.940 | **16235.090** | -- | -- | -- | 0.026 | -- | -- | 0.008 | -- | -- | 0.670 |
++++-| Smolny Cathedral St Petersburg | **21.806** | 22.143 | 0.554 | 2.111 | **2.099** | 0.051 | **11.228** | 12.526 | 1.660 | **131** | 131 | -- | **10362.595** | 10475.740 | -- | -- | -- | 0.033 | -- | -- | 0.006 | -- | -- | 0.810 |
++++-| Sri Mariamman Singapore | **1.219** | 6.479 | 2.302 | **0.382** | 2.188 | 0.683 | **10.088** | 17.854 | 4.130 | **222** | 222 | -- | **5985.540** | 6485.667 | -- | -- | -- | 0.077 | -- | -- | 0.023 | -- | -- | 0.910 |
++++-| Sri Thendayuthapani Singapore | **0.719** | 47.075 | 46.269 | **0.179** | 3.855 | 3.812 | **7.124** | 14.133 | 23.370 | **98** | 98 | -- | **13362.090** | 13808.440 | -- | -- | -- | 44.170 | -- | -- | 2.870 | -- | -- | 8.440 |
++++-| Sri Veeramakaliamman Singapore | **2.334** | 29.405 | 2.559 | **0.543** | 3.868 | 0.597 | **11.965** | 28.951 | 3.470 | **157** | 157 | -- | **12392.180** | 12544.690 | -- | -- | -- | 0.175 | -- | -- | 0.040 | -- | -- | 0.730 |
++++-| Statue Of Liberty | 52.358 | **39.859** | 46.887 | 28.503 | **18.503** | 20.012 | 235036.825 | **41.028** | 26.160 | **134** | 134 | -- | **99.900** | 12924 | -- | -- | -- | 9.091 | -- | -- | 4.122 | -- | -- | 6.970 |
++++-| The Pumpkin | **25.731** | 26.571 | 94.672 | **5.552** | 5.665 | 14.890 | **26.876** | 35.367 | 33.410 | **196** | 196 | -- | **4327.273** | 5790.597 | -- | -- | -- | 98.862 | -- | -- | 14.952 | -- | -- | 24.850 |
++++-| Thian Hook Keng Temple Singapore | **0.927** | 1.201 | 0.832 | **0.093** | 0.121 | 0.082 | **14.915** | 20.789 | 2.750 | **138** | 138 | -- | **3298.537** | 4612.373 | -- | -- | -- | 0.081 | -- | -- | 0.008 | -- | -- | 1.130 |
++++-| Tsar Nikolai I | **42.803** | 78.413 | 48.499 | **8.927** | 15.394 | 9.467 | **11.124** | 18.431 | 9.790 | **98** | 98 | -- | **4928.430** | 6646.845 | -- | -- | -- | 36.280 | -- | -- | 7.836 | -- | -- | 6.530 |
++++-| Urban II | 59.030 | -- | 47.490 | 11.008 | -- | 9.467 | 17.601 | -- | 9.380 | 96 | -- | -- | 8889.690 | -- | -- | -- | -- | 48.214 | -- | -- | 9.586 | -- | -- | 6.920 |
++++-| Vercingetorix | **82.952** | 86.088 | 69.328 | **10.195** | 10.890 | 8.788 | **7.257** | 8.681 | 5.080 | **69** | 69 | -- | **1223.970** | 3105.607 | -- | -- | -- | 17.706 | -- | -- | 3.104 | -- | -- | 1.500 |
++++-| Yueh Hai Ching Temple Singapore | 0.627 | -- | 0.720 | 0.084 | -- | 0.098 | 5.562 | -- | 0.940 | 43 | -- | -- | 1749.280 | -- | -- | -- | -- | 0.043 | -- | -- | 0.014 | -- | -- | 0.650 |
++++-| **Mean** | **16.146** | 23.697 | 18.023 | **3.121** | 4.014 | 2.912 | 6920.602 | **13.088** | 5.673 | 100.559 | **104.818** | -- | **5819.728** | 7358.187 | -- | -- | -- | 13.695 | -- | -- | 1.937 | -- | -- | 2.992 |
+++++| Alcatraz Courtyard | **0.362** | 10.132 | 0.619 | **0.093** | 2.219 | 0.160 | **3.435** | 7.272 | 1.640 | **133** | 133 | -- | **4826.713** | 6550.983 | -- | -- | -- | 0.049 | -- | -- | 0.015 | -- | -- | 0.810 |
+++++| Alcatraz Water Tower | **0.666** | 1.480 | 0.933 | **0.370** | 0.803 | 0.518 | **4.763** | 7.397 | 2.130 | **172** | 172 | -- | **2461.337** | 3795.877 | -- | -- | -- | 0.230 | -- | -- | 0.116 | -- | -- | 0.550 |
+++++| Buddah Tooth Relic Temple Singapore | **1.131** | 1.791 | 1.030 | **0.280** | 0.520 | 0.233 | **9.964** | 14.029 | 2.060 | **162** | 162 | -- | **2916.837** | 4289.503 | -- | -- | -- | 0.081 | -- | -- | 0.014 | -- | -- | 0.850 |
+++++| Doge Palace Venice | **0.753** | 1.121 | 1.163 | **0.209** | 0.350 | 0.342 | **6.359** | 10.496 | 3.620 | **241** | 241 | -- | 22522.737 | **20767.260** | -- | -- | -- | 0.211 | -- | -- | 0.029 | -- | -- | 1 |
+++++| Door Lund | **0.014** | 6.960 | 0.024 | **0.004** | 0.512 | 0.006 | **1.812** | 7.101 | 0.320 | **12** | 12 | -- | **4888.363** | 9252.735 | -- | -- | -- | 0.006 | -- | -- | 0.001 | -- | -- | 0.300 |
+++++| Drinking Fountain Somewhere In Zurich | **17.193** | 17.329 | 0.031 | 0.753 | **0.750** | 0.004 | **6.058** | 6.675 | 0.330 | **14** | 14 | -- | **883.837** | 5815.220 | -- | -- | -- | 0.007 | -- | -- | 0.002 | -- | -- | 0.310 |
+++++| East Indiaman Goteborg | **5.568** | 7.231 | 3.814 | **0.979** | 1.364 | 0.621 | **7.318** | 10.130 | 4.130 | **179** | 179 | -- | **3987.183** | 5198.093 | -- | -- | -- | 3.117 | -- | -- | 0.509 | -- | -- | 1.850 |
+++++| Ecole Superior De Guerre | **0.349** | 25.800 | 0.318 | **0.090** | 1.324 | 0.081 | **3.418** | 8.646 | 0.720 | **35** | 35 | -- | **2380.387** | 10416.080 | -- | -- | -- | 0.024 | -- | -- | 0.005 | -- | -- | 0.340 |
+++++| Eglise du dome | **0.386** | 0.773 | 0.808 | **0.098** | 0.213 | 0.205 | **3.987** | 6.018 | 0.910 | **85** | 85 | -- | 19631.293 | **9796.445** | -- | -- | -- | 0.037 | -- | -- | 0.010 | -- | -- | 0.270 |
+++++| Folke Filbyter | 85.831 | **82.656** | 74.596 | **0.129** | 0.131 | 0.125 | **19.275** | 28.225 | 10.370 | **40** | 40 | -- | **1581.897** | 3755.193 | -- | -- | -- | 70.157 | -- | -- | 0.118 | -- | -- | 4.290 |
+++++| Fort Channing Gate Singapore | **0.093** | 0.122 | 0.207 | **0.041** | 0.056 | 0.093 | **1.783** | 3.649 | 0.520 | **27** | 27 | -- | **3434.113** | 6183.925 | -- | -- | -- | 0.020 | -- | -- | 0.008 | -- | -- | 0.250 |
+++++| Golden Statue Somewhere In Hong Kong | 0.232 | -- | 0.292 | 0.053 | -- | 0.073 | 1.829 | -- | 0.400 | 18 | -- | -- | 4095.853 | -- | -- | -- | -- | 0.031 | -- | -- | 0.004 | -- | -- | 0.270 |
+++++| Gustav Vasa | **3.509** | 25.854 | 34.181 | **0.220** | 0.775 | 1.085 | **1.655** | 4.747 | 3.520 | **18** | 18 | -- | **709.867** | 3887.653 | -- | -- | -- | 32.266 | -- | -- | 1.145 | -- | -- | 3.150 |
+++++| GustavIIAdolf | **47.126** | 88.194 | 67.784 | **8.536** | 13.129 | 9.714 | **11.722** | 17.220 | 13.910 | **57** | 57 | -- | **1044.170** | 2981.017 | -- | -- | -- | 58.458 | -- | -- | 8.524 | -- | -- | 11.490 |
+++++| Jonas Ahlstromer | 48.740 | **46.584** | 50.190 | **9.823** | 10.245 | 10.888 | **11.136** | 11.144 | 10.820 | **40** | 40 | -- | **615.450** | 2414.653 | -- | -- | -- | 47.117 | -- | -- | 10.451 | -- | -- | 8.410 |
+++++| Kings College University Of Toronto | **9.025** | 14.284 | 0.989 | 2.044 | **1.639** | 0.235 | **3.565** | 6.351 | 0.900 | **77** | 77 | -- | **1078.980** | 2857.733 | -- | -- | -- | 0.085 | -- | -- | 0.017 | -- | -- | 0.340 |
+++++| Lund University Sphinx | **11.392** | 42.959 | 19.522 | **2.981** | 8.720 | 4.585 | **6.056** | 15.000 | 4.780 | **70** | 70 | -- | **2718.567** | 4705.610 | -- | -- | -- | 8.752 | -- | -- | 2.191 | -- | -- | 1.360 |
+++++| Nijo Castle Gate | **0.839** | 8.641 | 1.495 | **0.165** | 2.791 | 0.286 | **7.415** | 20.967 | 1.700 | **19** | 19 | -- | **1095.177** | 3241.330 | -- | -- | -- | 0.069 | -- | -- | 0.012 | -- | -- | 0.730 |
+++++| Pantheon Paris | -- | 0.509 | 0.192 | -- | 0.066 | 0.050 | -- | 7.493 | 1.470 | -- | 179 | -- | -- | 5565.670 | -- | -- | -- | 0.040 | -- | -- | 0.005 | -- | -- | 0.490 |
+++++| Park Gate Clermont Ferrand | **21.388** | 26.451 | 0.391 | **10.192** | 12.467 | 0.125 | **7.712** | 9.374 | 0.570 | **34** | 34 | -- | **1596.940** | 3659.900 | -- | -- | -- | 0.049 | -- | -- | 0.022 | -- | -- | 0.350 |
+++++| Plaza De Armas Santiago | 0.775 | -- | 6.782 | 0.337 | -- | 2.944 | 6.009 | -- | 7.400 | 240 | -- | -- | 6372.683 | -- | -- | -- | -- | 2.556 | -- | -- | 1.383 | -- | -- | 4.900 |
+++++| Porta San Donato Bologna | 0.592 | -- | 2.153 | 0.107 | -- | 0.388 | 5.603 | -- | 2.280 | 141 | -- | -- | 3754.163 | -- | -- | -- | -- | 0.095 | -- | -- | 0.046 | -- | -- | 0.750 |
+++++| Round Church Cambridge | 2.137 | -- | 2.451 | 0.927 | -- | 1.003 | 6.103 | -- | 2.660 | 92 | -- | -- | 9779.270 | -- | -- | -- | -- | 1.107 | -- | -- | 0.582 | -- | -- | 1.540 |
+++++| Skansen Kronan Gothenburg | 0.301 | -- | 0.736 | 0.102 | -- | 0.226 | 2.906 | -- | 1.240 | 131 | -- | -- | 17153.677 | -- | -- | -- | -- | 0.026 | -- | -- | 0.008 | -- | -- | 0.670 |
+++++| Smolny Cathedral St Petersburg | 21.775 | -- | 0.554 | 2.111 | -- | 0.051 | 11.239 | -- | 1.660 | 131 | -- | -- | 10362.787 | -- | -- | -- | -- | 0.033 | -- | -- | 0.006 | -- | -- | 0.810 |
+++++| Some Cathedral In Barcelona | 0.651 | -- | 0.880 | 0.233 | -- | 0.315 | 6.271 | -- | 2.870 | 177 | -- | -- | 5001.973 | -- | -- | -- | -- | 0.026 | -- | -- | 0.011 | -- | -- | 0.890 |
+++++| Sri Mariamman Singapore | 1.219 | -- | 2.302 | 0.382 | -- | 0.683 | 10.088 | -- | 4.130 | 222 | -- | -- | 5985.540 | -- | -- | -- | -- | 0.077 | -- | -- | 0.023 | -- | -- | 0.910 |
+++++| Sri Thendayuthapani Singapore | 0.593 | -- | 46.269 | 0.150 | -- | 3.812 | 7.016 | -- | 23.370 | 98 | -- | -- | 13338.167 | -- | -- | -- | -- | 44.170 | -- | -- | 2.870 | -- | -- | 8.440 |
+++++| Sri Veeramakaliamman Singapore | 2.404 | -- | 2.559 | 0.559 | -- | 0.597 | 12.234 | -- | 3.470 | 157 | -- | -- | 12377.927 | -- | -- | -- | -- | 0.175 | -- | -- | 0.040 | -- | -- | 0.730 |
+++++| Statue Of Liberty | 52.358 | -- | 46.887 | 28.503 | -- | 20.012 | 235036.825 | -- | 26.160 | 134 | -- | -- | 99.900 | -- | -- | -- | -- | 9.091 | -- | -- | 4.122 | -- | -- | 6.970 |
+++++| The Pumpkin | 25.731 | -- | 94.672 | 5.552 | -- | 14.890 | 26.876 | -- | 33.410 | 196 | -- | -- | 4327.273 | -- | -- | -- | -- | 98.862 | -- | -- | 14.952 | -- | -- | 24.850 |
+++++| Thian Hook Keng Temple Singapore | 0.927 | -- | 0.832 | 0.093 | -- | 0.082 | 14.915 | -- | 2.750 | 138 | -- | -- | 3298.537 | -- | -- | -- | -- | 0.081 | -- | -- | 0.008 | -- | -- | 1.130 |
+++++| Tsar Nikolai I | 42.166 | -- | 48.499 | 8.732 | -- | 9.467 | 11.210 | -- | 9.790 | 98 | -- | -- | 4929.200 | -- | -- | -- | -- | 36.280 | -- | -- | 7.836 | -- | -- | 6.530 |
+++++| Urban II | 58.919 | -- | 47.490 | 11.006 | -- | 9.467 | 17.538 | -- | 9.380 | 96 | -- | -- | 8889.617 | -- | -- | -- | -- | 48.214 | -- | -- | 9.586 | -- | -- | 6.920 |
+++++| Vercingetorix | 82.952 | -- | 69.328 | 10.195 | -- | 8.788 | 7.257 | -- | 5.080 | 69 | -- | -- | 1223.970 | -- | -- | -- | -- | 17.706 | -- | -- | 3.104 | -- | -- | 1.500 |
+++++| Yueh Hai Ching Temple Singapore | 0.544 | -- | 0.720 | 0.075 | -- | 0.098 | 5.596 | -- | 0.940 | 43 | -- | -- | 1748.667 | -- | -- | -- | -- | 0.043 | -- | -- | 0.014 | -- | -- | 0.650 |
+++++| **Mean** | **15.676** | 21.520 | 17.547 | **3.032** | 3.056 | 2.840 | 6723.056 | **10.628** | 5.595 | **102.743** | 83.895 | -- | **5460.373** | 6059.731 | -- | -- | -- | 13.315 | -- | -- | 1.883 | -- | -- | 2.933 |
++++ 
++++ _Seeds per cell: [0, 1, 2]._
++++ 
++++diff --git a/code/results/single_scene/summary_table.tex b/code/results/single_scene/summary_table.tex
++++index 258e3f2..42f87b3 100644
++++--- a/code/results/single_scene/summary_table.tex
+++++++ b/code/results/single_scene/summary_table.tex
++++@@ -7,45 +7,46 @@
++++ \begin{tabular}{lcccccccccccccccccccccccc}
++++ \toprule
++++  & \multicolumn{3}{c}{Rot (deg)} & \multicolumn{3}{c}{Trans} & \multicolumn{3}{c}{Reproj (px)} & \multicolumn{3}{c}{Nr} & \multicolumn{3}{c}{Time (s)} & \multicolumn{3}{c}{Rot-BA (deg)} & \multicolumn{3}{c}{Trans-BA} & \multicolumn{3}{c}{Reproj-BA (px)} \\
++++-Scene & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) \\
+++++Scene & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) \\
++++ \midrule
++++-Alcatraz Courtyard & \textbf{0.362} & 0.616 & 0.619 & \textbf{0.093} & 0.161 & 0.160 & \textbf{3.435} & 5.796 & 1.640 & \textbf{133} & 133 & -- & \textbf{4826.713} & 6481.757 & -- & -- & -- & 0.049 & -- & -- & 0.015 & -- & -- & 0.810 \\
++++-Alcatraz Water Tower & \textbf{0.666} & 7.413 & 0.933 & \textbf{0.370} & 3.437 & 0.518 & \textbf{4.763} & 8.607 & 2.130 & \textbf{172} & 172 & -- & \textbf{2461.337} & 3854.907 & -- & -- & -- & 0.230 & -- & -- & 0.116 & -- & -- & 0.550 \\
++++-Buddah Tooth Relic Temple Singapore & \textbf{1.131} & 12.080 & 1.030 & \textbf{0.280} & 2.308 & 0.233 & \textbf{9.964} & 19.634 & 2.060 & \textbf{162} & 162 & -- & \textbf{2916.837} & 4320.873 & -- & -- & -- & 0.081 & -- & -- & 0.014 & -- & -- & 0.850 \\
++++-Doge Palace Venice & \textbf{0.801} & 0.911 & 1.163 & \textbf{0.218} & 0.290 & 0.342 & \textbf{6.432} & 10.966 & 3.620 & \textbf{241} & 241 & -- & 28768.725 & \textbf{21074.950} & -- & -- & -- & 0.211 & -- & -- & 0.029 & -- & -- & 1 \\
++++-Door Lund & \textbf{0.014} & 14.172 & 0.024 & \textbf{0.004} & 0.952 & 0.006 & \textbf{1.812} & 11.922 & 0.320 & \textbf{12} & 12 & -- & \textbf{4888.363} & 9188.593 & -- & -- & -- & 0.006 & -- & -- & 0.001 & -- & -- & 0.300 \\
++++-Drinking Fountain Somewhere In Zurich & \textbf{17.193} & 17.536 & 0.031 & 0.753 & \textbf{0.740} & 0.004 & \textbf{6.058} & 6.511 & 0.330 & \textbf{14} & 14 & -- & \textbf{883.837} & 2963.647 & -- & -- & -- & 0.007 & -- & -- & 0.002 & -- & -- & 0.310 \\
++++-East Indiaman Goteborg & 5.568 & \textbf{5.310} & 3.814 & 0.979 & \textbf{0.959} & 0.621 & \textbf{7.318} & 9.840 & 4.130 & \textbf{179} & 179 & -- & \textbf{3987.183} & 5241.430 & -- & -- & -- & 3.117 & -- & -- & 0.509 & -- & -- & 1.850 \\
++++-Ecole Superior De Guerre & 0.349 & \textbf{0.284} & 0.318 & 0.090 & \textbf{0.077} & 0.081 & \textbf{3.418} & 5.349 & 0.720 & \textbf{35} & 35 & -- & \textbf{2380.387} & 4633.833 & -- & -- & -- & 0.024 & -- & -- & 0.005 & -- & -- & 0.340 \\
++++-Eglise du dome & \textbf{0.351} & 45.801 & 0.808 & \textbf{0.088} & 6.451 & 0.205 & \textbf{3.799} & 7.964 & 0.910 & \textbf{85} & 85 & -- & 25099.450 & \textbf{23371.730} & -- & -- & -- & 0.037 & -- & -- & 0.010 & -- & -- & 0.270 \\
++++-Folke Filbyter & 85.831 & \textbf{81.288} & 74.596 & \textbf{0.129} & 0.130 & 0.125 & \textbf{19.275} & 23.156 & 10.370 & \textbf{40} & 40 & -- & \textbf{1581.897} & 3703.187 & -- & -- & -- & 70.157 & -- & -- & 0.118 & -- & -- & 4.290 \\
++++-Fort Channing Gate Singapore & \textbf{0.093} & 11.128 & 0.207 & \textbf{0.041} & 4.004 & 0.093 & \textbf{1.783} & 6.651 & 0.520 & \textbf{27} & 27 & -- & \textbf{3434.113} & 6172.087 & -- & -- & -- & 0.020 & -- & -- & 0.008 & -- & -- & 0.250 \\
++++-Golden Statue Somewhere In Hong Kong & \textbf{0.232} & 0.371 & 0.292 & \textbf{0.053} & 0.074 & 0.073 & \textbf{1.829} & 2.835 & 0.400 & \textbf{18} & 18 & -- & \textbf{4095.853} & 7762.437 & -- & -- & -- & 0.031 & -- & -- & 0.004 & -- & -- & 0.270 \\
++++-Gustav Vasa & \textbf{3.509} & 6.099 & 34.181 & \textbf{0.220} & 0.345 & 1.085 & \textbf{1.655} & 3.087 & 3.520 & \textbf{18} & 18 & -- & \textbf{709.867} & 2543.553 & -- & -- & -- & 32.266 & -- & -- & 1.145 & -- & -- & 3.150 \\
++++-GustavIIAdolf & \textbf{47.126} & 77.423 & 67.784 & \textbf{8.536} & 11.962 & 9.714 & \textbf{11.722} & 15.851 & 13.910 & \textbf{57} & 57 & -- & \textbf{1044.170} & 2892.580 & -- & -- & -- & 58.458 & -- & -- & 8.524 & -- & -- & 11.490 \\
++++-Jonas Ahlstromer & 48.740 & \textbf{42.246} & 50.190 & \textbf{9.823} & 10.300 & 10.888 & 11.136 & \textbf{8.782} & 10.820 & \textbf{40} & 40 & -- & \textbf{615.450} & 2328.313 & -- & -- & -- & 47.117 & -- & -- & 10.451 & -- & -- & 8.410 \\
++++-Kings College University Of Toronto & \textbf{9.025} & 21.523 & 0.989 & 2.044 & \textbf{1.905} & 0.235 & \textbf{3.565} & 5.525 & 0.900 & \textbf{77} & 77 & -- & \textbf{1078.980} & 2886.243 & -- & -- & -- & 0.085 & -- & -- & 0.017 & -- & -- & 0.340 \\
++++-Lund University Sphinx & \textbf{11.392} & 33.023 & 19.522 & \textbf{2.981} & 6.889 & 4.585 & \textbf{6.056} & 12.060 & 4.780 & \textbf{70} & 70 & -- & \textbf{2718.567} & 4654.610 & -- & -- & -- & 8.752 & -- & -- & 2.191 & -- & -- & 1.360 \\
++++-Nijo Castle Gate & \textbf{0.839} & 1.682 & 1.495 & \textbf{0.165} & 0.326 & 0.286 & \textbf{7.415} & 12.138 & 1.700 & \textbf{19} & 19 & -- & \textbf{1095.177} & 3306.697 & -- & -- & -- & 0.069 & -- & -- & 0.012 & -- & -- & 0.730 \\
++++-Pantheon Paris & -- & 19.345 & 0.192 & -- & 3.065 & 0.050 & -- & 13.562 & 1.470 & -- & 179 & -- & -- & 5497.377 & -- & -- & -- & 0.040 & -- & -- & 0.005 & -- & -- & 0.490 \\
++++-Park Gate Clermont Ferrand & 21.388 & \textbf{19.302} & 0.391 & 10.192 & \textbf{9.397} & 0.125 & \textbf{7.712} & 7.852 & 0.570 & \textbf{34} & 34 & -- & \textbf{1596.940} & 3597.403 & -- & -- & -- & 0.049 & -- & -- & 0.022 & -- & -- & 0.350 \\
++++-Plaza De Armas Santiago & \textbf{0.775} & 3.327 & 6.782 & \textbf{0.337} & 1.435 & 2.944 & \textbf{6.009} & 9.463 & 7.400 & \textbf{240} & 240 & -- & \textbf{6372.683} & 6535.537 & -- & -- & -- & 2.556 & -- & -- & 1.383 & -- & -- & 4.900 \\
++++-Porta San Donato Bologna & \textbf{0.592} & 21.006 & 2.153 & \textbf{0.107} & 3.541 & 0.388 & \textbf{5.603} & 13.244 & 2.280 & \textbf{141} & 141 & -- & \textbf{3754.163} & 5118.030 & -- & -- & -- & 0.095 & -- & -- & 0.046 & -- & -- & 0.750 \\
++++-Round Church Cambridge & \textbf{2.129} & 2.490 & 2.451 & \textbf{0.920} & 1.002 & 1.003 & \textbf{6.097} & 8.546 & 2.660 & \textbf{92} & 92 & -- & \textbf{9776.630} & 12061.350 & -- & -- & -- & 1.107 & -- & -- & 0.582 & -- & -- & 1.540 \\
++++-Skansen Kronan Gothenburg & \textbf{0.338} & 0.375 & 0.736 & \textbf{0.116} & 0.124 & 0.226 & \textbf{3.048} & 4.799 & 1.240 & \textbf{131} & 131 & -- & 17163.940 & \textbf{16235.090} & -- & -- & -- & 0.026 & -- & -- & 0.008 & -- & -- & 0.670 \\
++++-Smolny Cathedral St Petersburg & \textbf{21.806} & 22.143 & 0.554 & 2.111 & \textbf{2.099} & 0.051 & \textbf{11.228} & 12.526 & 1.660 & \textbf{131} & 131 & -- & \textbf{10362.595} & 10475.740 & -- & -- & -- & 0.033 & -- & -- & 0.006 & -- & -- & 0.810 \\
++++-Sri Mariamman Singapore & \textbf{1.219} & 6.479 & 2.302 & \textbf{0.382} & 2.188 & 0.683 & \textbf{10.088} & 17.854 & 4.130 & \textbf{222} & 222 & -- & \textbf{5985.540} & 6485.667 & -- & -- & -- & 0.077 & -- & -- & 0.023 & -- & -- & 0.910 \\
++++-Sri Thendayuthapani Singapore & \textbf{0.719} & 47.075 & 46.269 & \textbf{0.179} & 3.855 & 3.812 & \textbf{7.124} & 14.133 & 23.370 & \textbf{98} & 98 & -- & \textbf{13362.090} & 13808.440 & -- & -- & -- & 44.170 & -- & -- & 2.870 & -- & -- & 8.440 \\
++++-Sri Veeramakaliamman Singapore & \textbf{2.334} & 29.405 & 2.559 & \textbf{0.543} & 3.868 & 0.597 & \textbf{11.965} & 28.951 & 3.470 & \textbf{157} & 157 & -- & \textbf{12392.180} & 12544.690 & -- & -- & -- & 0.175 & -- & -- & 0.040 & -- & -- & 0.730 \\
++++-Statue Of Liberty & 52.358 & \textbf{39.859} & 46.887 & 28.503 & \textbf{18.503} & 20.012 & 235036.825 & \textbf{41.028} & 26.160 & \textbf{134} & 134 & -- & \textbf{99.900} & 12924 & -- & -- & -- & 9.091 & -- & -- & 4.122 & -- & -- & 6.970 \\
++++-The Pumpkin & \textbf{25.731} & 26.571 & 94.672 & \textbf{5.552} & 5.665 & 14.890 & \textbf{26.876} & 35.367 & 33.410 & \textbf{196} & 196 & -- & \textbf{4327.273} & 5790.597 & -- & -- & -- & 98.862 & -- & -- & 14.952 & -- & -- & 24.850 \\
++++-Thian Hook Keng Temple Singapore & \textbf{0.927} & 1.201 & 0.832 & \textbf{0.093} & 0.121 & 0.082 & \textbf{14.915} & 20.789 & 2.750 & \textbf{138} & 138 & -- & \textbf{3298.537} & 4612.373 & -- & -- & -- & 0.081 & -- & -- & 0.008 & -- & -- & 1.130 \\
++++-Tsar Nikolai I & \textbf{42.803} & 78.413 & 48.499 & \textbf{8.927} & 15.394 & 9.467 & \textbf{11.124} & 18.431 & 9.790 & \textbf{98} & 98 & -- & \textbf{4928.430} & 6646.845 & -- & -- & -- & 36.280 & -- & -- & 7.836 & -- & -- & 6.530 \\
++++-Urban II & 59.030 & -- & 47.490 & 11.008 & -- & 9.467 & 17.601 & -- & 9.380 & 96 & -- & -- & 8889.690 & -- & -- & -- & -- & 48.214 & -- & -- & 9.586 & -- & -- & 6.920 \\
++++-Vercingetorix & \textbf{82.952} & 86.088 & 69.328 & \textbf{10.195} & 10.890 & 8.788 & \textbf{7.257} & 8.681 & 5.080 & \textbf{69} & 69 & -- & \textbf{1223.970} & 3105.607 & -- & -- & -- & 17.706 & -- & -- & 3.104 & -- & -- & 1.500 \\
++++-Yueh Hai Ching Temple Singapore & 0.627 & -- & 0.720 & 0.084 & -- & 0.098 & 5.562 & -- & 0.940 & 43 & -- & -- & 1749.280 & -- & -- & -- & -- & 0.043 & -- & -- & 0.014 & -- & -- & 0.650 \\
+++++Alcatraz Courtyard & \textbf{0.362} & 10.132 & 0.619 & \textbf{0.093} & 2.219 & 0.160 & \textbf{3.435} & 7.272 & 1.640 & \textbf{133} & 133 & -- & \textbf{4826.713} & 6550.983 & -- & -- & -- & 0.049 & -- & -- & 0.015 & -- & -- & 0.810 \\
+++++Alcatraz Water Tower & \textbf{0.666} & 1.480 & 0.933 & \textbf{0.370} & 0.803 & 0.518 & \textbf{4.763} & 7.397 & 2.130 & \textbf{172} & 172 & -- & \textbf{2461.337} & 3795.877 & -- & -- & -- & 0.230 & -- & -- & 0.116 & -- & -- & 0.550 \\
+++++Buddah Tooth Relic Temple Singapore & \textbf{1.131} & 1.791 & 1.030 & \textbf{0.280} & 0.520 & 0.233 & \textbf{9.964} & 14.029 & 2.060 & \textbf{162} & 162 & -- & \textbf{2916.837} & 4289.503 & -- & -- & -- & 0.081 & -- & -- & 0.014 & -- & -- & 0.850 \\
+++++Doge Palace Venice & \textbf{0.753} & 1.121 & 1.163 & \textbf{0.209} & 0.350 & 0.342 & \textbf{6.359} & 10.496 & 3.620 & \textbf{241} & 241 & -- & 22522.737 & \textbf{20767.260} & -- & -- & -- & 0.211 & -- & -- & 0.029 & -- & -- & 1 \\
+++++Door Lund & \textbf{0.014} & 6.960 & 0.024 & \textbf{0.004} & 0.512 & 0.006 & \textbf{1.812} & 7.101 & 0.320 & \textbf{12} & 12 & -- & \textbf{4888.363} & 9252.735 & -- & -- & -- & 0.006 & -- & -- & 0.001 & -- & -- & 0.300 \\
+++++Drinking Fountain Somewhere In Zurich & \textbf{17.193} & 17.329 & 0.031 & 0.753 & \textbf{0.750} & 0.004 & \textbf{6.058} & 6.675 & 0.330 & \textbf{14} & 14 & -- & \textbf{883.837} & 5815.220 & -- & -- & -- & 0.007 & -- & -- & 0.002 & -- & -- & 0.310 \\
+++++East Indiaman Goteborg & \textbf{5.568} & 7.231 & 3.814 & \textbf{0.979} & 1.364 & 0.621 & \textbf{7.318} & 10.130 & 4.130 & \textbf{179} & 179 & -- & \textbf{3987.183} & 5198.093 & -- & -- & -- & 3.117 & -- & -- & 0.509 & -- & -- & 1.850 \\
+++++Ecole Superior De Guerre & \textbf{0.349} & 25.800 & 0.318 & \textbf{0.090} & 1.324 & 0.081 & \textbf{3.418} & 8.646 & 0.720 & \textbf{35} & 35 & -- & \textbf{2380.387} & 10416.080 & -- & -- & -- & 0.024 & -- & -- & 0.005 & -- & -- & 0.340 \\
+++++Eglise du dome & \textbf{0.386} & 0.773 & 0.808 & \textbf{0.098} & 0.213 & 0.205 & \textbf{3.987} & 6.018 & 0.910 & \textbf{85} & 85 & -- & 19631.293 & \textbf{9796.445} & -- & -- & -- & 0.037 & -- & -- & 0.010 & -- & -- & 0.270 \\
+++++Folke Filbyter & 85.831 & \textbf{82.656} & 74.596 & \textbf{0.129} & 0.131 & 0.125 & \textbf{19.275} & 28.225 & 10.370 & \textbf{40} & 40 & -- & \textbf{1581.897} & 3755.193 & -- & -- & -- & 70.157 & -- & -- & 0.118 & -- & -- & 4.290 \\
+++++Fort Channing Gate Singapore & \textbf{0.093} & 0.122 & 0.207 & \textbf{0.041} & 0.056 & 0.093 & \textbf{1.783} & 3.649 & 0.520 & \textbf{27} & 27 & -- & \textbf{3434.113} & 6183.925 & -- & -- & -- & 0.020 & -- & -- & 0.008 & -- & -- & 0.250 \\
+++++Golden Statue Somewhere In Hong Kong & 0.232 & -- & 0.292 & 0.053 & -- & 0.073 & 1.829 & -- & 0.400 & 18 & -- & -- & 4095.853 & -- & -- & -- & -- & 0.031 & -- & -- & 0.004 & -- & -- & 0.270 \\
+++++Gustav Vasa & \textbf{3.509} & 25.854 & 34.181 & \textbf{0.220} & 0.775 & 1.085 & \textbf{1.655} & 4.747 & 3.520 & \textbf{18} & 18 & -- & \textbf{709.867} & 3887.653 & -- & -- & -- & 32.266 & -- & -- & 1.145 & -- & -- & 3.150 \\
+++++GustavIIAdolf & \textbf{47.126} & 88.194 & 67.784 & \textbf{8.536} & 13.129 & 9.714 & \textbf{11.722} & 17.220 & 13.910 & \textbf{57} & 57 & -- & \textbf{1044.170} & 2981.017 & -- & -- & -- & 58.458 & -- & -- & 8.524 & -- & -- & 11.490 \\
+++++Jonas Ahlstromer & 48.740 & \textbf{46.584} & 50.190 & \textbf{9.823} & 10.245 & 10.888 & \textbf{11.136} & 11.144 & 10.820 & \textbf{40} & 40 & -- & \textbf{615.450} & 2414.653 & -- & -- & -- & 47.117 & -- & -- & 10.451 & -- & -- & 8.410 \\
+++++Kings College University Of Toronto & \textbf{9.025} & 14.284 & 0.989 & 2.044 & \textbf{1.639} & 0.235 & \textbf{3.565} & 6.351 & 0.900 & \textbf{77} & 77 & -- & \textbf{1078.980} & 2857.733 & -- & -- & -- & 0.085 & -- & -- & 0.017 & -- & -- & 0.340 \\
+++++Lund University Sphinx & \textbf{11.392} & 42.959 & 19.522 & \textbf{2.981} & 8.720 & 4.585 & \textbf{6.056} & 15.000 & 4.780 & \textbf{70} & 70 & -- & \textbf{2718.567} & 4705.610 & -- & -- & -- & 8.752 & -- & -- & 2.191 & -- & -- & 1.360 \\
+++++Nijo Castle Gate & \textbf{0.839} & 8.641 & 1.495 & \textbf{0.165} & 2.791 & 0.286 & \textbf{7.415} & 20.967 & 1.700 & \textbf{19} & 19 & -- & \textbf{1095.177} & 3241.330 & -- & -- & -- & 0.069 & -- & -- & 0.012 & -- & -- & 0.730 \\
+++++Pantheon Paris & -- & 0.509 & 0.192 & -- & 0.066 & 0.050 & -- & 7.493 & 1.470 & -- & 179 & -- & -- & 5565.670 & -- & -- & -- & 0.040 & -- & -- & 0.005 & -- & -- & 0.490 \\
+++++Park Gate Clermont Ferrand & \textbf{21.388} & 26.451 & 0.391 & \textbf{10.192} & 12.467 & 0.125 & \textbf{7.712} & 9.374 & 0.570 & \textbf{34} & 34 & -- & \textbf{1596.940} & 3659.900 & -- & -- & -- & 0.049 & -- & -- & 0.022 & -- & -- & 0.350 \\
+++++Plaza De Armas Santiago & 0.775 & -- & 6.782 & 0.337 & -- & 2.944 & 6.009 & -- & 7.400 & 240 & -- & -- & 6372.683 & -- & -- & -- & -- & 2.556 & -- & -- & 1.383 & -- & -- & 4.900 \\
+++++Porta San Donato Bologna & 0.592 & -- & 2.153 & 0.107 & -- & 0.388 & 5.603 & -- & 2.280 & 141 & -- & -- & 3754.163 & -- & -- & -- & -- & 0.095 & -- & -- & 0.046 & -- & -- & 0.750 \\
+++++Round Church Cambridge & 2.137 & -- & 2.451 & 0.927 & -- & 1.003 & 6.103 & -- & 2.660 & 92 & -- & -- & 9779.270 & -- & -- & -- & -- & 1.107 & -- & -- & 0.582 & -- & -- & 1.540 \\
+++++Skansen Kronan Gothenburg & 0.301 & -- & 0.736 & 0.102 & -- & 0.226 & 2.906 & -- & 1.240 & 131 & -- & -- & 17153.677 & -- & -- & -- & -- & 0.026 & -- & -- & 0.008 & -- & -- & 0.670 \\
+++++Smolny Cathedral St Petersburg & 21.775 & -- & 0.554 & 2.111 & -- & 0.051 & 11.239 & -- & 1.660 & 131 & -- & -- & 10362.787 & -- & -- & -- & -- & 0.033 & -- & -- & 0.006 & -- & -- & 0.810 \\
+++++Some Cathedral In Barcelona & 0.651 & -- & 0.880 & 0.233 & -- & 0.315 & 6.271 & -- & 2.870 & 177 & -- & -- & 5001.973 & -- & -- & -- & -- & 0.026 & -- & -- & 0.011 & -- & -- & 0.890 \\
+++++Sri Mariamman Singapore & 1.219 & -- & 2.302 & 0.382 & -- & 0.683 & 10.088 & -- & 4.130 & 222 & -- & -- & 5985.540 & -- & -- & -- & -- & 0.077 & -- & -- & 0.023 & -- & -- & 0.910 \\
+++++Sri Thendayuthapani Singapore & 0.593 & -- & 46.269 & 0.150 & -- & 3.812 & 7.016 & -- & 23.370 & 98 & -- & -- & 13338.167 & -- & -- & -- & -- & 44.170 & -- & -- & 2.870 & -- & -- & 8.440 \\
+++++Sri Veeramakaliamman Singapore & 2.404 & -- & 2.559 & 0.559 & -- & 0.597 & 12.234 & -- & 3.470 & 157 & -- & -- & 12377.927 & -- & -- & -- & -- & 0.175 & -- & -- & 0.040 & -- & -- & 0.730 \\
+++++Statue Of Liberty & 52.358 & -- & 46.887 & 28.503 & -- & 20.012 & 235036.825 & -- & 26.160 & 134 & -- & -- & 99.900 & -- & -- & -- & -- & 9.091 & -- & -- & 4.122 & -- & -- & 6.970 \\
+++++The Pumpkin & 25.731 & -- & 94.672 & 5.552 & -- & 14.890 & 26.876 & -- & 33.410 & 196 & -- & -- & 4327.273 & -- & -- & -- & -- & 98.862 & -- & -- & 14.952 & -- & -- & 24.850 \\
+++++Thian Hook Keng Temple Singapore & 0.927 & -- & 0.832 & 0.093 & -- & 0.082 & 14.915 & -- & 2.750 & 138 & -- & -- & 3298.537 & -- & -- & -- & -- & 0.081 & -- & -- & 0.008 & -- & -- & 1.130 \\
+++++Tsar Nikolai I & 42.166 & -- & 48.499 & 8.732 & -- & 9.467 & 11.210 & -- & 9.790 & 98 & -- & -- & 4929.200 & -- & -- & -- & -- & 36.280 & -- & -- & 7.836 & -- & -- & 6.530 \\
+++++Urban II & 58.919 & -- & 47.490 & 11.006 & -- & 9.467 & 17.538 & -- & 9.380 & 96 & -- & -- & 8889.617 & -- & -- & -- & -- & 48.214 & -- & -- & 9.586 & -- & -- & 6.920 \\
+++++Vercingetorix & 82.952 & -- & 69.328 & 10.195 & -- & 8.788 & 7.257 & -- & 5.080 & 69 & -- & -- & 1223.970 & -- & -- & -- & -- & 17.706 & -- & -- & 3.104 & -- & -- & 1.500 \\
+++++Yueh Hai Ching Temple Singapore & 0.544 & -- & 0.720 & 0.075 & -- & 0.098 & 5.596 & -- & 0.940 & 43 & -- & -- & 1748.667 & -- & -- & -- & -- & 0.043 & -- & -- & 0.014 & -- & -- & 0.650 \\
++++ \midrule
++++-Mean & \textbf{16.146} & 23.697 & 18.023 & \textbf{3.121} & 4.014 & 2.912 & 6920.602 & \textbf{13.088} & 5.673 & 100.559 & \textbf{104.818} & -- & \textbf{5819.728} & 7358.187 & -- & -- & -- & 13.695 & -- & -- & 1.937 & -- & -- & 2.992 \\
+++++Mean & \textbf{15.676} & 21.520 & 17.547 & \textbf{3.032} & 3.056 & 2.840 & 6723.056 & \textbf{10.628} & 5.595 & \textbf{102.743} & 83.895 & -- & \textbf{5460.373} & 6059.731 & -- & -- & -- & 13.315 & -- & -- & 1.883 & -- & -- & 2.933 \\
++++ \bottomrule
++++ \end{tabular}}
++++ \end{table*}
+++ ```
+++ - untracked/modified files:
+++ ```
++++M code/results/single_scene/REPRO.md
++++ M code/results/single_scene/summary.csv
++++ M code/results/single_scene/summary_table.md
++++ M code/results/single_scene/summary_table.tex
+++ ?? MVG_Project_Report_Ortal_Dayan.pdf
+++ ?? "RESFM- ROBUST DEEP EQUIVARIANT STRUCTURE FROM MOTION.pdf"
+++ ?? "claude specs/SPEC_cvpr_experiments.md"
+++ ?? "claude specs/SPEC_sfm_datasets_setup.md"
+++ ?? "claude specs/SPEC_single_scene_experiments.md"
+++ ?? "claude specs/SPEC_uesfm_combined.md"
++++?? "claude specs/TASK_crossdataset_readiness.md"
+++ ?? code/datasets/Euclidean
+++ ?? tmp_ab_check/
+++ ```
+++ ### esfm-baseline (official ESFM)
+++ - path: `/home/projects/bagon/ortalda/MVG/final-project/esfm-baseline`
+++-- commit: `2c0079497bf5dda1eeb2a902b2f859c6387f213a`
++++- commit: `de666af53423c1dc3dd412e53c4e153c03d70922`
+++ - **WARNING: working tree dirty.** Diff:
+++ ```diff
+++ (untracked files only)
+++@@ -58,6 +2532,7 @@ Generated: 2026-07-13T20:41:04
+++ ?? code/confs/single_scene_generated/ss_esfm_Buddah_Tooth_Relic_Temple_Singapore_seed1.conf
+++ ?? code/confs/single_scene_generated/ss_esfm_Buddah_Tooth_Relic_Temple_Singapore_seed2.conf
+++ ?? code/confs/single_scene_generated/ss_esfm_Doge_Palace_Venice_seed1.conf
++++?? code/confs/single_scene_generated/ss_esfm_Doge_Palace_Venice_seed2.conf
+++ ?? code/confs/single_scene_generated/ss_esfm_Door_Lund_seed1.conf
+++ ?? code/confs/single_scene_generated/ss_esfm_Door_Lund_seed2.conf
+++ ?? code/confs/single_scene_generated/ss_esfm_Drinking_Fountain_Somewhere_In_Zurich_seed1.conf
+++@@ -70,6 +2545,7 @@ Generated: 2026-07-13T20:41:04
+++ ?? code/confs/single_scene_generated/ss_esfm_Ecole_Superior_De_Guerre_seed2.conf
+++ ?? code/confs/single_scene_generated/ss_esfm_Eglise_du_dome_seed0.conf
+++ ?? code/confs/single_scene_generated/ss_esfm_Eglise_du_dome_seed1.conf
++++?? code/confs/single_scene_generated/ss_esfm_Eglise_du_dome_seed2.conf
+++ ?? code/confs/single_scene_generated/ss_esfm_Folke_Filbyter_seed0.conf
+++ ?? code/confs/single_scene_generated/ss_esfm_Folke_Filbyter_seed1.conf
+++ ?? code/confs/single_scene_generated/ss_esfm_Folke_Filbyter_seed2.conf
+++@@ -114,16 +2590,22 @@ Generated: 2026-07-13T20:41:04
+++ ?? code/confs/single_scene_generated/ss_esfm_Round_Church_Cambridge_seed2.conf
+++ ?? code/confs/single_scene_generated/ss_esfm_Skansen_Kronan_Gothenburg_seed0.conf
+++ ?? code/confs/single_scene_generated/ss_esfm_Skansen_Kronan_Gothenburg_seed1.conf
++++?? code/confs/single_scene_generated/ss_esfm_Skansen_Kronan_Gothenburg_seed2.conf
+++ ?? code/confs/single_scene_generated/ss_esfm_Smolny_Cathedral_St_Petersburg_seed0.conf
+++ ?? code/confs/single_scene_generated/ss_esfm_Smolny_Cathedral_St_Petersburg_seed1.conf
++++?? code/confs/single_scene_generated/ss_esfm_Smolny_Cathedral_St_Petersburg_seed2.conf
+++ ?? code/confs/single_scene_generated/ss_esfm_Some_Cathedral_In_Barcelona_seed0.conf
++++?? code/confs/single_scene_generated/ss_esfm_Some_Cathedral_In_Barcelona_seed1.conf
++++?? code/confs/single_scene_generated/ss_esfm_Some_Cathedral_In_Barcelona_seed2.conf
+++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Mariamman_Singapore_seed0.conf
+++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Mariamman_Singapore_seed1.conf
+++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Mariamman_Singapore_seed2.conf
+++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Thendayuthapani_Singapore_seed0.conf
+++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Thendayuthapani_Singapore_seed1.conf
++++?? code/confs/single_scene_generated/ss_esfm_Sri_Thendayuthapani_Singapore_seed2.conf
+++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Veeramakaliamman_Singapore_seed0.conf
+++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Veeramakaliamman_Singapore_seed1.conf
++++?? code/confs/single_scene_generated/ss_esfm_Sri_Veeramakaliamman_Singapore_seed2.conf
+++ ?? code/confs/single_scene_generated/ss_esfm_Statue_Of_Liberty_seed0.conf
+++ ?? code/confs/single_scene_generated/ss_esfm_Statue_Of_Liberty_seed1.conf
+++ ?? code/confs/single_scene_generated/ss_esfm_Statue_Of_Liberty_seed2.conf
+++@@ -137,10 +2619,14 @@ Generated: 2026-07-13T20:41:04
+++ ?? code/confs/single_scene_generated/ss_esfm_Tsar_Nikolai_I_seed1.conf
+++ ?? code/confs/single_scene_generated/ss_esfm_Tsar_Nikolai_I_seed2.conf
+++ ?? code/confs/single_scene_generated/ss_esfm_Urban_II_seed0.conf
++++?? code/confs/single_scene_generated/ss_esfm_Urban_II_seed1.conf
++++?? code/confs/single_scene_generated/ss_esfm_Urban_II_seed2.conf
+++ ?? code/confs/single_scene_generated/ss_esfm_Vercingetorix_seed0.conf
+++ ?? code/confs/single_scene_generated/ss_esfm_Vercingetorix_seed1.conf
+++ ?? code/confs/single_scene_generated/ss_esfm_Vercingetorix_seed2.conf
+++ ?? code/confs/single_scene_generated/ss_esfm_Yueh_Hai_Ching_Temple_Singapore_seed0.conf
++++?? code/confs/single_scene_generated/ss_esfm_Yueh_Hai_Ching_Temple_Singapore_seed1.conf
++++?? code/confs/single_scene_generated/ss_esfm_Yueh_Hai_Ching_Temple_Singapore_seed2.conf
+++ ```
+++ 
+++ ## Environment (shared venv used for BOTH methods)
+++diff --git a/code/results/single_scene/summary.csv b/code/results/single_scene/summary.csv
+++index e4f9f5e..5ec842d 100644
+++--- a/code/results/single_scene/summary.csv
++++++ b/code/results/single_scene/summary.csv
+++@@ -2,171 +2,154 @@ method,scene,seed,rot_err_deg,pos_err,reproj_err_px,reproj_med_px,n_cams,runtime
+++ esfm,Alcatraz Courtyard,0,0.324288380204305,0.0854554642858637,3.379232694353053,1.9204072326560708,133,4821.4,4792.707528591156,99999.0
+++ esfm,Alcatraz Courtyard,1,0.36138678157391557,0.09112912757839253,3.3691035794153503,1.8536696768772378,133,4825.72,4804.594581127167,99999.0
+++ esfm,Alcatraz Courtyard,2,0.4015043649148433,0.10235190268069758,3.557242072642648,1.9198007399315369,133,4833.02,4812.242679834366,99999.0
+++-uesfm,Alcatraz Courtyard,0,0.6370266175309793,0.1671943803031883,5.956592777085197,3.6638282340102246,133,6478.06,6448.735792636871,99999.0
+++-uesfm,Alcatraz Courtyard,1,0.5812669610362707,0.15343994793419002,5.71686544495911,3.550076086474883,133,6483.48,6454.087207078934,99999.0
+++-uesfm,Alcatraz Courtyard,2,0.6293740753551684,0.16343948136157807,5.713459273661585,3.459422664018691,133,6483.73,6451.153926610947,99999.0
++++uesfm,Alcatraz Courtyard,0,0.5509063563081962,0.14050635120746804,5.528957815039945,3.569220349534855,133,6556.5,6511.41095662117,99999.0
++++uesfm,Alcatraz Courtyard,1,0.6823224031499351,0.17277455831381494,5.435931436314427,3.4771386471547348,133,6552.19,6506.419185638428,99999.0
++++uesfm,Alcatraz Courtyard,2,29.163200369074893,6.342299350838519,10.849871644564203,6.781661682554276,133,6544.26,6512.667362689972,99999.0
+++ esfm,Alcatraz Water Tower,0,0.6984432687370988,0.38694796813795407,4.741313233832226,2.1058898944784543,172,2461.59,2440.421297073364,99999.0
+++ esfm,Alcatraz Water Tower,1,0.5944425806062753,0.3287537445614408,4.5656682619915765,1.9703698309044646,172,2467.28,2454.429310798645,99999.0
+++ esfm,Alcatraz Water Tower,2,0.7051984712289041,0.39359395644708545,4.982802041506025,2.1318655747232063,172,2455.14,2441.783402681351,99999.0
+++-uesfm,Alcatraz Water Tower,0,18.69941055700858,8.189402666249999,12.38813738647375,6.481578998967327,172,3857.94,3831.41609287262,99999.0
+++-uesfm,Alcatraz Water Tower,1,1.3394060108035364,0.7384271683550214,6.577617375342355,3.200105455600137,172,3887.47,3865.343374729156,99999.0
+++-uesfm,Alcatraz Water Tower,2,2.200189184784068,1.3839747978399466,6.854463337738994,3.1730434024886014,172,3819.31,3799.390902757645,99999.0
++++uesfm,Alcatraz Water Tower,0,2.016029335283383,1.087082854315953,7.9502009315303575,4.413488326253207,172,3805.19,3775.908159971237,99999.0
++++uesfm,Alcatraz Water Tower,1,1.4259772151091428,0.7645994701417347,7.177871865477131,3.7683369835989504,172,3787.07,3763.27596282959,99999.0
++++uesfm,Alcatraz Water Tower,2,0.9982626119141534,0.558403777925872,7.063395425015731,3.814475911180562,172,3795.37,3773.579354286194,99999.0
+++ esfm,Buddah Tooth Relic Temple Singapore,0,1.3132762212436329,0.31652548993670354,9.903890396062208,4.634952217191846,162,2921.45,2903.059753894806,99999.0
+++ esfm,Buddah Tooth Relic Temple Singapore,1,0.9908742070528852,0.2335767172234499,9.836436953133868,4.661767906617449,162,2915.39,2904.387546062469,99999.0
+++ esfm,Buddah Tooth Relic Temple Singapore,2,1.089403690069201,0.2896515999941069,10.151103921611115,4.628183275251743,162,2913.67,2902.928613901138,99999.0
+++-uesfm,Buddah Tooth Relic Temple Singapore,0,1.9767734026333772,0.5777116480820644,15.584999477939833,8.800407928031486,162,4315.97,4296.739196300507,99999.0
+++-uesfm,Buddah Tooth Relic Temple Singapore,1,1.4532821126601407,0.345133635718661,16.753182673783428,9.555989756393366,162,4334.28,4311.474883317947,99999.0
+++-uesfm,Buddah Tooth Relic Temple Singapore,2,32.81105779154638,6.0024233739495125,26.564606536203048,13.925597303414058,162,4312.37,4291.042752504349,99999.0
++++uesfm,Buddah Tooth Relic Temple Singapore,0,1.5799948547864517,0.43886498334860274,13.484389374830494,7.719072745032001,162,4294.34,4266.538989067078,99999.0
++++uesfm,Buddah Tooth Relic Temple Singapore,1,1.934129180062467,0.5924991617860504,13.872322562031941,8.04977435062149,162,4303.62,4259.311170101166,99999.0
++++uesfm,Buddah Tooth Relic Temple Singapore,2,1.8591664615796053,0.5279087705483619,14.731758217446671,8.481044153007865,162,4270.55,4249.088565349579,99999.0
+++ esfm,Doge Palace Venice,0,0.9741701978804636,0.2578214102266912,6.511459231891615,3.6947627756511525,241,28795.18,28746.22650837898,99999.0
+++ esfm,Doge Palace Venice,1,0.627860879835449,0.1773446161890865,6.351890452986879,3.537139014829222,241,28742.27,28699.91308617592,99999.0
+++-uesfm,Doge Palace Venice,0,0.9110673544913167,0.2904399918743801,10.966183720673449,6.771416746021886,241,21074.95,21017.30448675156,99999.0
++++esfm,Doge Palace Venice,2,0.6576305681966796,0.19114542381679772,6.214841374149415,3.591638795332466,241,10030.76,9989.16938996315,99999.0
++++uesfm,Doge Palace Venice,0,1.120669822868001,0.3500031006198606,10.496216444068507,6.9730858315176345,241,20767.26,20701.21099281311,99999.0
+++ esfm,Door Lund,0,0.01167051570787235,0.003654424593160691,1.8552512563951393,0.9492901559591532,12,4892.41,4878.351888656616,99999.0
+++ esfm,Door Lund,1,0.009685491454411136,0.0030270923396151107,1.6462990657331118,0.9049495114630693,12,4884.51,4877.529450178146,99999.0
+++ esfm,Door Lund,2,0.02068285629921104,0.005193423834514985,1.933833706356197,0.9506754982524039,12,4888.17,4880.201772212982,99999.0
+++-uesfm,Door Lund,0,14.059454220682019,0.9733778175929079,11.85413335315025,6.517535151323047,12,9185.47,8715.343282699585,95000.0
+++-uesfm,Door Lund,1,14.111993883089163,0.9618908892808385,11.853785941610944,6.504290333453439,12,9203.34,8729.614821910858,95000.0
+++-uesfm,Door Lund,2,14.344083655704312,0.9193773651816372,12.058851721280973,6.643622749322514,12,9176.97,9162.127201795578,99999.0
++++uesfm,Door Lund,0,0.03084538453455712,0.007522151486969653,2.3711871496098955,1.6703501976219666,12,9262.05,9243.771874427795,99999.0
++++uesfm,Door Lund,1,13.889154901234134,1.0160235649641791,11.830787694675347,6.5775252925915195,12,9243.42,7848.296544790268,85000.0
+++ esfm,Drinking Fountain Somewhere In Zurich,0,25.804179845300204,1.1242096722251314,8.219025541657992,3.3464550934365884,14,888.07,874.4874215126038,99999.0
+++ esfm,Drinking Fountain Somewhere In Zurich,1,25.754577171528883,1.1271567319301725,8.245599860088616,3.352895635561939,14,882.33,873.2415659427643,99999.0
+++ esfm,Drinking Fountain Somewhere In Zurich,2,0.019604308735801684,0.006780729271172703,1.7090486008047308,0.6375575382268279,14,881.11,873.6504108905792,99999.0
+++-uesfm,Drinking Fountain Somewhere In Zurich,0,26.27952541668808,1.102546079947844,8.40971803772888,3.4593948661771203,14,2969.96,2950.480526924133,99999.0
+++-uesfm,Drinking Fountain Somewhere In Zurich,1,26.264849988219943,1.1078406736783482,8.547474098662555,3.585334207185317,14,2960.9,2504.42572927475,85000.0
+++-uesfm,Drinking Fountain Somewhere In Zurich,2,0.06235835436589852,0.011103715758719332,2.5743283079384534,1.5361135173722633,14,2960.08,2945.673992872238,99999.0
++++uesfm,Drinking Fountain Somewhere In Zurich,0,25.946100100284117,1.1178920745623426,8.471712388928934,3.6241397282821626,14,5817.37,5796.999963760376,99999.0
++++uesfm,Drinking Fountain Somewhere In Zurich,1,25.973070279327214,1.1166462082604052,8.538877295313494,3.6343523276029472,14,5821.94,5806.947921991348,99999.0
++++uesfm,Drinking Fountain Somewhere In Zurich,2,0.06926036271356541,0.01532826351119577,3.015375672575656,1.7451989053599506,14,5806.35,5792.085475206375,99999.0
+++ esfm,East Indiaman Goteborg,0,6.033696331509855,1.0773322362193534,7.427433157074123,2.292729458380311,179,3990.25,3970.732353687286,99999.0
+++ esfm,East Indiaman Goteborg,1,5.313967690277326,0.9356515362150493,7.350753973659207,2.2747477951684703,179,3986.66,3970.914752960205,99999.0
+++ esfm,East Indiaman Goteborg,2,5.357164002763642,0.9254557315236661,7.174538078735992,2.2493605774605823,179,3984.64,3967.810528755188,99999.0
+++-uesfm,East Indiaman Goteborg,0,4.992588850190341,0.9093886386013725,9.930437318497303,4.011819888270754,179,5247.16,5221.816437721252,99999.0
+++-uesfm,East Indiaman Goteborg,1,5.382092262718057,0.9927766021898647,10.035611563702533,3.912004668740913,179,5245.65,5210.653891563416,99999.0
+++-uesfm,East Indiaman Goteborg,2,5.554715231405653,0.9752252406138763,9.5551521706325,3.7417116053580193,179,5231.48,5207.815300703049,99999.0
++++uesfm,East Indiaman Goteborg,0,5.365772057886591,0.9552115949640507,9.655123223073948,4.174486084237091,179,5203.6,5178.753549814224,99999.0
++++uesfm,East Indiaman Goteborg,1,10.45533482189932,2.0687908277631517,9.98932850274456,4.110771460953736,179,5195.98,5174.958552598953,99999.0
++++uesfm,East Indiaman Goteborg,2,5.872735570127935,1.0672842053553182,10.745053220477157,4.480528977857714,179,5194.7,5173.540862560272,99999.0
+++ esfm,Ecole Superior De Guerre,0,0.4863901164687255,0.1239377922325492,3.750666445958877,2.0528113108131016,35,2385.33,2371.708672523499,99999.0
+++ esfm,Ecole Superior De Guerre,1,0.3031794677170201,0.07921628898965839,3.2456935905644406,1.730123845260616,35,2379.34,2371.191816806793,99999.0
+++ esfm,Ecole Superior De Guerre,2,0.2580439805296837,0.06772089648290676,3.257951412507941,1.687291979251782,35,2376.49,2368.935952663422,99999.0
+++-uesfm,Ecole Superior De Guerre,0,0.28243957678843373,0.0778675008788964,5.291358032458062,3.601773151002547,35,4647.51,4622.850021600723,99999.0
+++-uesfm,Ecole Superior De Guerre,1,0.25507313740145815,0.06695766472508183,5.170495431958795,3.5179325723573998,35,4636.07,4616.890106916428,99999.0
+++-uesfm,Ecole Superior De Guerre,2,0.3137499276035537,0.08683967313665406,5.583934687507232,3.7813953498055604,35,4617.92,4602.193422079086,99999.0
++++uesfm,Ecole Superior De Guerre,0,51.32681795107141,2.57407526776939,11.751384768771835,6.449293516058521,35,10426.77,10405.3174200058,99999.0
++++uesfm,Ecole Superior De Guerre,1,0.2738850931952059,0.0734625834578241,5.5413290518324265,3.89237755756953,35,10405.39,10384.56669402122,99999.0
+++ esfm,Eglise du dome,0,0.3242410065346163,0.0824797769992069,3.906129661328729,1.3512292775673158,85,25054.74,25028.64922428131,99999.0
+++ esfm,Eglise du dome,1,0.3771623368290559,0.09432763361451474,3.692490941110161,1.297032357665518,85,25144.16,25116.01952123642,99999.0
+++-uesfm,Eglise du dome,0,45.800562123362404,6.451383786272971,7.963674120656965,3.986938865913781,85,23371.73,23333.49408817291,99999.0
++++esfm,Eglise du dome,2,0.45623311880194967,0.11755570327842058,4.363674496823582,1.384257241090686,85,8694.98,8671.502047538757,99999.0
++++uesfm,Eglise du dome,0,0.8528700994829498,0.23193079249562099,5.944108200898114,2.9306630821877517,85,9803.94,9764.585114717484,99999.0
++++uesfm,Eglise du dome,1,0.6936386490465627,0.1932815436676001,6.091509620719386,2.9784748994578076,85,9788.95,9753.742706537247,99999.0
+++ esfm,Folke Filbyter,0,89.03873163576341,0.12895560134543632,17.986624161769267,7.423093846189277,40,1582.8,1573.360969305038,99999.0
+++ esfm,Folke Filbyter,1,84.74740427947457,0.13132264058966686,17.98706393015447,7.385900243615879,40,1579.13,1572.079373121262,99999.0
+++ esfm,Folke Filbyter,2,83.70817707245973,0.1261878953205589,21.85200129525089,6.324245998829542,40,1583.76,1576.207699537277,99999.0
+++-uesfm,Folke Filbyter,0,80.34590333370599,0.13169898199385566,22.83327162619912,11.711593158296864,40,3701.49,3129.452286243439,85000.0
+++-uesfm,Folke Filbyter,1,81.28676564061475,0.1282318052949447,24.480394314414323,9.74969805057316,40,3698.11,3681.891911745071,99999.0
+++-uesfm,Folke Filbyter,2,82.23270227083374,0.1312531386108397,22.153947807931974,10.961536903594071,40,3709.96,3695.84988451004,99999.0
++++uesfm,Folke Filbyter,0,84.55526303793802,0.13091275274611416,29.190267154194732,11.160073225547443,40,3747.65,3727.990335702896,99999.0
++++uesfm,Folke Filbyter,1,84.87047537526948,0.13112291273275417,32.36503566471496,13.743964264262598,40,3743.53,3728.497133255005,99999.0
++++uesfm,Folke Filbyter,2,78.54375401873024,0.1316690349103016,23.11854509839131,10.94439865744236,40,3774.4,3007.95537352562,80000.0
+++ esfm,Fort Channing Gate Singapore,0,0.10143646071220622,0.04531774433800129,1.7158776598376824,0.8165178663538488,27,3438.39,3428.288990497589,99999.0
+++ esfm,Fort Channing Gate Singapore,1,0.08873951454303304,0.039001318240256114,1.783712095722291,0.8156020254191567,27,3430.88,3424.490573883057,99999.0
+++ esfm,Fort Channing Gate Singapore,2,0.09031224636384508,0.03957289173255022,1.850800887931411,0.820658413919779,27,3433.07,3426.547735452652,99999.0
+++-uesfm,Fort Channing Gate Singapore,0,33.158343156383296,11.908537566474344,12.69248500308198,5.438758486477886,27,6209.29,6192.107179880142,99999.0
+++-uesfm,Fort Channing Gate Singapore,1,0.07319798057764343,0.03541606344760984,3.4819579894798927,2.2085534812612853,27,6159.7,6144.797720909119,99999.0
+++-uesfm,Fort Channing Gate Singapore,2,0.15394831153888047,0.0686769122909015,3.77993377770425,2.2995094868650368,27,6147.27,6132.745970487595,99999.0
++++uesfm,Fort Channing Gate Singapore,0,0.10214634402234797,0.048923384893640356,3.7907267588398517,2.449846070018404,27,6175.88,6156.048281908035,99999.0
++++uesfm,Fort Channing Gate Singapore,1,0.14219758622796153,0.06357969294387673,3.5066994916782015,2.1555120267718575,27,6191.97,6176.711992740631,99999.0
++++uesfm,Fort Channing Gate Singapore,2,0.217326908828172,0.10047378918560818,3.8470573886147417,2.340537393844086,27,6145.9,6130.889567136765,99999.0
+++ esfm,Golden Statue Somewhere In Hong Kong,0,0.23894829924734365,0.053817571637812096,1.7239352536484445,0.744089758310296,18,4096.74,4087.775670289993,99999.0
+++ esfm,Golden Statue Somewhere In Hong Kong,1,0.2233990458438394,0.053061028269618456,1.8936878914364637,0.7746436618607616,18,4096.28,4088.994391202927,99999.0
+++ esfm,Golden Statue Somewhere In Hong Kong,2,0.23397914670516012,0.051137498616263405,1.87070255923672,0.7763877145267694,18,4094.54,4087.674698591232,99999.0
+++-uesfm,Golden Statue Somewhere In Hong Kong,0,0.36074365888948035,0.06894135246785482,2.939092008443413,1.863811153764833,18,7761.69,7746.569520235062,99999.0
+++-uesfm,Golden Statue Somewhere In Hong Kong,1,0.35638930493960375,0.06132592684828702,2.686157398440538,1.735571387461727,18,7759.16,7745.389947891235,99999.0
+++-uesfm,Golden Statue Somewhere In Hong Kong,2,0.3955566726941678,0.09197790539367184,2.878341172309852,1.8164738202745854,18,7766.46,7751.458931922913,99999.0
+++ esfm,Gustav Vasa,0,3.4881363194229826,0.2273181104934767,1.7369784534725496,0.9521606063342205,18,711.53,703.3091380596161,99999.0
+++ esfm,Gustav Vasa,1,3.4307883498221035,0.21432466556133864,1.5556448596237022,0.9232184455756751,18,710.28,703.9906163215637,99999.0
+++ esfm,Gustav Vasa,2,3.6089062562451386,0.2196578287395002,1.673803634770729,0.9429755567757121,18,707.79,701.5903799533844,99999.0
+++-uesfm,Gustav Vasa,0,5.828154714648938,0.32510821331670736,3.0711042143581646,2.4170166616864357,18,2566.0,2547.931907176971,99999.0
+++-uesfm,Gustav Vasa,1,6.515239842974278,0.35941242702736925,3.089677100739632,2.4476428398062584,18,2530.9,1385.089058637619,55000.0
+++-uesfm,Gustav Vasa,2,5.954036423569478,0.34942942401850696,3.101296679577594,2.400483554095863,18,2533.76,2520.798699855804,99999.0
++++uesfm,Gustav Vasa,0,36.645536646380805,1.0070193571286277,4.980488290477233,3.197977230260218,18,2562.94,2419.297919034958,95000.0
++++uesfm,Gustav Vasa,1,36.4914104514929,1.03254185606701,5.377024289258223,3.4818421684593366,18,4566.81,4543.375423431396,99999.0
++++uesfm,Gustav Vasa,2,4.4238740734834625,0.284762443333143,3.8820183248893803,2.8663959682636566,18,4533.21,4516.61039352417,99999.0
+++ esfm,GustavIIAdolf,0,46.92758389264633,8.570499881283022,11.684017169857798,5.131125459552879,57,1047.82,1038.018162250519,99999.0
+++ esfm,GustavIIAdolf,1,47.21999673071185,8.515052769156952,11.695065125063977,5.091617086264161,57,1041.41,1034.5799472332,99999.0
+++ esfm,GustavIIAdolf,2,47.2312223396789,8.523261295519099,11.787583212607423,5.2715388868550574,57,1043.28,1036.478415489197,99999.0
+++-uesfm,GustavIIAdolf,0,89.37607436538163,13.276440526172165,15.53883714368065,5.864332719302596,57,2904.42,2881.687849521637,99999.0
+++-uesfm,GustavIIAdolf,1,70.16367503302509,11.555242253627961,15.162174576474532,6.9709512967490355,57,2884.62,2870.103493452072,99999.0
+++-uesfm,GustavIIAdolf,2,72.72996778765942,11.054260851523775,16.85269001998736,9.009102977660282,57,2888.7,2874.096453428268,99999.0
++++uesfm,GustavIIAdolf,0,85.51179144359693,12.792264407383987,17.018024925092185,8.196266635685827,57,2973.35,2956.916826248169,99999.0
++++uesfm,GustavIIAdolf,1,88.9494736685426,13.260723460498589,17.699982689478404,8.888010081099381,57,2995.3,2976.055320978165,99999.0
++++uesfm,GustavIIAdolf,2,90.12096202318821,13.335169188185864,16.941538974035925,7.772181288890227,57,2974.4,2960.110015630722,99999.0
+++ esfm,Jonas Ahlstromer,0,64.46662959289324,11.985605264945464,12.602584968179986,2.9168445332917745,40,614.88,607.3498430252075,99999.0
+++ esfm,Jonas Ahlstromer,1,38.42510202464649,8.741159740710629,10.25268858530355,3.7024742418884333,40,618.73,611.5567183494568,99999.0
+++ esfm,Jonas Ahlstromer,2,43.32891670826628,8.741053567623224,10.553031866494822,3.031717304343782,40,612.74,606.6802380084991,99999.0
+++-uesfm,Jonas Ahlstromer,0,42.30073879858729,10.272232427418913,9.098970988131883,3.992857074878733,40,2339.57,1280.604686737061,55000.0
+++-uesfm,Jonas Ahlstromer,1,42.18412804385421,10.310808592826955,8.453808908218651,3.4329852879523464,40,2334.33,2320.362742424011,99999.0
+++-uesfm,Jonas Ahlstromer,2,42.25301394187691,10.315519611163806,8.793789077520445,3.8046125347596442,40,2311.04,2297.139800786972,99999.0
++++uesfm,Jonas Ahlstromer,0,45.264596178655594,9.403212752141883,11.336617283269852,4.728596140494824,40,2421.11,2401.221394062042,99999.0
++++uesfm,Jonas Ahlstromer,1,51.490398556789884,10.860548616493958,12.821407884052217,5.012487046810724,40,2428.25,2415.139223337173,99999.0
++++uesfm,Jonas Ahlstromer,2,42.99784237868634,10.470910789899067,9.274184391894982,4.4612757046777345,40,2394.6,2381.585824012756,99999.0
+++ esfm,Kings College University Of Toronto,0,14.400461796497643,3.2653370282498866,4.064353916685246,1.6753245173381437,77,1074.94,1066.306197404861,99999.0
+++ esfm,Kings College University Of Toronto,1,11.956699445712928,2.705452710546247,3.9727144140636783,1.5797087908756111,77,1071.64,1064.129351377487,99999.0
+++ esfm,Kings College University Of Toronto,2,0.7191591237413282,0.16178494376153177,2.656807607821862,1.1949677972638288,77,1090.36,1082.987608671188,99999.0
+++-uesfm,Kings College University Of Toronto,0,31.932532001930838,2.9775501815475,6.099555590572752,3.8717713150174773,77,2884.23,2866.194714784622,99999.0
+++-uesfm,Kings College University Of Toronto,1,2.4241467960672063,0.41577835428574456,4.631696476065898,3.148046400399828,77,2899.51,2884.114916801453,99999.0
+++-uesfm,Kings College University Of Toronto,2,30.212552334890066,2.3209884971628156,5.844709739288878,3.754268841443037,77,2874.99,2860.901805400848,99999.0
++++uesfm,Kings College University Of Toronto,0,29.200261744017546,2.0893308856114428,7.037455848998898,4.8227457854711675,77,2856.72,2838.923613071442,99999.0
++++uesfm,Kings College University Of Toronto,1,7.107988087620218,1.3235014088907995,5.965363878755208,4.179664786063016,77,2852.71,2839.42994761467,99999.0
++++uesfm,Kings College University Of Toronto,2,6.543121223644844,1.5041055210367995,6.0501544660211914,4.127143945227024,77,2863.77,2850.061238765717,99999.0
+++ esfm,Lund University Sphinx,0,13.661465370705821,3.5591094267381553,6.242667318159146,2.715620555070548,70,2718.96,2709.5872194767,99999.0
+++ esfm,Lund University Sphinx,1,12.524302160109885,3.2130942199657664,6.1182420215295075,2.7468468987195953,70,2722.86,2715.36418390274,99999.0
+++ esfm,Lund University Sphinx,2,7.98961938632192,2.1710959765491817,5.807289242508196,2.893370267054225,70,2713.88,2706.478076219559,99999.0
+++-uesfm,Lund University Sphinx,0,15.165716414733652,3.5564827651349713,10.803408462752863,6.730422995453065,70,4644.92,4626.135177612305,99999.0
+++-uesfm,Lund University Sphinx,1,68.88899086055604,13.548881652837478,14.336626262363456,7.626356837439143,70,4703.31,4687.50173997879,99999.0
+++-uesfm,Lund University Sphinx,2,15.014326075576053,3.5607694164340526,11.039432108003597,6.81081192121135,70,4615.6,4600.294319868088,99999.0
++++uesfm,Lund University Sphinx,0,42.95863056778373,8.719684065980621,14.999945982069319,8.361965506089375,70,4705.61,4681.787243127823,99999.0
++++uesfm,Lund University Sphinx,1,14.202013501375653,3.5150197050401064,10.824261355110268,6.665393527268384,70,4616.97,4592.001487731934,99999.0
+++ esfm,Nijo Castle Gate,0,0.6521524412276166,0.13305866403975986,7.404953325705311,3.0629804954987674,19,1097.29,1084.301982402802,99999.0
+++ esfm,Nijo Castle Gate,1,0.8564043952718835,0.16950374865879153,7.358076475544799,3.1235223975185864,19,1094.89,1084.640900611877,99999.0
+++ esfm,Nijo Castle Gate,2,1.0089926380330574,0.19095470213426863,7.481049991287358,3.1671779658403687,19,1093.35,1083.446812152863,99999.0
+++-uesfm,Nijo Castle Gate,0,1.6923297320319277,0.3246009364794658,11.792009654594867,6.945622003724464,19,3310.13,3288.75860619545,99999.0
+++-uesfm,Nijo Castle Gate,1,1.6697791743236226,0.32440429939345133,12.704681259536075,7.466128811395402,19,3308.19,3290.209539175034,99999.0
+++-uesfm,Nijo Castle Gate,2,1.682403384345942,0.32817821817561266,11.917977420973372,7.102279949089778,19,3301.77,3284.964460611343,99999.0
+++-uesfm,Pantheon Paris,0,0.3823220210788846,0.05987691585642601,7.649321203280243,4.797127317249956,179,5506.88,5473.694003582001,99999.0
+++-uesfm,Pantheon Paris,1,45.61471478909275,6.671068486991568,16.912359249810606,6.602415648206237,179,5491.31,5459.329566717148,99999.0
+++-uesfm,Pantheon Paris,2,12.03796171625132,2.465102828965073,16.125095834055482,6.860965809143959,179,5493.94,5463.555550813675,99999.0
++++uesfm,Nijo Castle Gate,0,8.641303631787597,2.791258638759238,20.96738948203601,10.297911838853688,19,3241.33,3220.324777603149,99999.0
++++uesfm,Nijo Castle Gate,1,1.3055705335697472,0.2681929407818948,11.680266119036336,6.962610820698007,19,3234.66,3216.868148088455,99999.0
++++uesfm,Pantheon Paris,0,0.5453004964472504,0.06999784414241637,7.790870638815493,5.057509442480535,179,5594.6,5562.967379331589,99999.0
++++uesfm,Pantheon Paris,1,0.472097972346089,0.062163803587728925,7.194706475909503,4.639524309087925,179,5536.74,5501.721322774887,99999.0
++++uesfm,Pantheon Paris,2,44.569134625905356,6.741655820531256,16.92751887234327,6.718188212415848,179,5541.46,5510.693382978439,99999.0
+++ esfm,Park Gate Clermont Ferrand,0,19.399216821978534,9.438471236639753,7.521969683469249,3.892136696028138,34,1597.96,1588.07075381279,99999.0
+++ esfm,Park Gate Clermont Ferrand,1,19.325402297843297,9.417998214523331,7.465007292717891,3.849088919564891,34,1596.87,1589.703137874603,99999.0
+++ esfm,Park Gate Clermont Ferrand,2,25.440611579781365,11.720641869372841,8.15032188787613,4.0871739024940315,34,1595.99,1588.841923236847,99999.0
+++-uesfm,Park Gate Clermont Ferrand,0,19.27114964212729,9.381930985797803,7.856513901068924,4.317004881495189,34,3598.69,3580.396904230118,99999.0
+++-uesfm,Park Gate Clermont Ferrand,1,19.288145181777256,9.394362801959678,7.762082007624792,4.207851936960509,34,3592.66,3577.494926214218,99999.0
+++-uesfm,Park Gate Clermont Ferrand,2,19.345727373971844,9.415066249707772,7.936652645099164,4.425507919979627,34,3600.86,3581.473034620285,99999.0
++++uesfm,Park Gate Clermont Ferrand,0,26.45121972215746,12.46688671568752,9.374365736774523,5.572822711026232,34,3659.9,3637.290126085281,99999.0
++++uesfm,Park Gate Clermont Ferrand,1,25.126789970267566,11.629146345704141,8.445284973863814,4.644358918969844,34,3647.69,3633.016668319702,99999.0
+++ esfm,Plaza De Armas Santiago,0,0.7341628728579594,0.32156050159900956,6.000154308231358,2.679971945741675,240,6375.02,6333.455562829971,99999.0
+++ esfm,Plaza De Armas Santiago,1,0.5176165701053533,0.22744923951256435,5.661576462867733,2.533952088248649,240,6373.68,6336.030319213867,99999.0
+++ esfm,Plaza De Armas Santiago,2,1.0740077415291467,0.46304622323619105,6.365716833190761,2.733195456670397,240,6369.35,6331.436369419098,99999.0
+++-uesfm,Plaza De Armas Santiago,0,6.6772176498607925,2.909933722327364,11.384209379387066,5.623987151554922,240,6540.76,6491.619550704956,99999.0
+++-uesfm,Plaza De Armas Santiago,1,1.3364898126442175,0.5686650140764532,7.951759184923103,3.933795015892702,240,6536.61,6482.320575714111,99999.0
+++-uesfm,Plaza De Armas Santiago,2,1.9671483940705004,0.8276654865349118,9.054185015150745,4.441179120460681,240,6529.24,6477.723381996155,99999.0
+++ esfm,Porta San Donato Bologna,0,0.5746001989174886,0.10608894230214515,5.691906895070585,2.4183460353027986,141,3751.75,3734.281898736954,99999.0
+++ esfm,Porta San Donato Bologna,1,0.8978003419853328,0.14474404936935045,5.682379581165418,2.563141107750346,141,3758.51,3742.688964605331,99999.0
+++ esfm,Porta San Donato Bologna,2,0.30440272958620346,0.07026546193502164,5.435311727517131,2.1626312981864086,141,3752.23,3738.056202173233,99999.0
+++-uesfm,Porta San Donato Bologna,0,1.1269164513732723,0.19275737956525044,10.071225650489282,5.557466375146339,141,5111.57,5083.21807050705,99999.0
+++-uesfm,Porta San Donato Bologna,1,60.66366912240625,10.234123128225525,19.593360464637342,9.305574874002819,141,5125.57,5098.085618019104,99999.0
+++-uesfm,Porta San Donato Bologna,2,1.2271289404198877,0.19605254084885107,10.066750816092268,5.595831381287904,141,5116.95,5093.473650932312,99999.0
+++ esfm,Round Church Cambridge,0,2.1173198662081005,0.9129854085205968,5.854320175654564,1.9683268910531708,92,9788.65,9759.719386577606,99999.0
+++ esfm,Round Church Cambridge,1,2.1415279636494953,0.9267927094737857,6.33905951937897,2.0439997148640834,92,9764.61,9741.356333732605,99999.0
+++-uesfm,Round Church Cambridge,0,2.5434984075631974,1.0219048587087352,8.914080922081759,4.131734829440289,92,12088.06,12047.02147889137,99999.0
+++-uesfm,Round Church Cambridge,1,2.435781054756339,0.9827352719070902,8.178482224933683,3.8400051230290893,92,12034.64,11993.72608184814,99999.0
++++esfm,Round Church Cambridge,2,2.1507951248804917,0.9401429799233081,6.115230559713554,2.0517209910251797,92,9784.55,9763.525354623795,99999.0
+++ esfm,Skansen Kronan Gothenburg,0,0.3380139741078169,0.1158931606908392,3.0478862369293624,1.5340801117570733,131,17163.94,17140.15425729752,99999.0
+++-uesfm,Skansen Kronan Gothenburg,0,0.37472494768162085,0.1243676312212316,4.798736338080396,3.348916057055834,131,16235.09,16202.21351981163,99999.0
++++esfm,Skansen Kronan Gothenburg,1,0.21516831826827018,0.07458182746108981,2.786705021535272,1.4619994473169777,131,17129.62,17109.18022465706,99999.0
++++esfm,Skansen Kronan Gothenburg,2,0.3508356515622903,0.11667319277987132,2.8836839507901773,1.5216584205042176,131,17167.47,17142.66050887108,99999.0
+++ esfm,Smolny Cathedral St Petersburg,0,21.81278753416543,2.110667559831311,11.2315357584775,6.379259553776089,131,10364.67,10320.51852750778,99999.0
+++ esfm,Smolny Cathedral St Petersburg,1,21.799828748358873,2.111313741354268,11.225141339870355,6.36605939788532,131,10360.52,10319.01374697685,99999.0
+++-uesfm,Smolny Cathedral St Petersburg,0,22.143392894578028,2.0985584569799265,12.52645725687386,7.895265841408903,131,10475.74,10416.62969732285,99999.0
++++esfm,Smolny Cathedral St Petersburg,2,21.71090600320656,2.110307413725059,11.259799763291888,6.4136596726420905,131,10363.17,10320.08415436745,99999.0
++++esfm,Some Cathedral In Barcelona,0,0.6775058368464424,0.24064262464083658,6.20165095815897,3.7253636674062887,177,5002.81,4985.274734258652,99999.0
++++esfm,Some Cathedral In Barcelona,1,0.6722147495861831,0.2423973792150467,6.350786792749083,3.720536755931007,177,5009.15,4991.552013158798,99999.0
++++esfm,Some Cathedral In Barcelona,2,0.6046048207068493,0.21699252142655406,6.261075100347952,3.7258171467923633,177,4993.96,4975.845166683197,99999.0
+++ esfm,Sri Mariamman Singapore,0,1.4920493985513708,0.4412389781578158,10.406406306167431,6.369585247957454,222,5987.98,5970.066485881805,99999.0
+++ esfm,Sri Mariamman Singapore,1,1.1345564396944785,0.3608906720626337,9.703927906328842,5.885963785879842,222,5993.62,5975.365797519684,99999.0
+++ esfm,Sri Mariamman Singapore,2,1.0293707921546553,0.3426709250774071,10.15220066698994,6.067537729054635,222,5975.02,5958.786058187485,99999.0
+++-uesfm,Sri Mariamman Singapore,0,16.417883129790784,5.526406595652791,19.962711816743173,12.18830031311293,222,6531.45,6494.750840425491,99999.0
+++-uesfm,Sri Mariamman Singapore,1,1.3793173305528281,0.5014534676018023,16.10857603648069,10.651562750176396,222,6467.65,6426.201541185379,99999.0
+++-uesfm,Sri Mariamman Singapore,2,1.639001766282598,0.536397596737868,17.490398022274007,11.439337896248928,222,6457.9,6427.717175960541,99999.0
+++ esfm,Sri Thendayuthapani Singapore,0,0.7190396606626565,0.17922550285921848,7.12365374281766,2.231392236341273,98,13362.09,13320.09781932831,99999.0
+++-uesfm,Sri Thendayuthapani Singapore,0,47.0750455982536,3.8553135654690074,14.133068630999212,7.117179633590087,98,13808.44,13754.94554877281,99999.0
++++esfm,Sri Thendayuthapani Singapore,1,0.5350953449295449,0.1261925950852554,6.4467364012557615,2.04778550672059,98,13347.85,13309.03068971634,99999.0
++++esfm,Sri Thendayuthapani Singapore,2,0.5262093764563488,0.14374074415280041,7.476614790858722,2.2575164196527426,98,13304.56,13262.61834168434,99999.0
+++ esfm,Sri Veeramakaliamman Singapore,0,2.333869921178247,0.542847254574005,11.965270419451508,5.472877662905037,157,12392.18,12365.15180635452,99999.0
+++-uesfm,Sri Veeramakaliamman Singapore,0,29.40487246624615,3.8676026992817505,28.950500021302087,14.32295224594287,157,12544.69,12504.26410722733,99999.0
++++esfm,Sri Veeramakaliamman Singapore,1,2.4804161216418734,0.5731176371311499,12.150997217479388,5.5309200508775875,157,12387.11,12363.29210400581,99999.0
++++esfm,Sri Veeramakaliamman Singapore,2,2.3974569344813434,0.5599333960859275,12.58650507397198,5.711763361742226,157,12354.49,12330.03034973145,99999.0
+++ esfm,Statue Of Liberty,0,52.35559762033152,28.241594953600423,688383.1603142131,683624.423417858,134,102.64,43.32754778862,45000.0
+++ esfm,Statue Of Liberty,1,52.36920727661949,28.627801879821945,9277.715116667516,9303.546387753544,134,99.78,59.98358988761902,65000.0
+++ esfm,Statue Of Liberty,2,52.3478568435799,28.64052858422677,7449.598084265948,7416.803834848509,134,97.28,3.520912647247314,0.0
+++-uesfm,Statue Of Liberty,0,40.92379801166407,19.915794833689585,48.781849500680366,25.904882064330994,134,12927.11,12902.23110127449,99999.0
+++-uesfm,Statue Of Liberty,1,38.79484508812604,17.090491683338875,33.27459817133403,17.608726936833005,134,12920.89,12896.32496738434,99999.0
+++ esfm,The Pumpkin,0,27.59459354015253,5.94835892276036,27.52772578222832,9.11126349703041,196,4330.9,4316.109169960022,99999.0
+++ esfm,The Pumpkin,1,26.54397520858153,5.701234509093555,27.342758332715125,8.907676505669826,196,4327.98,4314.043692827225,99999.0
+++ esfm,The Pumpkin,2,23.05395646123413,5.007378502393024,25.75879879971585,7.898438464943201,196,4322.94,4310.356798410416,99999.0
+++-uesfm,The Pumpkin,0,25.490608822831348,5.440568114393512,34.198429037653845,15.6024365790899,196,5793.03,5767.982522964478,99999.0
+++-uesfm,The Pumpkin,1,27.26556900533122,5.820572144056583,35.6951097814524,16.572200360468877,196,5785.61,5760.521738529205,99999.0
+++-uesfm,The Pumpkin,2,26.95722616022698,5.73349875876088,36.207441893083136,17.07678356860021,196,5793.15,5768.243654251099,99999.0
+++ esfm,Thian Hook Keng Temple Singapore,0,1.0653505173922067,0.1079565219850111,15.054152885837562,5.81055046268767,138,3299.26,3281.187238931656,99999.0
+++ esfm,Thian Hook Keng Temple Singapore,1,0.8390930608109779,0.08272919436528102,14.587041005106803,5.705702663819475,138,3306.93,3283.257929086685,99999.0
+++ esfm,Thian Hook Keng Temple Singapore,2,0.8760513596629002,0.08764395630143504,15.102705362055175,5.7567965038255124,138,3289.42,3278.173619031906,99999.0
+++-uesfm,Thian Hook Keng Temple Singapore,0,1.146131162829257,0.10799243740327723,20.24888615229727,10.297158058757331,138,4626.48,4595.79723072052,99999.0
+++-uesfm,Thian Hook Keng Temple Singapore,1,1.2072956626750737,0.12821580405567637,21.242388560332195,10.655698470690366,138,4600.34,4578.699931621552,99999.0
+++-uesfm,Thian Hook Keng Temple Singapore,2,1.251054659622132,0.12561972972485672,20.875018460120653,10.416875106878496,138,4610.3,4581.792747020721,99999.0
+++ esfm,Tsar Nikolai I,0,42.507553768489565,8.869914103947062,10.846690028504929,4.007961025902478,98,4932.15,4920.938139438629,99999.0
+++ esfm,Tsar Nikolai I,1,43.09835370739786,8.983133733509343,11.402130572143294,4.032588092580996,98,4924.71,4914.929541349411,99999.0
+++-uesfm,Tsar Nikolai I,0,75.94880014543322,15.146573953653359,15.307363921859029,6.734065570805788,98,6652.35,6629.384099006653,99999.0
+++-uesfm,Tsar Nikolai I,1,80.87807139992418,15.64102228288136,21.553650044939403,9.179513384124702,98,6641.34,6623.797205209732,99999.0
++++esfm,Tsar Nikolai I,2,40.892180545819315,8.342265568650971,11.381967217434166,4.272306098513181,98,4930.74,4920.495186090469,99999.0
+++ esfm,Urban II,0,59.0297968076411,11.008489348760156,17.600594623525765,5.400933781867582,96,8889.69,8878.01452589035,99999.0
++++esfm,Urban II,1,58.947927470837534,11.025374954076108,17.465515658589908,5.68378594242137,96,8885.17,8876.618469953537,99999.0
++++esfm,Urban II,2,58.77798771298748,10.982786678935327,17.546556385759054,5.767405629316944,96,8893.99,8885.34243106842,99999.0
+++ esfm,Vercingetorix,0,79.07478057258302,9.776081681696928,6.649989325889541,1.7432748253588333,69,1223.75,1212.340968608856,99999.0
+++ esfm,Vercingetorix,1,80.09313371434186,9.858626097407651,6.426263444651325,1.6979463697629655,69,1222.8,1213.763489961624,99999.0
+++ esfm,Vercingetorix,2,89.68787080902364,10.950615257181747,8.695412468830675,3.360645595149667,69,1225.36,1215.808262825012,99999.0
+++-uesfm,Vercingetorix,0,78.95942988866727,10.356244187384963,8.07065603849916,3.372297052368834,69,3105.42,3085.208213567734,99999.0
+++-uesfm,Vercingetorix,1,89.99175121301754,11.178484677344343,8.929255781006072,4.302365177465691,69,3105.88,3091.564247369766,99999.0
+++-uesfm,Vercingetorix,2,89.31151857112468,11.134780185118505,9.044030033429575,4.351952473195928,69,3105.52,3089.0203332901,99999.0
+++ esfm,Yueh Hai Ching Temple Singapore,0,0.6268598783684821,0.08360284283022794,5.56223941139456,2.132543748253723,43,1749.28,1738.747783660889,99999.0
++++esfm,Yueh Hai Ching Temple Singapore,1,0.5064956072004329,0.07235771132992698,5.30294026596974,2.06113770612239,43,1750.4,1739.704204082489,99999.0
++++esfm,Yueh Hai Ching Temple Singapore,2,0.49998151063409024,0.07009863154178562,5.92393490371475,2.2317763685579015,43,1746.32,1738.905562877655,99999.0
+++diff --git a/code/results/single_scene/summary_table.md b/code/results/single_scene/summary_table.md
+++index bd3f673..803d0c2 100644
+++--- a/code/results/single_scene/summary_table.md
++++++ b/code/results/single_scene/summary_table.md
+++@@ -2,44 +2,45 @@
+++ 
+++ Mean over seeds per (scene, method); best value per scene in **bold** (lower is better except Nr).
+++ 
+++-| Scene | ESFM Rot (deg) | U-ESFM Rot (deg) | ESFM (paper) Rot (deg) | ESFM Trans | U-ESFM Trans | ESFM (paper) Trans | ESFM Reproj (px) | U-ESFM Reproj (px) | ESFM (paper) Reproj (px) | ESFM Nr | U-ESFM Nr | ESFM (paper) Nr | ESFM Time (s) | U-ESFM Time (s) | ESFM (paper) Time (s) | ESFM Rot-BA (deg) | U-ESFM Rot-BA (deg) | ESFM (paper) Rot-BA (deg) | ESFM Trans-BA | U-ESFM Trans-BA | ESFM (paper) Trans-BA | ESFM Reproj-BA (px) | U-ESFM Reproj-BA (px) | ESFM (paper) Reproj-BA (px) |
++++| Scene | ESFM (official code) Rot (deg) | U-ESFM Rot (deg) | ESFM (paper) Rot (deg) | ESFM (official code) Trans | U-ESFM Trans | ESFM (paper) Trans | ESFM (official code) Reproj (px) | U-ESFM Reproj (px) | ESFM (paper) Reproj (px) | ESFM (official code) Nr | U-ESFM Nr | ESFM (paper) Nr | ESFM (official code) Time (s) | U-ESFM Time (s) | ESFM (paper) Time (s) | ESFM (official code) Rot-BA (deg) | U-ESFM Rot-BA (deg) | ESFM (paper) Rot-BA (deg) | ESFM (official code) Trans-BA | U-ESFM Trans-BA | ESFM (paper) Trans-BA | ESFM (official code) Reproj-BA (px) | U-ESFM Reproj-BA (px) | ESFM (paper) Reproj-BA (px) |
+++ |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
+++-| Alcatraz Courtyard | **0.362** | 0.616 | 0.619 | **0.093** | 0.161 | 0.160 | **3.435** | 5.796 | 1.640 | **133** | 133 | -- | **4826.713** | 6481.757 | -- | -- | -- | 0.049 | -- | -- | 0.015 | -- | -- | 0.810 |
+++-| Alcatraz Water Tower | **0.666** | 7.413 | 0.933 | **0.370** | 3.437 | 0.518 | **4.763** | 8.607 | 2.130 | **172** | 172 | -- | **2461.337** | 3854.907 | -- | -- | -- | 0.230 | -- | -- | 0.116 | -- | -- | 0.550 |
+++-| Buddah Tooth Relic Temple Singapore | **1.131** | 12.080 | 1.030 | **0.280** | 2.308 | 0.233 | **9.964** | 19.634 | 2.060 | **162** | 162 | -- | **2916.837** | 4320.873 | -- | -- | -- | 0.081 | -- | -- | 0.014 | -- | -- | 0.850 |
+++-| Doge Palace Venice | **0.801** | 0.911 | 1.163 | **0.218** | 0.290 | 0.342 | **6.432** | 10.966 | 3.620 | **241** | 241 | -- | 28768.725 | **21074.950** | -- | -- | -- | 0.211 | -- | -- | 0.029 | -- | -- | 1 |
+++-| Door Lund | **0.014** | 14.172 | 0.024 | **0.004** | 0.952 | 0.006 | **1.812** | 11.922 | 0.320 | **12** | 12 | -- | **4888.363** | 9188.593 | -- | -- | -- | 0.006 | -- | -- | 0.001 | -- | -- | 0.300 |
+++-| Drinking Fountain Somewhere In Zurich | **17.193** | 17.536 | 0.031 | 0.753 | **0.740** | 0.004 | **6.058** | 6.511 | 0.330 | **14** | 14 | -- | **883.837** | 2963.647 | -- | -- | -- | 0.007 | -- | -- | 0.002 | -- | -- | 0.310 |
+++-| East Indiaman Goteborg | 5.568 | **5.310** | 3.814 | 0.979 | **0.959** | 0.621 | **7.318** | 9.840 | 4.130 | **179** | 179 | -- | **3987.183** | 5241.430 | -- | -- | -- | 3.117 | -- | -- | 0.509 | -- | -- | 1.850 |
+++-| Ecole Superior De Guerre | 0.349 | **0.284** | 0.318 | 0.090 | **0.077** | 0.081 | **3.418** | 5.349 | 0.720 | **35** | 35 | -- | **2380.387** | 4633.833 | -- | -- | -- | 0.024 | -- | -- | 0.005 | -- | -- | 0.340 |
+++-| Eglise du dome | **0.351** | 45.801 | 0.808 | **0.088** | 6.451 | 0.205 | **3.799** | 7.964 | 0.910 | **85** | 85 | -- | 25099.450 | **23371.730** | -- | -- | -- | 0.037 | -- | -- | 0.010 | -- | -- | 0.270 |
+++-| Folke Filbyter | 85.831 | **81.288** | 74.596 | **0.129** | 0.130 | 0.125 | **19.275** | 23.156 | 10.370 | **40** | 40 | -- | **1581.897** | 3703.187 | -- | -- | -- | 70.157 | -- | -- | 0.118 | -- | -- | 4.290 |
+++-| Fort Channing Gate Singapore | **0.093** | 11.128 | 0.207 | **0.041** | 4.004 | 0.093 | **1.783** | 6.651 | 0.520 | **27** | 27 | -- | **3434.113** | 6172.087 | -- | -- | -- | 0.020 | -- | -- | 0.008 | -- | -- | 0.250 |
+++-| Golden Statue Somewhere In Hong Kong | **0.232** | 0.371 | 0.292 | **0.053** | 0.074 | 0.073 | **1.829** | 2.835 | 0.400 | **18** | 18 | -- | **4095.853** | 7762.437 | -- | -- | -- | 0.031 | -- | -- | 0.004 | -- | -- | 0.270 |
+++-| Gustav Vasa | **3.509** | 6.099 | 34.181 | **0.220** | 0.345 | 1.085 | **1.655** | 3.087 | 3.520 | **18** | 18 | -- | **709.867** | 2543.553 | -- | -- | -- | 32.266 | -- | -- | 1.145 | -- | -- | 3.150 |
+++-| GustavIIAdolf | **47.126** | 77.423 | 67.784 | **8.536** | 11.962 | 9.714 | **11.722** | 15.851 | 13.910 | **57** | 57 | -- | **1044.170** | 2892.580 | -- | -- | -- | 58.458 | -- | -- | 8.524 | -- | -- | 11.490 |
+++-| Jonas Ahlstromer | 48.740 | **42.246** | 50.190 | **9.823** | 10.300 | 10.888 | 11.136 | **8.782** | 10.820 | **40** | 40 | -- | **615.450** | 2328.313 | -- | -- | -- | 47.117 | -- | -- | 10.451 | -- | -- | 8.410 |
+++-| Kings College University Of Toronto | **9.025** | 21.523 | 0.989 | 2.044 | **1.905** | 0.235 | **3.565** | 5.525 | 0.900 | **77** | 77 | -- | **1078.980** | 2886.243 | -- | -- | -- | 0.085 | -- | -- | 0.017 | -- | -- | 0.340 |
+++-| Lund University Sphinx | **11.392** | 33.023 | 19.522 | **2.981** | 6.889 | 4.585 | **6.056** | 12.060 | 4.780 | **70** | 70 | -- | **2718.567** | 4654.610 | -- | -- | -- | 8.752 | -- | -- | 2.191 | -- | -- | 1.360 |
+++-| Nijo Castle Gate | **0.839** | 1.682 | 1.495 | **0.165** | 0.326 | 0.286 | **7.415** | 12.138 | 1.700 | **19** | 19 | -- | **1095.177** | 3306.697 | -- | -- | -- | 0.069 | -- | -- | 0.012 | -- | -- | 0.730 |
+++-| Pantheon Paris | -- | 19.345 | 0.192 | -- | 3.065 | 0.050 | -- | 13.562 | 1.470 | -- | 179 | -- | -- | 5497.377 | -- | -- | -- | 0.040 | -- | -- | 0.005 | -- | -- | 0.490 |
+++-| Park Gate Clermont Ferrand | 21.388 | **19.302** | 0.391 | 10.192 | **9.397** | 0.125 | **7.712** | 7.852 | 0.570 | **34** | 34 | -- | **1596.940** | 3597.403 | -- | -- | -- | 0.049 | -- | -- | 0.022 | -- | -- | 0.350 |
+++-| Plaza De Armas Santiago | **0.775** | 3.327 | 6.782 | **0.337** | 1.435 | 2.944 | **6.009** | 9.463 | 7.400 | **240** | 240 | -- | **6372.683** | 6535.537 | -- | -- | -- | 2.556 | -- | -- | 1.383 | -- | -- | 4.900 |
+++-| Porta San Donato Bologna | **0.592** | 21.006 | 2.153 | **0.107** | 3.541 | 0.388 | **5.603** | 13.244 | 2.280 | **141** | 141 | -- | **3754.163** | 5118.030 | -- | -- | -- | 0.095 | -- | -- | 0.046 | -- | -- | 0.750 |
+++-| Round Church Cambridge | **2.129** | 2.490 | 2.451 | **0.920** | 1.002 | 1.003 | **6.097** | 8.546 | 2.660 | **92** | 92 | -- | **9776.630** | 12061.350 | -- | -- | -- | 1.107 | -- | -- | 0.582 | -- | -- | 1.540 |
+++-| Skansen Kronan Gothenburg | **0.338** | 0.375 | 0.736 | **0.116** | 0.124 | 0.226 | **3.048** | 4.799 | 1.240 | **131** | 131 | -- | 17163.940 | **16235.090** | -- | -- | -- | 0.026 | -- | -- | 0.008 | -- | -- | 0.670 |
+++-| Smolny Cathedral St Petersburg | **21.806** | 22.143 | 0.554 | 2.111 | **2.099** | 0.051 | **11.228** | 12.526 | 1.660 | **131** | 131 | -- | **10362.595** | 10475.740 | -- | -- | -- | 0.033 | -- | -- | 0.006 | -- | -- | 0.810 |
+++-| Sri Mariamman Singapore | **1.219** | 6.479 | 2.302 | **0.382** | 2.188 | 0.683 | **10.088** | 17.854 | 4.130 | **222** | 222 | -- | **5985.540** | 6485.667 | -- | -- | -- | 0.077 | -- | -- | 0.023 | -- | -- | 0.910 |
+++-| Sri Thendayuthapani Singapore | **0.719** | 47.075 | 46.269 | **0.179** | 3.855 | 3.812 | **7.124** | 14.133 | 23.370 | **98** | 98 | -- | **13362.090** | 13808.440 | -- | -- | -- | 44.170 | -- | -- | 2.870 | -- | -- | 8.440 |
+++-| Sri Veeramakaliamman Singapore | **2.334** | 29.405 | 2.559 | **0.543** | 3.868 | 0.597 | **11.965** | 28.951 | 3.470 | **157** | 157 | -- | **12392.180** | 12544.690 | -- | -- | -- | 0.175 | -- | -- | 0.040 | -- | -- | 0.730 |
+++-| Statue Of Liberty | 52.358 | **39.859** | 46.887 | 28.503 | **18.503** | 20.012 | 235036.825 | **41.028** | 26.160 | **134** | 134 | -- | **99.900** | 12924 | -- | -- | -- | 9.091 | -- | -- | 4.122 | -- | -- | 6.970 |
+++-| The Pumpkin | **25.731** | 26.571 | 94.672 | **5.552** | 5.665 | 14.890 | **26.876** | 35.367 | 33.410 | **196** | 196 | -- | **4327.273** | 5790.597 | -- | -- | -- | 98.862 | -- | -- | 14.952 | -- | -- | 24.850 |
+++-| Thian Hook Keng Temple Singapore | **0.927** | 1.201 | 0.832 | **0.093** | 0.121 | 0.082 | **14.915** | 20.789 | 2.750 | **138** | 138 | -- | **3298.537** | 4612.373 | -- | -- | -- | 0.081 | -- | -- | 0.008 | -- | -- | 1.130 |
+++-| Tsar Nikolai I | **42.803** | 78.413 | 48.499 | **8.927** | 15.394 | 9.467 | **11.124** | 18.431 | 9.790 | **98** | 98 | -- | **4928.430** | 6646.845 | -- | -- | -- | 36.280 | -- | -- | 7.836 | -- | -- | 6.530 |
+++-| Urban II | 59.030 | -- | 47.490 | 11.008 | -- | 9.467 | 17.601 | -- | 9.380 | 96 | -- | -- | 8889.690 | -- | -- | -- | -- | 48.214 | -- | -- | 9.586 | -- | -- | 6.920 |
+++-| Vercingetorix | **82.952** | 86.088 | 69.328 | **10.195** | 10.890 | 8.788 | **7.257** | 8.681 | 5.080 | **69** | 69 | -- | **1223.970** | 3105.607 | -- | -- | -- | 17.706 | -- | -- | 3.104 | -- | -- | 1.500 |
+++-| Yueh Hai Ching Temple Singapore | 0.627 | -- | 0.720 | 0.084 | -- | 0.098 | 5.562 | -- | 0.940 | 43 | -- | -- | 1749.280 | -- | -- | -- | -- | 0.043 | -- | -- | 0.014 | -- | -- | 0.650 |
+++-| **Mean** | **16.146** | 23.697 | 18.023 | **3.121** | 4.014 | 2.912 | 6920.602 | **13.088** | 5.673 | 100.559 | **104.818** | -- | **5819.728** | 7358.187 | -- | -- | -- | 13.695 | -- | -- | 1.937 | -- | -- | 2.992 |
++++| Alcatraz Courtyard | **0.362** | 10.132 | 0.619 | **0.093** | 2.219 | 0.160 | **3.435** | 7.272 | 1.640 | **133** | 133 | -- | **4826.713** | 6550.983 | -- | -- | -- | 0.049 | -- | -- | 0.015 | -- | -- | 0.810 |
++++| Alcatraz Water Tower | **0.666** | 1.480 | 0.933 | **0.370** | 0.803 | 0.518 | **4.763** | 7.397 | 2.130 | **172** | 172 | -- | **2461.337** | 3795.877 | -- | -- | -- | 0.230 | -- | -- | 0.116 | -- | -- | 0.550 |
++++| Buddah Tooth Relic Temple Singapore | **1.131** | 1.791 | 1.030 | **0.280** | 0.520 | 0.233 | **9.964** | 14.029 | 2.060 | **162** | 162 | -- | **2916.837** | 4289.503 | -- | -- | -- | 0.081 | -- | -- | 0.014 | -- | -- | 0.850 |
++++| Doge Palace Venice | **0.753** | 1.121 | 1.163 | **0.209** | 0.350 | 0.342 | **6.359** | 10.496 | 3.620 | **241** | 241 | -- | 22522.737 | **20767.260** | -- | -- | -- | 0.211 | -- | -- | 0.029 | -- | -- | 1 |
++++| Door Lund | **0.014** | 6.960 | 0.024 | **0.004** | 0.512 | 0.006 | **1.812** | 7.101 | 0.320 | **12** | 12 | -- | **4888.363** | 9252.735 | -- | -- | -- | 0.006 | -- | -- | 0.001 | -- | -- | 0.300 |
++++| Drinking Fountain Somewhere In Zurich | **17.193** | 17.329 | 0.031 | 0.753 | **0.750** | 0.004 | **6.058** | 6.675 | 0.330 | **14** | 14 | -- | **883.837** | 5815.220 | -- | -- | -- | 0.007 | -- | -- | 0.002 | -- | -- | 0.310 |
++++| East Indiaman Goteborg | **5.568** | 7.231 | 3.814 | **0.979** | 1.364 | 0.621 | **7.318** | 10.130 | 4.130 | **179** | 179 | -- | **3987.183** | 5198.093 | -- | -- | -- | 3.117 | -- | -- | 0.509 | -- | -- | 1.850 |
++++| Ecole Superior De Guerre | **0.349** | 25.800 | 0.318 | **0.090** | 1.324 | 0.081 | **3.418** | 8.646 | 0.720 | **35** | 35 | -- | **2380.387** | 10416.080 | -- | -- | -- | 0.024 | -- | -- | 0.005 | -- | -- | 0.340 |
++++| Eglise du dome | **0.386** | 0.773 | 0.808 | **0.098** | 0.213 | 0.205 | **3.987** | 6.018 | 0.910 | **85** | 85 | -- | 19631.293 | **9796.445** | -- | -- | -- | 0.037 | -- | -- | 0.010 | -- | -- | 0.270 |
++++| Folke Filbyter | 85.831 | **82.656** | 74.596 | **0.129** | 0.131 | 0.125 | **19.275** | 28.225 | 10.370 | **40** | 40 | -- | **1581.897** | 3755.193 | -- | -- | -- | 70.157 | -- | -- | 0.118 | -- | -- | 4.290 |
++++| Fort Channing Gate Singapore | **0.093** | 0.154 | 0.207 | **0.041** | 0.071 | 0.093 | **1.783** | 3.715 | 0.520 | **27** | 27 | -- | **3434.113** | 6171.250 | -- | -- | -- | 0.020 | -- | -- | 0.008 | -- | -- | 0.250 |
++++| Golden Statue Somewhere In Hong Kong | 0.232 | -- | 0.292 | 0.053 | -- | 0.073 | 1.829 | -- | 0.400 | 18 | -- | -- | 4095.853 | -- | -- | -- | -- | 0.031 | -- | -- | 0.004 | -- | -- | 0.270 |
++++| Gustav Vasa | **3.509** | 25.854 | 34.181 | **0.220** | 0.775 | 1.085 | **1.655** | 4.747 | 3.520 | **18** | 18 | -- | **709.867** | 3887.653 | -- | -- | -- | 32.266 | -- | -- | 1.145 | -- | -- | 3.150 |
++++| GustavIIAdolf | **47.126** | 88.194 | 67.784 | **8.536** | 13.129 | 9.714 | **11.722** | 17.220 | 13.910 | **57** | 57 | -- | **1044.170** | 2981.017 | -- | -- | -- | 58.458 | -- | -- | 8.524 | -- | -- | 11.490 |
++++| Jonas Ahlstromer | 48.740 | **46.584** | 50.190 | **9.823** | 10.245 | 10.888 | **11.136** | 11.144 | 10.820 | **40** | 40 | -- | **615.450** | 2414.653 | -- | -- | -- | 47.117 | -- | -- | 10.451 | -- | -- | 8.410 |
++++| Kings College University Of Toronto | **9.025** | 14.284 | 0.989 | 2.044 | **1.639** | 0.235 | **3.565** | 6.351 | 0.900 | **77** | 77 | -- | **1078.980** | 2857.733 | -- | -- | -- | 0.085 | -- | -- | 0.017 | -- | -- | 0.340 |
++++| Lund University Sphinx | **11.392** | 28.580 | 19.522 | **2.981** | 6.117 | 4.585 | **6.056** | 12.912 | 4.780 | **70** | 70 | -- | **2718.567** | 4661.290 | -- | -- | -- | 8.752 | -- | -- | 2.191 | -- | -- | 1.360 |
++++| Nijo Castle Gate | **0.839** | 4.973 | 1.495 | **0.165** | 1.530 | 0.286 | **7.415** | 16.324 | 1.700 | **19** | 19 | -- | **1095.177** | 3237.995 | -- | -- | -- | 0.069 | -- | -- | 0.012 | -- | -- | 0.730 |
++++| Pantheon Paris | -- | 15.196 | 0.192 | -- | 2.291 | 0.050 | -- | 10.638 | 1.470 | -- | 179 | -- | -- | 5557.600 | -- | -- | -- | 0.040 | -- | -- | 0.005 | -- | -- | 0.490 |
++++| Park Gate Clermont Ferrand | **21.388** | 25.789 | 0.391 | **10.192** | 12.048 | 0.125 | **7.712** | 8.910 | 0.570 | **34** | 34 | -- | **1596.940** | 3653.795 | -- | -- | -- | 0.049 | -- | -- | 0.022 | -- | -- | 0.350 |
++++| Plaza De Armas Santiago | 0.775 | -- | 6.782 | 0.337 | -- | 2.944 | 6.009 | -- | 7.400 | 240 | -- | -- | 6372.683 | -- | -- | -- | -- | 2.556 | -- | -- | 1.383 | -- | -- | 4.900 |
++++| Porta San Donato Bologna | 0.592 | -- | 2.153 | 0.107 | -- | 0.388 | 5.603 | -- | 2.280 | 141 | -- | -- | 3754.163 | -- | -- | -- | -- | 0.095 | -- | -- | 0.046 | -- | -- | 0.750 |
++++| Round Church Cambridge | 2.137 | -- | 2.451 | 0.927 | -- | 1.003 | 6.103 | -- | 2.660 | 92 | -- | -- | 9779.270 | -- | -- | -- | -- | 1.107 | -- | -- | 0.582 | -- | -- | 1.540 |
++++| Skansen Kronan Gothenburg | 0.301 | -- | 0.736 | 0.102 | -- | 0.226 | 2.906 | -- | 1.240 | 131 | -- | -- | 17153.677 | -- | -- | -- | -- | 0.026 | -- | -- | 0.008 | -- | -- | 0.670 |
++++| Smolny Cathedral St Petersburg | 21.775 | -- | 0.554 | 2.111 | -- | 0.051 | 11.239 | -- | 1.660 | 131 | -- | -- | 10362.787 | -- | -- | -- | -- | 0.033 | -- | -- | 0.006 | -- | -- | 0.810 |
++++| Some Cathedral In Barcelona | 0.651 | -- | 0.880 | 0.233 | -- | 0.315 | 6.271 | -- | 2.870 | 177 | -- | -- | 5001.973 | -- | -- | -- | -- | 0.026 | -- | -- | 0.011 | -- | -- | 0.890 |
++++| Sri Mariamman Singapore | 1.219 | -- | 2.302 | 0.382 | -- | 0.683 | 10.088 | -- | 4.130 | 222 | -- | -- | 5985.540 | -- | -- | -- | -- | 0.077 | -- | -- | 0.023 | -- | -- | 0.910 |
++++| Sri Thendayuthapani Singapore | 0.593 | -- | 46.269 | 0.150 | -- | 3.812 | 7.016 | -- | 23.370 | 98 | -- | -- | 13338.167 | -- | -- | -- | -- | 44.170 | -- | -- | 2.870 | -- | -- | 8.440 |
++++| Sri Veeramakaliamman Singapore | 2.404 | -- | 2.559 | 0.559 | -- | 0.597 | 12.234 | -- | 3.470 | 157 | -- | -- | 12377.927 | -- | -- | -- | -- | 0.175 | -- | -- | 0.040 | -- | -- | 0.730 |
++++| Statue Of Liberty | 52.358 | -- | 46.887 | 28.503 | -- | 20.012 | 235036.825 | -- | 26.160 | 134 | -- | -- | 99.900 | -- | -- | -- | -- | 9.091 | -- | -- | 4.122 | -- | -- | 6.970 |
++++| The Pumpkin | 25.731 | -- | 94.672 | 5.552 | -- | 14.890 | 26.876 | -- | 33.410 | 196 | -- | -- | 4327.273 | -- | -- | -- | -- | 98.862 | -- | -- | 14.952 | -- | -- | 24.850 |
++++| Thian Hook Keng Temple Singapore | 0.927 | -- | 0.832 | 0.093 | -- | 0.082 | 14.915 | -- | 2.750 | 138 | -- | -- | 3298.537 | -- | -- | -- | -- | 0.081 | -- | -- | 0.008 | -- | -- | 1.130 |
++++| Tsar Nikolai I | 42.166 | -- | 48.499 | 8.732 | -- | 9.467 | 11.210 | -- | 9.790 | 98 | -- | -- | 4929.200 | -- | -- | -- | -- | 36.280 | -- | -- | 7.836 | -- | -- | 6.530 |
++++| Urban II | 58.919 | -- | 47.490 | 11.006 | -- | 9.467 | 17.538 | -- | 9.380 | 96 | -- | -- | 8889.617 | -- | -- | -- | -- | 48.214 | -- | -- | 9.586 | -- | -- | 6.920 |
++++| Vercingetorix | 82.952 | -- | 69.328 | 10.195 | -- | 8.788 | 7.257 | -- | 5.080 | 69 | -- | -- | 1223.970 | -- | -- | -- | -- | 17.706 | -- | -- | 3.104 | -- | -- | 1.500 |
++++| Yueh Hai Ching Temple Singapore | 0.544 | -- | 0.720 | 0.075 | -- | 0.098 | 5.596 | -- | 0.940 | 43 | -- | -- | 1748.667 | -- | -- | -- | -- | 0.043 | -- | -- | 0.014 | -- | -- | 0.650 |
++++| **Mean** | **15.676** | 21.310 | 17.547 | 3.032 | **2.949** | 2.840 | 6723.056 | **10.418** | 5.595 | **102.743** | 83.895 | -- | **5460.373** | 6055.809 | -- | -- | -- | 13.315 | -- | -- | 1.883 | -- | -- | 2.933 |
+++ 
+++ _Seeds per cell: [0, 1, 2]._
+++ 
+++diff --git a/code/results/single_scene/summary_table.tex b/code/results/single_scene/summary_table.tex
+++index 258e3f2..76cdb32 100644
+++--- a/code/results/single_scene/summary_table.tex
++++++ b/code/results/single_scene/summary_table.tex
+++@@ -7,45 +7,46 @@
+++ \begin{tabular}{lcccccccccccccccccccccccc}
+++ \toprule
+++  & \multicolumn{3}{c}{Rot (deg)} & \multicolumn{3}{c}{Trans} & \multicolumn{3}{c}{Reproj (px)} & \multicolumn{3}{c}{Nr} & \multicolumn{3}{c}{Time (s)} & \multicolumn{3}{c}{Rot-BA (deg)} & \multicolumn{3}{c}{Trans-BA} & \multicolumn{3}{c}{Reproj-BA (px)} \\
+++-Scene & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) \\
++++Scene & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) \\
+++ \midrule
+++-Alcatraz Courtyard & \textbf{0.362} & 0.616 & 0.619 & \textbf{0.093} & 0.161 & 0.160 & \textbf{3.435} & 5.796 & 1.640 & \textbf{133} & 133 & -- & \textbf{4826.713} & 6481.757 & -- & -- & -- & 0.049 & -- & -- & 0.015 & -- & -- & 0.810 \\
+++-Alcatraz Water Tower & \textbf{0.666} & 7.413 & 0.933 & \textbf{0.370} & 3.437 & 0.518 & \textbf{4.763} & 8.607 & 2.130 & \textbf{172} & 172 & -- & \textbf{2461.337} & 3854.907 & -- & -- & -- & 0.230 & -- & -- & 0.116 & -- & -- & 0.550 \\
+++-Buddah Tooth Relic Temple Singapore & \textbf{1.131} & 12.080 & 1.030 & \textbf{0.280} & 2.308 & 0.233 & \textbf{9.964} & 19.634 & 2.060 & \textbf{162} & 162 & -- & \textbf{2916.837} & 4320.873 & -- & -- & -- & 0.081 & -- & -- & 0.014 & -- & -- & 0.850 \\
+++-Doge Palace Venice & \textbf{0.801} & 0.911 & 1.163 & \textbf{0.218} & 0.290 & 0.342 & \textbf{6.432} & 10.966 & 3.620 & \textbf{241} & 241 & -- & 28768.725 & \textbf{21074.950} & -- & -- & -- & 0.211 & -- & -- & 0.029 & -- & -- & 1 \\
+++-Door Lund & \textbf{0.014} & 14.172 & 0.024 & \textbf{0.004} & 0.952 & 0.006 & \textbf{1.812} & 11.922 & 0.320 & \textbf{12} & 12 & -- & \textbf{4888.363} & 9188.593 & -- & -- & -- & 0.006 & -- & -- & 0.001 & -- & -- & 0.300 \\
+++-Drinking Fountain Somewhere In Zurich & \textbf{17.193} & 17.536 & 0.031 & 0.753 & \textbf{0.740} & 0.004 & \textbf{6.058} & 6.511 & 0.330 & \textbf{14} & 14 & -- & \textbf{883.837} & 2963.647 & -- & -- & -- & 0.007 & -- & -- & 0.002 & -- & -- & 0.310 \\
+++-East Indiaman Goteborg & 5.568 & \textbf{5.310} & 3.814 & 0.979 & \textbf{0.959} & 0.621 & \textbf{7.318} & 9.840 & 4.130 & \textbf{179} & 179 & -- & \textbf{3987.183} & 5241.430 & -- & -- & -- & 3.117 & -- & -- & 0.509 & -- & -- & 1.850 \\
+++-Ecole Superior De Guerre & 0.349 & \textbf{0.284} & 0.318 & 0.090 & \textbf{0.077} & 0.081 & \textbf{3.418} & 5.349 & 0.720 & \textbf{35} & 35 & -- & \textbf{2380.387} & 4633.833 & -- & -- & -- & 0.024 & -- & -- & 0.005 & -- & -- & 0.340 \\
+++-Eglise du dome & \textbf{0.351} & 45.801 & 0.808 & \textbf{0.088} & 6.451 & 0.205 & \textbf{3.799} & 7.964 & 0.910 & \textbf{85} & 85 & -- & 25099.450 & \textbf{23371.730} & -- & -- & -- & 0.037 & -- & -- & 0.010 & -- & -- & 0.270 \\
+++-Folke Filbyter & 85.831 & \textbf{81.288} & 74.596 & \textbf{0.129} & 0.130 & 0.125 & \textbf{19.275} & 23.156 & 10.370 & \textbf{40} & 40 & -- & \textbf{1581.897} & 3703.187 & -- & -- & -- & 70.157 & -- & -- & 0.118 & -- & -- & 4.290 \\
+++-Fort Channing Gate Singapore & \textbf{0.093} & 11.128 & 0.207 & \textbf{0.041} & 4.004 & 0.093 & \textbf{1.783} & 6.651 & 0.520 & \textbf{27} & 27 & -- & \textbf{3434.113} & 6172.087 & -- & -- & -- & 0.020 & -- & -- & 0.008 & -- & -- & 0.250 \\
+++-Golden Statue Somewhere In Hong Kong & \textbf{0.232} & 0.371 & 0.292 & \textbf{0.053} & 0.074 & 0.073 & \textbf{1.829} & 2.835 & 0.400 & \textbf{18} & 18 & -- & \textbf{4095.853} & 7762.437 & -- & -- & -- & 0.031 & -- & -- & 0.004 & -- & -- & 0.270 \\
+++-Gustav Vasa & \textbf{3.509} & 6.099 & 34.181 & \textbf{0.220} & 0.345 & 1.085 & \textbf{1.655} & 3.087 & 3.520 & \textbf{18} & 18 & -- & \textbf{709.867} & 2543.553 & -- & -- & -- & 32.266 & -- & -- & 1.145 & -- & -- & 3.150 \\
+++-GustavIIAdolf & \textbf{47.126} & 77.423 & 67.784 & \textbf{8.536} & 11.962 & 9.714 & \textbf{11.722} & 15.851 & 13.910 & \textbf{57} & 57 & -- & \textbf{1044.170} & 2892.580 & -- & -- & -- & 58.458 & -- & -- & 8.524 & -- & -- & 11.490 \\
+++-Jonas Ahlstromer & 48.740 & \textbf{42.246} & 50.190 & \textbf{9.823} & 10.300 & 10.888 & 11.136 & \textbf{8.782} & 10.820 & \textbf{40} & 40 & -- & \textbf{615.450} & 2328.313 & -- & -- & -- & 47.117 & -- & -- & 10.451 & -- & -- & 8.410 \\
+++-Kings College University Of Toronto & \textbf{9.025} & 21.523 & 0.989 & 2.044 & \textbf{1.905} & 0.235 & \textbf{3.565} & 5.525 & 0.900 & \textbf{77} & 77 & -- & \textbf{1078.980} & 2886.243 & -- & -- & -- & 0.085 & -- & -- & 0.017 & -- & -- & 0.340 \\
+++-Lund University Sphinx & \textbf{11.392} & 33.023 & 19.522 & \textbf{2.981} & 6.889 & 4.585 & \textbf{6.056} & 12.060 & 4.780 & \textbf{70} & 70 & -- & \textbf{2718.567} & 4654.610 & -- & -- & -- & 8.752 & -- & -- & 2.191 & -- & -- & 1.360 \\
+++-Nijo Castle Gate & \textbf{0.839} & 1.682 & 1.495 & \textbf{0.165} & 0.326 & 0.286 & \textbf{7.415} & 12.138 & 1.700 & \textbf{19} & 19 & -- & \textbf{1095.177} & 3306.697 & -- & -- & -- & 0.069 & -- & -- & 0.012 & -- & -- & 0.730 \\
+++-Pantheon Paris & -- & 19.345 & 0.192 & -- & 3.065 & 0.050 & -- & 13.562 & 1.470 & -- & 179 & -- & -- & 5497.377 & -- & -- & -- & 0.040 & -- & -- & 0.005 & -- & -- & 0.490 \\
+++-Park Gate Clermont Ferrand & 21.388 & \textbf{19.302} & 0.391 & 10.192 & \textbf{9.397} & 0.125 & \textbf{7.712} & 7.852 & 0.570 & \textbf{34} & 34 & -- & \textbf{1596.940} & 3597.403 & -- & -- & -- & 0.049 & -- & -- & 0.022 & -- & -- & 0.350 \\
+++-Plaza De Armas Santiago & \textbf{0.775} & 3.327 & 6.782 & \textbf{0.337} & 1.435 & 2.944 & \textbf{6.009} & 9.463 & 7.400 & \textbf{240} & 240 & -- & \textbf{6372.683} & 6535.537 & -- & -- & -- & 2.556 & -- & -- & 1.383 & -- & -- & 4.900 \\
+++-Porta San Donato Bologna & \textbf{0.592} & 21.006 & 2.153 & \textbf{0.107} & 3.541 & 0.388 & \textbf{5.603} & 13.244 & 2.280 & \textbf{141} & 141 & -- & \textbf{3754.163} & 5118.030 & -- & -- & -- & 0.095 & -- & -- & 0.046 & -- & -- & 0.750 \\
+++-Round Church Cambridge & \textbf{2.129} & 2.490 & 2.451 & \textbf{0.920} & 1.002 & 1.003 & \textbf{6.097} & 8.546 & 2.660 & \textbf{92} & 92 & -- & \textbf{9776.630} & 12061.350 & -- & -- & -- & 1.107 & -- & -- & 0.582 & -- & -- & 1.540 \\
+++-Skansen Kronan Gothenburg & \textbf{0.338} & 0.375 & 0.736 & \textbf{0.116} & 0.124 & 0.226 & \textbf{3.048} & 4.799 & 1.240 & \textbf{131} & 131 & -- & 17163.940 & \textbf{16235.090} & -- & -- & -- & 0.026 & -- & -- & 0.008 & -- & -- & 0.670 \\
+++-Smolny Cathedral St Petersburg & \textbf{21.806} & 22.143 & 0.554 & 2.111 & \textbf{2.099} & 0.051 & \textbf{11.228} & 12.526 & 1.660 & \textbf{131} & 131 & -- & \textbf{10362.595} & 10475.740 & -- & -- & -- & 0.033 & -- & -- & 0.006 & -- & -- & 0.810 \\
+++-Sri Mariamman Singapore & \textbf{1.219} & 6.479 & 2.302 & \textbf{0.382} & 2.188 & 0.683 & \textbf{10.088} & 17.854 & 4.130 & \textbf{222} & 222 & -- & \textbf{5985.540} & 6485.667 & -- & -- & -- & 0.077 & -- & -- & 0.023 & -- & -- & 0.910 \\
+++-Sri Thendayuthapani Singapore & \textbf{0.719} & 47.075 & 46.269 & \textbf{0.179} & 3.855 & 3.812 & \textbf{7.124} & 14.133 & 23.370 & \textbf{98} & 98 & -- & \textbf{13362.090} & 13808.440 & -- & -- & -- & 44.170 & -- & -- & 2.870 & -- & -- & 8.440 \\
+++-Sri Veeramakaliamman Singapore & \textbf{2.334} & 29.405 & 2.559 & \textbf{0.543} & 3.868 & 0.597 & \textbf{11.965} & 28.951 & 3.470 & \textbf{157} & 157 & -- & \textbf{12392.180} & 12544.690 & -- & -- & -- & 0.175 & -- & -- & 0.040 & -- & -- & 0.730 \\
+++-Statue Of Liberty & 52.358 & \textbf{39.859} & 46.887 & 28.503 & \textbf{18.503} & 20.012 & 235036.825 & \textbf{41.028} & 26.160 & \textbf{134} & 134 & -- & \textbf{99.900} & 12924 & -- & -- & -- & 9.091 & -- & -- & 4.122 & -- & -- & 6.970 \\
+++-The Pumpkin & \textbf{25.731} & 26.571 & 94.672 & \textbf{5.552} & 5.665 & 14.890 & \textbf{26.876} & 35.367 & 33.410 & \textbf{196} & 196 & -- & \textbf{4327.273} & 5790.597 & -- & -- & -- & 98.862 & -- & -- & 14.952 & -- & -- & 24.850 \\
+++-Thian Hook Keng Temple Singapore & \textbf{0.927} & 1.201 & 0.832 & \textbf{0.093} & 0.121 & 0.082 & \textbf{14.915} & 20.789 & 2.750 & \textbf{138} & 138 & -- & \textbf{3298.537} & 4612.373 & -- & -- & -- & 0.081 & -- & -- & 0.008 & -- & -- & 1.130 \\
+++-Tsar Nikolai I & \textbf{42.803} & 78.413 & 48.499 & \textbf{8.927} & 15.394 & 9.467 & \textbf{11.124} & 18.431 & 9.790 & \textbf{98} & 98 & -- & \textbf{4928.430} & 6646.845 & -- & -- & -- & 36.280 & -- & -- & 7.836 & -- & -- & 6.530 \\
+++-Urban II & 59.030 & -- & 47.490 & 11.008 & -- & 9.467 & 17.601 & -- & 9.380 & 96 & -- & -- & 8889.690 & -- & -- & -- & -- & 48.214 & -- & -- & 9.586 & -- & -- & 6.920 \\
+++-Vercingetorix & \textbf{82.952} & 86.088 & 69.328 & \textbf{10.195} & 10.890 & 8.788 & \textbf{7.257} & 8.681 & 5.080 & \textbf{69} & 69 & -- & \textbf{1223.970} & 3105.607 & -- & -- & -- & 17.706 & -- & -- & 3.104 & -- & -- & 1.500 \\
+++-Yueh Hai Ching Temple Singapore & 0.627 & -- & 0.720 & 0.084 & -- & 0.098 & 5.562 & -- & 0.940 & 43 & -- & -- & 1749.280 & -- & -- & -- & -- & 0.043 & -- & -- & 0.014 & -- & -- & 0.650 \\
++++Alcatraz Courtyard & \textbf{0.362} & 10.132 & 0.619 & \textbf{0.093} & 2.219 & 0.160 & \textbf{3.435} & 7.272 & 1.640 & \textbf{133} & 133 & -- & \textbf{4826.713} & 6550.983 & -- & -- & -- & 0.049 & -- & -- & 0.015 & -- & -- & 0.810 \\
++++Alcatraz Water Tower & \textbf{0.666} & 1.480 & 0.933 & \textbf{0.370} & 0.803 & 0.518 & \textbf{4.763} & 7.397 & 2.130 & \textbf{172} & 172 & -- & \textbf{2461.337} & 3795.877 & -- & -- & -- & 0.230 & -- & -- & 0.116 & -- & -- & 0.550 \\
++++Buddah Tooth Relic Temple Singapore & \textbf{1.131} & 1.791 & 1.030 & \textbf{0.280} & 0.520 & 0.233 & \textbf{9.964} & 14.029 & 2.060 & \textbf{162} & 162 & -- & \textbf{2916.837} & 4289.503 & -- & -- & -- & 0.081 & -- & -- & 0.014 & -- & -- & 0.850 \\
++++Doge Palace Venice & \textbf{0.753} & 1.121 & 1.163 & \textbf{0.209} & 0.350 & 0.342 & \textbf{6.359} & 10.496 & 3.620 & \textbf{241} & 241 & -- & 22522.737 & \textbf{20767.260} & -- & -- & -- & 0.211 & -- & -- & 0.029 & -- & -- & 1 \\
++++Door Lund & \textbf{0.014} & 6.960 & 0.024 & \textbf{0.004} & 0.512 & 0.006 & \textbf{1.812} & 7.101 & 0.320 & \textbf{12} & 12 & -- & \textbf{4888.363} & 9252.735 & -- & -- & -- & 0.006 & -- & -- & 0.001 & -- & -- & 0.300 \\
++++Drinking Fountain Somewhere In Zurich & \textbf{17.193} & 17.329 & 0.031 & 0.753 & \textbf{0.750} & 0.004 & \textbf{6.058} & 6.675 & 0.330 & \textbf{14} & 14 & -- & \textbf{883.837} & 5815.220 & -- & -- & -- & 0.007 & -- & -- & 0.002 & -- & -- & 0.310 \\
++++East Indiaman Goteborg & \textbf{5.568} & 7.231 & 3.814 & \textbf{0.979} & 1.364 & 0.621 & \textbf{7.318} & 10.130 & 4.130 & \textbf{179} & 179 & -- & \textbf{3987.183} & 5198.093 & -- & -- & -- & 3.117 & -- & -- & 0.509 & -- & -- & 1.850 \\
++++Ecole Superior De Guerre & \textbf{0.349} & 25.800 & 0.318 & \textbf{0.090} & 1.324 & 0.081 & \textbf{3.418} & 8.646 & 0.720 & \textbf{35} & 35 & -- & \textbf{2380.387} & 10416.080 & -- & -- & -- & 0.024 & -- & -- & 0.005 & -- & -- & 0.340 \\
++++Eglise du dome & \textbf{0.386} & 0.773 & 0.808 & \textbf{0.098} & 0.213 & 0.205 & \textbf{3.987} & 6.018 & 0.910 & \textbf{85} & 85 & -- & 19631.293 & \textbf{9796.445} & -- & -- & -- & 0.037 & -- & -- & 0.010 & -- & -- & 0.270 \\
++++Folke Filbyter & 85.831 & \textbf{82.656} & 74.596 & \textbf{0.129} & 0.131 & 0.125 & \textbf{19.275} & 28.225 & 10.370 & \textbf{40} & 40 & -- & \textbf{1581.897} & 3755.193 & -- & -- & -- & 70.157 & -- & -- & 0.118 & -- & -- & 4.290 \\
++++Fort Channing Gate Singapore & \textbf{0.093} & 0.154 & 0.207 & \textbf{0.041} & 0.071 & 0.093 & \textbf{1.783} & 3.715 & 0.520 & \textbf{27} & 27 & -- & \textbf{3434.113} & 6171.250 & -- & -- & -- & 0.020 & -- & -- & 0.008 & -- & -- & 0.250 \\
++++Golden Statue Somewhere In Hong Kong & 0.232 & -- & 0.292 & 0.053 & -- & 0.073 & 1.829 & -- & 0.400 & 18 & -- & -- & 4095.853 & -- & -- & -- & -- & 0.031 & -- & -- & 0.004 & -- & -- & 0.270 \\
++++Gustav Vasa & \textbf{3.509} & 25.854 & 34.181 & \textbf{0.220} & 0.775 & 1.085 & \textbf{1.655} & 4.747 & 3.520 & \textbf{18} & 18 & -- & \textbf{709.867} & 3887.653 & -- & -- & -- & 32.266 & -- & -- & 1.145 & -- & -- & 3.150 \\
++++GustavIIAdolf & \textbf{47.126} & 88.194 & 67.784 & \textbf{8.536} & 13.129 & 9.714 & \textbf{11.722} & 17.220 & 13.910 & \textbf{57} & 57 & -- & \textbf{1044.170} & 2981.017 & -- & -- & -- & 58.458 & -- & -- & 8.524 & -- & -- & 11.490 \\
++++Jonas Ahlstromer & 48.740 & \textbf{46.584} & 50.190 & \textbf{9.823} & 10.245 & 10.888 & \textbf{11.136} & 11.144 & 10.820 & \textbf{40} & 40 & -- & \textbf{615.450} & 2414.653 & -- & -- & -- & 47.117 & -- & -- & 10.451 & -- & -- & 8.410 \\
++++Kings College University Of Toronto & \textbf{9.025} & 14.284 & 0.989 & 2.044 & \textbf{1.639} & 0.235 & \textbf{3.565} & 6.351 & 0.900 & \textbf{77} & 77 & -- & \textbf{1078.980} & 2857.733 & -- & -- & -- & 0.085 & -- & -- & 0.017 & -- & -- & 0.340 \\
++++Lund University Sphinx & \textbf{11.392} & 28.580 & 19.522 & \textbf{2.981} & 6.117 & 4.585 & \textbf{6.056} & 12.912 & 4.780 & \textbf{70} & 70 & -- & \textbf{2718.567} & 4661.290 & -- & -- & -- & 8.752 & -- & -- & 2.191 & -- & -- & 1.360 \\
++++Nijo Castle Gate & \textbf{0.839} & 4.973 & 1.495 & \textbf{0.165} & 1.530 & 0.286 & \textbf{7.415} & 16.324 & 1.700 & \textbf{19} & 19 & -- & \textbf{1095.177} & 3237.995 & -- & -- & -- & 0.069 & -- & -- & 0.012 & -- & -- & 0.730 \\
++++Pantheon Paris & -- & 15.196 & 0.192 & -- & 2.291 & 0.050 & -- & 10.638 & 1.470 & -- & 179 & -- & -- & 5557.600 & -- & -- & -- & 0.040 & -- & -- & 0.005 & -- & -- & 0.490 \\
++++Park Gate Clermont Ferrand & \textbf{21.388} & 25.789 & 0.391 & \textbf{10.192} & 12.048 & 0.125 & \textbf{7.712} & 8.910 & 0.570 & \textbf{34} & 34 & -- & \textbf{1596.940} & 3653.795 & -- & -- & -- & 0.049 & -- & -- & 0.022 & -- & -- & 0.350 \\
++++Plaza De Armas Santiago & 0.775 & -- & 6.782 & 0.337 & -- & 2.944 & 6.009 & -- & 7.400 & 240 & -- & -- & 6372.683 & -- & -- & -- & -- & 2.556 & -- & -- & 1.383 & -- & -- & 4.900 \\
++++Porta San Donato Bologna & 0.592 & -- & 2.153 & 0.107 & -- & 0.388 & 5.603 & -- & 2.280 & 141 & -- & -- & 3754.163 & -- & -- & -- & -- & 0.095 & -- & -- & 0.046 & -- & -- & 0.750 \\
++++Round Church Cambridge & 2.137 & -- & 2.451 & 0.927 & -- & 1.003 & 6.103 & -- & 2.660 & 92 & -- & -- & 9779.270 & -- & -- & -- & -- & 1.107 & -- & -- & 0.582 & -- & -- & 1.540 \\
++++Skansen Kronan Gothenburg & 0.301 & -- & 0.736 & 0.102 & -- & 0.226 & 2.906 & -- & 1.240 & 131 & -- & -- & 17153.677 & -- & -- & -- & -- & 0.026 & -- & -- & 0.008 & -- & -- & 0.670 \\
++++Smolny Cathedral St Petersburg & 21.775 & -- & 0.554 & 2.111 & -- & 0.051 & 11.239 & -- & 1.660 & 131 & -- & -- & 10362.787 & -- & -- & -- & -- & 0.033 & -- & -- & 0.006 & -- & -- & 0.810 \\
++++Some Cathedral In Barcelona & 0.651 & -- & 0.880 & 0.233 & -- & 0.315 & 6.271 & -- & 2.870 & 177 & -- & -- & 5001.973 & -- & -- & -- & -- & 0.026 & -- & -- & 0.011 & -- & -- & 0.890 \\
++++Sri Mariamman Singapore & 1.219 & -- & 2.302 & 0.382 & -- & 0.683 & 10.088 & -- & 4.130 & 222 & -- & -- & 5985.540 & -- & -- & -- & -- & 0.077 & -- & -- & 0.023 & -- & -- & 0.910 \\
++++Sri Thendayuthapani Singapore & 0.593 & -- & 46.269 & 0.150 & -- & 3.812 & 7.016 & -- & 23.370 & 98 & -- & -- & 13338.167 & -- & -- & -- & -- & 44.170 & -- & -- & 2.870 & -- & -- & 8.440 \\
++++Sri Veeramakaliamman Singapore & 2.404 & -- & 2.559 & 0.559 & -- & 0.597 & 12.234 & -- & 3.470 & 157 & -- & -- & 12377.927 & -- & -- & -- & -- & 0.175 & -- & -- & 0.040 & -- & -- & 0.730 \\
++++Statue Of Liberty & 52.358 & -- & 46.887 & 28.503 & -- & 20.012 & 235036.825 & -- & 26.160 & 134 & -- & -- & 99.900 & -- & -- & -- & -- & 9.091 & -- & -- & 4.122 & -- & -- & 6.970 \\
++++The Pumpkin & 25.731 & -- & 94.672 & 5.552 & -- & 14.890 & 26.876 & -- & 33.410 & 196 & -- & -- & 4327.273 & -- & -- & -- & -- & 98.862 & -- & -- & 14.952 & -- & -- & 24.850 \\
++++Thian Hook Keng Temple Singapore & 0.927 & -- & 0.832 & 0.093 & -- & 0.082 & 14.915 & -- & 2.750 & 138 & -- & -- & 3298.537 & -- & -- & -- & -- & 0.081 & -- & -- & 0.008 & -- & -- & 1.130 \\
++++Tsar Nikolai I & 42.166 & -- & 48.499 & 8.732 & -- & 9.467 & 11.210 & -- & 9.790 & 98 & -- & -- & 4929.200 & -- & -- & -- & -- & 36.280 & -- & -- & 7.836 & -- & -- & 6.530 \\
++++Urban II & 58.919 & -- & 47.490 & 11.006 & -- & 9.467 & 17.538 & -- & 9.380 & 96 & -- & -- & 8889.617 & -- & -- & -- & -- & 48.214 & -- & -- & 9.586 & -- & -- & 6.920 \\
++++Vercingetorix & 82.952 & -- & 69.328 & 10.195 & -- & 8.788 & 7.257 & -- & 5.080 & 69 & -- & -- & 1223.970 & -- & -- & -- & -- & 17.706 & -- & -- & 3.104 & -- & -- & 1.500 \\
++++Yueh Hai Ching Temple Singapore & 0.544 & -- & 0.720 & 0.075 & -- & 0.098 & 5.596 & -- & 0.940 & 43 & -- & -- & 1748.667 & -- & -- & -- & -- & 0.043 & -- & -- & 0.014 & -- & -- & 0.650 \\
+++ \midrule
+++-Mean & \textbf{16.146} & 23.697 & 18.023 & \textbf{3.121} & 4.014 & 2.912 & 6920.602 & \textbf{13.088} & 5.673 & 100.559 & \textbf{104.818} & -- & \textbf{5819.728} & 7358.187 & -- & -- & -- & 13.695 & -- & -- & 1.937 & -- & -- & 2.992 \\
++++Mean & \textbf{15.676} & 21.310 & 17.547 & 3.032 & \textbf{2.949} & 2.840 & 6723.056 & \textbf{10.418} & 5.595 & \textbf{102.743} & 83.895 & -- & \textbf{5460.373} & 6055.809 & -- & -- & -- & 13.315 & -- & -- & 1.883 & -- & -- & 2.933 \\
+++ \bottomrule
+++ \end{tabular}}
+++ \end{table*}
++ ```
++ - untracked/modified files:
++ ```
+++M code/results/single_scene/REPRO.md
+++ M code/results/single_scene/summary.csv
+++ M code/results/single_scene/summary_table.md
+++ M code/results/single_scene/summary_table.tex
++ ?? MVG_Project_Report_Ortal_Dayan.pdf
++ ?? "RESFM- ROBUST DEEP EQUIVARIANT STRUCTURE FROM MOTION.pdf"
++ ?? "claude specs/SPEC_cvpr_experiments.md"
++ ?? "claude specs/SPEC_sfm_datasets_setup.md"
++ ?? "claude specs/SPEC_single_scene_experiments.md"
++ ?? "claude specs/SPEC_uesfm_combined.md"
+++?? "claude specs/TASK_crossdataset_readiness.md"
++ ?? code/datasets/Euclidean
++ ?? tmp_ab_check/
++ ```
++ ### esfm-baseline (official ESFM)
++ - path: `/home/projects/bagon/ortalda/MVG/final-project/esfm-baseline`
++-- commit: `2c0079497bf5dda1eeb2a902b2f859c6387f213a`
+++- commit: `de666af53423c1dc3dd412e53c4e153c03d70922`
++ - **WARNING: working tree dirty.** Diff:
++ ```diff
++ (untracked files only)
++@@ -58,6 +3062,7 @@ Generated: 2026-07-13T20:41:04
++ ?? code/confs/single_scene_generated/ss_esfm_Buddah_Tooth_Relic_Temple_Singapore_seed1.conf
++ ?? code/confs/single_scene_generated/ss_esfm_Buddah_Tooth_Relic_Temple_Singapore_seed2.conf
++ ?? code/confs/single_scene_generated/ss_esfm_Doge_Palace_Venice_seed1.conf
+++?? code/confs/single_scene_generated/ss_esfm_Doge_Palace_Venice_seed2.conf
++ ?? code/confs/single_scene_generated/ss_esfm_Door_Lund_seed1.conf
++ ?? code/confs/single_scene_generated/ss_esfm_Door_Lund_seed2.conf
++ ?? code/confs/single_scene_generated/ss_esfm_Drinking_Fountain_Somewhere_In_Zurich_seed1.conf
++@@ -70,6 +3075,7 @@ Generated: 2026-07-13T20:41:04
++ ?? code/confs/single_scene_generated/ss_esfm_Ecole_Superior_De_Guerre_seed2.conf
++ ?? code/confs/single_scene_generated/ss_esfm_Eglise_du_dome_seed0.conf
++ ?? code/confs/single_scene_generated/ss_esfm_Eglise_du_dome_seed1.conf
+++?? code/confs/single_scene_generated/ss_esfm_Eglise_du_dome_seed2.conf
++ ?? code/confs/single_scene_generated/ss_esfm_Folke_Filbyter_seed0.conf
++ ?? code/confs/single_scene_generated/ss_esfm_Folke_Filbyter_seed1.conf
++ ?? code/confs/single_scene_generated/ss_esfm_Folke_Filbyter_seed2.conf
++@@ -114,16 +3120,22 @@ Generated: 2026-07-13T20:41:04
++ ?? code/confs/single_scene_generated/ss_esfm_Round_Church_Cambridge_seed2.conf
++ ?? code/confs/single_scene_generated/ss_esfm_Skansen_Kronan_Gothenburg_seed0.conf
++ ?? code/confs/single_scene_generated/ss_esfm_Skansen_Kronan_Gothenburg_seed1.conf
+++?? code/confs/single_scene_generated/ss_esfm_Skansen_Kronan_Gothenburg_seed2.conf
++ ?? code/confs/single_scene_generated/ss_esfm_Smolny_Cathedral_St_Petersburg_seed0.conf
++ ?? code/confs/single_scene_generated/ss_esfm_Smolny_Cathedral_St_Petersburg_seed1.conf
+++?? code/confs/single_scene_generated/ss_esfm_Smolny_Cathedral_St_Petersburg_seed2.conf
++ ?? code/confs/single_scene_generated/ss_esfm_Some_Cathedral_In_Barcelona_seed0.conf
+++?? code/confs/single_scene_generated/ss_esfm_Some_Cathedral_In_Barcelona_seed1.conf
+++?? code/confs/single_scene_generated/ss_esfm_Some_Cathedral_In_Barcelona_seed2.conf
++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Mariamman_Singapore_seed0.conf
++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Mariamman_Singapore_seed1.conf
++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Mariamman_Singapore_seed2.conf
++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Thendayuthapani_Singapore_seed0.conf
++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Thendayuthapani_Singapore_seed1.conf
+++?? code/confs/single_scene_generated/ss_esfm_Sri_Thendayuthapani_Singapore_seed2.conf
++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Veeramakaliamman_Singapore_seed0.conf
++ ?? code/confs/single_scene_generated/ss_esfm_Sri_Veeramakaliamman_Singapore_seed1.conf
+++?? code/confs/single_scene_generated/ss_esfm_Sri_Veeramakaliamman_Singapore_seed2.conf
++ ?? code/confs/single_scene_generated/ss_esfm_Statue_Of_Liberty_seed0.conf
++ ?? code/confs/single_scene_generated/ss_esfm_Statue_Of_Liberty_seed1.conf
++ ?? code/confs/single_scene_generated/ss_esfm_Statue_Of_Liberty_seed2.conf
++@@ -137,10 +3149,14 @@ Generated: 2026-07-13T20:41:04
++ ?? code/confs/single_scene_generated/ss_esfm_Tsar_Nikolai_I_seed1.conf
++ ?? code/confs/single_scene_generated/ss_esfm_Tsar_Nikolai_I_seed2.conf
++ ?? code/confs/single_scene_generated/ss_esfm_Urban_II_seed0.conf
+++?? code/confs/single_scene_generated/ss_esfm_Urban_II_seed1.conf
+++?? code/confs/single_scene_generated/ss_esfm_Urban_II_seed2.conf
++ ?? code/confs/single_scene_generated/ss_esfm_Vercingetorix_seed0.conf
++ ?? code/confs/single_scene_generated/ss_esfm_Vercingetorix_seed1.conf
++ ?? code/confs/single_scene_generated/ss_esfm_Vercingetorix_seed2.conf
++ ?? code/confs/single_scene_generated/ss_esfm_Yueh_Hai_Ching_Temple_Singapore_seed0.conf
+++?? code/confs/single_scene_generated/ss_esfm_Yueh_Hai_Ching_Temple_Singapore_seed1.conf
+++?? code/confs/single_scene_generated/ss_esfm_Yueh_Hai_Ching_Temple_Singapore_seed2.conf
++ ```
++ 
++ ## Environment (shared venv used for BOTH methods)
++diff --git a/code/results/single_scene/summary.csv b/code/results/single_scene/summary.csv
++index e4f9f5e..d98886f 100644
++--- a/code/results/single_scene/summary.csv
+++++ b/code/results/single_scene/summary.csv
++@@ -2,171 +2,198 @@ method,scene,seed,rot_err_deg,pos_err,reproj_err_px,reproj_med_px,n_cams,runtime
++ esfm,Alcatraz Courtyard,0,0.324288380204305,0.0854554642858637,3.379232694353053,1.9204072326560708,133,4821.4,4792.707528591156,99999.0
++ esfm,Alcatraz Courtyard,1,0.36138678157391557,0.09112912757839253,3.3691035794153503,1.8536696768772378,133,4825.72,4804.594581127167,99999.0
++ esfm,Alcatraz Courtyard,2,0.4015043649148433,0.10235190268069758,3.557242072642648,1.9198007399315369,133,4833.02,4812.242679834366,99999.0
++-uesfm,Alcatraz Courtyard,0,0.6370266175309793,0.1671943803031883,5.956592777085197,3.6638282340102246,133,6478.06,6448.735792636871,99999.0
++-uesfm,Alcatraz Courtyard,1,0.5812669610362707,0.15343994793419002,5.71686544495911,3.550076086474883,133,6483.48,6454.087207078934,99999.0
++-uesfm,Alcatraz Courtyard,2,0.6293740753551684,0.16343948136157807,5.713459273661585,3.459422664018691,133,6483.73,6451.153926610947,99999.0
+++uesfm,Alcatraz Courtyard,0,0.5509063563081962,0.14050635120746804,5.528957815039945,3.569220349534855,133,6556.5,6511.41095662117,99999.0
+++uesfm,Alcatraz Courtyard,1,0.6823224031499351,0.17277455831381494,5.435931436314427,3.4771386471547348,133,6552.19,6506.419185638428,99999.0
+++uesfm,Alcatraz Courtyard,2,29.163200369074893,6.342299350838519,10.849871644564203,6.781661682554276,133,6544.26,6512.667362689972,99999.0
++ esfm,Alcatraz Water Tower,0,0.6984432687370988,0.38694796813795407,4.741313233832226,2.1058898944784543,172,2461.59,2440.421297073364,99999.0
++ esfm,Alcatraz Water Tower,1,0.5944425806062753,0.3287537445614408,4.5656682619915765,1.9703698309044646,172,2467.28,2454.429310798645,99999.0
++ esfm,Alcatraz Water Tower,2,0.7051984712289041,0.39359395644708545,4.982802041506025,2.1318655747232063,172,2455.14,2441.783402681351,99999.0
++-uesfm,Alcatraz Water Tower,0,18.69941055700858,8.189402666249999,12.38813738647375,6.481578998967327,172,3857.94,3831.41609287262,99999.0
++-uesfm,Alcatraz Water Tower,1,1.3394060108035364,0.7384271683550214,6.577617375342355,3.200105455600137,172,3887.47,3865.343374729156,99999.0
++-uesfm,Alcatraz Water Tower,2,2.200189184784068,1.3839747978399466,6.854463337738994,3.1730434024886014,172,3819.31,3799.390902757645,99999.0
+++uesfm,Alcatraz Water Tower,0,2.016029335283383,1.087082854315953,7.9502009315303575,4.413488326253207,172,3805.19,3775.908159971237,99999.0
+++uesfm,Alcatraz Water Tower,1,1.4259772151091428,0.7645994701417347,7.177871865477131,3.7683369835989504,172,3787.07,3763.27596282959,99999.0
+++uesfm,Alcatraz Water Tower,2,0.9982626119141534,0.558403777925872,7.063395425015731,3.814475911180562,172,3795.37,3773.579354286194,99999.0
++ esfm,Buddah Tooth Relic Temple Singapore,0,1.3132762212436329,0.31652548993670354,9.903890396062208,4.634952217191846,162,2921.45,2903.059753894806,99999.0
++ esfm,Buddah Tooth Relic Temple Singapore,1,0.9908742070528852,0.2335767172234499,9.836436953133868,4.661767906617449,162,2915.39,2904.387546062469,99999.0
++ esfm,Buddah Tooth Relic Temple Singapore,2,1.089403690069201,0.2896515999941069,10.151103921611115,4.628183275251743,162,2913.67,2902.928613901138,99999.0
++-uesfm,Buddah Tooth Relic Temple Singapore,0,1.9767734026333772,0.5777116480820644,15.584999477939833,8.800407928031486,162,4315.97,4296.739196300507,99999.0
++-uesfm,Buddah Tooth Relic Temple Singapore,1,1.4532821126601407,0.345133635718661,16.753182673783428,9.555989756393366,162,4334.28,4311.474883317947,99999.0
++-uesfm,Buddah Tooth Relic Temple Singapore,2,32.81105779154638,6.0024233739495125,26.564606536203048,13.925597303414058,162,4312.37,4291.042752504349,99999.0
+++uesfm,Buddah Tooth Relic Temple Singapore,0,1.5799948547864517,0.43886498334860274,13.484389374830494,7.719072745032001,162,4294.34,4266.538989067078,99999.0
+++uesfm,Buddah Tooth Relic Temple Singapore,1,1.934129180062467,0.5924991617860504,13.872322562031941,8.04977435062149,162,4303.62,4259.311170101166,99999.0
+++uesfm,Buddah Tooth Relic Temple Singapore,2,1.8591664615796053,0.5279087705483619,14.731758217446671,8.481044153007865,162,4270.55,4249.088565349579,99999.0
++ esfm,Doge Palace Venice,0,0.9741701978804636,0.2578214102266912,6.511459231891615,3.6947627756511525,241,28795.18,28746.22650837898,99999.0
++ esfm,Doge Palace Venice,1,0.627860879835449,0.1773446161890865,6.351890452986879,3.537139014829222,241,28742.27,28699.91308617592,99999.0
++-uesfm,Doge Palace Venice,0,0.9110673544913167,0.2904399918743801,10.966183720673449,6.771416746021886,241,21074.95,21017.30448675156,99999.0
+++esfm,Doge Palace Venice,2,0.6576305681966796,0.19114542381679772,6.214841374149415,3.591638795332466,241,10030.76,9989.16938996315,99999.0
+++uesfm,Doge Palace Venice,0,1.120669822868001,0.3500031006198606,10.496216444068507,6.9730858315176345,241,20767.26,20701.21099281311,99999.0
++ esfm,Door Lund,0,0.01167051570787235,0.003654424593160691,1.8552512563951393,0.9492901559591532,12,4892.41,4878.351888656616,99999.0
++ esfm,Door Lund,1,0.009685491454411136,0.0030270923396151107,1.6462990657331118,0.9049495114630693,12,4884.51,4877.529450178146,99999.0
++ esfm,Door Lund,2,0.02068285629921104,0.005193423834514985,1.933833706356197,0.9506754982524039,12,4888.17,4880.201772212982,99999.0
++-uesfm,Door Lund,0,14.059454220682019,0.9733778175929079,11.85413335315025,6.517535151323047,12,9185.47,8715.343282699585,95000.0
++-uesfm,Door Lund,1,14.111993883089163,0.9618908892808385,11.853785941610944,6.504290333453439,12,9203.34,8729.614821910858,95000.0
++-uesfm,Door Lund,2,14.344083655704312,0.9193773651816372,12.058851721280973,6.643622749322514,12,9176.97,9162.127201795578,99999.0
+++uesfm,Door Lund,0,0.03084538453455712,0.007522151486969653,2.3711871496098955,1.6703501976219666,12,9262.05,9243.771874427795,99999.0
+++uesfm,Door Lund,1,13.889154901234134,1.0160235649641791,11.830787694675347,6.5775252925915195,12,9243.42,7848.296544790268,85000.0
+++uesfm,Door Lund,2,0.03345748760425064,0.00845459656629652,2.434197612189203,1.6957836630064358,12,9262.36,9247.514908313751,99999.0
++ esfm,Drinking Fountain Somewhere In Zurich,0,25.804179845300204,1.1242096722251314,8.219025541657992,3.3464550934365884,14,888.07,874.4874215126038,99999.0
++ esfm,Drinking Fountain Somewhere In Zurich,1,25.754577171528883,1.1271567319301725,8.245599860088616,3.352895635561939,14,882.33,873.2415659427643,99999.0
++ esfm,Drinking Fountain Somewhere In Zurich,2,0.019604308735801684,0.006780729271172703,1.7090486008047308,0.6375575382268279,14,881.11,873.6504108905792,99999.0
++-uesfm,Drinking Fountain Somewhere In Zurich,0,26.27952541668808,1.102546079947844,8.40971803772888,3.4593948661771203,14,2969.96,2950.480526924133,99999.0
++-uesfm,Drinking Fountain Somewhere In Zurich,1,26.264849988219943,1.1078406736783482,8.547474098662555,3.585334207185317,14,2960.9,2504.42572927475,85000.0
++-uesfm,Drinking Fountain Somewhere In Zurich,2,0.06235835436589852,0.011103715758719332,2.5743283079384534,1.5361135173722633,14,2960.08,2945.673992872238,99999.0
+++uesfm,Drinking Fountain Somewhere In Zurich,0,25.946100100284117,1.1178920745623426,8.471712388928934,3.6241397282821626,14,5817.37,5796.999963760376,99999.0
+++uesfm,Drinking Fountain Somewhere In Zurich,1,25.973070279327214,1.1166462082604052,8.538877295313494,3.6343523276029472,14,5821.94,5806.947921991348,99999.0
+++uesfm,Drinking Fountain Somewhere In Zurich,2,0.06926036271356541,0.01532826351119577,3.015375672575656,1.7451989053599506,14,5806.35,5792.085475206375,99999.0
++ esfm,East Indiaman Goteborg,0,6.033696331509855,1.0773322362193534,7.427433157074123,2.292729458380311,179,3990.25,3970.732353687286,99999.0
++ esfm,East Indiaman Goteborg,1,5.313967690277326,0.9356515362150493,7.350753973659207,2.2747477951684703,179,3986.66,3970.914752960205,99999.0
++ esfm,East Indiaman Goteborg,2,5.357164002763642,0.9254557315236661,7.174538078735992,2.2493605774605823,179,3984.64,3967.810528755188,99999.0
++-uesfm,East Indiaman Goteborg,0,4.992588850190341,0.9093886386013725,9.930437318497303,4.011819888270754,179,5247.16,5221.816437721252,99999.0
++-uesfm,East Indiaman Goteborg,1,5.382092262718057,0.9927766021898647,10.035611563702533,3.912004668740913,179,5245.65,5210.653891563416,99999.0
++-uesfm,East Indiaman Goteborg,2,5.554715231405653,0.9752252406138763,9.5551521706325,3.7417116053580193,179,5231.48,5207.815300703049,99999.0
+++uesfm,East Indiaman Goteborg,0,5.365772057886591,0.9552115949640507,9.655123223073948,4.174486084237091,179,5203.6,5178.753549814224,99999.0
+++uesfm,East Indiaman Goteborg,1,10.45533482189932,2.0687908277631517,9.98932850274456,4.110771460953736,179,5195.98,5174.958552598953,99999.0
+++uesfm,East Indiaman Goteborg,2,5.872735570127935,1.0672842053553182,10.745053220477157,4.480528977857714,179,5194.7,5173.540862560272,99999.0
++ esfm,Ecole Superior De Guerre,0,0.4863901164687255,0.1239377922325492,3.750666445958877,2.0528113108131016,35,2385.33,2371.708672523499,99999.0
++ esfm,Ecole Superior De Guerre,1,0.3031794677170201,0.07921628898965839,3.2456935905644406,1.730123845260616,35,2379.34,2371.191816806793,99999.0
++ esfm,Ecole Superior De Guerre,2,0.2580439805296837,0.06772089648290676,3.257951412507941,1.687291979251782,35,2376.49,2368.935952663422,99999.0
++-uesfm,Ecole Superior De Guerre,0,0.28243957678843373,0.0778675008788964,5.291358032458062,3.601773151002547,35,4647.51,4622.850021600723,99999.0
++-uesfm,Ecole Superior De Guerre,1,0.25507313740145815,0.06695766472508183,5.170495431958795,3.5179325723573998,35,4636.07,4616.890106916428,99999.0
++-uesfm,Ecole Superior De Guerre,2,0.3137499276035537,0.08683967313665406,5.583934687507232,3.7813953498055604,35,4617.92,4602.193422079086,99999.0
+++uesfm,Ecole Superior De Guerre,0,51.32681795107141,2.57407526776939,11.751384768771835,6.449293516058521,35,10426.77,10405.3174200058,99999.0
+++uesfm,Ecole Superior De Guerre,1,0.2738850931952059,0.0734625834578241,5.5413290518324265,3.89237755756953,35,10405.39,10384.56669402122,99999.0
++ esfm,Eglise du dome,0,0.3242410065346163,0.0824797769992069,3.906129661328729,1.3512292775673158,85,25054.74,25028.64922428131,99999.0
++ esfm,Eglise du dome,1,0.3771623368290559,0.09432763361451474,3.692490941110161,1.297032357665518,85,25144.16,25116.01952123642,99999.0
++-uesfm,Eglise du dome,0,45.800562123362404,6.451383786272971,7.963674120656965,3.986938865913781,85,23371.73,23333.49408817291,99999.0
+++esfm,Eglise du dome,2,0.45623311880194967,0.11755570327842058,4.363674496823582,1.384257241090686,85,8694.98,8671.502047538757,99999.0
+++uesfm,Eglise du dome,0,0.8528700994829498,0.23193079249562099,5.944108200898114,2.9306630821877517,85,9803.94,9764.585114717484,99999.0
+++uesfm,Eglise du dome,1,0.6936386490465627,0.1932815436676001,6.091509620719386,2.9784748994578076,85,9788.95,9753.742706537247,99999.0
+++uesfm,Eglise du dome,2,45.4797063436542,6.439471230615876,8.408073658981259,4.497252326534489,85,9815.16,9781.022309541702,99999.0
++ esfm,Folke Filbyter,0,89.03873163576341,0.12895560134543632,17.986624161769267,7.423093846189277,40,1582.8,1573.360969305038,99999.0
++ esfm,Folke Filbyter,1,84.74740427947457,0.13132264058966686,17.98706393015447,7.385900243615879,40,1579.13,1572.079373121262,99999.0
++ esfm,Folke Filbyter,2,83.70817707245973,0.1261878953205589,21.85200129525089,6.324245998829542,40,1583.76,1576.207699537277,99999.0
++-uesfm,Folke Filbyter,0,80.34590333370599,0.13169898199385566,22.83327162619912,11.711593158296864,40,3701.49,3129.452286243439,85000.0
++-uesfm,Folke Filbyter,1,81.28676564061475,0.1282318052949447,24.480394314414323,9.74969805057316,40,3698.11,3681.891911745071,99999.0
++-uesfm,Folke Filbyter,2,82.23270227083374,0.1312531386108397,22.153947807931974,10.961536903594071,40,3709.96,3695.84988451004,99999.0
+++uesfm,Folke Filbyter,0,84.55526303793802,0.13091275274611416,29.190267154194732,11.160073225547443,40,3747.65,3727.990335702896,99999.0
+++uesfm,Folke Filbyter,1,84.87047537526948,0.13112291273275417,32.36503566471496,13.743964264262598,40,3743.53,3728.497133255005,99999.0
+++uesfm,Folke Filbyter,2,78.54375401873024,0.1316690349103016,23.11854509839131,10.94439865744236,40,3774.4,3007.95537352562,80000.0
++ esfm,Fort Channing Gate Singapore,0,0.10143646071220622,0.04531774433800129,1.7158776598376824,0.8165178663538488,27,3438.39,3428.288990497589,99999.0
++ esfm,Fort Channing Gate Singapore,1,0.08873951454303304,0.039001318240256114,1.783712095722291,0.8156020254191567,27,3430.88,3424.490573883057,99999.0
++ esfm,Fort Channing Gate Singapore,2,0.09031224636384508,0.03957289173255022,1.850800887931411,0.820658413919779,27,3433.07,3426.547735452652,99999.0
++-uesfm,Fort Channing Gate Singapore,0,33.158343156383296,11.908537566474344,12.69248500308198,5.438758486477886,27,6209.29,6192.107179880142,99999.0
++-uesfm,Fort Channing Gate Singapore,1,0.07319798057764343,0.03541606344760984,3.4819579894798927,2.2085534812612853,27,6159.7,6144.797720909119,99999.0
++-uesfm,Fort Channing Gate Singapore,2,0.15394831153888047,0.0686769122909015,3.77993377770425,2.2995094868650368,27,6147.27,6132.745970487595,99999.0
+++uesfm,Fort Channing Gate Singapore,0,0.10214634402234797,0.048923384893640356,3.7907267588398517,2.449846070018404,27,6175.88,6156.048281908035,99999.0
+++uesfm,Fort Channing Gate Singapore,1,0.14219758622796153,0.06357969294387673,3.5066994916782015,2.1555120267718575,27,6191.97,6176.711992740631,99999.0
+++uesfm,Fort Channing Gate Singapore,2,0.217326908828172,0.10047378918560818,3.8470573886147417,2.340537393844086,27,6145.9,6130.889567136765,99999.0
++ esfm,Golden Statue Somewhere In Hong Kong,0,0.23894829924734365,0.053817571637812096,1.7239352536484445,0.744089758310296,18,4096.74,4087.775670289993,99999.0
++ esfm,Golden Statue Somewhere In Hong Kong,1,0.2233990458438394,0.053061028269618456,1.8936878914364637,0.7746436618607616,18,4096.28,4088.994391202927,99999.0
++ esfm,Golden Statue Somewhere In Hong Kong,2,0.23397914670516012,0.051137498616263405,1.87070255923672,0.7763877145267694,18,4094.54,4087.674698591232,99999.0
++-uesfm,Golden Statue Somewhere In Hong Kong,0,0.36074365888948035,0.06894135246785482,2.939092008443413,1.863811153764833,18,7761.69,7746.569520235062,99999.0
++-uesfm,Golden Statue Somewhere In Hong Kong,1,0.35638930493960375,0.06132592684828702,2.686157398440538,1.735571387461727,18,7759.16,7745.389947891235,99999.0
++-uesfm,Golden Statue Somewhere In Hong Kong,2,0.3955566726941678,0.09197790539367184,2.878341172309852,1.8164738202745854,18,7766.46,7751.458931922913,99999.0
+++uesfm,Golden Statue Somewhere In Hong Kong,0,0.6244166607301816,0.09872331853471474,3.161519043143039,2.067073705499029,18,19646.03,19620.43561792374,99999.0
++ esfm,Gustav Vasa,0,3.4881363194229826,0.2273181104934767,1.7369784534725496,0.9521606063342205,18,711.53,703.3091380596161,99999.0
++ esfm,Gustav Vasa,1,3.4307883498221035,0.21432466556133864,1.5556448596237022,0.9232184455756751,18,710.28,703.9906163215637,99999.0
++ esfm,Gustav Vasa,2,3.6089062562451386,0.2196578287395002,1.673803634770729,0.9429755567757121,18,707.79,701.5903799533844,99999.0
++-uesfm,Gustav Vasa,0,5.828154714648938,0.32510821331670736,3.0711042143581646,2.4170166616864357,18,2566.0,2547.931907176971,99999.0
++-uesfm,Gustav Vasa,1,6.515239842974278,0.35941242702736925,3.089677100739632,2.4476428398062584,18,2530.9,1385.089058637619,55000.0
++-uesfm,Gustav Vasa,2,5.954036423569478,0.34942942401850696,3.101296679577594,2.400483554095863,18,2533.76,2520.798699855804,99999.0
+++uesfm,Gustav Vasa,0,36.645536646380805,1.0070193571286277,4.980488290477233,3.197977230260218,18,2562.94,2419.297919034958,95000.0
+++uesfm,Gustav Vasa,1,36.4914104514929,1.03254185606701,5.377024289258223,3.4818421684593366,18,4566.81,4543.375423431396,99999.0
+++uesfm,Gustav Vasa,2,4.4238740734834625,0.284762443333143,3.8820183248893803,2.8663959682636566,18,4533.21,4516.61039352417,99999.0
++ esfm,GustavIIAdolf,0,46.92758389264633,8.570499881283022,11.684017169857798,5.131125459552879,57,1047.82,1038.018162250519,99999.0
++ esfm,GustavIIAdolf,1,47.21999673071185,8.515052769156952,11.695065125063977,5.091617086264161,57,1041.41,1034.5799472332,99999.0
++ esfm,GustavIIAdolf,2,47.2312223396789,8.523261295519099,11.787583212607423,5.2715388868550574,57,1043.28,1036.478415489197,99999.0
++-uesfm,GustavIIAdolf,0,89.37607436538163,13.276440526172165,15.53883714368065,5.864332719302596,57,2904.42,2881.687849521637,99999.0
++-uesfm,GustavIIAdolf,1,70.16367503302509,11.555242253627961,15.162174576474532,6.9709512967490355,57,2884.62,2870.103493452072,99999.0
++-uesfm,GustavIIAdolf,2,72.72996778765942,11.054260851523775,16.85269001998736,9.009102977660282,57,2888.7,2874.096453428268,99999.0
+++uesfm,GustavIIAdolf,0,85.51179144359693,12.792264407383987,17.018024925092185,8.196266635685827,57,2973.35,2956.916826248169,99999.0
+++uesfm,GustavIIAdolf,1,88.9494736685426,13.260723460498589,17.699982689478404,8.888010081099381,57,2995.3,2976.055320978165,99999.0
+++uesfm,GustavIIAdolf,2,90.12096202318821,13.335169188185864,16.941538974035925,7.772181288890227,57,2974.4,2960.110015630722,99999.0
++ esfm,Jonas Ahlstromer,0,64.46662959289324,11.985605264945464,12.602584968179986,2.9168445332917745,40,614.88,607.3498430252075,99999.0
++ esfm,Jonas Ahlstromer,1,38.42510202464649,8.741159740710629,10.25268858530355,3.7024742418884333,40,618.73,611.5567183494568,99999.0
++ esfm,Jonas Ahlstromer,2,43.32891670826628,8.741053567623224,10.553031866494822,3.031717304343782,40,612.74,606.6802380084991,99999.0
++-uesfm,Jonas Ahlstromer,0,42.30073879858729,10.272232427418913,9.098970988131883,3.992857074878733,40,2339.57,1280.604686737061,55000.0
++-uesfm,Jonas Ahlstromer,1,42.18412804385421,10.310808592826955,8.453808908218651,3.4329852879523464,40,2334.33,2320.362742424011,99999.0
++-uesfm,Jonas Ahlstromer,2,42.25301394187691,10.315519611163806,8.793789077520445,3.8046125347596442,40,2311.04,2297.139800786972,99999.0
+++uesfm,Jonas Ahlstromer,0,45.264596178655594,9.403212752141883,11.336617283269852,4.728596140494824,40,2421.11,2401.221394062042,99999.0
+++uesfm,Jonas Ahlstromer,1,51.490398556789884,10.860548616493958,12.821407884052217,5.012487046810724,40,2428.25,2415.139223337173,99999.0
+++uesfm,Jonas Ahlstromer,2,42.99784237868634,10.470910789899067,9.274184391894982,4.4612757046777345,40,2394.6,2381.585824012756,99999.0
++ esfm,Kings College University Of Toronto,0,14.400461796497643,3.2653370282498866,4.064353916685246,1.6753245173381437,77,1074.94,1066.306197404861,99999.0
++ esfm,Kings College University Of Toronto,1,11.956699445712928,2.705452710546247,3.9727144140636783,1.5797087908756111,77,1071.64,1064.129351377487,99999.0
++ esfm,Kings College University Of Toronto,2,0.7191591237413282,0.16178494376153177,2.656807607821862,1.1949677972638288,77,1090.36,1082.987608671188,99999.0
++-uesfm,Kings College University Of Toronto,0,31.932532001930838,2.9775501815475,6.099555590572752,3.8717713150174773,77,2884.23,2866.194714784622,99999.0
++-uesfm,Kings College University Of Toronto,1,2.4241467960672063,0.41577835428574456,4.631696476065898,3.148046400399828,77,2899.51,2884.114916801453,99999.0
++-uesfm,Kings College University Of Toronto,2,30.212552334890066,2.3209884971628156,5.844709739288878,3.754268841443037,77,2874.99,2860.901805400848,99999.0
+++uesfm,Kings College University Of Toronto,0,29.200261744017546,2.0893308856114428,7.037455848998898,4.8227457854711675,77,2856.72,2838.923613071442,99999.0
+++uesfm,Kings College University Of Toronto,1,7.107988087620218,1.3235014088907995,5.965363878755208,4.179664786063016,77,2852.71,2839.42994761467,99999.0
+++uesfm,Kings College University Of Toronto,2,6.543121223644844,1.5041055210367995,6.0501544660211914,4.127143945227024,77,2863.77,2850.061238765717,99999.0
++ esfm,Lund University Sphinx,0,13.661465370705821,3.5591094267381553,6.242667318159146,2.715620555070548,70,2718.96,2709.5872194767,99999.0
++ esfm,Lund University Sphinx,1,12.524302160109885,3.2130942199657664,6.1182420215295075,2.7468468987195953,70,2722.86,2715.36418390274,99999.0
++ esfm,Lund University Sphinx,2,7.98961938632192,2.1710959765491817,5.807289242508196,2.893370267054225,70,2713.88,2706.478076219559,99999.0
++-uesfm,Lund University Sphinx,0,15.165716414733652,3.5564827651349713,10.803408462752863,6.730422995453065,70,4644.92,4626.135177612305,99999.0
++-uesfm,Lund University Sphinx,1,68.88899086055604,13.548881652837478,14.336626262363456,7.626356837439143,70,4703.31,4687.50173997879,99999.0
++-uesfm,Lund University Sphinx,2,15.014326075576053,3.5607694164340526,11.039432108003597,6.81081192121135,70,4615.6,4600.294319868088,99999.0
+++uesfm,Lund University Sphinx,0,42.95863056778373,8.719684065980621,14.999945982069319,8.361965506089375,70,4705.61,4681.787243127823,99999.0
+++uesfm,Lund University Sphinx,1,14.202013501375653,3.5150197050401064,10.824261355110268,6.665393527268384,70,4616.97,4592.001487731934,99999.0
+++uesfm,Lund University Sphinx,2,7.993187283055365,2.2173389310010068,8.827116984475035,5.573703948682402,70,4605.39,4587.484831571579,99999.0
++ esfm,Nijo Castle Gate,0,0.6521524412276166,0.13305866403975986,7.404953325705311,3.0629804954987674,19,1097.29,1084.301982402802,99999.0
++ esfm,Nijo Castle Gate,1,0.8564043952718835,0.16950374865879153,7.358076475544799,3.1235223975185864,19,1094.89,1084.640900611877,99999.0
++ esfm,Nijo Castle Gate,2,1.0089926380330574,0.19095470213426863,7.481049991287358,3.1671779658403687,19,1093.35,1083.446812152863,99999.0
++-uesfm,Nijo Castle Gate,0,1.6923297320319277,0.3246009364794658,11.792009654594867,6.945622003724464,19,3310.13,3288.75860619545,99999.0
++-uesfm,Nijo Castle Gate,1,1.6697791743236226,0.32440429939345133,12.704681259536075,7.466128811395402,19,3308.19,3290.209539175034,99999.0
++-uesfm,Nijo Castle Gate,2,1.682403384345942,0.32817821817561266,11.917977420973372,7.102279949089778,19,3301.77,3284.964460611343,99999.0
++-uesfm,Pantheon Paris,0,0.3823220210788846,0.05987691585642601,7.649321203280243,4.797127317249956,179,5506.88,5473.694003582001,99999.0
++-uesfm,Pantheon Paris,1,45.61471478909275,6.671068486991568,16.912359249810606,6.602415648206237,179,5491.31,5459.329566717148,99999.0
++-uesfm,Pantheon Paris,2,12.03796171625132,2.465102828965073,16.125095834055482,6.860965809143959,179,5493.94,5463.555550813675,99999.0
+++uesfm,Nijo Castle Gate,0,8.641303631787597,2.791258638759238,20.96738948203601,10.297911838853688,19,3241.33,3220.324777603149,99999.0
+++uesfm,Nijo Castle Gate,1,1.3055705335697472,0.2681929407818948,11.680266119036336,6.962610820698007,19,3234.66,3216.868148088455,99999.0
+++uesfm,Nijo Castle Gate,2,1.8310744216680852,0.3558493818160386,12.75712078857825,7.906743075262264,19,3227.08,3213.229565620422,99999.0
+++uesfm,Pantheon Paris,0,0.5453004964472504,0.06999784414241637,7.790870638815493,5.057509442480535,179,5594.6,5562.967379331589,99999.0
+++uesfm,Pantheon Paris,1,0.472097972346089,0.062163803587728925,7.194706475909503,4.639524309087925,179,5536.74,5501.721322774887,99999.0
+++uesfm,Pantheon Paris,2,44.569134625905356,6.741655820531256,16.92751887234327,6.718188212415848,179,5541.46,5510.693382978439,99999.0
++ esfm,Park Gate Clermont Ferrand,0,19.399216821978534,9.438471236639753,7.521969683469249,3.892136696028138,34,1597.96,1588.07075381279,99999.0
++ esfm,Park Gate Clermont Ferrand,1,19.325402297843297,9.417998214523331,7.465007292717891,3.849088919564891,34,1596.87,1589.703137874603,99999.0
++ esfm,Park Gate Clermont Ferrand,2,25.440611579781365,11.720641869372841,8.15032188787613,4.0871739024940315,34,1595.99,1588.841923236847,99999.0
++-uesfm,Park Gate Clermont Ferrand,0,19.27114964212729,9.381930985797803,7.856513901068924,4.317004881495189,34,3598.69,3580.396904230118,99999.0
++-uesfm,Park Gate Clermont Ferrand,1,19.288145181777256,9.394362801959678,7.762082007624792,4.207851936960509,34,3592.66,3577.494926214218,99999.0
++-uesfm,Park Gate Clermont Ferrand,2,19.345727373971844,9.415066249707772,7.936652645099164,4.425507919979627,34,3600.86,3581.473034620285,99999.0
+++uesfm,Park Gate Clermont Ferrand,0,26.45121972215746,12.46688671568752,9.374365736774523,5.572822711026232,34,3659.9,3637.290126085281,99999.0
+++uesfm,Park Gate Clermont Ferrand,1,25.126789970267566,11.629146345704141,8.445284973863814,4.644358918969844,34,3647.69,3633.016668319702,99999.0
+++uesfm,Park Gate Clermont Ferrand,2,19.34531670045473,9.424313535878454,8.137652552146374,4.753910741641504,34,3642.92,3629.229806900024,99999.0
++ esfm,Plaza De Armas Santiago,0,0.7341628728579594,0.32156050159900956,6.000154308231358,2.679971945741675,240,6375.02,6333.455562829971,99999.0
++ esfm,Plaza De Armas Santiago,1,0.5176165701053533,0.22744923951256435,5.661576462867733,2.533952088248649,240,6373.68,6336.030319213867,99999.0
++ esfm,Plaza De Armas Santiago,2,1.0740077415291467,0.46304622323619105,6.365716833190761,2.733195456670397,240,6369.35,6331.436369419098,99999.0
++-uesfm,Plaza De Armas Santiago,0,6.6772176498607925,2.909933722327364,11.384209379387066,5.623987151554922,240,6540.76,6491.619550704956,99999.0
++-uesfm,Plaza De Armas Santiago,1,1.3364898126442175,0.5686650140764532,7.951759184923103,3.933795015892702,240,6536.61,6482.320575714111,99999.0
++-uesfm,Plaza De Armas Santiago,2,1.9671483940705004,0.8276654865349118,9.054185015150745,4.441179120460681,240,6529.24,6477.723381996155,99999.0
+++uesfm,Plaza De Armas Santiago,0,6.703928231459959,2.9117331629179497,11.237000025698872,5.853835350910829,240,6616.75,6560.115643978119,99999.0
+++uesfm,Plaza De Armas Santiago,1,6.276886659110257,2.7457580388752243,10.702703105569906,5.585481192919806,240,6612.47,6560.248650550842,99999.0
+++uesfm,Plaza De Armas Santiago,2,6.731807209592312,2.8977064717650864,11.656685600371235,6.070719216176278,240,6621.69,6564.684820175171,99999.0
++ esfm,Porta San Donato Bologna,0,0.5746001989174886,0.10608894230214515,5.691906895070585,2.4183460353027986,141,3751.75,3734.281898736954,99999.0
++ esfm,Porta San Donato Bologna,1,0.8978003419853328,0.14474404936935045,5.682379581165418,2.563141107750346,141,3758.51,3742.688964605331,99999.0
++ esfm,Porta San Donato Bologna,2,0.30440272958620346,0.07026546193502164,5.435311727517131,2.1626312981864086,141,3752.23,3738.056202173233,99999.0
++-uesfm,Porta San Donato Bologna,0,1.1269164513732723,0.19275737956525044,10.071225650489282,5.557466375146339,141,5111.57,5083.21807050705,99999.0
++-uesfm,Porta San Donato Bologna,1,60.66366912240625,10.234123128225525,19.593360464637342,9.305574874002819,141,5125.57,5098.085618019104,99999.0
++-uesfm,Porta San Donato Bologna,2,1.2271289404198877,0.19605254084885107,10.066750816092268,5.595831381287904,141,5116.95,5093.473650932312,99999.0
+++uesfm,Porta San Donato Bologna,0,4.020291451689462,1.1214786148434206,10.933993245401652,5.19237663800495,141,5137.62,5110.554834842682,99999.0
+++uesfm,Porta San Donato Bologna,1,0.7088473947757895,0.1505922603123855,8.617060379525427,4.684345536783661,141,5152.53,5128.584963560104,99999.0
+++uesfm,Porta San Donato Bologna,2,0.7136264396886388,0.1325505085909014,7.953155791163967,4.201659432858777,141,5212.46,5175.418401956558,99999.0
++ esfm,Round Church Cambridge,0,2.1173198662081005,0.9129854085205968,5.854320175654564,1.9683268910531708,92,9788.65,9759.719386577606,99999.0
++ esfm,Round Church Cambridge,1,2.1415279636494953,0.9267927094737857,6.33905951937897,2.0439997148640834,92,9764.61,9741.356333732605,99999.0
++-uesfm,Round Church Cambridge,0,2.5434984075631974,1.0219048587087352,8.914080922081759,4.131734829440289,92,12088.06,12047.02147889137,99999.0
++-uesfm,Round Church Cambridge,1,2.435781054756339,0.9827352719070902,8.178482224933683,3.8400051230290893,92,12034.64,11993.72608184814,99999.0
+++esfm,Round Church Cambridge,2,2.1507951248804917,0.9401429799233081,6.115230559713554,2.0517209910251797,92,9784.55,9763.525354623795,99999.0
+++uesfm,Round Church Cambridge,0,1.1275240742139254,0.32924483901066126,7.8063197416312855,3.992682305020409,92,12183.97,12142.68090772629,99999.0
+++uesfm,Round Church Cambridge,1,2.530477427180247,0.972585197332408,8.18733294883617,4.015376648566849,92,12114.44,12080.06272149086,99999.0
+++uesfm,Round Church Cambridge,2,2.6330412756122525,1.0135214760737334,8.856154239831417,4.349376095653312,92,12091.11,12053.28948831558,99999.0
++ esfm,Skansen Kronan Gothenburg,0,0.3380139741078169,0.1158931606908392,3.0478862369293624,1.5340801117570733,131,17163.94,17140.15425729752,99999.0
++-uesfm,Skansen Kronan Gothenburg,0,0.37472494768162085,0.1243676312212316,4.798736338080396,3.348916057055834,131,16235.09,16202.21351981163,99999.0
+++esfm,Skansen Kronan Gothenburg,1,0.21516831826827018,0.07458182746108981,2.786705021535272,1.4619994473169777,131,17129.62,17109.18022465706,99999.0
+++esfm,Skansen Kronan Gothenburg,2,0.3508356515622903,0.11667319277987132,2.8836839507901773,1.5216584205042176,131,17167.47,17142.66050887108,99999.0
+++uesfm,Skansen Kronan Gothenburg,0,0.38108224181004346,0.13301438128895593,4.623321785995157,3.2918868818818052,131,16316.64,16276.98657560349,99999.0
+++uesfm,Skansen Kronan Gothenburg,1,0.37345373662160575,0.12470260098166054,4.764112581596325,3.4119305236076505,131,7073.79,7040.064019680023,99999.0
++ esfm,Smolny Cathedral St Petersburg,0,21.81278753416543,2.110667559831311,11.2315357584775,6.379259553776089,131,10364.67,10320.51852750778,99999.0
++ esfm,Smolny Cathedral St Petersburg,1,21.799828748358873,2.111313741354268,11.225141339870355,6.36605939788532,131,10360.52,10319.01374697685,99999.0
++-uesfm,Smolny Cathedral St Petersburg,0,22.143392894578028,2.0985584569799265,12.52645725687386,7.895265841408903,131,10475.74,10416.62969732285,99999.0
+++esfm,Smolny Cathedral St Petersburg,2,21.71090600320656,2.110307413725059,11.259799763291888,6.4136596726420905,131,10363.17,10320.08415436745,99999.0
+++uesfm,Smolny Cathedral St Petersburg,0,22.2854186118672,2.094744890313659,12.901706098015099,8.330992137880555,131,10444.11,10386.77505373955,99999.0
+++uesfm,Smolny Cathedral St Petersburg,1,22.12557249893661,2.0983352220394447,12.972911403502199,8.373490866652297,131,10440.89,10388.390884161,99999.0
+++uesfm,Smolny Cathedral St Petersburg,2,22.3108596899731,2.0940347709664024,12.978138438291575,8.138999834900323,131,10450.27,10395.72995519638,99999.0
+++esfm,Some Cathedral In Barcelona,0,0.6775058368464424,0.24064262464083658,6.20165095815897,3.7253636674062887,177,5002.81,4985.274734258652,99999.0
+++esfm,Some Cathedral In Barcelona,1,0.6722147495861831,0.2423973792150467,6.350786792749083,3.720536755931007,177,5009.15,4991.552013158798,99999.0
+++esfm,Some Cathedral In Barcelona,2,0.6046048207068493,0.21699252142655406,6.261075100347952,3.7258171467923633,177,4993.96,4975.845166683197,99999.0
+++uesfm,Some Cathedral In Barcelona,0,3.68833616720384,1.476348113803727,14.728491750889885,9.144995894634686,177,5865.45,5836.537386417389,99999.0
+++uesfm,Some Cathedral In Barcelona,1,1.3699265043959112,0.4939901267450982,10.514669959569153,7.150937287439337,177,5931.0,5904.003969430923,99999.0
+++uesfm,Some Cathedral In Barcelona,2,39.16330988748253,12.556902649670175,22.516615012125715,12.905943206593893,177,5872.86,5843.584012508392,99999.0
++ esfm,Sri Mariamman Singapore,0,1.4920493985513708,0.4412389781578158,10.406406306167431,6.369585247957454,222,5987.98,5970.066485881805,99999.0
++ esfm,Sri Mariamman Singapore,1,1.1345564396944785,0.3608906720626337,9.703927906328842,5.885963785879842,222,5993.62,5975.365797519684,99999.0
++ esfm,Sri Mariamman Singapore,2,1.0293707921546553,0.3426709250774071,10.15220066698994,6.067537729054635,222,5975.02,5958.786058187485,99999.0
++-uesfm,Sri Mariamman Singapore,0,16.417883129790784,5.526406595652791,19.962711816743173,12.18830031311293,222,6531.45,6494.750840425491,99999.0
++-uesfm,Sri Mariamman Singapore,1,1.3793173305528281,0.5014534676018023,16.10857603648069,10.651562750176396,222,6467.65,6426.201541185379,99999.0
++-uesfm,Sri Mariamman Singapore,2,1.639001766282598,0.536397596737868,17.490398022274007,11.439337896248928,222,6457.9,6427.717175960541,99999.0
+++uesfm,Sri Mariamman Singapore,0,2.4686335799916477,0.7739007732343344,14.605464813669228,9.97923516189929,222,14748.08,14705.00972270966,99999.0
+++uesfm,Sri Mariamman Singapore,1,2.078550795026382,0.6787961741334789,13.788432610497317,9.377924514787443,222,14759.28,14715.18007564545,99999.0
+++uesfm,Sri Mariamman Singapore,2,1.7907753932427595,0.6052602851109582,12.444814769922951,8.402147028063522,222,14729.1,14697.75046205521,99999.0
++ esfm,Sri Thendayuthapani Singapore,0,0.7190396606626565,0.17922550285921848,7.12365374281766,2.231392236341273,98,13362.09,13320.09781932831,99999.0
++-uesfm,Sri Thendayuthapani Singapore,0,47.0750455982536,3.8553135654690074,14.133068630999212,7.117179633590087,98,13808.44,13754.94554877281,99999.0
+++esfm,Sri Thendayuthapani Singapore,1,0.5350953449295449,0.1261925950852554,6.4467364012557615,2.04778550672059,98,13347.85,13309.03068971634,99999.0
+++esfm,Sri Thendayuthapani Singapore,2,0.5262093764563488,0.14374074415280041,7.476614790858722,2.2575164196527426,98,13304.56,13262.61834168434,99999.0
++ esfm,Sri Veeramakaliamman Singapore,0,2.333869921178247,0.542847254574005,11.965270419451508,5.472877662905037,157,12392.18,12365.15180635452,99999.0
++-uesfm,Sri Veeramakaliamman Singapore,0,29.40487246624615,3.8676026992817505,28.950500021302087,14.32295224594287,157,12544.69,12504.26410722733,99999.0
+++esfm,Sri Veeramakaliamman Singapore,1,2.4804161216418734,0.5731176371311499,12.150997217479388,5.5309200508775875,157,12387.11,12363.29210400581,99999.0
+++esfm,Sri Veeramakaliamman Singapore,2,2.3974569344813434,0.5599333960859275,12.58650507397198,5.711763361742226,157,12354.49,12330.03034973145,99999.0
+++uesfm,Sri Veeramakaliamman Singapore,0,2.2697869665184376,0.5625570243353476,17.34053107468558,8.599916604868525,157,12591.85,12539.44258975983,99999.0
+++uesfm,Sri Veeramakaliamman Singapore,1,2.4543260332472085,0.6034427994130044,15.868105070421757,7.655614154170212,157,12556.37,12497.1868751049,99999.0
++ esfm,Statue Of Liberty,0,52.35559762033152,28.241594953600423,688383.1603142131,683624.423417858,134,102.64,43.32754778862,45000.0
++ esfm,Statue Of Liberty,1,52.36920727661949,28.627801879821945,9277.715116667516,9303.546387753544,134,99.78,59.98358988761902,65000.0
++ esfm,Statue Of Liberty,2,52.3478568435799,28.64052858422677,7449.598084265948,7416.803834848509,134,97.28,3.520912647247314,0.0
++-uesfm,Statue Of Liberty,0,40.92379801166407,19.915794833689585,48.781849500680366,25.904882064330994,134,12927.11,12902.23110127449,99999.0
++-uesfm,Statue Of Liberty,1,38.79484508812604,17.090491683338875,33.27459817133403,17.608726936833005,134,12920.89,12896.32496738434,99999.0
+++uesfm,Statue Of Liberty,0,42.72583129860769,17.96588742153844,35.11813579810792,19.871970350798126,134,5574.75,4164.912250995636,75000.0
+++uesfm,Statue Of Liberty,1,40.08181077917127,17.10193566857949,38.50494847120855,23.533461831836124,134,5591.94,5563.9591152668,99999.0
+++uesfm,Statue Of Liberty,2,77.10685903857735,28.579866483385974,68.32023407938978,27.795793203242106,134,5579.6,5550.581798315048,99999.0
++ esfm,The Pumpkin,0,27.59459354015253,5.94835892276036,27.52772578222832,9.11126349703041,196,4330.9,4316.109169960022,99999.0
++ esfm,The Pumpkin,1,26.54397520858153,5.701234509093555,27.342758332715125,8.907676505669826,196,4327.98,4314.043692827225,99999.0
++ esfm,The Pumpkin,2,23.05395646123413,5.007378502393024,25.75879879971585,7.898438464943201,196,4322.94,4310.356798410416,99999.0
++-uesfm,The Pumpkin,0,25.490608822831348,5.440568114393512,34.198429037653845,15.6024365790899,196,5793.03,5767.982522964478,99999.0
++-uesfm,The Pumpkin,1,27.26556900533122,5.820572144056583,35.6951097814524,16.572200360468877,196,5785.61,5760.521738529205,99999.0
++-uesfm,The Pumpkin,2,26.95722616022698,5.73349875876088,36.207441893083136,17.07678356860021,196,5793.15,5768.243654251099,99999.0
+++uesfm,The Pumpkin,0,28.724486849374728,6.049382168756297,32.623684990680346,13.080799898606461,196,5715.4,5688.280281066895,99999.0
+++uesfm,The Pumpkin,1,88.35244660282751,14.480787037348948,41.33437120165238,18.995341966051914,196,13267.96,13227.80508255959,99999.0
++ esfm,Thian Hook Keng Temple Singapore,0,1.0653505173922067,0.1079565219850111,15.054152885837562,5.81055046268767,138,3299.26,3281.187238931656,99999.0
++ esfm,Thian Hook Keng Temple Singapore,1,0.8390930608109779,0.08272919436528102,14.587041005106803,5.705702663819475,138,3306.93,3283.257929086685,99999.0
++ esfm,Thian Hook Keng Temple Singapore,2,0.8760513596629002,0.08764395630143504,15.102705362055175,5.7567965038255124,138,3289.42,3278.173619031906,99999.0
++-uesfm,Thian Hook Keng Temple Singapore,0,1.146131162829257,0.10799243740327723,20.24888615229727,10.297158058757331,138,4626.48,4595.79723072052,99999.0
++-uesfm,Thian Hook Keng Temple Singapore,1,1.2072956626750737,0.12821580405567637,21.242388560332195,10.655698470690366,138,4600.34,4578.699931621552,99999.0
++-uesfm,Thian Hook Keng Temple Singapore,2,1.251054659622132,0.12561972972485672,20.875018460120653,10.416875106878496,138,4610.3,4581.792747020721,99999.0
+++uesfm,Thian Hook Keng Temple Singapore,0,0.8764983375088615,0.08202347764005188,16.212666761500287,8.391224013223542,138,4645.6,4621.513728618622,99999.0
+++uesfm,Thian Hook Keng Temple Singapore,1,0.9645123709523686,0.08977989341780128,17.411430835446634,7.905836840367921,138,4707.79,4683.081910133362,99999.0
+++uesfm,Thian Hook Keng Temple Singapore,2,0.9175998563737782,0.09144633731084849,18.520173505219134,8.485294370641896,138,4744.84,4725.761254310608,99999.0
++ esfm,Tsar Nikolai I,0,42.507553768489565,8.869914103947062,10.846690028504929,4.007961025902478,98,4932.15,4920.938139438629,99999.0
++ esfm,Tsar Nikolai I,1,43.09835370739786,8.983133733509343,11.402130572143294,4.032588092580996,98,4924.71,4914.929541349411,99999.0
++-uesfm,Tsar Nikolai I,0,75.94880014543322,15.146573953653359,15.307363921859029,6.734065570805788,98,6652.35,6629.384099006653,99999.0
++-uesfm,Tsar Nikolai I,1,80.87807139992418,15.64102228288136,21.553650044939403,9.179513384124702,98,6641.34,6623.797205209732,99999.0
+++esfm,Tsar Nikolai I,2,40.892180545819315,8.342265568650971,11.381967217434166,4.272306098513181,98,4930.74,4920.495186090469,99999.0
+++uesfm,Tsar Nikolai I,0,78.76617790472244,15.241035141808007,22.562667761517744,10.059176205736136,98,6651.48,6624.56773352623,99999.0
+++uesfm,Tsar Nikolai I,1,77.86748197534685,15.28454292365267,22.935722021884366,9.48322850302597,98,6683.57,6660.621834993362,99999.0
+++uesfm,Tsar Nikolai I,2,80.52498876805318,15.650443058307074,22.159589059526215,10.075605477720723,98,6705.71,6687.593456506729,99999.0
++ esfm,Urban II,0,59.0297968076411,11.008489348760156,17.600594623525765,5.400933781867582,96,8889.69,8878.01452589035,99999.0
+++esfm,Urban II,1,58.947927470837534,11.025374954076108,17.465515658589908,5.68378594242137,96,8885.17,8876.618469953537,99999.0
+++esfm,Urban II,2,58.77798771298748,10.982786678935327,17.546556385759054,5.767405629316944,96,8893.99,8885.34243106842,99999.0
+++uesfm,Urban II,0,70.72651677542252,10.846285449637286,20.932695061288186,9.499807603806392,96,4648.82,4629.565636634827,99999.0
+++uesfm,Urban II,1,76.69172898118858,11.840995205937814,23.160820370282533,9.824556493964668,96,10328.85,10307.27435564995,99999.0
++ esfm,Vercingetorix,0,79.07478057258302,9.776081681696928,6.649989325889541,1.7432748253588333,69,1223.75,1212.340968608856,99999.0
++ esfm,Vercingetorix,1,80.09313371434186,9.858626097407651,6.426263444651325,1.6979463697629655,69,1222.8,1213.763489961624,99999.0
++ esfm,Vercingetorix,2,89.68787080902364,10.950615257181747,8.695412468830675,3.360645595149667,69,1225.36,1215.808262825012,99999.0
++-uesfm,Vercingetorix,0,78.95942988866727,10.356244187384963,8.07065603849916,3.372297052368834,69,3105.42,3085.208213567734,99999.0
++-uesfm,Vercingetorix,1,89.99175121301754,11.178484677344343,8.929255781006072,4.302365177465691,69,3105.88,3091.564247369766,99999.0
++-uesfm,Vercingetorix,2,89.31151857112468,11.134780185118505,9.044030033429575,4.351952473195928,69,3105.52,3089.0203332901,99999.0
+++uesfm,Vercingetorix,0,90.40583622245013,11.219354185124304,13.021170300998648,6.058136758958224,69,3147.28,3128.323934793472,99999.0
+++uesfm,Vercingetorix,1,86.14504993152221,10.346921123080264,10.20083592178795,4.889104941829656,69,3189.9,3161.61762547493,99999.0
+++uesfm,Vercingetorix,2,82.105622694467,10.494716884620729,9.246952910064651,4.3504763439652825,69,3124.47,3102.738929271698,99999.0
++ esfm,Yueh Hai Ching Temple Singapore,0,0.6268598783684821,0.08360284283022794,5.56223941139456,2.132543748253723,43,1749.28,1738.747783660889,99999.0
+++esfm,Yueh Hai Ching Temple Singapore,1,0.5064956072004329,0.07235771132992698,5.30294026596974,2.06113770612239,43,1750.4,1739.704204082489,99999.0
+++esfm,Yueh Hai Ching Temple Singapore,2,0.49998151063409024,0.07009863154178562,5.92393490371475,2.2317763685579015,43,1746.32,1738.905562877655,99999.0
++diff --git a/code/results/single_scene/summary_table.md b/code/results/single_scene/summary_table.md
++index bd3f673..67b542f 100644
++--- a/code/results/single_scene/summary_table.md
+++++ b/code/results/single_scene/summary_table.md
++@@ -2,44 +2,45 @@
++ 
++ Mean over seeds per (scene, method); best value per scene in **bold** (lower is better except Nr).
++ 
++-| Scene | ESFM Rot (deg) | U-ESFM Rot (deg) | ESFM (paper) Rot (deg) | ESFM Trans | U-ESFM Trans | ESFM (paper) Trans | ESFM Reproj (px) | U-ESFM Reproj (px) | ESFM (paper) Reproj (px) | ESFM Nr | U-ESFM Nr | ESFM (paper) Nr | ESFM Time (s) | U-ESFM Time (s) | ESFM (paper) Time (s) | ESFM Rot-BA (deg) | U-ESFM Rot-BA (deg) | ESFM (paper) Rot-BA (deg) | ESFM Trans-BA | U-ESFM Trans-BA | ESFM (paper) Trans-BA | ESFM Reproj-BA (px) | U-ESFM Reproj-BA (px) | ESFM (paper) Reproj-BA (px) |
+++| Scene | ESFM (official code) Rot (deg) | U-ESFM Rot (deg) | ESFM (paper) Rot (deg) | ESFM (official code) Trans | U-ESFM Trans | ESFM (paper) Trans | ESFM (official code) Reproj (px) | U-ESFM Reproj (px) | ESFM (paper) Reproj (px) | ESFM (official code) Nr | U-ESFM Nr | ESFM (paper) Nr | ESFM (official code) Time (s) | U-ESFM Time (s) | ESFM (paper) Time (s) | ESFM (official code) Rot-BA (deg) | U-ESFM Rot-BA (deg) | ESFM (paper) Rot-BA (deg) | ESFM (official code) Trans-BA | U-ESFM Trans-BA | ESFM (paper) Trans-BA | ESFM (official code) Reproj-BA (px) | U-ESFM Reproj-BA (px) | ESFM (paper) Reproj-BA (px) |
++ |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
++-| Alcatraz Courtyard | **0.362** | 0.616 | 0.619 | **0.093** | 0.161 | 0.160 | **3.435** | 5.796 | 1.640 | **133** | 133 | -- | **4826.713** | 6481.757 | -- | -- | -- | 0.049 | -- | -- | 0.015 | -- | -- | 0.810 |
++-| Alcatraz Water Tower | **0.666** | 7.413 | 0.933 | **0.370** | 3.437 | 0.518 | **4.763** | 8.607 | 2.130 | **172** | 172 | -- | **2461.337** | 3854.907 | -- | -- | -- | 0.230 | -- | -- | 0.116 | -- | -- | 0.550 |
++-| Buddah Tooth Relic Temple Singapore | **1.131** | 12.080 | 1.030 | **0.280** | 2.308 | 0.233 | **9.964** | 19.634 | 2.060 | **162** | 162 | -- | **2916.837** | 4320.873 | -- | -- | -- | 0.081 | -- | -- | 0.014 | -- | -- | 0.850 |
++-| Doge Palace Venice | **0.801** | 0.911 | 1.163 | **0.218** | 0.290 | 0.342 | **6.432** | 10.966 | 3.620 | **241** | 241 | -- | 28768.725 | **21074.950** | -- | -- | -- | 0.211 | -- | -- | 0.029 | -- | -- | 1 |
++-| Door Lund | **0.014** | 14.172 | 0.024 | **0.004** | 0.952 | 0.006 | **1.812** | 11.922 | 0.320 | **12** | 12 | -- | **4888.363** | 9188.593 | -- | -- | -- | 0.006 | -- | -- | 0.001 | -- | -- | 0.300 |
++-| Drinking Fountain Somewhere In Zurich | **17.193** | 17.536 | 0.031 | 0.753 | **0.740** | 0.004 | **6.058** | 6.511 | 0.330 | **14** | 14 | -- | **883.837** | 2963.647 | -- | -- | -- | 0.007 | -- | -- | 0.002 | -- | -- | 0.310 |
++-| East Indiaman Goteborg | 5.568 | **5.310** | 3.814 | 0.979 | **0.959** | 0.621 | **7.318** | 9.840 | 4.130 | **179** | 179 | -- | **3987.183** | 5241.430 | -- | -- | -- | 3.117 | -- | -- | 0.509 | -- | -- | 1.850 |
++-| Ecole Superior De Guerre | 0.349 | **0.284** | 0.318 | 0.090 | **0.077** | 0.081 | **3.418** | 5.349 | 0.720 | **35** | 35 | -- | **2380.387** | 4633.833 | -- | -- | -- | 0.024 | -- | -- | 0.005 | -- | -- | 0.340 |
++-| Eglise du dome | **0.351** | 45.801 | 0.808 | **0.088** | 6.451 | 0.205 | **3.799** | 7.964 | 0.910 | **85** | 85 | -- | 25099.450 | **23371.730** | -- | -- | -- | 0.037 | -- | -- | 0.010 | -- | -- | 0.270 |
++-| Folke Filbyter | 85.831 | **81.288** | 74.596 | **0.129** | 0.130 | 0.125 | **19.275** | 23.156 | 10.370 | **40** | 40 | -- | **1581.897** | 3703.187 | -- | -- | -- | 70.157 | -- | -- | 0.118 | -- | -- | 4.290 |
++-| Fort Channing Gate Singapore | **0.093** | 11.128 | 0.207 | **0.041** | 4.004 | 0.093 | **1.783** | 6.651 | 0.520 | **27** | 27 | -- | **3434.113** | 6172.087 | -- | -- | -- | 0.020 | -- | -- | 0.008 | -- | -- | 0.250 |
++-| Golden Statue Somewhere In Hong Kong | **0.232** | 0.371 | 0.292 | **0.053** | 0.074 | 0.073 | **1.829** | 2.835 | 0.400 | **18** | 18 | -- | **4095.853** | 7762.437 | -- | -- | -- | 0.031 | -- | -- | 0.004 | -- | -- | 0.270 |
++-| Gustav Vasa | **3.509** | 6.099 | 34.181 | **0.220** | 0.345 | 1.085 | **1.655** | 3.087 | 3.520 | **18** | 18 | -- | **709.867** | 2543.553 | -- | -- | -- | 32.266 | -- | -- | 1.145 | -- | -- | 3.150 |
++-| GustavIIAdolf | **47.126** | 77.423 | 67.784 | **8.536** | 11.962 | 9.714 | **11.722** | 15.851 | 13.910 | **57** | 57 | -- | **1044.170** | 2892.580 | -- | -- | -- | 58.458 | -- | -- | 8.524 | -- | -- | 11.490 |
++-| Jonas Ahlstromer | 48.740 | **42.246** | 50.190 | **9.823** | 10.300 | 10.888 | 11.136 | **8.782** | 10.820 | **40** | 40 | -- | **615.450** | 2328.313 | -- | -- | -- | 47.117 | -- | -- | 10.451 | -- | -- | 8.410 |
++-| Kings College University Of Toronto | **9.025** | 21.523 | 0.989 | 2.044 | **1.905** | 0.235 | **3.565** | 5.525 | 0.900 | **77** | 77 | -- | **1078.980** | 2886.243 | -- | -- | -- | 0.085 | -- | -- | 0.017 | -- | -- | 0.340 |
++-| Lund University Sphinx | **11.392** | 33.023 | 19.522 | **2.981** | 6.889 | 4.585 | **6.056** | 12.060 | 4.780 | **70** | 70 | -- | **2718.567** | 4654.610 | -- | -- | -- | 8.752 | -- | -- | 2.191 | -- | -- | 1.360 |
++-| Nijo Castle Gate | **0.839** | 1.682 | 1.495 | **0.165** | 0.326 | 0.286 | **7.415** | 12.138 | 1.700 | **19** | 19 | -- | **1095.177** | 3306.697 | -- | -- | -- | 0.069 | -- | -- | 0.012 | -- | -- | 0.730 |
++-| Pantheon Paris | -- | 19.345 | 0.192 | -- | 3.065 | 0.050 | -- | 13.562 | 1.470 | -- | 179 | -- | -- | 5497.377 | -- | -- | -- | 0.040 | -- | -- | 0.005 | -- | -- | 0.490 |
++-| Park Gate Clermont Ferrand | 21.388 | **19.302** | 0.391 | 10.192 | **9.397** | 0.125 | **7.712** | 7.852 | 0.570 | **34** | 34 | -- | **1596.940** | 3597.403 | -- | -- | -- | 0.049 | -- | -- | 0.022 | -- | -- | 0.350 |
++-| Plaza De Armas Santiago | **0.775** | 3.327 | 6.782 | **0.337** | 1.435 | 2.944 | **6.009** | 9.463 | 7.400 | **240** | 240 | -- | **6372.683** | 6535.537 | -- | -- | -- | 2.556 | -- | -- | 1.383 | -- | -- | 4.900 |
++-| Porta San Donato Bologna | **0.592** | 21.006 | 2.153 | **0.107** | 3.541 | 0.388 | **5.603** | 13.244 | 2.280 | **141** | 141 | -- | **3754.163** | 5118.030 | -- | -- | -- | 0.095 | -- | -- | 0.046 | -- | -- | 0.750 |
++-| Round Church Cambridge | **2.129** | 2.490 | 2.451 | **0.920** | 1.002 | 1.003 | **6.097** | 8.546 | 2.660 | **92** | 92 | -- | **9776.630** | 12061.350 | -- | -- | -- | 1.107 | -- | -- | 0.582 | -- | -- | 1.540 |
++-| Skansen Kronan Gothenburg | **0.338** | 0.375 | 0.736 | **0.116** | 0.124 | 0.226 | **3.048** | 4.799 | 1.240 | **131** | 131 | -- | 17163.940 | **16235.090** | -- | -- | -- | 0.026 | -- | -- | 0.008 | -- | -- | 0.670 |
++-| Smolny Cathedral St Petersburg | **21.806** | 22.143 | 0.554 | 2.111 | **2.099** | 0.051 | **11.228** | 12.526 | 1.660 | **131** | 131 | -- | **10362.595** | 10475.740 | -- | -- | -- | 0.033 | -- | -- | 0.006 | -- | -- | 0.810 |
++-| Sri Mariamman Singapore | **1.219** | 6.479 | 2.302 | **0.382** | 2.188 | 0.683 | **10.088** | 17.854 | 4.130 | **222** | 222 | -- | **5985.540** | 6485.667 | -- | -- | -- | 0.077 | -- | -- | 0.023 | -- | -- | 0.910 |
++-| Sri Thendayuthapani Singapore | **0.719** | 47.075 | 46.269 | **0.179** | 3.855 | 3.812 | **7.124** | 14.133 | 23.370 | **98** | 98 | -- | **13362.090** | 13808.440 | -- | -- | -- | 44.170 | -- | -- | 2.870 | -- | -- | 8.440 |
++-| Sri Veeramakaliamman Singapore | **2.334** | 29.405 | 2.559 | **0.543** | 3.868 | 0.597 | **11.965** | 28.951 | 3.470 | **157** | 157 | -- | **12392.180** | 12544.690 | -- | -- | -- | 0.175 | -- | -- | 0.040 | -- | -- | 0.730 |
++-| Statue Of Liberty | 52.358 | **39.859** | 46.887 | 28.503 | **18.503** | 20.012 | 235036.825 | **41.028** | 26.160 | **134** | 134 | -- | **99.900** | 12924 | -- | -- | -- | 9.091 | -- | -- | 4.122 | -- | -- | 6.970 |
++-| The Pumpkin | **25.731** | 26.571 | 94.672 | **5.552** | 5.665 | 14.890 | **26.876** | 35.367 | 33.410 | **196** | 196 | -- | **4327.273** | 5790.597 | -- | -- | -- | 98.862 | -- | -- | 14.952 | -- | -- | 24.850 |
++-| Thian Hook Keng Temple Singapore | **0.927** | 1.201 | 0.832 | **0.093** | 0.121 | 0.082 | **14.915** | 20.789 | 2.750 | **138** | 138 | -- | **3298.537** | 4612.373 | -- | -- | -- | 0.081 | -- | -- | 0.008 | -- | -- | 1.130 |
++-| Tsar Nikolai I | **42.803** | 78.413 | 48.499 | **8.927** | 15.394 | 9.467 | **11.124** | 18.431 | 9.790 | **98** | 98 | -- | **4928.430** | 6646.845 | -- | -- | -- | 36.280 | -- | -- | 7.836 | -- | -- | 6.530 |
++-| Urban II | 59.030 | -- | 47.490 | 11.008 | -- | 9.467 | 17.601 | -- | 9.380 | 96 | -- | -- | 8889.690 | -- | -- | -- | -- | 48.214 | -- | -- | 9.586 | -- | -- | 6.920 |
++-| Vercingetorix | **82.952** | 86.088 | 69.328 | **10.195** | 10.890 | 8.788 | **7.257** | 8.681 | 5.080 | **69** | 69 | -- | **1223.970** | 3105.607 | -- | -- | -- | 17.706 | -- | -- | 3.104 | -- | -- | 1.500 |
++-| Yueh Hai Ching Temple Singapore | 0.627 | -- | 0.720 | 0.084 | -- | 0.098 | 5.562 | -- | 0.940 | 43 | -- | -- | 1749.280 | -- | -- | -- | -- | 0.043 | -- | -- | 0.014 | -- | -- | 0.650 |
++-| **Mean** | **16.146** | 23.697 | 18.023 | **3.121** | 4.014 | 2.912 | 6920.602 | **13.088** | 5.673 | 100.559 | **104.818** | -- | **5819.728** | 7358.187 | -- | -- | -- | 13.695 | -- | -- | 1.937 | -- | -- | 2.992 |
+++| Alcatraz Courtyard | **0.362** | 10.132 | 0.619 | **0.093** | 2.219 | 0.160 | **3.435** | 7.272 | 1.640 | **133** | 133 | -- | **4826.713** | 6550.983 | -- | -- | -- | 0.049 | -- | -- | 0.015 | -- | -- | 0.810 |
+++| Alcatraz Water Tower | **0.666** | 1.480 | 0.933 | **0.370** | 0.803 | 0.518 | **4.763** | 7.397 | 2.130 | **172** | 172 | -- | **2461.337** | 3795.877 | -- | -- | -- | 0.230 | -- | -- | 0.116 | -- | -- | 0.550 |
+++| Buddah Tooth Relic Temple Singapore | **1.131** | 1.791 | 1.030 | **0.280** | 0.520 | 0.233 | **9.964** | 14.029 | 2.060 | **162** | 162 | -- | **2916.837** | 4289.503 | -- | -- | -- | 0.081 | -- | -- | 0.014 | -- | -- | 0.850 |
+++| Doge Palace Venice | **0.753** | 1.121 | 1.163 | **0.209** | 0.350 | 0.342 | **6.359** | 10.496 | 3.620 | **241** | 241 | -- | 22522.737 | **20767.260** | -- | -- | -- | 0.211 | -- | -- | 0.029 | -- | -- | 1 |
+++| Door Lund | **0.014** | 4.651 | 0.024 | **0.004** | 0.344 | 0.006 | **1.812** | 5.545 | 0.320 | **12** | 12 | -- | **4888.363** | 9255.943 | -- | -- | -- | 0.006 | -- | -- | 0.001 | -- | -- | 0.300 |
+++| Drinking Fountain Somewhere In Zurich | **17.193** | 17.329 | 0.031 | 0.753 | **0.750** | 0.004 | **6.058** | 6.675 | 0.330 | **14** | 14 | -- | **883.837** | 5815.220 | -- | -- | -- | 0.007 | -- | -- | 0.002 | -- | -- | 0.310 |
+++| East Indiaman Goteborg | **5.568** | 7.231 | 3.814 | **0.979** | 1.364 | 0.621 | **7.318** | 10.130 | 4.130 | **179** | 179 | -- | **3987.183** | 5198.093 | -- | -- | -- | 3.117 | -- | -- | 0.509 | -- | -- | 1.850 |
+++| Ecole Superior De Guerre | **0.349** | 25.800 | 0.318 | **0.090** | 1.324 | 0.081 | **3.418** | 8.646 | 0.720 | **35** | 35 | -- | **2380.387** | 10416.080 | -- | -- | -- | 0.024 | -- | -- | 0.005 | -- | -- | 0.340 |
+++| Eglise du dome | **0.386** | 15.675 | 0.808 | **0.098** | 2.288 | 0.205 | **3.987** | 6.815 | 0.910 | **85** | 85 | -- | 19631.293 | **9802.683** | -- | -- | -- | 0.037 | -- | -- | 0.010 | -- | -- | 0.270 |
+++| Folke Filbyter | 85.831 | **82.656** | 74.596 | **0.129** | 0.131 | 0.125 | **19.275** | 28.225 | 10.370 | **40** | 40 | -- | **1581.897** | 3755.193 | -- | -- | -- | 70.157 | -- | -- | 0.118 | -- | -- | 4.290 |
+++| Fort Channing Gate Singapore | **0.093** | 0.154 | 0.207 | **0.041** | 0.071 | 0.093 | **1.783** | 3.715 | 0.520 | **27** | 27 | -- | **3434.113** | 6171.250 | -- | -- | -- | 0.020 | -- | -- | 0.008 | -- | -- | 0.250 |
+++| Golden Statue Somewhere In Hong Kong | **0.232** | 0.624 | 0.292 | **0.053** | 0.099 | 0.073 | **1.829** | 3.162 | 0.400 | **18** | 18 | -- | **4095.853** | 19646.030 | -- | -- | -- | 0.031 | -- | -- | 0.004 | -- | -- | 0.270 |
+++| Gustav Vasa | **3.509** | 25.854 | 34.181 | **0.220** | 0.775 | 1.085 | **1.655** | 4.747 | 3.520 | **18** | 18 | -- | **709.867** | 3887.653 | -- | -- | -- | 32.266 | -- | -- | 1.145 | -- | -- | 3.150 |
+++| GustavIIAdolf | **47.126** | 88.194 | 67.784 | **8.536** | 13.129 | 9.714 | **11.722** | 17.220 | 13.910 | **57** | 57 | -- | **1044.170** | 2981.017 | -- | -- | -- | 58.458 | -- | -- | 8.524 | -- | -- | 11.490 |
+++| Jonas Ahlstromer | 48.740 | **46.584** | 50.190 | **9.823** | 10.245 | 10.888 | **11.136** | 11.144 | 10.820 | **40** | 40 | -- | **615.450** | 2414.653 | -- | -- | -- | 47.117 | -- | -- | 10.451 | -- | -- | 8.410 |
+++| Kings College University Of Toronto | **9.025** | 14.284 | 0.989 | 2.044 | **1.639** | 0.235 | **3.565** | 6.351 | 0.900 | **77** | 77 | -- | **1078.980** | 2857.733 | -- | -- | -- | 0.085 | -- | -- | 0.017 | -- | -- | 0.340 |
+++| Lund University Sphinx | **11.392** | 21.718 | 19.522 | **2.981** | 4.817 | 4.585 | **6.056** | 11.550 | 4.780 | **70** | 70 | -- | **2718.567** | 4642.657 | -- | -- | -- | 8.752 | -- | -- | 2.191 | -- | -- | 1.360 |
+++| Nijo Castle Gate | **0.839** | 3.926 | 1.495 | **0.165** | 1.138 | 0.286 | **7.415** | 15.135 | 1.700 | **19** | 19 | -- | **1095.177** | 3234.357 | -- | -- | -- | 0.069 | -- | -- | 0.012 | -- | -- | 0.730 |
+++| Pantheon Paris | -- | 15.196 | 0.192 | -- | 2.291 | 0.050 | -- | 10.638 | 1.470 | -- | 179 | -- | -- | 5557.600 | -- | -- | -- | 0.040 | -- | -- | 0.005 | -- | -- | 0.490 |
+++| Park Gate Clermont Ferrand | **21.388** | 23.641 | 0.391 | **10.192** | 11.173 | 0.125 | **7.712** | 8.652 | 0.570 | **34** | 34 | -- | **1596.940** | 3650.170 | -- | -- | -- | 0.049 | -- | -- | 0.022 | -- | -- | 0.350 |
+++| Plaza De Armas Santiago | **0.775** | 6.571 | 6.782 | **0.337** | 2.852 | 2.944 | **6.009** | 11.199 | 7.400 | **240** | 240 | -- | **6372.683** | 6616.970 | -- | -- | -- | 2.556 | -- | -- | 1.383 | -- | -- | 4.900 |
+++| Porta San Donato Bologna | **0.592** | 1.814 | 2.153 | **0.107** | 0.468 | 0.388 | **5.603** | 9.168 | 2.280 | **141** | 141 | -- | **3754.163** | 5167.537 | -- | -- | -- | 0.095 | -- | -- | 0.046 | -- | -- | 0.750 |
+++| Round Church Cambridge | 2.137 | **2.097** | 2.451 | 0.927 | **0.772** | 1.003 | **6.103** | 8.283 | 2.660 | **92** | 92 | -- | **9779.270** | 12129.840 | -- | -- | -- | 1.107 | -- | -- | 0.582 | -- | -- | 1.540 |
+++| Skansen Kronan Gothenburg | **0.301** | 0.377 | 0.736 | **0.102** | 0.129 | 0.226 | **2.906** | 4.694 | 1.240 | **131** | 131 | -- | 17153.677 | **11695.215** | -- | -- | -- | 0.026 | -- | -- | 0.008 | -- | -- | 0.670 |
+++| Smolny Cathedral St Petersburg | **21.775** | 22.241 | 0.554 | 2.111 | **2.096** | 0.051 | **11.239** | 12.951 | 1.660 | **131** | 131 | -- | **10362.787** | 10445.090 | -- | -- | -- | 0.033 | -- | -- | 0.006 | -- | -- | 0.810 |
+++| Some Cathedral In Barcelona | **0.651** | 14.741 | 0.880 | **0.233** | 4.842 | 0.315 | **6.271** | 15.920 | 2.870 | **177** | 177 | -- | **5001.973** | 5889.770 | -- | -- | -- | 0.026 | -- | -- | 0.011 | -- | -- | 0.890 |
+++| Sri Mariamman Singapore | **1.219** | 2.113 | 2.302 | **0.382** | 0.686 | 0.683 | **10.088** | 13.613 | 4.130 | **222** | 222 | -- | **5985.540** | 14745.487 | -- | -- | -- | 0.077 | -- | -- | 0.023 | -- | -- | 0.910 |
+++| Sri Thendayuthapani Singapore | 0.593 | -- | 46.269 | 0.150 | -- | 3.812 | 7.016 | -- | 23.370 | 98 | -- | -- | 13338.167 | -- | -- | -- | -- | 44.170 | -- | -- | 2.870 | -- | -- | 8.440 |
+++| Sri Veeramakaliamman Singapore | 2.404 | **2.362** | 2.559 | **0.559** | 0.583 | 0.597 | **12.234** | 16.604 | 3.470 | **157** | 157 | -- | **12377.927** | 12574.110 | -- | -- | -- | 0.175 | -- | -- | 0.040 | -- | -- | 0.730 |
+++| Statue Of Liberty | **52.358** | 53.305 | 46.887 | 28.503 | **21.216** | 20.012 | 235036.825 | **47.314** | 26.160 | **134** | 134 | -- | **99.900** | 5582.097 | -- | -- | -- | 9.091 | -- | -- | 4.122 | -- | -- | 6.970 |
+++| The Pumpkin | **25.731** | 58.538 | 94.672 | **5.552** | 10.265 | 14.890 | **26.876** | 36.979 | 33.410 | **196** | 196 | -- | **4327.273** | 9491.680 | -- | -- | -- | 98.862 | -- | -- | 14.952 | -- | -- | 24.850 |
+++| Thian Hook Keng Temple Singapore | 0.927 | **0.920** | 0.832 | 0.093 | **0.088** | 0.082 | **14.915** | 17.381 | 2.750 | **138** | 138 | -- | **3298.537** | 4699.410 | -- | -- | -- | 0.081 | -- | -- | 0.008 | -- | -- | 1.130 |
+++| Tsar Nikolai I | **42.166** | 79.053 | 48.499 | **8.732** | 15.392 | 9.467 | **11.210** | 22.553 | 9.790 | **98** | 98 | -- | **4929.200** | 6680.253 | -- | -- | -- | 36.280 | -- | -- | 7.836 | -- | -- | 6.530 |
+++| Urban II | **58.919** | 73.709 | 47.490 | **11.006** | 11.344 | 9.467 | **17.538** | 22.047 | 9.380 | **96** | 96 | -- | 8889.617 | **7488.835** | -- | -- | -- | 48.214 | -- | -- | 9.586 | -- | -- | 6.920 |
+++| Vercingetorix | **82.952** | 86.219 | 69.328 | **10.195** | 10.687 | 8.788 | **7.257** | 10.823 | 5.080 | **69** | 69 | -- | **1223.970** | 3153.883 | -- | -- | -- | 17.706 | -- | -- | 3.104 | -- | -- | 1.500 |
+++| Yueh Hai Ching Temple Singapore | 0.544 | -- | 0.720 | 0.075 | -- | 0.098 | 5.596 | -- | 0.940 | 43 | -- | -- | 1748.667 | -- | -- | -- | -- | 0.043 | -- | -- | 0.014 | -- | -- | 0.650 |
+++| **Mean** | **15.676** | 23.885 | 17.547 | **3.032** | 4.026 | 2.840 | 6723.056 | **13.149** | 5.595 | 102.743 | **106.882** | -- | **5460.373** | 7383.827 | -- | -- | -- | 13.315 | -- | -- | 1.883 | -- | -- | 2.933 |
++ 
++ _Seeds per cell: [0, 1, 2]._
++ 
++diff --git a/code/results/single_scene/summary_table.tex b/code/results/single_scene/summary_table.tex
++index 258e3f2..c2a3dea 100644
++--- a/code/results/single_scene/summary_table.tex
+++++ b/code/results/single_scene/summary_table.tex
++@@ -7,45 +7,46 @@
++ \begin{tabular}{lcccccccccccccccccccccccc}
++ \toprule
++  & \multicolumn{3}{c}{Rot (deg)} & \multicolumn{3}{c}{Trans} & \multicolumn{3}{c}{Reproj (px)} & \multicolumn{3}{c}{Nr} & \multicolumn{3}{c}{Time (s)} & \multicolumn{3}{c}{Rot-BA (deg)} & \multicolumn{3}{c}{Trans-BA} & \multicolumn{3}{c}{Reproj-BA (px)} \\
++-Scene & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) \\
+++Scene & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) & ESFM (official code) & U-ESFM & ESFM (paper) \\
++ \midrule
++-Alcatraz Courtyard & \textbf{0.362} & 0.616 & 0.619 & \textbf{0.093} & 0.161 & 0.160 & \textbf{3.435} & 5.796 & 1.640 & \textbf{133} & 133 & -- & \textbf{4826.713} & 6481.757 & -- & -- & -- & 0.049 & -- & -- & 0.015 & -- & -- & 0.810 \\
++-Alcatraz Water Tower & \textbf{0.666} & 7.413 & 0.933 & \textbf{0.370} & 3.437 & 0.518 & \textbf{4.763} & 8.607 & 2.130 & \textbf{172} & 172 & -- & \textbf{2461.337} & 3854.907 & -- & -- & -- & 0.230 & -- & -- & 0.116 & -- & -- & 0.550 \\
++-Buddah Tooth Relic Temple Singapore & \textbf{1.131} & 12.080 & 1.030 & \textbf{0.280} & 2.308 & 0.233 & \textbf{9.964} & 19.634 & 2.060 & \textbf{162} & 162 & -- & \textbf{2916.837} & 4320.873 & -- & -- & -- & 0.081 & -- & -- & 0.014 & -- & -- & 0.850 \\
++-Doge Palace Venice & \textbf{0.801} & 0.911 & 1.163 & \textbf{0.218} & 0.290 & 0.342 & \textbf{6.432} & 10.966 & 3.620 & \textbf{241} & 241 & -- & 28768.725 & \textbf{21074.950} & -- & -- & -- & 0.211 & -- & -- & 0.029 & -- & -- & 1 \\
++-Door Lund & \textbf{0.014} & 14.172 & 0.024 & \textbf{0.004} & 0.952 & 0.006 & \textbf{1.812} & 11.922 & 0.320 & \textbf{12} & 12 & -- & \textbf{4888.363} & 9188.593 & -- & -- & -- & 0.006 & -- & -- & 0.001 & -- & -- & 0.300 \\
++-Drinking Fountain Somewhere In Zurich & \textbf{17.193} & 17.536 & 0.031 & 0.753 & \textbf{0.740} & 0.004 & \textbf{6.058} & 6.511 & 0.330 & \textbf{14} & 14 & -- & \textbf{883.837} & 2963.647 & -- & -- & -- & 0.007 & -- & -- & 0.002 & -- & -- & 0.310 \\
++-East Indiaman Goteborg & 5.568 & \textbf{5.310} & 3.814 & 0.979 & \textbf{0.959} & 0.621 & \textbf{7.318} & 9.840 & 4.130 & \textbf{179} & 179 & -- & \textbf{3987.183} & 5241.430 & -- & -- & -- & 3.117 & -- & -- & 0.509 & -- & -- & 1.850 \\
++-Ecole Superior De Guerre & 0.349 & \textbf{0.284} & 0.318 & 0.090 & \textbf{0.077} & 0.081 & \textbf{3.418} & 5.349 & 0.720 & \textbf{35} & 35 & -- & \textbf{2380.387} & 4633.833 & -- & -- & -- & 0.024 & -- & -- & 0.005 & -- & -- & 0.340 \\
++-Eglise du dome & \textbf{0.351} & 45.801 & 0.808 & \textbf{0.088} & 6.451 & 0.205 & \textbf{3.799} & 7.964 & 0.910 & \textbf{85} & 85 & -- & 25099.450 & \textbf{23371.730} & -- & -- & -- & 0.037 & -- & -- & 0.010 & -- & -- & 0.270 \\
++-Folke Filbyter & 85.831 & \textbf{81.288} & 74.596 & \textbf{0.129} & 0.130 & 0.125 & \textbf{19.275} & 23.156 & 10.370 & \textbf{40} & 40 & -- & \textbf{1581.897} & 3703.187 & -- & -- & -- & 70.157 & -- & -- & 0.118 & -- & -- & 4.290 \\
++-Fort Channing Gate Singapore & \textbf{0.093} & 11.128 & 0.207 & \textbf{0.041} & 4.004 & 0.093 & \textbf{1.783} & 6.651 & 0.520 & \textbf{27} & 27 & -- & \textbf{3434.113} & 6172.087 & -- & -- & -- & 0.020 & -- & -- & 0.008 & -- & -- & 0.250 \\
++-Golden Statue Somewhere In Hong Kong & \textbf{0.232} & 0.371 & 0.292 & \textbf{0.053} & 0.074 & 0.073 & \textbf{1.829} & 2.835 & 0.400 & \textbf{18} & 18 & -- & \textbf{4095.853} & 7762.437 & -- & -- & -- & 0.031 & -- & -- & 0.004 & -- & -- & 0.270 \\
++-Gustav Vasa & \textbf{3.509} & 6.099 & 34.181 & \textbf{0.220} & 0.345 & 1.085 & \textbf{1.655} & 3.087 & 3.520 & \textbf{18} & 18 & -- & \textbf{709.867} & 2543.553 & -- & -- & -- & 32.266 & -- & -- & 1.145 & -- & -- & 3.150 \\
++-GustavIIAdolf & \textbf{47.126} & 77.423 & 67.784 & \textbf{8.536} & 11.962 & 9.714 & \textbf{11.722} & 15.851 & 13.910 & \textbf{57} & 57 & -- & \textbf{1044.170} & 2892.580 & -- & -- & -- & 58.458 & -- & -- & 8.524 & -- & -- & 11.490 \\
++-Jonas Ahlstromer & 48.740 & \textbf{42.246} & 50.190 & \textbf{9.823} & 10.300 & 10.888 & 11.136 & \textbf{8.782} & 10.820 & \textbf{40} & 40 & -- & \textbf{615.450} & 2328.313 & -- & -- & -- & 47.117 & -- & -- & 10.451 & -- & -- & 8.410 \\
++-Kings College University Of Toronto & \textbf{9.025} & 21.523 & 0.989 & 2.044 & \textbf{1.905} & 0.235 & \textbf{3.565} & 5.525 & 0.900 & \textbf{77} & 77 & -- & \textbf{1078.980} & 2886.243 & -- & -- & -- & 0.085 & -- & -- & 0.017 & -- & -- & 0.340 \\
++-Lund University Sphinx & \textbf{11.392} & 33.023 & 19.522 & \textbf{2.981} & 6.889 & 4.585 & \textbf{6.056} & 12.060 & 4.780 & \textbf{70} & 70 & -- & \textbf{2718.567} & 4654.610 & -- & -- & -- & 8.752 & -- & -- & 2.191 & -- & -- & 1.360 \\
++-Nijo Castle Gate & \textbf{0.839} & 1.682 & 1.495 & \textbf{0.165} & 0.326 & 0.286 & \textbf{7.415} & 12.138 & 1.700 & \textbf{19} & 19 & -- & \textbf{1095.177} & 3306.697 & -- & -- & -- & 0.069 & -- & -- & 0.012 & -- & -- & 0.730 \\
++-Pantheon Paris & -- & 19.345 & 0.192 & -- & 3.065 & 0.050 & -- & 13.562 & 1.470 & -- & 179 & -- & -- & 5497.377 & -- & -- & -- & 0.040 & -- & -- & 0.005 & -- & -- & 0.490 \\
++-Park Gate Clermont Ferrand & 21.388 & \textbf{19.302} & 0.391 & 10.192 & \textbf{9.397} & 0.125 & \textbf{7.712} & 7.852 & 0.570 & \textbf{34} & 34 & -- & \textbf{1596.940} & 3597.403 & -- & -- & -- & 0.049 & -- & -- & 0.022 & -- & -- & 0.350 \\
++-Plaza De Armas Santiago & \textbf{0.775} & 3.327 & 6.782 & \textbf{0.337} & 1.435 & 2.944 & \textbf{6.009} & 9.463 & 7.400 & \textbf{240} & 240 & -- & \textbf{6372.683} & 6535.537 & -- & -- & -- & 2.556 & -- & -- & 1.383 & -- & -- & 4.900 \\
++-Porta San Donato Bologna & \textbf{0.592} & 21.006 & 2.153 & \textbf{0.107} & 3.541 & 0.388 & \textbf{5.603} & 13.244 & 2.280 & \textbf{141} & 141 & -- & \textbf{3754.163} & 5118.030 & -- & -- & -- & 0.095 & -- & -- & 0.046 & -- & -- & 0.750 \\
++-Round Church Cambridge & \textbf{2.129} & 2.490 & 2.451 & \textbf{0.920} & 1.002 & 1.003 & \textbf{6.097} & 8.546 & 2.660 & \textbf{92} & 92 & -- & \textbf{9776.630} & 12061.350 & -- & -- & -- & 1.107 & -- & -- & 0.582 & -- & -- & 1.540 \\
++-Skansen Kronan Gothenburg & \textbf{0.338} & 0.375 & 0.736 & \textbf{0.116} & 0.124 & 0.226 & \textbf{3.048} & 4.799 & 1.240 & \textbf{131} & 131 & -- & 17163.940 & \textbf{16235.090} & -- & -- & -- & 0.026 & -- & -- & 0.008 & -- & -- & 0.670 \\
++-Smolny Cathedral St Petersburg & \textbf{21.806} & 22.143 & 0.554 & 2.111 & \textbf{2.099} & 0.051 & \textbf{11.228} & 12.526 & 1.660 & \textbf{131} & 131 & -- & \textbf{10362.595} & 10475.740 & -- & -- & -- & 0.033 & -- & -- & 0.006 & -- & -- & 0.810 \\
++-Sri Mariamman Singapore & \textbf{1.219} & 6.479 & 2.302 & \textbf{0.382} & 2.188 & 0.683 & \textbf{10.088} & 17.854 & 4.130 & \textbf{222} & 222 & -- & \textbf{5985.540} & 6485.667 & -- & -- & -- & 0.077 & -- & -- & 0.023 & -- & -- & 0.910 \\
++-Sri Thendayuthapani Singapore & \textbf{0.719} & 47.075 & 46.269 & \textbf{0.179} & 3.855 & 3.812 & \textbf{7.124} & 14.133 & 23.370 & \textbf{98} & 98 & -- & \textbf{13362.090} & 13808.440 & -- & -- & -- & 44.170 & -- & -- & 2.870 & -- & -- & 8.440 \\
++-Sri Veeramakaliamman Singapore & \textbf{2.334} & 29.405 & 2.559 & \textbf{0.543} & 3.868 & 0.597 & \textbf{11.965} & 28.951 & 3.470 & \textbf{157} & 157 & -- & \textbf{12392.180} & 12544.690 & -- & -- & -- & 0.175 & -- & -- & 0.040 & -- & -- & 0.730 \\
++-Statue Of Liberty & 52.358 & \textbf{39.859} & 46.887 & 28.503 & \textbf{18.503} & 20.012 & 235036.825 & \textbf{41.028} & 26.160 & \textbf{134} & 134 & -- & \textbf{99.900} & 12924 & -- & -- & -- & 9.091 & -- & -- & 4.122 & -- & -- & 6.970 \\
++-The Pumpkin & \textbf{25.731} & 26.571 & 94.672 & \textbf{5.552} & 5.665 & 14.890 & \textbf{26.876} & 35.367 & 33.410 & \textbf{196} & 196 & -- & \textbf{4327.273} & 5790.597 & -- & -- & -- & 98.862 & -- & -- & 14.952 & -- & -- & 24.850 \\
++-Thian Hook Keng Temple Singapore & \textbf{0.927} & 1.201 & 0.832 & \textbf{0.093} & 0.121 & 0.082 & \textbf{14.915} & 20.789 & 2.750 & \textbf{138} & 138 & -- & \textbf{3298.537} & 4612.373 & -- & -- & -- & 0.081 & -- & -- & 0.008 & -- & -- & 1.130 \\
++-Tsar Nikolai I & \textbf{42.803} & 78.413 & 48.499 & \textbf{8.927} & 15.394 & 9.467 & \textbf{11.124} & 18.431 & 9.790 & \textbf{98} & 98 & -- & \textbf{4928.430} & 6646.845 & -- & -- & -- & 36.280 & -- & -- & 7.836 & -- & -- & 6.530 \\
++-Urban II & 59.030 & -- & 47.490 & 11.008 & -- & 9.467 & 17.601 & -- & 9.380 & 96 & -- & -- & 8889.690 & -- & -- & -- & -- & 48.214 & -- & -- & 9.586 & -- & -- & 6.920 \\
++-Vercingetorix & \textbf{82.952} & 86.088 & 69.328 & \textbf{10.195} & 10.890 & 8.788 & \textbf{7.257} & 8.681 & 5.080 & \textbf{69} & 69 & -- & \textbf{1223.970} & 3105.607 & -- & -- & -- & 17.706 & -- & -- & 3.104 & -- & -- & 1.500 \\
++-Yueh Hai Ching Temple Singapore & 0.627 & -- & 0.720 & 0.084 & -- & 0.098 & 5.562 & -- & 0.940 & 43 & -- & -- & 1749.280 & -- & -- & -- & -- & 0.043 & -- & -- & 0.014 & -- & -- & 0.650 \\
+++Alcatraz Courtyard & \textbf{0.362} & 10.132 & 0.619 & \textbf{0.093} & 2.219 & 0.160 & \textbf{3.435} & 7.272 & 1.640 & \textbf{133} & 133 & -- & \textbf{4826.713} & 6550.983 & -- & -- & -- & 0.049 & -- & -- & 0.015 & -- & -- & 0.810 \\
+++Alcatraz Water Tower & \textbf{0.666} & 1.480 & 0.933 & \textbf{0.370} & 0.803 & 0.518 & \textbf{4.763} & 7.397 & 2.130 & \textbf{172} & 172 & -- & \textbf{2461.337} & 3795.877 & -- & -- & -- & 0.230 & -- & -- & 0.116 & -- & -- & 0.550 \\
+++Buddah Tooth Relic Temple Singapore & \textbf{1.131} & 1.791 & 1.030 & \textbf{0.280} & 0.520 & 0.233 & \textbf{9.964} & 14.029 & 2.060 & \textbf{162} & 162 & -- & \textbf{2916.837} & 4289.503 & -- & -- & -- & 0.081 & -- & -- & 0.014 & -- & -- & 0.850 \\
+++Doge Palace Venice & \textbf{0.753} & 1.121 & 1.163 & \textbf{0.209} & 0.350 & 0.342 & \textbf{6.359} & 10.496 & 3.620 & \textbf{241} & 241 & -- & 22522.737 & \textbf{20767.260} & -- & -- & -- & 0.211 & -- & -- & 0.029 & -- & -- & 1 \\
+++Door Lund & \textbf{0.014} & 4.651 & 0.024 & \textbf{0.004} & 0.344 & 0.006 & \textbf{1.812} & 5.545 & 0.320 & \textbf{12} & 12 & -- & \textbf{4888.363} & 9255.943 & -- & -- & -- & 0.006 & -- & -- & 0.001 & -- & -- & 0.300 \\
+++Drinking Fountain Somewhere In Zurich & \textbf{17.193} & 17.329 & 0.031 & 0.753 & \textbf{0.750} & 0.004 & \textbf{6.058} & 6.675 & 0.330 & \textbf{14} & 14 & -- & \textbf{883.837} & 5815.220 & -- & -- & -- & 0.007 & -- & -- & 0.002 & -- & -- & 0.310 \\
+++East Indiaman Goteborg & \textbf{5.568} & 7.231 & 3.814 & \textbf{0.979} & 1.364 & 0.621 & \textbf{7.318} & 10.130 & 4.130 & \textbf{179} & 179 & -- & \textbf{3987.183} & 5198.093 & -- & -- & -- & 3.117 & -- & -- & 0.509 & -- & -- & 1.850 \\
+++Ecole Superior De Guerre & \textbf{0.349} & 25.800 & 0.318 & \textbf{0.090} & 1.324 & 0.081 & \textbf{3.418} & 8.646 & 0.720 & \textbf{35} & 35 & -- & \textbf{2380.387} & 10416.080 & -- & -- & -- & 0.024 & -- & -- & 0.005 & -- & -- & 0.340 \\
+++Eglise du dome & \textbf{0.386} & 15.675 & 0.808 & \textbf{0.098} & 2.288 & 0.205 & \textbf{3.987} & 6.815 & 0.910 & \textbf{85} & 85 & -- & 19631.293 & \textbf{9802.683} & -- & -- & -- & 0.037 & -- & -- & 0.010 & -- & -- & 0.270 \\
+++Folke Filbyter & 85.831 & \textbf{82.656} & 74.596 & \textbf{0.129} & 0.131 & 0.125 & \textbf{19.275} & 28.225 & 10.370 & \textbf{40} & 40 & -- & \textbf{1581.897} & 3755.193 & -- & -- & -- & 70.157 & -- & -- & 0.118 & -- & -- & 4.290 \\
+++Fort Channing Gate Singapore & \textbf{0.093} & 0.154 & 0.207 & \textbf{0.041} & 0.071 & 0.093 & \textbf{1.783} & 3.715 & 0.520 & \textbf{27} & 27 & -- & \textbf{3434.113} & 6171.250 & -- & -- & -- & 0.020 & -- & -- & 0.008 & -- & -- & 0.250 \\
+++Golden Statue Somewhere In Hong Kong & \textbf{0.232} & 0.624 & 0.292 & \textbf{0.053} & 0.099 & 0.073 & \textbf{1.829} & 3.162 & 0.400 & \textbf{18} & 18 & -- & \textbf{4095.853} & 19646.030 & -- & -- & -- & 0.031 & -- & -- & 0.004 & -- & -- & 0.270 \\
+++Gustav Vasa & \textbf{3.509} & 25.854 & 34.181 & \textbf{0.220} & 0.775 & 1.085 & \textbf{1.655} & 4.747 & 3.520 & \textbf{18} & 18 & -- & \textbf{709.867} & 3887.653 & -- & -- & -- & 32.266 & -- & -- & 1.145 & -- & -- & 3.150 \\
+++GustavIIAdolf & \textbf{47.126} & 88.194 & 67.784 & \textbf{8.536} & 13.129 & 9.714 & \textbf{11.722} & 17.220 & 13.910 & \textbf{57} & 57 & -- & \textbf{1044.170} & 2981.017 & -- & -- & -- & 58.458 & -- & -- & 8.524 & -- & -- & 11.490 \\
+++Jonas Ahlstromer & 48.740 & \textbf{46.584} & 50.190 & \textbf{9.823} & 10.245 & 10.888 & \textbf{11.136} & 11.144 & 10.820 & \textbf{40} & 40 & -- & \textbf{615.450} & 2414.653 & -- & -- & -- & 47.117 & -- & -- & 10.451 & -- & -- & 8.410 \\
+++Kings College University Of Toronto & \textbf{9.025} & 14.284 & 0.989 & 2.044 & \textbf{1.639} & 0.235 & \textbf{3.565} & 6.351 & 0.900 & \textbf{77} & 77 & -- & \textbf{1078.980} & 2857.733 & -- & -- & -- & 0.085 & -- & -- & 0.017 & -- & -- & 0.340 \\
+++Lund University Sphinx & \textbf{11.392} & 21.718 & 19.522 & \textbf{2.981} & 4.817 & 4.585 & \textbf{6.056} & 11.550 & 4.780 & \textbf{70} & 70 & -- & \textbf{2718.567} & 4642.657 & -- & -- & -- & 8.752 & -- & -- & 2.191 & -- & -- & 1.360 \\
+++Nijo Castle Gate & \textbf{0.839} & 3.926 & 1.495 & \textbf{0.165} & 1.138 & 0.286 & \textbf{7.415} & 15.135 & 1.700 & \textbf{19} & 19 & -- & \textbf{1095.177} & 3234.357 & -- & -- & -- & 0.069 & -- & -- & 0.012 & -- & -- & 0.730 \\
+++Pantheon Paris & -- & 15.196 & 0.192 & -- & 2.291 & 0.050 & -- & 10.638 & 1.470 & -- & 179 & -- & -- & 5557.600 & -- & -- & -- & 0.040 & -- & -- & 0.005 & -- & -- & 0.490 \\
+++Park Gate Clermont Ferrand & \textbf{21.388} & 23.641 & 0.391 & \textbf{10.192} & 11.173 & 0.125 & \textbf{7.712} & 8.652 & 0.570 & \textbf{34} & 34 & -- & \textbf{1596.940} & 3650.170 & -- & -- & -- & 0.049 & -- & -- & 0.022 & -- & -- & 0.350 \\
+++Plaza De Armas Santiago & \textbf{0.775} & 6.571 & 6.782 & \textbf{0.337} & 2.852 & 2.944 & \textbf{6.009} & 11.199 & 7.400 & \textbf{240} & 240 & -- & \textbf{6372.683} & 6616.970 & -- & -- & -- & 2.556 & -- & -- & 1.383 & -- & -- & 4.900 \\
+++Porta San Donato Bologna & \textbf{0.592} & 1.814 & 2.153 & \textbf{0.107} & 0.468 & 0.388 & \textbf{5.603} & 9.168 & 2.280 & \textbf{141} & 141 & -- & \textbf{3754.163} & 5167.537 & -- & -- & -- & 0.095 & -- & -- & 0.046 & -- & -- & 0.750 \\
+++Round Church Cambridge & 2.137 & \textbf{2.097} & 2.451 & 0.927 & \textbf{0.772} & 1.003 & \textbf{6.103} & 8.283 & 2.660 & \textbf{92} & 92 & -- & \textbf{9779.270} & 12129.840 & -- & -- & -- & 1.107 & -- & -- & 0.582 & -- & -- & 1.540 \\
+++Skansen Kronan Gothenburg & \textbf{0.301} & 0.377 & 0.736 & \textbf{0.102} & 0.129 & 0.226 & \textbf{2.906} & 4.694 & 1.240 & \textbf{131} & 131 & -- & 17153.677 & \textbf{11695.215} & -- & -- & -- & 0.026 & -- & -- & 0.008 & -- & -- & 0.670 \\
+++Smolny Cathedral St Petersburg & \textbf{21.775} & 22.241 & 0.554 & 2.111 & \textbf{2.096} & 0.051 & \textbf{11.239} & 12.951 & 1.660 & \textbf{131} & 131 & -- & \textbf{10362.787} & 10445.090 & -- & -- & -- & 0.033 & -- & -- & 0.006 & -- & -- & 0.810 \\
+++Some Cathedral In Barcelona & \textbf{0.651} & 14.741 & 0.880 & \textbf{0.233} & 4.842 & 0.315 & \textbf{6.271} & 15.920 & 2.870 & \textbf{177} & 177 & -- & \textbf{5001.973} & 5889.770 & -- & -- & -- & 0.026 & -- & -- & 0.011 & -- & -- & 0.890 \\
+++Sri Mariamman Singapore & \textbf{1.219} & 2.113 & 2.302 & \textbf{0.382} & 0.686 & 0.683 & \textbf{10.088} & 13.613 & 4.130 & \textbf{222} & 222 & -- & \textbf{5985.540} & 14745.487 & -- & -- & -- & 0.077 & -- & -- & 0.023 & -- & -- & 0.910 \\
+++Sri Thendayuthapani Singapore & 0.593 & -- & 46.269 & 0.150 & -- & 3.812 & 7.016 & -- & 23.370 & 98 & -- & -- & 13338.167 & -- & -- & -- & -- & 44.170 & -- & -- & 2.870 & -- & -- & 8.440 \\
+++Sri Veeramakaliamman Singapore & 2.404 & \textbf{2.362} & 2.559 & \textbf{0.559} & 0.583 & 0.597 & \textbf{12.234} & 16.604 & 3.470 & \textbf{157} & 157 & -- & \textbf{12377.927} & 12574.110 & -- & -- & -- & 0.175 & -- & -- & 0.040 & -- & -- & 0.730 \\
+++Statue Of Liberty & \textbf{52.358} & 53.305 & 46.887 & 28.503 & \textbf{21.216} & 20.012 & 235036.825 & \textbf{47.314} & 26.160 & \textbf{134} & 134 & -- & \textbf{99.900} & 5582.097 & -- & -- & -- & 9.091 & -- & -- & 4.122 & -- & -- & 6.970 \\
+++The Pumpkin & \textbf{25.731} & 58.538 & 94.672 & \textbf{5.552} & 10.265 & 14.890 & \textbf{26.876} & 36.979 & 33.410 & \textbf{196} & 196 & -- & \textbf{4327.273} & 9491.680 & -- & -- & -- & 98.862 & -- & -- & 14.952 & -- & -- & 24.850 \\
+++Thian Hook Keng Temple Singapore & 0.927 & \textbf{0.920} & 0.832 & 0.093 & \textbf{0.088} & 0.082 & \textbf{14.915} & 17.381 & 2.750 & \textbf{138} & 138 & -- & \textbf{3298.537} & 4699.410 & -- & -- & -- & 0.081 & -- & -- & 0.008 & -- & -- & 1.130 \\
+++Tsar Nikolai I & \textbf{42.166} & 79.053 & 48.499 & \textbf{8.732} & 15.392 & 9.467 & \textbf{11.210} & 22.553 & 9.790 & \textbf{98} & 98 & -- & \textbf{4929.200} & 6680.253 & -- & -- & -- & 36.280 & -- & -- & 7.836 & -- & -- & 6.530 \\
+++Urban II & \textbf{58.919} & 73.709 & 47.490 & \textbf{11.006} & 11.344 & 9.467 & \textbf{17.538} & 22.047 & 9.380 & \textbf{96} & 96 & -- & 8889.617 & \textbf{7488.835} & -- & -- & -- & 48.214 & -- & -- & 9.586 & -- & -- & 6.920 \\
+++Vercingetorix & \textbf{82.952} & 86.219 & 69.328 & \textbf{10.195} & 10.687 & 8.788 & \textbf{7.257} & 10.823 & 5.080 & \textbf{69} & 69 & -- & \textbf{1223.970} & 3153.883 & -- & -- & -- & 17.706 & -- & -- & 3.104 & -- & -- & 1.500 \\
+++Yueh Hai Ching Temple Singapore & 0.544 & -- & 0.720 & 0.075 & -- & 0.098 & 5.596 & -- & 0.940 & 43 & -- & -- & 1748.667 & -- & -- & -- & -- & 0.043 & -- & -- & 0.014 & -- & -- & 0.650 \\
++ \midrule
++-Mean & \textbf{16.146} & 23.697 & 18.023 & \textbf{3.121} & 4.014 & 2.912 & 6920.602 & \textbf{13.088} & 5.673 & 100.559 & \textbf{104.818} & -- & \textbf{5819.728} & 7358.187 & -- & -- & -- & 13.695 & -- & -- & 1.937 & -- & -- & 2.992 \\
+++Mean & \textbf{15.676} & 23.885 & 17.547 & \textbf{3.032} & 4.026 & 2.840 & 6723.056 & \textbf{13.149} & 5.595 & 102.743 & \textbf{106.882} & -- & \textbf{5460.373} & 7383.827 & -- & -- & -- & 13.315 & -- & -- & 1.883 & -- & -- & 2.933 \\
++ \bottomrule
++ \end{tabular}}
++ \end{table*}
+ ```
+ - untracked/modified files:
+ ```
++M code/results/single_scene/REPRO.md
++ M code/results/single_scene/summary.csv
++ M code/results/single_scene/summary_table.md
++ M code/results/single_scene/summary_table.tex
+ ?? MVG_Project_Report_Ortal_Dayan.pdf
+ ?? "RESFM- ROBUST DEEP EQUIVARIANT STRUCTURE FROM MOTION.pdf"
+ ?? "claude specs/SPEC_cvpr_experiments.md"
+ ?? "claude specs/SPEC_sfm_datasets_setup.md"
+ ?? "claude specs/SPEC_single_scene_experiments.md"
+ ?? "claude specs/SPEC_uesfm_combined.md"
++?? "claude specs/TASK_crossdataset_readiness.md"
+ ?? code/datasets/Euclidean
+ ?? tmp_ab_check/
+ ```
+ ### esfm-baseline (official ESFM)
+ - path: `/home/projects/bagon/ortalda/MVG/final-project/esfm-baseline`
+-- commit: `2c0079497bf5dda1eeb2a902b2f859c6387f213a`
++- commit: `de666af53423c1dc3dd412e53c4e153c03d70922`
+ - **WARNING: working tree dirty.** Diff:
+ ```diff
+ (untracked files only)
+@@ -58,6 +3636,7 @@ Generated: 2026-07-13T20:41:04
+ ?? code/confs/single_scene_generated/ss_esfm_Buddah_Tooth_Relic_Temple_Singapore_seed1.conf
+ ?? code/confs/single_scene_generated/ss_esfm_Buddah_Tooth_Relic_Temple_Singapore_seed2.conf
+ ?? code/confs/single_scene_generated/ss_esfm_Doge_Palace_Venice_seed1.conf
++?? code/confs/single_scene_generated/ss_esfm_Doge_Palace_Venice_seed2.conf
+ ?? code/confs/single_scene_generated/ss_esfm_Door_Lund_seed1.conf
+ ?? code/confs/single_scene_generated/ss_esfm_Door_Lund_seed2.conf
+ ?? code/confs/single_scene_generated/ss_esfm_Drinking_Fountain_Somewhere_In_Zurich_seed1.conf
+@@ -70,6 +3649,7 @@ Generated: 2026-07-13T20:41:04
+ ?? code/confs/single_scene_generated/ss_esfm_Ecole_Superior_De_Guerre_seed2.conf
+ ?? code/confs/single_scene_generated/ss_esfm_Eglise_du_dome_seed0.conf
+ ?? code/confs/single_scene_generated/ss_esfm_Eglise_du_dome_seed1.conf
++?? code/confs/single_scene_generated/ss_esfm_Eglise_du_dome_seed2.conf
+ ?? code/confs/single_scene_generated/ss_esfm_Folke_Filbyter_seed0.conf
+ ?? code/confs/single_scene_generated/ss_esfm_Folke_Filbyter_seed1.conf
+ ?? code/confs/single_scene_generated/ss_esfm_Folke_Filbyter_seed2.conf
+@@ -114,16 +3694,22 @@ Generated: 2026-07-13T20:41:04
+ ?? code/confs/single_scene_generated/ss_esfm_Round_Church_Cambridge_seed2.conf
+ ?? code/confs/single_scene_generated/ss_esfm_Skansen_Kronan_Gothenburg_seed0.conf
+ ?? code/confs/single_scene_generated/ss_esfm_Skansen_Kronan_Gothenburg_seed1.conf
++?? code/confs/single_scene_generated/ss_esfm_Skansen_Kronan_Gothenburg_seed2.conf
+ ?? code/confs/single_scene_generated/ss_esfm_Smolny_Cathedral_St_Petersburg_seed0.conf
+ ?? code/confs/single_scene_generated/ss_esfm_Smolny_Cathedral_St_Petersburg_seed1.conf
++?? code/confs/single_scene_generated/ss_esfm_Smolny_Cathedral_St_Petersburg_seed2.conf
+ ?? code/confs/single_scene_generated/ss_esfm_Some_Cathedral_In_Barcelona_seed0.conf
++?? code/confs/single_scene_generated/ss_esfm_Some_Cathedral_In_Barcelona_seed1.conf
++?? code/confs/single_scene_generated/ss_esfm_Some_Cathedral_In_Barcelona_seed2.conf
+ ?? code/confs/single_scene_generated/ss_esfm_Sri_Mariamman_Singapore_seed0.conf
+ ?? code/confs/single_scene_generated/ss_esfm_Sri_Mariamman_Singapore_seed1.conf
+ ?? code/confs/single_scene_generated/ss_esfm_Sri_Mariamman_Singapore_seed2.conf
+ ?? code/confs/single_scene_generated/ss_esfm_Sri_Thendayuthapani_Singapore_seed0.conf
+ ?? code/confs/single_scene_generated/ss_esfm_Sri_Thendayuthapani_Singapore_seed1.conf
++?? code/confs/single_scene_generated/ss_esfm_Sri_Thendayuthapani_Singapore_seed2.conf
+ ?? code/confs/single_scene_generated/ss_esfm_Sri_Veeramakaliamman_Singapore_seed0.conf
+ ?? code/confs/single_scene_generated/ss_esfm_Sri_Veeramakaliamman_Singapore_seed1.conf
++?? code/confs/single_scene_generated/ss_esfm_Sri_Veeramakaliamman_Singapore_seed2.conf
+ ?? code/confs/single_scene_generated/ss_esfm_Statue_Of_Liberty_seed0.conf
+ ?? code/confs/single_scene_generated/ss_esfm_Statue_Of_Liberty_seed1.conf
+ ?? code/confs/single_scene_generated/ss_esfm_Statue_Of_Liberty_seed2.conf
+@@ -137,10 +3723,14 @@ Generated: 2026-07-13T20:41:04
+ ?? code/confs/single_scene_generated/ss_esfm_Tsar_Nikolai_I_seed1.conf
+ ?? code/confs/single_scene_generated/ss_esfm_Tsar_Nikolai_I_seed2.conf
+ ?? code/confs/single_scene_generated/ss_esfm_Urban_II_seed0.conf
++?? code/confs/single_scene_generated/ss_esfm_Urban_II_seed1.conf
++?? code/confs/single_scene_generated/ss_esfm_Urban_II_seed2.conf
+ ?? code/confs/single_scene_generated/ss_esfm_Vercingetorix_seed0.conf
+ ?? code/confs/single_scene_generated/ss_esfm_Vercingetorix_seed1.conf
+ ?? code/confs/single_scene_generated/ss_esfm_Vercingetorix_seed2.conf
+ ?? code/confs/single_scene_generated/ss_esfm_Yueh_Hai_Ching_Temple_Singapore_seed0.conf
++?? code/confs/single_scene_generated/ss_esfm_Yueh_Hai_Ching_Temple_Singapore_seed1.conf
++?? code/confs/single_scene_generated/ss_esfm_Yueh_Hai_Ching_Temple_Singapore_seed2.conf
+ ```
+ 
+ ## Environment (shared venv used for BOTH methods)
+diff --git a/code/results/single_scene/summary.csv b/code/results/single_scene/summary.csv
+index e4f9f5e..4e9f7a5 100644
+--- a/code/results/single_scene/summary.csv
++++ b/code/results/single_scene/summary.csv
+@@ -1,172 +1,427 @@
+-method,scene,seed,rot_err_deg,pos_err,reproj_err_px,reproj_med_px,n_cams,runtime_s,convergence_time_s,best_epoch
+-esfm,Alcatraz Courtyard,0,0.324288380204305,0.0854554642858637,3.379232694353053,1.9204072326560708,133,4821.4,4792.707528591156,99999.0
+-esfm,Alcatraz Courtyard,1,0.36138678157391557,0.09112912757839253,3.3691035794153503,1.8536696768772378,133,4825.72,4804.594581127167,99999.0
+-esfm,Alcatraz Courtyard,2,0.4015043649148433,0.10235190268069758,3.557242072642648,1.9198007399315369,133,4833.02,4812.242679834366,99999.0
+-uesfm,Alcatraz Courtyard,0,0.6370266175309793,0.1671943803031883,5.956592777085197,3.6638282340102246,133,6478.06,6448.735792636871,99999.0
+-uesfm,Alcatraz Courtyard,1,0.5812669610362707,0.15343994793419002,5.71686544495911,3.550076086474883,133,6483.48,6454.087207078934,99999.0
+-uesfm,Alcatraz Courtyard,2,0.6293740753551684,0.16343948136157807,5.713459273661585,3.459422664018691,133,6483.73,6451.153926610947,99999.0
+-esfm,Alcatraz Water Tower,0,0.6984432687370988,0.38694796813795407,4.741313233832226,2.1058898944784543,172,2461.59,2440.421297073364,99999.0
+-esfm,Alcatraz Water Tower,1,0.5944425806062753,0.3287537445614408,4.5656682619915765,1.9703698309044646,172,2467.28,2454.429310798645,99999.0
+-esfm,Alcatraz Water Tower,2,0.7051984712289041,0.39359395644708545,4.982802041506025,2.1318655747232063,172,2455.14,2441.783402681351,99999.0
+-uesfm,Alcatraz Water Tower,0,18.69941055700858,8.189402666249999,12.38813738647375,6.481578998967327,172,3857.94,3831.41609287262,99999.0
+-uesfm,Alcatraz Water Tower,1,1.3394060108035364,0.7384271683550214,6.577617375342355,3.200105455600137,172,3887.47,3865.343374729156,99999.0
+-uesfm,Alcatraz Water Tower,2,2.200189184784068,1.3839747978399466,6.854463337738994,3.1730434024886014,172,3819.31,3799.390902757645,99999.0
+-esfm,Buddah Tooth Relic Temple Singapore,0,1.3132762212436329,0.31652548993670354,9.903890396062208,4.634952217191846,162,2921.45,2903.059753894806,99999.0
+-esfm,Buddah Tooth Relic Temple Singapore,1,0.9908742070528852,0.2335767172234499,9.836436953133868,4.661767906617449,162,2915.39,2904.387546062469,99999.0
+-esfm,Buddah Tooth Relic Temple Singapore,2,1.089403690069201,0.2896515999941069,10.151103921611115,4.628183275251743,162,2913.67,2902.928613901138,99999.0
+-uesfm,Buddah Tooth Relic Temple Singapore,0,1.9767734026333772,0.5777116480820644,15.584999477939833,8.800407928031486,162,4315.97,4296.739196300507,99999.0
+-uesfm,Buddah Tooth Relic Temple Singapore,1,1.4532821126601407,0.345133635718661,16.753182673783428,9.555989756393366,162,4334.28,4311.474883317947,99999.0
+-uesfm,Buddah Tooth Relic Temple Singapore,2,32.81105779154638,6.0024233739495125,26.564606536203048,13.925597303414058,162,4312.37,4291.042752504349,99999.0
+-esfm,Doge Palace Venice,0,0.9741701978804636,0.2578214102266912,6.511459231891615,3.6947627756511525,241,28795.18,28746.22650837898,99999.0
+-esfm,Doge Palace Venice,1,0.627860879835449,0.1773446161890865,6.351890452986879,3.537139014829222,241,28742.27,28699.91308617592,99999.0
+-uesfm,Doge Palace Venice,0,0.9110673544913167,0.2904399918743801,10.966183720673449,6.771416746021886,241,21074.95,21017.30448675156,99999.0
+-esfm,Door Lund,0,0.01167051570787235,0.003654424593160691,1.8552512563951393,0.9492901559591532,12,4892.41,4878.351888656616,99999.0
+-esfm,Door Lund,1,0.009685491454411136,0.0030270923396151107,1.6462990657331118,0.9049495114630693,12,4884.51,4877.529450178146,99999.0
+-esfm,Door Lund,2,0.02068285629921104,0.005193423834514985,1.933833706356197,0.9506754982524039,12,4888.17,4880.201772212982,99999.0
+-uesfm,Door Lund,0,14.059454220682019,0.9733778175929079,11.85413335315025,6.517535151323047,12,9185.47,8715.343282699585,95000.0
+-uesfm,Door Lund,1,14.111993883089163,0.9618908892808385,11.853785941610944,6.504290333453439,12,9203.34,8729.614821910858,95000.0
+-uesfm,Door Lund,2,14.344083655704312,0.9193773651816372,12.058851721280973,6.643622749322514,12,9176.97,9162.127201795578,99999.0
+-esfm,Drinking Fountain Somewhere In Zurich,0,25.804179845300204,1.1242096722251314,8.219025541657992,3.3464550934365884,14,888.07,874.4874215126038,99999.0
+-esfm,Drinking Fountain Somewhere In Zurich,1,25.754577171528883,1.1271567319301725,8.245599860088616,3.352895635561939,14,882.33,873.2415659427643,99999.0
+-esfm,Drinking Fountain Somewhere In Zurich,2,0.019604308735801684,0.006780729271172703,1.7090486008047308,0.6375575382268279,14,881.11,873.6504108905792,99999.0
+-uesfm,Drinking Fountain Somewhere In Zurich,0,26.27952541668808,1.102546079947844,8.40971803772888,3.4593948661771203,14,2969.96,2950.480526924133,99999.0
+-uesfm,Drinking Fountain Somewhere In Zurich,1,26.264849988219943,1.1078406736783482,8.547474098662555,3.585334207185317,14,2960.9,2504.42572927475,85000.0
+-uesfm,Drinking Fountain Somewhere In Zurich,2,0.06235835436589852,0.011103715758719332,2.5743283079384534,1.5361135173722633,14,2960.08,2945.673992872238,99999.0
+-esfm,East Indiaman Goteborg,0,6.033696331509855,1.0773322362193534,7.427433157074123,2.292729458380311,179,3990.25,3970.732353687286,99999.0
+-esfm,East Indiaman Goteborg,1,5.313967690277326,0.9356515362150493,7.350753973659207,2.2747477951684703,179,3986.66,3970.914752960205,99999.0
+-esfm,East Indiaman Goteborg,2,5.357164002763642,0.9254557315236661,7.174538078735992,2.2493605774605823,179,3984.64,3967.810528755188,99999.0
+-uesfm,East Indiaman Goteborg,0,4.992588850190341,0.9093886386013725,9.930437318497303,4.011819888270754,179,5247.16,5221.816437721252,99999.0
+-uesfm,East Indiaman Goteborg,1,5.382092262718057,0.9927766021898647,10.035611563702533,3.912004668740913,179,5245.65,5210.653891563416,99999.0
+-uesfm,East Indiaman Goteborg,2,5.554715231405653,0.9752252406138763,9.5551521706325,3.7417116053580193,179,5231.48,5207.815300703049,99999.0
+-esfm,Ecole Superior De Guerre,0,0.4863901164687255,0.1239377922325492,3.750666445958877,2.0528113108131016,35,2385.33,2371.708672523499,99999.0
+-esfm,Ecole Superior De Guerre,1,0.3031794677170201,0.07921628898965839,3.2456935905644406,1.730123845260616,35,2379.34,2371.191816806793,99999.0
+-esfm,Ecole Superior De Guerre,2,0.2580439805296837,0.06772089648290676,3.257951412507941,1.687291979251782,35,2376.49,2368.935952663422,99999.0
+-uesfm,Ecole Superior De Guerre,0,0.28243957678843373,0.0778675008788964,5.291358032458062,3.601773151002547,35,4647.51,4622.850021600723,99999.0
+-uesfm,Ecole Superior De Guerre,1,0.25507313740145815,0.06695766472508183,5.170495431958795,3.5179325723573998,35,4636.07,4616.890106916428,99999.0
+-uesfm,Ecole Superior De Guerre,2,0.3137499276035537,0.08683967313665406,5.583934687507232,3.7813953498055604,35,4617.92,4602.193422079086,99999.0
+-esfm,Eglise du dome,0,0.3242410065346163,0.0824797769992069,3.906129661328729,1.3512292775673158,85,25054.74,25028.64922428131,99999.0
+-esfm,Eglise du dome,1,0.3771623368290559,0.09432763361451474,3.692490941110161,1.297032357665518,85,25144.16,25116.01952123642,99999.0
+-uesfm,Eglise du dome,0,45.800562123362404,6.451383786272971,7.963674120656965,3.986938865913781,85,23371.73,23333.49408817291,99999.0
+-esfm,Folke Filbyter,0,89.03873163576341,0.12895560134543632,17.986624161769267,7.423093846189277,40,1582.8,1573.360969305038,99999.0
+-esfm,Folke Filbyter,1,84.74740427947457,0.13132264058966686,17.98706393015447,7.385900243615879,40,1579.13,1572.079373121262,99999.0
+-esfm,Folke Filbyter,2,83.70817707245973,0.1261878953205589,21.85200129525089,6.324245998829542,40,1583.76,1576.207699537277,99999.0
+-uesfm,Folke Filbyter,0,80.34590333370599,0.13169898199385566,22.83327162619912,11.711593158296864,40,3701.49,3129.452286243439,85000.0
+-uesfm,Folke Filbyter,1,81.28676564061475,0.1282318052949447,24.480394314414323,9.74969805057316,40,3698.11,3681.891911745071,99999.0
+-uesfm,Folke Filbyter,2,82.23270227083374,0.1312531386108397,22.153947807931974,10.961536903594071,40,3709.96,3695.84988451004,99999.0
+-esfm,Fort Channing Gate Singapore,0,0.10143646071220622,0.04531774433800129,1.7158776598376824,0.8165178663538488,27,3438.39,3428.288990497589,99999.0
+-esfm,Fort Channing Gate Singapore,1,0.08873951454303304,0.039001318240256114,1.783712095722291,0.8156020254191567,27,3430.88,3424.490573883057,99999.0
+-esfm,Fort Channing Gate Singapore,2,0.09031224636384508,0.03957289173255022,1.850800887931411,0.820658413919779,27,3433.07,3426.547735452652,99999.0
+-uesfm,Fort Channing Gate Singapore,0,33.158343156383296,11.908537566474344,12.69248500308198,5.438758486477886,27,6209.29,6192.107179880142,99999.0
+-uesfm,Fort Channing Gate Singapore,1,0.07319798057764343,0.03541606344760984,3.4819579894798927,2.2085534812612853,27,6159.7,6144.797720909119,99999.0
+-uesfm,Fort Channing Gate Singapore,2,0.15394831153888047,0.0686769122909015,3.77993377770425,2.2995094868650368,27,6147.27,6132.745970487595,99999.0
+-esfm,Golden Statue Somewhere In Hong Kong,0,0.23894829924734365,0.053817571637812096,1.7239352536484445,0.744089758310296,18,4096.74,4087.775670289993,99999.0
+-esfm,Golden Statue Somewhere In Hong Kong,1,0.2233990458438394,0.053061028269618456,1.8936878914364637,0.7746436618607616,18,4096.28,4088.994391202927,99999.0
+-esfm,Golden Statue Somewhere In Hong Kong,2,0.23397914670516012,0.051137498616263405,1.87070255923672,0.7763877145267694,18,4094.54,4087.674698591232,99999.0
+-uesfm,Golden Statue Somewhere In Hong Kong,0,0.36074365888948035,0.06894135246785482,2.939092008443413,1.863811153764833,18,7761.69,7746.569520235062,99999.0
+-uesfm,Golden Statue Somewhere In Hong Kong,1,0.35638930493960375,0.06132592684828702,2.686157398440538,1.735571387461727,18,7759.16,7745.389947891235,99999.0
+-uesfm,Golden Statue Somewhere In Hong Kong,2,0.3955566726941678,0.09197790539367184,2.878341172309852,1.8164738202745854,18,7766.46,7751.458931922913,99999.0
+-esfm,Gustav Vasa,0,3.4881363194229826,0.2273181104934767,1.7369784534725496,0.9521606063342205,18,711.53,703.3091380596161,99999.0
+-esfm,Gustav Vasa,1,3.4307883498221035,0.21432466556133864,1.5556448596237022,0.9232184455756751,18,710.28,703.9906163215637,99999.0
+-esfm,Gustav Vasa,2,3.6089062562451386,0.2196578287395002,1.673803634770729,0.9429755567757121,18,707.79,701.5903799533844,99999.0
+-uesfm,Gustav Vasa,0,5.828154714648938,0.32510821331670736,3.0711042143581646,2.4170166616864357,18,2566.0,2547.931907176971,99999.0
+-uesfm,Gustav Vasa,1,6.515239842974278,0.35941242702736925,3.089677100739632,2.4476428398062584,18,2530.9,1385.089058637619,55000.0
+-uesfm,Gustav Vasa,2,5.954036423569478,0.34942942401850696,3.101296679577594,2.400483554095863,18,2533.76,2520.798699855804,99999.0
+-esfm,GustavIIAdolf,0,46.92758389264633,8.570499881283022,11.684017169857798,5.131125459552879,57,1047.82,1038.018162250519,99999.0
+-esfm,GustavIIAdolf,1,47.21999673071185,8.515052769156952,11.695065125063977,5.091617086264161,57,1041.41,1034.5799472332,99999.0
+-esfm,GustavIIAdolf,2,47.2312223396789,8.523261295519099,11.787583212607423,5.2715388868550574,57,1043.28,1036.478415489197,99999.0
+-uesfm,GustavIIAdolf,0,89.37607436538163,13.276440526172165,15.53883714368065,5.864332719302596,57,2904.42,2881.687849521637,99999.0
+-uesfm,GustavIIAdolf,1,70.16367503302509,11.555242253627961,15.162174576474532,6.9709512967490355,57,2884.62,2870.103493452072,99999.0
+-uesfm,GustavIIAdolf,2,72.72996778765942,11.054260851523775,16.85269001998736,9.009102977660282,57,2888.7,2874.096453428268,99999.0
+-esfm,Jonas Ahlstromer,0,64.46662959289324,11.985605264945464,12.602584968179986,2.9168445332917745,40,614.88,607.3498430252075,99999.0
+-esfm,Jonas Ahlstromer,1,38.42510202464649,8.741159740710629,10.25268858530355,3.7024742418884333,40,618.73,611.5567183494568,99999.0
+-esfm,Jonas Ahlstromer,2,43.32891670826628,8.741053567623224,10.553031866494822,3.031717304343782,40,612.74,606.6802380084991,99999.0
+-uesfm,Jonas Ahlstromer,0,42.30073879858729,10.272232427418913,9.098970988131883,3.992857074878733,40,2339.57,1280.604686737061,55000.0
+-uesfm,Jonas Ahlstromer,1,42.18412804385421,10.310808592826955,8.453808908218651,3.4329852879523464,40,2334.33,2320.362742424011,99999.0
+-uesfm,Jonas Ahlstromer,2,42.25301394187691,10.315519611163806,8.793789077520445,3.8046125347596442,40,2311.04,2297.139800786972,99999.0
+-esfm,Kings College University Of Toronto,0,14.400461796497643,3.2653370282498866,4.064353916685246,1.6753245173381437,77,1074.94,1066.306197404861,99999.0
+-esfm,Kings College University Of Toronto,1,11.956699445712928,2.705452710546247,3.9727144140636783,1.5797087908756111,77,1071.64,1064.129351377487,99999.0
+-esfm,Kings College University Of Toronto,2,0.7191591237413282,0.16178494376153177,2.656807607821862,1.1949677972638288,77,1090.36,1082.987608671188,99999.0
+-uesfm,Kings College University Of Toronto,0,31.932532001930838,2.9775501815475,6.099555590572752,3.8717713150174773,77,2884.23,2866.194714784622,99999.0
+-uesfm,Kings College University Of Toronto,1,2.4241467960672063,0.41577835428574456,4.631696476065898,3.148046400399828,77,2899.51,2884.114916801453,99999.0
+-uesfm,Kings College University Of Toronto,2,30.212552334890066,2.3209884971628156,5.844709739288878,3.754268841443037,77,2874.99,2860.901805400848,99999.0
+-esfm,Lund University Sphinx,0,13.661465370705821,3.5591094267381553,6.242667318159146,2.715620555070548,70,2718.96,2709.5872194767,99999.0
+-esfm,Lund University Sphinx,1,12.524302160109885,3.2130942199657664,6.1182420215295075,2.7468468987195953,70,2722.86,2715.36418390274,99999.0
+-esfm,Lund University Sphinx,2,7.98961938632192,2.1710959765491817,5.807289242508196,2.893370267054225,70,2713.88,2706.478076219559,99999.0
+-uesfm,Lund University Sphinx,0,15.165716414733652,3.5564827651349713,10.803408462752863,6.730422995453065,70,4644.92,4626.135177612305,99999.0
+-uesfm,Lund University Sphinx,1,68.88899086055604,13.548881652837478,14.336626262363456,7.626356837439143,70,4703.31,4687.50173997879,99999.0
+-uesfm,Lund University Sphinx,2,15.014326075576053,3.5607694164340526,11.039432108003597,6.81081192121135,70,4615.6,4600.294319868088,99999.0
+-esfm,Nijo Castle Gate,0,0.6521524412276166,0.13305866403975986,7.404953325705311,3.0629804954987674,19,1097.29,1084.301982402802,99999.0
+-esfm,Nijo Castle Gate,1,0.8564043952718835,0.16950374865879153,7.358076475544799,3.1235223975185864,19,1094.89,1084.640900611877,99999.0
+-esfm,Nijo Castle Gate,2,1.0089926380330574,0.19095470213426863,7.481049991287358,3.1671779658403687,19,1093.35,1083.446812152863,99999.0
+-uesfm,Nijo Castle Gate,0,1.6923297320319277,0.3246009364794658,11.792009654594867,6.945622003724464,19,3310.13,3288.75860619545,99999.0
+-uesfm,Nijo Castle Gate,1,1.6697791743236226,0.32440429939345133,12.704681259536075,7.466128811395402,19,3308.19,3290.209539175034,99999.0
+-uesfm,Nijo Castle Gate,2,1.682403384345942,0.32817821817561266,11.917977420973372,7.102279949089778,19,3301.77,3284.964460611343,99999.0
+-uesfm,Pantheon Paris,0,0.3823220210788846,0.05987691585642601,7.649321203280243,4.797127317249956,179,5506.88,5473.694003582001,99999.0
+-uesfm,Pantheon Paris,1,45.61471478909275,6.671068486991568,16.912359249810606,6.602415648206237,179,5491.31,5459.329566717148,99999.0
+-uesfm,Pantheon Paris,2,12.03796171625132,2.465102828965073,16.125095834055482,6.860965809143959,179,5493.94,5463.555550813675,99999.0
+-esfm,Park Gate Clermont Ferrand,0,19.399216821978534,9.438471236639753,7.521969683469249,3.892136696028138,34,1597.96,1588.07075381279,99999.0
+-esfm,Park Gate Clermont Ferrand,1,19.325402297843297,9.417998214523331,7.465007292717891,3.849088919564891,34,1596.87,1589.703137874603,99999.0
+-esfm,Park Gate Clermont Ferrand,2,25.440611579781365,11.720641869372841,8.15032188787613,4.0871739024940315,34,1595.99,1588.841923236847,99999.0
+-uesfm,Park Gate Clermont Ferrand,0,19.27114964212729,9.381930985797803,7.856513901068924,4.317004881495189,34,3598.69,3580.396904230118,99999.0
+-uesfm,Park Gate Clermont Ferrand,1,19.288145181777256,9.394362801959678,7.762082007624792,4.207851936960509,34,3592.66,3577.494926214218,99999.0
+-uesfm,Park Gate Clermont Ferrand,2,19.345727373971844,9.415066249707772,7.936652645099164,4.425507919979627,34,3600.86,3581.473034620285,99999.0
+-esfm,Plaza De Armas Santiago,0,0.7341628728579594,0.32156050159900956,6.000154308231358,2.679971945741675,240,6375.02,6333.455562829971,99999.0
+-esfm,Plaza De Armas Santiago,1,0.5176165701053533,0.22744923951256435,5.661576462867733,2.533952088248649,240,6373.68,6336.030319213867,99999.0
+-esfm,Plaza De Armas Santiago,2,1.0740077415291467,0.46304622323619105,6.365716833190761,2.733195456670397,240,6369.35,6331.436369419098,99999.0
+-uesfm,Plaza De Armas Santiago,0,6.6772176498607925,2.909933722327364,11.384209379387066,5.623987151554922,240,6540.76,6491.619550704956,99999.0
+-uesfm,Plaza De Armas Santiago,1,1.3364898126442175,0.5686650140764532,7.951759184923103,3.933795015892702,240,6536.61,6482.320575714111,99999.0
+-uesfm,Plaza De Armas Santiago,2,1.9671483940705004,0.8276654865349118,9.054185015150745,4.441179120460681,240,6529.24,6477.723381996155,99999.0
+-esfm,Porta San Donato Bologna,0,0.5746001989174886,0.10608894230214515,5.691906895070585,2.4183460353027986,141,3751.75,3734.281898736954,99999.0
+-esfm,Porta San Donato Bologna,1,0.8978003419853328,0.14474404936935045,5.682379581165418,2.563141107750346,141,3758.51,3742.688964605331,99999.0
+-esfm,Porta San Donato Bologna,2,0.30440272958620346,0.07026546193502164,5.435311727517131,2.1626312981864086,141,3752.23,3738.056202173233,99999.0
+-uesfm,Porta San Donato Bologna,0,1.1269164513732723,0.19275737956525044,10.071225650489282,5.557466375146339,141,5111.57,5083.21807050705,99999.0
+-uesfm,Porta San Donato Bologna,1,60.66366912240625,10.234123128225525,19.593360464637342,9.305574874002819,141,5125.57,5098.085618019104,99999.0
+-uesfm,Porta San Donato Bologna,2,1.2271289404198877,0.19605254084885107,10.066750816092268,5.595831381287904,141,5116.95,5093.473650932312,99999.0
+-esfm,Round Church Cambridge,0,2.1173198662081005,0.9129854085205968,5.854320175654564,1.9683268910531708,92,9788.65,9759.719386577606,99999.0
+-esfm,Round Church Cambridge,1,2.1415279636494953,0.9267927094737857,6.33905951937897,2.0439997148640834,92,9764.61,9741.356333732605,99999.0
+-uesfm,Round Church Cambridge,0,2.5434984075631974,1.0219048587087352,8.914080922081759,4.131734829440289,92,12088.06,12047.02147889137,99999.0
+-uesfm,Round Church Cambridge,1,2.435781054756339,0.9827352719070902,8.178482224933683,3.8400051230290893,92,12034.64,11993.72608184814,99999.0
+-esfm,Skansen Kronan Gothenburg,0,0.3380139741078169,0.1158931606908392,3.0478862369293624,1.5340801117570733,131,17163.94,17140.15425729752,99999.0
+-uesfm,Skansen Kronan Gothenburg,0,0.37472494768162085,0.1243676312212316,4.798736338080396,3.348916057055834,131,16235.09,16202.21351981163,99999.0
+-esfm,Smolny Cathedral St Petersburg,0,21.81278753416543,2.110667559831311,11.2315357584775,6.379259553776089,131,10364.67,10320.51852750778,99999.0
+-esfm,Smolny Cathedral St Petersburg,1,21.799828748358873,2.111313741354268,11.225141339870355,6.36605939788532,131,10360.52,10319.01374697685,99999.0
+-uesfm,Smolny Cathedral St Petersburg,0,22.143392894578028,2.0985584569799265,12.52645725687386,7.895265841408903,131,10475.74,10416.62969732285,99999.0
+-esfm,Sri Mariamman Singapore,0,1.4920493985513708,0.4412389781578158,10.406406306167431,6.369585247957454,222,5987.98,5970.066485881805,99999.0
+-esfm,Sri Mariamman Singapore,1,1.1345564396944785,0.3608906720626337,9.703927906328842,5.885963785879842,222,5993.62,5975.365797519684,99999.0
+-esfm,Sri Mariamman Singapore,2,1.0293707921546553,0.3426709250774071,10.15220066698994,6.067537729054635,222,5975.02,5958.786058187485,99999.0
+-uesfm,Sri Mariamman Singapore,0,16.417883129790784,5.526406595652791,19.962711816743173,12.18830031311293,222,6531.45,6494.750840425491,99999.0
+-uesfm,Sri Mariamman Singapore,1,1.3793173305528281,0.5014534676018023,16.10857603648069,10.651562750176396,222,6467.65,6426.201541185379,99999.0
+-uesfm,Sri Mariamman Singapore,2,1.639001766282598,0.536397596737868,17.490398022274007,11.439337896248928,222,6457.9,6427.717175960541,99999.0
+-esfm,Sri Thendayuthapani Singapore,0,0.7190396606626565,0.17922550285921848,7.12365374281766,2.231392236341273,98,13362.09,13320.09781932831,99999.0
+-uesfm,Sri Thendayuthapani Singapore,0,47.0750455982536,3.8553135654690074,14.133068630999212,7.117179633590087,98,13808.44,13754.94554877281,99999.0
+-esfm,Sri Veeramakaliamman Singapore,0,2.333869921178247,0.542847254574005,11.965270419451508,5.472877662905037,157,12392.18,12365.15180635452,99999.0
+-uesfm,Sri Veeramakaliamman Singapore,0,29.40487246624615,3.8676026992817505,28.950500021302087,14.32295224594287,157,12544.69,12504.26410722733,99999.0
+-esfm,Statue Of Liberty,0,52.35559762033152,28.241594953600423,688383.1603142131,683624.423417858,134,102.64,43.32754778862,45000.0
+-esfm,Statue Of Liberty,1,52.36920727661949,28.627801879821945,9277.715116667516,9303.546387753544,134,99.78,59.98358988761902,65000.0
+-esfm,Statue Of Liberty,2,52.3478568435799,28.64052858422677,7449.598084265948,7416.803834848509,134,97.28,3.520912647247314,0.0
+-uesfm,Statue Of Liberty,0,40.92379801166407,19.915794833689585,48.781849500680366,25.904882064330994,134,12927.11,12902.23110127449,99999.0
+-uesfm,Statue Of Liberty,1,38.79484508812604,17.090491683338875,33.27459817133403,17.608726936833005,134,12920.89,12896.32496738434,99999.0
+-esfm,The Pumpkin,0,27.59459354015253,5.94835892276036,27.52772578222832,9.11126349703041,196,4330.9,4316.109169960022,99999.0
+-esfm,The Pumpkin,1,26.54397520858153,5.701234509093555,27.342758332715125,8.907676505669826,196,4327.98,4314.043692827225,99999.0
+-esfm,The Pumpkin,2,23.05395646123413,5.007378502393024,25.75879879971585,7.898438464943201,196,4322.94,4310.356798410416,99999.0
+-uesfm,The Pumpkin,0,25.490608822831348,5.440568114393512,34.198429037653845,15.6024365790899,196,5793.03,5767.982522964478,99999.0
+-uesfm,The Pumpkin,1,27.26556900533122,5.820572144056583,35.6951097814524,16.572200360468877,196,5785.61,5760.521738529205,99999.0
+-uesfm,The Pumpkin,2,26.95722616022698,5.73349875876088,36.207441893083136,17.07678356860021,196,5793.15,5768.243654251099,99999.0
+-esfm,Thian Hook Keng Temple Singapore,0,1.0653505173922067,0.1079565219850111,15.054152885837562,5.81055046268767,138,3299.26,3281.187238931656,99999.0
+-esfm,Thian Hook Keng Temple Singapore,1,0.8390930608109779,0.08272919436528102,14.587041005106803,5.705702663819475,138,3306.93,3283.257929086685,99999.0
+-esfm,Thian Hook Keng Temple Singapore,2,0.8760513596629002,0.08764395630143504,15.102705362055175,5.7567965038255124,138,3289.42,3278.173619031906,99999.0
+-uesfm,Thian Hook Keng Temple Singapore,0,1.146131162829257,0.10799243740327723,20.24888615229727,10.297158058757331,138,4626.48,4595.79723072052,99999.0
+-uesfm,Thian Hook Keng Temple Singapore,1,1.2072956626750737,0.12821580405567637,21.242388560332195,10.655698470690366,138,4600.34,4578.699931621552,99999.0
+-uesfm,Thian Hook Keng Temple Singapore,2,1.251054659622132,0.12561972972485672,20.875018460120653,10.416875106878496,138,4610.3,4581.792747020721,99999.0
+-esfm,Tsar Nikolai I,0,42.507553768489565,8.869914103947062,10.846690028504929,4.007961025902478,98,4932.15,4920.938139438629,99999.0
+-esfm,Tsar Nikolai I,1,43.09835370739786,8.983133733509343,11.402130572143294,4.032588092580996,98,4924.71,4914.929541349411,99999.0
+-uesfm,Tsar Nikolai I,0,75.94880014543322,15.146573953653359,15.307363921859029,6.734065570805788,98,6652.35,6629.384099006653,99999.0
+-uesfm,Tsar Nikolai I,1,80.87807139992418,15.64102228288136,21.553650044939403,9.179513384124702,98,6641.34,6623.797205209732,99999.0
+-esfm,Urban II,0,59.0297968076411,11.008489348760156,17.600594623525765,5.400933781867582,96,8889.69,8878.01452589035,99999.0
+-esfm,Vercingetorix,0,79.07478057258302,9.776081681696928,6.649989325889541,1.7432748253588333,69,1223.75,1212.340968608856,99999.0
+-esfm,Vercingetorix,1,80.09313371434186,9.858626097407651,6.426263444651325,1.6979463697629655,69,1222.8,1213.763489961624,99999.0
+-esfm,Vercingetorix,2,89.68787080902364,10.950615257181747,8.695412468830675,3.360645595149667,69,1225.36,1215.808262825012,99999.0
+-uesfm,Vercingetorix,0,78.95942988866727,10.356244187384963,8.07065603849916,3.372297052368834,69,3105.42,3085.208213567734,99999.0
+-uesfm,Vercingetorix,1,89.99175121301754,11.178484677344343,8.929255781006072,4.302365177465691,69,3105.88,3091.564247369766,99999.0
+-uesfm,Vercingetorix,2,89.31151857112468,11.134780185118505,9.044030033429575,4.351952473195928,69,3105.52,3089.0203332901,99999.0
+-esfm,Yueh Hai Ching Temple Singapore,0,0.6268598783684821,0.08360284283022794,5.56223941139456,2.132543748253723,43,1749.28,1738.747783660889,99999.0
++method,scene,seed,rot_err_deg,pos_err,reproj_err_px,reproj_med_px,n_cams,runtime_s,convergence_time_s,best_epoch,rot_err_deg_ba,pos_err_ba,reproj_err_px_ba
++esfm,Alcatraz Courtyard,0,0.324288380204305,0.0854554642858637,3.379232694353053,1.9204072326560708,133,4821.4,4792.707528591156,99999.0,0.04336579548315269,0.01736823028852939,0.8927630229850309
++esfm,Alcatraz Courtyard,1,0.36138678157391557,0.09112912757839253,3.3691035794153503,1.8536696768772378,133,4825.72,4804.594581127167,99999.0,0.04361630830506929,0.01730885636433267,1.11229287420467
++esfm,Alcatraz Courtyard,2,0.4015043649148433,0.10235190268069758,3.557242072642648,1.9198007399315369,133,4833.02,4812.242679834366,99999.0,0.04372553753273131,0.01736657742692317,0.8899897580452283
++esfm_rc,Alcatraz Courtyard,0,0.40021217864007563,0.10366874053275751,3.6525220170666364,2.116170425959351,133,5678.34,5647.516944408417,99999.0,,,
++esfm_rc,Alcatraz Courtyard,1,0.4164699362576677,0.10735288670743451,3.553972406799646,2.0080654095255728,133,5692.91,5662.111888885498,99999.0,,,
++esfm_rc,Alcatraz Courtyard,2,0.443321991451076,0.11848499800783799,3.8114044579296285,2.1404981515498216,133,5681.11,5649.007675886154,99999.0,,,
++uesfm,Alcatraz Courtyard,0,0.5509063563081962,0.14050635120746804,5.528957815039945,3.569220349534855,133,6556.5,6511.41095662117,99999.0,,,
++uesfm,Alcatraz Courtyard,1,0.6823224031499351,0.17277455831381494,5.435931436314427,3.4771386471547348,133,6552.19,6506.419185638428,99999.0,,,
++uesfm,Alcatraz Courtyard,2,29.163200369074893,6.342299350838519,10.849871644564203,6.781661682554276,133,6544.26,6512.667362689972,99999.0,,,
++uesfm_abl,Alcatraz Courtyard,0,0.40021217864007563,0.10366874053275751,3.6525220170666364,2.116170425959351,133,5706.16,5667.877521038055,99999.0,,,
++uesfm_abl,Alcatraz Courtyard,1,0.4164699362576677,0.10735288670743451,3.553972406799646,2.0080654095255728,133,5691.01,5660.514966249466,99999.0,,,
++uesfm_abl,Alcatraz Courtyard,2,0.443321991451076,0.11848499800783799,3.8114044579296285,2.1404981515498216,133,5711.81,5680.801262140274,99999.0,,,
++esfm,Alcatraz Water Tower,0,0.6984432687370988,0.38694796813795407,4.741313233832226,2.1058898944784543,172,2461.59,2440.421297073364,99999.0,0.22768614198983136,0.11526567721949761,0.77419350794262
++esfm,Alcatraz Water Tower,1,0.5944425806062753,0.3287537445614408,4.5656682619915765,1.9703698309044646,172,2467.28,2454.429310798645,99999.0,0.2277265457963764,0.11534572613922207,0.658390729293639
++esfm,Alcatraz Water Tower,2,0.7051984712289041,0.39359395644708545,4.982802041506025,2.1318655747232063,172,2455.14,2441.783402681351,99999.0,0.2264345069130526,0.11472488848384464,0.7908639514877002
++esfm_rc,Alcatraz Water Tower,0,0.9705085738488239,0.5332154449748415,5.147471061152484,2.310649130340069,172,3074.8,3046.602040290833,99999.0,,,
++esfm_rc,Alcatraz Water Tower,1,0.7440758749948073,0.4145867394737888,4.961714044617586,2.211091062168503,172,3061.2,3039.165870904922,99999.0,,,
++esfm_rc,Alcatraz Water Tower,2,0.9302629778601009,0.51517638469428,5.227327015979294,2.2724984979322977,172,3056.55,3035.032587766647,99999.0,,,
++uesfm,Alcatraz Water Tower,0,2.016029335283383,1.087082854315953,7.9502009315303575,4.413488326253207,172,3805.19,3775.908159971237,99999.0,,,
++uesfm,Alcatraz Water Tower,1,1.4259772151091428,0.7645994701417347,7.177871865477131,3.7683369835989504,172,3787.07,3763.27596282959,99999.0,,,
++uesfm,Alcatraz Water Tower,2,0.9982626119141534,0.558403777925872,7.063395425015731,3.814475911180562,172,3795.37,3773.579354286194,99999.0,,,
++uesfm_abl,Alcatraz Water Tower,0,0.9705085738488239,0.5332154449748415,5.147471061152484,2.310649130340069,172,3108.65,3076.951498508453,99999.0,,,
++uesfm_abl,Alcatraz Water Tower,1,0.7440758749948073,0.4145867394737888,4.961714044617586,2.211091062168503,172,3209.17,3181.968911647797,99999.0,,,
++uesfm_abl,Alcatraz Water Tower,2,0.9302629778601009,0.51517638469428,5.227327015979294,2.2724984979322977,172,3092.82,3063.191012859344,99999.0,,,
++esfm,Buddah Tooth Relic Temple Singapore,0,1.3132762212436329,0.31652548993670354,9.903890396062208,4.634952217191846,162,2921.45,2903.059753894806,99999.0,0.08653940053085564,0.016105041532649904,1.9758619604429803
++esfm,Buddah Tooth Relic Temple Singapore,1,0.9908742070528852,0.2335767172234499,9.836436953133868,4.661767906617449,162,2915.39,2904.387546062469,99999.0,0.0849468509455147,0.015653062410540777,1.9626282085900335
++esfm,Buddah Tooth Relic Temple Singapore,2,1.089403690069201,0.2896515999941069,10.151103921611115,4.628183275251743,162,2913.67,2902.928613901138,99999.0,0.08666865718094915,0.016308330450717633,2.2978504476586723
++esfm_rc,Buddah Tooth Relic Temple Singapore,0,1.4156219496055027,0.3849696818920549,10.910671400016733,5.192142953606055,162,3558.33,3528.124534845352,99999.0,,,
++esfm_rc,Buddah Tooth Relic Temple Singapore,1,1.1046287069544334,0.2802093099622587,10.446776494877133,5.042054116563641,162,3542.82,3523.402928352356,99999.0,,,
++esfm_rc,Buddah Tooth Relic Temple Singapore,2,1.2816861979534584,0.34711677570751104,10.873035865566612,5.101037865376439,162,3533.26,3513.161389827728,99999.0,,,
++uesfm,Buddah Tooth Relic Temple Singapore,0,1.5799948547864517,0.43886498334860274,13.484389374830494,7.719072745032001,162,4294.34,4266.538989067078,99999.0,,,
++uesfm,Buddah Tooth Relic Temple Singapore,1,1.934129180062467,0.5924991617860504,13.872322562031941,8.04977435062149,162,4303.62,4259.311170101166,99999.0,,,
++uesfm,Buddah Tooth Relic Temple Singapore,2,1.8591664615796053,0.5279087705483619,14.731758217446671,8.481044153007865,162,4270.55,4249.088565349579,99999.0,,,
++uesfm_abl,Buddah Tooth Relic Temple Singapore,0,1.4156219496055027,0.3849696818920549,10.910671400016733,5.192142953606055,162,3537.56,3511.595267057419,99999.0,,,
++uesfm_abl,Buddah Tooth Relic Temple Singapore,1,1.1046287069544334,0.2802093099622587,10.446776494877133,5.042054116563641,162,3525.65,3502.759250164032,99999.0,,,
++uesfm_abl,Buddah Tooth Relic Temple Singapore,2,1.2816861979534584,0.34711677570751104,10.873035865566612,5.101037865376439,162,3516.21,3493.906430959702,99999.0,,,
++esfm,Doge Palace Venice,0,0.9741701978804636,0.2578214102266912,6.511459231891615,3.6947627756511525,241,28795.18,28746.22650837898,99999.0,0.07156237283091285,0.01915571946233311,1.1580904809936006
++esfm,Doge Palace Venice,1,0.627860879835449,0.1773446161890865,6.351890452986879,3.537139014829222,241,28742.27,28699.91308617592,99999.0,0.050631712294427526,0.017261384802431806,1.1986047342873392
++esfm,Doge Palace Venice,2,0.6576305681966796,0.19114542381679772,6.214841374149415,3.591638795332466,241,10030.76,9989.16938996315,99999.0,0.0477845991264688,0.01734493189317512,1.1742938182004765
++esfm_rc,Doge Palace Venice,0,1.0500985034194232,0.2872157262518388,6.790948674732308,4.010160174021759,241,7952.55,7887.024859905243,99999.0,,,
++esfm_rc,Doge Palace Venice,1,0.822972055147261,0.234503198836285,6.887079459513741,3.9072413922164637,241,7919.49,7863.557105779648,99999.0,,,
++esfm_rc,Doge Palace Venice,2,0.7680084929628118,0.22652177619664943,6.815847430201609,3.913545550189034,241,7919.45,7864.596030473709,99999.0,,,
++uesfm,Doge Palace Venice,0,1.120669822868001,0.3500031006198606,10.496216444068507,6.9730858315176345,241,20767.26,20701.21099281311,99999.0,,,
++uesfm_abl,Doge Palace Venice,0,1.0500985034194232,0.2872157262518388,6.790948674732308,4.010160174021759,241,8102.36,8027.503306865692,99999.0,,,
++uesfm_abl,Doge Palace Venice,1,0.822972055147261,0.234503198836285,6.887079459513741,3.9072413922164637,241,8000.13,7943.610855102539,99999.0,,,
++uesfm_abl,Doge Palace Venice,2,0.7680084929628118,0.22652177619664943,6.815847430201609,3.913545550189034,241,8036.55,7978.251197338104,99999.0,,,
++esfm,Door Lund,0,0.01167051570787235,0.003654424593160691,1.8552512563951393,0.9492901559591532,12,4892.41,4878.351888656616,99999.0,0.002863516481658609,0.0005677797973481404,0.3034458379531829
++esfm,Door Lund,1,0.009685491454411136,0.0030270923396151107,1.6462990657331118,0.9049495114630693,12,4884.51,4877.529450178146,99999.0,0.0028855670751485583,0.0005696749289261176,0.3034450379552205
++esfm,Door Lund,2,0.02068285629921104,0.005193423834514985,1.933833706356197,0.9506754982524039,12,4888.17,4880.201772212982,99999.0,0.0028793804217640893,0.0005687319198631606,0.30345769639617903
++esfm_rc,Door Lund,0,0.013192395227632247,0.004661037914540053,1.778271540426845,0.9602560994879148,12,8464.92,8445.741333007812,99999.0,,,
++esfm_rc,Door Lund,1,0.019167063815313256,0.0029864726992216563,1.8741769497138485,0.9511493437953005,12,8450.0,8434.659740924835,99999.0,,,
++esfm_rc,Door Lund,2,0.02286084013237112,0.006025105467312505,2.0198271044543015,0.9855463158253012,12,8438.17,8421.555475234985,99999.0,,,
++uesfm,Door Lund,0,0.03084538453455712,0.007522151486969653,2.3711871496098955,1.6703501976219666,12,9262.05,9243.771874427795,99999.0,,,
++uesfm,Door Lund,1,13.889154901234134,1.0160235649641791,11.830787694675347,6.5775252925915195,12,9243.42,7848.296544790268,85000.0,,,
++uesfm,Door Lund,2,0.03345748760425064,0.00845459656629652,2.434197612189203,1.6957836630064358,12,9262.36,9247.514908313751,99999.0,,,
++uesfm_abl,Door Lund,0,0.013192395227632247,0.004661037914540053,1.778271540426845,0.9602560994879148,12,8562.39,8544.837433815002,99999.0,,,
++uesfm_abl,Door Lund,1,0.019167063815313256,0.0029864726992216563,1.8741769497138485,0.9511493437953005,12,8478.99,8459.191262245178,99999.0,,,
++uesfm_abl,Door Lund,2,0.02286084013237112,0.006025105467312505,2.0198271044543015,0.9855463158253012,12,8485.62,8462.715673446655,99999.0,,,
++esfm,Drinking Fountain Somewhere In Zurich,0,25.804179845300204,1.1242096722251314,8.219025541657992,3.3464550934365884,14,888.07,874.4874215126038,99999.0,23.41403219049492,1.200238386449365,7.652263637098719
++esfm,Drinking Fountain Somewhere In Zurich,1,25.754577171528883,1.1271567319301725,8.245599860088616,3.352895635561939,14,882.33,873.2415659427643,99999.0,23.424386659175696,1.1914927737957968,7.588314595112779
++esfm,Drinking Fountain Somewhere In Zurich,2,0.019604308735801684,0.006780729271172703,1.7090486008047308,0.6375575382268279,14,881.11,873.6504108905792,99999.0,0.004928146172176949,0.001084436268196593,0.3140688404104962
++esfm_rc,Drinking Fountain Somewhere In Zurich,0,25.966275912639762,1.1148303838604945,8.295888343149453,3.323889859222449,14,2336.21,2202.958295583725,95000.0,,,
++esfm_rc,Drinking Fountain Somewhere In Zurich,1,25.82082253609568,1.1243203937617456,8.27258982311774,3.337923918211656,14,2332.06,2203.267946481705,95000.0,,,
++esfm_rc,Drinking Fountain Somewhere In Zurich,2,0.02664156912090857,0.008991520744418306,1.817273903936106,0.7075879334430649,14,2330.18,2316.842426300049,99999.0,,,
++uesfm,Drinking Fountain Somewhere In Zurich,0,25.946100100284117,1.1178920745623426,8.471712388928934,3.6241397282821626,14,5817.37,5796.999963760376,99999.0,,,
++uesfm,Drinking Fountain Somewhere In Zurich,1,25.973070279327214,1.1166462082604052,8.538877295313494,3.6343523276029472,14,5821.94,5806.947921991348,99999.0,,,
++uesfm,Drinking Fountain Somewhere In Zurich,2,0.06926036271356541,0.01532826351119577,3.015375672575656,1.7451989053599506,14,5806.35,5792.085475206375,99999.0,,,
++uesfm_abl,Drinking Fountain Somewhere In Zurich,0,25.966275912639762,1.1148303838604945,8.295888343149453,3.323889859222449,14,2449.75,2298.556520700455,95000.0,,,
++uesfm_abl,Drinking Fountain Somewhere In Zurich,1,25.82082253609568,1.1243203937617456,8.27258982311774,3.337923918211656,14,2424.52,2287.711306810379,95000.0,,,
++uesfm_abl,Drinking Fountain Somewhere In Zurich,2,0.02664156912090857,0.008991520744418306,1.817273903936106,0.7075879334430649,14,2429.07,2408.744810342789,99999.0,,,
++esfm,East Indiaman Goteborg,0,6.033696331509855,1.0773322362193534,7.427433157074123,2.292729458380311,179,3990.25,3970.732353687286,99999.0,5.5436529958372045,1.0684143055593625,2.580925145978092
++esfm,East Indiaman Goteborg,1,5.313967690277326,0.9356515362150493,7.350753973659207,2.2747477951684703,179,3986.66,3970.914752960205,99999.0,5.452634397810775,1.0476183365286567,2.625738084551857
++esfm,East Indiaman Goteborg,2,5.357164002763642,0.9254557315236661,7.174538078735992,2.2493605774605823,179,3984.64,3967.810528755188,99999.0,4.7401879289520705,0.8636100794526042,2.737932350716639
++esfm_rc,East Indiaman Goteborg,0,5.174865473862546,0.9113234827275644,7.487193880318395,2.400937113397191,179,4375.89,4350.625731945038,99999.0,,,
++esfm_rc,East Indiaman Goteborg,1,6.224967804118117,1.1150553113186616,7.636857918099955,2.3906451402388336,179,4385.52,4363.705677032471,99999.0,,,
++esfm_rc,East Indiaman Goteborg,2,5.188808159202813,0.9073521965754248,7.4841004661072486,2.3403976043137757,179,4396.21,4371.83106970787,99999.0,,,
++uesfm,East Indiaman Goteborg,0,5.365772057886591,0.9552115949640507,9.655123223073948,4.174486084237091,179,5203.6,5178.753549814224,99999.0,,,
++uesfm,East Indiaman Goteborg,1,10.45533482189932,2.0687908277631517,9.98932850274456,4.110771460953736,179,5195.98,5174.958552598953,99999.0,,,
++uesfm,East Indiaman Goteborg,2,5.872735570127935,1.0672842053553182,10.745053220477157,4.480528977857714,179,5194.7,5173.540862560272,99999.0,,,
++uesfm_abl,East Indiaman Goteborg,0,5.174865473862546,0.9113234827275644,7.487193880318395,2.400937113397191,179,4549.74,4433.853548288345,99999.0,,,
++uesfm_abl,East Indiaman Goteborg,1,6.224967804118117,1.1150553113186616,7.636857918099955,2.3906451402388336,179,4290.55,4266.563747644424,99999.0,,,
++uesfm_abl,East Indiaman Goteborg,2,5.188808159202813,0.9073521965754248,7.4841004661072486,2.3403976043137757,179,4299.65,4276.496908426285,99999.0,,,
++esfm,Ecole Superior De Guerre,0,0.4863901164687255,0.1239377922325492,3.750666445958877,2.0528113108131016,35,2385.33,2371.708672523499,99999.0,0.013820220558282399,0.002882938121219649,0.35561699442321343
++esfm,Ecole Superior De Guerre,1,0.3031794677170201,0.07921628898965839,3.2456935905644406,1.730123845260616,35,2379.34,2371.191816806793,99999.0,0.013839621308859032,0.002888897351408931,0.35520606627666845
++esfm,Ecole Superior De Guerre,2,0.2580439805296837,0.06772089648290676,3.257951412507941,1.687291979251782,35,2376.49,2368.935952663422,99999.0,0.013741554515737818,0.002868803314661506,0.35531179719423556
++esfm_rc,Ecole Superior De Guerre,0,6.0615095626482445,1.3902441953970521,6.582603890419177,2.5874556376476794,35,3932.24,3914.314348936081,99999.0,,,
++esfm_rc,Ecole Superior De Guerre,1,0.5222448808102489,0.13311780686828037,3.858194363564449,2.0941111590578783,35,3930.97,3916.866501569748,99999.0,,,
++esfm_rc,Ecole Superior De Guerre,2,0.42670768047814917,0.11109013616311524,3.738376401049213,1.9806437286978857,35,3932.83,3915.012129306793,99999.0,,,
++uesfm,Ecole Superior De Guerre,0,51.32681795107141,2.57407526776939,11.751384768771835,6.449293516058521,35,10426.77,10405.3174200058,99999.0,,,
++uesfm,Ecole Superior De Guerre,1,0.2738850931952059,0.0734625834578241,5.5413290518324265,3.89237755756953,35,10405.39,10384.56669402122,99999.0,,,
++uesfm_abl,Ecole Superior De Guerre,0,6.0615095626482445,1.3902441953970521,6.582603890419177,2.5874556376476794,35,4013.39,3989.901808023453,99999.0,,,
++uesfm_abl,Ecole Superior De Guerre,1,0.5222448808102489,0.13311780686828037,3.858194363564449,2.0941111590578783,35,3911.79,3894.292152404785,99999.0,,,
++uesfm_abl,Ecole Superior De Guerre,2,0.42670768047814917,0.11109013616311524,3.738376401049213,1.9806437286978857,35,3914.42,3897.570469856262,99999.0,,,
++esfm,Eglise du dome,0,0.3242410065346163,0.0824797769992069,3.906129661328729,1.3512292775673158,85,25054.74,25028.64922428131,99999.0,0.03400419508287424,0.00968361713071315,2.1529226593308155
++esfm,Eglise du dome,1,0.3771623368290559,0.09432763361451474,3.692490941110161,1.297032357665518,85,25144.16,25116.01952123642,99999.0,0.03329023612659896,0.009627024817575195,1.843711309152907
++esfm,Eglise du dome,2,0.45623311880194967,0.11755570327842058,4.363674496823582,1.384257241090686,85,8694.98,8671.502047538757,99999.0,0.03489358171094041,0.010000209116698642,2.2467882207119256
++esfm_rc,Eglise du dome,0,0.5557206668193069,0.14201663162258846,4.387076179747346,1.5005097595973045,85,8784.41,8746.477850675583,99999.0,,,
++esfm_rc,Eglise du dome,1,0.5674000958153612,0.14488795138247493,4.186122825560066,1.4261565747278528,85,8806.29,8769.514946460724,99999.0,,,
++esfm_rc,Eglise du dome,2,0.5841919346497452,0.15295064836207942,4.177169486298827,1.4755743828038193,85,8796.97,8757.282470941544,99999.0,,,
++uesfm,Eglise du dome,0,0.8528700994829498,0.23193079249562099,5.944108200898114,2.9306630821877517,85,9803.94,9764.585114717484,99999.0,,,
++uesfm,Eglise du dome,1,0.6936386490465627,0.1932815436676001,6.091509620719386,2.9784748994578076,85,9788.95,9753.742706537247,99999.0,,,
++uesfm,Eglise du dome,2,45.4797063436542,6.439471230615876,8.408073658981259,4.497252326534489,85,9815.16,9781.022309541702,99999.0,,,
++uesfm_abl,Eglise du dome,0,0.6358533835189901,0.15875620748423402,4.568749457625584,1.500653586857206,85,8887.81,8839.908431768417,99999.0,,,
++uesfm_abl,Eglise du dome,1,0.4055403078104767,0.10680074618232935,4.157231512595432,1.4012026303579939,85,8835.0,8793.938161611557,99999.0,,,
++uesfm_abl,Eglise du dome,2,0.5564269559867648,0.13948261513340296,4.352396467434004,1.4848570308333404,85,8783.56,8744.307095527649,99999.0,,,
++esfm,Folke Filbyter,0,89.03873163576341,0.12895560134543632,17.986624161769267,7.423093846189277,40,1582.8,1573.360969305038,99999.0,75.51315528045146,0.13175161206780092,20.394291737162312
++esfm,Folke Filbyter,1,84.74740427947457,0.13132264058966686,17.98706393015447,7.385900243615879,40,1579.13,1572.079373121262,99999.0,70.97231562213899,0.12890045131878497,18.182171069573492
++esfm,Folke Filbyter,2,83.70817707245973,0.1261878953205589,21.85200129525089,6.324245998829542,40,1583.76,1576.207699537277,99999.0,67.83491986963047,0.12837305355978038,21.46453761850234
++esfm_rc,Folke Filbyter,0,89.37676108904198,0.12860403401017323,18.22099901069209,7.74644308432037,40,3082.48,3064.089830160141,99999.0,,,
++esfm_rc,Folke Filbyter,1,85.29802780650462,0.1312655372498663,18.31496063485457,7.58936290577374,40,3067.08,3052.338712692261,99999.0,,,
++esfm_rc,Folke Filbyter,2,83.0243809466643,0.12750491742097297,23.157774763962923,7.009627030137372,40,3069.05,3053.145490646362,99999.0,,,
++uesfm,Folke Filbyter,0,84.55526303793802,0.13091275274611416,29.190267154194732,11.160073225547443,40,3747.65,3727.990335702896,99999.0,,,
++uesfm,Folke Filbyter,1,84.87047537526948,0.13112291273275417,32.36503566471496,13.743964264262598,40,3743.53,3728.497133255005,99999.0,,,
++uesfm,Folke Filbyter,2,78.54375401873024,0.1316690349103016,23.11854509839131,10.94439865744236,40,3774.4,3007.95537352562,80000.0,,,
++uesfm_abl,Folke Filbyter,0,89.37676108904198,0.12860403401017323,18.22099901069209,7.74644308432037,40,3097.72,3074.491502285004,99999.0,,,
++uesfm_abl,Folke Filbyter,1,85.29802780650462,0.1312655372498663,18.31496063485457,7.58936290577374,40,3007.04,2989.949685811996,99999.0,,,
++uesfm_abl,Folke Filbyter,2,83.0243809466643,0.12750491742097297,23.157774763962923,7.009627030137372,40,3004.95,2987.753928661346,99999.0,,,
++esfm,Fort Channing Gate Singapore,0,0.10143646071220622,0.04531774433800129,1.7158776598376824,0.8165178663538488,27,3438.39,3428.288990497589,99999.0,0.01831129170730475,0.007384689295328118,0.2597632094033597
++esfm,Fort Channing Gate Singapore,1,0.08873951454303304,0.039001318240256114,1.783712095722291,0.8156020254191567,27,3430.88,3424.490573883057,99999.0,0.018321633248369797,0.007389903495856605,0.2597556139620049
++esfm,Fort Channing Gate Singapore,2,0.09031224636384508,0.03957289173255022,1.850800887931411,0.820658413919779,27,3433.07,3426.547735452652,99999.0,0.018353600365247733,0.0074018125678656016,0.25972738760938635
++esfm_rc,Fort Channing Gate Singapore,0,0.12919721972508352,0.05643946161236995,1.8665780418242388,0.9262975062395102,27,5450.14,5430.581563711166,99999.0,,,
++esfm_rc,Fort Channing Gate Singapore,1,0.0961175836225673,0.04352638473532752,1.7634485231989456,0.8832348541778796,27,5476.97,5456.584360361099,99999.0,,,
++esfm_rc,Fort Channing Gate Singapore,2,0.1269205407593247,0.05699742664239027,1.9815335382658048,0.9205728692828943,27,5423.49,5403.335438489914,99999.0,,,
++uesfm,Fort Channing Gate Singapore,0,0.10214634402234797,0.048923384893640356,3.7907267588398517,2.449846070018404,27,6175.88,6156.048281908035,99999.0,,,
++uesfm,Fort Channing Gate Singapore,1,0.14219758622796153,0.06357969294387673,3.5066994916782015,2.1555120267718575,27,6191.97,6176.711992740631,99999.0,,,
++uesfm,Fort Channing Gate Singapore,2,0.217326908828172,0.10047378918560818,3.8470573886147417,2.340537393844086,27,6145.9,6130.889567136765,99999.0,,,
++uesfm_abl,Fort Channing Gate Singapore,0,0.12919721972508352,0.05643946161236995,1.8665780418242388,0.9262975062395102,27,5443.13,5416.156220436096,99999.0,,,
++uesfm_abl,Fort Channing Gate Singapore,1,0.0961175836225673,0.04352638473532752,1.7634485231989456,0.8832348541778796,27,5447.14,5431.220237016678,99999.0,,,
++uesfm_abl,Fort Channing Gate Singapore,2,0.1269205407593247,0.05699742664239027,1.9815335382658048,0.9205728692828943,27,5422.08,5404.235495090485,99999.0,,,
++esfm,Golden Statue Somewhere In Hong Kong,0,0.23894829924734365,0.053817571637812096,1.7239352536484445,0.744089758310296,18,4096.74,4087.775670289993,99999.0,0.026250976813289254,0.003462321884642883,0.2822531708961522
++esfm,Golden Statue Somewhere In Hong Kong,1,0.2233990458438394,0.053061028269618456,1.8936878914364637,0.7746436618607616,18,4096.28,4088.994391202927,99999.0,0.026089105914074914,0.003445687328266814,0.2822884560119062
++esfm,Golden Statue Somewhere In Hong Kong,2,0.23397914670516012,0.051137498616263405,1.87070255923672,0.7763877145267694,18,4094.54,4087.674698591232,99999.0,0.026019507414979055,0.0034393367985468748,0.28218109003255504
++esfm_rc,Golden Statue Somewhere In Hong Kong,0,0.23746483548049216,0.052245486770656324,1.8202139930821122,0.7929927966477024,18,7039.12,7017.501612186432,99999.0,,,
++esfm_rc,Golden Statue Somewhere In Hong Kong,1,0.25835446587605726,0.061292299836508156,1.9316593984355397,0.8363119479886987,18,7014.04,6999.158929586411,99999.0,,,
++esfm_rc,Golden Statue Somewhere In Hong Kong,2,0.3060791978287713,0.07361327882523418,2.0787911799501324,0.8493789100421191,18,7119.11,7103.271198987961,99999.0,,,
++uesfm,Golden Statue Somewhere In Hong Kong,0,0.6244166607301816,0.09872331853471474,3.161519043143039,2.067073705499029,18,19646.03,19620.43561792374,99999.0,,,
++uesfm,Golden Statue Somewhere In Hong Kong,1,0.267468385274811,0.060691033227620865,2.8287683223467064,1.8282220692313509,18,7727.72,7710.336065292358,99999.0,,,
++uesfm,Golden Statue Somewhere In Hong Kong,2,0.6315484707179251,0.08671732527042503,3.075728070835161,1.9442441573584643,18,7724.17,7706.132658720016,99999.0,,,
++uesfm_abl,Golden Statue Somewhere In Hong Kong,0,0.23746483548049216,0.052245486770656324,1.8202139930821122,0.7929927966477024,18,7080.25,7058.794875144958,99999.0,,,
++uesfm_abl,Golden Statue Somewhere In Hong Kong,1,0.25835446587605726,0.061292299836508156,1.9316593984355397,0.8363119479886987,18,7063.84,7045.639441013336,99999.0,,,
++uesfm_abl,Golden Statue Somewhere In Hong Kong,2,0.3060791978287713,0.07361327882523418,2.0787911799501324,0.8493789100421191,18,7087.1,7070.110768318176,99999.0,,,
++esfm,Gustav Vasa,0,3.4881363194229826,0.2273181104934767,1.7369784534725496,0.9521606063342205,18,711.53,703.3091380596161,99999.0,0.9784891824482792,0.10778420441012125,0.5210309534376089
++esfm,Gustav Vasa,1,3.4307883498221035,0.21432466556133864,1.5556448596237022,0.9232184455756751,18,710.28,703.9906163215637,99999.0,0.9831979134139665,0.10803454074848035,0.5250933657575036
++esfm,Gustav Vasa,2,3.6089062562451386,0.2196578287395002,1.673803634770729,0.9429755567757121,18,707.79,701.5903799533844,99999.0,1.0495249332227277,0.12320327825606991,0.5686977966530726
++esfm_rc,Gustav Vasa,0,4.479928423821196,0.26813796016443026,1.9801034973174125,1.1078604015859557,18,1944.19,1905.260601043701,99999.0,,,
++esfm_rc,Gustav Vasa,1,3.6300123089239613,0.2270854488008436,1.7892038194948103,1.0620796169846443,18,1918.69,1900.605086088181,99999.0,,,
++esfm_rc,Gustav Vasa,2,4.256624192410625,0.25269996179819426,1.8539674990623372,1.0610032766895823,18,1923.76,1897.99094748497,99999.0,,,
++uesfm,Gustav Vasa,0,36.645536646380805,1.0070193571286277,4.980488290477233,3.197977230260218,18,2562.94,2419.297919034958,95000.0,,,
++uesfm,Gustav Vasa,1,36.4914104514929,1.03254185606701,5.377024289258223,3.4818421684593366,18,4566.81,4543.375423431396,99999.0,,,
++uesfm,Gustav Vasa,2,4.4238740734834625,0.284762443333143,3.8820183248893803,2.8663959682636566,18,4533.21,4516.61039352417,99999.0,,,
++uesfm_abl,Gustav Vasa,0,4.479928423821196,0.26813796016443026,1.9801034973174125,1.1078604015859557,18,1942.38,1920.377394676208,99999.0,,,
++uesfm_abl,Gustav Vasa,1,3.6300123089239613,0.2270854488008436,1.7892038194948103,1.0620796169846443,18,1939.3,1923.45086979866,99999.0,,,
++uesfm_abl,Gustav Vasa,2,4.256624192410625,0.25269996179819426,1.8539674990623372,1.0610032766895823,18,1975.57,1961.386824131012,99999.0,,,
++esfm,GustavIIAdolf,0,46.92758389264633,8.570499881283022,11.684017169857798,5.131125459552879,57,1047.82,1038.018162250519,99999.0,24.71817789017971,5.446736853565307,16.261776511982404
++esfm,GustavIIAdolf,1,47.21999673071185,8.515052769156952,11.695065125063977,5.091617086264161,57,1041.41,1034.5799472332,99999.0,24.783445519335448,5.485252890255178,16.122725952233722
++esfm,GustavIIAdolf,2,47.2312223396789,8.523261295519099,11.787583212607423,5.2715388868550574,57,1043.28,1036.478415489197,99999.0,24.71364130473506,5.447948021143418,16.671847441164118
++esfm_rc,GustavIIAdolf,0,47.35567772792213,8.62293127729445,11.946761161586606,5.4045343866458575,57,2284.75,2245.40932559967,99999.0,,,
++esfm_rc,GustavIIAdolf,1,47.65353986419624,8.589683791795656,11.844361575387735,5.303722447527598,57,2258.26,2237.952227830887,99999.0,,,
++esfm_rc,GustavIIAdolf,2,47.396282103638164,8.509630444500928,11.963746746326924,5.434993473177883,57,2252.6,2235.428824663162,99999.0,,,
++uesfm,GustavIIAdolf,0,85.51179144359693,12.792264407383987,17.018024925092185,8.196266635685827,57,2973.35,2956.916826248169,99999.0,,,
++uesfm,GustavIIAdolf,1,88.9494736685426,13.260723460498589,17.699982689478404,8.888010081099381,57,2995.3,2976.055320978165,99999.0,,,
++uesfm,GustavIIAdolf,2,90.12096202318821,13.335169188185864,16.941538974035925,7.772181288890227,57,2974.4,2960.110015630722,99999.0,,,
++uesfm_abl,GustavIIAdolf,0,47.35567772792213,8.62293127729445,11.946761161586606,5.4045343866458575,57,2296.29,2276.984587669373,99999.0,,,
++uesfm_abl,GustavIIAdolf,1,47.65353986419624,8.589683791795656,11.844361575387735,5.303722447527598,57,2317.06,2297.372611522675,99999.0,,,
++uesfm_abl,GustavIIAdolf,2,47.396282103638164,8.509630444500928,11.963746746326924,5.434993473177883,57,2251.14,2236.10190486908,99999.0,,,
++esfm,Jonas Ahlstromer,0,64.46662959289324,11.985605264945464,12.602584968179986,2.9168445332917745,40,614.88,607.3498430252075,99999.0,56.233065938789125,10.435407172303073,11.885955230403729
++esfm,Jonas Ahlstromer,1,38.42510202464649,8.741159740710629,10.25268858530355,3.7024742418884333,40,618.73,611.5567183494568,99999.0,0.03712241399969036,0.010988262348231177,0.22755727926611335
++esfm,Jonas Ahlstromer,2,43.32891670826628,8.741053567623224,10.553031866494822,3.031717304343782,40,612.74,606.6802380084991,99999.0,42.27144254755807,8.48578501886873,10.391663840228917
++esfm_rc,Jonas Ahlstromer,0,63.4390388558475,11.542340942042635,12.704406945971337,3.2390607659681296,40,1732.58,1605.343009233475,95000.0,,,
++esfm_rc,Jonas Ahlstromer,1,38.177570703294535,8.672488375453444,10.465133389728726,3.83336868366082,40,1706.01,1687.714308977127,99999.0,,,
++esfm_rc,Jonas Ahlstromer,2,43.62862719300082,8.8014510569871,10.815011628378553,3.3581265722454883,40,1699.67,1682.403540372849,99999.0,,,
++uesfm,Jonas Ahlstromer,0,45.264596178655594,9.403212752141883,11.336617283269852,4.728596140494824,40,2421.11,2401.221394062042,99999.0,,,
++uesfm,Jonas Ahlstromer,1,51.490398556789884,10.860548616493958,12.821407884052217,5.012487046810724,40,2428.25,2415.139223337173,99999.0,,,
++uesfm,Jonas Ahlstromer,2,42.99784237868634,10.470910789899067,9.274184391894982,4.4612757046777345,40,2394.6,2381.585824012756,99999.0,,,
++uesfm_abl,Jonas Ahlstromer,0,63.4390388558475,11.542340942042635,12.704406945971337,3.2390607659681296,40,1748.31,1643.204682588577,95000.0,,,
++uesfm_abl,Jonas Ahlstromer,1,38.177570703294535,8.672488375453444,10.465133389728726,3.83336868366082,40,1731.4,1717.937728643417,99999.0,,,
++uesfm_abl,Jonas Ahlstromer,2,43.62862719300082,8.8014510569871,10.815011628378553,3.3581265722454883,40,1734.1,1719.565098524094,99999.0,,,
++esfm,Kings College University Of Toronto,0,14.400461796497643,3.2653370282498866,4.064353916685246,1.6753245173381437,77,1074.94,1066.306197404861,99999.0,9.635756937410923,2.001374412854319,2.4237261655963693
++esfm,Kings College University Of Toronto,1,11.956699445712928,2.705452710546247,3.9727144140636783,1.5797087908756111,77,1071.64,1064.129351377487,99999.0,7.641765083108614,1.6563433329963133,2.511022238246807
++esfm,Kings College University Of Toronto,2,0.7191591237413282,0.16178494376153177,2.656807607821862,1.1949677972638288,77,1090.36,1082.987608671188,99999.0,0.08065822304086778,0.013824042665226435,0.3683414375557499
++esfm_rc,Kings College University Of Toronto,0,14.749850754568351,3.390834150797208,4.273460375035421,1.881809737201061,77,2235.19,2086.953983068466,95000.0,,,
++esfm_rc,Kings College University Of Toronto,1,12.24434343895146,2.727340976741191,4.101496648756234,1.7269690990930942,77,2196.25,2068.98152923584,95000.0,,,
++esfm_rc,Kings College University Of Toronto,2,0.9409038453179062,0.20480434027922273,2.819943864062075,1.3255016011048806,77,2197.64,2180.024880647659,99999.0,,,
++uesfm,Kings College University Of Toronto,0,29.200261744017546,2.0893308856114428,7.037455848998898,4.8227457854711675,77,2856.72,2838.923613071442,99999.0,,,
++uesfm,Kings College University Of Toronto,1,7.107988087620218,1.3235014088907995,5.965363878755208,4.179664786063016,77,2852.71,2839.42994761467,99999.0,,,
++uesfm,Kings College University Of Toronto,2,6.543121223644844,1.5041055210367995,6.0501544660211914,4.127143945227024,77,2863.77,2850.061238765717,99999.0,,,
++uesfm_abl,Kings College University Of Toronto,0,14.749850754568351,3.390834150797208,4.273460375035421,1.881809737201061,77,2198.51,2068.722495794296,95000.0,,,
++uesfm_abl,Kings College University Of Toronto,1,12.24434343895146,2.727340976741191,4.101496648756234,1.7269690990930942,77,2197.87,2073.158569335938,95000.0,,,
++uesfm_abl,Kings College University Of Toronto,2,0.9409038453179062,0.20480434027922273,2.819943864062075,1.3255016011048806,77,2195.63,2180.702211618423,99999.0,,,
++esfm,Lund University Sphinx,0,13.661465370705821,3.5591094267381553,6.242667318159146,2.715620555070548,70,2718.96,2709.5872194767,99999.0,8.460013856708239,2.11467464964552,1.4565174740087705
++esfm,Lund University Sphinx,1,12.524302160109885,3.2130942199657664,6.1182420215295075,2.7468468987195953,70,2722.86,2715.36418390274,99999.0,8.12353912383344,2.074906927104935,1.325091071719341
++esfm,Lund University Sphinx,2,7.98961938632192,2.1710959765491817,5.807289242508196,2.893370267054225,70,2713.88,2706.478076219559,99999.0,4.449918670952723,1.2322626694001688,0.8213585400994716
++esfm_rc,Lund University Sphinx,0,16.25886053825766,4.102962816505069,6.869889153124891,3.100467608594146,70,3942.66,3901.573514938354,99999.0,,,
++esfm_rc,Lund University Sphinx,1,13.156269680075315,3.4132660053719683,6.667310697819336,3.1453748872389977,70,3914.43,3893.653036355972,99999.0,,,
++esfm_rc,Lund University Sphinx,2,9.604992163703725,2.6373578340271524,6.742167958027671,3.346047584137487,70,3916.03,3895.086451530457,99999.0,,,
++uesfm,Lund University Sphinx,0,42.95863056778373,8.719684065980621,14.999945982069319,8.361965506089375,70,4705.61,4681.787243127823,99999.0,,,
++uesfm,Lund University Sphinx,1,14.202013501375653,3.5150197050401064,10.824261355110268,6.665393527268384,70,4616.97,4592.001487731934,99999.0,,,
++uesfm,Lund University Sphinx,2,7.993187283055365,2.2173389310010068,8.827116984475035,5.573703948682402,70,4605.39,4587.484831571579,99999.0,,,
++uesfm_abl,Lund University Sphinx,0,16.25886053825766,4.102962816505069,6.869889153124891,3.100467608594146,70,3907.32,3882.466088533401,99999.0,,,
++uesfm_abl,Lund University Sphinx,1,13.156269680075315,3.4132660053719683,6.667310697819336,3.1453748872389977,70,3915.29,3898.995554447174,99999.0,,,
++uesfm_abl,Lund University Sphinx,2,9.604992163703725,2.6373578340271524,6.742167958027671,3.346047584137487,70,3912.47,3894.494997739792,99999.0,,,
++esfm,Nijo Castle Gate,0,0.6521524412276166,0.13305866403975986,7.404953325705311,3.0629804954987674,19,1097.29,1084.301982402802,99999.0,0.08036972152423151,0.012796484603062593,0.7615804699553756
++esfm,Nijo Castle Gate,1,0.8564043952718835,0.16950374865879153,7.358076475544799,3.1235223975185864,19,1094.89,1084.640900611877,99999.0,0.08192404770421978,0.013101136558935796,0.760865657995869
++esfm,Nijo Castle Gate,2,1.0089926380330574,0.19095470213426863,7.481049991287358,3.1671779658403687,19,1093.35,1083.446812152863,99999.0,0.08102331686497327,0.01291191910447522,0.7612254771474661
++esfm_rc,Nijo Castle Gate,0,1.1608620390074227,0.22835706972308775,8.292447452530368,3.6544156383512774,19,2653.78,2485.451326608658,95000.0,,,
++esfm_rc,Nijo Castle Gate,1,1.4421159803060983,0.2748016105660959,8.250799662340748,3.7148544612954977,19,2619.96,2603.008899211884,99999.0,,,
++esfm_rc,Nijo Castle Gate,2,1.3714350481742656,0.2608522893427154,8.124524832882802,3.5384916107814472,19,2614.73,2598.148197889328,99999.0,,,
++uesfm,Nijo Castle Gate,0,8.641303631787597,2.791258638759238,20.96738948203601,10.297911838853688,19,3241.33,3220.324777603149,99999.0,,,
++uesfm,Nijo Castle Gate,1,1.3055705335697472,0.2681929407818948,11.680266119036336,6.962610820698007,19,3234.66,3216.868148088455,99999.0,,,
++uesfm,Nijo Castle Gate,2,1.8310744216680852,0.3558493818160386,12.75712078857825,7.906743075262264,19,3227.08,3213.229565620422,99999.0,,,
++uesfm_abl,Nijo Castle Gate,0,1.1608620390074227,0.22835706972308775,8.292447452530368,3.6544156383512774,19,2638.45,2489.672921419144,95000.0,,,
++uesfm_abl,Nijo Castle Gate,1,1.4421159803060983,0.2748016105660959,8.250799662340748,3.7148544612954977,19,2632.14,2618.752884149551,99999.0,,,
++uesfm_abl,Nijo Castle Gate,2,1.3714350481742656,0.2608522893427154,8.124524832882802,3.5384916107814472,19,2622.82,2607.735609292984,99999.0,,,
++esfm_rc,Pantheon Paris,0,0.21324226532150278,0.0556382184158685,4.83107981482966,3.002373021388077,179,4653.95,4598.05153131485,99999.0,,,
++esfm_rc,Pantheon Paris,1,0.212491875340622,0.05529075860063325,4.8846908181959074,2.9588861166627645,179,4628.46,4593.19188284874,99999.0,,,
++esfm_rc,Pantheon Paris,2,0.3340659445994911,0.05750962417018217,4.964330493411554,2.9691476134237185,179,4639.41,4604.334435939789,99999.0,,,
++uesfm,Pantheon Paris,0,0.5453004964472504,0.06999784414241637,7.790870638815493,5.057509442480535,179,5594.6,5562.967379331589,99999.0,,,
++uesfm,Pantheon Paris,1,0.472097972346089,0.062163803587728925,7.194706475909503,4.639524309087925,179,5536.74,5501.721322774887,99999.0,,,
++uesfm,Pantheon Paris,2,44.569134625905356,6.741655820531256,16.92751887234327,6.718188212415848,179,5541.46,5510.693382978439,99999.0,,,
++uesfm_abl,Pantheon Paris,0,0.21324226532150278,0.0556382184158685,4.83107981482966,3.002373021388077,179,4624.92,4588.305618524551,99999.0,,,
++uesfm_abl,Pantheon Paris,1,0.212491875340622,0.05529075860063325,4.8846908181959074,2.9588861166627645,179,4612.54,4580.788692235947,99999.0,,,
++uesfm_abl,Pantheon Paris,2,0.3340659445994911,0.05750962417018217,4.964330493411554,2.9691476134237185,179,4614.44,4582.374843358994,99999.0,,,
++esfm,Park Gate Clermont Ferrand,0,19.399216821978534,9.438471236639753,7.521969683469249,3.892136696028138,34,1597.96,1588.07075381279,99999.0,7.005947814792836,2.8784023682563795,14.489115644047233
++esfm,Park Gate Clermont Ferrand,1,19.325402297843297,9.417998214523331,7.465007292717891,3.849088919564891,34,1596.87,1589.703137874603,99999.0,7.04585897473849,2.8955291629401523,14.422988510386801
++esfm,Park Gate Clermont Ferrand,2,25.440611579781365,11.720641869372841,8.15032188787613,4.0871739024940315,34,1595.99,1588.841923236847,99999.0,16.730405105853063,7.240660871176633,13.929099740909901
++esfm_rc,Park Gate Clermont Ferrand,0,19.41015735759728,9.44545450483485,7.5655698947009755,3.9376102285883072,34,2969.64,2931.905954837799,99999.0,,,
++esfm_rc,Park Gate Clermont Ferrand,1,19.33326285032967,9.425195186043549,7.520162585346567,3.890911800001084,34,2946.52,2928.374010801315,99999.0,,,
++esfm_rc,Park Gate Clermont Ferrand,2,25.18721114681958,11.659048612833406,8.156533346326963,4.159109875326399,34,2941.72,2923.031396150589,99999.0,,,
++uesfm,Park Gate Clermont Ferrand,0,26.45121972215746,12.46688671568752,9.374365736774523,5.572822711026232,34,3659.9,3637.290126085281,99999.0,,,
++uesfm,Park Gate Clermont Ferrand,1,25.126789970267566,11.629146345704141,8.445284973863814,4.644358918969844,34,3647.69,3633.016668319702,99999.0,,,
++uesfm,Park Gate Clermont Ferrand,2,19.34531670045473,9.424313535878454,8.137652552146374,4.753910741641504,34,3642.92,3629.229806900024,99999.0,,,
++uesfm_abl,Park Gate Clermont Ferrand,0,19.41015735759728,9.44545450483485,7.5655698947009755,3.9376102285883072,34,3001.15,2982.093791484833,99999.0,,,
++uesfm_abl,Park Gate Clermont Ferrand,1,19.33326285032967,9.425195186043549,7.520162585346567,3.890911800001084,34,2964.98,2949.696373462677,99999.0,,,
++uesfm_abl,Park Gate Clermont Ferrand,2,25.18721114681958,11.659048612833406,8.156533346326963,4.159109875326399,34,2965.47,2949.038843154907,99999.0,,,
++esfm,Plaza De Armas Santiago,0,0.7341628728579594,0.32156050159900956,6.000154308231358,2.679971945741675,240,6375.02,6333.455562829971,99999.0,0.13274758032014072,0.05309931531294862,2.606822533335194
++esfm,Plaza De Armas Santiago,1,0.5176165701053533,0.22744923951256435,5.661576462867733,2.533952088248649,240,6373.68,6336.030319213867,99999.0,0.13819907741152881,0.05533560554064368,2.759268616966368
++esfm,Plaza De Armas Santiago,2,1.0740077415291467,0.46304622323619105,6.365716833190761,2.733195456670397,240,6369.35,6331.436369419098,99999.0,0.13512690139003744,0.05415457227659273,3.4062212958788285
++esfm_rc,Plaza De Armas Santiago,0,1.0741847479970412,0.4714298140713374,6.407094111087262,2.8609073397853417,240,5662.18,5611.049166440964,99999.0,,,
++esfm_rc,Plaza De Armas Santiago,1,0.8428560730717238,0.3711994064496868,5.976446140682491,2.748210015625399,240,5659.43,5609.024231433868,99999.0,,,
++esfm_rc,Plaza De Armas Santiago,2,1.4478823177517481,0.6177865944942146,6.59706564230341,2.915296192751213,240,5714.62,5664.147575616837,99999.0,,,
++uesfm,Plaza De Armas Santiago,0,6.703928231459959,2.9117331629179497,11.237000025698872,5.853835350910829,240,6616.75,6560.115643978119,99999.0,,,
++uesfm,Plaza De Armas Santiago,1,6.276886659110257,2.7457580388752243,10.702703105569906,5.585481192919806,240,6612.47,6560.248650550842,99999.0,,,
++uesfm,Plaza De Armas Santiago,2,6.731807209592312,2.8977064717650864,11.656685600371235,6.070719216176278,240,6621.69,6564.684820175171,99999.0,,,
++uesfm_abl,Plaza De Armas Santiago,0,1.0741847479970412,0.4714298140713374,6.407094111087262,2.8609073397853417,240,5627.44,5572.398021221161,99999.0,,,
++uesfm_abl,Plaza De Armas Santiago,1,0.8428560730717238,0.3711994064496868,5.976446140682491,2.748210015625399,240,5575.14,5524.294562578201,99999.0,,,
++uesfm_abl,Plaza De Armas Santiago,2,1.4478823177517481,0.6177865944942146,6.59706564230341,2.915296192751213,240,5583.46,5533.742848873138,99999.0,,,
++esfm,Porta San Donato Bologna,0,0.5746001989174886,0.10608894230214515,5.691906895070585,2.4183460353027986,141,3751.75,3734.281898736954,99999.0,0.10991271062835306,0.05551366141500488,4.218662994987151
++esfm,Porta San Donato Bologna,1,0.8978003419853328,0.14474404936935045,5.682379581165418,2.563141107750346,141,3758.51,3742.688964605331,99999.0,0.1128666236829697,0.05602911579225738,2.211912679347602
++esfm,Porta San Donato Bologna,2,0.30440272958620346,0.07026546193502164,5.435311727517131,2.1626312981864086,141,3752.23,3738.056202173233,99999.0,0.11448509746230341,0.056754942188861325,3.2403286864511167
++esfm_rc,Porta San Donato Bologna,0,0.4080484041025974,0.07831590997213768,5.6144822077580585,2.3712185771771375,141,4331.14,4297.81409239769,99999.0,,,
++esfm_rc,Porta San Donato Bologna,1,0.4654680308021159,0.09236185451721962,5.643652792745585,2.4272418315858557,141,4309.37,4283.085269212723,99999.0,,,
++esfm_rc,Porta San Donato Bologna,2,0.9559717082151379,0.1643078286741575,6.078014738464724,2.61821191294824,141,4313.36,4286.725072145462,99999.0,,,
++uesfm,Porta San Donato Bologna,0,4.020291451689462,1.1214786148434206,10.933993245401652,5.19237663800495,141,5137.62,5110.554834842682,99999.0,,,
++uesfm,Porta San Donato Bologna,1,0.7088473947757895,0.1505922603123855,8.617060379525427,4.684345536783661,141,5152.53,5128.584963560104,99999.0,,,
++uesfm,Porta San Donato Bologna,2,0.7136264396886388,0.1325505085909014,7.953155791163967,4.201659432858777,141,5212.46,5175.418401956558,99999.0,,,
++uesfm_abl,Porta San Donato Bologna,0,0.4080484041025974,0.07831590997213768,5.6144822077580585,2.3712185771771375,141,4363.98,4335.123957633972,99999.0,,,
++uesfm_abl,Porta San Donato Bologna,1,0.4654680308021159,0.09236185451721962,5.643652792745585,2.4272418315858557,141,4338.27,4313.383106946945,99999.0,,,
++uesfm_abl,Porta San Donato Bologna,2,0.9559717082151379,0.1643078286741575,6.078014738464724,2.61821191294824,141,4342.05,4317.41094493866,99999.0,,,
++esfm,Round Church Cambridge,0,2.1173198662081005,0.9129854085205968,5.854320175654564,1.9683268910531708,92,9788.65,9759.719386577606,99999.0,0.9342212598130449,0.5515227006059547,4.054067325360582
++esfm,Round Church Cambridge,1,2.1415279636494953,0.9267927094737857,6.33905951937897,2.0439997148640834,92,9764.61,9741.356333732605,99999.0,1.115336663170846,0.5722632781970113,5.541826341161163
++esfm,Round Church Cambridge,2,2.1507951248804917,0.9401429799233081,6.115230559713554,2.0517209910251797,92,9784.55,9763.525354623795,99999.0,0.9104797992475266,0.5364933276394585,5.035679124124719
++esfm_rc,Round Church Cambridge,0,2.2826201708417098,0.9657710484746329,6.16310348787557,2.185542741510192,92,10880.38,10823.98271656036,99999.0,,,
++esfm_rc,Round Church Cambridge,1,2.2206190556774503,0.9437039356840937,6.581626868211357,2.1839542860976264,92,10884.45,10840.10469603539,99999.0,,,
++esfm_rc,Round Church Cambridge,2,2.3743814339345155,0.9912916631176563,6.734437930520745,2.258930633937234,92,10886.78,10842.68440675735,99999.0,,,
++uesfm,Round Church Cambridge,0,1.1275240742139254,0.32924483901066126,7.8063197416312855,3.992682305020409,92,12183.97,12142.68090772629,99999.0,,,
++uesfm,Round Church Cambridge,1,2.530477427180247,0.972585197332408,8.18733294883617,4.015376648566849,92,12114.44,12080.06272149086,99999.0,,,
++uesfm,Round Church Cambridge,2,2.6330412756122525,1.0135214760737334,8.856154239831417,4.349376095653312,92,12091.11,12053.28948831558,99999.0,,,
++uesfm_abl,Round Church Cambridge,0,2.2826201708417098,0.9657710484746329,6.16310348787557,2.185542741510192,92,10938.68,10901.49413061142,99999.0,,,
++uesfm_abl,Round Church Cambridge,1,2.2206190556774503,0.9437039356840937,6.581626868211357,2.1839542860976264,92,10884.14,10849.70382070541,99999.0,,,
++uesfm_abl,Round Church Cambridge,2,2.3743814339345155,0.9912916631176563,6.734437930520745,2.258930633937234,92,10887.63,10854.02958774567,99999.0,,,
++esfm,Skansen Kronan Gothenburg,0,0.3380139741078169,0.1158931606908392,3.0478862369293624,1.5340801117570733,131,17163.94,17140.15425729752,99999.0,0.025387020977857267,0.008390321550547433,0.6788480725266395
++esfm,Skansen Kronan Gothenburg,1,0.21516831826827018,0.07458182746108981,2.786705021535272,1.4619994473169777,131,17129.62,17109.18022465706,99999.0,0.025231970575166494,0.008360587055798304,0.6833298341122367
++esfm,Skansen Kronan Gothenburg,2,0.3508356515622903,0.11667319277987132,2.8836839507901773,1.5216584205042176,131,17167.47,17142.66050887108,99999.0,0.025087644890282812,0.008293159108503236,0.6831238442687219
++esfm_rc,Skansen Kronan Gothenburg,0,0.2958520465895253,0.10029016058373366,3.0067714626699646,1.6085514748163654,131,6130.39,6091.164911746979,99999.0,,,
++esfm_rc,Skansen Kronan Gothenburg,1,0.2916975925764745,0.09931739923314177,3.007556973630657,1.5796531296605096,131,6127.58,6094.12020111084,99999.0,,,
++esfm_rc,Skansen Kronan Gothenburg,2,0.34947792833332825,0.11898136948566021,3.0445611018681613,1.6310446575058488,131,6125.12,6091.680647850037,99999.0,,,
++uesfm,Skansen Kronan Gothenburg,0,0.38108224181004346,0.13301438128895593,4.623321785995157,3.2918868818818052,131,16316.64,16276.98657560349,99999.0,,,
++uesfm,Skansen Kronan Gothenburg,1,0.37345373662160575,0.12470260098166054,4.764112581596325,3.4119305236076505,131,7073.79,7040.064019680023,99999.0,,,
++uesfm,Skansen Kronan Gothenburg,2,0.2703573639284502,0.0962556036770187,4.872246482920416,3.4297600302558404,131,7079.8,7048.02818608284,99999.0,,,
++uesfm_abl,Skansen Kronan Gothenburg,0,0.3533584640939546,0.1208300122579018,3.045592631019205,1.6601247980904001,131,6178.83,6138.532582998276,99999.0,,,
++uesfm_abl,Skansen Kronan Gothenburg,1,0.2837532261305787,0.09579775777282211,2.9567373831366788,1.5617381428579997,131,6166.53,6133.822659730911,99999.0,,,
++uesfm_abl,Skansen Kronan Gothenburg,2,0.3162961160751722,0.10814799705691167,3.0168436405283536,1.6130503814274837,131,6162.94,6132.723542928696,99999.0,,,
++esfm,Smolny Cathedral St Petersburg,0,21.81278753416543,2.110667559831311,11.2315357584775,6.379259553776089,131,10364.67,10320.51852750778,99999.0,21.423862086052477,2.1555190029458697,10.301353476554675
++esfm,Smolny Cathedral St Petersburg,1,21.799828748358873,2.111313741354268,11.225141339870355,6.36605939788532,131,10360.52,10319.01374697685,99999.0,21.406459319996742,2.160222344201535,10.317138393648838
++esfm,Smolny Cathedral St Petersburg,2,21.71090600320656,2.110307413725059,11.259799763291888,6.4136596726420905,131,10363.17,10320.08415436745,99999.0,21.419154026165344,2.157488960966909,10.311901836556512
++esfm_rc,Smolny Cathedral St Petersburg,0,21.82091176814388,2.109838123236979,11.34747421311657,6.5257275032038065,131,9392.23,9330.660197257996,99999.0,,,
++esfm_rc,Smolny Cathedral St Petersburg,1,21.8691455338683,2.106636325604001,11.385698170334294,6.528914486700552,131,9365.77,9312.483961582184,99999.0,,,
++esfm_rc,Smolny Cathedral St Petersburg,2,21.852976622993765,2.104890865572913,11.527275577228112,6.60848088276277,131,9396.26,9340.038203716278,99999.0,,,
++uesfm,Smolny Cathedral St Petersburg,0,22.2854186118672,2.094744890313659,12.901706098015099,8.330992137880555,131,10444.11,10386.77505373955,99999.0,,,
++uesfm,Smolny Cathedral St Petersburg,1,22.12557249893661,2.0983352220394447,12.972911403502199,8.373490866652297,131,10440.89,10388.390884161,99999.0,,,
++uesfm,Smolny Cathedral St Petersburg,2,22.3108596899731,2.0940347709664024,12.978138438291575,8.138999834900323,131,10450.27,10395.72995519638,99999.0,,,
++uesfm_abl,Smolny Cathedral St Petersburg,0,21.82091176814388,2.109838123236979,11.34747421311657,6.5257275032038065,131,9323.21,9264.469937562943,99999.0,,,
++uesfm_abl,Smolny Cathedral St Petersburg,1,21.8691455338683,2.106636325604001,11.385698170334294,6.528914486700552,131,9318.22,9263.867384433746,99999.0,,,
++uesfm_abl,Smolny Cathedral St Petersburg,2,21.852976622993765,2.104890865572913,11.527275577228112,6.60848088276277,131,9319.64,9267.756978750229,99999.0,,,
++esfm,Some Cathedral In Barcelona,0,0.6775058368464424,0.24064262464083658,6.20165095815897,3.7253636674062887,177,5002.81,4985.274734258652,99999.0,0.03317243582351888,0.013609437166710361,0.9277887218183031
++esfm,Some Cathedral In Barcelona,1,0.6722147495861831,0.2423973792150467,6.350786792749083,3.720536755931007,177,5009.15,4991.552013158798,99999.0,0.033053432001706344,0.013569847698698592,1.4229226408233036
++esfm,Some Cathedral In Barcelona,2,0.6046048207068493,0.21699252142655406,6.261075100347952,3.7258171467923633,177,4993.96,4975.845166683197,99999.0,0.03281389435149025,0.013509575767387335,0.9486644088357065
++esfm_rc,Some Cathedral In Barcelona,0,0.8749233797042759,0.3145000897825549,6.881244292669395,4.213158668265296,177,4973.63,4937.959746360779,99999.0,,,
++esfm_rc,Some Cathedral In Barcelona,1,0.7676684854847832,0.2755901489197553,6.761455876131694,4.075310939831256,177,4959.11,4927.540645837784,99999.0,,,
++esfm_rc,Some Cathedral In Barcelona,2,0.71858178037652,0.25641289780654003,6.744656202748352,4.080713738243981,177,4952.16,4923.896674156189,99999.0,,,
++uesfm,Some Cathedral In Barcelona,0,3.68833616720384,1.476348113803727,14.728491750889885,9.144995894634686,177,5865.45,5836.537386417389,99999.0,,,
++uesfm,Some Cathedral In Barcelona,1,1.3699265043959112,0.4939901267450982,10.514669959569153,7.150937287439337,177,5931.0,5904.003969430923,99999.0,,,
++uesfm,Some Cathedral In Barcelona,2,39.16330988748253,12.556902649670175,22.516615012125715,12.905943206593893,177,5872.86,5843.584012508392,99999.0,,,
++uesfm_abl,Some Cathedral In Barcelona,0,0.8749233797042759,0.3145000897825549,6.881244292669395,4.213158668265296,177,4960.1,4929.568880319595,99999.0,,,
++uesfm_abl,Some Cathedral In Barcelona,1,0.7676684854847832,0.2755901489197553,6.761455876131694,4.075310939831256,177,4962.99,4935.773297548294,99999.0,,,
++uesfm_abl,Some Cathedral In Barcelona,2,0.71858178037652,0.25641289780654003,6.744656202748352,4.080713738243981,177,4966.51,4937.314158678055,99999.0,,,
++esfm,Sri Mariamman Singapore,0,1.4920493985513708,0.4412389781578158,10.406406306167431,6.369585247957454,222,5987.98,5970.066485881805,99999.0,0.08136203430698868,0.024448880002067186,0.9292989040444514
++esfm,Sri Mariamman Singapore,1,1.1345564396944785,0.3608906720626337,9.703927906328842,5.885963785879842,222,5993.62,5975.365797519684,99999.0,0.07981309127265196,0.023957917778313187,0.9234854997565143
++esfm,Sri Mariamman Singapore,2,1.0293707921546553,0.3426709250774071,10.15220066698994,6.067537729054635,222,5975.02,5958.786058187485,99999.0,0.07902485326574453,0.023758403592938876,0.9232224972615053
++esfm_rc,Sri Mariamman Singapore,0,2.074526913277199,0.5981095947062628,11.463860875975854,7.123711976607464,222,5510.81,5473.720585107803,99999.0,,,
++esfm_rc,Sri Mariamman Singapore,1,1.7658826525662057,0.5591108569640473,10.926621607114187,6.741925171900791,222,5496.08,5465.121775150299,99999.0,,,
++esfm_rc,Sri Mariamman Singapore,2,1.4009372521572498,0.44931018686538454,11.01432197073524,6.709336637494123,222,5499.38,5468.996921539307,99999.0,,,
++uesfm,Sri Mariamman Singapore,0,2.4686335799916477,0.7739007732343344,14.605464813669228,9.97923516189929,222,14748.08,14705.00972270966,99999.0,,,
++uesfm,Sri Mariamman Singapore,1,2.078550795026382,0.6787961741334789,13.788432610497317,9.377924514787443,222,14759.28,14715.18007564545,99999.0,,,
++uesfm,Sri Mariamman Singapore,2,1.7907753932427595,0.6052602851109582,12.444814769922951,8.402147028063522,222,14729.1,14697.75046205521,99999.0,,,
++uesfm_abl,Sri Mariamman Singapore,0,2.074526913277199,0.5981095947062628,11.463860875975854,7.123711976607464,222,5519.36,5485.930455446243,99999.0,,,
++uesfm_abl,Sri Mariamman Singapore,1,1.7658826525662057,0.5591108569640473,10.926621607114187,6.741925171900791,222,5513.22,5479.612489461899,99999.0,,,
++uesfm_abl,Sri Mariamman Singapore,2,1.4009372521572498,0.44931018686538454,11.01432197073524,6.709336637494123,222,5505.44,5478.000994443893,99999.0,,,
++esfm,Sri Thendayuthapani Singapore,0,0.7190396606626565,0.17922550285921848,7.12365374281766,2.231392236341273,98,13362.09,13320.09781932831,99999.0,0.19739404993046333,0.057925157042210476,6.111012693196153
++esfm,Sri Thendayuthapani Singapore,1,0.5350953449295449,0.1261925950852554,6.4467364012557615,2.04778550672059,98,13347.85,13309.03068971634,99999.0,0.19200876982588352,0.055981285294245835,8.004904549888494
++esfm,Sri Thendayuthapani Singapore,2,0.5262093764563488,0.14374074415280041,7.476614790858722,2.2575164196527426,98,13304.56,13262.61834168434,99999.0,,,
++esfm_rc,Sri Thendayuthapani Singapore,0,0.7371639218203122,0.18650202548448253,7.499935996618937,2.3110604008688562,98,12532.48,12472.05219936371,99999.0,,,
++esfm_rc,Sri Thendayuthapani Singapore,1,0.7298957622388771,0.15775896959686408,6.855850988025784,2.2114171261230844,98,12550.85,12494.93173718452,99999.0,,,
++esfm_rc,Sri Thendayuthapani Singapore,2,0.6332570037033005,0.1670204347365887,7.4898727605029665,2.3242398420277945,98,12530.0,12474.00808596611,99999.0,,,
++uesfm,Sri Thendayuthapani Singapore,0,0.6231213322199768,0.21686608350881142,10.153314817239993,4.077818514354337,98,13885.62,13829.64960861206,99999.0,,,
++uesfm,Sri Thendayuthapani Singapore,1,0.8606139427206572,0.2295356611884548,9.279712988615236,3.705865828185615,98,13902.5,13848.30702185631,99999.0,,,
++uesfm,Sri Thendayuthapani Singapore,2,0.8685610884058521,0.229402954616869,9.929333644165053,3.7851203401221336,98,13914.59,13860.98727893829,99999.0,,,
++uesfm_abl,Sri Thendayuthapani Singapore,0,0.6424755506443429,0.17578891022993118,7.22996967140421,2.317747114870694,98,12521.45,12465.29735732079,99999.0,,,
++uesfm_abl,Sri Thendayuthapani Singapore,1,0.8013752523965639,0.1735753109027402,7.09269635292555,2.2748608542934687,98,12590.42,12531.87466025352,99999.0,,,
++uesfm_abl,Sri Thendayuthapani Singapore,2,0.6803901454070272,0.16819990121758815,7.760963408769563,2.373228054708062,98,12572.88,12519.08800196648,99999.0,,,
++esfm,Sri Veeramakaliamman Singapore,0,2.333869921178247,0.542847254574005,11.965270419451508,5.472877662905037,157,12392.18,12365.15180635452,99999.0,,,
++esfm,Sri Veeramakaliamman Singapore,1,2.4804161216418734,0.5731176371311499,12.150997217479388,5.5309200508775875,157,12387.11,12363.29210400581,99999.0,,,
++esfm,Sri Veeramakaliamman Singapore,2,2.3974569344813434,0.5599333960859275,12.58650507397198,5.711763361742226,157,12354.49,12330.03034973145,99999.0,,,
++esfm_rc,Sri Veeramakaliamman Singapore,0,2.509821389378875,0.5815511527755557,13.072703350607625,6.052101493759435,157,11248.53,11196.92873668671,99999.0,,,
++esfm_rc,Sri Veeramakaliamman Singapore,1,2.8830400712922857,0.6676084122286379,13.610449126665129,6.2077999257192324,157,11257.08,11204.25902700424,99999.0,,,
++esfm_rc,Sri Veeramakaliamman Singapore,2,2.7715395728193424,0.64246713807485,13.500340673383525,6.122046272477576,157,11224.0,11178.18259263039,99999.0,,,
++uesfm,Sri Veeramakaliamman Singapore,0,2.2697869665184376,0.5625570243353476,17.34053107468558,8.599916604868525,157,12591.85,12539.44258975983,99999.0,,,
++uesfm,Sri Veeramakaliamman Singapore,1,2.4543260332472085,0.6034427994130044,15.868105070421757,7.655614154170212,157,12556.37,12497.1868751049,99999.0,,,
++uesfm,Sri Veeramakaliamman Singapore,2,2.2402378081453076,0.5362922009551214,14.34751765070624,6.779024945733527,157,12542.85,12490.60071396828,99999.0,,,
++uesfm_abl,Sri Veeramakaliamman Singapore,0,2.509821389378875,0.5815511527755557,13.072703350607625,6.052101493759435,157,11246.04,11199.69246673584,99999.0,,,
++uesfm_abl,Sri Veeramakaliamman Singapore,1,2.8830400712922857,0.6676084122286379,13.610449126665129,6.2077999257192324,157,11238.67,11192.63931441307,99999.0,,,
++uesfm_abl,Sri Veeramakaliamman Singapore,2,2.7715395728193424,0.64246713807485,13.500340673383525,6.122046272477576,157,11252.95,11208.22395801544,99999.0,,,
++esfm,Statue Of Liberty,0,52.35559762033152,28.241594953600423,688383.1603142131,683624.423417858,134,102.64,43.32754778862,45000.0,,,
++esfm,Statue Of Liberty,1,52.36920727661949,28.627801879821945,9277.715116667516,9303.546387753544,134,99.78,59.98358988761902,65000.0,,,
++esfm,Statue Of Liberty,2,52.3478568435799,28.64052858422677,7449.598084265948,7416.803834848509,134,97.28,3.520912647247314,0.0,,,
++esfm_rc,Statue Of Liberty,0,41.218124829650854,16.35503230171343,29.08277227297952,15.1834694799134,134,4803.59,4780.802038192749,99999.0,,,
++esfm_rc,Statue Of Liberty,1,39.57596391441381,15.93235966523679,27.37986618426645,14.290432615449056,134,4885.84,4862.140880584717,99999.0,,,
++esfm_rc,Statue Of Liberty,2,77.96860956365762,27.11543600094153,57.757423269004605,24.765649053609003,134,4900.73,4877.64520072937,99999.0,,,
++uesfm,Statue Of Liberty,0,42.72583129860769,17.96588742153844,35.11813579810792,19.871970350798126,134,5574.75,4164.912250995636,75000.0,,,
++uesfm,Statue Of Liberty,1,40.08181077917127,17.10193566857949,38.50494847120855,23.533461831836124,134,5591.94,5563.9591152668,99999.0,,,
++uesfm,Statue Of Liberty,2,77.10685903857735,28.579866483385974,68.32023407938978,27.795793203242106,134,5579.6,5550.581798315048,99999.0,,,
++uesfm_abl,Statue Of Liberty,0,41.218124829650854,16.35503230171343,29.08277227297952,15.1834694799134,134,4809.67,4784.786497354507,99999.0,,,
++uesfm_abl,Statue Of Liberty,1,39.57596391441381,15.93235966523679,27.37986618426645,14.290432615449056,134,4806.18,4784.308579206467,99999.0,,,
++uesfm_abl,Statue Of Liberty,2,77.96860956365762,27.11543600094153,57.757423269004605,24.765649053609003,134,4799.39,4778.599727869034,99999.0,,,
++esfm,The Pumpkin,0,27.59459354015253,5.94835892276036,27.52772578222832,9.11126349703041,196,4330.9,4316.109169960022,99999.0,,,
++esfm,The Pumpkin,1,26.54397520858153,5.701234509093555,27.342758332715125,8.907676505669826,196,4327.98,4314.043692827225,99999.0,,,
++esfm,The Pumpkin,2,23.05395646123413,5.007378502393024,25.75879879971585,7.898438464943201,196,4322.94,4310.356798410416,99999.0,,,
++esfm_rc,The Pumpkin,0,29.050661771479646,6.26570371316686,29.316909181775795,10.313120532688693,196,4910.6,4880.431479454041,99999.0,,,
++esfm_rc,The Pumpkin,1,26.919880388678763,5.774443198443047,27.896215187210355,9.357701886344401,196,4888.76,4859.359884023666,99999.0,,,
++esfm_rc,The Pumpkin,2,23.23435517664717,5.033668075029946,26.250953898370245,8.401411316898391,196,5035.56,5005.703458309174,99999.0,,,
++uesfm,The Pumpkin,0,28.724486849374728,6.049382168756297,32.623684990680346,13.080799898606461,196,5715.4,5688.280281066895,99999.0,,,
++uesfm,The Pumpkin,1,88.35244660282751,14.480787037348948,41.33437120165238,18.995341966051914,196,13267.96,13227.80508255959,99999.0,,,
++uesfm,The Pumpkin,2,25.512327800837042,5.499412600712255,29.975920040125473,11.811169023090269,196,5778.52,5752.150703191757,99999.0,,,
++uesfm_abl,The Pumpkin,0,29.050661771479646,6.26570371316686,29.316909181775795,10.313120532688693,196,4983.49,4954.032075881958,99999.0,,,
++uesfm_abl,The Pumpkin,1,26.919880388678763,5.774443198443047,27.896215187210355,9.357701886344401,196,4967.14,4940.374793052673,99999.0,,,
++uesfm_abl,The Pumpkin,2,23.23435517664717,5.033668075029946,26.250953898370245,8.401411316898391,196,4976.94,4951.61617064476,99999.0,,,
++esfm,Thian Hook Keng Temple Singapore,0,1.0653505173922067,0.1079565219850111,15.054152885837562,5.81055046268767,138,3299.26,3281.187238931656,99999.0,,,
++esfm,Thian Hook Keng Temple Singapore,1,0.8390930608109779,0.08272919436528102,14.587041005106803,5.705702663819475,138,3306.93,3283.257929086685,99999.0,,,
++esfm,Thian Hook Keng Temple Singapore,2,0.8760513596629002,0.08764395630143504,15.102705362055175,5.7567965038255124,138,3289.42,3278.173619031906,99999.0,,,
++esfm_rc,Thian Hook Keng Temple Singapore,0,1.0708584393011769,0.10688781753476269,15.9008082124771,6.214673486605086,138,3870.91,3843.877260684967,99999.0,,,
++esfm_rc,Thian Hook Keng Temple Singapore,1,0.9337315322680709,0.09457601793548075,15.699154845940917,6.103426802531022,138,3856.04,3835.938611507416,99999.0,,,
++esfm_rc,Thian Hook Keng Temple Singapore,2,0.9801090998477185,0.09872310339916696,15.76967975306522,6.19466789547437,138,3869.01,3848.04944896698,99999.0,,,
++uesfm,Thian Hook Keng Temple Singapore,0,0.8764983375088615,0.08202347764005188,16.212666761500287,8.391224013223542,138,4645.6,4621.513728618622,99999.0,,,
++uesfm,Thian Hook Keng Temple Singapore,1,0.9645123709523686,0.08977989341780128,17.411430835446634,7.905836840367921,138,4707.79,4683.081910133362,99999.0,,,
++uesfm,Thian Hook Keng Temple Singapore,2,0.9175998563737782,0.09144633731084849,18.520173505219134,8.485294370641896,138,4744.84,4725.761254310608,99999.0,,,
++uesfm_abl,Thian Hook Keng Temple Singapore,0,1.0708584393011769,0.10688781753476269,15.9008082124771,6.214673486605086,138,3889.87,3864.646636009216,99999.0,,,
++uesfm_abl,Thian Hook Keng Temple Singapore,1,0.9337315322680709,0.09457601793548075,15.699154845940917,6.103426802531022,138,3893.17,3865.924063205719,99999.0,,,
++uesfm_abl,Thian Hook Keng Temple Singapore,2,0.9801090998477185,0.09872310339916696,15.76967975306522,6.19466789547437,138,3912.57,3890.681934595108,99999.0,,,
++esfm,Tsar Nikolai I,0,42.507553768489565,8.869914103947062,10.846690028504929,4.007961025902478,98,4932.15,4920.938139438629,99999.0,,,
++esfm,Tsar Nikolai I,1,43.09835370739786,8.983133733509343,11.402130572143294,4.032588092580996,98,4924.71,4914.929541349411,99999.0,,,
++esfm,Tsar Nikolai I,2,40.892180545819315,8.342265568650971,11.381967217434166,4.272306098513181,98,4930.74,4920.495186090469,99999.0,,,
++esfm_rc,Tsar Nikolai I,0,42.9819031611806,8.992392580392774,11.136519361295402,4.275788532431734,98,5820.43,5790.713532209396,99999.0,,,
++esfm_rc,Tsar Nikolai I,1,42.9676733807449,8.915056020619012,11.454988629980559,4.1737271261332385,98,5855.22,5834.582437753677,99999.0,,,
++esfm_rc,Tsar Nikolai I,2,41.42806728678177,8.459777152786799,11.79264110004773,4.455551649314186,98,5812.33,5793.210029363632,99999.0,,,
++uesfm,Tsar Nikolai I,0,78.76617790472244,15.241035141808007,22.562667761517744,10.059176205736136,98,6651.48,6624.56773352623,99999.0,,,
++uesfm,Tsar Nikolai I,1,77.86748197534685,15.28454292365267,22.935722021884366,9.48322850302597,98,6683.57,6660.621834993362,99999.0,,,
++uesfm,Tsar Nikolai I,2,80.52498876805318,15.650443058307074,22.159589059526215,10.075605477720723,98,6705.71,6687.593456506729,99999.0,,,
++uesfm_abl,Tsar Nikolai I,0,42.9819031611806,8.992392580392774,11.136519361295402,4.275788532431734,98,5806.14,5773.554664373398,99999.0,,,
++uesfm_abl,Tsar Nikolai I,1,42.9676733807449,8.915056020619012,11.454988629980559,4.1737271261332385,98,5795.38,5776.308238744736,99999.0,,,
++uesfm_abl,Tsar Nikolai I,2,41.42806728678177,8.459777152786799,11.79264110004773,4.455551649314186,98,5820.44,5792.472865819931,99999.0,,,
++esfm,Urban II,0,59.0297968076411,11.008489348760156,17.600594623525765,5.400933781867582,96,8889.69,8878.01452589035,99999.0,,,
++esfm,Urban II,1,58.947927470837534,11.025374954076108,17.465515658589908,5.68378594242137,96,8885.17,8876.618469953537,99999.0,,,
++esfm,Urban II,2,58.77798771298748,10.982786678935327,17.546556385759054,5.767405629316944,96,8893.99,8885.34243106842,99999.0,,,
++esfm_rc,Urban II,0,59.216536616067906,11.08663748585098,17.82614031873468,6.107131646325687,96,3941.4,3917.114440202713,99999.0,,,
++esfm_rc,Urban II,1,59.212380355922555,11.11858086716632,17.69328746099667,6.184432905776519,96,3928.12,3913.32084608078,99999.0,,,
++esfm_rc,Urban II,2,59.03738533505208,11.026151607617244,18.25007970328771,6.5714286359830405,96,3928.76,3910.048242330551,99999.0,,,
++uesfm,Urban II,0,70.72651677542252,10.846285449637286,20.932695061288186,9.499807603806392,96,4648.82,4629.565636634827,99999.0,,,
++uesfm,Urban II,1,76.69172898118858,11.840995205937814,23.160820370282533,9.824556493964668,96,10328.85,10307.27435564995,99999.0,,,
++uesfm,Urban II,2,71.54029457581474,13.635035860571998,17.833596140295615,10.443322691822011,96,4688.65,4667.598724603653,99999.0,,,
++uesfm_abl,Urban II,0,59.216536616067906,11.08663748585098,17.82614031873468,6.107131646325687,96,3997.91,3976.541655540466,99999.0,,,
++uesfm_abl,Urban II,1,59.212380355922555,11.11858086716632,17.69328746099667,6.184432905776519,96,4013.5,3996.176211357117,99999.0,,,
++uesfm_abl,Urban II,2,59.03738533505208,11.026151607617244,18.25007970328771,6.5714286359830405,96,3985.19,3967.987709999084,99999.0,,,
++esfm,Vercingetorix,0,79.07478057258302,9.776081681696928,6.649989325889541,1.7432748253588333,69,1223.75,1212.340968608856,99999.0,,,
++esfm,Vercingetorix,1,80.09313371434186,9.858626097407651,6.426263444651325,1.6979463697629655,69,1222.8,1213.763489961624,99999.0,,,
++esfm,Vercingetorix,2,89.68787080902364,10.950615257181747,8.695412468830675,3.360645595149667,69,1225.36,1215.808262825012,99999.0,,,
++esfm_rc,Vercingetorix,0,77.58116004246067,9.538739741724191,6.629853166022587,1.863546939626005,69,2490.94,2466.239528179169,99999.0,,,
++esfm_rc,Vercingetorix,1,80.59735041412574,9.964918624137537,6.844226785033953,1.8768109759688332,69,2466.57,2448.244215965271,99999.0,,,
++esfm_rc,Vercingetorix,2,78.38409445587499,10.071797617255122,8.813177642424675,3.6575396142269336,69,2464.57,2447.203853368759,99999.0,,,
++uesfm,Vercingetorix,0,90.40583622245013,11.219354185124304,13.021170300998648,6.058136758958224,69,3147.28,3128.323934793472,99999.0,,,
++uesfm,Vercingetorix,1,86.14504993152221,10.346921123080264,10.20083592178795,4.889104941829656,69,3189.9,3161.61762547493,99999.0,,,
++uesfm,Vercingetorix,2,82.105622694467,10.494716884620729,9.246952910064651,4.3504763439652825,69,3124.47,3102.738929271698,99999.0,,,
++uesfm_abl,Vercingetorix,0,77.58116004246067,9.538739741724191,6.629853166022587,1.863546939626005,69,2471.07,2450.949246168137,99999.0,,,
++uesfm_abl,Vercingetorix,1,80.59735041412574,9.964918624137537,6.844226785033953,1.8768109759688332,69,2469.29,2452.077670097351,99999.0,,,
++uesfm_abl,Vercingetorix,2,78.38409445587499,10.071797617255122,8.813177642424675,3.6575396142269336,69,2471.57,2450.715873003006,99999.0,,,
++esfm,Yueh Hai Ching Temple Singapore,0,0.6268598783684821,0.08360284283022794,5.56223941139456,2.132543748253723,43,1749.28,1738.747783660889,99999.0,,,
++esfm,Yueh Hai Ching Temple Singapore,1,0.5064956072004329,0.07235771132992698,5.30294026596974,2.06113770612239,43,1750.4,1739.704204082489,99999.0,,,
++esfm,Yueh Hai Ching Temple Singapore,2,0.49998151063409024,0.07009863154178562,5.92393490371475,2.2317763685579015,43,1746.32,1738.905562877655,99999.0,,,
++esfm_rc,Yueh Hai Ching Temple Singapore,0,0.9825210445873275,0.12976980064181087,5.840029450443526,2.3826902247979316,43,3055.68,3033.551073074341,99999.0,,,
++esfm_rc,Yueh Hai Ching Temple Singapore,1,0.43780611024916033,0.06298154479762619,5.398989840788992,2.105985514464995,43,3050.01,3030.62473154068,99999.0,,,
++esfm_rc,Yueh Hai Ching Temple Singapore,2,0.6724636819858062,0.09232893673280672,6.135782446114854,2.362032898968259,43,3056.82,3030.568630933762,99999.0,,,
++uesfm,Yueh Hai Ching Temple Singapore,0,1.011090286407443,0.14577536562871568,8.910101741972593,4.774050044333404,43,3800.63,3779.992084264755,99999.0,,,
++uesfm,Yueh Hai Ching Temple Singapore,1,0.9987973483341249,0.13337675637303437,8.021267579781536,4.39681234232059,43,3801.31,3781.260888576508,99999.0,,,
++uesfm,Yueh Hai Ching Temple Singapore,2,0.8036578112160735,0.11221136471697148,8.679131282807896,4.609032159824265,43,3795.78,3775.22882604599,99999.0,,,
++uesfm_abl,Yueh Hai Ching Temple Singapore,0,0.9825210445873275,0.12976980064181087,5.840029450443526,2.3826902247979316,43,3061.04,3040.403587818146,99999.0,,,
++uesfm_abl,Yueh Hai Ching Temple Singapore,1,0.43780611024916033,0.06298154479762619,5.398989840788992,2.105985514464995,43,3053.44,3038.016949653625,99999.0,,,
++uesfm_abl,Yueh Hai Ching Temple Singapore,2,0.6724636819858062,0.09232893673280672,6.135782446114854,2.362032898968259,43,3056.79,3041.021898984909,99999.0,,,
+diff --git a/code/results/single_scene/summary_table.md b/code/results/single_scene/summary_table.md
+index bd3f673..664bf0b 100644
+--- a/code/results/single_scene/summary_table.md
++++ b/code/results/single_scene/summary_table.md
+@@ -2,44 +2,45 @@
+ 
+ Mean over seeds per (scene, method); best value per scene in **bold** (lower is better except Nr).
+ 
+-| Scene | ESFM Rot (deg) | U-ESFM Rot (deg) | ESFM (paper) Rot (deg) | ESFM Trans | U-ESFM Trans | ESFM (paper) Trans | ESFM Reproj (px) | U-ESFM Reproj (px) | ESFM (paper) Reproj (px) | ESFM Nr | U-ESFM Nr | ESFM (paper) Nr | ESFM Time (s) | U-ESFM Time (s) | ESFM (paper) Time (s) | ESFM Rot-BA (deg) | U-ESFM Rot-BA (deg) | ESFM (paper) Rot-BA (deg) | ESFM Trans-BA | U-ESFM Trans-BA | ESFM (paper) Trans-BA | ESFM Reproj-BA (px) | U-ESFM Reproj-BA (px) | ESFM (paper) Reproj-BA (px) |
+-|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
+-| Alcatraz Courtyard | **0.362** | 0.616 | 0.619 | **0.093** | 0.161 | 0.160 | **3.435** | 5.796 | 1.640 | **133** | 133 | -- | **4826.713** | 6481.757 | -- | -- | -- | 0.049 | -- | -- | 0.015 | -- | -- | 0.810 |
+-| Alcatraz Water Tower | **0.666** | 7.413 | 0.933 | **0.370** | 3.437 | 0.518 | **4.763** | 8.607 | 2.130 | **172** | 172 | -- | **2461.337** | 3854.907 | -- | -- | -- | 0.230 | -- | -- | 0.116 | -- | -- | 0.550 |
+-| Buddah Tooth Relic Temple Singapore | **1.131** | 12.080 | 1.030 | **0.280** | 2.308 | 0.233 | **9.964** | 19.634 | 2.060 | **162** | 162 | -- | **2916.837** | 4320.873 | -- | -- | -- | 0.081 | -- | -- | 0.014 | -- | -- | 0.850 |
+-| Doge Palace Venice | **0.801** | 0.911 | 1.163 | **0.218** | 0.290 | 0.342 | **6.432** | 10.966 | 3.620 | **241** | 241 | -- | 28768.725 | **21074.950** | -- | -- | -- | 0.211 | -- | -- | 0.029 | -- | -- | 1 |
+-| Door Lund | **0.014** | 14.172 | 0.024 | **0.004** | 0.952 | 0.006 | **1.812** | 11.922 | 0.320 | **12** | 12 | -- | **4888.363** | 9188.593 | -- | -- | -- | 0.006 | -- | -- | 0.001 | -- | -- | 0.300 |
+-| Drinking Fountain Somewhere In Zurich | **17.193** | 17.536 | 0.031 | 0.753 | **0.740** | 0.004 | **6.058** | 6.511 | 0.330 | **14** | 14 | -- | **883.837** | 2963.647 | -- | -- | -- | 0.007 | -- | -- | 0.002 | -- | -- | 0.310 |
+-| East Indiaman Goteborg | 5.568 | **5.310** | 3.814 | 0.979 | **0.959** | 0.621 | **7.318** | 9.840 | 4.130 | **179** | 179 | -- | **3987.183** | 5241.430 | -- | -- | -- | 3.117 | -- | -- | 0.509 | -- | -- | 1.850 |
+-| Ecole Superior De Guerre | 0.349 | **0.284** | 0.318 | 0.090 | **0.077** | 0.081 | **3.418** | 5.349 | 0.720 | **35** | 35 | -- | **2380.387** | 4633.833 | -- | -- | -- | 0.024 | -- | -- | 0.005 | -- | -- | 0.340 |
+-| Eglise du dome | **0.351** | 45.801 | 0.808 | **0.088** | 6.451 | 0.205 | **3.799** | 7.964 | 0.910 | **85** | 85 | -- | 25099.450 | **23371.730** | -- | -- | -- | 0.037 | -- | -- | 0.010 | -- | -- | 0.270 |
+-| Folke Filbyter | 85.831 | **81.288** | 74.596 | **0.129** | 0.130 | 0.125 | **19.275** | 23.156 | 10.370 | **40** | 40 | -- | **1581.897** | 3703.187 | -- | -- | -- | 70.157 | -- | -- | 0.118 | -- | -- | 4.290 |
+-| Fort Channing Gate Singapore | **0.093** | 11.128 | 0.207 | **0.041** | 4.004 | 0.093 | **1.783** | 6.651 | 0.520 | **27** | 27 | -- | **3434.113** | 6172.087 | -- | -- | -- | 0.020 | -- | -- | 0.008 | -- | -- | 0.250 |
+-| Golden Statue Somewhere In Hong Kong | **0.232** | 0.371 | 0.292 | **0.053** | 0.074 | 0.073 | **1.829** | 2.835 | 0.400 | **18** | 18 | -- | **4095.853** | 7762.437 | -- | -- | -- | 0.031 | -- | -- | 0.004 | -- | -- | 0.270 |
+-| Gustav Vasa | **3.509** | 6.099 | 34.181 | **0.220** | 0.345 | 1.085 | **1.655** | 3.087 | 3.520 | **18** | 18 | -- | **709.867** | 2543.553 | -- | -- | -- | 32.266 | -- | -- | 1.145 | -- | -- | 3.150 |
+-| GustavIIAdolf | **47.126** | 77.423 | 67.784 | **8.536** | 11.962 | 9.714 | **11.722** | 15.851 | 13.910 | **57** | 57 | -- | **1044.170** | 2892.580 | -- | -- | -- | 58.458 | -- | -- | 8.524 | -- | -- | 11.490 |
+-| Jonas Ahlstromer | 48.740 | **42.246** | 50.190 | **9.823** | 10.300 | 10.888 | 11.136 | **8.782** | 10.820 | **40** | 40 | -- | **615.450** | 2328.313 | -- | -- | -- | 47.117 | -- | -- | 10.451 | -- | -- | 8.410 |
+-| Kings College University Of Toronto | **9.025** | 21.523 | 0.989 | 2.044 | **1.905** | 0.235 | **3.565** | 5.525 | 0.900 | **77** | 77 | -- | **1078.980** | 2886.243 | -- | -- | -- | 0.085 | -- | -- | 0.017 | -- | -- | 0.340 |
+-| Lund University Sphinx | **11.392** | 33.023 | 19.522 | **2.981** | 6.889 | 4.585 | **6.056** | 12.060 | 4.780 | **70** | 70 | -- | **2718.567** | 4654.610 | -- | -- | -- | 8.752 | -- | -- | 2.191 | -- | -- | 1.360 |
+-| Nijo Castle Gate | **0.839** | 1.682 | 1.495 | **0.165** | 0.326 | 0.286 | **7.415** | 12.138 | 1.700 | **19** | 19 | -- | **1095.177** | 3306.697 | -- | -- | -- | 0.069 | -- | -- | 0.012 | -- | -- | 0.730 |
+-| Pantheon Paris | -- | 19.345 | 0.192 | -- | 3.065 | 0.050 | -- | 13.562 | 1.470 | -- | 179 | -- | -- | 5497.377 | -- | -- | -- | 0.040 | -- | -- | 0.005 | -- | -- | 0.490 |
+-| Park Gate Clermont Ferrand | 21.388 | **19.302** | 0.391 | 10.192 | **9.397** | 0.125 | **7.712** | 7.852 | 0.570 | **34** | 34 | -- | **1596.940** | 3597.403 | -- | -- | -- | 0.049 | -- | -- | 0.022 | -- | -- | 0.350 |
+-| Plaza De Armas Santiago | **0.775** | 3.327 | 6.782 | **0.337** | 1.435 | 2.944 | **6.009** | 9.463 | 7.400 | **240** | 240 | -- | **6372.683** | 6535.537 | -- | -- | -- | 2.556 | -- | -- | 1.383 | -- | -- | 4.900 |
+-| Porta San Donato Bologna | **0.592** | 21.006 | 2.153 | **0.107** | 3.541 | 0.388 | **5.603** | 13.244 | 2.280 | **141** | 141 | -- | **3754.163** | 5118.030 | -- | -- | -- | 0.095 | -- | -- | 0.046 | -- | -- | 0.750 |
+-| Round Church Cambridge | **2.129** | 2.490 | 2.451 | **0.920** | 1.002 | 1.003 | **6.097** | 8.546 | 2.660 | **92** | 92 | -- | **9776.630** | 12061.350 | -- | -- | -- | 1.107 | -- | -- | 0.582 | -- | -- | 1.540 |
+-| Skansen Kronan Gothenburg | **0.338** | 0.375 | 0.736 | **0.116** | 0.124 | 0.226 | **3.048** | 4.799 | 1.240 | **131** | 131 | -- | 17163.940 | **16235.090** | -- | -- | -- | 0.026 | -- | -- | 0.008 | -- | -- | 0.670 |
+-| Smolny Cathedral St Petersburg | **21.806** | 22.143 | 0.554 | 2.111 | **2.099** | 0.051 | **11.228** | 12.526 | 1.660 | **131** | 131 | -- | **10362.595** | 10475.740 | -- | -- | -- | 0.033 | -- | -- | 0.006 | -- | -- | 0.810 |
+-| Sri Mariamman Singapore | **1.219** | 6.479 | 2.302 | **0.382** | 2.188 | 0.683 | **10.088** | 17.854 | 4.130 | **222** | 222 | -- | **5985.540** | 6485.667 | -- | -- | -- | 0.077 | -- | -- | 0.023 | -- | -- | 0.910 |
+-| Sri Thendayuthapani Singapore | **0.719** | 47.075 | 46.269 | **0.179** | 3.855 | 3.812 | **7.124** | 14.133 | 23.370 | **98** | 98 | -- | **13362.090** | 13808.440 | -- | -- | -- | 44.170 | -- | -- | 2.870 | -- | -- | 8.440 |
+-| Sri Veeramakaliamman Singapore | **2.334** | 29.405 | 2.559 | **0.543** | 3.868 | 0.597 | **11.965** | 28.951 | 3.470 | **157** | 157 | -- | **12392.180** | 12544.690 | -- | -- | -- | 0.175 | -- | -- | 0.040 | -- | -- | 0.730 |
+-| Statue Of Liberty | 52.358 | **39.859** | 46.887 | 28.503 | **18.503** | 20.012 | 235036.825 | **41.028** | 26.160 | **134** | 134 | -- | **99.900** | 12924 | -- | -- | -- | 9.091 | -- | -- | 4.122 | -- | -- | 6.970 |
+-| The Pumpkin | **25.731** | 26.571 | 94.672 | **5.552** | 5.665 | 14.890 | **26.876** | 35.367 | 33.410 | **196** | 196 | -- | **4327.273** | 5790.597 | -- | -- | -- | 98.862 | -- | -- | 14.952 | -- | -- | 24.850 |
+-| Thian Hook Keng Temple Singapore | **0.927** | 1.201 | 0.832 | **0.093** | 0.121 | 0.082 | **14.915** | 20.789 | 2.750 | **138** | 138 | -- | **3298.537** | 4612.373 | -- | -- | -- | 0.081 | -- | -- | 0.008 | -- | -- | 1.130 |
+-| Tsar Nikolai I | **42.803** | 78.413 | 48.499 | **8.927** | 15.394 | 9.467 | **11.124** | 18.431 | 9.790 | **98** | 98 | -- | **4928.430** | 6646.845 | -- | -- | -- | 36.280 | -- | -- | 7.836 | -- | -- | 6.530 |
+-| Urban II | 59.030 | -- | 47.490 | 11.008 | -- | 9.467 | 17.601 | -- | 9.380 | 96 | -- | -- | 8889.690 | -- | -- | -- | -- | 48.214 | -- | -- | 9.586 | -- | -- | 6.920 |
+-| Vercingetorix | **82.952** | 86.088 | 69.328 | **10.195** | 10.890 | 8.788 | **7.257** | 8.681 | 5.080 | **69** | 69 | -- | **1223.970** | 3105.607 | -- | -- | -- | 17.706 | -- | -- | 3.104 | -- | -- | 1.500 |
+-| Yueh Hai Ching Temple Singapore | 0.627 | -- | 0.720 | 0.084 | -- | 0.098 | 5.562 | -- | 0.940 | 43 | -- | -- | 1749.280 | -- | -- | -- | -- | 0.043 | -- | -- | 0.014 | -- | -- | 0.650 |
+-| **Mean** | **16.146** | 23.697 | 18.023 | **3.121** | 4.014 | 2.912 | 6920.602 | **13.088** | 5.673 | 100.559 | **104.818** | -- | **5819.728** | 7358.187 | -- | -- | -- | 13.695 | -- | -- | 1.937 | -- | -- | 2.992 |
++| Scene | ESFM (official code) Rot (deg) | ESFM (RESfM code) Rot (deg) | U-ESFM Rot (deg) | U-ESFM arch + ESFMLoss Rot (deg) | ESFM (paper) Rot (deg) | ESFM (official code) Trans | ESFM (RESfM code) Trans | U-ESFM Trans | U-ESFM arch + ESFMLoss Trans | ESFM (paper) Trans | ESFM (official code) Reproj (px) | ESFM (RESfM code) Reproj (px) | U-ESFM Reproj (px) | U-ESFM arch + ESFMLoss Reproj (px) | ESFM (paper) Reproj (px) | ESFM (official code) Nr | ESFM (RESfM code) Nr | U-ESFM Nr | U-ESFM arch + ESFMLoss Nr | ESFM (paper) Nr | ESFM (official code) Time (s) | ESFM (RESfM code) Time (s) | U-ESFM Time (s) | U-ESFM arch + ESFMLoss Time (s) | ESFM (paper) Time (s) | ESFM (official code) Rot-BA (deg) | ESFM (RESfM code) Rot-BA (deg) | U-ESFM Rot-BA (deg) | U-ESFM arch + ESFMLoss Rot-BA (deg) | ESFM (paper) Rot-BA (deg) | ESFM (official code) Trans-BA | ESFM (RESfM code) Trans-BA | U-ESFM Trans-BA | U-ESFM arch + ESFMLoss Trans-BA | ESFM (paper) Trans-BA | ESFM (official code) Reproj-BA (px) | ESFM (RESfM code) Reproj-BA (px) | U-ESFM Reproj-BA (px) | U-ESFM arch + ESFMLoss Reproj-BA (px) | ESFM (paper) Reproj-BA (px) |
++|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
++| Alcatraz Courtyard | **0.362** | 0.420 | 10.132 | 0.420 | 0.619 | **0.093** | 0.110 | 2.219 | 0.110 | 0.160 | **3.435** | 3.673 | 7.272 | 3.673 | 1.640 | **133** | 133 | 133 | 133 | -- | **4826.713** | 5684.120 | 6550.983 | 5702.993 | -- | 0.044 | -- | -- | -- | 0.049 | 0.017 | -- | -- | -- | 0.015 | 0.965 | -- | -- | -- | 0.810 |
++| Alcatraz Water Tower | **0.666** | 0.882 | 1.480 | 0.882 | 0.933 | **0.370** | 0.488 | 0.803 | 0.488 | 0.518 | **4.763** | 5.112 | 7.397 | 5.112 | 2.130 | **172** | 172 | 172 | 172 | -- | **2461.337** | 3064.183 | 3795.877 | 3136.880 | -- | 0.227 | -- | -- | -- | 0.230 | 0.115 | -- | -- | -- | 0.116 | 0.741 | -- | -- | -- | 0.550 |
++| Buddah Tooth Relic Temple Singapore | **1.131** | 1.267 | 1.791 | 1.267 | 1.030 | **0.280** | 0.337 | 0.520 | 0.337 | 0.233 | **9.964** | 10.743 | 14.029 | 10.743 | 2.060 | **162** | 162 | 162 | 162 | -- | **2916.837** | 3544.803 | 4289.503 | 3526.473 | -- | 0.086 | -- | -- | -- | 0.081 | 0.016 | -- | -- | -- | 0.014 | 2.079 | -- | -- | -- | 0.850 |
++| Doge Palace Venice | **0.753** | 0.880 | 1.121 | 0.880 | 1.163 | **0.209** | 0.249 | 0.350 | 0.249 | 0.342 | **6.359** | 6.831 | 10.496 | 6.831 | 3.620 | **241** | 241 | 241 | 241 | -- | 22522.737 | **7930.497** | 20767.260 | 8046.347 | -- | 0.057 | -- | -- | -- | 0.211 | 0.018 | -- | -- | -- | 0.029 | 1.177 | -- | -- | -- | 1 |
++| Door Lund | **0.014** | 0.018 | 4.651 | 0.018 | 0.024 | **0.004** | 0.005 | 0.344 | 0.005 | 0.006 | **1.812** | 1.891 | 5.545 | 1.891 | 0.320 | **12** | 12 | 12 | 12 | -- | **4888.363** | 8451.030 | 9255.943 | 8509 | -- | 0.003 | -- | -- | -- | 0.006 | 0.001 | -- | -- | -- | 0.001 | 0.303 | -- | -- | -- | 0.300 |
++| Drinking Fountain Somewhere In Zurich | **17.193** | 17.271 | 17.329 | 17.271 | 0.031 | 0.753 | **0.749** | 0.750 | 0.749 | 0.004 | **6.058** | 6.129 | 6.675 | 6.129 | 0.330 | **14** | 14 | 14 | 14 | -- | **883.837** | 2332.817 | 5815.220 | 2434.447 | -- | 15.614 | -- | -- | -- | 0.007 | 0.798 | -- | -- | -- | 0.002 | 5.185 | -- | -- | -- | 0.310 |
++| East Indiaman Goteborg | 5.568 | **5.530** | 7.231 | 5.530 | 3.814 | 0.979 | **0.978** | 1.364 | 0.978 | 0.621 | **7.318** | 7.536 | 10.130 | 7.536 | 4.130 | **179** | 179 | 179 | 179 | -- | **3987.183** | 4385.873 | 5198.093 | 4379.980 | -- | 5.245 | -- | -- | -- | 3.117 | 0.993 | -- | -- | -- | 0.509 | 2.648 | -- | -- | -- | 1.850 |
++| Ecole Superior De Guerre | **0.349** | 2.337 | 25.800 | 2.337 | 0.318 | **0.090** | 0.545 | 1.324 | 0.545 | 0.081 | **3.418** | 4.726 | 8.646 | 4.726 | 0.720 | **35** | 35 | 35 | 35 | -- | **2380.387** | 3932.013 | 10416.080 | 3946.533 | -- | 0.014 | -- | -- | -- | 0.024 | 0.003 | -- | -- | -- | 0.005 | 0.355 | -- | -- | -- | 0.340 |
++| Eglise du dome | **0.386** | 0.569 | 15.675 | 0.533 | 0.808 | **0.098** | 0.147 | 2.288 | 0.135 | 0.205 | **3.987** | 4.250 | 6.815 | 4.359 | 0.910 | **85** | 85 | 85 | 85 | -- | 19631.293 | **8795.890** | 9802.683 | 8835.457 | -- | 0.034 | -- | -- | -- | 0.037 | 0.010 | -- | -- | -- | 0.010 | 2.081 | -- | -- | -- | 0.270 |
++| Folke Filbyter | 85.831 | 85.900 | **82.656** | 85.900 | 74.596 | **0.129** | 0.129 | 0.131 | 0.129 | 0.125 | **19.275** | 19.898 | 28.225 | 19.898 | 10.370 | **40** | 40 | 40 | 40 | -- | **1581.897** | 3072.870 | 3755.193 | 3036.570 | -- | 71.440 | -- | -- | -- | 70.157 | 0.130 | -- | -- | -- | 0.118 | 20.014 | -- | -- | -- | 4.290 |
++| Fort Channing Gate Singapore | **0.093** | 0.117 | 0.154 | 0.117 | 0.207 | **0.041** | 0.052 | 0.071 | 0.052 | 0.093 | **1.783** | 1.871 | 3.715 | 1.871 | 0.520 | **27** | 27 | 27 | 27 | -- | **3434.113** | 5450.200 | 6171.250 | 5437.450 | -- | 0.018 | -- | -- | -- | 0.020 | 0.007 | -- | -- | -- | 0.008 | 0.260 | -- | -- | -- | 0.250 |
++| Golden Statue Somewhere In Hong Kong | **0.232** | 0.267 | 0.508 | 0.267 | 0.292 | **0.053** | 0.062 | 0.082 | 0.062 | 0.073 | **1.829** | 1.944 | 3.022 | 1.944 | 0.400 | **18** | 18 | 18 | 18 | -- | **4095.853** | 7057.423 | 11699.307 | 7077.063 | -- | 0.026 | -- | -- | -- | 0.031 | 0.003 | -- | -- | -- | 0.004 | 0.282 | -- | -- | -- | 0.270 |
++| Gustav Vasa | **3.509** | 4.122 | 25.854 | 4.122 | 34.181 | **0.220** | 0.249 | 0.775 | 0.249 | 1.085 | **1.655** | 1.874 | 4.747 | 1.874 | 3.520 | **18** | 18 | 18 | 18 | -- | **709.867** | 1928.880 | 3887.653 | 1952.417 | -- | 1.004 | -- | -- | -- | 32.266 | 0.113 | -- | -- | -- | 1.145 | 0.538 | -- | -- | -- | 3.150 |
++| GustavIIAdolf | **47.126** | 47.468 | 88.194 | 47.468 | 67.784 | **8.536** | 8.574 | 13.129 | 8.574 | 9.714 | **11.722** | 11.918 | 17.220 | 11.918 | 13.910 | **57** | 57 | 57 | 57 | -- | **1044.170** | 2265.203 | 2981.017 | 2288.163 | -- | 24.738 | -- | -- | -- | 58.458 | 5.460 | -- | -- | -- | 8.524 | 16.352 | -- | -- | -- | 11.490 |
++| Jonas Ahlstromer | 48.740 | 48.415 | **46.584** | 48.415 | 50.190 | 9.823 | **9.672** | 10.245 | 9.672 | 10.888 | **11.136** | 11.328 | 11.144 | 11.328 | 10.820 | **40** | 40 | 40 | 40 | -- | **615.450** | 1712.753 | 2414.653 | 1737.937 | -- | 32.847 | -- | -- | -- | 47.117 | 6.311 | -- | -- | -- | 10.451 | 7.502 | -- | -- | -- | 8.410 |
++| Kings College University Of Toronto | **9.025** | 9.312 | 14.284 | 9.312 | 0.989 | 2.044 | 2.108 | **1.639** | 2.108 | 0.235 | **3.565** | 3.732 | 6.351 | 3.732 | 0.900 | **77** | 77 | 77 | 77 | -- | **1078.980** | 2209.693 | 2857.733 | 2197.337 | -- | 5.786 | -- | -- | -- | 0.085 | 1.224 | -- | -- | -- | 0.017 | 1.768 | -- | -- | -- | 0.340 |
++| Lund University Sphinx | **11.392** | 13.007 | 21.718 | 13.007 | 19.522 | **2.981** | 3.385 | 4.817 | 3.385 | 4.585 | **6.056** | 6.760 | 11.550 | 6.760 | 4.780 | **70** | 70 | 70 | 70 | -- | **2718.567** | 3924.373 | 4642.657 | 3911.693 | -- | 7.011 | -- | -- | -- | 8.752 | 1.807 | -- | -- | -- | 2.191 | 1.201 | -- | -- | -- | 1.360 |
++| Nijo Castle Gate | **0.839** | 1.325 | 3.926 | 1.325 | 1.495 | **0.165** | 0.255 | 1.138 | 0.255 | 0.286 | **7.415** | 8.223 | 15.135 | 8.223 | 1.700 | **19** | 19 | 19 | 19 | -- | **1095.177** | 2629.490 | 3234.357 | 2631.137 | -- | 0.081 | -- | -- | -- | 0.069 | 0.013 | -- | -- | -- | 0.012 | 0.761 | -- | -- | -- | 0.730 |
++| Pantheon Paris | -- | **0.253** | 15.196 | 0.253 | 0.192 | -- | **0.056** | 2.291 | 0.056 | 0.050 | -- | **4.893** | 10.638 | 4.893 | 1.470 | -- | **179** | 179 | 179 | -- | -- | 4640.607 | 5557.600 | **4617.300** | -- | -- | -- | -- | -- | 0.040 | -- | -- | -- | -- | 0.005 | -- | -- | -- | -- | 0.490 |
++| Park Gate Clermont Ferrand | 21.388 | **21.310** | 23.641 | 21.310 | 0.391 | 10.192 | **10.177** | 11.173 | 10.177 | 0.125 | **7.712** | 7.747 | 8.652 | 7.747 | 0.570 | **34** | 34 | 34 | 34 | -- | **1596.940** | 2952.627 | 3650.170 | 2977.200 | -- | 10.261 | -- | -- | -- | 0.049 | 4.338 | -- | -- | -- | 0.022 | 14.280 | -- | -- | -- | 0.350 |
++| Plaza De Armas Santiago | **0.775** | 1.122 | 6.571 | 1.122 | 6.782 | **0.337** | 0.487 | 2.852 | 0.487 | 2.944 | **6.009** | 6.327 | 11.199 | 6.327 | 7.400 | **240** | 240 | 240 | 240 | -- | 6372.683 | 5678.743 | 6616.970 | **5595.347** | -- | 0.135 | -- | -- | -- | 2.556 | 0.054 | -- | -- | -- | 1.383 | 2.924 | -- | -- | -- | 4.900 |
++| Porta San Donato Bologna | **0.592** | 0.610 | 1.814 | 0.610 | 2.153 | **0.107** | 0.112 | 0.468 | 0.112 | 0.388 | **5.603** | 5.779 | 9.168 | 5.779 | 2.280 | **141** | 141 | 141 | 141 | -- | **3754.163** | 4317.957 | 5167.537 | 4348.100 | -- | 0.112 | -- | -- | -- | 0.095 | 0.056 | -- | -- | -- | 0.046 | 3.224 | -- | -- | -- | 0.750 |
++| Round Church Cambridge | 2.137 | 2.293 | **2.097** | 2.293 | 2.451 | 0.927 | 0.967 | **0.772** | 0.967 | 1.003 | **6.103** | 6.493 | 8.283 | 6.493 | 2.660 | **92** | 92 | 92 | 92 | -- | **9779.270** | 10883.870 | 12129.840 | 10903.483 | -- | 0.987 | -- | -- | -- | 1.107 | 0.553 | -- | -- | -- | 0.582 | 4.877 | -- | -- | -- | 1.540 |
++| Skansen Kronan Gothenburg | **0.301** | 0.312 | 0.342 | 0.318 | 0.736 | **0.102** | 0.106 | 0.118 | 0.108 | 0.226 | **2.906** | 3.020 | 4.753 | 3.006 | 1.240 | **131** | 131 | 131 | 131 | -- | 17153.677 | **6127.697** | 10156.743 | 6169.433 | -- | 0.025 | -- | -- | -- | 0.026 | 0.008 | -- | -- | -- | 0.008 | 0.682 | -- | -- | -- | 0.670 |
++| Smolny Cathedral St Petersburg | **21.775** | 21.848 | 22.241 | 21.848 | 0.554 | 2.111 | 2.107 | **2.096** | 2.107 | 0.051 | **11.239** | 11.420 | 12.951 | 11.420 | 1.660 | **131** | 131 | 131 | 131 | -- | 10362.787 | 9384.753 | 10445.090 | **9320.357** | -- | 21.416 | -- | -- | -- | 0.033 | 2.158 | -- | -- | -- | 0.006 | 10.310 | -- | -- | -- | 0.810 |
++| Some Cathedral In Barcelona | **0.651** | 0.787 | 14.741 | 0.787 | 0.880 | **0.233** | 0.282 | 4.842 | 0.282 | 0.315 | **6.271** | 6.796 | 15.920 | 6.796 | 2.870 | **177** | 177 | 177 | 177 | -- | 5001.973 | **4961.633** | 5889.770 | 4963.200 | -- | 0.033 | -- | -- | -- | 0.026 | 0.014 | -- | -- | -- | 0.011 | 1.100 | -- | -- | -- | 0.890 |
++| Sri Mariamman Singapore | **1.219** | 1.747 | 2.113 | 1.747 | 2.302 | **0.382** | 0.536 | 0.686 | 0.536 | 0.683 | **10.088** | 11.135 | 13.613 | 11.135 | 4.130 | **222** | 222 | 222 | 222 | -- | 5985.540 | **5502.090** | 14745.487 | 5512.673 | -- | 0.080 | -- | -- | -- | 0.077 | 0.024 | -- | -- | -- | 0.023 | 0.925 | -- | -- | -- | 0.910 |
++| Sri Thendayuthapani Singapore | **0.593** | 0.700 | 0.784 | 0.708 | 46.269 | **0.150** | 0.170 | 0.225 | 0.173 | 3.812 | **7.016** | 7.282 | 9.787 | 7.361 | 23.370 | **98** | 98 | 98 | 98 | -- | 13338.167 | **12537.777** | 13900.903 | 12561.583 | -- | 0.195 | -- | -- | -- | 44.170 | 0.057 | -- | -- | -- | 2.870 | 7.058 | -- | -- | -- | 8.440 |
++| Sri Veeramakaliamman Singapore | 2.404 | 2.721 | **2.321** | 2.721 | 2.559 | **0.559** | 0.631 | 0.567 | 0.631 | 0.597 | **12.234** | 13.394 | 15.852 | 13.394 | 3.470 | **157** | 157 | 157 | 157 | -- | 12377.927 | **11243.203** | 12563.690 | 11245.887 | -- | -- | -- | -- | -- | 0.175 | -- | -- | -- | -- | 0.040 | -- | -- | -- | -- | 0.730 |
++| Statue Of Liberty | **52.358** | 52.921 | 53.305 | 52.921 | 46.887 | 28.503 | **19.801** | 21.216 | 19.801 | 20.012 | 235036.825 | **38.073** | 47.314 | 38.073 | 26.160 | **134** | 134 | 134 | 134 | -- | **99.900** | 4863.387 | 5582.097 | 4805.080 | -- | -- | -- | -- | -- | 9.091 | -- | -- | -- | -- | 4.122 | -- | -- | -- | -- | 6.970 |
++| The Pumpkin | **25.731** | 26.402 | 47.530 | 26.402 | 94.672 | **5.552** | 5.691 | 8.677 | 5.691 | 14.890 | **26.876** | 27.821 | 34.645 | 27.821 | 33.410 | **196** | 196 | 196 | 196 | -- | **4327.273** | 4944.973 | 8253.960 | 4975.857 | -- | -- | -- | -- | -- | 98.862 | -- | -- | -- | -- | 14.952 | -- | -- | -- | -- | 24.850 |
++| Thian Hook Keng Temple Singapore | 0.927 | 0.995 | **0.920** | 0.995 | 0.832 | 0.093 | 0.100 | **0.088** | 0.100 | 0.082 | **14.915** | 15.790 | 17.381 | 15.790 | 2.750 | **138** | 138 | 138 | 138 | -- | **3298.537** | 3865.320 | 4699.410 | 3898.537 | -- | -- | -- | -- | -- | 0.081 | -- | -- | -- | -- | 0.008 | -- | -- | -- | -- | 1.130 |
++| Tsar Nikolai I | **42.166** | 42.459 | 79.053 | 42.459 | 48.499 | **8.732** | 8.789 | 15.392 | 8.789 | 9.467 | **11.210** | 11.461 | 22.553 | 11.461 | 9.790 | **98** | 98 | 98 | 98 | -- | **4929.200** | 5829.327 | 6680.253 | 5807.320 | -- | -- | -- | -- | -- | 36.280 | -- | -- | -- | -- | 7.836 | -- | -- | -- | -- | 6.530 |
++| Urban II | **58.919** | 59.155 | 72.986 | 59.155 | 47.490 | **11.006** | 11.077 | 12.107 | 11.077 | 9.467 | **17.538** | 17.923 | 20.642 | 17.923 | 9.380 | **96** | 96 | 96 | 96 | -- | 8889.617 | **3932.760** | 6555.440 | 3998.867 | -- | -- | -- | -- | -- | 48.214 | -- | -- | -- | -- | 9.586 | -- | -- | -- | -- | 6.920 |
++| Vercingetorix | 82.952 | **78.854** | 86.219 | 78.854 | 69.328 | 10.195 | **9.858** | 10.687 | 9.858 | 8.788 | **7.257** | 7.429 | 10.823 | 7.429 | 5.080 | **69** | 69 | 69 | 69 | -- | **1223.970** | 2474.027 | 3153.883 | 2470.643 | -- | -- | -- | -- | -- | 17.706 | -- | -- | -- | -- | 3.104 | -- | -- | -- | -- | 1.500 |
++| Yueh Hai Ching Temple Singapore | **0.544** | 0.698 | 0.938 | 0.698 | 0.720 | **0.075** | 0.095 | 0.130 | 0.095 | 0.098 | **5.596** | 5.792 | 8.537 | 5.792 | 0.940 | **43** | 43 | 43 | 43 | -- | **1748.667** | 3054.170 | 3799.240 | 3057.090 | -- | -- | -- | -- | -- | 0.043 | -- | -- | -- | -- | 0.014 | -- | -- | -- | -- | 0.650 |
++| **Mean** | 15.676 | 15.397 | 22.275 | **15.396** | 17.547 | 3.032 | 2.754 | 3.788 | **2.754** | 2.840 | 6723.056 | **9.084** | 12.801 | 9.089 | 5.595 | 102.743 | **104.861** | 104.861 | 104.861 | -- | 5460.373 | **5154.640** | 7141.210 | 5167.106 | -- | 7.316 | -- | -- | -- | 13.315 | 0.900 | -- | -- | -- | 1.883 | 4.059 | -- | -- | -- | 2.933 |
+ 
+ _Seeds per cell: [0, 1, 2]._
+ 
+diff --git a/code/results/single_scene/summary_table.tex b/code/results/single_scene/summary_table.tex
+index 258e3f2..70d4642 100644
+--- a/code/results/single_scene/summary_table.tex
++++ b/code/results/single_scene/summary_table.tex
+@@ -4,48 +4,49 @@
+ \caption{Single-scene optimization on the Olsson dataset (calibrated setting). Mean over [0, 1, 2] seeds; best per scene in bold. ESFM (paper) reproduces the published per-scene numbers of \cite{Moran_2021_ICCV} (post-BA values use their BA) and is excluded from the bold-best comparison.}
+ \label{tab:single_scene_olsson}
+ \resizebox{\textwidth}{!}{
+-\begin{tabular}{lcccccccccccccccccccccccc}
++\begin{tabular}{lcccccccccccccccccccccccccccccccccccccccc}
+ \toprule
+- & \multicolumn{3}{c}{Rot (deg)} & \multicolumn{3}{c}{Trans} & \multicolumn{3}{c}{Reproj (px)} & \multicolumn{3}{c}{Nr} & \multicolumn{3}{c}{Time (s)} & \multicolumn{3}{c}{Rot-BA (deg)} & \multicolumn{3}{c}{Trans-BA} & \multicolumn{3}{c}{Reproj-BA (px)} \\
+-Scene & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) \\
++ & \multicolumn{5}{c}{Rot (deg)} & \multicolumn{5}{c}{Trans} & \multicolumn{5}{c}{Reproj (px)} & \multicolumn{5}{c}{Nr} & \multicolumn{5}{c}{Time (s)} & \multicolumn{5}{c}{Rot-BA (deg)} & \multicolumn{5}{c}{Trans-BA} & \multicolumn{5}{c}{Reproj-BA (px)} \\
++Scene & ESFM (official code) & ESFM (RESfM code) & U-ESFM & U-ESFM arch + ESFMLoss & ESFM (paper) & ESFM (official code) & ESFM (RESfM code) & U-ESFM & U-ESFM arch + ESFMLoss & ESFM (paper) & ESFM (official code) & ESFM (RESfM code) & U-ESFM & U-ESFM arch + ESFMLoss & ESFM (paper) & ESFM (official code) & ESFM (RESfM code) & U-ESFM & U-ESFM arch + ESFMLoss & ESFM (paper) & ESFM (official code) & ESFM (RESfM code) & U-ESFM & U-ESFM arch + ESFMLoss & ESFM (paper) & ESFM (official code) & ESFM (RESfM code) & U-ESFM & U-ESFM arch + ESFMLoss & ESFM (paper) & ESFM (official code) & ESFM (RESfM code) & U-ESFM & U-ESFM arch + ESFMLoss & ESFM (paper) & ESFM (official code) & ESFM (RESfM code) & U-ESFM & U-ESFM arch + ESFMLoss & ESFM (paper) \\
+ \midrule
+-Alcatraz Courtyard & \textbf{0.362} & 0.616 & 0.619 & \textbf{0.093} & 0.161 & 0.160 & \textbf{3.435} & 5.796 & 1.640 & \textbf{133} & 133 & -- & \textbf{4826.713} & 6481.757 & -- & -- & -- & 0.049 & -- & -- & 0.015 & -- & -- & 0.810 \\
+-Alcatraz Water Tower & \textbf{0.666} & 7.413 & 0.933 & \textbf{0.370} & 3.437 & 0.518 & \textbf{4.763} & 8.607 & 2.130 & \textbf{172} & 172 & -- & \textbf{2461.337} & 3854.907 & -- & -- & -- & 0.230 & -- & -- & 0.116 & -- & -- & 0.550 \\
+-Buddah Tooth Relic Temple Singapore & \textbf{1.131} & 12.080 & 1.030 & \textbf{0.280} & 2.308 & 0.233 & \textbf{9.964} & 19.634 & 2.060 & \textbf{162} & 162 & -- & \textbf{2916.837} & 4320.873 & -- & -- & -- & 0.081 & -- & -- & 0.014 & -- & -- & 0.850 \\
+-Doge Palace Venice & \textbf{0.801} & 0.911 & 1.163 & \textbf{0.218} & 0.290 & 0.342 & \textbf{6.432} & 10.966 & 3.620 & \textbf{241} & 241 & -- & 28768.725 & \textbf{21074.950} & -- & -- & -- & 0.211 & -- & -- & 0.029 & -- & -- & 1 \\
+-Door Lund & \textbf{0.014} & 14.172 & 0.024 & \textbf{0.004} & 0.952 & 0.006 & \textbf{1.812} & 11.922 & 0.320 & \textbf{12} & 12 & -- & \textbf{4888.363} & 9188.593 & -- & -- & -- & 0.006 & -- & -- & 0.001 & -- & -- & 0.300 \\
+-Drinking Fountain Somewhere In Zurich & \textbf{17.193} & 17.536 & 0.031 & 0.753 & \textbf{0.740} & 0.004 & \textbf{6.058} & 6.511 & 0.330 & \textbf{14} & 14 & -- & \textbf{883.837} & 2963.647 & -- & -- & -- & 0.007 & -- & -- & 0.002 & -- & -- & 0.310 \\
+-East Indiaman Goteborg & 5.568 & \textbf{5.310} & 3.814 & 0.979 & \textbf{0.959} & 0.621 & \textbf{7.318} & 9.840 & 4.130 & \textbf{179} & 179 & -- & \textbf{3987.183} & 5241.430 & -- & -- & -- & 3.117 & -- & -- & 0.509 & -- & -- & 1.850 \\
+-Ecole Superior De Guerre & 0.349 & \textbf{0.284} & 0.318 & 0.090 & \textbf{0.077} & 0.081 & \textbf{3.418} & 5.349 & 0.720 & \textbf{35} & 35 & -- & \textbf{2380.387} & 4633.833 & -- & -- & -- & 0.024 & -- & -- & 0.005 & -- & -- & 0.340 \\
+-Eglise du dome & \textbf{0.351} & 45.801 & 0.808 & \textbf{0.088} & 6.451 & 0.205 & \textbf{3.799} & 7.964 & 0.910 & \textbf{85} & 85 & -- & 25099.450 & \textbf{23371.730} & -- & -- & -- & 0.037 & -- & -- & 0.010 & -- & -- & 0.270 \\
+-Folke Filbyter & 85.831 & \textbf{81.288} & 74.596 & \textbf{0.129} & 0.130 & 0.125 & \textbf{19.275} & 23.156 & 10.370 & \textbf{40} & 40 & -- & \textbf{1581.897} & 3703.187 & -- & -- & -- & 70.157 & -- & -- & 0.118 & -- & -- & 4.290 \\
+-Fort Channing Gate Singapore & \textbf{0.093} & 11.128 & 0.207 & \textbf{0.041} & 4.004 & 0.093 & \textbf{1.783} & 6.651 & 0.520 & \textbf{27} & 27 & -- & \textbf{3434.113} & 6172.087 & -- & -- & -- & 0.020 & -- & -- & 0.008 & -- & -- & 0.250 \\
+-Golden Statue Somewhere In Hong Kong & \textbf{0.232} & 0.371 & 0.292 & \textbf{0.053} & 0.074 & 0.073 & \textbf{1.829} & 2.835 & 0.400 & \textbf{18} & 18 & -- & \textbf{4095.853} & 7762.437 & -- & -- & -- & 0.031 & -- & -- & 0.004 & -- & -- & 0.270 \\
+-Gustav Vasa & \textbf{3.509} & 6.099 & 34.181 & \textbf{0.220} & 0.345 & 1.085 & \textbf{1.655} & 3.087 & 3.520 & \textbf{18} & 18 & -- & \textbf{709.867} & 2543.553 & -- & -- & -- & 32.266 & -- & -- & 1.145 & -- & -- & 3.150 \\
+-GustavIIAdolf & \textbf{47.126} & 77.423 & 67.784 & \textbf{8.536} & 11.962 & 9.714 & \textbf{11.722} & 15.851 & 13.910 & \textbf{57} & 57 & -- & \textbf{1044.170} & 2892.580 & -- & -- & -- & 58.458 & -- & -- & 8.524 & -- & -- & 11.490 \\
+-Jonas Ahlstromer & 48.740 & \textbf{42.246} & 50.190 & \textbf{9.823} & 10.300 & 10.888 & 11.136 & \textbf{8.782} & 10.820 & \textbf{40} & 40 & -- & \textbf{615.450} & 2328.313 & -- & -- & -- & 47.117 & -- & -- & 10.451 & -- & -- & 8.410 \\
+-Kings College University Of Toronto & \textbf{9.025} & 21.523 & 0.989 & 2.044 & \textbf{1.905} & 0.235 & \textbf{3.565} & 5.525 & 0.900 & \textbf{77} & 77 & -- & \textbf{1078.980} & 2886.243 & -- & -- & -- & 0.085 & -- & -- & 0.017 & -- & -- & 0.340 \\
+-Lund University Sphinx & \textbf{11.392} & 33.023 & 19.522 & \textbf{2.981} & 6.889 & 4.585 & \textbf{6.056} & 12.060 & 4.780 & \textbf{70} & 70 & -- & \textbf{2718.567} & 4654.610 & -- & -- & -- & 8.752 & -- & -- & 2.191 & -- & -- & 1.360 \\
+-Nijo Castle Gate & \textbf{0.839} & 1.682 & 1.495 & \textbf{0.165} & 0.326 & 0.286 & \textbf{7.415} & 12.138 & 1.700 & \textbf{19} & 19 & -- & \textbf{1095.177} & 3306.697 & -- & -- & -- & 0.069 & -- & -- & 0.012 & -- & -- & 0.730 \\
+-Pantheon Paris & -- & 19.345 & 0.192 & -- & 3.065 & 0.050 & -- & 13.562 & 1.470 & -- & 179 & -- & -- & 5497.377 & -- & -- & -- & 0.040 & -- & -- & 0.005 & -- & -- & 0.490 \\
+-Park Gate Clermont Ferrand & 21.388 & \textbf{19.302} & 0.391 & 10.192 & \textbf{9.397} & 0.125 & \textbf{7.712} & 7.852 & 0.570 & \textbf{34} & 34 & -- & \textbf{1596.940} & 3597.403 & -- & -- & -- & 0.049 & -- & -- & 0.022 & -- & -- & 0.350 \\
+-Plaza De Armas Santiago & \textbf{0.775} & 3.327 & 6.782 & \textbf{0.337} & 1.435 & 2.944 & \textbf{6.009} & 9.463 & 7.400 & \textbf{240} & 240 & -- & \textbf{6372.683} & 6535.537 & -- & -- & -- & 2.556 & -- & -- & 1.383 & -- & -- & 4.900 \\
+-Porta San Donato Bologna & \textbf{0.592} & 21.006 & 2.153 & \textbf{0.107} & 3.541 & 0.388 & \textbf{5.603} & 13.244 & 2.280 & \textbf{141} & 141 & -- & \textbf{3754.163} & 5118.030 & -- & -- & -- & 0.095 & -- & -- & 0.046 & -- & -- & 0.750 \\
+-Round Church Cambridge & \textbf{2.129} & 2.490 & 2.451 & \textbf{0.920} & 1.002 & 1.003 & \textbf{6.097} & 8.546 & 2.660 & \textbf{92} & 92 & -- & \textbf{9776.630} & 12061.350 & -- & -- & -- & 1.107 & -- & -- & 0.582 & -- & -- & 1.540 \\
+-Skansen Kronan Gothenburg & \textbf{0.338} & 0.375 & 0.736 & \textbf{0.116} & 0.124 & 0.226 & \textbf{3.048} & 4.799 & 1.240 & \textbf{131} & 131 & -- & 17163.940 & \textbf{16235.090} & -- & -- & -- & 0.026 & -- & -- & 0.008 & -- & -- & 0.670 \\
+-Smolny Cathedral St Petersburg & \textbf{21.806} & 22.143 & 0.554 & 2.111 & \textbf{2.099} & 0.051 & \textbf{11.228} & 12.526 & 1.660 & \textbf{131} & 131 & -- & \textbf{10362.595} & 10475.740 & -- & -- & -- & 0.033 & -- & -- & 0.006 & -- & -- & 0.810 \\
+-Sri Mariamman Singapore & \textbf{1.219} & 6.479 & 2.302 & \textbf{0.382} & 2.188 & 0.683 & \textbf{10.088} & 17.854 & 4.130 & \textbf{222} & 222 & -- & \textbf{5985.540} & 6485.667 & -- & -- & -- & 0.077 & -- & -- & 0.023 & -- & -- & 0.910 \\
+-Sri Thendayuthapani Singapore & \textbf{0.719} & 47.075 & 46.269 & \textbf{0.179} & 3.855 & 3.812 & \textbf{7.124} & 14.133 & 23.370 & \textbf{98} & 98 & -- & \textbf{13362.090} & 13808.440 & -- & -- & -- & 44.170 & -- & -- & 2.870 & -- & -- & 8.440 \\
+-Sri Veeramakaliamman Singapore & \textbf{2.334} & 29.405 & 2.559 & \textbf{0.543} & 3.868 & 0.597 & \textbf{11.965} & 28.951 & 3.470 & \textbf{157} & 157 & -- & \textbf{12392.180} & 12544.690 & -- & -- & -- & 0.175 & -- & -- & 0.040 & -- & -- & 0.730 \\
+-Statue Of Liberty & 52.358 & \textbf{39.859} & 46.887 & 28.503 & \textbf{18.503} & 20.012 & 235036.825 & \textbf{41.028} & 26.160 & \textbf{134} & 134 & -- & \textbf{99.900} & 12924 & -- & -- & -- & 9.091 & -- & -- & 4.122 & -- & -- & 6.970 \\
+-The Pumpkin & \textbf{25.731} & 26.571 & 94.672 & \textbf{5.552} & 5.665 & 14.890 & \textbf{26.876} & 35.367 & 33.410 & \textbf{196} & 196 & -- & \textbf{4327.273} & 5790.597 & -- & -- & -- & 98.862 & -- & -- & 14.952 & -- & -- & 24.850 \\
+-Thian Hook Keng Temple Singapore & \textbf{0.927} & 1.201 & 0.832 & \textbf{0.093} & 0.121 & 0.082 & \textbf{14.915} & 20.789 & 2.750 & \textbf{138} & 138 & -- & \textbf{3298.537} & 4612.373 & -- & -- & -- & 0.081 & -- & -- & 0.008 & -- & -- & 1.130 \\
+-Tsar Nikolai I & \textbf{42.803} & 78.413 & 48.499 & \textbf{8.927} & 15.394 & 9.467 & \textbf{11.124} & 18.431 & 9.790 & \textbf{98} & 98 & -- & \textbf{4928.430} & 6646.845 & -- & -- & -- & 36.280 & -- & -- & 7.836 & -- & -- & 6.530 \\
+-Urban II & 59.030 & -- & 47.490 & 11.008 & -- & 9.467 & 17.601 & -- & 9.380 & 96 & -- & -- & 8889.690 & -- & -- & -- & -- & 48.214 & -- & -- & 9.586 & -- & -- & 6.920 \\
+-Vercingetorix & \textbf{82.952} & 86.088 & 69.328 & \textbf{10.195} & 10.890 & 8.788 & \textbf{7.257} & 8.681 & 5.080 & \textbf{69} & 69 & -- & \textbf{1223.970} & 3105.607 & -- & -- & -- & 17.706 & -- & -- & 3.104 & -- & -- & 1.500 \\
+-Yueh Hai Ching Temple Singapore & 0.627 & -- & 0.720 & 0.084 & -- & 0.098 & 5.562 & -- & 0.940 & 43 & -- & -- & 1749.280 & -- & -- & -- & -- & 0.043 & -- & -- & 0.014 & -- & -- & 0.650 \\
++Alcatraz Courtyard & \textbf{0.362} & 0.420 & 10.132 & 0.420 & 0.619 & \textbf{0.093} & 0.110 & 2.219 & 0.110 & 0.160 & \textbf{3.435} & 3.673 & 7.272 & 3.673 & 1.640 & \textbf{133} & 133 & 133 & 133 & -- & \textbf{4826.713} & 5684.120 & 6550.983 & 5702.993 & -- & 0.044 & -- & -- & -- & 0.049 & 0.017 & -- & -- & -- & 0.015 & 0.965 & -- & -- & -- & 0.810 \\
++Alcatraz Water Tower & \textbf{0.666} & 0.882 & 1.480 & 0.882 & 0.933 & \textbf{0.370} & 0.488 & 0.803 & 0.488 & 0.518 & \textbf{4.763} & 5.112 & 7.397 & 5.112 & 2.130 & \textbf{172} & 172 & 172 & 172 & -- & \textbf{2461.337} & 3064.183 & 3795.877 & 3136.880 & -- & 0.227 & -- & -- & -- & 0.230 & 0.115 & -- & -- & -- & 0.116 & 0.741 & -- & -- & -- & 0.550 \\
++Buddah Tooth Relic Temple Singapore & \textbf{1.131} & 1.267 & 1.791 & 1.267 & 1.030 & \textbf{0.280} & 0.337 & 0.520 & 0.337 & 0.233 & \textbf{9.964} & 10.743 & 14.029 & 10.743 & 2.060 & \textbf{162} & 162 & 162 & 162 & -- & \textbf{2916.837} & 3544.803 & 4289.503 & 3526.473 & -- & 0.086 & -- & -- & -- & 0.081 & 0.016 & -- & -- & -- & 0.014 & 2.079 & -- & -- & -- & 0.850 \\
++Doge Palace Venice & \textbf{0.753} & 0.880 & 1.121 & 0.880 & 1.163 & \textbf{0.209} & 0.249 & 0.350 & 0.249 & 0.342 & \textbf{6.359} & 6.831 & 10.496 & 6.831 & 3.620 & \textbf{241} & 241 & 241 & 241 & -- & 22522.737 & \textbf{7930.497} & 20767.260 & 8046.347 & -- & 0.057 & -- & -- & -- & 0.211 & 0.018 & -- & -- & -- & 0.029 & 1.177 & -- & -- & -- & 1 \\
++Door Lund & \textbf{0.014} & 0.018 & 4.651 & 0.018 & 0.024 & \textbf{0.004} & 0.005 & 0.344 & 0.005 & 0.006 & \textbf{1.812} & 1.891 & 5.545 & 1.891 & 0.320 & \textbf{12} & 12 & 12 & 12 & -- & \textbf{4888.363} & 8451.030 & 9255.943 & 8509 & -- & 0.003 & -- & -- & -- & 0.006 & 0.001 & -- & -- & -- & 0.001 & 0.303 & -- & -- & -- & 0.300 \\
++Drinking Fountain Somewhere In Zurich & \textbf{17.193} & 17.271 & 17.329 & 17.271 & 0.031 & 0.753 & \textbf{0.749} & 0.750 & 0.749 & 0.004 & \textbf{6.058} & 6.129 & 6.675 & 6.129 & 0.330 & \textbf{14} & 14 & 14 & 14 & -- & \textbf{883.837} & 2332.817 & 5815.220 & 2434.447 & -- & 15.614 & -- & -- & -- & 0.007 & 0.798 & -- & -- & -- & 0.002 & 5.185 & -- & -- & -- & 0.310 \\
++East Indiaman Goteborg & 5.568 & \textbf{5.530} & 7.231 & 5.530 & 3.814 & 0.979 & \textbf{0.978} & 1.364 & 0.978 & 0.621 & \textbf{7.318} & 7.536 & 10.130 & 7.536 & 4.130 & \textbf{179} & 179 & 179 & 179 & -- & \textbf{3987.183} & 4385.873 & 5198.093 & 4379.980 & -- & 5.245 & -- & -- & -- & 3.117 & 0.993 & -- & -- & -- & 0.509 & 2.648 & -- & -- & -- & 1.850 \\
++Ecole Superior De Guerre & \textbf{0.349} & 2.337 & 25.800 & 2.337 & 0.318 & \textbf{0.090} & 0.545 & 1.324 & 0.545 & 0.081 & \textbf{3.418} & 4.726 & 8.646 & 4.726 & 0.720 & \textbf{35} & 35 & 35 & 35 & -- & \textbf{2380.387} & 3932.013 & 10416.080 & 3946.533 & -- & 0.014 & -- & -- & -- & 0.024 & 0.003 & -- & -- & -- & 0.005 & 0.355 & -- & -- & -- & 0.340 \\
++Eglise du dome & \textbf{0.386} & 0.569 & 15.675 & 0.533 & 0.808 & \textbf{0.098} & 0.147 & 2.288 & 0.135 & 0.205 & \textbf{3.987} & 4.250 & 6.815 & 4.359 & 0.910 & \textbf{85} & 85 & 85 & 85 & -- & 19631.293 & \textbf{8795.890} & 9802.683 & 8835.457 & -- & 0.034 & -- & -- & -- & 0.037 & 0.010 & -- & -- & -- & 0.010 & 2.081 & -- & -- & -- & 0.270 \\
++Folke Filbyter & 85.831 & 85.900 & \textbf{82.656} & 85.900 & 74.596 & \textbf{0.129} & 0.129 & 0.131 & 0.129 & 0.125 & \textbf{19.275} & 19.898 & 28.225 & 19.898 & 10.370 & \textbf{40} & 40 & 40 & 40 & -- & \textbf{1581.897} & 3072.870 & 3755.193 & 3036.570 & -- & 71.440 & -- & -- & -- & 70.157 & 0.130 & -- & -- & -- & 0.118 & 20.014 & -- & -- & -- & 4.290 \\
++Fort Channing Gate Singapore & \textbf{0.093} & 0.117 & 0.154 & 0.117 & 0.207 & \textbf{0.041} & 0.052 & 0.071 & 0.052 & 0.093 & \textbf{1.783} & 1.871 & 3.715 & 1.871 & 0.520 & \textbf{27} & 27 & 27 & 27 & -- & \textbf{3434.113} & 5450.200 & 6171.250 & 5437.450 & -- & 0.018 & -- & -- & -- & 0.020 & 0.007 & -- & -- & -- & 0.008 & 0.260 & -- & -- & -- & 0.250 \\
++Golden Statue Somewhere In Hong Kong & \textbf{0.232} & 0.267 & 0.508 & 0.267 & 0.292 & \textbf{0.053} & 0.062 & 0.082 & 0.062 & 0.073 & \textbf{1.829} & 1.944 & 3.022 & 1.944 & 0.400 & \textbf{18} & 18 & 18 & 18 & -- & \textbf{4095.853} & 7057.423 & 11699.307 & 7077.063 & -- & 0.026 & -- & -- & -- & 0.031 & 0.003 & -- & -- & -- & 0.004 & 0.282 & -- & -- & -- & 0.270 \\
++Gustav Vasa & \textbf{3.509} & 4.122 & 25.854 & 4.122 & 34.181 & \textbf{0.220} & 0.249 & 0.775 & 0.249 & 1.085 & \textbf{1.655} & 1.874 & 4.747 & 1.874 & 3.520 & \textbf{18} & 18 & 18 & 18 & -- & \textbf{709.867} & 1928.880 & 3887.653 & 1952.417 & -- & 1.004 & -- & -- & -- & 32.266 & 0.113 & -- & -- & -- & 1.145 & 0.538 & -- & -- & -- & 3.150 \\
++GustavIIAdolf & \textbf{47.126} & 47.468 & 88.194 & 47.468 & 67.784 & \textbf{8.536} & 8.574 & 13.129 & 8.574 & 9.714 & \textbf{11.722} & 11.918 & 17.220 & 11.918 & 13.910 & \textbf{57} & 57 & 57 & 57 & -- & \textbf{1044.170} & 2265.203 & 2981.017 & 2288.163 & -- & 24.738 & -- & -- & -- & 58.458 & 5.460 & -- & -- & -- & 8.524 & 16.352 & -- & -- & -- & 11.490 \\
++Jonas Ahlstromer & 48.740 & 48.415 & \textbf{46.584} & 48.415 & 50.190 & 9.823 & \textbf{9.672} & 10.245 & 9.672 & 10.888 & \textbf{11.136} & 11.328 & 11.144 & 11.328 & 10.820 & \textbf{40} & 40 & 40 & 40 & -- & \textbf{615.450} & 1712.753 & 2414.653 & 1737.937 & -- & 32.847 & -- & -- & -- & 47.117 & 6.311 & -- & -- & -- & 10.451 & 7.502 & -- & -- & -- & 8.410 \\
++Kings College University Of Toronto & \textbf{9.025} & 9.312 & 14.284 & 9.312 & 0.989 & 2.044 & 2.108 & \textbf{1.639} & 2.108 & 0.235 & \textbf{3.565} & 3.732 & 6.351 & 3.732 & 0.900 & \textbf{77} & 77 & 77 & 77 & -- & \textbf{1078.980} & 2209.693 & 2857.733 & 2197.337 & -- & 5.786 & -- & -- & -- & 0.085 & 1.224 & -- & -- & -- & 0.017 & 1.768 & -- & -- & -- & 0.340 \\
++Lund University Sphinx & \textbf{11.392} & 13.007 & 21.718 & 13.007 & 19.522 & \textbf{2.981} & 3.385 & 4.817 & 3.385 & 4.585 & \textbf{6.056} & 6.760 & 11.550 & 6.760 & 4.780 & \textbf{70} & 70 & 70 & 70 & -- & \textbf{2718.567} & 3924.373 & 4642.657 & 3911.693 & -- & 7.011 & -- & -- & -- & 8.752 & 1.807 & -- & -- & -- & 2.191 & 1.201 & -- & -- & -- & 1.360 \\
++Nijo Castle Gate & \textbf{0.839} & 1.325 & 3.926 & 1.325 & 1.495 & \textbf{0.165} & 0.255 & 1.138 & 0.255 & 0.286 & \textbf{7.415} & 8.223 & 15.135 & 8.223 & 1.700 & \textbf{19} & 19 & 19 & 19 & -- & \textbf{1095.177} & 2629.490 & 3234.357 & 2631.137 & -- & 0.081 & -- & -- & -- & 0.069 & 0.013 & -- & -- & -- & 0.012 & 0.761 & -- & -- & -- & 0.730 \\
++Pantheon Paris & -- & \textbf{0.253} & 15.196 & 0.253 & 0.192 & -- & \textbf{0.056} & 2.291 & 0.056 & 0.050 & -- & \textbf{4.893} & 10.638 & 4.893 & 1.470 & -- & \textbf{179} & 179 & 179 & -- & -- & 4640.607 & 5557.600 & \textbf{4617.300} & -- & -- & -- & -- & -- & 0.040 & -- & -- & -- & -- & 0.005 & -- & -- & -- & -- & 0.490 \\
++Park Gate Clermont Ferrand & 21.388 & \textbf{21.310} & 23.641 & 21.310 & 0.391 & 10.192 & \textbf{10.177} & 11.173 & 10.177 & 0.125 & \textbf{7.712} & 7.747 & 8.652 & 7.747 & 0.570 & \textbf{34} & 34 & 34 & 34 & -- & \textbf{1596.940} & 2952.627 & 3650.170 & 2977.200 & -- & 10.261 & -- & -- & -- & 0.049 & 4.338 & -- & -- & -- & 0.022 & 14.280 & -- & -- & -- & 0.350 \\
++Plaza De Armas Santiago & \textbf{0.775} & 1.122 & 6.571 & 1.122 & 6.782 & \textbf{0.337} & 0.487 & 2.852 & 0.487 & 2.944 & \textbf{6.009} & 6.327 & 11.199 & 6.327 & 7.400 & \textbf{240} & 240 & 240 & 240 & -- & 6372.683 & 5678.743 & 6616.970 & \textbf{5595.347} & -- & 0.135 & -- & -- & -- & 2.556 & 0.054 & -- & -- & -- & 1.383 & 2.924 & -- & -- & -- & 4.900 \\
++Porta San Donato Bologna & \textbf{0.592} & 0.610 & 1.814 & 0.610 & 2.153 & \textbf{0.107} & 0.112 & 0.468 & 0.112 & 0.388 & \textbf{5.603} & 5.779 & 9.168 & 5.779 & 2.280 & \textbf{141} & 141 & 141 & 141 & -- & \textbf{3754.163} & 4317.957 & 5167.537 & 4348.100 & -- & 0.112 & -- & -- & -- & 0.095 & 0.056 & -- & -- & -- & 0.046 & 3.224 & -- & -- & -- & 0.750 \\
++Round Church Cambridge & 2.137 & 2.293 & \textbf{2.097} & 2.293 & 2.451 & 0.927 & 0.967 & \textbf{0.772} & 0.967 & 1.003 & \textbf{6.103} & 6.493 & 8.283 & 6.493 & 2.660 & \textbf{92} & 92 & 92 & 92 & -- & \textbf{9779.270} & 10883.870 & 12129.840 & 10903.483 & -- & 0.987 & -- & -- & -- & 1.107 & 0.553 & -- & -- & -- & 0.582 & 4.877 & -- & -- & -- & 1.540 \\
++Skansen Kronan Gothenburg & \textbf{0.301} & 0.312 & 0.342 & 0.318 & 0.736 & \textbf{0.102} & 0.106 & 0.118 & 0.108 & 0.226 & \textbf{2.906} & 3.020 & 4.753 & 3.006 & 1.240 & \textbf{131} & 131 & 131 & 131 & -- & 17153.677 & \textbf{6127.697} & 10156.743 & 6169.433 & -- & 0.025 & -- & -- & -- & 0.026 & 0.008 & -- & -- & -- & 0.008 & 0.682 & -- & -- & -- & 0.670 \\
++Smolny Cathedral St Petersburg & \textbf{21.775} & 21.848 & 22.241 & 21.848 & 0.554 & 2.111 & 2.107 & \textbf{2.096} & 2.107 & 0.051 & \textbf{11.239} & 11.420 & 12.951 & 11.420 & 1.660 & \textbf{131} & 131 & 131 & 131 & -- & 10362.787 & 9384.753 & 10445.090 & \textbf{9320.357} & -- & 21.416 & -- & -- & -- & 0.033 & 2.158 & -- & -- & -- & 0.006 & 10.310 & -- & -- & -- & 0.810 \\
++Some Cathedral In Barcelona & \textbf{0.651} & 0.787 & 14.741 & 0.787 & 0.880 & \textbf{0.233} & 0.282 & 4.842 & 0.282 & 0.315 & \textbf{6.271} & 6.796 & 15.920 & 6.796 & 2.870 & \textbf{177} & 177 & 177 & 177 & -- & 5001.973 & \textbf{4961.633} & 5889.770 & 4963.200 & -- & 0.033 & -- & -- & -- & 0.026 & 0.014 & -- & -- & -- & 0.011 & 1.100 & -- & -- & -- & 0.890 \\
++Sri Mariamman Singapore & \textbf{1.219} & 1.747 & 2.113 & 1.747 & 2.302 & \textbf{0.382} & 0.536 & 0.686 & 0.536 & 0.683 & \textbf{10.088} & 11.135 & 13.613 & 11.135 & 4.130 & \textbf{222} & 222 & 222 & 222 & -- & 5985.540 & \textbf{5502.090} & 14745.487 & 5512.673 & -- & 0.080 & -- & -- & -- & 0.077 & 0.024 & -- & -- & -- & 0.023 & 0.925 & -- & -- & -- & 0.910 \\
++Sri Thendayuthapani Singapore & \textbf{0.593} & 0.700 & 0.784 & 0.708 & 46.269 & \textbf{0.150} & 0.170 & 0.225 & 0.173 & 3.812 & \textbf{7.016} & 7.282 & 9.787 & 7.361 & 23.370 & \textbf{98} & 98 & 98 & 98 & -- & 13338.167 & \textbf{12537.777} & 13900.903 & 12561.583 & -- & 0.195 & -- & -- & -- & 44.170 & 0.057 & -- & -- & -- & 2.870 & 7.058 & -- & -- & -- & 8.440 \\
++Sri Veeramakaliamman Singapore & 2.404 & 2.721 & \textbf{2.321} & 2.721 & 2.559 & \textbf{0.559} & 0.631 & 0.567 & 0.631 & 0.597 & \textbf{12.234} & 13.394 & 15.852 & 13.394 & 3.470 & \textbf{157} & 157 & 157 & 157 & -- & 12377.927 & \textbf{11243.203} & 12563.690 & 11245.887 & -- & -- & -- & -- & -- & 0.175 & -- & -- & -- & -- & 0.040 & -- & -- & -- & -- & 0.730 \\
++Statue Of Liberty & \textbf{52.358} & 52.921 & 53.305 & 52.921 & 46.887 & 28.503 & \textbf{19.801} & 21.216 & 19.801 & 20.012 & 235036.825 & \textbf{38.073} & 47.314 & 38.073 & 26.160 & \textbf{134} & 134 & 134 & 134 & -- & \textbf{99.900} & 4863.387 & 5582.097 & 4805.080 & -- & -- & -- & -- & -- & 9.091 & -- & -- & -- & -- & 4.122 & -- & -- & -- & -- & 6.970 \\
++The Pumpkin & \textbf{25.731} & 26.402 & 47.530 & 26.402 & 94.672 & \textbf{5.552} & 5.691 & 8.677 & 5.691 & 14.890 & \textbf{26.876} & 27.821 & 34.645 & 27.821 & 33.410 & \textbf{196} & 196 & 196 & 196 & -- & \textbf{4327.273} & 4944.973 & 8253.960 & 4975.857 & -- & -- & -- & -- & -- & 98.862 & -- & -- & -- & -- & 14.952 & -- & -- & -- & -- & 24.850 \\
++Thian Hook Keng Temple Singapore & 0.927 & 0.995 & \textbf{0.920} & 0.995 & 0.832 & 0.093 & 0.100 & \textbf{0.088} & 0.100 & 0.082 & \textbf{14.915} & 15.790 & 17.381 & 15.790 & 2.750 & \textbf{138} & 138 & 138 & 138 & -- & \textbf{3298.537} & 3865.320 & 4699.410 & 3898.537 & -- & -- & -- & -- & -- & 0.081 & -- & -- & -- & -- & 0.008 & -- & -- & -- & -- & 1.130 \\
++Tsar Nikolai I & \textbf{42.166} & 42.459 & 79.053 & 42.459 & 48.499 & \textbf{8.732} & 8.789 & 15.392 & 8.789 & 9.467 & \textbf{11.210} & 11.461 & 22.553 & 11.461 & 9.790 & \textbf{98} & 98 & 98 & 98 & -- & \textbf{4929.200} & 5829.327 & 6680.253 & 5807.320 & -- & -- & -- & -- & -- & 36.280 & -- & -- & -- & -- & 7.836 & -- & -- & -- & -- & 6.530 \\
++Urban II & \textbf{58.919} & 59.155 & 72.986 & 59.155 & 47.490 & \textbf{11.006} & 11.077 & 12.107 & 11.077 & 9.467 & \textbf{17.538} & 17.923 & 20.642 & 17.923 & 9.380 & \textbf{96} & 96 & 96 & 96 & -- & 8889.617 & \textbf{3932.760} & 6555.440 & 3998.867 & -- & -- & -- & -- & -- & 48.214 & -- & -- & -- & -- & 9.586 & -- & -- & -- & -- & 6.920 \\
++Vercingetorix & 82.952 & \textbf{78.854} & 86.219 & 78.854 & 69.328 & 10.195 & \textbf{9.858} & 10.687 & 9.858 & 8.788 & \textbf{7.257} & 7.429 & 10.823 & 7.429 & 5.080 & \textbf{69} & 69 & 69 & 69 & -- & \textbf{1223.970} & 2474.027 & 3153.883 & 2470.643 & -- & -- & -- & -- & -- & 17.706 & -- & -- & -- & -- & 3.104 & -- & -- & -- & -- & 1.500 \\
++Yueh Hai Ching Temple Singapore & \textbf{0.544} & 0.698 & 0.938 & 0.698 & 0.720 & \textbf{0.075} & 0.095 & 0.130 & 0.095 & 0.098 & \textbf{5.596} & 5.792 & 8.537 & 5.792 & 0.940 & \textbf{43} & 43 & 43 & 43 & -- & \textbf{1748.667} & 3054.170 & 3799.240 & 3057.090 & -- & -- & -- & -- & -- & 0.043 & -- & -- & -- & -- & 0.014 & -- & -- & -- & -- & 0.650 \\
+ \midrule
+-Mean & \textbf{16.146} & 23.697 & 18.023 & \textbf{3.121} & 4.014 & 2.912 & 6920.602 & \textbf{13.088} & 5.673 & 100.559 & \textbf{104.818} & -- & \textbf{5819.728} & 7358.187 & -- & -- & -- & 13.695 & -- & -- & 1.937 & -- & -- & 2.992 \\
++Mean & 15.676 & 15.397 & 22.275 & \textbf{15.396} & 17.547 & 3.032 & 2.754 & 3.788 & \textbf{2.754} & 2.840 & 6723.056 & \textbf{9.084} & 12.801 & 9.089 & 5.595 & 102.743 & \textbf{104.861} & 104.861 & 104.861 & -- & 5460.373 & \textbf{5154.640} & 7141.210 & 5167.106 & -- & 7.316 & -- & -- & -- & 13.315 & 0.900 & -- & -- & -- & 1.883 & 4.059 & -- & -- & -- & 2.933 \\
+ \bottomrule
+ \end{tabular}}
+ \end{table*}
 ```
 - untracked/modified files:
 ```
+M code/results/single_scene/REPRO.md
+ M code/results/single_scene/summary.csv
+ M code/results/single_scene/summary_table.md
+ M code/results/single_scene/summary_table.tex
 ?? MVG_Project_Report_Ortal_Dayan.pdf
 ?? "RESFM- ROBUST DEEP EQUIVARIANT STRUCTURE FROM MOTION.pdf"
 ?? "claude specs/SPEC_cvpr_experiments.md"
 ?? "claude specs/SPEC_sfm_datasets_setup.md"
 ?? "claude specs/SPEC_single_scene_experiments.md"
 ?? "claude specs/SPEC_uesfm_combined.md"
+?? "claude specs/TASK_crossdataset_readiness.md"
 ?? code/datasets/Euclidean
 ?? tmp_ab_check/
 ```
 ### esfm-baseline (official ESFM)
 - path: `/home/projects/bagon/ortalda/MVG/final-project/esfm-baseline`
-- commit: `2c0079497bf5dda1eeb2a902b2f859c6387f213a`
+- commit: `de666af53423c1dc3dd412e53c4e153c03d70922`
 - **WARNING: working tree dirty.** Diff:
 ```diff
 (untracked files only)
@@ -58,6 +4541,7 @@ Generated: 2026-07-13T20:41:04
 ?? code/confs/single_scene_generated/ss_esfm_Buddah_Tooth_Relic_Temple_Singapore_seed1.conf
 ?? code/confs/single_scene_generated/ss_esfm_Buddah_Tooth_Relic_Temple_Singapore_seed2.conf
 ?? code/confs/single_scene_generated/ss_esfm_Doge_Palace_Venice_seed1.conf
+?? code/confs/single_scene_generated/ss_esfm_Doge_Palace_Venice_seed2.conf
 ?? code/confs/single_scene_generated/ss_esfm_Door_Lund_seed1.conf
 ?? code/confs/single_scene_generated/ss_esfm_Door_Lund_seed2.conf
 ?? code/confs/single_scene_generated/ss_esfm_Drinking_Fountain_Somewhere_In_Zurich_seed1.conf
@@ -70,6 +4554,7 @@ Generated: 2026-07-13T20:41:04
 ?? code/confs/single_scene_generated/ss_esfm_Ecole_Superior_De_Guerre_seed2.conf
 ?? code/confs/single_scene_generated/ss_esfm_Eglise_du_dome_seed0.conf
 ?? code/confs/single_scene_generated/ss_esfm_Eglise_du_dome_seed1.conf
+?? code/confs/single_scene_generated/ss_esfm_Eglise_du_dome_seed2.conf
 ?? code/confs/single_scene_generated/ss_esfm_Folke_Filbyter_seed0.conf
 ?? code/confs/single_scene_generated/ss_esfm_Folke_Filbyter_seed1.conf
 ?? code/confs/single_scene_generated/ss_esfm_Folke_Filbyter_seed2.conf
@@ -114,16 +4599,22 @@ Generated: 2026-07-13T20:41:04
 ?? code/confs/single_scene_generated/ss_esfm_Round_Church_Cambridge_seed2.conf
 ?? code/confs/single_scene_generated/ss_esfm_Skansen_Kronan_Gothenburg_seed0.conf
 ?? code/confs/single_scene_generated/ss_esfm_Skansen_Kronan_Gothenburg_seed1.conf
+?? code/confs/single_scene_generated/ss_esfm_Skansen_Kronan_Gothenburg_seed2.conf
 ?? code/confs/single_scene_generated/ss_esfm_Smolny_Cathedral_St_Petersburg_seed0.conf
 ?? code/confs/single_scene_generated/ss_esfm_Smolny_Cathedral_St_Petersburg_seed1.conf
+?? code/confs/single_scene_generated/ss_esfm_Smolny_Cathedral_St_Petersburg_seed2.conf
 ?? code/confs/single_scene_generated/ss_esfm_Some_Cathedral_In_Barcelona_seed0.conf
+?? code/confs/single_scene_generated/ss_esfm_Some_Cathedral_In_Barcelona_seed1.conf
+?? code/confs/single_scene_generated/ss_esfm_Some_Cathedral_In_Barcelona_seed2.conf
 ?? code/confs/single_scene_generated/ss_esfm_Sri_Mariamman_Singapore_seed0.conf
 ?? code/confs/single_scene_generated/ss_esfm_Sri_Mariamman_Singapore_seed1.conf
 ?? code/confs/single_scene_generated/ss_esfm_Sri_Mariamman_Singapore_seed2.conf
 ?? code/confs/single_scene_generated/ss_esfm_Sri_Thendayuthapani_Singapore_seed0.conf
 ?? code/confs/single_scene_generated/ss_esfm_Sri_Thendayuthapani_Singapore_seed1.conf
+?? code/confs/single_scene_generated/ss_esfm_Sri_Thendayuthapani_Singapore_seed2.conf
 ?? code/confs/single_scene_generated/ss_esfm_Sri_Veeramakaliamman_Singapore_seed0.conf
 ?? code/confs/single_scene_generated/ss_esfm_Sri_Veeramakaliamman_Singapore_seed1.conf
+?? code/confs/single_scene_generated/ss_esfm_Sri_Veeramakaliamman_Singapore_seed2.conf
 ?? code/confs/single_scene_generated/ss_esfm_Statue_Of_Liberty_seed0.conf
 ?? code/confs/single_scene_generated/ss_esfm_Statue_Of_Liberty_seed1.conf
 ?? code/confs/single_scene_generated/ss_esfm_Statue_Of_Liberty_seed2.conf
@@ -137,10 +4628,14 @@ Generated: 2026-07-13T20:41:04
 ?? code/confs/single_scene_generated/ss_esfm_Tsar_Nikolai_I_seed1.conf
 ?? code/confs/single_scene_generated/ss_esfm_Tsar_Nikolai_I_seed2.conf
 ?? code/confs/single_scene_generated/ss_esfm_Urban_II_seed0.conf
+?? code/confs/single_scene_generated/ss_esfm_Urban_II_seed1.conf
+?? code/confs/single_scene_generated/ss_esfm_Urban_II_seed2.conf
 ?? code/confs/single_scene_generated/ss_esfm_Vercingetorix_seed0.conf
 ?? code/confs/single_scene_generated/ss_esfm_Vercingetorix_seed1.conf
 ?? code/confs/single_scene_generated/ss_esfm_Vercingetorix_seed2.conf
 ?? code/confs/single_scene_generated/ss_esfm_Yueh_Hai_Ching_Temple_Singapore_seed0.conf
+?? code/confs/single_scene_generated/ss_esfm_Yueh_Hai_Ching_Temple_Singapore_seed1.conf
+?? code/confs/single_scene_generated/ss_esfm_Yueh_Hai_Ching_Temple_Singapore_seed2.conf
 ```
 
 ## Environment (shared venv used for BOTH methods)
diff --git a/code/results/single_scene/summary.csv b/code/results/single_scene/summary.csv
index e4f9f5e..ec1ee18 100644
--- a/code/results/single_scene/summary.csv
+++ b/code/results/single_scene/summary.csv
@@ -1,172 +1,319 @@
-method,scene,seed,rot_err_deg,pos_err,reproj_err_px,reproj_med_px,n_cams,runtime_s,convergence_time_s,best_epoch
-esfm,Alcatraz Courtyard,0,0.324288380204305,0.0854554642858637,3.379232694353053,1.9204072326560708,133,4821.4,4792.707528591156,99999.0
-esfm,Alcatraz Courtyard,1,0.36138678157391557,0.09112912757839253,3.3691035794153503,1.8536696768772378,133,4825.72,4804.594581127167,99999.0
-esfm,Alcatraz Courtyard,2,0.4015043649148433,0.10235190268069758,3.557242072642648,1.9198007399315369,133,4833.02,4812.242679834366,99999.0
-uesfm,Alcatraz Courtyard,0,0.6370266175309793,0.1671943803031883,5.956592777085197,3.6638282340102246,133,6478.06,6448.735792636871,99999.0
-uesfm,Alcatraz Courtyard,1,0.5812669610362707,0.15343994793419002,5.71686544495911,3.550076086474883,133,6483.48,6454.087207078934,99999.0
-uesfm,Alcatraz Courtyard,2,0.6293740753551684,0.16343948136157807,5.713459273661585,3.459422664018691,133,6483.73,6451.153926610947,99999.0
-esfm,Alcatraz Water Tower,0,0.6984432687370988,0.38694796813795407,4.741313233832226,2.1058898944784543,172,2461.59,2440.421297073364,99999.0
-esfm,Alcatraz Water Tower,1,0.5944425806062753,0.3287537445614408,4.5656682619915765,1.9703698309044646,172,2467.28,2454.429310798645,99999.0
-esfm,Alcatraz Water Tower,2,0.7051984712289041,0.39359395644708545,4.982802041506025,2.1318655747232063,172,2455.14,2441.783402681351,99999.0
-uesfm,Alcatraz Water Tower,0,18.69941055700858,8.189402666249999,12.38813738647375,6.481578998967327,172,3857.94,3831.41609287262,99999.0
-uesfm,Alcatraz Water Tower,1,1.3394060108035364,0.7384271683550214,6.577617375342355,3.200105455600137,172,3887.47,3865.343374729156,99999.0
-uesfm,Alcatraz Water Tower,2,2.200189184784068,1.3839747978399466,6.854463337738994,3.1730434024886014,172,3819.31,3799.390902757645,99999.0
-esfm,Buddah Tooth Relic Temple Singapore,0,1.3132762212436329,0.31652548993670354,9.903890396062208,4.634952217191846,162,2921.45,2903.059753894806,99999.0
-esfm,Buddah Tooth Relic Temple Singapore,1,0.9908742070528852,0.2335767172234499,9.836436953133868,4.661767906617449,162,2915.39,2904.387546062469,99999.0
-esfm,Buddah Tooth Relic Temple Singapore,2,1.089403690069201,0.2896515999941069,10.151103921611115,4.628183275251743,162,2913.67,2902.928613901138,99999.0
-uesfm,Buddah Tooth Relic Temple Singapore,0,1.9767734026333772,0.5777116480820644,15.584999477939833,8.800407928031486,162,4315.97,4296.739196300507,99999.0
-uesfm,Buddah Tooth Relic Temple Singapore,1,1.4532821126601407,0.345133635718661,16.753182673783428,9.555989756393366,162,4334.28,4311.474883317947,99999.0
-uesfm,Buddah Tooth Relic Temple Singapore,2,32.81105779154638,6.0024233739495125,26.564606536203048,13.925597303414058,162,4312.37,4291.042752504349,99999.0
-esfm,Doge Palace Venice,0,0.9741701978804636,0.2578214102266912,6.511459231891615,3.6947627756511525,241,28795.18,28746.22650837898,99999.0
-esfm,Doge Palace Venice,1,0.627860879835449,0.1773446161890865,6.351890452986879,3.537139014829222,241,28742.27,28699.91308617592,99999.0
-uesfm,Doge Palace Venice,0,0.9110673544913167,0.2904399918743801,10.966183720673449,6.771416746021886,241,21074.95,21017.30448675156,99999.0
-esfm,Door Lund,0,0.01167051570787235,0.003654424593160691,1.8552512563951393,0.9492901559591532,12,4892.41,4878.351888656616,99999.0
-esfm,Door Lund,1,0.009685491454411136,0.0030270923396151107,1.6462990657331118,0.9049495114630693,12,4884.51,4877.529450178146,99999.0
-esfm,Door Lund,2,0.02068285629921104,0.005193423834514985,1.933833706356197,0.9506754982524039,12,4888.17,4880.201772212982,99999.0
-uesfm,Door Lund,0,14.059454220682019,0.9733778175929079,11.85413335315025,6.517535151323047,12,9185.47,8715.343282699585,95000.0
-uesfm,Door Lund,1,14.111993883089163,0.9618908892808385,11.853785941610944,6.504290333453439,12,9203.34,8729.614821910858,95000.0
-uesfm,Door Lund,2,14.344083655704312,0.9193773651816372,12.058851721280973,6.643622749322514,12,9176.97,9162.127201795578,99999.0
-esfm,Drinking Fountain Somewhere In Zurich,0,25.804179845300204,1.1242096722251314,8.219025541657992,3.3464550934365884,14,888.07,874.4874215126038,99999.0
-esfm,Drinking Fountain Somewhere In Zurich,1,25.754577171528883,1.1271567319301725,8.245599860088616,3.352895635561939,14,882.33,873.2415659427643,99999.0
-esfm,Drinking Fountain Somewhere In Zurich,2,0.019604308735801684,0.006780729271172703,1.7090486008047308,0.6375575382268279,14,881.11,873.6504108905792,99999.0
-uesfm,Drinking Fountain Somewhere In Zurich,0,26.27952541668808,1.102546079947844,8.40971803772888,3.4593948661771203,14,2969.96,2950.480526924133,99999.0
-uesfm,Drinking Fountain Somewhere In Zurich,1,26.264849988219943,1.1078406736783482,8.547474098662555,3.585334207185317,14,2960.9,2504.42572927475,85000.0
-uesfm,Drinking Fountain Somewhere In Zurich,2,0.06235835436589852,0.011103715758719332,2.5743283079384534,1.5361135173722633,14,2960.08,2945.673992872238,99999.0
-esfm,East Indiaman Goteborg,0,6.033696331509855,1.0773322362193534,7.427433157074123,2.292729458380311,179,3990.25,3970.732353687286,99999.0
-esfm,East Indiaman Goteborg,1,5.313967690277326,0.9356515362150493,7.350753973659207,2.2747477951684703,179,3986.66,3970.914752960205,99999.0
-esfm,East Indiaman Goteborg,2,5.357164002763642,0.9254557315236661,7.174538078735992,2.2493605774605823,179,3984.64,3967.810528755188,99999.0
-uesfm,East Indiaman Goteborg,0,4.992588850190341,0.9093886386013725,9.930437318497303,4.011819888270754,179,5247.16,5221.816437721252,99999.0
-uesfm,East Indiaman Goteborg,1,5.382092262718057,0.9927766021898647,10.035611563702533,3.912004668740913,179,5245.65,5210.653891563416,99999.0
-uesfm,East Indiaman Goteborg,2,5.554715231405653,0.9752252406138763,9.5551521706325,3.7417116053580193,179,5231.48,5207.815300703049,99999.0
-esfm,Ecole Superior De Guerre,0,0.4863901164687255,0.1239377922325492,3.750666445958877,2.0528113108131016,35,2385.33,2371.708672523499,99999.0
-esfm,Ecole Superior De Guerre,1,0.3031794677170201,0.07921628898965839,3.2456935905644406,1.730123845260616,35,2379.34,2371.191816806793,99999.0
-esfm,Ecole Superior De Guerre,2,0.2580439805296837,0.06772089648290676,3.257951412507941,1.687291979251782,35,2376.49,2368.935952663422,99999.0
-uesfm,Ecole Superior De Guerre,0,0.28243957678843373,0.0778675008788964,5.291358032458062,3.601773151002547,35,4647.51,4622.850021600723,99999.0
-uesfm,Ecole Superior De Guerre,1,0.25507313740145815,0.06695766472508183,5.170495431958795,3.5179325723573998,35,4636.07,4616.890106916428,99999.0
-uesfm,Ecole Superior De Guerre,2,0.3137499276035537,0.08683967313665406,5.583934687507232,3.7813953498055604,35,4617.92,4602.193422079086,99999.0
-esfm,Eglise du dome,0,0.3242410065346163,0.0824797769992069,3.906129661328729,1.3512292775673158,85,25054.74,25028.64922428131,99999.0
-esfm,Eglise du dome,1,0.3771623368290559,0.09432763361451474,3.692490941110161,1.297032357665518,85,25144.16,25116.01952123642,99999.0
-uesfm,Eglise du dome,0,45.800562123362404,6.451383786272971,7.963674120656965,3.986938865913781,85,23371.73,23333.49408817291,99999.0
-esfm,Folke Filbyter,0,89.03873163576341,0.12895560134543632,17.986624161769267,7.423093846189277,40,1582.8,1573.360969305038,99999.0
-esfm,Folke Filbyter,1,84.74740427947457,0.13132264058966686,17.98706393015447,7.385900243615879,40,1579.13,1572.079373121262,99999.0
-esfm,Folke Filbyter,2,83.70817707245973,0.1261878953205589,21.85200129525089,6.324245998829542,40,1583.76,1576.207699537277,99999.0
-uesfm,Folke Filbyter,0,80.34590333370599,0.13169898199385566,22.83327162619912,11.711593158296864,40,3701.49,3129.452286243439,85000.0
-uesfm,Folke Filbyter,1,81.28676564061475,0.1282318052949447,24.480394314414323,9.74969805057316,40,3698.11,3681.891911745071,99999.0
-uesfm,Folke Filbyter,2,82.23270227083374,0.1312531386108397,22.153947807931974,10.961536903594071,40,3709.96,3695.84988451004,99999.0
-esfm,Fort Channing Gate Singapore,0,0.10143646071220622,0.04531774433800129,1.7158776598376824,0.8165178663538488,27,3438.39,3428.288990497589,99999.0
-esfm,Fort Channing Gate Singapore,1,0.08873951454303304,0.039001318240256114,1.783712095722291,0.8156020254191567,27,3430.88,3424.490573883057,99999.0
-esfm,Fort Channing Gate Singapore,2,0.09031224636384508,0.03957289173255022,1.850800887931411,0.820658413919779,27,3433.07,3426.547735452652,99999.0
-uesfm,Fort Channing Gate Singapore,0,33.158343156383296,11.908537566474344,12.69248500308198,5.438758486477886,27,6209.29,6192.107179880142,99999.0
-uesfm,Fort Channing Gate Singapore,1,0.07319798057764343,0.03541606344760984,3.4819579894798927,2.2085534812612853,27,6159.7,6144.797720909119,99999.0
-uesfm,Fort Channing Gate Singapore,2,0.15394831153888047,0.0686769122909015,3.77993377770425,2.2995094868650368,27,6147.27,6132.745970487595,99999.0
-esfm,Golden Statue Somewhere In Hong Kong,0,0.23894829924734365,0.053817571637812096,1.7239352536484445,0.744089758310296,18,4096.74,4087.775670289993,99999.0
-esfm,Golden Statue Somewhere In Hong Kong,1,0.2233990458438394,0.053061028269618456,1.8936878914364637,0.7746436618607616,18,4096.28,4088.994391202927,99999.0
-esfm,Golden Statue Somewhere In Hong Kong,2,0.23397914670516012,0.051137498616263405,1.87070255923672,0.7763877145267694,18,4094.54,4087.674698591232,99999.0
-uesfm,Golden Statue Somewhere In Hong Kong,0,0.36074365888948035,0.06894135246785482,2.939092008443413,1.863811153764833,18,7761.69,7746.569520235062,99999.0
-uesfm,Golden Statue Somewhere In Hong Kong,1,0.35638930493960375,0.06132592684828702,2.686157398440538,1.735571387461727,18,7759.16,7745.389947891235,99999.0
-uesfm,Golden Statue Somewhere In Hong Kong,2,0.3955566726941678,0.09197790539367184,2.878341172309852,1.8164738202745854,18,7766.46,7751.458931922913,99999.0
-esfm,Gustav Vasa,0,3.4881363194229826,0.2273181104934767,1.7369784534725496,0.9521606063342205,18,711.53,703.3091380596161,99999.0
-esfm,Gustav Vasa,1,3.4307883498221035,0.21432466556133864,1.5556448596237022,0.9232184455756751,18,710.28,703.9906163215637,99999.0
-esfm,Gustav Vasa,2,3.6089062562451386,0.2196578287395002,1.673803634770729,0.9429755567757121,18,707.79,701.5903799533844,99999.0
-uesfm,Gustav Vasa,0,5.828154714648938,0.32510821331670736,3.0711042143581646,2.4170166616864357,18,2566.0,2547.931907176971,99999.0
-uesfm,Gustav Vasa,1,6.515239842974278,0.35941242702736925,3.089677100739632,2.4476428398062584,18,2530.9,1385.089058637619,55000.0
-uesfm,Gustav Vasa,2,5.954036423569478,0.34942942401850696,3.101296679577594,2.400483554095863,18,2533.76,2520.798699855804,99999.0
-esfm,GustavIIAdolf,0,46.92758389264633,8.570499881283022,11.684017169857798,5.131125459552879,57,1047.82,1038.018162250519,99999.0
-esfm,GustavIIAdolf,1,47.21999673071185,8.515052769156952,11.695065125063977,5.091617086264161,57,1041.41,1034.5799472332,99999.0
-esfm,GustavIIAdolf,2,47.2312223396789,8.523261295519099,11.787583212607423,5.2715388868550574,57,1043.28,1036.478415489197,99999.0
-uesfm,GustavIIAdolf,0,89.37607436538163,13.276440526172165,15.53883714368065,5.864332719302596,57,2904.42,2881.687849521637,99999.0
-uesfm,GustavIIAdolf,1,70.16367503302509,11.555242253627961,15.162174576474532,6.9709512967490355,57,2884.62,2870.103493452072,99999.0
-uesfm,GustavIIAdolf,2,72.72996778765942,11.054260851523775,16.85269001998736,9.009102977660282,57,2888.7,2874.096453428268,99999.0
-esfm,Jonas Ahlstromer,0,64.46662959289324,11.985605264945464,12.602584968179986,2.9168445332917745,40,614.88,607.3498430252075,99999.0
-esfm,Jonas Ahlstromer,1,38.42510202464649,8.741159740710629,10.25268858530355,3.7024742418884333,40,618.73,611.5567183494568,99999.0
-esfm,Jonas Ahlstromer,2,43.32891670826628,8.741053567623224,10.553031866494822,3.031717304343782,40,612.74,606.6802380084991,99999.0
-uesfm,Jonas Ahlstromer,0,42.30073879858729,10.272232427418913,9.098970988131883,3.992857074878733,40,2339.57,1280.604686737061,55000.0
-uesfm,Jonas Ahlstromer,1,42.18412804385421,10.310808592826955,8.453808908218651,3.4329852879523464,40,2334.33,2320.362742424011,99999.0
-uesfm,Jonas Ahlstromer,2,42.25301394187691,10.315519611163806,8.793789077520445,3.8046125347596442,40,2311.04,2297.139800786972,99999.0
-esfm,Kings College University Of Toronto,0,14.400461796497643,3.2653370282498866,4.064353916685246,1.6753245173381437,77,1074.94,1066.306197404861,99999.0
-esfm,Kings College University Of Toronto,1,11.956699445712928,2.705452710546247,3.9727144140636783,1.5797087908756111,77,1071.64,1064.129351377487,99999.0
-esfm,Kings College University Of Toronto,2,0.7191591237413282,0.16178494376153177,2.656807607821862,1.1949677972638288,77,1090.36,1082.987608671188,99999.0
-uesfm,Kings College University Of Toronto,0,31.932532001930838,2.9775501815475,6.099555590572752,3.8717713150174773,77,2884.23,2866.194714784622,99999.0
-uesfm,Kings College University Of Toronto,1,2.4241467960672063,0.41577835428574456,4.631696476065898,3.148046400399828,77,2899.51,2884.114916801453,99999.0
-uesfm,Kings College University Of Toronto,2,30.212552334890066,2.3209884971628156,5.844709739288878,3.754268841443037,77,2874.99,2860.901805400848,99999.0
-esfm,Lund University Sphinx,0,13.661465370705821,3.5591094267381553,6.242667318159146,2.715620555070548,70,2718.96,2709.5872194767,99999.0
-esfm,Lund University Sphinx,1,12.524302160109885,3.2130942199657664,6.1182420215295075,2.7468468987195953,70,2722.86,2715.36418390274,99999.0
-esfm,Lund University Sphinx,2,7.98961938632192,2.1710959765491817,5.807289242508196,2.893370267054225,70,2713.88,2706.478076219559,99999.0
-uesfm,Lund University Sphinx,0,15.165716414733652,3.5564827651349713,10.803408462752863,6.730422995453065,70,4644.92,4626.135177612305,99999.0
-uesfm,Lund University Sphinx,1,68.88899086055604,13.548881652837478,14.336626262363456,7.626356837439143,70,4703.31,4687.50173997879,99999.0
-uesfm,Lund University Sphinx,2,15.014326075576053,3.5607694164340526,11.039432108003597,6.81081192121135,70,4615.6,4600.294319868088,99999.0
-esfm,Nijo Castle Gate,0,0.6521524412276166,0.13305866403975986,7.404953325705311,3.0629804954987674,19,1097.29,1084.301982402802,99999.0
-esfm,Nijo Castle Gate,1,0.8564043952718835,0.16950374865879153,7.358076475544799,3.1235223975185864,19,1094.89,1084.640900611877,99999.0
-esfm,Nijo Castle Gate,2,1.0089926380330574,0.19095470213426863,7.481049991287358,3.1671779658403687,19,1093.35,1083.446812152863,99999.0
-uesfm,Nijo Castle Gate,0,1.6923297320319277,0.3246009364794658,11.792009654594867,6.945622003724464,19,3310.13,3288.75860619545,99999.0
-uesfm,Nijo Castle Gate,1,1.6697791743236226,0.32440429939345133,12.704681259536075,7.466128811395402,19,3308.19,3290.209539175034,99999.0
-uesfm,Nijo Castle Gate,2,1.682403384345942,0.32817821817561266,11.917977420973372,7.102279949089778,19,3301.77,3284.964460611343,99999.0
-uesfm,Pantheon Paris,0,0.3823220210788846,0.05987691585642601,7.649321203280243,4.797127317249956,179,5506.88,5473.694003582001,99999.0
-uesfm,Pantheon Paris,1,45.61471478909275,6.671068486991568,16.912359249810606,6.602415648206237,179,5491.31,5459.329566717148,99999.0
-uesfm,Pantheon Paris,2,12.03796171625132,2.465102828965073,16.125095834055482,6.860965809143959,179,5493.94,5463.555550813675,99999.0
-esfm,Park Gate Clermont Ferrand,0,19.399216821978534,9.438471236639753,7.521969683469249,3.892136696028138,34,1597.96,1588.07075381279,99999.0
-esfm,Park Gate Clermont Ferrand,1,19.325402297843297,9.417998214523331,7.465007292717891,3.849088919564891,34,1596.87,1589.703137874603,99999.0
-esfm,Park Gate Clermont Ferrand,2,25.440611579781365,11.720641869372841,8.15032188787613,4.0871739024940315,34,1595.99,1588.841923236847,99999.0
-uesfm,Park Gate Clermont Ferrand,0,19.27114964212729,9.381930985797803,7.856513901068924,4.317004881495189,34,3598.69,3580.396904230118,99999.0
-uesfm,Park Gate Clermont Ferrand,1,19.288145181777256,9.394362801959678,7.762082007624792,4.207851936960509,34,3592.66,3577.494926214218,99999.0
-uesfm,Park Gate Clermont Ferrand,2,19.345727373971844,9.415066249707772,7.936652645099164,4.425507919979627,34,3600.86,3581.473034620285,99999.0
-esfm,Plaza De Armas Santiago,0,0.7341628728579594,0.32156050159900956,6.000154308231358,2.679971945741675,240,6375.02,6333.455562829971,99999.0
-esfm,Plaza De Armas Santiago,1,0.5176165701053533,0.22744923951256435,5.661576462867733,2.533952088248649,240,6373.68,6336.030319213867,99999.0
-esfm,Plaza De Armas Santiago,2,1.0740077415291467,0.46304622323619105,6.365716833190761,2.733195456670397,240,6369.35,6331.436369419098,99999.0
-uesfm,Plaza De Armas Santiago,0,6.6772176498607925,2.909933722327364,11.384209379387066,5.623987151554922,240,6540.76,6491.619550704956,99999.0
-uesfm,Plaza De Armas Santiago,1,1.3364898126442175,0.5686650140764532,7.951759184923103,3.933795015892702,240,6536.61,6482.320575714111,99999.0
-uesfm,Plaza De Armas Santiago,2,1.9671483940705004,0.8276654865349118,9.054185015150745,4.441179120460681,240,6529.24,6477.723381996155,99999.0
-esfm,Porta San Donato Bologna,0,0.5746001989174886,0.10608894230214515,5.691906895070585,2.4183460353027986,141,3751.75,3734.281898736954,99999.0
-esfm,Porta San Donato Bologna,1,0.8978003419853328,0.14474404936935045,5.682379581165418,2.563141107750346,141,3758.51,3742.688964605331,99999.0
-esfm,Porta San Donato Bologna,2,0.30440272958620346,0.07026546193502164,5.435311727517131,2.1626312981864086,141,3752.23,3738.056202173233,99999.0
-uesfm,Porta San Donato Bologna,0,1.1269164513732723,0.19275737956525044,10.071225650489282,5.557466375146339,141,5111.57,5083.21807050705,99999.0
-uesfm,Porta San Donato Bologna,1,60.66366912240625,10.234123128225525,19.593360464637342,9.305574874002819,141,5125.57,5098.085618019104,99999.0
-uesfm,Porta San Donato Bologna,2,1.2271289404198877,0.19605254084885107,10.066750816092268,5.595831381287904,141,5116.95,5093.473650932312,99999.0
-esfm,Round Church Cambridge,0,2.1173198662081005,0.9129854085205968,5.854320175654564,1.9683268910531708,92,9788.65,9759.719386577606,99999.0
-esfm,Round Church Cambridge,1,2.1415279636494953,0.9267927094737857,6.33905951937897,2.0439997148640834,92,9764.61,9741.356333732605,99999.0
-uesfm,Round Church Cambridge,0,2.5434984075631974,1.0219048587087352,8.914080922081759,4.131734829440289,92,12088.06,12047.02147889137,99999.0
-uesfm,Round Church Cambridge,1,2.435781054756339,0.9827352719070902,8.178482224933683,3.8400051230290893,92,12034.64,11993.72608184814,99999.0
-esfm,Skansen Kronan Gothenburg,0,0.3380139741078169,0.1158931606908392,3.0478862369293624,1.5340801117570733,131,17163.94,17140.15425729752,99999.0
-uesfm,Skansen Kronan Gothenburg,0,0.37472494768162085,0.1243676312212316,4.798736338080396,3.348916057055834,131,16235.09,16202.21351981163,99999.0
-esfm,Smolny Cathedral St Petersburg,0,21.81278753416543,2.110667559831311,11.2315357584775,6.379259553776089,131,10364.67,10320.51852750778,99999.0
-esfm,Smolny Cathedral St Petersburg,1,21.799828748358873,2.111313741354268,11.225141339870355,6.36605939788532,131,10360.52,10319.01374697685,99999.0
-uesfm,Smolny Cathedral St Petersburg,0,22.143392894578028,2.0985584569799265,12.52645725687386,7.895265841408903,131,10475.74,10416.62969732285,99999.0
-esfm,Sri Mariamman Singapore,0,1.4920493985513708,0.4412389781578158,10.406406306167431,6.369585247957454,222,5987.98,5970.066485881805,99999.0
-esfm,Sri Mariamman Singapore,1,1.1345564396944785,0.3608906720626337,9.703927906328842,5.885963785879842,222,5993.62,5975.365797519684,99999.0
-esfm,Sri Mariamman Singapore,2,1.0293707921546553,0.3426709250774071,10.15220066698994,6.067537729054635,222,5975.02,5958.786058187485,99999.0
-uesfm,Sri Mariamman Singapore,0,16.417883129790784,5.526406595652791,19.962711816743173,12.18830031311293,222,6531.45,6494.750840425491,99999.0
-uesfm,Sri Mariamman Singapore,1,1.3793173305528281,0.5014534676018023,16.10857603648069,10.651562750176396,222,6467.65,6426.201541185379,99999.0
-uesfm,Sri Mariamman Singapore,2,1.639001766282598,0.536397596737868,17.490398022274007,11.439337896248928,222,6457.9,6427.717175960541,99999.0
-esfm,Sri Thendayuthapani Singapore,0,0.7190396606626565,0.17922550285921848,7.12365374281766,2.231392236341273,98,13362.09,13320.09781932831,99999.0
-uesfm,Sri Thendayuthapani Singapore,0,47.0750455982536,3.8553135654690074,14.133068630999212,7.117179633590087,98,13808.44,13754.94554877281,99999.0
-esfm,Sri Veeramakaliamman Singapore,0,2.333869921178247,0.542847254574005,11.965270419451508,5.472877662905037,157,12392.18,12365.15180635452,99999.0
-uesfm,Sri Veeramakaliamman Singapore,0,29.40487246624615,3.8676026992817505,28.950500021302087,14.32295224594287,157,12544.69,12504.26410722733,99999.0
-esfm,Statue Of Liberty,0,52.35559762033152,28.241594953600423,688383.1603142131,683624.423417858,134,102.64,43.32754778862,45000.0
-esfm,Statue Of Liberty,1,52.36920727661949,28.627801879821945,9277.715116667516,9303.546387753544,134,99.78,59.98358988761902,65000.0
-esfm,Statue Of Liberty,2,52.3478568435799,28.64052858422677,7449.598084265948,7416.803834848509,134,97.28,3.520912647247314,0.0
-uesfm,Statue Of Liberty,0,40.92379801166407,19.915794833689585,48.781849500680366,25.904882064330994,134,12927.11,12902.23110127449,99999.0
-uesfm,Statue Of Liberty,1,38.79484508812604,17.090491683338875,33.27459817133403,17.608726936833005,134,12920.89,12896.32496738434,99999.0
-esfm,The Pumpkin,0,27.59459354015253,5.94835892276036,27.52772578222832,9.11126349703041,196,4330.9,4316.109169960022,99999.0
-esfm,The Pumpkin,1,26.54397520858153,5.701234509093555,27.342758332715125,8.907676505669826,196,4327.98,4314.043692827225,99999.0
-esfm,The Pumpkin,2,23.05395646123413,5.007378502393024,25.75879879971585,7.898438464943201,196,4322.94,4310.356798410416,99999.0
-uesfm,The Pumpkin,0,25.490608822831348,5.440568114393512,34.198429037653845,15.6024365790899,196,5793.03,5767.982522964478,99999.0
-uesfm,The Pumpkin,1,27.26556900533122,5.820572144056583,35.6951097814524,16.572200360468877,196,5785.61,5760.521738529205,99999.0
-uesfm,The Pumpkin,2,26.95722616022698,5.73349875876088,36.207441893083136,17.07678356860021,196,5793.15,5768.243654251099,99999.0
-esfm,Thian Hook Keng Temple Singapore,0,1.0653505173922067,0.1079565219850111,15.054152885837562,5.81055046268767,138,3299.26,3281.187238931656,99999.0
-esfm,Thian Hook Keng Temple Singapore,1,0.8390930608109779,0.08272919436528102,14.587041005106803,5.705702663819475,138,3306.93,3283.257929086685,99999.0
-esfm,Thian Hook Keng Temple Singapore,2,0.8760513596629002,0.08764395630143504,15.102705362055175,5.7567965038255124,138,3289.42,3278.173619031906,99999.0
-uesfm,Thian Hook Keng Temple Singapore,0,1.146131162829257,0.10799243740327723,20.24888615229727,10.297158058757331,138,4626.48,4595.79723072052,99999.0
-uesfm,Thian Hook Keng Temple Singapore,1,1.2072956626750737,0.12821580405567637,21.242388560332195,10.655698470690366,138,4600.34,4578.699931621552,99999.0
-uesfm,Thian Hook Keng Temple Singapore,2,1.251054659622132,0.12561972972485672,20.875018460120653,10.416875106878496,138,4610.3,4581.792747020721,99999.0
-esfm,Tsar Nikolai I,0,42.507553768489565,8.869914103947062,10.846690028504929,4.007961025902478,98,4932.15,4920.938139438629,99999.0
-esfm,Tsar Nikolai I,1,43.09835370739786,8.983133733509343,11.402130572143294,4.032588092580996,98,4924.71,4914.929541349411,99999.0
-uesfm,Tsar Nikolai I,0,75.94880014543322,15.146573953653359,15.307363921859029,6.734065570805788,98,6652.35,6629.384099006653,99999.0
-uesfm,Tsar Nikolai I,1,80.87807139992418,15.64102228288136,21.553650044939403,9.179513384124702,98,6641.34,6623.797205209732,99999.0
-esfm,Urban II,0,59.0297968076411,11.008489348760156,17.600594623525765,5.400933781867582,96,8889.69,8878.01452589035,99999.0
-esfm,Vercingetorix,0,79.07478057258302,9.776081681696928,6.649989325889541,1.7432748253588333,69,1223.75,1212.340968608856,99999.0
-esfm,Vercingetorix,1,80.09313371434186,9.858626097407651,6.426263444651325,1.6979463697629655,69,1222.8,1213.763489961624,99999.0
-esfm,Vercingetorix,2,89.68787080902364,10.950615257181747,8.695412468830675,3.360645595149667,69,1225.36,1215.808262825012,99999.0
-uesfm,Vercingetorix,0,78.95942988866727,10.356244187384963,8.07065603849916,3.372297052368834,69,3105.42,3085.208213567734,99999.0
-uesfm,Vercingetorix,1,89.99175121301754,11.178484677344343,8.929255781006072,4.302365177465691,69,3105.88,3091.564247369766,99999.0
-uesfm,Vercingetorix,2,89.31151857112468,11.134780185118505,9.044030033429575,4.351952473195928,69,3105.52,3089.0203332901,99999.0
-esfm,Yueh Hai Ching Temple Singapore,0,0.6268598783684821,0.08360284283022794,5.56223941139456,2.132543748253723,43,1749.28,1738.747783660889,99999.0
+method,scene,seed,rot_err_deg,pos_err,reproj_err_px,reproj_med_px,n_cams,runtime_s,convergence_time_s,best_epoch,rot_err_deg_ba,pos_err_ba,reproj_err_px_ba
+esfm,Alcatraz Courtyard,0,0.324288380204305,0.0854554642858637,3.379232694353053,1.9204072326560708,133,4821.4,4792.707528591156,99999.0,0.04336579548315269,0.01736823028852939,0.8927630229850309
+esfm,Alcatraz Courtyard,1,0.36138678157391557,0.09112912757839253,3.3691035794153503,1.8536696768772378,133,4825.72,4804.594581127167,99999.0,0.04361630830506929,0.01730885636433267,1.11229287420467
+esfm,Alcatraz Courtyard,2,0.4015043649148433,0.10235190268069758,3.557242072642648,1.9198007399315369,133,4833.02,4812.242679834366,99999.0,0.04372553753273131,0.01736657742692317,0.8899897580452283
+esfm_rc,Alcatraz Courtyard,0,0.40021217864007563,0.10366874053275751,3.6525220170666364,2.116170425959351,133,5678.34,5647.516944408417,99999.0,,,
+esfm_rc,Alcatraz Courtyard,1,0.4164699362576677,0.10735288670743451,3.553972406799646,2.0080654095255728,133,5692.91,5662.111888885498,99999.0,,,
+esfm_rc,Alcatraz Courtyard,2,0.443321991451076,0.11848499800783799,3.8114044579296285,2.1404981515498216,133,5681.11,5649.007675886154,99999.0,,,
+uesfm,Alcatraz Courtyard,0,0.5509063563081962,0.14050635120746804,5.528957815039945,3.569220349534855,133,6556.5,6511.41095662117,99999.0,,,
+uesfm,Alcatraz Courtyard,1,0.6823224031499351,0.17277455831381494,5.435931436314427,3.4771386471547348,133,6552.19,6506.419185638428,99999.0,,,
+uesfm,Alcatraz Courtyard,2,29.163200369074893,6.342299350838519,10.849871644564203,6.781661682554276,133,6544.26,6512.667362689972,99999.0,,,
+esfm,Alcatraz Water Tower,0,0.6984432687370988,0.38694796813795407,4.741313233832226,2.1058898944784543,172,2461.59,2440.421297073364,99999.0,0.22768614198983136,0.11526567721949761,0.77419350794262
+esfm,Alcatraz Water Tower,1,0.5944425806062753,0.3287537445614408,4.5656682619915765,1.9703698309044646,172,2467.28,2454.429310798645,99999.0,0.2277265457963764,0.11534572613922207,0.658390729293639
+esfm,Alcatraz Water Tower,2,0.7051984712289041,0.39359395644708545,4.982802041506025,2.1318655747232063,172,2455.14,2441.783402681351,99999.0,0.2264345069130526,0.11472488848384464,0.7908639514877002
+esfm_rc,Alcatraz Water Tower,0,0.9705085738488239,0.5332154449748415,5.147471061152484,2.310649130340069,172,3074.8,3046.602040290833,99999.0,,,
+esfm_rc,Alcatraz Water Tower,1,0.7440758749948073,0.4145867394737888,4.961714044617586,2.211091062168503,172,3061.2,3039.165870904922,99999.0,,,
+esfm_rc,Alcatraz Water Tower,2,0.9302629778601009,0.51517638469428,5.227327015979294,2.2724984979322977,172,3056.55,3035.032587766647,99999.0,,,
+uesfm,Alcatraz Water Tower,0,2.016029335283383,1.087082854315953,7.9502009315303575,4.413488326253207,172,3805.19,3775.908159971237,99999.0,,,
+uesfm,Alcatraz Water Tower,1,1.4259772151091428,0.7645994701417347,7.177871865477131,3.7683369835989504,172,3787.07,3763.27596282959,99999.0,,,
+uesfm,Alcatraz Water Tower,2,0.9982626119141534,0.558403777925872,7.063395425015731,3.814475911180562,172,3795.37,3773.579354286194,99999.0,,,
+esfm,Buddah Tooth Relic Temple Singapore,0,1.3132762212436329,0.31652548993670354,9.903890396062208,4.634952217191846,162,2921.45,2903.059753894806,99999.0,0.08653940053085564,0.016105041532649904,1.9758619604429803
+esfm,Buddah Tooth Relic Temple Singapore,1,0.9908742070528852,0.2335767172234499,9.836436953133868,4.661767906617449,162,2915.39,2904.387546062469,99999.0,0.0849468509455147,0.015653062410540777,1.9626282085900335
+esfm,Buddah Tooth Relic Temple Singapore,2,1.089403690069201,0.2896515999941069,10.151103921611115,4.628183275251743,162,2913.67,2902.928613901138,99999.0,0.08666865718094915,0.016308330450717633,2.2978504476586723
+esfm_rc,Buddah Tooth Relic Temple Singapore,0,1.4156219496055027,0.3849696818920549,10.910671400016733,5.192142953606055,162,3558.33,3528.124534845352,99999.0,,,
+esfm_rc,Buddah Tooth Relic Temple Singapore,1,1.1046287069544334,0.2802093099622587,10.446776494877133,5.042054116563641,162,3542.82,3523.402928352356,99999.0,,,
+esfm_rc,Buddah Tooth Relic Temple Singapore,2,1.2816861979534584,0.34711677570751104,10.873035865566612,5.101037865376439,162,3533.26,3513.161389827728,99999.0,,,
+uesfm,Buddah Tooth Relic Temple Singapore,0,1.5799948547864517,0.43886498334860274,13.484389374830494,7.719072745032001,162,4294.34,4266.538989067078,99999.0,,,
+uesfm,Buddah Tooth Relic Temple Singapore,1,1.934129180062467,0.5924991617860504,13.872322562031941,8.04977435062149,162,4303.62,4259.311170101166,99999.0,,,
+uesfm,Buddah Tooth Relic Temple Singapore,2,1.8591664615796053,0.5279087705483619,14.731758217446671,8.481044153007865,162,4270.55,4249.088565349579,99999.0,,,
+esfm,Doge Palace Venice,0,0.9741701978804636,0.2578214102266912,6.511459231891615,3.6947627756511525,241,28795.18,28746.22650837898,99999.0,0.07156237283091285,0.01915571946233311,1.1580904809936006
+esfm,Doge Palace Venice,1,0.627860879835449,0.1773446161890865,6.351890452986879,3.537139014829222,241,28742.27,28699.91308617592,99999.0,0.050631712294427526,0.017261384802431806,1.1986047342873392
+esfm,Doge Palace Venice,2,0.6576305681966796,0.19114542381679772,6.214841374149415,3.591638795332466,241,10030.76,9989.16938996315,99999.0,0.0477845991264688,0.01734493189317512,1.1742938182004765
+esfm_rc,Doge Palace Venice,0,1.0500985034194232,0.2872157262518388,6.790948674732308,4.010160174021759,241,7952.55,7887.024859905243,99999.0,,,
+esfm_rc,Doge Palace Venice,1,0.822972055147261,0.234503198836285,6.887079459513741,3.9072413922164637,241,7919.49,7863.557105779648,99999.0,,,
+esfm_rc,Doge Palace Venice,2,0.7680084929628118,0.22652177619664943,6.815847430201609,3.913545550189034,241,7919.45,7864.596030473709,99999.0,,,
+uesfm,Doge Palace Venice,0,1.120669822868001,0.3500031006198606,10.496216444068507,6.9730858315176345,241,20767.26,20701.21099281311,99999.0,,,
+esfm,Door Lund,0,0.01167051570787235,0.003654424593160691,1.8552512563951393,0.9492901559591532,12,4892.41,4878.351888656616,99999.0,0.002863516481658609,0.0005677797973481404,0.3034458379531829
+esfm,Door Lund,1,0.009685491454411136,0.0030270923396151107,1.6462990657331118,0.9049495114630693,12,4884.51,4877.529450178146,99999.0,0.0028855670751485583,0.0005696749289261176,0.3034450379552205
+esfm,Door Lund,2,0.02068285629921104,0.005193423834514985,1.933833706356197,0.9506754982524039,12,4888.17,4880.201772212982,99999.0,0.0028793804217640893,0.0005687319198631606,0.30345769639617903
+esfm_rc,Door Lund,0,0.013192395227632247,0.004661037914540053,1.778271540426845,0.9602560994879148,12,8464.92,8445.741333007812,99999.0,,,
+esfm_rc,Door Lund,1,0.019167063815313256,0.0029864726992216563,1.8741769497138485,0.9511493437953005,12,8450.0,8434.659740924835,99999.0,,,
+esfm_rc,Door Lund,2,0.02286084013237112,0.006025105467312505,2.0198271044543015,0.9855463158253012,12,8438.17,8421.555475234985,99999.0,,,
+uesfm,Door Lund,0,0.03084538453455712,0.007522151486969653,2.3711871496098955,1.6703501976219666,12,9262.05,9243.771874427795,99999.0,,,
+uesfm,Door Lund,1,13.889154901234134,1.0160235649641791,11.830787694675347,6.5775252925915195,12,9243.42,7848.296544790268,85000.0,,,
+uesfm,Door Lund,2,0.03345748760425064,0.00845459656629652,2.434197612189203,1.6957836630064358,12,9262.36,9247.514908313751,99999.0,,,
+esfm,Drinking Fountain Somewhere In Zurich,0,25.804179845300204,1.1242096722251314,8.219025541657992,3.3464550934365884,14,888.07,874.4874215126038,99999.0,23.41403219049492,1.200238386449365,7.652263637098719
+esfm,Drinking Fountain Somewhere In Zurich,1,25.754577171528883,1.1271567319301725,8.245599860088616,3.352895635561939,14,882.33,873.2415659427643,99999.0,23.424386659175696,1.1914927737957968,7.588314595112779
+esfm,Drinking Fountain Somewhere In Zurich,2,0.019604308735801684,0.006780729271172703,1.7090486008047308,0.6375575382268279,14,881.11,873.6504108905792,99999.0,0.004928146172176949,0.001084436268196593,0.3140688404104962
+esfm_rc,Drinking Fountain Somewhere In Zurich,0,25.966275912639762,1.1148303838604945,8.295888343149453,3.323889859222449,14,2336.21,2202.958295583725,95000.0,,,
+esfm_rc,Drinking Fountain Somewhere In Zurich,1,25.82082253609568,1.1243203937617456,8.27258982311774,3.337923918211656,14,2332.06,2203.267946481705,95000.0,,,
+esfm_rc,Drinking Fountain Somewhere In Zurich,2,0.02664156912090857,0.008991520744418306,1.817273903936106,0.7075879334430649,14,2330.18,2316.842426300049,99999.0,,,
+uesfm,Drinking Fountain Somewhere In Zurich,0,25.946100100284117,1.1178920745623426,8.471712388928934,3.6241397282821626,14,5817.37,5796.999963760376,99999.0,,,
+uesfm,Drinking Fountain Somewhere In Zurich,1,25.973070279327214,1.1166462082604052,8.538877295313494,3.6343523276029472,14,5821.94,5806.947921991348,99999.0,,,
+uesfm,Drinking Fountain Somewhere In Zurich,2,0.06926036271356541,0.01532826351119577,3.015375672575656,1.7451989053599506,14,5806.35,5792.085475206375,99999.0,,,
+esfm,East Indiaman Goteborg,0,6.033696331509855,1.0773322362193534,7.427433157074123,2.292729458380311,179,3990.25,3970.732353687286,99999.0,5.5436529958372045,1.0684143055593625,2.580925145978092
+esfm,East Indiaman Goteborg,1,5.313967690277326,0.9356515362150493,7.350753973659207,2.2747477951684703,179,3986.66,3970.914752960205,99999.0,5.452634397810775,1.0476183365286567,2.625738084551857
+esfm,East Indiaman Goteborg,2,5.357164002763642,0.9254557315236661,7.174538078735992,2.2493605774605823,179,3984.64,3967.810528755188,99999.0,4.7401879289520705,0.8636100794526042,2.737932350716639
+esfm_rc,East Indiaman Goteborg,0,5.174865473862546,0.9113234827275644,7.487193880318395,2.400937113397191,179,4375.89,4350.625731945038,99999.0,,,
+esfm_rc,East Indiaman Goteborg,1,6.224967804118117,1.1150553113186616,7.636857918099955,2.3906451402388336,179,4385.52,4363.705677032471,99999.0,,,
+esfm_rc,East Indiaman Goteborg,2,5.188808159202813,0.9073521965754248,7.4841004661072486,2.3403976043137757,179,4396.21,4371.83106970787,99999.0,,,
+uesfm,East Indiaman Goteborg,0,5.365772057886591,0.9552115949640507,9.655123223073948,4.174486084237091,179,5203.6,5178.753549814224,99999.0,,,
+uesfm,East Indiaman Goteborg,1,10.45533482189932,2.0687908277631517,9.98932850274456,4.110771460953736,179,5195.98,5174.958552598953,99999.0,,,
+uesfm,East Indiaman Goteborg,2,5.872735570127935,1.0672842053553182,10.745053220477157,4.480528977857714,179,5194.7,5173.540862560272,99999.0,,,
+esfm,Ecole Superior De Guerre,0,0.4863901164687255,0.1239377922325492,3.750666445958877,2.0528113108131016,35,2385.33,2371.708672523499,99999.0,0.013820220558282399,0.002882938121219649,0.35561699442321343
+esfm,Ecole Superior De Guerre,1,0.3031794677170201,0.07921628898965839,3.2456935905644406,1.730123845260616,35,2379.34,2371.191816806793,99999.0,0.013839621308859032,0.002888897351408931,0.35520606627666845
+esfm,Ecole Superior De Guerre,2,0.2580439805296837,0.06772089648290676,3.257951412507941,1.687291979251782,35,2376.49,2368.935952663422,99999.0,0.013741554515737818,0.002868803314661506,0.35531179719423556
+esfm_rc,Ecole Superior De Guerre,0,6.0615095626482445,1.3902441953970521,6.582603890419177,2.5874556376476794,35,3932.24,3914.314348936081,99999.0,,,
+esfm_rc,Ecole Superior De Guerre,1,0.5222448808102489,0.13311780686828037,3.858194363564449,2.0941111590578783,35,3930.97,3916.866501569748,99999.0,,,
+esfm_rc,Ecole Superior De Guerre,2,0.42670768047814917,0.11109013616311524,3.738376401049213,1.9806437286978857,35,3932.83,3915.012129306793,99999.0,,,
+uesfm,Ecole Superior De Guerre,0,51.32681795107141,2.57407526776939,11.751384768771835,6.449293516058521,35,10426.77,10405.3174200058,99999.0,,,
+uesfm,Ecole Superior De Guerre,1,0.2738850931952059,0.0734625834578241,5.5413290518324265,3.89237755756953,35,10405.39,10384.56669402122,99999.0,,,
+esfm,Eglise du dome,0,0.3242410065346163,0.0824797769992069,3.906129661328729,1.3512292775673158,85,25054.74,25028.64922428131,99999.0,0.03400419508287424,0.00968361713071315,2.1529226593308155
+esfm,Eglise du dome,1,0.3771623368290559,0.09432763361451474,3.692490941110161,1.297032357665518,85,25144.16,25116.01952123642,99999.0,0.03329023612659896,0.009627024817575195,1.843711309152907
+esfm,Eglise du dome,2,0.45623311880194967,0.11755570327842058,4.363674496823582,1.384257241090686,85,8694.98,8671.502047538757,99999.0,0.03489358171094041,0.010000209116698642,2.2467882207119256
+esfm_rc,Eglise du dome,0,0.5557206668193069,0.14201663162258846,4.387076179747346,1.5005097595973045,85,8784.41,8746.477850675583,99999.0,,,
+esfm_rc,Eglise du dome,1,0.5674000958153612,0.14488795138247493,4.186122825560066,1.4261565747278528,85,8806.29,8769.514946460724,99999.0,,,
+esfm_rc,Eglise du dome,2,0.5841919346497452,0.15295064836207942,4.177169486298827,1.4755743828038193,85,8796.97,8757.282470941544,99999.0,,,
+uesfm,Eglise du dome,0,0.8528700994829498,0.23193079249562099,5.944108200898114,2.9306630821877517,85,9803.94,9764.585114717484,99999.0,,,
+uesfm,Eglise du dome,1,0.6936386490465627,0.1932815436676001,6.091509620719386,2.9784748994578076,85,9788.95,9753.742706537247,99999.0,,,
+uesfm,Eglise du dome,2,45.4797063436542,6.439471230615876,8.408073658981259,4.497252326534489,85,9815.16,9781.022309541702,99999.0,,,
+esfm,Folke Filbyter,0,89.03873163576341,0.12895560134543632,17.986624161769267,7.423093846189277,40,1582.8,1573.360969305038,99999.0,75.51315528045146,0.13175161206780092,20.394291737162312
+esfm,Folke Filbyter,1,84.74740427947457,0.13132264058966686,17.98706393015447,7.385900243615879,40,1579.13,1572.079373121262,99999.0,70.97231562213899,0.12890045131878497,18.182171069573492
+esfm,Folke Filbyter,2,83.70817707245973,0.1261878953205589,21.85200129525089,6.324245998829542,40,1583.76,1576.207699537277,99999.0,67.83491986963047,0.12837305355978038,21.46453761850234
+esfm_rc,Folke Filbyter,0,89.37676108904198,0.12860403401017323,18.22099901069209,7.74644308432037,40,3082.48,3064.089830160141,99999.0,,,
+esfm_rc,Folke Filbyter,1,85.29802780650462,0.1312655372498663,18.31496063485457,7.58936290577374,40,3067.08,3052.338712692261,99999.0,,,
+esfm_rc,Folke Filbyter,2,83.0243809466643,0.12750491742097297,23.157774763962923,7.009627030137372,40,3069.05,3053.145490646362,99999.0,,,
+uesfm,Folke Filbyter,0,84.55526303793802,0.13091275274611416,29.190267154194732,11.160073225547443,40,3747.65,3727.990335702896,99999.0,,,
+uesfm,Folke Filbyter,1,84.87047537526948,0.13112291273275417,32.36503566471496,13.743964264262598,40,3743.53,3728.497133255005,99999.0,,,
+uesfm,Folke Filbyter,2,78.54375401873024,0.1316690349103016,23.11854509839131,10.94439865744236,40,3774.4,3007.95537352562,80000.0,,,
+esfm,Fort Channing Gate Singapore,0,0.10143646071220622,0.04531774433800129,1.7158776598376824,0.8165178663538488,27,3438.39,3428.288990497589,99999.0,0.01831129170730475,0.007384689295328118,0.2597632094033597
+esfm,Fort Channing Gate Singapore,1,0.08873951454303304,0.039001318240256114,1.783712095722291,0.8156020254191567,27,3430.88,3424.490573883057,99999.0,0.018321633248369797,0.007389903495856605,0.2597556139620049
+esfm,Fort Channing Gate Singapore,2,0.09031224636384508,0.03957289173255022,1.850800887931411,0.820658413919779,27,3433.07,3426.547735452652,99999.0,0.018353600365247733,0.0074018125678656016,0.25972738760938635
+esfm_rc,Fort Channing Gate Singapore,0,0.12919721972508352,0.05643946161236995,1.8665780418242388,0.9262975062395102,27,5450.14,5430.581563711166,99999.0,,,
+esfm_rc,Fort Channing Gate Singapore,1,0.0961175836225673,0.04352638473532752,1.7634485231989456,0.8832348541778796,27,5476.97,5456.584360361099,99999.0,,,
+esfm_rc,Fort Channing Gate Singapore,2,0.1269205407593247,0.05699742664239027,1.9815335382658048,0.9205728692828943,27,5423.49,5403.335438489914,99999.0,,,
+uesfm,Fort Channing Gate Singapore,0,0.10214634402234797,0.048923384893640356,3.7907267588398517,2.449846070018404,27,6175.88,6156.048281908035,99999.0,,,
+uesfm,Fort Channing Gate Singapore,1,0.14219758622796153,0.06357969294387673,3.5066994916782015,2.1555120267718575,27,6191.97,6176.711992740631,99999.0,,,
+uesfm,Fort Channing Gate Singapore,2,0.217326908828172,0.10047378918560818,3.8470573886147417,2.340537393844086,27,6145.9,6130.889567136765,99999.0,,,
+esfm,Golden Statue Somewhere In Hong Kong,0,0.23894829924734365,0.053817571637812096,1.7239352536484445,0.744089758310296,18,4096.74,4087.775670289993,99999.0,0.026250976813289254,0.003462321884642883,0.2822531708961522
+esfm,Golden Statue Somewhere In Hong Kong,1,0.2233990458438394,0.053061028269618456,1.8936878914364637,0.7746436618607616,18,4096.28,4088.994391202927,99999.0,0.026089105914074914,0.003445687328266814,0.2822884560119062
+esfm,Golden Statue Somewhere In Hong Kong,2,0.23397914670516012,0.051137498616263405,1.87070255923672,0.7763877145267694,18,4094.54,4087.674698591232,99999.0,0.026019507414979055,0.0034393367985468748,0.28218109003255504
+esfm_rc,Golden Statue Somewhere In Hong Kong,0,0.23746483548049216,0.052245486770656324,1.8202139930821122,0.7929927966477024,18,7039.12,7017.501612186432,99999.0,,,
+esfm_rc,Golden Statue Somewhere In Hong Kong,1,0.25835446587605726,0.061292299836508156,1.9316593984355397,0.8363119479886987,18,7014.04,6999.158929586411,99999.0,,,
+esfm_rc,Golden Statue Somewhere In Hong Kong,2,0.3060791978287713,0.07361327882523418,2.0787911799501324,0.8493789100421191,18,7119.11,7103.271198987961,99999.0,,,
+uesfm,Golden Statue Somewhere In Hong Kong,0,0.6244166607301816,0.09872331853471474,3.161519043143039,2.067073705499029,18,19646.03,19620.43561792374,99999.0,,,
+uesfm,Golden Statue Somewhere In Hong Kong,1,0.267468385274811,0.060691033227620865,2.8287683223467064,1.8282220692313509,18,7727.72,7710.336065292358,99999.0,,,
+uesfm,Golden Statue Somewhere In Hong Kong,2,0.6315484707179251,0.08671732527042503,3.075728070835161,1.9442441573584643,18,7724.17,7706.132658720016,99999.0,,,
+esfm,Gustav Vasa,0,3.4881363194229826,0.2273181104934767,1.7369784534725496,0.9521606063342205,18,711.53,703.3091380596161,99999.0,0.9784891824482792,0.10778420441012125,0.5210309534376089
+esfm,Gustav Vasa,1,3.4307883498221035,0.21432466556133864,1.5556448596237022,0.9232184455756751,18,710.28,703.9906163215637,99999.0,0.9831979134139665,0.10803454074848035,0.5250933657575036
+esfm,Gustav Vasa,2,3.6089062562451386,0.2196578287395002,1.673803634770729,0.9429755567757121,18,707.79,701.5903799533844,99999.0,1.0495249332227277,0.12320327825606991,0.5686977966530726
+esfm_rc,Gustav Vasa,0,4.479928423821196,0.26813796016443026,1.9801034973174125,1.1078604015859557,18,1944.19,1905.260601043701,99999.0,,,
+esfm_rc,Gustav Vasa,1,3.6300123089239613,0.2270854488008436,1.7892038194948103,1.0620796169846443,18,1918.69,1900.605086088181,99999.0,,,
+esfm_rc,Gustav Vasa,2,4.256624192410625,0.25269996179819426,1.8539674990623372,1.0610032766895823,18,1923.76,1897.99094748497,99999.0,,,
+uesfm,Gustav Vasa,0,36.645536646380805,1.0070193571286277,4.980488290477233,3.197977230260218,18,2562.94,2419.297919034958,95000.0,,,
+uesfm,Gustav Vasa,1,36.4914104514929,1.03254185606701,5.377024289258223,3.4818421684593366,18,4566.81,4543.375423431396,99999.0,,,
+uesfm,Gustav Vasa,2,4.4238740734834625,0.284762443333143,3.8820183248893803,2.8663959682636566,18,4533.21,4516.61039352417,99999.0,,,
+esfm,GustavIIAdolf,0,46.92758389264633,8.570499881283022,11.684017169857798,5.131125459552879,57,1047.82,1038.018162250519,99999.0,24.71817789017971,5.446736853565307,16.261776511982404
+esfm,GustavIIAdolf,1,47.21999673071185,8.515052769156952,11.695065125063977,5.091617086264161,57,1041.41,1034.5799472332,99999.0,24.783445519335448,5.485252890255178,16.122725952233722
+esfm,GustavIIAdolf,2,47.2312223396789,8.523261295519099,11.787583212607423,5.2715388868550574,57,1043.28,1036.478415489197,99999.0,24.71364130473506,5.447948021143418,16.671847441164118
+esfm_rc,GustavIIAdolf,0,47.35567772792213,8.62293127729445,11.946761161586606,5.4045343866458575,57,2284.75,2245.40932559967,99999.0,,,
+esfm_rc,GustavIIAdolf,1,47.65353986419624,8.589683791795656,11.844361575387735,5.303722447527598,57,2258.26,2237.952227830887,99999.0,,,
+esfm_rc,GustavIIAdolf,2,47.396282103638164,8.509630444500928,11.963746746326924,5.434993473177883,57,2252.6,2235.428824663162,99999.0,,,
+uesfm,GustavIIAdolf,0,85.51179144359693,12.792264407383987,17.018024925092185,8.196266635685827,57,2973.35,2956.916826248169,99999.0,,,
+uesfm,GustavIIAdolf,1,88.9494736685426,13.260723460498589,17.699982689478404,8.888010081099381,57,2995.3,2976.055320978165,99999.0,,,
+uesfm,GustavIIAdolf,2,90.12096202318821,13.335169188185864,16.941538974035925,7.772181288890227,57,2974.4,2960.110015630722,99999.0,,,
+esfm,Jonas Ahlstromer,0,64.46662959289324,11.985605264945464,12.602584968179986,2.9168445332917745,40,614.88,607.3498430252075,99999.0,56.233065938789125,10.435407172303073,11.885955230403729
+esfm,Jonas Ahlstromer,1,38.42510202464649,8.741159740710629,10.25268858530355,3.7024742418884333,40,618.73,611.5567183494568,99999.0,0.03712241399969036,0.010988262348231177,0.22755727926611335
+esfm,Jonas Ahlstromer,2,43.32891670826628,8.741053567623224,10.553031866494822,3.031717304343782,40,612.74,606.6802380084991,99999.0,42.27144254755807,8.48578501886873,10.391663840228917
+esfm_rc,Jonas Ahlstromer,0,63.4390388558475,11.542340942042635,12.704406945971337,3.2390607659681296,40,1732.58,1605.343009233475,95000.0,,,
+esfm_rc,Jonas Ahlstromer,1,38.177570703294535,8.672488375453444,10.465133389728726,3.83336868366082,40,1706.01,1687.714308977127,99999.0,,,
+esfm_rc,Jonas Ahlstromer,2,43.62862719300082,8.8014510569871,10.815011628378553,3.3581265722454883,40,1699.67,1682.403540372849,99999.0,,,
+uesfm,Jonas Ahlstromer,0,45.264596178655594,9.403212752141883,11.336617283269852,4.728596140494824,40,2421.11,2401.221394062042,99999.0,,,
+uesfm,Jonas Ahlstromer,1,51.490398556789884,10.860548616493958,12.821407884052217,5.012487046810724,40,2428.25,2415.139223337173,99999.0,,,
+uesfm,Jonas Ahlstromer,2,42.99784237868634,10.470910789899067,9.274184391894982,4.4612757046777345,40,2394.6,2381.585824012756,99999.0,,,
+esfm,Kings College University Of Toronto,0,14.400461796497643,3.2653370282498866,4.064353916685246,1.6753245173381437,77,1074.94,1066.306197404861,99999.0,9.635756937410923,2.001374412854319,2.4237261655963693
+esfm,Kings College University Of Toronto,1,11.956699445712928,2.705452710546247,3.9727144140636783,1.5797087908756111,77,1071.64,1064.129351377487,99999.0,7.641765083108614,1.6563433329963133,2.511022238246807
+esfm,Kings College University Of Toronto,2,0.7191591237413282,0.16178494376153177,2.656807607821862,1.1949677972638288,77,1090.36,1082.987608671188,99999.0,0.08065822304086778,0.013824042665226435,0.3683414375557499
+esfm_rc,Kings College University Of Toronto,0,14.749850754568351,3.390834150797208,4.273460375035421,1.881809737201061,77,2235.19,2086.953983068466,95000.0,,,
+esfm_rc,Kings College University Of Toronto,1,12.24434343895146,2.727340976741191,4.101496648756234,1.7269690990930942,77,2196.25,2068.98152923584,95000.0,,,
+esfm_rc,Kings College University Of Toronto,2,0.9409038453179062,0.20480434027922273,2.819943864062075,1.3255016011048806,77,2197.64,2180.024880647659,99999.0,,,
+uesfm,Kings College University Of Toronto,0,29.200261744017546,2.0893308856114428,7.037455848998898,4.8227457854711675,77,2856.72,2838.923613071442,99999.0,,,
+uesfm,Kings College University Of Toronto,1,7.107988087620218,1.3235014088907995,5.965363878755208,4.179664786063016,77,2852.71,2839.42994761467,99999.0,,,
+uesfm,Kings College University Of Toronto,2,6.543121223644844,1.5041055210367995,6.0501544660211914,4.127143945227024,77,2863.77,2850.061238765717,99999.0,,,
+esfm,Lund University Sphinx,0,13.661465370705821,3.5591094267381553,6.242667318159146,2.715620555070548,70,2718.96,2709.5872194767,99999.0,8.460013856708239,2.11467464964552,1.4565174740087705
+esfm,Lund University Sphinx,1,12.524302160109885,3.2130942199657664,6.1182420215295075,2.7468468987195953,70,2722.86,2715.36418390274,99999.0,8.12353912383344,2.074906927104935,1.325091071719341
+esfm,Lund University Sphinx,2,7.98961938632192,2.1710959765491817,5.807289242508196,2.893370267054225,70,2713.88,2706.478076219559,99999.0,4.449918670952723,1.2322626694001688,0.8213585400994716
+esfm_rc,Lund University Sphinx,0,16.25886053825766,4.102962816505069,6.869889153124891,3.100467608594146,70,3942.66,3901.573514938354,99999.0,,,
+esfm_rc,Lund University Sphinx,1,13.156269680075315,3.4132660053719683,6.667310697819336,3.1453748872389977,70,3914.43,3893.653036355972,99999.0,,,
+esfm_rc,Lund University Sphinx,2,9.604992163703725,2.6373578340271524,6.742167958027671,3.346047584137487,70,3916.03,3895.086451530457,99999.0,,,
+uesfm,Lund University Sphinx,0,42.95863056778373,8.719684065980621,14.999945982069319,8.361965506089375,70,4705.61,4681.787243127823,99999.0,,,
+uesfm,Lund University Sphinx,1,14.202013501375653,3.5150197050401064,10.824261355110268,6.665393527268384,70,4616.97,4592.001487731934,99999.0,,,
+uesfm,Lund University Sphinx,2,7.993187283055365,2.2173389310010068,8.827116984475035,5.573703948682402,70,4605.39,4587.484831571579,99999.0,,,
+esfm,Nijo Castle Gate,0,0.6521524412276166,0.13305866403975986,7.404953325705311,3.0629804954987674,19,1097.29,1084.301982402802,99999.0,0.08036972152423151,0.012796484603062593,0.7615804699553756
+esfm,Nijo Castle Gate,1,0.8564043952718835,0.16950374865879153,7.358076475544799,3.1235223975185864,19,1094.89,1084.640900611877,99999.0,0.08192404770421978,0.013101136558935796,0.760865657995869
+esfm,Nijo Castle Gate,2,1.0089926380330574,0.19095470213426863,7.481049991287358,3.1671779658403687,19,1093.35,1083.446812152863,99999.0,0.08102331686497327,0.01291191910447522,0.7612254771474661
+esfm_rc,Nijo Castle Gate,0,1.1608620390074227,0.22835706972308775,8.292447452530368,3.6544156383512774,19,2653.78,2485.451326608658,95000.0,,,
+esfm_rc,Nijo Castle Gate,1,1.4421159803060983,0.2748016105660959,8.250799662340748,3.7148544612954977,19,2619.96,2603.008899211884,99999.0,,,
+esfm_rc,Nijo Castle Gate,2,1.3714350481742656,0.2608522893427154,8.124524832882802,3.5384916107814472,19,2614.73,2598.148197889328,99999.0,,,
+uesfm,Nijo Castle Gate,0,8.641303631787597,2.791258638759238,20.96738948203601,10.297911838853688,19,3241.33,3220.324777603149,99999.0,,,
+uesfm,Nijo Castle Gate,1,1.3055705335697472,0.2681929407818948,11.680266119036336,6.962610820698007,19,3234.66,3216.868148088455,99999.0,,,
+uesfm,Nijo Castle Gate,2,1.8310744216680852,0.3558493818160386,12.75712078857825,7.906743075262264,19,3227.08,3213.229565620422,99999.0,,,
+esfm_rc,Pantheon Paris,0,0.21324226532150278,0.0556382184158685,4.83107981482966,3.002373021388077,179,4653.95,4598.05153131485,99999.0,,,
+esfm_rc,Pantheon Paris,1,0.212491875340622,0.05529075860063325,4.8846908181959074,2.9588861166627645,179,4628.46,4593.19188284874,99999.0,,,
+esfm_rc,Pantheon Paris,2,0.3340659445994911,0.05750962417018217,4.964330493411554,2.9691476134237185,179,4639.41,4604.334435939789,99999.0,,,
+uesfm,Pantheon Paris,0,0.5453004964472504,0.06999784414241637,7.790870638815493,5.057509442480535,179,5594.6,5562.967379331589,99999.0,,,
+uesfm,Pantheon Paris,1,0.472097972346089,0.062163803587728925,7.194706475909503,4.639524309087925,179,5536.74,5501.721322774887,99999.0,,,
+uesfm,Pantheon Paris,2,44.569134625905356,6.741655820531256,16.92751887234327,6.718188212415848,179,5541.46,5510.693382978439,99999.0,,,
+esfm,Park Gate Clermont Ferrand,0,19.399216821978534,9.438471236639753,7.521969683469249,3.892136696028138,34,1597.96,1588.07075381279,99999.0,7.005947814792836,2.8784023682563795,14.489115644047233
+esfm,Park Gate Clermont Ferrand,1,19.325402297843297,9.417998214523331,7.465007292717891,3.849088919564891,34,1596.87,1589.703137874603,99999.0,7.04585897473849,2.8955291629401523,14.422988510386801
+esfm,Park Gate Clermont Ferrand,2,25.440611579781365,11.720641869372841,8.15032188787613,4.0871739024940315,34,1595.99,1588.841923236847,99999.0,16.730405105853063,7.240660871176633,13.929099740909901
+esfm_rc,Park Gate Clermont Ferrand,0,19.41015735759728,9.44545450483485,7.5655698947009755,3.9376102285883072,34,2969.64,2931.905954837799,99999.0,,,
+esfm_rc,Park Gate Clermont Ferrand,1,19.33326285032967,9.425195186043549,7.520162585346567,3.890911800001084,34,2946.52,2928.374010801315,99999.0,,,
+esfm_rc,Park Gate Clermont Ferrand,2,25.18721114681958,11.659048612833406,8.156533346326963,4.159109875326399,34,2941.72,2923.031396150589,99999.0,,,
+uesfm,Park Gate Clermont Ferrand,0,26.45121972215746,12.46688671568752,9.374365736774523,5.572822711026232,34,3659.9,3637.290126085281,99999.0,,,
+uesfm,Park Gate Clermont Ferrand,1,25.126789970267566,11.629146345704141,8.445284973863814,4.644358918969844,34,3647.69,3633.016668319702,99999.0,,,
+uesfm,Park Gate Clermont Ferrand,2,19.34531670045473,9.424313535878454,8.137652552146374,4.753910741641504,34,3642.92,3629.229806900024,99999.0,,,
+esfm,Plaza De Armas Santiago,0,0.7341628728579594,0.32156050159900956,6.000154308231358,2.679971945741675,240,6375.02,6333.455562829971,99999.0,0.13274758032014072,0.05309931531294862,2.606822533335194
+esfm,Plaza De Armas Santiago,1,0.5176165701053533,0.22744923951256435,5.661576462867733,2.533952088248649,240,6373.68,6336.030319213867,99999.0,0.13819907741152881,0.05533560554064368,2.759268616966368
+esfm,Plaza De Armas Santiago,2,1.0740077415291467,0.46304622323619105,6.365716833190761,2.733195456670397,240,6369.35,6331.436369419098,99999.0,0.13512690139003744,0.05415457227659273,3.4062212958788285
+esfm_rc,Plaza De Armas Santiago,0,1.0741847479970412,0.4714298140713374,6.407094111087262,2.8609073397853417,240,5662.18,5611.049166440964,99999.0,,,
+esfm_rc,Plaza De Armas Santiago,1,0.8428560730717238,0.3711994064496868,5.976446140682491,2.748210015625399,240,5659.43,5609.024231433868,99999.0,,,
+esfm_rc,Plaza De Armas Santiago,2,1.4478823177517481,0.6177865944942146,6.59706564230341,2.915296192751213,240,5714.62,5664.147575616837,99999.0,,,
+uesfm,Plaza De Armas Santiago,0,6.703928231459959,2.9117331629179497,11.237000025698872,5.853835350910829,240,6616.75,6560.115643978119,99999.0,,,
+uesfm,Plaza De Armas Santiago,1,6.276886659110257,2.7457580388752243,10.702703105569906,5.585481192919806,240,6612.47,6560.248650550842,99999.0,,,
+uesfm,Plaza De Armas Santiago,2,6.731807209592312,2.8977064717650864,11.656685600371235,6.070719216176278,240,6621.69,6564.684820175171,99999.0,,,
+esfm,Porta San Donato Bologna,0,0.5746001989174886,0.10608894230214515,5.691906895070585,2.4183460353027986,141,3751.75,3734.281898736954,99999.0,0.10991271062835306,0.05551366141500488,4.218662994987151
+esfm,Porta San Donato Bologna,1,0.8978003419853328,0.14474404936935045,5.682379581165418,2.563141107750346,141,3758.51,3742.688964605331,99999.0,0.1128666236829697,0.05602911579225738,2.211912679347602
+esfm,Porta San Donato Bologna,2,0.30440272958620346,0.07026546193502164,5.435311727517131,2.1626312981864086,141,3752.23,3738.056202173233,99999.0,0.11448509746230341,0.056754942188861325,3.2403286864511167
+esfm_rc,Porta San Donato Bologna,0,0.4080484041025974,0.07831590997213768,5.6144822077580585,2.3712185771771375,141,4331.14,4297.81409239769,99999.0,,,
+esfm_rc,Porta San Donato Bologna,1,0.4654680308021159,0.09236185451721962,5.643652792745585,2.4272418315858557,141,4309.37,4283.085269212723,99999.0,,,
+esfm_rc,Porta San Donato Bologna,2,0.9559717082151379,0.1643078286741575,6.078014738464724,2.61821191294824,141,4313.36,4286.725072145462,99999.0,,,
+uesfm,Porta San Donato Bologna,0,4.020291451689462,1.1214786148434206,10.933993245401652,5.19237663800495,141,5137.62,5110.554834842682,99999.0,,,
+uesfm,Porta San Donato Bologna,1,0.7088473947757895,0.1505922603123855,8.617060379525427,4.684345536783661,141,5152.53,5128.584963560104,99999.0,,,
+uesfm,Porta San Donato Bologna,2,0.7136264396886388,0.1325505085909014,7.953155791163967,4.201659432858777,141,5212.46,5175.418401956558,99999.0,,,
+esfm,Round Church Cambridge,0,2.1173198662081005,0.9129854085205968,5.854320175654564,1.9683268910531708,92,9788.65,9759.719386577606,99999.0,0.9342212598130449,0.5515227006059547,4.054067325360582
+esfm,Round Church Cambridge,1,2.1415279636494953,0.9267927094737857,6.33905951937897,2.0439997148640834,92,9764.61,9741.356333732605,99999.0,1.115336663170846,0.5722632781970113,5.541826341161163
+esfm,Round Church Cambridge,2,2.1507951248804917,0.9401429799233081,6.115230559713554,2.0517209910251797,92,9784.55,9763.525354623795,99999.0,0.9104797992475266,0.5364933276394585,5.035679124124719
+esfm_rc,Round Church Cambridge,0,2.2826201708417098,0.9657710484746329,6.16310348787557,2.185542741510192,92,10880.38,10823.98271656036,99999.0,,,
+esfm_rc,Round Church Cambridge,1,2.2206190556774503,0.9437039356840937,6.581626868211357,2.1839542860976264,92,10884.45,10840.10469603539,99999.0,,,
+esfm_rc,Round Church Cambridge,2,2.3743814339345155,0.9912916631176563,6.734437930520745,2.258930633937234,92,10886.78,10842.68440675735,99999.0,,,
+uesfm,Round Church Cambridge,0,1.1275240742139254,0.32924483901066126,7.8063197416312855,3.992682305020409,92,12183.97,12142.68090772629,99999.0,,,
+uesfm,Round Church Cambridge,1,2.530477427180247,0.972585197332408,8.18733294883617,4.015376648566849,92,12114.44,12080.06272149086,99999.0,,,
+uesfm,Round Church Cambridge,2,2.6330412756122525,1.0135214760737334,8.856154239831417,4.349376095653312,92,12091.11,12053.28948831558,99999.0,,,
+esfm,Skansen Kronan Gothenburg,0,0.3380139741078169,0.1158931606908392,3.0478862369293624,1.5340801117570733,131,17163.94,17140.15425729752,99999.0,0.025387020977857267,0.008390321550547433,0.6788480725266395
+esfm,Skansen Kronan Gothenburg,1,0.21516831826827018,0.07458182746108981,2.786705021535272,1.4619994473169777,131,17129.62,17109.18022465706,99999.0,0.025231970575166494,0.008360587055798304,0.6833298341122367
+esfm,Skansen Kronan Gothenburg,2,0.3508356515622903,0.11667319277987132,2.8836839507901773,1.5216584205042176,131,17167.47,17142.66050887108,99999.0,0.025087644890282812,0.008293159108503236,0.6831238442687219
+esfm_rc,Skansen Kronan Gothenburg,0,0.2958520465895253,0.10029016058373366,3.0067714626699646,1.6085514748163654,131,6130.39,6091.164911746979,99999.0,,,
+esfm_rc,Skansen Kronan Gothenburg,1,0.2916975925764745,0.09931739923314177,3.007556973630657,1.5796531296605096,131,6127.58,6094.12020111084,99999.0,,,
+esfm_rc,Skansen Kronan Gothenburg,2,0.34947792833332825,0.11898136948566021,3.0445611018681613,1.6310446575058488,131,6125.12,6091.680647850037,99999.0,,,
+uesfm,Skansen Kronan Gothenburg,0,0.38108224181004346,0.13301438128895593,4.623321785995157,3.2918868818818052,131,16316.64,16276.98657560349,99999.0,,,
+uesfm,Skansen Kronan Gothenburg,1,0.37345373662160575,0.12470260098166054,4.764112581596325,3.4119305236076505,131,7073.79,7040.064019680023,99999.0,,,
+uesfm,Skansen Kronan Gothenburg,2,0.2703573639284502,0.0962556036770187,4.872246482920416,3.4297600302558404,131,7079.8,7048.02818608284,99999.0,,,
+esfm,Smolny Cathedral St Petersburg,0,21.81278753416543,2.110667559831311,11.2315357584775,6.379259553776089,131,10364.67,10320.51852750778,99999.0,21.423862086052477,2.1555190029458697,10.301353476554675
+esfm,Smolny Cathedral St Petersburg,1,21.799828748358873,2.111313741354268,11.225141339870355,6.36605939788532,131,10360.52,10319.01374697685,99999.0,21.406459319996742,2.160222344201535,10.317138393648838
+esfm,Smolny Cathedral St Petersburg,2,21.71090600320656,2.110307413725059,11.259799763291888,6.4136596726420905,131,10363.17,10320.08415436745,99999.0,21.419154026165344,2.157488960966909,10.311901836556512
+esfm_rc,Smolny Cathedral St Petersburg,0,21.82091176814388,2.109838123236979,11.34747421311657,6.5257275032038065,131,9392.23,9330.660197257996,99999.0,,,
+esfm_rc,Smolny Cathedral St Petersburg,1,21.8691455338683,2.106636325604001,11.385698170334294,6.528914486700552,131,9365.77,9312.483961582184,99999.0,,,
+esfm_rc,Smolny Cathedral St Petersburg,2,21.852976622993765,2.104890865572913,11.527275577228112,6.60848088276277,131,9396.26,9340.038203716278,99999.0,,,
+uesfm,Smolny Cathedral St Petersburg,0,22.2854186118672,2.094744890313659,12.901706098015099,8.330992137880555,131,10444.11,10386.77505373955,99999.0,,,
+uesfm,Smolny Cathedral St Petersburg,1,22.12557249893661,2.0983352220394447,12.972911403502199,8.373490866652297,131,10440.89,10388.390884161,99999.0,,,
+uesfm,Smolny Cathedral St Petersburg,2,22.3108596899731,2.0940347709664024,12.978138438291575,8.138999834900323,131,10450.27,10395.72995519638,99999.0,,,
+esfm,Some Cathedral In Barcelona,0,0.6775058368464424,0.24064262464083658,6.20165095815897,3.7253636674062887,177,5002.81,4985.274734258652,99999.0,0.03317243582351888,0.013609437166710361,0.9277887218183031
+esfm,Some Cathedral In Barcelona,1,0.6722147495861831,0.2423973792150467,6.350786792749083,3.720536755931007,177,5009.15,4991.552013158798,99999.0,0.033053432001706344,0.013569847698698592,1.4229226408233036
+esfm,Some Cathedral In Barcelona,2,0.6046048207068493,0.21699252142655406,6.261075100347952,3.7258171467923633,177,4993.96,4975.845166683197,99999.0,0.03281389435149025,0.013509575767387335,0.9486644088357065
+esfm_rc,Some Cathedral In Barcelona,0,0.8749233797042759,0.3145000897825549,6.881244292669395,4.213158668265296,177,4973.63,4937.959746360779,99999.0,,,
+esfm_rc,Some Cathedral In Barcelona,1,0.7676684854847832,0.2755901489197553,6.761455876131694,4.075310939831256,177,4959.11,4927.540645837784,99999.0,,,
+esfm_rc,Some Cathedral In Barcelona,2,0.71858178037652,0.25641289780654003,6.744656202748352,4.080713738243981,177,4952.16,4923.896674156189,99999.0,,,
+uesfm,Some Cathedral In Barcelona,0,3.68833616720384,1.476348113803727,14.728491750889885,9.144995894634686,177,5865.45,5836.537386417389,99999.0,,,
+uesfm,Some Cathedral In Barcelona,1,1.3699265043959112,0.4939901267450982,10.514669959569153,7.150937287439337,177,5931.0,5904.003969430923,99999.0,,,
+uesfm,Some Cathedral In Barcelona,2,39.16330988748253,12.556902649670175,22.516615012125715,12.905943206593893,177,5872.86,5843.584012508392,99999.0,,,
+esfm,Sri Mariamman Singapore,0,1.4920493985513708,0.4412389781578158,10.406406306167431,6.369585247957454,222,5987.98,5970.066485881805,99999.0,0.08136203430698868,0.024448880002067186,0.9292989040444514
+esfm,Sri Mariamman Singapore,1,1.1345564396944785,0.3608906720626337,9.703927906328842,5.885963785879842,222,5993.62,5975.365797519684,99999.0,0.07981309127265196,0.023957917778313187,0.9234854997565143
+esfm,Sri Mariamman Singapore,2,1.0293707921546553,0.3426709250774071,10.15220066698994,6.067537729054635,222,5975.02,5958.786058187485,99999.0,0.07902485326574453,0.023758403592938876,0.9232224972615053
+esfm_rc,Sri Mariamman Singapore,0,2.074526913277199,0.5981095947062628,11.463860875975854,7.123711976607464,222,5510.81,5473.720585107803,99999.0,,,
+esfm_rc,Sri Mariamman Singapore,1,1.7658826525662057,0.5591108569640473,10.926621607114187,6.741925171900791,222,5496.08,5465.121775150299,99999.0,,,
+esfm_rc,Sri Mariamman Singapore,2,1.4009372521572498,0.44931018686538454,11.01432197073524,6.709336637494123,222,5499.38,5468.996921539307,99999.0,,,
+uesfm,Sri Mariamman Singapore,0,2.4686335799916477,0.7739007732343344,14.605464813669228,9.97923516189929,222,14748.08,14705.00972270966,99999.0,,,
+uesfm,Sri Mariamman Singapore,1,2.078550795026382,0.6787961741334789,13.788432610497317,9.377924514787443,222,14759.28,14715.18007564545,99999.0,,,
+uesfm,Sri Mariamman Singapore,2,1.7907753932427595,0.6052602851109582,12.444814769922951,8.402147028063522,222,14729.1,14697.75046205521,99999.0,,,
+esfm,Sri Thendayuthapani Singapore,0,0.7190396606626565,0.17922550285921848,7.12365374281766,2.231392236341273,98,13362.09,13320.09781932831,99999.0,0.19739404993046333,0.057925157042210476,6.111012693196153
+esfm,Sri Thendayuthapani Singapore,1,0.5350953449295449,0.1261925950852554,6.4467364012557615,2.04778550672059,98,13347.85,13309.03068971634,99999.0,0.19200876982588352,0.055981285294245835,8.004904549888494
+esfm,Sri Thendayuthapani Singapore,2,0.5262093764563488,0.14374074415280041,7.476614790858722,2.2575164196527426,98,13304.56,13262.61834168434,99999.0,0.19342864205084717,0.05646279918433841,7.499358041956718
+esfm_rc,Sri Thendayuthapani Singapore,0,0.7371639218203122,0.18650202548448253,7.499935996618937,2.3110604008688562,98,12532.48,12472.05219936371,99999.0,,,
+esfm_rc,Sri Thendayuthapani Singapore,1,0.7298957622388771,0.15775896959686408,6.855850988025784,2.2114171261230844,98,12550.85,12494.93173718452,99999.0,,,
+esfm_rc,Sri Thendayuthapani Singapore,2,0.6332570037033005,0.1670204347365887,7.4898727605029665,2.3242398420277945,98,12530.0,12474.00808596611,99999.0,,,
+uesfm,Sri Thendayuthapani Singapore,0,0.6231213322199768,0.21686608350881142,10.153314817239993,4.077818514354337,98,13885.62,13829.64960861206,99999.0,,,
+uesfm,Sri Thendayuthapani Singapore,1,0.8606139427206572,0.2295356611884548,9.279712988615236,3.705865828185615,98,13902.5,13848.30702185631,99999.0,,,
+uesfm,Sri Thendayuthapani Singapore,2,0.8685610884058521,0.229402954616869,9.929333644165053,3.7851203401221336,98,13914.59,13860.98727893829,99999.0,,,
+esfm,Sri Veeramakaliamman Singapore,0,2.333869921178247,0.542847254574005,11.965270419451508,5.472877662905037,157,12392.18,12365.15180635452,99999.0,0.18474384851666648,0.045708292122046836,2.2926940013925408
+esfm,Sri Veeramakaliamman Singapore,1,2.4804161216418734,0.5731176371311499,12.150997217479388,5.5309200508775875,157,12387.11,12363.29210400581,99999.0,,,
+esfm,Sri Veeramakaliamman Singapore,2,2.3974569344813434,0.5599333960859275,12.58650507397198,5.711763361742226,157,12354.49,12330.03034973145,99999.0,,,
+esfm_rc,Sri Veeramakaliamman Singapore,0,2.509821389378875,0.5815511527755557,13.072703350607625,6.052101493759435,157,11248.53,11196.92873668671,99999.0,,,
+esfm_rc,Sri Veeramakaliamman Singapore,1,2.8830400712922857,0.6676084122286379,13.610449126665129,6.2077999257192324,157,11257.08,11204.25902700424,99999.0,,,
+esfm_rc,Sri Veeramakaliamman Singapore,2,2.7715395728193424,0.64246713807485,13.500340673383525,6.122046272477576,157,11224.0,11178.18259263039,99999.0,,,
+uesfm,Sri Veeramakaliamman Singapore,0,2.2697869665184376,0.5625570243353476,17.34053107468558,8.599916604868525,157,12591.85,12539.44258975983,99999.0,,,
+uesfm,Sri Veeramakaliamman Singapore,1,2.4543260332472085,0.6034427994130044,15.868105070421757,7.655614154170212,157,12556.37,12497.1868751049,99999.0,,,
+uesfm,Sri Veeramakaliamman Singapore,2,2.2402378081453076,0.5362922009551214,14.34751765070624,6.779024945733527,157,12542.85,12490.60071396828,99999.0,,,
+esfm,Statue Of Liberty,0,52.35559762033152,28.241594953600423,688383.1603142131,683624.423417858,134,102.64,43.32754778862,45000.0,,,
+esfm,Statue Of Liberty,1,52.36920727661949,28.627801879821945,9277.715116667516,9303.546387753544,134,99.78,59.98358988761902,65000.0,,,
+esfm,Statue Of Liberty,2,52.3478568435799,28.64052858422677,7449.598084265948,7416.803834848509,134,97.28,3.520912647247314,0.0,,,
+esfm_rc,Statue Of Liberty,0,41.218124829650854,16.35503230171343,29.08277227297952,15.1834694799134,134,4803.59,4780.802038192749,99999.0,,,
+esfm_rc,Statue Of Liberty,1,39.57596391441381,15.93235966523679,27.37986618426645,14.290432615449056,134,4885.84,4862.140880584717,99999.0,,,
+esfm_rc,Statue Of Liberty,2,77.96860956365762,27.11543600094153,57.757423269004605,24.765649053609003,134,4900.73,4877.64520072937,99999.0,,,
+uesfm,Statue Of Liberty,0,42.72583129860769,17.96588742153844,35.11813579810792,19.871970350798126,134,5574.75,4164.912250995636,75000.0,,,
+uesfm,Statue Of Liberty,1,40.08181077917127,17.10193566857949,38.50494847120855,23.533461831836124,134,5591.94,5563.9591152668,99999.0,,,
+uesfm,Statue Of Liberty,2,77.10685903857735,28.579866483385974,68.32023407938978,27.795793203242106,134,5579.6,5550.581798315048,99999.0,,,
+esfm,The Pumpkin,0,27.59459354015253,5.94835892276036,27.52772578222832,9.11126349703041,196,4330.9,4316.109169960022,99999.0,,,
+esfm,The Pumpkin,1,26.54397520858153,5.701234509093555,27.342758332715125,8.907676505669826,196,4327.98,4314.043692827225,99999.0,,,
+esfm,The Pumpkin,2,23.05395646123413,5.007378502393024,25.75879879971585,7.898438464943201,196,4322.94,4310.356798410416,99999.0,,,
+esfm_rc,The Pumpkin,0,29.050661771479646,6.26570371316686,29.316909181775795,10.313120532688693,196,4910.6,4880.431479454041,99999.0,,,
+esfm_rc,The Pumpkin,1,26.919880388678763,5.774443198443047,27.896215187210355,9.357701886344401,196,4888.76,4859.359884023666,99999.0,,,
+esfm_rc,The Pumpkin,2,23.23435517664717,5.033668075029946,26.250953898370245,8.401411316898391,196,5035.56,5005.703458309174,99999.0,,,
+uesfm,The Pumpkin,0,28.724486849374728,6.049382168756297,32.623684990680346,13.080799898606461,196,5715.4,5688.280281066895,99999.0,,,
+uesfm,The Pumpkin,1,88.35244660282751,14.480787037348948,41.33437120165238,18.995341966051914,196,13267.96,13227.80508255959,99999.0,,,
+uesfm,The Pumpkin,2,25.512327800837042,5.499412600712255,29.975920040125473,11.811169023090269,196,5778.52,5752.150703191757,99999.0,,,
+esfm,Thian Hook Keng Temple Singapore,0,1.0653505173922067,0.1079565219850111,15.054152885837562,5.81055046268767,138,3299.26,3281.187238931656,99999.0,,,
+esfm,Thian Hook Keng Temple Singapore,1,0.8390930608109779,0.08272919436528102,14.587041005106803,5.705702663819475,138,3306.93,3283.257929086685,99999.0,,,
+esfm,Thian Hook Keng Temple Singapore,2,0.8760513596629002,0.08764395630143504,15.102705362055175,5.7567965038255124,138,3289.42,3278.173619031906,99999.0,,,
+esfm_rc,Thian Hook Keng Temple Singapore,0,1.0708584393011769,0.10688781753476269,15.9008082124771,6.214673486605086,138,3870.91,3843.877260684967,99999.0,,,
+esfm_rc,Thian Hook Keng Temple Singapore,1,0.9337315322680709,0.09457601793548075,15.699154845940917,6.103426802531022,138,3856.04,3835.938611507416,99999.0,,,
+esfm_rc,Thian Hook Keng Temple Singapore,2,0.9801090998477185,0.09872310339916696,15.76967975306522,6.19466789547437,138,3869.01,3848.04944896698,99999.0,,,
+uesfm,Thian Hook Keng Temple Singapore,0,0.8764983375088615,0.08202347764005188,16.212666761500287,8.391224013223542,138,4645.6,4621.513728618622,99999.0,,,
+uesfm,Thian Hook Keng Temple Singapore,1,0.9645123709523686,0.08977989341780128,17.411430835446634,7.905836840367921,138,4707.79,4683.081910133362,99999.0,,,
+uesfm,Thian Hook Keng Temple Singapore,2,0.9175998563737782,0.09144633731084849,18.520173505219134,8.485294370641896,138,4744.84,4725.761254310608,99999.0,,,
+esfm,Tsar Nikolai I,0,42.507553768489565,8.869914103947062,10.846690028504929,4.007961025902478,98,4932.15,4920.938139438629,99999.0,,,
+esfm,Tsar Nikolai I,1,43.09835370739786,8.983133733509343,11.402130572143294,4.032588092580996,98,4924.71,4914.929541349411,99999.0,,,
+esfm,Tsar Nikolai I,2,40.892180545819315,8.342265568650971,11.381967217434166,4.272306098513181,98,4930.74,4920.495186090469,99999.0,,,
+esfm_rc,Tsar Nikolai I,0,42.9819031611806,8.992392580392774,11.136519361295402,4.275788532431734,98,5820.43,5790.713532209396,99999.0,,,
+esfm_rc,Tsar Nikolai I,1,42.9676733807449,8.915056020619012,11.454988629980559,4.1737271261332385,98,5855.22,5834.582437753677,99999.0,,,
+esfm_rc,Tsar Nikolai I,2,41.42806728678177,8.459777152786799,11.79264110004773,4.455551649314186,98,5812.33,5793.210029363632,99999.0,,,
+uesfm,Tsar Nikolai I,0,78.76617790472244,15.241035141808007,22.562667761517744,10.059176205736136,98,6651.48,6624.56773352623,99999.0,,,
+uesfm,Tsar Nikolai I,1,77.86748197534685,15.28454292365267,22.935722021884366,9.48322850302597,98,6683.57,6660.621834993362,99999.0,,,
+uesfm,Tsar Nikolai I,2,80.52498876805318,15.650443058307074,22.159589059526215,10.075605477720723,98,6705.71,6687.593456506729,99999.0,,,
+esfm,Urban II,0,59.0297968076411,11.008489348760156,17.600594623525765,5.400933781867582,96,8889.69,8878.01452589035,99999.0,,,
+esfm,Urban II,1,58.947927470837534,11.025374954076108,17.465515658589908,5.68378594242137,96,8885.17,8876.618469953537,99999.0,,,
+esfm,Urban II,2,58.77798771298748,10.982786678935327,17.546556385759054,5.767405629316944,96,8893.99,8885.34243106842,99999.0,,,
+esfm_rc,Urban II,0,59.216536616067906,11.08663748585098,17.82614031873468,6.107131646325687,96,3941.4,3917.114440202713,99999.0,,,
+esfm_rc,Urban II,1,59.212380355922555,11.11858086716632,17.69328746099667,6.184432905776519,96,3928.12,3913.32084608078,99999.0,,,
+esfm_rc,Urban II,2,59.03738533505208,11.026151607617244,18.25007970328771,6.5714286359830405,96,3928.76,3910.048242330551,99999.0,,,
+uesfm,Urban II,0,70.72651677542252,10.846285449637286,20.932695061288186,9.499807603806392,96,4648.82,4629.565636634827,99999.0,,,
+uesfm,Urban II,1,76.69172898118858,11.840995205937814,23.160820370282533,9.824556493964668,96,10328.85,10307.27435564995,99999.0,,,
+uesfm,Urban II,2,71.54029457581474,13.635035860571998,17.833596140295615,10.443322691822011,96,4688.65,4667.598724603653,99999.0,,,
+esfm,Vercingetorix,0,79.07478057258302,9.776081681696928,6.649989325889541,1.7432748253588333,69,1223.75,1212.340968608856,99999.0,,,
+esfm,Vercingetorix,1,80.09313371434186,9.858626097407651,6.426263444651325,1.6979463697629655,69,1222.8,1213.763489961624,99999.0,,,
+esfm,Vercingetorix,2,89.68787080902364,10.950615257181747,8.695412468830675,3.360645595149667,69,1225.36,1215.808262825012,99999.0,,,
+esfm_rc,Vercingetorix,0,77.58116004246067,9.538739741724191,6.629853166022587,1.863546939626005,69,2490.94,2466.239528179169,99999.0,,,
+esfm_rc,Vercingetorix,1,80.59735041412574,9.964918624137537,6.844226785033953,1.8768109759688332,69,2466.57,2448.244215965271,99999.0,,,
+esfm_rc,Vercingetorix,2,78.38409445587499,10.071797617255122,8.813177642424675,3.6575396142269336,69,2464.57,2447.203853368759,99999.0,,,
+uesfm,Vercingetorix,0,90.40583622245013,11.219354185124304,13.021170300998648,6.058136758958224,69,3147.28,3128.323934793472,99999.0,,,
+uesfm,Vercingetorix,1,86.14504993152221,10.346921123080264,10.20083592178795,4.889104941829656,69,3189.9,3161.61762547493,99999.0,,,
+uesfm,Vercingetorix,2,82.105622694467,10.494716884620729,9.246952910064651,4.3504763439652825,69,3124.47,3102.738929271698,99999.0,,,
+esfm,Yueh Hai Ching Temple Singapore,0,0.6268598783684821,0.08360284283022794,5.56223941139456,2.132543748253723,43,1749.28,1738.747783660889,99999.0,,,
+esfm,Yueh Hai Ching Temple Singapore,1,0.5064956072004329,0.07235771132992698,5.30294026596974,2.06113770612239,43,1750.4,1739.704204082489,99999.0,,,
+esfm,Yueh Hai Ching Temple Singapore,2,0.49998151063409024,0.07009863154178562,5.92393490371475,2.2317763685579015,43,1746.32,1738.905562877655,99999.0,,,
+esfm_rc,Yueh Hai Ching Temple Singapore,0,0.9825210445873275,0.12976980064181087,5.840029450443526,2.3826902247979316,43,3055.68,3033.551073074341,99999.0,,,
+esfm_rc,Yueh Hai Ching Temple Singapore,1,0.43780611024916033,0.06298154479762619,5.398989840788992,2.105985514464995,43,3050.01,3030.62473154068,99999.0,,,
+esfm_rc,Yueh Hai Ching Temple Singapore,2,0.6724636819858062,0.09232893673280672,6.135782446114854,2.362032898968259,43,3056.82,3030.568630933762,99999.0,,,
+uesfm,Yueh Hai Ching Temple Singapore,0,1.011090286407443,0.14577536562871568,8.910101741972593,4.774050044333404,43,3800.63,3779.992084264755,99999.0,,,
+uesfm,Yueh Hai Ching Temple Singapore,1,0.9987973483341249,0.13337675637303437,8.021267579781536,4.39681234232059,43,3801.31,3781.260888576508,99999.0,,,
+uesfm,Yueh Hai Ching Temple Singapore,2,0.8036578112160735,0.11221136471697148,8.679131282807896,4.609032159824265,43,3795.78,3775.22882604599,99999.0,,,
diff --git a/code/results/single_scene/summary_table.md b/code/results/single_scene/summary_table.md
index bd3f673..3d7af62 100644
--- a/code/results/single_scene/summary_table.md
+++ b/code/results/single_scene/summary_table.md
@@ -2,44 +2,45 @@
 
 Mean over seeds per (scene, method); best value per scene in **bold** (lower is better except Nr).
 
-| Scene | ESFM Rot (deg) | U-ESFM Rot (deg) | ESFM (paper) Rot (deg) | ESFM Trans | U-ESFM Trans | ESFM (paper) Trans | ESFM Reproj (px) | U-ESFM Reproj (px) | ESFM (paper) Reproj (px) | ESFM Nr | U-ESFM Nr | ESFM (paper) Nr | ESFM Time (s) | U-ESFM Time (s) | ESFM (paper) Time (s) | ESFM Rot-BA (deg) | U-ESFM Rot-BA (deg) | ESFM (paper) Rot-BA (deg) | ESFM Trans-BA | U-ESFM Trans-BA | ESFM (paper) Trans-BA | ESFM Reproj-BA (px) | U-ESFM Reproj-BA (px) | ESFM (paper) Reproj-BA (px) |
-|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
-| Alcatraz Courtyard | **0.362** | 0.616 | 0.619 | **0.093** | 0.161 | 0.160 | **3.435** | 5.796 | 1.640 | **133** | 133 | -- | **4826.713** | 6481.757 | -- | -- | -- | 0.049 | -- | -- | 0.015 | -- | -- | 0.810 |
-| Alcatraz Water Tower | **0.666** | 7.413 | 0.933 | **0.370** | 3.437 | 0.518 | **4.763** | 8.607 | 2.130 | **172** | 172 | -- | **2461.337** | 3854.907 | -- | -- | -- | 0.230 | -- | -- | 0.116 | -- | -- | 0.550 |
-| Buddah Tooth Relic Temple Singapore | **1.131** | 12.080 | 1.030 | **0.280** | 2.308 | 0.233 | **9.964** | 19.634 | 2.060 | **162** | 162 | -- | **2916.837** | 4320.873 | -- | -- | -- | 0.081 | -- | -- | 0.014 | -- | -- | 0.850 |
-| Doge Palace Venice | **0.801** | 0.911 | 1.163 | **0.218** | 0.290 | 0.342 | **6.432** | 10.966 | 3.620 | **241** | 241 | -- | 28768.725 | **21074.950** | -- | -- | -- | 0.211 | -- | -- | 0.029 | -- | -- | 1 |
-| Door Lund | **0.014** | 14.172 | 0.024 | **0.004** | 0.952 | 0.006 | **1.812** | 11.922 | 0.320 | **12** | 12 | -- | **4888.363** | 9188.593 | -- | -- | -- | 0.006 | -- | -- | 0.001 | -- | -- | 0.300 |
-| Drinking Fountain Somewhere In Zurich | **17.193** | 17.536 | 0.031 | 0.753 | **0.740** | 0.004 | **6.058** | 6.511 | 0.330 | **14** | 14 | -- | **883.837** | 2963.647 | -- | -- | -- | 0.007 | -- | -- | 0.002 | -- | -- | 0.310 |
-| East Indiaman Goteborg | 5.568 | **5.310** | 3.814 | 0.979 | **0.959** | 0.621 | **7.318** | 9.840 | 4.130 | **179** | 179 | -- | **3987.183** | 5241.430 | -- | -- | -- | 3.117 | -- | -- | 0.509 | -- | -- | 1.850 |
-| Ecole Superior De Guerre | 0.349 | **0.284** | 0.318 | 0.090 | **0.077** | 0.081 | **3.418** | 5.349 | 0.720 | **35** | 35 | -- | **2380.387** | 4633.833 | -- | -- | -- | 0.024 | -- | -- | 0.005 | -- | -- | 0.340 |
-| Eglise du dome | **0.351** | 45.801 | 0.808 | **0.088** | 6.451 | 0.205 | **3.799** | 7.964 | 0.910 | **85** | 85 | -- | 25099.450 | **23371.730** | -- | -- | -- | 0.037 | -- | -- | 0.010 | -- | -- | 0.270 |
-| Folke Filbyter | 85.831 | **81.288** | 74.596 | **0.129** | 0.130 | 0.125 | **19.275** | 23.156 | 10.370 | **40** | 40 | -- | **1581.897** | 3703.187 | -- | -- | -- | 70.157 | -- | -- | 0.118 | -- | -- | 4.290 |
-| Fort Channing Gate Singapore | **0.093** | 11.128 | 0.207 | **0.041** | 4.004 | 0.093 | **1.783** | 6.651 | 0.520 | **27** | 27 | -- | **3434.113** | 6172.087 | -- | -- | -- | 0.020 | -- | -- | 0.008 | -- | -- | 0.250 |
-| Golden Statue Somewhere In Hong Kong | **0.232** | 0.371 | 0.292 | **0.053** | 0.074 | 0.073 | **1.829** | 2.835 | 0.400 | **18** | 18 | -- | **4095.853** | 7762.437 | -- | -- | -- | 0.031 | -- | -- | 0.004 | -- | -- | 0.270 |
-| Gustav Vasa | **3.509** | 6.099 | 34.181 | **0.220** | 0.345 | 1.085 | **1.655** | 3.087 | 3.520 | **18** | 18 | -- | **709.867** | 2543.553 | -- | -- | -- | 32.266 | -- | -- | 1.145 | -- | -- | 3.150 |
-| GustavIIAdolf | **47.126** | 77.423 | 67.784 | **8.536** | 11.962 | 9.714 | **11.722** | 15.851 | 13.910 | **57** | 57 | -- | **1044.170** | 2892.580 | -- | -- | -- | 58.458 | -- | -- | 8.524 | -- | -- | 11.490 |
-| Jonas Ahlstromer | 48.740 | **42.246** | 50.190 | **9.823** | 10.300 | 10.888 | 11.136 | **8.782** | 10.820 | **40** | 40 | -- | **615.450** | 2328.313 | -- | -- | -- | 47.117 | -- | -- | 10.451 | -- | -- | 8.410 |
-| Kings College University Of Toronto | **9.025** | 21.523 | 0.989 | 2.044 | **1.905** | 0.235 | **3.565** | 5.525 | 0.900 | **77** | 77 | -- | **1078.980** | 2886.243 | -- | -- | -- | 0.085 | -- | -- | 0.017 | -- | -- | 0.340 |
-| Lund University Sphinx | **11.392** | 33.023 | 19.522 | **2.981** | 6.889 | 4.585 | **6.056** | 12.060 | 4.780 | **70** | 70 | -- | **2718.567** | 4654.610 | -- | -- | -- | 8.752 | -- | -- | 2.191 | -- | -- | 1.360 |
-| Nijo Castle Gate | **0.839** | 1.682 | 1.495 | **0.165** | 0.326 | 0.286 | **7.415** | 12.138 | 1.700 | **19** | 19 | -- | **1095.177** | 3306.697 | -- | -- | -- | 0.069 | -- | -- | 0.012 | -- | -- | 0.730 |
-| Pantheon Paris | -- | 19.345 | 0.192 | -- | 3.065 | 0.050 | -- | 13.562 | 1.470 | -- | 179 | -- | -- | 5497.377 | -- | -- | -- | 0.040 | -- | -- | 0.005 | -- | -- | 0.490 |
-| Park Gate Clermont Ferrand | 21.388 | **19.302** | 0.391 | 10.192 | **9.397** | 0.125 | **7.712** | 7.852 | 0.570 | **34** | 34 | -- | **1596.940** | 3597.403 | -- | -- | -- | 0.049 | -- | -- | 0.022 | -- | -- | 0.350 |
-| Plaza De Armas Santiago | **0.775** | 3.327 | 6.782 | **0.337** | 1.435 | 2.944 | **6.009** | 9.463 | 7.400 | **240** | 240 | -- | **6372.683** | 6535.537 | -- | -- | -- | 2.556 | -- | -- | 1.383 | -- | -- | 4.900 |
-| Porta San Donato Bologna | **0.592** | 21.006 | 2.153 | **0.107** | 3.541 | 0.388 | **5.603** | 13.244 | 2.280 | **141** | 141 | -- | **3754.163** | 5118.030 | -- | -- | -- | 0.095 | -- | -- | 0.046 | -- | -- | 0.750 |
-| Round Church Cambridge | **2.129** | 2.490 | 2.451 | **0.920** | 1.002 | 1.003 | **6.097** | 8.546 | 2.660 | **92** | 92 | -- | **9776.630** | 12061.350 | -- | -- | -- | 1.107 | -- | -- | 0.582 | -- | -- | 1.540 |
-| Skansen Kronan Gothenburg | **0.338** | 0.375 | 0.736 | **0.116** | 0.124 | 0.226 | **3.048** | 4.799 | 1.240 | **131** | 131 | -- | 17163.940 | **16235.090** | -- | -- | -- | 0.026 | -- | -- | 0.008 | -- | -- | 0.670 |
-| Smolny Cathedral St Petersburg | **21.806** | 22.143 | 0.554 | 2.111 | **2.099** | 0.051 | **11.228** | 12.526 | 1.660 | **131** | 131 | -- | **10362.595** | 10475.740 | -- | -- | -- | 0.033 | -- | -- | 0.006 | -- | -- | 0.810 |
-| Sri Mariamman Singapore | **1.219** | 6.479 | 2.302 | **0.382** | 2.188 | 0.683 | **10.088** | 17.854 | 4.130 | **222** | 222 | -- | **5985.540** | 6485.667 | -- | -- | -- | 0.077 | -- | -- | 0.023 | -- | -- | 0.910 |
-| Sri Thendayuthapani Singapore | **0.719** | 47.075 | 46.269 | **0.179** | 3.855 | 3.812 | **7.124** | 14.133 | 23.370 | **98** | 98 | -- | **13362.090** | 13808.440 | -- | -- | -- | 44.170 | -- | -- | 2.870 | -- | -- | 8.440 |
-| Sri Veeramakaliamman Singapore | **2.334** | 29.405 | 2.559 | **0.543** | 3.868 | 0.597 | **11.965** | 28.951 | 3.470 | **157** | 157 | -- | **12392.180** | 12544.690 | -- | -- | -- | 0.175 | -- | -- | 0.040 | -- | -- | 0.730 |
-| Statue Of Liberty | 52.358 | **39.859** | 46.887 | 28.503 | **18.503** | 20.012 | 235036.825 | **41.028** | 26.160 | **134** | 134 | -- | **99.900** | 12924 | -- | -- | -- | 9.091 | -- | -- | 4.122 | -- | -- | 6.970 |
-| The Pumpkin | **25.731** | 26.571 | 94.672 | **5.552** | 5.665 | 14.890 | **26.876** | 35.367 | 33.410 | **196** | 196 | -- | **4327.273** | 5790.597 | -- | -- | -- | 98.862 | -- | -- | 14.952 | -- | -- | 24.850 |
-| Thian Hook Keng Temple Singapore | **0.927** | 1.201 | 0.832 | **0.093** | 0.121 | 0.082 | **14.915** | 20.789 | 2.750 | **138** | 138 | -- | **3298.537** | 4612.373 | -- | -- | -- | 0.081 | -- | -- | 0.008 | -- | -- | 1.130 |
-| Tsar Nikolai I | **42.803** | 78.413 | 48.499 | **8.927** | 15.394 | 9.467 | **11.124** | 18.431 | 9.790 | **98** | 98 | -- | **4928.430** | 6646.845 | -- | -- | -- | 36.280 | -- | -- | 7.836 | -- | -- | 6.530 |
-| Urban II | 59.030 | -- | 47.490 | 11.008 | -- | 9.467 | 17.601 | -- | 9.380 | 96 | -- | -- | 8889.690 | -- | -- | -- | -- | 48.214 | -- | -- | 9.586 | -- | -- | 6.920 |
-| Vercingetorix | **82.952** | 86.088 | 69.328 | **10.195** | 10.890 | 8.788 | **7.257** | 8.681 | 5.080 | **69** | 69 | -- | **1223.970** | 3105.607 | -- | -- | -- | 17.706 | -- | -- | 3.104 | -- | -- | 1.500 |
-| Yueh Hai Ching Temple Singapore | 0.627 | -- | 0.720 | 0.084 | -- | 0.098 | 5.562 | -- | 0.940 | 43 | -- | -- | 1749.280 | -- | -- | -- | -- | 0.043 | -- | -- | 0.014 | -- | -- | 0.650 |
-| **Mean** | **16.146** | 23.697 | 18.023 | **3.121** | 4.014 | 2.912 | 6920.602 | **13.088** | 5.673 | 100.559 | **104.818** | -- | **5819.728** | 7358.187 | -- | -- | -- | 13.695 | -- | -- | 1.937 | -- | -- | 2.992 |
+| Scene | ESFM (official code) Rot (deg) | ESFM (RESfM code) Rot (deg) | U-ESFM Rot (deg) | ESFM (paper) Rot (deg) | ESFM (official code) Trans | ESFM (RESfM code) Trans | U-ESFM Trans | ESFM (paper) Trans | ESFM (official code) Reproj (px) | ESFM (RESfM code) Reproj (px) | U-ESFM Reproj (px) | ESFM (paper) Reproj (px) | ESFM (official code) Nr | ESFM (RESfM code) Nr | U-ESFM Nr | ESFM (paper) Nr | ESFM (official code) Time (s) | ESFM (RESfM code) Time (s) | U-ESFM Time (s) | ESFM (paper) Time (s) | ESFM (official code) Rot-BA (deg) | ESFM (RESfM code) Rot-BA (deg) | U-ESFM Rot-BA (deg) | ESFM (paper) Rot-BA (deg) | ESFM (official code) Trans-BA | ESFM (RESfM code) Trans-BA | U-ESFM Trans-BA | ESFM (paper) Trans-BA | ESFM (official code) Reproj-BA (px) | ESFM (RESfM code) Reproj-BA (px) | U-ESFM Reproj-BA (px) | ESFM (paper) Reproj-BA (px) |
+|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
+| Alcatraz Courtyard | **0.362** | 0.420 | 10.132 | 0.619 | **0.093** | 0.110 | 2.219 | 0.160 | **3.435** | 3.673 | 7.272 | 1.640 | **133** | 133 | 133 | -- | **4826.713** | 5684.120 | 6550.983 | -- | 0.044 | -- | -- | 0.049 | 0.017 | -- | -- | 0.015 | 0.965 | -- | -- | 0.810 |
+| Alcatraz Water Tower | **0.666** | 0.882 | 1.480 | 0.933 | **0.370** | 0.488 | 0.803 | 0.518 | **4.763** | 5.112 | 7.397 | 2.130 | **172** | 172 | 172 | -- | **2461.337** | 3064.183 | 3795.877 | -- | 0.227 | -- | -- | 0.230 | 0.115 | -- | -- | 0.116 | 0.741 | -- | -- | 0.550 |
+| Buddah Tooth Relic Temple Singapore | **1.131** | 1.267 | 1.791 | 1.030 | **0.280** | 0.337 | 0.520 | 0.233 | **9.964** | 10.743 | 14.029 | 2.060 | **162** | 162 | 162 | -- | **2916.837** | 3544.803 | 4289.503 | -- | 0.086 | -- | -- | 0.081 | 0.016 | -- | -- | 0.014 | 2.079 | -- | -- | 0.850 |
+| Doge Palace Venice | **0.753** | 0.880 | 1.121 | 1.163 | **0.209** | 0.249 | 0.350 | 0.342 | **6.359** | 6.831 | 10.496 | 3.620 | **241** | 241 | 241 | -- | 22522.737 | **7930.497** | 20767.260 | -- | 0.057 | -- | -- | 0.211 | 0.018 | -- | -- | 0.029 | 1.177 | -- | -- | 1 |
+| Door Lund | **0.014** | 0.018 | 4.651 | 0.024 | **0.004** | 0.005 | 0.344 | 0.006 | **1.812** | 1.891 | 5.545 | 0.320 | **12** | 12 | 12 | -- | **4888.363** | 8451.030 | 9255.943 | -- | 0.003 | -- | -- | 0.006 | 0.001 | -- | -- | 0.001 | 0.303 | -- | -- | 0.300 |
+| Drinking Fountain Somewhere In Zurich | **17.193** | 17.271 | 17.329 | 0.031 | 0.753 | **0.749** | 0.750 | 0.004 | **6.058** | 6.129 | 6.675 | 0.330 | **14** | 14 | 14 | -- | **883.837** | 2332.817 | 5815.220 | -- | 15.614 | -- | -- | 0.007 | 0.798 | -- | -- | 0.002 | 5.185 | -- | -- | 0.310 |
+| East Indiaman Goteborg | 5.568 | **5.530** | 7.231 | 3.814 | 0.979 | **0.978** | 1.364 | 0.621 | **7.318** | 7.536 | 10.130 | 4.130 | **179** | 179 | 179 | -- | **3987.183** | 4385.873 | 5198.093 | -- | 5.245 | -- | -- | 3.117 | 0.993 | -- | -- | 0.509 | 2.648 | -- | -- | 1.850 |
+| Ecole Superior De Guerre | **0.349** | 2.337 | 25.800 | 0.318 | **0.090** | 0.545 | 1.324 | 0.081 | **3.418** | 4.726 | 8.646 | 0.720 | **35** | 35 | 35 | -- | **2380.387** | 3932.013 | 10416.080 | -- | 0.014 | -- | -- | 0.024 | 0.003 | -- | -- | 0.005 | 0.355 | -- | -- | 0.340 |
+| Eglise du dome | **0.386** | 0.569 | 15.675 | 0.808 | **0.098** | 0.147 | 2.288 | 0.205 | **3.987** | 4.250 | 6.815 | 0.910 | **85** | 85 | 85 | -- | 19631.293 | **8795.890** | 9802.683 | -- | 0.034 | -- | -- | 0.037 | 0.010 | -- | -- | 0.010 | 2.081 | -- | -- | 0.270 |
+| Folke Filbyter | 85.831 | 85.900 | **82.656** | 74.596 | **0.129** | 0.129 | 0.131 | 0.125 | **19.275** | 19.898 | 28.225 | 10.370 | **40** | 40 | 40 | -- | **1581.897** | 3072.870 | 3755.193 | -- | 71.440 | -- | -- | 70.157 | 0.130 | -- | -- | 0.118 | 20.014 | -- | -- | 4.290 |
+| Fort Channing Gate Singapore | **0.093** | 0.117 | 0.154 | 0.207 | **0.041** | 0.052 | 0.071 | 0.093 | **1.783** | 1.871 | 3.715 | 0.520 | **27** | 27 | 27 | -- | **3434.113** | 5450.200 | 6171.250 | -- | 0.018 | -- | -- | 0.020 | 0.007 | -- | -- | 0.008 | 0.260 | -- | -- | 0.250 |
+| Golden Statue Somewhere In Hong Kong | **0.232** | 0.267 | 0.508 | 0.292 | **0.053** | 0.062 | 0.082 | 0.073 | **1.829** | 1.944 | 3.022 | 0.400 | **18** | 18 | 18 | -- | **4095.853** | 7057.423 | 11699.307 | -- | 0.026 | -- | -- | 0.031 | 0.003 | -- | -- | 0.004 | 0.282 | -- | -- | 0.270 |
+| Gustav Vasa | **3.509** | 4.122 | 25.854 | 34.181 | **0.220** | 0.249 | 0.775 | 1.085 | **1.655** | 1.874 | 4.747 | 3.520 | **18** | 18 | 18 | -- | **709.867** | 1928.880 | 3887.653 | -- | 1.004 | -- | -- | 32.266 | 0.113 | -- | -- | 1.145 | 0.538 | -- | -- | 3.150 |
+| GustavIIAdolf | **47.126** | 47.468 | 88.194 | 67.784 | **8.536** | 8.574 | 13.129 | 9.714 | **11.722** | 11.918 | 17.220 | 13.910 | **57** | 57 | 57 | -- | **1044.170** | 2265.203 | 2981.017 | -- | 24.738 | -- | -- | 58.458 | 5.460 | -- | -- | 8.524 | 16.352 | -- | -- | 11.490 |
+| Jonas Ahlstromer | 48.740 | 48.415 | **46.584** | 50.190 | 9.823 | **9.672** | 10.245 | 10.888 | **11.136** | 11.328 | 11.144 | 10.820 | **40** | 40 | 40 | -- | **615.450** | 1712.753 | 2414.653 | -- | 32.847 | -- | -- | 47.117 | 6.311 | -- | -- | 10.451 | 7.502 | -- | -- | 8.410 |
+| Kings College University Of Toronto | **9.025** | 9.312 | 14.284 | 0.989 | 2.044 | 2.108 | **1.639** | 0.235 | **3.565** | 3.732 | 6.351 | 0.900 | **77** | 77 | 77 | -- | **1078.980** | 2209.693 | 2857.733 | -- | 5.786 | -- | -- | 0.085 | 1.224 | -- | -- | 0.017 | 1.768 | -- | -- | 0.340 |
+| Lund University Sphinx | **11.392** | 13.007 | 21.718 | 19.522 | **2.981** | 3.385 | 4.817 | 4.585 | **6.056** | 6.760 | 11.550 | 4.780 | **70** | 70 | 70 | -- | **2718.567** | 3924.373 | 4642.657 | -- | 7.011 | -- | -- | 8.752 | 1.807 | -- | -- | 2.191 | 1.201 | -- | -- | 1.360 |
+| Nijo Castle Gate | **0.839** | 1.325 | 3.926 | 1.495 | **0.165** | 0.255 | 1.138 | 0.286 | **7.415** | 8.223 | 15.135 | 1.700 | **19** | 19 | 19 | -- | **1095.177** | 2629.490 | 3234.357 | -- | 0.081 | -- | -- | 0.069 | 0.013 | -- | -- | 0.012 | 0.761 | -- | -- | 0.730 |
+| Pantheon Paris | -- | **0.253** | 15.196 | 0.192 | -- | **0.056** | 2.291 | 0.050 | -- | **4.893** | 10.638 | 1.470 | -- | **179** | 179 | -- | -- | **4640.607** | 5557.600 | -- | -- | -- | -- | 0.040 | -- | -- | -- | 0.005 | -- | -- | -- | 0.490 |
+| Park Gate Clermont Ferrand | 21.388 | **21.310** | 23.641 | 0.391 | 10.192 | **10.177** | 11.173 | 0.125 | **7.712** | 7.747 | 8.652 | 0.570 | **34** | 34 | 34 | -- | **1596.940** | 2952.627 | 3650.170 | -- | 10.261 | -- | -- | 0.049 | 4.338 | -- | -- | 0.022 | 14.280 | -- | -- | 0.350 |
+| Plaza De Armas Santiago | **0.775** | 1.122 | 6.571 | 6.782 | **0.337** | 0.487 | 2.852 | 2.944 | **6.009** | 6.327 | 11.199 | 7.400 | **240** | 240 | 240 | -- | 6372.683 | **5678.743** | 6616.970 | -- | 0.135 | -- | -- | 2.556 | 0.054 | -- | -- | 1.383 | 2.924 | -- | -- | 4.900 |
+| Porta San Donato Bologna | **0.592** | 0.610 | 1.814 | 2.153 | **0.107** | 0.112 | 0.468 | 0.388 | **5.603** | 5.779 | 9.168 | 2.280 | **141** | 141 | 141 | -- | **3754.163** | 4317.957 | 5167.537 | -- | 0.112 | -- | -- | 0.095 | 0.056 | -- | -- | 0.046 | 3.224 | -- | -- | 0.750 |
+| Round Church Cambridge | 2.137 | 2.293 | **2.097** | 2.451 | 0.927 | 0.967 | **0.772** | 1.003 | **6.103** | 6.493 | 8.283 | 2.660 | **92** | 92 | 92 | -- | **9779.270** | 10883.870 | 12129.840 | -- | 0.987 | -- | -- | 1.107 | 0.553 | -- | -- | 0.582 | 4.877 | -- | -- | 1.540 |
+| Skansen Kronan Gothenburg | **0.301** | 0.312 | 0.342 | 0.736 | **0.102** | 0.106 | 0.118 | 0.226 | **2.906** | 3.020 | 4.753 | 1.240 | **131** | 131 | 131 | -- | 17153.677 | **6127.697** | 10156.743 | -- | 0.025 | -- | -- | 0.026 | 0.008 | -- | -- | 0.008 | 0.682 | -- | -- | 0.670 |
+| Smolny Cathedral St Petersburg | **21.775** | 21.848 | 22.241 | 0.554 | 2.111 | 2.107 | **2.096** | 0.051 | **11.239** | 11.420 | 12.951 | 1.660 | **131** | 131 | 131 | -- | 10362.787 | **9384.753** | 10445.090 | -- | 21.416 | -- | -- | 0.033 | 2.158 | -- | -- | 0.006 | 10.310 | -- | -- | 0.810 |
+| Some Cathedral In Barcelona | **0.651** | 0.787 | 14.741 | 0.880 | **0.233** | 0.282 | 4.842 | 0.315 | **6.271** | 6.796 | 15.920 | 2.870 | **177** | 177 | 177 | -- | 5001.973 | **4961.633** | 5889.770 | -- | 0.033 | -- | -- | 0.026 | 0.014 | -- | -- | 0.011 | 1.100 | -- | -- | 0.890 |
+| Sri Mariamman Singapore | **1.219** | 1.747 | 2.113 | 2.302 | **0.382** | 0.536 | 0.686 | 0.683 | **10.088** | 11.135 | 13.613 | 4.130 | **222** | 222 | 222 | -- | 5985.540 | **5502.090** | 14745.487 | -- | 0.080 | -- | -- | 0.077 | 0.024 | -- | -- | 0.023 | 0.925 | -- | -- | 0.910 |
+| Sri Thendayuthapani Singapore | **0.593** | 0.700 | 0.784 | 46.269 | **0.150** | 0.170 | 0.225 | 3.812 | **7.016** | 7.282 | 9.787 | 23.370 | **98** | 98 | 98 | -- | 13338.167 | **12537.777** | 13900.903 | -- | 0.194 | -- | -- | 44.170 | 0.057 | -- | -- | 2.870 | 7.205 | -- | -- | 8.440 |
+| Sri Veeramakaliamman Singapore | 2.404 | 2.721 | **2.321** | 2.559 | **0.559** | 0.631 | 0.567 | 0.597 | **12.234** | 13.394 | 15.852 | 3.470 | **157** | 157 | 157 | -- | 12377.927 | **11243.203** | 12563.690 | -- | 0.185 | -- | -- | 0.175 | 0.046 | -- | -- | 0.040 | 2.293 | -- | -- | 0.730 |
+| Statue Of Liberty | **52.358** | 52.921 | 53.305 | 46.887 | 28.503 | **19.801** | 21.216 | 20.012 | 235036.825 | **38.073** | 47.314 | 26.160 | **134** | 134 | 134 | -- | **99.900** | 4863.387 | 5582.097 | -- | -- | -- | -- | 9.091 | -- | -- | -- | 4.122 | -- | -- | -- | 6.970 |
+| The Pumpkin | **25.731** | 26.402 | 47.530 | 94.672 | **5.552** | 5.691 | 8.677 | 14.890 | **26.876** | 27.821 | 34.645 | 33.410 | **196** | 196 | 196 | -- | **4327.273** | 4944.973 | 8253.960 | -- | -- | -- | -- | 98.862 | -- | -- | -- | 14.952 | -- | -- | -- | 24.850 |
+| Thian Hook Keng Temple Singapore | 0.927 | 0.995 | **0.920** | 0.832 | 0.093 | 0.100 | **0.088** | 0.082 | **14.915** | 15.790 | 17.381 | 2.750 | **138** | 138 | 138 | -- | **3298.537** | 3865.320 | 4699.410 | -- | -- | -- | -- | 0.081 | -- | -- | -- | 0.008 | -- | -- | -- | 1.130 |
+| Tsar Nikolai I | **42.166** | 42.459 | 79.053 | 48.499 | **8.732** | 8.789 | 15.392 | 9.467 | **11.210** | 11.461 | 22.553 | 9.790 | **98** | 98 | 98 | -- | **4929.200** | 5829.327 | 6680.253 | -- | -- | -- | -- | 36.280 | -- | -- | -- | 7.836 | -- | -- | -- | 6.530 |
+| Urban II | **58.919** | 59.155 | 72.986 | 47.490 | **11.006** | 11.077 | 12.107 | 9.467 | **17.538** | 17.923 | 20.642 | 9.380 | **96** | 96 | 96 | -- | 8889.617 | **3932.760** | 6555.440 | -- | -- | -- | -- | 48.214 | -- | -- | -- | 9.586 | -- | -- | -- | 6.920 |
+| Vercingetorix | 82.952 | **78.854** | 86.219 | 69.328 | 10.195 | **9.858** | 10.687 | 8.788 | **7.257** | 7.429 | 10.823 | 5.080 | **69** | 69 | 69 | -- | **1223.970** | 2474.027 | 3153.883 | -- | -- | -- | -- | 17.706 | -- | -- | -- | 3.104 | -- | -- | -- | 1.500 |
+| Yueh Hai Ching Temple Singapore | **0.544** | 0.698 | 0.938 | 0.720 | **0.075** | 0.095 | 0.130 | 0.098 | **5.596** | 5.792 | 8.537 | 0.940 | **43** | 43 | 43 | -- | **1748.667** | 3054.170 | 3799.240 | -- | -- | -- | -- | 0.043 | -- | -- | -- | 0.014 | -- | -- | -- | 0.650 |
+| **Mean** | 15.676 | **15.397** | 22.275 | 17.547 | 3.032 | **2.754** | 3.788 | 2.840 | 6723.056 | **9.084** | 12.801 | 5.595 | 102.743 | **104.861** | 104.861 | -- | 5460.373 | **5154.640** | 7141.210 | -- | 7.061 | -- | -- | 13.315 | 0.870 | -- | -- | 1.883 | 4.001 | -- | -- | 2.933 |
 
 _Seeds per cell: [0, 1, 2]._
 
diff --git a/code/results/single_scene/summary_table.tex b/code/results/single_scene/summary_table.tex
index 258e3f2..b7677a5 100644
--- a/code/results/single_scene/summary_table.tex
+++ b/code/results/single_scene/summary_table.tex
@@ -4,48 +4,49 @@
 \caption{Single-scene optimization on the Olsson dataset (calibrated setting). Mean over [0, 1, 2] seeds; best per scene in bold. ESFM (paper) reproduces the published per-scene numbers of \cite{Moran_2021_ICCV} (post-BA values use their BA) and is excluded from the bold-best comparison.}
 \label{tab:single_scene_olsson}
 \resizebox{\textwidth}{!}{
-\begin{tabular}{lcccccccccccccccccccccccc}
+\begin{tabular}{lcccccccccccccccccccccccccccccccc}
 \toprule
- & \multicolumn{3}{c}{Rot (deg)} & \multicolumn{3}{c}{Trans} & \multicolumn{3}{c}{Reproj (px)} & \multicolumn{3}{c}{Nr} & \multicolumn{3}{c}{Time (s)} & \multicolumn{3}{c}{Rot-BA (deg)} & \multicolumn{3}{c}{Trans-BA} & \multicolumn{3}{c}{Reproj-BA (px)} \\
-Scene & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) & ESFM & U-ESFM & ESFM (paper) \\
+ & \multicolumn{4}{c}{Rot (deg)} & \multicolumn{4}{c}{Trans} & \multicolumn{4}{c}{Reproj (px)} & \multicolumn{4}{c}{Nr} & \multicolumn{4}{c}{Time (s)} & \multicolumn{4}{c}{Rot-BA (deg)} & \multicolumn{4}{c}{Trans-BA} & \multicolumn{4}{c}{Reproj-BA (px)} \\
+Scene & ESFM (official code) & ESFM (RESfM code) & U-ESFM & ESFM (paper) & ESFM (official code) & ESFM (RESfM code) & U-ESFM & ESFM (paper) & ESFM (official code) & ESFM (RESfM code) & U-ESFM & ESFM (paper) & ESFM (official code) & ESFM (RESfM code) & U-ESFM & ESFM (paper) & ESFM (official code) & ESFM (RESfM code) & U-ESFM & ESFM (paper) & ESFM (official code) & ESFM (RESfM code) & U-ESFM & ESFM (paper) & ESFM (official code) & ESFM (RESfM code) & U-ESFM & ESFM (paper) & ESFM (official code) & ESFM (RESfM code) & U-ESFM & ESFM (paper) \\
 \midrule
-Alcatraz Courtyard & \textbf{0.362} & 0.616 & 0.619 & \textbf{0.093} & 0.161 & 0.160 & \textbf{3.435} & 5.796 & 1.640 & \textbf{133} & 133 & -- & \textbf{4826.713} & 6481.757 & -- & -- & -- & 0.049 & -- & -- & 0.015 & -- & -- & 0.810 \\
-Alcatraz Water Tower & \textbf{0.666} & 7.413 & 0.933 & \textbf{0.370} & 3.437 & 0.518 & \textbf{4.763} & 8.607 & 2.130 & \textbf{172} & 172 & -- & \textbf{2461.337} & 3854.907 & -- & -- & -- & 0.230 & -- & -- & 0.116 & -- & -- & 0.550 \\
-Buddah Tooth Relic Temple Singapore & \textbf{1.131} & 12.080 & 1.030 & \textbf{0.280} & 2.308 & 0.233 & \textbf{9.964} & 19.634 & 2.060 & \textbf{162} & 162 & -- & \textbf{2916.837} & 4320.873 & -- & -- & -- & 0.081 & -- & -- & 0.014 & -- & -- & 0.850 \\
-Doge Palace Venice & \textbf{0.801} & 0.911 & 1.163 & \textbf{0.218} & 0.290 & 0.342 & \textbf{6.432} & 10.966 & 3.620 & \textbf{241} & 241 & -- & 28768.725 & \textbf{21074.950} & -- & -- & -- & 0.211 & -- & -- & 0.029 & -- & -- & 1 \\
-Door Lund & \textbf{0.014} & 14.172 & 0.024 & \textbf{0.004} & 0.952 & 0.006 & \textbf{1.812} & 11.922 & 0.320 & \textbf{12} & 12 & -- & \textbf{4888.363} & 9188.593 & -- & -- & -- & 0.006 & -- & -- & 0.001 & -- & -- & 0.300 \\
-Drinking Fountain Somewhere In Zurich & \textbf{17.193} & 17.536 & 0.031 & 0.753 & \textbf{0.740} & 0.004 & \textbf{6.058} & 6.511 & 0.330 & \textbf{14} & 14 & -- & \textbf{883.837} & 2963.647 & -- & -- & -- & 0.007 & -- & -- & 0.002 & -- & -- & 0.310 \\
-East Indiaman Goteborg & 5.568 & \textbf{5.310} & 3.814 & 0.979 & \textbf{0.959} & 0.621 & \textbf{7.318} & 9.840 & 4.130 & \textbf{179} & 179 & -- & \textbf{3987.183} & 5241.430 & -- & -- & -- & 3.117 & -- & -- & 0.509 & -- & -- & 1.850 \\
-Ecole Superior De Guerre & 0.349 & \textbf{0.284} & 0.318 & 0.090 & \textbf{0.077} & 0.081 & \textbf{3.418} & 5.349 & 0.720 & \textbf{35} & 35 & -- & \textbf{2380.387} & 4633.833 & -- & -- & -- & 0.024 & -- & -- & 0.005 & -- & -- & 0.340 \\
-Eglise du dome & \textbf{0.351} & 45.801 & 0.808 & \textbf{0.088} & 6.451 & 0.205 & \textbf{3.799} & 7.964 & 0.910 & \textbf{85} & 85 & -- & 25099.450 & \textbf{23371.730} & -- & -- & -- & 0.037 & -- & -- & 0.010 & -- & -- & 0.270 \\
-Folke Filbyter & 85.831 & \textbf{81.288} & 74.596 & \textbf{0.129} & 0.130 & 0.125 & \textbf{19.275} & 23.156 & 10.370 & \textbf{40} & 40 & -- & \textbf{1581.897} & 3703.187 & -- & -- & -- & 70.157 & -- & -- & 0.118 & -- & -- & 4.290 \\
-Fort Channing Gate Singapore & \textbf{0.093} & 11.128 & 0.207 & \textbf{0.041} & 4.004 & 0.093 & \textbf{1.783} & 6.651 & 0.520 & \textbf{27} & 27 & -- & \textbf{3434.113} & 6172.087 & -- & -- & -- & 0.020 & -- & -- & 0.008 & -- & -- & 0.250 \\
-Golden Statue Somewhere In Hong Kong & \textbf{0.232} & 0.371 & 0.292 & \textbf{0.053} & 0.074 & 0.073 & \textbf{1.829} & 2.835 & 0.400 & \textbf{18} & 18 & -- & \textbf{4095.853} & 7762.437 & -- & -- & -- & 0.031 & -- & -- & 0.004 & -- & -- & 0.270 \\
-Gustav Vasa & \textbf{3.509} & 6.099 & 34.181 & \textbf{0.220} & 0.345 & 1.085 & \textbf{1.655} & 3.087 & 3.520 & \textbf{18} & 18 & -- & \textbf{709.867} & 2543.553 & -- & -- & -- & 32.266 & -- & -- & 1.145 & -- & -- & 3.150 \\
-GustavIIAdolf & \textbf{47.126} & 77.423 & 67.784 & \textbf{8.536} & 11.962 & 9.714 & \textbf{11.722} & 15.851 & 13.910 & \textbf{57} & 57 & -- & \textbf{1044.170} & 2892.580 & -- & -- & -- & 58.458 & -- & -- & 8.524 & -- & -- & 11.490 \\
-Jonas Ahlstromer & 48.740 & \textbf{42.246} & 50.190 & \textbf{9.823} & 10.300 & 10.888 & 11.136 & \textbf{8.782} & 10.820 & \textbf{40} & 40 & -- & \textbf{615.450} & 2328.313 & -- & -- & -- & 47.117 & -- & -- & 10.451 & -- & -- & 8.410 \\
-Kings College University Of Toronto & \textbf{9.025} & 21.523 & 0.989 & 2.044 & \textbf{1.905} & 0.235 & \textbf{3.565} & 5.525 & 0.900 & \textbf{77} & 77 & -- & \textbf{1078.980} & 2886.243 & -- & -- & -- & 0.085 & -- & -- & 0.017 & -- & -- & 0.340 \\
-Lund University Sphinx & \textbf{11.392} & 33.023 & 19.522 & \textbf{2.981} & 6.889 & 4.585 & \textbf{6.056} & 12.060 & 4.780 & \textbf{70} & 70 & -- & \textbf{2718.567} & 4654.610 & -- & -- & -- & 8.752 & -- & -- & 2.191 & -- & -- & 1.360 \\
-Nijo Castle Gate & \textbf{0.839} & 1.682 & 1.495 & \textbf{0.165} & 0.326 & 0.286 & \textbf{7.415} & 12.138 & 1.700 & \textbf{19} & 19 & -- & \textbf{1095.177} & 3306.697 & -- & -- & -- & 0.069 & -- & -- & 0.012 & -- & -- & 0.730 \\
-Pantheon Paris & -- & 19.345 & 0.192 & -- & 3.065 & 0.050 & -- & 13.562 & 1.470 & -- & 179 & -- & -- & 5497.377 & -- & -- & -- & 0.040 & -- & -- & 0.005 & -- & -- & 0.490 \\
-Park Gate Clermont Ferrand & 21.388 & \textbf{19.302} & 0.391 & 10.192 & \textbf{9.397} & 0.125 & \textbf{7.712} & 7.852 & 0.570 & \textbf{34} & 34 & -- & \textbf{1596.940} & 3597.403 & -- & -- & -- & 0.049 & -- & -- & 0.022 & -- & -- & 0.350 \\
-Plaza De Armas Santiago & \textbf{0.775} & 3.327 & 6.782 & \textbf{0.337} & 1.435 & 2.944 & \textbf{6.009} & 9.463 & 7.400 & \textbf{240} & 240 & -- & \textbf{6372.683} & 6535.537 & -- & -- & -- & 2.556 & -- & -- & 1.383 & -- & -- & 4.900 \\
-Porta San Donato Bologna & \textbf{0.592} & 21.006 & 2.153 & \textbf{0.107} & 3.541 & 0.388 & \textbf{5.603} & 13.244 & 2.280 & \textbf{141} & 141 & -- & \textbf{3754.163} & 5118.030 & -- & -- & -- & 0.095 & -- & -- & 0.046 & -- & -- & 0.750 \\
-Round Church Cambridge & \textbf{2.129} & 2.490 & 2.451 & \textbf{0.920} & 1.002 & 1.003 & \textbf{6.097} & 8.546 & 2.660 & \textbf{92} & 92 & -- & \textbf{9776.630} & 12061.350 & -- & -- & -- & 1.107 & -- & -- & 0.582 & -- & -- & 1.540 \\
-Skansen Kronan Gothenburg & \textbf{0.338} & 0.375 & 0.736 & \textbf{0.116} & 0.124 & 0.226 & \textbf{3.048} & 4.799 & 1.240 & \textbf{131} & 131 & -- & 17163.940 & \textbf{16235.090} & -- & -- & -- & 0.026 & -- & -- & 0.008 & -- & -- & 0.670 \\
-Smolny Cathedral St Petersburg & \textbf{21.806} & 22.143 & 0.554 & 2.111 & \textbf{2.099} & 0.051 & \textbf{11.228} & 12.526 & 1.660 & \textbf{131} & 131 & -- & \textbf{10362.595} & 10475.740 & -- & -- & -- & 0.033 & -- & -- & 0.006 & -- & -- & 0.810 \\
-Sri Mariamman Singapore & \textbf{1.219} & 6.479 & 2.302 & \textbf{0.382} & 2.188 & 0.683 & \textbf{10.088} & 17.854 & 4.130 & \textbf{222} & 222 & -- & \textbf{5985.540} & 6485.667 & -- & -- & -- & 0.077 & -- & -- & 0.023 & -- & -- & 0.910 \\
-Sri Thendayuthapani Singapore & \textbf{0.719} & 47.075 & 46.269 & \textbf{0.179} & 3.855 & 3.812 & \textbf{7.124} & 14.133 & 23.370 & \textbf{98} & 98 & -- & \textbf{13362.090} & 13808.440 & -- & -- & -- & 44.170 & -- & -- & 2.870 & -- & -- & 8.440 \\
-Sri Veeramakaliamman Singapore & \textbf{2.334} & 29.405 & 2.559 & \textbf{0.543} & 3.868 & 0.597 & \textbf{11.965} & 28.951 & 3.470 & \textbf{157} & 157 & -- & \textbf{12392.180} & 12544.690 & -- & -- & -- & 0.175 & -- & -- & 0.040 & -- & -- & 0.730 \\
-Statue Of Liberty & 52.358 & \textbf{39.859} & 46.887 & 28.503 & \textbf{18.503} & 20.012 & 235036.825 & \textbf{41.028} & 26.160 & \textbf{134} & 134 & -- & \textbf{99.900} & 12924 & -- & -- & -- & 9.091 & -- & -- & 4.122 & -- & -- & 6.970 \\
-The Pumpkin & \textbf{25.731} & 26.571 & 94.672 & \textbf{5.552} & 5.665 & 14.890 & \textbf{26.876} & 35.367 & 33.410 & \textbf{196} & 196 & -- & \textbf{4327.273} & 5790.597 & -- & -- & -- & 98.862 & -- & -- & 14.952 & -- & -- & 24.850 \\
-Thian Hook Keng Temple Singapore & \textbf{0.927} & 1.201 & 0.832 & \textbf{0.093} & 0.121 & 0.082 & \textbf{14.915} & 20.789 & 2.750 & \textbf{138} & 138 & -- & \textbf{3298.537} & 4612.373 & -- & -- & -- & 0.081 & -- & -- & 0.008 & -- & -- & 1.130 \\
-Tsar Nikolai I & \textbf{42.803} & 78.413 & 48.499 & \textbf{8.927} & 15.394 & 9.467 & \textbf{11.124} & 18.431 & 9.790 & \textbf{98} & 98 & -- & \textbf{4928.430} & 6646.845 & -- & -- & -- & 36.280 & -- & -- & 7.836 & -- & -- & 6.530 \\
-Urban II & 59.030 & -- & 47.490 & 11.008 & -- & 9.467 & 17.601 & -- & 9.380 & 96 & -- & -- & 8889.690 & -- & -- & -- & -- & 48.214 & -- & -- & 9.586 & -- & -- & 6.920 \\
-Vercingetorix & \textbf{82.952} & 86.088 & 69.328 & \textbf{10.195} & 10.890 & 8.788 & \textbf{7.257} & 8.681 & 5.080 & \textbf{69} & 69 & -- & \textbf{1223.970} & 3105.607 & -- & -- & -- & 17.706 & -- & -- & 3.104 & -- & -- & 1.500 \\
-Yueh Hai Ching Temple Singapore & 0.627 & -- & 0.720 & 0.084 & -- & 0.098 & 5.562 & -- & 0.940 & 43 & -- & -- & 1749.280 & -- & -- & -- & -- & 0.043 & -- & -- & 0.014 & -- & -- & 0.650 \\
+Alcatraz Courtyard & \textbf{0.362} & 0.420 & 10.132 & 0.619 & \textbf{0.093} & 0.110 & 2.219 & 0.160 & \textbf{3.435} & 3.673 & 7.272 & 1.640 & \textbf{133} & 133 & 133 & -- & \textbf{4826.713} & 5684.120 & 6550.983 & -- & 0.044 & -- & -- & 0.049 & 0.017 & -- & -- & 0.015 & 0.965 & -- & -- & 0.810 \\
+Alcatraz Water Tower & \textbf{0.666} & 0.882 & 1.480 & 0.933 & \textbf{0.370} & 0.488 & 0.803 & 0.518 & \textbf{4.763} & 5.112 & 7.397 & 2.130 & \textbf{172} & 172 & 172 & -- & \textbf{2461.337} & 3064.183 & 3795.877 & -- & 0.227 & -- & -- & 0.230 & 0.115 & -- & -- & 0.116 & 0.741 & -- & -- & 0.550 \\
+Buddah Tooth Relic Temple Singapore & \textbf{1.131} & 1.267 & 1.791 & 1.030 & \textbf{0.280} & 0.337 & 0.520 & 0.233 & \textbf{9.964} & 10.743 & 14.029 & 2.060 & \textbf{162} & 162 & 162 & -- & \textbf{2916.837} & 3544.803 & 4289.503 & -- & 0.086 & -- & -- & 0.081 & 0.016 & -- & -- & 0.014 & 2.079 & -- & -- & 0.850 \\
+Doge Palace Venice & \textbf{0.753} & 0.880 & 1.121 & 1.163 & \textbf{0.209} & 0.249 & 0.350 & 0.342 & \textbf{6.359} & 6.831 & 10.496 & 3.620 & \textbf{241} & 241 & 241 & -- & 22522.737 & \textbf{7930.497} & 20767.260 & -- & 0.057 & -- & -- & 0.211 & 0.018 & -- & -- & 0.029 & 1.177 & -- & -- & 1 \\
+Door Lund & \textbf{0.014} & 0.018 & 4.651 & 0.024 & \textbf{0.004} & 0.005 & 0.344 & 0.006 & \textbf{1.812} & 1.891 & 5.545 & 0.320 & \textbf{12} & 12 & 12 & -- & \textbf{4888.363} & 8451.030 & 9255.943 & -- & 0.003 & -- & -- & 0.006 & 0.001 & -- & -- & 0.001 & 0.303 & -- & -- & 0.300 \\
+Drinking Fountain Somewhere In Zurich & \textbf{17.193} & 17.271 & 17.329 & 0.031 & 0.753 & \textbf{0.749} & 0.750 & 0.004 & \textbf{6.058} & 6.129 & 6.675 & 0.330 & \textbf{14} & 14 & 14 & -- & \textbf{883.837} & 2332.817 & 5815.220 & -- & 15.614 & -- & -- & 0.007 & 0.798 & -- & -- & 0.002 & 5.185 & -- & -- & 0.310 \\
+East Indiaman Goteborg & 5.568 & \textbf{5.530} & 7.231 & 3.814 & 0.979 & \textbf{0.978} & 1.364 & 0.621 & \textbf{7.318} & 7.536 & 10.130 & 4.130 & \textbf{179} & 179 & 179 & -- & \textbf{3987.183} & 4385.873 & 5198.093 & -- & 5.245 & -- & -- & 3.117 & 0.993 & -- & -- & 0.509 & 2.648 & -- & -- & 1.850 \\
+Ecole Superior De Guerre & \textbf{0.349} & 2.337 & 25.800 & 0.318 & \textbf{0.090} & 0.545 & 1.324 & 0.081 & \textbf{3.418} & 4.726 & 8.646 & 0.720 & \textbf{35} & 35 & 35 & -- & \textbf{2380.387} & 3932.013 & 10416.080 & -- & 0.014 & -- & -- & 0.024 & 0.003 & -- & -- & 0.005 & 0.355 & -- & -- & 0.340 \\
+Eglise du dome & \textbf{0.386} & 0.569 & 15.675 & 0.808 & \textbf{0.098} & 0.147 & 2.288 & 0.205 & \textbf{3.987} & 4.250 & 6.815 & 0.910 & \textbf{85} & 85 & 85 & -- & 19631.293 & \textbf{8795.890} & 9802.683 & -- & 0.034 & -- & -- & 0.037 & 0.010 & -- & -- & 0.010 & 2.081 & -- & -- & 0.270 \\
+Folke Filbyter & 85.831 & 85.900 & \textbf{82.656} & 74.596 & \textbf{0.129} & 0.129 & 0.131 & 0.125 & \textbf{19.275} & 19.898 & 28.225 & 10.370 & \textbf{40} & 40 & 40 & -- & \textbf{1581.897} & 3072.870 & 3755.193 & -- & 71.440 & -- & -- & 70.157 & 0.130 & -- & -- & 0.118 & 20.014 & -- & -- & 4.290 \\
+Fort Channing Gate Singapore & \textbf{0.093} & 0.117 & 0.154 & 0.207 & \textbf{0.041} & 0.052 & 0.071 & 0.093 & \textbf{1.783} & 1.871 & 3.715 & 0.520 & \textbf{27} & 27 & 27 & -- & \textbf{3434.113} & 5450.200 & 6171.250 & -- & 0.018 & -- & -- & 0.020 & 0.007 & -- & -- & 0.008 & 0.260 & -- & -- & 0.250 \\
+Golden Statue Somewhere In Hong Kong & \textbf{0.232} & 0.267 & 0.508 & 0.292 & \textbf{0.053} & 0.062 & 0.082 & 0.073 & \textbf{1.829} & 1.944 & 3.022 & 0.400 & \textbf{18} & 18 & 18 & -- & \textbf{4095.853} & 7057.423 & 11699.307 & -- & 0.026 & -- & -- & 0.031 & 0.003 & -- & -- & 0.004 & 0.282 & -- & -- & 0.270 \\
+Gustav Vasa & \textbf{3.509} & 4.122 & 25.854 & 34.181 & \textbf{0.220} & 0.249 & 0.775 & 1.085 & \textbf{1.655} & 1.874 & 4.747 & 3.520 & \textbf{18} & 18 & 18 & -- & \textbf{709.867} & 1928.880 & 3887.653 & -- & 1.004 & -- & -- & 32.266 & 0.113 & -- & -- & 1.145 & 0.538 & -- & -- & 3.150 \\
+GustavIIAdolf & \textbf{47.126} & 47.468 & 88.194 & 67.784 & \textbf{8.536} & 8.574 & 13.129 & 9.714 & \textbf{11.722} & 11.918 & 17.220 & 13.910 & \textbf{57} & 57 & 57 & -- & \textbf{1044.170} & 2265.203 & 2981.017 & -- & 24.738 & -- & -- & 58.458 & 5.460 & -- & -- & 8.524 & 16.352 & -- & -- & 11.490 \\
+Jonas Ahlstromer & 48.740 & 48.415 & \textbf{46.584} & 50.190 & 9.823 & \textbf{9.672} & 10.245 & 10.888 & \textbf{11.136} & 11.328 & 11.144 & 10.820 & \textbf{40} & 40 & 40 & -- & \textbf{615.450} & 1712.753 & 2414.653 & -- & 32.847 & -- & -- & 47.117 & 6.311 & -- & -- & 10.451 & 7.502 & -- & -- & 8.410 \\
+Kings College University Of Toronto & \textbf{9.025} & 9.312 & 14.284 & 0.989 & 2.044 & 2.108 & \textbf{1.639} & 0.235 & \textbf{3.565} & 3.732 & 6.351 & 0.900 & \textbf{77} & 77 & 77 & -- & \textbf{1078.980} & 2209.693 & 2857.733 & -- & 5.786 & -- & -- & 0.085 & 1.224 & -- & -- & 0.017 & 1.768 & -- & -- & 0.340 \\
+Lund University Sphinx & \textbf{11.392} & 13.007 & 21.718 & 19.522 & \textbf{2.981} & 3.385 & 4.817 & 4.585 & \textbf{6.056} & 6.760 & 11.550 & 4.780 & \textbf{70} & 70 & 70 & -- & \textbf{2718.567} & 3924.373 & 4642.657 & -- & 7.011 & -- & -- & 8.752 & 1.807 & -- & -- & 2.191 & 1.201 & -- & -- & 1.360 \\
+Nijo Castle Gate & \textbf{0.839} & 1.325 & 3.926 & 1.495 & \textbf{0.165} & 0.255 & 1.138 & 0.286 & \textbf{7.415} & 8.223 & 15.135 & 1.700 & \textbf{19} & 19 & 19 & -- & \textbf{1095.177} & 2629.490 & 3234.357 & -- & 0.081 & -- & -- & 0.069 & 0.013 & -- & -- & 0.012 & 0.761 & -- & -- & 0.730 \\
+Pantheon Paris & -- & \textbf{0.253} & 15.196 & 0.192 & -- & \textbf{0.056} & 2.291 & 0.050 & -- & \textbf{4.893} & 10.638 & 1.470 & -- & \textbf{179} & 179 & -- & -- & \textbf{4640.607} & 5557.600 & -- & -- & -- & -- & 0.040 & -- & -- & -- & 0.005 & -- & -- & -- & 0.490 \\
+Park Gate Clermont Ferrand & 21.388 & \textbf{21.310} & 23.641 & 0.391 & 10.192 & \textbf{10.177} & 11.173 & 0.125 & \textbf{7.712} & 7.747 & 8.652 & 0.570 & \textbf{34} & 34 & 34 & -- & \textbf{1596.940} & 2952.627 & 3650.170 & -- & 10.261 & -- & -- & 0.049 & 4.338 & -- & -- & 0.022 & 14.280 & -- & -- & 0.350 \\
+Plaza De Armas Santiago & \textbf{0.775} & 1.122 & 6.571 & 6.782 & \textbf{0.337} & 0.487 & 2.852 & 2.944 & \textbf{6.009} & 6.327 & 11.199 & 7.400 & \textbf{240} & 240 & 240 & -- & 6372.683 & \textbf{5678.743} & 6616.970 & -- & 0.135 & -- & -- & 2.556 & 0.054 & -- & -- & 1.383 & 2.924 & -- & -- & 4.900 \\
+Porta San Donato Bologna & \textbf{0.592} & 0.610 & 1.814 & 2.153 & \textbf{0.107} & 0.112 & 0.468 & 0.388 & \textbf{5.603} & 5.779 & 9.168 & 2.280 & \textbf{141} & 141 & 141 & -- & \textbf{3754.163} & 4317.957 & 5167.537 & -- & 0.112 & -- & -- & 0.095 & 0.056 & -- & -- & 0.046 & 3.224 & -- & -- & 0.750 \\
+Round Church Cambridge & 2.137 & 2.293 & \textbf{2.097} & 2.451 & 0.927 & 0.967 & \textbf{0.772} & 1.003 & \textbf{6.103} & 6.493 & 8.283 & 2.660 & \textbf{92} & 92 & 92 & -- & \textbf{9779.270} & 10883.870 & 12129.840 & -- & 0.987 & -- & -- & 1.107 & 0.553 & -- & -- & 0.582 & 4.877 & -- & -- & 1.540 \\
+Skansen Kronan Gothenburg & \textbf{0.301} & 0.312 & 0.342 & 0.736 & \textbf{0.102} & 0.106 & 0.118 & 0.226 & \textbf{2.906} & 3.020 & 4.753 & 1.240 & \textbf{131} & 131 & 131 & -- & 17153.677 & \textbf{6127.697} & 10156.743 & -- & 0.025 & -- & -- & 0.026 & 0.008 & -- & -- & 0.008 & 0.682 & -- & -- & 0.670 \\
+Smolny Cathedral St Petersburg & \textbf{21.775} & 21.848 & 22.241 & 0.554 & 2.111 & 2.107 & \textbf{2.096} & 0.051 & \textbf{11.239} & 11.420 & 12.951 & 1.660 & \textbf{131} & 131 & 131 & -- & 10362.787 & \textbf{9384.753} & 10445.090 & -- & 21.416 & -- & -- & 0.033 & 2.158 & -- & -- & 0.006 & 10.310 & -- & -- & 0.810 \\
+Some Cathedral In Barcelona & \textbf{0.651} & 0.787 & 14.741 & 0.880 & \textbf{0.233} & 0.282 & 4.842 & 0.315 & \textbf{6.271} & 6.796 & 15.920 & 2.870 & \textbf{177} & 177 & 177 & -- & 5001.973 & \textbf{4961.633} & 5889.770 & -- & 0.033 & -- & -- & 0.026 & 0.014 & -- & -- & 0.011 & 1.100 & -- & -- & 0.890 \\
+Sri Mariamman Singapore & \textbf{1.219} & 1.747 & 2.113 & 2.302 & \textbf{0.382} & 0.536 & 0.686 & 0.683 & \textbf{10.088} & 11.135 & 13.613 & 4.130 & \textbf{222} & 222 & 222 & -- & 5985.540 & \textbf{5502.090} & 14745.487 & -- & 0.080 & -- & -- & 0.077 & 0.024 & -- & -- & 0.023 & 0.925 & -- & -- & 0.910 \\
+Sri Thendayuthapani Singapore & \textbf{0.593} & 0.700 & 0.784 & 46.269 & \textbf{0.150} & 0.170 & 0.225 & 3.812 & \textbf{7.016} & 7.282 & 9.787 & 23.370 & \textbf{98} & 98 & 98 & -- & 13338.167 & \textbf{12537.777} & 13900.903 & -- & 0.194 & -- & -- & 44.170 & 0.057 & -- & -- & 2.870 & 7.205 & -- & -- & 8.440 \\
+Sri Veeramakaliamman Singapore & 2.404 & 2.721 & \textbf{2.321} & 2.559 & \textbf{0.559} & 0.631 & 0.567 & 0.597 & \textbf{12.234} & 13.394 & 15.852 & 3.470 & \textbf{157} & 157 & 157 & -- & 12377.927 & \textbf{11243.203} & 12563.690 & -- & 0.185 & -- & -- & 0.175 & 0.046 & -- & -- & 0.040 & 2.293 & -- & -- & 0.730 \\
+Statue Of Liberty & \textbf{52.358} & 52.921 & 53.305 & 46.887 & 28.503 & \textbf{19.801} & 21.216 & 20.012 & 235036.825 & \textbf{38.073} & 47.314 & 26.160 & \textbf{134} & 134 & 134 & -- & \textbf{99.900} & 4863.387 & 5582.097 & -- & -- & -- & -- & 9.091 & -- & -- & -- & 4.122 & -- & -- & -- & 6.970 \\
+The Pumpkin & \textbf{25.731} & 26.402 & 47.530 & 94.672 & \textbf{5.552} & 5.691 & 8.677 & 14.890 & \textbf{26.876} & 27.821 & 34.645 & 33.410 & \textbf{196} & 196 & 196 & -- & \textbf{4327.273} & 4944.973 & 8253.960 & -- & -- & -- & -- & 98.862 & -- & -- & -- & 14.952 & -- & -- & -- & 24.850 \\
+Thian Hook Keng Temple Singapore & 0.927 & 0.995 & \textbf{0.920} & 0.832 & 0.093 & 0.100 & \textbf{0.088} & 0.082 & \textbf{14.915} & 15.790 & 17.381 & 2.750 & \textbf{138} & 138 & 138 & -- & \textbf{3298.537} & 3865.320 & 4699.410 & -- & -- & -- & -- & 0.081 & -- & -- & -- & 0.008 & -- & -- & -- & 1.130 \\
+Tsar Nikolai I & \textbf{42.166} & 42.459 & 79.053 & 48.499 & \textbf{8.732} & 8.789 & 15.392 & 9.467 & \textbf{11.210} & 11.461 & 22.553 & 9.790 & \textbf{98} & 98 & 98 & -- & \textbf{4929.200} & 5829.327 & 6680.253 & -- & -- & -- & -- & 36.280 & -- & -- & -- & 7.836 & -- & -- & -- & 6.530 \\
+Urban II & \textbf{58.919} & 59.155 & 72.986 & 47.490 & \textbf{11.006} & 11.077 & 12.107 & 9.467 & \textbf{17.538} & 17.923 & 20.642 & 9.380 & \textbf{96} & 96 & 96 & -- & 8889.617 & \textbf{3932.760} & 6555.440 & -- & -- & -- & -- & 48.214 & -- & -- & -- & 9.586 & -- & -- & -- & 6.920 \\
+Vercingetorix & 82.952 & \textbf{78.854} & 86.219 & 69.328 & 10.195 & \textbf{9.858} & 10.687 & 8.788 & \textbf{7.257} & 7.429 & 10.823 & 5.080 & \textbf{69} & 69 & 69 & -- & \textbf{1223.970} & 2474.027 & 3153.883 & -- & -- & -- & -- & 17.706 & -- & -- & -- & 3.104 & -- & -- & -- & 1.500 \\
+Yueh Hai Ching Temple Singapore & \textbf{0.544} & 0.698 & 0.938 & 0.720 & \textbf{0.075} & 0.095 & 0.130 & 0.098 & \textbf{5.596} & 5.792 & 8.537 & 0.940 & \textbf{43} & 43 & 43 & -- & \textbf{1748.667} & 3054.170 & 3799.240 & -- & -- & -- & -- & 0.043 & -- & -- & -- & 0.014 & -- & -- & -- & 0.650 \\
 \midrule
-Mean & \textbf{16.146} & 23.697 & 18.023 & \textbf{3.121} & 4.014 & 2.912 & 6920.602 & \textbf{13.088} & 5.673 & 100.559 & \textbf{104.818} & -- & \textbf{5819.728} & 7358.187 & -- & -- & -- & 13.695 & -- & -- & 1.937 & -- & -- & 2.992 \\
+Mean & 15.676 & \textbf{15.397} & 22.275 & 17.547 & 3.032 & \textbf{2.754} & 3.788 & 2.840 & 6723.056 & \textbf{9.084} & 12.801 & 5.595 & 102.743 & \textbf{104.861} & 104.861 & -- & 5460.373 & \textbf{5154.640} & 7141.210 & -- & 7.061 & -- & -- & 13.315 & 0.870 & -- & -- & 1.883 & 4.001 & -- & -- & 2.933 \\
 \bottomrule
 \end{tabular}}
 \end{table*}
```
- untracked/modified files:
```
M code/aggregate_single_scene.py
 M code/results/single_scene/REPRO.md
 M code/results/single_scene/summary.csv
 M code/results/single_scene/summary_table.md
 M code/results/single_scene/summary_table.tex
?? MVG_Project_Report_Ortal_Dayan.pdf
?? "RESFM- ROBUST DEEP EQUIVARIANT STRUCTURE FROM MOTION.pdf"
?? "claude specs/SPEC_cvpr_experiments.md"
?? "claude specs/SPEC_sfm_datasets_setup.md"
?? "claude specs/SPEC_single_scene_experiments.md"
?? "claude specs/SPEC_uesfm_combined.md"
?? "claude specs/TASK_crossdataset_readiness.md"
?? code/datasets/Euclidean
?? tmp_ab_check/
```
### esfm-baseline (official ESFM)
- path: `/home/projects/bagon/ortalda/MVG/final-project/esfm-baseline`
- commit: `de666af53423c1dc3dd412e53c4e153c03d70922`
- **WARNING: working tree dirty.** Diff:
```diff
(untracked files only)
```
- untracked/modified files:
```
?? code/confs/single_scene_generated/ss_esfm_Alcatraz_Courtyard_seed1.conf
?? code/confs/single_scene_generated/ss_esfm_Alcatraz_Courtyard_seed2.conf
?? code/confs/single_scene_generated/ss_esfm_Alcatraz_Water_Tower_seed1.conf
?? code/confs/single_scene_generated/ss_esfm_Alcatraz_Water_Tower_seed2.conf
?? code/confs/single_scene_generated/ss_esfm_Buddah_Tooth_Relic_Temple_Singapore_seed1.conf
?? code/confs/single_scene_generated/ss_esfm_Buddah_Tooth_Relic_Temple_Singapore_seed2.conf
?? code/confs/single_scene_generated/ss_esfm_Doge_Palace_Venice_seed1.conf
?? code/confs/single_scene_generated/ss_esfm_Doge_Palace_Venice_seed2.conf
?? code/confs/single_scene_generated/ss_esfm_Door_Lund_seed1.conf
?? code/confs/single_scene_generated/ss_esfm_Door_Lund_seed2.conf
?? code/confs/single_scene_generated/ss_esfm_Drinking_Fountain_Somewhere_In_Zurich_seed1.conf
?? code/confs/single_scene_generated/ss_esfm_Drinking_Fountain_Somewhere_In_Zurich_seed2.conf
?? code/confs/single_scene_generated/ss_esfm_East_Indiaman_Goteborg_seed0.conf
?? code/confs/single_scene_generated/ss_esfm_East_Indiaman_Goteborg_seed1.conf
?? code/confs/single_scene_generated/ss_esfm_East_Indiaman_Goteborg_seed2.conf
?? code/confs/single_scene_generated/ss_esfm_Ecole_Superior_De_Guerre_seed0.conf
?? code/confs/single_scene_generated/ss_esfm_Ecole_Superior_De_Guerre_seed1.conf
?? code/confs/single_scene_generated/ss_esfm_Ecole_Superior_De_Guerre_seed2.conf
?? code/confs/single_scene_generated/ss_esfm_Eglise_du_dome_seed0.conf
?? code/confs/single_scene_generated/ss_esfm_Eglise_du_dome_seed1.conf
?? code/confs/single_scene_generated/ss_esfm_Eglise_du_dome_seed2.conf
?? code/confs/single_scene_generated/ss_esfm_Folke_Filbyter_seed0.conf
?? code/confs/single_scene_generated/ss_esfm_Folke_Filbyter_seed1.conf
?? code/confs/single_scene_generated/ss_esfm_Folke_Filbyter_seed2.conf
?? code/confs/single_scene_generated/ss_esfm_Fort_Channing_Gate_Singapore_seed0.conf
?? code/confs/single_scene_generated/ss_esfm_Fort_Channing_Gate_Singapore_seed1.conf
?? code/confs/single_scene_generated/ss_esfm_Fort_Channing_Gate_Singapore_seed2.conf
?? code/confs/single_scene_generated/ss_esfm_Golden_Statue_Somewhere_In_Hong_Kong_seed0.conf
?? code/confs/single_scene_generated/ss_esfm_Golden_Statue_Somewhere_In_Hong_Kong_seed1.conf
?? code/confs/single_scene_generated/ss_esfm_Golden_Statue_Somewhere_In_Hong_Kong_seed2.conf
?? code/confs/single_scene_generated/ss_esfm_GustavIIAdolf_seed0.conf
?? code/confs/single_scene_generated/ss_esfm_GustavIIAdolf_seed1.conf
?? code/confs/single_scene_generated/ss_esfm_GustavIIAdolf_seed2.conf
?? code/confs/single_scene_generated/ss_esfm_Gustav_Vasa_seed0.conf
?? code/confs/single_scene_generated/ss_esfm_Gustav_Vasa_seed1.conf
?? code/confs/single_scene_generated/ss_esfm_Gustav_Vasa_seed2.conf
?? code/confs/single_scene_generated/ss_esfm_Jonas_Ahlstromer_seed0.conf
?? code/confs/single_scene_generated/ss_esfm_Jonas_Ahlstromer_seed1.conf
?? code/confs/single_scene_generated/ss_esfm_Jonas_Ahlstromer_seed2.conf
?? code/confs/single_scene_generated/ss_esfm_Kings_College_University_Of_Toronto_seed0.conf
?? code/confs/single_scene_generated/ss_esfm_Kings_College_University_Of_Toronto_seed1.conf
?? code/confs/single_scene_generated/ss_esfm_Kings_College_University_Of_Toronto_seed2.conf
?? code/confs/single_scene_generated/ss_esfm_Lund_University_Sphinx_seed0.conf
?? code/confs/single_scene_generated/ss_esfm_Lund_University_Sphinx_seed1.conf
?? code/confs/single_scene_generated/ss_esfm_Lund_University_Sphinx_seed2.conf
?? code/confs/single_scene_generated/ss_esfm_Nijo_Castle_Gate_seed0.conf
?? code/confs/single_scene_generated/ss_esfm_Nijo_Castle_Gate_seed1.conf
?? code/confs/single_scene_generated/ss_esfm_Nijo_Castle_Gate_seed2.conf
?? code/confs/single_scene_generated/ss_esfm_Pantheon_Paris_seed0.conf
?? code/confs/single_scene_generated/ss_esfm_Pantheon_Paris_seed1.conf
?? code/confs/single_scene_generated/ss_esfm_Pantheon_Paris_seed2.conf
?? code/confs/single_scene_generated/ss_esfm_Park_Gate_Clermont_Ferrand_seed0.conf
?? code/confs/single_scene_generated/ss_esfm_Park_Gate_Clermont_Ferrand_seed1.conf
?? code/confs/single_scene_generated/ss_esfm_Park_Gate_Clermont_Ferrand_seed2.conf
?? code/confs/single_scene_generated/ss_esfm_Plaza_De_Armas_Santiago_seed0.conf
?? code/confs/single_scene_generated/ss_esfm_Plaza_De_Armas_Santiago_seed1.conf
?? code/confs/single_scene_generated/ss_esfm_Plaza_De_Armas_Santiago_seed2.conf
?? code/confs/single_scene_generated/ss_esfm_Porta_San_Donato_Bologna_seed0.conf
?? code/confs/single_scene_generated/ss_esfm_Porta_San_Donato_Bologna_seed1.conf
?? code/confs/single_scene_generated/ss_esfm_Porta_San_Donato_Bologna_seed2.conf
?? code/confs/single_scene_generated/ss_esfm_Round_Church_Cambridge_seed0.conf
?? code/confs/single_scene_generated/ss_esfm_Round_Church_Cambridge_seed1.conf
?? code/confs/single_scene_generated/ss_esfm_Round_Church_Cambridge_seed2.conf
?? code/confs/single_scene_generated/ss_esfm_Skansen_Kronan_Gothenburg_seed0.conf
?? code/confs/single_scene_generated/ss_esfm_Skansen_Kronan_Gothenburg_seed1.conf
?? code/confs/single_scene_generated/ss_esfm_Skansen_Kronan_Gothenburg_seed2.conf
?? code/confs/single_scene_generated/ss_esfm_Smolny_Cathedral_St_Petersburg_seed0.conf
?? code/confs/single_scene_generated/ss_esfm_Smolny_Cathedral_St_Petersburg_seed1.conf
?? code/confs/single_scene_generated/ss_esfm_Smolny_Cathedral_St_Petersburg_seed2.conf
?? code/confs/single_scene_generated/ss_esfm_Some_Cathedral_In_Barcelona_seed0.conf
?? code/confs/single_scene_generated/ss_esfm_Some_Cathedral_In_Barcelona_seed1.conf
?? code/confs/single_scene_generated/ss_esfm_Some_Cathedral_In_Barcelona_seed2.conf
?? code/confs/single_scene_generated/ss_esfm_Sri_Mariamman_Singapore_seed0.conf
?? code/confs/single_scene_generated/ss_esfm_Sri_Mariamman_Singapore_seed1.conf
?? code/confs/single_scene_generated/ss_esfm_Sri_Mariamman_Singapore_seed2.conf
?? code/confs/single_scene_generated/ss_esfm_Sri_Thendayuthapani_Singapore_seed0.conf
?? code/confs/single_scene_generated/ss_esfm_Sri_Thendayuthapani_Singapore_seed1.conf
?? code/confs/single_scene_generated/ss_esfm_Sri_Thendayuthapani_Singapore_seed2.conf
?? code/confs/single_scene_generated/ss_esfm_Sri_Veeramakaliamman_Singapore_seed0.conf
?? code/confs/single_scene_generated/ss_esfm_Sri_Veeramakaliamman_Singapore_seed1.conf
?? code/confs/single_scene_generated/ss_esfm_Sri_Veeramakaliamman_Singapore_seed2.conf
?? code/confs/single_scene_generated/ss_esfm_Statue_Of_Liberty_seed0.conf
?? code/confs/single_scene_generated/ss_esfm_Statue_Of_Liberty_seed1.conf
?? code/confs/single_scene_generated/ss_esfm_Statue_Of_Liberty_seed2.conf
?? code/confs/single_scene_generated/ss_esfm_The_Pumpkin_seed0.conf
?? code/confs/single_scene_generated/ss_esfm_The_Pumpkin_seed1.conf
?? code/confs/single_scene_generated/ss_esfm_The_Pumpkin_seed2.conf
?? code/confs/single_scene_generated/ss_esfm_Thian_Hook_Keng_Temple_Singapore_seed0.conf
?? code/confs/single_scene_generated/ss_esfm_Thian_Hook_Keng_Temple_Singapore_seed1.conf
?? code/confs/single_scene_generated/ss_esfm_Thian_Hook_Keng_Temple_Singapore_seed2.conf
?? code/confs/single_scene_generated/ss_esfm_Tsar_Nikolai_I_seed0.conf
?? code/confs/single_scene_generated/ss_esfm_Tsar_Nikolai_I_seed1.conf
?? code/confs/single_scene_generated/ss_esfm_Tsar_Nikolai_I_seed2.conf
?? code/confs/single_scene_generated/ss_esfm_Urban_II_seed0.conf
?? code/confs/single_scene_generated/ss_esfm_Urban_II_seed1.conf
?? code/confs/single_scene_generated/ss_esfm_Urban_II_seed2.conf
?? code/confs/single_scene_generated/ss_esfm_Vercingetorix_seed0.conf
?? code/confs/single_scene_generated/ss_esfm_Vercingetorix_seed1.conf
?? code/confs/single_scene_generated/ss_esfm_Vercingetorix_seed2.conf
?? code/confs/single_scene_generated/ss_esfm_Yueh_Hai_Ching_Temple_Singapore_seed0.conf
?? code/confs/single_scene_generated/ss_esfm_Yueh_Hai_Ching_Temple_Singapore_seed1.conf
?? code/confs/single_scene_generated/ss_esfm_Yueh_Hai_Ching_Temple_Singapore_seed2.conf
```

## Environment (shared venv used for BOTH methods)
- python: `Python 3.9.0`
```

```

## Hardware
- lgn14.wexac.weizmann.ac.il | NVIDIA H100 NVL, 560.35.05, 95830 MiB
- lgn15.wexac.weizmann.ac.il | NVIDIA H100 NVL, 570.195.03, 95830 MiB
- CUDA (login node view): n/a

## Dataset manifest (sha256)
```
799402d5a3d08aad103c1f2fc5cdd8daeaf2eef6edfb60b7ee252d4b738f1719  olsson_download/esfm_datasets.zip
e2058d44460a932c302106143772d8e37cf406cc4537fa0607ed39a9f139ecec  Euclidean/Alcatraz Courtyard.npz
0aaca4dd5c60ad24f2a54ae73f280202d3b8a15f0ca83389429ac55bdd71db8c  Euclidean/Alcatraz Water Tower.npz
d4cac54b5c1ce637022db5481297b63c86cd16b811d5aba60ca5f8809dfa77e0  Euclidean/Buddah Tooth Relic Temple Singapore.npz
42452bfa2cde4e55561340442088d6d3d7d791816277f5c2439e8f457db9c019  Euclidean/Doge Palace Venice.npz
9c38f48796fd435a607c867ca3e82a559bd44e8f4398e80b6e777fba5c4a41d2  Euclidean/Door Lund.npz
0af84736a1fc45d35678d11a14df7168009a1198853489025d2a7dac2b09649a  Euclidean/Drinking Fountain Somewhere In Zurich.npz
9d712bac7a00a4659011bd01c420f482f297d2103fbf92376c9fc8bdc5ef8db3  Euclidean/dtu106.npz
d26068ad67f05937cb3ca22be36fd3ee038205796eb8c56b816e72107d20a7dc  Euclidean/dtu500.npz
de1ea9e7baca66e7f81feebbfbdc1326f40d9c51fd1d74bac12e3c70aa6bb5b3  Euclidean/dtu700.npz
684f11391f85a7e25881579241c16609555b9e4d1371aecab2820d777665f923  Euclidean/East Indiaman Goteborg.npz
ded30b6c81ca8daa7cb517b672a01e6af7b8b58303e092c475bf9ec446488010  Euclidean/Ecole Superior De Guerre.npz
d85e209613e297d454155e79140cadfefaa2e1294f22bbe17f5b339669be0598  Euclidean/Eglise du dome.npz
1c11b2d3759b7ffb5e88f101830289d0437d68b13f327e73ce8c3b5b7c09d7c9  Euclidean/Folke Filbyter.npz
0bca1e48e71bd0cfbbc8c654667b0a309f21a7ef9877bfaca4b7919d5c3da503  Euclidean/Fort Channing Gate Singapore.npz
4d3c1b3dd2d5ffe15c9dede67e58a4cb368207566e68b83f778376d76edd8145  Euclidean/Golden Statue Somewhere In Hong Kong.npz
928dbeb7c9c93d5069c52e0c2c9727e7ed369c321cfb4f8b71e2847ccc2b05bc  Euclidean/GustavIIAdolf.npz
6f7c9e03eae8b6b9dbfcc215a7ceffd9e8135315cb9b24d70db9f1708d56b929  Euclidean/Gustav Vasa.npz
d1bb13397a68cf1c43aa9ea72263b417dc71dc3711c1cca5c34ebbf07031ee04  Euclidean/Jonas Ahlstromer.npz
c2c6697f28c444ba89843b28c11764a1108fcea642d887846405384319f60929  Euclidean/Kings College University Of Toronto.npz
3f4d73d843dc788a42b3485ab114b1b955717ad101723ca557ce5a379f5aa854  Euclidean/Lund University Sphinx.npz
fd5340b1380ffae67ccebe24fb3586c854623e475b30aa2f879d5dc2a846dbe5  Euclidean/Nijo Castle Gate.npz
284ba94ee09e00ef3beae6be8dc8521a5f348c09bf9744b8c648ba2d4f8c962b  Euclidean/Pantheon Paris.npz
fbf4f0e96f40a2070094cd4f70d8bd5900d46fd8e0b93801001eb2830f82bfb8  Euclidean/Park Gate Clermont Ferrand.npz
05f9831454edb44ee524291c8a21222f6cc3ee9666224a3ef1ef869865e199c5  Euclidean/Plaza De Armas Santiago.npz
96851160f3bdb3ddfb65b9151b7527c260f17111bd846d9a89f9f3a353aa7776  Euclidean/Porta San Donato Bologna.npz
e1a737df42c0be87fdd308d99f4d75a1d1fde783745a251c8124dbad21d4ff63  Euclidean/Round Church Cambridge.npz
f8972dd92863010459510bef43858a69b9b1fc8a7bfc2e3b99e5f4f35a225bf5  Euclidean/Skansen Kronan Gothenburg.npz
0a03118e28056177d7d28044f86ab87e3dbc9fea1f6c825ba0837db3a0a1ed07  Euclidean/Smolny Cathedral St Petersburg.npz
e03da80e8e6bf1a2a298222fd65f9f9629c3117ba61aa6721f968a1a8b819a08  Euclidean/Some Cathedral In Barcelona.npz
67bd4fbda49dccdf21fb929bcfd271f0c4e5c3a206b1a7abdeeb15848e851293  Euclidean/Sri Mariamman Singapore.npz
991e5e1dbe48e83f7ee67711d4efc1be4705607fc1d610a58c59dcb581e41d34  Euclidean/Sri Thendayuthapani Singapore.npz
46fcf89db465d455bfa6dca0e0888de0b0c96c9b230663a5f5f3f667de8c8a53  Euclidean/Sri Veeramakaliamman Singapore.npz
bfd68465cedc22b42f3660ec377ae981e78d4a6b2b02d890ed1656a98a441d6e  Euclidean/Statue Of Liberty.npz
f981a9aa5c6c633a175537af217e96fb561324dfdff2d0039c8834b75d7e9b7c  Euclidean/The Pumpkin.npz
db2284ee52e92f5355c5db8d5776308d4ff3157b5dfe07a188fc884a1b68aa21  Euclidean/Thian Hook Keng Temple Singapore.npz
817d1737e50c8bb7a2997baa4ddaba190abb9c0772f213e649409fbd2e1873d0  Euclidean/Tsar Nikolai I.npz
545ddb115d63e7da1ad45d1f4e754d4ca3108d4f4eabdc46dfd4c9f06e816d7c  Euclidean/Urban II.npz
da229f1439d3c77d8ed1d4a3d145c1bda2f27a5a27139b2312b733a5732057b5  Euclidean/Vercingetorix.npz
509221dd42e7f433c05b6be6d51ffda08225995b8318c109c71eb4ea3511a252  Euclidean/Yueh Hai Ching Temple Singapore.npz
```
