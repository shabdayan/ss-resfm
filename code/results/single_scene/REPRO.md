# Reproducibility record — single-scene ESFM vs U-ESFM (Olsson, calibrated)
Generated: 2026-07-13T20:41:04

## Fairness policy (SPEC ground rules)
- Identical inputs: both methods read the same shared `u-esfm/datasets/Euclidean/<scene>.npz` (symlinked into both repos).
- Post-processing: BOTH methods optimized with `ba.run_ba = false`; pre-BA numbers come from each method's raw cameras. Post-BA numbers (when present) come from ONE shared pycolmap BA applied by `evaluate_single_scene.py --ba` to both methods identically. Neither method's own BA was used. Pre- and post-BA are reported separately.
- Identical budgets: same epochs / eval intervals / lr / schedule per run (recorded in each run's `run.conf` and below).
- Same hardware, sequential: for each (scene, seed) the two methods run back-to-back in the same LSF job on the same GPU (see per-run `run_meta.json` host/gpu fields).
- Seeds: per-run `random_seed` in the conf; raw per-seed numbers kept in `summary.csv`.

## Sweep actually run
- methods: ['esfm', 'uesfm']
- scenes (35): ['Alcatraz Courtyard', 'Alcatraz Water Tower', 'Buddah Tooth Relic Temple Singapore', 'Doge Palace Venice', 'Door Lund', 'Drinking Fountain Somewhere In Zurich', 'East Indiaman Goteborg', 'Ecole Superior De Guerre', 'Eglise du dome', 'Folke Filbyter', 'Fort Channing Gate Singapore', 'Golden Statue Somewhere In Hong Kong', 'Gustav Vasa', 'GustavIIAdolf', 'Jonas Ahlstromer', 'Kings College University Of Toronto', 'Lund University Sphinx', 'Nijo Castle Gate', 'Pantheon Paris', 'Park Gate Clermont Ferrand', 'Plaza De Armas Santiago', 'Porta San Donato Bologna', 'Round Church Cambridge', 'Skansen Kronan Gothenburg', 'Smolny Cathedral St Petersburg', 'Sri Mariamman Singapore', 'Sri Thendayuthapani Singapore', 'Sri Veeramakaliamman Singapore', 'Statue Of Liberty', 'The Pumpkin', 'Thian Hook Keng Temple Singapore', 'Tsar Nikolai I', 'Urban II', 'Vercingetorix', 'Yueh Hai Ching Temple Singapore']
- seeds: [0, 1, 2]

## Budgets / exact commands
### ESFM
- epochs: 100000  eval_intervals: 5000
- example command: `/home/projects/bagon/ortalda/MVG/final-project/u-esfm/.venv/bin/python single_scene_optimization.py --conf single_scene_generated/ss_esfm_Jonas_Ahlstromer_seed1.conf --exp_version Jonas_Ahlstromer_seed1` (cwd: `/home/projects/bagon/ortalda/MVG/final-project/esfm-baseline/code`)
- full config: `<run_dir>/run.conf` in every run directory
### U-ESFM
- epochs: 100000  eval_intervals: 5000
- example command: `/home/projects/bagon/ortalda/MVG/final-project/u-esfm/.venv/bin/python single_scene_optimization.py --conf confs/single_scene_generated/ss_uesfm_Jonas_Ahlstromer_seed1.conf --wandb 0 --stage 1 --architecture_type single_scene_bench --results_aggregation_file /home/projects/bagon/ortalda/MVG/final-project/u-esfm/code/results/single_scene/uesfm/Jonas_Ahlstromer/seed1/raw/aggregated_results.xlsx` (cwd: `/home/projects/bagon/ortalda/MVG/final-project/u-esfm/code`)
- full config: `<run_dir>/run.conf` in every run directory

## Code versions
### u-esfm (U-ESFM)
- path: `/home/projects/bagon/ortalda/MVG/final-project/u-esfm`
- commit: `3f871108d3e843b4698d6af48adb12f7ad835f6d`
- **WARNING: working tree dirty.** Diff:
```diff
(untracked files only)
```
- untracked/modified files:
```
?? MVG_Project_Report_Ortal_Dayan.pdf
?? "RESFM- ROBUST DEEP EQUIVARIANT STRUCTURE FROM MOTION.pdf"
?? "claude specs/SPEC_cvpr_experiments.md"
?? "claude specs/SPEC_sfm_datasets_setup.md"
?? "claude specs/SPEC_single_scene_experiments.md"
?? "claude specs/SPEC_uesfm_combined.md"
?? code/datasets/Euclidean
?? tmp_ab_check/
```
### esfm-baseline (official ESFM)
- path: `/home/projects/bagon/ortalda/MVG/final-project/esfm-baseline`
- commit: `2c0079497bf5dda1eeb2a902b2f859c6387f213a`
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
?? code/confs/single_scene_generated/ss_esfm_Smolny_Cathedral_St_Petersburg_seed0.conf
?? code/confs/single_scene_generated/ss_esfm_Smolny_Cathedral_St_Petersburg_seed1.conf
?? code/confs/single_scene_generated/ss_esfm_Some_Cathedral_In_Barcelona_seed0.conf
?? code/confs/single_scene_generated/ss_esfm_Sri_Mariamman_Singapore_seed0.conf
?? code/confs/single_scene_generated/ss_esfm_Sri_Mariamman_Singapore_seed1.conf
?? code/confs/single_scene_generated/ss_esfm_Sri_Mariamman_Singapore_seed2.conf
?? code/confs/single_scene_generated/ss_esfm_Sri_Thendayuthapani_Singapore_seed0.conf
?? code/confs/single_scene_generated/ss_esfm_Sri_Thendayuthapani_Singapore_seed1.conf
?? code/confs/single_scene_generated/ss_esfm_Sri_Veeramakaliamman_Singapore_seed0.conf
?? code/confs/single_scene_generated/ss_esfm_Sri_Veeramakaliamman_Singapore_seed1.conf
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
?? code/confs/single_scene_generated/ss_esfm_Vercingetorix_seed0.conf
?? code/confs/single_scene_generated/ss_esfm_Vercingetorix_seed1.conf
?? code/confs/single_scene_generated/ss_esfm_Vercingetorix_seed2.conf
?? code/confs/single_scene_generated/ss_esfm_Yueh_Hai_Ching_Temple_Singapore_seed0.conf
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
