# U-ESfM 

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Unsupervised Robust Structure from Motion via Self-Supervised Outlier Handling and Deep Equivariant Network**

This repository contains the official implementation of U-RESfM, developed as part of the Multiple View Geometry (MVG) course project at the Weizmann Institute of Science.

## Abstract

We investigate deeper equivariant architectures and unsupervised outlier handling methods for Structure from Motion on scenes with varying outlier characteristics (11.7-43.9%). Our evaluation on 9 MegaDepth scenes reveals that **network depth is the most consistent performance driver**, with deep architectures achieving **47% mean translation error improvement** over baseline ESFM in 7 of 9 scenes without performance degradation.

Combining deep architectures with adaptive confidence-weighted outlier loss achieves the **lowest overall mean translation error (53% improvement)**, with transformative results on high-outlier scenes (>25% outliers).

## Key Contributions

1. **Deep Equivariant Architecture** — Networks with 10+ layers using:
   - Residual connections for gradient flow
   - Sparse Layer Normalization for sparse tensors
   - Sparse Dropout for regularization

2. **Unsupervised Outlier Handling** — No labeled data required:
   - MAD-based threshold (robust to heavy outlier contamination)
   - STD-based threshold (for approximately Gaussian distributions)
   - Huber-style soft weighting (continuous down-weighting of outliers)

3. **Adaptive Confidence-Weighted Outlier Loss** — Enables multi-scene learning:
   - Dynamic per-scene adaptive thresholds using percentiles
   - Self-supervised pseudo-labeling for outlier detection
   - Warmup schedule for training stability

## Results

### Translation Error (Lower is Better)

| Method | Mean Error | Improvement |
|--------|-----------|-------------|
| ESFM (baseline) | 1.055 | — |
| ESFM + MAD | 0.939 | 11% |
| Deep ESFM | 0.548 | **47%** |
| Deep RESfM + OutlierLoss | 0.493 | **53%** |

### Rotation Error (Lower is Better)

| Method | Mean Error | Improvement |
|--------|-----------|-------------|
| ESFM (baseline) | 5.884° | — |
| ESFM + MAD | 3.943° | **33%** |
| Deep ESFM | 4.363° | 26% |
| Deep RESfM OutlierLoss | 5.696° | 3% |

### Key Findings

- **For translation accuracy**: Deep architectures provide the most consistent improvement
- **For rotation accuracy**: Statistical MAD-based outlier removal outperforms deep methods
- **Scene-dependent effects**: Deep RESfM OutlierLoss excels on high-outlier scenes (>25%) but may degrade on low-outlier scenes (<21%)

## Installation

```bash
# Clone the repository
git clone https://github.com/OrtalDayan/u-resfm.git
cd u-resfm

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install with PyTorch CUDA support
uv venv
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
uv pip install -e 
```


## Requirements

- Python 3.8+
- PyTorch 2.x with CUDA
- CUDA 11.8+ (for GPU training)

See `pyproject.toml` for full dependencies.

## Usage

### Train Original ESFM (Baseline)

```bash
./run_single_scene_optimization.sh \
  --queue risk --num_epochs 1e5 --eval_intervals 5e2 \
  --architecture_type esfm_outliers_deep \
  --scans "0007,0015,0024,0147,0327,0411,1001,0063,0104" \
  --scheduler_milestone "5000,7000,9000" \
  --block_number 1 --block_size 3 --lr 1e-4
```

### Train Deep ESFM

```bash
./run_single_scene_optimization.sh \
  --queue risk --num_epochs 1e5 --eval_intervals 5e2 \
  --architecture_type esfm_outliers_deep \
  --scans "0007,0015,0024,0147,0327,0411,1001,0063,0104" \
  --scheduler_milestone "5000,7000,9000" \
  --block_number 5 --block_size 2 --lr 1e-4
```

### Two-Stage with MAD/STD/Huber Outlier Handling

**Stage 1:** Run either ESFM or Deep ESFM command above.

**Stage 2:** Run with outlier weighting:

```bash
./run_single_scene_optimization.sh \
  --queue risk --num_epochs 1e5 --eval_intervals 5e2 \
  --architecture_type esfm_outliers_deep \
  --scans "0007,0015,0024,0147,0327,0411,1001,0063,0104" \
  --scheduler_milestone "5000,7000,9000" \
  --block_number 1 --block_size 3 --lr 1e-4 \
  --weight_method mad --alpha 2
```

Options for `--weight_method`: `mad`, `std`, `huber`

### Deep RESfM + Adaptive Confidence-Weighted Outlier Loss

```bash
./run_single_scene_optimization.sh \
  --queue risk --num_epochs 1e5 --eval_intervals 5e2 \
  --architecture_type esfm_outliers_deep \
  --scans "0007,0015,0024,0147,0327,0411,1001,0063,0104" \
  --scheduler_milestone "5000,7000,9000" \
  --block_number 1 --block_size 3 --lr 1e-4 \
  --loss_function CombinedLoss \
  --reproj_loss_weight 1.0 --classification_loss_weight 1.0
```



## Dataset

We use the [MegaDepth dataset](https://www.cs.cornell.edu/projects/megadepth/) for evaluation. The dataset provides:
- Dense 2D point correspondences across multiple views
- Ground truth camera poses for evaluation
- Challenging outdoor scenes with varying illumination

### Scenes Used

| Scene | Images | Outlier % |
|-------|--------|-----------|
| 7 | <1000 | 11.7% |
| 63 | >1000 | 14.5% |
| 104 | >1000 | 16.2% |
| 15 | >1000 | 20.6% |
| 327 | >1000 | 21.0% |
| 24 | >1000 | 23.0% |
| 147 | >1000 | 24.6% |
| 411 | >1000 | 29.9% |
| 1001 | >1000 | 43.9% |

## Architecture

### 1. ESFM (Baseline)
<img src="images/1_esfm_baseline.png" alt="ESFM Architecture" width="40%">

### 2. Deep ESFM
<img src="images/2_deep_esfm.png" alt="ESFM Architecture" width="40%">

### 3. Two-Stage ESFM + Outlier Handling
<img src="images/3_two_stage_esfm.png" alt="ESFM Architecture" width="40%">

### 4. RESfM + Adaptive Confidence-Weighted Outlier Loss
<img src="images/4_resfm_outlierloss.png" alt="ESFM Architecture" width="40%">


### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 1e-4 |
| Epochs | 10,000 |
| Number of Features | 256 |
| Scheduler Milestones | [5000, 7000, 9000] |

### Outlier Handling

| Method | Parameter | Recommended Value |
|--------|-----------|-------------------|
| MAD | δ | 2.0 |
| STD | δ | 3.0 |
| Huber | δ | 1.0 |


## Project Structure

```
u-resfm/
├── code/
│   ├── confs/                     # Configuration files
│   ├── datasets/                  # Dataset utilities
│   ├── hyperparameter_tuning/     # Hyperparameter tuning scripts
│   ├── lsf_output/                # LSF cluster job outputs
│   ├── megadepth/                 # MegaDepth dataset processing
│   ├── models/                    # Network architectures
│   ├── results/                   # Experiment results
│   ├── utils/                     # Utility functions
│   ├── evaluation.py              # Evaluation & outlier detection functions
│   ├── loss_functions.py          # Loss functions including CombinedLoss
│   ├── run_single_scene_optimization.sh
│   ├── SceneData.py
│   ├── ScenesDataSet.py
│   └── single_scene_optimization.py  # Main training script
├── .gitignore
├── .python-version
├── pyproject.toml
└── uv.lock
```

## Method Selection Guide

| Scenario | Recommended Method |
|----------|-------------------|
| Unknown scene characteristics | Deep ESFM |
| High outlier scenes (>25%) | Deep RESfM OutlierLoss |
| Low outlier scenes (<21%) | Deep ESFM |
| Rotation accuracy priority | ESFM + MAD or Deep ESFM + MAD |
| Translation accuracy priority | Deep RESfM OutlierLoss |

## Citation

```bibtex
@misc{dayan2025uresfm,
  author = {Dayan, Ortal-Shabnam},
  title = {U-RESfM: Unsupervised Robust Structure from Motion via Self-Supervised Outlier Handling and Deep Equivariant Network},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/OrtalDayan/u-resfm}}
}
```

## Acknowledgments

This work builds upon:
- [ESFM](https://github.com/drormoran/ESFM) by Moran et al. (ICCV 2021) — Deep Permutation Equivariant Structure from Motion
- [RESfM](https://github.com/) by Khatib et al. (2024) — Robust Equivariant Structure from Motion
- [MegaDepth](https://www.cs.cornell.edu/projects/megadepth/) dataset by Li & Snavely (CVPR 2018)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Ortal-Shabnam Dayan**  
Deep Learning Researcher, Core AI Research Center  
Weizmann Institute of Science  
[LinkedIn](https://www.linkedin.com/in/ortaldayan/)# u-resfm
