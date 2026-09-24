set -e
cd /home/projects/bagon/ortalda/MVG/final-project/u-esfm
export PATH="$HOME/.local/bin:$PATH"
uv venv .venv-gasfm --python 3.10
source .venv-gasfm/bin/activate
# modern torch (H100-compatible cu121) — sidesteps GASFM's pinned cu116
uv pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
# core deps for GASFM inference (no ceres, no pytorch3d)
uv pip install torch_geometric numpy scipy pandas opencv-python-headless pyhocon openpyxl scikit-learn matplotlib tqdm
# compiled PyG extensions matched to torch 2.3.0+cu121
uv pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
python -c "import torch, torch_geometric, torch_scatter, torch_sparse; print('torch',torch.__version__,'pyg ok')"
