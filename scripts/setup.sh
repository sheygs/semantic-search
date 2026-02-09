#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/venv"
PYTHON="/usr/bin/python3"
JUPYTER_PORT="${1:-8888}"

echo "=== Semantic Search - GPU Setup ==="
echo "Project directory: ${PROJECT_DIR}"

# ── 1. Request a GPU node via SLURM (skip if already on one) ──
if ! nvidia-smi &>/dev/null; then
    echo ""
    echo "No GPU detected. Requesting a GPU node via SLURM..."
    echo "Re-running this script on a GPU node."
    exec srun \
        --partition=gpu \
        --gres=gpu:1 \
        --cpus-per-task=16 \
        --mem=32G \
        --time=04:00:00 \
        --pty bash -c "source /etc/profile && bash ${SCRIPT_DIR}/setup.sh ${JUPYTER_PORT}"
fi

echo "GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
NODE_NAME=$(hostname)

# ── 2. Load CUDA module ──
echo ""
echo "Loading CUDA 12.4..."
module load cuda-12.4
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PATH="${PATH:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

# ── 3. Create virtual environment ──
if [ -d "${VENV_DIR}" ]; then
    echo ""
    read -rp "Existing venv found. Remove and recreate? [y/N]: " answer
    if [[ "${answer}" =~ ^[Yy]$ ]]; then
        echo "Removing old venv..."
        rm -rf "${VENV_DIR}"
    else
        echo "Keeping existing venv."
    fi
fi

if [ ! -d "${VENV_DIR}" ]; then
    echo ""
    echo "Creating virtual environment with $(${PYTHON} --version)..."
    "${PYTHON}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
echo "Activated venv: ${VENV_DIR}"

# ── 4. Install dependencies ──
echo ""
echo "Upgrading pip..."
pip install --upgrade pip --quiet

echo "Installing PyTorch with CUDA 12.4 support..."
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124 --quiet

echo "Installing project dependencies from requirements.txt..."
pip install -r "${PROJECT_DIR}/requirements.txt" --quiet

echo ""
echo "Verifying installation..."
python3 -c "
import torch
print(f'  PyTorch:  {torch.__version__}')
print(f'  CUDA:     {torch.cuda.is_available()} (device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"})')
import faiss
print(f'  Faiss:    {faiss.__version__} (GPU count: {faiss.get_num_gpus()})')
import sentence_transformers
print(f'  SentenceTransformers: {sentence_transformers.__version__}')
"

# ── 5. Start Jupyter Notebook ──
echo ""
echo "============================================="
echo "  Setup complete!"
echo "============================================="
echo ""
echo "Jupyter will start on ${NODE_NAME}:${JUPYTER_PORT}"
echo ""
echo "To access from your local machine, run:"
echo "  ssh -L ${JUPYTER_PORT}:${NODE_NAME}:${JUPYTER_PORT} <username>@<cluster-login-node>"
echo ""
echo "Then open http://localhost:${JUPYTER_PORT} in your browser."
echo "============================================="
echo ""

jupyter notebook \
    --no-browser \
    --port="${JUPYTER_PORT}" \
    --ip=0.0.0.0 \
    --notebook-dir="${PROJECT_DIR}"
