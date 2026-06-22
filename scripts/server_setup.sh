#!/bin/bash
# VLA-Robotics server one-click setup
# Usage: bash scripts/server_setup.sh

set -e
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
ENV_NAME="VLM_Robotics"

echo "============================================"
echo " VLA-Robotics Server Setup"
echo " Project: $PROJECT_DIR"
echo " Date:    $(date)"
echo " GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'N/A')"
echo "============================================"

# ---- 1. Conda environment ----
echo ""
echo "[1/4] Conda environment..."
if command -v conda &> /dev/null; then
    if conda info --envs 2>/dev/null | grep -q "$ENV_NAME"; then
        echo "  Env '$ENV_NAME' exists, updating..."
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$ENV_NAME"
    else
        echo "  Creating env '$ENV_NAME'..."
        conda create -n "$ENV_NAME" python=3.11 -y
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$ENV_NAME"
    fi
else
    echo "  No conda, using venv..."
    python3 -m venv venv
    source venv/bin/activate
fi

# ---- 2. Dependencies ----
echo ""
echo "[2/4] Installing dependencies..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 -q
pip install -r "$PROJECT_DIR/requirements.txt" -q
pip install qwen-vl-utils rosbags opencv-python pyarrow -q
echo "  Done. PyTorch CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

# ---- 3. Model check ----
echo ""
echo "[3/4] Model check..."
MODEL_DIR="$PROJECT_DIR/model-embedding"
if [ -f "$MODEL_DIR/config.json" ]; then
    echo "  Model found: $MODEL_DIR ($(du -sh $MODEL_DIR | cut -f1))"
else
    echo "  ERROR: Model not found at $MODEL_DIR"
    echo "  Please upload model-embedding/ directory to server first."
    exit 1
fi

# ---- 4. Data ----
echo ""
echo "[4/4] Data preparation..."
echo "  Download RoboMIND Franka data:"
echo "    python scripts/download_robomind.py --robot franka"
echo ""
echo "  Convert to training format:"
echo "    python scripts/convert_robomind.py --hdf5 data/raw/robomind/data/franka/trajectory.hdf5 --output data/processed/metadata.json"
echo ""
echo "============================================"
echo " Setup complete!"
echo "============================================"
echo ""
echo " Next steps:"
echo "  1. Download data:  python scripts/download_let_tasks.py --all-claw --max-bags 10"
echo "  2. Convert data:   python scripts/convert_let_bag.py --bag-dir data/raw/let_dataset/quick_sort-P4-claw --output data/processed/metadata.json"
echo "  3. Train:          bash scripts/run_train.sh"
echo ""
