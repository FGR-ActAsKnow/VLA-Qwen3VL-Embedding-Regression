#!/bin/bash
# Launch VLA training
# Usage: bash scripts/run_train.sh [--resume checkpoints/checkpoint-best]

set -e
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# Activate conda
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate VLM_Robotics 2>/dev/null || true
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

mkdir -p logs checkpoints

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/train_${TIMESTAMP}.log"

NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
echo "GPUs: $NUM_GPUS"
echo "Log:  $LOG_FILE"

CMD="python scripts/train.py --config configs/config.yaml --metadata data/processed/metadata.json"

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Launching DeepSpeed ($NUM_GPUS GPUs)..."
    deepspeed --num_gpus=$NUM_GPUS scripts/train.py \
        --config configs/config.yaml \
        --metadata data/processed/metadata.json \
        --deepspeed
else
    echo "Launching single-GPU..."
    echo "Command: $CMD"
    nohup $CMD >> "$LOG_FILE" 2>&1 &
    PID=$!
    echo "PID: $PID"
    echo "Monitor: tail -f $LOG_FILE"
    echo "Stop:   kill $PID"
fi
