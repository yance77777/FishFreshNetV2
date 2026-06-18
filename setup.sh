#!/bin/bash
# FishFreshNetV2 AutoDL/Linux setup helper.
# Expected layout:
#   /root/autodl-tmp/FishFreshNetV2/
#   /root/autodl-tmp/Multistage Fish Eye Dataset/
# Usage:
#   cd /root/autodl-tmp/FishFreshNetV2 && chmod +x setup.sh && bash setup.sh

set -e

PROJECT_DIR="/root/autodl-tmp/FishFreshNetV2"
DATASET_DIR="/root/autodl-tmp/Multistage Fish Eye Dataset"

echo "============================================"
echo "  FishFreshNetV2 AutoDL Setup"
echo "============================================"

cd "$PROJECT_DIR" || {
    echo "ERROR: Project not found at $PROJECT_DIR"
    echo "Please upload the project to /root/autodl-tmp/FishFreshNetV2/"
    exit 1
}

echo ""
echo "[1/4] Installing dependencies..."
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

echo ""
echo "[2/4] Verifying GPU..."
python -c "
import torch
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
else:
    print('  WARNING: No GPU detected.')
"

echo ""
echo "[3/4] Checking dataset..."
if [ -d "$DATASET_DIR/Highly Fresh" ]; then
    N_HIGH=$(ls "$DATASET_DIR/Highly Fresh/" | wc -l)
    N_FRESH=$(ls "$DATASET_DIR/Fresh/" | wc -l)
    N_NOT=$(ls "$DATASET_DIR/Not Fresh/" | wc -l)
    echo "  Dataset found at: $DATASET_DIR"
    echo "  Highly Fresh: $N_HIGH images"
    echo "  Fresh:        $N_FRESH images"
    echo "  Not Fresh:    $N_NOT images"
    echo "  Total:        $((N_HIGH + N_FRESH + N_NOT)) images"
else
    echo "  WARNING: Dataset not found at $DATASET_DIR"
    echo "  Please upload the MFED dataset to /root/autodl-tmp/"
fi

echo ""
echo "[4/4] Running quick test (1 epoch, 1 run)..."
python FishFreshNetV2.py --model fishfreshnet_v2 --epochs 1 --runs 1 --output-dir artifacts/setup_test

echo ""
echo "============================================"
echo "  Setup Complete"
echo "============================================"
echo ""
echo "Project dir : $PROJECT_DIR"
echo "Dataset dir : $DATASET_DIR"
echo ""
echo "To start full training:"
echo "  cd $PROJECT_DIR"
echo "  python FishFreshNetV2.py --model fishfreshnet_v2"
echo ""
echo "To train both public variants:"
echo "  for m in fishfreshnet_v2 fishfreshnet_v2_lite; do"
echo "    python FishFreshNetV2.py --model \$m --output-dir runs/\$m"
echo "  done"
echo ""
echo "To train in background:"
echo "  screen -S train"
echo "  python FishFreshNetV2.py --model fishfreshnet_v2"
echo "  # Press Ctrl+A then D to detach"
echo "  # screen -r train to reconnect"
