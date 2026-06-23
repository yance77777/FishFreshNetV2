#!/bin/bash
# FishFreshNetV2 setup helper for Linux/macOS.
# Verifies dependencies, GPU availability, and dataset layout.
#
# Usage:
#   chmod +x setup.sh && bash setup.sh
#
# Expected dataset layout (one of):
#   Multistage Fish Eye Dataset/Highly Fresh/
#   Multistage Fish Eye Dataset/Fresh/
#   Multistage Fish Eye Dataset/Not Fresh/
#   (or the same names with underscores, e.g. Multistage_Fish_Eye_Dataset)

set -e

echo "============================================"
echo "  FishFreshNetV2 Setup"
echo "============================================"

echo ""
echo "[1/3] Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "[2/3] Verifying GPU..."
python -c "
import torch
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
else:
    print('  WARNING: No GPU detected. Training will run on CPU.')
"

echo ""
echo "[3/3] Checking dataset..."
DATASET_DIR="Multistage Fish Eye Dataset"
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
    echo "  WARNING: Dataset not found in the current directory."
    echo "  Place your dataset folder here or pass --data-dir explicitly."
fi

echo ""
echo "============================================"
echo "  Setup Complete"
echo "============================================"
echo ""
echo "To start training:"
echo "  python FishFreshNetV2.py --model fishfreshnet_v2"
echo ""
echo "If the dataset is elsewhere, specify it:"
echo "  python FishFreshNetV2.py --model fishfreshnet_v2 --data-dir /path/to/dataset"
echo ""
echo "To train both public variants:"
echo "  for m in fishfreshnet_v2 fishfreshnet_v2_lite; do"
echo "    python FishFreshNetV2.py --model \$m --output-dir runs/\$m"
echo "  done"
