"""
FishFreshNetV2: Lightweight and Efficient Neural Network for Fish Freshness Assessment
======================================================================================

Quick Start:
    python FishFreshNetV2.py --model fishfreshnet_v2
    python FishFreshNetV2.py --model fishfreshnet_v2_lite
    python FishFreshNetV2.py --model fishfreshnet_v2 --no-cra
    python FishFreshNetV2.py --input-size 192          # Train with 192x192 input

Model Variants:
    fishfreshnet_v2       EfficientNet-B0 + ECA + Light CRA
    fishfreshnet_v2_lite  FishFreshNetV2-Lite

Key Parameters (edit below or pass via command line):
    --model         Model variant (default: fishfreshnet_v2)
    --epochs        Training epochs (default: 60)
    --batch-size    Batch size (default: 512, tune to your GPU memory)
    --learning-rate Learning rate (default: 3e-4)
    --label-smoothing Cross-entropy label smoothing (default: 0.05)
    --grad-clip     Max gradient norm (default: 1.0)
    --scheduler-monitor Monitor val_loss or val_acc (default: val_loss)
    --cra-type      light or original CRA (default: light)
    --no-benchmark  Skip per-run inference timing for faster formal training
    --runs          Number of independent runs (default: 5)
    --input-size    Image resolution (default: 224)
    --no-cra        Disable Circular Region Attention
    --no-pretrained Disable ImageNet pretrained weights
    --no-amp        Disable mixed precision training
    --data-dir      Path to dataset (auto-detected if not specified)
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from fishfreshnet_v2.train import main

if __name__ == "__main__":
    main()
