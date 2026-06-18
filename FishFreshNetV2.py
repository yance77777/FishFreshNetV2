"""
FishFreshNetV2: Lightweight and Efficient Neural Network for Fish Freshness Assessment
======================================================================================

Quick Start (Local):
    python FishFreshNetV2.py --model fishfreshnet_v2
    python FishFreshNetV2.py --model fishfreshnet_v2_lite
    python FishFreshNetV2.py --model fishfreshnet_v2 --no-cra
    python FishFreshNetV2.py --input-size 192          # Train with 192x192 input

Quick Start (AutoDL):
    # 1. Upload project to /root/autodl-tmp/FishFreshNetV2/
    # 2. Upload MFED to /root/autodl-tmp/MFED and FFE to /root/autodl-tmp/FFE
    # 3. cd /root/autodl-tmp/FishFreshNetV2
    # 4. python FishFreshNetV2.py --model fishfreshnet_v2

Model Variants:
    fishfreshnet_v2       EfficientNet-B0 + ECA + Light CRA
    fishfreshnet_v2_lite  FishFreshNetV2-Lite

Key Parameters (edit below or pass via command line):
    --model         Model variant (default: fishfreshnet_v2)
    --epochs        Training epochs (default: 60)
    --batch-size    Batch size (default: 512, optimized for RTX 5090)
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
    --data-dir      Path to MFED dataset (auto-detected if not specified)
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from fishfreshnet_v2.train import main

if __name__ == "__main__":
    main()
