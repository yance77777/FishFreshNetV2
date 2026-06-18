# FishFreshNetV2

FishFreshNetV2 is a lightweight fish-eye freshness classification project focused on two public model variants:

- `fishfreshnet_v2`: EfficientNet-B0 + Lightweight CRA + ECA.
- `fishfreshnet_v2_lite`: MobileNetV3-Small + ECA.

The public repository contains only the code needed to train, evaluate, benchmark, and visualize FishFreshNetV2 and FishFreshNetV2-Lite. Datasets, trained weights, experiment outputs, manuscripts, notes, and other private project materials are intentionally excluded.

## Repository Layout

```text
fishfreshnet_v2/     Core model, dataset, training, and utility code
scripts/             Training, benchmarking, plotting, and XAI scripts
FishFreshNetV2.py    Main training entry point
requirements.txt     Python dependencies
setup.sh             Optional AutoDL/Linux setup helper
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Train FishFreshNetV2:

```bash
python FishFreshNetV2.py --model fishfreshnet_v2 --data-dir "Multistage Fish Eye Dataset"
```

Train FishFreshNetV2-Lite:

```bash
python FishFreshNetV2.py --model fishfreshnet_v2_lite --data-dir "Multistage Fish Eye Dataset"
```

Generate publication-style result assets from a local run directory:

```bash
python scripts/generate_publication_assets.py --run-root runs/v2_suite_20260617_190256 --output-dir runs/v2_suite_20260617_190256/publication_assets
python scripts/generate_xai_visualizations.py --data-dir "Multistage Fish Eye Dataset" --output-dir runs/v2_suite_20260617_190256/publication_assets/xai --device cpu
```

## Open Source Scope

This repository does not include:

- datasets such as MFED or FFE;
- trained `.pth` weights;
- `runs/` experiment outputs;
- manuscripts, notes, or review materials;
- private application materials or packaged binaries.

Place datasets locally using the class-folder layout expected by the training code: `Highly Fresh/`, `Fresh/`, and `Not Fresh/`.
