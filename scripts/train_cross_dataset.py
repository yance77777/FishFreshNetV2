"""Train and evaluate FishFreshNetV2 across MFED and FFE datasets.

Examples:
  python scripts/train_cross_dataset.py --experiment all
  python scripts/train_cross_dataset.py --experiment mfed_to_ffe --runs 5
  python scripts/train_cross_dataset.py --experiment ffe_indomain --model fishfreshnet_v2_lite
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support

from fishfreshnet_v2.data import (
    CLASS_NAMES,
    create_dataloaders,
    create_full_dataloader,
    create_split_indices,
    create_subset_dataloader,
    describe_dataset,
    get_labels_from_dataset,
)
from fishfreshnet_v2.models import build_model
from fishfreshnet_v2.train import model_complexity, prepare_device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = torch.cuda.is_available()


def autocast_context(use_amp: bool):
    return torch.amp.autocast("cuda") if use_amp else nullcontext()


def move_inputs(x: torch.Tensor, device: torch.device, channels_last: bool) -> torch.Tensor:
    x = x.to(device, non_blocking=device.type == "cuda")
    if channels_last and device.type == "cuda":
        x = x.contiguous(memory_format=torch.channels_last)
    return x


def evaluate(model: nn.Module, loader, device: torch.device, use_amp: bool, channels_last: bool) -> dict[str, float]:
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = move_inputs(inputs, device, channels_last)
            labels = labels.to(device, non_blocking=device.type == "cuda")
            with autocast_context(use_amp):
                outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    per_class = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
    }
    for i, name in enumerate(CLASS_NAMES):
        metrics[f"{name} Recall"] = per_class[1][i]
        metrics[f"{name} F1"] = per_class[2][i]
    return metrics


def build_trainable_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    model = build_model(
        args.model,
        num_classes=len(CLASS_NAMES),
        pretrained=not args.no_pretrained,
        attention=args.attention,
        use_cra=not args.no_cra,
        cra_type=args.cra_type,
        cra_rings=args.cra_rings,
        fusion_channels=args.fusion_channels,
    ).to(device)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model


MODEL_DISPLAY_NAMES = {
    "fishfreshnet_v2": "FishFreshNetV2",
    "fishfreshnet_v2_lite": "FishFreshNetV2-Lite",
}


def train_on_split(
    args: argparse.Namespace,
    train_dir: Path,
    split: dict[str, list[int]],
    device: torch.device,
    seed: int,
) -> nn.Module:
    set_seed(seed)
    loaders = create_dataloaders(
        data_dir=train_dir,
        split_indices=split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        input_size=args.input_size,
        prefetch_factor=args.prefetch_factor,
        cache_images=args.cache_images,
    )
    model = build_trainable_model(args, device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    best_weights = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        for phase in ("train", "val"):
            model.train(phase == "train")
            running_loss = 0.0
            for inputs, labels in loaders[phase]:
                inputs = move_inputs(inputs, device, args.channels_last)
                labels = labels.to(device, non_blocking=device.type == "cuda")
                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase == "train"):
                    with autocast_context(use_amp):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    if phase == "train":
                        if use_amp:
                            scaler.scale(loss).backward()
                            if args.grad_clip > 0:
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            if args.grad_clip > 0:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                            optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(loaders[phase].dataset)
            if phase == "val":
                scheduler.step(epoch_loss)
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_weights = copy.deepcopy(model.state_dict())

        if epoch == 0 or (epoch + 1) % args.log_interval == 0:
            print(f"seed={seed} epoch={epoch + 1}/{args.epochs} val_loss={best_val_loss:.4f}")

    model.load_state_dict(best_weights)
    return model


def create_eval_loader(
    args: argparse.Namespace,
    train_dir: Path,
    test_dir: Path,
    split: dict[str, list[int]],
    device: torch.device,
):
    """Use held-out test split in-domain, and full target dataset cross-domain."""
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "input_size": args.input_size,
        "prefetch_factor": args.prefetch_factor,
        "cache_images": args.cache_images,
    }
    if train_dir.resolve() == test_dir.resolve():
        return create_subset_dataloader(test_dir, split["test"], **loader_args), "heldout_test"
    return create_full_dataloader(test_dir, **loader_args), "full_target"


def run_experiment(args: argparse.Namespace, name: str, train_dir: Path, test_dir: Path) -> list[dict[str, float]]:
    device = prepare_device()
    labels = get_labels_from_dataset(train_dir)
    splits = create_split_indices(len(labels), labels, runs=args.runs, seed=args.seed)
    rows = []

    print("=" * 72)
    print(f"Experiment: {name}")
    print(f"Train: {train_dir} {describe_dataset(train_dir)}")
    print(f"Test : {test_dir} {describe_dataset(test_dir)}")
    print("=" * 72)

    for run_index, split in enumerate(splits):
        seed = args.seed + run_index
        model = train_on_split(args, train_dir, split, device, seed)
        test_loader, eval_protocol = create_eval_loader(args, train_dir, test_dir, split, device)
        metrics = evaluate(
            model,
            test_loader,
            device,
            use_amp=device.type == "cuda" and not args.no_amp,
            channels_last=args.channels_last,
        )
        params_m, flops_g = model_complexity(model, device, args.input_size)
        row = {
            "Experiment": name,
            "Run": run_index + 1,
            "Model": MODEL_DISPLAY_NAMES.get(args.model, args.model),
            "Train Dataset": train_dir.name,
            "Test Dataset": test_dir.name,
            "Eval Protocol": eval_protocol,
            "Params (M)": params_m,
            "FLOPs (G)": flops_g,
            **metrics,
        }
        rows.append(row)
        print(f"run={run_index + 1} acc={metrics['Accuracy']:.4f} macro_f1={metrics['F1-Score']:.4f}")

    return rows


def save_results(rows: list[dict[str, float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "all_runs_metrics.csv", index=False)
    numeric_cols = [c for c in df.columns if c not in {"Experiment", "Model", "Train Dataset", "Test Dataset", "Eval Protocol"}]
    summary = df.groupby("Experiment")[numeric_cols].agg(["mean", "std"])
    summary.to_csv(output_dir / "summary_metrics.csv")

    lines = ["# Cross-Dataset Results", ""]
    for experiment, group in df.groupby("Experiment"):
        lines.append(f"## {experiment}")
        lines.append("")
        protocols = ", ".join(sorted(group["Eval Protocol"].dropna().unique())) if "Eval Protocol" in group else "unknown"
        lines.append(f"Evaluation protocol: {protocols}")
        lines.append("")
        lines.append("| Metric | Mean +/- Std |")
        lines.append("|--------|------------|")
        for col in ["Accuracy", "Balanced Accuracy", "F1-Score", "Precision", "Recall", "Params (M)", "FLOPs (G)"]:
            if col in group.columns:
                mean = group[col].mean()
                std = group[col].std()
                if pd.isna(mean):
                    value = "N/A"
                elif pd.isna(std):
                    value = f"{mean * 100:.2f}%" if col not in {"Params (M)", "FLOPs (G)"} else f"{mean:.3f}"
                elif col in {"Params (M)", "FLOPs (G)"}:
                    value = f"{mean:.3f} +/- {std:.3f}"
                else:
                    value = f"{mean * 100:.2f} +/- {std * 100:.2f}%"
                lines.append(f"| {col} | {value} |")
        lines.append("")
    (output_dir / "summary_table.md").write_text("\n".join(lines), encoding="utf-8")


def find_dataset(default_name: str, candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists() and (path / "Highly Fresh").is_dir():
            return path
    return Path(default_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-dataset MFED/FFE experiments")
    parser.add_argument(
        "--mfed-dir",
        type=Path,
        default=find_dataset(
            "Multistage Fish Eye Dataset",
            [
                Path("/root/autodl-tmp/MFED"),
                Path("/root/autodl-tmp/Multistage Fish Eye Dataset"),
                Path("/root/autodl-tmo/MFED"),
                Path("/root/autodl-tmo/Multistage Fish Eye Dataset"),
                Path("MFED"),
                Path("Multistage Fish Eye Dataset"),
            ],
        ),
    )
    parser.add_argument(
        "--ffe-dir",
        type=Path,
        default=find_dataset(
            "FFE dataset",
            [
                Path("/root/autodl-tmp/FFE"),
                Path("/root/autodl-tmp/FFE dataset"),
                Path("/root/autodl-tmo/FFE"),
                Path("/root/autodl-tmo/FFE dataset"),
                Path("FFE"),
                Path("FFE dataset"),
            ],
        ),
    )
    parser.add_argument(
        "--experiment",
        choices=["all", "mfed_indomain", "ffe_indomain", "mfed_to_ffe", "ffe_to_mfed"],
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/cross_dataset"))
    parser.add_argument(
        "--model",
        default="fishfreshnet_v2",
        choices=[
            "fishfreshnet_v2",
            "fishfreshnet_v2_lite",
        ],
    )
    parser.add_argument("--attention", default="eca", choices=["eca", "se", "cbam", "none"])
    parser.add_argument("--no-cra", action="store_true")
    parser.add_argument("--cra-type", default="light", choices=["light", "original"])
    parser.add_argument("--cra-rings", type=int, default=3)
    parser.add_argument("--fusion-channels", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=min(24, os.cpu_count() or 4))
    parser.add_argument("--prefetch-factor", type=int, default=8)
    parser.add_argument("--cache-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--log-interval", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments = {
        "mfed_indomain": (args.mfed_dir, args.mfed_dir),
        "ffe_indomain": (args.ffe_dir, args.ffe_dir),
        "mfed_to_ffe": (args.mfed_dir, args.ffe_dir),
        "ffe_to_mfed": (args.ffe_dir, args.mfed_dir),
    }
    selected = experiments.keys() if args.experiment == "all" else [args.experiment]

    rows = []
    for name in selected:
        train_dir, test_dir = experiments[name]
        rows.extend(run_experiment(args, name, train_dir.resolve(), test_dir.resolve()))
    save_results(rows, args.output_dir)
    print(f"Saved cross-dataset results to {args.output_dir}")


if __name__ == "__main__":
    main()
