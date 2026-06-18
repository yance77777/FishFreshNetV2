"""Training, evaluation, and benchmarking for FishFreshNetV2."""

import argparse
import copy
import os
import random
import warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", message=".*tracemalloc.*")
from contextlib import nullcontext
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .data import CLASS_NAMES, MFEDImageFolder, create_dataloaders, create_split_indices, get_labels_from_dataset
from .models import build_model

try:
    from thop import profile
except ImportError:
    profile = None


# Publication-quality plot style
PUB_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
}


MODEL_DISPLAY_NAMES = {
    "fishfreshnet_v2": "FishFreshNetV2",
    "fishfreshnet_v2_lite": "FishFreshNetV2-Lite",
}


CRA_CAPABLE_MODELS = {"fishfreshnet_v2"}


def _find_mfed_dataset() -> Path | None:
    """Auto-detect MFED dataset in common locations.

    Searches in:
    - Current working directory (local dev / same dir as project)
    - Parent directory (local dev)
    - /root/autodl-tmp/ (AutoDL data disk - recommended)
    - /root/ (AutoDL system disk)
    """
    candidates = [
        Path("/root/autodl-tmp/MFED"),
        Path("/root/autodl-tmp/Multistage Fish Eye Dataset"),
        Path("/root/autodl-tmo/MFED"),
        Path("/root/autodl-tmo/Multistage Fish Eye Dataset"),
        # Same directory as project (autodl-tmp or local)
        Path.cwd() / "Multistage Fish Eye Dataset",
        Path.cwd() / "MFED",
        # Parent directory
        Path.cwd().parent / "Multistage Fish Eye Dataset",
        Path.cwd().parent / "MFED",
        # AutoDL system disk fallback
        Path("/root/Multistage Fish Eye Dataset"),
        Path("/root/MFED"),
    ]
    for p in candidates:
        if p.exists() and (p / "Highly Fresh").is_dir():
            return p.resolve()
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train FishFreshNetV2 on MFED.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python FishFreshNetV2.py                           # FishFreshNetV2: EffB0 + Light CRA + ECA
  python FishFreshNetV2.py --model fishfreshnet_v2_lite  # FishFreshNetV2-Lite
  python FishFreshNetV2.py --model fishfreshnet_v2 --no-cra  # V2 without CRA
  python FishFreshNetV2.py --input-size 192 --epochs 30  # Quick experiment
  python FishFreshNetV2.py --learning-rate 3e-4 --label-smoothing 0.05
        """,
    )
    default_data = _find_mfed_dataset()
    parser.add_argument("--data-dir", type=Path,
                        default=default_data,
                        help="Path to MFED root directory. Auto-detected if not specified.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/fishfreshnet_v2"))
    parser.add_argument("--model", type=str, default="fishfreshnet_v2",
                        choices=[
                            "fishfreshnet_v2",
                            "fishfreshnet_v2_lite",
                        ],
                        help="Model variant: fishfreshnet_v2 or fishfreshnet_v2_lite")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs (default: 60)")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size (default: 512, RTX 5090)")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    parser.add_argument("--label-smoothing", type=float, default=0.05,
                        help="Cross-entropy label smoothing. Use 0 to disable (default: 0.05)")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Max gradient norm. Use 0 to disable (default: 1.0)")
    parser.add_argument("--scheduler-monitor", type=str, default="val_loss",
                        choices=["val_loss", "val_acc"],
                        help="Metric monitored by ReduceLROnPlateau and checkpointing")
    parser.add_argument("--runs", type=int, default=5, help="Number of independent runs (default: 5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-size", type=int, default=224, help="Input image size (default: 224)")
    parser.add_argument("--attention", type=str, default="eca", choices=["eca", "se", "cbam", "none"],
                        help="Attention mechanism (default: eca)")
    parser.add_argument("--use-cra", action="store_true", default=True)
    parser.add_argument("--no-cra", action="store_true", help="Disable Circular Region Attention")
    parser.add_argument("--cra-rings", type=int, default=3, help="Number of CRA rings (default: 3)")
    parser.add_argument("--cra-type", type=str, default="light", choices=["light", "original"],
                        help="CRA implementation for models that support it (default: light)")
    parser.add_argument("--fusion-channels", type=int, default=256,
                        help="Output channels for multi-scale fusion in V2 (default: 256)")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretrained weights")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for faster training")
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True,
                        help="Use channels_last memory format on CUDA (default: true)")
    parser.add_argument("--no-benchmark", action="store_true",
                        help="Skip inference latency benchmark at the end of each run")
    parser.add_argument("--save-plots", action=argparse.BooleanOptionalAction, default=True,
                        help="Save learning curves and confusion matrices (default: true)")
    parser.add_argument("--cache-images", action=argparse.BooleanOptionalAction, default=True,
                        help="Cache decoded MFED images in RAM to improve GPU utilization (default: true)")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=8)
    return parser.parse_args()


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = torch.cuda.is_available() and not deterministic


def prepare_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    return device


def move_inputs(inputs: torch.Tensor, device: torch.device, channels_last: bool = True) -> torch.Tensor:
    inputs = inputs.to(device, non_blocking=device.type == "cuda")
    if channels_last and device.type == "cuda" and inputs.ndim == 4:
        return inputs.contiguous(memory_format=torch.channels_last)
    return inputs


def autocast_context(use_amp: bool):
    if use_amp:
        return torch.amp.autocast("cuda")
    return nullcontext()


def model_complexity(model: nn.Module, device: torch.device, input_size: int = 224) -> tuple[float | None, float | None]:
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    if profile is None:
        return params_m, None
    was_training = model.training
    model.eval()
    dummy = torch.randn(1, 3, input_size, input_size, device=device)
    if device.type == "cuda":
        dummy = dummy.contiguous(memory_format=torch.channels_last)
    flops, params = profile(model, inputs=(dummy,), verbose=False)
    model.train(was_training)
    return params / 1e6, flops / 1e9


def save_learning_curves(history: dict[str, list[float]], run_index: int, output_dir: Path, model_name: str = "") -> None:  # noqa: ARG001
    """Save publication-quality learning curves."""
    with plt.rc_context(PUB_STYLE):
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Loss curve
        axes[0].plot(epochs, history["train_loss"], marker="o", markersize=4, label="Train", color="#2171b5")
        axes[0].plot(epochs, history["val_loss"], marker="s", markersize=4, label="Validation", color="#cb181d")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend(frameon=True, edgecolor="0.8", fancybox=False)
        axes[0].grid(True, alpha=0.3)

        # Accuracy curve
        axes[1].plot(epochs, np.array(history["train_acc"]) * 100, marker="o", markersize=4, label="Train", color="#2171b5")
        axes[1].plot(epochs, np.array(history["val_acc"]) * 100, marker="s", markersize=4, label="Validation", color="#cb181d")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_ylim(0, 100)
        axes[1].yaxis.set_major_locator(mticker.MultipleLocator(20))
        axes[1].legend(frameon=True, edgecolor="0.8", fancybox=False)
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_dir / f"learning_curves_run_{run_index + 1}.png")
        plt.close(fig)


def save_confusion_matrix(y_true: list[int], y_pred: list[int], run_index: int, output_dir: Path, model_name: str = "") -> None:  # noqa: ARG001
    """Save publication-quality normalized confusion matrix."""
    with plt.rc_context(PUB_STYLE):
        matrix = confusion_matrix(y_true, y_pred, normalize="true")
        fig, ax = plt.subplots(figsize=(5, 4.5))
        sns.heatmap(
            matrix, annot=True, fmt=".3f", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            vmin=0.0, vmax=1.0, square=True, ax=ax,
            linewidths=0.5, linecolor="white",
            cbar_kws={"shrink": 0.8, "label": "Proportion"},
            annot_kws={"size": 11},
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        fig.tight_layout()
        fig.savefig(output_dir / f"confusion_matrix_run_{run_index + 1}.png")
        plt.close(fig)


def evaluate(
    model: nn.Module,
    dataloader,
    device: torch.device,
    use_amp: bool,
    channels_last: bool = True,
) -> tuple[dict[str, float], list[int], list[int]]:
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = move_inputs(inputs, device, channels_last)
            labels = labels.to(device, non_blocking=device.type == "cuda")
            with autocast_context(use_amp):
                outputs = model(inputs)
            predictions = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
    }, y_true, y_pred


def train_one_run(
    args: argparse.Namespace,
    split_indices: dict[str, list[int]],
    run_index: int,
    device: torch.device,
) -> dict[str, float]:
    display_name = MODEL_DISPLAY_NAMES.get(args.model, args.model)
    dataloaders = create_dataloaders(
        data_dir=args.data_dir,
        split_indices=split_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        input_size=args.input_size,
        prefetch_factor=args.prefetch_factor,
        cache_images=args.cache_images,
    )

    use_cra = args.use_cra and not args.no_cra
    model = build_model(
        model_name=args.model,
        num_classes=len(CLASS_NAMES),
        pretrained=not args.no_pretrained,
        attention=args.attention,
        use_cra=use_cra,
        cra_rings=args.cra_rings,
        cra_type=args.cra_type,
        fusion_channels=args.fusion_channels,
    ).to(device)

    # Keep raw model for complexity measurement (torch.compile breaks thop)
    raw_model = model

    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if args.compile and device.type == "cuda" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="default")
            print("  torch.compile() enabled")
        except Exception:
            print("  torch.compile() skipped (not supported)")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler_mode = "min" if args.scheduler_monitor == "val_loss" else "max"
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=scheduler_mode, factor=0.5, patience=5)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_weights = copy.deepcopy(model.state_dict())
    best_score = float("inf") if args.scheduler_monitor == "val_loss" else -float("inf")

    for epoch in range(args.epochs):
        for phase in ["train", "val"]:
            model.train(phase == "train")
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = move_inputs(inputs, device, args.channels_last)
                labels = labels.to(device, non_blocking=device.type == "cuda")
                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(phase == "train"):
                    with autocast_context(use_amp):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    predictions = outputs.argmax(dim=1)
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
                running_corrects += (predictions == labels).sum().item()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)
            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)

            if phase == "val":
                monitor_value = epoch_loss if args.scheduler_monitor == "val_loss" else epoch_acc
                scheduler.step(monitor_value)
                is_better = monitor_value < best_score if args.scheduler_monitor == "val_loss" else monitor_value > best_score
                if is_better:
                    best_score = monitor_value
                    best_weights = copy.deepcopy(model.state_dict())

        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(
                f"Run {run_index + 1} | Epoch {epoch + 1}/{args.epochs} | "
                f"train_loss={history['train_loss'][-1]:.4f} train_acc={history['train_acc'][-1]:.4f} | "
                f"val_loss={history['val_loss'][-1]:.4f} val_acc={history['val_acc'][-1]:.4f}"
            )

    model.load_state_dict(best_weights)
    metrics, y_true, y_pred = evaluate(model, dataloaders["test"], device, use_amp, args.channels_last)
    params_m, flops_g = model_complexity(raw_model, device, args.input_size)

    # Measure inference time
    from .utils import measure_inference_time
    timing = {} if args.no_benchmark else measure_inference_time(
        model,
        device,
        args.input_size,
        channels_last=args.channels_last,
    )

    if args.save_plots:
        save_learning_curves(history, run_index, args.output_dir, display_name)
        save_confusion_matrix(y_true, y_pred, run_index, args.output_dir, display_name)
    safe_model_name = display_name.lower().replace("+", "plus").replace(" ", "_").replace("-", "_")
    torch.save(best_weights, args.output_dir / f"best_{safe_model_name}_run{run_index + 1}.pth",
               _use_new_zipfile_serialization=True)

    row = {
        "Run": run_index + 1,
        "Params (M)": params_m,
        "FLOPs (G)": flops_g,
        "CPU ms/img": timing.get("cpu_ms"),
        "GPU ms/img": timing.get("gpu_ms"),
        **metrics,
    }
    params_str = f"{params_m:.2f}M" if params_m is not None else "N/A"
    flops_str = f"{flops_g:.4f}G" if flops_g is not None else "N/A"
    print(
        f"Run {run_index + 1} Test | {display_name} | "
        f"accuracy={metrics['Accuracy']:.4f} f1={metrics['F1-Score']:.4f} | "
        f"params={params_str} flops={flops_str}"
    )
    return row


def _format_mean_std(mean: float, std: float, decimals: int = 2) -> str:
    """Format mean ± std for publication tables."""
    if pd.isna(std) or std == 0:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def save_summary_table(metrics_df: pd.DataFrame, output_dir: Path, model_name: str) -> None:
    """Generate a publication-ready LaTeX and Markdown summary table."""
    numeric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "Params (M)", "FLOPs (G)", "CPU ms/img"]
    existing_cols = [c for c in numeric_cols if c in metrics_df.columns]

    means = metrics_df[existing_cols].mean()
    stds = metrics_df[existing_cols].std()

    # Markdown table
    lines = [f"### {model_name} Results (mean ± std, {len(metrics_df)} runs)\n"]
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for col in existing_cols:
        if col in ("Accuracy", "Precision", "Recall", "F1-Score"):
            lines.append(f"| {col} (%) | {_format_mean_std(means[col]*100, stds[col]*100, 2)} |")
        elif col == "Params (M)":
            lines.append(f"| {col} | {_format_mean_std(means[col], stds[col], 2)} |")
        elif col == "FLOPs (G)":
            lines.append(f"| {col} | {_format_mean_std(means[col], stds[col], 4)} |")
        else:
            lines.append(f"| {col} | {_format_mean_std(means[col], stds[col], 1)} |")
    lines.append("")

    md_path = output_dir / "summary_table.md"
    with open(md_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # LaTeX table snippet
    latex_lines = [
        f"% {model_name} results",
        "\\begin{tabular}{lc}",
        "\\toprule",
        "Metric & Value \\\\",
        "\\midrule",
    ]
    for col in existing_cols:
        if col in ("Accuracy", "Precision", "Recall", "F1-Score"):
            val = _format_mean_std(means[col]*100, stds[col]*100, 2) + "\\%"
        elif col == "Params (M)":
            val = _format_mean_std(means[col], stds[col], 2) + "M"
        elif col == "FLOPs (G)":
            val = _format_mean_std(means[col], stds[col], 4) + "G"
        else:
            val = _format_mean_std(means[col], stds[col], 1)
        latex_lines.append(f"{col} & {val} \\\\")
    latex_lines += ["\\bottomrule", "\\end{tabular}", ""]

    tex_path = output_dir / "summary_table.tex"
    with open(tex_path, "a", encoding="utf-8") as f:
        f.write("\n".join(latex_lines) + "\n")


def main() -> None:
    args = parse_args()

    if args.data_dir is None:
        raise FileNotFoundError(
            "MFED dataset not found. Please specify --data-dir manually.\n"
            "Example: python FishFreshNetV2.py --data-dir /path/to/MFED"
        )
    args.data_dir = args.data_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.num_workers is None:
        args.num_workers = min(24, os.cpu_count() or 4)

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {args.data_dir}")

    set_seed(args.seed)
    sns.set_theme(style="whitegrid", context="paper")
    device = prepare_device()

    labels = get_labels_from_dataset(args.data_dir)
    dataset_size = len(labels)
    split_cache = create_split_indices(dataset_size, labels, runs=args.runs, seed=args.seed)

    print("=" * 60)
    print(f"  FishFreshNetV2 Training")
    print("=" * 60)
    display_name = MODEL_DISPLAY_NAMES.get(args.model, args.model)
    print(f"  Model     : {display_name}")
    print(f"  Dataset   : {args.data_dir} ({dataset_size} samples)")
    print(f"  Classes   : {CLASS_NAMES}")
    print(f"  Device    : {device}")
    print(f"  Runs      : {args.runs}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR        : {args.learning_rate}")
    print(f"  Label sm. : {args.label_smoothing}")
    print(f"  Grad clip : {args.grad_clip}")
    print(f"  Scheduler : {args.scheduler_monitor}")
    print(f"  Input size: {args.input_size}")
    effective_cra = args.model in CRA_CAPABLE_MODELS and not args.no_cra
    print(f"  CRA       : {effective_cra} (rings={args.cra_rings})")
    print(f"  CRA type  : {args.cra_type}")
    print(f"  Attention : {args.attention}")
    print(f"  Workers   : {args.num_workers}")
    print(f"  Prefetch  : {args.prefetch_factor}")
    print(f"  Cache img : {args.cache_images}")
    print(f"  Benchmark : {not args.no_benchmark}")
    print("=" * 60)

    rows = []
    for run_index, split_indices in enumerate(split_cache):
        set_seed(args.seed + run_index)
        rows.append(train_one_run(args, split_indices, run_index, device))

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(args.output_dir / "all_runs_metrics.csv", index=False)

    # Generate summary statistics
    summary = metrics_df.drop(columns=["Run"]).agg(["mean", "std"]).T
    summary.to_csv(args.output_dir / "summary_metrics.csv")

    # Generate publication-quality tables
    save_summary_table(metrics_df, args.output_dir, display_name)

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(summary.to_string())
    print("=" * 60)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - all_runs_metrics.csv")
    print(f"  - summary_metrics.csv")
    print(f"  - summary_table.md (Markdown)")
    print(f"  - summary_table.tex (LaTeX)")
    print(f"  - learning_curves_run_*.png")
    print(f"  - confusion_matrix_run_*.png")
    print(f"  - best_{display_name.lower().replace('+', 'plus').replace(' ', '_').replace('-', '_')}_run_*.pth")


if __name__ == "__main__":
    main()
