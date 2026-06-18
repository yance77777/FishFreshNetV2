"""Generate public result tables and figures for FishFreshNetV2 variants."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_LABELS = {
    "fishfreshnet_v2": "FishFreshNetV2",
    "fishfreshnet_v2_lite": "FishFreshNetV2-Lite",
}

PUB_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.4,
}

COLORS = {
    "FishFreshNetV2": "#3A6EA5",
    "FishFreshNetV2-Lite": "#2A9D8F",
}


def load_metrics(path: Path) -> pd.DataFrame | None:
    csv_path = path / "all_runs_metrics.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def collect_public_results(run_root: Path) -> pd.DataFrame:
    search_roots = [run_root / "formal", run_root]
    rows = []
    seen = set()

    for parent in search_roots:
        if not parent.exists():
            continue
        for model_key, label in MODEL_LABELS.items():
            if model_key in seen:
                continue
            df = load_metrics(parent / model_key)
            if df is None:
                continue
            row = {"Model": label}
            for col in ["Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1-Score", "Params (M)", "FLOPs (G)"]:
                if col in df.columns:
                    row[f"{col} Mean"] = df[col].mean()
                    row[f"{col} Std"] = df[col].std()
            rows.append(row)
            seen.add(model_key)

    return pd.DataFrame(rows)


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=False)
    try:
        path.write_text(df.to_markdown(index=False), encoding="utf-8")
    except ImportError:
        path.write_text(df.to_string(index=False), encoding="utf-8")


def make_results_table(results: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    for _, row in results.iterrows():
        rows.append(
            {
                "Model": row["Model"],
                "Accuracy (%)": f"{row.get('Accuracy Mean', np.nan) * 100:.2f}",
                "F1-Score (%)": f"{row.get('F1-Score Mean', np.nan) * 100:.2f}",
                "Params (M)": f"{row.get('Params (M) Mean', np.nan):.3f}",
                "FLOPs (G)": "N/A" if pd.isna(row.get("FLOPs (G) Mean", np.nan)) else f"{row.get('FLOPs (G) Mean'):.3f}",
            }
        )
    save_table(pd.DataFrame(rows), output_dir / "table_public_results.md")


def plot_accuracy(results: pd.DataFrame, output_dir: Path) -> None:
    if results.empty or "Accuracy Mean" not in results.columns:
        return
    plot_df = results.copy()
    plot_df["Accuracy (%)"] = plot_df["Accuracy Mean"] * 100

    with plt.rc_context(PUB_STYLE):
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        x = np.arange(len(plot_df))
        colors = [COLORS.get(model, "#8E8E8E") for model in plot_df["Model"]]
        bars = ax.bar(x, plot_df["Accuracy (%)"], color=colors, edgecolor="#222222", linewidth=0.7, width=0.55)
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_ylim(max(90, plot_df["Accuracy (%)"].min() - 2), 100.5)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["Model"])
        ax.grid(axis="y", alpha=0.35, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, acc in zip(bars, plot_df["Accuracy (%)"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15, f"{acc:.2f}%", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        fig.savefig(output_dir / "fig_public_accuracy.png")
        fig.savefig(output_dir / "fig_public_accuracy.pdf")
        plt.close(fig)


def plot_complexity(results: pd.DataFrame, output_dir: Path) -> None:
    if results.empty or "Params (M) Mean" not in results.columns:
        return
    plot_df = results.copy()

    with plt.rc_context(PUB_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
        y = np.arange(len(plot_df))
        colors = [COLORS.get(model, "#8E8E8E") for model in plot_df["Model"]]

        axes[0].barh(y, plot_df["Params (M) Mean"], color=colors, edgecolor="#222222", linewidth=0.7, height=0.5)
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(plot_df["Model"])
        axes[0].set_xlabel("Parameters (M)")
        axes[0].invert_yaxis()
        axes[0].grid(axis="x", alpha=0.3, linestyle="--")

        if "FLOPs (G) Mean" in plot_df.columns:
            axes[1].barh(y, plot_df["FLOPs (G) Mean"], color=colors, edgecolor="#222222", linewidth=0.7, height=0.5)
            axes[1].set_yticks(y)
            axes[1].set_yticklabels([""] * len(plot_df))
            axes[1].set_xlabel("FLOPs (G)")
            axes[1].invert_yaxis()
            axes[1].grid(axis="x", alpha=0.3, linestyle="--")
        else:
            axes[1].axis("off")

        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.tight_layout()
        fig.savefig(output_dir / "fig_public_complexity.png")
        fig.savefig(output_dir / "fig_public_complexity.pdf")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate public FishFreshNetV2 result assets")
    parser.add_argument("--run-root", type=Path, default=Path("runs/v2_suite_latest"))
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.run_root / "public_assets"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = collect_public_results(args.run_root)
    if results.empty:
        raise SystemExit(f"No public model metrics found under {args.run_root}")

    make_results_table(results, output_dir)
    plot_accuracy(results, output_dir)
    plot_complexity(results, output_dir)
    print(f"Public assets saved to {output_dir}")


if __name__ == "__main__":
    main()
