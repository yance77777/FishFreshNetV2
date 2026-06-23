"""Generate Grad-CAM and CRA ring-importance figures from trained V2 weights."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from fishfreshnet_v2.models import build_model


CLASS_NAMES = ["Highly Fresh", "Fresh", "Not Fresh"]
CLASS_COLORS = {
    "Highly Fresh": "#2A9D8F",
    "Fresh": "#3A6EA5",
    "Not Fresh": "#E76F51",
}
RING_LABELS = ["Center\n(Pupil)", "Middle\n(Iris)", "Outer\n(Sclera)"]

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
}


def load_state(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)


def build_trained_model(model_name: str, weights: Path, device: torch.device) -> torch.nn.Module:
    model = build_model(
        model_name,
        num_classes=3,
        pretrained=False,
        attention="eca",
        cra_type="light",
        cra_rings=3,
    )
    load_state(model, weights, device)
    model.to(device)
    model.eval()
    return model


def image_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def collect_images(data_dir: Path, per_class: int) -> list[tuple[Path, str]]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    samples: list[tuple[Path, str]] = []
    for class_name in CLASS_NAMES:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            continue
        images = [p for p in sorted(class_dir.iterdir()) if p.suffix.lower() in suffixes]
        if not images:
            continue
        if len(images) <= per_class:
            chosen = images
        else:
            idx = np.linspace(0, len(images) - 1, per_class, dtype=int)
            chosen = [images[i] for i in idx]
        samples.extend((path, class_name) for path in chosen)
    return samples


def normalize_cam(cam: torch.Tensor) -> torch.Tensor:
    cam = cam.detach()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam


def gradcam_for_model(model: torch.nn.Module, tensor: torch.Tensor) -> tuple[np.ndarray, int, np.ndarray]:
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_module, _inputs, output):
        activations.append(output)

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer = model.features[-1]
    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0].detach().cpu().numpy()
        pred = int(outputs.argmax(dim=1).item())
        outputs[0, pred].backward()
        act = activations[-1]
        grad = gradients[-1]
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * act).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
        return normalize_cam(cam).cpu().numpy(), pred, probs
    finally:
        handle_f.remove()
        handle_b.remove()


def overlay_cam(image: Image.Image, cam: np.ndarray, alpha: float = 0.42) -> Image.Image:
    base = image.resize((224, 224)).convert("RGB")
    base_np = np.asarray(base).astype(np.float32) / 255.0
    cmap = plt.get_cmap("turbo")
    heat = cmap(cam)[..., :3].astype(np.float32)
    overlay = np.clip((1 - alpha) * base_np + alpha * heat, 0, 1)
    return Image.fromarray((overlay * 255).astype(np.uint8))


def v2_ring_importance(model: torch.nn.Module, tensor: torch.Tensor) -> tuple[np.ndarray, int]:
    features = model.features(tensor)
    features.retain_grad()
    x = model.cra(features) if getattr(model, "use_cra", False) else features
    x = model.attention(x)
    outputs = model.classifier(x)
    pred = int(outputs.argmax(dim=1).item())
    model.zero_grad(set_to_none=True)
    outputs[0, pred].backward()

    grad = features.grad
    weights = grad.mean(dim=(2, 3), keepdim=True)
    saliency = torch.relu((weights * features).sum(dim=1, keepdim=False))[0]
    masks = model.cra._create_ring_masks(saliency.shape[0], saliency.shape[1], saliency.device)
    vals = []
    for mask in masks:
        m = mask[0, 0]
        vals.append(float((saliency * m).sum().detach().cpu().item()))
    vals_np = np.array(vals, dtype=np.float64)
    if vals_np.sum() <= 1e-12:
        vals_np = np.ones_like(vals_np) / len(vals_np)
    else:
        vals_np = vals_np / vals_np.sum()
    return vals_np, pred


def plot_ring_importance(records: list[dict], output_dir: Path) -> None:
    df = pd.DataFrame(records)
    df.to_csv(output_dir / "cra_ring_importance.csv", index=False)
    means = df.groupby("true_class")[["center", "middle", "outer"]].mean().reindex(CLASS_NAMES)

    with plt.rc_context(PUB_STYLE):
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        x = np.arange(3)
        width = 0.24
        offsets = [-width, 0, width]
        colors = [CLASS_COLORS[c] for c in CLASS_NAMES]
        for i, class_name in enumerate(CLASS_NAMES):
            vals = means.loc[class_name].to_numpy(dtype=float)
            ax.bar(
                x + offsets[i],
                vals,
                width=width,
                label=class_name,
                color=colors[i],
                edgecolor="#222222",
                linewidth=0.6,
            )
            for j, value in enumerate(vals):
                ax.text(x[j] + offsets[i], value + 0.01, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(RING_LABELS)
        ax.set_ylabel("Gradient-Weighted Importance")
        ax.set_ylim(0, max(0.75, float(means.to_numpy().max()) + 0.08))
        ax.set_title("CRA Ring Importance on Real Fish-Eye Images", pad=10)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(framealpha=0.95)
        fig.tight_layout()
        fig.savefig(output_dir / "fig_cra_ring_importance.png")
        fig.savefig(output_dir / "fig_cra_ring_importance.pdf")
        plt.close(fig)


def plot_gradcam_grid(rows: list[dict], output_dir: Path) -> None:
    if not rows:
        return
    with plt.rc_context(PUB_STYLE):
        fig, axes = plt.subplots(len(rows), 3, figsize=(8.2, 2.35 * len(rows)))
        if len(rows) == 1:
            axes = np.expand_dims(axes, axis=0)
        for r, row in enumerate(rows):
            original = Image.open(row["path"]).convert("RGB").resize((224, 224))
            axes[r, 0].imshow(original)
            axes[r, 0].set_title(f"Original\n{row['true_class']}")
            axes[r, 1].imshow(row["lite_overlay"])
            axes[r, 1].set_title(f"V2-Lite Grad-CAM\nPred: {CLASS_NAMES[row['lite_pred']]}")
            axes[r, 2].imshow(row["v2_overlay"])
            axes[r, 2].set_title(f"FishFreshNetV2 Grad-CAM\nPred: {CLASS_NAMES[row['v2_pred']]}")
            for c in range(3):
                axes[r, c].axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / "fig_gradcam_examples.png")
        fig.savefig(output_dir / "fig_gradcam_examples.pdf")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate XAI figures for FishFreshNetV2")
    parser.add_argument("--v2-weights", type=Path, default=Path("runs/best_fishfreshnetv2.pth"))
    parser.add_argument("--lite-weights", type=Path, default=Path("runs/best_fishfreshnetv2_lite.pth"))
    parser.add_argument("--data-dir", type=Path, default=Path("Multistage Fish Eye Dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/xai"))
    parser.add_argument("--per-class", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    transform = image_transform()

    v2 = build_trained_model("fishfreshnet_v2", args.v2_weights, device)
    lite = build_trained_model("fishfreshnet_v2_lite", args.lite_weights, device)
    samples = collect_images(args.data_dir, args.per_class)
    if not samples:
        raise FileNotFoundError(f"No class images found under {args.data_dir}")

    ring_records = []
    gradcam_rows = []
    selected_for_grid: set[str] = set()

    for path, true_class in samples:
        image = Image.open(path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        importance, v2_pred = v2_ring_importance(v2, tensor.clone())
        ring_records.append(
            {
                "image": str(path),
                "true_class": true_class,
                "pred_class": CLASS_NAMES[v2_pred],
                "center": importance[0],
                "middle": importance[1],
                "outer": importance[2],
            }
        )

        # One Grad-CAM example per class keeps the paper figure compact.
        if true_class not in selected_for_grid:
            lite.zero_grad(set_to_none=True)
            v2.zero_grad(set_to_none=True)
            lite_cam, lite_pred, _ = gradcam_for_model(lite, tensor.clone())
            v2_cam, v2_pred_cam, _ = gradcam_for_model(v2, tensor.clone())
            lite_overlay = overlay_cam(image, lite_cam)
            v2_overlay = overlay_cam(image, v2_cam)
            lite_overlay.save(args.output_dir / f"gradcam_lite_{true_class.replace(' ', '_')}.png")
            v2_overlay.save(args.output_dir / f"gradcam_v2_{true_class.replace(' ', '_')}.png")
            gradcam_rows.append(
                {
                    "path": path,
                    "true_class": true_class,
                    "lite_overlay": lite_overlay,
                    "v2_overlay": v2_overlay,
                    "lite_pred": lite_pred,
                    "v2_pred": v2_pred_cam,
                }
            )
            selected_for_grid.add(true_class)

    plot_ring_importance(ring_records, args.output_dir)
    plot_gradcam_grid(gradcam_rows, args.output_dir)
    print(f"Saved XAI figures to {args.output_dir}")


if __name__ == "__main__":
    main()
