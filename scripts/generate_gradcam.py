"""Generate Grad-CAM visualizations for FishFreshNetV2 models.

Usage:
    python scripts/generate_gradcam.py --data-dir /path/to/MFED --output-dir runs/gradcam

This script loads trained model weights and generates Grad-CAM heatmaps
for selected test images, showing which regions drive the prediction.
"""

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
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from fishfreshnet_v2.models import build_model


# Grad-CAM implementation
class GradCAM:
    """Grad-CAM for the last convolutional layer."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Generate Grad-CAM heatmap for a specific class."""
        self.model.zero_grad()
        output = self.model(input_tensor)
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1.0
        output.backward(gradient=one_hot)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.squeeze().cpu().numpy()


def get_last_conv_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Find the last convolutional layer in the model."""
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    return last_conv


def load_image(image_path: Path, transform: transforms.Compose) -> tuple[torch.Tensor, np.ndarray]:
    """Load and preprocess an image, return tensor and original."""
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    img_np = np.array(img.resize((224, 224))) / 255.0
    return img_tensor, img_np


def generate_gradcam_for_model(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Generate Grad-CAM heatmap for a model and image."""
    model.eval()
    image_tensor = image_tensor.to(device)

    # Get target layer
    last_conv = get_last_conv_layer(model)
    if last_conv is None:
        raise ValueError("No convolutional layer found in model")

    grad_cam = GradCAM(model, last_conv)

    # Forward pass
    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()

    # Generate heatmap
    heatmap = grad_cam.generate(image_tensor, pred_class)

    return pred_class, confidence, heatmap


def overlay_heatmap(img_np: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay Grad-CAM heatmap on image."""
    import matplotlib.cm as cm

    heatmap_resized = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize((224, 224))) / 255.0
    colormap = cm.jet(heatmap_resized)[:, :, :3]
    overlay = (1 - alpha) * img_np + alpha * colormap
    return np.clip(overlay, 0, 1)


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations")
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to MFED dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/gradcam"))
    parser.add_argument("--v2-weights", type=Path, default=None, help="FishFreshNetV2 weights (.pth)")
    parser.add_argument("--lite-weights", type=Path, default=None, help="V2-Lite weights (.pth)")
    parser.add_argument("--num-images", type=int, default=6, help="Number of images per class")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Build models
    models = {}
    if args.v2_weights and args.v2_weights.exists():
        m = build_model("fishfreshnet_v2", num_classes=3, pretrained=False, attention="eca", cra_type="light", cra_rings=3)
        m.load_state_dict(torch.load(args.v2_weights, map_location=device, weights_only=True))
        m.to(device)
        models["FishFreshNetV2"] = m

    if args.lite_weights and args.lite_weights.exists():
        m = build_model("fishfreshnet_v2_lite", num_classes=3, pretrained=False, attention="eca")
        m.load_state_dict(torch.load(args.lite_weights, map_location=device, weights_only=True))
        m.to(device)
        models["V2-Lite"] = m

    if not models:
        print("No model weights provided. Use --v2-weights and/or --lite-weights.")
        return

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Find test images
    class_names = ["Highly Fresh", "Fresh", "Not Fresh"]
    test_dir = args.data_dir / "test"
    if not test_dir.exists():
        # Try flat structure
        test_dir = args.data_dir

    images_by_class = {}
    for cls in class_names:
        cls_dir = test_dir / cls
        if cls_dir.exists():
            imgs = sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.png"))
            images_by_class[cls] = imgs[: args.num_images]

    if not images_by_class:
        print(f"No images found in {test_dir}")
        return

    # Generate Grad-CAM for each model
    for model_name, model in models.items():
        print(f"\nGenerating Grad-CAM for {model_name}...")

        for cls, images in images_by_class.items():
            for i, img_path in enumerate(images):
                img_tensor, img_np = load_image(img_path, transform)
                pred_class, confidence, heatmap = generate_gradcam_for_model(model, img_tensor, device)
                overlay = overlay_heatmap(img_np, heatmap)

                # Save
                fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

                axes[0].imshow(img_np)
                axes[0].set_title("Original", fontsize=10)
                axes[0].axis("off")

                axes[1].imshow(heatmap, cmap="jet")
                axes[1].set_title("Grad-CAM", fontsize=10)
                axes[1].axis("off")

                axes[2].imshow(overlay)
                pred_label = class_names[pred_class]
                axes[2].set_title(f"Pred: {pred_label}\nConf: {confidence:.1%}", fontsize=10)
                axes[2].axis("off")

                fig.suptitle(f"{model_name} | True: {cls}", fontsize=11, y=1.02)
                fig.tight_layout()

                safe_name = cls.replace(" ", "_").lower()
                out_path = args.output_dir / f"{model_name.replace(' ', '_').lower()}_{safe_name}_{i+1}.png"
                fig.savefig(out_path, dpi=200, bbox_inches="tight")
                plt.close(fig)

        print(f"  Saved to {args.output_dir}")

    print("\nDone!")


if __name__ == "__main__":
    main()
