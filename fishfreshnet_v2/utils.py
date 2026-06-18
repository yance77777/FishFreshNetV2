"""Utility functions for FishFreshNetV2: Grad-CAM, visualization, timing."""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def measure_inference_time(
    model: nn.Module,
    device: torch.device,
    input_size: int = 224,
    warmup: int = 10,
    repeats: int = 100,
    channels_last: bool = True,
) -> dict[str, float]:
    """Measure average inference time on CPU and GPU.

    Args:
        model: The model to benchmark.
        device: Device to run on.
        input_size: Input image size.
        warmup: Number of warmup iterations.
        repeats: Number of timed iterations.

    Returns:
        Dict with 'cpu_ms' and 'gpu_ms' (if available).
    """
    model.eval()
    original_device = next(model.parameters()).device
    results = {}

    # CPU timing
    cpu_model = model.cpu()
    dummy = torch.randn(1, 3, input_size, input_size)
    with torch.no_grad():
        for _ in range(warmup):
            cpu_model(dummy)
        start = time.perf_counter()
        for _ in range(repeats):
            cpu_model(dummy)
        elapsed = (time.perf_counter() - start) / repeats * 1000
    results["cpu_ms"] = elapsed

    # GPU timing (if available)
    if device.type == "cuda":
        gpu_model = model.to(device)
        dummy_gpu = dummy.to(device)
        # channels_last format
        if channels_last and dummy_gpu.ndim == 4:
            dummy_gpu = dummy_gpu.contiguous(memory_format=torch.channels_last)
        with torch.no_grad():
            for _ in range(warmup):
                gpu_model(dummy_gpu)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(repeats):
                gpu_model(dummy_gpu)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) / repeats * 1000
        results["gpu_ms"] = elapsed

    # Restore model to original device
    model.to(original_device)
    return results


class GradCAM:
    """Grad-CAM visualization for FishFreshNetV2.

    Generates gradient-weighted class activation maps to show
    which regions the model focuses on for its predictions.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap.

        Args:
            input_tensor: Input image tensor (1, 3, H, W).
            target_class: Target class index. If None, uses predicted class.

        Returns:
            Heatmap as numpy array, normalized to [0, 1].
        """
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        # Weight activations by global-average-pooled gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = torch.relu(cam)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-6)

        return cam.squeeze().cpu().numpy()


def get_target_layer(model: nn.Module) -> nn.Module:
    """Get the last convolutional layer for Grad-CAM.

    Args:
        model: The model.

    Returns:
        The target layer module.
    """
    # For MobileNetV3-based models
    if hasattr(model, "features"):
        return model.features[-1]
    raise ValueError("Cannot find target layer. Please specify manually.")


def visualize_gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    class_names: list[str],
    predicted_class: int,
    save_path: Path | None = None,
    title: str = "Grad-CAM",
) -> plt.Figure:
    """Visualize Grad-CAM heatmap overlaid on the original image.

    Args:
        model: The model.
        image_tensor: Input image tensor (1, 3, H, W).
        class_names: List of class names.
        predicted_class: Predicted class index.
        save_path: Path to save the figure. If None, not saved.
        title: Figure title.

    Returns:
        matplotlib Figure object.
    """
    target_layer = get_target_layer(model)
    gradcam = GradCAM(model, target_layer)
    heatmap = gradcam.generate(image_tensor)

    # Denormalize image for display
    img = image_tensor.squeeze().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)

    # Resize heatmap to match image
    from scipy.ndimage import zoom
    h, w = img.shape[:2]
    heatmap_resized = zoom(heatmap, (h / heatmap.shape[0], w / heatmap.shape[1]))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap_resized, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    axes[2].imshow(img)
    axes[2].imshow(heatmap_resized, cmap="jet", alpha=0.5)
    axes[2].set_title(f"Overlay: {class_names[predicted_class]}")
    axes[2].axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def visualize_ring_weights(
    ring_weights: torch.Tensor,
    class_names: list[str],
    ring_labels: list[str] | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Visualize CRA ring weights as a bar chart.

    Args:
        ring_weights: Tensor of shape (num_classes, num_rings) or (num_rings,).
        class_names: List of class names.
        ring_labels: Labels for each ring. Defaults to ['Center', 'Mid', 'Outer'].
        save_path: Path to save the figure.

    Returns:
        matplotlib Figure object.
    """
    if ring_labels is None:
        ring_labels = [f"Ring {i+1}" for i in range(ring_weights.shape[-1])]

    weights = ring_weights.cpu().numpy()

    if weights.ndim == 1:
        weights = weights.reshape(1, -1)

    num_classes = weights.shape[0]
    fig, axes = plt.subplots(1, num_classes, figsize=(4 * num_classes, 4), squeeze=False)

    for i in range(num_classes):
        ax = axes[0, i]
        bars = ax.bar(ring_labels, weights[i], color=["#4CAF50", "#FFC107", "#F44336"][:len(ring_labels)])
        ax.set_title(class_names[i] if i < len(class_names) else f"Class {i}")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Weight")
        for bar, val in zip(bars, weights[i]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", fontsize=9)

    fig.suptitle("CRA Ring Weights by Class", fontsize=12)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig
