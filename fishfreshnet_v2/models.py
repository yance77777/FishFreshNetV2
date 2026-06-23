import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ECA(nn.Module):
    """Efficient Channel Attention (ECA-Net).

    Uses 1D convolution over channel dimension instead of FC layers.
    Much lighter than SE-Net or CBAM while maintaining competitive accuracy.
    """

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).squeeze(-1).squeeze(-1)  # (B, C)
        y = y.unsqueeze(1)  # (B, 1, C)
        y = self.conv(y)  # (B, 1, C)
        y = self.sigmoid(y).squeeze(1)  # (B, C)
        return x * y.unsqueeze(-1).unsqueeze(-1)


class SE(nn.Module):
    """Squeeze-and-Excitation attention module."""

    def __init__(self, channels: int, ratio: int = 16) -> None:
        super().__init__()
        hidden = max(1, channels // ratio)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class CRA(nn.Module):
    """Circular Region Attention (CRA).

    Divides the feature map into concentric ring regions (center, mid, outer),
    pools each ring separately, learns region weights via a small MLP,
    and applies the weights back to the feature map.

    This module exploits the circular structure of fish eye images:
    - Center ring: pupil area
    - Mid ring: iris/cornea area
    - Outer ring: sclera/background area
    """

    def __init__(self, channels: int, num_rings: int = 3, learnable: bool = True) -> None:
        super().__init__()
        self.num_rings = num_rings
        self.learnable = learnable

        # Ring weight predictor: small MLP that outputs per-ring importance
        self.ring_mlp = nn.Sequential(
            nn.Linear(channels * num_rings, channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels, num_rings, bias=False),
            nn.Softmax(dim=-1),
        )

        # Per-ring channel projection (1x1 conv per ring)
        self.ring_projs = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=1, bias=False)
            for _ in range(num_rings)
        ])

    def _create_ring_masks(self, h: int, w: int, device: torch.device) -> list[torch.Tensor]:
        """Create soft ring masks for the given spatial dimensions.

        Divides the feature map into concentric rings based on distance from center.
        Returns a list of num_rings masks, each of shape (1, 1, H, W).
        """
        cy, cx = h / 2.0, w / 2.0
        y_grid = torch.arange(h, device=device).float().unsqueeze(1).expand(h, w)
        x_grid = torch.arange(w, device=device).float().unsqueeze(0).expand(h, w)
        dist = torch.sqrt((y_grid - cy) ** 2 + (x_grid - cx) ** 2)
        max_dist = (cy ** 2 + cx ** 2) ** 0.5

        # Normalize distance to [0, 1]
        dist_norm = dist / (max_dist + 1e-6)

        # Create ring masks using distance binning
        masks = []
        for i in range(self.num_rings):
            lower = i / self.num_rings
            upper = (i + 1) / self.num_rings
            mask = ((dist_norm >= lower) & (dist_norm < upper)).float()
            masks.append(mask.unsqueeze(0).unsqueeze(0))  # (1, 1, H, W)

        # Normalize masks to sum to 1 at each spatial location
        mask_sum = torch.sum(torch.cat(masks, dim=0), dim=0, keepdim=True) + 1e-6
        masks = [m / mask_sum for m in masks]

        return masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        masks = self._create_ring_masks(h, w, x.device)

        # Pool each ring region
        ring_features = []
        ring_pooled = []
        for i in range(self.num_rings):
            masked = x * masks[i]  # (B, C, H, W)
            ring_features.append(self.ring_projs[i](masked))
            pooled = masked.sum(dim=[2, 3]) / (masks[i].sum() + 1e-6)  # (B, C)
            ring_pooled.append(pooled)

        # Concatenate ring features and predict weights
        concat_pooled = torch.cat(ring_pooled, dim=1)  # (B, C * num_rings)
        ring_weights = self.ring_mlp(concat_pooled)  # (B, num_rings)

        # Weighted sum of ring features
        out = torch.zeros_like(x)
        for i in range(self.num_rings):
            w_i = ring_weights[:, i].view(b, 1, 1, 1)  # (B, 1, 1, 1)
            out = out + w_i * ring_features[i]

        return out

    def get_ring_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return ring weights for visualization. Shape: (B, num_rings)."""
        b, c, h, w = x.size()
        masks = self._create_ring_masks(h, w, x.device)
        ring_pooled = []
        for i in range(self.num_rings):
            masked = x * masks[i]
            pooled = masked.sum(dim=[2, 3]) / (masks[i].sum() + 1e-6)
            ring_pooled.append(pooled)
        concat_pooled = torch.cat(ring_pooled, dim=1)
        return self.ring_mlp(concat_pooled)


class LightweightCRA(nn.Module):
    """Parameter-controlled Circular Region Attention.

    Unlike the original CRA, this module uses a depthwise shared 1x1 projection
    and a compact ring predictor. This keeps the module lightweight even on
    high-channel EfficientNet-B0 features while preserving ring-weight
    interpretability.
    """

    def __init__(self, channels: int, num_rings: int = 3, hidden: int = 64) -> None:
        super().__init__()
        self.num_rings = num_rings
        self.shared_proj = nn.Conv2d(channels, channels, kernel_size=1, groups=channels, bias=False)
        self.ring_mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_rings, bias=False),
            nn.Softmax(dim=-1),
        )

    def _create_ring_masks(self, h: int, w: int, device: torch.device) -> list[torch.Tensor]:
        cy, cx = h / 2.0, w / 2.0
        y_grid = torch.arange(h, device=device).float().unsqueeze(1).expand(h, w)
        x_grid = torch.arange(w, device=device).float().unsqueeze(0).expand(h, w)
        dist = torch.sqrt((y_grid - cy) ** 2 + (x_grid - cx) ** 2)
        max_dist = (cy ** 2 + cx ** 2) ** 0.5
        dist_norm = dist / (max_dist + 1e-6)

        masks = []
        for i in range(self.num_rings):
            lower = i / self.num_rings
            upper = (i + 1) / self.num_rings
            mask = ((dist_norm >= lower) & (dist_norm < upper)).float()
            masks.append(mask.unsqueeze(0).unsqueeze(0))
        mask_sum = torch.sum(torch.cat(masks, dim=0), dim=0, keepdim=True) + 1e-6
        return [m / mask_sum for m in masks]

    def _ring_weights(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        _, _, h, w = x.size()
        masks = self._create_ring_masks(h, w, x.device)

        descriptors = []
        for mask in masks:
            masked = x * mask
            descriptors.append(masked.sum(dim=[2, 3]) / (mask.sum() + 1e-6))
        descriptor = torch.stack(descriptors, dim=1).mean(dim=1)
        return self.ring_mlp(descriptor), masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, _, _ = x.size()
        ring_weights, masks = self._ring_weights(x)
        projected = self.shared_proj(x)

        attention = torch.zeros_like(projected[:, :1])
        for i, mask in enumerate(masks):
            weight = ring_weights[:, i].view(b, 1, 1, 1)
            attention = attention + weight * mask
        return projected * attention * self.num_rings

    def get_ring_weights(self, x: torch.Tensor) -> torch.Tensor:
        weights, _ = self._ring_weights(x)
        return weights


class ChannelAttention(nn.Module):
    """Standard channel attention (SE-style)."""

    def __init__(self, channels: int, ratio: int = 16) -> None:
        super().__init__()
        hidden = max(1, channels // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Standard spatial attention."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    """CBAM attention module."""

    def __init__(self, channels: int, ratio: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.ca = ChannelAttention(channels, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)
        return x * self.sa(x)


def build_attention(attention: str, channels: int) -> nn.Module:
    """Build an attention module with a shared interface for all backbones."""
    if attention == "eca":
        return ECA(channels)
    if attention == "se":
        return SE(channels)
    if attention == "cbam":
        return CBAM(channels)
    if attention == "none":
        return nn.Identity()
    raise ValueError(f"Unknown attention: {attention}. Choose from eca, se, cbam, none.")


class CompactClassifier(nn.Module):
    """Lightweight classifier head: Global Average Pool + Dropout + Linear."""

    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


class MultiScaleFusion(nn.Module):
    """Fuse low/mid/high EfficientNet feature maps into a compact feature map."""

    def __init__(self, in_channels: tuple[int, int, int] = (40, 112, 1280), out_channels: int = 256) -> None:
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
            )
            for ch in in_channels
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        target_size = features[0].shape[-2:]
        projected = []
        for feat, proj in zip(features, self.projections):
            feat = proj(feat)
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            projected.append(feat)
        return self.fuse(torch.cat(projected, dim=1))


class FishFreshNetV2(nn.Module):
    """FishFreshNetV2: EfficientNet-B0 + lightweight CRA + ECA.

    This is the main V2 model. It keeps the strong EfficientNet-B0 backbone,
    replaces heavy attention with ECA, and adds lightweight CRA for ring-based
    interpretability without using multi-scale fusion.
    """

    def __init__(
        self,
        num_classes: int = 3,
        dropout: float = 0.3,
        pretrained: bool = True,
        attention: str = "eca",
        use_cra: bool = True,
        cra_rings: int = 3,
        cra_type: str = "light",
        fusion_channels: int = 256,
    ) -> None:
        super().__init__()
        self.use_cra = use_cra
        self.attention_type = attention
        _ = fusion_channels

        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        self.features = backbone.features
        self.last_channels = 1280

        if use_cra:
            if cra_type == "light":
                self.cra = LightweightCRA(self.last_channels, num_rings=cra_rings)
            elif cra_type == "original":
                self.cra = CRA(self.last_channels, num_rings=cra_rings)
            else:
                raise ValueError("cra_type must be 'light' or 'original'")

        self.attention = build_attention(attention, self.last_channels)
        self.classifier = CompactClassifier(self.last_channels, num_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)

        if self.use_cra:
            x = self.cra(x)

        x = self.attention(x)
        return self.classifier(x)

    def get_ring_weights(self, x: torch.Tensor) -> torch.Tensor | None:
        if not self.use_cra:
            return None
        with torch.no_grad():
            feat = self.features(x)
            return self.cra.get_ring_weights(feat)


class FishFreshNetV2A(nn.Module):
    """FishFreshNetV2-Lite: MobileNetV3-Small + ECA."""

    def __init__(
        self,
        num_classes: int = 3,
        dropout: float = 0.3,
        pretrained: bool = True,
        attention: str = "eca",
    ) -> None:
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        self.features = backbone.features
        self.attention = build_attention(attention, 576)
        self.classifier = CompactClassifier(576, num_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.attention(x)
        return self.classifier(x)


MODEL_REGISTRY = {
    "fishfreshnet_v2": FishFreshNetV2,
    "fishfreshnet_v2_lite": FishFreshNetV2A,
}


def build_model(
    model_name: str = "v2",
    num_classes: int = 3,
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    """Build a model by name.

    Args:
        model_name: One of 'fishfreshnet_v2' or 'fishfreshnet_v2_lite'.
        num_classes: Number of output classes.
        pretrained: Whether to load ImageNet pretrained weights.
        **kwargs: Additional arguments (only passed to models that accept them).

    Returns:
        The constructed model.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_REGISTRY.keys())}")

    import inspect
    cls = MODEL_REGISTRY[model_name]
    valid_params = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return cls(num_classes=num_classes, pretrained=pretrained, **filtered_kwargs)
