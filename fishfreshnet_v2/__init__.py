from .models import (
    FishFreshNetV2,
    FishFreshNetV2A,
    build_model,
)

FishFreshNetV2Lite = FishFreshNetV2A

__all__ = [
    "FishFreshNetV2",
    "FishFreshNetV2Lite",
    "FishFreshNetV2A",
    "build_model",
]
