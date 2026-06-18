"""Benchmark model latency, size, params, and FLOPs.

This measures raw PyTorch model inference for FishFreshNetV2 and
FishFreshNetV2-Lite efficiency comparisons.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch

from fishfreshnet_v2.models import build_model
from fishfreshnet_v2.train import model_complexity, prepare_device


MODEL_CONFIGS = {
    "FishFreshNetV2-Lite": {"model_name": "fishfreshnet_v2_lite", "attention": "eca"},
    "FishFreshNetV2": {"model_name": "fishfreshnet_v2", "attention": "eca", "cra_type": "light", "cra_rings": 3},
}


def autocast_context(use_amp: bool, device: torch.device):
    if use_amp and device.type == "cuda":
        return torch.amp.autocast("cuda")
    return nullcontext()


def percentile(values: list[float], q: float) -> float:
    tensor = torch.tensor(values)
    return float(torch.quantile(tensor, q).item())


def benchmark_one(
    name: str,
    config: dict,
    device: torch.device,
    input_size: int,
    batch_size: int,
    warmup: int,
    repeats: int,
    channels_last: bool,
    use_amp: bool,
) -> dict[str, float | str]:
    model = build_model(config.pop("model_name"), num_classes=3, pretrained=False, **config).to(device)
    if channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model.eval()

    params_m, flops_g = model_complexity(model, device, input_size)
    dummy = torch.randn(batch_size, 3, input_size, input_size, device=device)
    if channels_last and device.type == "cuda":
        dummy = dummy.contiguous(memory_format=torch.channels_last)

    with torch.no_grad():
        for _ in range(warmup):
            with autocast_context(use_amp, device):
                model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            with autocast_context(use_amp, device):
                model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000 / batch_size)

    state_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return {
        "Model": name,
        "Device": str(device),
        "Batch Size": batch_size,
        "Params (M)": params_m,
        "FLOPs (G)": flops_g,
        "Model Size (MB)": state_bytes / (1024 * 1024),
        "Latency Mean (ms/img)": sum(times) / len(times),
        "Latency P50 (ms/img)": percentile(times, 0.50),
        "Latency P95 (ms/img)": percentile(times, 0.95),
        "Latency P99 (ms/img)": percentile(times, 0.99),
        "Throughput (img/s)": 1000 / (sum(times) / len(times)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark FishFreshNet model latency")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/benchmark_models"))
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = prepare_device()
    rows = []
    for name, config in MODEL_CONFIGS.items():
        print(f"Benchmarking {name}...")
        rows.append(
            benchmark_one(
                name,
                config.copy(),
                device,
                args.input_size,
                args.batch_size,
                args.warmup,
                args.repeats,
                args.channels_last,
                args.amp,
            )
        )
    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "model_latency.csv", index=False)
    (args.output_dir / "model_latency.md").write_text(df.to_markdown(index=False), encoding="utf-8")
    print(df.to_string(index=False))
    print(f"Saved benchmark results to {args.output_dir}")


if __name__ == "__main__":
    main()
