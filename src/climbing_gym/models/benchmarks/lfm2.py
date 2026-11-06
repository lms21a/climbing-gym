from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from climbing_gym.models.registry import build_model


@dataclass(frozen=True, slots=True)
class TimingStats:
    mean_ms: float
    p50_ms: float
    p90_ms: float


@dataclass(frozen=True, slots=True)
class TorchCompileBenchmarkResult:
    compiled_forward: TimingStats
    compiled_fwdbwd: TimingStats
    compile_wall_s: float


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = int(round((len(values) - 1) * q))
    idx = max(0, min(idx, len(values) - 1))
    return values[idx]


def _resolve_model_path(model_path: str) -> str:
    candidate = Path(model_path)
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    return str(candidate)


def _resolve_torch_dtype(dtype: str) -> torch.dtype:
    normalized = str(dtype).strip().lower()
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {dtype!r}") from exc


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_lfm2_model(
    model_path: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    config_overrides: dict[str, Any] | None = None,
) -> torch.nn.Module:
    resolved_dtype = torch.float32 if device.type != "cuda" and dtype != torch.float32 else dtype
    model = build_model("lfm2", model_dir=model_path, **(config_overrides or {}))
    return model.to(device=device, dtype=resolved_dtype)


def _timed_forward(
    module: torch.nn.Module,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    *,
    device: torch.device,
    warmup: int,
    iters: int,
) -> TimingStats:
    module.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = module(input_ids, positions=positions)
        _sync_device(device)

        times: list[float] = []
        for _ in range(iters):
            start = time.perf_counter()
            _ = module(input_ids, positions=positions)
            _sync_device(device)
            times.append((time.perf_counter() - start) * 1e3)

    return TimingStats(
        mean_ms=statistics.mean(times),
        p50_ms=_percentile(times, 0.50),
        p90_ms=_percentile(times, 0.90),
    )


def _timed_fwdbwd(
    module: torch.nn.Module,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    targets: torch.Tensor,
    *,
    device: torch.device,
    warmup: int,
    iters: int,
) -> TimingStats:
    module.train()
    for _ in range(warmup):
        module.zero_grad(set_to_none=True)
        logits = module(input_ids, positions=positions)
        loss = torch.nn.functional.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
    _sync_device(device)

    times: list[float] = []
    for _ in range(iters):
        module.zero_grad(set_to_none=True)
        start = time.perf_counter()
        logits = module(input_ids, positions=positions)
        loss = torch.nn.functional.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        _sync_device(device)
        times.append((time.perf_counter() - start) * 1e3)

    return TimingStats(
        mean_ms=statistics.mean(times),
        p50_ms=_percentile(times, 0.50),
        p90_ms=_percentile(times, 0.90),
    )


def run_torch_compile_benchmark(
    model_path: str,
    *,
    batch_size: int = 1,
    seq_len: int = 512,
    warmup: int = 5,
    iters: int = 20,
    device: str | None = None,
    dtype: str = "bfloat16",
    compile_mode: str = "default",
    fullgraph: bool = False,
    config_overrides: dict[str, Any] | None = None,
) -> TorchCompileBenchmarkResult:
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    resolved_dtype = _resolve_torch_dtype(dtype)
    model = _load_lfm2_model(
        _resolve_model_path(model_path),
        device=resolved_device,
        dtype=resolved_dtype,
        config_overrides=config_overrides,
    )

    vocab_size = int(model.config.vocab_size)
    input_ids = torch.randint(vocab_size, (batch_size, seq_len), device=resolved_device)
    positions = torch.arange(seq_len, device=resolved_device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    targets = torch.randint(vocab_size, (batch_size, seq_len), device=resolved_device)

    compile_start = time.perf_counter()
    compiled_model = torch.compile(model, backend="inductor", mode=compile_mode, fullgraph=fullgraph)
    _ = compiled_model(input_ids, positions=positions)
    _sync_device(resolved_device)
    compile_wall_s = time.perf_counter() - compile_start

    compiled_forward = _timed_forward(
        compiled_model,
        input_ids,
        positions,
        device=resolved_device,
        warmup=warmup,
        iters=iters,
    )
    compiled_fwdbwd = _timed_fwdbwd(
        compiled_model,
        input_ids,
        positions,
        targets,
        device=resolved_device,
        warmup=max(1, warmup),
        iters=iters,
    )
    return TorchCompileBenchmarkResult(
        compiled_forward=compiled_forward,
        compiled_fwdbwd=compiled_fwdbwd,
        compile_wall_s=compile_wall_s,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark climbing_gym LFM2 compile performance.")
    parser.add_argument("--model-path", type=str, default="saved_models/LFM2.5-1.2B-Base")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--compile-mode", type=str, default="default")
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument("--json-output", type=str, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    payload: dict[str, Any] = {"model_path": _resolve_model_path(args.model_path)}

    torch_result = run_torch_compile_benchmark(
        args.model_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        warmup=args.warmup,
        iters=args.iters,
        device=args.device,
        dtype=args.dtype,
        compile_mode=args.compile_mode,
        fullgraph=bool(args.fullgraph),
    )
    payload["torch_compile"] = asdict(torch_result)
    print("=== Torch Compile Benchmark ===")
    print(f"compile_wall_s: {torch_result.compile_wall_s:.3f}")
    print(f"compiled_forward_mean_ms: {torch_result.compiled_forward.mean_ms:.3f}")
    print(f"compiled_fwdbwd_mean_ms: {torch_result.compiled_fwdbwd.mean_ms:.3f}")

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
