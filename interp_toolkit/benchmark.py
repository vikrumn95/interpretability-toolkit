"""Lightweight measurement of activation-capture overhead.

What each number means:

* ``mean_ms`` / ``std_ms`` / ``min_ms``: wall-clock time per forward pass, after
  warmup, with device synchronisation before and after (CUDA/MPS are async).
* ``retained_bytes``: exact bytes held by the captured activations, counting
  shared storage once (aliases). This is the most reliable memory number.
* ``rss_delta_bytes``: change in process resident memory across one run while
  its outputs are still referenced. Approximate: the allocator may reuse freed
  memory, or keep memory after it is freed.
* ``process_peak_rss_bytes``: the process's lifetime peak RSS (``ru_maxrss``).
  It cannot be reset, so it only shows whether *some* run raised the high-water
  mark; it is not a per-configuration peak.
* ``accelerator_peak_bytes``: exact per-run peak allocated by PyTorch on CUDA
  (reset before each configuration). On MPS this is the current allocation after
  the run (MPS has no peak counter); on CPU it is None.
"""

from __future__ import annotations

import gc
import resource
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from typing import Mapping

import psutil
import torch

from .capture import CaptureSpec, capture


@dataclass
class BenchmarkResult:
    label: str
    repeats: int
    mean_ms: float
    std_ms: float
    min_ms: float
    n_tensors: int
    retained_bytes: int
    rss_delta_bytes: int
    process_peak_rss_bytes: int
    device: str
    accelerator_peak_bytes: int | None

    def as_dict(self) -> dict:
        return asdict(self)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _peak_rss() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024  # macOS reports bytes, Linux KiB


@torch.no_grad()
def benchmark_capture(model, tokens: torch.Tensor, specs: Mapping[str, CaptureSpec | None],
                      repeats: int = 5, warmup: int = 1) -> list[BenchmarkResult]:
    """Time and measure one forward pass per configuration.

    ``specs`` maps a label to a ``CaptureSpec``, or to ``None`` for an
    uninstrumented forward pass (the baseline). Example::

        benchmark_capture(model, tokens, {
            "no capture": None,
            "selective": CaptureSpec(components=["resid_pre"]),
            "full": CaptureSpec.all(),
        })
    """
    device = next(model.parameters()).device
    tokens = tokens.to(device)
    proc = psutil.Process()

    def once(spec):
        if spec is None:
            return model(tokens), None
        return capture(model, tokens, spec)

    results = []
    for label, spec in specs.items():
        for _ in range(warmup):
            once(spec)
        times = []
        for _ in range(repeats):
            _sync(device)
            t0 = time.perf_counter()
            out = once(spec)
            _sync(device)
            times.append((time.perf_counter() - t0) * 1e3)
            del out

        # One extra measured run for memory, keeping its outputs alive while measuring.
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        rss0 = proc.memory_info().rss
        logits, acts = once(spec)
        _sync(device)
        rss_delta = proc.memory_info().rss - rss0
        if device.type == "cuda":
            accel = torch.cuda.max_memory_allocated(device)
        elif device.type == "mps":
            accel = torch.mps.current_allocated_memory()
        else:
            accel = None
        results.append(BenchmarkResult(
            label=label, repeats=repeats, mean_ms=statistics.fmean(times),
            std_ms=statistics.stdev(times) if len(times) > 1 else 0.0, min_ms=min(times),
            n_tensors=0 if acts is None else len(acts),
            retained_bytes=0 if acts is None else acts.nbytes(),
            rss_delta_bytes=rss_delta, process_peak_rss_bytes=_peak_rss(),
            device=str(device), accelerator_peak_bytes=accel))
        del logits, acts
    return results


def format_results(results: list[BenchmarkResult]) -> str:
    """Plain-text table for terminal output.

    ``min ms`` is the least noise-sensitive timing for short runs; compare
    configurations on it. ``rss_delta_bytes`` is left out of the table because
    allocator reuse usually makes it ~0 on CPU (it stays in the result object).
    """
    head = (f"{'config':<22}{'min ms':>8}{'mean ms':>9}{'± std':>7}{'tensors':>9}"
            f"{'retained MB':>13}{'accel peak MB':>15}")
    lines = [head, "-" * len(head)]
    for r in results:
        accel = "n/a" if r.accelerator_peak_bytes is None else f"{r.accelerator_peak_bytes / 1e6:.1f}"
        lines.append(f"{r.label:<22}{r.min_ms:>8.1f}{r.mean_ms:>9.1f}{r.std_ms:>7.1f}{r.n_tensors:>9}"
                     f"{r.retained_bytes / 1e6:>13.2f}{accel:>15}")
    peak = max((r.process_peak_rss_bytes for r in results), default=0)
    lines.append(f"process peak RSS (lifetime high-water mark, not per config): {peak / 1e6:.0f} MB")
    return "\n".join(lines)
