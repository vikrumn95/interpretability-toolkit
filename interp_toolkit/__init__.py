"""interp_toolkit: activation capture, patching, and experiment utilities on TransformerBridge."""

from .benchmark import BenchmarkResult, benchmark_capture, format_results
from .capture import Activations, CaptureSpec, capture
from .components import COMPONENTS, hook_name, resolve_component
from .experiment import (ExperimentResult, Intervention, PatchRecord, PromptPair, Readout,
                         SweepConfig, enumerate_interventions, run_patching_experiment)
from .inspection import activation_shapes, list_components, list_hook_points, model_info
from .interventions import Patch, active_hook_count, patch_hooks, run_with_patches
from .metrics import aggregate, logit_diff, recovery, score, target_prob, to_grid
from .model import DEFAULT_MODEL, RunResult, load_model, run, set_determinism, to_tokens, top_predictions

__all__ = [
    "load_model", "set_determinism", "run", "RunResult", "to_tokens", "top_predictions", "DEFAULT_MODEL",
    "model_info", "list_hook_points", "list_components", "activation_shapes",
    "COMPONENTS", "hook_name", "resolve_component",
    "CaptureSpec", "Activations", "capture",
    "Patch", "patch_hooks", "run_with_patches", "active_hook_count",
    "PromptPair", "SweepConfig", "Intervention", "Readout", "PatchRecord", "ExperimentResult",
    "enumerate_interventions", "run_patching_experiment",
    "logit_diff", "target_prob", "recovery", "score", "aggregate", "to_grid",
    "benchmark_capture", "format_results", "BenchmarkResult",
]
