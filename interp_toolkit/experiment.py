"""Generic activation-patching experiment runner.

For each clean/corrupted prompt pair:

1. tokenize both (they must have equal length so positions align);
2. run the clean prompt once, capturing *only* the hooks the sweep will patch;
3. run the corrupted prompt once (baseline);
4. for every intervention in the sweep grid, re-run the corrupted prompt with
   that slice of the clean activation patched in, and record a readout.

The runner is metric-agnostic. Each forward pass is reduced to a ``Readout``
(the logits of the two target tokens, the log-partition function, and the top-1
token at the readout position), which is enough to compute logit differences,
target probabilities, and recovery later (see ``metrics.py``) without
re-running the model or storing full-vocabulary logits.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import torch

from .capture import CaptureSpec, capture
from .components import get_component
from .interventions import Patch, run_with_patches


@dataclass(frozen=True)
class PromptPair:
    """A clean/corrupted prompt pair and the two tokens whose logits are read out.

    ``clean_target`` is the token the clean prompt is expected to favour and
    ``corrupted_target`` the one the corrupted prompt is expected to favour.
    Each must be a single token (include the leading space for GPT-2, e.g. " Paris").
    """

    clean: str
    corrupted: str
    clean_target: str
    corrupted_target: str


@dataclass(frozen=True)
class SweepConfig:
    """Which interventions to run.

    The grid is ``components x layers x positions x heads``:

    * ``layers=None``: all layers.
    * ``positions=None``: all positions. With ``per_position=False`` the selected
      positions are patched together in a single run (recorded as ``position=None``).
    * ``per_head=True`` sweeps heads individually for per-head components
      (q, k, v, z, pattern); otherwise all heads are patched together.
    * ``readout_position``: where target logits are read (default: last token).
    """

    components: tuple[str, ...] = ("resid_pre",)
    layers: tuple[int, ...] | None = None
    positions: tuple[int, ...] | None = None
    per_position: bool = True
    per_head: bool = False
    readout_position: int = -1
    prepend_bos: bool = True

    def __post_init__(self):
        object.__setattr__(self, "components", tuple(self.components))
        for f in ("layers", "positions"):
            v = getattr(self, f)
            if v is not None:
                object.__setattr__(self, f, tuple(v))
        for c in self.components:
            get_component(c)


@dataclass(frozen=True)
class Intervention:
    component: str
    layer: int
    position: int | None  # None: all selected positions at once
    head: int | None  # None: all heads (or not a per-head component)


@dataclass
class Readout:
    """What is recorded from one forward pass, at the readout position."""

    target_logits: list[float]  # [clean_target, corrupted_target]
    logsumexp: float  # log of the softmax normaliser, so p(t) = exp(logit_t - logsumexp)
    top_token_id: int
    top_token: str


@dataclass
class PairBaseline:
    pair: int
    clean: Readout
    corrupted: Readout
    target_ids: list[int]
    clean_str_tokens: list[str]
    corrupted_str_tokens: list[str]


@dataclass
class PatchRecord:
    pair: int
    component: str
    layer: int
    position: int | None
    head: int | None
    readout: Readout


@dataclass
class ExperimentResult:
    config: SweepConfig
    pairs: list[PromptPair]
    baselines: list[PairBaseline]
    records: list[PatchRecord]
    metadata: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=1))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentResult":
        d = json.loads(Path(path).read_text())
        cfg = d["config"]
        return cls(
            config=SweepConfig(**cfg),
            pairs=[PromptPair(**p) for p in d["pairs"]],
            baselines=[PairBaseline(**{**b, "clean": Readout(**b["clean"]),
                                       "corrupted": Readout(**b["corrupted"])})
                       for b in d["baselines"]],
            records=[PatchRecord(**{**r, "readout": Readout(**r["readout"])}) for r in d["records"]],
            metadata=d["metadata"],
        )


def single_token_id(model, s: str) -> int:
    ids = model.to_tokens(s, prepend_bos=False)[0]
    if len(ids) != 1:
        raise ValueError(f"Target {s!r} is {len(ids)} tokens {model.to_str_tokens(s, prepend_bos=False)}; "
                         "targets must be single tokens")
    return int(ids[0])


def make_readout(model, logits: torch.Tensor, target_ids: Sequence[int], pos: int) -> Readout:
    row = logits[0, pos].float()
    top = int(row.argmax())
    return Readout([float(row[t]) for t in target_ids], float(torch.logsumexp(row, -1)),
                   top, model.to_string(top))


def enumerate_interventions(config: SweepConfig, n_layers: int, n_heads: int,
                            seq_len: int) -> list[Intervention]:
    """The sweep grid for one prompt length, in deterministic order."""
    layers = range(n_layers) if config.layers is None else config.layers
    for layer in layers:
        if not 0 <= layer < n_layers:
            raise ValueError(f"Layer {layer} out of range for {n_layers}-layer model")
    if config.positions is None:
        positions = list(range(seq_len))
    else:
        positions = []
        for p in config.positions:
            if not -seq_len <= p < seq_len:
                raise ValueError(f"Position {p} out of range for sequence length {seq_len}")
            positions.append(p % seq_len)
    pos_axis: list[int | None] = positions if config.per_position else [None]
    out = []
    for comp in config.components:
        heads: list[int | None] = (list(range(n_heads))
                                   if config.per_head and get_component(comp).head_dim is not None
                                   else [None])
        for layer in layers:
            for pos in pos_axis:
                for head in heads:
                    out.append(Intervention(comp, layer, pos, head))
    return out


@torch.no_grad()
def run_patching_experiment(
    model,
    pairs: Sequence[PromptPair],
    config: SweepConfig = SweepConfig(),
    progress: Callable[[int, int], None] | None = None,
) -> ExperimentResult:
    """Run baselines and the full patching sweep for every prompt pair.

    ``progress(done, total)`` is called after each patched run if given.
    """
    from importlib.metadata import version

    started = time.time()
    device = next(model.parameters()).device
    baselines, records = [], []
    for i, pair in enumerate(pairs):
        clean = model.to_tokens(pair.clean, prepend_bos=config.prepend_bos).to(device)
        corrupted = model.to_tokens(pair.corrupted, prepend_bos=config.prepend_bos).to(device)
        if clean.shape != corrupted.shape:
            raise ValueError(f"Pair {i}: clean and corrupted prompts tokenize to different lengths "
                             f"({clean.shape[1]} vs {corrupted.shape[1]}); positions would not align")
        target_ids = [single_token_id(model, pair.clean_target),
                      single_token_id(model, pair.corrupted_target)]
        grid = enumerate_interventions(config, model.cfg.n_layers, model.cfg.n_heads, clean.shape[1])

        # Selective capture: only the hooks this sweep patches, all positions (patch
        # sources must be full-shape).
        needed = tuple(dict.fromkeys(get_component(iv.component).hook_name(iv.layer) for iv in grid))
        clean_logits, clean_acts = capture(model, clean, CaptureSpec(hook_names=needed))
        corrupted_logits = model(corrupted)
        pos = config.readout_position
        baselines.append(PairBaseline(
            i, make_readout(model, clean_logits, target_ids, pos),
            make_readout(model, corrupted_logits, target_ids, pos), target_ids,
            model.to_str_tokens(clean[0]), model.to_str_tokens(corrupted[0])))

        for n, iv in enumerate(grid):
            hook = get_component(iv.component).hook_name(iv.layer)
            positions = ([iv.position] if iv.position is not None
                         else None if config.positions is None else list(config.positions))
            patch = Patch(hook, clean_acts[hook], positions=positions,
                          heads=None if iv.head is None else [iv.head])
            logits = run_with_patches(model, corrupted, [patch])
            records.append(PatchRecord(i, iv.component, iv.layer, iv.position, iv.head,
                                       make_readout(model, logits, target_ids, pos)))
            if progress:
                progress(n + 1, len(grid))
        del clean_acts  # release captured activations before the next pair

    metadata = {
        "model": getattr(model.cfg, "model_name", "?"),
        "device": str(device),
        "dtype": str(next(model.parameters()).dtype),
        "versions": {p: version(p) for p in ("torch", "transformer-lens", "transformers")},
        "python": platform.python_version(),
        "n_forward_passes": len(records) + 2 * len(pairs),
        "seconds": round(time.time() - started, 3),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    return ExperimentResult(config, list(pairs), baselines, records, metadata)
