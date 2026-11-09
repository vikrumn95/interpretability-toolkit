"""Run a small patching sweep, score it, save results, and plot.

The prompt pairs are illustrative inputs for exercising the pipeline, not a study.
"""

from pathlib import Path

from interp_toolkit import (ExperimentResult, PromptPair, SweepConfig, aggregate, load_model,
                            run_patching_experiment, score)
from interp_toolkit import plotting

OUT = Path("outputs")
model = load_model()

pairs = [
    PromptPair("The capital of France is", "The capital of Italy is", " Paris", " Rome"),
    PromptPair("The capital of Japan is", "The capital of Spain is", " Tokyo", " Madrid"),
]
config = SweepConfig(components=["resid_pre", "attn_out", "mlp_out"])

result = run_patching_experiment(model, pairs, config)
path = result.save(OUT / "sweep.json")
print(f"{len(result.records)} patched runs in {result.metadata['seconds']}s -> {path}")
for b in result.baselines:
    print(f"pair {b.pair}: clean top={b.clean.top_token!r} corrupted top={b.corrupted.top_token!r}")

# Metrics are computed from saved readouts; reloading needs no model.
result = ExperimentResult.load(path)
rows = score(result, metric="logit_diff")
agg = aggregate(rows, by=("component", "layer", "position"))
print(f"{len(agg)} aggregated cells; first: {agg[0]}")

tokens = result.baselines[0].clean_str_tokens
for comp in config.components:
    plotting.save_figure(plotting.plot_recovery_heatmap(rows, comp, str_tokens=tokens),
                         OUT / f"recovery_{comp}.png")
plotting.save_figure(plotting.plot_prediction_comparison(plotting.baseline_readouts(result, 0)),
                     OUT / "predictions_pair0.png")
plotting.save_figure(plotting.plot_metric_by_config(rows, x="layer", position=len(tokens) - 1,
                                                    title="Recovery at the final position"),
                     OUT / "recovery_by_layer.png")
print("figures written to", OUT.resolve())
