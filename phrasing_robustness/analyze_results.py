"""Analyze saved phrasing-robustness patching results WITHOUT loading GPT-2.

    python phrasing_robustness/analyze_results.py
    python phrasing_robustness/analyze_results.py --out phrasing_robustness/results
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from interp_toolkit import ExperimentResult, score, to_grid
from interp_toolkit import plotting


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    valid = a[np.isfinite(a)]
    if valid.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "std": float("nan")}
    return {"n": int(valid.size), "mean": float(valid.mean()),
            "median": float(np.median(valid)), "std": float(valid.std(ddof=1))
            if valid.size > 1 else 0.0}


def load_scored(path: Path, min_gap: float) -> list[dict]:
    result = ExperimentResult.load(path)
    return score(result, metric="logit_diff", min_denominator=min_gap)


def plot_semantic_layers(summary: list[dict], role: str, templates: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for template in templates:
        points = sorted((r for r in summary if r["template"] == template and
                         r["role"] == role), key=lambda r: r["layer"])
        if not points:
            continue
        ax.errorbar([r["layer"] for r in points], [r["mean"] for r in points],
                    yerr=[r["std"] / np.sqrt(r["n"]) if r["n"] > 1 else 0
                          for r in points], label=template, marker="o", capsize=2)
    ax.axhline(0, linewidth=0.8, linestyle="--")
    ax.axhline(1, linewidth=0.8, linestyle=":")
    ax.set(title=f"Patching recovery at {role.replace('_', ' ')}",
           xlabel="Transformer layer (0-based)",
           ylabel="Mean normalized recovery (error bars: SE)")
    ax.legend()
    fig.tight_layout()
    plotting.save_figure(fig, path)


def spearman(a: list[float], b: list[float]) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    valid = np.isfinite(aa) & np.isfinite(bb)
    aa, bb = aa[valid], bb[valid]
    if aa.size < 3 or np.std(aa) == 0 or np.std(bb) == 0:
        return float("nan")
    # Rank data rather than raw magnitudes; ties are unlikely for these floats.
    ra = np.argsort(np.argsort(aa)).astype(float)
    rb = np.argsort(np.argsort(bb)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    out = args.out
    manifest = json.loads((out / "manifest.json").read_text())
    templates = list(manifest["templates"])
    selected = manifest["selected"]
    min_gap = float(manifest["min_gap"])
    if not selected:
        raise SystemExit("No passing pairs in manifest")

    all_rows: list[dict] = []
    semantic_groups: dict[tuple, list[float]] = defaultdict(list)
    head_groups: dict[tuple, list[float]] = defaultdict(list)
    controls: list[dict] = []
    baseline_rows: list[dict] = []
    head_rows_by_template: dict[str, list[dict]] = {}

    for template in templates:
        filename = out / f"residual_{template}.json"
        result = ExperimentResult.load(filename)
        rows = score(result, metric="logit_diff", min_denominator=min_gap)
        for i, baseline in enumerate(result.baselines):
            p = selected[i]
            baseline_rows.append({
                "template": template, "id": p["id"],
                "clean_prompt": result.pairs[i].clean,
                "corrupted_prompt": result.pairs[i].corrupted,
                "clean_target": result.pairs[i].clean_target,
                "corrupted_target": result.pairs[i].corrupted_target,
                "clean_top1": baseline.clean.top_token,
                "corrupted_top1": baseline.corrupted.top_token,
                "clean_logit_diff": baseline.clean.target_logits[0] - baseline.clean.target_logits[1],
                "corrupted_logit_diff": baseline.corrupted.target_logits[0] - baseline.corrupted.target_logits[1],
            })
        for r in rows:
            p = selected[r["pair"]]
            positions = p["per_template"][template]
            base = {"template": template, "id": p["id"], "kind": "residual", **r}
            all_rows.append(base)
            for role, pos in (("country_token", positions["country_position"]),
                              ("final_token", positions["final_position"])):
                if r["position"] == pos:
                    semantic_groups[(template, role, r["layer"])].append(r["recovery"])
            if r["layer"] == 0 and r["position"] == positions["country_position"]:
                delta = abs(r["score"] - r["clean_score"])
                controls.append({
                    "template": template, "id": p["id"],
                    "control": "patch country token residual at layer 0",
                    "clean_logit_diff": r["clean_score"],
                    "patched_logit_diff": r["score"],
                    "abs_score_error": delta,
                    "recovery": r["recovery"],
                    "passed": delta < 0.001,
                })

        head_path = out / f"heads_{template}.json"
        if manifest["heads"]:
            if not head_path.exists():
                raise FileNotFoundError(f"Missing head results requested by manifest: {head_path}")
            hrows = load_scored(head_path, min_gap)
            head_rows_by_template[template] = hrows
            for r in hrows:
                p = selected[r["pair"]]
                all_rows.append({"template": template, "id": p["id"], "kind": "head", **r})
                head_groups[(template, r["layer"], r["head"])].append(r["recovery"])
            grid, layers, heads = to_grid(
                hrows, row_key="layer", col_key="head", component="z")
            plotting.save_figure(plotting.plot_heatmap(
                grid, layers, heads, title=f"Attention-head recovery — {template}",
                xlabel="Head", ylabel="Layer"), out / f"heads_{template}.png")

    semantic_summary = [
        {"template": t, "role": role, "layer": layer, **summarize(values)}
        for (t, role, layer), values in sorted(semantic_groups.items())
    ]
    attention_summary = [
        {"template": t, "layer": layer, "head": head, **summarize(values)}
        for (t, layer, head), values in sorted(head_groups.items())
    ]
    write_csv(out / "all_patch_scores.csv", all_rows)
    write_csv(out / "baselines.csv", baseline_rows)
    write_csv(out / "semantic_layer_summary.csv", semantic_summary)
    write_csv(out / "attention_head_summary.csv", attention_summary)
    write_csv(out / "sanity_checks.csv", controls)
    for role in ("country_token", "final_token"):
        plot_semantic_layers(semantic_summary, role, templates,
                             out / f"compare_{role}.png")

    head_comparisons = []
    if attention_summary:
        by_template = defaultdict(dict)
        for r in attention_summary:
            by_template[r["template"]][(r["layer"], r["head"])] = r["mean"]
        for a, b in combinations(templates, 2):
            if a not in by_template or b not in by_template:
                continue
            common = sorted(set(by_template[a]) & set(by_template[b]))
            if not common:
                continue
            top_a = set(sorted(common, key=lambda k: by_template[a][k], reverse=True)[:10])
            top_b = set(sorted(common, key=lambda k: by_template[b][k], reverse=True)[:10])
            head_comparisons.append({
                "template_a": a, "template_b": b, "n_heads": len(common),
                "spearman_rank_correlation": spearman(
                    [by_template[a][k] for k in common],
                    [by_template[b][k] for k in common]),
                "top10_overlap_count": len(top_a & top_b),
                "top10_jaccard": len(top_a & top_b) / len(top_a | top_b),
            })
    write_csv(out / "head_comparisons.csv", head_comparisons)

    lines = [
        "# Capital-city phrasing robustness — results",
        "",
        "## Design",
        f"- Model: {manifest['model']}",
        f"- Prompt templates: {', '.join(templates)}",
        f"- Passing pairs (same across all templates): {len(selected)} / {len(manifest['candidate_pool'])}",
        f"- Baseline threshold: opposite preferences, gap >= {min_gap} logits",
        f"- Repository commit: {manifest.get('git_sha') or 'not recorded'}",
        "- Hypothesis (recorded before patches): " + manifest["hypothesis"],
        "",
        "## Data quality / controls",
        f"- Layer-0 country-token restoration: {sum(bool(c['passed']) for c in controls)} / {len(controls)} passed (abs logit-difference error < 0.001).",
        "- See screening.csv for **all** candidates and exclusion reasons.",
        "- See baselines.csv for the actual top predicted tokens (pairwise preference is not top-1 correctness).",
        "",
        "## Residual-stream results (descriptive)",
        "Each mean is across the same passing country pairs. Recovery is not a fraction of a circuit explained.",
        "",
    ]
    for t in templates:
        lines.append(f"### {t}")
        for role in ("country_token", "final_token"):
            relevant = [r for r in semantic_summary if r["template"] == t and
                        r["role"] == role and np.isfinite(r["mean"])]
            relevant.sort(key=lambda r: r["mean"], reverse=True)
            top = ", ".join(f"L{r['layer']} ({r['mean']:.2f})" for r in relevant[:3]) or "none"
            lines.append(f"- {role}: highest mean recoveries: {top}")
        lines.append("")
    if attention_summary:
        lines += ["## Attention-head results", ""]
        for t in templates:
            entries = [r for r in attention_summary if r["template"] == t and
                       np.isfinite(r["mean"])]
            entries.sort(key=lambda r: r["mean"], reverse=True)
            top = ", ".join(f"L{r['layer']}H{r['head']} ({r['mean']:.2f})"
                            for r in entries[:5]) or "none"
            lines.append(f"- {t}: {top}")
        lines += ["", "### Similarity across templates", ""]
        for r in head_comparisons:
            lines.append(f"- {r['template_a']} vs {r['template_b']}: Spearman "
                         f"{r['spearman_rank_correlation']:.2f}, "
                         f"top-10 overlap {r['top10_overlap_count']}/10")
    lines += [
        "", "## Interpretation / follow-up (fill in after inspecting figures)",
        "", "- Observation: TBD", "- Plausible alternative explanations: TBD",
        "- Next controlled test: TBD", "",
        "## Limitations", "",
        "- Results are conditional on pre-screened prompts that work in all three templates; this can exaggerate apparent robustness.",
        "- Input token positions differ across templates; cross-template plots compare semantic locations (country/final).",
        "- Attention-head scores are for z at the final token only; they do not isolate a full circuit.",
        "- Patching can change predictions without directly identifying human-readable stored knowledge.",
        "- Country pairs are not a representative random sample; no generalization or novelty claim is justified.",
        "", "## Reproducibility", "",
        "- Original forward-pass readouts and configs: residual_*.json and heads_*.json (when enabled)",
        "- Flat patch-level outcomes: all_patch_scores.csv", "- Screening decision record: screening.csv",
        "- Controls: sanity_checks.csv", "- Metadata and predeclared hypothesis: manifest.json",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Analyzed {len(all_rows)} patch records across {len(selected)} pairs and {len(templates)} templates.")
    print(f"Saved report: {out / 'report.md'}")
    if controls and not all(c["passed"] for c in controls):
        print("WARNING: Some layer-0 country-token sanity checks failed. Investigate before interpreting results.")


if __name__ == "__main__":
    main()
