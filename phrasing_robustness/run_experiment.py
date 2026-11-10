"""Run a controlled GPT-2 capital-city phrasing-robustness patching study.

Run from the interpretability-toolkit repository root after `pip install -e .`:
    python phrasing_robustness/run_experiment.py --pilot
    python phrasing_robustness/run_experiment.py
    python phrasing_robustness/run_experiment.py --heads

The runner pre-screens candidate pairs BEFORE running patch sweeps. A pair must pass
all three templates so comparisons use exactly the same set of countries.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from interp_toolkit import PromptPair, SweepConfig, load_model, run_patching_experiment, score
from interp_toolkit import plotting

TEMPLATES = {
    "capital_of": "The capital of {country} is",
    "capital_city": "The capital city of {country} is",
    "possessive": "{country}'s capital is",
}

# Fixed candidate pool, selected before examining any patching results.
# City labels must be SINGLE GPT-2 next tokens (including their leading space).
# Some candidates may be excluded by the model/tokenizer validation below.
CANDIDATES = [
    ("france_italy", "France", "Paris", "Italy", "Rome"),
    ("germany_spain", "Germany", "Berlin", "Spain", "Madrid"),
    ("japan_china", "Japan", "Tokyo", "China", "Beijing"),
    ("norway_sweden", "Norway", "Oslo", "Sweden", "Stockholm"),
    ("austria_belgium", "Austria", "Vienna", "Belgium", "Brussels"),
    ("portugal_greece", "Portugal", "Lisbon", "Greece", "Athens"),
    ("finland_denmark", "Finland", "Helsinki", "Denmark", "Copenhagen"),
    ("hungary_romania", "Hungary", "Budapest", "Romania", "Bucharest"),
    ("russia_canada", "Russia", "Moscow", "Canada", "Ottawa"),
    ("vietnam_thailand", "Vietnam", "Hanoi", "Thailand", "Bangkok"),
    ("cuba_peru", "Cuba", "Havana", "Peru", "Lima"),
    ("iran_iraq", "Iran", "Tehran", "Iraq", "Baghdad"),
    ("turkey_poland", "Turkey", "Ankara", "Poland", "Warsaw"),
    ("netherlands_switzerland", "Netherlands", "Amsterdam", "Switzerland", "Bern"),
]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def target_token_id(model, target: str) -> int | None:
    ids = model.to_tokens(target, prepend_bos=False)[0].tolist()
    return ids[0] if len(ids) == 1 else None


@torch.no_grad()
def screen_one(model, candidate: tuple[str, str, str, str, str],
               template_name: str, template: str, min_gap: float) -> dict:
    pair_id, clean_country, clean_city, corrupted_country, corrupted_city = candidate
    clean_prompt = template.format(country=clean_country)
    corrupted_prompt = template.format(country=corrupted_country)
    row = {
        "id": pair_id, "template": template_name,
        "clean_prompt": clean_prompt, "corrupted_prompt": corrupted_prompt,
        "clean_target": " " + clean_city, "corrupted_target": " " + corrupted_city,
        "accepted": False, "reason": "",
    }

    clean_id = target_token_id(model, row["clean_target"])
    corrupted_id = target_token_id(model, row["corrupted_target"])
    if clean_id is None or corrupted_id is None:
        row["reason"] = "capital name is not a single GPT-2 token"
        return row

    clean_tokens = model.to_tokens(clean_prompt, prepend_bos=True)
    corrupted_tokens = model.to_tokens(corrupted_prompt, prepend_bos=True)
    if clean_tokens.shape != corrupted_tokens.shape:
        row["reason"] = "different prompt token lengths"
        return row
    # The input should differ at EXACTLY ONE token position, representing the country.
    mismatches = (clean_tokens[0] != corrupted_tokens[0]).nonzero(as_tuple=True)[0]
    if mismatches.numel() != 1:
        row["reason"] = f"prompts differ at {mismatches.numel()} token positions (not one)"
        return row
    row["country_position"] = int(mismatches.item())
    row["final_position"] = int(clean_tokens.shape[1] - 1)
    row["seq_len"] = int(clean_tokens.shape[1])

    # The model and the tokens are on the same device.
    device = next(model.parameters()).device
    clean_logits = model(clean_tokens.to(device))[0, -1].float()
    corrupted_logits = model(corrupted_tokens.to(device))[0, -1].float()
    clean_score = float(clean_logits[clean_id] - clean_logits[corrupted_id])
    corrupted_score = float(corrupted_logits[clean_id] - corrupted_logits[corrupted_id])

    row.update({
        "clean_score": clean_score,
        "corrupted_score": corrupted_score,
        "baseline_gap": clean_score - corrupted_score,
        "clean_target_rank": int((clean_logits > clean_logits[clean_id]).sum()) + 1,
        "corrupted_target_rank": int((corrupted_logits > corrupted_logits[corrupted_id]).sum()) + 1,
        "clean_top_token": model.to_string(int(clean_logits.argmax())),
        "corrupted_top_token": model.to_string(int(corrupted_logits.argmax())),
    })
    # Relative preference, NOT necessarily top-1 correctness. Record ranks above.
    if clean_score <= 0 or corrupted_score >= 0:
        row["reason"] = "baseline does not favor opposite capital targets"
    elif clean_score - corrupted_score < min_gap:
        row["reason"] = "clean-corrupted baseline gap below prespecified minimum"
    else:
        row["accepted"] = True
        row["reason"] = "passes prespecified checks"
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="Screen just the first three candidate pairs")
    parser.add_argument("--heads", action="store_true", help="Also sweep 12 x 12 individual attention heads at final token")
    parser.add_argument("--device", default="cpu", help="cpu, auto, cuda, or mps")
    parser.add_argument("--min-gap", type=float, default=1.0,
                        help="Minimum clean-minus-corrupted logit-difference gap (default 1.0)")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Maximum number of passing pairs, in predetermined candidate order")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    if args.min_gap <= 0 or (args.max_pairs is not None and args.max_pairs < 1):
        parser.error("--min-gap and --max-pairs must be positive")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(device=args.device)
    candidates = CANDIDATES[:3] if args.pilot else CANDIDATES
    screen_rows = [
        screen_one(model, candidate, name, template, args.min_gap)
        for candidate in candidates for name, template in TEMPLATES.items()
    ]
    write_csv(out / "screening.csv", screen_rows)

    by_key = {(r["id"], r["template"]): r for r in screen_rows}
    selected = []
    for candidate in candidates:
        pair_id, clean_country, clean_city, corrupted_country, corrupted_city = candidate
        checks = {name: by_key[(pair_id, name)] for name in TEMPLATES}
        if not all(row["accepted"] for row in checks.values()):
            continue
        selected.append({
            "id": pair_id, "clean_country": clean_country, "clean_city": clean_city,
            "corrupted_country": corrupted_country, "corrupted_city": corrupted_city,
            "per_template": {
                name: {
                    "country_position": row["country_position"],
                    "final_position": row["final_position"],
                    "seq_len": row["seq_len"],
                } for name, row in checks.items()
            },
        })
    if args.max_pairs is not None:
        selected = selected[:args.max_pairs]

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "model": "openai-community/gpt2", "device": args.device,
        "pilot": args.pilot, "heads": args.heads, "min_gap": args.min_gap,
        "screening_rule": "exactly one input-token mismatch, single-token capital targets, "
                          "clean logit diff > 0, corrupted logit diff < 0, gap >= min_gap; "
                          "must pass all three templates",
        "important_caveat": "Screening is conditional on baseline behavior; findings apply to this selected subset, "
                            "not all country facts. Target need not be top-1 prediction.",
        "hypothesis": "Some influential layer/head locations for capital-city recall will be shared "
                      "across phrasings when the same country pairs are used; similarity metrics "
                      "are descriptive and not statistical significance tests.",
        "templates": TEMPLATES,
        "candidate_pool": [dict(id=c[0], clean_country=c[1], clean_city=c[2],
                                corrupted_country=c[3], corrupted_city=c[4]) for c in candidates],
        "selected": selected,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Screened {len(candidates)} candidate pairs x {len(TEMPLATES)} templates; "
          f"{len(selected)} pairs pass all templates.")
    print(f"Screening details: {out / 'screening.csv'}")
    if not selected:
        raise SystemExit("No complete-case pairs passed. Review screening.csv; "
                         "do not silently loosen criteria after inspecting patch results.")

    for template_name, template in TEMPLATES.items():
        pairs = [PromptPair(
            clean=template.format(country=p["clean_country"]),
            corrupted=template.format(country=p["corrupted_country"]),
            clean_target=" " + p["clean_city"],
            corrupted_target=" " + p["corrupted_city"],
        ) for p in selected]
        # Stage 1: all layers and token positions, one residual vector at a time.
        resid_cfg = SweepConfig(components=("resid_pre",), per_position=True)
        print(f"[{template_name}] Residual sweep: {len(pairs)} pair(s) ...", flush=True)
        resid_result = run_patching_experiment(model, pairs, resid_cfg)
        resid_result.save(out / f"residual_{template_name}.json")
        resid_rows = score(resid_result, metric="logit_diff", min_denominator=args.min_gap)
        write_csv(out / f"residual_{template_name}.csv", resid_rows)
        plotting.save_figure(
            plotting.plot_recovery_heatmap(
                resid_rows, "resid_pre", title=f"Residual recovery — {template_name}"
            ), out / f"residual_{template_name}.png"
        )
        print(f"  {len(resid_result.records)} patches, {resid_result.metadata['seconds']}s")

        if args.heads:
            # Stage 2: every head (z) at the FINAL token position, not all positions.
            head_cfg = SweepConfig(components=("z",), positions=(-1,), per_head=True)
            print(f"[{template_name}] Head sweep: {len(pairs)} pair(s) ...", flush=True)
            head_result = run_patching_experiment(model, pairs, head_cfg)
            head_result.save(out / f"heads_{template_name}.json")
            head_rows = score(head_result, metric="logit_diff", min_denominator=args.min_gap)
            write_csv(out / f"heads_{template_name}.csv", head_rows)
            print(f"  {len(head_result.records)} patches, {head_result.metadata['seconds']}s")

    print(f"Done. Run: python phrasing_robustness/analyze_results.py --out {out}")


if __name__ == "__main__":
    main()
