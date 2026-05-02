"""Aggregate metrics across all runs and write a markdown summary."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def load(name: str):
    p = RESULTS / name / "metrics.json"
    if not p.exists():
        return None
    with open(p, "r") as f:
        return json.load(f)


def main():
    runs = {
        "Baseline (Crystal Transformer)": load("baseline_v2") or load("baseline"),
        "DAST (ours)": load("improved_v2") or load("improved"),
        "Local-only ablation": load("ablate_local_only"),
        "DAST -no virtual": load("ablate_no_virtual"),
        "DAST -no lattice": load("ablate_no_lattice"),
    }
    runs = {k: v for k, v in runs.items() if v}
    if not runs:
        print("No metrics.json files found.", file=sys.stderr)
        return

    out = ROOT / "results" / "summary.md"
    lines = ["# Experiment summary", ""]
    lines.append("| Model | Params (M) | Best val MAE | Test MAE | Test RMSE |")
    lines.append("|---|---|---|---|---|")
    for name, data in runs.items():
        params_m = data["n_params"] / 1e6
        lines.append(
            f"| {name} | {params_m:.3f} | {data['best_val_mae']:.4f} | "
            f"{data['test_mae']:.4f} | {data['test_rmse']:.4f} |"
        )
    lines.append("")
    lines.append("## Validation MAE per epoch")
    for name, data in runs.items():
        history = data.get("history") or []
        if not history:
            continue
        lines.append(f"### {name}")
        lines.append("epoch | train MAE | val MAE | val RMSE | lr")
        for h in history:
            lines.append(
                f"{h['epoch']} | {h['train_mae']:.4f} | {h['val_mae']:.4f} | "
                f"{h['val_rmse']:.4f} | {h['lr']:.2e}"
            )
        lines.append("")

    out.write_text("\n".join(lines))
    print("Wrote", out)


if __name__ == "__main__":
    main()
