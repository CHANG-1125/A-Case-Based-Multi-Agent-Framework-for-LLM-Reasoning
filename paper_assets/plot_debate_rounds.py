#!/usr/bin/env python3
"""Plot impact of debate rounds (ablation style)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot(round_labels: list[str], acc: list[float], output: Path) -> None:
    # publication-ish style
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["grid.linestyle"] = "--"

    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=300)
    ax.plot(
        round_labels,
        acc,
        marker="o",
        markersize=9,
        linestyle="-",
        linewidth=2.4,
        color="#ED8536",
        markerfacecolor="#5A9E54",
        markeredgecolor="black",
        markeredgewidth=0.9,
    )
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_xlabel("Maximum Allowed Debate Iterations", fontsize=11)
    ax.set_title("Impact of Multi-Agent Debate Length", fontsize=13, pad=10)
    lo = min(acc) - 0.03
    hi = max(acc) + 0.02
    ax.set_ylim(max(0.0, lo), min(1.0, hi))

    for x, y in zip(round_labels, acc):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("paper_assets/fig8_debate_rounds.png"))
    ap.add_argument(
        "--use_real",
        action="store_true",
        help="Use real numbers from provided json files (rag as round 0, ours files for rounds >=1).",
    )
    ap.add_argument("--rag_json", type=Path, default=Path("results/run.json"))
    ap.add_argument("--round1_json", type=Path, default=Path("results/run_v5_ours500_gpt4omini.json"))
    ap.add_argument("--round2_json", type=Path, default=Path("results/debate2.json"))
    ap.add_argument("--round3_json", type=Path, default=Path("results/debate3.json"))
    args = ap.parse_args()

    if args.use_real:
        labels = ["0\n(RAG)", "1 Round", "2 Rounds", "3 Rounds"]
        vals: list[float] = []
        files = [args.rag_json, args.round1_json, args.round2_json, args.round3_json]
        keys = ["rag", "ours", "ours", "ours"]
        for p, k in zip(files, keys):
            if not p.exists():
                raise SystemExit(f"Missing file for real plotting: {p}")
            d = json.loads(p.read_text(encoding="utf-8"))
            if k not in d.get("methods", {}):
                raise SystemExit(f"File {p} has no methods.{k}")
            vals.append(float(d["methods"][k]["accuracy"]))
        plot(labels, vals, args.output)
        print(f"Saved real-data plot to {args.output}")
        return

    # default placeholder curve (replace with real data when available)
    labels = ["0\n(RAG Baseline)", "1 Round", "2 Rounds", "3 Rounds", "≥4 Rounds"]
    vals = [0.926, 0.935, 0.942, 0.912, 0.885]
    plot(labels, vals, args.output)
    print(f"Saved placeholder plot to {args.output}")


if __name__ == "__main__":
    main()

