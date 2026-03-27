#!/usr/bin/env python3
"""Plot failure-mode distribution as a donut chart."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def parse_counts(s: str) -> list[int]:
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    if len(vals) != 4:
        raise ValueError("counts must contain exactly 4 integers, e.g. '18,13,9,4'")
    if any(v < 0 for v in vals):
        raise ValueError("counts must be non-negative")
    if sum(vals) == 0:
        raise ValueError("counts sum cannot be zero")
    return vals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("paper_assets/fig9_error_distribution.png"))
    ap.add_argument(
        "--counts",
        type=str,
        default="40,30,20,10",
        help="Either percentages summing ~100 or raw counts for the 4 categories.",
    )
    ap.add_argument(
        "--title",
        type=str,
        default="Distribution of Failure Modes (v5 Framework)",
    )
    args = ap.parse_args()

    labels = [
        "Negative Transfer\n(Bad Retrieval)",
        "Critic Missed Error\n(Echo Chamber)",
        "Critic Hallucinated\n(Over-correction)",
        "Other Calculation Errors",
    ]
    values = parse_counts(args.counts)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
        }
    )
    fig = plt.figure(figsize=(10.8, 5.8), dpi=400)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.02)
    ax = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1])

    colors = ["#8DA0CB", "#5A9E54", "#ED8536", "#E5C494"]
    explode = (0.05, 0, 0, 0)

    total = sum(values)
    legend_labels = [f"{lab}  ({v/total*100:.1f}%)" for lab, v in zip(labels, values)]

    wedges, _ = ax.pie(
        values,
        explode=explode,
        labels=None,  # keep chart clean; use legend outside
        colors=colors,
        autopct=None,  # percentages moved to legend for better readability
        startangle=140,
        wedgeprops={"linewidth": 1, "edgecolor": "black"},
        textprops={"fontsize": 11},
    )

    centre_circle = plt.Circle((0, 0), 0.70, fc="white")
    ax.add_artist(centre_circle)
    ax.set_aspect("equal")

    ax.set_title(args.title, fontsize=13, pad=12)
    ax_leg.axis("off")
    ax_leg.legend(
        wedges,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(0.0, 0.5),
        frameon=False,
        fontsize=10.0,
    )
    fig.subplots_adjust(left=0.03, right=0.99, top=0.9, bottom=0.06)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

