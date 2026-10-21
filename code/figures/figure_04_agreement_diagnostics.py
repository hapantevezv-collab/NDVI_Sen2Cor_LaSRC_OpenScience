#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rebuild Figure 4 as a legible 3 x 2 agreement figure.

Rows:
1. LaSRC vs L1C
2. Sen2Cor vs L1C
3. LaSRC vs Sen2Cor

Columns:
1. Deming regression
2. Bland-Altman plot

The script requires a record-level CSV containing the three NDVI columns.
It auto-detects common column names, or they can be supplied explicitly.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def find_column(df: pd.DataFrame, explicit: str | None, groups: list[list[str]]) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Column not found: {explicit}")
        return explicit

    normalized = {column: norm(column) for column in df.columns}
    for alternatives in groups:
        for column, key in normalized.items():
            if all(token in key for token in alternatives):
                return column

    raise ValueError(
        "Could not identify a required NDVI column.\n"
        f"Available columns:\n{list(df.columns)}"
    )


def lin_ccc(x: np.ndarray, y: np.ndarray) -> float:
    mx, my = np.mean(x), np.mean(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    cov = np.cov(x, y, ddof=1)[0, 1]
    denominator = vx + vy + (mx - my) ** 2
    return float(2 * cov / denominator) if denominator > 0 else np.nan


def deming_fit(x: np.ndarray, y: np.ndarray, error_ratio: float = 1.0) -> tuple[float, float]:
    xm, ym = np.mean(x), np.mean(y)
    sxx = np.var(x, ddof=1)
    syy = np.var(y, ddof=1)
    sxy = np.cov(x, y, ddof=1)[0, 1]

    if np.isclose(sxy, 0.0):
        return np.nan, np.nan

    slope = (
        syy
        - error_ratio * sxx
        + np.sqrt((syy - error_ratio * sxx) ** 2 + 4 * error_ratio * sxy**2)
    ) / (2 * sxy)
    intercept = ym - slope * xm
    return float(intercept), float(slope)


def agreement_stats(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    difference = y - x
    mean_difference = float(np.mean(difference))
    sd_difference = float(np.std(difference, ddof=1))
    rmsd = float(np.sqrt(np.mean(difference**2)))
    ccc = lin_ccc(x, y)
    intercept, slope = deming_fit(x, y)
    return {
        "n": len(x),
        "mean_difference": mean_difference,
        "rmsd": rmsd,
        "ccc": ccc,
        "intercept": intercept,
        "slope": slope,
        "loa_low": mean_difference - 1.96 * sd_difference,
        "loa_high": mean_difference + 1.96 * sd_difference,
    }


def clean_pair(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    pair = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"),
                         "y": pd.to_numeric(y, errors="coerce")}).dropna()
    return pair["x"].to_numpy(), pair["y"].to_numpy()


def draw_deming(ax, x: np.ndarray, y: np.ndarray, title: str, panel: str) -> None:
    stats = agreement_stats(x, y)
    lower = min(np.quantile(x, 0.001), np.quantile(y, 0.001))
    upper = max(np.quantile(x, 0.999), np.quantile(y, 0.999))
    padding = max(0.015, 0.04 * (upper - lower))
    limits = (lower - padding, upper + padding)

    ax.scatter(x, y, s=7, alpha=0.28, linewidths=0, rasterized=True)
    ax.plot(limits, limits, linestyle="--", linewidth=1.0, label="1:1")
    xx = np.linspace(*limits, 200)
    ax.plot(xx, stats["intercept"] + stats["slope"] * xx, linewidth=1.4, label="Deming")

    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Reference-pipeline NDVI", fontsize=9)
    ax.set_ylabel("Comparison-pipeline NDVI", fontsize=9)
    ax.set_title(f"{panel}  {title}", loc="left", fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)

    annotation = (
        f"n = {stats['n']:,}\n"
        f"CCC = {stats['ccc']:.3f}\n"
        f"slope = {stats['slope']:.3f}\n"
        f"intercept = {stats['intercept']:+.3f}"
    )
    ax.text(
        0.03, 0.97, annotation,
        transform=ax.transAxes,
        ha="left", va="top", fontsize=7.5,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
              "edgecolor": "0.75", "alpha": 0.88},
    )
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_bland_altman(ax, x: np.ndarray, y: np.ndarray, title: str, panel: str) -> None:
    stats = agreement_stats(x, y)
    mean_xy = (x + y) / 2
    difference = y - x

    ax.scatter(mean_xy, difference, s=7, alpha=0.28, linewidths=0, rasterized=True)
    ax.axhline(0, linestyle=":", linewidth=0.9)
    ax.axhline(stats["mean_difference"], linewidth=1.3)
    ax.axhline(stats["loa_low"], linestyle="--", linewidth=1.0)
    ax.axhline(stats["loa_high"], linestyle="--", linewidth=1.0)

    ymin = min(np.quantile(difference, 0.002), stats["loa_low"])
    ymax = max(np.quantile(difference, 0.998), stats["loa_high"])
    padding = max(0.01, 0.08 * (ymax - ymin))
    ax.set_ylim(ymin - padding, ymax + padding)

    ax.set_xlabel("Mean NDVI of the two pipelines", fontsize=9)
    ax.set_ylabel("Paired NDVI difference", fontsize=9)
    ax.set_title(f"{panel}  {title}", loc="left", fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)

    annotation = (
        f"mean = {stats['mean_difference']:+.3f}\n"
        f"RMSD = {stats['rmsd']:.3f}\n"
        f"LoA = [{stats['loa_low']:+.3f}, {stats['loa_high']:+.3f}]"
    )
    ax.text(
        0.03, 0.97, annotation,
        transform=ax.transAxes,
        ha="left", va="top", fontsize=7.5,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
              "edgecolor": "0.75", "alpha": 0.88},
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Record-level common three-pipeline CSV")
    parser.add_argument("--l1c-col")
    parser.add_argument("--sen2cor-col")
    parser.add_argument("--lasrc-col")
    parser.add_argument("--out", required=True)
    parser.add_argument("--pdf")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    l1c_col = find_column(
        df, args.l1c_col,
        [["ndvi", "l1c"], ["l1c", "mean"], ["l1c"]],
    )
    sen2cor_col = find_column(
        df, args.sen2cor_col,
        [["ndvi", "sen2cor"], ["sen2cor", "mean"], ["sen2cor"], ["s2c"]],
    )
    lasrc_col = find_column(
        df, args.lasrc_col,
        [["ndvi", "lasrc"], ["lasrc", "mean"], ["lasrc"]],
    )

    pairs = [
        ("LaSRC vs L1C", l1c_col, lasrc_col),
        ("Sen2Cor vs L1C", l1c_col, sen2cor_col),
        ("LaSRC vs Sen2Cor", sen2cor_col, lasrc_col),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(8.25, 10.4))
    panels = iter(["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"])

    for row, (title, reference, comparison) in enumerate(pairs):
        x, y = clean_pair(df[reference], df[comparison])
        draw_deming(axes[row, 0], x, y, f"{title} — Deming", next(panels))
        draw_bland_altman(axes[row, 1], x, y, f"{title} — Bland–Altman", next(panels))

    fig.subplots_adjust(left=0.10, right=0.98, top=0.98, bottom=0.06,
                        hspace=0.34, wspace=0.27)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight")
    if args.pdf:
        fig.savefig(args.pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure written to: {output}")
    print(f"Columns used: L1C={l1c_col}; Sen2Cor={sen2cor_col}; LaSRC={lasrc_col}")


if __name__ == "__main__":
    main()
