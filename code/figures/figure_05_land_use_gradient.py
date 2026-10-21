#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 5 — V14.

Reads the canonical land-use output (data/derived/land_use_gradient_2019_2020.csv).
Intervals are 95% farm-cluster bootstrap percentile intervals.
"""

from __future__ import annotations

import argparse
import re
import textwrap
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CATEGORY_ORDER = ["UT2", "UT3", "UT4", "UT7", "UT6", "UT8"]

CATEGORY_LABELS = {
    "UT2": "Secondary forest or wetlands undergoing restoration",
    "UT3": "Scattered trees in pasture and vegetative succession",
    "UT4": "Agroforestry crops",
    "UT7": "Intensive silvopastoral system",
    "UT6": "Agricultural land with vegetation cover greater than 80%",
    "UT8": "Other agricultural practices",
}

PAIR_ORDER = ["LaSRC − L1C", "Sen2Cor − L1C", "LaSRC − Sen2Cor"]


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def normalize_category(value: object) -> str:
    match = re.search(r"ut\s*([0-9]+)", str(value), flags=re.IGNORECASE)
    if match:
        return f"UT{match.group(1)}"
    value = str(value).strip().upper()
    return value if value.startswith("UT") else f"UT{value}"


def normalize_pair(value: object) -> str:
    key = normalize_text(value)

    has_lasrc = "lasrc" in key
    has_l1c = "l1c" in key or "toa" in key
    has_sen2cor = "sen2cor" in key or "s2c" in key

    if has_lasrc and has_l1c:
        return "LaSRC − L1C"
    if has_sen2cor and has_l1c:
        return "Sen2Cor − L1C"
    if has_lasrc and has_sen2cor:
        return "LaSRC − Sen2Cor"

    raise ValueError(f"Unrecognized comparison label in column 'par': {value!r}")


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Accept the canonical schema (data/derived/land_use_gradient_2019_2020.csv)."""
    required = {
        "comparison", "land_use_code", "observation_count", "farm_count",
        "management_polygon_count", "mean_difference",
        "mean_difference_ci_low", "mean_difference_ci_high",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            "The land-use CSV does not have the canonical schema.\n"
            f"Missing columns: {missing}\nAvailable columns: {list(df.columns)}"
        )

    out = pd.DataFrame({
        "comparison": df["comparison"].map(normalize_pair),
        "category": df["land_use_code"].map(normalize_category),
        "mean_difference": pd.to_numeric(df["mean_difference"], errors="coerce"),
        "ci_low": pd.to_numeric(df["mean_difference_ci_low"], errors="coerce"),
        "ci_high": pd.to_numeric(df["mean_difference_ci_high"], errors="coerce"),
        "observations": pd.to_numeric(df["observation_count"], errors="coerce"),
        "farms": pd.to_numeric(df["farm_count"], errors="coerce"),
        "polygons": pd.to_numeric(df["management_polygon_count"], errors="coerce"),
    })

    out = out[
        out["category"].isin(CATEGORY_ORDER)
        & out["comparison"].isin(PAIR_ORDER)
    ].dropna(subset=["mean_difference", "ci_low", "ci_high"])

    duplicated = out.duplicated(["comparison", "category"], keep=False)
    if duplicated.any():
        raise ValueError(
            "More than one record was found for the same comparison/category:\n"
            + out.loc[duplicated, ["comparison", "category"]].to_string(index=False)
        )
    return out


def wrapped_category_label(row: pd.Series) -> str:
    category = row["category"]
    label = f"{category}  {CATEGORY_LABELS[category]}"
    wrapped = textwrap.fill(
        label,
        width=48,
        subsequent_indent="     ",
        break_long_words=False,
    )
    counts = (
        f"{int(row['observations']):,} observations; "
        f"{int(row['farms'])} farms; {int(row['polygons'])} polygons"
    )
    return f"{wrapped}\n{counts}"


def compute_limits(data: pd.DataFrame, pairs: list[str]) -> tuple[float, float]:
    subset = data[data["comparison"].isin(pairs)]
    low = float(subset["ci_low"].min())
    high = float(subset["ci_high"].max())
    span = max(high - low, 0.01)
    lower = min(0.0, low - 0.10 * span)
    upper = high + 0.12 * span
    return lower, upper


def draw_panel(
    ax: plt.Axes,
    subset: pd.DataFrame,
    comparison: str,
    panel: str,
    x_limits: tuple[float, float],
) -> None:
    subset = (
        subset.set_index("category")
        .reindex(CATEGORY_ORDER)
        .dropna(subset=["mean_difference"])
        .reset_index()
    )

    y = np.arange(len(subset))
    mean = subset["mean_difference"].to_numpy(float)
    low = subset["ci_low"].to_numpy(float)
    high = subset["ci_high"].to_numpy(float)
    xerr = np.vstack([mean - low, high - mean])

    ax.errorbar(
        mean,
        y,
        xerr=xerr,
        fmt="o",
        markersize=5.5,
        linewidth=1.15,
        capsize=3.0,
    )
    ax.axvline(0.0, linestyle=":", linewidth=0.9)

    ax.set_yticks(y)
    ax.set_yticklabels(
        [wrapped_category_label(row) for _, row in subset.iterrows()],
        fontsize=8.2,
        linespacing=1.12,
    )
    ax.invert_yaxis()
    ax.set_xlim(*x_limits)

    ax.set_title(
        f"{panel}  {comparison}",
        loc="left",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    ax.set_xlabel("Paired mean NDVI difference", fontsize=9.5)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.tick_params(axis="y", length=0, pad=7)

    ax.grid(axis="x", linewidth=0.55, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pdf")
    args = parser.parse_args()

    source = pd.read_csv(args.csv)
    data = prepare_data(source)

    missing_pairs = [
        pair for pair in PAIR_ORDER
        if data[data["comparison"] == pair].empty
    ]
    if missing_pairs:
        raise ValueError(f"Missing comparison(s): {missing_pairs}")

    toa_sr_limits = compute_limits(data, PAIR_ORDER[:2])
    sr_sr_limits = compute_limits(data, [PAIR_ORDER[2]])

    fig, axes = plt.subplots(3, 1, figsize=(8.27, 11.2))

    for ax, comparison, panel in zip(
        axes,
        PAIR_ORDER,
        ["(a)", "(b)", "(c)"],
    ):
        limits = (
            toa_sr_limits
            if comparison in PAIR_ORDER[:2]
            else sr_sr_limits
        )
        draw_panel(
            ax,
            data[data["comparison"] == comparison].copy(),
            comparison,
            panel,
            limits,
        )

    fig.subplots_adjust(
        left=0.48,
        right=0.97,
        top=0.98,
        bottom=0.06,
        hspace=0.48,
    )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight")
    if args.pdf:
        fig.savefig(args.pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure written to: {output}")
    print("\nData used:")
    print(
        data.sort_values(["comparison", "category"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
