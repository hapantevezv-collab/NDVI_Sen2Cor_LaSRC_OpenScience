#!/usr/bin/env python3
"""Figure 3 - Sentinel-2 NDVI for farm 3948 on 7 August 2019 (T18NXJ) across L1C/TOA,
Sen2Cor L2A/SR and LaSRC L2A/SR.

Same processing as the original script `article/plot_ndvi_three_pipelines_finca_20190807.py`
(L1C 10 m grid as reference, bilinear reprojection of Sen2Cor and LaSRC, 300 m context buffer,
common valid-pixel mask, means within the farm polygon, common NDVI scale 0-1, viridis).
Layout changes only (P011): figure built at the final print width (170 mm), no suptitle (the
caption carries it), pipeline name and mean on separate lines with explicit spacing, horizontal
colour bar with its label below the ticks, and an automatic overlap check that aborts if any
text overlaps.

Usage (from A3_lasrc/):
    python3 NDVI_Sen2Cor_LaSRC_OpenScience/code/figures/figure_03_ndvi_comparison.py \
        --raster-dir article/ndvi_local --vector data/PO_LDO_EI_2019.shp \
        --out-dir NDVI_Sen2Cor_LaSRC_OpenScience/manuscript/source/figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.plot import plotting_extent
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window, from_bounds

DATE_TAG, TILE = "20190807", "T18NXJ"
DATE_LABEL = "7 August 2019"
FARM_ID = 3948
ID_FIELDS = ("ID_Predio", "OBJECTID", "ID_POL", "FID")
CONTEXT_BUFFER_M = 300.0
NDVI_MIN, NDVI_MAX, CMAP, DPI = 0.0, 1.0, "viridis", 600
WIDTH_MM = 170.0
PIPELINES = [("L1C", "L1C / TOA"), ("Sen2Cor", "Sen2Cor L2A / SR"), ("LaSRC", "LaSRC L2A / SR")]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})


def read_reference_window(ds, geom):
    w = from_bounds(*geom.bounds, transform=ds.transform).round_offsets().round_lengths()
    w = w.intersection(Window(0, 0, ds.width, ds.height))
    a = ds.read(1, window=w, masked=True).astype("float32").filled(np.nan)
    if ds.nodata is not None:
        a[a == ds.nodata] = np.nan
    a[(a < -1) | (a > 1)] = np.nan
    return a, ds.window_transform(w)


def reproject_to(ds, shape, transform, crs):
    dst = np.full(shape, np.nan, dtype=np.float32)
    reproject(source=rasterio.band(ds, 1), destination=dst, src_transform=ds.transform,
              src_crs=ds.crs, src_nodata=ds.nodata, dst_transform=transform, dst_crs=crs,
              dst_nodata=np.nan, resampling=Resampling.bilinear)
    dst[(dst < -1) | (dst > 1)] = np.nan
    return dst


def bbox_mm(artist, renderer, fig):
    bb = artist.get_window_extent(renderer)
    return bb.transformed(fig.dpi_scale_trans.inverted())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raster-dir", required=True)
    ap.add_argument("--vector", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    rdir, out = Path(args.raster_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tifs = {k: rdir / f"NDVI_{k}_{DATE_TAG}_{TILE}.tif" for k, _ in PIPELINES}
    for k, p in tifs.items():
        if not p.is_file():
            sys.exit(f"Missing raster {k}: {p}")

    gdf = gpd.read_file(args.vector)
    id_field = next((f for f in ID_FIELDS if f in gdf.columns), None)
    if id_field is None:
        sys.exit(f"No ID field {ID_FIELDS} in {args.vector}; fields: {list(gdf.columns)}")
    try:
        sel = gdf[gdf[id_field].astype(float) == float(FARM_ID)]
    except (TypeError, ValueError):
        sel = gdf[gdf[id_field].astype(str) == str(FARM_ID)]
    if sel.empty:
        sys.exit(f"No feature with {id_field} = {FARM_ID}")
    print(f"Vector: {args.vector} | field {id_field} = {FARM_ID} | features matched: {len(sel)}")

    with rasterio.open(tifs["L1C"]) as ref:
        sel = sel.to_crs(ref.crs)
        geom = sel.geometry.union_all() if hasattr(sel.geometry, "union_all") else sel.geometry.unary_union
        l1c, tr = read_reference_window(ref, geom.buffer(CONTEXT_BUFFER_M))
        crs, shape = ref.crs, l1c.shape
        extent = plotting_extent(l1c, tr)
    arrs = {"L1C": l1c}
    for k in ("Sen2Cor", "LaSRC"):
        with rasterio.open(tifs[k]) as ds:
            arrs[k] = reproject_to(ds, shape, tr, crs)

    common = np.isfinite(arrs["L1C"]) & np.isfinite(arrs["Sen2Cor"]) & np.isfinite(arrs["LaSRC"])
    poly = geometry_mask([geom], out_shape=shape, transform=tr, invert=True) & common
    stats = {"farm_id": FARM_ID, "id_field": id_field, "features": len(sel), "date": DATE_LABEL,
             "tile": TILE, "n_common_pixels_within_polygon": int(poly.sum())}
    for k, _ in PIPELINES:
        v = arrs[k][poly]
        stats[f"mean_ndvi_{k.lower()}"] = float(v.mean())
        stats[f"median_ndvi_{k.lower()}"] = float(np.median(v))
    stats_csv = out / "figure_03_ndvi_comparison_stats.csv"
    pd.DataFrame([stats]).to_csv(stats_csv, index=False)
    print("Common pixels in polygon:", stats["n_common_pixels_within_polygon"])
    for k, _ in PIPELINES:
        print(f"  mean NDVI {k}: {stats[f'mean_ndvi_{k.lower()}']:.3f}")

    # ---------------- figure at final print width ----------------
    h, w = shape
    panel_w_in = (WIDTH_MM / 25.4 - 0.25) / 3
    fig_h_in = panel_w_in * h / w + 1.15          # titles + colour bar band
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, fig_h_in))
    gs = fig.add_gridspec(2, 3, height_ratios=[h / w * panel_w_in, 0.08], hspace=0.30,
                          wspace=0.04, left=0.01, right=0.99, top=0.84, bottom=0.17)
    sel_plot = sel.copy()
    titles, subtitles, im = [], [], None
    for i, (k, name) in enumerate(PIPELINES):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(np.where(common, arrs[k], np.nan), extent=extent, origin="upper",
                       cmap=CMAP, vmin=NDVI_MIN, vmax=NDVI_MAX, interpolation="nearest")
        sel_plot.boundary.plot(ax=ax, color="white", linewidth=1.6, zorder=5)
        sel_plot.boundary.plot(ax=ax, color="black", linewidth=0.5, zorder=6)
        ax.text(0.02, 0.98, f"({chr(97 + i)})", transform=ax.transAxes, ha="left", va="top",
                fontsize=8.5, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.9), zorder=10)
        titles.append(ax.text(0.5, 1.0, name, transform=ax.transAxes, ha="center", va="bottom",
                              fontsize=9, fontweight="bold"))
        subtitles.append(ax.text(0.5, 1.0, f"Mean NDVI within farm = {stats[f'mean_ndvi_{k.lower()}']:.3f}",
                                 transform=ax.transAxes, ha="center", va="bottom", fontsize=8))
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        for s in ax.spines.values():
            s.set_linewidth(0.4)
    # stack: subtitle 2 pt above the map, title 3 pt above the subtitle (offsets in points)
    from matplotlib.transforms import offset_copy
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for t, s in zip(titles, subtitles):
        s.set_transform(offset_copy(s.get_transform(), fig=fig, y=2.5, units="points"))
        sub_h_pt = s.get_window_extent(r).height * 72 / fig.dpi
        t.set_transform(offset_copy(t.get_transform(), fig=fig, y=2.5 + sub_h_pt + 3.0, units="points"))

    cax = fig.add_subplot(gs[1, :])
    pos = cax.get_position()
    cax.set_position([pos.x0 + pos.width * 0.25, pos.y0, pos.width * 0.5, pos.height])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=7.5, length=2.5, width=0.4, pad=1.5)
    cb.outline.set_linewidth(0.4)
    cb.set_label("NDVI", fontsize=8.5, labelpad=3)

    # ---------------- overlap check (aborts on failure) ----------------
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = {}
    for i, (t, s) in enumerate(zip(titles, subtitles)):
        boxes[f"title{i}"] = t.get_window_extent(r)
        boxes[f"subtitle{i}"] = s.get_window_extent(r)
    boxes["cbar_label"] = cb.ax.xaxis.label.get_window_extent(r)
    tick_boxes = [tl.get_window_extent(r) for tl in cb.ax.get_xticklabels() if tl.get_text()]
    fig_bb = fig.bbox
    problems = []
    names = list(boxes)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            if boxes[names[a]].overlaps(boxes[names[b]]):
                problems.append(f"{names[a]} overlaps {names[b]}")
    for j, tb in enumerate(tick_boxes):
        if tb.overlaps(boxes["cbar_label"]):
            problems.append(f"colour-bar tick {j} overlaps colour-bar label")
    for n, bb in boxes.items():
        if bb.x0 < 0 or bb.y0 < 0 or bb.x1 > fig_bb.x1 or bb.y1 > fig_bb.y1:
            problems.append(f"{n} outside the canvas")
    print(f"Overlap check: {len(problems)} problems")
    for p in problems:
        print("  -", p)
    if problems:
        sys.exit("Figure 3 NOT written.")

    png, pdf = out / "figure_03_ndvi_comparison.png", out / "figure_03_ndvi_comparison.pdf"
    fig.savefig(png, dpi=DPI, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Figure size: {WIDTH_MM:.0f} x {fig_h_in * 25.4:.0f} mm")
    print("Written:", png, pdf, stats_csv, sep="\n  ")


if __name__ == "__main__":
    main()
