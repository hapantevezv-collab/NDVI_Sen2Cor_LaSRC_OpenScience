#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 1 R4 CIERRE — Study area in cascade of scales.
Article A3 (target journal: Remote Sensing Applications: Society and Environment, Elsevier).
Instruction basis: INSTRUCCION_FIGURAS_V2_REF_CHRAIBI.md (2026-10-15/20)
with data/palette/QC inherited from INSTRUCCION_FIGURA1_V1.md.

Execution root: the A3_lasrc workspace (parent folder of this repository)
First-delivery mode (default): script + console + preview PNG only.
Final mode (--final): canonical PDF/SVG/PNG + provenance JSON.
R4 closure: only five prescribed changes; all other R3 components frozen.

No web tiles. No recursive disk searches. No downloads are performed by this script.
All contextual data must already exist under data/external/.
"""

from __future__ import annotations

import argparse
import os
import hashlib
import importlib.util
import json
import math
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import geopandas as gpd
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LinearSegmentedColormap, LightSource, to_rgb
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import ConnectionPatch, Patch, Rectangle
from pyproj import Geod, Transformer
from rasterio.merge import merge
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds
from shapely.geometry import Point, box
from shapely.ops import unary_union

try:
    from shapely import force_2d
except Exception:  # Shapely < 2 fallback
    force_2d = None


# -----------------------------------------------------------------------------
# Fixed publication geometry and style
# -----------------------------------------------------------------------------
MM_PER_IN = 25.4
FIG_W_MM = 170.0
FIG_H_MM = 80.0
MARGIN_MM = 2.0
PANEL_AB_GAP_MM = 6.0
PANEL_BC_GAP_MM = 3.0
MAP_Y0_MM = 17.0
MAP_H_MM = 81.0
PANEL_A_W_MM = 31.0
PANEL_B_W_MM = 74.0
PANEL_C_W_MM = 45.0
PANEL_A_X_MM = 2.0
PANEL_B_X_MM = PANEL_A_X_MM + PANEL_A_W_MM + PANEL_AB_GAP_MM
PANEL_C_X_MM = PANEL_B_X_MM + PANEL_B_W_MM + PANEL_BC_GAP_MM

FONT_FAMILY = "Liberation Sans"
PANEL_LABEL_PT = 9.0
PANEL_TITLE_PT = 8.0
AXIS_PT = 7.0
TICK_PT = 6.5
ANNOT_PT = 6.5
ATTR_PT = 5.5
FRAME_LW = 0.4

COL = {
    "farm": "#1F4E79",
    "accent": "#C0392B",
    "tile": "#6C3483",
    "river": "#5B8DB8",
    "municipal": "#7F7F7F",
    "frame": "#4D4D4D",
    "sea": "#EAF2F8",
    "country": "#EDEDED",
    "meta": "#BFC9D4",
    "text": "#333333",
    "ut2": "#1B7837",
    "ut3": "#7FBF7B",
    "ut4": "#A6611A",
    "ut6": "#DFC27D",
    "ut7": "#018571",
    "ut8": "#BABABA",
    "ut0": "#FFFFFF",
}

UT_COLORS = {
    "0": COL["ut0"],
    "2": COL["ut2"],
    "3": COL["ut3"],
    "4": COL["ut4"],
    "6": COL["ut6"],
    "7": COL["ut7"],
    "8": COL["ut8"],
}

mpl.rcParams.update(
    {
        "font.family": FONT_FAMILY,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "hatch.linewidth": 0.4,
    }
)


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def mm_to_fig_x(mm: float) -> float:
    return mm / FIG_W_MM


def mm_to_fig_y(mm: float) -> float:
    return mm / FIG_H_MM


def axes_rect_mm(x: float, y: float, w: float, h: float):
    return [mm_to_fig_x(x), mm_to_fig_y(y), mm_to_fig_x(w), mm_to_fig_y(h)]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_id(v) -> Optional[str]:
    if pd.isna(v):
        return None
    s = str(v).strip()
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def norm_text(v: str) -> str:
    s = unicodedata.normalize("NFKD", str(v))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().strip()


def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise SystemExit(f"Missing required {label}: {path}")
    return path


def optional_file(path: Path, label: str) -> Optional[Path]:
    if path.exists():
        return path
    print(f"WARNING [{label}] unavailable; using documented fallback: {path}")
    return None


def force_gdf_2d(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    if force_2d is not None:
        out.geometry = out.geometry.map(lambda g: force_2d(g) if g is not None else None)
    return out


def find_name_col(gdf: gpd.GeoDataFrame, candidates: Iterable[str]) -> str:
    lut = {c.lower(): c for c in gdf.columns}
    for c in candidates:
        if c.lower() in lut:
            return lut[c.lower()]
    raise SystemExit(f"No name column found. Columns: {list(gdf.columns)}")


def expand_bounds(bounds, pad_frac=0.08, min_pad=1000.0):
    xmin, ymin, xmax, ymax = map(float, bounds)
    dx = xmax - xmin
    dy = ymax - ymin
    padx = max(dx * pad_frac, min_pad)
    pady = max(dy * pad_frac, min_pad)
    return (xmin - padx, ymin - pady, xmax + padx, ymax + pady)


def fit_bounds_to_panel_aspect(bounds, panel_w_mm: float, panel_h_mm: float):
    """Expand metric bounds so equal data scaling fills a fixed panel box."""
    xmin, ymin, xmax, ymax = map(float, bounds)
    dx, dy = xmax - xmin, ymax - ymin
    if dx <= 0 or dy <= 0:
        raise SystemExit(f"Invalid bounds: {bounds}")
    target = panel_w_mm / panel_h_mm
    current = dx / dy
    cx, cy = (xmin+xmax)/2.0, (ymin+ymax)/2.0
    if current < target:
        dx2 = dy * target
        xmin, xmax = cx-dx2/2.0, cx+dx2/2.0
    elif current > target:
        dy2 = dx / target
        ymin, ymax = cy-dy2/2.0, cy+dy2/2.0
    return (xmin, ymin, xmax, ymax)


def deuteranopia_palette_check():
    """Simple linear deuteranopia simulation; report minimum pairwise RGB distance."""
    # Approximate Machado-style deuteranopia matrix for a qualitative publication check.
    M = np.array([[0.367, 0.861, -0.228],
                  [0.280, 0.673,  0.047],
                  [-0.012, 0.043, 0.969]], dtype=float)
    cols = {k: np.array(to_rgb(UT_COLORS[k])) for k in ["2","3","4","6","7","8"]}
    sims = {k: np.clip(M @ v, 0, 1) for k, v in cols.items()}
    pairs = []
    keys = list(sims)
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            d = float(np.linalg.norm(sims[keys[i]]-sims[keys[j]]))
            pairs.append((d, keys[i], keys[j]))
    d, a, b = min(pairs)
    print(f"  Deuteranopia palette check: minimum simulated RGB distance = {d:.3f} (UT{a} vs UT{b})")
    if d < 0.06:
        print("  WARNING: two UT colours are close under the simple deuteranopia simulation; codes remain explicit in legend.")


def set_frame(ax):
    for sp in ax.spines.values():
        sp.set_linewidth(FRAME_LW)
        sp.set_color(COL["frame"])
        sp.set_visible(True)
    ax.set_facecolor("white")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


def add_panel_heading(fig, x_mm: float, label: str, title: str):
    y = mm_to_fig_y(MAP_Y0_MM + MAP_H_MM + 2.0)
    fig.text(mm_to_fig_x(x_mm), y, label, fontsize=PANEL_LABEL_PT, weight="bold",
             ha="left", va="bottom", color=COL["text"])
    fig.text(mm_to_fig_x(x_mm + 6.5), y, title, fontsize=PANEL_TITLE_PT, weight="bold",
             ha="left", va="bottom", color=COL["text"])


def bbox_intersects(a, b, pad_px=0.0) -> bool:
    return not (
        a.x1 + pad_px <= b.x0 or b.x1 + pad_px <= a.x0 or
        a.y1 + pad_px <= b.y0 or b.y1 + pad_px <= a.y0
    )


def data_box_to_display(ax, geom_box):
    xmin, ymin, xmax, ymax = geom_box.bounds
    p0 = ax.transData.transform((xmin, ymin))
    p1 = ax.transData.transform((xmax, ymax))
    from matplotlib.transforms import Bbox
    return Bbox.from_extents(min(p0[0], p1[0]), min(p0[1], p1[1]),
                             max(p0[0], p1[0]), max(p0[1], p1[1]))


# -----------------------------------------------------------------------------
# Import Figure 2 palette constants (required by common-style contract)
# -----------------------------------------------------------------------------
def import_figure2_colors(root: Path) -> dict:
    p = root / "NDVI_Sen2Cor_LaSRC_OpenScience/code/figures/figure_02_processing_workflow.py"
    require_file(p, "Figure 2 script")
    spec = importlib.util.spec_from_file_location("figure2_palette", p)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot import Figure 2 palette from {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "COL"):
        raise SystemExit("Figure 2 script does not expose COL palette.")
    needed = {"l1c", "s2c", "lasrc"}
    if not needed.issubset(set(mod.COL)):
        raise SystemExit(f"Figure 2 COL is missing required keys: {needed - set(mod.COL)}")
    return {k: mod.COL[k] for k in needed}


# -----------------------------------------------------------------------------
# Data loading / canonical controls
# -----------------------------------------------------------------------------
@dataclass
class Inputs:
    farms_path: Path
    pgo_path: Path
    tiles_path: Path
    adm1_path: Optional[Path]
    adm2_path: Optional[Path]
    rivers_path: Optional[Path]
    countries_path: Optional[Path]
    dem_paths: list[Path]


def resolve_inputs(root: Path) -> Inputs:
    external = root / "data/external"
    dem_dir = external / "copernicus_dem_glo90"
    return Inputs(
        farms_path=require_file(root / "data/PO_LDO_EI_2019.shp", "farm layer"),
        pgo_path=require_file(root / "data/PGO_EI_2019.shp", "management-polygon layer"),
        tiles_path=require_file(root / "data/sentinel2_tiles_18NXJ_18NXK.geojson", "Sentinel-2 tile layer"),
        adm1_path=optional_file(external / "geoboundaries_COL_ADM1.geojson", "ADM1"),
        adm2_path=optional_file(external / "geoboundaries_COL_ADM2.geojson", "ADM2"),
        rivers_path=optional_file(external / "ne_10m_rivers_lake_centerlines/ne_10m_rivers_lake_centerlines.shp", "rivers"),
        countries_path=optional_file(external / "ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp", "countries"),
        dem_paths=sorted(dem_dir.glob("*.tif")) if dem_dir.exists() else [],
    )


def load_project_data(inp: Inputs):
    farms_raw = gpd.read_file(inp.farms_path)
    if farms_raw.crs is None:
        raise SystemExit("Farm layer has no CRS.")
    farms_raw = farms_raw.to_crs(9377)
    farms_raw["_farm"] = farms_raw["ID_Predio"].map(norm_id)
    farms_raw["_area"] = pd.to_numeric(farms_raw["AREA_HA"], errors="coerce")

    if len(farms_raw) != 23:
        raise SystemExit(f"Expected 23 farm features; found {len(farms_raw)}")
    if farms_raw["_farm"].nunique() != 22:
        raise SystemExit(f"Expected 22 unique farms; found {farms_raw['_farm'].nunique()}")

    # Attribute area counted once per farm ID; duplicate feature 438 must not double count.
    area_once = float(farms_raw.groupby("_farm")["_area"].first().sum())
    farms = farms_raw[["_farm", "geometry"]].dissolve(by="_farm", as_index=False)

    if abs(area_once - 855.23) > 0.01:
        raise SystemExit(f"Expected unique-farm AREA_HA sum 855.23; observed {area_once:.3f}")

    farm3948 = farms[farms["_farm"] == "3948"].copy()
    if len(farm3948) != 1:
        raise SystemExit(f"Expected one dissolved farm 3948; found {len(farm3948)}")

    pgo = force_gdf_2d(gpd.read_file(inp.pgo_path))
    if pgo.crs is None:
        raise SystemExit("Management-polygon layer has no CRS.")
    pgo = pgo.to_crs(9377)
    pgo["_farm"] = pgo["ID_Predio"].map(norm_id)
    pgo["_ut"] = pgo["TIPO_UT"].map(norm_id)
    p3948 = pgo[pgo["_farm"] == "3948"].copy()

    if len(p3948) != 14:
        raise SystemExit(f"Expected 14 management polygons for farm 3948; found {len(p3948)}")
    expected_ut = {"0", "2", "3", "6"}
    observed_ut = set(p3948["_ut"].dropna())
    if observed_ut != expected_ut:
        raise SystemExit(f"Farm 3948 UT mismatch. Expected {sorted(expected_ut)}, observed {sorted(observed_ut)}")

    farm_geom = farm3948.geometry.iloc[0]
    n_intersect = int(p3948.geometry.intersects(farm_geom).sum())
    if n_intersect != len(p3948):
        raise SystemExit(f"Spatial verification failed: {n_intersect}/{len(p3948)} polygons intersect farm 3948")

    tiles = gpd.read_file(inp.tiles_path)
    if tiles.crs is None:
        raise SystemExit("Sentinel-2 tile layer has no CRS.")
    if len(tiles) != 2 or set(tiles["tile"].astype(str)) != {"T18NXJ", "T18NXK"}:
        raise SystemExit("Sentinel-2 tile layer must contain exactly T18NXJ and T18NXK.")
    tiles_32618 = tiles.to_crs(32618)
    expected = {
        "T18NXJ": (600000.0, 290220.0, 709800.0, 400020.0),
        "T18NXK": (600000.0, 390220.0, 709800.0, 500020.0),
    }
    for _, r in tiles_32618.iterrows():
        obs = tuple(map(float, r.geometry.bounds))
        exp = expected[str(r["tile"])]
        if any(abs(a - b) > 0.10 for a, b in zip(obs, exp)):
            raise SystemExit(f"Tile {r['tile']} bounds mismatch: {obs} != {exp}")
    tiles = tiles.to_crs(9377)

    return farms_raw, farms, farm3948, p3948, tiles, area_once


# -----------------------------------------------------------------------------
# Optional context layers
# -----------------------------------------------------------------------------
def load_context(inp: Inputs, panel_b_bounds):
    xmin, ymin, xmax, ymax = panel_b_bounds
    roi = gpd.GeoDataFrame(geometry=[box(xmin, ymin, xmax, ymax)], crs=9377)

    adm1 = None
    meta = None
    if inp.adm1_path:
        adm1 = gpd.read_file(inp.adm1_path)
        if adm1.crs is None:
            raise SystemExit("ADM1 has no CRS.")
        name_col = find_name_col(adm1, ["shapeName", "name", "NAME_1", "admin1Name"])
        mask = adm1[name_col].astype(str).map(norm_text).str.contains("meta", na=False)
        meta = adm1[mask].copy()
        if len(meta) != 1:
            print(f"WARNING [ADM1] expected one Meta feature; found {len(meta)}")
        adm1 = adm1.to_crs(9377)
        if len(meta):
            meta = meta.to_crs(9377)

    adm2 = None
    if inp.adm2_path:
        adm2 = gpd.read_file(inp.adm2_path)
        if adm2.crs is None:
            raise SystemExit("ADM2 has no CRS.")
        adm2 = adm2.to_crs(9377)
        adm2 = gpd.clip(adm2, roi)

    rivers = None
    if inp.rivers_path:
        rivers = gpd.read_file(inp.rivers_path)
        if rivers.crs is None:
            raise SystemExit("River layer has no CRS.")
        rivers = rivers.to_crs(9377)

        # Natural Earth may contain geometries with non-finite coordinates.
        from shapely import get_coordinates

        def _finite_geometry(geom):
            if geom is None or geom.is_empty:
                return False
            try:
                coords = get_coordinates(geom, include_z=False)
                return coords.size > 0 and bool(np.isfinite(coords).all())
            except Exception:
                return False

        n_rivers_before = len(rivers)
        rivers = rivers[rivers.geometry.map(_finite_geometry)].copy()
        n_nonfinite_removed = n_rivers_before - len(rivers)
        print(
            "River geometry QA:",
            f"{n_rivers_before} input -> {len(rivers)} finite geometries",
            f"(removed {n_nonfinite_removed})",
        )

        if len(rivers):
            rivers = gpd.clip(rivers, roi)

    countries = None
    if inp.countries_path:
        countries = gpd.read_file(inp.countries_path)
        if countries.crs is None:
            raise SystemExit("Countries layer has no CRS.")
        countries = countries.to_crs(4326)

    return adm1, meta, adm2, rivers, countries


# -----------------------------------------------------------------------------
# Relief
# -----------------------------------------------------------------------------
def build_relief(dem_paths: list[Path], panel_b_bounds):
    if not dem_paths:
        print("Relief: FALLBACK WITHOUT RELIEF")
        return None

    print(f"Relief: using {len(dem_paths)} Copernicus DEM files")
    b4326 = transform_bounds(9377, 4326, *panel_b_bounds, densify_pts=21)
    srcs = [rasterio.open(p) for p in dem_paths]
    try:
        mosaic, src_transform = merge(srcs, bounds=b4326)
        elev = mosaic[0].astype("float32")
        nodata_vals = [s.nodata for s in srcs if s.nodata is not None]
        if nodata_vals:
            for nd in set(nodata_vals):
                elev[elev == nd] = np.nan
        elev[~np.isfinite(elev)] = np.nan

        src_bounds = array_bounds(elev.shape[0], elev.shape[1], src_transform)
        dst_transform, dst_w, dst_h = calculate_default_transform(
            srcs[0].crs, "EPSG:9377", elev.shape[1], elev.shape[0], *src_bounds,
            resolution=180.0,
        )
        dst = np.full((dst_h, dst_w), np.nan, dtype="float32")
        reproject(
            source=elev,
            destination=dst,
            src_transform=src_transform,
            src_crs=srcs[0].crs,
            dst_transform=dst_transform,
            dst_crs="EPSG:9377",
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        dst_bounds = array_bounds(dst_h, dst_w, dst_transform)
        valid = np.isfinite(dst)
        if not valid.any():
            print("WARNING [relief] reprojected DEM has no finite values; fallback without relief")
            return None
        fill = dst.copy()
        fill[~valid] = float(np.nanmedian(dst))
        cmap = LinearSegmentedColormap.from_list(
            "piedmont_hypsometry",
            ["#d8dfd0", "#d7d1bd", "#c7b18d", "#b79a75"],
        )
        ls = LightSource(azdeg=315, altdeg=45)
        rgb = ls.shade(fill, cmap=cmap, vert_exag=2.5, dx=180, dy=180, blend_mode="soft")
        rgb[..., 3] = np.where(valid, 0.70, 0.0)
        return rgb, dst_bounds
    finally:
        for s in srcs:
            s.close()


# -----------------------------------------------------------------------------
# Cartographic helpers
# -----------------------------------------------------------------------------
def add_metric_scale_bar(ax, xlim, ylim, segments_km, corner="lower right", zorder=20):
    total_m = float(max(segments_km)) * 1000.0
    x0, x1 = xlim
    y0, y1 = ylim
    dx = x1 - x0
    dy = y1 - y0
    bar_h = dy * 0.018
    pad_x = dx * 0.035
    pad_y = dy * 0.045

    if corner == "lower right":
        left = x1 - pad_x - total_m
        bottom = y0 + pad_y
    elif corner == "lower left":
        left = x0 + pad_x
        bottom = y0 + pad_y
    elif corner == "upper right":
        left = x1 - pad_x - total_m
        bottom = y1 - pad_y - bar_h * 3.5
    else:
        left = x0 + pad_x
        bottom = y1 - pad_y - bar_h * 3.5

    bg = box(left - total_m * 0.04, bottom - bar_h * 2.3,
             left + total_m * 1.04, bottom + bar_h * 3.8)
    rect_bg = Rectangle((bg.bounds[0], bg.bounds[1]), bg.bounds[2]-bg.bounds[0], bg.bounds[3]-bg.bounds[1],
                        facecolor="white", edgecolor="none", alpha=0.85, zorder=zorder)
    ax.add_patch(rect_bg)

    vals = list(map(float, segments_km))
    for i in range(len(vals)-1):
        xa = left + vals[i]*1000.0
        xb = left + vals[i+1]*1000.0
        ax.add_patch(Rectangle((xa, bottom), xb-xa, bar_h,
                               facecolor="black" if i % 2 == 0 else "white",
                               edgecolor="black", linewidth=0.5, zorder=zorder+1))
    for v in vals:
        ax.text(left + v*1000.0, bottom - bar_h*0.65, f"{v:g}", fontsize=TICK_PT,
                ha="center", va="top", zorder=zorder+2, color=COL["text"])
    ax.text(left + total_m/2, bottom + bar_h*1.15, "km", fontsize=TICK_PT,
            ha="center", va="bottom", zorder=zorder+2, color=COL["text"])
    return bg


def add_geo_scale_bar(ax, lonlat_bounds, km=500):
    xmin, ymin, xmax, ymax = lonlat_bounds
    lon0 = xmin + 0.08*(xmax-xmin)
    lat0 = ymin + 0.09*(ymax-ymin)
    geod = Geod(ellps="WGS84")
    lon1, lat1, _ = geod.fwd(lon0, lat0, 90.0, km*1000.0)
    h = 0.012*(ymax-ymin)
    ax.add_patch(Rectangle((lon0-0.2, lat0-h*3.0), (lon1-lon0)+0.4, h*6.0,
                           facecolor="white", edgecolor="none", alpha=0.85, zorder=20))
    mid = (lon0 + lon1)/2
    ax.add_patch(Rectangle((lon0, lat0), mid-lon0, h, facecolor="black", edgecolor="black", lw=0.5, zorder=21))
    ax.add_patch(Rectangle((mid, lat0), lon1-mid, h, facecolor="white", edgecolor="black", lw=0.5, zorder=21))
    ax.text(lon0, lat0-h*0.6, "0", fontsize=TICK_PT, ha="center", va="top", zorder=22)
    ax.text(lon1, lat0-h*0.6, str(km), fontsize=TICK_PT, ha="center", va="top", zorder=22)
    ax.text(mid, lat0+h*1.25, "km", fontsize=TICK_PT, ha="center", va="bottom", zorder=22)


def place_north_arrow(ax, xlim, ylim, forbidden):
    """Place north arrow, preferring upper-right and avoiding map/text conflicts."""
    x0, x1 = xlim
    y0, y1 = ylim
    dx, dy = x1-x0, y1-y0
    w, h = dx*0.050, dy*0.105

    # Strict upper-right preference, then progressively inward fallbacks.
    candidates = []
    for xin in (0.035, 0.07, 0.11, 0.15, 0.20):
        for yin in (0.035, 0.07, 0.11, 0.15):
            candidates.append((x1 - dx*xin - w, y1 - dy*yin - h))
    candidates.extend([
        (x1 - dx*0.07 - w, y0 + dy*0.08),
        (x0 + dx*0.05, y1 - dy*0.07 - h),
    ])

    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()

    text_boxes = []
    for txt in ax.texts:
        if not txt.get_visible() or not txt.get_text().strip():
            continue
        try:
            text_boxes.append(txt.get_window_extent(renderer=renderer).expanded(1.08, 1.12))
        except Exception:
            pass

    chosen = None
    for left, bottom in candidates:
        b = box(left, bottom, left+w, bottom+h)
        if any(b.intersects(g) for g in forbidden if g is not None and not g.is_empty):
            continue

        corners = np.array([
            [left, bottom],
            [left+w, bottom],
            [left+w, bottom+h],
            [left, bottom+h],
        ])
        disp = ax.transData.transform(corners)
        cx0, cy0 = disp[:, 0].min(), disp[:, 1].min()
        cx1, cy1 = disp[:, 0].max(), disp[:, 1].max()

        if any(
            cx1 > bb.x0 and cx0 < bb.x1 and cy1 > bb.y0 and cy0 < bb.y1
            for bb in text_boxes
        ):
            continue

        chosen = (left, bottom, b)
        break

    if chosen is None:
        raise SystemExit("Cannot place north arrow without touching mapped features or labels.")

    left, bottom, b = chosen
    print(
        "North arrow placement:",
        f"x={left:.1f}, y={bottom:.1f}",
        "(upper-right preferred; collision-free)",
    )
    ax.add_patch(Rectangle(
        (left, bottom), w, h,
        facecolor="white", edgecolor="none", alpha=0.88, zorder=20
    ))
    cx = left + w/2
    ax.annotate(
        "", xy=(cx, bottom+h*0.70), xytext=(cx, bottom+h*0.24),
        arrowprops=dict(arrowstyle="-|>", lw=0.7, color="black"),
        zorder=22,
    )
    ax.text(
        cx, bottom+h*0.84, "N",
        fontsize=AXIS_PT, weight="bold",
        ha="center", va="center", zorder=22,
    )
    return b


def choose_scale_bar_corner(xlim, ylim, segments, forbidden):
    total_m = max(segments) * 1000.0
    x0, x1 = xlim
    y0, y1 = ylim
    dx, dy = x1-x0, y1-y0
    bar_h = dy*0.018
    pad_x, pad_y = dx*0.035, dy*0.045
    corners = ["lower right", "lower left", "upper right", "upper left"]
    for corner in corners:
        if corner == "lower right":
            left, bottom = x1-pad_x-total_m, y0+pad_y
        elif corner == "lower left":
            left, bottom = x0+pad_x, y0+pad_y
        elif corner == "upper right":
            left, bottom = x1-pad_x-total_m, y1-pad_y-bar_h*3.5
        else:
            left, bottom = x0+pad_x, y1-pad_y-bar_h*3.5
        b = box(left-total_m*0.04, bottom-bar_h*2.3, left+total_m*1.04, bottom+bar_h*3.8)
        if not any(b.intersects(g) for g in forbidden if g is not None and not g.is_empty):
            return corner
    raise SystemExit("Cannot place scale bar without touching farms, tile boundaries, or rivers.")


def add_bottom_left_graticule(ax, panel_bounds, step=0.2):
    """Draw bottom longitude labels and left latitude labels outside panel B."""
    xmin, ymin, xmax, ymax = panel_bounds
    to_ll = Transformer.from_crs(9377, 4326, always_xy=True)
    to_xy = Transformer.from_crs(4326, 9377, always_xy=True)
    lon_min, lat_min = to_ll.transform(xmin, ymin)
    lon_max, lat_max = to_ll.transform(xmax, ymax)
    lo0, lo1 = sorted([lon_min, lon_max])
    la0, la1 = sorted([lat_min, lat_max])

    lons = np.arange(math.ceil(lo0/step)*step, lo1+1e-9, step)
    lats = np.arange(math.ceil(la0/step)*step, la1+1e-9, step)
    texts = []
    lat_texts = []

    # Convert requested physical offsets into data units.
    tick_len_y = (ymax-ymin) * (1.0 / MAP_H_MM)      # 1 mm inward
    tick_len_x = (xmax-xmin) * (1.0 / PANEL_B_W_MM) # 1 mm inward
    lat_gap_x = (xmax-xmin) * (1.0 / PANEL_B_W_MM)  # 1 mm outside

    for lon in lons:
        x, _ = to_xy.transform(lon, (la0+la1)/2)
        if xmin <= x <= xmax:
            ax.plot(
                [x, x], [ymin, ymin+tick_len_y],
                color=COL["frame"], lw=0.4, zorder=30,
            )
            t = ax.text(
                x, ymin-tick_len_y*0.70,
                f"{abs(lon):.1f}°W",
                fontsize=TICK_PT,
                ha="center", va="top",
                clip_on=False, color=COL["text"], zorder=31,
            )
            t.set_gid("longitude_label")
            texts.append(t)

    for lat in lats:
        _, y = to_xy.transform((lo0+lo1)/2, lat)
        if ymin <= y <= ymax:
            ax.plot(
                [xmin, xmin+tick_len_x], [y, y],
                color=COL["frame"], lw=0.4, zorder=30,
            )
            t = ax.text(
                xmin-lat_gap_x, y,
                f"{abs(lat):.1f}°N",
                fontsize=TICK_PT,
                ha="right", va="center",
                clip_on=False, color=COL["text"], zorder=31,
            )
            t.set_gid("latitude_label")
            texts.append(t)
            lat_texts.append(t)

    return texts, lat_texts


def smart_place_name(value: str) -> str:
    """Sentence-style municipality names while preserving leading articles."""
    words = str(value).strip().split()
    if not words:
        return ""
    minor = {"de", "del", "la", "las", "los", "y"}
    out = []
    for i, w in enumerate(words):
        wl = w.lower()
        if i > 0 and wl in minor:
            out.append(wl)
        else:
            out.append(wl.capitalize())
    return " ".join(out)


def label_river(ax, geom, name, color):
    try:
        if geom.geom_type == "MultiLineString":
            line = max(geom.geoms, key=lambda g: g.length)
        else:
            line = geom
        if line.length <= 0:
            return None
        d = line.length * 0.52
        p = line.interpolate(d)
        p0 = line.interpolate(max(0, d-line.length*0.03))
        p1 = line.interpolate(min(line.length, d+line.length*0.03))
        angle = math.degrees(math.atan2(p1.y-p0.y, p1.x-p0.x))
        if angle > 90:
            angle -= 180
        if angle < -90:
            angle += 180
        return ax.text(p.x, p.y, name, fontsize=6.0, style="italic", color=color,
                       rotation=angle, rotation_mode="anchor", ha="center", va="center",
                       path_effects=[pe.withStroke(linewidth=1.5, foreground="white")], zorder=13)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Figure drawing
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# R3 FINAL helpers
# -----------------------------------------------------------------------------
def _largest_polygon(geom):
    if geom is None or geom.is_empty:
        return geom
    if geom.geom_type == "Polygon":
        return geom
    if geom.geom_type == "MultiPolygon":
        return max(geom.geoms, key=lambda g: g.area)
    polys = [g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon"]
    return max(polys, key=lambda g: g.area) if polys else geom


def _pad_bounds_fraction(bounds, frac=0.04):
    x0, y0, x1, y1 = map(float, bounds)
    dx, dy = x1-x0, y1-y0
    return (x0-dx*frac, y0-dy*frac, x1+dx*frac, y1+dy*frac)


def _display_bbox_to_data_polygon(ax, bb, pad_px=0.0):
    from shapely.geometry import Polygon
    pts = np.array([
        [bb.x0-pad_px, bb.y0-pad_px],
        [bb.x1+pad_px, bb.y0-pad_px],
        [bb.x1+pad_px, bb.y1+pad_px],
        [bb.x0-pad_px, bb.y1+pad_px],
    ])
    dat = ax.transData.inverted().transform(pts)
    return Polygon(dat)


def _bbox_hits_samples(bb, samples, pad_px=0.0):
    if samples is None or len(samples) == 0:
        return False
    return bool(np.any(
        (samples[:, 0] >= bb.x0-pad_px)
        & (samples[:, 0] <= bb.x1+pad_px)
        & (samples[:, 1] >= bb.y0-pad_px)
        & (samples[:, 1] <= bb.y1+pad_px)
    ))


def _line_samples_display(fig, ax0, xy0, tr0, ax1, xy1, tr1, n=300):
    p0 = tr0.transform(xy0)
    p1 = tr1.transform(xy1)
    ts = np.linspace(0.0, 1.0, n)
    return np.array([p0*(1.0-t)+p1*t for t in ts])


def _plot_two_segment_metric_scale(ax, xlim, ylim, total_km, corner="lower left", zorder=20):
    """Two-segment metric scale bar with endpoint labels only; returns background box."""
    total_m = float(total_km) * 1000.0
    x0, x1 = xlim; y0, y1 = ylim
    dx, dy = x1-x0, y1-y0
    bar_h = dy*0.018
    pad_x, pad_y = dx*0.045, dy*0.045
    if corner == "lower left":
        left, bottom = x0+pad_x, y0+pad_y
    elif corner == "lower right":
        left, bottom = x1-pad_x-total_m, y0+pad_y
    elif corner == "upper left":
        left, bottom = x0+pad_x, y1-pad_y-bar_h*3.5
    else:
        left, bottom = x1-pad_x-total_m, y1-pad_y-bar_h*3.5
    bg = box(left-total_m*0.05, bottom-bar_h*2.3,
             left+total_m*1.05, bottom+bar_h*3.8)
    ax.add_patch(Rectangle((bg.bounds[0], bg.bounds[1]), bg.bounds[2]-bg.bounds[0], bg.bounds[3]-bg.bounds[1],
                           facecolor="white", edgecolor="none", alpha=0.85, zorder=zorder))
    half=total_m/2.0
    ax.add_patch(Rectangle((left,bottom),half,bar_h,facecolor="black",edgecolor="black",linewidth=0.5,zorder=zorder+1))
    ax.add_patch(Rectangle((left+half,bottom),half,bar_h,facecolor="white",edgecolor="black",linewidth=0.5,zorder=zorder+1))
    ax.text(left, bottom-bar_h*0.65, "0", fontsize=TICK_PT, ha="center", va="top", zorder=zorder+2)
    ax.text(left+total_m, bottom-bar_h*0.65, f"{total_km:g}", fontsize=TICK_PT, ha="center", va="top", zorder=zorder+2)
    ax.text(left+half, bottom+bar_h*1.15, "km", fontsize=TICK_PT, ha="center", va="bottom", zorder=zorder+2)
    return bg



def _add_panel_a_scale_bar_r4(ax, xlim, ylim, panel_w_mm=42.0, panel_h_mm=50.0, total_km=500.0, zorder=20):
    """R4 panel-A scale: 3 mm from left, bar 4 mm above bottom, labels below on one line."""
    x0,x1=xlim; y0,y1=ylim
    ux=(x1-x0)/panel_w_mm; uy=(y1-y0)/panel_h_mm
    total_m=float(total_km)*1000.0
    left=x0+3.0*ux
    bottom=y0+4.0*uy
    bar_h=0.9*uy
    # Keep labels immediately below the bar while preserving the
    # mandatory >=1.5-mm clearance from the panel frame.
    label_y=bottom-0.10*uy
    half=total_m/2.0

    # White 85% background behind bar + labels.  Its lower edge is allowed
    # closer to the frame; the 1.5-mm requirement applies to the labels.
    bg_left=left-1.0*ux
    bg_right=left+total_m+1.0*ux
    bg_bottom=y0+0.50*uy
    bg_top=bottom+bar_h+1.0*uy
    bg=box(bg_left,bg_bottom,bg_right,bg_top)
    ax.add_patch(Rectangle((bg_left,bg_bottom),bg_right-bg_left,bg_top-bg_bottom,
                           facecolor="white",edgecolor="none",alpha=0.85,zorder=zorder))
    ax.add_patch(Rectangle((left,bottom),half,bar_h,facecolor="black",edgecolor="black",linewidth=0.5,zorder=zorder+1))
    ax.add_patch(Rectangle((left+half,bottom),half,bar_h,facecolor="white",edgecolor="black",linewidth=0.5,zorder=zorder+1))
    t0=ax.text(left,label_y,"0",fontsize=6.0,ha="left",va="top",zorder=zorder+2,color=COL["text"])
    t1=ax.text(left+total_m,label_y,f"{total_km:g} km",fontsize=6.0,ha="right",va="top",zorder=zorder+2,color=COL["text"])
    return bg,[t0,t1]


def _add_panel_c_scale_bar(ax, xlim, ylim, outline, segments_km=(0,0.5,1), zorder=20):
    """Place the frozen 0-0.5-1 km scale in the lower-left sector without touching the farm."""
    total_m=float(max(segments_km))*1000.0
    x0,x1=xlim; y0,y1=ylim; dx=x1-x0; dy=y1-y0
    bar_h=dy*0.018
    candidates=[]
    for yf in (0.060,0.075,0.095,0.12,0.15,0.18,0.22):
        for xf in (0.035,0.055,0.075,0.10):
            left=x0+dx*xf; bottom=y0+dy*yf
            bg=box(left-total_m*0.04,bottom-bar_h*2.3,left+total_m*1.04,bottom+bar_h*3.8)
            if bg.bounds[0] < x0 or bg.bounds[2] > x1 or bg.bounds[1] < y0 or bg.bounds[3] > y1:
                continue
            if bg.intersects(outline):
                continue
            candidates.append((yf,xf,left,bottom,bg))
    if not candidates:
        return None
    _,_,left,bottom,bg=sorted(candidates,key=lambda z:(z[0],z[1]))[0]
    ax.add_patch(Rectangle((bg.bounds[0],bg.bounds[1]),bg.bounds[2]-bg.bounds[0],bg.bounds[3]-bg.bounds[1],
                           facecolor="white",edgecolor="none",alpha=0.85,zorder=zorder))
    vals=list(segments_km)
    for i in range(len(vals)-1):
        xa=left+vals[i]*1000.0; xb=left+vals[i+1]*1000.0
        ax.add_patch(Rectangle((xa,bottom),xb-xa,bar_h,facecolor="black" if i%2==0 else "white",
                               edgecolor="black",linewidth=0.5,zorder=zorder+1))
    for v in vals:
        ax.text(left+v*1000.0,bottom-bar_h*0.65,f"{v:g}",fontsize=TICK_PT,ha="center",va="top",zorder=zorder+2,color=COL["text"])
    ax.text(left+total_m/2.0,bottom+bar_h*1.15,"km",fontsize=TICK_PT,ha="center",va="bottom",zorder=zorder+2,color=COL["text"])
    return bg


def _river_segments(geom):
    lines=[]
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type=="LineString": lines=[geom]
    elif geom.geom_type=="MultiLineString": lines=list(geom.geoms)
    else: lines=[g for g in getattr(geom,"geoms",[]) if g.geom_type=="LineString"]
    segs=[]
    for line in lines:
        coords=list(line.coords)
        for a,b in zip(coords[:-1],coords[1:]):
            L=math.hypot(b[0]-a[0],b[1]-a[1])
            if L>0: segs.append((L,a,b))
    return sorted(segs,key=lambda z:z[0],reverse=True)


def _oriented_text_polygon_data(ax, fig, text_artist, angle_deg, pad_px=0.0):
    """Approximate the actual rotated text footprint, not its axis-aligned display bbox."""
    original_rotation=text_artist.get_rotation()
    text_artist.set_rotation(0)
    fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    bb0=text_artist.get_window_extent(renderer=renderer)
    text_artist.set_rotation(original_rotation)
    fig.canvas.draw()
    center=ax.transData.transform(text_artist.get_position())
    hw=bb0.width/2.0+pad_px; hh=bb0.height/2.0+pad_px
    corners=np.array([[-hw,-hh],[hw,-hh],[hw,hh],[-hw,hh]],dtype=float)
    th=math.radians(angle_deg); R=np.array([[math.cos(th),-math.sin(th)],[math.sin(th),math.cos(th)]])
    disp=corners@R.T+center
    dat=ax.transData.inverted().transform(disp)
    from shapely.geometry import Polygon
    return Polygon(dat)


def _add_graticule_r3(ax, panel_bounds, panel_w_mm, panel_h_mm, step=0.2):
    """Longitude/latitude labels outside B frame, preserving R2 style."""
    xmin,ymin,xmax,ymax=panel_bounds
    to_ll=Transformer.from_crs(9377,4326,always_xy=True)
    to_xy=Transformer.from_crs(4326,9377,always_xy=True)
    lon_min,lat_min=to_ll.transform(xmin,ymin); lon_max,lat_max=to_ll.transform(xmax,ymax)
    lo0,lo1=sorted([lon_min,lon_max]); la0,la1=sorted([lat_min,lat_max])
    lons=np.arange(math.ceil(lo0/step)*step,lo1+1e-9,step)
    lats=np.arange(math.ceil(la0/step)*step,la1+1e-9,step)
    tick_y=(ymax-ymin)*(1.0/panel_h_mm)
    tick_x=(xmax-xmin)*(1.0/panel_w_mm)
    lat_gap=(xmax-xmin)*(1.0/panel_w_mm)
    texts=[]; lat_texts=[]
    for lon in lons:
        x,_=to_xy.transform(lon,(la0+la1)/2)
        if xmin<=x<=xmax:
            ax.plot([x,x],[ymin,ymin+tick_y],color=COL["frame"],lw=0.4,zorder=30)
            t=ax.text(x,ymin-tick_y*0.70,f"{abs(lon):.1f}°W",fontsize=TICK_PT,ha="center",va="top",clip_on=False,color=COL["text"],zorder=31)
            t.set_gid("longitude_label"); texts.append(t)
    for lat in lats:
        _,y=to_xy.transform((lo0+lo1)/2,lat)
        if ymin<=y<=ymax:
            ax.plot([xmin,xmin+tick_x],[y,y],color=COL["frame"],lw=0.4,zorder=30)
            t=ax.text(xmin-lat_gap,y,f"{abs(lat):.1f}°N",fontsize=TICK_PT,ha="right",va="center",clip_on=False,color=COL["text"],zorder=31)
            t.set_gid("latitude_label"); texts.append(t); lat_texts.append(t)
    return texts,lat_texts


def _separate_dot_symbols(ax, data_points, min_mm, px_per_mm):
    """Displace only rendered symbol centers; underlying farm geometries remain unchanged."""
    if len(data_points) < 2:
        return list(data_points), 0, float("inf")
    disp=np.array([ax.transData.transform((p.x,p.y)) for p in data_points],dtype=float)
    threshold=min_mm*px_per_mm
    pairs=set()
    for _ in range(80):
        changed=False
        for i in range(len(disp)):
            for j in range(i+1,len(disp)):
                v=disp[j]-disp[i]; d=float(np.hypot(v[0],v[1]))
                if d+1e-9 >= threshold:
                    continue
                pairs.add((i,j)); changed=True
                if d < 1e-6:
                    ang=(i*37+j*53)%360
                    u=np.array([math.cos(math.radians(ang)),math.sin(math.radians(ang))])
                else:
                    u=v/d
                shift=(threshold-d)/2.0+0.05
                disp[i]-=u*shift; disp[j]+=u*shift
        if not changed:
            break
    mins=[]
    for i in range(len(disp)):
        for j in range(i+1,len(disp)):
            mins.append(float(np.linalg.norm(disp[j]-disp[i])))
    outdat=ax.transData.inverted().transform(disp)
    out=[Point(float(x),float(y)) for x,y in outdat]
    min_dist_mm=(min(mins)/px_per_mm) if mins else float("inf")
    return out,len(pairs),min_dist_mm


def _river_longest_segment(geom):
    lines=[]
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type=="LineString": lines=[geom]
    elif geom.geom_type=="MultiLineString": lines=list(geom.geoms)
    else:
        lines=[g for g in getattr(geom,"geoms",[]) if g.geom_type=="LineString"]
    best=None
    for line in lines:
        coords=list(line.coords)
        for a,b in zip(coords[:-1],coords[1:]):
            dx=b[0]-a[0]; dy=b[1]-a[1]; L=math.hypot(dx,dy)
            if L<=0: continue
            if best is None or L>best[0]: best=(L,a,b)
    return best


def _place_river_label_r3(ax, fig, river_geom, raw_name, tile_boundary, farm_geoms, px_per_mm):
    """Place source river name parallel to a straight segment, offset exactly 1.2 mm."""
    segs=_river_segments(river_geom)
    if not segs:
        return None,None
    farm_centers=np.array([ax.transData.transform((g.representative_point().x,g.representative_point().y))
                           for g in farm_geoms if g is not None and not g.is_empty])
    display_name=str(raw_name)
    # Try the longest/straightest available source segments first.
    for _,a,b in segs[:120]:
        p0=ax.transData.transform(a); p1=ax.transData.transform(b)
        v=p1-p0; L=float(np.linalg.norm(v))
        if L<=0: continue
        u=v/L; normal=np.array([-u[1],u[0]])
        mid=(p0+p1)/2.0
        angle=math.degrees(math.atan2(v[1],v[0]))
        if angle>90: angle-=180
        if angle<-90: angle+=180
        signs=[]
        for sign in (1.0,-1.0):
            pos=mid+normal*(1.2*px_per_mm*sign)
            score=float(np.min(np.linalg.norm(farm_centers-pos,axis=1))) if len(farm_centers) else 1e9
            signs.append((score,sign,pos))
        signs.sort(key=lambda z:z[0],reverse=True)
        for _,sign,pos in signs:
            dat=ax.transData.inverted().transform(pos)
            t=ax.text(float(dat[0]),float(dat[1]),display_name,fontsize=6.0,style="italic",color=COL["river"],
                      rotation=angle,rotation_mode="anchor",ha="center",va="center",
                      path_effects=[pe.withStroke(linewidth=1.5,foreground="white")],zorder=13)
            poly=_oriented_text_polygon_data(ax,fig,t,angle,pad_px=0.15*px_per_mm)
            if poly.intersects(river_geom) or poly.intersects(tile_boundary):
                t.remove(); continue
            return t,poly
    return None,None


def _candidate_offsets_mm():
    vals=[(0,0)]
    for d in (3,6):
        vals.extend([(d,0),(-d,0),(0,d),(0,-d),(d,d),(d,-d),(-d,d),(-d,-d)])
    return vals


def draw_figure(root: Path, preview_path: Path, final: bool):
    fonts={f.name for f in fm.fontManager.ttflist}
    if FONT_FAMILY not in fonts:
        raise SystemExit(f"Required font not installed: {FONT_FAMILY}")

    fig2_cols=import_figure2_colors(root)
    print("Figure 2 pipeline colours imported:",fig2_cols)
    inp=resolve_inputs(root)
    farms_raw,farms,farm3948,p3948,tiles,area_once=load_project_data(inp)

    # Frozen R2 panel-B extent: compute with the R2 80 x 81 mm aspect, regardless of R3 layout.
    all_farms_geom=unary_union(list(farms.geometry))
    b0=expand_bounds(all_farms_geom.bounds,pad_frac=0.08,min_pad=8000.0)
    braw=(b0[0]-9000.0,b0[1],b0[2],b0[3])
    panel_b_bounds=fit_bounds_to_panel_aspect(braw,80.0,81.0)
    xmin_b,ymin_b,xmax_b,ymax_b=panel_b_bounds
    print("Regional extent west extension: 9.0 km (existing DEM coverage only)")

    adm1,meta,adm2,rivers,countries=load_context(inp,panel_b_bounds)
    relief=build_relief(inp.dem_paths,panel_b_bounds)

    # R3-2: farm outline from source farm features; morphological closing fuses <2 m slivers.
    farm3948_features=farms_raw[farms_raw["_farm"]=="3948"].copy().to_crs(9377)
    outline=farm3948_features.geometry.buffer(1.0).union_all().buffer(-1.0)
    outline_components = list(outline.geoms) if outline.geom_type == "MultiPolygon" else [outline]
    outline_component_hits = [int(p3948.geometry.intersects(g).sum()) for g in outline_components]
    # Source-faithful rule: discard a component only when it is demonstrably spurious.
    # Here a component is substantive when at least one management polygon intersects it.
    substantive = [g for g,h in zip(outline_components, outline_component_hits) if h > 0]
    if outline.geom_type == "MultiPolygon" and len(substantive) == 1:
        outline = substantive[0]
        outline_components = [outline]
        outline_component_hits = [int(p3948.geometry.intersects(outline).sum())]
    outline_area_ha=float(outline.area/10000.0)
    outline_component_count=len(outline_components)

    # Panel C: retain 6% side/top margin and reserve a lower-left strip for the 1-km scale bar.
    fx0,fy0,fx1,fy1=map(float,outline.bounds); fdx, fdy=fx1-fx0, fy1-fy0
    c_bounds=(fx0-fdx*0.06,fy0-fdy*0.18,fx1+fdx*0.06,fy1+fdy*0.06)
    xmin_c,ymin_c,xmax_c,ymax_c=c_bounds

    # R4 symmetric layout in millimetres.
    # Side maps are identical 42 x 50 mm boxes, top-aligned with panel B.
    A_X0,A_X1=2.0,44.0
    B_X0,B_X1=52.0,120.0
    C_X0,C_X1=126.0,168.0
    A_W=A_X1-A_X0; B_W=B_X1-B_X0; C_W=C_X1-C_X0
    SIDE_H=50.0
    PANEL_TOP=74.0
    b_ratio=(xmax_b-xmin_b)/(ymax_b-ymin_b)
    b_h=B_W/b_ratio
    b_y=PANEL_TOP-b_h
    a_x=A_X0; a_y=PANEL_TOP-SIDE_H
    c_x=C_X0; c_y=PANEL_TOP-SIDE_H

    # Text blocks below side maps, both ending at panel-B bottom.
    attr_y0=b_y
    attr_y1=a_y-3.0
    legend_y0=b_y
    legend_y1=c_y-3.0
    if attr_y1<=attr_y0 or legend_y1<=legend_y0:
        raise SystemExit("R4 layout failure: insufficient text-block space below side panels.")

    # R4 panel A: mainland Colombia in EPSG:9377, centered in a fixed 42 x 50 mm box.
    if countries is None:
        raise SystemExit("R4 requires the existing Natural Earth countries layer.")
    name_col=find_name_col(countries,["ADMIN","NAME","NAME_EN","SOVEREIGNT"])
    cmask=countries[name_col].astype(str).map(norm_text).str.contains("colombia",na=False)
    col4326=countries[cmask].copy()
    if len(col4326)!=1:
        raise SystemExit(f"Could not uniquely identify Colombia; found {len(col4326)}")
    col9377=col4326.to_crs(9377)
    mainland=_largest_polygon(col9377.geometry.iloc[0])
    a_bounds=_pad_bounds_fraction(mainland.bounds,0.04)
    ax0,ay0,ax1,ay1=a_bounds
    a_ratio=(ax1-ax0)/(ay1-ay0)

    # Mandatory console baseline.
    geometry_area_ha=float(farms.geometry.area.sum()/10000.0)
    ut_present=sorted(set(p3948["_ut"]),key=lambda x:int(x))
    print("\nCanonical map controls:")
    print(f"  Farms: {len(farms)} (features: {len(farms_raw)})")
    print(f"  Total farm area (AREA_HA): {area_once:.2f} ha")
    print(f"  Total farm area (geometry, EPSG:9377): {geometry_area_ha:.3f} ha")
    print(f"  Farm 3948 features: {len(farm3948_features)} ; outline components: {outline_component_count} ; outline area: {outline_area_ha:.3f} ha ; UT present: {','.join(ut_present)}")
    if outline_component_count > 1:
        print("  Farm 3948 component QA : " + "; ".join(
            f"{i+1}:{g.area/10000.0:.3f} ha/{outline_component_hits[i]} management polygons"
            for i,g in enumerate(outline_components)
        ))
    print(f"  Management polygons (farm 3948): {len(p3948)}")
    print("  Farm 3948 publication authorization: PENDING SCR CONFIRMATION")
    deuteranopia_palette_check()

    fig=plt.figure(figsize=(FIG_W_MM/MM_PER_IN,FIG_H_MM/MM_PER_IN),dpi=160,facecolor="white")
    ax_a=fig.add_axes(axes_rect_mm(a_x,a_y,A_W,SIDE_H),label="panel_a")
    ax_b=fig.add_axes(axes_rect_mm(B_X0,b_y,B_W,b_h),label="panel_b")
    ax_c=fig.add_axes(axes_rect_mm(c_x,c_y,C_W,SIDE_H),label="panel_c")
    ax_attr=fig.add_axes(axes_rect_mm(A_X0,attr_y0,A_W,attr_y1-attr_y0),label="attribution_column")
    ax_leg=fig.add_axes(axes_rect_mm(C_X0,legend_y0,C_W,legend_y1-legend_y0),label="legend_column")
    ax_attr.axis("off"); ax_leg.axis("off")
    for ax in (ax_a,ax_b,ax_c): set_frame(ax)
    for sp in ax_c.spines.values(): sp.set_color(COL["accent"]); sp.set_linewidth(0.8)

    # Headings aligned above actual panels.
    heading_y=mm_to_fig_y(PANEL_TOP+2.0)
    for x_mm,lab,title in [(A_X0,"(a)","Colombia"),(B_X0,"(b)","Regional context"),(C_X0,"(c)","Farm 3948")]:
        fig.text(mm_to_fig_x(x_mm),heading_y,lab,fontsize=PANEL_LABEL_PT,weight="bold",ha="left",va="bottom",color=COL["text"])
        fig.text(mm_to_fig_x(x_mm+6.5),heading_y,title,fontsize=PANEL_TITLE_PT,weight="bold",ha="left",va="bottom",color=COL["text"])

    manual_texts=[]; lat_texts=[]; connectors=[]; problems=[]

    # ---------------- Panel A: metric/equal-aspect ----------------
    # R4-2: sea is the AXES BACKGROUND, never a separate polygon.
    ax_a.set_facecolor("#EAF2F8")
    countries9377=countries.to_crs(9377)
    try:
        countries9377.cx[ax0:ax1,ay0:ay1].plot(ax=ax_a,facecolor="white",edgecolor="#BDBDBD",linewidth=0.35,zorder=1)
    except Exception:
        pass
    gpd.GeoSeries([mainland],crs=9377).plot(ax=ax_a,facecolor=COL["country"],edgecolor=COL["frame"],linewidth=0.4,zorder=2)
    meta_geom=None
    if meta is not None and len(meta):
        meta_geom=_largest_polygon(unary_union(list(meta.geometry)))
        gpd.GeoSeries([meta_geom],crs=9377).plot(ax=ax_a,facecolor=COL["meta"],edgecolor=COL["frame"],linewidth=0.4,zorder=3)
    reg_rect_geom=box(xmin_b,ymin_b,xmax_b,ymax_b)
    ax_a.add_patch(Rectangle((xmin_b,ymin_b),xmax_b-xmin_b,ymax_b-ymin_b,fill=False,edgecolor=COL["accent"],linewidth=0.8,zorder=11))
    ax_a.set_xlim(ax0,ax1); ax_a.set_ylim(ay0,ay1)
    ax_a.set_aspect("equal",adjustable="datalim")
    fig.canvas.draw()
    ax0,ax1=ax_a.get_xlim(); ay0,ay1=ax_a.get_ylim()
    a_bounds=(ax0,ay0,ax1,ay1)
    scale_a_geom,scale_a_texts=_add_panel_a_scale_bar_r4(
        ax_a,(ax0,ax1),(ay0,ay1),panel_w_mm=A_W,panel_h_mm=SIDE_H,total_km=500
    )

    # Precompute the exact two connector trajectories in display space.
    fig.canvas.draw(); px_per_mm=fig.dpi/MM_PER_IN
    connector_samples_a=[]
    for source,target in [((xmax_b,ymax_b),(0.0,1.0)),((xmax_b,ymin_b),(0.0,0.0))]:
        connector_samples_a.append(_line_samples_display(fig,ax_a,source,ax_a.transData,ax_b,target,ax_b.transAxes,n=360))
    all_conn_samples=np.vstack(connector_samples_a)

    # R3-8 COLOMBIA: constrained polylabel-led placement.
    # Search in DISPLAY space because the requirement is physical:
    # the rendered text box + 1 mm must fit inside mainland Colombia
    # while avoiding Meta and both A->B connectors.
    from shapely.ops import polylabel, nearest_points
    from matplotlib.transforms import Bbox

    cp = polylabel(mainland, tolerance=500.0)

    colombia_label = ax_a.text(
        cp.x,
        cp.y,
        "C O L O M B I A",
        fontsize=7.0,
        color="#4D4D4D",
        ha="center",
        va="center",
        path_effects=[
            pe.withStroke(
                linewidth=1.6,
                foreground="white"
            )
        ],
        zorder=10,
    )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Rendered dimensions are fixed for all candidate positions.
    bb0 = colombia_label.get_window_extent(renderer=renderer)
    half_w = bb0.width / 2.0
    half_h = bb0.height / 2.0

    pad_px = 1.0 * px_per_mm
    axes_bb = ax_a.get_window_extent(renderer=renderer)

    def _candidate_is_valid(cx, cy):
        bb = Bbox.from_extents(
            cx - half_w,
            cy - half_h,
            cx + half_w,
            cy + half_h,
        )

        colpoly = _display_bbox_to_data_polygon(
            ax_a,
            bb,
            pad_px=pad_px,
        )

        # Exact R3 tests.
        if not mainland.contains(colpoly):
            return None

        if (
            meta_geom is not None
            and colpoly.intersects(meta_geom)
        ):
            return None

        if _bbox_hits_samples(
            bb,
            all_conn_samples,
            pad_px=pad_px,
        ):
            return None

        xdat, ydat = ax_a.transData.inverted().transform(
            (cx, cy)
        )

        pt = Point(
            float(xdat),
            float(ydat)
        )

        if not mainland.contains(pt):
            return None

        clearance = float(
            pt.distance(mainland.boundary)
        )

        return clearance, pt, bb

    # First test the actual polylabel point, exactly as requested.
    cp_disp = ax_a.transData.transform(
        (cp.x, cp.y)
    )

    valid = _candidate_is_valid(
        float(cp_disp[0]),
        float(cp_disp[1])
    )

    chosen_col = None

    if valid is not None:
        _, chosen_col, _ = valid
        placement_mode = "polylabel"

    else:
        # Fine physical search at 0.25-mm spacing.
        # This avoids missing a feasible position because of a coarse
        # geographic grid.
        step_px = 0.25 * px_per_mm

        xmin_px = axes_bb.x0 + half_w + pad_px
        xmax_px = axes_bb.x1 - half_w - pad_px
        ymin_px = axes_bb.y0 + half_h + pad_px
        ymax_px = axes_bb.y1 - half_h - pad_px

        feasible = []

        if xmax_px > xmin_px and ymax_px > ymin_px:
            xs = np.arange(
                xmin_px,
                xmax_px + step_px*0.25,
                step_px
            )

            ys = np.arange(
                ymin_px,
                ymax_px + step_px*0.25,
                step_px
            )

            for cy in ys:
                for cx in xs:
                    result = _candidate_is_valid(
                        float(cx),
                        float(cy)
                    )

                    if result is not None:
                        feasible.append(result)

        if feasible:
            # Constrained equivalent of polylabel:
            # among all QA-valid positions choose the point with
            # maximum distance from Colombia's boundary.
            feasible.sort(
                key=lambda z: z[0],
                reverse=True
            )

            _, chosen_col, _ = feasible[0]
            placement_mode = (
                "constrained polylabel; "
                "0.25-mm display search"
            )

    r3_8_status = "PASS"
    r3_8_outside_fraction = 0.0

    if chosen_col is None:
        # ------------------------------------------------------------------
        # R3-8 documented exception
        # ------------------------------------------------------------------
        step_px = 0.25 * px_per_mm

        xmin_px = axes_bb.x0 + half_w + pad_px
        xmax_px = axes_bb.x1 - half_w - pad_px
        ymin_px = axes_bb.y0 + half_h + pad_px
        ymax_px = axes_bb.y1 - half_h - pad_px

        fallback = []

        if xmax_px > xmin_px and ymax_px > ymin_px:
            xs = np.arange(xmin_px, xmax_px + step_px*0.25, step_px)
            ys = np.arange(ymin_px, ymax_px + step_px*0.25, step_px)

            for cy in ys:
                for cx in xs:
                    bb = Bbox.from_extents(cx-half_w, cy-half_h, cx+half_w, cy+half_h)
                    colpoly = _display_bbox_to_data_polygon(ax_a, bb, pad_px=pad_px)
                    xdat, ydat = ax_a.transData.inverted().transform((cx, cy))
                    pt = Point(float(xdat), float(ydat))
                    if not mainland.contains(pt):
                        continue
                    if meta_geom is not None and colpoly.intersects(meta_geom):
                        continue
                    if _bbox_hits_samples(bb, all_conn_samples, pad_px=pad_px):
                        continue
                    area = float(colpoly.area)
                    if area <= 0:
                        continue
                    outside = float(colpoly.difference(mainland).area)
                    outside_frac = outside / area
                    clearance = float(pt.distance(mainland.boundary))
                    fallback.append((outside_frac, -clearance, pt))

        if not fallback:
            raise SystemExit(
                "R3-8 exception failed: no position keeps the "
                "COLOMBIA anchor inside Colombia while remaining "
                "clear of Meta and A-B connectors."
            )

        fallback.sort(key=lambda z: (z[0], z[1]))
        r3_8_outside_fraction, _, chosen_col = fallback[0]
        placement_mode = "R3-8 documented geometric-infeasibility fallback"
        r3_8_status = "WAIVED_GEOMETRIC_INFEASIBILITY"

        print("  R3-8 COLOMBIA containment:", r3_8_status)
        print("  R3-8 minimum bbox outside Colombia:", f"{100.0*r3_8_outside_fraction:.2f}%")
        print("  R3-8 retained tests:", "anchor inside Colombia; Meta clear; A-B connectors clear")

    colombia_label.set_position((chosen_col.x, chosen_col.y))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    print(
        "  COLOMBIA label placement:",
        f"x={chosen_col.x:.1f},",
        f"y={chosen_col.y:.1f}",
        f"({placement_mode}; QA-clear)"
    )

    manual_texts.append(colombia_label)

    # R3-9 Meta east of department, outside Meta but inside mainland, short leader.
    meta_label=None; meta_leader=None
    if meta_geom is not None:
        mx0,my0,mx1,my1=meta_geom.bounds; mw,mh=mx1-mx0,my1-my0
        rp=meta_geom.representative_point()
        candidates=[]
        for ex in (0.06,0.10,0.14,0.18,0.24):
            for ey in (0.00,0.12,-0.12,0.24,-0.24):
                p=Point(mx1+mw*ex,rp.y+mh*ey)
                if mainland.contains(p) and not meta_geom.contains(p): candidates.append(p)
        meta_label=ax_a.text(0,0,"Meta",fontsize=6.5,ha="left",va="center",color=COL["text"],path_effects=[pe.withStroke(linewidth=1.5,foreground="white")],zorder=10)
        chosen=None
        for p in candidates:
            meta_label.set_position((p.x,p.y)); fig.canvas.draw(); renderer=fig.canvas.get_renderer(); bb=meta_label.get_window_extent(renderer=renderer)
            poly=_display_bbox_to_data_polygon(ax_a,bb,pad_px=0.45*px_per_mm)
            if poly.intersects(reg_rect_geom) or poly.intersects(meta_geom.boundary): continue
            if _bbox_hits_samples(bb,all_conn_samples,pad_px=1.0): continue
            if not mainland.contains(poly): continue
            chosen=p; break
        if chosen is None:
            raise SystemExit("R3 Meta label: no valid eastern external placement.")
        anchor=nearest_points(meta_geom.boundary,chosen)[0]
        meta_leader=ax_a.annotate("",xy=(anchor.x,anchor.y),xytext=(chosen.x,chosen.y),arrowprops=dict(arrowstyle="-",linewidth=0.3,color="#777777",shrinkA=5.0,shrinkB=1.0),zorder=9)
        manual_texts.append(meta_label)

    # ---------------- Panel B ----------------
    ax_b.set_xlim(xmin_b,xmax_b); ax_b.set_ylim(ymin_b,ymax_b); ax_b.set_aspect("equal",adjustable="box")
    if relief is not None:
        rgb,rbounds=relief
        ax_b.imshow(rgb,extent=[rbounds[0],rbounds[2],rbounds[1],rbounds[3]],origin="upper",zorder=0)
    if adm2 is not None and len(adm2):
        adm2.boundary.plot(ax=ax_b,color=COL["municipal"],linewidth=0.35,linestyle=(0,(3,2)),zorder=4)

    # R4-1: river layer removed. Natural Earth misidentifies the local trace as Guainía at this scale.
    river_union=None
    river_name=""
    print("  River layer: removed (Natural Earth name 'Guainía' incorrect at this scale)")

    # Tiles / overlap frozen except R3 alpha=0.06.
    panel_b_box=box(*panel_b_bounds); tile_geoms={str(r["tile"]):r.geometry for _,r in tiles.iterrows()}
    overlap_geom=tile_geoms["T18NXJ"].intersection(tile_geoms["T18NXK"]).intersection(panel_b_box)
    if overlap_geom is None or overlap_geom.is_empty: raise SystemExit("Sentinel-2 tile overlap is empty in panel B.")
    overlap_alpha=0.06
    gpd.GeoSeries([overlap_geom],crs=tiles.crs).plot(ax=ax_b,facecolor=COL["tile"],edgecolor="none",alpha=overlap_alpha,zorder=2)
    tiles.boundary.plot(ax=ax_b,color=COL["tile"],linewidth=0.7,linestyle=(0,(5,3)),zorder=6)
    tile_boundary=unary_union(list(tiles.boundary))
    ob=overlap_geom.bounds; dx_b=xmax_b-xmin_b; dy_b=ymax_b-ymin_b
    tile_label_x=xmax_b-dx_b*0.025; y_nxk=min(ymax_b-dy_b*0.035,ob[3]+dy_b*0.025); y_nxj=max(ymin_b+dy_b*0.035,ob[1]-dy_b*0.025)
    for name,yy in [("T18NXK",y_nxk),("T18NXJ",y_nxj)]:
        manual_texts.append(ax_b.text(tile_label_x,yy,name,fontsize=ANNOT_PT,color=COL["tile"],ha="right",va="center",path_effects=[pe.withStroke(linewidth=1.5,foreground="white")],zorder=12))

    farms.plot(ax=ax_b,facecolor=COL["farm"],edgecolor="white",linewidth=0.3,alpha=0.85,zorder=8)
    farm3948.plot(ax=ax_b,facecolor=COL["farm"],edgecolor=COL["accent"],linewidth=0.9,alpha=0.90,zorder=9)

    # Small farms -> dots, then R3-11 separate overlaps to >=1.8 mm.
    fig.canvas.draw(); renderer=fig.canvas.get_renderer(); px_per_mm=fig.dpi/MM_PER_IN
    small_rows=[]; small_points=[]
    for idx,r in farms.iterrows():
        g=r.geometry; gx0,gy0,gx1,gy1=g.bounds
        q0=ax_b.transData.transform((gx0,gy0)); q1=ax_b.transData.transform((gx1,gy1))
        wmm=abs(q1[0]-q0[0])/px_per_mm; hmm=abs(q1[1]-q0[1])/px_per_mm
        if min(wmm,hmm)<1.2:
            small_rows.append(idx); small_points.append(g.representative_point())
    adjusted_points,dot_pairs_separated,min_dot_mm=_separate_dot_symbols(ax_b,small_points,1.8,px_per_mm)
    dot_artists=[]
    for p in adjusted_points:
        ln=ax_b.plot(p.x,p.y,marker="o",markersize=1.6/MM_PER_IN*72.0,markerfacecolor=COL["farm"],markeredgecolor="white",markeredgewidth=0.2,zorder=10)[0]
        dot_artists.append(ln)

    # R3 farm-3948 red box = source farm outline + 1.5 mm.
    map_units_per_mm_b=dx_b/B_W; farm_box_pad=1.5*map_units_per_mm_b
    fb=outline.bounds; fb_box=box(fb[0]-farm_box_pad,fb[1]-farm_box_pad,fb[2]+farm_box_pad,fb[3]+farm_box_pad); fbb=fb_box.bounds
    ax_b.add_patch(Rectangle((fbb[0],fbb[1]),fbb[2]-fbb[0],fbb[3]-fbb[1],fill=False,edgecolor=COL["accent"],linewidth=0.8,zorder=14))

    # B scale bar / graticule / north arrow frozen.
    forbidden_b=[all_farms_geom,tile_boundary,fb_box]
    corner=choose_scale_bar_corner((xmin_b,xmax_b),(ymin_b,ymax_b),[0,5,10],forbidden_b)
    scale_b_geom=add_metric_scale_bar(ax_b,(xmin_b,xmax_b),(ymin_b,ymax_b),[0,5,10],corner=corner)
    grat_texts,lat_texts=_add_graticule_r3(ax_b,panel_b_bounds,B_W,b_h,step=0.2); manual_texts.extend(grat_texts)
    north_geom=place_north_arrow(ax_b,(xmin_b,xmax_b),(ymin_b,ymax_b),forbidden_b+[scale_b_geom])

    # Farm symbol display boxes for R3 c-label and municipality tests.
    fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    from matplotlib.transforms import Bbox
    farm_display=[]
    for _,r in farms.iterrows():
        farm_display.append(data_box_to_display(ax_b,r.geometry))
    dot_half=0.8*px_per_mm
    for p in adjusted_points:
        x,y=ax_b.transData.transform((p.x,p.y)); farm_display.append(Bbox.from_extents(x-dot_half,y-dot_half,x+dot_half,y+dot_half))

    # R3-4 place red c OUTSIDE the farm box; test farms +0.5 mm.
    c_candidates=[(fbb[0],fbb[3],"right","bottom"),(fbb[0],fbb[1],"right","top"),(fbb[2],fbb[1],"left","top"),(fbb[2],fbb[3],"left","bottom")]
    c_label=None
    for x,y,ha,va in c_candidates:
        t=ax_b.text(x,y,"c",fontsize=7.0,weight="bold",color=COL["accent"],ha=ha,va=va,path_effects=[pe.withStroke(linewidth=1.5,foreground="white")],zorder=15)
        fig.canvas.draw(); bb=t.get_window_extent(renderer=fig.canvas.get_renderer()); pad=0.5*px_per_mm
        if any(bbox_intersects(bb,fbx,pad_px=pad) for fbx in farm_display): t.remove(); continue
        c_label=t; manual_texts.append(t); break
    if c_label is None: problems.append('label "c" has no farm-clear outside-box position')

    # R4-1: no river label or river object.
    river_label=None; river_label_poly=None

    # R3-5 municipality labels: collision-aware; omit if no valid candidate.
    omitted=[]; municipality_texts=[]
    fig.canvas.draw(); renderer=fig.canvas.get_renderer(); axbbb=ax_b.get_window_extent(renderer=renderer)
    scale_b_display=data_box_to_display(ax_b,scale_b_geom); north_display=data_box_to_display(ax_b,north_geom); fb_display=data_box_to_display(ax_b,fb_box)
    tile_display_geoms=[g for g in tiles.boundary]
    if adm2 is not None and len(adm2):
        mname=find_name_col(adm2,["shapeName","name","NAME_2","admin2Name"])
        x_units_per_mm=dx_b/B_W; y_units_per_mm=dy_b/b_h
        for _,r in adm2.iterrows():
            if r.geometry is None or r.geometry.is_empty: continue
            txt=smart_place_name(str(r[mname])); rp=r.geometry.representative_point(); chosen=None; chosen_text=None
            for ox,oy in _candidate_offsets_mm():
                p=Point(rp.x+ox*x_units_per_mm,rp.y+oy*y_units_per_mm)
                if not r.geometry.contains(p): continue
                t=ax_b.text(p.x,p.y,txt,fontsize=ANNOT_PT,style="italic",color="#4F4F4F",ha="center",va="center",path_effects=[pe.withStroke(linewidth=1.5,foreground="white")],zorder=12)
                fig.canvas.draw(); rr=fig.canvas.get_renderer(); bb=t.get_window_extent(renderer=rr); pad=0.5*px_per_mm
                # >=1 mm inside panel frame.
                if not (bb.x0>=axbbb.x0+1.0*px_per_mm and bb.x1<=axbbb.x1-1.0*px_per_mm and bb.y0>=axbbb.y0+1.0*px_per_mm and bb.y1<=axbbb.y1-1.0*px_per_mm): t.remove(); continue
                poly=_display_bbox_to_data_polygon(ax_b,bb,pad_px=pad)
                bad=poly.intersects(tile_boundary)
                bad=bad or poly.intersects(fb_box) or poly.intersects(scale_b_geom) or poly.intersects(north_geom)
                if any(bbox_intersects(bb,fx,pad_px=pad) for fx in farm_display): bad=True
                if bad: t.remove(); continue
                chosen=p; chosen_text=t; break
            if chosen_text is None: omitted.append(txt)
            else: municipality_texts.append(chosen_text); manual_texts.append(chosen_text)
    print(f"  Municipality labels omitted: {omitted}")

    # ---------------- Panel C ----------------
    ax_c.set_xlim(xmin_c,xmax_c); ax_c.set_ylim(ymin_c,ymax_c)
    ax_c.set_aspect("equal",adjustable="datalim")
    fig.canvas.draw()
    xmin_c,xmax_c=ax_c.get_xlim(); ymin_c,ymax_c=ax_c.get_ylim()
    c_bounds=(xmin_c,ymin_c,xmax_c,ymax_c)
    for ut in ["0","2","3","4","6","7","8"]:
        sub=p3948[p3948["_ut"]==ut]
        if not len(sub): continue
        if ut=="0":
            sub.plot(ax=ax_c,facecolor="white",edgecolor="#7F7F7F",linewidth=0.3,hatch="////",zorder=5)
        else:
            sub.plot(ax=ax_c,facecolor=UT_COLORS[ut],edgecolor="white",linewidth=0.25,zorder=5)
    gpd.GeoSeries([outline],crs=9377).boundary.plot(ax=ax_c,color="black",linewidth=0.6,zorder=8)
    scale_c_geom=_add_panel_c_scale_bar(ax_c,(xmin_c,xmax_c),(ymin_c,ymax_c),outline,segments_km=(0,0.5,1))
    if scale_c_geom is None:
        problems.append("panel-c scale bar has no collision-free lower-left placement")
    elif scale_c_geom.intersects(outline):
        problems.append("panel-c scale bar intersects farm 3948 outline")

    # ---------------- A -> B connectors only (R3-10) ----------------
    for source_xy,target_xy in [((xmax_b,ymax_b),(0.0,1.0)),((xmax_b,ymin_b),(0.0,0.0))]:
        cpobj=ConnectionPatch(xyA=source_xy,coordsA=ax_a.transData,xyB=target_xy,coordsB=ax_b.transAxes,color="#C0392B",alpha=0.6,linewidth=0.5,zorder=5.5,clip_on=False)
        fig.add_artist(cpobj); connectors.append(cpobj)

    # ---------------- Legend in right column below C ----------------
    farms_patch=Patch(facecolor=COL["farm"],edgecolor="white")
    farms_dot=Line2D([0],[0],marker="o",markersize=4.0,markerfacecolor=COL["farm"],markeredgecolor="white",markeredgewidth=0.4,linestyle="None")
    farms_handle=(farms_patch,farms_dot)
    handles=[farms_handle,Patch(facecolor=COL["farm"],edgecolor=COL["accent"],linewidth=1.0),Line2D([0],[0],color=COL["tile"],lw=0.8,linestyle=(0,(5,3))),Patch(facecolor=COL["tile"],edgecolor="none",alpha=0.06),Line2D([0],[0],color=COL["municipal"],lw=0.5,linestyle=(0,(3,2)))]
    labels=["Farms (n = 22; small farms as dots)","Farm 3948","Sentinel-2 tile boundary","Tile overlap","Municipal boundary"]
    for ut in ut_present:
        if ut=="0": handles.append(Patch(facecolor="white",edgecolor="#7F7F7F",hatch="////",linewidth=0.3))
        else: handles.append(Patch(facecolor=UT_COLORS[ut],edgecolor="#555555",linewidth=0.2))
        labels.append(f"UT{ut}")
    legend=ax_leg.legend(handles=handles,labels=labels,handler_map={tuple:HandlerTuple(ndivide=None,pad=0.35)},loc="lower left",bbox_to_anchor=(0.0,0.0),ncol=1,frameon=False,fontsize=6.0,handlelength=1.6,labelspacing=-0.12,columnspacing=0.5,handletextpad=0.45,borderaxespad=0.0)

    attr=("Relief: Copernicus DEM GLO-90 © DLR e.V.\n"
          "2010–2014 and © Airbus Defence and Space GmbH\n"
          "2014–2018, provided under COPERNICUS by the\n"
          "European Union and ESA.\n"
          "Boundaries: geoBoundaries gbOpen.")
    attr_text=ax_attr.text(0.0,0.0,attr,transform=ax_attr.transAxes,fontsize=ATTR_PT,color="#666666",ha="left",va="bottom",linespacing=1.0)

    # ---------------- R3 automated tests ----------------
    fig.canvas.draw(); renderer=fig.canvas.get_renderer(); px_per_mm=fig.dpi/MM_PER_IN
    from matplotlib.transforms import Bbox

    # R4-4 symmetric side panels and aligned text blocks.
    lbb=legend.get_window_extent(renderer=renderer); bbb=ax_b.get_window_extent(renderer=renderer); cbb=ax_c.get_window_extent(renderer=renderer); abb=ax_a.get_window_extent(renderer=renderer)
    attrbb=attr_text.get_window_extent(renderer=renderer)
    wa,ha=abb.width/px_per_mm,abb.height/px_per_mm
    wb,hb=bbb.width/px_per_mm,bbb.height/px_per_mm
    wc,hc=cbb.width/px_per_mm,cbb.height/px_per_mm
    if abs(wa-wc)>=0.2: problems.append(f"R4 side-panel width mismatch: a={wa:.2f} mm c={wc:.2f} mm")
    if abs(ha-hc)>=0.2: problems.append(f"R4 side-panel height mismatch: a={ha:.2f} mm c={hc:.2f} mm")
    if abs(abb.y1-bbb.y1)>0.2*px_per_mm or abs(cbb.y1-bbb.y1)>0.2*px_per_mm: problems.append("R4 panel tops are not aligned within 0.2 mm")
    if abs(lbb.y0-bbb.y0)>1.0*px_per_mm: problems.append(f"legend bottom misaligned with panel B by {abs(lbb.y0-bbb.y0)/px_per_mm:.2f} mm")
    if lbb.x0 < cbb.x0-0.5*px_per_mm or lbb.x1 > cbb.x1+0.5*px_per_mm: problems.append("legend exceeds panel-C column")
    if abs(attrbb.y0-bbb.y0)>1.0*px_per_mm: problems.append(f"attribution bottom misaligned with panel B by {abs(attrbb.y0-bbb.y0)/px_per_mm:.2f} mm")
    if attrbb.x0 < abb.x0-0.5*px_per_mm or attrbb.x1 > abb.x1+0.5*px_per_mm: problems.append("attribution exceeds panel-A column")

    # R3-2 outline geometry: preserve all substantive source components.
    if outline.geom_type == "Polygon":
        pass
    elif outline.geom_type == "MultiPolygon":
        if not all(h > 0 for h in outline_component_hits):
            problems.append("farm 3948 MultiPolygon contains a component with no management-polygon support")
        source_area=float(farm3948_features.geometry.union_all().area)
        if abs(outline.area-source_area) > max(1.0, source_area*1e-5):
            problems.append("farm 3948 multipart outline area does not match source geometry")
    else:
        problems.append(f"farm 3948 outline has unsupported geometry type {outline.geom_type}")

    # R3-4 c label vs symbols with 0.5-mm buffer.
    if c_label is not None:
        cbbx=c_label.get_window_extent(renderer=renderer); pad=0.5*px_per_mm
        if any(bbox_intersects(cbbx,fx,pad_px=pad) for fx in farm_display): problems.append('label "c" intersects a farm symbol')

    # R4-1 river removed, R4-2 sea background, R4-3 UT0 hatch.
    if river_union is not None or river_label is not None: problems.append("R4 river object still present")
    if str(mpl.colors.to_hex(ax_a.get_facecolor(),keep_alpha=False)).lower() != "#eaf2f8": problems.append("panel-A sea background is not #EAF2F8")
    if abs(float(mpl.rcParams.get("hatch.linewidth",0.0))-0.4)>1e-9: problems.append("UT0 hatch linewidth is not 0.4")

    # R3-7 overlap alpha exact.
    if abs(overlap_alpha-0.06)>1e-12: problems.append("tile overlap alpha is not 0.06")

    # Recompute actual connector display samples after artists exist.
    conn_samples=[]
    for cpobj in connectors:
        path=cpobj.get_path().transformed(cpobj.get_transform()); vv=path.vertices
        sm=[]
        for aa,bbv in zip(vv[:-1],vv[1:]):
            for tt in np.linspace(0,1,160): sm.append(aa*(1-tt)+bbv*tt)
        conn_samples.append(np.asarray(sm))
    conn_all=np.vstack(conn_samples) if conn_samples else np.empty((0,2))

    # R3-8 Colombia full bbox +1 mm contained in mainland, avoids Meta/connectors.
    colbb=colombia_label.get_window_extent(renderer=renderer); colpoly=_display_bbox_to_data_polygon(ax_a,colbb,pad_px=1.0*px_per_mm)
    if not mainland.contains(colpoly):
        if r3_8_status == "WAIVED_GEOMETRIC_INFEASIBILITY":
            print(
                "  R3-8 automated containment test:",
                "WAIVED_GEOMETRIC_INFEASIBILITY",
                f"(minimum outside = {100.0*r3_8_outside_fraction:.2f}%)"
            )
        else:
            problems.append(
                "COLOMBIA label +1 mm is not contained in mainland Colombia"
            )
    if meta_geom is not None and colpoly.intersects(meta_geom):
        problems.append("COLOMBIA label intersects Meta")
    if _bbox_hits_samples(colbb,conn_all,pad_px=1.0*px_per_mm):
        problems.append("COLOMBIA label intersects A-B connector")

    # R3-9 Meta tests.
    if meta_label is not None:
        mbb=meta_label.get_window_extent(renderer=renderer); mpoly=_display_bbox_to_data_polygon(ax_a,mbb,pad_px=0.0)
        if mpoly.intersects(reg_rect_geom): problems.append("Meta label intersects regional red box")
        if mpoly.intersects(meta_geom.boundary): problems.append("Meta label intersects Meta boundary")
        if _bbox_hits_samples(mbb,conn_all,pad_px=0.5): problems.append("Meta label intersects A-B connector")

    # R3-10 connectors do not intersect any panel-A text and do not enter panel B interior.
    a_texts=[t for t in manual_texts if getattr(t,'axes',None) is ax_a and t.get_visible()]
    for k,samps in enumerate(conn_samples,1):
        for t in a_texts:
            bb=t.get_window_extent(renderer=renderer)
            if _bbox_hits_samples(bb,samps,pad_px=0.4): problems.append(f"connector {k} intersects panel-A text {t.get_text()!r}")
        body=samps[:-4] if len(samps)>4 else samps
        if np.any((body[:,0]>bbb.x0+0.5)&(body[:,0]<bbb.x1)&(body[:,1]>bbb.y0)&(body[:,1]<bbb.y1)): problems.append(f"connector {k} enters panel B interior")

    # R3-11 dot separation.
    if len(adjusted_points)>1 and min_dot_mm < 1.8-1e-6: problems.append(f"minimum farm-dot distance {min_dot_mm:.3f} mm < 1.8 mm")

    # Latitudes/longitudes remain outside frames (frozen R2 behavior).
    for t in lat_texts:
        bb=t.get_window_extent(renderer=renderer)
        if bb.x1>bbb.x0-0.2: problems.append(f"latitude label not outside panel B: {t.get_text()!r}")

    # R4-5 scale-bar label/frame clearance and connector avoidance.
    for nm,ax,geom,txts in [("a",ax_a,scale_a_geom,scale_a_texts)]:
        axbb=ax.get_window_extent(renderer=renderer)
        for t in txts:
            bb=t.get_window_extent(renderer=renderer)
            if bb.x0 < axbb.x0+1.5*px_per_mm or bb.x1 > axbb.x1-1.5*px_per_mm or bb.y0 < axbb.y0+1.5*px_per_mm or bb.y1 > axbb.y1-1.5*px_per_mm:
                problems.append(f"panel-{nm} scale label lacks 1.5-mm frame margin")
            if _bbox_hits_samples(bb,conn_all,pad_px=0.2*px_per_mm):
                problems.append(f"panel-{nm} scale label intersects A-B connector")
    scale_a_disp=data_box_to_display(ax_a,scale_a_geom)
    for samps in conn_samples:
        if np.any((samps[:,0]>=scale_a_disp.x0)&(samps[:,0]<=scale_a_disp.x1)&(samps[:,1]>=scale_a_disp.y0)&(samps[:,1]<=scale_a_disp.y1)):
            problems.append("A-B connector intersects panel-A scale bar")

    # R3-13 equal-aspect test exactly as specified.
    for ax in (ax_a,ax_b,ax_c):
        x0,x1=ax.get_xlim(); y0,y1=ax.get_ylim(); bb=ax.get_window_extent(renderer=renderer)
        data_ratio=(x1-x0)/(y1-y0); box_ratio=bb.width/bb.height
        if abs(data_ratio/box_ratio-1)>=0.01: problems.append(f"{ax.get_label()}: distorted ({data_ratio:.3f} vs {box_ratio:.3f})")

    # Figure margin safeguard for panels and core text.
    figbb=fig.bbox; margin=1.0*px_per_mm
    for nm,ax in [("a",ax_a),("b",ax_b),("c",ax_c)]:
        bb=ax.get_window_extent(renderer=renderer)
        if bb.x0<figbb.x0+margin or bb.x1>figbb.x1-margin or bb.y0<figbb.y0+margin or bb.y1>figbb.y1-margin: problems.append(f"panel {nm} outside 1-mm figure margin")

    # Mandatory console lines before decision.
    print(f"  Municipality labels omitted: {omitted}")
    print(f"  Dot pairs separated: {dot_pairs_separated}")
    print(f"  Minimum dot-center distance: {min_dot_mm:.3f} mm" if np.isfinite(min_dot_mm) else "  Minimum dot-center distance: n/a")
    print(f"  Panel (a) Colombia width/height: {a_ratio:.3f}")
    print(f"  Panel sizes (mm): a = {wa:.1f} x {ha:.1f} ; b = {wb:.1f} x {hb:.1f} ; c = {wc:.1f} x {hc:.1f}")
    print(f"  Figure size: {FIG_W_MM:.0f} x {FIG_H_MM:.1f} mm")

    if problems:
        unique=list(dict.fromkeys(problems))
        print("\nOverlap check: FAILED")
        for item in unique[:80]: print("  -",item)
        raise SystemExit(f"Overlap check found {len(unique)} problems; outputs not saved.")
    print("Overlap check: 0 problems")

    # Provenance / hashes.
    primary=[inp.farms_path,inp.pgo_path,inp.tiles_path]
    for q in [inp.adm1_path,inp.adm2_path,inp.countries_path]:
        if q is not None: primary.append(q)
    primary.extend(inp.dem_paths)
    print("SHA-256 of inputs:")
    input_hashes=[]
    for q in primary:
        h=sha256(q); input_hashes.append({"path":os.path.relpath(str(q), str(root)),"sha256":h}); print(f"  {h}  {q}")

    caption=("Study area in the Orinoco Piedmont, Department of Meta, Colombia. (a) Location in Colombia. "
             "(b) The 22 farms (small farms shown as dots), municipal boundaries and Sentinel-2 tiles 18NXJ and 18NXK (shaded: tile overlap) over shaded relief. "
             "(c) Management polygons of farm 3948 (red box in b; shown in Figure 3) by SCR land-use category (Table 1). Relief: Copernicus DEM GLO-90.")
    provenance={
        "figure":"Figure 1 — study area, R4 CIERRE",
        "instruction":"INSTRUCCION_FIGURA1_R4_CIERRE.md",
        "figure_size_mm":[FIG_W_MM,FIG_H_MM],
        "font":FONT_FAMILY,"caption":caption,
        "counts":{"farm_features":int(len(farms_raw)),"unique_farms":int(len(farms)),"area_ha_attribute_once_per_farm":round(area_once,2),"area_ha_geometry_epsg9377":round(geometry_area_ha,3),"farm_3948_features":int(len(farm3948_features)),"farm_3948_outline_components":int(outline_component_count),"farm_3948_outline_area_ha":round(outline_area_ha,3),"farm_3948_management_polygons":int(len(p3948)),"farm_3948_ut":ut_present,"dot_pairs_separated":int(dot_pairs_separated),"municipality_labels_omitted":omitted},
        "cartography":{"panel_b_extent_frozen_from_R2":list(map(float,panel_b_bounds)),"regional_extent_west_extension_m":9000,"relief_vertical_exaggeration":2.5,"relief_alpha":0.70,"tile_overlap_alpha":0.06,"panel_c_margin_fraction":0.06,"panel_bc_connectors":"none; red box/frame relation"},
        "restricted_geometry":{"farm_3948_shown":True,"publication_authorization":"pending SCR confirmation","coordinates_suppressed_in_panel_c":True},
        "inputs":input_hashes,"pipeline_colors_imported_from_figure2":fig2_cols,
    }

    # Save only after all tests pass.
    preview_path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(preview_path,dpi=600,facecolor="white")
    print(f"PREVIEW PNG: {preview_path}")
    if final:
        outdir=root/"NDVI_Sen2Cor_LaSRC_OpenScience"/"manuscript/source/figures"; outdir.mkdir(parents=True,exist_ok=True)
        pdf=outdir/"figure_01_study_area.pdf"; svg=outdir/"figure_01_study_area.svg"; png=outdir/"figure_01_study_area.png"; prov=outdir/"figure_01_study_area_provenance.json"
        fig.savefig(pdf,facecolor="white"); fig.savefig(svg,facecolor="white"); fig.savefig(png,dpi=600,facecolor="white")
        prov.write_text(json.dumps(provenance,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
        print("FINAL OUTPUTS:"); print(" ",pdf); print(" ",svg); print(" ",png); print(" ",prov)
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description="Generate Figure 1 R3 FINAL study-area map.")
    ap.add_argument("--root",type=Path,default=Path.cwd(),help="A3_lasrc root. Default: current directory.")
    ap.add_argument("--preview",type=Path,default=Path("NDVI_Sen2Cor_LaSRC_OpenScience/manuscript/source/figures/figure_01_study_area_preview.png"),help="Preview PNG path, relative to --root unless absolute.")
    ap.add_argument("--final",action="store_true",help="Also write canonical PDF/SVG/PNG and provenance JSON.")
    args=ap.parse_args(); root=args.root.resolve(); preview=args.preview if args.preview.is_absolute() else root/args.preview
    print("="*96); print("FIGURE 1 R4 CIERRE — STUDY AREA IN CASCADE OF SCALES"); print("="*96); print("ROOT:",root); print("MODE:","FINAL" if args.final else "R4 PREVIEW"); print("No downloads; no recursive searches; no web tiles.")
    draw_figure(root,preview,args.final)
    print("="*96); print("END"); print("="*96)


if __name__=="__main__":
    main()
