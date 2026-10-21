# Version: Figure 2 V4.1 — Sentinel-2 NDVI Comparison Workflow
# Date: 2026-10-15
# Status: proposed — corrected for the real project schema; authoritative validation must run in the project itself
# Instruction implemented: Dictamen Figura 2 V4 + corrección V4.1 — real-file parser, fixed V4 geometry defects, and expanded arrow/collision QA.
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.transforms import Bbox
import pandas as pd

# -----------------------------------------------------------------------------
# 0. Fixed physical layout (millimetres)
# -----------------------------------------------------------------------------
MM_TO_IN = 1.0 / 25.4
CANVAS_W_MM = 170.0
CANVAS_H_MM = 225.0
MARGIN_MM = 3.0

# Required grid
X_PRODUCT_0, X_PRODUCT_1 = 3.0, 40.0
X_LEFT_GUTTER_0, X_LEFT_GUTTER_1 = 40.0, 46.0
X_SPINE_0, X_SPINE_1 = 46.0, 132.0
X_RIGHT_GUTTER_0, X_RIGHT_GUTTER_1 = 132.0, 139.0
X_MASK_CHANNEL = 135.5
X_TOOLS_0, X_TOOLS_1 = 139.0, 167.0

SOURCE_Y0, SOURCE_Y1 = 3.0, 17.0
BLOCK_GAP_MM = 4.0
HEADER_H_MM = 6.0
BODY_MARGIN_MM = 2.0
LEGEND_H_MM = 7.0

# Tool bars: 4 × 6 mm + 3 × 1.3 mm = 27.9 mm
TOOL_BAR_W_MM = 6.0
TOOL_BAR_GAP_MM = 1.3

# Typography
FONT_FAMILY = "Liberation Sans Narrow"
HEADER_PT = 8.0
PIPELINE_PT = 6.5
BODY_PT = 6.5
FOOT_PT = 6.2
PRODUCT_TITLE_PT = 7.0
PRODUCT_BODY_PT = 6.5
CARDINALITY_PT = 7.5
TOOL_PT = 6.5
MIN_PT = 6.0

# Approximate line geometry used to calculate heights; actual fit is verified by renderer.
LINE_H_MM = 2.90
BULLET_LINE_H_MM = 2.90
PIPE_HEAD_H_MM = 4.3
FOOT_LINE_H_MM = 3.6
CHIP_H_MM = 5.0
FORMULA_H_MM = 2.85

# Confirmed in V4 content specification.
L1C_FMASK_TOOL = "sen2lasrc"

mpl.rcParams.update(
    {
        "font.family": FONT_FAMILY,
        "font.size": BODY_PT,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
    }
)

COL = {
    "text": "#23272b",
    "border": "#747c83",
    "muted": "#657078",
    "data_header": "#d7e9f6",
    "process_header": "#f6e7b3",
    "qc_header": "#dcefd8",
    "analysis_header": "#e8dcf5",
    "release_header": "#d8e0e8",
    "l1c": "#2f72b5",
    "l1c_fill": "#edf4fb",
    "s2c": "#4d913f",
    "s2c_fill": "#eef6ea",
    "lasrc": "#cf7921",
    "lasrc_fill": "#fcf0e4",
    "mask": "#c53c3c",
    "mask_fill": "#fff7f7",
    "product": "#f8f9fa",
    "black": "#202428",
    "gray_arrow": "#7c858b",
    "tool_python": "#7d6aa7",
}

# -----------------------------------------------------------------------------
# 1. Canonical data gate
# -----------------------------------------------------------------------------

def _read_csv(path: Path, label: str) -> pd.DataFrame:
    print(f"READ [{label}] {path}")
    if not path.exists():
        raise SystemExit(f"Missing required source file: {path}")
    return pd.read_csv(path, low_memory=False)


def _read_json(path: Path, label: str) -> dict:
    print(f"READ [{label}] {path}")
    if not path.exists():
        raise SystemExit(f"Missing required source file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_canonical(root: Path) -> dict:
    """Read every Figure-2 count from the real project files documented in V4.1.

    No synthetic snapshot fields are accepted. The gate fails if the canonical
    common-set counts differ from N=1,797 / 106 polygons / 22 farms.
    """
    agreement_path = root / "data/derived/agreement_three_pipelines_2019_2020.csv"
    masking_path = root / "data/derived/masking_sensitivity_2019_2020.csv"
    land_path = root / "data/derived/land_use_gradient_2019_2020.csv"
    common_path = root / "local_only/workflow_contract_draft/canonical_build/common_three_pipeline_records_2019_2020.csv"
    scene_path = root / "local_only/workflow_contract_draft/data/metadata/input_scene_manifest.csv"
    aux_path = root / "local_only/workflow_contract_draft/data/metadata/auxiliary_data_manifest.csv"
    geom_path = root / "local_only/workflow_contract_draft/data/metadata/geometry_manifest.json"
    acceptance_path = root / "local_only/workflow_contract_draft/canonical_build/acceptance_report.json"
    low_pixel_path = root / "local_only/workflow_contract_draft/metadata/lasrc_below_10_pixel_records.csv"
    exclusion_path = root / "local_only/workflow_contract_draft/data/metadata/exclusion_ledger.csv"

    agreement = _read_csv(agreement_path, "agreement")
    masking = _read_csv(masking_path, "masking")
    land = _read_csv(land_path, "land_use")
    common = _read_csv(common_path, "common_records")
    scene = _read_csv(scene_path, "input_scene_manifest")
    aux = _read_csv(aux_path, "auxiliary_data_manifest")
    low_pixel = _read_csv(low_pixel_path, "low_pixel_records")
    exclusions = _read_csv(exclusion_path, "exclusion_ledger")
    geom = _read_json(geom_path, "geometry_manifest")
    acceptance = _read_json(acceptance_path, "acceptance_report")

    # 1) Canonical three-pipeline set: real period label is 2019-2020.
    pooled = agreement.loc[agreement["period"].astype(str).eq("2019-2020")].copy()
    if len(pooled) != 3:
        raise SystemExit(f"Expected 3 agreement rows with period == '2019-2020'; observed {len(pooled)}")
    for col in ["observation_count", "management_polygon_count", "farm_count"]:
        vals = pooled[col].dropna().astype(int).unique().tolist()
        if len(vals) != 1:
            raise SystemExit(f"Agreement pooled rows disagree on {col}: {vals}")

    common_n = int(pooled["observation_count"].iloc[0])
    common_polygons = int(pooled["management_polygon_count"].iloc[0])
    common_farms = int(pooled["farm_count"].iloc[0])
    if (common_n, common_polygons, common_farms) != (1797, 106, 22):
        raise SystemExit(
            "Canonical count mismatch: expected (1797, 106, 22); observed "
            f"{(common_n, common_polygons, common_farms)}"
        )

    # 2) Masking sensitivity.
    m_primary = masking.loc[masking["analysis_set"].astype(str).eq("primary")]
    m_strict = masking.loc[masking["analysis_set"].astype(str).eq("strict_10_pixel")]
    if len(m_primary) != 1 or len(m_strict) != 1:
        raise SystemExit("Masking table must contain exactly one primary and one strict_10_pixel row")
    masking_n = int(m_primary.iloc[0]["observation_count"])
    masking_polygons = int(m_primary.iloc[0]["management_polygon_count"])
    masking_farms = int(m_primary.iloc[0]["farm_count"])
    masking_strict_n = int(m_strict.iloc[0]["observation_count"])
    if (masking_n, masking_strict_n) != (2585, 2521):
        raise SystemExit(f"Masking counts mismatch: observed primary={masking_n}, strict={masking_strict_n}")

    # 3) Land-use N once per UT (the table repeats counts by comparison).
    land_once = land.sort_values(["land_use_code", "comparison"]).drop_duplicates("land_use_code")
    land_n = int(land_once["observation_count"].sum())
    if land_n != 1527:
        raise SystemExit(f"Land-use count mismatch: expected 1527; observed {land_n}")

    # Farms for land use are derived from the real canonical common records, excluding UT0.
    if not {"CLAVE_POLIGONO", "farm"}.issubset(common.columns):
        raise SystemExit("Common-record file lacks required columns CLAVE_POLIGONO and farm")
    if len(common) != common_n:
        raise SystemExit(f"Common-record row count mismatch: agreement N={common_n}; common file rows={len(common)}")
    if int(common["farm"].nunique()) != common_farms:
        raise SystemExit(f"Common-record farm count mismatch: expected {common_farms}; observed {common['farm'].nunique()}")
    if int(common["CLAVE_POLIGONO"].nunique()) != common_polygons:
        raise SystemExit(f"Common-record polygon-key count mismatch: expected {common_polygons}; observed {common['CLAVE_POLIGONO'].nunique()}")
    non_ut0 = common.loc[~common["CLAVE_POLIGONO"].astype(str).str.contains(r"__UT0$", regex=True)].copy()
    land_farms = int(non_ut0["farm"].nunique())
    if land_farms != 21:
        raise SystemExit(f"Land-use farm count mismatch: expected 21; observed {land_farms}")

    # 4) Input-scene manifest. Do not assume provider-specific fields; only the real 'pipeline' column.
    if "pipeline" not in scene.columns:
        raise SystemExit(f"input_scene_manifest.csv lacks 'pipeline'; columns={list(scene.columns)}")
    pipe_counts = scene["pipeline"].astype(str).value_counts().to_dict()

    def count_pipeline(token: str) -> int:
        matches = {k: int(v) for k, v in pipe_counts.items() if token.lower() in k.lower()}
        if len(matches) != 1:
            raise SystemExit(f"Could not uniquely resolve pipeline token {token!r}: {matches}; all={pipe_counts}")
        return next(iter(matches.values()))

    scene_l1c = count_pipeline("L1C")
    scene_s2c = count_pipeline("Sen2Cor")
    scene_lasrc = count_pipeline("LaSRC")
    scene_total = int(len(scene))
    if (scene_total, scene_l1c, scene_s2c, scene_lasrc) != (708, 157, 175, 376):
        raise SystemExit(
            "Input-scene manifest mismatch: expected (708,157,175,376); observed "
            f"{(scene_total, scene_l1c, scene_s2c, scene_lasrc)}"
        )

    # 5) Atmospheric auxiliaries are the row count of the real manifest.
    aux_n = int(len(aux))
    if aux_n != 217:
        raise SystemExit(f"Auxiliary manifest mismatch: expected 217 rows; observed {aux_n}")

    # 6) Geometry manifest: real field structure.
    try:
        geom_farms = int(geom["selection"]["n_farms"])
        geom_polygons = int(geom["analytical_key"]["unique_keys_in_selection"])
        analytical_chain = geom["analytical_chain"]
        inner_join_keys = int(analytical_chain["three_pipeline_inner_join_keys"])
        final_keys = int(analytical_chain["final_three_pipeline_keys"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"geometry_manifest.json lacks the V4.1 required structure: {exc}")
    if (geom_farms, geom_polygons, inner_join_keys, final_keys) != (22, 141, 107, 106):
        raise SystemExit(
            "Geometry-chain mismatch: expected (22,141,107,106); observed "
            f"{(geom_farms, geom_polygons, inner_join_keys, final_keys)}"
        )

    # 7) Recompute 1,810 -> 1,809 -> 1,797 from record-level evidence.
    #    final common records + 12 records excluded by the >=10-pixel gate = 1,809;
    #    + the single documented cross-product exclusion = 1,810.
    low_pixel_n = int(len(low_pixel))
    documented_exclusions = int(len(exclusions))
    if low_pixel_n != 12:
        raise SystemExit(f"Expected 12 low-pixel records; observed {low_pixel_n}")
    if documented_exclusions != 1:
        raise SystemExit(f"Expected 1 documented cross-product exclusion; observed {documented_exclusions}")
    after_exclusion_n = common_n + low_pixel_n
    before_exclusion_n = after_exclusion_n + documented_exclusions
    if (before_exclusion_n, after_exclusion_n, common_n) != (1810, 1809, 1797):
        raise SystemExit(
            "Record-chain mismatch: expected 1810 -> 1809 -> 1797; observed "
            f"{before_exclusion_n} -> {after_exclusion_n} -> {common_n}"
        )

    # 8) Acceptance report: only real top-level keys are used.
    status = str(acceptance.get("status", ""))
    failures = acceptance.get("failures")
    dataset_counts = acceptance.get("dataset_counts", {})
    if status != "PASS" or failures != []:
        raise SystemExit(f"Acceptance report is not clean PASS: status={status!r}; failures={failures!r}")
    expected_dataset_counts = {
        "three_pipeline": 1797,
        "three_pipeline_farms": 22,
        "three_pipeline_management_polygons": 106,
        "masking_primary": 2585,
        "masking_strict": 2521,
    }
    for key, val in expected_dataset_counts.items():
        if int(dataset_counts.get(key, -1)) != val:
            raise SystemExit(f"acceptance_report dataset_counts[{key!r}] mismatch: {dataset_counts.get(key)!r} != {val}")

    return {
        "agreement": agreement,
        "masking": masking,
        "land": land,
        "common": common,
        "scene_manifest": scene,
        "aux_manifest": aux,
        "geometry_manifest": geom,
        "acceptance": acceptance,
        "paths_read": [
            agreement_path, masking_path, land_path, common_path, scene_path,
            aux_path, geom_path, acceptance_path, low_pixel_path, exclusion_path,
        ],
        "common_n": common_n,
        "common_polygons": common_polygons,
        "common_farms": common_farms,
        "scene_total": scene_total,
        "scene_l1c": scene_l1c,
        "scene_s2c": scene_s2c,
        "scene_lasrc": scene_lasrc,
        "aux_n": aux_n,
        "land_n": land_n,
        "land_farms": land_farms,
        "masking_n": masking_n,
        "masking_polygons": masking_polygons,
        "masking_farms": masking_farms,
        "masking_strict_n": masking_strict_n,
        "geom_farms": geom_farms,
        "geom_polygons": geom_polygons,
        "inner_join_keys": inner_join_keys,
        "final_keys": final_keys,
        "low_pixel_n": low_pixel_n,
        "documented_exclusions": documented_exclusions,
        "before_exclusion_n": before_exclusion_n,
        "after_exclusion_n": after_exclusion_n,
        "acceptance_status": status,
    }

# -----------------------------------------------------------------------------
# 2. Geometry / registry for QA
# -----------------------------------------------------------------------------

@dataclass
class BoxRec:
    name: str
    rect: Tuple[float, float, float, float]  # x0, y0, x1, y1 in mm; y increases downward
    parent: Optional[str] = None
    overlap_group: Optional[str] = None


@dataclass
class TextRec:
    name: str
    artist: object
    container: str
    margin_mm: float


@dataclass
class ArrowRec:
    name: str
    points: List[Tuple[float, float]]
    origin: str
    destination: str
    extra_allowed_boxes: Tuple[str, ...] = ()


class Registry:
    def __init__(self):
        self.boxes: Dict[str, BoxRec] = {}
        self.texts: List[TextRec] = []
        self.arrows: List[ArrowRec] = []

    def add_box(self, name: str, x: float, y: float, w: float, h: float, *, parent=None, overlap_group=None):
        self.boxes[name] = BoxRec(name, (x, y, x + w, y + h), parent=parent, overlap_group=overlap_group)

    def add_text(self, name: str, artist, container: str, margin_mm: float = 1.0):
        self.texts.append(TextRec(name, artist, container, margin_mm))

    def add_arrow(self, name: str, points: Sequence[Tuple[float, float]], origin: str, destination: str, extra_allowed_boxes=()):
        self.arrows.append(ArrowRec(name, list(points), origin, destination, tuple(extra_allowed_boxes)))


def rect_contains(outer, inner, eps=1e-6):
    return outer[0] - eps <= inner[0] and outer[1] - eps <= inner[1] and outer[2] + eps >= inner[2] and outer[3] + eps >= inner[3]


def rect_intersection(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def is_ancestor(reg: Registry, maybe_parent: str, child: str) -> bool:
    cur = reg.boxes.get(child)
    while cur and cur.parent:
        if cur.parent == maybe_parent:
            return True
        cur = reg.boxes.get(cur.parent)
    return False

# -----------------------------------------------------------------------------
# 3. Drawing helpers
# -----------------------------------------------------------------------------

def mm_box_to_display(ax, rect):
    (x0, y0), (x1, y1) = ax.transData.transform([(rect[0], rect[1]), (rect[2], rect[3])])
    return Bbox.from_extents(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def margin_px(ax, mm: float) -> float:
    p0 = ax.transData.transform((0.0, 0.0))
    p1 = ax.transData.transform((mm, 0.0))
    return abs(p1[0] - p0[0])


def fit_text(fig, ax, reg: Registry, name: str, container: str, text: str, x: float, y: float,
             *, fontsize: float, weight="normal", color=COL["text"], ha="left", va="top",
             rotation=0, style="normal", stretch="normal", xscale=1.0, margin_mm_required=1.5, linespacing=1.0, zorder=8):
    """Draw text and shrink only as needed; abort below 6 pt if it cannot fit."""
    if container not in reg.boxes:
        raise KeyError(f"Unknown text container: {container}")

    artist = ax.text(
        x, y, text,
        fontsize=fontsize,
        fontfamily=FONT_FAMILY,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        rotation=rotation,
        fontstyle=style,
        fontstretch=stretch,
        linespacing=max(linespacing, 1.15),
        zorder=zorder,
    )
    if not math.isclose(float(xscale), 1.0, abs_tol=1e-9):
        # Mild horizontal compression around the anchor; font size remains >= 6 pt.
        artist.set_transform(
            mpl.transforms.Affine2D().translate(-x, 0).scale(float(xscale), 1.0).translate(x, 0) + ax.transData
        )

    fs = float(fontsize)
    while True:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bb = artist.get_window_extent(renderer=renderer)
        cb = mm_box_to_display(ax, reg.boxes[container].rect)
        m = margin_px(ax, margin_mm_required)
        allowed = Bbox.from_extents(cb.x0 + m, cb.y0 + m, cb.x1 - m, cb.y1 - m)
        fits = allowed.contains(bb.x0, bb.y0) and allowed.contains(bb.x1, bb.y1)
        if fits:
            reg.add_text(name, artist, container, margin_mm_required)
            return artist
        fs = round(fs - 0.1, 2)
        if fs < MIN_PT - 1e-9:
            raise SystemExit(
                f"fit_text failed: '{name}' does not fit inside '{container}' at minimum {MIN_PT:.1f} pt. "
                f"Text: {text!r}"
            )
        artist.set_fontsize(fs)


def add_rect(ax, reg: Registry, name: str, x: float, y: float, w: float, h: float,
             *, edge=COL["border"], face="white", lw=0.7, ls="-", parent=None, overlap_group=None, zorder=2):
    r = Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=lw, linestyle=ls, zorder=zorder)
    ax.add_patch(r)
    reg.add_box(name, x, y, w, h, parent=parent, overlap_group=overlap_group)
    return r


def add_round_rect(ax, reg: Registry, name: str, x: float, y: float, w: float, h: float,
                   *, edge=COL["border"], face="white", lw=0.7, radius=1.4, parent=None, zorder=2):
    r = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=zorder,
    )
    ax.add_patch(r)
    reg.add_box(name, x, y, w, h, parent=parent)
    return r


def draw_orthogonal_arrow(ax, reg: Registry, name: str, points: Sequence[Tuple[float, float]],
                          *, color, lw=0.85, ls="-", origin: str, destination: str, extra_allowed_boxes=()):
    """Draw only horizontal/vertical segments; arrowhead appears on final straight segment."""
    pts = list(points)
    for a, b in zip(pts[:-1], pts[1:]):
        if not (math.isclose(a[0], b[0], abs_tol=1e-9) or math.isclose(a[1], b[1], abs_tol=1e-9)):
            raise SystemExit(f"Non-orthogonal connector requested: {name}: {a} -> {b}")

    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.add_line(Line2D([a[0], b[0]], [a[1], b[1]], color=color, linewidth=lw, linestyle=ls, zorder=5))

    a, b = pts[-2], pts[-1]
    arrow = FancyArrowPatch(
        a, b,
        arrowstyle="-|>",
        mutation_scale=7.5,
        linewidth=lw,
        linestyle=ls,
        color=color,
        shrinkA=0,
        shrinkB=0,
        zorder=6,
    )
    ax.add_patch(arrow)
    reg.add_arrow(name, pts, origin, destination, extra_allowed_boxes=extra_allowed_boxes)
    return arrow

# -----------------------------------------------------------------------------
# 4. Content-driven block heights
# -----------------------------------------------------------------------------

def block_heights_mm() -> Dict[int, float]:
    """Required rule: header + content lines × line spacing + 2 × 2 mm margin."""
    # Additional structural rows are accounted for explicitly (chips, pipeline headers, formula, footer).
    return {
        1: HEADER_H_MM + 2 * BODY_MARGIN_MM + 2 * LINE_H_MM + CHIP_H_MM,
        2: HEADER_H_MM + 2 * BODY_MARGIN_MM + PIPE_HEAD_H_MM + 3 * LINE_H_MM,
        3: HEADER_H_MM + 2 * BODY_MARGIN_MM + PIPE_HEAD_H_MM + 3 * LINE_H_MM + FOOT_LINE_H_MM + 0.9,
        4: HEADER_H_MM + 2 * BODY_MARGIN_MM + FORMULA_H_MM + PIPE_HEAD_H_MM + 2 * LINE_H_MM + FOOT_LINE_H_MM + 2.3,
        5: HEADER_H_MM + 2 * BODY_MARGIN_MM + 3 * BULLET_LINE_H_MM,
        6: HEADER_H_MM + 2 * BODY_MARGIN_MM + 3 * BULLET_LINE_H_MM + 1.5,
        7: HEADER_H_MM + 2 * BODY_MARGIN_MM + 3.5 + 3 * LINE_H_MM,
        8: HEADER_H_MM + 2 * BODY_MARGIN_MM + 2 * BULLET_LINE_H_MM,
    }

# -----------------------------------------------------------------------------
# 5. Figure components
# -----------------------------------------------------------------------------

def draw_source_box(fig, ax, reg, name, x, y, w, h, text, edge):
    add_rect(ax, reg, name, x, y, w, h, edge=edge, face="white", lw=0.8)
    fit_text(fig, ax, reg, name+":text", name, text, x+w/2, y+h/2,
             fontsize=6.5, weight="normal", ha="center", va="center", stretch="normal", xscale=0.76, margin_mm_required=1.0, linespacing=1.0)


def draw_step_shell(fig, ax, reg, step, x, y, w, h, title, header_fill):
    cid = f"step{step}"
    add_rect(ax, reg, cid, x, y, w, h, edge=COL["border"], face="white", lw=0.75)
    add_rect(ax, reg, cid+":header", x, y, w, HEADER_H_MM, edge=COL["border"], face=header_fill,
             lw=0.65, parent=cid)
    fit_text(fig, ax, reg, cid+":title", cid+":header", title, x+w/2, y+HEADER_H_MM/2,
             fontsize=HEADER_PT, weight="bold", ha="center", va="center", margin_mm_required=1.0)
    return cid


def draw_pipeline_cells(fig, ax, reg, parent, x, y, w, h, contents, *, dashed_mask_index=None):
    gap = 1.3
    cw = (w - 2 * gap) / 3.0
    labels = ["L1C / TOA", "Sen2Cor", "LaSRC"]
    edges = [COL["l1c"], COL["s2c"], COL["lasrc"]]
    fills = [COL["l1c_fill"], COL["s2c_fill"], COL["lasrc_fill"]]
    out = []
    for i in range(3):
        xi = x + i * (cw + gap)
        name = f"{parent}:pipe{i}"
        ls = "--" if dashed_mask_index == i else "-"
        edge = COL["mask"] if dashed_mask_index == i else "#c7ccd0"
        add_rect(ax, reg, name, xi, y, cw, h, edge=edge, face=fills[i], lw=0.65, ls=ls, parent=parent)
        add_rect(ax, reg, name+":head", xi, y, cw, PIPE_HEAD_H_MM, edge="none", face=edges[i], lw=0, parent=name)
        fit_text(fig, ax, reg, name+":label", name+":head", labels[i], xi+cw/2, y+PIPE_HEAD_H_MM/2,
                 fontsize=PIPELINE_PT, weight="bold", color="white", ha="center", va="center", margin_mm_required=0.8)
        # First body line begins >=0.8 mm below the coloured strip; full cell remains the text container.
        body_gap = 0.8
        fit_text(fig, ax, reg, name+":body", name, contents[i], xi+1.5, y+PIPE_HEAD_H_MM+body_gap,
                 fontsize=PIPELINE_PT, ha="left", va="top", margin_mm_required=1.0, linespacing=1.15)
        out.append((name, (xi, y, xi+cw, y+h)))
    return out


def draw_two_columns(fig, ax, reg, parent, x, y, w, h, left_text, right_text):
    gap = 3.0
    cw = (w-gap)/2
    boxes = []
    for idx, (xi, text) in enumerate([(x, left_text), (x+cw+gap, right_text)]):
        name = f"{parent}:col{idx}"
        add_rect(ax, reg, name, xi, y, cw, h, edge="none", face="white", lw=0, parent=parent)
        fit_text(fig, ax, reg, name+":text", name, text, xi+1.6, y+1.0, fontsize=BODY_PT,
                 ha="left", va="top", margin_mm_required=1.0, linespacing=1.15)
        boxes.append(name)
    return boxes


def draw_product(fig, ax, reg, step, block_y, block_h, count, title, body):
    # Fixed stack height chosen so that the mandated x=21 mm next-step elbow stops before the next product.
    front_h = 8.8 if step == 8 else 10.8
    x, w = X_PRODUCT_0 + 1.5, X_PRODUCT_1 - X_PRODUCT_0 - 4.0
    y = block_y + (block_h - front_h) / 2.0
    stack_group = f"product{step}:stack"
    # Back sheets intentionally overlap; declared via overlap_group.
    add_rect(ax, reg, f"product{step}:back2", x+1.6, y+1.6, w, front_h, edge="#c5cbd0", face="#fafbfc",
             lw=0.55, overlap_group=stack_group)
    add_rect(ax, reg, f"product{step}:back1", x+0.8, y+0.8, w, front_h, edge="#b7bec4", face="#f7f9fa",
             lw=0.55, overlap_group=stack_group)
    front = f"product{step}"
    add_rect(ax, reg, front, x, y, w, front_h, edge="#7d858c", face=COL["product"], lw=0.75, overlap_group=stack_group)
    title_y = y + (1.4 if step == 8 else 2.0)
    body_y = y + (4.2 if step == 8 else 5.2)
    fit_text(fig, ax, reg, front+":title", front, title, x+w/2, title_y,
             fontsize=PRODUCT_TITLE_PT, weight="bold", ha="center", va="top", margin_mm_required=1.0)
    fit_text(fig, ax, reg, front+":body", front, body, x+1.5, body_y,
             fontsize=PRODUCT_BODY_PT, ha="left", va="top", margin_mm_required=1.0, linespacing=1.0)

    # Reference-style cardinality: number and slash are one compact label, attached to the stack corner.
    count_container = f"product{step}:countzone"
    add_rect(ax, reg, count_container, X_PRODUCT_0, y-3.2, 20.0, 3.0, edge="none", face="none", lw=0)
    fit_text(fig, ax, reg, count_container+":text", count_container, f"{count} /",
             X_PRODUCT_0+0.5, y-1.7, fontsize=CARDINALITY_PT, weight="bold", ha="left", va="center", margin_mm_required=0.1)
    return front, (x, y, x+w, y+front_h)


def draw_tool_bar(fig, ax, reg, idx, label, first_step, last_step, steps, color):
    x = X_TOOLS_0 + idx * (TOOL_BAR_W_MM + TOOL_BAR_GAP_MM)
    y0 = steps[first_step][0]
    y1 = steps[last_step][0] + steps[last_step][1]
    name = f"tool{idx}"
    add_round_rect(ax, reg, name, x, y0, TOOL_BAR_W_MM, y1-y0, edge=color, face=color, lw=0.55, radius=1.2)
    fit_text(fig, ax, reg, name+":text", name, label, x+TOOL_BAR_W_MM/2, (y0+y1)/2,
             fontsize=TOOL_PT, weight="bold", color="white", ha="center", va="center", rotation=90,
             margin_mm_required=1.0)

# -----------------------------------------------------------------------------
# 6. Build figure
# -----------------------------------------------------------------------------

def build_figure(data: dict):
    global CANVAS_H_MM
    fig = plt.figure(figsize=(CANVAS_W_MM*MM_TO_IN, CANVAS_H_MM*MM_TO_IN), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, CANVAS_W_MM)
    ax.set_ylim(CANVAS_H_MM, 0)  # top-down millimetre coordinates
    ax.set_aspect("equal")
    ax.axis("off")
    reg = Registry()

    # Source row: all three boxes stay inside the spine width (46–132 mm), equally distributed.
    source_gap = 0.0
    source_w = ((X_SPINE_1 - X_SPINE_0) - 2 * source_gap) / 3.0
    src_specs = [
        ("source:S1", X_SPINE_0, source_w, "S1  Sentinel-2 L1C\nCopernicus Data Space", COL["l1c"]),
        ("source:S2", X_SPINE_0 + source_w + source_gap, source_w, "S2  Sentinel-2 L2A (Sen2Cor)\nGEE S2_SR_HARMONIZED", COL["s2c"]),
        ("source:S3", X_SPINE_0 + 2 * (source_w + source_gap), source_w, "S3  SCR 2019 polygons\n(restricted)\n22 farms · 141 polygons", COL["border"]),
    ]
    for name, x, w, text, edge in src_specs:
        draw_source_box(fig, ax, reg, name, x, SOURCE_Y0, w, SOURCE_Y1-SOURCE_Y0, text, edge)

    # Calculate block heights and positions.
    heights = block_heights_mm()
    start_y = SOURCE_Y1 + BLOCK_GAP_MM
    steps: Dict[int, Tuple[float, float]] = {}
    y = start_y
    for step in range(1, 9):
        steps[step] = (y, heights[step])
        y += heights[step]
        if step < 8:
            y += BLOCK_GAP_MM
    legend_y = y + 3.0
    CANVAS_H_MM = legend_y + LEGEND_H_MM + MARGIN_MM
    if CANVAS_H_MM > 240.0:
        raise SystemExit(f'Canvas would be {CANVAS_H_MM:.1f} mm (> 240 mm); shorten content')
    fig.set_size_inches(CANVAS_W_MM*MM_TO_IN, CANVAS_H_MM*MM_TO_IN)
    ax.set_ylim(CANVAS_H_MM, 0)
    print(f'Canvas height (content-driven): {CANVAS_H_MM:.1f} mm')
    if legend_y + LEGEND_H_MM + MARGIN_MM > CANVAS_H_MM:
        raise SystemExit(
            f"Computed layout exceeds canvas: legend bottom = {legend_y + LEGEND_H_MM:.2f} mm > {CANVAS_H_MM-MARGIN_MM:.2f} mm"
        )

    # --- Step 1 ----------------------------------------------------------------
    y1, h1 = steps[1]
    cid = draw_step_shell(fig, ax, reg, 1, X_SPINE_0, y1, X_SPINE_1-X_SPINE_0, h1,
                          "1. Data acquisition", COL["data_header"])
    body = cid+":body"
    add_rect(ax, reg, body, X_SPINE_0, y1+HEADER_H_MM, X_SPINE_1-X_SPINE_0, h1-HEADER_H_MM,
             edge="none", face="white", lw=0, parent=cid)
    fit_text(fig, ax, reg, body+":line1", body,
             "2019-01-01 to 2020-12-31 · tiles 18NXJ, 18NXK",
             X_SPINE_0+(X_SPINE_1-X_SPINE_0)/2, y1+HEADER_H_MM+2.2,
             fontsize=BODY_PT, ha="center", va="top", margin_mm_required=1.0)
    fit_text(fig, ax, reg, body+":line2", body,
             "Exact-date matching across pipelines",
             X_SPINE_0+(X_SPINE_1-X_SPINE_0)/2, y1+HEADER_H_MM+5.0,
             fontsize=BODY_PT, ha="center", va="top", margin_mm_required=1.0)
    chip_y = y1+h1-CHIP_H_MM-1.4
    chip_gap = 1.6
    chip_w = ((X_SPINE_1-X_SPINE_0)-4.0-2*chip_gap)/3
    for i, (lab, edge, fill) in enumerate([
        ("L1C / TOA", COL["l1c"], COL["l1c_fill"]),
        ("Sen2Cor", COL["s2c"], COL["s2c_fill"]),
        ("LaSRC", COL["lasrc"], COL["lasrc_fill"]),
    ]):
        x = X_SPINE_0+2.0+i*(chip_w+chip_gap)
        nm = f"step1:chip{i}"
        add_round_rect(ax, reg, nm, x, chip_y, chip_w, CHIP_H_MM, edge=edge, face=fill, lw=0.55, radius=0.8, parent=body)
        fit_text(fig, ax, reg, nm+":text", nm, lab, x+chip_w/2, chip_y+CHIP_H_MM/2,
                 fontsize=PIPELINE_PT, weight="bold", color=edge, ha="center", va="center", margin_mm_required=0.8)

    # --- Step 2 ----------------------------------------------------------------
    y2, h2 = steps[2]
    cid2 = draw_step_shell(fig, ax, reg, 2, X_SPINE_0, y2, X_SPINE_1-X_SPINE_0, h2,
                           "2. Radiometric and atmospheric correction", COL["process_header"])
    cell_y = y2+HEADER_H_MM+BODY_MARGIN_MM
    cell_h = h2-HEADER_H_MM-2*BODY_MARGIN_MM
    draw_pipeline_cells(fig, ax, reg, cid2, X_SPINE_0+2.0, cell_y, (X_SPINE_1-X_SPINE_0)-4.0, cell_h, [
        "TOA reflectance\nPB ≥ 04.00:\n−1000 DN offset",
        "ESA Sen2Cor L2A\narchived; not re-run",
        "LaSRC 3.2.0\n(sen2lasrc 2.0.2)\nMODIS_FUSED (217)",
    ])

    # --- Step 3 ----------------------------------------------------------------
    y3, h3 = steps[3]
    cid3 = draw_step_shell(fig, ax, reg, 3, X_SPINE_0, y3, X_SPINE_1-X_SPINE_0, h3,
                           "3. Cloud and shadow screening", COL["process_header"])
    foot_y3 = y3+h3-BODY_MARGIN_MM-FOOT_LINE_H_MM
    cell_y3 = y3+HEADER_H_MM+BODY_MARGIN_MM
    cell_h3 = foot_y3-cell_y3-0.8
    draw_pipeline_cells(fig, ax, reg, cid3, X_SPINE_0+2.0, cell_y3, (X_SPINE_1-X_SPINE_0)-4.0, cell_h3, [
        "Fmask 4.6 +\ns2cloudless",
        "SCL 1, 3, 8–11\nexcluded",
        "Fmask 4.6 +\ns2cloudless\nmasked + unmasked",
    ])
    foot3 = cid3+":foot"
    add_rect(ax, reg, foot3, X_SPINE_0+2.0, foot_y3, (X_SPINE_1-X_SPINE_0)-4.0, FOOT_LINE_H_MM,
             edge="none", face="white", lw=0, parent=cid3)
    fit_text(fig, ax, reg, foot3+":text", foot3,
             "L1C and LaSRC share one mask (sen2lasrc, 20 m $\\rightarrow$ 10 m)",
             (X_SPINE_0+X_SPINE_1)/2, foot_y3+FOOT_LINE_H_MM/2,
             fontsize=FOOT_PT, style="italic", ha="center", va="center", margin_mm_required=0.3)

    # --- Step 4 ----------------------------------------------------------------
    y4, h4 = steps[4]
    cid4 = draw_step_shell(fig, ax, reg, 4, X_SPINE_0, y4, X_SPINE_1-X_SPINE_0, h4,
                           "4. NDVI and zonal aggregation", COL["process_header"])
    formula_box = cid4+":formula"
    formula_y = y4+HEADER_H_MM+1.0
    add_rect(ax, reg, formula_box, X_SPINE_0+2.0, formula_y, (X_SPINE_1-X_SPINE_0)-4.0, FORMULA_H_MM+0.8,
             edge="none", face="white", lw=0, parent=cid4)
    fit_text(fig, ax, reg, formula_box+":text", formula_box,
             "NDVI = ($\\rho_{B8}$ − $\\rho_{B4}$) / ($\\rho_{B8}$ + $\\rho_{B4}$)",
             (X_SPINE_0+X_SPINE_1)/2, formula_y+(FORMULA_H_MM+0.8)/2,
             fontsize=BODY_PT, ha="center", va="center", margin_mm_required=0.3)
    foot_y4 = y4+h4-BODY_MARGIN_MM-FOOT_LINE_H_MM
    cell_y4 = formula_y+FORMULA_H_MM+1.0
    cell_h4 = foot_y4-cell_y4-0.7
    step4_cells = draw_pipeline_cells(fig, ax, reg, cid4, X_SPINE_0+2.0, cell_y4, (X_SPINE_1-X_SPINE_0)-4.0, cell_h4, [
        "rasterio mean\n10 m",
        "reduceRegions\nnative B8, 10 m",
        "rasterio mean\nINT16 scaled, 10 m",
    ])
    foot4 = cid4+":foot"
    add_rect(ax, reg, foot4, X_SPINE_0+2.0, foot_y4, (X_SPINE_1-X_SPINE_0)-4.0, FOOT_LINE_H_MM,
             edge="none", face="white", lw=0, parent=cid4)
    fit_text(fig, ax, reg, foot4+":text", foot4,
             "Per record: mean NDVI + valid-pixel count · polygons from S3",
             (X_SPINE_0+X_SPINE_1)/2, foot_y4+FOOT_LINE_H_MM/2,
             fontsize=FOOT_PT, style="italic", ha="center", va="center", margin_mm_required=0.3)

    # --- Step 5 ----------------------------------------------------------------
    y5, h5 = steps[5]
    cid5 = draw_step_shell(fig, ax, reg, 5, X_SPINE_0, y5, X_SPINE_1-X_SPINE_0, h5,
                           "5. Harmonization and quality control", COL["qc_header"])
    draw_two_columns(fig, ax, reg, cid5, X_SPINE_0+2.0, y5+HEADER_H_MM+1.0, (X_SPINE_1-X_SPINE_0)-4.0,
                     h5-HEADER_H_MM-2.0,
                     "• Key: farm_polygon__UT\n• Duplicates excluded\n• Exact-date inner join",
                     "• Cross-product QC (−1)\n• ≥ 10 valid px, L1C + LaSRC\n  (−12; 107 $\\rightarrow$ 106 polygons)")

    # --- Step 6 ----------------------------------------------------------------
    y6, h6 = steps[6]
    cid6 = draw_step_shell(fig, ax, reg, 6, X_SPINE_0, y6, X_SPINE_1-X_SPINE_0, h6,
                           "6. Agreement analysis", COL["analysis_header"])
    draw_two_columns(fig, ax, reg, cid6, X_SPINE_0+2.0, y6+HEADER_H_MM+1.0, (X_SPINE_1-X_SPINE_0)-4.0,
                     h6-HEADER_H_MM-2.0,
                     "• d = Y − X; RMSD; uRMSD\n• Lin's CCC; Deming ($\\lambda$ = 1)\n• LoA = $\\bar{d}$ ± 1.96 SD",
                     "• 3 pairs × 2019, 2020, pooled\n• Farm-cluster bootstrap\n  B = 2,000; seed 2026")

    # --- Step 7 ----------------------------------------------------------------
    y7, h7 = steps[7]
    cid7 = draw_step_shell(fig, ax, reg, 7, X_SPINE_0, y7, X_SPINE_1-X_SPINE_0, h7,
                           "7. Stratified analyses", COL["analysis_header"])
    gap7 = 2.0
    cw7 = ((X_SPINE_1-X_SPINE_0)-4.0-gap7)/2
    sub_y7 = y7+HEADER_H_MM+1.5
    sub_h7 = h7-HEADER_H_MM-3.0
    land_box = cid7+":land"
    mask_box = cid7+":mask"
    add_rect(ax, reg, land_box, X_SPINE_0+2.0, sub_y7, cw7, sub_h7, edge="#c8c4d0", face="#faf8fd", lw=0.6, parent=cid7)
    add_rect(ax, reg, mask_box, X_SPINE_0+2.0+cw7+gap7, sub_y7, cw7, sub_h7, edge=COL["mask"], face=COL["mask_fill"], lw=0.8, ls="--", parent=cid7)
    fit_text(fig, ax, reg, land_box+":title", land_box, "Land use", X_SPINE_0+3.5, sub_y7+1.3,
             fontsize=6.5, weight="bold", color=COL["s2c"], ha="left", va="top", margin_mm_required=1.0)
    fit_text(fig, ax, reg, land_box+":body", land_box,
             "6 SCR classes (UT2–UT8)\nUT0 excluded (−270)\nN = 1,527 · 21 farms",
             X_SPINE_0+3.5, sub_y7+4.3, fontsize=BODY_PT, ha="left", va="top", margin_mm_required=1.0, linespacing=1.0)
    mask_x0 = X_SPINE_0+2.0+cw7+gap7
    fit_text(fig, ax, reg, mask_box+":title", mask_box, "Masking sensitivity", mask_x0+1.5, sub_y7+1.3,
             fontsize=6.5, weight="bold", color=COL["mask"], ha="left", va="top", margin_mm_required=1.0)
    fit_text(fig, ax, reg, mask_box+":body", mask_box,
             "Masked vs unmasked LaSRC\nvs same Sen2Cor\nN = 2,585 (strict 2,521)",
             mask_x0+1.5, sub_y7+4.3, fontsize=BODY_PT, ha="left", va="top", margin_mm_required=1.0, linespacing=1.0)

    # --- Step 8 ----------------------------------------------------------------
    y8, h8 = steps[8]
    cid8 = draw_step_shell(fig, ax, reg, 8, X_SPINE_0, y8, X_SPINE_1-X_SPINE_0, h8,
                           "8. Acceptance and release", COL["release_header"])
    draw_two_columns(fig, ax, reg, cid8, X_SPINE_0+2.0, y8+HEADER_H_MM+1.0, (X_SPINE_1-X_SPINE_0)-4.0,
                     h8-HEADER_H_MM-2.0,
                     "• Expected counts + metrics\n• Tolerance 1e-12 $\\rightarrow$ PASS",
                     "• SHA-256 · Zenodo package\n• Restricted geometry excluded")

    # Products, centered vertically with their blocks.
    products = {}
    product_content = {
        1: ("708", "Scene records", "L1C 157 · Sen2Cor 175\nLaSRC 376 outputs"),
        2: ("3", "Reflectance stacks", "L1C · Sen2Cor · LaSRC"),
        3: ("4", "Screened reflectance", "3 masked\n+ LaSRC unmasked"),
        4: ("4", "Polygon-date tables", "L1C · Sen2Cor\nLaSRC masked + unmasked"),
        5: ("1,797", "Common set", "1,810 $\\rightarrow$ 1,809 $\\rightarrow$ 1,797\n106 polygons · 22 farms"),
        6: ("9", "Agreement table", "3 pairs × 3 periods"),
        7: ("18 + 2", "Stratified tables", "Land use · Masking"),
        8: ("3", "Canonical outputs", "+ release metadata"),
    }
    for step in range(1, 9):
        by, bh = steps[step]
        products[step] = draw_product(fig, ax, reg, step, by, bh, *product_content[step])

    # Process -> product: black, simple, single-head, at the vertical centre of the product.
    # This is intentionally separated from the grey next-step arrow that enters the next header.
    output_arrow_y = {}
    for step in range(1, 9):
        _, prect = products[step]
        yy = (prect[1] + prect[3]) / 2.0
        output_arrow_y[step] = yy
        draw_orthogonal_arrow(
            ax, reg, f"arrow:step{step}->product{step}",
            [(X_SPINE_0, yy), (prect[2], yy)],
            color=COL["black"], origin=f"step{step}", destination=f"product{step}"
        )

    # Product -> next process: grey orthogonal elbow.
    # Start down x≈21 mm only to 2 mm above the next cardinality, then move into x=42 mm gutter.
    X_ELBOW_PRODUCT = 21.0
    X_ELBOW_GUTTER = 42.0
    for step in range(1, 8):
        _, prect = products[step]
        next_y, _ = steps[step+1]
        target_y = next_y + HEADER_H_MM/2
        next_count_rect = reg.boxes[f"product{step+1}:countzone"].rect
        turn_y = next_count_rect[1] - 2.0
        start = ((prect[0]+prect[2])/2, prect[3])
        if turn_y <= start[1] + 0.5:
            raise SystemExit(f"Insufficient vertical room for next-step connector after product {step}")
        draw_orthogonal_arrow(
            ax, reg, f"arrow:product{step}->step{step+1}",
            [start, (X_ELBOW_PRODUCT, turn_y), (X_ELBOW_GUTTER, turn_y),
             (X_ELBOW_GUTTER, target_y), (X_SPINE_0, target_y)],
            color=COL["gray_arrow"], origin=f"product{step}", destination=f"step{step+1}"
        )

        # V4.1 requirement: output and incoming arrowheads on each next block must be >=3 mm apart.
        if abs(output_arrow_y[step+1] - target_y) < 3.0:
            raise SystemExit(
                f"Arrowhead separation below 3 mm at Step {step+1}: "
                f"output={output_arrow_y[step+1]:.2f}, incoming={target_y:.2f}"
            )

    # Source -> Step 1: three short straight vertical arrows, none below Step 1.
    for name, x, w, _, _ in src_specs:
        cx = x+w/2
        draw_orthogonal_arrow(
            ax, reg, f"arrow:{name}->step1", [(cx, SOURCE_Y1), (cx, y1)],
            color=COL["gray_arrow"], origin=name, destination="step1"
        )

    # Masking-sensitivity branch: leave Step-4 LaSRC, use only x=135.5 gutter,
    # approach above the Step-7 masking cell, then enter its TOP border away from text.
    lasrc_rect = step4_cells[2][1]
    branch_y0 = (lasrc_rect[1]+lasrc_rect[3])/2
    mask_rect = reg.boxes[mask_box].rect
    approach_y = mask_rect[1] - 0.8
    entry_x = mask_rect[2] - 3.0
    draw_orthogonal_arrow(
        ax, reg, "arrow:masking-branch",
        [(lasrc_rect[2], branch_y0), (X_MASK_CHANNEL, branch_y0),
         (X_MASK_CHANNEL, approach_y), (entry_x, approach_y), (entry_x, mask_rect[1])],
        color=COL["mask"], lw=0.95, ls="--", origin=step4_cells[2][0], destination=mask_box,
        extra_allowed_boxes=("step4", "step7")
    )

    # Tools: vertical text inside rounded bars, aligned exactly to covered steps.
    draw_tool_bar(fig, ax, reg, 0, "EODAG / CDSE", 1, 2, steps, COL["l1c"])
    draw_tool_bar(fig, ax, reg, 1, "Earth Engine (Python API)", 1, 4, steps, COL["s2c"])
    draw_tool_bar(fig, ax, reg, 2, "sen2lasrc / Docker · USGS M2M", 1, 3, steps, COL["lasrc"])
    draw_tool_bar(fig, ax, reg, 3, "Python: rasterio, pandas, numpy", 1, 8, steps, COL["tool_python"])

    # Legend: one row immediately below Step 8, no isolated empty lower area.
    legend_box = "legend"
    add_rect(ax, reg, legend_box, X_PRODUCT_0, legend_y, X_TOOLS_1-X_PRODUCT_0, LEGEND_H_MM,
             edge="none", face="white", lw=0)
    lx = X_PRODUCT_0+1.0
    ly = legend_y+LEGEND_H_MM/2
    entries = [(COL["l1c"], "L1C / TOA"), (COL["s2c"], "Sen2Cor"), (COL["lasrc"], "LaSRC")]
    legend_steps = [24.0, 20.0, 18.0]
    for i, (c, lab) in enumerate(entries):
        sq = f"legend:sq{i}"
        add_rect(ax, reg, sq, lx, ly-1.8, 3.6, 3.6, edge=c, face=c, lw=0.4, parent=legend_box)
        fit_text(fig, ax, reg, sq+":label", legend_box, lab, lx+4.6, ly, fontsize=6.2,
                 ha="left", va="center", margin_mm_required=0.1)
        lx += legend_steps[i]
    ax.add_line(Line2D([lx, lx+8.0], [ly, ly], color=COL["mask"], linewidth=1.0, linestyle="--", zorder=8))
    fit_text(fig, ax, reg, "legend:mask", legend_box, "Masking-sensitivity branch", lx+9.5, ly,
             fontsize=6.2, ha="left", va="center", margin_mm_required=0.1)
    lx += 45.0
    # Output symbol and next-step elbow symbol.
    ax.add_patch(FancyArrowPatch((lx, ly), (lx+7.0, ly), arrowstyle="-|>", mutation_scale=7.0,
                                 linewidth=0.8, color=COL["black"], zorder=8))
    fit_text(fig, ax, reg, "legend:output", legend_box, "output", lx+8.2, ly, fontsize=6.2,
             ha="left", va="center", margin_mm_required=0.1)
    lx += 19.0
    ax.add_line(Line2D([lx, lx, lx+6.0], [ly-2.0, ly, ly], color=COL["gray_arrow"], linewidth=0.8, zorder=8))
    ax.add_patch(FancyArrowPatch((lx+3.8, ly), (lx+6.0, ly), arrowstyle="-|>", mutation_scale=6.5,
                                 linewidth=0.8, color=COL["gray_arrow"], zorder=8))
    fit_text(fig, ax, reg, "legend:next", legend_box, "next step", lx+7.0, ly, fontsize=6.2,
             ha="left", va="center", margin_mm_required=0.1)

    return fig, ax, reg, steps, legend_y

# -----------------------------------------------------------------------------
# 7. Collision and containment QA
# -----------------------------------------------------------------------------

def sample_segment(a, b, step=0.5):
    dx, dy = b[0]-a[0], b[1]-a[1]
    dist = abs(dx)+abs(dy)
    n = max(1, int(math.ceil(dist/step)))
    for i in range(n+1):
        t = i/n
        yield (a[0]+t*dx, a[1]+t*dy)


def point_in_rect(p, r, margin=0.0):
    return r[0]-margin <= p[0] <= r[2]+margin and r[1]-margin <= p[1] <= r[3]+margin


def point_on_rect_border(p, r, tol=0.3):
    x, y = p
    within_x = r[0]-tol <= x <= r[2]+tol
    within_y = r[1]-tol <= y <= r[3]+tol
    on_left = abs(x-r[0]) <= tol and within_y
    on_right = abs(x-r[2]) <= tol and within_y
    on_top = abs(y-r[1]) <= tol and within_x
    on_bottom = abs(y-r[3]) <= tol and within_x
    return on_left or on_right or on_top or on_bottom


def _close_point(a, b, tol=1e-6):
    return abs(a[0]-b[0]) <= tol and abs(a[1]-b[1]) <= tol


def _segment_intersection(a0, a1, b0, b1, tol=1e-9):
    """Return None, ('point', (x,y)), or ('overlap', ((x0,y0),(x1,y1)))."""
    av = math.isclose(a0[0], a1[0], abs_tol=tol)
    bv = math.isclose(b0[0], b1[0], abs_tol=tol)
    if av and bv:
        if not math.isclose(a0[0], b0[0], abs_tol=tol):
            return None
        a_lo, a_hi = sorted([a0[1], a1[1]])
        b_lo, b_hi = sorted([b0[1], b1[1]])
        lo, hi = max(a_lo, b_lo), min(a_hi, b_hi)
        if hi < lo - tol:
            return None
        if abs(hi-lo) <= tol:
            return ("point", (a0[0], (lo+hi)/2))
        return ("overlap", ((a0[0], lo), (a0[0], hi)))
    if (not av) and (not bv):
        if not math.isclose(a0[1], b0[1], abs_tol=tol):
            return None
        a_lo, a_hi = sorted([a0[0], a1[0]])
        b_lo, b_hi = sorted([b0[0], b1[0]])
        lo, hi = max(a_lo, b_lo), min(a_hi, b_hi)
        if hi < lo - tol:
            return None
        if abs(hi-lo) <= tol:
            return ("point", ((lo+hi)/2, a0[1]))
        return ("overlap", ((lo, a0[1]), (hi, a0[1])))

    # one vertical, one horizontal
    if av:
        v0, v1, h0, h1 = a0, a1, b0, b1
    else:
        v0, v1, h0, h1 = b0, b1, a0, a1
    vx = v0[0]
    hy = h0[1]
    v_lo, v_hi = sorted([v0[1], v1[1]])
    h_lo, h_hi = sorted([h0[0], h1[0]])
    if h_lo-tol <= vx <= h_hi+tol and v_lo-tol <= hy <= v_hi+tol:
        return ("point", (vx, hy))
    return None


def _arrow_endpoint_for_node(ar: ArrowRec, node: str):
    if ar.destination == node:
        return ar.points[-1]
    if ar.origin == node:
        return ar.points[0]
    return None


def collision_check(fig, ax, reg: Registry, legend_y: float) -> List[str]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    conflicts: List[str] = []

    # 1. Every text inside declared container with >= requested margin.
    text_bboxes_data = {}
    text_by_name = {}
    for tr in reg.texts:
        bb = tr.artist.get_window_extent(renderer=renderer)
        inv = ax.transData.inverted()
        (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        tb = (min(x0,x1), min(y0,y1), max(x0,x1), max(y0,y1))
        text_bboxes_data[tr.name] = tb
        text_by_name[tr.name] = tr
        cb = reg.boxes[tr.container].rect
        inset = (cb[0]+tr.margin_mm, cb[1]+tr.margin_mm, cb[2]-tr.margin_mm, cb[3]-tr.margin_mm)
        if not rect_contains(inset, tb, eps=0.05):
            conflicts.append(f"TEXT OUTSIDE: {tr.name} not inside {tr.container} with {tr.margin_mm:.1f} mm margin")

    # 2. No pair of texts intersects.
    names = list(text_bboxes_data)
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a, b = names[i], names[j]
            if rect_intersection(text_bboxes_data[a], text_bboxes_data[b]) is not None:
                conflicts.append(f"TEXT INTERSECTION: {a} <> {b}")

    # 3. No unintended box intersections.
    box_names = list(reg.boxes)
    for i in range(len(box_names)):
        for j in range(i+1, len(box_names)):
            a, b = reg.boxes[box_names[i]], reg.boxes[box_names[j]]
            inter = rect_intersection(a.rect, b.rect)
            if inter is None:
                continue
            if is_ancestor(reg, a.name, b.name) or is_ancestor(reg, b.name, a.name):
                continue
            if a.overlap_group and a.overlap_group == b.overlap_group:
                continue
            if a.name.endswith(":countzone") or b.name.endswith(":countzone"):
                continue
            conflicts.append(f"BOX INTERSECTION: {a.name} <> {b.name}")

    # 4a. Every arrow endpoint must land on the DESTINATION border (tolerance 0.3 mm).
    for ar in reg.arrows:
        if ar.destination not in reg.boxes:
            conflicts.append(f"ARROW DESTINATION UNKNOWN: {ar.name} -> {ar.destination}")
            continue
        endpoint = ar.points[-1]
        dest_rect = reg.boxes[ar.destination].rect
        if not point_on_rect_border(endpoint, dest_rect, tol=0.3):
            conflicts.append(
                f"ARROW ENDPOINT OFF BORDER: {ar.name} ends at {endpoint[0]:.2f},{endpoint[1]:.2f} mm; "
                f"destination={ar.destination} rect={dest_rect}"
            )

    # 4b. Arrow path does not cross unrelated boxes or ANY text, including destination text.
    for ar in reg.arrows:
        allowed_boxes = {ar.origin, ar.destination, *ar.extra_allowed_boxes}
        # Parent boxes of explicitly allowed endpoints may be traversed only where geometry requires it.
        for node in list(allowed_boxes):
            cur = reg.boxes.get(node)
            while cur and cur.parent:
                allowed_boxes.add(cur.parent)
                cur = reg.boxes.get(cur.parent)

        all_points = []
        for p0, p1 in zip(ar.points[:-1], ar.points[1:]):
            pts = list(sample_segment(p0, p1, 0.5))
            if all_points and pts and _close_point(all_points[-1], pts[0]):
                pts = pts[1:]
            all_points.extend(pts)

        for p in all_points:
            for bn, br in reg.boxes.items():
                if bn in allowed_boxes:
                    continue
                if br.overlap_group and (
                    ar.origin.startswith(br.overlap_group.split(':')[0]) or
                    ar.destination.startswith(br.overlap_group.split(':')[0])
                ):
                    continue
                if point_in_rect(p, br.rect, margin=-0.10):
                    conflicts.append(f"ARROW-BOX: {ar.name} crosses {bn} at {p[0]:.1f},{p[1]:.1f} mm")
                    break
            else:
                continue
            break

        # No destination-text exemption: touching a destination border is allowed, entering its text is not.
        for p in all_points:
            hit = False
            for tn, tb in text_bboxes_data.items():
                if point_in_rect(p, tb, margin=0.05):
                    conflicts.append(f"ARROW-TEXT: {ar.name} crosses {tn} at {p[0]:.1f},{p[1]:.1f} mm")
                    hit = True
                    break
            if hit:
                break

    # 4c. Arrow-arrow: no shared segment and no crossing, except a declared shared node endpoint.
    for i in range(len(reg.arrows)):
        a = reg.arrows[i]
        for j in range(i+1, len(reg.arrows)):
            b = reg.arrows[j]
            shared_nodes = {a.origin, a.destination}.intersection({b.origin, b.destination})
            for a0, a1 in zip(a.points[:-1], a.points[1:]):
                for b0, b1 in zip(b.points[:-1], b.points[1:]):
                    inter = _segment_intersection(a0, a1, b0, b1)
                    if inter is None:
                        continue
                    kind, geom = inter
                    if kind == "overlap":
                        conflicts.append(f"ARROW-ARROW OVERLAP: {a.name} <> {b.name}: {geom}")
                        continue
                    p = geom
                    allowed = False
                    for node in shared_nodes:
                        ea = _arrow_endpoint_for_node(a, node)
                        eb = _arrow_endpoint_for_node(b, node)
                        if ea is not None and eb is not None and _close_point(p, ea, 1e-5) and _close_point(p, eb, 1e-5):
                            allowed = True
                            break
                    if not allowed:
                        conflicts.append(f"ARROW-ARROW CROSSING: {a.name} <> {b.name} at {p[0]:.2f},{p[1]:.2f} mm")

    # 4d. Arrowhead separation: >=2 mm for every pair of final endpoints.
    for i in range(len(reg.arrows)):
        a = reg.arrows[i]
        pa = a.points[-1]
        for j in range(i+1, len(reg.arrows)):
            b = reg.arrows[j]
            pb = b.points[-1]
            d = math.hypot(pa[0]-pb[0], pa[1]-pb[1])
            if d < 2.0 - 1e-6:
                conflicts.append(f"ARROWHEAD SEPARATION: {a.name} <> {b.name}: {d:.2f} mm < 2 mm")

    # 5. Everything within canvas, and empty space below legend <= 5 mm.
    for bn, br in reg.boxes.items():
        r = br.rect
        if r[0] < -1e-6 or r[1] < -1e-6 or r[2] > CANVAS_W_MM+1e-6 or r[3] > CANVAS_H_MM+1e-6:
            conflicts.append(f"OUTSIDE CANVAS: {bn} {r}")
    legend_bottom = legend_y + LEGEND_H_MM
    empty_bottom = CANVAS_H_MM - legend_bottom
    if empty_bottom > 5.0 + 1e-6:
        conflicts.append(f"BOTTOM EMPTY SPACE: {empty_bottom:.2f} mm > 5 mm")

    # Deduplicate while preserving order.
    seen = set(); unique = []
    for c in conflicts:
        if c not in seen:
            unique.append(c); seen.add(c)
    return unique

# -----------------------------------------------------------------------------
# 8. Provenance and CLI
# -----------------------------------------------------------------------------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_provenance(root: Path, command: str, outputs: Sequence[Path]):
    out = root / "docs/provenance/figure_02_processing_workflow.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Figure 2 processing-workflow provenance",
        "",
        "## Command",
        "```bash",
        command,
        "```",
        "",
        f"Python: {platform.python_version()}",
        "",
        f"Matplotlib: {mpl.__version__}",
        "",
        f"L1C_FMASK_TOOL: `{L1C_FMASK_TOOL}`",
        "",
        "## Output SHA-256",
    ]
    for p in outputs:
        lines.append(f"- `{p.name}`: `{sha256(p)}`")
    out.write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None, help="Project root. Defaults to the repository root inferred from this script path.")
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--svg", required=True)
    parser.add_argument("--png", required=True)
    args = parser.parse_args()

    script = Path(__file__).resolve()
    root = Path(args.root).resolve() if args.root else script.parents[2]
    pdf, svg, png = map(lambda s: Path(s).resolve(), [args.pdf, args.svg, args.png])
    for p in [pdf, svg, png]:
        p.parent.mkdir(parents=True, exist_ok=True)

    data = load_canonical(root)

    print("Canonical figures read from REAL project source files:")
    print(f"  Common set: N = {data['common_n']:,} | polygons = {data['common_polygons']} | farms = {data['common_farms']}")
    print(f"  Record chain: {data['before_exclusion_n']:,} -> {data['after_exclusion_n']:,} -> {data['common_n']:,}")
    print(f"    documented exclusions = {data['documented_exclusions']} | low-pixel exclusions = {data['low_pixel_n']}")
    print(f"  Input-scene manifest: {data['scene_total']} records")
    print(f"    L1C = {data['scene_l1c']} scenes")
    print(f"    Sen2Cor = {data['scene_s2c']} dates")
    print(f"    LaSRC = {data['scene_lasrc']} outputs")
    print(f"  MODIS_FUSED auxiliaries = {data['aux_n']}")
    print(f"  Land-use set: N = {data['land_n']:,} | farms = {data['land_farms']}")
    print(f"  Masking set: N = {data['masking_n']:,} | polygons = {data['masking_polygons']} | farms = {data['masking_farms']} | strict = {data['masking_strict_n']:,}")
    print(f"  Geometry source: {data['geom_farms']} farms | {data['geom_polygons']} polygons")
    print(f"  Geometry analytical chain: {data['inner_join_keys']} -> {data['final_keys']} polygons")
    print(f"  Acceptance status: {data['acceptance_status']}")
    print(f"  L1C_FMASK_TOOL = {L1C_FMASK_TOOL}")

    heights = block_heights_mm()
    print("Calculated block heights (mm):")
    for step in range(1, 9):
        print(f"  Step {step}: {heights[step]:.2f}")

    fig, ax, reg, steps, legend_y = build_figure(data)
    conflicts = collision_check(fig, ax, reg, legend_y)
    print(f"Collision check: {len(conflicts)} conflicts")
    if conflicts:
        for c in conflicts:
            print("  -", c)
        plt.close(fig)
        raise SystemExit("Collision check failed; exports not written.")

    fig.savefig(pdf, format="pdf", dpi=600, facecolor="white")
    fig.savefig(svg, format="svg", dpi=600, facecolor="white")
    fig.savefig(png, format="png", dpi=600, facecolor="white")
    plt.close(fig)

    command = (
        "python3 code/figures/figure_02_processing_workflow.py "
        f"--pdf {pdf.name} --svg {svg.name} --png {png.name}"
    )
    write_provenance(root, command, [pdf, svg, png])

    print("Outputs written:")
    print(" ", pdf)
    print(" ", svg)
    print(" ", png)


if __name__ == "__main__":
    main()
