#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Patch the V12 QMD to use the corrected V13 figures and safer float positions."""

from __future__ import annotations

import re
import shutil
import sys
from datetime import datetime
from pathlib import Path


if len(sys.argv) != 2:
    raise SystemExit(
        "Usage: python3 patch_qmd_figures_v13.py ARTICLE_NDVI_V12_A3_MDPI.qmd"
    )

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"QMD not found: {path}")

backup = path.with_name(
    path.name + ".bak_plots_v13_" + datetime.now().strftime("%Y%m%d_%H%M%S")
)
shutil.copy2(path, backup)

text = path.read_text(encoding="utf-8")

replacements = {
    "figures/fig3_ndvi_three_pipelines_v12.png":
        "figures/fig3_ndvi_three_pipelines_v13.png",
    "figures/fig4_agreement_3x2_v12.png":
        "figures/fig4_agreement_v13.png",
    "figures/fig5_gradient_2plus1_v12.png":
        "figures/fig5_gradient_v13.png",
}

for old, new in replacements.items():
    if old not in text:
        print(f"Warning: source path not found in QMD: {old}")
    text = text.replace(old, new)

# Use flexible float placement to avoid the blank pages produced by fig-pos="H".
for figure_id in ("fig-method-workflow", "fig-three-pipelines",
                  "fig-agreement-3vias", "fig-gradiente-ut"):
    pattern = rf'(\{{#{re.escape(figure_id)}[^}}]*?)fig-pos="[^"]+"'
    text, count = re.subn(pattern, rf'\1fig-pos="htbp"', text)
    if count == 0:
        print(f"Warning: could not patch float position for {figure_id}")

# Figure-specific widths.
width_map = {
    "fig-three-pipelines": "96%",
    "fig-agreement-3vias": "100%",
    "fig-gradiente-ut": "100%",
}
for figure_id, width in width_map.items():
    pattern = rf'(\{{#{re.escape(figure_id)}[^}}]*?)width="[^"]+"'
    text, count = re.subn(pattern, rf'\1width="{width}"', text)
    if count == 0:
        print(f"Warning: could not patch width for {figure_id}")

path.write_text(text, encoding="utf-8")

print(f"Patched QMD: {path}")
print(f"Backup: {backup}")
