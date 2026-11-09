#!/usr/bin/env bash
# Render the Elsevier (RSASE) version of article A3.
set -euo pipefail
cd "$(dirname "$0")"
QMD="ARTICLE_NDVI_V15_A3_RSASE.qmd"
quarto render "$QMD" --to elsevier-pdf --output-dir _render_v15
echo "PDF: $(pwd)/_render_v15/${QMD%.qmd}.pdf"
