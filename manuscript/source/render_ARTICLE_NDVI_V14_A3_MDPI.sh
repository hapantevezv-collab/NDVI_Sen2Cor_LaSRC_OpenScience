#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QMD="ARTICLE_NDVI_V14_A3_MDPI.qmd"
OUT="_render_v14"

cd "$ROOT"

required=(
  "$QMD"
  "mdpi-quarto-template-v13.tex"
  "references-v12.bib"
  "figures/figure_01_study_area.png"
  "figures/figure_02_processing_workflow.pdf"
  "figures/figure_03_ndvi_comparison.png"
  "figures/figure_04_agreement_diagnostics.pdf"
  "figures/figure_05_land_use_gradient.pdf"
)

missing=0

echo "=== V14 manuscript input gate ==="

for file in "${required[@]}"; do
    if [[ -s "$file" ]]; then
        printf 'OK:      %s\n' "$file"
    else
        printf 'MISSING: %s\n' "$file" >&2
        missing=1
    fi
done

if [[ "$missing" -ne 0 ]]; then
    echo
    echo "Render not started: final manuscript inputs are incomplete." >&2
    exit 1
fi

rm -rf "$OUT"
mkdir -p "$OUT"

quarto render "$QMD" --output-dir "$OUT"

echo
echo "=== V14 render products ==="

find "$OUT"   -maxdepth 2   -type f   -printf '%p\t%k KB\n' |
sort
