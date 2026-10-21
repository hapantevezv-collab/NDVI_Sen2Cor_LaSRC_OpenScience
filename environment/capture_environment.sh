#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/environment"
mkdir -p "$OUT"

{
  echo "captured_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "kernel=$(uname -a)"
  echo "os_release:"
  cat /etc/os-release 2>/dev/null || true
} > "$OUT/system-info.txt"

{
  echo "python=$(python3 --version 2>&1 || true)"
  echo "quarto=$(quarto --version 2>&1 || true)"
  echo "pdflatex=$(pdflatex --version 2>&1 | head -1 || true)"
  echo "r=$(R --version 2>&1 | head -1 || true)"
  echo "gdal=$(gdalinfo --version 2>&1 || true)"
  echo "docker=$(docker --version 2>&1 || true)"
} > "$OUT/tool-versions.txt"

python3 -m pip freeze > "$OUT/requirements-freeze.txt" 2>/dev/null || true
conda list --explicit > "$OUT/conda-explicit.txt" 2>/dev/null || true
Rscript -e 'writeLines(capture.output(sessionInfo()))' > "$OUT/R-sessionInfo.txt" 2>/dev/null || true

{
  echo "container_id=$(hostname)"
  echo "image_digest_or_tag=ADD_FROM_DOCKER_HOST_OR_REGISTRY"
  echo "sen2lasrc_version=2.0.2"
  echo "lasrc_version=3.2.0"
  echo "eodag_version=3.10.2"
} > "$OUT/container-provenance.txt"

echo "Environment captured in $OUT"
