# Reproducibility guide

Version 1.0.0.

## Scope

The public repository reproduces manuscript rendering, final figures, and reported summary tables from the canonical derived outputs. Full reprocessing from raw satellite products additionally requires external services, governed geometries, and local caches that are not redistributed.

## Recommended workflow

1. Copy `config/project_config.example.yml` to `project_config.local.yml` and set local paths.
2. Run `bash environment/capture_environment.sh` inside the analysis container.
3. Verify `data/derived/` against the release checksums.
4. Generate publication figures with the scripts in `code/figures/` (`figure_01` … `figure_05`; `fig2_preflight.py` checks Figure 2 without writing files).
5. Render the manuscript from `manuscript/source/` with `render_ARTICLE_NDVI_V14_A3_MDPI.sh`.
6. Verify file integrity with `sha256sum -c release/SHA256SUMS.txt`.
7. Build Zenodo packages with `python3 code/utilities/build_zenodo_packages.py --root . --version 1.0.0`.

## Non-public prerequisites

- SCR management-polygon and farm-boundary geometries (restricted).
- Local Sentinel-2 L1C and LaSRC-derived raster products.
- Google Earth Engine access to `COPERNICUS/S2_SR_HARMONIZED` (Python API).
- USGS M2M access for MODIS_FUSED auxiliary data.
- The `sen2lasrc` 2.0.2 environment: the LaSRC/Fmask Docker image pinned by digest (`environment/container-provenance.txt`), driven from a conda environment named `sen2lasrc` under WSL2.

## Determinism

Farm-cluster bootstrap with B = 2,000 replicates and seed 2026; percentile intervals. Software versions, container image digest, input-scene and auxiliary-data manifests, and SHA-256 checksums are recorded in `environment/`, `data/metadata/` and `release/`.
