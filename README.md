# Category-Specific Sentinel-2 NDVI Differences among Level-1C, Sen2Cor, and LaSRC in a Tropical Silvopastoral Landscape: Open Science Reproducibility Compendium

**Version:** 1.0.1 · **Author:** Heiber Alexander Pantevez Villamil ([ORCID 0000-0003-1797-0057](https://orcid.org/0000-0003-1797-0057)) · **DOI (all versions):** [10.5281/zenodo.24089625](https://doi.org/10.5281/zenodo.24089625)

This repository contains the manuscript source, analysis and figure code, derived summary data, figures, metadata, provenance and reproducibility documentation supporting the comparison of Sentinel-2 NDVI among Level-1C/TOA, Sen2Cor L2A/SR and LaSRC L2A/SR.

## Scientific scope

The study evaluates processor-associated NDVI differences for 2019–2020 across **106 land-use management polygons on 22 farms** (854 ha) in the Colombian Orinoco Piedmont, Department of Meta. The main common dataset contains **1,797 polygon-by-date observations** (2019: 668; 2020: 1,129). Stratified analyses use 1,527 observations for the six SCR land-use categories and 2,585 paired records for the LaSRC masking sensitivity. The package is a reproducibility compendium for a method comparison; it is not a ground-truth validation dataset.

Processing pipelines:
- **L1C/TOA:** Copernicus Data Space via EODAG 3.10.2 (1000-DN offset removed for processing baseline ≥ 04.00).
- **Sen2Cor L2A/SR:** Google Earth Engine `COPERNICUS/S2_SR_HARMONIZED`, accessed through the Earth Engine Python API; SCL classes 1, 3, 8–11 excluded.
- **LaSRC L2A/SR:** `sen2lasrc` 2.0.2 (LaSRC 3.2.0) in a digest-pinned Docker image; MODIS_FUSED auxiliary data from the USGS M2M API.
- L1C and LaSRC share one Fmask 4.6 + s2cloudless mask.

## Repository map

- `manuscript/` — source (`.qmd`), journal-template dependencies, final figures and rendered drafts.
- `code/` — analysis, figure-generation, render and utility scripts.
- `data/derived/` — public summary tables (canonical outputs, SHA-256 in the release manifest).
- `data/metadata/` — data dictionary, input-scene and auxiliary-data manifests, exclusion ledger.
- `data/external/` — external sources referenced but not redistributed.
- `data/restricted/` — documentation of governed data that are excluded.
- `docs/` — reproducibility, provenance, data-governance and Zenodo guidance.
- `environment/` — environment capture scripts and container provenance (full Docker image digest).
- `metadata/`, `provenance/` — creators, record relationships and provenance records.
- `release/` — validation reports, manifests and checksums.

`local_only/` (private source documents and working archives) is excluded from all public packages.

## Access and privacy

Farm-boundary and management-polygon geometries belong to the Sustainable Cattle Ranching project (*Proyecto Ganadería Colombiana Sostenible*) and are not publicly distributed. Sentinel-2 scenes are not redistributed; they are identified in `data/metadata/` and available from the Copernicus Data Space Ecosystem and Google Earth Engine.

## Citation

Citation metadata are in `CITATION.cff`. Please cite the Zenodo record of this version and the associated article.

## Licensing

- Code: MIT (`LICENSES/LICENSE-CODE-MIT.txt`)
- Text, figures and derived data: CC BY 4.0 (`LICENSES/LICENSE-DATA-DOCS-CC-BY-4.0.md`)
- Metadata: CC0 1.0 (`LICENSES/LICENSE-METADATA-CC0-1.0.txt`)

## Use of generative AI

Generative AI assistants (Claude, Anthropic; ChatGPT, OpenAI) were used between January and October 2026, as described in the manuscript. The author ran all code, verified every reported result against the canonical outputs, and takes full responsibility for the content.

## Reproduction

```bash
bash environment/capture_environment.sh
sha256sum -c release/SHA256SUMS.txt
python3 code/utilities/build_zenodo_packages.py --root . --version 1.0.1
```
