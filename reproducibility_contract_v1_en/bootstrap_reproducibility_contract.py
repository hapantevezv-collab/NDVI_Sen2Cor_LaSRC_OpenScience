#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Initialize a non-public draft reproducibility contract for the Sentinel-2
L1C/Sen2Cor/LaSRC NDVI study.

The script is non-destructive and writes only under:
    local_only/workflow_contract_draft/

All generated filenames, labels, headers, and documentation are in English.
Unknown values are left blank rather than fabricated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


KNOWN_SCRIPTS = [
    ("acquisition", "L1C inventory and extraction",
     "code/analysis/extract_l1c_ndvi_2019_2020.py"),
    ("processing", "LaSRC processing",
     "code/analysis/run_lasrc_processing_2016_2020.py"),
    ("analysis", "Agreement statistics",
     "code/analysis/compute_agreement_statistics.py"),
    ("analysis", "Land-use category analysis",
     "code/analysis/compute_land_use_gradient.py"),
    ("visualization", "Study-area figure",
     "code/figures/figure_01_study_area.py"),
    ("visualization", "Processing workflow figure",
     "code/figures/figure_02_processing_workflow.py"),
    ("visualization", "NDVI comparison figure",
     "code/figures/figure_03_ndvi_comparison.py"),
    ("visualization", "Agreement diagnostics figure",
     "code/figures/figure_04_agreement_diagnostics.py"),
    ("visualization", "Land-use gradient figure",
     "code/figures/figure_05_land_use_gradient.py"),
    ("release", "Zenodo package builder",
     "code/utilities/build_zenodo_packages.py"),
]

PUBLIC_DERIVED_TABLES = [
    "data/derived/agreement_three_pipelines_2019_2020.csv",
    "data/derived/agreement_sen2cor_lasrc_2019_2020.csv",
    "data/derived/land_use_gradient_2019_2020.csv",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for row in reader if any(cell.strip() for cell in row))


def csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader, [])


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_if_missing(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Open Science repository root")
    parser.add_argument(
        "--source-workspace",
        default="",
        help="Optional source workspace used only for candidate discovery",
    )
    parser.add_argument(
        "--draft-dir",
        default="local_only/workflow_contract_draft",
        help="Draft contract location relative to the repository root",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not (root / "manuscript").exists():
        raise SystemExit(f"Repository root not recognized: {root}")

    draft = root / args.draft_dir
    config_dir = draft / "config"
    metadata_dir = draft / "data" / "metadata"
    contract_dir = draft / "metadata"
    acceptance_dir = draft / "acceptance"
    docs_dir = draft / "docs" / "provenance"

    for directory in (
        config_dir, metadata_dir, contract_dir, acceptance_dir, docs_dir
    ):
        directory.mkdir(parents=True, exist_ok=True)

    workflow_config = """# Reproducibility contract — draft
# Public labels and documentation are in English.
# Unknown operational values remain blank until verified.

project:
  short_name: "NDVI Sen2Cor-LaSRC comparison"
  repository_version: "0.9.1"
  study_period:
    start_date: "2019-01-01"
    end_date: "2020-12-31"
  mgrs_tiles:
    - "18NXJ"
    - "18NXK"
  farms: 22
  management_polygons: 91
  target_resolution_m: 10

spectral_design:
  red_band: "B4"
  near_infrared_band: "B8"
  ndvi_formula: "(B8 - B4) / (B8 + B4)"
  no_data_policy: "Remove product no-data values before NDVI calculation."

l1c_pipeline:
  source: "Copernicus Data Space Ecosystem"
  client: "EODAG"
  client_version: "3.10.2"
  maximum_cloud_cover_percent: 20
  processing_baseline_offset:
    applies_from: "04.00"
    digital_number_offset_removed: 1000
  screening:
    fmask_version: "4.6"
    s2cloudless: true
  zonal_aggregation: "Local raster masking and polygon statistics"

sen2cor_pipeline:
  source_collection: "COPERNICUS/S2_SR_HARMONIZED"
  platform: "Google Earth Engine"
  screening:
    method: "Scene Classification Layer"
    excluded_classes:
      - 0
      - 1
      - 3
      - 8
      - 9
      - 10
      - 11
  grid: "Native B8 projection at 10 m"
  zonal_aggregation: "reduceRegions"

lasrc_pipeline:
  sen2lasrc_version: "2.0.2"
  lasrc_version: "3.2.0"
  auxiliary_data_service: "USGS Machine-to-Machine API"
  screening:
    fmask_version: "4.6"
    s2cloudless: true
  zonal_aggregation: "Local raster masking and polygon statistics"

record_definition:
  key_fields:
    - "acquisition_date"
    - "polygon_identifier"
  local_minimum_valid_pixels: 10
  common_join: "Exact-date and polygon-key inner join"

analysis_sets:
  common_three_pipeline_records: 1809
  land_use_records: 1528
  land_use_polygons: 50
  masking_sensitivity_records: 2585

statistical_analysis:
  metrics:
    - "mean paired difference"
    - "RMSD"
    - "uRMSD"
    - "Lin CCC"
    - "Deming regression"
    - "Bland-Altman limits of agreement"
  deming_error_variance_ratio: 1
  bland_altman_multiplier: 1.96
  bootstrap:
    unit: "polygon"
    replicates: 2000
    random_seed:
    interval_method:
  missing_value_policy:
  software_implementation:

access:
  farm_geometry: "restricted"
  public_derived_tables: true
  public_analysis_code: true
"""
    write_if_missing(config_dir / "workflow_config.yml", workflow_config)

    scene_fields = [
        "pipeline",
        "product_id",
        "system_index",
        "acquisition_datetime_utc",
        "mgrs_tile",
        "processing_baseline",
        "source_collection",
        "source_service",
        "cloud_cover_percent",
        "local_snapshot_relative_path",
        "sha256",
        "access_status",
        "notes",
    ]
    scene_path = metadata_dir / "input_scene_manifest.csv"
    if not scene_path.exists():
        write_csv(scene_path, scene_fields, [])

    auxiliary_fields = [
        "pipeline",
        "acquisition_datetime_utc",
        "auxiliary_product_id",
        "auxiliary_product_type",
        "source_service",
        "retrieval_date_utc",
        "local_snapshot_relative_path",
        "sha256",
        "access_status",
        "notes",
    ]
    auxiliary_path = metadata_dir / "auxiliary_data_manifest.csv"
    if not auxiliary_path.exists():
        write_csv(auxiliary_path, auxiliary_fields, [])

    exclusion_fields = [
        "farm_identifier",
        "polygon_identifier",
        "acquisition_date",
        "analysis_set",
        "l1c_ndvi",
        "sen2cor_ndvi",
        "lasrc_ndvi",
        "decision_type",
        "reason",
        "pre_specified_rule",
        "included_in_sensitivity_analysis",
        "source_reference",
    ]
    exclusion_path = metadata_dir / "exclusion_ledger.csv"
    if not exclusion_path.exists():
        write_csv(
            exclusion_path,
            exclusion_fields,
            [{
                "farm_identifier": "442",
                "polygon_identifier": "P7",
                "acquisition_date": "2020-12-21",
                "analysis_set": "common three-pipeline comparison",
                "l1c_ndvi": "0.333",
                "sen2cor_ndvi": "0.066",
                "lasrc_ndvi": "0.468",
                "decision_type": "cross-product quality control",
                "reason": "Data-dependent exclusion documented in the manuscript.",
                "pre_specified_rule": "false",
                "included_in_sensitivity_analysis": "",
                "source_reference": "Manuscript Section 3.6",
            }],
        )

    geometry_manifest = {
        "schema_version": "1.0",
        "generated_utc": utc_now(),
        "object_type": "restricted management-polygon geometry",
        "inventory_year": 2019,
        "farm_count": 22,
        "polygon_count": 91,
        "municipalities": [
            "Granada",
            "Fuente de Oro",
            "San Juan de Arama",
            "El Castillo",
        ],
        "access_status": "restricted",
        "public_distribution": False,
        "governance_statement": (
            "Farm-boundary geometries are subject to Sustainable Cattle "
            "Ranching project data-governance conditions."
        ),
        "authoritative_file_logical_identifier": "",
        "crs": "",
        "geometry_type": "",
        "primary_key": "",
        "sha256": "",
        "restricted_storage_reference": "",
        "verification_status": "incomplete",
    }
    geometry_path = metadata_dir / "geometry_manifest.json"
    if not geometry_path.exists():
        geometry_path.write_text(
            json.dumps(geometry_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    script_rows = []
    for stage, role, relative in KNOWN_SCRIPTS:
        path = root / relative
        script_rows.append({
            "stage": stage,
            "role": role,
            "relative_path": relative,
            "present": str(path.is_file()).lower(),
            "sha256": sha256_file(path) if path.is_file() else "",
            "notes": "",
        })
    write_csv(
        contract_dir / "script_registry.tsv",
        ["stage", "role", "relative_path", "present", "sha256", "notes"],
        script_rows,
    )
    # Convert comma-delimited output to TSV.
    csv_path = contract_dir / "script_registry.tsv"
    text = csv_path.read_text(encoding="utf-8")
    rows = list(csv.reader(text.splitlines()))
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerows(rows)

    output_rows = []
    for relative in PUBLIC_DERIVED_TABLES:
        path = root / relative
        output_rows.append({
            "artifact_role": "public derived statistical table",
            "relative_path": relative,
            "present": str(path.is_file()).lower(),
            "observed_row_count": csv_row_count(path) if path.is_file() else "",
            "header": "|".join(csv_header(path)) if path.is_file() else "",
            "sha256": sha256_file(path) if path.is_file() else "",
            "generating_script": "",
            "acceptance_rule": "",
        })
    write_csv(
        contract_dir / "output_contract.tsv",
        [
            "artifact_role",
            "relative_path",
            "present",
            "observed_row_count",
            "header",
            "sha256",
            "generating_script",
            "acceptance_rule",
        ],
        output_rows,
    )
    csv_path = contract_dir / "output_contract.tsv"
    rows = list(csv.reader(csv_path.read_text(encoding="utf-8").splitlines()))
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerows(rows)

    expected_counts = {
        "schema_version": "1.0",
        "common_three_pipeline_records": {
            "expected": 1809,
            "acceptance_rule": "observed == expected",
        },
        "land_use_records": {
            "expected": 1528,
            "acceptance_rule": "observed == expected",
        },
        "land_use_polygons": {
            "expected": 50,
            "acceptance_rule": "observed == expected",
        },
        "masking_sensitivity_records": {
            "expected": 2585,
            "acceptance_rule": "observed == expected",
        },
    }
    (acceptance_dir / "expected_counts.json").write_text(
        json.dumps(expected_counts, indent=2) + "\n",
        encoding="utf-8",
    )

    expected_metrics = {
        "schema_version": "1.0",
        "acceptance_rule": (
            "The reproduced unrounded value must round to the reported "
            "manuscript value at the stated number of decimals."
        ),
        "comparisons": {
            "LaSRC-L1C": {
                "mean_difference": 0.115,
                "rmsd": 0.121,
                "urmsd": 0.035,
                "ccc": 0.653,
                "deming_slope": 1.093,
                "deming_intercept": 0.061,
                "loa_low": 0.046,
                "loa_high": 0.185,
            },
            "Sen2Cor-L1C": {
                "mean_difference": 0.094,
                "rmsd": 0.100,
                "urmsd": 0.034,
                "ccc": 0.743,
                "deming_slope": 1.155,
                "deming_intercept": 0.004,
                "loa_low": 0.027,
                "loa_high": 0.162,
            },
            "LaSRC-Sen2Cor": {
                "mean_difference": 0.021,
                "rmsd": 0.036,
                "urmsd": 0.029,
                "ccc": 0.961,
                "deming_slope": 0.946,
                "deming_intercept": 0.058,
                "loa_low": -0.036,
                "loa_high": 0.078,
            },
        },
    }
    (acceptance_dir / "expected_metrics.json").write_text(
        json.dumps(expected_metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    readme = """# Workflow reproducibility contract — draft

This directory is intentionally stored under `local_only/` while the contract
is incomplete. All public-facing labels, filenames, headers, and documentation
are in English.

The draft must not be promoted to the public repository until:

1. the scene manifest contains every acquisition used by each pipeline;
2. the restricted geometry manifest contains a verified logical identifier,
   CRS, primary key, and SHA-256 checksum;
3. the LaSRC auxiliary-data manifest is complete;
4. the bootstrap seed, interval method, missing-value policy, and software
   implementation are fixed;
5. every public output is mapped to its generating script and acceptance rule;
6. the audit status is `READY_FOR_PROMOTION`.

No unknown value should be invented merely to satisfy the gate.
"""
    write_if_missing(draft / "README.md", readme)

    # Run the bundled auditor when available.
    audit_script = Path(__file__).with_name("audit_reproducibility_contract.py")
    if audit_script.is_file():
        import subprocess
        subprocess.run(
            [
                "python3",
                str(audit_script),
                "--root",
                str(root),
                "--draft-dir",
                str(Path(args.draft_dir)),
            ],
            check=False,
        )

    print(f"Draft reproducibility contract: {draft}")
    print(f"Workflow configuration: {config_dir / 'workflow_config.yml'}")
    print(f"Scene manifest: {scene_path}")
    print(f"Geometry manifest: {geometry_path}")
    print(f"Auxiliary-data manifest: {auxiliary_path}")
    print(f"Exclusion ledger: {exclusion_path}")


if __name__ == "__main__":
    main()
