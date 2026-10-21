#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Audit the draft workflow reproducibility contract."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def count_data_rows(path: Path, delimiter: str = ",") -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        next(reader, None)
        return sum(1 for row in reader if any(cell.strip() for cell in row))


def read_geometry(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def yes_no(value: bool) -> str:
    return "PASS" if value else "FAIL"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--draft-dir",
        default="local_only/workflow_contract_draft",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    draft = root / args.draft_dir

    scene_manifest = draft / "data/metadata/input_scene_manifest.csv"
    auxiliary_manifest = draft / "data/metadata/auxiliary_data_manifest.csv"
    exclusion_ledger = draft / "data/metadata/exclusion_ledger.csv"
    geometry_manifest = draft / "data/metadata/geometry_manifest.json"
    workflow_config = draft / "config/workflow_config.yml"
    script_registry = draft / "metadata/script_registry.tsv"
    output_contract = draft / "metadata/output_contract.tsv"
    expected_counts = draft / "acceptance/expected_counts.json"
    expected_metrics = draft / "acceptance/expected_metrics.json"

    scene_rows = count_data_rows(scene_manifest)
    auxiliary_rows = count_data_rows(auxiliary_manifest)
    exclusion_rows = count_data_rows(exclusion_ledger)
    geometry = read_geometry(geometry_manifest)

    required_geometry_fields = (
        "authoritative_file_logical_identifier",
        "crs",
        "geometry_type",
        "primary_key",
        "sha256",
        "restricted_storage_reference",
    )
    missing_geometry = [
        field for field in required_geometry_fields
        if not str(geometry.get(field, "")).strip()
    ]
    geometry_hash_ok = bool(
        SHA256_RE.fullmatch(str(geometry.get("sha256", "")).strip())
    )

    environment_files = {
        "container_provenance": root / "environment/container-provenance.txt",
        "requirements_freeze": root / "environment/requirements-freeze.txt",
        "tool_versions": root / "environment/tool-versions.txt",
    }

    required_files = [
        workflow_config,
        scene_manifest,
        auxiliary_manifest,
        exclusion_ledger,
        geometry_manifest,
        script_registry,
        output_contract,
        expected_counts,
        expected_metrics,
    ]

    missing_files = [path for path in required_files if not path.is_file()]

    config_text = (
        workflow_config.read_text(encoding="utf-8")
        if workflow_config.is_file()
        else ""
    )
    config_missing_values = [
        label for label in (
            "random_seed:",
            "interval_method:",
            "missing_value_policy:",
            "software_implementation:",
        )
        if re.search(rf"^\s*{re.escape(label)}\s*$", config_text, re.MULTILINE)
    ]

    registry_missing_scripts = []
    if script_registry.is_file():
        with script_registry.open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                if row.get("present", "").lower() != "true":
                    registry_missing_scripts.append(row.get("relative_path", ""))

    output_unmapped = []
    if output_contract.is_file():
        with output_contract.open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                if not row.get("generating_script", "").strip():
                    output_unmapped.append(row.get("relative_path", ""))

    blockers = []
    if missing_files:
        blockers.append("Required draft contract files are missing.")
    if scene_rows == 0:
        blockers.append("The input scene manifest has no acquisition records.")
    if auxiliary_rows == 0:
        blockers.append("The auxiliary-data manifest has no LaSRC records.")
    if exclusion_rows == 0:
        blockers.append("The exclusion ledger is empty.")
    if missing_geometry:
        blockers.append(
            "The restricted geometry manifest is incomplete: "
            + ", ".join(missing_geometry)
        )
    if not geometry_hash_ok:
        blockers.append("The restricted geometry SHA-256 is absent or invalid.")
    if config_missing_values:
        blockers.append(
            "Statistical implementation fields are not fixed: "
            + ", ".join(config_missing_values)
        )
    if registry_missing_scripts:
        blockers.append(
            "Canonical scripts are missing: "
            + ", ".join(registry_missing_scripts)
        )
    if output_unmapped:
        blockers.append(
            "Public outputs are not mapped to generating scripts: "
            + ", ".join(output_unmapped)
        )
    for name, path in environment_files.items():
        if not path.is_file() or path.stat().st_size == 0:
            blockers.append(f"Environment evidence is missing: {path.relative_to(root)}")

    status = "READY_FOR_PROMOTION" if not blockers else "INCOMPLETE"

    report = draft / "docs/provenance/WORKFLOW_REPRODUCIBILITY_AUDIT.md"
    report.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Workflow reproducibility audit",
        "",
        f"- Status: **{status}**",
        f"- Generated: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Contract inventory",
        "",
        f"- Workflow configuration: **{yes_no(workflow_config.is_file())}**",
        f"- Scene-manifest records: **{scene_rows}**",
        f"- Auxiliary-data records: **{auxiliary_rows}**",
        f"- Exclusion-ledger records: **{exclusion_rows}**",
        f"- Geometry-manifest checksum: **{yes_no(geometry_hash_ok)}**",
        f"- Script registry: **{yes_no(script_registry.is_file())}**",
        f"- Output contract: **{yes_no(output_contract.is_file())}**",
        "",
        "## Environment evidence",
        "",
    ]

    for name, path in environment_files.items():
        lines.append(
            f"- `{path.relative_to(root)}`: "
            f"**{yes_no(path.is_file() and path.stat().st_size > 0)}**"
        )

    lines.extend(["", "## Blocking findings", ""])
    if blockers:
        lines.extend([f"- {item}" for item in blockers])
    else:
        lines.append("- None.")

    lines.extend([
        "",
        "## Promotion rule",
        "",
        "The contract may be promoted to the public repository only when the "
        "status is `READY_FOR_PROMOTION`. Unknown values must be verified, "
        "not invented.",
        "",
    ])
    report.write_text("\n".join(lines), encoding="utf-8")

    print(report)
    print(f"Status: {status}")


if __name__ == "__main__":
    main()
