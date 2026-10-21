#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Promote a completed draft reproducibility contract into the public repository.
The script refuses to run unless the draft audit status is READY_FOR_PROMOTION.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


MAPPINGS = {
    "config/workflow_config.yml": "config/workflow_config.yml",
    "data/metadata/input_scene_manifest.csv":
        "data/metadata/input_scene_manifest.csv",
    "data/metadata/auxiliary_data_manifest.csv":
        "data/metadata/auxiliary_data_manifest.csv",
    "data/metadata/exclusion_ledger.csv":
        "data/metadata/exclusion_ledger.csv",
    "data/metadata/geometry_manifest.json":
        "data/metadata/geometry_manifest.json",
    "metadata/script_registry.tsv": "metadata/script_registry.tsv",
    "metadata/output_contract.tsv": "metadata/output_contract.tsv",
    "acceptance/expected_counts.json":
        "metadata/expected_counts.json",
    "acceptance/expected_metrics.json":
        "metadata/expected_metrics.json",
    "docs/provenance/WORKFLOW_REPRODUCIBILITY_AUDIT.md":
        "docs/provenance/WORKFLOW_REPRODUCIBILITY_AUDIT.md",
}


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
    audit = draft / "docs/provenance/WORKFLOW_REPRODUCIBILITY_AUDIT.md"

    if not audit.is_file():
        raise SystemExit("Audit report is missing.")

    text = audit.read_text(encoding="utf-8")
    if "Status: **READY_FOR_PROMOTION**" not in text:
        raise SystemExit(
            "Promotion refused: draft audit status is not READY_FOR_PROMOTION."
        )

    for source_relative, destination_relative in MAPPINGS.items():
        source = draft / source_relative
        destination = root / destination_relative
        if not source.is_file():
            raise SystemExit(f"Required draft file is missing: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"PROMOTED: {source_relative} -> {destination_relative}")


if __name__ == "__main__":
    main()
