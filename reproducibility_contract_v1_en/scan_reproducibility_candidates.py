#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scan a source workspace for likely scene, auxiliary-data, geometry, and
reproducibility artifacts. The output is an inventory only; no file is copied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path


TOKENS = (
    "scene", "catalog", "manifest", "inventory", "reconciliation",
    "aux", "m2m", "usgs", "lasrc", "sen2cor", "l1c", "ndvi",
    "polygon", "geometry", "exclusion", "bootstrap", "agreement",
)

SUFFIXES = {
    ".csv", ".tsv", ".json", ".geojson", ".gpkg", ".shp",
    ".parquet", ".txt", ".yml", ".yaml", ".py", ".sh",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-files", type=int, default=500)
    parser.add_argument(
        "--hash-limit-mb",
        type=int,
        default=50,
        help="Only hash files at or below this size.",
    )
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    matches = []
    for path in workspace.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUFFIXES:
            continue
        lower = path.name.lower()
        if not any(token in lower for token in TOKENS):
            continue
        matches.append(path)
        if len(matches) >= args.max_files:
            break

    with output.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "relative_path",
            "suffix",
            "bytes",
            "modified_utc",
            "sha256",
            "candidate_type",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for path in sorted(matches):
            size = path.stat().st_size
            name = path.name.lower()
            if any(x in name for x in ("scene", "catalog", "inventory", "manifest")):
                candidate_type = "scene or product inventory"
            elif any(x in name for x in ("aux", "m2m", "usgs")):
                candidate_type = "auxiliary-data evidence"
            elif any(x in name for x in ("polygon", "geometry")):
                candidate_type = "geometry evidence"
            elif "exclusion" in name:
                candidate_type = "exclusion evidence"
            else:
                candidate_type = "analysis or provenance candidate"

            writer.writerow({
                "relative_path": str(path.relative_to(workspace)),
                "suffix": path.suffix.lower(),
                "bytes": size,
                "modified_utc": datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
                "sha256": (
                    sha256_file(path)
                    if size <= args.hash_limit_mb * 1024 * 1024
                    else ""
                ),
                "candidate_type": candidate_type,
            })

    print(output)
    print(f"Candidates: {len(matches)}")


if __name__ == "__main__":
    main()
