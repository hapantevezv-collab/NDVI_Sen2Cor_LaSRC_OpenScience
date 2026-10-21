#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build curated Zenodo packages with package-specific metadata, manifests, and checksums."""

from __future__ import annotations
import argparse
import csv
import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def should_exclude(rel: Path) -> bool:
    if not rel.parts:
        return False
    if rel.parts[0] in {'local_only', 'release', '.git'}:
        return True
    if rel.parts[:2] == ('docs', 'review'):
        return True
    if rel.name in {'migration_report.tsv', 'source_workspace_inventory.tsv', 'duplicate_files.tsv'}:
        return True
    return False


def copy_item(root: Path, rel: Path, staging: Path) -> None:
    source = root / rel
    if not source.exists():
        return
    if source.is_dir():
        for path in source.rglob('*'):
            if path.is_file():
                sub_rel = path.relative_to(root)
                if should_exclude(sub_rel):
                    continue
                target = staging / sub_rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    else:
        target = staging / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def write_manifest(staging: Path) -> None:
    files = sorted(path for path in staging.rglob('*') if path.is_file() and path.name not in {'MANIFEST.tsv', 'CHECKSUMS.sha256'})
    with (staging / 'MANIFEST.tsv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(['relative_path', 'bytes', 'sha256'])
        for path in files:
            writer.writerow([str(path.relative_to(staging)), path.stat().st_size, sha256(path)])
    with (staging / 'CHECKSUMS.sha256').open('w', encoding='utf-8') as handle:
        for path in files:
            handle.write(f'{sha256(path)}  {path.relative_to(staging)}\n')


def build_package(root: Path, output_dir: Path, name: str, version: str, includes: list[Path], zenodo_metadata: Path) -> Path:
    with tempfile.TemporaryDirectory(prefix=f'{name}_') as tmp:
        staging = Path(tmp)
        for rel in includes:
            copy_item(root, rel, staging)
        if zenodo_metadata.exists():
            shutil.copy2(zenodo_metadata, staging / 'ZENODO_METADATA.json')
        write_manifest(staging)
        output = output_dir / f'ndvi_sen2cor_lasrc_{name}_v{version}.zip'
        with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(p for p in staging.rglob('*') if p.is_file()):
                archive.write(path, arcname=str(path.relative_to(staging)))
        return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='.')
    parser.add_argument('--version')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    version = args.version or (root / 'VERSION').read_text(encoding='utf-8').strip()
    output_dir = root / 'release' / 'packages'
    output_dir.mkdir(parents=True, exist_ok=True)

    common = [
        Path('README.md'), Path('CITATION.cff'), Path('VERSION'), Path('CHANGELOG.md'), Path('LICENSES'),
    ]
    specs = {
        'software': common + [
            Path('codemeta.json'), Path('code'), Path('config'), Path('docs/open_science'),
            Path('docs/provenance'), Path('environment'), Path('metadata/creators.tsv'),
            Path('metadata/contributors.tsv'), Path('metadata/public_provenance_summary.tsv'),
            Path('metadata/record_relationships.md'),
        ],
        'dataset': common + [
            Path('data/derived'), Path('data/metadata'), Path('docs/open_science/DATA_AVAILABILITY.md'),
            Path('docs/open_science/LANGUAGE_POLICY.md'), Path('metadata/creators.tsv'),
            Path('metadata/contributors.tsv'), Path('metadata/public_provenance_summary.tsv'),
        ],
        'manuscript': common + [
            Path('manuscript/source'), Path('manuscript/rendered'),
            Path('docs/open_science/DATA_AVAILABILITY.md'), Path('docs/open_science/LANGUAGE_POLICY.md'),
            Path('metadata/creators.tsv'), Path('metadata/contributors.tsv'),
        ],
        'compendium': common + [
            Path('codemeta.json'), Path('.zenodo.json'), Path('manuscript'), Path('code'), Path('config'),
            Path('data'), Path('docs/open_science'), Path('docs/provenance'), Path('environment'), Path('metadata'),
        ],
    }

    outputs = []
    for name, includes in specs.items():
        metadata = root / 'metadata' / f'zenodo_{name}.json'
        outputs.append(build_package(root, output_dir, name, version, includes, metadata))
    for output in outputs:
        print(output)


if __name__ == '__main__':
    main()
