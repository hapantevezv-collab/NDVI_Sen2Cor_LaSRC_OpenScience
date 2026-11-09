#!/usr/bin/env python3
"""P023 - Add Zenodo concept DOI and bump to v1.0.1. Run from repo root."""
import json, re, sys
from pathlib import Path
C = "10.5281/zenodo.24089625"; V1 = "10.5281/zenodo.24089626"; NEW = "1.0.1"; DATE = "2026-11-09"
if not Path("CITATION.cff").exists(): sys.exit("Run from the repository root.")
Path("VERSION").write_text(NEW + "\n")
# CITATION.cff
s = Path("CITATION.cff").read_text(encoding="utf-8")
s = re.sub(r'^version: ".*"$', f'version: "{NEW}"', s, flags=re.M)
s = re.sub(r'^date-released: ".*"$', f'date-released: "{DATE}"', s, flags=re.M)
if "\ndoi:" not in s:
    s = s.replace(f'version: "{NEW}"', f'version: "{NEW}"\ndoi: "{C}"\nidentifiers:\n  - type: doi\n    value: "{C}"\n    description: "Concept DOI (all versions)"\n  - type: doi\n    value: "{V1}"\n    description: "Version 1.0.0"', 1)
Path("CITATION.cff").write_text(s, encoding="utf-8")
# .zenodo.json (version only; Zenodo assigns the new version DOI)
z = json.loads(Path(".zenodo.json").read_text(encoding="utf-8")); z["version"] = NEW
Path(".zenodo.json").write_text(json.dumps(z, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
# codemeta.json
c = json.loads(Path("codemeta.json").read_text(encoding="utf-8"))
c["version"] = NEW; c["dateModified"] = DATE; c["identifier"] = f"https://doi.org/{C}"
Path("codemeta.json").write_text(json.dumps(c, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
# README
r = Path("README.md").read_text(encoding="utf-8")
r = r.replace("**Version:** 1.0.0", f"**Version:** {NEW}")
r = r.replace("**DOI:** assigned by Zenodo at release", f"**DOI (all versions):** [{C}](https://doi.org/{C})")
r = r.replace("--version 1.0.0", f"--version {NEW}")
Path("README.md").write_text(r, encoding="utf-8")
# CHANGELOG
ch = Path("CHANGELOG.md").read_text(encoding="utf-8")
if f"[{NEW}]" not in ch:
    ch = ch.replace("# Changelog\n", f"# Changelog\n\n## [{NEW}] - {DATE}\n\n- Added the Zenodo concept DOI ({C}) to the manuscript Data Availability Statement, README, CITATION.cff and codemeta.json.\n- Acknowledged the SCR data team and the SNAPP Colombia Sustainable Cattle Ranching Working Group.\n- Version 1.0.0: {V1}.\n", 1)
Path("CHANGELOG.md").write_text(ch, encoding="utf-8")
print("Updated VERSION, CITATION.cff, .zenodo.json, codemeta.json, README.md, CHANGELOG.md ->", NEW)
