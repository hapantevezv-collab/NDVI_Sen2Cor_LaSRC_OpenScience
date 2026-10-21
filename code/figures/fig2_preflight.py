"""Preflight for Figure 2 V4.1 - reports ALL fit_text failures and ALL collisions in one run.
Writes nothing (no PDF/SVG/PNG, no provenance). Usage, from the repository root:
    python3 code/figures/fig2_preflight.py
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "code/figures/figure_02_processing_workflow.py"
spec = importlib.util.spec_from_file_location("fig2", SCRIPT)
fig2 = importlib.util.module_from_spec(spec)
sys.modules["fig2"] = fig2
spec.loader.exec_module(fig2)

failures = []
_orig = fig2.fit_text


def tolerant_fit_text(*args, **kwargs):
    try:
        return _orig(*args, **kwargs)
    except SystemExit as exc:
        failures.append(str(exc))
        return None


fig2.fit_text = tolerant_fit_text

print("Font:", fig2.FONT_FAMILY)
data = fig2.load_canonical(ROOT)
fig, ax, reg, steps, legend_y = fig2.build_figure(data)
print(f"\nfit_text failures: {len(failures)}")
for f in failures:
    print("  -", f)
conflicts = fig2.collision_check(fig, ax, reg, legend_y)
print(f"\nCollision check: {len(conflicts)} conflicts")
for c in conflicts:
    print("  -", c)
print("\nPREFLIGHT:", "PASS" if not failures and not conflicts else "FAIL (nothing was written)")
sys.exit(0)
