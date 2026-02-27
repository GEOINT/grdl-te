# -*- coding: utf-8 -*-
"""Root conftest — saves per-run coverage reports for validation tests."""

import re
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
_COVERAGE_TMP = _PROJECT_ROOT / ".coverage.tmp.json"
_REPORTS_DIR = _PROJECT_ROOT / "coverage" / "reports"
_EXCLUDE_DIRS = {"benchmarking"}


def pytest_sessionfinish(session, exitstatus):
    """Save coverage JSON with a descriptive name, skip for excluded dirs."""
    # pytest-cov writes the tmp file relative to cwd, which may differ
    # from the project root when pytest is invoked from another directory.
    tmp = _COVERAGE_TMP
    if not tmp.exists():
        tmp = Path.cwd() / ".coverage.tmp.json"
    if not tmp.exists():
        return

    coverage_db = tmp.parent / ".coverage"

    if session.items and all(
        _EXCLUDE_DIRS & set(item.path.parts) for item in session.items
    ):
        tmp.unlink(missing_ok=True)
        coverage_db.unlink(missing_ok=True)
        return

    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    label = _label_from_items(session.items)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest = _REPORTS_DIR / f"coverage_{label}_{ts}.json"
    tmp.rename(dest)
    coverage_db.unlink(missing_ok=True)

    tw = session.config.get_terminal_writer()
    tw.line()
    tw.line(f"Coverage JSON saved to {dest.relative_to(_PROJECT_ROOT)}")


def _label_from_items(items) -> str:
    """Build a short label from the test files that actually ran."""
    files = list(dict.fromkeys(item.path for item in items))
    names = [f.stem.removeprefix("test_") for f in files]

    if len(names) > 2:
        names = list(dict.fromkeys(f.parent.name for f in files))

    label = "_".join(names[:2]) or "all"
    return re.sub(r"[^\w\-.]", "_", label)[:80]
