#!/usr/bin/env python3
"""
Coverage summary grouped by package directory.

Reads the JSON coverage report produced by pytest-cov and displays
a rolled-up view by directory. Run after pytest to get the summary.

Usage
-----
    pytest                          # generates .coverage.json
    python coverage_summary.py      # compact directory summary
    python coverage_summary.py -v   # expand to show individual files

Author
------
Ava Courtney

License
-------
MIT License
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

COVERAGE_FILE = Path(__file__).parent / ".coverage.json"


def load_coverage(path: Path) -> dict:
    """Load coverage JSON report."""
    if not path.exists():
        print(f"Coverage file not found: {path}")
        print("Run pytest first to generate the report.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def group_by_directory(files: dict, depth: int = 3) -> dict:
    """Group file coverage stats by parent directory.

    Parameters
    ----------
    files : dict
        coverage.json ``files`` section keyed by file path.
    depth : int
        Number of path components (from the package root) to group by.
        Default 3 gives groups like ``grdl/IO/sar``.
    """
    groups = defaultdict(lambda: {"stmts": 0, "miss": 0, "files": []})

    for filepath, data in files.items():
        summary = data.get("summary", {})
        stmts = summary.get("num_statements", 0)
        miss = summary.get("missing_lines", 0)
        if stmts == 0:
            continue

        # Derive group key from path relative to package root
        parts = Path(filepath).parts
        # Find 'grdl' in path to anchor the grouping
        try:
            anchor = parts.index("grdl")
        except ValueError:
            anchor = 0
        key_parts = parts[anchor:anchor + depth]
        key = "/".join(key_parts)

        pct = round(100 * (stmts - miss) / stmts) if stmts else 0
        groups[key]["stmts"] += stmts
        groups[key]["miss"] += miss
        groups[key]["files"].append((filepath, stmts, miss, pct))

    return dict(groups)


def print_summary(groups: dict, verbose: bool = False) -> None:
    """Print coverage summary table."""
    # Sort by coverage ascending (worst first)
    sorted_groups = sorted(
        groups.items(),
        key=lambda kv: (kv[1]["stmts"] - kv[1]["miss"]) / max(kv[1]["stmts"], 1),
    )

    total_stmts = 0
    total_miss = 0

    # Header
    print()
    print(f"{'Package':<45} {'Stmts':>6} {'Miss':>6} {'Cover':>6}")
    print("-" * 67)

    for key, data in sorted_groups:
        stmts = data["stmts"]
        miss = data["miss"]
        pct = round(100 * (stmts - miss) / stmts) if stmts else 0
        total_stmts += stmts
        total_miss += miss

        # Color: red < 50%, yellow 50-79%, green >= 80%
        print(f"{key:<45} {stmts:>6} {miss:>6} {pct:>5}%")

        if verbose:
            for fpath, f_stmts, f_miss, f_pct in sorted(
                data["files"], key=lambda x: x[3]
            ):
                short = Path(fpath).name
                print(f"  {short:<43} {f_stmts:>6} {f_miss:>6} {f_pct:>5}%")

    # Total
    total_pct = round(100 * (total_stmts - total_miss) / total_stmts) if total_stmts else 0
    print("-" * 67)
    print(f"{'TOTAL':<45} {total_stmts:>6} {total_miss:>6} {total_pct:>5}%")
    print()


def main():
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    data = load_coverage(COVERAGE_FILE)
    files = data.get("files", {})
    if not files:
        print("No coverage data found in report.")
        sys.exit(1)
    groups = group_by_directory(files)
    print_summary(groups, verbose=verbose)


if __name__ == "__main__":
    main()