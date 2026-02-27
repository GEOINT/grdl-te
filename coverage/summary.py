#!/usr/bin/env python3
"""
Coverage summary grouped by package directory.

Reads JSON coverage reports produced by pytest-cov and displays
a rolled-up view by directory.

Usage
-----
    python coverage/summary.py              # latest report
    python coverage/summary.py --all        # merge all reports
    python coverage/summary.py FILE         # summarise a specific JSON file
    python coverage/summary.py -v           # expand to show individual files

Author
------
Ava Courtney

License
-------
MIT License
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPORTS_DIR = Path(__file__).parent / "reports"


def resolve_coverage_paths(file_arg: str | None, all_flag: bool) -> list[Path]:
    """Determine which coverage JSON file(s) to load.

    Returns a list of paths (one for a single file, many for --all).
    """
    if file_arg:
        p = Path(file_arg)
        if not p.exists():
            print(f"Coverage file not found: {p}")
            sys.exit(1)
        return [p]

    if not REPORTS_DIR.is_dir():
        print(f"Coverage directory not found: {REPORTS_DIR}")
        print("Run pytest first to generate a report.")
        sys.exit(1)

    reports = sorted(REPORTS_DIR.glob("*.json"))
    if not reports:
        print(f"No JSON reports found in {REPORTS_DIR}")
        sys.exit(1)

    if all_flag:
        return reports

    # Default: most recent file (last when sorted by name/timestamp)
    return [reports[-1]]


def load_coverage(paths: list[Path]) -> dict:
    """Load and merge coverage JSON report(s).

    When multiple paths are provided the file-level data is merged,
    keeping the entry with the highest statement count for each file.
    """
    merged_files: dict = {}
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        for filepath, fdata in data.get("files", {}).items():
            existing = merged_files.get(filepath)
            if existing is None:
                merged_files[filepath] = fdata
            else:
                # Keep whichever run had more statements covered
                old_stmts = existing.get("summary", {}).get("num_statements", 0)
                new_stmts = fdata.get("summary", {}).get("num_statements", 0)
                if new_stmts > old_stmts:
                    merged_files[filepath] = fdata
    return merged_files


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
    parser = argparse.ArgumentParser(
        description="Coverage summary grouped by package directory.",
    )
    parser.add_argument(
        "file", nargs="?", default=None,
        help="Path to a specific coverage JSON file. "
             "Default: latest report in coverage/reports/",
    )
    parser.add_argument(
        "-a", "--all", action="store_true",
        help="Merge all reports in coverage/reports/ instead of using the latest.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show individual files within each package group.",
    )
    args = parser.parse_args()

    paths = resolve_coverage_paths(args.file, args.all)
    print(f"Loading {len(paths)} report(s): {', '.join(p.name for p in paths)}")

    files = load_coverage(paths)
    if not files:
        print("No coverage data found in report(s).")
        sys.exit(1)
    groups = group_by_directory(files)
    print_summary(groups, verbose=args.verbose)


if __name__ == "__main__":
    main()
