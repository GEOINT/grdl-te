# -*- coding: utf-8 -*-
"""
CLI entry point for the GRDL-TE benchmark suite.

Run with::

    python -m grdl_te                          # medium, 10 iterations
    python -m grdl_te --size small -n 5        # small, 5 iterations
    python -m grdl_te --only filters io        # specific groups
    python -m grdl_te --store-dir ./results    # custom output

Author
------
Ava Courtney

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2026-02-13

Modified
--------
2026-02-13
"""

# Standard library
import argparse
import sys
from pathlib import Path
from typing import List

from grdl_te.benchmarking.suite import (
    ARRAY_SIZES,
    BENCHMARK_GROUPS,
    DEFAULT_ITERATIONS,
    DEFAULT_SIZE,
    DEFAULT_WARMUP,
    run_suite,
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    valid_groups = list(BENCHMARK_GROUPS.keys()) + ["workflow"]

    parser = argparse.ArgumentParser(
        prog="python -m grdl_te",
        description="GRDL Component Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Array sizes:
  small   {ARRAY_SIZES['small'][0]} x {ARRAY_SIZES['small'][1]}
  medium  {ARRAY_SIZES['medium'][0]} x {ARRAY_SIZES['medium'][1]}
  large   {ARRAY_SIZES['large'][0]} x {ARRAY_SIZES['large'][1]}

Benchmark groups:
  {', '.join(valid_groups)}

Examples:
  python -m grdl_te                            # medium, 10 iterations
  python -m grdl_te --size small -n 5          # quick run
  python -m grdl_te --size large -n 20         # thorough run
  python -m grdl_te --only filters intensity   # specific groups
  python -m grdl_te --skip-workflow             # component-only
  python -m grdl_te --store-dir ./results      # custom output dir
  python -m grdl_te --report                   # print report to terminal
  python -m grdl_te --report ./reports/        # save report to directory
  python -m grdl_te --report ./my_report.txt   # save report to file
  python -m grdl_te --view .benchmarks/        # view saved records in GUI
  python -m grdl_te --view rec1.json rec2.json # view specific JSON files
""",
    )
    parser.add_argument(
        "--size", choices=ARRAY_SIZES.keys(), default=DEFAULT_SIZE,
        help=f"Array size preset (default: {DEFAULT_SIZE})",
    )
    parser.add_argument(
        "-n", "--iterations", type=int, default=DEFAULT_ITERATIONS,
        help=f"Measurement iterations (default: {DEFAULT_ITERATIONS})",
    )
    parser.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP,
        help=f"Warmup iterations (default: {DEFAULT_WARMUP})",
    )
    parser.add_argument(
        "--store-dir", type=Path, default=None,
        help="Benchmark store directory (default: .benchmarks/)",
    )
    parser.add_argument(
        "--skip-workflow", action="store_true",
        help="Skip the workflow-level benchmark",
    )
    parser.add_argument(
        "--only", nargs="+", choices=valid_groups, default=None,
        help="Run only specific benchmark groups",
    )
    parser.add_argument(
        "--report", nargs="?", const=True, default=None,
        metavar="PATH",
        help=(
            "Generate a comprehensive report. Without a path, prints to "
            "terminal. With a path, writes to the specified file or directory."
        ),
    )
    parser.add_argument(
        "--view", nargs="+", metavar="PATH",
        help=(
            "View saved benchmark records in the interactive GUI dashboard. "
            "Accepts a store directory, a directory of JSON files, "
            "or individual JSON file paths."
        ),
    )
    parser.add_argument(
        "--port", type=int, default=8080,
        help="Port for the GUI dashboard (default: 8080). Used with --view.",
    )
    return parser


def _load_records(paths: List[Path]) -> List["BenchmarkRecord"]:  # noqa: F821
    """Load benchmark records from files or directories.

    Parameters
    ----------
    paths : list of Path
        Each path may be a ``.json`` file, a ``JSONBenchmarkStore``
        directory (contains a ``records/`` subdirectory), or a plain
        directory of ``.json`` files.

    Returns
    -------
    list of BenchmarkRecord
        De-duplicated by ``benchmark_id``.
    """
    from grdl_te.benchmarking.models import BenchmarkRecord
    from grdl_te.benchmarking.store import JSONBenchmarkStore

    seen_ids: set = set()
    records: List[BenchmarkRecord] = []

    def _add(rec: BenchmarkRecord) -> None:
        if rec.benchmark_id not in seen_ids:
            seen_ids.add(rec.benchmark_id)
            records.append(rec)

    for p in paths:
        p = p.expanduser().resolve()
        if p.is_file() and p.suffix == ".json":
            try:
                _add(BenchmarkRecord.from_json(p.read_text()))
            except Exception as exc:
                print(f"Warning: could not load {p}: {exc}", file=sys.stderr)
        elif p.is_dir():
            if (p / "records").is_dir():
                # Store directory
                store = JSONBenchmarkStore(base_dir=p)
                for rec in store.list_records(limit=10_000):
                    _add(rec)
            else:
                # Loose JSON files
                json_files = sorted(p.glob("*.json"))
                if not json_files:
                    print(f"Warning: no .json files in {p}", file=sys.stderr)
                for jf in json_files:
                    try:
                        _add(BenchmarkRecord.from_json(jf.read_text()))
                    except Exception as exc:
                        print(
                            f"Warning: could not load {jf}: {exc}",
                            file=sys.stderr,
                        )
        else:
            print(f"Warning: {p} is not a file or directory", file=sys.stderr)

    return records


def main() -> None:
    """Parse arguments and run the benchmark suite."""
    args = _build_parser().parse_args()

    if args.view:
        records = _load_records([Path(p) for p in args.view])
        if not records:
            print("Error: no benchmark records found.", file=sys.stderr)
            sys.exit(1)
        print(f"Loaded {len(records)} record(s). Launching dashboard...")
        from collections import OrderedDict

        from grdl_te.benchmarking.report_gui import launch_ui

        # Group by workflow name for tabbed view
        groups: OrderedDict[str, List] = OrderedDict()
        for rec in records:
            groups.setdefault(rec.workflow_name, []).append(rec)

        if len(groups) > 1:
            launch_ui(
                [(name, recs) for name, recs in groups.items()],
                port=args.port,
            )
        else:
            launch_ui(records, port=args.port)
        sys.exit(0)

    results = run_suite(
        size=args.size,
        iterations=args.iterations,
        warmup=args.warmup,
        store_dir=args.store_dir,
        only=args.only,
        skip_workflow=args.skip_workflow,
    )

    if args.report is not None and results:
        from grdl_te.benchmarking.report import print_report, save_report

        if args.report is True:
            print_report(results)
        else:
            output_path = save_report(results, Path(args.report))
            print(f"\nReport saved to: {output_path}")

    sys.exit(0 if results else 1)


if __name__ == "__main__":
    main()
