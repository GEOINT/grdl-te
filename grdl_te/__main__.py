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
    return parser


def main() -> None:
    """Parse arguments and run the benchmark suite."""
    args = _build_parser().parse_args()

    results = run_suite(
        size=args.size,
        iterations=args.iterations,
        warmup=args.warmup,
        store_dir=args.store_dir,
        only=args.only,
        skip_workflow=args.skip_workflow,
    )

    sys.exit(0 if results else 1)


if __name__ == "__main__":
    main()
