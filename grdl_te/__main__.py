# -*- coding: utf-8 -*-
"""
CLI entry point for the GRDL-TE benchmark suite.

Run with::

    python -m grdl_te                          # medium, 10 iterations
    python -m grdl_te --size small -n 5        # small, 5 iterations
    python -m grdl_te --only filters io        # specific groups
    python -m grdl_te --store-dir ./results    # custom output
    python -m grdl_te --stress-test            # stress test with defaults
    python -m grdl_te --stress-test --stress-concurrency 8 --stress-steps 4

Author
------
Ava Courtney <courtney-ava@zai.com>

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2026-02-13

Modified
--------
2026-04-07
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
from grdl_te.benchmarking.stress_models import (
    DEFAULT_DURATION_PER_STEP_S,
    DEFAULT_FAILURE_ESCALATION_MODE,
    DEFAULT_FAILURE_THRESHOLD_PCT,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_ESCALATION_CONCURRENCY,
    DEFAULT_MAX_ESCALATION_PAYLOAD_DIM,
    DEFAULT_RAMP_MODE,
    DEFAULT_RAMP_STEPS,
    StressTestConfig,
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
  python -m grdl_te --stress-gui                # interactive stress test GUI
  python -m grdl_te --stress-gui --port 8081    # custom port
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
        help="Port for the GUI dashboard (default: 8080). Used with --view and --stress-gui.",
    )
    parser.add_argument(
        "--stress-gui", action="store_true",
        help="Launch the interactive stress test GUI dashboard.",
    )

    # ---- Stress test options ----
    stress = parser.add_argument_group(
        "stress testing",
        description=(
            "Run a concurrency ramp stress test instead of the standard "
            "benchmark suite.  Results are stored separately under "
            "<store-dir>/stress/ and produce a dedicated report."
        ),
    )
    stress.add_argument(
        "--stress-test", action="store_true",
        help="Run stress tests instead of standard benchmarks.",
    )
    stress.add_argument(
        "--stress-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY,
        metavar="N",
        help=(
            f"Maximum worker concurrency for the stress ramp "
            f"(default: {DEFAULT_MAX_CONCURRENCY})."
        ),
    )
    stress.add_argument(
        "--stress-steps", type=int, default=DEFAULT_RAMP_STEPS,
        metavar="N",
        help=(
            f"Number of ramp steps (default: {DEFAULT_RAMP_STEPS}). "
            "Uses geometric doubling from 1 to --stress-concurrency."
        ),
    )
    stress.add_argument(
        "--stress-duration", type=float, default=DEFAULT_DURATION_PER_STEP_S,
        metavar="S",
        help=(
            f"Seconds to sustain each concurrency level "
            f"(default: {DEFAULT_DURATION_PER_STEP_S})."
        ),
    )
    stress.add_argument(
        "--run-until-failure", action="store_true",
        help=(
            "Stop early when failure_threshold_pct%% of calls fail at a step "
            "or the wall-time budget is reached.  Works with both ramp modes."
        ),
    )
    stress.add_argument(
        "--ramp-mode",
        choices=("concurrency", "payload"),
        default=DEFAULT_RAMP_MODE,
        metavar="MODE",
        help=(
            "Which axis to ramp.  'concurrency' (default) ramps parallel workers "
            "at a fixed payload size.  'payload' ramps array dimensions geometrically "
            "at a fixed concurrency level, skipping the worker ramp."
        ),
    )
    # Legacy alias kept for backward-compatible scripts
    stress.add_argument(
        "--escalate",
        choices=("concurrency", "payload"),
        default=None,
        metavar="MODE",
        help=argparse.SUPPRESS,  # hidden; prefer --ramp-mode
    )
    stress.add_argument(
        "--failure-threshold", type=float, default=DEFAULT_FAILURE_THRESHOLD_PCT,
        metavar="PCT",
        help=(
            f"Percentage of calls that must fail at a level to declare "
            f"saturation when --run-until-failure is active "
            f"(default: {DEFAULT_FAILURE_THRESHOLD_PCT})."
        ),
    )
    stress.add_argument(
        "--max-wall-time", type=float, default=None,
        metavar="S",
        help=(
            "Maximum total wall-clock seconds for the entire run "
            "(ramp + escalation).  Omit for no hard limit."
        ),
    )
    stress.add_argument(
        "--max-escalation-concurrency", type=int,
        default=DEFAULT_MAX_ESCALATION_CONCURRENCY, metavar="N",
        help=(
            f"Hard ceiling on concurrent workers during escalation "
            f"(default: {DEFAULT_MAX_ESCALATION_CONCURRENCY})."
        ),
    )
    stress.add_argument(
        "--max-escalation-payload-dim", type=int,
        default=DEFAULT_MAX_ESCALATION_PAYLOAD_DIM, metavar="PX",
        help=(
            f"Hard ceiling on payload linear dimension in pixels during escalation "
            f"(default: {DEFAULT_MAX_ESCALATION_PAYLOAD_DIM})."
        ),
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


def _run_stress(args) -> None:
    """Execute the stress test suite and report results."""
    from pathlib import Path

    from grdl_te.benchmarking.stress_runner import ComponentStressTester
    from grdl_te.benchmarking.stress_store import JSONStressTestStore
    from grdl_te.benchmarking.stress_report import (
        print_stress_report,
        save_stress_report,
        save_stress_report_md,
    )

    config = StressTestConfig(
        max_concurrency=args.stress_concurrency,
        ramp_steps=args.stress_steps,
        duration_per_step_s=args.stress_duration,
        payload_size=args.size,
        run_until_failure=args.run_until_failure,
        ramp_mode=args.escalate if args.escalate is not None else args.ramp_mode,
        failure_escalation_mode=args.escalate if args.escalate is not None else args.ramp_mode,
        failure_threshold_pct=args.failure_threshold,
        max_wall_time_s=args.max_wall_time,
        max_escalation_concurrency=args.max_escalation_concurrency,
        max_escalation_payload_dim=args.max_escalation_payload_dim,
    )

    store_dir = args.store_dir or (Path.cwd() / ".benchmarks")
    store = JSONStressTestStore(base_dir=store_dir)

    print("GRDL Stress Test Suite")
    print(f"  Payload size:    {config.payload_size}")
    print(f"  Max concurrency: {config.max_concurrency}")
    print(f"  Ramp steps:      {config.ramp_steps}")
    print(f"  Step duration:   {config.duration_per_step_s}s")
    print(f"  Store:           {store._stress_dir}")
    print()

    # Run stress tests on representative lightweight components
    records = []
    try:
        import numpy as np
        from grdl.data_prep import Normalizer

        norm = Normalizer(method="minmax")
        tester = ComponentStressTester(
            "Normalizer.minmax",
            norm.normalize,
            store=store,
            tags={"group": "data_prep"},
        )
        print("  Running: Normalizer.minmax ...")
        record = tester.run(config)
        records.append(record)
        s = record.summary
        print(
            f"  Done. max_sustained={s.max_sustained_concurrency}  "
            f"saturation={s.saturation_concurrency}  "
            f"failures={s.failed_calls}/{s.total_calls}"
        )
    except Exception as exc:
        print(f"  SKIP  Normalizer.minmax: {exc}")

    if not records:
        print("\nNo stress tests completed.")
        return

    for record in records:
        if args.report is not None:
            if args.report is True:
                print_stress_report(record)
            else:
                report_path = save_stress_report(record, Path(args.report))
                md_path = save_stress_report_md(record, Path(args.report))
                print(f"\n  Text report:     {report_path}")
                print(f"  Markdown report: {md_path}")
        else:
            print_stress_report(record)


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

    if args.stress_gui:
        from grdl_te.benchmarking.stress_gui import launch_stress_gui
        store_dir = args.store_dir or (Path.cwd() / ".benchmarks")
        print(f"Launching Stress Test GUI on http://localhost:{args.port} ...")
        launch_stress_gui(port=args.port, store_dir=store_dir)
        sys.exit(0)

    if args.stress_test:
        _run_stress(args)
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
