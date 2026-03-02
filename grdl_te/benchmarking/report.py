# -*- coding: utf-8 -*-
"""
Benchmark Report Generator — comprehensive human-readable reports.

Formats benchmark results into a detailed text report covering hardware
configuration, per-benchmark statistics, per-step breakdowns, module-level
aggregation, and overall summary.  Reports can be printed to stdout or
written to a file.

Dependencies
------------
numpy

Author
------
Ava Courtney

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2026-02-18

Modified
--------
2026-02-18
"""

# Standard library
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

# Third-party
import numpy as np

# Internal
from grdl_te.benchmarking.models import (
    AggregatedMetrics,
    BenchmarkRecord,
    HardwareSnapshot,
    StepBenchmarkResult,
)

LINE_WIDTH = 80
RULE_CHAR = "="
THIN_RULE_CHAR = "-"


# ---------------------------------------------------------------------------
# Unit formatting helpers
# ---------------------------------------------------------------------------
def _fmt_bytes(value_bytes: float) -> str:
    """Format a byte count as a human-readable string.

    Parameters
    ----------
    value_bytes : float
        Value in bytes.

    Returns
    -------
    str
        Formatted string with unit suffix (B, KB, MB, or GB).
    """
    abs_val = abs(value_bytes)
    if abs_val < 1024:
        return f"{value_bytes:.0f} B"
    if abs_val < 1024 ** 2:
        return f"{value_bytes / 1024:.1f} KB"
    if abs_val < 1024 ** 3:
        return f"{value_bytes / 1024 ** 2:.1f} MB"
    return f"{value_bytes / 1024 ** 3:.2f} GB"


def _fmt_time(seconds: float) -> str:
    """Format a duration in seconds with appropriate precision.

    Parameters
    ----------
    seconds : float
        Duration in seconds.

    Returns
    -------
    str
        Formatted string with ``s`` suffix.
    """
    if abs(seconds) < 1.0:
        return f"{seconds:.4f}s"
    return f"{seconds:.2f}s"


# ---------------------------------------------------------------------------
# Section formatters
# ---------------------------------------------------------------------------
def _format_header(record_count: int) -> List[str]:
    """Build the report header section.

    Parameters
    ----------
    record_count : int
        Number of benchmark records in the report.

    Returns
    -------
    List[str]
    """
    now = datetime.now(timezone.utc).isoformat()
    return [
        RULE_CHAR * LINE_WIDTH,
        f"  GRDL BENCHMARK REPORT",
        f"  Generated: {now}",
        f"  Records:   {record_count}",
        RULE_CHAR * LINE_WIDTH,
    ]


def _format_hardware(hardware: HardwareSnapshot) -> List[str]:
    """Build the hardware summary section.

    Parameters
    ----------
    hardware : HardwareSnapshot
        Hardware snapshot from the benchmark run.

    Returns
    -------
    List[str]
    """
    mem_str = _fmt_bytes(hardware.total_memory_bytes)
    python_short = hardware.python_version.split("\n")[0]

    lines = [
        "",
        f"  HARDWARE",
        f"  {THIN_RULE_CHAR * 40}",
        f"  Hostname:        {hardware.hostname}",
        f"  Platform:        {hardware.platform_info}",
        f"  Python:          {python_short}",
        f"  CPUs:            {hardware.cpu_count}",
        f"  System Memory:   {mem_str}",
        f"  GPU Available:   {'Yes' if hardware.gpu_available else 'No'}",
    ]

    for dev in hardware.gpu_devices:
        name = dev.get("name", "Unknown")
        mem = dev.get("memory_bytes", 0)
        idx = dev.get("device_index", "?")
        lines.append(f"    Device {idx}:      {name} ({_fmt_bytes(mem)})")

    lines.append(f"  Captured:        {hardware.captured_at}")

    return lines


def _format_configuration(records: List[BenchmarkRecord]) -> List[str]:
    """Build the run configuration section.

    Parameters
    ----------
    records : List[BenchmarkRecord]
        All benchmark records.

    Returns
    -------
    List[str]
    """
    first = records[0]

    # Collect unique benchmark types
    types = sorted({r.benchmark_type for r in records})

    # Extract array size from tags
    size_tag = first.tags.get("array_size", "unknown")
    rows_tag = first.tags.get("rows", "?")
    cols_tag = first.tags.get("cols", "?")

    # Time span
    timestamps = [r.created_at for r in records if r.created_at]
    time_start = min(timestamps) if timestamps else "N/A"
    time_end = max(timestamps) if timestamps else "N/A"

    lines = [
        "",
        f"  RUN CONFIGURATION",
        f"  {THIN_RULE_CHAR * 40}",
        f"  Iterations:      {first.iterations}",
        f"  Array Size:      {size_tag} ({rows_tag} x {cols_tag})",
        f"  Benchmark Types: {', '.join(types)}",
        f"  Total Records:   {len(records)}",
        f"  Time Span:       {time_start}",
        f"                   to {time_end}",
    ]
    return lines


def _format_metrics_table(
    wall: AggregatedMetrics,
    cpu: AggregatedMetrics,
    mem: AggregatedMetrics,
    indent: str = "     ",
) -> List[str]:
    """Build an aligned statistics table for wall/cpu/memory metrics.

    Parameters
    ----------
    wall : AggregatedMetrics
        Wall-clock time statistics in seconds.
    cpu : AggregatedMetrics
        CPU time statistics in seconds.
    mem : AggregatedMetrics
        Memory statistics in bytes.
    indent : str
        Whitespace prefix for each line.

    Returns
    -------
    List[str]
    """
    hdr = (
        f"{indent}{'':12s}  {'Mean':>10s}  {'Median':>10s}  "
        f"{'StdDev':>10s}  {'P95':>10s}  {'Min':>10s}  {'Max':>10s}"
    )
    sep = f"{indent}{THIN_RULE_CHAR * 78}"

    def _row(label: str, m: AggregatedMetrics, divisor: float = 1.0,
             fmt: str = ".4f") -> str:
        return (
            f"{indent}{label:<12s}  "
            f"{m.mean / divisor:>10{fmt}}  "
            f"{m.median / divisor:>10{fmt}}  "
            f"{m.stddev / divisor:>10{fmt}}  "
            f"{m.p95 / divisor:>10{fmt}}  "
            f"{m.min / divisor:>10{fmt}}  "
            f"{m.max / divisor:>10{fmt}}"
        )

    return [
        hdr,
        sep,
        _row("Wall (s)", wall),
        _row("CPU  (s)", cpu),
        _row("Mem  (KB)", mem, divisor=1024.0, fmt=".1f"),
    ]


def _format_metrics_table_partial(
    wall: AggregatedMetrics,
    cpu: AggregatedMetrics,
    mem: AggregatedMetrics,
    indent: str = "     ",
) -> List[str]:
    """Build a statistics table for a concurrent step.

    Wall and CPU rows are shown normally.  The memory row shows only
    the mean value with a ``(shared)`` annotation because per-step
    memory cannot be isolated when threads share a process.

    Parameters
    ----------
    wall : AggregatedMetrics
        Wall-clock time statistics in seconds.
    cpu : AggregatedMetrics
        CPU time statistics in seconds.
    mem : AggregatedMetrics
        Level-wide peak memory (shared across concurrent steps).
    indent : str
        Whitespace prefix for each line.

    Returns
    -------
    List[str]
    """
    hdr = (
        f"{indent}{'':12s}  {'Mean':>10s}  {'Median':>10s}  "
        f"{'StdDev':>10s}  {'P95':>10s}  {'Min':>10s}  {'Max':>10s}"
    )
    sep = f"{indent}{THIN_RULE_CHAR * 78}"

    def _row(label: str, m: AggregatedMetrics, divisor: float = 1.0,
             fmt: str = ".4f") -> str:
        return (
            f"{indent}{label:<12s}  "
            f"{m.mean / divisor:>10{fmt}}  "
            f"{m.median / divisor:>10{fmt}}  "
            f"{m.stddev / divisor:>10{fmt}}  "
            f"{m.p95 / divisor:>10{fmt}}  "
            f"{m.min / divisor:>10{fmt}}  "
            f"{m.max / divisor:>10{fmt}}"
        )

    mem_kb = mem.mean / 1024.0
    if mem_kb >= 1024 * 1024:
        mem_str = f"{mem_kb / 1024 / 1024:.2f} GB"
    elif mem_kb >= 1024:
        mem_str = f"{mem_kb / 1024:.1f} MB"
    else:
        mem_str = f"{mem_kb:.1f} KB"

    return [
        hdr,
        sep,
        _row("Wall (s)", wall),
        _row("CPU  (s)", cpu),
        f"{indent}{'Mem':12s}  {mem_str:>10s}  (shared across parallel steps)",
    ]


def _step_was_skipped(step: StepBenchmarkResult) -> bool:
    """Return True if a step was skipped (all wall and CPU times are zero)."""
    return step.wall_time_s.max == 0.0 and step.cpu_time_s.max == 0.0


def _build_branch_chains(
    step_results: List[StepBenchmarkResult],
) -> List[List[StepBenchmarkResult]]:
    """Group steps into branch chains using dependency information.

    A branch chain is the path from a root step (no intra-workflow
    parents) through all sequential descendants that have a single
    parent.  Steps with multiple parents (merge points) terminate a
    chain.

    Returns an empty list when branch structure cannot be determined
    (missing ``step_id`` or ``depends_on`` data).

    Parameters
    ----------
    step_results : List[StepBenchmarkResult]

    Returns
    -------
    List[List[StepBenchmarkResult]]
        One list per branch, ordered from root to leaf.
    """
    active = [s for s in step_results if not _step_was_skipped(s)]
    if not all(s.step_id for s in active):
        return []
    if not any(s.depends_on is not None for s in active):
        return []

    by_id: Dict[str, StepBenchmarkResult] = {s.step_id: s for s in active}

    # Forward edges: parent → [children]
    children: Dict[str, List[str]] = {sid: [] for sid in by_id}
    for sid, step in by_id.items():
        for dep in (step.depends_on or []):
            if dep in children:
                children[dep].append(sid)

    def n_intra_parents(sid: str) -> int:
        return sum(1 for d in (by_id[sid].depends_on or []) if d in by_id)

    roots = [s for s in active if n_intra_parents(s.step_id) == 0]
    if len(roots) < 2:
        return []

    def walk_chain(root_id: str) -> List[str]:
        chain = [root_id]
        current = root_id
        while True:
            kids = children.get(current, [])
            if len(kids) == 1 and n_intra_parents(kids[0]) == 1:
                chain.append(kids[0])
                current = kids[0]
            else:
                break
        return chain

    return [[by_id[sid] for sid in walk_chain(r.step_id)] for r in roots]


def _format_branch_chains(record: BenchmarkRecord) -> List[str]:
    """Build the parallel-branch comparison block for a single record.

    Emitted only when the record has concurrent steps and dependency
    info is present to reconstruct branch chains.

    Parameters
    ----------
    record : BenchmarkRecord

    Returns
    -------
    List[str]
    """
    has_parallel = any(
        s.concurrent for s in record.step_results if not _step_was_skipped(s)
    )
    if not has_parallel:
        return []

    chains = _build_branch_chains(record.step_results)
    if len(chains) < 2:
        return []

    chain_times = [sum(s.wall_time_s.mean for s in c) for c in chains]
    critical_time = max(chain_times)

    active = [s for s in record.step_results if not _step_was_skipped(s)]
    # Sum of step wall times measured *during* parallel execution.  Because
    # steps competed for CPU/memory while co-running, each individual time is
    # inflated relative to what that step would take in isolation.  The ratio
    # below is NOT a true speedup — compare the parallel wall time against a
    # dedicated sequential benchmark to get the real wall-clock gain.
    contended_sum = sum(s.wall_time_s.mean for s in active)
    actual_wall = record.total_wall_time.mean
    contended_ratio = contended_sum / actual_wall if actual_wall > 0 else 1.0

    indent = "     "
    lines = [
        "",
        f"{indent}PARALLEL BRANCHES ({len(chains)} branches,"
        f" critical path: {_fmt_time(critical_time)})",
        f"{indent}{THIN_RULE_CHAR * 72}",
    ]

    for i, (chain, ct) in enumerate(zip(chains, chain_times)):
        step_names = " -> ".join(s.processor_name for s in chain)
        if ct >= critical_time - 1e-9:
            suffix = "  [critical path *]"
        else:
            suffix = f"  (idle {_fmt_time(critical_time - ct)})"
        lines.append(f"{indent}  Branch {i + 1}: {step_names}")
        lines.append(f"{indent}             chain mean: {_fmt_time(ct)}{suffix}")

    lines.extend([
        "",
        f"{indent}Contended step-sum: {_fmt_time(contended_sum)}"
        f"  ->  Parallel: {_fmt_time(actual_wall)}"
        f"  |  Ratio: {contended_ratio:.2f}x",
        f"{indent}(Step times measured under resource contention."
        f" For true speedup, compare parallel wall time to a sequential benchmark.)",
    ])

    return lines


def _format_step_detail(step: StepBenchmarkResult) -> List[str]:
    """Build the detail block for a single workflow step.

    Parameters
    ----------
    step : StepBenchmarkResult
        Per-step aggregated metrics.

    Returns
    -------
    List[str]
    """
    indent = "       "
    gpu_tag = f"GPU: {'Yes' if step.gpu_used else 'No'}"
    lines = [
        f"       [{step.step_index}] {step.processor_name}"
        f"  ({gpu_tag})",
    ]
    if step.concurrent:
        # Concurrent steps: show wall/CPU per-step (accurate) but
        # annotate memory as the shared level-wide peak.
        lines.extend(
            _format_metrics_table_partial(
                step.wall_time_s, step.cpu_time_s, step.peak_rss_bytes,
                indent=indent,
            )
        )
    else:
        lines.extend(
            _format_metrics_table(
                step.wall_time_s, step.cpu_time_s, step.peak_rss_bytes,
                indent=indent,
            )
        )
    if step.gpu_memory_bytes is not None:
        gm = step.gpu_memory_bytes
        lines.append(
            f"{indent}GPU Mem (KB)  "
            f"{gm.mean / 1024:>10.1f}  "
            f"{gm.median / 1024:>10.1f}  "
            f"{gm.stddev / 1024:>10.1f}  "
            f"{gm.p95 / 1024:>10.1f}  "
            f"{gm.min / 1024:>10.1f}  "
            f"{gm.max / 1024:>10.1f}"
        )
    return lines


def _format_record_detail(
    index: int,
    record: BenchmarkRecord,
) -> List[str]:
    """Build the detail block for a single benchmark record.

    Parameters
    ----------
    index : int
        One-based display index.
    record : BenchmarkRecord
        The benchmark record.

    Returns
    -------
    List[str]
    """
    lines = [
        "",
        f"  {index}. {record.workflow_name}",
        f"     Type: {record.benchmark_type}    "
        f"Version: {record.workflow_version}    "
        f"Iterations: {record.iterations}",
    ]

    if record.tags:
        tag_str = ", ".join(f"{k}={v}" for k, v in sorted(record.tags.items()))
        lines.append(f"     Tags: {tag_str}")

    lines.append(f"     {THIN_RULE_CHAR * 72}")

    lines.extend(
        _format_metrics_table(
            record.total_wall_time,
            record.total_cpu_time,
            record.total_peak_rss,
        )
    )

    lines.extend(_format_branch_chains(record))

    if record.step_results:
        ran = [s for s in record.step_results if not _step_was_skipped(s)]
        skipped = [s for s in record.step_results if _step_was_skipped(s)]
        parallel = [s for s in ran if s.concurrent]

        lines.append("")
        summary_parts = [f"{len(ran)} ran"]
        if parallel:
            summary_parts.append(f"{len(parallel)} parallel")
        summary_parts.append(f"{len(skipped)} skipped")
        lines.append(
            f"     Steps ({', '.join(summary_parts)}"
            f" / {len(record.step_results)} total):"
        )
        for step in record.step_results:
            if _step_was_skipped(step):
                lines.append(
                    f"       [{step.step_index}] {step.processor_name}"
                    f"  -- SKIPPED (condition not met)"
                )
            else:
                lines.extend(_format_step_detail(step))

    return lines


def _format_detailed_results(
    sorted_records: List[BenchmarkRecord],
) -> List[str]:
    """Build the detailed results section for all records.

    Parameters
    ----------
    sorted_records : List[BenchmarkRecord]
        Records sorted by wall time descending (slowest first).

    Returns
    -------
    List[str]
    """
    lines = [
        "",
        RULE_CHAR * LINE_WIDTH,
        f"  DETAILED RESULTS",
        RULE_CHAR * LINE_WIDTH,
    ]

    for i, record in enumerate(sorted_records, start=1):
        lines.extend(_format_record_detail(i, record))

    return lines


def _format_module_summary(records: List[BenchmarkRecord]) -> List[str]:
    """Build the module-level aggregation section.

    Parameters
    ----------
    records : List[BenchmarkRecord]
        All benchmark records.

    Returns
    -------
    List[str]
    """
    module_data: Dict[str, List[float]] = {}
    for record in records:
        mod = record.tags.get("module", "unknown")
        module_data.setdefault(mod, []).append(record.total_wall_time.mean)

    if len(module_data) <= 1:
        return []

    lines = [
        "",
        RULE_CHAR * LINE_WIDTH,
        f"  MODULE SUMMARY",
        RULE_CHAR * LINE_WIDTH,
        "",
        f"  {'Module':<40s}  {'Count':>6s}  "
        f"{'Total Wall (s)':>14s}  {'Avg Wall (s)':>12s}",
        f"  {THIN_RULE_CHAR * 40}  {THIN_RULE_CHAR * 6}  "
        f"{THIN_RULE_CHAR * 14}  {THIN_RULE_CHAR * 12}",
    ]

    sorted_modules = sorted(
        module_data.items(),
        key=lambda item: sum(item[1]),
        reverse=True,
    )

    for mod, times in sorted_modules:
        total = sum(times)
        avg = total / len(times)
        lines.append(
            f"  {mod:<40s}  {len(times):>6d}  "
            f"{total:>14.4f}  {avg:>12.4f}"
        )

    return lines


def _format_overall_summary(records: List[BenchmarkRecord]) -> List[str]:
    """Build the overall summary section.

    For a single record the summary shows the iteration-level spread
    (min/max wall time and memory across iterations).  For multiple
    records it compares across benchmarks.

    Parameters
    ----------
    records : List[BenchmarkRecord]
        All benchmark records.

    Returns
    -------
    List[str]
    """
    lines = [
        "",
        RULE_CHAR * LINE_WIDTH,
        f"  OVERALL SUMMARY",
        RULE_CHAR * LINE_WIDTH,
        "",
    ]

    if len(records) == 1:
        r = records[0]
        w = r.total_wall_time
        c = r.total_cpu_time
        m = r.total_peak_rss
        lines.extend([
            f"  Workflow:           {r.workflow_name} v{r.workflow_version}",
            f"  Iterations:         {r.iterations}",
            f"  Wall Time:          {_fmt_time(w.mean)} mean"
            f"  |  {_fmt_time(w.min)} min  |  {_fmt_time(w.max)} max"
            f"  |  {_fmt_time(w.stddev)} stddev",
            f"  CPU Time:           {_fmt_time(c.mean)} mean"
            f"  |  {_fmt_time(c.min)} min  |  {_fmt_time(c.max)} max"
            f"  |  {_fmt_time(c.stddev)} stddev",
            f"  Peak Memory:        {_fmt_bytes(m.mean)} mean"
            f"  |  {_fmt_bytes(m.min)} min  |  {_fmt_bytes(m.max)} max",
        ])
    else:
        wall_means = np.array([r.total_wall_time.mean for r in records])
        total_wall = float(np.sum(wall_means))
        median_wall = float(np.median(wall_means))
        mean_wall = float(np.mean(wall_means))

        fastest = min(records, key=lambda r: r.total_wall_time.mean)
        slowest = max(records, key=lambda r: r.total_wall_time.mean)
        least_mem = min(records, key=lambda r: r.total_peak_rss.mean)
        most_mem = max(records, key=lambda r: r.total_peak_rss.mean)

        lines.extend([
            f"  Total Benchmarks:   {len(records)}",
            f"  Total Wall Time:    {_fmt_time(total_wall)}",
            f"  Mean Wall Time:     {_fmt_time(mean_wall)}",
            f"  Median Wall Time:   {_fmt_time(median_wall)}",
            f"  Fastest:            {fastest.workflow_name} "
            f"({_fmt_time(fastest.total_wall_time.mean)})",
            f"  Slowest:            {slowest.workflow_name} "
            f"({_fmt_time(slowest.total_wall_time.mean)})",
            f"  Least Peak Mem:     {least_mem.workflow_name} "
            f"({_fmt_bytes(least_mem.total_peak_rss.mean)})",
            f"  Most Peak Mem:      {most_mem.workflow_name} "
            f"({_fmt_bytes(most_mem.total_peak_rss.mean)})",
        ])

    lines.extend([
        "",
        RULE_CHAR * LINE_WIDTH,
    ])

    return lines


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def format_report(records: List[BenchmarkRecord]) -> str:
    """Generate a comprehensive benchmark report as a string.

    Produces a human-readable text report covering hardware configuration,
    run parameters, per-benchmark statistical breakdowns, per-step details,
    module-level aggregation, and an overall summary.

    Parameters
    ----------
    records : List[BenchmarkRecord]
        Benchmark records to include in the report.  Must not be empty.

    Returns
    -------
    str
        The complete formatted report text.

    Raises
    ------
    ValueError
        If *records* is empty.
    """
    if not records:
        raise ValueError("Cannot generate report from empty records list.")

    lines: List[str] = []

    lines.extend(_format_header(len(records)))
    lines.extend(_format_hardware(records[0].hardware))
    lines.extend(_format_configuration(records))

    # Sort by wall time descending (slowest first)
    sorted_records = sorted(
        records,
        key=lambda r: r.total_wall_time.mean,
        reverse=True,
    )

    lines.extend(_format_detailed_results(sorted_records))
    lines.extend(_format_module_summary(records))
    lines.extend(_format_overall_summary(records))

    return "\n".join(lines)


def print_report(records: List[BenchmarkRecord]) -> None:
    """Print a comprehensive benchmark report to stdout.

    Parameters
    ----------
    records : List[BenchmarkRecord]
        Benchmark records to include in the report.
    """
    print(format_report(records))


def save_report(records: List[BenchmarkRecord], path: Path) -> Path:
    """Write a comprehensive benchmark report to a file.

    Creates parent directories if they do not exist.  If *path* is an
    existing directory (or has no file extension and does not exist), a
    timestamped filename ``benchmark_report_YYYYMMDD_HHMMSS.txt`` is
    generated inside it.

    Parameters
    ----------
    records : List[BenchmarkRecord]
        Benchmark records to include in the report.
    path : Path
        Output file path or directory.

    Returns
    -------
    Path
        The path to the written report file.
    """
    report_text = format_report(records)
    path = Path(path)

    if path.is_dir() or (not path.suffix and not path.exists()):
        path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = path / f"benchmark_report_{timestamp}.txt"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(report_text, encoding="utf-8")
    return path
