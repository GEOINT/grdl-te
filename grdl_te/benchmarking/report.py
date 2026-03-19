# -*- coding: utf-8 -*-
"""
Benchmark Report Generator — comprehensive human-readable reports.

Formats benchmark results into a detailed text report covering an
executive summary with bottleneck identification, hardware configuration,
per-benchmark statistics, per-step breakdowns with latency and memory
contribution percentages, time decomposition, branch analysis, memory
profile, and overall summary.  Reports can be printed to stdout or
written to a file.

This module is a thin presenter.  All data derivations are performed
by ``report_engine.build_report_data()``.

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
2026-03-19
"""

# Standard library
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Internal
from grdl_te.benchmarking._formatting import (
    fmt_bytes as _fmt_bytes,
    fmt_throughput as _fmt_throughput,
    fmt_time as _fmt_time,
)
from grdl_te.benchmarking.models import (
    AggregatedMetrics,
    BenchmarkRecord,
    HardwareSnapshot,
    StepBenchmarkResult,
)
from grdl_te.benchmarking.report_engine import (
    RecordReportData,
    ReportData,
    StepReportData,
    build_branch_chains,
    build_report_data,
    step_was_skipped,
)

LINE_WIDTH = 80
RULE_CHAR = "="
THIN_RULE_CHAR = "-"


# ---------------------------------------------------------------------------
# Backward-compatible wrappers (used by report_md.py / report_gui.py)
# ---------------------------------------------------------------------------
def _step_was_skipped(step: StepBenchmarkResult) -> bool:
    """Return True if a step was skipped (all wall and CPU times are zero)."""
    return step_was_skipped(step)


def _build_branch_chains(
    step_results: List[StepBenchmarkResult],
) -> List[List[StepBenchmarkResult]]:
    """Group steps into branch chains using dependency information.

    Delegates to ``report_engine.build_branch_chains``.
    """
    return build_branch_chains(step_results)


# ---------------------------------------------------------------------------
# Section formatters
# ---------------------------------------------------------------------------
def _format_header(record_count: int) -> List[str]:
    """Build the report header section."""
    now = datetime.now(timezone.utc).isoformat()
    return [
        RULE_CHAR * LINE_WIDTH,
        f"  GRDL BENCHMARK REPORT",
        f"  Generated: {now}",
        f"  Records:   {record_count}",
        RULE_CHAR * LINE_WIDTH,
    ]


def _format_hardware(hardware: HardwareSnapshot) -> List[str]:
    """Build the hardware summary section."""
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
    """Build the run configuration section."""
    first = records[0]

    types = sorted({r.benchmark_type for r in records})

    size_tag = first.tags.get("array_size", "")
    rows_tag = first.tags.get("rows", "?")
    cols_tag = first.tags.get("cols", "?")

    timestamps = [r.created_at for r in records if r.created_at]
    time_start = min(timestamps) if timestamps else "N/A"
    time_end = max(timestamps) if timestamps else "N/A"

    lines = [
        "",
        f"  RUN CONFIGURATION",
        f"  {THIN_RULE_CHAR * 40}",
        f"  Iterations:      {first.iterations}",
    ]
    if size_tag:
        lines.append(f"  Array Size:      {size_tag} ({rows_tag} x {cols_tag})")
    lines.extend([
        f"  Benchmark Types: {', '.join(types)}",
        f"  Total Records:   {len(records)}",
        f"  Time Span:       {time_start}",
        f"                   to {time_end}",
    ])
    return lines


def _format_executive_summary(data: ReportData) -> List[str]:
    """Build the executive summary with bottleneck identification."""
    lines = [
        "",
        RULE_CHAR * LINE_WIDTH,
        f"  EXECUTIVE SUMMARY",
        RULE_CHAR * LINE_WIDTH,
        "",
    ]

    bns = data.bottlenecks
    if not bns:
        lines.append("  No bottlenecks identified.")
        return lines

    top = bns[0]
    lines.append(
        f"  Top Bottleneck: {top.step_name} accounts for "
        f"{top.latency_pct:.1f}% of latency"
        f" ({_fmt_time(top.wall_time_s)} mean wall time)"
        f" in {top.workflow}."
    )
    lines.append("")

    col_rank = 4
    col_step = 24
    col_lat = 10
    col_wf = 20
    col_wall = 10

    lines.append(
        f"  {'Rank':<{col_rank}}  {'Step':<{col_step}}  "
        f"{'Latency%':>{col_lat}}  "
        f"{'Workflow':<{col_wf}}  {'Mean Wall':>{col_wall}}"
    )
    lines.append(
        f"  {THIN_RULE_CHAR * col_rank}  {THIN_RULE_CHAR * col_step}  "
        f"{THIN_RULE_CHAR * col_lat}  "
        f"{THIN_RULE_CHAR * col_wf}  {THIN_RULE_CHAR * col_wall}"
    )

    for bn in bns:
        lat = "--" if bn.latency_pct == 0.0 else f"{bn.latency_pct:.1f}%"
        step_name = bn.step_name
        if len(step_name) > col_step:
            step_name = step_name[:col_step - 3] + "..."
        wf = bn.workflow
        if len(wf) > col_wf:
            wf = wf[:col_wf - 3] + "..."
        lines.append(
            f"  {bn.rank:<{col_rank}}  {step_name:<{col_step}}  "
            f"{lat:>{col_lat}}  "
            f"{wf:<{col_wf}}  {_fmt_time(bn.wall_time_s):>{col_wall}}"
        )

    return lines


def _format_time_decomposition(rd: RecordReportData) -> List[str]:
    """Build time decomposition section for parallel/mixed workflows."""
    td = rd.time_decomposition
    if td is None:
        return []

    indent = "     "
    return [
        "",
        f"{indent}TIME DECOMPOSITION",
        f"{indent}{THIN_RULE_CHAR * 50}",
        f"{indent}  Wall Clock (actual elapsed):   {_fmt_time(td.wall_clock_s)}",
        f"{indent}  Critical Path (longest chain): {_fmt_time(td.critical_path_s)}",
        f"{indent}  Contended Step Sum:            {_fmt_time(td.contended_step_sum_s)}",
        f"{indent}  (Step times measured under resource contention —"
        f" not a valid sequential baseline.)",
    ]


def _format_metrics_table(
    wall: AggregatedMetrics,
    cpu: AggregatedMetrics,
    indent: str = "     ",
    iterations: int = 1,
) -> List[str]:
    """Build an aligned statistics table for wall/cpu metrics."""
    # N=1: scalar layout
    if wall.count == 1:
        return [
            f"{indent}Wall (s):    {wall.mean:.4f}",
            f"{indent}CPU  (s):    {cpu.mean:.4f}",
        ]

    show_p95 = iterations > 10

    if show_p95:
        hdr = (
            f"{indent}{'':12s}  {'Mean':>10s}  {'Median':>10s}  "
            f"{'StdDev':>10s}  {'P95':>10s}  {'Min':>10s}  {'Max':>10s}"
        )
        sep = f"{indent}{THIN_RULE_CHAR * 78}"

        def _row(label: str, m: AggregatedMetrics) -> str:
            return (
                f"{indent}{label:<12s}  "
                f"{m.mean:>10.4f}  "
                f"{m.median:>10.4f}  "
                f"{m.stddev:>10.4f}  "
                f"{m.p95:>10.4f}  "
                f"{m.min:>10.4f}  "
                f"{m.max:>10.4f}"
            )
    else:
        hdr = (
            f"{indent}{'':12s}  {'Mean':>10s}  {'Median':>10s}  "
            f"{'StdDev':>10s}  {'Min':>10s}  {'Max':>10s}"
        )
        sep = f"{indent}{THIN_RULE_CHAR * 66}"

        def _row(label: str, m: AggregatedMetrics) -> str:
            return (
                f"{indent}{label:<12s}  "
                f"{m.mean:>10.4f}  "
                f"{m.median:>10.4f}  "
                f"{m.stddev:>10.4f}  "
                f"{m.min:>10.4f}  "
                f"{m.max:>10.4f}"
            )

    return [
        hdr,
        sep,
        _row("Wall (s)", wall),
        _row("CPU  (s)", cpu),
    ]


def _format_step_detail(
    sd: StepReportData,
    iterations: int = 1,
    show_gpu: bool = False,
) -> List[str]:
    """Build the detail block for a single workflow step."""
    indent = "       "
    gpu_tag = f"  GPU: {'Yes' if sd.gpu_used else 'No'}" if show_gpu else ""
    contended_tag = "  [contended]" if sd.concurrent else ""
    path_tag = f"  [{sd.path_classification}]"
    lines = [
        f"       [{sd.step_index}] {sd.processor_name}"
        f"{contended_tag}{path_tag}{gpu_tag}",
    ]

    # Latency%
    if sd.concurrent and not sd.on_critical_path:
        lat_str = "--"
    else:
        lat_str = f"{sd.latency_pct:.1f}%"
    lines.append(f"{indent}Latency: {lat_str}")

    lines.extend(
        _format_metrics_table(
            sd.wall_time, sd.cpu_time,
            indent=indent,
            iterations=iterations,
        )
    )

    # Throughput — from pre-computed engine data
    if sd.throughput_scalar is not None:
        if iterations == 1:
            lines.append(f"{indent}Throughput:  {_fmt_throughput(sd.throughput_scalar)}")
        elif sd.throughput_stats is not None:
            lines.append(
                f"{indent}Throughput:  {_fmt_throughput(sd.throughput_stats.mean)} mean"
                f"  |  {_fmt_throughput(sd.throughput_stats.min)} min"
                f"  |  {_fmt_throughput(sd.throughput_stats.max)} max"
            )

    return lines


def _format_branch_chains_from_data(rd: RecordReportData) -> List[str]:
    """Build the parallel-branch comparison block from pre-computed data."""
    branches = rd.branches
    if len(branches) < 2:
        return []

    critical_time = max(b.chain_time_s for b in branches)

    rec = rd.record
    active = [s for s in rd.steps if not s.skipped]
    contended_sum = sum(s.wall_time.mean for s in active)
    actual_wall = rec.total_wall_time.mean

    indent = "     "
    lines = [
        "",
        f"{indent}PARALLEL BRANCHES ({len(branches)} branches,"
        f" critical path: {_fmt_time(critical_time)})",
        f"{indent}{THIN_RULE_CHAR * 72}",
    ]

    for b in branches:
        step_names = " -> ".join(s.processor_name for s in b.steps)
        if b.is_critical:
            suffix = "  [critical path *]"
        else:
            suffix = f"  (idle {_fmt_time(b.idle_time_s)})"
        lines.append(f"{indent}  Branch {b.branch_index}: {step_names}")
        lines.append(f"{indent}             chain mean: {_fmt_time(b.chain_time_s)}{suffix}")

    lines.extend([
        "",
        f"{indent}Contended Step Sum: {_fmt_time(contended_sum)}"
        f"  ->  Parallel Wall: {_fmt_time(actual_wall)}",
        f"{indent}(Step times measured under resource contention."
        f" For true speedup, compare parallel wall time to a sequential benchmark.)",
    ])

    return lines


def _format_record_detail(
    index: int,
    rd: RecordReportData,
) -> List[str]:
    """Build the detail block for a single benchmark record."""
    rec = rd.record
    lines = [
        "",
        f"  {index}. {rd.workflow_name} ({rd.topology_label})",
        f"     Type: {rd.benchmark_type}    "
        f"Version: {rd.workflow_version}    "
        f"Iterations: {rd.iterations}",
        f"     Wall: {_fmt_time(rec.total_wall_time.mean)}    "
        f"CPU: {_fmt_time(rec.total_cpu_time.mean)}    "
        f"Memory: {_fmt_bytes(rec.total_peak_rss.mean)}",
    ]

    if rec.tags:
        tag_str = ", ".join(f"{k}={v}" for k, v in sorted(rec.tags.items()))
        lines.append(f"     Tags: {tag_str}")

    lines.append(f"     {THIN_RULE_CHAR * 72}")

    # Time decomposition (new — parity with MD/GUI)
    lines.extend(_format_time_decomposition(rd))

    # Branch analysis
    lines.extend(_format_branch_chains_from_data(rd))

    if rd.steps:
        has_gpu = any(s.gpu_used for s in rd.steps if not s.skipped)

        lines.append("")
        if rd.skipped_step_count or rd.parallel_step_count:
            summary_parts = [f"{rd.active_step_count} ran"]
            if rd.parallel_step_count:
                summary_parts.append(f"{rd.parallel_step_count} parallel")
            if rd.skipped_step_count:
                summary_parts.append(f"{rd.skipped_step_count} skipped")
            lines.append(
                f"     Steps ({', '.join(summary_parts)}"
                f" / {len(rd.steps)} total):"
            )
        else:
            lines.append(f"     Steps:")

        for sd in rd.steps:
            if sd.skipped:
                lines.append(
                    f"       [{sd.step_index}] {sd.processor_name}"
                    f"  -- SKIPPED (condition not met)"
                )
            else:
                lines.extend(_format_step_detail(
                    sd, iterations=rd.iterations, show_gpu=has_gpu,
                ))

    return lines


def _format_detailed_results(data: ReportData) -> List[str]:
    """Build the detailed results section for all records."""
    lines = [
        "",
        RULE_CHAR * LINE_WIDTH,
        f"  DETAILED RESULTS",
        RULE_CHAR * LINE_WIDTH,
    ]

    for i, rd in enumerate(data.records, start=1):
        lines.extend(_format_record_detail(i, rd))

    return lines


def _format_comparison(data: ReportData) -> List[str]:
    """Build the workflow comparison section for multi-record reports."""
    col_name = 34
    col_topo = 12
    col_time = 10
    col_mem = 12

    lines = [
        "",
        RULE_CHAR * LINE_WIDTH,
        f"  WORKFLOW COMPARISON",
        RULE_CHAR * LINE_WIDTH,
        "",
        f"  {'Workflow':<{col_name}}  {'Topology':<{col_topo}}  "
        f"{'Wall Time':>{col_time}}  {'CPU Time':>{col_time}}  "
        f"{'Peak Memory':>{col_mem}}  Steps",
        f"  {THIN_RULE_CHAR * col_name}  {THIN_RULE_CHAR * col_topo}  "
        f"{THIN_RULE_CHAR * col_time}  {THIN_RULE_CHAR * col_time}  "
        f"{THIN_RULE_CHAR * col_mem}  {THIN_RULE_CHAR * 5}",
    ]

    for rd in data.records:
        rec = rd.record
        name = rd.workflow_name
        if len(name) > col_name:
            name = name[:col_name - 3] + "..."
        lines.append(
            f"  {name:<{col_name}}  {rd.topology_label:<{col_topo}}  "
            f"{_fmt_time(rec.total_wall_time.mean):>{col_time}}  "
            f"{_fmt_time(rec.total_cpu_time.mean):>{col_time}}  "
            f"{_fmt_bytes(rec.total_peak_rss.mean):>{col_mem}}  {rd.active_step_count}"
        )

    return lines


def _format_overall_summary(data: ReportData) -> List[str]:
    """Build the overall summary section."""
    lines = [
        "",
        RULE_CHAR * LINE_WIDTH,
        f"  OVERALL SUMMARY",
        RULE_CHAR * LINE_WIDTH,
        "",
    ]

    if data.record_count == 1:
        rd = data.records[0]
        rec = rd.record
        w = rec.total_wall_time
        c = rec.total_cpu_time
        m = rec.total_peak_rss
        if rd.iterations == 1:
            lines.extend([
                f"  Workflow:           {rd.workflow_name} v{rd.workflow_version}",
                f"  Wall Time:          {_fmt_time(w.mean)}",
                f"  CPU Time:           {_fmt_time(c.mean)}",
                f"  Peak Memory:        {_fmt_bytes(m.mean)}",
            ])
        else:
            lines.extend([
                f"  Workflow:           {rd.workflow_name} v{rd.workflow_version}",
                f"  Iterations:         {rd.iterations}",
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
        s = data.overall_summary
        if s is not None:
            lines.extend([
                f"  Total Benchmarks:   {s.total_benchmarks}",
                f"  Total Wall Time:    {_fmt_time(s.total_wall_time_s)}",
                f"  Fastest:            {s.fastest_name} "
                f"({_fmt_time(s.fastest_wall_s)})",
                f"  Slowest:            {s.slowest_name} "
                f"({_fmt_time(s.slowest_wall_s)})",
                f"  Least Memory:       {s.least_memory_name} "
                f"({_fmt_bytes(s.least_memory_bytes)})",
                f"  Most Memory:        {s.most_memory_name} "
                f"({_fmt_bytes(s.most_memory_bytes)})",
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
    run parameters, executive summary with bottleneck identification,
    per-benchmark statistical breakdowns, per-step details with latency
    and memory percentages, time decomposition, workflow comparison
    (for multi-record runs), and an overall summary.

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

    data = build_report_data(records)

    lines: List[str] = []

    lines.extend(_format_header(data.record_count))
    if data.hardware is not None:
        lines.extend(_format_hardware(data.hardware))
    else:
        lines.append("Hardware:        [information missing — traces predate hardware capture]")
    lines.extend(_format_configuration(records))

    # Executive summary (new — parity with MD/GUI)
    lines.extend(_format_executive_summary(data))

    lines.extend(_format_detailed_results(data))
    if data.record_count > 1:
        lines.extend(_format_comparison(data))
        lines.extend(_format_overall_summary(data))

    return "\n".join(lines)


def print_report(records: List[BenchmarkRecord]) -> None:
    """Print a comprehensive benchmark report to stdout."""
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
