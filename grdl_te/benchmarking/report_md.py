# -*- coding: utf-8 -*-
"""
Markdown Benchmark Report Generator — comprehensive Markdown reports.

Formats benchmark results into a detailed Markdown report covering an
executive summary with bottleneck identification, hardware configuration,
per-benchmark step-level breakdowns with latency and memory contribution
percentages, branch analysis, time decomposition, and cross-workflow
comparison tables.

Every report includes full per-step data for all workflows benchmarked.
Comparisons are an additional summary section, never a replacement.

This module is a thin presenter.  All data derivations (bottleneck
ranking, throughput, branch chains, path classification, etc.) are
performed by ``report_engine.build_report_data()``.  Section formatters
here only convert numeric values into Markdown text.

Dependencies
------------
numpy

Author
------
Claude Code (Anthropic)

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2026-03-04

Modified
--------
2026-03-19
"""

# Standard library
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Third-party
import numpy as np

# Internal
from grdl_te.benchmarking._formatting import (
    fmt_bytes as _fmt_bytes,
    fmt_throughput as _fmt_throughput,
    fmt_time as _fmt_time,
    short_name as _short_name,
)
from grdl_te.benchmarking.comparison import ComparisonResult, compare_records
from grdl_te.benchmarking.models import (
    AggregatedMetrics,
    BenchmarkRecord,
    HardwareSnapshot,
    StepBenchmarkResult,
    TopologyDescriptor,
    WorkflowTopology,
)
from grdl_te.benchmarking.report_engine import (
    BottleneckEntry,
    RecordReportData,
    ReportData,
    StepReportData,
    build_report_data,
    compute_throughput as _compute_throughput,
    step_throughput_scalar as _step_throughput_scalar,
    step_throughput_stats as _step_throughput_stats,
    step_was_skipped as _step_was_skipped,
)


# ---------------------------------------------------------------------------
# Section formatters — pure presentation
# ---------------------------------------------------------------------------
def _md_header(record_count: int) -> str:
    """Build the report header."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return (
        f"# GRDL Benchmark Report\n\n"
        f"**Generated**: {now} | **Records**: {record_count}\n"
    )


def _resolve_array_size(record: BenchmarkRecord) -> str:
    """Derive a human-readable array size from record tags."""
    return record.tags.get("array_size", "")


def _md_executive_summary(
    data: ReportData,
) -> str:
    """Build the executive summary from pre-computed bottlenecks."""
    lines = ["## Executive Summary\n"]

    bns = data.bottlenecks
    if bns:
        top = bns[0]
        lines.append(
            f"> **Top Bottleneck**: `{top.step_name}` accounts for "
            f"**{top.latency_pct:.1f}%** of latency"
            f" ({_fmt_time(top.wall_time_s)} mean wall time)"
            f" in *{top.workflow}*.\n"
        )
        lines.append("| Rank | Step | Latency % | Workflow | Mean Wall |")
        lines.append("|------|------|-----------|----------|-----------|")
        for bn in bns:
            lat = "--" if bn.latency_pct == 0.0 else f"{bn.latency_pct:.1f}%"
            lines.append(
                f"| {bn.rank} | `{bn.step_name}` | {lat} | "
                f"{bn.workflow} | "
                f"{_fmt_time(bn.wall_time_s)} |"
            )
        lines.append("")

    return "\n".join(lines)


def _md_hardware(hardware: HardwareSnapshot) -> str:
    """Build the hardware & configuration section."""
    lines = ["## Hardware & Configuration\n"]
    python_short = hardware.python_version.split("\n")[0]

    lines.append("| Property | Value |")
    lines.append("|----------|-------|")
    lines.append(f"| Hostname | {hardware.hostname} |")
    lines.append(f"| Platform | {hardware.platform_info} |")
    lines.append(f"| CPUs | {hardware.cpu_count} |")
    lines.append(f"| System Memory | {_fmt_bytes(hardware.total_memory_bytes)} |")
    lines.append(f"| GPU Available | {'Yes' if hardware.gpu_available else 'No'} |")
    for dev in hardware.gpu_devices:
        name = dev.get("name", "Unknown")
        mem = dev.get("memory_bytes", 0)
        idx = dev.get("device_index", "?")
        lines.append(f"| GPU Device {idx} | {name} ({_fmt_bytes(mem)}) |")
    lines.append(f"| Python | {python_short} |")
    lines.append(f"| Captured | {hardware.captured_at} |")
    lines.append("")

    return "\n".join(lines)


def _md_configuration(records: List[BenchmarkRecord]) -> str:
    """Build the run configuration section."""
    first = records[0]
    array_size = _resolve_array_size(first)

    parts = [f"**Iterations**: {first.iterations}"]
    if array_size:
        parts.append(f"**Array Size**: {array_size}")
    parts.append(f"**Records**: {len(records)}")

    lines = [" | ".join(parts) + "\n"]
    return "\n".join(lines)


def _md_step_table(rd: RecordReportData) -> str:
    """Build per-step performance table from pre-computed step data."""
    active = [s for s in rd.steps if not s.skipped]
    skipped = [s for s in rd.steps if s.skipped]
    parallel = [s for s in active if s.concurrent]

    lines = ["#### Step Performance\n"]

    if skipped or parallel:
        summary_parts = [f"{len(active)} ran"]
        if parallel:
            summary_parts.append(f"{len(parallel)} parallel")
        if skipped:
            summary_parts.append(f"{len(skipped)} skipped")
        lines.append(
            f"*{', '.join(summary_parts)} / "
            f"{len(rd.steps)} total*\n"
        )

    is_single_run = rd.iterations == 1
    show_p95 = rd.iterations > 10
    has_throughput = any(s.throughput_scalar is not None for s in active)
    has_gpu = any(s.gpu_used for s in active)

    # Optional column fragments
    tp_hdr = " Throughput |" if has_throughput else ""
    tp_sep = "------------|" if has_throughput else ""
    gpu_hdr = " GPU |" if has_gpu else ""
    gpu_sep = "-----|" if has_gpu else ""

    # Table header
    if is_single_run:
        lines.append(f"| # | Step | Wall Time |{tp_hdr} Latency% | Path |{gpu_hdr}")
        lines.append(f"|---|------|-----------|{tp_sep}----------|------|{gpu_sep}")
    elif show_p95:
        lines.append(
            f"| # | Step | Mean | Median | StdDev | P95 | Min | Max |"
            f"{tp_hdr} Latency% | Path |{gpu_hdr}"
        )
        lines.append(
            f"|---|------|------|--------|--------|-----|-----|-----|"
            f"{tp_sep}----------|------|{gpu_sep}"
        )
    else:
        lines.append(
            f"| # | Step | Mean | Median | StdDev | Min | Max |"
            f"{tp_hdr} Latency% | Path |{gpu_hdr}"
        )
        lines.append(
            f"|---|------|------|--------|--------|-----|-----|"
            f"{tp_sep}----------|------|{gpu_sep}"
        )

    for step in rd.steps:
        tp_cell = ""
        if has_throughput:
            tp_cell = " -- |"
        gpu_cell = " |" if has_gpu else ""

        if step.skipped:
            if is_single_run:
                skip_dashes = "-- |"
            elif show_p95:
                skip_dashes = "-- | -- | -- | -- | -- | -- |"
            else:
                skip_dashes = "-- | -- | -- | -- | -- |"
            lines.append(
                f"| {step.step_index} | `{step.short_name}` "
                f"| {skip_dashes}{tp_cell} -- | *skipped* |{gpu_cell}"
            )
            continue

        w = step.wall_time
        mean_str = _fmt_time(w.mean)
        if step.concurrent:
            mean_str += " ‡"

        # Throughput cell
        if has_throughput:
            tp_cell = f" {_fmt_throughput(step.throughput_scalar)} |"

        # GPU cell
        if has_gpu:
            gpu_cell = " \u2714 |" if step.gpu_used else " |"

        # Path label (pre-computed by engine)
        path = step.path_classification

        # Latency%: non-critical concurrent steps show --
        if step.concurrent and not step.on_critical_path:
            lat = "--"
        else:
            lat = f"{step.latency_pct:.1f}%"

        if is_single_run:
            lines.append(
                f"| {step.step_index} | `{step.short_name}` "
                f"| {mean_str} |{tp_cell} {lat} | {path} |{gpu_cell}"
            )
        elif show_p95:
            lines.append(
                f"| {step.step_index} | `{step.short_name}` "
                f"| {mean_str} | {_fmt_time(w.median)} | "
                f"{_fmt_time(w.stddev)} | {_fmt_time(w.p95)} | "
                f"{_fmt_time(w.min)} | {_fmt_time(w.max)} |"
                f"{tp_cell} {lat} | {path} |{gpu_cell}"
            )
        else:
            lines.append(
                f"| {step.step_index} | `{step.short_name}` "
                f"| {mean_str} | {_fmt_time(w.median)} | "
                f"{_fmt_time(w.stddev)} | "
                f"{_fmt_time(w.min)} | {_fmt_time(w.max)} |"
                f"{tp_cell} {lat} | {path} |{gpu_cell}"
            )

    lines.append("")

    # Footnotes
    if parallel:
        lines.append(
            "*‡ Wall time measured under resource contention — "
            "not comparable to isolated standalone execution time.*\n"
        )

    return "\n".join(lines)


def _md_time_decomposition(rd: RecordReportData) -> str:
    """Build time decomposition table from pre-computed data."""
    td = rd.time_decomposition
    if td is None:
        return ""

    lines = ["#### Time Decomposition\n"]
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Wall Clock (actual elapsed) | {_fmt_time(td.wall_clock_s)} |")
    lines.append(f"| Critical Path (longest chain) | {_fmt_time(td.critical_path_s)} |")
    lines.append(f"| Contended Step Sum ‡ | {_fmt_time(td.contended_step_sum_s)} |")
    lines.append("")
    lines.append(
        "*‡ Step times were measured under resource contention — "
        "not a valid sequential baseline.*\n"
    )

    return "\n".join(lines)


def _md_branch_analysis(rd: RecordReportData) -> str:
    """Build branch analysis section from pre-computed branches."""
    branches = rd.branches
    if len(branches) < 2:
        return ""

    critical_time = max(b.chain_time_s for b in branches)

    lines = ["#### Branch Analysis\n"]
    lines.append(f"*{len(branches)} branches, critical path: {_fmt_time(critical_time)}*\n")
    lines.append("| Branch | Steps | Chain Time | Status |")
    lines.append("|--------|-------|------------|--------|")

    for b in branches:
        step_names = " → ".join(b.step_names)
        if b.is_critical:
            status = "**critical path**"
        else:
            status = f"idle {_fmt_time(b.idle_time_s)}"
        lines.append(
            f"| {b.branch_index} | {step_names} | {_fmt_time(b.chain_time_s)} | {status} |"
        )

    lines.append("")
    return "\n".join(lines)


def _md_record_detail(index: int, rd: RecordReportData) -> str:
    """Build the complete detail section for a single record."""
    topo_label = f" ({rd.topology_label})" if rd.topology_label else ""

    lines = [f"### {index}. {rd.workflow_name}{topo_label}\n"]

    rec = rd.record
    lines.append(
        f"**Type**: {rd.benchmark_type} | "
        f"**Version**: {rd.workflow_version} | "
        f"**Iterations**: {rd.iterations} | "
        f"**Wall**: {_fmt_time(rec.total_wall_time.mean)} | "
        f"**CPU**: {_fmt_time(rec.total_cpu_time.mean)} | "
        f"**Memory**: {_fmt_bytes(rec.total_peak_rss.mean)}"
    )

    if rec.tags:
        tag_str = ", ".join(f"`{k}={v}`" for k, v in sorted(rec.tags.items()))
        lines.append(f"\n**Tags**: {tag_str}")

    lines.append("")
    lines.append(_md_step_table(rd))
    lines.append(_md_time_decomposition(rd))
    lines.append(_md_branch_analysis(rd))

    return "\n".join(lines)


def _md_comparison_section(comparison: ComparisonResult) -> str:
    """Build the cross-workflow comparison section."""
    if len(comparison.records) < 2:
        return ""

    lines = ["## Comparison\n"]

    lines.append("### Workflow Summary\n")
    lines.append("| Workflow | Topology | Wall Time | CPU Time | Peak Memory | Steps |")
    lines.append("|----------|----------|-----------|----------|-------------|-------|")

    for label, rec in zip(comparison.record_labels, comparison.records):
        topo = rec.topology.topology.value if rec.topology else "unknown"
        active_count = sum(
            1 for s in rec.step_results if not _step_was_skipped(s)
        )
        lines.append(
            f"| {label} | {topo} | {_fmt_time(rec.total_wall_time.mean)} | "
            f"{_fmt_time(rec.total_cpu_time.mean)} | "
            f"{_fmt_bytes(rec.total_peak_rss.mean)} | {active_count} |"
        )
    lines.append("")

    return "\n".join(lines)


def _md_overall_summary(data: ReportData) -> str:
    """Build the overall summary section from pre-computed data."""
    s = data.overall_summary
    if s is None:
        return ""

    lines = ["## Overall Summary\n"]
    lines.append(f"- **Total Benchmarks**: {s.total_benchmarks}")
    lines.append(f"- **Total Wall Time**: {_fmt_time(s.total_wall_time_s)}")
    lines.append(
        f"- **Fastest**: {s.fastest_name} "
        f"({_fmt_time(s.fastest_wall_s)})"
    )
    lines.append(
        f"- **Slowest**: {s.slowest_name} "
        f"({_fmt_time(s.slowest_wall_s)})"
    )
    lines.append(
        f"- **Least Memory**: {s.least_memory_name} "
        f"({_fmt_bytes(s.least_memory_bytes)})"
    )
    lines.append(
        f"- **Most Memory**: {s.most_memory_name} "
        f"({_fmt_bytes(s.most_memory_bytes)})"
    )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def format_report_md(
    records: List[BenchmarkRecord],
    comparison: Optional[ComparisonResult] = None,
) -> str:
    """Generate a comprehensive Markdown benchmark report.

    Every report includes full per-step data for all workflows.
    Comparisons are an additional summary section when multiple
    records are present.

    Parameters
    ----------
    records : List[BenchmarkRecord]
        Benchmark records to include.  Must not be empty.
    comparison : ComparisonResult, optional
        Pre-computed comparison.  If ``None`` and ``len(records) > 1``,
        one is computed automatically.

    Returns
    -------
    str
        Complete Markdown report text.

    Raises
    ------
    ValueError
        If *records* is empty.
    """
    if not records:
        raise ValueError("Cannot generate report from empty records list.")

    data = build_report_data(records, comparison)

    parts: List[str] = []

    parts.append(_md_header(data.record_count))
    parts.append("---\n")
    parts.append(_md_executive_summary(data))
    parts.append("---\n")
    if data.hardware is not None:
        parts.append(_md_hardware(data.hardware))
    else:
        parts.append("## Hardware\n\n> Hardware information missing.\n")
    parts.append(_md_configuration(records))
    parts.append("---\n")

    # Detailed results (already sorted slowest-first by engine)
    parts.append("## Detailed Results\n")
    for i, rd in enumerate(data.records, start=1):
        parts.append(_md_record_detail(i, rd))

    if data.comparison and data.record_count > 1:
        parts.append("---\n")
        parts.append(_md_comparison_section(data.comparison))

    if data.record_count > 1:
        parts.append("---\n")
        parts.append(_md_overall_summary(data))

    return "\n".join(parts)


def save_report_md(
    records: List[BenchmarkRecord],
    path: Path,
    comparison: Optional[ComparisonResult] = None,
) -> Path:
    """Write a comprehensive Markdown benchmark report to a file.

    Creates parent directories if they do not exist.  If *path* is
    a directory, a timestamped filename is generated.

    Parameters
    ----------
    records : List[BenchmarkRecord]
        Benchmark records to include.
    path : Path
        Output file path or directory.
    comparison : ComparisonResult, optional
        Pre-computed comparison.

    Returns
    -------
    Path
        The path to the written report file.
    """
    report_text = format_report_md(records, comparison)
    path = Path(path)

    if path.is_dir() or (not path.suffix and not path.exists()):
        path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = path / f"benchmark_report_{timestamp}.md"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(report_text, encoding="utf-8")
    return path
