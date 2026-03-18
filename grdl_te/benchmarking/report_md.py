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
2026-03-04
"""

# Standard library
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Third-party
import numpy as np

# Internal
from grdl_te.benchmarking.models import (
    AggregatedMetrics,
    BenchmarkRecord,
    HardwareSnapshot,
    StepBenchmarkResult,
    TopologyDescriptor,
    WorkflowTopology,
)
from grdl_te.benchmarking.comparison import ComparisonResult, compare_records
from grdl_te.benchmarking.report import _build_branch_chains


# ---------------------------------------------------------------------------
# Unit formatting helpers
# ---------------------------------------------------------------------------
def _fmt_bytes(value_bytes: float) -> str:
    """Format a byte count as a human-readable string."""
    abs_val = abs(value_bytes)
    if abs_val < 1024:
        return f"{value_bytes:.0f} B"
    if abs_val < 1024 ** 2:
        return f"{value_bytes / 1024:.1f} KB"
    if abs_val < 1024 ** 3:
        return f"{value_bytes / 1024 ** 2:.1f} MB"
    return f"{value_bytes / 1024 ** 3:.2f} GB"


def _fmt_time(seconds: float) -> str:
    """Format a duration in seconds with appropriate precision."""
    if abs(seconds) < 1.0:
        return f"{seconds:.4f}s"
    return f"{seconds:.2f}s"


def _step_was_skipped(step: StepBenchmarkResult) -> bool:
    """Return True if a step was skipped."""
    return step.wall_time_s.max == 0.0 and step.cpu_time_s.max == 0.0


def _short_name(processor_name: str) -> str:
    """Extract short processor name."""
    return processor_name.rsplit(".", 1)[-1]


# ---------------------------------------------------------------------------
# Section formatters
# ---------------------------------------------------------------------------
def _md_header(record_count: int) -> str:
    """Build the report header."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return (
        f"# GRDL Benchmark Report\n\n"
        f"**Generated**: {now} | **Records**: {record_count}\n"
    )


def _md_executive_summary(
    records: List[BenchmarkRecord],
    comparison: Optional[ComparisonResult],
) -> str:
    """Build the executive summary with bottleneck identification."""
    lines = ["## Executive Summary\n"]

    # Topology summary
    topo_counts: Dict[str, int] = {}
    for rec in records:
        if rec.topology:
            name = rec.topology.topology.value
        else:
            name = "unknown"
        topo_counts[name] = topo_counts.get(name, 0) + 1

    topo_parts = []
    for topo_name, count in sorted(topo_counts.items()):
        topo_parts.append(f"{count} {topo_name}")
    lines.append(f"**Topology**: {', '.join(topo_parts)}\n")

    # Hardware one-liner
    hw = records[0].hardware
    mem_str = _fmt_bytes(hw.total_memory_bytes)
    gpu_str = f", {len(hw.gpu_devices)} GPU(s)" if hw.gpu_available else ""
    lines.append(f"**Hardware**: {hw.cpu_count} CPUs, {mem_str} RAM{gpu_str}\n")

    # Bottleneck table
    if comparison and comparison.bottlenecks:
        bns = comparison.bottlenecks[:5]
    else:
        # Single record — collect from step results directly
        bns = []
        for rec in records:
            for step in rec.step_results:
                if _step_was_skipped(step):
                    continue
                bns.append({
                    "step_name": _short_name(step.processor_name),
                    "latency_pct": step.latency_pct,
                    "memory_pct": step.memory_pct,
                    "workflow": rec.workflow_name,
                    "wall_time_s": step.wall_time_s.mean,
                })
        bns.sort(key=lambda e: (e["latency_pct"], e["memory_pct"]), reverse=True)
        bns = bns[:5]

    if bns:
        top = bns[0]
        lines.append(
            f"> **Top Bottleneck**: `{top['step_name']}` accounts for "
            f"**{top['latency_pct']:.1f}%** of latency"
            f" ({_fmt_time(top['wall_time_s'])} mean wall time)"
            f" in *{top['workflow']}*.\n"
        )
        lines.append("| Rank | Step | Latency % | Memory % | Workflow | Mean Wall |")
        lines.append("|------|------|-----------|----------|----------|-----------|")
        for i, bn in enumerate(bns, 1):
            lat = "--" if bn["latency_pct"] == 0.0 else f"{bn['latency_pct']:.1f}%"
            lines.append(
                f"| {i} | `{bn['step_name']}` | {lat} | "
                f"{bn['memory_pct']:.1f}% | {bn['workflow']} | "
                f"{_fmt_time(bn['wall_time_s'])} |"
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
    size_tag = first.tags.get("array_size", "unknown")

    lines = [
        f"**Iterations**: {first.iterations} | "
        f"**Array Size**: {size_tag} | "
        f"**Records**: {len(records)}\n",
    ]
    return "\n".join(lines)


def _md_step_table(record: BenchmarkRecord) -> str:
    """Build per-step performance table with latency% and memory%."""
    active = [s for s in record.step_results if not _step_was_skipped(s)]
    skipped = [s for s in record.step_results if _step_was_skipped(s)]
    parallel = [s for s in active if s.concurrent]

    lines = ["#### Step Performance\n"]

    # Summary line — only shown when there is something to say
    # (skipped steps or a parallel sub-count to report).
    if skipped or parallel:
        summary_parts = [f"{len(active)} ran"]
        if parallel:
            summary_parts.append(f"{len(parallel)} parallel")
        if skipped:
            summary_parts.append(f"{len(skipped)} skipped")
        lines.append(
            f"*{', '.join(summary_parts)} / "
            f"{len(record.step_results)} total*\n"
        )

    # Determine critical path set
    cp_set = set()
    if record.topology:
        cp_set = set(record.topology.critical_path_step_ids)

    is_single_run = record.iterations == 1
    show_p95 = record.iterations > 10

    # Table header — layout depends on N=1 (scalar) vs N>1 (statistics)
    if is_single_run:
        lines.append("| # | Step | Wall Time | Latency% | Memory% | Path |")
        lines.append("|---|------|-----------|----------|---------|------|")
    elif show_p95:
        lines.append(
            "| # | Step | Mean | Median | StdDev | P95 | Min | Max | "
            "Latency% | Memory% | Path |"
        )
        lines.append(
            "|---|------|------|--------|--------|-----|-----|-----|"
            "----------|---------|------|"
        )
    else:
        lines.append(
            "| # | Step | Mean | Median | StdDev | Min | Max | "
            "Latency% | Memory% | Path |"
        )
        lines.append(
            "|---|------|------|--------|--------|-----|-----|"
            "----------|---------|------|"
        )

    for step in record.step_results:
        if _step_was_skipped(step):
            if is_single_run:
                skip_dashes = "-- | -- | --"
            elif show_p95:
                skip_dashes = "-- | -- | -- | -- | -- | -- | -- | --"
            else:
                skip_dashes = "-- | -- | -- | -- | -- | -- | --"
            lines.append(
                f"| {step.step_index} | `{_short_name(step.processor_name)}` "
                f"| {skip_dashes} | *skipped* |"
            )
            continue

        w = step.wall_time_s
        mean_str = _fmt_time(w.mean)
        if step.concurrent:
            mean_str += " ‡"

        # Path label
        key = step.step_id or f"__idx_{step.step_index}"
        if key in cp_set:
            path = "critical"
        elif step.concurrent:
            path = "parallel"
        else:
            path = "sequential"

        # Latency%: non-critical concurrent steps show -- (their wall time is
        # hidden by the critical path — showing 0% would be misleading).
        if step.concurrent and key not in cp_set:
            lat = "--"
        else:
            lat = f"{step.latency_pct:.1f}%"

        mem = f"{step.memory_pct:.1f}%"
        if step.concurrent:
            mem += "†"

        if is_single_run:
            lines.append(
                f"| {step.step_index} | `{_short_name(step.processor_name)}` "
                f"| {mean_str} | {lat} | {mem} | {path} |"
            )
        elif show_p95:
            lines.append(
                f"| {step.step_index} | `{_short_name(step.processor_name)}` "
                f"| {mean_str} | {_fmt_time(w.median)} | "
                f"{_fmt_time(w.stddev)} | {_fmt_time(w.p95)} | "
                f"{_fmt_time(w.min)} | {_fmt_time(w.max)} | "
                f"{lat} | {mem} | {path} |"
            )
        else:
            lines.append(
                f"| {step.step_index} | `{_short_name(step.processor_name)}` "
                f"| {mean_str} | {_fmt_time(w.median)} | "
                f"{_fmt_time(w.stddev)} | "
                f"{_fmt_time(w.min)} | {_fmt_time(w.max)} | "
                f"{lat} | {mem} | {path} |"
            )

    lines.append("")

    # Footnotes — only for concurrent steps
    has_parallel = bool(parallel)
    if has_parallel:
        lines.append(
            "*† Memory shared across concurrent steps "
            "(tracemalloc is process-wide)*"
        )
        lines.append(
            "*‡ Wall time measured under resource contention — "
            "not comparable to isolated standalone execution time.*\n"
        )

    return "\n".join(lines)


def _md_time_decomposition(record: BenchmarkRecord) -> str:
    """Build time decomposition table for parallel/mixed workflows."""
    if not record.topology:
        return ""
    topo = record.topology
    if topo.topology in (WorkflowTopology.COMPONENT, WorkflowTopology.SEQUENTIAL):
        return ""

    lines = ["#### Time Decomposition\n"]
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Wall Clock (actual elapsed) | {_fmt_time(record.total_wall_time.mean)} |")
    lines.append(f"| Critical Path (longest chain) | {_fmt_time(topo.critical_path_wall_time_s)} |")
    lines.append(f"| Contended Step Sum ‡ | {_fmt_time(topo.sum_of_steps_wall_time_s)} |")
    lines.append("")
    lines.append(
        "*‡ Step times were measured under resource contention — "
        "not a valid sequential baseline.*\n"
    )

    return "\n".join(lines)


def _md_branch_analysis(record: BenchmarkRecord) -> str:
    """Build branch analysis section from depends_on graph."""
    if not record.topology:
        return ""
    topo = record.topology
    if topo.topology in (WorkflowTopology.COMPONENT, WorkflowTopology.SEQUENTIAL):
        return ""

    chains = _build_branch_chains(record.step_results)
    if len(chains) < 2:
        return ""

    chain_times = [sum(s.wall_time_s.mean for s in c) for c in chains]
    critical_time = max(chain_times) if chain_times else 0.0

    lines = ["#### Branch Analysis\n"]
    lines.append(f"*{len(chains)} branches, critical path: {_fmt_time(critical_time)}*\n")
    lines.append("| Branch | Steps | Chain Time | Status |")
    lines.append("|--------|-------|------------|--------|")

    for i, (chain, ct) in enumerate(zip(chains, chain_times)):
        step_names = " → ".join(
            _short_name(s.processor_name) for s in chain
        )
        if ct >= critical_time - 1e-9:
            status = "**critical path**"
        else:
            idle = critical_time - ct
            status = f"idle {_fmt_time(idle)}"
        lines.append(
            f"| {i + 1} | {step_names} | {_fmt_time(ct)} | {status} |"
        )

    lines.append("")
    return "\n".join(lines)


def _md_memory_profile(record: BenchmarkRecord) -> str:
    """Build memory profile table."""
    active = [s for s in record.step_results if not _step_was_skipped(s)]
    has_new_mem = any(
        s.peak_overhead_bytes is not None
        or s.end_of_step_footprint_bytes is not None
        for s in active
    )
    if not has_new_mem:
        return ""

    lines = ["#### Memory Profile\n"]
    lines.append("| Step | Peak Overhead | End-of-Step Footprint | Memory% |")
    lines.append("|------|-------------|----------------------|---------|")

    for step in active:
        name = _short_name(step.processor_name)
        overhead = (
            _fmt_bytes(step.peak_overhead_bytes.mean)
            if step.peak_overhead_bytes is not None
            else "N/A"
        )
        footprint = (
            _fmt_bytes(step.end_of_step_footprint_bytes.mean)
            if step.end_of_step_footprint_bytes is not None
            else "N/A"
        )
        mem_str = f"{step.memory_pct:.1f}%"
        if step.concurrent:
            mem_str += " (concurrent)"
        lines.append(
            f"| `{name}` | {overhead} | {footprint} | {mem_str} |"
        )

    overall = _fmt_bytes(record.total_peak_rss.mean)
    lines.append(f"| **Overall Workflow Peak** | **{overall}** | *(high-water mark)* | |")
    lines.append("")

    return "\n".join(lines)


def _md_record_detail(index: int, record: BenchmarkRecord) -> str:
    """Build the complete detail section for a single benchmark record."""
    topo_label = ""
    if record.topology:
        topo_label = f" ({record.topology.topology.value})"

    lines = [f"### {index}. {record.workflow_name}{topo_label}\n"]

    lines.append(
        f"**Type**: {record.benchmark_type} | "
        f"**Version**: {record.workflow_version} | "
        f"**Iterations**: {record.iterations} | "
        f"**Wall**: {_fmt_time(record.total_wall_time.mean)} | "
        f"**CPU**: {_fmt_time(record.total_cpu_time.mean)} | "
        f"**Memory**: {_fmt_bytes(record.total_peak_rss.mean)}"
    )

    if record.tags:
        tag_str = ", ".join(f"`{k}={v}`" for k, v in sorted(record.tags.items()))
        lines.append(f"\n**Tags**: {tag_str}")

    lines.append("")
    lines.append(_md_step_table(record))
    lines.append(_md_time_decomposition(record))
    lines.append(_md_branch_analysis(record))
    lines.append(_md_memory_profile(record))

    return "\n".join(lines)


def _md_comparison_section(comparison: ComparisonResult) -> str:
    """Build the cross-workflow comparison section."""
    if len(comparison.records) < 2:
        return ""

    lines = ["## Comparison\n"]

    # Workflow summary
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


def _md_overall_summary(records: List[BenchmarkRecord]) -> str:
    """Build the overall summary section."""
    lines = ["## Overall Summary\n"]

    if len(records) == 1:
        r = records[0]
        w = r.total_wall_time
        c = r.total_cpu_time
        m = r.total_peak_rss
        lines.append(
            f"**Workflow**: {r.workflow_name} v{r.workflow_version}\n"
        )
        if r.iterations == 1:
            lines.append(f"- **Wall Time**: {_fmt_time(w.mean)}")
            lines.append(f"- **CPU Time**: {_fmt_time(c.mean)}")
            lines.append(f"- **Peak Memory**: {_fmt_bytes(m.mean)}")
        else:
            lines.append(
                f"- **Wall Time**: {_fmt_time(w.mean)} mean | "
                f"{_fmt_time(w.min)} min | {_fmt_time(w.max)} max | "
                f"{_fmt_time(w.stddev)} stddev"
            )
            lines.append(
                f"- **CPU Time**: {_fmt_time(c.mean)} mean | "
                f"{_fmt_time(c.min)} min | {_fmt_time(c.max)} max | "
                f"{_fmt_time(c.stddev)} stddev"
            )
            lines.append(
                f"- **Peak Memory**: {_fmt_bytes(m.mean)} mean | "
                f"{_fmt_bytes(m.min)} min | {_fmt_bytes(m.max)} max"
            )
    else:
        wall_means = [r.total_wall_time.mean for r in records]
        total_wall = sum(wall_means)

        fastest = min(records, key=lambda r: r.total_wall_time.mean)
        slowest = max(records, key=lambda r: r.total_wall_time.mean)
        least_mem = min(records, key=lambda r: r.total_peak_rss.mean)
        most_mem = max(records, key=lambda r: r.total_peak_rss.mean)

        lines.append(f"- **Total Benchmarks**: {len(records)}")
        lines.append(f"- **Total Wall Time**: {_fmt_time(total_wall)}")
        lines.append(
            f"- **Fastest**: {fastest.workflow_name} "
            f"({_fmt_time(fastest.total_wall_time.mean)})"
        )
        lines.append(
            f"- **Slowest**: {slowest.workflow_name} "
            f"({_fmt_time(slowest.total_wall_time.mean)})"
        )
        lines.append(
            f"- **Least Memory**: {least_mem.workflow_name} "
            f"({_fmt_bytes(least_mem.total_peak_rss.mean)})"
        )
        lines.append(
            f"- **Most Memory**: {most_mem.workflow_name} "
            f"({_fmt_bytes(most_mem.total_peak_rss.mean)})"
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

    if comparison is None and len(records) > 1:
        comparison = compare_records(records)

    parts: List[str] = []

    parts.append(_md_header(len(records)))
    parts.append("---\n")
    parts.append(_md_executive_summary(records, comparison))
    parts.append("---\n")
    if records[0].hardware is not None:
        parts.append(_md_hardware(records[0].hardware))
    else:
        parts.append("## Hardware\n\n> Hardware information missing.\n")
    parts.append(_md_configuration(records))
    parts.append("---\n")

    # Detailed results — sorted by wall time descending (slowest first)
    sorted_records = sorted(
        records,
        key=lambda r: r.total_wall_time.mean,
        reverse=True,
    )

    parts.append("## Detailed Results\n")
    for i, record in enumerate(sorted_records, start=1):
        parts.append(_md_record_detail(i, record))

    if comparison and len(records) > 1:
        parts.append("---\n")
        parts.append(_md_comparison_section(comparison))

    if len(records) > 1:
        parts.append("---\n")
        parts.append(_md_overall_summary(records))

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
