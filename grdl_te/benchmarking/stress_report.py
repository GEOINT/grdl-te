# -*- coding: utf-8 -*-
"""
Stress Test Report Generator — text and Markdown reports for stress runs.

Provides ``format_stress_report`` (plain text) and
``format_stress_report_md`` (Markdown) for presenting ``StressTestRecord``
results.

Report structure:
  1. Header — component, grdl version, run timestamp
  2. Hardware — machine and memory configuration
  3. Stress Configuration — ramp parameters
  4. Saturation Curve — concurrency vs. error rate vs. p99 latency table
  5. Failure Analysis — ordered failure points with memory and error details
  6. Summary — headline statistics

When no failures occurred, sections 5 and the failure columns in section 4
are replaced with a clean "No failures detected" note.

Benchmark cross-reference (section 7, optional) is included when
*related_benchmark_id* is set on the record.

Dependencies
------------
numpy

Author
------
Ava Courtney <courtney-ava@zai.com>

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2026-04-07

Modified
--------
2026-04-07
"""

# Standard library
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Third-party
import numpy as np

# Internal
from grdl_te.benchmarking._formatting import fmt_bytes as _fmt_bytes
from grdl_te.benchmarking.stress_models import (
    FailurePoint,
    StressTestRecord,
    StressTestSummary,
)

LINE_WIDTH = 80
RULE = "=" * LINE_WIDTH
THIN_RULE = "-" * LINE_WIDTH


# ---------------------------------------------------------------------------
# Shared data derivation helpers
# ---------------------------------------------------------------------------

def _per_level_stats(
    record: StressTestRecord,
) -> List[Tuple[int, int, int, float, float]]:
    """Compute per-concurrency statistics from events.

    Returns
    -------
    List of (concurrency, total_calls, failed_calls, error_rate_pct, p99_latency_s)
    """
    from collections import defaultdict

    buckets: Dict[int, List] = defaultdict(list)
    for event in record.events:
        buckets[event.concurrency_level].append(event)

    rows = []
    for level in sorted(buckets):
        events = buckets[level]
        total = len(events)
        failed = sum(1 for e in events if not e.success)
        error_rate = (failed / total * 100.0) if total > 0 else 0.0
        latencies = [e.latency_s for e in events if e.success]
        p99 = float(np.percentile(latencies, 99)) if latencies else 0.0
        rows.append((level, total, failed, error_rate, p99))
    return rows


# ---------------------------------------------------------------------------
# Plain-text report
# ---------------------------------------------------------------------------

def format_stress_report(record: StressTestRecord) -> str:
    """Format a stress test record as a plain-text report.

    Parameters
    ----------
    record : StressTestRecord

    Returns
    -------
    str
    """
    lines: List[str] = []
    a = lines.append

    # 1. Header
    now = datetime.now(timezone.utc).isoformat()
    a(RULE)
    a("  GRDL STRESS TEST REPORT")
    a(f"  Generated:  {now}")
    a(f"  Component:  {record.component_name}  (v{record.component_version})")
    a(f"  GRDL:       {record.grdl_version}")
    a(f"  Run ID:     {record.stress_test_id}")
    a(f"  Run At:     {record.created_at}")
    a(RULE)

    # 2. Hardware
    hw = record.hardware
    if hw is not None:
        a("")
        a("  HARDWARE")
        a(f"  {THIN_RULE[:40]}")
        a(f"  Hostname:        {hw.hostname}")
        a(f"  Platform:        {hw.platform_info}")
        a(f"  CPUs:            {hw.cpu_count}")
        a(f"  System Memory:   {_fmt_bytes(hw.total_memory_bytes)}")
        a(f"  GPU Available:   {'Yes' if hw.gpu_available else 'No'}")

    # 3. Stress Configuration
    cfg = record.config
    a("")
    a("  STRESS CONFIGURATION")
    a(f"  {THIN_RULE[:40]}")
    a(f"  Payload Size:    {cfg.payload_size}")
    a(f"  Start Workers:   {cfg.start_concurrency}")
    a(f"  Max Workers:     {cfg.max_concurrency}")
    a(f"  Ramp Steps:      {cfg.ramp_steps}")
    a(f"  Step Duration:   {cfg.duration_per_step_s:.1f}s")
    a(f"  Call Timeout:    {cfg.timeout_per_call_s:.1f}s")
    levels = cfg.concurrency_levels()
    a(f"  Levels:          {levels}")

    # 4. Saturation Curve
    a("")
    a("  SATURATION CURVE")
    a(f"  {THIN_RULE[:40]}")
    stat_rows = _per_level_stats(record)
    if stat_rows:
        header = (
            f"  {'Workers':>8}  {'Calls':>7}  {'Errors':>7}  "
            f"{'Err%':>6}  {'p99 Lat':>9}"
        )
        a(header)
        a(f"  {THIN_RULE[:60]}")
        for level, total, failed, err_pct, p99 in stat_rows:
            flag = " <-- FAILURE" if failed > 0 else ""
            a(
                f"  {level:>8}  {total:>7}  {failed:>7}  "
                f"{err_pct:>5.1f}%  {p99:>8.3f}s{flag}"
            )
    else:
        a("  (no events recorded)")

    # 5. Failure Analysis
    a("")
    a("  FAILURE ANALYSIS")
    a(f"  {THIN_RULE[:40]}")
    if record.failure_points:
        for i, fp in enumerate(record.failure_points, 1):
            a(f"  [{i}] {fp.error_type} @ concurrency={fp.concurrency_level}")
            a(f"      Payload:   {list(fp.payload_shape)}")
            a(f"      Memory:    {_fmt_bytes(fp.memory_bytes_at_failure)}")
            a(f"      At:        {fp.first_occurrence_at}")
            a(f"      Message:   {fp.error_message[:120]}")
    else:
        a("  No failures detected across the full ramp.")

    # 6. Summary
    s = record.summary
    a("")
    a("  SUMMARY")
    a(f"  {THIN_RULE[:40]}")
    a(f"  Max Sustained Concurrency:  {s.max_sustained_concurrency}")
    if s.saturation_concurrency is not None:
        a(f"  Saturation Point:           {s.saturation_concurrency} workers")
        a(f"  First Failure Mode:         {s.first_failure_mode}")
    else:
        a("  Saturation Point:           None (component held under full load)")
    a(f"  Memory High-Water Mark:     {_fmt_bytes(s.memory_high_water_mark_bytes)}")
    a(f"  Total Calls:                {s.total_calls}")
    a(f"  Failed Calls:               {s.failed_calls}")
    success_pct = (
        (s.total_calls - s.failed_calls) / s.total_calls * 100.0
        if s.total_calls > 0
        else 0.0
    )
    a(f"  Success Rate:               {success_pct:.1f}%")
    a(f"  p99 Latency (success):      {s.p99_latency_s:.3f}s")

    # 7. Cross-reference
    if record.related_benchmark_id:
        a("")
        a("  CROSS-REFERENCE")
        a(f"  {THIN_RULE[:40]}")
        a(
            f"  Per-call timing statistics are available in benchmark record:"
        )
        a(f"  {record.related_benchmark_id}")

    a("")
    a(RULE)
    return "\n".join(lines)


def print_stress_report(record: StressTestRecord) -> None:
    """Print a stress test report to stdout.

    Parameters
    ----------
    record : StressTestRecord
    """
    print(format_stress_report(record))


def save_stress_report(
    record: StressTestRecord,
    path: Path,
) -> Path:
    """Save a plain-text stress test report to a file or directory.

    Parameters
    ----------
    record : StressTestRecord
    path : Path
        If *path* is a directory, the report is written to
        ``<path>/stress_<stress_test_id[:8]>.txt``.
        If *path* is a file path, it is written directly.

    Returns
    -------
    Path
        The file path where the report was written.
    """
    path = Path(path)
    if path.is_dir():
        short_id = record.stress_test_id[:8]
        path = path / f"stress_{short_id}.txt"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_stress_report(record), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def format_stress_report_md(record: StressTestRecord) -> str:
    """Format a stress test record as a Markdown report.

    Parameters
    ----------
    record : StressTestRecord

    Returns
    -------
    str
    """
    lines: List[str] = []
    a = lines.append

    # Header
    now = datetime.now(timezone.utc).isoformat()
    a(f"# GRDL Stress Test Report")
    a(f"")
    a(f"| Field | Value |")
    a(f"|-------|-------|")
    a(f"| Component | `{record.component_name}` v{record.component_version} |")
    a(f"| GRDL | {record.grdl_version} |")
    a(f"| Run ID | `{record.stress_test_id}` |")
    a(f"| Run At | {record.created_at} |")
    a(f"| Generated | {now} |")
    a(f"")

    # Hardware
    hw = record.hardware
    if hw is not None:
        a("## Hardware")
        a(f"")
        a(f"| Field | Value |")
        a(f"|-------|-------|")
        a(f"| Hostname | {hw.hostname} |")
        a(f"| Platform | {hw.platform_info} |")
        a(f"| CPUs | {hw.cpu_count} |")
        a(f"| System Memory | {_fmt_bytes(hw.total_memory_bytes)} |")
        a(f"| GPU | {'Yes' if hw.gpu_available else 'No'} |")
        a(f"")

    # Config
    cfg = record.config
    a("## Stress Configuration")
    a(f"")
    a(f"| Parameter | Value |")
    a(f"|-----------|-------|")
    a(f"| Payload Size | {cfg.payload_size} |")
    a(f"| Worker Range | {cfg.start_concurrency} → {cfg.max_concurrency} |")
    a(f"| Ramp Steps | {cfg.ramp_steps} |")
    a(f"| Step Duration | {cfg.duration_per_step_s:.1f}s |")
    a(f"| Call Timeout | {cfg.timeout_per_call_s:.1f}s |")
    levels = cfg.concurrency_levels()
    a(f"| Levels Tested | {levels} |")
    a(f"")

    # Saturation Curve
    a("## Saturation Curve")
    a(f"")
    stat_rows = _per_level_stats(record)
    if stat_rows:
        a("| Workers | Total Calls | Failed | Error Rate | p99 Latency |")
        a("|---------|-------------|--------|------------|-------------|")
        for level, total, failed, err_pct, p99 in stat_rows:
            flag = " ⚠" if failed > 0 else ""
            a(
                f"| {level} | {total} | {failed} | {err_pct:.1f}%{flag} "
                f"| {p99:.3f}s |"
            )
    else:
        a("_No events recorded._")
    a(f"")

    # Failure Analysis
    a("## Failure Analysis")
    a(f"")
    if record.failure_points:
        for i, fp in enumerate(record.failure_points, 1):
            a(f"### Failure {i}: `{fp.error_type}`")
            a(f"")
            a(f"| Field | Value |")
            a(f"|-------|-------|")
            a(f"| Concurrency | {fp.concurrency_level} workers |")
            a(f"| Payload | {list(fp.payload_shape)} |")
            a(f"| Memory At Failure | {_fmt_bytes(fp.memory_bytes_at_failure)} |")
            a(f"| First Occurrence | {fp.first_occurrence_at} |")
            a(f"| Message | `{fp.error_message[:120]}` |")
            a(f"")
    else:
        a("_No failures detected across the full ramp._")
        a(f"")

    # Summary
    s = record.summary
    a("## Summary")
    a(f"")
    a(f"| Metric | Value |")
    a(f"|--------|-------|")
    a(f"| Max Sustained Concurrency | **{s.max_sustained_concurrency}** |")
    if s.saturation_concurrency is not None:
        a(f"| Saturation Point | {s.saturation_concurrency} workers |")
        a(f"| First Failure Mode | `{s.first_failure_mode}` |")
    else:
        a("| Saturation Point | None (held under full load) |")
    a(f"| Memory High-Water Mark | {_fmt_bytes(s.memory_high_water_mark_bytes)} |")
    a(f"| Total Calls | {s.total_calls} |")
    a(f"| Failed Calls | {s.failed_calls} |")
    success_pct = (
        (s.total_calls - s.failed_calls) / s.total_calls * 100.0
        if s.total_calls > 0
        else 0.0
    )
    a(f"| Success Rate | {success_pct:.1f}% |")
    a(f"| p99 Latency (success) | {s.p99_latency_s:.3f}s |")
    a(f"")

    # Cross-reference
    if record.related_benchmark_id:
        a("## Cross-Reference")
        a(f"")
        a(
            "Per-call timing statistics are available in benchmark record "
            f"`{record.related_benchmark_id}`."
        )
        a(f"")

    return "\n".join(lines)


def save_stress_report_md(
    record: StressTestRecord,
    path: Path,
) -> Path:
    """Save a Markdown stress test report to a file or directory.

    Parameters
    ----------
    record : StressTestRecord
    path : Path
        If a directory, written to ``<path>/stress_<id[:8]>.md``.

    Returns
    -------
    Path
    """
    path = Path(path)
    if path.is_dir():
        short_id = record.stress_test_id[:8]
        path = path / f"stress_{short_id}.md"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_stress_report_md(record), encoding="utf-8")
    return path
