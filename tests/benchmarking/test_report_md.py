# -*- coding: utf-8 -*-
"""
Tests for Markdown benchmark report generator.

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
import tempfile
from pathlib import Path

# Third-party
import pytest

# Internal
from grdl_te.benchmarking.models import (
    AggregatedMetrics,
    BenchmarkRecord,
    HardwareSnapshot,
    StepBenchmarkResult,
    TopologyDescriptor,
    WorkflowTopology,
)
from grdl_te.benchmarking.comparison import compare_records
from grdl_te.benchmarking.report_md import format_report_md, save_report_md


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Module-level hardware cache — captured once per module to avoid
# redundant OS/platform calls across ~14 tests.
_CACHED_HW: "HardwareSnapshot | None" = None


def _hw() -> HardwareSnapshot:
    global _CACHED_HW
    if _CACHED_HW is None:
        _CACHED_HW = HardwareSnapshot.capture()
    return _CACHED_HW


def _agg(mean: float) -> AggregatedMetrics:
    return AggregatedMetrics.from_values([mean])


def _step(
    index: int,
    name: str,
    wall: float,
    mem: float = 1000.0,
    concurrent: bool = False,
    step_id: str = None,
    depends_on: list = None,
    latency_pct: float = 0.0,
    memory_pct: float = 0.0,
) -> StepBenchmarkResult:
    s = StepBenchmarkResult(
        step_index=index,
        processor_name=name,
        wall_time_s=_agg(wall),
        cpu_time_s=_agg(wall * 0.5),
        peak_rss_bytes=_agg(mem),
        gpu_used=False,
        sample_count=1,
        concurrent=concurrent,
        step_id=step_id,
        depends_on=depends_on or [],
        latency_pct=latency_pct,
        memory_pct=memory_pct,
    )
    return s


def _record(
    name: str,
    steps: list,
    topology: TopologyDescriptor = None,
    total_wall: float = None,
    benchmark_type: str = "active",
) -> BenchmarkRecord:
    if total_wall is None:
        total_wall = sum(s.wall_time_s.mean for s in steps)
    total_cpu = sum(s.cpu_time_s.mean for s in steps)
    # Use sum for concurrent steps (overlapping footprints) vs max for
    # sequential (process high-water mark is the single largest step).
    if any(s.concurrent for s in steps):
        total_mem = sum(s.peak_rss_bytes.mean for s in steps)
    else:
        total_mem = max((s.peak_rss_bytes.mean for s in steps), default=0.0)

    rec = BenchmarkRecord.create(
        benchmark_type=benchmark_type,
        workflow_name=name,
        workflow_version="1.0.0",
        iterations=5,
        hardware=_hw(),
        total_wall_time=_agg(total_wall),
        total_cpu_time=_agg(total_cpu),
        total_peak_rss=_agg(total_mem),
        step_results=steps,
        raw_metrics=[],
        tags={"array_size": "medium"},
    )
    if topology:
        rec.topology = topology
    return rec


# ---------------------------------------------------------------------------
# format_report_md
# ---------------------------------------------------------------------------
class TestFormatReportMd:
    """Tests for format_report_md()."""

    def test_empty_raises(self):
        """Empty records list raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            format_report_md([])

    def test_single_sequential_record(self):
        """Single sequential record produces valid markdown."""
        steps = [
            _step(0, "StepA", wall=3.0, latency_pct=30.0, memory_pct=40.0),
            _step(1, "StepB", wall=7.0, latency_pct=70.0, memory_pct=60.0),
        ]
        rec = _record(
            "SeqWorkflow", steps,
            topology=TopologyDescriptor(topology=WorkflowTopology.SEQUENTIAL),
        )
        md = format_report_md([rec])

        assert "# GRDL Benchmark Report" in md
        assert "## Executive Summary" in md
        assert "## Hardware & Configuration" in md
        assert "## Detailed Results" in md
        # Single-record reports omit Overall Summary (data is in Detailed Results)
        assert "## Overall Summary" not in md
        assert "SeqWorkflow" in md
        assert "StepA" in md
        assert "StepB" in md

    def test_single_component_record(self):
        """Component record shows correct topology label."""
        steps = [
            _step(0, "MyFunc", wall=0.5, latency_pct=100.0, memory_pct=100.0),
        ]
        rec = _record(
            "MyFunc", steps,
            topology=TopologyDescriptor(topology=WorkflowTopology.COMPONENT),
            benchmark_type="component",
        )
        md = format_report_md([rec])

        assert "component" in md
        assert "MyFunc" in md

    def test_parallel_workflow_annotations(self):
        """Parallel workflow shows time decomposition and path labels."""
        steps = [
            _step(0, "CritStep", wall=5.0, concurrent=True,
                  step_id="crit", latency_pct=100.0, memory_pct=50.0),
            _step(1, "NonCritStep", wall=2.0, concurrent=True,
                  step_id="noncrit", latency_pct=0.0, memory_pct=50.0),
        ]
        topo = TopologyDescriptor(
            topology=WorkflowTopology.PARALLEL,
            num_branches=2,
            critical_path_step_ids=("crit",),
            critical_path_wall_time_s=5.0,
            sum_of_steps_wall_time_s=7.0,
        )
        rec = _record("ParallelWF", steps, topology=topo, total_wall=5.0)
        md = format_report_md([rec])

        assert "Time Decomposition" in md
        assert "Contended Step Sum" in md
        assert "Parallelism Ratio" not in md
        assert "critical" in md
        # Non-critical step latency should show '--' (hidden by critical path)
        assert "--" in md
        # Concurrent wall times have contention marker
        assert "‡" in md
        # Memory has dagger
        assert "†" in md

    def test_comparison_section_with_two_records(self):
        """Two records trigger comparison section."""
        rec_a = _record("WorkflowA", [
            _step(0, "mod.Step1", wall=3.0, latency_pct=100.0),
        ], topology=TopologyDescriptor(topology=WorkflowTopology.SEQUENTIAL))
        rec_b = _record("WorkflowB", [
            _step(0, "pkg.Step1", wall=6.0, latency_pct=100.0),
        ], topology=TopologyDescriptor(topology=WorkflowTopology.SEQUENTIAL))

        md = format_report_md([rec_a, rec_b])

        assert "## Comparison" in md
        assert "Workflow Summary" in md
        assert "Step Comparison" not in md
        assert "Speedup Matrix" not in md
        assert "WorkflowA" in md
        assert "WorkflowB" in md

    def test_no_comparison_for_single_record(self):
        """Single record doesn't produce comparison section."""
        rec = _record("Solo", [_step(0, "S", wall=1.0)])
        md = format_report_md([rec])

        assert "## Comparison" not in md

    def test_executive_summary_bottleneck(self):
        """Executive summary highlights top bottleneck."""
        steps = [
            _step(0, "Small", wall=0.5, latency_pct=5.0, memory_pct=10.0),
            _step(1, "BigBottleneck", wall=9.5, latency_pct=95.0, memory_pct=90.0),
        ]
        rec = _record(
            "BenchWF", steps,
            topology=TopologyDescriptor(topology=WorkflowTopology.SEQUENTIAL),
        )
        md = format_report_md([rec])

        assert "Top Bottleneck" in md
        assert "BigBottleneck" in md

    def test_skipped_steps_shown(self):
        """Skipped steps appear in table with '--' values."""
        steps = [
            _step(0, "Active", wall=1.0, latency_pct=100.0),
            StepBenchmarkResult(
                step_index=1,
                processor_name="SkippedProc",
                wall_time_s=_agg(0.0),
                cpu_time_s=_agg(0.0),
                peak_rss_bytes=_agg(0.0),
                gpu_used=False,
                sample_count=1,
            ),
        ]
        rec = _record("WF", steps)
        md = format_report_md([rec])

        assert "SkippedProc" in md
        assert "skipped" in md

    def test_hardware_section(self):
        """Hardware section includes CPU, memory, and platform info."""
        rec = _record("WF", [_step(0, "S", wall=1.0)])
        md = format_report_md([rec])

        assert "CPUs" in md
        assert "System Memory" in md
        assert "Platform" in md

    def test_no_overall_summary_single(self):
        """Single-record report omits Overall Summary (data is in Detailed Results)."""
        rec = _record("WF", [_step(0, "S", wall=5.0)])
        md = format_report_md([rec])

        assert "## Overall Summary" not in md
        # Metrics still appear in the Detailed Results header
        assert "**Wall**:" in md
        assert "**CPU**:" in md
        assert "**Memory**:" in md

    def test_overall_summary_multi(self):
        """Multi-record summary shows fastest/slowest/memory extremes."""
        rec_a = _record("Fast", [_step(0, "S", wall=1.0)], total_wall=1.0)
        rec_b = _record("Slow", [_step(0, "S", wall=10.0)], total_wall=10.0)
        md = format_report_md([rec_a, rec_b])

        assert "Fastest" in md
        assert "Slowest" in md
        assert "Least Memory" in md
        assert "Most Memory" in md

    def test_memory_profile_with_overhead(self):
        """Memory Profile section renders when steps have peak_overhead_bytes."""
        overhead = AggregatedMetrics.from_values([2_000_000.0])
        footprint = AggregatedMetrics.from_values([5_000_000.0])
        steps = [
            StepBenchmarkResult(
                step_index=0,
                processor_name="Loader",
                wall_time_s=_agg(1.0),
                cpu_time_s=_agg(0.5),
                peak_rss_bytes=_agg(8_000_000.0),
                gpu_used=False,
                sample_count=1,
                peak_overhead_bytes=overhead,
                end_of_step_footprint_bytes=footprint,
                latency_pct=50.0,
                memory_pct=40.0,
            ),
            StepBenchmarkResult(
                step_index=1,
                processor_name="Processor",
                wall_time_s=_agg(2.0),
                cpu_time_s=_agg(1.0),
                peak_rss_bytes=_agg(12_000_000.0),
                gpu_used=False,
                sample_count=1,
                peak_overhead_bytes=AggregatedMetrics.from_values([4_000_000.0]),
                end_of_step_footprint_bytes=None,
                latency_pct=50.0,
                memory_pct=60.0,
            ),
        ]
        rec = _record(
            "MemWF", steps,
            topology=TopologyDescriptor(topology=WorkflowTopology.SEQUENTIAL),
        )
        md = format_report_md([rec])

        assert "Memory Profile" in md
        assert "Peak Overhead" in md
        assert "End-of-Step Footprint" in md
        assert "Loader" in md
        assert "Processor" in md
        # Step with footprint=None renders "N/A"
        assert "N/A" in md
        # Overall workflow peak line
        assert "Overall Workflow Peak" in md

    def test_memory_profile_concurrent_shared_annotation(self):
        """Concurrent steps show '(concurrent)' annotation in Memory Profile."""
        steps = [
            StepBenchmarkResult(
                step_index=0,
                processor_name="BranchA",
                wall_time_s=_agg(3.0),
                cpu_time_s=_agg(1.5),
                peak_rss_bytes=_agg(6_000_000.0),
                gpu_used=False,
                sample_count=1,
                concurrent=True,
                step_id="a",
                peak_overhead_bytes=AggregatedMetrics.from_values([1_000_000.0]),
                latency_pct=100.0,
                memory_pct=60.0,
            ),
            StepBenchmarkResult(
                step_index=1,
                processor_name="BranchB",
                wall_time_s=_agg(2.0),
                cpu_time_s=_agg(1.0),
                peak_rss_bytes=_agg(4_000_000.0),
                gpu_used=False,
                sample_count=1,
                concurrent=True,
                step_id="b",
                peak_overhead_bytes=AggregatedMetrics.from_values([500_000.0]),
                latency_pct=0.0,
                memory_pct=40.0,
            ),
        ]
        topo = TopologyDescriptor(
            topology=WorkflowTopology.PARALLEL,
            critical_path_step_ids=("a",),
            critical_path_wall_time_s=3.0,
            sum_of_steps_wall_time_s=5.0,
            num_branches=2,
        )
        rec = _record("ConcurrentWF", steps, topology=topo, total_wall=3.0)
        md = format_report_md([rec])

        assert "Memory Profile" in md
        assert "(concurrent)" in md

    def test_memory_profile_absent_without_overhead(self):
        """Memory Profile section is absent when no steps have overhead data."""
        steps = [
            _step(0, "Plain", wall=1.0, latency_pct=100.0, memory_pct=100.0),
        ]
        rec = _record("PlainWF", steps)
        md = format_report_md([rec])

        assert "Memory Profile" not in md


# ---------------------------------------------------------------------------
# save_report_md
# ---------------------------------------------------------------------------
class TestSaveReportMd:
    """Tests for save_report_md()."""

    def test_save_to_file(self):
        """Saves report to specified file path."""
        rec = _record("WF", [_step(0, "S", wall=1.0)])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.md"
            result = save_report_md([rec], path)

            assert result == path
            assert path.exists()
            content = path.read_text()
            assert "# GRDL Benchmark Report" in content

    def test_save_to_directory(self):
        """Saves with auto-generated filename when given a directory."""
        rec = _record("WF", [_step(0, "S", wall=1.0)])
        with tempfile.TemporaryDirectory() as tmpdir:
            result = save_report_md([rec], Path(tmpdir))

            assert result.exists()
            assert result.suffix == ".md"
            assert "benchmark_report_" in result.name

    def test_creates_parent_dirs(self):
        """Creates parent directories if they don't exist."""
        rec = _record("WF", [_step(0, "S", wall=1.0)])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sub" / "dir" / "report.md"
            result = save_report_md([rec], path)

            assert result.exists()
            assert result == path
