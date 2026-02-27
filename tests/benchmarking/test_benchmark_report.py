# -*- coding: utf-8 -*-
"""
Tests for benchmark report generation.

Validates ``format_report``, ``print_report``, and ``save_report`` against
synthetic benchmark records covering single-record, multi-record,
step-detail, and GPU-memory scenarios.

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
from pathlib import Path

# Third-party
import pytest

# Internal
from grdl_te.benchmarking.models import (
    AggregatedMetrics,
    BenchmarkRecord,
    HardwareSnapshot,
    StepBenchmarkResult,
)
from grdl_te.benchmarking.report import format_report, print_report, save_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_hw() -> HardwareSnapshot:
    """Create a deterministic HardwareSnapshot for testing."""
    return HardwareSnapshot(
        cpu_count=8,
        total_memory_bytes=16 * 1024 ** 3,
        gpu_available=False,
        gpu_devices=(),
        gpu_memory_bytes=0,
        platform_info="Linux-5.14.0-test",
        python_version="3.11.0 (test)",
        hostname="testhost",
        captured_at="2026-02-18T12:00:00+00:00",
    )


def _make_record(
    workflow_name: str = "TestBenchmark",
    benchmark_type: str = "component",
    version: str = "1.0.0",
    tags: dict = None,
    step_results: list = None,
) -> BenchmarkRecord:
    """Create a minimal valid BenchmarkRecord for testing."""
    return BenchmarkRecord.create(
        benchmark_type=benchmark_type,
        workflow_name=workflow_name,
        workflow_version=version,
        iterations=5,
        hardware=_make_hw(),
        total_wall_time=AggregatedMetrics.from_values(
            [0.10, 0.12, 0.11, 0.13, 0.09]
        ),
        total_cpu_time=AggregatedMetrics.from_values(
            [0.08, 0.10, 0.09, 0.11, 0.07]
        ),
        total_peak_rss=AggregatedMetrics.from_values(
            [50000.0, 52000.0, 51000.0, 53000.0, 49000.0]
        ),
        tags=tags or {"module": "test_module", "array_size": "small",
                       "rows": "512", "cols": "512"},
        step_results=step_results,
    )


def _make_step(
    step_index: int = 0,
    processor_name: str = "TestProcessor",
    gpu_used: bool = False,
    gpu_memory_bytes: AggregatedMetrics = None,
) -> StepBenchmarkResult:
    """Create a minimal StepBenchmarkResult for testing."""
    return StepBenchmarkResult(
        step_index=step_index,
        processor_name=processor_name,
        wall_time_s=AggregatedMetrics.from_values([0.05, 0.06, 0.04]),
        cpu_time_s=AggregatedMetrics.from_values([0.04, 0.05, 0.03]),
        peak_rss_bytes=AggregatedMetrics.from_values(
            [25000.0, 26000.0, 24000.0]
        ),
        gpu_used=gpu_used,
        gpu_memory_bytes=gpu_memory_bytes,
        sample_count=3,
    )


# ---------------------------------------------------------------------------
# Tests: format_report
# ---------------------------------------------------------------------------
class TestFormatReport:
    """Tests for format_report()."""

    def test_returns_nonempty_string(self):
        """format_report returns a non-empty string."""
        report = format_report([_make_record()])
        assert isinstance(report, str)
        assert len(report) > 0

    def test_empty_records_raises(self):
        """Empty records list raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            format_report([])

    def test_contains_header(self):
        """Report contains the header section."""
        report = format_report([_make_record()])
        assert "GRDL BENCHMARK REPORT" in report

    def test_contains_hardware_section(self):
        """Report includes hardware information from the snapshot."""
        report = format_report([_make_record()])
        assert "HARDWARE" in report
        assert "testhost" in report
        assert "Linux-5.14.0-test" in report
        assert "CPUs:" in report

    def test_contains_configuration(self):
        """Report includes run configuration details."""
        report = format_report([_make_record()])
        assert "RUN CONFIGURATION" in report
        assert "Iterations:" in report
        assert "small" in report

    def test_contains_all_benchmark_names(self):
        """Every record's workflow_name appears in the report."""
        r1 = _make_record(workflow_name="AlphaBench")
        r2 = _make_record(workflow_name="BetaBench")
        report = format_report([r1, r2])
        assert "AlphaBench" in report
        assert "BetaBench" in report

    def test_contains_statistics_columns(self):
        """Report includes full statistical column headers."""
        report = format_report([_make_record()])
        assert "Mean" in report
        assert "Median" in report
        assert "StdDev" in report
        assert "P95" in report
        assert "Min" in report
        assert "Max" in report

    def test_contains_detailed_results(self):
        """Report includes the detailed results section."""
        report = format_report([_make_record()])
        assert "DETAILED RESULTS" in report

    def test_contains_module_summary(self):
        """Report includes module aggregation section."""
        report = format_report([_make_record()])
        assert "MODULE SUMMARY" in report
        assert "test_module" in report

    def test_contains_overall_summary(self):
        """Report includes overall summary with fastest/slowest."""
        report = format_report([_make_record()])
        assert "OVERALL SUMMARY" in report
        assert "Fastest:" in report
        assert "Slowest:" in report
        assert "Total Benchmarks:" in report

    def test_single_record(self):
        """Report works correctly with exactly one record."""
        report = format_report([_make_record()])
        assert "Total Benchmarks:   1" in report

    def test_records_without_steps(self):
        """Records with no step_results still format cleanly."""
        record = _make_record(step_results=[])
        report = format_report([record])
        assert "TestBenchmark" in report
        # Should not contain "Steps" sub-section
        assert "Steps (" not in report

    def test_records_with_steps(self):
        """Records with step_results show per-step breakdown."""
        step = _make_step(processor_name="MyFilter")
        record = _make_record(step_results=[step])
        report = format_report([record])
        assert "1 ran, 0 skipped / 1 total" in report
        assert "MyFilter" in report

    def test_skipped_steps_shown_inline(self):
        """Skipped steps appear as one-liners in step order."""
        ran_step = _make_step(step_index=0, processor_name="RanFilter")
        skipped_step = StepBenchmarkResult(
            step_index=1,
            processor_name="SkippedProcessor",
            wall_time_s=AggregatedMetrics.from_values([0.0, 0.0, 0.0]),
            cpu_time_s=AggregatedMetrics.from_values([0.0, 0.0, 0.0]),
            peak_rss_bytes=AggregatedMetrics.from_values([0.0, 0.0, 0.0]),
            gpu_used=False,
            sample_count=3,
        )
        record = _make_record(step_results=[ran_step, skipped_step])
        report = format_report([record])
        assert "1 ran, 1 skipped / 2 total" in report
        assert "RanFilter" in report
        assert "SkippedProcessor" in report
        assert "SKIPPED (condition not met)" in report

    def test_records_with_gpu_memory(self):
        """GPU memory statistics appear when present in steps."""
        gpu_mem = AggregatedMetrics.from_values(
            [1000000.0, 1100000.0, 1050000.0]
        )
        step = _make_step(gpu_used=True, gpu_memory_bytes=gpu_mem)
        record = _make_record(step_results=[step])
        report = format_report([record])
        assert "GPU: Yes" in report
        assert "GPU Mem" in report

    def test_gpu_not_shown_when_absent(self):
        """GPU memory row absent when step has no GPU memory."""
        step = _make_step(gpu_used=False)
        record = _make_record(step_results=[step])
        report = format_report([record])
        assert "GPU: No" in report
        assert "GPU Mem" not in report

    def test_hardware_gpu_devices(self):
        """GPU device details appear when hardware has GPUs."""
        hw = HardwareSnapshot(
            cpu_count=16,
            total_memory_bytes=64 * 1024 ** 3,
            gpu_available=True,
            gpu_devices=(
                {"name": "NVIDIA A100", "memory_bytes": 40 * 1024 ** 3,
                 "device_index": 0},
            ),
            gpu_memory_bytes=40 * 1024 ** 3,
            platform_info="Linux-test",
            python_version="3.11.0",
            hostname="gpu-host",
            captured_at="2026-02-18T12:00:00+00:00",
        )
        record = BenchmarkRecord.create(
            benchmark_type="component",
            workflow_name="GpuTest",
            workflow_version="1.0.0",
            iterations=3,
            hardware=hw,
            total_wall_time=AggregatedMetrics.from_values([1.0, 2.0, 3.0]),
            total_cpu_time=AggregatedMetrics.from_values([0.5, 1.0, 1.5]),
            total_peak_rss=AggregatedMetrics.from_values([1000.0, 2000.0, 3000.0]),
        )
        report = format_report([record])
        assert "GPU Available:   Yes" in report
        assert "NVIDIA A100" in report

    def test_tags_displayed(self):
        """Record tags are shown in the detail section."""
        record = _make_record(tags={"module": "io", "data": "real"})
        report = format_report([record])
        assert "data=real" in report
        assert "module=io" in report

    def test_multiple_modules_aggregated(self):
        """Module summary groups records by their module tag."""
        r1 = _make_record(workflow_name="A", tags={"module": "filters"})
        r2 = _make_record(workflow_name="B", tags={"module": "filters"})
        r3 = _make_record(workflow_name="C", tags={"module": "io"})
        report = format_report([r1, r2, r3])
        assert "filters" in report
        assert "io" in report

    def test_sorted_slowest_first(self):
        """Detailed results are sorted slowest-first."""
        slow = BenchmarkRecord.create(
            benchmark_type="component",
            workflow_name="SlowBench",
            workflow_version="1.0.0",
            iterations=3,
            hardware=_make_hw(),
            total_wall_time=AggregatedMetrics.from_values([10.0, 11.0, 12.0]),
            total_cpu_time=AggregatedMetrics.from_values([5.0, 6.0, 7.0]),
            total_peak_rss=AggregatedMetrics.from_values([1000.0, 2000.0, 3000.0]),
        )
        fast = _make_record(workflow_name="FastBench")
        report = format_report([fast, slow])
        # SlowBench should appear before FastBench in detailed section
        slow_pos = report.index("SlowBench")
        fast_pos = report.index("FastBench")
        assert slow_pos < fast_pos


# ---------------------------------------------------------------------------
# Tests: print_report
# ---------------------------------------------------------------------------
class TestPrintReport:
    """Tests for print_report()."""

    def test_prints_to_stdout(self, capsys):
        """print_report writes the full report to stdout."""
        print_report([_make_record()])
        captured = capsys.readouterr()
        assert "GRDL BENCHMARK REPORT" in captured.out
        assert "OVERALL SUMMARY" in captured.out


# ---------------------------------------------------------------------------
# Tests: save_report
# ---------------------------------------------------------------------------
class TestSaveReport:
    """Tests for save_report()."""

    def test_writes_to_file(self, tmp_path):
        """save_report creates a file with report content."""
        out_file = tmp_path / "report.txt"
        result = save_report([_make_record()], out_file)
        assert result == out_file
        assert out_file.exists()
        content = out_file.read_text(encoding="utf-8")
        assert "GRDL BENCHMARK REPORT" in content

    def test_writes_to_directory(self, tmp_path):
        """When path is a directory, generates timestamped filename."""
        out_dir = tmp_path / "reports"
        out_dir.mkdir()
        result = save_report([_make_record()], out_dir)
        assert result.parent == out_dir
        assert result.name.startswith("benchmark_report_")
        assert result.name.endswith(".txt")
        assert result.exists()

    def test_creates_parent_directories(self, tmp_path):
        """save_report creates intermediate directories."""
        out_file = tmp_path / "deep" / "nested" / "report.txt"
        result = save_report([_make_record()], out_file)
        assert result.exists()
        assert result.read_text(encoding="utf-8") != ""

    def test_returns_path(self, tmp_path):
        """save_report returns a Path object."""
        out_file = tmp_path / "report.txt"
        result = save_report([_make_record()], out_file)
        assert isinstance(result, Path)

    def test_directory_without_extension(self, tmp_path):
        """Path without extension that doesn't exist is treated as directory."""
        out_dir = tmp_path / "new_reports"
        result = save_report([_make_record()], out_dir)
        assert out_dir.is_dir()
        assert result.parent == out_dir
        assert result.name.startswith("benchmark_report_")
