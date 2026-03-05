# -*- coding: utf-8 -*-
"""
Tests for comparison engine — cross-workflow benchmark comparison.

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
2026-03-05
"""

# Third-party
import pytest

# Internal
from grdl_te.benchmarking.models import (
    AggregatedMetrics,
    BenchmarkRecord,
    HardwareSnapshot,
    StepBenchmarkResult,
)
from grdl_te.benchmarking.comparison import compare_records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Module-level hardware cache — captured once per module to avoid
# redundant OS/platform calls across ~11 tests.
_CACHED_HW: "HardwareSnapshot | None" = None


def _hw() -> HardwareSnapshot:
    global _CACHED_HW
    if _CACHED_HW is None:
        _CACHED_HW = HardwareSnapshot.capture()
    return _CACHED_HW


def _agg(mean: float) -> AggregatedMetrics:
    return AggregatedMetrics.from_values([mean])


def _step(index: int, name: str, wall: float, mem: float = 1000.0) -> StepBenchmarkResult:
    return StepBenchmarkResult(
        step_index=index,
        processor_name=name,
        wall_time_s=_agg(wall),
        cpu_time_s=_agg(wall * 0.5),
        peak_rss_bytes=_agg(mem),
        gpu_used=False,
        sample_count=1,
    )


def _record(
    name: str,
    steps: list,
    total_wall: float = None,
) -> BenchmarkRecord:
    if total_wall is None:
        total_wall = sum(s.wall_time_s.mean for s in steps)
    total_cpu = sum(s.cpu_time_s.mean for s in steps)
    # Use sum for concurrent steps (overlapping footprints) vs max for
    # sequential (process high-water mark is the single largest step).
    if any(getattr(s, 'concurrent', False) for s in steps):
        total_mem = sum(s.peak_rss_bytes.mean for s in steps)
    else:
        total_mem = max((s.peak_rss_bytes.mean for s in steps), default=0.0)

    return BenchmarkRecord.create(
        benchmark_type="active",
        workflow_name=name,
        workflow_version="1.0.0",
        iterations=1,
        hardware=_hw(),
        total_wall_time=_agg(total_wall),
        total_cpu_time=_agg(total_cpu),
        total_peak_rss=_agg(total_mem),
        step_results=steps,
        raw_metrics=[],
        tags={},
    )


# ---------------------------------------------------------------------------
# compare_records
# ---------------------------------------------------------------------------
class TestCompareRecords:
    """Tests for compare_records()."""

    def test_empty_records(self):
        """Empty list returns empty ComparisonResult."""
        result = compare_records([])
        assert result.record_labels == []
        assert result.bottlenecks == []

    def test_single_record(self):
        """Single record still produces valid result."""
        rec = _record("WF1", [_step(0, "StepA", wall=1.0)])
        result = compare_records([rec])
        assert len(result.record_labels) == 1
        assert result.record_labels[0] == "WF1"

    def test_wall_time_summary(self):
        """Wall time summary maps labels to total wall mean."""
        rec_a = _record("WF1", [_step(0, "S", wall=5.0)], total_wall=5.0)
        rec_b = _record("WF2", [_step(0, "S", wall=10.0)], total_wall=10.0)
        result = compare_records([rec_a, rec_b])

        assert result.wall_time_summary["WF1"] == pytest.approx(5.0)
        assert result.wall_time_summary["WF2"] == pytest.approx(10.0)

    def test_bottleneck_ranking(self):
        """Bottlenecks ranked by latency_pct descending."""
        rec = _record("WF", [
            _step(0, "Small", wall=1.0),
            _step(1, "Big", wall=9.0),
        ])
        # Give them latency_pct for bottleneck ranking
        rec.step_results[0].latency_pct = 10.0
        rec.step_results[1].latency_pct = 90.0

        result = compare_records([rec])
        assert result.bottlenecks[0]["step_name"] == "Big"
        assert result.bottlenecks[0]["latency_pct"] == 90.0

    def test_custom_labels(self):
        """Custom labels override auto-generated ones."""
        rec_a = _record("WF1", [_step(0, "S", wall=1.0)])
        rec_b = _record("WF2", [_step(0, "S", wall=2.0)])
        result = compare_records([rec_a, rec_b], labels=["Alpha", "Beta"])

        assert result.record_labels == ["Alpha", "Beta"]
        assert "Alpha" in result.wall_time_summary
        assert "Beta" in result.wall_time_summary

    def test_duplicate_workflow_names(self):
        """Duplicate names get numeric suffixes."""
        rec_a = _record("Pipeline", [_step(0, "S", wall=1.0)])
        rec_b = _record("Pipeline", [_step(0, "S", wall=2.0)])
        result = compare_records([rec_a, rec_b])

        assert result.record_labels[0] == "Pipeline (1)"
        assert result.record_labels[1] == "Pipeline (2)"

    def test_skipped_steps_excluded(self):
        """Skipped steps (zero wall+cpu) are excluded from bottlenecks."""
        rec = _record("WF", [
            _step(0, "Active", wall=1.0),
            StepBenchmarkResult(
                step_index=1,
                processor_name="Skipped",
                wall_time_s=_agg(0.0),
                cpu_time_s=_agg(0.0),
                peak_rss_bytes=_agg(0.0),
                gpu_used=False,
                sample_count=1,
            ),
        ])
        rec.step_results[0].latency_pct = 100.0
        result = compare_records([rec])
        step_names = {b["step_name"] for b in result.bottlenecks}
        assert "Active" in step_names
        assert "Skipped" not in step_names
