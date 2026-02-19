# -*- coding: utf-8 -*-
"""
Tests for benchmark data models.

Validates construction, aggregation, and JSON round-tripping for
``AggregatedMetrics``, ``HardwareSnapshot``, ``StepBenchmarkResult``,
and ``BenchmarkRecord``.

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
import json

# Third-party
import numpy as np
import pytest

# Internal
from grdl_te.benchmarking.models import (
    AggregatedMetrics,
    BenchmarkRecord,
    HardwareSnapshot,
    StepBenchmarkResult,
)

# Optional: grdl-runtime types for StepBenchmarkResult.from_step_metrics
try:
    from grdl_rt.execution.metrics import StepMetrics
    _HAS_RUNTIME = True
except ImportError:
    _HAS_RUNTIME = False


# ---------------------------------------------------------------------------
# AggregatedMetrics
# ---------------------------------------------------------------------------

class TestAggregatedMetrics:
    """Tests for AggregatedMetrics dataclass."""

    def test_from_values_normal(self):
        """Multiple values produce correct statistics."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        agg = AggregatedMetrics.from_values(values)

        assert agg.count == 5
        assert agg.min == 1.0
        assert agg.max == 5.0
        assert agg.mean == pytest.approx(3.0)
        assert agg.median == pytest.approx(3.0)
        assert agg.stddev == pytest.approx(np.std(values, ddof=1))
        assert agg.p95 == pytest.approx(np.percentile(values, 95))
        assert agg.values == tuple(values)

    def test_from_values_single(self):
        """Single value produces degenerate statistics."""
        agg = AggregatedMetrics.from_values([42.0])

        assert agg.count == 1
        assert agg.min == 42.0
        assert agg.max == 42.0
        assert agg.mean == 42.0
        assert agg.median == 42.0
        assert agg.stddev == 0.0  # ddof=0 for single value
        assert agg.p95 == 42.0

    def test_from_values_identical(self):
        """All-identical values produce zero stddev."""
        agg = AggregatedMetrics.from_values([7.0, 7.0, 7.0])

        assert agg.mean == 7.0
        assert agg.stddev == 0.0
        assert agg.min == agg.max == 7.0

    def test_from_values_empty_raises(self):
        """Empty list raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            AggregatedMetrics.from_values([])

    def test_to_dict_from_dict_roundtrip(self):
        """Serialization round-trip preserves all fields."""
        original = AggregatedMetrics.from_values([1.0, 2.0, 3.0])
        restored = AggregatedMetrics.from_dict(original.to_dict())

        assert restored.count == original.count
        assert restored.min == original.min
        assert restored.max == original.max
        assert restored.mean == pytest.approx(original.mean)
        assert restored.median == pytest.approx(original.median)
        assert restored.stddev == pytest.approx(original.stddev)
        assert restored.p95 == pytest.approx(original.p95)
        assert restored.values == original.values

    def test_frozen(self):
        """AggregatedMetrics is immutable."""
        agg = AggregatedMetrics.from_values([1.0, 2.0])
        with pytest.raises(AttributeError):
            agg.mean = 999.0


# ---------------------------------------------------------------------------
# HardwareSnapshot
# ---------------------------------------------------------------------------

class TestHardwareSnapshot:
    """Tests for HardwareSnapshot dataclass."""

    def test_capture_produces_valid_snapshot(self):
        """capture() returns a populated snapshot."""
        snap = HardwareSnapshot.capture()

        assert snap.cpu_count >= 1
        assert isinstance(snap.gpu_available, bool)
        assert isinstance(snap.gpu_devices, tuple)
        assert snap.platform_info  # non-empty string
        assert snap.python_version  # non-empty string
        assert snap.hostname  # non-empty string
        assert snap.captured_at  # ISO 8601 string

    def test_to_dict_from_dict_roundtrip(self):
        """Serialization round-trip preserves all fields."""
        original = HardwareSnapshot.capture()
        restored = HardwareSnapshot.from_dict(original.to_dict())

        assert restored.cpu_count == original.cpu_count
        assert restored.total_memory_bytes == original.total_memory_bytes
        assert restored.gpu_available == original.gpu_available
        assert restored.gpu_memory_bytes == original.gpu_memory_bytes
        assert restored.platform_info == original.platform_info
        assert restored.python_version == original.python_version
        assert restored.hostname == original.hostname
        assert restored.captured_at == original.captured_at

    def test_frozen(self):
        """HardwareSnapshot is immutable."""
        snap = HardwareSnapshot.capture()
        with pytest.raises(AttributeError):
            snap.cpu_count = 999


# ---------------------------------------------------------------------------
# StepBenchmarkResult
# ---------------------------------------------------------------------------

class TestStepBenchmarkResult:
    """Tests for StepBenchmarkResult dataclass."""

    @pytest.mark.skipif(not _HAS_RUNTIME, reason="grdl-runtime not installed")
    def test_from_step_metrics(self):
        """Aggregation from StepMetrics list works correctly."""
        metrics = [
            StepMetrics(
                step_index=0,
                processor_name="TestProcessor",
                wall_time_s=1.0 + i * 0.1,
                cpu_time_s=0.5 + i * 0.05,
                peak_rss_bytes=1000 + i * 100,
                gpu_used=False,
            )
            for i in range(5)
        ]

        result = StepBenchmarkResult.from_step_metrics(metrics)

        assert result.step_index == 0
        assert result.processor_name == "TestProcessor"
        assert result.sample_count == 5
        assert result.wall_time_s.count == 5
        assert result.wall_time_s.min == pytest.approx(1.0)
        assert result.wall_time_s.max == pytest.approx(1.4)
        assert result.gpu_used is False
        assert result.gpu_memory_bytes is None

    @pytest.mark.skipif(not _HAS_RUNTIME, reason="grdl-runtime not installed")
    def test_from_step_metrics_with_gpu(self):
        """GPU metrics are aggregated when present."""
        metrics = [
            StepMetrics(
                step_index=1,
                processor_name="GpuProcessor",
                wall_time_s=0.5,
                cpu_time_s=0.1,
                peak_rss_bytes=2000,
                gpu_used=True,
                gpu_memory_bytes=1_000_000 + i * 100_000,
            )
            for i in range(3)
        ]

        result = StepBenchmarkResult.from_step_metrics(metrics)

        assert result.gpu_used is True
        assert result.gpu_memory_bytes is not None
        assert result.gpu_memory_bytes.count == 3

    @pytest.mark.skipif(not _HAS_RUNTIME, reason="grdl-runtime not installed")
    def test_from_step_metrics_empty_raises(self):
        """Empty metrics list raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            StepBenchmarkResult.from_step_metrics([])

    def test_to_dict_from_dict_roundtrip(self):
        """Serialization round-trip preserves all fields."""
        original = StepBenchmarkResult(
            step_index=0,
            processor_name="TestProc",
            wall_time_s=AggregatedMetrics.from_values([1.0, 2.0]),
            cpu_time_s=AggregatedMetrics.from_values([0.5, 1.0]),
            peak_rss_bytes=AggregatedMetrics.from_values([1000.0, 2000.0]),
            gpu_used=False,
            sample_count=2,
        )
        restored = StepBenchmarkResult.from_dict(original.to_dict())

        assert restored.step_index == original.step_index
        assert restored.processor_name == original.processor_name
        assert restored.wall_time_s.mean == pytest.approx(
            original.wall_time_s.mean
        )
        assert restored.sample_count == original.sample_count

    def test_to_dict_from_dict_with_gpu(self):
        """Round-trip with GPU memory metrics."""
        original = StepBenchmarkResult(
            step_index=0,
            processor_name="GpuProc",
            wall_time_s=AggregatedMetrics.from_values([1.0]),
            cpu_time_s=AggregatedMetrics.from_values([0.5]),
            peak_rss_bytes=AggregatedMetrics.from_values([1000.0]),
            gpu_used=True,
            gpu_memory_bytes=AggregatedMetrics.from_values([500000.0]),
            sample_count=1,
        )
        restored = StepBenchmarkResult.from_dict(original.to_dict())

        assert restored.gpu_used is True
        assert restored.gpu_memory_bytes is not None
        assert restored.gpu_memory_bytes.mean == pytest.approx(500000.0)


# ---------------------------------------------------------------------------
# BenchmarkRecord
# ---------------------------------------------------------------------------

class TestBenchmarkRecord:
    """Tests for BenchmarkRecord dataclass."""

    def _make_record(self) -> BenchmarkRecord:
        """Helper to create a minimal valid record."""
        return BenchmarkRecord.create(
            benchmark_type="active",
            workflow_name="TestWorkflow",
            workflow_version="1.0.0",
            iterations=3,
            hardware=HardwareSnapshot.capture(),
            total_wall_time=AggregatedMetrics.from_values([1.0, 2.0, 3.0]),
            total_cpu_time=AggregatedMetrics.from_values([0.5, 1.0, 1.5]),
            total_peak_rss=AggregatedMetrics.from_values(
                [1000.0, 2000.0, 3000.0]
            ),
            step_results=[
                StepBenchmarkResult(
                    step_index=0,
                    processor_name="Step0",
                    wall_time_s=AggregatedMetrics.from_values([0.5, 1.0, 1.5]),
                    cpu_time_s=AggregatedMetrics.from_values([0.2, 0.5, 0.7]),
                    peak_rss_bytes=AggregatedMetrics.from_values(
                        [500.0, 1000.0, 1500.0]
                    ),
                    gpu_used=False,
                    sample_count=3,
                ),
            ],
            tags={"branch": "main"},
            metadata={"note": "test record"},
        )

    def test_create_generates_id_and_timestamp(self):
        """create() produces a unique ID and timestamp."""
        record = self._make_record()

        assert record.benchmark_id  # non-empty UUID
        assert record.created_at  # non-empty ISO timestamp
        assert record.benchmark_type == "active"
        assert record.workflow_name == "TestWorkflow"
        assert record.iterations == 3

    def test_to_dict_from_dict_roundtrip(self):
        """Dictionary serialization round-trip."""
        original = self._make_record()
        restored = BenchmarkRecord.from_dict(original.to_dict())

        assert restored.benchmark_id == original.benchmark_id
        assert restored.benchmark_type == original.benchmark_type
        assert restored.workflow_name == original.workflow_name
        assert restored.workflow_version == original.workflow_version
        assert restored.iterations == original.iterations
        assert restored.total_wall_time.mean == pytest.approx(
            original.total_wall_time.mean
        )
        assert len(restored.step_results) == len(original.step_results)
        assert restored.tags == original.tags
        assert restored.metadata == original.metadata

    def test_to_json_from_json_roundtrip(self):
        """JSON string serialization round-trip."""
        original = self._make_record()
        json_str = original.to_json()

        # Valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

        restored = BenchmarkRecord.from_json(json_str)
        assert restored.benchmark_id == original.benchmark_id
        assert restored.workflow_name == original.workflow_name

    def test_unique_ids(self):
        """Each create() call produces a unique benchmark_id."""
        r1 = self._make_record()
        r2 = self._make_record()
        assert r1.benchmark_id != r2.benchmark_id
