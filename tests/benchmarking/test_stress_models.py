# -*- coding: utf-8 -*-
"""
Tests for Stress Test Data Models.

Validates ``StressTestConfig``, ``StressTestEvent``, ``FailurePoint``,
``StressTestSummary``, and ``StressTestRecord`` serialization round-trips,
field validation, and the "loadable" comparison contract — two records
produced from identical configs must produce the same JSON structure.

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

# Third-party
import pytest

# Internal
from grdl_te.benchmarking.stress_models import (
    STRESS_SCHEMA_VERSION,
    FailurePoint,
    StressTestConfig,
    StressTestEvent,
    StressTestRecord,
    StressTestSummary,
)

pytestmark = pytest.mark.stress_test


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_event(
    concurrency: int = 2,
    success: bool = True,
    error_type: str = None,
    latency_s: float = 0.05,
    peak_rss_bytes: int = 1024,
) -> StressTestEvent:
    return StressTestEvent(
        concurrency_level=concurrency,
        payload_shape=(512, 512),
        success=success,
        latency_s=latency_s,
        peak_rss_bytes=peak_rss_bytes,
        error_type=error_type,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _make_failure_point(concurrency: int = 4) -> FailurePoint:
    return FailurePoint(
        concurrency_level=concurrency,
        payload_shape=(512, 512),
        error_type="MemoryError",
        error_message="Out of memory",
        memory_bytes_at_failure=512 * 1024 * 1024,
        first_occurrence_at=datetime.now(timezone.utc).isoformat(),
    )


def _make_record(
    *,
    config: StressTestConfig = None,
    events=None,
    failure_points=None,
) -> StressTestRecord:
    if config is None:
        config = StressTestConfig(
            max_concurrency=4,
            ramp_steps=2,
            duration_per_step_s=1.0,
        )
    if events is None:
        events = [_make_event(1), _make_event(2)]
    if failure_points is None:
        failure_points = []
    return StressTestRecord.create(
        component_name="TestComponent",
        component_version="1.0.0",
        hardware=None,
        config=config,
        events=events,
        failure_points=failure_points,
        related_benchmark_id=None,
        tags={"test": "true"},
    )


# ---------------------------------------------------------------------------
# StressTestConfig
# ---------------------------------------------------------------------------

class TestStressTestConfig:
    """Tests for StressTestConfig model."""

    def test_defaults(self):
        """Default config has valid fields."""
        cfg = StressTestConfig()
        assert cfg.start_concurrency >= 1
        assert cfg.max_concurrency >= cfg.start_concurrency
        assert cfg.ramp_steps >= 1
        assert cfg.duration_per_step_s > 0
        assert cfg.timeout_per_call_s > 0

    def test_round_trip(self):
        """Config survives to_dict → from_dict round-trip."""
        cfg = StressTestConfig(
            start_concurrency=2,
            max_concurrency=16,
            ramp_steps=4,
            duration_per_step_s=5.0,
            payload_size="small",
            timeout_per_call_s=20.0,
        )
        restored = StressTestConfig.from_dict(cfg.to_dict())
        assert restored == cfg

    def test_validation_start_concurrency(self):
        """Rejects start_concurrency < 1."""
        with pytest.raises(ValueError, match="start_concurrency"):
            StressTestConfig(start_concurrency=0)

    def test_validation_max_lt_start(self):
        """Rejects max_concurrency < start_concurrency."""
        with pytest.raises(ValueError, match="max_concurrency"):
            StressTestConfig(start_concurrency=8, max_concurrency=4)

    def test_validation_ramp_steps(self):
        """Rejects ramp_steps < 1."""
        with pytest.raises(ValueError, match="ramp_steps"):
            StressTestConfig(ramp_steps=0)

    def test_validation_duration(self):
        """Rejects duration_per_step_s <= 0."""
        with pytest.raises(ValueError, match="duration_per_step_s"):
            StressTestConfig(duration_per_step_s=0.0)

    def test_concurrency_levels_sorted(self):
        """concurrency_levels() returns a sorted list."""
        cfg = StressTestConfig(
            start_concurrency=1, max_concurrency=8, ramp_steps=4
        )
        levels = cfg.concurrency_levels()
        assert levels == sorted(levels)
        assert levels[0] >= cfg.start_concurrency
        assert levels[-1] <= cfg.max_concurrency

    def test_concurrency_levels_respects_max(self):
        """No level exceeds max_concurrency."""
        cfg = StressTestConfig(
            start_concurrency=1, max_concurrency=10, ramp_steps=10
        )
        levels = cfg.concurrency_levels()
        assert all(l <= 10 for l in levels)

    def test_concurrency_levels_single_step(self):
        """Single step config returns exactly one level."""
        cfg = StressTestConfig(
            start_concurrency=4, max_concurrency=4, ramp_steps=1
        )
        levels = cfg.concurrency_levels()
        assert len(levels) == 1
        assert levels[0] == 4


# ---------------------------------------------------------------------------
# StressTestEvent
# ---------------------------------------------------------------------------

class TestStressTestEvent:
    """Tests for StressTestEvent model."""

    def test_successful_event_round_trip(self):
        """Successful event survives round-trip."""
        evt = _make_event(success=True)
        restored = StressTestEvent.from_dict(evt.to_dict())
        assert restored.concurrency_level == evt.concurrency_level
        assert restored.success is True
        assert restored.error_type is None

    def test_failed_event_round_trip(self):
        """Failed event preserves error_type."""
        evt = _make_event(success=False, error_type="MemoryError")
        d = evt.to_dict()
        restored = StressTestEvent.from_dict(d)
        assert restored.success is False
        assert restored.error_type == "MemoryError"

    def test_error_type_absent_for_success(self):
        """Successful events do not include error_type key in dict."""
        evt = _make_event(success=True)
        d = evt.to_dict()
        assert "error_type" not in d

    def test_payload_shape_preserved(self):
        """payload_shape survives serialization as tuple."""
        evt = StressTestEvent(
            concurrency_level=1,
            payload_shape=(1024, 2048),
            success=True,
            latency_s=0.1,
            peak_rss_bytes=0,
            error_type=None,
            timestamp="2026-04-07T00:00:00+00:00",
        )
        restored = StressTestEvent.from_dict(evt.to_dict())
        assert restored.payload_shape == (1024, 2048)


# ---------------------------------------------------------------------------
# FailurePoint
# ---------------------------------------------------------------------------

class TestFailurePoint:
    """Tests for FailurePoint model."""

    def test_round_trip(self):
        """FailurePoint survives to_dict → from_dict."""
        fp = _make_failure_point()
        restored = FailurePoint.from_dict(fp.to_dict())
        assert restored.concurrency_level == fp.concurrency_level
        assert restored.error_type == fp.error_type
        assert restored.memory_bytes_at_failure == fp.memory_bytes_at_failure
        assert restored.payload_shape == fp.payload_shape

    def test_error_message_preserved(self):
        """Long error messages are preserved in full."""
        long_msg = "x" * 500
        fp = FailurePoint(
            concurrency_level=4,
            payload_shape=(512, 512),
            error_type="RuntimeError",
            error_message=long_msg,
            memory_bytes_at_failure=0,
            first_occurrence_at="2026-04-07T00:00:00+00:00",
        )
        restored = FailurePoint.from_dict(fp.to_dict())
        assert restored.error_message == long_msg


# ---------------------------------------------------------------------------
# StressTestSummary
# ---------------------------------------------------------------------------

class TestStressTestSummary:
    """Tests for StressTestSummary derivation and serialization."""

    def test_from_events_no_failures(self):
        """All-success events produce None failure fields."""
        events = [_make_event(1), _make_event(2), _make_event(4)]
        summary = StressTestSummary.from_events(events, [])
        assert summary.saturation_concurrency is None
        assert summary.first_failure_mode is None
        assert summary.failed_calls == 0
        assert summary.total_calls == 3
        assert summary.max_sustained_concurrency == 4

    def test_from_events_with_failures(self):
        """Failure events populate saturation fields."""
        events = [
            _make_event(1, success=True),
            _make_event(2, success=True),
            _make_event(4, success=False, error_type="MemoryError"),
        ]
        fp = _make_failure_point(concurrency=4)
        summary = StressTestSummary.from_events(events, [fp])
        assert summary.failed_calls == 1
        assert summary.saturation_concurrency == 4
        assert summary.first_failure_mode == "MemoryError"
        assert summary.max_sustained_concurrency == 2

    def test_from_events_empty(self):
        """Empty events list produces zeroed summary."""
        s = StressTestSummary.from_events([], [])
        assert s.total_calls == 0
        assert s.max_sustained_concurrency == 0

    def test_p99_latency_computed(self):
        """p99 is derived from successful call latencies."""
        events = [
            _make_event(1, latency_s=float(i) * 0.01)
            for i in range(1, 101)
        ]
        summary = StressTestSummary.from_events(events, [])
        assert summary.p99_latency_s > 0.0

    def test_round_trip(self):
        """Summary survives to_dict → from_dict."""
        events = [_make_event(2), _make_event(2, success=False, error_type="OSError")]
        fp = _make_failure_point(2)
        s = StressTestSummary.from_events(events, [fp])
        restored = StressTestSummary.from_dict(s.to_dict())
        assert restored.total_calls == s.total_calls
        assert restored.saturation_concurrency == s.saturation_concurrency


# ---------------------------------------------------------------------------
# StressTestRecord
# ---------------------------------------------------------------------------

class TestStressTestRecord:
    """Tests for StressTestRecord serialization and fields."""

    def test_create_sets_schema_version(self):
        """Created record carries the current schema version."""
        record = _make_record()
        assert record.schema_version == STRESS_SCHEMA_VERSION

    def test_create_generates_unique_id(self):
        """Two records created in sequence have different IDs."""
        r1 = _make_record()
        r2 = _make_record()
        assert r1.stress_test_id != r2.stress_test_id

    def test_create_sets_timestamp(self):
        """Record has a non-empty created_at timestamp."""
        record = _make_record()
        assert record.created_at
        # Should be parseable as ISO 8601
        datetime.fromisoformat(record.created_at.replace("Z", "+00:00"))

    def test_json_round_trip(self):
        """Record survives to_json → from_json."""
        orig = _make_record(
            events=[_make_event(1), _make_event(2, success=False, error_type="ValueError")],
            failure_points=[_make_failure_point(2)],
        )
        restored = StressTestRecord.from_json(orig.to_json())
        assert restored.stress_test_id == orig.stress_test_id
        assert restored.component_name == orig.component_name
        assert restored.schema_version == orig.schema_version
        assert len(restored.events) == len(orig.events)
        assert len(restored.failure_points) == len(orig.failure_points)

    def test_summary_embedded(self):
        """Record summary is consistent with events."""
        events = [_make_event(1), _make_event(2)]
        record = _make_record(events=events)
        assert record.summary.total_calls == 2
        assert record.summary.failed_calls == 0

    def test_tags_preserved(self):
        """Custom tags survive round-trip."""
        record = _make_record()
        assert record.tags.get("test") == "true"
        restored = StressTestRecord.from_json(record.to_json())
        assert restored.tags.get("test") == "true"

    def test_related_benchmark_id_optional(self):
        """related_benchmark_id is None by default and preserved when set."""
        record = _make_record()
        assert record.related_benchmark_id is None
        d = record.to_dict()
        assert "related_benchmark_id" not in d

        record2 = StressTestRecord.create(
            component_name="X",
            component_version="1.0",
            hardware=None,
            config=StressTestConfig(max_concurrency=2, ramp_steps=1),
            events=[],
            failure_points=[],
            related_benchmark_id="abc-123",
        )
        assert record2.related_benchmark_id == "abc-123"
        assert StressTestRecord.from_json(record2.to_json()).related_benchmark_id == "abc-123"

    def test_comparable_structure(self):
        """Two records from identical configs have the same JSON key structure."""
        config = StressTestConfig(max_concurrency=4, ramp_steps=2)
        r1 = _make_record(config=config)
        r2 = _make_record(config=config)
        keys1 = set(r1.to_dict().keys())
        keys2 = set(r2.to_dict().keys())
        assert keys1 == keys2, "Key structures must match for cross-run comparison"

    def test_config_round_trip_in_record(self):
        """Config embedded in record is restored exactly."""
        config = StressTestConfig(
            start_concurrency=2,
            max_concurrency=8,
            ramp_steps=3,
            duration_per_step_s=2.5,
            payload_size="large",
        )
        record = _make_record(config=config)
        restored = StressTestRecord.from_json(record.to_json())
        assert restored.config == config
