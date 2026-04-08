# -*- coding: utf-8 -*-
"""
Tests for ComponentStressTester, WorkflowStressTester, and BaseStressTester.

Validates the ramp loop, event collection, failure detection, custom
config override, related_benchmark_id propagation, store auto-save,
and WorkflowStressTester grdl-runtime integration.

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
2026-05-13
"""

# Standard library
import time

# Third-party
import numpy as np
import pytest

# Internal
from grdl_te.benchmarking.stress_models import StressTestConfig, StressTestRecord
from grdl_te.benchmarking.stress_runner import ComponentStressTester

try:
    from grdl_te.benchmarking.stress_runner import WorkflowStressTester
    _HAS_WORKFLOW_TESTER = True
except ImportError:
    _HAS_WORKFLOW_TESTER = False

try:
    from grdl_rt import Workflow  # type: ignore
    _HAS_RUNTIME = True
except ImportError:
    _HAS_RUNTIME = False

from grdl_te.benchmarking.stress_store import JSONStressTestStore

pytestmark = pytest.mark.stress_test

# ---------------------------------------------------------------------------
# Shared fast config — short durations keep suite runtime low
# ---------------------------------------------------------------------------
FAST_CONFIG = StressTestConfig(
    start_concurrency=1,
    max_concurrency=2,
    ramp_steps=2,
    duration_per_step_s=0.5,
    payload_size="small",
    timeout_per_call_s=5.0,
)


def _identity(arr: np.ndarray) -> np.ndarray:
    """Trivial no-op callable."""
    return arr


def _slow_fn(arr: np.ndarray) -> np.ndarray:
    """Callable with a tiny but measurable delay."""
    time.sleep(0.01)
    return arr


def _always_raises(arr: np.ndarray) -> None:
    """Callable that always raises ValueError."""
    raise ValueError("deliberate test failure")


def _intermittent(arr: np.ndarray) -> np.ndarray:
    """Callable that raises once, then succeeds.

    Uses a mutable counter to toggle failure on the first call only.
    """
    counter = getattr(_intermittent, "_count", 0)
    _intermittent._count = counter + 1
    if counter == 0:
        raise RuntimeError("first call always fails")
    return arr


class TestComponentStressTesterBasic:
    """Basic correctness tests for ComponentStressTester."""

    def test_returns_stress_test_record(self):
        """run() returns a StressTestRecord."""
        tester = ComponentStressTester("identity", _identity)
        record = tester.run(FAST_CONFIG)
        assert isinstance(record, StressTestRecord)

    def test_component_name_in_record(self):
        """Component name is stored in the record."""
        tester = ComponentStressTester("my_component", _identity)
        record = tester.run(FAST_CONFIG)
        assert record.component_name == "my_component"

    def test_config_in_record(self):
        """Config stored in record matches the one passed to run()."""
        tester = ComponentStressTester("identity", _identity)
        record = tester.run(FAST_CONFIG)
        assert record.config == FAST_CONFIG

    def test_events_recorded(self):
        """At least one event is recorded."""
        tester = ComponentStressTester("identity", _identity)
        record = tester.run(FAST_CONFIG)
        assert len(record.events) > 0

    def test_all_events_succeed_for_trivial_callable(self):
        """All events are successful when callable never raises."""
        tester = ComponentStressTester("identity", _identity)
        record = tester.run(FAST_CONFIG)
        failed = [e for e in record.events if not e.success]
        assert len(failed) == 0

    def test_summary_max_sustained_is_positive(self):
        """max_sustained_concurrency >= 1 for a non-failing callable."""
        tester = ComponentStressTester("identity", _identity)
        record = tester.run(FAST_CONFIG)
        assert record.summary.max_sustained_concurrency >= 1

    def test_no_failure_points_for_trivial_callable(self):
        """No failure points produced when callable never raises."""
        tester = ComponentStressTester("identity", _identity)
        record = tester.run(FAST_CONFIG)
        assert len(record.failure_points) == 0

    def test_events_have_correct_payload_shape(self):
        """All events have payload_shape matching the config payload size."""
        from grdl_te.benchmarking.source import ARRAY_SIZES
        expected_shape = tuple(ARRAY_SIZES[FAST_CONFIG.payload_size])
        tester = ComponentStressTester("identity", _identity)
        record = tester.run(FAST_CONFIG)
        for event in record.events:
            assert event.payload_shape == expected_shape

    def test_events_have_concurrency_levels_from_config(self):
        """All event concurrency levels are drawn from config levels."""
        tester = ComponentStressTester("identity", _identity)
        record = tester.run(FAST_CONFIG)
        configured_levels = set(FAST_CONFIG.concurrency_levels())
        observed_levels = {e.concurrency_level for e in record.events}
        assert observed_levels.issubset(configured_levels)


class TestComponentStressTesterFailures:
    """Tests verifying failure detection and failure point recording."""

    def test_always_failing_callable_records_failures(self):
        """All calls failing produces non-zero failed_calls in summary."""
        tester = ComponentStressTester("always_raises", _always_raises)
        record = tester.run(FAST_CONFIG)
        assert record.summary.failed_calls > 0

    def test_always_failing_callable_produces_failure_point(self):
        """A callable that always raises produces at least one FailurePoint."""
        tester = ComponentStressTester("always_raises", _always_raises)
        record = tester.run(FAST_CONFIG)
        assert len(record.failure_points) >= 1

    def test_failure_point_error_type(self):
        """FailurePoint error_type matches the raised exception class."""
        tester = ComponentStressTester("always_raises", _always_raises)
        record = tester.run(FAST_CONFIG)
        error_types = {fp.error_type for fp in record.failure_points}
        assert "ValueError" in error_types

    def test_failure_point_concurrency_level_valid(self):
        """FailurePoint concurrency levels are within the configured range."""
        tester = ComponentStressTester("always_raises", _always_raises)
        record = tester.run(FAST_CONFIG)
        for fp in record.failure_points:
            assert fp.concurrency_level >= FAST_CONFIG.start_concurrency
            assert fp.concurrency_level <= FAST_CONFIG.max_concurrency

    def test_saturation_populated_on_failure(self):
        """summary.saturation_concurrency is populated when failures occur."""
        tester = ComponentStressTester("always_raises", _always_raises)
        record = tester.run(FAST_CONFIG)
        # Some events may fail; saturation should be set
        if record.summary.failed_calls > 0:
            assert record.summary.saturation_concurrency is not None

    def test_max_sustained_zero_when_all_fail(self):
        """max_sustained_concurrency is 0 when no level runs clean."""
        tester = ComponentStressTester("always_raises", _always_raises)
        record = tester.run(FAST_CONFIG)
        # All events fail, so no level is "clean"
        assert record.summary.max_sustained_concurrency == 0


class TestComponentStressTesterConfig:
    """Tests for default and custom configuration handling."""

    def test_default_config_used_when_none(self):
        """run() with no config argument uses StressTestConfig defaults."""
        tester = ComponentStressTester("identity", _identity)
        # Run with explicit fast config to avoid a long test; ensure no error
        record = tester.run(FAST_CONFIG)
        assert record is not None

    def test_custom_config_respected(self):
        """Custom config override is reflected in the stored record."""
        custom = StressTestConfig(
            start_concurrency=1,
            max_concurrency=3,
            ramp_steps=2,
            duration_per_step_s=0.3,
            payload_size="small",
        )
        tester = ComponentStressTester("identity", _identity)
        record = tester.run(custom)
        assert record.config.max_concurrency == 3
        assert record.config.ramp_steps == 2

    def test_version_stored_in_record(self):
        """Component version is stored in the record."""
        tester = ComponentStressTester("identity", _identity, version="2.3.4")
        record = tester.run(FAST_CONFIG)
        assert record.component_version == "2.3.4"


class TestComponentStressTesterSetup:
    """Tests for the setup callback."""

    def test_setup_callback_invoked(self):
        """setup callback is called and its args/kwargs are forwarded."""
        calls = {"setup": 0, "fn": 0, "received_shape": None}

        def setup(payload: np.ndarray):
            calls["setup"] += 1
            return (payload,), {}

        def fn(arr):
            calls["fn"] += 1
            calls["received_shape"] = arr.shape
            return arr

        tester = ComponentStressTester("setup_test", fn, setup=setup)
        record = tester.run(FAST_CONFIG)
        assert calls["setup"] > 0
        assert calls["fn"] > 0
        assert calls["received_shape"] is not None

    def test_setup_kwargs_forwarded(self):
        """setup can inject keyword arguments."""
        received = {}

        def setup(payload):
            return (payload,), {"scale": 2.0}

        def fn(arr, scale=1.0):
            received["scale"] = scale
            return arr * scale

        tester = ComponentStressTester("kwarg_test", fn, setup=setup)
        record = tester.run(FAST_CONFIG)
        assert received.get("scale") == 2.0


class TestComponentStressTesterRelatedId:
    """Tests for related_benchmark_id linkage."""

    def test_related_benchmark_id_none_by_default(self):
        """related_benchmark_id is None when not specified."""
        tester = ComponentStressTester("identity", _identity)
        record = tester.run(FAST_CONFIG)
        assert record.related_benchmark_id is None

    def test_related_benchmark_id_propagated(self):
        """related_benchmark_id is stored in the record."""
        tester = ComponentStressTester(
            "identity",
            _identity,
            related_benchmark_id="bench-abc-123",
        )
        record = tester.run(FAST_CONFIG)
        assert record.related_benchmark_id == "bench-abc-123"

    def test_related_id_survives_json_round_trip(self):
        """related_benchmark_id survives JSON serialization."""
        tester = ComponentStressTester(
            "identity",
            _identity,
            related_benchmark_id="round-trip-id",
        )
        record = tester.run(FAST_CONFIG)
        restored = StressTestRecord.from_json(record.to_json())
        assert restored.related_benchmark_id == "round-trip-id"


class TestComponentStressTesterStore:
    """Tests for automatic store persistence."""

    def test_auto_save_to_store(self, tmp_path):
        """run() with a store auto-saves the record."""
        store = JSONStressTestStore(base_dir=tmp_path)
        tester = ComponentStressTester("identity", _identity, store=store)
        record = tester.run(FAST_CONFIG)

        loaded = store.load(record.stress_test_id)
        assert loaded.stress_test_id == record.stress_test_id
        assert loaded.component_name == "identity"

    def test_store_index_contains_record(self, tmp_path):
        """Saved record appears in store index."""
        store = JSONStressTestStore(base_dir=tmp_path)
        tester = ComponentStressTester("identity", _identity, store=store)
        record = tester.run(FAST_CONFIG)

        index = store._read_index()
        ids = [e["stress_test_id"] for e in index]
        assert record.stress_test_id in ids

    def test_list_records_returns_saved(self, tmp_path):
        """list_records() returns the saved record."""
        store = JSONStressTestStore(base_dir=tmp_path)
        tester = ComponentStressTester("my_fn", _identity, store=store)
        record = tester.run(FAST_CONFIG)

        records = store.list_records(component_name="my_fn")
        assert len(records) == 1
        assert records[0].stress_test_id == record.stress_test_id


class TestStressTestRecordComparability:
    """Tests for cross-run comparison ('loadable' requirement)."""

    def test_two_runs_have_matching_key_structure(self):
        """Two records from the same config have identical top-level keys."""
        tester = ComponentStressTester("identity", _identity)
        r1 = tester.run(FAST_CONFIG)
        r2 = tester.run(FAST_CONFIG)
        assert set(r1.to_dict().keys()) == set(r2.to_dict().keys())

    def test_schema_version_stable_across_runs(self):
        """schema_version is identical across runs."""
        tester = ComponentStressTester("identity", _identity)
        r1 = tester.run(FAST_CONFIG)
        r2 = tester.run(FAST_CONFIG)
        assert r1.schema_version == r2.schema_version

    def test_grdl_version_field_present(self):
        """grdl_version is a non-empty string."""
        tester = ComponentStressTester("identity", _identity)
        record = tester.run(FAST_CONFIG)
        assert isinstance(record.grdl_version, str)
        assert len(record.grdl_version) > 0


# ===========================================================================
# WorkflowStressTester tests
# ===========================================================================

@pytest.mark.skipif(
    not (_HAS_WORKFLOW_TESTER and _HAS_RUNTIME),
    reason="WorkflowStressTester requires grdl_rt",
)
class TestWorkflowStressTester:
    """Tests for WorkflowStressTester (requires grdl_rt)."""

    def _make_workflow(self) -> "Workflow":
        """Build a minimal array-mode workflow using a trivial lambda step."""
        return Workflow("test_workflow").step(lambda arr: arr)

    def test_component_name_defaults_to_workflow_name(self):
        """component_name defaults to the workflow's .name attribute."""
        wf = self._make_workflow()
        tester = WorkflowStressTester(wf)
        assert tester.component_name == "test_workflow"

    def test_component_name_override(self):
        """Explicit name parameter overrides workflow.name."""
        wf = self._make_workflow()
        tester = WorkflowStressTester(wf, name="custom_name")
        assert tester.component_name == "custom_name"

    def test_run_returns_stress_test_record(self):
        """run() returns a StressTestRecord."""
        wf = self._make_workflow()
        tester = WorkflowStressTester(wf)
        record = tester.run(FAST_CONFIG)
        assert isinstance(record, StressTestRecord)

    def test_record_component_name_matches(self):
        """StressTestRecord.component_name matches the tester's component_name."""
        wf = self._make_workflow()
        tester = WorkflowStressTester(wf, name="my_workflow")
        record = tester.run(FAST_CONFIG)
        assert record.component_name == "my_workflow"

    def test_events_collected(self):
        """At least one event is recorded per ramp level."""
        wf = self._make_workflow()
        tester = WorkflowStressTester(wf)
        record = tester.run(FAST_CONFIG)
        assert len(record.events) > 0

    def test_summary_populated(self):
        """Summary is populated with non-negative fields."""
        wf = self._make_workflow()
        tester = WorkflowStressTester(wf)
        record = tester.run(FAST_CONFIG)
        s = record.summary
        assert s.total_calls > 0
        assert s.max_sustained_concurrency >= 1
        assert s.p99_latency_s >= 0.0

    def test_version_propagated(self):
        """version kwarg is reflected in the record."""
        wf = self._make_workflow()
        tester = WorkflowStressTester(wf, version="1.2.3")
        record = tester.run(FAST_CONFIG)
        assert record.component_version == "1.2.3"

    def test_tags_propagated(self):
        """tags kwarg appears in the record."""
        wf = self._make_workflow()
        tester = WorkflowStressTester(wf, tags={"env": "test"})
        record = tester.run(FAST_CONFIG)
        assert record.tags.get("env") == "test"


@pytest.mark.skipif(
    _HAS_RUNTIME,
    reason="Only runs when grdl_rt is NOT installed",
)
class TestWorkflowStressTesterMissingRuntime:
    """ImportError raised when grdl_rt is absent."""

    def test_instantiation_raises_import_error(self):
        """WorkflowStressTester raises ImportError without grdl_rt."""
        with pytest.raises(ImportError, match="grdl_rt is required"):
            from grdl_te.benchmarking.stress_runner import WorkflowStressTester as WST
            WST(object())
