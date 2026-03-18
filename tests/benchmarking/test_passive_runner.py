# -*- coding: utf-8 -*-
"""
Tests for PassiveBenchmarkRunner and ForensicTraceReader.

Covers forensic benchmarking end-to-end: loading traces from JSON files,
directories, and in-memory dicts; aggregation correctness; active/passive
metric parity; hardware resolution policy; store integration; and the
grdl-runtime WorkflowMetrics schema additions (hardware, step_depends_on).

Dependencies
------------
grdl-runtime

Author
------
Ava Courtney

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2026-03-17

Modified
--------
2026-03-17
"""

# Standard library
from __future__ import annotations

import json
import uuid
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Third-party
import numpy as np
import pytest

# Conditionally import grdl-runtime types
try:
    from grdl_rt.execution.metrics import StepMetrics, WorkflowMetrics
    from grdl_rt.execution.result import WorkflowResult
    _HAS_RUNTIME = True
except ImportError:
    _HAS_RUNTIME = False

pytestmark = pytest.mark.skipif(
    not _HAS_RUNTIME, reason="grdl-runtime not installed"
)

# Internal (only import after skip guard is set)
from grdl_te.benchmarking.active import ActiveBenchmarkRunner
from grdl_te.benchmarking.forensic import ForensicTraceReader
from grdl_te.benchmarking.models import HardwareSnapshot
from grdl_te.benchmarking.passive import PassiveBenchmarkRunner
from grdl_te.benchmarking.source import BenchmarkSource
from grdl_te.benchmarking.store import JSONBenchmarkStore


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_workflow_metrics(
    name: str = "MockWorkflow",
    version: str = "1.0.0",
    n_steps: int = 2,
    wall_time_base: float = 0.1,
    step_ids: Optional[List[str]] = None,
    step_depends_on: Optional[Dict[str, List[str]]] = None,
) -> WorkflowMetrics:
    """Create synthetic WorkflowMetrics."""
    step_metrics = [
        StepMetrics(
            step_index=i,
            processor_name=f"Step{i}",
            step_id=step_ids[i] if step_ids else None,
            wall_time_s=wall_time_base * (i + 1),
            cpu_time_s=wall_time_base * (i + 1) * 0.8,
            peak_rss_bytes=1000 * (i + 1),
            gpu_used=False,
        )
        for i in range(n_steps)
    ]
    return WorkflowMetrics(
        workflow_id=f"{name}:{version}",
        run_id=str(uuid.uuid4()),
        workflow_name=name,
        workflow_version=version,
        total_wall_time_s=sum(s.wall_time_s for s in step_metrics),
        total_cpu_time_s=sum(s.cpu_time_s for s in step_metrics),
        peak_rss_bytes=max(s.peak_rss_bytes for s in step_metrics),
        step_metrics=step_metrics,
        started_at=datetime.now(timezone.utc).isoformat(),
        completed_at=datetime.now(timezone.utc).isoformat(),
        step_depends_on=step_depends_on,
    )


def _make_hw_dict(
    cpu_count: int = 4,
    total_memory_bytes: int = 8 * 1024 ** 3,
    gpu_available: bool = False,
) -> Dict[str, Any]:
    """Build a hardware dict matching LocalHardwareContext().to_dict() schema."""
    return {
        "cpu_count": cpu_count,
        "total_memory_bytes": total_memory_bytes,
        "available_memory_bytes": total_memory_bytes // 2,
        "gpu_available": gpu_available,
        "gpu_devices": [],
        "gpu_memory_bytes": 0,
    }


class _MockWorkflow:
    """Mock Workflow that returns synthetic WorkflowResults."""

    def __init__(
        self,
        name: str = "MockWorkflow",
        version: str = "1.0.0",
        n_steps: int = 2,
    ) -> None:
        self.name = name
        self.version = version
        self._n_steps = n_steps
        self.steps = []

    def execute(self, source=None, **kwargs) -> "WorkflowResult":
        """Return a synthetic WorkflowResult."""
        metrics = _make_workflow_metrics(
            name=self.name, version=self.version, n_steps=self._n_steps
        )
        return WorkflowResult(result=np.zeros((8, 8)), metrics=metrics)


_SMALL_SOURCE = BenchmarkSource.synthetic("small")


# ---------------------------------------------------------------------------
# ForensicTraceReader — from_json_file
# ---------------------------------------------------------------------------

class TestForensicTraceReaderJsonFile:
    """ForensicTraceReader.from_json_file loading and validation."""

    def test_loads_single_trace(self, tmp_path):
        """Single valid JSON file → exactly one ForensicExecutionTrace."""
        p = tmp_path / "run.json"
        p.write_text(json.dumps(_make_workflow_metrics().to_dict()), encoding="utf-8")

        traces = ForensicTraceReader.from_json_file(p)

        assert len(traces) == 1
        assert traces[0].workflow_name == "MockWorkflow"
        assert traces[0].source_type == "json_file"

    def test_missing_file_raises(self, tmp_path):
        """FileNotFoundError when path does not exist."""
        with pytest.raises(FileNotFoundError):
            ForensicTraceReader.from_json_file(tmp_path / "nonexistent.json")

    def test_invalid_json_raises(self, tmp_path):
        """JSONDecodeError for malformed JSON content."""
        p = tmp_path / "bad.json"
        p.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(Exception):
            ForensicTraceReader.from_json_file(p)

    def test_missing_required_fields_raises(self, tmp_path):
        """ValueError when required WorkflowMetrics fields are absent."""
        p = tmp_path / "partial.json"
        p.write_text(json.dumps({"total_wall_time_s": 1.0}), encoding="utf-8")
        with pytest.raises(ValueError, match="Missing fields"):
            ForensicTraceReader.from_json_file(p)


# ---------------------------------------------------------------------------
# ForensicTraceReader — from_json_directory
# ---------------------------------------------------------------------------

class TestForensicTraceReaderJsonDirectory:
    """ForensicTraceReader.from_json_directory loading, filtering, and sorting."""

    def test_loads_all_json_files(self, tmp_path):
        """All valid JSON files in a directory are loaded."""
        for i in range(5):
            m = _make_workflow_metrics(wall_time_base=0.1 * (i + 1))
            (tmp_path / f"run_{i:02d}.json").write_text(
                json.dumps(m.to_dict()), encoding="utf-8"
            )

        traces = ForensicTraceReader.from_json_directory(tmp_path)

        assert len(traces) == 5

    def test_status_filter_excludes_failures(self, tmp_path):
        """status_filter='success' skips traces with status='failed'."""
        d_fail = _make_workflow_metrics().to_dict()
        d_fail["status"] = "failed"
        (tmp_path / "failed.json").write_text(json.dumps(d_fail), encoding="utf-8")

        d_ok = _make_workflow_metrics().to_dict()
        d_ok["status"] = "success"
        (tmp_path / "success.json").write_text(json.dumps(d_ok), encoding="utf-8")

        traces = ForensicTraceReader.from_json_directory(tmp_path, status_filter="success")

        assert len(traces) == 1
        assert traces[0].status == "success"

    def test_status_filter_none_includes_all(self, tmp_path):
        """status_filter=None includes all statuses."""
        for status in ("success", "failed", "cancelled"):
            d = _make_workflow_metrics().to_dict()
            d["status"] = status
            (tmp_path / f"{status}.json").write_text(json.dumps(d), encoding="utf-8")

        traces = ForensicTraceReader.from_json_directory(tmp_path, status_filter=None)

        assert len(traces) == 3

    def test_malformed_files_skipped_with_warning(self, tmp_path):
        """Malformed JSON files are skipped; valid files are still loaded."""
        (tmp_path / "bad.json").write_text("{broken", encoding="utf-8")
        d = _make_workflow_metrics().to_dict()
        (tmp_path / "good.json").write_text(json.dumps(d), encoding="utf-8")

        traces = ForensicTraceReader.from_json_directory(tmp_path)

        assert len(traces) == 1

    def test_missing_directory_raises(self, tmp_path):
        """FileNotFoundError when directory does not exist."""
        with pytest.raises(FileNotFoundError):
            ForensicTraceReader.from_json_directory(tmp_path / "no_such_dir")


# ---------------------------------------------------------------------------
# ForensicTraceReader — from_memory
# ---------------------------------------------------------------------------

class TestForensicTraceReaderFromMemory:
    """ForensicTraceReader.from_memory wrapping and validation."""

    def test_wraps_dicts_as_traces(self):
        """Each dict is wrapped as a ForensicExecutionTrace with source_type='memory'."""
        dicts = [_make_workflow_metrics().to_dict() for _ in range(3)]

        traces = ForensicTraceReader.from_memory(dicts)

        assert len(traces) == 3
        assert all(t.source_type == "memory" for t in traces)

    def test_missing_fields_raises(self):
        """ValueError when a dict is missing required WorkflowMetrics fields."""
        with pytest.raises(ValueError, match="Missing fields"):
            ForensicTraceReader.from_memory([{"total_wall_time_s": 1.0}])


# ---------------------------------------------------------------------------
# PassiveBenchmarkRunner — smoke test
# ---------------------------------------------------------------------------

class TestPassiveBenchmarkRunnerSmoke:
    """Basic PassiveBenchmarkRunner correctness."""

    def test_single_trace_produces_valid_record(self, tmp_path):
        """One trace → BenchmarkRecord with correct type and iteration count."""
        p = tmp_path / "run.json"
        p.write_text(json.dumps(_make_workflow_metrics().to_dict()), encoding="utf-8")

        traces = ForensicTraceReader.from_json_file(p)
        record = PassiveBenchmarkRunner(traces).run()

        assert record.benchmark_type == "passive"
        assert record.iterations == 1
        assert record.total_wall_time.count == 1
        assert record.topology is not None

    def test_benchmark_type_property(self):
        """benchmark_type returns 'passive'."""
        traces = ForensicTraceReader.from_memory([_make_workflow_metrics().to_dict()])
        runner = PassiveBenchmarkRunner(traces)
        assert runner.benchmark_type == "passive"

    def test_workflow_identity_captured(self):
        """workflow_name and workflow_version come from the traces."""
        m = _make_workflow_metrics(name="SAR Pipeline", version="2.1.0")
        traces = ForensicTraceReader.from_memory([m.to_dict()])
        record = PassiveBenchmarkRunner(traces).run()

        assert record.workflow_name == "SAR Pipeline"
        assert record.workflow_version == "2.1.0"


# ---------------------------------------------------------------------------
# PassiveBenchmarkRunner — multi-trace aggregation
# ---------------------------------------------------------------------------

class TestPassiveBenchmarkRunnerAggregation:
    """Aggregation over multiple traces."""

    def test_iteration_count_matches_trace_count(self, tmp_path):
        """record.iterations equals the number of traces loaded."""
        n = 5
        for i in range(n):
            m = _make_workflow_metrics(wall_time_base=0.1 * (i + 1))
            (tmp_path / f"run_{i:02d}.json").write_text(
                json.dumps(m.to_dict()), encoding="utf-8"
            )

        traces = ForensicTraceReader.from_json_directory(tmp_path)
        record = PassiveBenchmarkRunner(traces).run()

        assert record.iterations == n
        assert record.total_wall_time.count == n

    def test_user_tags_and_provenance_tags_merged(self, tmp_path):
        """User tags and auto-added forensic_source/trace_count are all present."""
        for i in range(3):
            (tmp_path / f"run_{i}.json").write_text(
                json.dumps(_make_workflow_metrics().to_dict()), encoding="utf-8"
            )

        traces = ForensicTraceReader.from_json_directory(tmp_path)
        record = PassiveBenchmarkRunner(traces, tags={"env": "prod"}).run()

        assert record.tags["env"] == "prod"
        assert record.tags["trace_count"] == "3"
        assert "forensic_source" in record.tags

    def test_latency_contributions_populated(self, tmp_path):
        """latency_pct is non-negative for every step result."""
        for i in range(3):
            (tmp_path / f"run_{i}.json").write_text(
                json.dumps(_make_workflow_metrics().to_dict()), encoding="utf-8"
            )

        traces = ForensicTraceReader.from_json_directory(tmp_path)
        record = PassiveBenchmarkRunner(traces).run()

        assert all(sr.latency_pct >= 0 for sr in record.step_results)


# ---------------------------------------------------------------------------
# PassiveBenchmarkRunner — active/passive metric parity
# ---------------------------------------------------------------------------

class TestActivePassiveParity:
    """PassiveBenchmarkRunner fed active raw_metrics must match exactly."""

    def test_step_values_identical(self):
        """Per-step wall_time_s values are bit-for-bit identical."""
        wf = _MockWorkflow()
        active_record = ActiveBenchmarkRunner(
            wf, _SMALL_SOURCE, iterations=3, warmup=0
        ).run()

        traces = ForensicTraceReader.from_memory(active_record.raw_metrics)
        passive_record = PassiveBenchmarkRunner(traces).run()

        assert passive_record.iterations == active_record.iterations
        assert len(passive_record.step_results) == len(active_record.step_results)
        for a_step, p_step in zip(active_record.step_results, passive_record.step_results):
            assert a_step.processor_name == p_step.processor_name
            assert a_step.wall_time_s.values == p_step.wall_time_s.values
            assert a_step.wall_time_s.mean == p_step.wall_time_s.mean
            assert a_step.wall_time_s.stddev == p_step.wall_time_s.stddev

    def test_workflow_totals_identical(self):
        """Workflow-level total_wall_time is bit-for-bit identical."""
        wf = _MockWorkflow(n_steps=3)
        active_record = ActiveBenchmarkRunner(
            wf, _SMALL_SOURCE, iterations=4, warmup=0
        ).run()

        traces = ForensicTraceReader.from_memory(active_record.raw_metrics)
        passive_record = PassiveBenchmarkRunner(traces).run()

        assert active_record.total_wall_time.mean == passive_record.total_wall_time.mean
        assert active_record.total_wall_time.values == passive_record.total_wall_time.values


# ---------------------------------------------------------------------------
# PassiveBenchmarkRunner — store integration
# ---------------------------------------------------------------------------

class TestPassiveRunnerStoreIntegration:
    """BenchmarkStore round-trip for passive records."""

    def test_record_persisted_and_reloaded(self, tmp_path):
        """Record saved by store= param can be loaded back with matching fields."""
        traces = ForensicTraceReader.from_memory([_make_workflow_metrics().to_dict()])
        store = JSONBenchmarkStore(tmp_path)
        record = PassiveBenchmarkRunner(traces, store=store).run()

        loaded = store.load(record.benchmark_id)

        assert loaded.benchmark_type == "passive"
        assert loaded.benchmark_id == record.benchmark_id
        assert loaded.iterations == 1

    def test_record_appears_in_list(self, tmp_path):
        """Saved record is returned by store.list_records(benchmark_type='passive')."""
        traces = ForensicTraceReader.from_memory([_make_workflow_metrics().to_dict()])
        store = JSONBenchmarkStore(tmp_path)
        record = PassiveBenchmarkRunner(traces, store=store).run()

        records = store.list_records(benchmark_type="passive")

        assert any(r.benchmark_id == record.benchmark_id for r in records)


# ---------------------------------------------------------------------------
# PassiveBenchmarkRunner — hardware resolution policy
# ---------------------------------------------------------------------------

class TestHardwareResolution:
    """_resolve_hardware() policy: compare hardware details, not hostname."""

    def test_hardware_none_when_no_embedded_hardware(self):
        """Traces with no hardware field → record.hardware is None."""
        d = _make_workflow_metrics().to_dict()
        d.pop("hardware", None)
        traces = ForensicTraceReader.from_memory([d])

        record = PassiveBenchmarkRunner(traces).run()

        assert record.hardware is None

    def test_hardware_none_when_some_traces_missing_hardware(self):
        """Mixed present/absent hardware across traces → RuntimeWarning + None."""
        d1 = _make_workflow_metrics().to_dict()
        d1["hardware"] = _make_hw_dict()
        d2 = _make_workflow_metrics().to_dict()
        d2.pop("hardware", None)

        traces = ForensicTraceReader.from_memory([d1, d2])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            record = PassiveBenchmarkRunner(traces).run()

        assert record.hardware is None
        assert any(issubclass(x.category, RuntimeWarning) for x in w)

    def test_hardware_none_when_cpu_count_differs(self):
        """Different cpu_count across traces → RuntimeWarning + None."""
        d1 = _make_workflow_metrics().to_dict()
        d1["hardware"] = _make_hw_dict(cpu_count=4)
        d2 = _make_workflow_metrics().to_dict()
        d2["hardware"] = _make_hw_dict(cpu_count=8)

        traces = ForensicTraceReader.from_memory([d1, d2])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            record = PassiveBenchmarkRunner(traces).run()

        assert record.hardware is None
        assert any(issubclass(x.category, RuntimeWarning) for x in w)

    def test_hardware_none_when_memory_differs(self):
        """Different total_memory_bytes across traces → RuntimeWarning + None."""
        d1 = _make_workflow_metrics().to_dict()
        d1["hardware"] = _make_hw_dict(total_memory_bytes=8 * 1024 ** 3)
        d2 = _make_workflow_metrics().to_dict()
        d2["hardware"] = _make_hw_dict(total_memory_bytes=16 * 1024 ** 3)

        traces = ForensicTraceReader.from_memory([d1, d2])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            record = PassiveBenchmarkRunner(traces).run()

        assert record.hardware is None
        assert any(issubclass(x.category, RuntimeWarning) for x in w)

    def test_hardware_none_when_embedded_hardware_lacks_snapshot_fields(self):
        """Consistent LocalHardwareContext dicts pass fingerprint check but
        HardwareSnapshot.from_dict() fails on missing hostname/platform_info/etc.
        → None without warning."""
        d1 = _make_workflow_metrics().to_dict()
        d1["hardware"] = _make_hw_dict()
        d2 = _make_workflow_metrics().to_dict()
        d2["hardware"] = _make_hw_dict()

        traces = ForensicTraceReader.from_memory([d1, d2])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            record = PassiveBenchmarkRunner(traces).run()

        assert record.hardware is None
        assert not any(issubclass(x.category, RuntimeWarning) for x in w)

    def test_explicit_hardware_override(self):
        """Explicit hardware= param is used as-is regardless of trace content."""
        explicit_hw = HardwareSnapshot.capture()
        traces = ForensicTraceReader.from_memory([_make_workflow_metrics().to_dict()])

        record = PassiveBenchmarkRunner(traces, hardware=explicit_hw).run()

        assert record.hardware is explicit_hw


# ---------------------------------------------------------------------------
# PassiveBenchmarkRunner — constructor validation
# ---------------------------------------------------------------------------

class TestPassiveRunnerValidation:
    """Constructor guards for invalid trace combinations."""

    def test_empty_traces_raises(self):
        """ValueError when no traces are provided."""
        with pytest.raises(ValueError, match="at least one"):
            PassiveBenchmarkRunner([])

    def test_mixed_workflow_names_raises(self):
        """ValueError when traces span multiple workflow names."""
        traces = ForensicTraceReader.from_memory([
            _make_workflow_metrics(name="WF-A").to_dict(),
            _make_workflow_metrics(name="WF-B").to_dict(),
        ])
        with pytest.raises(ValueError, match="same workflow"):
            PassiveBenchmarkRunner(traces)

    def test_mixed_workflow_versions_raises(self):
        """ValueError when traces span multiple workflow versions."""
        traces = ForensicTraceReader.from_memory([
            _make_workflow_metrics(version="1.0.0").to_dict(),
            _make_workflow_metrics(version="2.0.0").to_dict(),
        ])
        with pytest.raises(ValueError, match="version"):
            PassiveBenchmarkRunner(traces)


# ---------------------------------------------------------------------------
# Step dependency injection
# ---------------------------------------------------------------------------

class TestDependencyInjection:
    """step_depends_on from trace is injected into StepBenchmarkResult."""

    def test_depends_on_injected(self):
        """step_depends_on in the metrics dict populates StepBenchmarkResult.depends_on."""
        step_ids = ["step-A", "step-B"]
        dep_map = {"step-B": ["step-A"]}
        m = _make_workflow_metrics(n_steps=2, step_ids=step_ids, step_depends_on=dep_map)

        traces = ForensicTraceReader.from_memory([m.to_dict()])
        record = PassiveBenchmarkRunner(traces).run()

        step_b = next(sr for sr in record.step_results if sr.step_id == "step-B")
        assert "step-A" in step_b.depends_on


# ---------------------------------------------------------------------------
# WorkflowMetrics schema additions (grdl-runtime)
# ---------------------------------------------------------------------------

class TestWorkflowMetricsSchemaAdditions:
    """hardware and step_depends_on fields added to WorkflowMetrics."""

    def test_hardware_field_round_trips(self):
        """hardware dict is serialized and present in to_dict() output."""
        hw = {"cpu_count": 4, "total_memory_bytes": 8 * 1024 ** 3}
        m = _make_workflow_metrics()
        m_with_hw = WorkflowMetrics(
            workflow_id=m.workflow_id,
            run_id=m.run_id,
            workflow_name=m.workflow_name,
            workflow_version=m.workflow_version,
            total_wall_time_s=m.total_wall_time_s,
            total_cpu_time_s=m.total_cpu_time_s,
            peak_rss_bytes=m.peak_rss_bytes,
            step_metrics=m.step_metrics,
            started_at=m.started_at,
            completed_at=m.completed_at,
            hardware=hw,
        )

        assert m_with_hw.to_dict().get("hardware") == hw

    def test_step_depends_on_round_trips(self):
        """step_depends_on is serialized when set."""
        dep = {"step-B": ["step-A"]}
        m = _make_workflow_metrics(step_depends_on=dep)

        assert m.to_dict().get("step_depends_on") == dep

    def test_hardware_none_omitted_from_dict(self):
        """hardware=None is absent (or None) in to_dict() output."""
        m = _make_workflow_metrics()

        assert m.to_dict().get("hardware") is None
