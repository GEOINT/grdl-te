# -*- coding: utf-8 -*-
"""
Tests for ActiveBenchmarkRunner.

Uses mock Workflow objects that return synthetic WorkflowResults
to test iteration counting, warmup exclusion, aggregation, and
store integration without requiring real imagery.

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
2026-02-13

Modified
--------
2026-02-13
"""

# Standard library
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

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

# Internal (only import after skip check is set)
from grdl_te.benchmarking.active import ActiveBenchmarkRunner
from grdl_te.benchmarking.store import JSONBenchmarkStore


def _make_workflow_metrics(
    name: str = "MockWorkflow",
    version: str = "1.0.0",
    n_steps: int = 2,
    wall_time_base: float = 0.1,
) -> WorkflowMetrics:
    """Create synthetic WorkflowMetrics."""
    step_metrics = [
        StepMetrics(
            step_index=i,
            processor_name=f"Step{i}",
            wall_time_s=wall_time_base * (i + 1),
            cpu_time_s=wall_time_base * (i + 1) * 0.8,
            peak_rss_bytes=1000 * (i + 1),
            gpu_used=False,
        )
        for i in range(n_steps)
    ]

    total_wall = sum(s.wall_time_s for s in step_metrics)
    total_cpu = sum(s.cpu_time_s for s in step_metrics)
    peak_rss = max(s.peak_rss_bytes for s in step_metrics)

    return WorkflowMetrics(
        workflow_id=f"{name}:{version}",
        run_id=str(uuid.uuid4()),
        workflow_name=name,
        workflow_version=version,
        total_wall_time_s=total_wall,
        total_cpu_time_s=total_cpu,
        peak_rss_bytes=peak_rss,
        step_metrics=step_metrics,
        started_at=datetime.now(timezone.utc).isoformat(),
        completed_at=datetime.now(timezone.utc).isoformat(),
    )


class _MockWorkflow:
    """Mock Workflow that returns synthetic results."""

    def __init__(
        self,
        name: str = "MockWorkflow",
        version: str = "1.0.0",
        n_steps: int = 2,
    ) -> None:
        self.name = name
        self.version = version
        self._n_steps = n_steps
        self.execute_count = 0

    def execute(self, source=None, **kwargs) -> 'WorkflowResult':
        """Return a synthetic WorkflowResult."""
        self.execute_count += 1
        metrics = _make_workflow_metrics(
            name=self.name,
            version=self.version,
            n_steps=self._n_steps,
        )
        return WorkflowResult(
            result=np.zeros((8, 8)),
            metrics=metrics,
        )


class TestActiveBenchmarkRunner:
    """Tests for ActiveBenchmarkRunner."""

    def test_correct_iteration_count(self):
        """Record reports correct iteration count."""
        wf = _MockWorkflow()
        runner = ActiveBenchmarkRunner(wf, iterations=5, warmup=2)
        record = runner.run()

        assert record.iterations == 5
        assert record.benchmark_type == "active"

    def test_warmup_excluded(self):
        """Warmup runs execute but don't appear in results."""
        wf = _MockWorkflow()
        runner = ActiveBenchmarkRunner(wf, iterations=3, warmup=2)
        record = runner.run()

        # Total calls = warmup + iterations
        assert wf.execute_count == 5
        # But record only has 3 iterations worth of data
        assert record.iterations == 3
        assert record.total_wall_time.count == 3

    def test_no_warmup(self):
        """Runner works with warmup=0."""
        wf = _MockWorkflow()
        runner = ActiveBenchmarkRunner(wf, iterations=3, warmup=0)
        record = runner.run()

        assert wf.execute_count == 3
        assert record.iterations == 3

    def test_step_results_aggregated(self):
        """Per-step metrics are aggregated correctly."""
        wf = _MockWorkflow(n_steps=3)
        runner = ActiveBenchmarkRunner(wf, iterations=5, warmup=0)
        record = runner.run()

        assert len(record.step_results) == 3
        for i, step in enumerate(record.step_results):
            assert step.step_index == i
            assert step.processor_name == f"Step{i}"
            assert step.sample_count == 5

    def test_workflow_identity(self):
        """Record captures workflow name and version."""
        wf = _MockWorkflow(name="SAR Pipeline", version="2.1.0")
        runner = ActiveBenchmarkRunner(wf, iterations=1, warmup=0)
        record = runner.run()

        assert record.workflow_name == "SAR Pipeline"
        assert record.workflow_version == "2.1.0"

    def test_raw_metrics_stored(self):
        """Raw WorkflowMetrics dicts are preserved."""
        wf = _MockWorkflow()
        runner = ActiveBenchmarkRunner(wf, iterations=3, warmup=0)
        record = runner.run()

        assert len(record.raw_metrics) == 3
        for raw in record.raw_metrics:
            assert "workflow_id" in raw
            assert "step_metrics" in raw

    def test_store_integration(self, tmp_path):
        """Record is persisted when store is provided."""
        store = JSONBenchmarkStore(base_dir=tmp_path)
        wf = _MockWorkflow()
        runner = ActiveBenchmarkRunner(
            wf, iterations=2, warmup=0, store=store
        )
        record = runner.run()

        loaded = store.load(record.benchmark_id)
        assert loaded.benchmark_id == record.benchmark_id
        assert loaded.workflow_name == record.workflow_name

    def test_tags_preserved(self):
        """User-defined tags are attached to the record."""
        wf = _MockWorkflow()
        runner = ActiveBenchmarkRunner(
            wf, iterations=1, warmup=0,
            tags={"branch": "feature/x", "env": "ci"},
        )
        record = runner.run()

        assert record.tags == {"branch": "feature/x", "env": "ci"}

    def test_progress_callback(self):
        """Progress callback is called for each measurement iteration."""
        wf = _MockWorkflow()
        calls = []
        runner = ActiveBenchmarkRunner(wf, iterations=3, warmup=1)
        runner.run(progress_callback=lambda i, t: calls.append((i, t)))

        assert calls == [(1, 3), (2, 3), (3, 3)]

    def test_hardware_snapshot_captured(self):
        """Record includes a valid hardware snapshot."""
        wf = _MockWorkflow()
        runner = ActiveBenchmarkRunner(wf, iterations=1, warmup=0)
        record = runner.run()

        assert record.hardware.cpu_count >= 1
        assert record.hardware.captured_at

    def test_invalid_iterations_raises(self):
        """iterations < 1 raises ValueError."""
        wf = _MockWorkflow()
        with pytest.raises(ValueError, match="iterations"):
            ActiveBenchmarkRunner(wf, iterations=0)

    def test_invalid_warmup_raises(self):
        """warmup < 0 raises ValueError."""
        wf = _MockWorkflow()
        with pytest.raises(ValueError, match="warmup"):
            ActiveBenchmarkRunner(wf, warmup=-1)

    def test_execute_kwargs_forwarded(self):
        """Extra kwargs are passed to workflow.execute()."""
        wf = _MockWorkflow()
        # Replace execute to capture kwargs
        received_kwargs = {}
        original_execute = wf.execute

        def capturing_execute(source=None, **kwargs):
            received_kwargs.update(kwargs)
            return original_execute(source, **kwargs)

        wf.execute = capturing_execute

        runner = ActiveBenchmarkRunner(wf, iterations=1, warmup=0)
        runner.run(source="test.nitf", prefer_gpu=True)

        assert received_kwargs.get("prefer_gpu") is True
