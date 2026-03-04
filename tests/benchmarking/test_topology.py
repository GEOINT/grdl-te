# -*- coding: utf-8 -*-
"""
Tests for topology classification, critical path, and contribution analysis.

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
from grdl_te.benchmarking.topology import (
    classify_topology,
    compute_critical_path,
    compute_latency_contributions,
    compute_memory_contributions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Module-level hardware cache — captured once per module to avoid
# redundant OS/platform calls across ~23 tests.
_CACHED_HW: "HardwareSnapshot | None" = None


def _hw() -> HardwareSnapshot:
    global _CACHED_HW
    if _CACHED_HW is None:
        _CACHED_HW = HardwareSnapshot.capture()
    return _CACHED_HW


def _agg(mean: float) -> AggregatedMetrics:
    """Create AggregatedMetrics from a single value."""
    return AggregatedMetrics.from_values([mean])


def _step(
    index: int,
    name: str,
    wall: float,
    cpu: float = 0.5,
    mem: float = 1000.0,
    concurrent: bool = False,
    step_id: str = None,
    depends_on: list = None,
) -> StepBenchmarkResult:
    """Create a StepBenchmarkResult with sensible defaults."""
    return StepBenchmarkResult(
        step_index=index,
        processor_name=name,
        wall_time_s=_agg(wall),
        cpu_time_s=_agg(cpu),
        peak_rss_bytes=_agg(mem),
        gpu_used=False,
        sample_count=1,
        concurrent=concurrent,
        step_id=step_id,
        depends_on=depends_on or [],
    )


def _record(
    steps: list,
    total_wall: float = None,
    total_cpu: float = None,
    total_mem: float = None,
    benchmark_type: str = "active",
    name: str = "TestWorkflow",
) -> BenchmarkRecord:
    """Create a BenchmarkRecord with given steps."""
    if total_wall is None:
        total_wall = sum(s.wall_time_s.mean for s in steps)
    if total_cpu is None:
        total_cpu = sum(s.cpu_time_s.mean for s in steps)
    if total_mem is None:
        # Use sum for concurrent steps (overlapping footprints) vs max for
        # sequential (process high-water mark is the single largest step).
        if any(s.concurrent for s in steps):
            total_mem = sum(s.peak_rss_bytes.mean for s in steps)
        else:
            total_mem = max((s.peak_rss_bytes.mean for s in steps), default=0.0)

    return BenchmarkRecord.create(
        benchmark_type=benchmark_type,
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
# classify_topology
# ---------------------------------------------------------------------------
class TestClassifyTopology:
    """Tests for classify_topology()."""

    def test_component(self):
        """Component benchmark_type → COMPONENT."""
        steps = [_step(0, "MyFunc", wall=1.0)]
        rec = _record(steps, benchmark_type="component")
        topo = classify_topology(rec)
        assert topo.topology == WorkflowTopology.COMPONENT

    def test_sequential_no_concurrent(self):
        """All sequential steps → SEQUENTIAL."""
        steps = [
            _step(0, "Step0", wall=1.0),
            _step(1, "Step1", wall=2.0),
            _step(2, "Step2", wall=3.0),
        ]
        rec = _record(steps)
        topo = classify_topology(rec)
        assert topo.topology == WorkflowTopology.SEQUENTIAL

    def test_parallel_all_concurrent(self):
        """All concurrent with branching → PARALLEL."""
        steps = [
            _step(0, "A", wall=1.0, concurrent=True, step_id="a", depends_on=[]),
            _step(1, "B", wall=2.0, concurrent=True, step_id="b", depends_on=[]),
        ]
        rec = _record(steps, total_wall=2.0)
        topo = classify_topology(rec)
        assert topo.topology == WorkflowTopology.PARALLEL

    def test_mixed_concurrent_and_sequential(self):
        """Mix of concurrent and non-concurrent → MIXED."""
        steps = [
            _step(0, "Reader", wall=1.0, step_id="reader"),
            _step(1, "BranchA", wall=3.0, concurrent=True,
                  step_id="branch_a", depends_on=["reader"]),
            _step(2, "BranchB", wall=2.0, concurrent=True,
                  step_id="branch_b", depends_on=["reader"]),
        ]
        rec = _record(steps, total_wall=4.0)
        topo = classify_topology(rec)
        assert topo.topology == WorkflowTopology.MIXED

    def test_empty_steps(self):
        """No active steps → SEQUENTIAL (default)."""
        rec = _record([])
        topo = classify_topology(rec)
        assert topo.topology == WorkflowTopology.SEQUENTIAL

    def test_skipped_steps_ignored(self):
        """Skipped steps (all zero) are ignored in classification."""
        steps = [
            _step(0, "Active", wall=1.0),
            _step(1, "Skipped", wall=0.0, cpu=0.0),
        ]
        rec = _record(steps, total_wall=1.0)
        topo = classify_topology(rec)
        assert topo.topology == WorkflowTopology.SEQUENTIAL

    def test_parallelism_ratio(self):
        """Parallelism ratio = sum_of_steps / wall_clock."""
        steps = [
            _step(0, "A", wall=5.0, concurrent=True, step_id="a"),
            _step(1, "B", wall=5.0, concurrent=True, step_id="b"),
        ]
        rec = _record(steps, total_wall=5.0)
        topo = classify_topology(rec)
        assert topo.parallelism_ratio == pytest.approx(2.0)

    def test_num_branches(self):
        """Branch count reflects number of root nodes."""
        steps = [
            _step(0, "A", wall=1.0, concurrent=True, step_id="a"),
            _step(1, "B", wall=2.0, concurrent=True, step_id="b"),
            _step(2, "C", wall=1.5, concurrent=True, step_id="c"),
        ]
        rec = _record(steps, total_wall=2.0)
        topo = classify_topology(rec)
        assert topo.num_branches == 3


# ---------------------------------------------------------------------------
# compute_critical_path
# ---------------------------------------------------------------------------
class TestComputeCriticalPath:
    """Tests for compute_critical_path()."""

    def test_linear_chain(self):
        """Linear chain: all steps on critical path."""
        steps = [
            _step(0, "A", wall=1.0),
            _step(1, "B", wall=2.0),
            _step(2, "C", wall=3.0),
        ]
        path, total = compute_critical_path(steps)
        assert len(path) == 3
        assert total == pytest.approx(6.0)

    def test_fan_out_longest_branch(self):
        """Fan-out: critical path follows the heaviest branch."""
        steps = [
            _step(0, "Root", wall=1.0, step_id="root"),
            _step(1, "Light", wall=2.0, step_id="light", depends_on=["root"]),
            _step(2, "Heavy", wall=10.0, step_id="heavy", depends_on=["root"]),
        ]
        path, total = compute_critical_path(steps)
        assert "heavy" in path
        assert "root" in path
        assert total == pytest.approx(11.0)

    def test_diamond_merge(self):
        """Diamond DAG: Root → A, B → Merge.  Critical path via heavier branch."""
        steps = [
            _step(0, "Root", wall=1.0, step_id="root"),
            _step(1, "BranchA", wall=5.0, step_id="a", depends_on=["root"]),
            _step(2, "BranchB", wall=2.0, step_id="b", depends_on=["root"]),
            _step(3, "Merge", wall=1.0, step_id="merge", depends_on=["a", "b"]),
        ]
        path, total = compute_critical_path(steps)
        assert path == ["root", "a", "merge"]
        assert total == pytest.approx(7.0)

    def test_nested_branches(self):
        """Branch within a branch: critical path through deepest heavy chain."""
        steps = [
            _step(0, "Root", wall=1.0, step_id="root"),
            # Branch 1: root → x → y (light)
            _step(1, "X", wall=1.0, step_id="x", depends_on=["root"]),
            _step(2, "Y", wall=1.0, step_id="y", depends_on=["x"]),
            # Branch 2: root → z (heavy)
            _step(3, "Z", wall=10.0, step_id="z", depends_on=["root"]),
        ]
        path, total = compute_critical_path(steps)
        assert "z" in path
        assert total == pytest.approx(11.0)

    def test_empty_steps(self):
        """No steps → empty path."""
        path, total = compute_critical_path([])
        assert path == []
        assert total == 0.0

    def test_single_step(self):
        """Single step → that step is the entire path."""
        steps = [_step(0, "Only", wall=5.0)]
        path, total = compute_critical_path(steps)
        assert len(path) == 1
        assert total == pytest.approx(5.0)

    def test_all_roots_no_deps(self):
        """Multiple root nodes with no dependencies."""
        steps = [
            _step(0, "A", wall=3.0, step_id="a"),
            _step(1, "B", wall=5.0, step_id="b"),
            _step(2, "C", wall=1.0, step_id="c"),
        ]
        path, total = compute_critical_path(steps)
        # Heaviest single node is the critical path
        assert path == ["b"]
        assert total == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# compute_latency_contributions
# ---------------------------------------------------------------------------
class TestLatencyContributions:
    """Tests for compute_latency_contributions()."""

    def test_sequential_proportional(self):
        """Sequential: each step proportional to total wall time."""
        steps = [
            _step(0, "A", wall=3.0),
            _step(1, "B", wall=7.0),
        ]
        rec = _record(steps, total_wall=10.0)
        topo = TopologyDescriptor(topology=WorkflowTopology.SEQUENTIAL)
        pcts = compute_latency_contributions(rec, topo)

        assert pcts["__idx_0"] == pytest.approx(30.0)
        assert pcts["__idx_1"] == pytest.approx(70.0)

    def test_component_100pct(self):
        """Component: single step = 100%."""
        steps = [_step(0, "MyFunc", wall=5.0)]
        rec = _record(steps, benchmark_type="component")
        topo = TopologyDescriptor(topology=WorkflowTopology.COMPONENT)
        pcts = compute_latency_contributions(rec, topo)
        assert pcts["__idx_0"] == pytest.approx(100.0)

    def test_parallel_critical_path_only(self):
        """Parallel: only critical path steps get latency%."""
        steps = [
            _step(0, "CritA", wall=5.0, step_id="crit_a", concurrent=True),
            _step(1, "NonCrit", wall=3.0, step_id="non_crit", concurrent=True),
        ]
        rec = _record(steps, total_wall=5.0)
        topo = TopologyDescriptor(
            topology=WorkflowTopology.PARALLEL,
            critical_path_step_ids=("crit_a",),
            critical_path_wall_time_s=5.0,
        )
        pcts = compute_latency_contributions(rec, topo)
        assert pcts["crit_a"] == pytest.approx(100.0)
        assert pcts["non_crit"] == pytest.approx(0.0)

    def test_empty_steps(self):
        """No active steps → empty dict."""
        rec = _record([])
        topo = TopologyDescriptor(topology=WorkflowTopology.SEQUENTIAL)
        pcts = compute_latency_contributions(rec, topo)
        assert pcts == {}


# ---------------------------------------------------------------------------
# compute_memory_contributions
# ---------------------------------------------------------------------------
class TestMemoryContributions:
    """Tests for compute_memory_contributions()."""

    def test_all_steps_contribute(self):
        """All steps contribute to memory regardless of critical path."""
        steps = [
            _step(0, "A", wall=5.0, mem=3000.0, step_id="a", concurrent=True),
            _step(1, "B", wall=2.0, mem=7000.0, step_id="b", concurrent=True),
        ]
        rec = _record(steps, total_mem=10000.0)
        pcts = compute_memory_contributions(rec)
        assert pcts["a"] == pytest.approx(30.0)
        assert pcts["b"] == pytest.approx(70.0)

    def test_sequential_memory(self):
        """Sequential steps each get proportional memory%."""
        steps = [
            _step(0, "A", wall=1.0, mem=2000.0),
            _step(1, "B", wall=1.0, mem=8000.0),
        ]
        rec = _record(steps, total_mem=10000.0)
        pcts = compute_memory_contributions(rec)
        assert pcts["__idx_0"] == pytest.approx(20.0)
        assert pcts["__idx_1"] == pytest.approx(80.0)

    def test_empty_steps(self):
        """No active steps → empty dict."""
        rec = _record([])
        pcts = compute_memory_contributions(rec)
        assert pcts == {}

    def test_zero_peak(self):
        """Zero workflow peak → all steps get 0%."""
        steps = [_step(0, "A", wall=1.0, mem=1000.0)]
        rec = _record(steps, total_mem=0.0)
        pcts = compute_memory_contributions(rec)
        assert pcts["__idx_0"] == pytest.approx(0.0)
