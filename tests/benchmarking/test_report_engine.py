# -*- coding: utf-8 -*-
"""
Tests for the report data engine.

Validates ``build_report_data`` and its supporting functions against
synthetic benchmark records covering sequential, parallel, mixed,
and component topologies.

Author
------
Claude Code (Anthropic)

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2026-03-19

Modified
--------
2026-03-19
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
from grdl_te.benchmarking.report_engine import (
    BranchReportData,
    OverallSummary,
    ReportData,
    StepReportData,
    build_branch_chains,
    build_report_data,
    compute_throughput,
    step_throughput_scalar,
    step_throughput_stats,
    step_was_skipped,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_CACHED_HW = None


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
    input_shape: tuple = None,
) -> StepBenchmarkResult:
    return StepBenchmarkResult(
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
        input_shape=input_shape,
    )


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
# step_was_skipped
# ---------------------------------------------------------------------------
class TestStepWasSkipped:
    def test_active_step(self):
        s = _step(0, "Active", wall=1.0)
        assert not step_was_skipped(s)

    def test_skipped_step(self):
        s = _step(0, "Skip", wall=0.0)
        assert step_was_skipped(s)


# ---------------------------------------------------------------------------
# Throughput
# ---------------------------------------------------------------------------
class TestThroughput:
    def test_compute_throughput_basic(self):
        assert compute_throughput((512, 512), 1.0) == 512 * 512

    def test_compute_throughput_none_shape(self):
        assert compute_throughput(None, 1.0) is None

    def test_compute_throughput_zero_time(self):
        assert compute_throughput((10, 10), 0.0) is None

    def test_step_throughput_scalar_with_shape(self):
        s = _step(0, "X", wall=1.0, input_shape=(100, 100))
        assert step_throughput_scalar(s) == 10_000.0

    def test_step_throughput_scalar_no_shape(self):
        s = _step(0, "X", wall=1.0)
        assert step_throughput_scalar(s) is None

    def test_step_throughput_stats_returns_aggregated(self):
        s = _step(0, "X", wall=2.0, input_shape=(100, 100))
        stats = step_throughput_stats(s)
        assert stats is not None
        assert stats.mean == pytest.approx(5_000.0)


# ---------------------------------------------------------------------------
# build_branch_chains
# ---------------------------------------------------------------------------
class TestBuildBranchChains:
    def test_parallel_branches_detected(self):
        steps = [
            _step(0, "A", wall=1.0, step_id="a"),
            _step(1, "B", wall=2.0, step_id="b"),
            _step(2, "A2", wall=1.0, step_id="a2", depends_on=["a"]),
            _step(3, "B2", wall=2.0, step_id="b2", depends_on=["b"]),
        ]
        chains = build_branch_chains(steps)
        assert len(chains) == 2

    def test_no_deps_returns_empty(self):
        steps = [_step(0, "A", wall=1.0, step_id="a")]
        assert build_branch_chains(steps) == []

    def test_linear_chain_returns_empty(self):
        steps = [
            _step(0, "A", wall=1.0, step_id="a"),
            _step(1, "B", wall=2.0, step_id="b", depends_on=["a"]),
        ]
        assert build_branch_chains(steps) == []


# ---------------------------------------------------------------------------
# build_report_data
# ---------------------------------------------------------------------------
class TestBuildReportData:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            build_report_data([])

    def test_single_sequential_record(self):
        steps = [
            _step(0, "StepA", wall=3.0, latency_pct=30.0),
            _step(1, "StepB", wall=7.0, latency_pct=70.0),
        ]
        rec = _record(
            "SeqWorkflow", steps,
            topology=TopologyDescriptor(topology=WorkflowTopology.SEQUENTIAL),
        )
        data = build_report_data([rec])

        assert data.record_count == 1
        assert len(data.records) == 1
        assert data.overall_summary is None
        assert data.hardware is not None

        rd = data.records[0]
        assert rd.workflow_name == "SeqWorkflow"
        assert rd.topology_label == "sequential"
        assert rd.active_step_count == 2
        assert rd.skipped_step_count == 0
        assert rd.time_decomposition is None  # Sequential has no decomposition

    def test_single_component_record(self):
        steps = [_step(0, "grdl.filters.Gaussian", wall=0.5, latency_pct=100.0)]
        rec = _record(
            "Gaussian", steps,
            topology=TopologyDescriptor(topology=WorkflowTopology.COMPONENT),
            benchmark_type="component",
        )
        data = build_report_data([rec])
        rd = data.records[0]
        assert rd.topology_label == "component"
        assert rd.steps[0].short_name == "Gaussian"

    def test_parallel_workflow_branches(self):
        steps = [
            _step(0, "A", wall=1.0, concurrent=True, step_id="a",
                  latency_pct=50.0),
            _step(1, "B", wall=2.0, concurrent=True, step_id="b",
                  latency_pct=50.0),
            _step(2, "A2", wall=1.0, concurrent=True, step_id="a2",
                  depends_on=["a"], latency_pct=0.0),
            _step(3, "B2", wall=2.0, concurrent=True, step_id="b2",
                  depends_on=["b"], latency_pct=0.0),
        ]
        topo = TopologyDescriptor(
            topology=WorkflowTopology.PARALLEL,
            num_branches=2,
            critical_path_step_ids=("b", "b2"),
            critical_path_wall_time_s=4.0,
            sum_of_steps_wall_time_s=6.0,
        )
        rec = _record("ParWorkflow", steps, topology=topo, total_wall=4.0)
        data = build_report_data([rec])
        rd = data.records[0]

        assert rd.topology_label == "parallel"
        assert rd.parallel_step_count == 4
        assert len(rd.branches) == 2
        assert rd.time_decomposition is not None
        assert rd.time_decomposition.wall_clock_s == 4.0
        assert rd.time_decomposition.critical_path_s == 4.0

        # Verify path classification
        b_step = [s for s in rd.steps if s.processor_name == "B"][0]
        assert b_step.on_critical_path is True
        assert b_step.path_classification == "critical"

        a_step = [s for s in rd.steps if s.processor_name == "A"][0]
        assert a_step.path_classification == "parallel"

    def test_multi_record_overall_summary(self):
        s1 = [_step(0, "X", wall=1.0, latency_pct=100.0)]
        s2 = [_step(0, "Y", wall=3.0, latency_pct=100.0)]
        r1 = _record("Fast", s1, topology=TopologyDescriptor(
            topology=WorkflowTopology.COMPONENT))
        r2 = _record("Slow", s2, topology=TopologyDescriptor(
            topology=WorkflowTopology.COMPONENT))

        data = build_report_data([r1, r2])

        assert data.overall_summary is not None
        assert data.overall_summary.fastest_name == "Fast"
        assert data.overall_summary.slowest_name == "Slow"
        assert data.overall_summary.total_benchmarks == 2

        # Records sorted slowest-first
        assert data.records[0].workflow_name == "Slow"
        assert data.records[1].workflow_name == "Fast"

    def test_bottleneck_ranking(self):
        steps = [
            _step(0, "Low", wall=1.0, latency_pct=10.0),
            _step(1, "High", wall=9.0, latency_pct=90.0),
        ]
        rec = _record("W", steps, topology=TopologyDescriptor(
            topology=WorkflowTopology.SEQUENTIAL))
        data = build_report_data([rec])

        assert len(data.bottlenecks) == 2
        assert data.bottlenecks[0].step_name == "High"
        assert data.bottlenecks[0].rank == 1
        assert data.bottlenecks[1].step_name == "Low"

    def test_skipped_steps(self):
        steps = [
            _step(0, "Active", wall=1.0, latency_pct=100.0),
            _step(1, "Skipped", wall=0.0),
        ]
        rec = _record("W", steps, topology=TopologyDescriptor(
            topology=WorkflowTopology.SEQUENTIAL))
        data = build_report_data([rec])
        rd = data.records[0]

        assert rd.active_step_count == 1
        assert rd.skipped_step_count == 1
        skipped = [s for s in rd.steps if s.skipped]
        assert len(skipped) == 1
        assert skipped[0].path_classification == "skipped"

    def test_throughput_in_steps(self):
        steps = [
            _step(0, "X", wall=1.0, latency_pct=100.0,
                  input_shape=(256, 256)),
        ]
        rec = _record("W", steps, topology=TopologyDescriptor(
            topology=WorkflowTopology.COMPONENT))
        data = build_report_data([rec])
        s = data.records[0].steps[0]

        assert s.throughput_scalar == pytest.approx(256 * 256)
        assert s.throughput_stats is not None

    def test_comparison_auto_computed(self):
        s1 = [_step(0, "X", wall=1.0, latency_pct=100.0)]
        s2 = [_step(0, "Y", wall=2.0, latency_pct=100.0)]
        r1 = _record("A", s1, topology=TopologyDescriptor(
            topology=WorkflowTopology.COMPONENT))
        r2 = _record("B", s2, topology=TopologyDescriptor(
            topology=WorkflowTopology.COMPONENT))

        data = build_report_data([r1, r2])
        assert data.comparison is not None
        assert len(data.comparison.records) == 2
