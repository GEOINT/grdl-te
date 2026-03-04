# -*- coding: utf-8 -*-
"""
Integration tests — end-to-end topology classification and reporting.

Verifies that records created without topology information are correctly
classified downstream, that the full pipeline from classification through
contribution analysis to report generation works, and that topology
survives store round-trips.

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
from grdl_te.benchmarking.topology import (
    classify_topology,
    compute_latency_contributions,
    compute_memory_contributions,
)
from grdl_te.benchmarking.comparison import compare_records
from grdl_te.benchmarking.report_md import format_report_md
from grdl_te.benchmarking.store import JSONBenchmarkStore


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
    name: str,
    steps: list,
    total_wall: float = None,
    total_mem: float = None,
    benchmark_type: str = "active",
) -> BenchmarkRecord:
    """Create a record WITHOUT topology (simulates pre-existing or external data)."""
    if total_wall is None:
        total_wall = sum(s.wall_time_s.mean for s in steps)
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
        tags={"array_size": "small"},
    )


# ---------------------------------------------------------------------------
# Auto-classification when topology is None
# ---------------------------------------------------------------------------
class TestAutoClassification:
    """Verify topology is determined correctly on records that lack it."""

    def test_compare_records_auto_classifies(self):
        """compare_records() fills in topology on records that have None."""
        rec = _record("WF", [
            _step(0, "A", wall=1.0),
            _step(1, "B", wall=2.0),
        ])
        assert rec.topology is None

        result = compare_records([rec])

        # After compare_records, topology should be populated
        assert rec.topology is not None
        assert rec.topology.topology == WorkflowTopology.SEQUENTIAL

    def test_compare_records_auto_classifies_parallel(self):
        """compare_records() correctly classifies a parallel record."""
        rec = _record("ParWF", [
            _step(0, "Root", wall=1.0, step_id="root"),
            _step(1, "BranchA", wall=3.0, concurrent=True,
                  step_id="a", depends_on=["root"]),
            _step(2, "BranchB", wall=2.0, concurrent=True,
                  step_id="b", depends_on=["root"]),
        ], total_wall=4.0)
        assert rec.topology is None

        compare_records([rec])

        assert rec.topology is not None
        assert rec.topology.topology == WorkflowTopology.MIXED

    def test_classify_deserialized_record(self):
        """A record loaded from JSON (no topology) can be classified."""
        original = _record("WF", [
            _step(0, "A", wall=1.0),
            _step(1, "B", wall=2.0),
        ])
        # Simulate old record format: serialize without topology
        assert original.topology is None
        json_str = original.to_json()
        loaded = BenchmarkRecord.from_json(json_str)

        assert loaded.topology is None

        topo = classify_topology(loaded)
        assert topo.topology == WorkflowTopology.SEQUENTIAL

    def test_classify_then_contributions_then_report(self):
        """Full pipeline: no topology → classify → contributions → report."""
        steps = [
            _step(0, "Read", wall=2.0, mem=5000.0),
            _step(1, "Process", wall=8.0, mem=15000.0),
        ]
        rec = _record("Pipeline", steps, total_wall=10.0, total_mem=20000.0)
        assert rec.topology is None

        # Step 1: Classify
        topo = classify_topology(rec)
        rec.topology = topo
        assert topo.topology == WorkflowTopology.SEQUENTIAL

        # Step 2: Compute contributions
        lat_pcts = compute_latency_contributions(rec, topo)
        mem_pcts = compute_memory_contributions(rec)

        assert lat_pcts["__idx_0"] == pytest.approx(20.0)
        assert lat_pcts["__idx_1"] == pytest.approx(80.0)
        assert mem_pcts["__idx_0"] == pytest.approx(25.0)
        assert mem_pcts["__idx_1"] == pytest.approx(75.0)

        # Wire contributions onto steps
        for sr in rec.step_results:
            key = sr.step_id or f"__idx_{sr.step_index}"
            sr.latency_pct = lat_pcts.get(key, 0.0)
            sr.memory_pct = mem_pcts.get(key, 0.0)

        # Step 3: Generate report
        md = format_report_md([rec])
        assert "# GRDL Benchmark Report" in md
        assert "Pipeline" in md
        assert "Read" in md
        assert "Process" in md
        assert "80.0%" in md
        assert "75.0%" in md


# ---------------------------------------------------------------------------
# End-to-end: parallel DAG classification and reporting
# ---------------------------------------------------------------------------
class TestParallelEndToEnd:
    """Full pipeline for parallel/mixed workflows."""

    def test_mixed_dag_classify_contribute_report(self):
        """Mixed DAG: Reader → [BranchA, BranchB] — full pipeline."""
        steps = [
            _step(0, "Reader", wall=1.0, mem=2000.0, step_id="reader"),
            _step(1, "BranchA", wall=5.0, mem=8000.0, concurrent=True,
                  step_id="branch_a", depends_on=["reader"]),
            _step(2, "BranchB", wall=3.0, mem=6000.0, concurrent=True,
                  step_id="branch_b", depends_on=["reader"]),
        ]
        rec = _record("MixedDAG", steps, total_wall=6.0, total_mem=16000.0)

        # Classify
        topo = classify_topology(rec)
        rec.topology = topo
        assert topo.topology == WorkflowTopology.MIXED
        # Reader fans out to 2 children → 2 branches
        assert topo.num_branches == 2

        # Critical path should be Reader → BranchA (heaviest)
        assert "reader" in topo.critical_path_step_ids
        assert "branch_a" in topo.critical_path_step_ids
        assert "branch_b" not in topo.critical_path_step_ids

        # Contributions
        lat_pcts = compute_latency_contributions(rec, topo)
        mem_pcts = compute_memory_contributions(rec)

        # BranchB is NOT on critical path → latency 0%
        assert lat_pcts["branch_b"] == pytest.approx(0.0)
        # But BranchB still has memory contribution
        assert mem_pcts["branch_b"] > 0

        # Wire and report
        for sr in rec.step_results:
            key = sr.step_id or f"__idx_{sr.step_index}"
            sr.latency_pct = lat_pcts.get(key, 0.0)
            sr.memory_pct = mem_pcts.get(key, 0.0)

        md = format_report_md([rec])
        assert "Time Decomposition" in md
        assert "Parallelism Ratio" in md
        # Non-critical step should have asterisk annotation
        assert "0.0%*" in md

    def test_comparison_two_workflows_no_topology(self):
        """Two records with no topology → compare → report with comparison."""
        rec_seq = _record("Sequential", [
            _step(0, "pkg.StepA", wall=3.0),
            _step(1, "pkg.StepB", wall=7.0),
        ], total_wall=10.0)

        rec_fast = _record("Optimized", [
            _step(0, "pkg.StepA", wall=1.0),
            _step(1, "pkg.StepB", wall=4.0),
        ], total_wall=5.0)

        assert rec_seq.topology is None
        assert rec_fast.topology is None

        # compare_records should auto-classify both
        comparison = compare_records([rec_seq, rec_fast])

        assert rec_seq.topology is not None
        assert rec_fast.topology is not None
        assert rec_seq.topology.topology == WorkflowTopology.SEQUENTIAL
        assert rec_fast.topology.topology == WorkflowTopology.SEQUENTIAL

        # Speedup should be 2x
        assert comparison.speedup_matrix["Sequential_vs_Optimized"] == pytest.approx(2.0)

        # Report should have comparison section
        md = format_report_md([rec_seq, rec_fast], comparison=comparison)
        assert "## Comparison" in md
        assert "Speedup Matrix" in md
        assert "Sequential" in md
        assert "Optimized" in md


# ---------------------------------------------------------------------------
# Store round-trip with topology
# ---------------------------------------------------------------------------
class TestStoreTopologyRoundTrip:
    """Verify topology persists through JSONBenchmarkStore."""

    def test_save_load_preserves_topology(self):
        """Topology survives save → load cycle."""
        steps = [
            _step(0, "A", wall=1.0),
            _step(1, "B", wall=2.0),
        ]
        rec = _record("WF", steps)
        topo = classify_topology(rec)
        rec.topology = topo

        with tempfile.TemporaryDirectory() as tmpdir:
            store = JSONBenchmarkStore(base_dir=Path(tmpdir))
            bid = store.save(rec)

            loaded = store.load(bid)
            assert loaded.topology is not None
            assert loaded.topology.topology == rec.topology.topology
            assert loaded.topology.critical_path_step_ids == rec.topology.critical_path_step_ids
            assert loaded.topology.sum_of_steps_wall_time_s == pytest.approx(
                rec.topology.sum_of_steps_wall_time_s
            )

    def test_index_contains_topology(self):
        """Store index includes topology string for filtering."""
        steps = [
            _step(0, "A", wall=1.0, concurrent=True, step_id="a"),
            _step(1, "B", wall=2.0, concurrent=True, step_id="b"),
        ]
        rec = _record("ParWF", steps, total_wall=2.0)
        topo = classify_topology(rec)
        rec.topology = topo

        with tempfile.TemporaryDirectory() as tmpdir:
            store = JSONBenchmarkStore(base_dir=Path(tmpdir))
            store.save(rec)

            index = store._read_index()
            assert len(index) == 1
            assert index[0]["topology"] == topo.topology.value

    def test_filter_by_topology(self):
        """list_records(topology=...) filters correctly."""
        seq_rec = _record("SeqWF", [
            _step(0, "A", wall=1.0),
        ])
        seq_rec.topology = TopologyDescriptor(topology=WorkflowTopology.SEQUENTIAL)

        par_rec = _record("ParWF", [
            _step(0, "A", wall=1.0, concurrent=True, step_id="a"),
            _step(1, "B", wall=2.0, concurrent=True, step_id="b"),
        ], total_wall=2.0)
        par_rec.topology = TopologyDescriptor(topology=WorkflowTopology.PARALLEL)

        with tempfile.TemporaryDirectory() as tmpdir:
            store = JSONBenchmarkStore(base_dir=Path(tmpdir))
            store.save(seq_rec)
            store.save(par_rec)

            seq_results = store.list_records(topology="sequential")
            assert len(seq_results) == 1
            assert seq_results[0].workflow_name == "SeqWF"

            par_results = store.list_records(topology="parallel")
            assert len(par_results) == 1
            assert par_results[0].workflow_name == "ParWF"

            all_results = store.list_records()
            assert len(all_results) == 2

    def test_load_record_without_topology_then_classify(self):
        """Record saved without topology can be loaded and classified after."""
        rec = _record("OldWF", [
            _step(0, "A", wall=1.0),
            _step(1, "B", wall=2.0),
        ])
        assert rec.topology is None

        with tempfile.TemporaryDirectory() as tmpdir:
            store = JSONBenchmarkStore(base_dir=Path(tmpdir))
            bid = store.save(rec)

            loaded = store.load(bid)
            assert loaded.topology is None

            # Classify post-load
            topo = classify_topology(loaded)
            loaded.topology = topo
            assert topo.topology == WorkflowTopology.SEQUENTIAL

            # Contributions work on the loaded record
            lat_pcts = compute_latency_contributions(loaded, topo)
            assert len(lat_pcts) == 2
            total_pct = sum(lat_pcts.values())
            assert total_pct == pytest.approx(100.0)

    def test_rebuild_index_includes_topology(self):
        """rebuild_index() picks up topology from record files."""
        rec = _record("WF", [_step(0, "A", wall=1.0)])
        rec.topology = TopologyDescriptor(topology=WorkflowTopology.SEQUENTIAL)

        with tempfile.TemporaryDirectory() as tmpdir:
            store = JSONBenchmarkStore(base_dir=Path(tmpdir))
            store.save(rec)

            # Corrupt the index by clearing it
            store._write_index([])
            assert store._read_index() == []

            # Rebuild
            count = store.rebuild_index()
            assert count == 1

            index = store._read_index()
            assert index[0]["topology"] == "sequential"
