# -*- coding: utf-8 -*-
"""
Report Data Engine — single source of truth for all report derivations.

Computes every derived metric that report presenters need (throughput,
branch chains, bottleneck rankings, path classification, time
decomposition, etc.) and packages them into a ``ReportData`` object.
Presenters call ``build_report_data()`` and focus solely on formatting.

The engine is ephemeral — it produces in-memory data structures at
report-generation time.  Nothing it computes is persisted to JSON.

Dependencies
------------
numpy

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

# Standard library
from dataclasses import dataclass, field
from math import prod
from typing import Dict, List, Optional, Tuple

# Internal
from grdl_te.benchmarking._formatting import short_name
from grdl_te.benchmarking.comparison import ComparisonResult, compare_records
from grdl_te.benchmarking.models import (
    AggregatedMetrics,
    BenchmarkRecord,
    HardwareSnapshot,
    StepBenchmarkResult,
    TopologyDescriptor,
    WorkflowTopology,
)
from grdl_te.benchmarking.topology import classify_topology


# ---------------------------------------------------------------------------
# Primitive helpers
# ---------------------------------------------------------------------------
def step_was_skipped(step: StepBenchmarkResult) -> bool:
    """Return True if a step was skipped (zero wall and CPU time)."""
    return step.wall_time_s.max == 0.0 and step.cpu_time_s.max == 0.0


def compute_throughput(
    input_shape: Optional[tuple],
    wall_time_s: Optional[float],
) -> Optional[float]:
    """Return elements/sec or ``None`` if inputs are missing/invalid."""
    if input_shape is None or wall_time_s is None or wall_time_s <= 0:
        return None
    return prod(input_shape) / wall_time_s


def step_throughput_scalar(step: StepBenchmarkResult) -> Optional[float]:
    """Scalar throughput from mean wall time.  Returns float or ``None``."""
    return compute_throughput(step.input_shape, step.wall_time_s.mean)


def step_throughput_stats(
    step: StepBenchmarkResult,
) -> Optional[AggregatedMetrics]:
    """Per-iteration throughput as ``AggregatedMetrics``.

    Returns ``None`` when input_shape is unavailable.
    """
    if step.input_shape is None:
        return None
    n_elements = prod(step.input_shape)
    values = [n_elements / wt for wt in step.wall_time_s.values if wt > 0]
    return AggregatedMetrics.from_values(values) if values else None


def build_branch_chains(
    step_results: List[StepBenchmarkResult],
) -> List[List[StepBenchmarkResult]]:
    """Group steps into branch chains using dependency information.

    A branch chain is the path from a root step (no intra-workflow
    parents) through all sequential descendants that have a single
    parent.  Steps with multiple parents (merge points) terminate a
    chain.

    Returns an empty list when branch structure cannot be determined
    (missing ``step_id`` or ``depends_on`` data).
    """
    active = [s for s in step_results if not step_was_skipped(s)]
    if not all(s.step_id for s in active):
        return []
    if not any(s.depends_on is not None for s in active):
        return []

    by_id: Dict[str, StepBenchmarkResult] = {s.step_id: s for s in active}

    # Forward edges: parent → [children]
    children: Dict[str, List[str]] = {sid: [] for sid in by_id}
    for sid, step in by_id.items():
        for dep in (step.depends_on or []):
            if dep in children:
                children[dep].append(sid)

    def n_intra_parents(sid: str) -> int:
        return sum(1 for d in (by_id[sid].depends_on or []) if d in by_id)

    roots = [s for s in active if n_intra_parents(s.step_id) == 0]
    if len(roots) < 2:
        return []

    def walk_chain(root_id: str) -> List[str]:
        chain = [root_id]
        current = root_id
        while True:
            kids = children.get(current, [])
            if len(kids) == 1 and n_intra_parents(kids[0]) == 1:
                chain.append(kids[0])
                current = kids[0]
            else:
                break
        return chain

    return [[by_id[sid] for sid in walk_chain(r.step_id)] for r in roots]


# ---------------------------------------------------------------------------
# Report data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StepReportData:
    """All derived data for a single step, ready for presentation."""

    step_index: int
    processor_name: str
    short_name: str
    step_id: Optional[str]
    skipped: bool

    # Raw aggregated metrics (pass-through)
    wall_time: AggregatedMetrics
    cpu_time: AggregatedMetrics
    peak_rss: AggregatedMetrics
    gpu_used: bool
    concurrent: bool
    input_shape: Optional[tuple]

    # Derived values
    latency_pct: float
    path_classification: str  # "critical" | "parallel" | "sequential" | "skipped"
    on_critical_path: bool

    # Throughput
    throughput_scalar: Optional[float]
    throughput_stats: Optional[AggregatedMetrics]


@dataclass(frozen=True)
class BranchReportData:
    """Derived branch chain data."""

    branch_index: int
    steps: List[StepBenchmarkResult]
    step_names: List[str]
    chain_time_s: float
    is_critical: bool
    idle_time_s: float


@dataclass(frozen=True)
class TimeDecomposition:
    """Time decomposition for parallel/mixed workflows."""

    wall_clock_s: float
    critical_path_s: float
    contended_step_sum_s: float


@dataclass(frozen=True)
class BottleneckEntry:
    """A single bottleneck ranking entry."""

    rank: int
    step_name: str
    latency_pct: float
    workflow: str
    wall_time_s: float


@dataclass(frozen=True)
class OverallSummary:
    """Overall summary for multi-record reports."""

    total_benchmarks: int
    total_wall_time_s: float
    fastest_name: str
    fastest_wall_s: float
    slowest_name: str
    slowest_wall_s: float
    least_memory_name: str
    least_memory_bytes: float
    most_memory_name: str
    most_memory_bytes: float


@dataclass(frozen=True)
class RecordReportData:
    """All derived data for a single BenchmarkRecord."""

    record: BenchmarkRecord
    workflow_name: str
    benchmark_type: str
    workflow_version: str
    topology_label: str
    iterations: int
    steps: Tuple[StepReportData, ...]
    active_step_count: int
    skipped_step_count: int
    parallel_step_count: int
    branches: Tuple[BranchReportData, ...]
    time_decomposition: Optional[TimeDecomposition]


@dataclass(frozen=True)
class ReportData:
    """Complete derived data for an entire report."""

    records: Tuple[RecordReportData, ...]
    bottlenecks: Tuple[BottleneckEntry, ...]
    comparison: Optional[ComparisonResult]
    overall_summary: Optional[OverallSummary]
    hardware: Optional[HardwareSnapshot]
    record_count: int


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------
def _classify_step_path(
    step: StepBenchmarkResult,
    cp_set: frozenset,
    skipped: bool,
) -> Tuple[str, bool]:
    """Return (path_classification, on_critical_path) for a step."""
    if skipped:
        return ("skipped", False)
    key = step.step_id or f"__idx_{step.step_index}"
    if key in cp_set:
        return ("critical", True)
    if step.concurrent:
        return ("parallel", False)
    return ("sequential", False)


def _build_step_report(
    step: StepBenchmarkResult,
    cp_set: frozenset,
) -> StepReportData:
    """Build derived data for one step."""
    skipped = step_was_skipped(step)
    path_cls, on_cp = _classify_step_path(step, cp_set, skipped)

    return StepReportData(
        step_index=step.step_index,
        processor_name=step.processor_name,
        short_name=short_name(step.processor_name),
        step_id=step.step_id,
        skipped=skipped,
        wall_time=step.wall_time_s,
        cpu_time=step.cpu_time_s,
        peak_rss=step.peak_rss_bytes,
        gpu_used=step.gpu_used,
        concurrent=step.concurrent,
        input_shape=step.input_shape,
        latency_pct=step.latency_pct if not skipped else 0.0,
        path_classification=path_cls,
        on_critical_path=on_cp,
        throughput_scalar=step_throughput_scalar(step) if not skipped else None,
        throughput_stats=step_throughput_stats(step) if not skipped else None,
    )


def _build_branches(
    record: BenchmarkRecord,
) -> Tuple[BranchReportData, ...]:
    """Build branch chain data for a record."""
    if not record.topology:
        return ()
    topo = record.topology
    if topo.topology in (WorkflowTopology.COMPONENT, WorkflowTopology.SEQUENTIAL):
        return ()

    has_parallel = any(
        s.concurrent for s in record.step_results if not step_was_skipped(s)
    )
    if not has_parallel:
        return ()

    chains = build_branch_chains(record.step_results)
    if len(chains) < 2:
        return ()

    chain_times = [sum(s.wall_time_s.mean for s in c) for c in chains]
    critical_time = max(chain_times) if chain_times else 0.0

    branches = []
    for i, (chain, ct) in enumerate(zip(chains, chain_times)):
        is_critical = ct >= critical_time - 1e-9
        branches.append(BranchReportData(
            branch_index=i + 1,
            steps=chain,
            step_names=[short_name(s.processor_name) for s in chain],
            chain_time_s=ct,
            is_critical=is_critical,
            idle_time_s=0.0 if is_critical else critical_time - ct,
        ))

    return tuple(branches)


def _build_time_decomposition(
    record: BenchmarkRecord,
) -> Optional[TimeDecomposition]:
    """Build time decomposition for parallel/mixed workflows."""
    if not record.topology:
        return None
    topo = record.topology
    if topo.topology in (WorkflowTopology.COMPONENT, WorkflowTopology.SEQUENTIAL):
        return None

    return TimeDecomposition(
        wall_clock_s=record.total_wall_time.mean,
        critical_path_s=topo.critical_path_wall_time_s,
        contended_step_sum_s=topo.sum_of_steps_wall_time_s,
    )


def _build_bottlenecks(
    records: List[BenchmarkRecord],
    comparison: Optional[ComparisonResult],
) -> Tuple[BottleneckEntry, ...]:
    """Build top-5 bottleneck ranking."""
    if comparison and comparison.bottlenecks:
        raw = comparison.bottlenecks[:5]
    else:
        raw = []
        for rec in records:
            for step in rec.step_results:
                if step_was_skipped(step):
                    continue
                raw.append({
                    "step_name": short_name(step.processor_name),
                    "latency_pct": step.latency_pct,
                    "workflow": rec.workflow_name,
                    "wall_time_s": step.wall_time_s.mean,
                })
        raw.sort(
            key=lambda e: e["latency_pct"],
            reverse=True,
        )
        raw = raw[:5]

    return tuple(
        BottleneckEntry(
            rank=i + 1,
            step_name=bn["step_name"],
            latency_pct=bn["latency_pct"],
            workflow=bn["workflow"],
            wall_time_s=bn["wall_time_s"],
        )
        for i, bn in enumerate(raw)
    )


def _build_overall_summary(
    records: List[BenchmarkRecord],
) -> Optional[OverallSummary]:
    """Build overall summary for multi-record reports."""
    if len(records) <= 1:
        return None

    total_wall = sum(r.total_wall_time.mean for r in records)
    fastest = min(records, key=lambda r: r.total_wall_time.mean)
    slowest = max(records, key=lambda r: r.total_wall_time.mean)
    least_mem = min(records, key=lambda r: r.total_peak_rss.mean)
    most_mem = max(records, key=lambda r: r.total_peak_rss.mean)

    return OverallSummary(
        total_benchmarks=len(records),
        total_wall_time_s=total_wall,
        fastest_name=fastest.workflow_name,
        fastest_wall_s=fastest.total_wall_time.mean,
        slowest_name=slowest.workflow_name,
        slowest_wall_s=slowest.total_wall_time.mean,
        least_memory_name=least_mem.workflow_name,
        least_memory_bytes=least_mem.total_peak_rss.mean,
        most_memory_name=most_mem.workflow_name,
        most_memory_bytes=most_mem.total_peak_rss.mean,
    )


def _build_record_report(record: BenchmarkRecord) -> RecordReportData:
    """Build all derived data for a single record."""
    # Ensure topology is computed
    if record.topology is None:
        record.topology = classify_topology(record)

    cp_set = frozenset(
        record.topology.critical_path_step_ids
        if record.topology else ()
    )

    steps = tuple(_build_step_report(s, cp_set) for s in record.step_results)
    active = [s for s in steps if not s.skipped]
    skipped = [s for s in steps if s.skipped]
    parallel = [s for s in active if s.concurrent]

    topo_label = (
        record.topology.topology.value if record.topology else "unknown"
    )

    return RecordReportData(
        record=record,
        workflow_name=record.workflow_name,
        benchmark_type=record.benchmark_type,
        workflow_version=record.workflow_version,
        topology_label=topo_label,
        iterations=record.iterations,
        steps=steps,
        active_step_count=len(active),
        skipped_step_count=len(skipped),
        parallel_step_count=len(parallel),
        branches=_build_branches(record),
        time_decomposition=_build_time_decomposition(record),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_report_data(
    records: List[BenchmarkRecord],
    comparison: Optional[ComparisonResult] = None,
) -> ReportData:
    """Derive all report data from benchmark records.

    This is the single entry point for all report generators.
    Computes throughput, branch chains, bottleneck rankings,
    path classifications, time decomposition, and summary
    statistics.

    Parameters
    ----------
    records : List[BenchmarkRecord]
        Benchmark records to process.  Must not be empty.
    comparison : ComparisonResult, optional
        Pre-computed comparison.  If ``None`` and ``len(records) > 1``,
        computed automatically.

    Returns
    -------
    ReportData
        Complete derived data consumed by all presenters.

    Raises
    ------
    ValueError
        If *records* is empty.
    """
    if not records:
        raise ValueError("Cannot build report data from empty records list.")

    if comparison is None and len(records) > 1:
        comparison = compare_records(records)

    # Sort records by wall time descending (slowest first)
    sorted_records = sorted(
        records,
        key=lambda r: r.total_wall_time.mean,
        reverse=True,
    )

    record_reports = tuple(
        _build_record_report(r) for r in sorted_records
    )

    return ReportData(
        records=record_reports,
        bottlenecks=_build_bottlenecks(records, comparison),
        comparison=comparison,
        overall_summary=_build_overall_summary(records),
        hardware=records[0].hardware,
        record_count=len(records),
    )
