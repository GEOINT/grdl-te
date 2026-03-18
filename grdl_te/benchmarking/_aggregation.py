# -*- coding: utf-8 -*-
"""
Shared Aggregation Utilities — metric aggregation shared by all benchmark runners.

Extracts the aggregation logic from ``ActiveBenchmarkRunner`` into three
functions used identically by both ``ActiveBenchmarkRunner`` and
``PassiveBenchmarkRunner``.  Keeping this logic in one place is the
primary mechanism guaranteeing metric parity between live and forensic
benchmark runs.

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
from collections import defaultdict
from typing import Any, Dict, List, Tuple

# Internal
from grdl_te.benchmarking.models import (
    AggregatedMetrics,
    BenchmarkRecord,
    StepBenchmarkResult,
)


def _step_key(sm: Any) -> str:
    """Return a stable grouping key for a StepMetrics-like object.

    Prefers ``step_id`` (set by DAG executors) for deterministic
    grouping of parallel steps.  Falls back to ``step_index`` for
    linear workflows that do not set ``step_id``.

    Parameters
    ----------
    sm : StepMetrics-like
        Any object with ``step_id`` and ``step_index`` attributes.

    Returns
    -------
    str
    """
    if getattr(sm, "step_id", None) is not None:
        return sm.step_id
    return f"__idx_{sm.step_index}"


def aggregate_workflow_totals(
    all_workflow_metrics: List[Any],
) -> Tuple[AggregatedMetrics, AggregatedMetrics, AggregatedMetrics]:
    """Aggregate workflow-level timing and memory across N runs.

    Parameters
    ----------
    all_workflow_metrics : List[Any]
        One WorkflowMetrics-like object per iteration.  Each must
        expose ``total_wall_time_s``, ``total_cpu_time_s``, and
        ``peak_rss_bytes`` attributes (duck-typed).

    Returns
    -------
    Tuple[AggregatedMetrics, AggregatedMetrics, AggregatedMetrics]
        ``(total_wall_time, total_cpu_time, total_peak_rss)``
    """
    total_wall = AggregatedMetrics.from_values(
        [m.total_wall_time_s for m in all_workflow_metrics]
    )
    total_cpu = AggregatedMetrics.from_values(
        [m.total_cpu_time_s for m in all_workflow_metrics]
    )
    total_rss = AggregatedMetrics.from_values(
        [float(m.peak_rss_bytes) for m in all_workflow_metrics]
    )
    return total_wall, total_cpu, total_rss


def aggregate_step_metrics(
    all_workflow_metrics: List[Any],
) -> List[StepBenchmarkResult]:
    """Group StepMetrics by step key across N runs and aggregate.

    Groups each step's measurements across all iterations, then calls
    ``StepBenchmarkResult.from_step_metrics()`` to produce per-step
    statistical aggregations.

    Parameters
    ----------
    all_workflow_metrics : List[Any]
        One WorkflowMetrics-like object per iteration.  Each must
        expose a ``step_metrics`` attribute (list of StepMetrics-like
        objects).

    Returns
    -------
    List[StepBenchmarkResult]
        Sorted by ``step_index`` (execution order).
    """
    steps_by_key: Dict[str, List[Any]] = defaultdict(list)
    for wf_metrics in all_workflow_metrics:
        for sm in wf_metrics.step_metrics:
            steps_by_key[_step_key(sm)].append(sm)

    step_results = [
        StepBenchmarkResult.from_step_metrics(metrics)
        for metrics in steps_by_key.values()
    ]
    step_results.sort(key=lambda s: s.step_index)
    return step_results


def apply_topology_and_contributions(record: BenchmarkRecord) -> None:
    """Classify topology and populate contribution percentages in-place.

    Calls :func:`classify_topology`, :func:`compute_latency_contributions`,
    and :func:`compute_memory_contributions` from ``topology.py``, then
    injects the per-step ``latency_pct`` and ``memory_pct`` values into
    each ``StepBenchmarkResult``.

    Mutates ``record.topology``, ``record.step_latency_pct``,
    ``record.step_memory_pct``, and each step result's ``latency_pct``
    and ``memory_pct``.

    Parameters
    ----------
    record : BenchmarkRecord
        The record to annotate in-place.
    """
    from grdl_te.benchmarking.topology import (
        classify_topology,
        compute_latency_contributions,
        compute_memory_contributions,
    )

    topo = classify_topology(record)
    record.topology = topo
    record.step_latency_pct = compute_latency_contributions(record, topo)
    record.step_memory_pct = compute_memory_contributions(record)
    for sr in record.step_results:
        key = sr.step_id or f"__idx_{sr.step_index}"
        sr.latency_pct = record.step_latency_pct.get(key, 0.0)
        sr.memory_pct = record.step_memory_pct.get(key, 0.0)
