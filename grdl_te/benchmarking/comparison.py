# -*- coding: utf-8 -*-
"""
Comparison Engine — structured cross-workflow benchmark comparison.

Matches steps across multiple ``BenchmarkRecord`` instances by processor
name, computes wall-time deltas, and identifies latency and memory
bottlenecks.  Produces a ``ComparisonResult`` consumed by the Markdown
report generator.

Dependencies
------------
(none beyond grdl-te)

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
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Internal
from grdl_te.benchmarking.models import (
    BenchmarkRecord,
    StepBenchmarkResult,
    TopologyDescriptor,
)
from grdl_te.benchmarking.topology import classify_topology


def _short_name(processor_name: str) -> str:
    """Extract short processor name for cross-record matching."""
    return processor_name.rsplit(".", 1)[-1]


@dataclass
class StepComparison:
    """Side-by-side comparison of a single step across benchmark records.

    Attributes
    ----------
    step_name : str
        Short processor name used for matching.
    records : Dict[str, StepBenchmarkResult]
        ``{record_label: step_result}`` for each record containing
        this step.
    wall_time_deltas : Dict[str, float]
        Pairwise wall-time delta percentages.
        ``{"A_vs_B": (B.mean - A.mean) / A.mean * 100}``.
    latency_pcts : Dict[str, float]
        ``{label: latency_pct}`` for each record.
    memory_pcts : Dict[str, float]
        ``{label: memory_pct}`` for each record.
    """

    step_name: str
    records: Dict[str, StepBenchmarkResult] = field(default_factory=dict)
    wall_time_deltas: Dict[str, float] = field(default_factory=dict)
    latency_pcts: Dict[str, float] = field(default_factory=dict)
    memory_pcts: Dict[str, float] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Structured comparison across two or more BenchmarkRecords.

    Attributes
    ----------
    record_labels : List[str]
        Labels for each record (workflow name or user-supplied).
    records : List[BenchmarkRecord]
        The records being compared.
    step_comparisons : List[StepComparison]
        Per-step cross-record comparisons.
    bottlenecks : List[Dict[str, Any]]
        Top bottleneck steps ranked by latency and memory.
        Each entry: ``{step_name, latency_pct, memory_pct,
        workflow, wall_time_s}``.
    wall_time_summary : Dict[str, float]
        ``{label: total_wall_time.mean}`` for quick reference.
    speedup_matrix : Dict[str, float]
        Pairwise speedup ratios.
        ``{"A_vs_B": A.wall / B.wall}`` (>1 means B is faster).
    """

    record_labels: List[str] = field(default_factory=list)
    records: List[BenchmarkRecord] = field(default_factory=list)
    step_comparisons: List[StepComparison] = field(default_factory=list)
    bottlenecks: List[Dict[str, Any]] = field(default_factory=list)
    wall_time_summary: Dict[str, float] = field(default_factory=dict)
    speedup_matrix: Dict[str, float] = field(default_factory=dict)


def compare_records(
    records: List[BenchmarkRecord],
    labels: Optional[List[str]] = None,
) -> ComparisonResult:
    """Build a structured comparison across 2+ BenchmarkRecords.

    Steps are matched across records by short processor name
    (e.g., ``SublookDecomposition``).  Pairwise wall-time deltas,
    latency/memory contributions, and a speedup matrix are computed.

    Parameters
    ----------
    records : List[BenchmarkRecord]
        Records to compare.  Must contain at least one record.
    labels : List[str], optional
        Human-readable labels for each record.  If ``None``,
        workflow names are used (with numeric suffix for duplicates).

    Returns
    -------
    ComparisonResult
    """
    if not records:
        return ComparisonResult()

    # Ensure all records have topology
    for rec in records:
        if rec.topology is None:
            topo = classify_topology(rec)
            rec.topology = topo

    # Generate labels
    if labels is None:
        labels = _generate_labels(records)

    label_map = {label: rec for label, rec in zip(labels, records)}

    # Wall time summary
    wall_summary = {
        label: rec.total_wall_time.mean
        for label, rec in zip(labels, records)
    }

    # Match steps by short processor name
    step_map: Dict[str, Dict[str, StepBenchmarkResult]] = defaultdict(dict)
    for label, rec in zip(labels, records):
        for step in rec.step_results:
            if step.wall_time_s.max == 0.0 and step.cpu_time_s.max == 0.0:
                continue  # skip skipped steps
            short = _short_name(step.processor_name)
            step_map[short][label] = step

    # Build step comparisons
    step_comparisons: List[StepComparison] = []
    for step_name, step_records in sorted(step_map.items()):
        sc = StepComparison(step_name=step_name, records=step_records)

        for label, step in step_records.items():
            sc.latency_pcts[label] = step.latency_pct
            sc.memory_pcts[label] = step.memory_pct

        # Pairwise deltas
        label_list = sorted(step_records.keys())
        for i, la in enumerate(label_list):
            for lb in label_list[i + 1:]:
                a_wall = step_records[la].wall_time_s.mean
                b_wall = step_records[lb].wall_time_s.mean
                if a_wall > 0:
                    delta = ((b_wall - a_wall) / a_wall) * 100.0
                    sc.wall_time_deltas[f"{la}_vs_{lb}"] = delta

        step_comparisons.append(sc)

    # Compute bottlenecks (all steps across all records, ranked)
    bottleneck_entries: List[Dict[str, Any]] = []
    for label, rec in zip(labels, records):
        for step in rec.step_results:
            if step.wall_time_s.max == 0.0 and step.cpu_time_s.max == 0.0:
                continue
            bottleneck_entries.append({
                "step_name": _short_name(step.processor_name),
                "latency_pct": step.latency_pct,
                "memory_pct": step.memory_pct,
                "workflow": label,
                "wall_time_s": step.wall_time_s.mean,
            })

    # Deduplicate by step name — keep the entry with highest latency_pct
    seen: Dict[str, Dict[str, Any]] = {}
    for entry in bottleneck_entries:
        key = f"{entry['step_name']}_{entry['workflow']}"
        if key not in seen or entry["latency_pct"] > seen[key]["latency_pct"]:
            seen[key] = entry
    bottlenecks = sorted(
        seen.values(),
        key=lambda e: (e["latency_pct"], e["memory_pct"]),
        reverse=True,
    )

    # Speedup matrix
    speedup = {}
    for i, la in enumerate(labels):
        for lb in labels[i + 1:]:
            a_wall = wall_summary[la]
            b_wall = wall_summary[lb]
            if b_wall > 0:
                speedup[f"{la}_vs_{lb}"] = a_wall / b_wall

    return ComparisonResult(
        record_labels=labels,
        records=records,
        step_comparisons=step_comparisons,
        bottlenecks=bottlenecks,
        wall_time_summary=wall_summary,
        speedup_matrix=speedup,
    )


def _generate_labels(records: List[BenchmarkRecord]) -> List[str]:
    """Generate unique labels from workflow names."""
    counts: Dict[str, int] = {}
    labels: List[str] = []
    for rec in records:
        name = rec.workflow_name
        count = counts.get(name, 0)
        counts[name] = count + 1
        if count > 0:
            labels.append(f"{name} ({count + 1})")
        else:
            labels.append(name)

    # If any name was duplicated, retroactively fix the first occurrence
    for i, rec in enumerate(records):
        name = rec.workflow_name
        if counts[name] > 1 and not labels[i].endswith(")"):
            labels[i] = f"{name} (1)"

    return labels
