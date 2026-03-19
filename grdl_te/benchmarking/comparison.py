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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Internal
from grdl_te.benchmarking.models import BenchmarkRecord
from grdl_te.benchmarking._formatting import short_name as _short_name
from grdl_te.benchmarking.topology import classify_topology


@dataclass
class ComparisonResult:
    """Structured comparison across two or more BenchmarkRecords.

    Attributes
    ----------
    record_labels : List[str]
        Labels for each record (workflow name or user-supplied).
    records : List[BenchmarkRecord]
        The records being compared.
    bottlenecks : List[Dict[str, Any]]
        Top bottleneck steps ranked by latency and memory.
        Each entry: ``{step_name, latency_pct,
        workflow, wall_time_s}``.
    wall_time_summary : Dict[str, float]
        ``{label: total_wall_time.mean}`` for quick reference.
    """

    record_labels: List[str] = field(default_factory=list)
    records: List[BenchmarkRecord] = field(default_factory=list)
    bottlenecks: List[Dict[str, Any]] = field(default_factory=list)
    wall_time_summary: Dict[str, float] = field(default_factory=dict)


def compare_records(
    records: List[BenchmarkRecord],
    labels: Optional[List[str]] = None,
) -> ComparisonResult:
    """Build a structured comparison across 2+ BenchmarkRecords.

    Computes wall-time summaries and identifies latency/memory
    bottlenecks across records.

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

    # Wall time summary
    wall_summary = {
        label: rec.total_wall_time.mean
        for label, rec in zip(labels, records)
    }

    # Compute bottlenecks (all steps across all records, ranked)
    bottleneck_entries: List[Dict[str, Any]] = []
    for label, rec in zip(labels, records):
        for step in rec.step_results:
            if step.wall_time_s.max == 0.0 and step.cpu_time_s.max == 0.0:
                continue
            bottleneck_entries.append({
                "step_name": _short_name(step.processor_name),
                "latency_pct": step.latency_pct,
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
        key=lambda e: e["latency_pct"],
        reverse=True,
    )

    return ComparisonResult(
        record_labels=labels,
        records=records,
        bottlenecks=bottlenecks,
        wall_time_summary=wall_summary,
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
