# -*- coding: utf-8 -*-
"""
Topology Classification — classify workflow structure and compute contributions.

Analyzes ``BenchmarkRecord`` step results to classify the workflow
execution topology (sequential, parallel, mixed, or component),
compute the critical path through the DAG, and calculate per-step
latency and memory contribution percentages.

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
from typing import Any, Dict, List, Optional, Set, Tuple

# Internal
from grdl_te.benchmarking.models import (
    BenchmarkRecord,
    StepBenchmarkResult,
    TopologyDescriptor,
    WorkflowTopology,
)


def _step_was_skipped(step: StepBenchmarkResult) -> bool:
    """Return True if a step was skipped (all wall and CPU times zero)."""
    return step.wall_time_s.max == 0.0 and step.cpu_time_s.max == 0.0


def _step_key(step: StepBenchmarkResult) -> str:
    """Return a stable key for a step (step_id or fallback index)."""
    return step.step_id if step.step_id is not None else f"__idx_{step.step_index}"


def compute_critical_path(
    steps: List[StepBenchmarkResult],
) -> Tuple[List[str], float]:
    """Compute the critical path through a DAG of step results.

    The critical path is the dependency chain with the **longest
    accumulated wall time**, not the most steps.  A single 10-second
    step beats three 2-second steps.

    Uses a topological-sort-based longest-path algorithm:

    1. Build adjacency from ``depends_on``.
    2. For each node in topological order,
       ``dist[node] = max(dist[parent] + parent.wall_time.mean)``.
    3. The terminal node with the highest ``dist`` is the critical
       path endpoint.
    4. Trace back through predecessors to reconstruct the path.

    Parameters
    ----------
    steps : List[StepBenchmarkResult]
        Non-skipped steps with ``step_id`` and ``depends_on`` set.

    Returns
    -------
    Tuple[List[str], float]
        ``(step_ids_on_critical_path, critical_path_wall_time)``.
        Returns ``([], 0.0)`` if no dependency information is
        available or no steps are provided.
    """
    active = [s for s in steps if not _step_was_skipped(s)]
    if not active:
        return [], 0.0

    # Require step_id for DAG analysis
    if not all(s.step_id for s in active):
        # Linear workflow — the whole chain is the critical path
        ids = [_step_key(s) for s in active]
        total = sum(s.wall_time_s.mean for s in active)
        return ids, total

    by_id: Dict[str, StepBenchmarkResult] = {s.step_id: s for s in active}  # type: ignore[misc]

    # dist[node] = longest path ending at node (inclusive of node time)
    dist: Dict[str, float] = {}
    pred: Dict[str, Optional[str]] = {}

    # Topological order: process nodes whose parents are all processed
    processed: Set[str] = set()
    order: List[str] = []

    def _topo_sort() -> List[str]:
        in_degree: Dict[str, int] = {sid: 0 for sid in by_id}
        for sid, step in by_id.items():
            for dep in (step.depends_on or []):
                if dep in by_id:
                    in_degree[sid] = in_degree.get(sid, 0) + 1

        queue = [sid for sid, deg in in_degree.items() if deg == 0]
        result: List[str] = []
        while queue:
            queue.sort()  # deterministic ordering
            node = queue.pop(0)
            result.append(node)
            for sid, step in by_id.items():
                if node in (step.depends_on or []):
                    in_degree[sid] -= 1
                    if in_degree[sid] == 0:
                        queue.append(sid)
        return result

    order = _topo_sort()

    for sid in order:
        step = by_id[sid]
        deps = [d for d in (step.depends_on or []) if d in by_id]

        if not deps:
            # Root node
            dist[sid] = step.wall_time_s.mean
            pred[sid] = None
        else:
            best_parent = max(deps, key=lambda d: dist.get(d, 0.0))
            dist[sid] = dist.get(best_parent, 0.0) + step.wall_time_s.mean
            pred[sid] = best_parent

    if not dist:
        return [], 0.0

    # Find terminal with maximum distance
    terminal = max(dist, key=lambda sid: dist[sid])
    critical_time = dist[terminal]

    # Trace back
    path: List[str] = []
    current: Optional[str] = terminal
    while current is not None:
        path.append(current)
        current = pred.get(current)
    path.reverse()

    return path, critical_time


def _count_branches(steps: List[StepBenchmarkResult]) -> int:
    """Count the number of parallel branches in a DAG.

    A branch is a parallel execution path.  Counted by starting with
    the number of root nodes and adding ``children - 1`` at each
    fan-out point (a node with more than one child).

    Examples:

    - ``A → B → C`` → 0 (sequential, no branching)
    - ``Root → [A, B]`` → 2
    - ``[A, B]`` (two roots) → 2
    - ``Root → [A, B]``, ``A → [C, D]`` → 3
    """
    active = [s for s in steps if not _step_was_skipped(s)]
    if not all(s.step_id for s in active):
        return 0

    by_id: Dict[str, StepBenchmarkResult] = {
        s.step_id: s for s in active  # type: ignore[misc]
    }

    # Count children per node
    child_count: Dict[str, int] = {sid: 0 for sid in by_id}
    for sid, step in by_id.items():
        for dep in (step.depends_on or []):
            if dep in child_count:
                child_count[dep] += 1

    # Start with number of roots (nodes with no intra-DAG parents)
    roots = 0
    for sid, step in by_id.items():
        deps = [d for d in (step.depends_on or []) if d in by_id]
        if not deps:
            roots += 1

    # Each fan-out adds (children - 1) additional paths
    branches = roots
    for sid, count in child_count.items():
        if count > 1:
            branches += count - 1

    return branches if branches > 1 else 0


def classify_topology(record: BenchmarkRecord) -> TopologyDescriptor:
    """Classify a BenchmarkRecord's execution topology.

    Classification rules:

    1. ``benchmark_type == "component"`` → ``COMPONENT``
    2. No step has ``concurrent=True`` and no branching in
       ``depends_on`` → ``SEQUENTIAL``
    3. All non-skipped steps have ``concurrent=True`` → ``PARALLEL``
    4. Mix of concurrent and non-concurrent → ``MIXED``

    Also computes critical path, sum-of-steps, parallelism ratio,
    and branch count.

    Parameters
    ----------
    record : BenchmarkRecord

    Returns
    -------
    TopologyDescriptor
    """
    if record.benchmark_type == "component":
        total = sum(
            s.wall_time_s.mean
            for s in record.step_results
            if not _step_was_skipped(s)
        )
        return TopologyDescriptor(
            topology=WorkflowTopology.COMPONENT,
            sum_of_steps_wall_time_s=total,
        )

    active = [s for s in record.step_results if not _step_was_skipped(s)]
    if not active:
        return TopologyDescriptor(topology=WorkflowTopology.SEQUENTIAL)

    has_concurrent = any(s.concurrent for s in active)
    all_concurrent = all(s.concurrent for s in active)

    # Check for branching in depends_on
    has_branching = False
    if any(s.step_id for s in active):
        by_id = {s.step_id for s in active if s.step_id}
        for s in active:
            deps = [d for d in (s.depends_on or []) if d in by_id]
            if not deps and s.step_id:
                # Potential root — count how many roots exist
                pass
        has_branching = _count_branches(active) > 1

    if not has_concurrent and not has_branching:
        topo = WorkflowTopology.SEQUENTIAL
    elif all_concurrent:
        topo = WorkflowTopology.PARALLEL
    elif has_concurrent or has_branching:
        topo = WorkflowTopology.MIXED
    else:
        topo = WorkflowTopology.SEQUENTIAL

    # Compute critical path and metrics
    cp_ids, cp_time = compute_critical_path(active)
    sum_of_steps = sum(s.wall_time_s.mean for s in active)
    wall_clock = record.total_wall_time.mean
    ratio = sum_of_steps / wall_clock if wall_clock > 0 else 1.0
    num_branches = _count_branches(active)

    return TopologyDescriptor(
        topology=topo,
        num_branches=num_branches,
        critical_path_step_ids=tuple(cp_ids),
        critical_path_wall_time_s=cp_time,
        sum_of_steps_wall_time_s=sum_of_steps,
        parallelism_ratio=ratio,
    )


def compute_latency_contributions(
    record: BenchmarkRecord,
    topology: TopologyDescriptor,
) -> Dict[str, float]:
    """Compute each step's percentage contribution to total wall-clock time.

    For sequential workflows, every step contributes proportionally.
    For parallel workflows, only critical-path steps contribute;
    non-critical steps get ``0.0`` (their wall time is hidden by
    the critical path).

    Parameters
    ----------
    record : BenchmarkRecord
    topology : TopologyDescriptor

    Returns
    -------
    Dict[str, float]
        ``{step_key: percentage}`` where percentage is in ``[0, 100]``.
    """
    active = [s for s in record.step_results if not _step_was_skipped(s)]
    if not active:
        return {}

    result: Dict[str, float] = {}

    if topology.topology == WorkflowTopology.COMPONENT:
        for s in active:
            result[_step_key(s)] = 100.0
        return result

    if topology.topology == WorkflowTopology.SEQUENTIAL:
        total_wall = record.total_wall_time.mean
        if total_wall <= 0:
            return {_step_key(s): 0.0 for s in active}
        for s in active:
            result[_step_key(s)] = (s.wall_time_s.mean / total_wall) * 100.0
        return result

    # PARALLEL or MIXED — use critical path
    cp_set = set(topology.critical_path_step_ids)
    cp_wall = topology.critical_path_wall_time_s

    if cp_wall <= 0:
        return {_step_key(s): 0.0 for s in active}

    for s in active:
        key = _step_key(s)
        if key in cp_set:
            result[key] = (s.wall_time_s.mean / cp_wall) * 100.0
        else:
            result[key] = 0.0

    return result


def compute_memory_contributions(
    record: BenchmarkRecord,
) -> Dict[str, float]:
    """Compute each step's percentage contribution to peak memory.

    All steps contribute regardless of whether they are on the
    critical path — memory is process-wide.  For concurrent steps,
    the ``peak_rss_bytes`` is the shared level-wide peak and is
    attributed to each step in that level with a ``(shared)``
    annotation in reports.

    Parameters
    ----------
    record : BenchmarkRecord

    Returns
    -------
    Dict[str, float]
        ``{step_key: percentage}`` where percentage is in ``[0, 100]``.
    """
    active = [s for s in record.step_results if not _step_was_skipped(s)]
    if not active:
        return {}

    workflow_peak = record.total_peak_rss.mean
    if workflow_peak <= 0:
        return {_step_key(s): 0.0 for s in active}

    result: Dict[str, float] = {}
    for s in active:
        pct = (s.peak_rss_bytes.mean / workflow_peak) * 100.0
        result[_step_key(s)] = pct

    return result
