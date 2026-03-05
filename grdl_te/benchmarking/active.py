# -*- coding: utf-8 -*-
"""
Active Benchmark Runner — run a Workflow N times and aggregate metrics.

Wraps a grdl-runtime ``Workflow`` object, executes it multiple times
(with optional warmup), collects per-step ``StepMetrics`` from each
run, and aggregates them into a ``BenchmarkRecord``.

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
2026-02-18
"""

# Standard library
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

# Optional: grdl-runtime
try:
    from grdl_rt.execution.dag_executor import DAGExecutor
    from grdl_rt.execution.gpu import GpuBackend
    from grdl_rt.execution.result import WorkflowResult
    _HAS_RUNTIME = True
except ImportError:
    _HAS_RUNTIME = False

# Internal
from grdl_te.benchmarking.base import BenchmarkRunner, BenchmarkStore
from grdl_te.benchmarking.models import (
    AggregatedMetrics,
    BenchmarkRecord,
    HardwareSnapshot,
    StepBenchmarkResult,
)
from grdl_te.benchmarking.source import BenchmarkSource


class ActiveBenchmarkRunner(BenchmarkRunner):
    """Run a Workflow N times and aggregate per-step metrics.

    Parameters
    ----------
    workflow : Workflow
        The grdl-runtime ``Workflow`` to benchmark.
    source : BenchmarkSource
        Data source for the benchmark.  Use
        ``BenchmarkSource.synthetic()``, ``.from_file()``, or
        ``.from_array()`` to create one.  Data is resolved once
        and reused across all warmup and measurement iterations.
    iterations : int
        Number of measurement iterations (excluding warmup).  Default 5.
    warmup : int
        Number of warmup iterations whose results are discarded.
        Default 1.
    store : BenchmarkStore, optional
        If provided, the resulting ``BenchmarkRecord`` is persisted
        automatically after each ``run()`` call.
    tags : Dict[str, str], optional
        User-defined labels attached to the record.

    Raises
    ------
    ImportError
        If grdl-runtime is not installed.
    ValueError
        If *iterations* < 1 or *warmup* < 0.

    Examples
    --------
    >>> from grdl_rt import Workflow
    >>> from grdl_te.benchmarking import ActiveBenchmarkRunner, BenchmarkSource
    >>>
    >>> wf = (
    ...     Workflow("SAR Pipeline", modalities=["SAR"])
    ...     .reader(SICDReader)
    ...     .step(SublookDecomposition, num_looks=3)
    ...     .step(ToDecibels)
    ... )
    >>> source = BenchmarkSource.synthetic("medium")
    >>> runner = ActiveBenchmarkRunner(wf, source, iterations=10, warmup=2)
    >>> record = runner.run(prefer_gpu=True)
    """

    def __init__(
        self,
        workflow: Any,
        source: BenchmarkSource,
        iterations: int = 5,
        warmup: int = 1,
        store: Optional[BenchmarkStore] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        if not _HAS_RUNTIME:
            raise ImportError(
                "ActiveBenchmarkRunner requires grdl-runtime. "
                "Install it with: pip install grdl-runtime"
            )
        if iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {iterations}")
        if warmup < 0:
            raise ValueError(f"warmup must be >= 0, got {warmup}")

        self._workflow = workflow
        self._source = source
        self._iterations = iterations
        self._warmup = warmup
        self._store = store
        self._tags = tags or {}

    @property
    def benchmark_type(self) -> str:
        """Return ``"active"``."""
        return "active"

    def run(
        self,
        *,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **execute_kwargs: Any,
    ) -> BenchmarkRecord:
        """Execute the workflow N times and aggregate results.

        Parameters
        ----------
        progress_callback : callable, optional
            Called with ``(current_iteration, total_iterations)`` after
            each measurement run.  Warmup iterations are not counted.
        **execute_kwargs
            Additional keyword arguments forwarded to
            ``Workflow.execute()`` (e.g. ``prefer_gpu``,
            ``auto_tap_out``, ``metadata``).

        Returns
        -------
        BenchmarkRecord
            Aggregated benchmark results.
        """
        hardware = HardwareSnapshot.capture()

        # Resolve source once — reused across all iterations
        resolved = self._source.resolve()

        # Auto-populate array dimension tags from source
        merged_tags = dict(self._tags)
        shape = self._source.shape_hint
        if shape is not None:
            merged_tags.setdefault("rows", str(shape[0]))
            merged_tags.setdefault("cols", str(shape[1]))
            merged_tags.setdefault("array_size", f"{shape[0]}x{shape[1]}")

        # Build execute arguments
        exec_kwargs: Dict[str, Any] = dict(execute_kwargs)

        # Use the workflow's own execute() to support both Workflow (builder)
        # and WorkflowDefinition (DAG) objects.
        if hasattr(self._workflow, 'execute'):
            run_fn = self._workflow.execute
        else:
            prefer_gpu = exec_kwargs.pop("prefer_gpu", False)
            executor = DAGExecutor(
                self._workflow, gpu=GpuBackend(prefer_gpu=prefer_gpu),
            )
            run_fn = executor.execute

        # Warmup
        for i in range(self._warmup):
            run_fn(resolved, **exec_kwargs)

        # Measurement runs
        all_workflow_metrics: List[Any] = []
        for i in range(self._iterations):
            result = run_fn(resolved, **exec_kwargs)
            all_workflow_metrics.append(result.metrics)
            if progress_callback is not None:
                progress_callback(i + 1, self._iterations)

        # Aggregate workflow-level metrics
        total_wall_times = [m.total_wall_time_s for m in all_workflow_metrics]
        total_cpu_times = [m.total_cpu_time_s for m in all_workflow_metrics]
        total_peak_rss = [float(m.peak_rss_bytes) for m in all_workflow_metrics]

        # Aggregate per-step metrics.
        # Prefer step_id (set by DAG executors) for deterministic grouping
        # of parallel steps.  Fall back to step_index for backward
        # compatibility with linear workflows that don't set step_id.
        def _step_key(sm: Any) -> str:
            if getattr(sm, "step_id", None) is not None:
                return sm.step_id
            return f"__idx_{sm.step_index}"

        steps_by_key: Dict[str, List[Any]] = defaultdict(list)
        for wf_metrics in all_workflow_metrics:
            for sm in wf_metrics.step_metrics:
                steps_by_key[_step_key(sm)].append(sm)

        step_results = [
            StepBenchmarkResult.from_step_metrics(metrics)
            for _, metrics in steps_by_key.items()
        ]
        # Sort by topological step_index (execution order), not
        # alphabetical step_id, so reports display steps in the
        # order they actually run.
        step_results.sort(key=lambda s: s.step_index)

        # Inject step dependency graph from the workflow so the report can
        # reconstruct branch chains (which sequential steps follow which
        # concurrent roots).
        if hasattr(self._workflow, 'steps'):
            known_ids = {
                getattr(s, 'id', None)
                for s in self._workflow.steps
                if getattr(s, 'id', None)
            }
            dep_map: Dict[str, List[Any]] = {}
            for s in self._workflow.steps:
                sid = getattr(s, 'id', None)
                if sid:
                    raw_deps = getattr(s, 'depends_on', None) or []
                    dep_map[sid] = [d for d in raw_deps if d in known_ids]
            for sr in step_results:
                if sr.step_id and sr.step_id in dep_map:
                    sr.depends_on = dep_map[sr.step_id]

        # Raw metrics for lossless storage
        raw_metrics = [m.to_dict() for m in all_workflow_metrics]

        # Workflow identity
        wf_name = getattr(self._workflow, "name", "unknown")
        wf_version = getattr(self._workflow, "version", "0.0.0")

        record = BenchmarkRecord.create(
            benchmark_type=self.benchmark_type,
            workflow_name=wf_name,
            workflow_version=wf_version,
            iterations=self._iterations,
            hardware=hardware,
            total_wall_time=AggregatedMetrics.from_values(total_wall_times),
            total_cpu_time=AggregatedMetrics.from_values(total_cpu_times),
            total_peak_rss=AggregatedMetrics.from_values(total_peak_rss),
            step_results=step_results,
            raw_metrics=raw_metrics,
            tags=merged_tags,
        )

        # Topology classification and contribution analysis
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

        if self._store is not None:
            self._store.save(record)

        return record
