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
Steven Siebert

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
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union

# Optional: grdl-runtime
try:
    from grdl_rt.execution.builder import Workflow
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


class ActiveBenchmarkRunner(BenchmarkRunner):
    """Run a Workflow N times and aggregate per-step metrics.

    Parameters
    ----------
    workflow : Workflow
        The grdl-runtime ``Workflow`` to benchmark.
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
    >>> from grdl_te.benchmarking import ActiveBenchmarkRunner
    >>>
    >>> wf = (
    ...     Workflow("SAR Pipeline", modalities=["SAR"])
    ...     .reader(SICDReader)
    ...     .step(SublookDecomposition, num_looks=3)
    ...     .step(ToDecibels)
    ... )
    >>> runner = ActiveBenchmarkRunner(wf, iterations=10, warmup=2)
    >>> record = runner.run(source="image.nitf", prefer_gpu=True)
    """

    def __init__(
        self,
        workflow: Any,
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
        source: Any = None,
        *,
        prefer_gpu: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **execute_kwargs: Any,
    ) -> BenchmarkRecord:
        """Execute the workflow N times and aggregate results.

        Parameters
        ----------
        source : Any, optional
            Passed to ``Workflow.execute()``.  Can be a filepath string,
            ``np.ndarray``, or ``None`` (uses configured source).
        prefer_gpu : bool
            GPU preference forwarded to ``Workflow.execute()``.
        progress_callback : callable, optional
            Called with ``(current_iteration, total_iterations)`` after
            each measurement run.  Warmup iterations are not counted.
        **execute_kwargs
            Additional keyword arguments forwarded to
            ``Workflow.execute()``.

        Returns
        -------
        BenchmarkRecord
            Aggregated benchmark results.
        """
        hardware = HardwareSnapshot.capture()
        total = self._warmup + self._iterations

        # Build execute arguments
        exec_kwargs: Dict[str, Any] = {"prefer_gpu": prefer_gpu}
        exec_kwargs.update(execute_kwargs)

        # Warmup
        for i in range(self._warmup):
            self._workflow.execute(source, **exec_kwargs)

        # Measurement runs
        all_workflow_metrics: List[Any] = []
        for i in range(self._iterations):
            result = self._workflow.execute(source, **exec_kwargs)
            all_workflow_metrics.append(result.metrics)
            if progress_callback is not None:
                progress_callback(i + 1, self._iterations)

        # Aggregate workflow-level metrics
        total_wall_times = [m.total_wall_time_s for m in all_workflow_metrics]
        total_cpu_times = [m.total_cpu_time_s for m in all_workflow_metrics]
        total_peak_rss = [float(m.peak_rss_bytes) for m in all_workflow_metrics]

        # Aggregate per-step metrics
        steps_by_index: Dict[int, List[Any]] = defaultdict(list)
        for wf_metrics in all_workflow_metrics:
            for sm in wf_metrics.step_metrics:
                steps_by_index[sm.step_index].append(sm)

        step_results = [
            StepBenchmarkResult.from_step_metrics(metrics)
            for _, metrics in sorted(steps_by_index.items())
        ]

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
            tags=self._tags,
        )

        if self._store is not None:
            self._store.save(record)

        return record
