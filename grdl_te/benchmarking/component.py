# -*- coding: utf-8 -*-
"""
Component Benchmark — profile individual functions outside workflow context.

Benchmarks a single callable (function, method, or processor) with the
same measurement stack grdl-runtime uses (``time.perf_counter``,
``time.process_time``, ``tracemalloc``), producing a ``BenchmarkRecord``
compatible with the rest of the benchmarking infrastructure.

Does NOT require grdl-runtime — works with pure Python callables.

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
import time
import tracemalloc
from typing import Any, Callable, Dict, List, Optional, Tuple

# Internal
from grdl_te.benchmarking.base import BenchmarkRunner, BenchmarkStore
from grdl_te.benchmarking.models import (
    AggregatedMetrics,
    BenchmarkRecord,
    HardwareSnapshot,
    StepBenchmarkResult,
)


class ComponentBenchmark(BenchmarkRunner):
    """Benchmark a single callable with timing and memory instrumentation.

    Wraps any callable with the same measurement approach that
    grdl-runtime uses in ``Workflow._run_pipeline()``, so that
    component-level and workflow-level benchmarks are directly
    comparable.

    Parameters
    ----------
    name : str
        Human-readable name for the component (e.g.,
        ``"Normalizer.minmax.4k"``).
    fn : callable
        The function or method to benchmark.
    setup : callable, optional
        Called before each iteration.  Must return a tuple
        ``(args, kwargs)`` to pass to *fn*.  If ``None``, *fn* is
        called with no arguments.
    teardown : callable, optional
        Called after each iteration for cleanup.
    iterations : int
        Number of measurement iterations.  Default 10.
    warmup : int
        Number of warmup iterations (discarded).  Default 2.
    store : BenchmarkStore, optional
        If provided, result is persisted automatically.
    tags : Dict[str, str], optional
        User-defined labels.
    version : str
        Version string for the component.  Default ``"0.0.0"``.

    Raises
    ------
    ValueError
        If *iterations* < 1 or *warmup* < 0.

    Examples
    --------
    >>> import numpy as np
    >>> from grdl.data_prep import Normalizer
    >>> from grdl_te.benchmarking import ComponentBenchmark
    >>>
    >>> image = np.random.rand(4096, 4096).astype(np.float32)
    >>> norm = Normalizer(method='minmax')
    >>> bench = ComponentBenchmark(
    ...     "Normalizer.minmax.4k",
    ...     norm.normalize,
    ...     setup=lambda: ((image,), {}),
    ...     iterations=20,
    ...     warmup=3,
    ... )
    >>> record = bench.run()
    """

    def __init__(
        self,
        name: str,
        fn: Callable[..., Any],
        setup: Optional[Callable[[], Tuple[tuple, dict]]] = None,
        teardown: Optional[Callable[[], None]] = None,
        iterations: int = 10,
        warmup: int = 2,
        store: Optional[BenchmarkStore] = None,
        tags: Optional[Dict[str, str]] = None,
        version: str = "0.0.0",
    ) -> None:
        if iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {iterations}")
        if warmup < 0:
            raise ValueError(f"warmup must be >= 0, got {warmup}")

        self._name = name
        self._fn = fn
        self._setup = setup
        self._teardown = teardown
        self._iterations = iterations
        self._warmup = warmup
        self._store = store
        self._tags = tags or {}
        self._version = version

    @property
    def benchmark_type(self) -> str:
        """Return ``"component"``."""
        return "component"

    def run(self, **kwargs: Any) -> BenchmarkRecord:
        """Execute the callable N times and aggregate timing/memory.

        Returns
        -------
        BenchmarkRecord
            With ``benchmark_type="component"`` and a single
            ``StepBenchmarkResult`` entry.
        """
        hardware = HardwareSnapshot.capture()

        # Ensure tracemalloc is active
        was_tracing = tracemalloc.is_tracing()
        if not was_tracing:
            tracemalloc.start()

        try:
            # Warmup
            for _ in range(self._warmup):
                args, kw = self._get_args()
                self._fn(*args, **kw)
                if self._teardown is not None:
                    self._teardown()

            # Measurement
            wall_times: List[float] = []
            cpu_times: List[float] = []
            peak_rss_list: List[float] = []

            for _ in range(self._iterations):
                args, kw = self._get_args()

                snap_before = tracemalloc.take_snapshot()
                wall_t0 = time.perf_counter()
                cpu_t0 = time.process_time()

                self._fn(*args, **kw)

                wall_elapsed = time.perf_counter() - wall_t0
                cpu_elapsed = time.process_time() - cpu_t0
                snap_after = tracemalloc.take_snapshot()

                # Memory delta (sum of positive diffs)
                stats = snap_after.compare_to(snap_before, "lineno")
                mem_delta = sum(s.size_diff for s in stats if s.size_diff > 0)

                wall_times.append(wall_elapsed)
                cpu_times.append(cpu_elapsed)
                peak_rss_list.append(float(mem_delta))

                if self._teardown is not None:
                    self._teardown()

        finally:
            if not was_tracing:
                tracemalloc.stop()

        # Build a single StepBenchmarkResult manually
        step_result = StepBenchmarkResult(
            step_index=0,
            processor_name=self._name,
            wall_time_s=AggregatedMetrics.from_values(wall_times),
            cpu_time_s=AggregatedMetrics.from_values(cpu_times),
            peak_rss_bytes=AggregatedMetrics.from_values(peak_rss_list),
            gpu_used=False,
            gpu_memory_bytes=None,
            sample_count=self._iterations,
        )

        record = BenchmarkRecord.create(
            benchmark_type=self.benchmark_type,
            workflow_name=self._name,
            workflow_version=self._version,
            iterations=self._iterations,
            hardware=hardware,
            total_wall_time=AggregatedMetrics.from_values(wall_times),
            total_cpu_time=AggregatedMetrics.from_values(cpu_times),
            total_peak_rss=AggregatedMetrics.from_values(peak_rss_list),
            step_results=[step_result],
            raw_metrics=[],
            tags=self._tags,
        )

        if self._store is not None:
            self._store.save(record)

        return record

    def _get_args(self) -> Tuple[tuple, dict]:
        """Get arguments for the callable from the setup function.

        Returns
        -------
        Tuple[tuple, dict]
            ``(args, kwargs)`` to pass to the callable.
        """
        if self._setup is not None:
            return self._setup()
        return ((), {})


def as_pytest_benchmark(
    component: ComponentBenchmark,
    benchmark_fixture: Any,
) -> BenchmarkRecord:
    """Run a ``ComponentBenchmark`` through pytest-benchmark's fixture.

    Bridges the component benchmark into pytest-benchmark for CI
    integration.  The function under test is run through the
    pytest-benchmark ``benchmark`` fixture, and a ``BenchmarkRecord``
    is also produced for storage.

    Parameters
    ----------
    component : ComponentBenchmark
        The component benchmark configuration.
    benchmark_fixture : pytest_benchmark.fixture.BenchmarkFixture
        The pytest ``benchmark`` fixture from a test function parameter.

    Returns
    -------
    BenchmarkRecord
        The benchmark result (also captured by pytest-benchmark).
    """
    # Run through pytest-benchmark
    if component._setup is not None:
        args, kwargs = component._setup()
        benchmark_fixture(component._fn, *args, **kwargs)
    else:
        benchmark_fixture(component._fn)

    # Run through our own infrastructure for the BenchmarkRecord
    return component.run()
