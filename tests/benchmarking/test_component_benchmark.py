# -*- coding: utf-8 -*-
"""
Tests for ComponentBenchmark.

Validates timing measurement, setup/teardown callbacks, record
structure, warmup exclusion, and store integration using trivial
callables.

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
2026-02-13
"""

# Standard library
import time

# Third-party
import numpy as np
import pytest

# Internal
from grdl_te.benchmarking.component import ComponentBenchmark
from grdl_te.benchmarking.store import JSONBenchmarkStore


def _trivial_fn(x: np.ndarray) -> np.ndarray:
    """A trivial function to benchmark."""
    return x * 2.0


def _slow_fn() -> None:
    """A function with measurable wall time."""
    time.sleep(0.01)


class TestComponentBenchmark:
    """Tests for ComponentBenchmark."""

    def test_wall_time_positive(self):
        """Benchmarked function has positive wall time."""
        bench = ComponentBenchmark(
            name="slow_fn",
            fn=_slow_fn,
            iterations=3,
            warmup=0,
        )
        record = bench.run()

        assert record.total_wall_time.mean > 0
        assert record.total_wall_time.min > 0
        assert record.total_wall_time.count == 3

    def test_benchmark_type(self):
        """Record has benchmark_type='component'."""
        bench = ComponentBenchmark(
            name="test", fn=lambda: None, iterations=1, warmup=0
        )
        record = bench.run()
        assert record.benchmark_type == "component"

    def test_single_step_result(self):
        """Component produces exactly one step result."""
        bench = ComponentBenchmark(
            name="test_proc",
            fn=lambda: None,
            iterations=3,
            warmup=0,
        )
        record = bench.run()

        assert len(record.step_results) == 1
        step = record.step_results[0]
        assert step.step_index == 0
        assert step.processor_name == "test_proc"
        assert step.sample_count == 3
        assert step.gpu_used is False

    def test_setup_called(self):
        """Setup function is called before each iteration."""
        call_count = {"setup": 0, "fn": 0}

        def setup():
            call_count["setup"] += 1
            return ((), {})

        def fn():
            call_count["fn"] += 1

        bench = ComponentBenchmark(
            name="test",
            fn=fn,
            setup=setup,
            iterations=3,
            warmup=2,
        )
        bench.run()

        # setup called for warmup + measurement iterations
        assert call_count["setup"] == 5
        assert call_count["fn"] == 5

    def test_teardown_called(self):
        """Teardown function is called after each iteration."""
        teardown_count = {"count": 0}

        def teardown():
            teardown_count["count"] += 1

        bench = ComponentBenchmark(
            name="test",
            fn=lambda: None,
            teardown=teardown,
            iterations=3,
            warmup=1,
        )
        bench.run()

        # teardown called for warmup + measurement
        assert teardown_count["count"] == 4

    def test_setup_provides_args(self):
        """Setup function provides args to the callable."""
        received = {"value": None}

        def fn(x, multiplier=1):
            received["value"] = x * multiplier

        def setup():
            return ((np.array([1.0, 2.0]),), {"multiplier": 3})

        bench = ComponentBenchmark(
            name="test",
            fn=fn,
            setup=setup,
            iterations=1,
            warmup=0,
        )
        bench.run()

        np.testing.assert_array_equal(
            received["value"], np.array([3.0, 6.0])
        )

    def test_warmup_excluded(self):
        """Warmup iterations are not included in results."""
        bench = ComponentBenchmark(
            name="test",
            fn=lambda: None,
            iterations=3,
            warmup=5,
        )
        record = bench.run()

        assert record.iterations == 3
        assert record.total_wall_time.count == 3

    def test_store_integration(self, tmp_path):
        """Record is persisted when store is provided."""
        store = JSONBenchmarkStore(base_dir=tmp_path)
        bench = ComponentBenchmark(
            name="stored_test",
            fn=lambda: None,
            iterations=2,
            warmup=0,
            store=store,
        )
        record = bench.run()

        loaded = store.load(record.benchmark_id)
        assert loaded.benchmark_id == record.benchmark_id
        assert loaded.workflow_name == "stored_test"

    def test_tags_preserved(self):
        """User-defined tags are attached to the record."""
        bench = ComponentBenchmark(
            name="test",
            fn=lambda: None,
            iterations=1,
            warmup=0,
            tags={"size": "4096x4096"},
        )
        record = bench.run()
        assert record.tags == {"size": "4096x4096"}

    def test_version(self):
        """Version string is captured in the record."""
        bench = ComponentBenchmark(
            name="test",
            fn=lambda: None,
            iterations=1,
            warmup=0,
            version="2.0.0",
        )
        record = bench.run()
        assert record.workflow_version == "2.0.0"

    def test_hardware_snapshot_captured(self):
        """Record includes a valid hardware snapshot."""
        bench = ComponentBenchmark(
            name="test",
            fn=lambda: None,
            iterations=1,
            warmup=0,
        )
        record = bench.run()

        assert record.hardware.cpu_count >= 1
        assert record.hardware.captured_at

    def test_invalid_iterations_raises(self):
        """iterations < 1 raises ValueError."""
        with pytest.raises(ValueError, match="iterations"):
            ComponentBenchmark(
                name="test", fn=lambda: None, iterations=0
            )

    def test_invalid_warmup_raises(self):
        """warmup < 0 raises ValueError."""
        with pytest.raises(ValueError, match="warmup"):
            ComponentBenchmark(
                name="test", fn=lambda: None, warmup=-1
            )

    def test_numpy_operation_benchmark(self):
        """End-to-end benchmark of a real numpy operation."""
        arr = np.random.rand(256, 256).astype(np.float32)

        bench = ComponentBenchmark(
            name="numpy_multiply",
            fn=_trivial_fn,
            setup=lambda: ((arr,), {}),
            iterations=5,
            warmup=1,
        )
        record = bench.run()

        assert record.iterations == 5
        assert record.total_wall_time.mean > 0
        assert record.total_cpu_time.mean >= 0
        assert len(record.step_results) == 1

    def test_memory_tracking(self):
        """Memory usage is tracked via tracemalloc."""

        def allocate_memory():
            # Allocate ~1MB
            _ = np.zeros(250_000, dtype=np.float32)

        bench = ComponentBenchmark(
            name="memory_test",
            fn=allocate_memory,
            iterations=3,
            warmup=0,
        )
        record = bench.run()

        # peak_rss_bytes should be non-negative
        assert record.total_peak_rss.mean >= 0
