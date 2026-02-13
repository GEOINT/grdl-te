# -*- coding: utf-8 -*-
"""
GRDL Testing & Evaluation — benchmarking and validation package.

Provides benchmarking infrastructure for profiling GRDL workflows and
individual components, with structured result storage and aggregation.

Dependencies
------------
numpy

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

from grdl_te.benchmarking import (
    AggregatedMetrics,
    BenchmarkRecord,
    BenchmarkRunner,
    BenchmarkStore,
    ComponentBenchmark,
    HardwareSnapshot,
    JSONBenchmarkStore,
    StepBenchmarkResult,
    as_pytest_benchmark,
)

__all__ = [
    "ActiveBenchmarkRunner",
    "AggregatedMetrics",
    "BenchmarkRecord",
    "BenchmarkRunner",
    "BenchmarkStore",
    "ComponentBenchmark",
    "HardwareSnapshot",
    "JSONBenchmarkStore",
    "StepBenchmarkResult",
    "as_pytest_benchmark",
]


def __getattr__(name: str):
    """Lazy import for ActiveBenchmarkRunner (requires grdl-runtime)."""
    if name == "ActiveBenchmarkRunner":
        from grdl_te.benchmarking.active import ActiveBenchmarkRunner
        return ActiveBenchmarkRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
