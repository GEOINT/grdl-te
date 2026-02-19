# -*- coding: utf-8 -*-
"""
Benchmarking subpackage — performance evaluation infrastructure.

Provides data models for benchmark results, abstract base classes for
runners and stores, and concrete implementations for active workflow
benchmarking, component-level profiling, and JSON file persistence.

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

from grdl_te.benchmarking.models import (
    AggregatedMetrics,
    BenchmarkRecord,
    HardwareSnapshot,
    StepBenchmarkResult,
)
from grdl_te.benchmarking.base import BenchmarkRunner, BenchmarkStore
from grdl_te.benchmarking.source import ARRAY_SIZES, BenchmarkSource
from grdl_te.benchmarking.store import JSONBenchmarkStore
from grdl_te.benchmarking.component import ComponentBenchmark, as_pytest_benchmark
from grdl_te.benchmarking.report import format_report, print_report, save_report
from grdl_te.benchmarking.suite import run_suite

__all__ = [
    "ARRAY_SIZES",
    "ActiveBenchmarkRunner",
    "AggregatedMetrics",
    "BenchmarkRecord",
    "BenchmarkRunner",
    "BenchmarkSource",
    "BenchmarkStore",
    "ComponentBenchmark",
    "HardwareSnapshot",
    "JSONBenchmarkStore",
    "StepBenchmarkResult",
    "as_pytest_benchmark",
    "format_report",
    "print_report",
    "run_suite",
    "save_report",
]


def __getattr__(name: str):
    """Lazy import for ActiveBenchmarkRunner (requires grdl-runtime)."""
    if name == "ActiveBenchmarkRunner":
        from grdl_te.benchmarking.active import ActiveBenchmarkRunner
        return ActiveBenchmarkRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
