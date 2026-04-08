# -*- coding: utf-8 -*-
"""
Benchmarking subpackage — performance evaluation infrastructure.

Provides data models for benchmark results, abstract base classes for
runners and stores, and concrete implementations for active workflow
benchmarking, passive forensic benchmarking from pre-recorded traces,
component-level profiling, and JSON file persistence.

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
2026-03-17
"""

from grdl_te.benchmarking.models import (
    AggregatedMetrics,
    BenchmarkRecord,
    HardwareSnapshot,
    StepBenchmarkResult,
    TopologyDescriptor,
    WorkflowTopology,
)
from grdl_te.benchmarking.base import BenchmarkRunner, BenchmarkStore
from grdl_te.benchmarking.source import ARRAY_SIZES, BenchmarkSource
from grdl_te.benchmarking.store import JSONBenchmarkStore
from grdl_te.benchmarking.component import ComponentBenchmark, as_pytest_benchmark
from grdl_te.benchmarking.report import format_report, print_report, save_report
from grdl_te.benchmarking.report_md import format_report_md, save_report_md
from grdl_te.benchmarking.topology import (
    classify_topology,
    compute_critical_path,
    compute_latency_contributions,
)
from grdl_te.benchmarking.comparison import (
    ComparisonResult,
    compare_records,
)
from grdl_te.benchmarking.report_engine import (
    ReportData,
    build_report_data,
)
from grdl_te.benchmarking.suite import run_suite
from grdl_te.benchmarking.forensic import ForensicExecutionTrace, ForensicTraceReader
from grdl_te.benchmarking.stress_models import (
    FailurePoint,
    StressTestConfig,
    StressTestEvent,
    StressTestRecord,
    StressTestSummary,
)
from grdl_te.benchmarking.stress_base import BaseStressTester
from grdl_te.benchmarking.stress_runner import ComponentStressTester
from grdl_te.benchmarking.stress_store import JSONStressTestStore
from grdl_te.benchmarking.unified_store import GRDLStore
from grdl_te.benchmarking.stress_report import (
    format_stress_report,
    format_stress_report_md,
    print_stress_report,
    save_stress_report,
    save_stress_report_md,
)

__all__ = [
    "ARRAY_SIZES",
    "ActiveBenchmarkRunner",
    "AggregatedMetrics",
    "BenchmarkRecord",
    "BenchmarkRunner",
    "BenchmarkSource",
    "BenchmarkStore",
    "ReportData",
    "build_report_data",
    "ComparisonResult",
    "ComponentBenchmark",
    "ForensicExecutionTrace",
    "ForensicTraceReader",
    "HardwareSnapshot",
    "JSONBenchmarkStore",
    "PassiveBenchmarkRunner",
    "StepBenchmarkResult",
    "TopologyDescriptor",
    "WorkflowTopology",
    "as_pytest_benchmark",
    "classify_topology",
    "compare_records",
    "compute_critical_path",
    "compute_latency_contributions",
    "format_report",
    "format_report_md",
    "launch_ui",
    "print_report",
    "run_suite",
    "save_report",
    "save_report_md",
    # Stress testing
    "BaseStressTester",
    "ComponentStressTester",
    "FailurePoint",
    "GRDLStore",
    "JSONStressTestStore",
    "WorkflowStressTester",
    "StressTestConfig",
    "StressTestEvent",
    "StressTestRecord",
    "StressTestSummary",
    "format_stress_report",
    "format_stress_report_md",
    "print_stress_report",
    "save_stress_report",
    "save_stress_report_md",
    "launch_stress_gui",
]


def __getattr__(name: str):
    """Lazy import for runners that require grdl-runtime."""
    if name == "ActiveBenchmarkRunner":
        from grdl_te.benchmarking.active import ActiveBenchmarkRunner
        return ActiveBenchmarkRunner
    if name == "launch_ui":
        from grdl_te.benchmarking.report_gui import launch_ui
        return launch_ui
    if name == "launch_stress_gui":
        from grdl_te.benchmarking.stress_gui import launch_stress_gui
        return launch_stress_gui
    if name == "PassiveBenchmarkRunner":
        from grdl_te.benchmarking.passive import PassiveBenchmarkRunner
        return PassiveBenchmarkRunner
    if name == "WorkflowStressTester":
        from grdl_te.benchmarking.stress_runner import WorkflowStressTester
        return WorkflowStressTester
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
