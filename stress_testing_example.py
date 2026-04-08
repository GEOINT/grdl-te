#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stress Testing Example — complete demonstration of GRDL-TE stress testing.

This script shows every feature of the stress testing engine:

  1.  Default configuration — run with built-in defaults
  2.  Custom configuration — override concurrency range, duration, payload
  3.  Setup callbacks — control exactly what arguments reach the callable
  4.  Failure detection — observe a deliberately fragile component break
  5.  Persistent storage — save via GRDLStore, reload, and compare
  6.  Benchmark linkage — link a stress record to a ComponentBenchmark record
  7.  Subclassing BaseStressTester — create a custom stress tester class
  8.  Text and Markdown reports — save reports to files
  9.  Cross-run comparison — load two records and compare their summaries
 10.  Tiler pipeline stress test — realistic multi-step pipeline
 11.  WorkflowStressTester — stress test a grdl-runtime Workflow directly

Run with::

    conda activate grdx
    python stress_testing_example.py

The script completes in under 60 seconds.  It writes output to
``./stress_example_output/`` and prints a summary for each scenario.

Dependencies
------------
grdl
grdl-te
numpy
psutil (optional — used for memory sampling when available)

Author
------
Ava Courtney <courtney-ava@zai.com>

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2026-04-08

Modified
--------
2026-05-13
"""

from __future__ import annotations

# Standard library
import time
from pathlib import Path
from typing import Any, Tuple

# Third-party
import numpy as np

# grdl components
from grdl.data_prep import ChipExtractor, Normalizer, Tiler
from grdl.image_processing.filters import GaussianFilter, MeanFilter
from grdl.image_processing.intensity import ToDecibels

# grdl-te stress testing engine
from grdl_te.benchmarking import (
    BaseStressTester,
    ComponentBenchmark,
    ComponentStressTester,
    FailurePoint,
    GRDLStore,
    StressTestConfig,
    StressTestRecord,
    format_stress_report,
    format_stress_report_md,
    print_stress_report,
    save_stress_report,
    save_stress_report_md,
)

# grdl-runtime workflow integration (optional — skipped if not installed)
try:
    from grdl_rt import Workflow  # type: ignore
    _HAS_RUNTIME = True
except ImportError:
    _HAS_RUNTIME = False

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("./stress_example_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STORE_DIR = OUTPUT_DIR / "store"

# ---------------------------------------------------------------------------
# Shared banner helpers
# ---------------------------------------------------------------------------

def _banner(n: int, title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  Scenario {n}: {title}")
    print(f"{'=' * 70}")

def _ok(msg: str) -> None:
    print(f"  [OK]  {msg}")

def _info(msg: str) -> None:
    print(f"        {msg}")

# ---------------------------------------------------------------------------
# Shared fast config used throughout the examples (keeps total runtime short)
# ---------------------------------------------------------------------------
FAST_CONFIG = StressTestConfig(
    start_concurrency=1,
    max_concurrency=4,
    ramp_steps=3,
    duration_per_step_s=2.0,   # 2 s per ramp level -- fast enough to demo
    payload_size="small",       # 512 x 512
    timeout_per_call_s=10.0,
)


# ===========================================================================
# Scenario 1 — Default Configuration
#
# The simplest possible usage: wrap any callable with ComponentStressTester
# and call run() with no arguments.  The engine uses StressTestConfig
# defaults (up to 16 workers, medium payload, 10 s per step).
# ===========================================================================

def scenario_1_default_config() -> StressTestRecord:
    _banner(1, "Default Configuration")

    norm = Normalizer(method="minmax")

    tester = ComponentStressTester(
        name="Normalizer.minmax",
        fn=norm.normalize,
    )

    # Use fast config here; in production you would call tester.run() with
    # no arguments and StressTestConfig() defaults would apply.
    record = tester.run(FAST_CONFIG)

    s = record.summary
    _ok(f"Component:               {record.component_name}")
    _info(f"Total calls:             {s.total_calls}")
    _info(f"Failed calls:            {s.failed_calls}")
    _info(f"Max sustained workers:   {s.max_sustained_concurrency}")
    _info(f"Saturation point:        {s.saturation_concurrency}")
    _info(f"p99 latency:             {s.p99_latency_s:.3f}s")
    _info(f"Memory high-water mark:  {s.memory_high_water_mark_bytes / 1024:.0f} KB")
    return record


# ===========================================================================
# Scenario 2 — Custom Configuration
#
# Override the key ramp parameters via StressTestConfig.  Pass it explicitly
# to run().  This is the standard way to control the ramp shape, payload
# size, per-step duration, and timeout.
# ===========================================================================

def scenario_2_custom_config() -> StressTestRecord:
    _banner(2, "Custom Configuration")

    filt = GaussianFilter(sigma=1.5)

    custom_config = StressTestConfig(
        start_concurrency=1,
        max_concurrency=8,      # test up to 8 concurrent workers
        ramp_steps=4,           # 1 → 2 → 4 → 8
        duration_per_step_s=2.0,
        payload_size="small",   # 512 x 512 float32
        timeout_per_call_s=15.0,
    )

    tester = ComponentStressTester(
        name="GaussianFilter.sigma1.5",
        fn=filt.apply,
        version="0.4.0",
    )

    record = tester.run(custom_config)

    s = record.summary
    _ok(f"Component:  {record.component_name}")
    _ok(f"Config:     {len(custom_config.concurrency_levels())} ramp levels "
        f"{custom_config.concurrency_levels()}")
    _info(f"Max sustained workers:  {s.max_sustained_concurrency}")
    _info(f"Total calls:            {s.total_calls}")
    _info(f"p99 latency:            {s.p99_latency_s:.3f}s")
    return record


# ===========================================================================
# Scenario 3 — Setup Callback
#
# The setup callback is called once per call attempt and receives the
# payload array.  It returns (args, kwargs) that are forwarded to fn().
# Use this when your callable needs extra arguments beyond the payload,
# or when you want to pre-process the payload before each call.
# ===========================================================================

def scenario_3_setup_callback() -> StressTestRecord:
    _banner(3, "Setup Callback")

    tdb = ToDecibels(floor_db=-80.0)

    # The setup callback receives the payload ndarray and returns
    # (args, kwargs).  Here we force positive values and pass the result.
    def setup(payload: np.ndarray) -> Tuple[tuple, dict]:
        positive = np.abs(payload) + 1e-9   # ensure all values > 0 for dB
        return (positive,), {}

    tester = ComponentStressTester(
        name="ToDecibels.floor_minus80",
        fn=tdb.apply,
        setup=setup,
    )

    record = tester.run(FAST_CONFIG)
    s = record.summary
    _ok(f"Setup callback invoked across {s.total_calls} calls")
    _info(f"All events successful: {s.failed_calls == 0}")
    _info(f"p99 latency:          {s.p99_latency_s:.4f}s")
    return record


# ===========================================================================
# Scenario 4 — Failure Detection
#
# Pass a function that raises an exception when called concurrently above
# a threshold.  The engine records FailurePoints and populates summary
# saturation fields.
#
# Here we simulate a component that degrades under load by counting
# concurrent callers and failing when the thread pool is too loaded.
# ===========================================================================

def scenario_4_failure_detection() -> StressTestRecord:
    _banner(4, "Failure Detection")

    import threading

    _active_count = 0
    _lock = threading.Lock()
    _CONCURRENCY_LIMIT = 3  # simulated limit

    def fragile_processor(arr: np.ndarray) -> np.ndarray:
        """Simulates a component that fails when too many threads hit it."""
        nonlocal _active_count
        with _lock:
            _active_count += 1
            current = _active_count
        try:
            if current > _CONCURRENCY_LIMIT:
                raise RuntimeError(
                    f"Resource exhausted at {current} concurrent callers "
                    f"(limit={_CONCURRENCY_LIMIT})"
                )
            time.sleep(0.05)  # simulate processing time
            return arr * 2.0
        finally:
            with _lock:
                _active_count -= 1

    failure_config = StressTestConfig(
        start_concurrency=1,
        max_concurrency=6,
        ramp_steps=4,
        duration_per_step_s=1.5,
        payload_size="small",
        timeout_per_call_s=5.0,
    )

    tester = ComponentStressTester(
        name="FragileProcessor",
        fn=fragile_processor,
        tags={"test_type": "failure_detection"},
    )

    record = tester.run(failure_config)
    s = record.summary

    _ok(f"Failure detection complete")
    _info(f"Total calls:             {s.total_calls}")
    _info(f"Failed calls:            {s.failed_calls}")
    _info(f"Max sustained workers:   {s.max_sustained_concurrency}")

    if s.saturation_concurrency is not None:
        _info(f"Saturation point:        {s.saturation_concurrency} workers")
        _info(f"First failure mode:      {s.first_failure_mode}")
    else:
        _info(f"No saturation detected (component held under full load)")

    for fp in record.failure_points:
        _info(
            f"FailurePoint: {fp.error_type} @ concurrency={fp.concurrency_level}  "
            f"memory={fp.memory_bytes_at_failure / 1024:.0f}KB"
        )

    return record


# ===========================================================================
# Scenario 5 — Persistent Storage
#
# Save a stress record via GRDLStore, reload it by ID, and verify it
# survives the round-trip.  GRDLStore handles both BenchmarkRecord and
# StressTestRecord under one root directory — no need to manage two
# separate store objects.
# ===========================================================================

def scenario_5_persistent_storage(base_record: StressTestRecord) -> None:
    _banner(5, "Persistent Storage (GRDLStore)")

    store = GRDLStore(base_dir=STORE_DIR)

    # Save
    saved_id = store.save(base_record)
    _ok(f"Saved stress record: {saved_id}")

    # Reload by ID
    loaded = store.load_stress(saved_id)
    assert loaded.stress_test_id == base_record.stress_test_id
    assert loaded.config == base_record.config
    _ok(f"Reloaded and verified round-trip for {loaded.component_name}")

    # List with filtering
    matches = store.list_stress_tests(component_name=base_record.component_name)
    _info(f"list_stress_tests() returned {len(matches)} record(s) for "
          f"'{base_record.component_name}'")

    # Show storage layout
    stress_dir = STORE_DIR / "stress"
    if stress_dir.exists():
        record_files = list((stress_dir / "records").glob("*.json"))
        _info(f"Files on disk: {stress_dir / 'records'} ({len(record_files)} JSON file(s))")


# ===========================================================================
# Scenario 6 — Benchmark Linkage
#
# Run a ComponentBenchmark first, then run a stress test for the same
# component and link the two via related_benchmark_id.  Both records are
# saved to the SAME GRDLStore — benchmark records go to records/ and stress
# records go to stress/records/ under the same root.
# ===========================================================================

def scenario_6_benchmark_linkage() -> StressTestRecord:
    _banner(6, "Benchmark Linkage (single GRDLStore)")

    mean_filter = MeanFilter(kernel_size=5)
    image = np.random.rand(512, 512).astype(np.float32)

    # ---- Step 1: standard benchmark ----------------------------------------
    # GRDLStore handles both record types — no need for a separate bench store
    store = GRDLStore(base_dir=STORE_DIR)

    bench = ComponentBenchmark(
        name="MeanFilter.5x5",
        fn=mean_filter.apply,
        setup=lambda: ((image,), {}),
        iterations=5,
        warmup=1,
        store=store,
        tags={"module": "filters", "array_size": "small"},
    )
    bench_record = bench.run()
    _ok(f"Benchmark complete:  mean wall={bench_record.total_wall_time.mean:.4f}s  "
        f"p95={bench_record.total_wall_time.p95:.4f}s")

    # ---- Step 2: stress test linked to the benchmark -----------------------
    tester = ComponentStressTester(
        name="MeanFilter.5x5",
        fn=mean_filter.apply,
        version="0.4.0",
        related_benchmark_id=bench_record.benchmark_id,  # <-- linkage
        store=store,
        tags={"module": "filters"},
    )

    record = tester.run(FAST_CONFIG)

    _ok(f"Stress test complete: max_sustained={record.summary.max_sustained_concurrency}")
    _info(f"Linked to benchmark:  {record.related_benchmark_id}")

    # Demonstrate that both record types are in the same store directory
    bench_files = list((STORE_DIR / "records").glob("*.json")) if (STORE_DIR / "records").exists() else []
    stress_files = list((STORE_DIR / "stress" / "records").glob("*.json")) if (STORE_DIR / "stress" / "records").exists() else []
    _info(f"Benchmark records in store: {len(bench_files)}")
    _info(f"Stress records in store:    {len(stress_files)}")

    # Demonstrate the linkage in a report snippet
    report = format_stress_report(record)
    has_xref = "CROSS-REFERENCE" in report
    _info(f"Report contains cross-reference section: {has_xref}")

    return record


# ===========================================================================
# Scenario 7 — Subclassing BaseStressTester
#
# For components that require richer set-up (e.g. objects with state, or
# pipeline chains), subclass BaseStressTester and implement call_once().
# The base class handles the entire ramp loop automatically.
# ===========================================================================

class NormalizeThenFilterStressTester(BaseStressTester):
    """Stress test a normalise → filter pipeline chain.

    Demonstrates how to subclass BaseStressTester for a multi-step
    operation.  The ramp loop, failure detection, and record creation
    are all handled by the base class.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._norm = Normalizer(method="zscore")
        self._filt = MeanFilter(kernel_size=3)

    @property
    def component_name(self) -> str:
        return "NormalizeThenFilter.pipeline"

    def call_once(self, payload: np.ndarray) -> np.ndarray:
        """Run normalize → filter as a single atomic call."""
        normalized = self._norm.normalize(payload)
        return self._filt.apply(normalized)


def scenario_7_subclass_base_stress_tester() -> StressTestRecord:
    _banner(7, "Subclassing BaseStressTester")

    tester = NormalizeThenFilterStressTester(
        version="0.4.0",
        tags={"pipeline": "normalize_filter"},
    )

    record = tester.run(FAST_CONFIG)
    s = record.summary

    _ok(f"Custom tester class:  {record.component_name}")
    _info(f"Total calls:          {s.total_calls}")
    _info(f"Max sustained:        {s.max_sustained_concurrency}")
    _info(f"p99 latency:          {s.p99_latency_s:.4f}s")
    return record


# ===========================================================================
# Scenario 8 — Text and Markdown Reports
#
# Save plain-text and Markdown reports to files.  Shows both the directory
# form (auto-names the file) and the explicit file path form.
# ===========================================================================

def scenario_8_reports(record: StressTestRecord) -> None:
    _banner(8, "Text and Markdown Reports")

    # ---- Print to terminal -------------------------------------------------
    print()
    print_stress_report(record)

    # ---- Save to directory (auto-named) ------------------------------------
    txt_path = save_stress_report(record, OUTPUT_DIR)
    md_path = save_stress_report_md(record, OUTPUT_DIR)
    _ok(f"Text report saved:     {txt_path}")
    _ok(f"Markdown report saved: {md_path}")

    # ---- Save to explicit file path ----------------------------------------
    explicit_txt = OUTPUT_DIR / "scenario8_explicit.txt"
    explicit_md = OUTPUT_DIR / "scenario8_explicit.md"
    save_stress_report(record, explicit_txt)
    save_stress_report_md(record, explicit_md)
    _ok(f"Explicit text path:    {explicit_txt}")
    _ok(f"Explicit md path:      {explicit_md}")

    # ---- Get as string (for embedding in a larger report) ------------------
    text = format_stress_report(record)
    md = format_stress_report_md(record)
    _info(f"format_stress_report()    → {len(text)} characters")
    _info(f"format_stress_report_md() → {len(md)} characters")


# ===========================================================================
# Scenario 9 — Cross-Run Comparison
#
# Run the same component twice with different configs, save both records,
# then reload and compare.  Demonstrates the "loadable" property —
# save a JSON, change grdl code, run again, compare two structures.
# ===========================================================================

def scenario_9_cross_run_comparison() -> None:
    _banner(9, "Cross-Run Comparison")

    store = GRDLStore(base_dir=STORE_DIR)
    norm = Normalizer(method="minmax")

    # Run A — narrow ramp
    config_a = StressTestConfig(
        start_concurrency=1,
        max_concurrency=2,
        ramp_steps=2,
        duration_per_step_s=1.0,
        payload_size="small",
    )
    record_a = ComponentStressTester(
        "Normalizer.minmax", norm.normalize, store=store,
        tags={"run": "baseline"},
    ).run(config_a)
    _ok(f"Run A (baseline):  max_sustained={record_a.summary.max_sustained_concurrency}  "
        f"calls={record_a.summary.total_calls}")

    # Run B — wider ramp
    config_b = StressTestConfig(
        start_concurrency=1,
        max_concurrency=4,
        ramp_steps=3,
        duration_per_step_s=1.0,
        payload_size="small",
    )
    record_b = ComponentStressTester(
        "Normalizer.minmax", norm.normalize, store=store,
        tags={"run": "extended"},
    ).run(config_b)
    _ok(f"Run B (extended):  max_sustained={record_b.summary.max_sustained_concurrency}  "
        f"calls={record_b.summary.total_calls}")

    # Reload both from disk and compare
    loaded_a = store.load_stress(record_a.stress_test_id)
    loaded_b = store.load_stress(record_b.stress_test_id)

    assert set(loaded_a.to_dict().keys()) == set(loaded_b.to_dict().keys()), (
        "Schema keys must match for cross-run comparison"
    )
    _ok("Key structure identical across both records")

    # Compare summaries
    _info(
        f"Schema version:       A={loaded_a.schema_version}  B={loaded_b.schema_version}"
    )
    _info(
        f"GRDL version:         A={loaded_a.grdl_version}  B={loaded_b.grdl_version}"
    )
    _info(
        f"Max sustained:        A={loaded_a.summary.max_sustained_concurrency}  "
        f"B={loaded_b.summary.max_sustained_concurrency}"
    )
    _info(
        f"p99 latency:          A={loaded_a.summary.p99_latency_s:.4f}s  "
        f"B={loaded_b.summary.p99_latency_s:.4f}s"
    )

    # JSON round-trip verification
    restored = StressTestRecord.from_json(loaded_a.to_json())
    assert restored.stress_test_id == loaded_a.stress_test_id
    _ok("JSON round-trip verified: from_json(to_json()) → identical record")


# ===========================================================================
# Scenario 10 — WorkflowStressTester (grdl-runtime integration)
#
# WorkflowStressTester wraps a grdl-runtime Workflow object directly.
#
# Requires grdl_rt to be installed.  Skipped automatically if not present.
# ===========================================================================

def scenario_10_workflow_stress_tester() -> "StressTestRecord | None":
    _banner(11, "WorkflowStressTester (grdl-runtime Workflow)")

    if not _HAS_RUNTIME:
        _info("grdl_rt not installed — skipping Scenario 11.")
        _info("Install grdl-runtime and re-run to see WorkflowStressTester in action.")
        return None

    from grdl_te.benchmarking import WorkflowStressTester  # lazy import

    # Build a simple array-mode workflow (no reader required — array mode).
    # The workflow receives the payload ndarray from call_once directly.
    norm = Normalizer(method="minmax")
    filt = GaussianFilter(sigma=1.0)

    wf = (
        Workflow("NormFilter Pipeline", version="0.1.0")
        .step(norm.normalize)
        .step(filt.apply)
    )

    tester = WorkflowStressTester(
        wf,
        name="NormFilter.pipeline",
        prefer_gpu=False,
        version="0.1.0",
        tags={"source": "example", "type": "workflow"},
    )

    config = StressTestConfig(
        start_concurrency=1,
        max_concurrency=4,
        ramp_steps=3,
        duration_per_step_s=2.0,
        payload_size="small",
        timeout_per_call_s=10.0,
    )

    record = tester.run(config)
    s = record.summary

    _ok(f"WorkflowStressTester complete: {record.component_name}")
    _info(f"Total calls:           {s.total_calls}")
    _info(f"Failed calls:          {s.failed_calls}")
    _info(f"Max sustained workers: {s.max_sustained_concurrency}")
    _info(f"p99 latency:           {s.p99_latency_s:.4f}s")

    # Save via GRDLStore
    store = GRDLStore(base_dir=STORE_DIR)
    store.save(record)
    _ok(f"Record saved: {record.stress_test_id}")

    # save_report dispatches on record type — no format selection needed
    txt_path = OUTPUT_DIR / "scenario11_workflow_stress.txt"
    store.save_report(record, txt_path)
    _ok(f"Report saved: {txt_path}")

    return record


# ===========================================================================
# CLI Equivalent Reference
# ===========================================================================

def print_cli_reference() -> None:
    _banner("CLI", "Equivalent Command-Line Invocations")

    print("""
  The stress testing engine is also accessible via the CLI.

  Run with defaults (StressTestConfig defaults):
    python -m grdl_te --stress-test

  Custom concurrency and duration:
    python -m grdl_te --stress-test --stress-concurrency 8 --stress-duration 5.0

  Custom ramp steps:
    python -m grdl_te --stress-test --stress-concurrency 16 --stress-steps 5

  Use a small payload:
    python -m grdl_te --stress-test --size small --stress-concurrency 4

  Save a report to a directory:
    python -m grdl_te --stress-test --report ./reports/

  Save to a specific file:
    python -m grdl_te --stress-test --report ./stress_report.txt

  Combine with store directory:
    python -m grdl_te --stress-test --store-dir ./my_store --report ./my_store/

  --stress-test is mutually exclusive with the normal benchmark suite;
  the two modes produce separate record types and separate output directories:
    .benchmarks/records/      <- BenchmarkRecord (normal benchmarks)
    .benchmarks/stress/records/ <- StressTestRecord (stress tests)
""")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print("\nGRDL-TE Stress Testing — Complete Feature Demo")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(f"Fast config: {FAST_CONFIG.concurrency_levels()} workers  "
          f"{FAST_CONFIG.duration_per_step_s}s/level  "
          f"payload={FAST_CONFIG.payload_size}")

    # Run all scenarios
    record_1  = scenario_1_default_config()
    record_2  = scenario_2_custom_config()
    record_3  = scenario_3_setup_callback()
    record_4  = scenario_4_failure_detection()
    scenario_5_persistent_storage(record_1)
    record_6  = scenario_6_benchmark_linkage()
    record_7  = scenario_7_subclass_base_stress_tester()
    scenario_8_reports(record_4)   # use the failure record for interesting report output
    scenario_9_cross_run_comparison()
    record_10 = scenario_10_workflow_stress_tester()
    print_cli_reference()

    # Final summary
    print("\n" + "=" * 70)
    print("  ALL SCENARIOS COMPLETE")
    print("=" * 70)
    print(f"\n  Output written to: {OUTPUT_DIR.resolve()}")
    print(f"\n  Records produced:")
    base_records = [record_1, record_2, record_3, record_4, record_6, record_7, record_10]
    if record_11 is not None:
        base_records.append(record_11)
    for i, r in enumerate(base_records, 1):
        s = r.summary
        status = "CLEAN" if s.failed_calls == 0 else f"FAILED@{s.saturation_concurrency}w"
        print(
            f"    {i:2d}. {r.component_name:<45s}  "
            f"max_sustained={s.max_sustained_concurrency:<3}  {status}"
        )
    print()


if __name__ == "__main__":
    main()
