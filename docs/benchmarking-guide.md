# Getting Started with GRDL-TE Benchmarking

This guide walks you through the three benchmarking modes in GRDL-TE — **component**, **active**, and **passive** — and shows you how to save, compare, and report on your results. Each section includes both Python script examples and CLI commands so you can pick whichever fits your workflow.

---

## Prerequisites

```bash
conda activate grdl
pip install -e ".[all,dev]"
```

For **component benchmarking**, only `grdl-te` itself is needed. For **active benchmarking**, you also need `grdl-runtime` (`grdl-rt`). **Passive benchmarking** works without `grdl-runtime` as long as you have JSON trace files to analyze.

---

## 1. Component Benchmarking

Component benchmarking profiles a single function or method in isolation — no workflow, no pipeline. This is the best starting point when you want to measure one specific operation.

### Python Script

```python
import numpy as np
from grdl_te.benchmarking import ComponentBenchmark

# The function you want to benchmark
def median_filter(image):
    from scipy.ndimage import median_filter as mf
    return mf(image, size=3)

# Create a test image once; the setup callback hands it to each iteration
image = np.random.rand(2048, 2048).astype(np.float32)

bench = ComponentBenchmark(
    name="MedianFilter.3x3.2k",
    fn=median_filter,
    setup=lambda: ((image,), {}),   # returns (args, kwargs)
    iterations=10,                   # measured runs
    warmup=2,                        # discarded warm-up runs
    tags={"module": "filters", "array_size": "2048x2048"},
)

record = bench.run()

# Inspect the results
print(f"Mean wall time:  {record.total_wall_time.mean:.4f} s")
print(f"Mean CPU time:   {record.total_cpu_time.mean:.4f} s")
print(f"Peak memory:     {record.total_peak_rss.mean / 1e6:.1f} MB")
print(f"Std dev (wall):  {record.total_wall_time.stddev:.4f} s")
print(f"P95 (wall):      {record.total_wall_time.p95:.4f} s")
```

A few things to notice:

- **`setup`** is called before *every* iteration and must return `(args, kwargs)`. This keeps data preparation out of the timing window.
- **`warmup`** iterations let caches and JIT compilation settle before measurement begins.
- **`teardown`** (not shown) is an optional callback for cleanup after each iteration.
- The record contains a single `StepBenchmarkResult` since there's only one function.

### Benchmarking a GRDL Component

```python
from grdl.data_prep import Normalizer
from grdl_te.benchmarking import ComponentBenchmark

image = np.random.rand(4096, 4096).astype(np.float32)
norm = Normalizer(method="minmax")

bench = ComponentBenchmark(
    name="Normalizer.minmax.4k",
    fn=norm.normalize,
    setup=lambda: ((image,), {}),
    iterations=20,
    warmup=3,
    version="1.2.0",
    tags={"module": "data_prep", "array_size": "large"},
)

record = bench.run()
```

### pytest Integration

If you run benchmarks in CI, you can bridge into `pytest-benchmark`:

```python
import pytest
from grdl_te.benchmarking import ComponentBenchmark, as_pytest_benchmark

def test_normalizer_perf(benchmark):
    image = np.random.rand(512, 512).astype(np.float32)
    norm = Normalizer(method="minmax")

    bench = ComponentBenchmark(
        name="Normalizer.minmax.small",
        fn=norm.normalize,
        setup=lambda: ((image,), {}),
        iterations=5,
        warmup=1,
    )

    record = as_pytest_benchmark(bench, benchmark)
    assert record.total_wall_time.mean < 1.0  # sanity gate
```

### CLI — Running the Built-in Suite

The CLI runs GRDL-TE's full component benchmark suite against the library's processing modules:

```bash
# Default: medium arrays (2048x2048), 10 iterations
python -m grdl_te

# Quick smoke test
python -m grdl_te --size small -n 5

# Thorough run with large arrays
python -m grdl_te --size large -n 20

# Only benchmark specific module groups
python -m grdl_te --only filters intensity data_prep

# Skip the workflow-level benchmark (component-only)
python -m grdl_te --skip-workflow
```

Available array size presets:

| Preset   | Dimensions    | Approx. Size (float32) |
|----------|---------------|------------------------|
| `small`  | 512 x 512     | ~1 MB                  |
| `medium` | 2048 x 2048   | ~16 MB                 |
| `large`  | 4096 x 4096   | ~64 MB                 |

Available benchmark groups: `filters`, `intensity`, `decomposition`, `detection`, `pipeline`, `data_prep`, `io`, `geolocation`, `coregistration`, `ortho`, `sar`, `interpolation`, `image_formation`, `workflow`.

---

## 2. Active Benchmarking

Active benchmarking runs a full `grdl-runtime` Workflow end-to-end, multiple times, and aggregates per-step metrics. This is what you want when you need to measure a real pipeline — including step dependencies, GPU offload, and memory pressure across the entire execution graph.

> **Requires:** `grdl-runtime` (`pip install grdl-runtime`)

### Python Script

```python
from grdl_rt import Workflow
from grdl_te.benchmarking import ActiveBenchmarkRunner, BenchmarkSource

# Build a workflow
wf = (
    Workflow("SAR Pipeline", modalities=["SAR"])
    .reader(SICDReader)
    .step(SublookDecomposition, num_looks=3)
    .step(ToDecibels)
)

# Create a synthetic data source
source = BenchmarkSource.synthetic("medium")

# Set up the runner
runner = ActiveBenchmarkRunner(
    workflow=wf,
    source=source,
    iterations=10,
    warmup=2,
    tags={"pipeline": "sar", "gpu": "enabled"},
)

# Run the benchmark — pass workflow execution kwargs here
record = runner.run(prefer_gpu=True)

# Inspect workflow-level totals
print(f"Total wall time (mean): {record.total_wall_time.mean:.3f} s")
print(f"Total CPU time (mean):  {record.total_cpu_time.mean:.3f} s")
print(f"Total peak RSS (mean):  {record.total_peak_rss.mean / 1e6:.1f} MB")
print(f"Topology:               {record.topology.topology.value}")

# Inspect per-step breakdowns
for step in record.step_results:
    print(
        f"  [{step.step_index}] {step.processor_name:30s} "
        f"wall={step.wall_time_s.mean:.4f}s  "
        f"mem={step.peak_rss_bytes.mean / 1e6:.1f}MB  "
        f"latency_pct={step.latency_pct:.1f}%"
    )
```

### Data Sources

You have three ways to feed data into an active benchmark:

```python
# 1. Synthetic — generated arrays with reproducible seed
source = BenchmarkSource.synthetic("medium")                     # preset
source = BenchmarkSource.synthetic((1024, 1024), seed=99)        # custom shape

# 2. From file — for benchmarking with real data
source = BenchmarkSource.from_file("data/umbra/sample.nitf")

# 3. From an existing array
arr = np.random.rand(2048, 2048).astype(np.float32)
source = BenchmarkSource.from_array(arr)
```

Data is resolved once at the start and cached across all warmup + measurement iterations.

### Progress Tracking

For long-running benchmarks, pass a progress callback:

```python
def on_progress(current, total):
    print(f"  Iteration {current}/{total}")

record = runner.run(progress_callback=on_progress, prefer_gpu=True)
```

### Topology and Critical Path

Active benchmarks automatically classify the workflow topology and compute the critical path for parallel workflows:

```python
from grdl_te.benchmarking import classify_topology

topo = record.topology
print(f"Topology:            {topo.topology.value}")       # sequential, parallel, mixed
print(f"Branches:            {topo.num_branches}")
print(f"Critical path:       {topo.critical_path_step_ids}")
print(f"Critical path time:  {topo.critical_path_wall_time_s:.3f} s")

# Per-step latency contributions (as percentages)
for name, pct in record.step_latency_pct.items():
    print(f"  {name}: {pct:.1f}% of total latency")
```

---

## 3. Passive Benchmarking

Passive benchmarking analyzes *pre-recorded* execution traces — it never runs any workflow. This is useful when you want to benchmark historical runs, analyze production traces, or compare runs from different machines without re-executing them.

### Loading Traces

The `ForensicTraceReader` loads traces from four source types:

```python
from grdl_te.benchmarking import ForensicTraceReader

# 1. Single JSON file (from WorkflowMetrics.to_json())
traces = ForensicTraceReader.from_json_file("run_output.json")

# 2. Directory of JSON files
traces = ForensicTraceReader.from_json_directory(
    "/exports/sar_runs/",
    glob_pattern="*.json",        # default
    status_filter="success",      # only successful runs (default); None for all
)

# 3. From grdl-runtime's execution history database
traces = ForensicTraceReader.from_history_db(
    run_ids=["abc-123", "def-456"],
    db_path=None,                 # defaults to ~/.grdl_rt/history.db
)

# 4. In-memory dicts (e.g., from an active benchmark's raw_metrics)
traces = ForensicTraceReader.from_memory(active_record.raw_metrics)
```

### Running the Passive Benchmark

```python
from grdl_te.benchmarking import PassiveBenchmarkRunner

# Load traces from exported JSON files
traces = ForensicTraceReader.from_json_directory("/exports/sar_runs/")

runner = PassiveBenchmarkRunner(
    traces=traces,
    tags={"environment": "production", "cluster": "gpu-01"},
)

record = runner.run()

print(f"Benchmark type: {record.benchmark_type}")        # "passive"
print(f"Workflow:       {record.workflow_name}")
print(f"Iterations:     {record.iterations}")             # = number of traces
print(f"Trace sources:  {record.tags['forensic_source']}")
print(f"Trace count:    {record.tags['trace_count']}")

# Same step-level access as active benchmarks
for step in record.step_results:
    print(f"  {step.processor_name}: {step.wall_time_s.mean:.4f}s mean")
```

### Hardware Resolution

Passive benchmarks resolve hardware info from the traces themselves:

1. If you pass an explicit `hardware` snapshot, that is used.
2. If all traces have embedded hardware data from the same machine, it is reconstructed automatically.
3. Otherwise, `hardware` is `None` — reports will show "hardware information missing."

```python
from grdl_te.benchmarking import HardwareSnapshot

# Option A: let the runner figure it out from the traces
record = PassiveBenchmarkRunner(traces).run()

# Option B: override with an explicit snapshot
hw = HardwareSnapshot.capture()
record = PassiveBenchmarkRunner(traces, hardware=hw).run()
```

### Comparing Active vs. Passive

A common pattern is to re-analyze an active benchmark's raw metrics as a passive benchmark to verify parity:

```python
# Run an active benchmark
active_record = runner.run()

# Re-analyze the same data passively
traces = ForensicTraceReader.from_memory(active_record.raw_metrics)
passive_record = PassiveBenchmarkRunner(traces).run()

# These should produce equivalent aggregated metrics
assert abs(active_record.total_wall_time.mean - passive_record.total_wall_time.mean) < 1e-9
```

---

## 4. Saving and Exporting Results

GRDL-TE provides several ways to persist and share your benchmark results.

### JSON Store (Structured Persistence)

The `JSONBenchmarkStore` saves each record as a JSON file with an index for fast lookups. This is the primary way to build up a history of benchmarks over time.

```python
from pathlib import Path
from grdl_te.benchmarking import JSONBenchmarkStore

# Create a store (defaults to .benchmarks/ in the current directory)
store = JSONBenchmarkStore(base_dir=Path("./my_benchmarks"))

# Save a record
benchmark_id = store.save(record)
print(f"Saved: {benchmark_id}")

# Load it back
loaded = store.load(benchmark_id)

# List stored records with filtering
all_records = store.list_records()
active_only = store.list_records(benchmark_type="active")
sar_records = store.list_records(workflow_name="SAR Pipeline")
parallel_records = store.list_records(topology="parallel", limit=10)

# Delete a record
store.delete(benchmark_id)

# Rebuild the index if files were modified externally
store.rebuild_index()
```

**Storage layout on disk:**

```
my_benchmarks/
    index.json              # lightweight index for fast filtering
    records/
        <uuid>.json         # one file per benchmark record
```

#### Auto-Save During Benchmarking

You can pass the store directly to any runner so results are saved automatically:

```python
store = JSONBenchmarkStore(base_dir=Path("./results"))

# Component — auto-saves on run()
bench = ComponentBenchmark(
    name="MedianFilter.3x3.2k",
    fn=median_filter,
    setup=lambda: ((image,), {}),
    iterations=10,
    store=store,
)
record = bench.run()  # saved automatically

# Active — auto-saves on run()
runner = ActiveBenchmarkRunner(
    workflow=wf,
    source=source,
    iterations=10,
    store=store,
)
record = runner.run()  # saved automatically

# Passive — auto-saves on run()
runner = PassiveBenchmarkRunner(traces, store=store)
record = runner.run()  # saved automatically
```

#### CLI — Custom Store Directory

```bash
python -m grdl_te --store-dir ./results
```

### Text Reports

Generate human-readable plain-text reports from one or more records.

```python
from grdl_te.benchmarking import format_report, print_report, save_report
from pathlib import Path

records = store.list_records()

# Print to terminal
print_report(records)

# Get the report as a string
text = format_report(records)

# Save to a specific file
save_report(records, Path("./benchmark_report.txt"))

# Save to a directory (generates a timestamped filename)
save_report(records, Path("./reports/"))
# -> ./reports/benchmark_report_20260318_143000.txt
```

#### CLI — Report Generation

```bash
# Print report to terminal
python -m grdl_te --report

# Save to a file
python -m grdl_te --report ./my_report.txt

# Save to a directory (auto-named with timestamp)
python -m grdl_te --report ./reports/
```

### Markdown Reports

For sharing in pull requests, wikis, or documentation, use the Markdown report format:

```python
from grdl_te.benchmarking import format_report_md, save_report_md
from pathlib import Path

records = store.list_records()

# Get Markdown as a string
md = format_report_md(records)

# Save to file
save_report_md(records, Path("./benchmark_report.md"))
```

Markdown reports include:

- Executive summary with top bottlenecks
- Hardware configuration details
- Per-benchmark step-level tables with latency and memory percentages
- Branch analysis for parallel workflows
- Cross-workflow comparison tables

### Direct JSON Serialization

Every `BenchmarkRecord` can be serialized and deserialized directly:

```python
# Serialize to JSON string
json_str = record.to_json()

# Deserialize
from grdl_te.benchmarking import BenchmarkRecord
loaded = BenchmarkRecord.from_json(json_str)

# Or work with dicts
d = record.to_dict()
loaded = BenchmarkRecord.from_dict(d)
```

---

## 5. Comparing Benchmarks

When you have multiple benchmark records, you can compare them side by side:

```python
from grdl_te.benchmarking import compare_records

records = store.list_records(limit=3)

result = compare_records(
    records=records,
    labels=["baseline", "optimized", "gpu-enabled"],  # optional
)

# Overall wall time comparison
for label, wall_time in result.wall_time_summary.items():
    print(f"{label}: {wall_time:.3f}s")

# Top bottleneck steps across all records
for bottleneck in result.bottlenecks:
    print(f"  {bottleneck}")
```

---

## 6. Putting It All Together

Here's a complete script that runs all three benchmark types, saves them to a shared store, and generates a report:

```python
#!/usr/bin/env python
"""End-to-end benchmarking example."""

import numpy as np
from pathlib import Path

from grdl_te.benchmarking import (
    ActiveBenchmarkRunner,
    BenchmarkSource,
    ComponentBenchmark,
    ForensicTraceReader,
    JSONBenchmarkStore,
    PassiveBenchmarkRunner,
    compare_records,
    print_report,
    save_report_md,
)

# -- Setup ------------------------------------------------------------------
store = JSONBenchmarkStore(base_dir=Path("./benchmark_results"))
image = np.random.rand(2048, 2048).astype(np.float32)


# -- 1. Component benchmark -------------------------------------------------
def apply_threshold(img):
    return (img > 0.5).astype(np.float32)

comp_record = ComponentBenchmark(
    name="threshold.2k",
    fn=apply_threshold,
    setup=lambda: ((image,), {}),
    iterations=10,
    warmup=2,
    store=store,
    tags={"module": "custom", "array_size": "medium"},
).run()

print(f"Component: {comp_record.total_wall_time.mean:.4f}s mean")


# -- 2. Active benchmark (requires grdl-runtime) ----------------------------
try:
    from grdl_rt import Workflow

    wf = (
        Workflow("Example Pipeline")
        .step(SublookDecomposition, num_looks=3)
        .step(ToDecibels)
    )

    active_record = ActiveBenchmarkRunner(
        workflow=wf,
        source=BenchmarkSource.synthetic("medium"),
        iterations=5,
        warmup=1,
        store=store,
        tags={"pipeline": "example"},
    ).run()

    print(f"Active:    {active_record.total_wall_time.mean:.3f}s mean")
except ImportError:
    print("Skipping active benchmark (grdl-runtime not installed)")
    active_record = None


# -- 3. Passive benchmark ---------------------------------------------------
# Analyze previously exported traces
trace_dir = Path("/exports/pipeline_runs/")
if trace_dir.exists():
    traces = ForensicTraceReader.from_json_directory(trace_dir)
    passive_record = PassiveBenchmarkRunner(
        traces,
        store=store,
        tags={"source": "production"},
    ).run()

    print(f"Passive:   {passive_record.total_wall_time.mean:.3f}s mean")
else:
    print("Skipping passive benchmark (no trace directory found)")
    passive_record = None


# -- 4. Report and compare --------------------------------------------------
all_records = store.list_records()

# Print summary to terminal
print_report(all_records)

# Save Markdown report for sharing
save_report_md(all_records, Path("./benchmark_results/report.md"))

print(f"\nDone. {len(all_records)} records saved to ./benchmark_results/")
```

---

## 7. Stress Testing

Stress testing answers a different question than benchmarking: not *"how fast is this?"* but *"at what concurrency level does this break, and what breaks first?"*

Stress tests use a concurrency ramp: the engine submits batches of concurrent calls, doubling the worker count at each step, and records every success and failure event.  When failures are detected, the engine records `FailurePoint` objects describing the concurrency level, memory at failure, and error type.

**Key difference from benchmarking:**
- Benchmark records (`BenchmarkRecord`) track per-step timing distributions at controlled load.
- Stress records (`StressTestRecord`) track failure points, saturation curves, and memory high-water marks under increasing concurrency.

The two record types are stored separately and produce separate reports.  A `related_benchmark_id` field on the stress record optionally links to the corresponding benchmark for cross-reference.

---

### 7.1 Default Usage

The simplest form: wrap any callable with `ComponentStressTester` and call `run()`.

```python
import numpy as np
from grdl.data_prep import Normalizer
from grdl_te.benchmarking import ComponentStressTester

norm = Normalizer(method="minmax")

tester = ComponentStressTester(
    name="Normalizer.minmax",
    fn=norm.normalize,
)

# run() with no args uses StressTestConfig defaults:
#   start_concurrency=1, max_concurrency=16, ramp_steps=5,
#   duration_per_step_s=10.0, payload_size="medium", timeout_per_call_s=30.0
record = tester.run()

s = record.summary
print(f"Max sustained concurrency: {s.max_sustained_concurrency}")
print(f"Saturation point:          {s.saturation_concurrency}")
print(f"First failure mode:        {s.first_failure_mode}")
print(f"p99 latency (success):     {s.p99_latency_s:.3f}s")
print(f"Memory high-water mark:    {s.memory_high_water_mark_bytes / 1e6:.1f} MB")
```

---

### 7.2 Custom Configuration

Override any parameter via `StressTestConfig`:

```python
from grdl_te.benchmarking import ComponentStressTester, StressTestConfig

config = StressTestConfig(
    start_concurrency=1,       # lowest concurrency level tested
    max_concurrency=16,        # highest concurrency level tested
    ramp_steps=5,              # levels: 1 → 2 → 4 → 8 → 16 (geometric doubling)
    duration_per_step_s=10.0,  # sustain each level for 10 seconds
    payload_size="medium",     # 2048x2048 float32 array
    timeout_per_call_s=30.0,   # calls exceeding this count as TimeoutError
)

tester = ComponentStressTester("MyComponent", my_fn)
record = tester.run(config)
```

**Payload size** accepts the same presets as benchmarking or an explicit `"ROWSxCOLS"` string:

```python
StressTestConfig(payload_size="small")       # 512 x 512
StressTestConfig(payload_size="medium")      # 2048 x 2048
StressTestConfig(payload_size="large")       # 4096 x 4096
StressTestConfig(payload_size="1024x768")    # custom dimensions
```

**Concurrency levels** are computed via geometric doubling:

```python
config = StressTestConfig(start_concurrency=1, max_concurrency=16, ramp_steps=5)
print(config.concurrency_levels())  # [1, 2, 4, 8, 16]

config = StressTestConfig(start_concurrency=2, max_concurrency=10, ramp_steps=3)
print(config.concurrency_levels())  # [2, 4, 8] capped at 10 → [2, 4, 8]
```

---

### 7.3 Setup Callback

When your callable needs extra arguments, use the `setup` callback — the same convention as `ComponentBenchmark`:

```python
from grdl.image_processing.intensity import ToDecibels

tdb = ToDecibels(floor_db=-80.0)

def my_setup(payload):
    # Ensure all values are positive before converting to dB
    positive = abs(payload) + 1e-9
    return (positive,), {}   # (args, kwargs)

tester = ComponentStressTester(
    name="ToDecibels",
    fn=tdb.apply,
    setup=my_setup,
)
record = tester.run(config)
```

The `setup` function receives the payload `np.ndarray` and returns `(args, kwargs)`.  `fn` is then called as `fn(*args, **kwargs)`.  When `setup=None` (default), `fn(payload)` is called directly.

---

### 7.4 Subclassing BaseStressTester

For more complex scenarios — components with state, multi-step pipeline chains, or objects requiring initialization — subclass `BaseStressTester` directly:

```python
from grdl_te.benchmarking import BaseStressTester
from grdl.data_prep import Normalizer
from grdl.image_processing.filters import MeanFilter

class NormalizeThenFilterStressTester(BaseStressTester):
    """Stress test a normalise → filter pipeline chain."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._norm = Normalizer(method="zscore")
        self._filt = MeanFilter(kernel_size=3)

    @property
    def component_name(self) -> str:
        return "NormalizeThenFilter.pipeline"

    def call_once(self, payload):
        """Run normalize → filter as a single atomic call."""
        normalized = self._norm.normalize(payload)
        return self._filt.apply(normalized)

tester = NormalizeThenFilterStressTester(version="0.4.0")
record = tester.run(config)
```

The base class handles the entire ramp loop, event collection, failure detection, early abort on catastrophic failure (consecutive `MemoryError` / `OSError`), and `StressTestRecord` construction.  Subclasses only implement `component_name` and `call_once()`.

---

### 7.5 Failure Detection

The engine automatically detects failure points when `call_once()` raises any exception or times out.  `FailurePoint` objects are created at the first failure of each error type at each concurrency level:

```python
from grdl_te.benchmarking import ComponentStressTester, StressTestConfig

def fragile_fn(arr):
    import threading
    # Simulate a component with limited resource pool:
    # fails when too many threads call it simultaneously
    raise RuntimeError("resource pool exhausted")

tester = ComponentStressTester("fragile", fragile_fn)
record = tester.run(StressTestConfig(max_concurrency=8, ramp_steps=4))

print(f"Failed calls:      {record.summary.failed_calls}")
print(f"Saturation point:  {record.summary.saturation_concurrency} workers")
print(f"First failure:     {record.summary.first_failure_mode}")
print()
for fp in record.failure_points:
    print(f"  [{fp.error_type}] @ concurrency={fp.concurrency_level}")
    print(f"    Memory at failure: {fp.memory_bytes_at_failure / 1e6:.1f} MB")
    print(f"    Message:           {fp.error_message}")
```

**Early abort:** if 3 consecutive calls raise `MemoryError` or `OSError`, the ramp stops early to prevent OOM conditions.

---

### 7.6 Benchmark Linkage

Link a stress record to a `BenchmarkRecord` to connect "normal-load timing statistics" with "where it breaks under pressure".  Both record types can be saved to the same `GRDLStore` — no need to manage two separate store objects:

```python
from grdl_te.benchmarking import ComponentBenchmark, ComponentStressTester, GRDLStore

store = GRDLStore(base_dir=Path("./my_store"))

# Step 1: benchmark for per-call timing statistics
bench = ComponentBenchmark(
    name="MyFilter",
    fn=my_filter.apply,
    setup=lambda: ((image,), {}),
    iterations=10,
    warmup=2,
    store=store,   # GRDLStore handles BenchmarkRecord automatically
)
bench_record = bench.run()
print(f"Benchmark mean: {bench_record.total_wall_time.mean:.4f}s")

# Step 2: stress test, linked to the benchmark — same store
tester = ComponentStressTester(
    name="MyFilter",
    fn=my_filter.apply,
    related_benchmark_id=bench_record.benchmark_id,  # <-- link
    store=store,                                      # same GRDLStore
)
stress_record = tester.run(config)

# The stress report will include a cross-reference section pointing to
# bench_record.benchmark_id for detailed timing statistics.
print(f"Stress max_sustained: {stress_record.summary.max_sustained_concurrency}")
print(f"Linked benchmark:     {stress_record.related_benchmark_id}")
```

---

### 7.6b WorkflowStressTester (grdl-runtime Integration)

`WorkflowStressTester` wraps a grdl-runtime `Workflow` object directly — you do not write `call_once()` by hand.  The tester calls `workflow.execute(payload, prefer_gpu=...)` internally for each concurrent slot.

```python
from grdl_rt import Workflow
from grdl.image_processing.intensity import ToDecibels
from grdl.image_processing.filters import GaussianFilter
from grdl_te.benchmarking import WorkflowStressTester, StressTestConfig

# Build a Workflow in array mode (no reader required)
wf = (
    Workflow("SAR Preprocessing", version="0.1.0")
    .step(ToDecibels())
    .step(GaussianFilter(sigma=1.0))
)

# WorkflowStressTester handles call_once automatically
tester = WorkflowStressTester(
    wf,
    prefer_gpu=False,    # forwarded to workflow.execute()
    version="0.1.0",
    tags={"pipeline": "sar_preprocess"},
)

config = StressTestConfig(max_concurrency=8, ramp_steps=4)
record = tester.run(config)
print(f"Workflow: {record.component_name}")
print(f"Max sustained concurrency: {record.summary.max_sustained_concurrency}")
```

**Requirements:** `grdl_rt` must be installed.  The `Workflow` must be usable in *array mode* — i.e., `workflow.execute(array)` must succeed.  Workflows that require a file-mode reader cannot be stress tested with `WorkflowStressTester` unless you wrap them in `ComponentStressTester` with a custom `call_once` that loads the file separately.

---

### 7.7 Saving and Loading Records — GRDLStore

`GRDLStore` is the unified store for both `BenchmarkRecord` and `StressTestRecord`.  Both record types are saved under a single root directory — no need to manage two separate stores.

```python
from pathlib import Path
from grdl_te.benchmarking import GRDLStore

store = GRDLStore(base_dir=Path("./my_store"))

# Save either record type with the same call:
bench_id = store.save(bench_record)      # → <base_dir>/records/<benchmark_id>.json
stress_id = store.save(stress_record)    # → <base_dir>/stress/records/<stress_test_id>.json

# Load by type:
loaded_bench  = store.load_benchmark(bench_id)
loaded_stress = store.load_stress(stress_id)

# List with optional filtering:
benchmarks   = store.list_benchmarks(workflow_name="MyFilter", limit=20)
stress_tests = store.list_stress_tests(component_name="MyFilter", limit=20)

# Delete:
store.delete_benchmark(bench_id)
store.delete_stress(stress_id)

# Rebuild indexes (useful after copying or corrupting the index):
store.rebuild_benchmark_index()
store.rebuild_stress_index()
```

**Storage layout:**

```
<base_dir>/
    index.json                      ← benchmark index
    records/
        <benchmark_id>.json         ← BenchmarkRecord files
    stress/
        index.json                  ← stress test index
        records/
            <stress_test_id>.json   ← StressTestRecord files
```

This layout is backward-compatible with the previous `JSONBenchmarkStore` and `JSONStressTestStore` — existing on-disk data can be opened directly with `GRDLStore`.

**Unified report saving:**

`GRDLStore.save_report()` dispatches on record type automatically:

```python
# Saves the correct report format based on what record is passed:
store.save_report(bench_record,   Path("bench.txt"),  fmt="text")
store.save_report(stress_record,  Path("stress.md"),  fmt="markdown")
```

# List records with optional filtering
all_records = store.list_records()
for_component = store.list_records(component_name="MyComponent")

# Delete a record
store.delete(saved_id)
```

**Storage layout:**

```
my_store/
    stress/
        index.json              # lightweight index
        records/
            <uuid>.json         # one file per StressTestRecord
    records/                    # BenchmarkRecord files (from JSONBenchmarkStore)
    index.json
```

---

### 7.8 Reports

Stress test records produce their own separate reports — never merged with benchmark output:

```python
from grdl_te.benchmarking import (
    format_stress_report,
    format_stress_report_md,
    print_stress_report,
    save_stress_report,
    save_stress_report_md,
)

# Print to terminal
print_stress_report(record)

# Get as a string
text = format_stress_report(record)
md = format_stress_report_md(record)

# Save to directory (auto-named: stress_<id[:8]>.txt / .md)
save_stress_report(record, Path("./reports/"))
save_stress_report_md(record, Path("./reports/"))

# Save to explicit path
save_stress_report(record, Path("./my_stress_report.txt"))
save_stress_report_md(record, Path("./my_stress_report.md"))
```

The text report includes:

1. **Header** — component name, grdl version, run ID, timestamp
2. **Hardware** — hostname, CPUs, total memory, GPU availability
3. **Stress Configuration** — ramp parameters and computed concurrency levels
4. **Saturation Curve** — table of workers vs. error rate vs. p99 latency
5. **Failure Analysis** — each `FailurePoint` with error type, memory, and message
6. **Summary** — headline statistics (max sustained concurrency, saturation point, memory high-water mark)
7. **Cross-Reference** (optional) — link to the associated `BenchmarkRecord`

---

### 7.9 Cross-Run Comparison

`StressTestRecord` is designed to be "loadable" — two records produced from different grdl versions have identical JSON key structures and can be compared field-by-field.  The `schema_version` field (currently `1`) is incremented when the schema changes, and `from_dict()` handles forward-compatible loading:

```python
# Run A: baseline grdl version
record_a = tester.run(config)
Path("./record_a.json").write_text(record_a.to_json())

# Upgrade grdl, run again
record_b = tester.run(config)
Path("./record_b.json").write_text(record_b.to_json())

# Compare
from grdl_te.benchmarking import StressTestRecord

a = StressTestRecord.from_json(Path("./record_a.json").read_text())
b = StressTestRecord.from_json(Path("./record_b.json").read_text())

print(f"grdl version A:     {a.grdl_version}")
print(f"grdl version B:     {b.grdl_version}")
print(f"Max sustained A:    {a.summary.max_sustained_concurrency}")
print(f"Max sustained B:    {b.summary.max_sustained_concurrency}")
print(f"p99 latency A:      {a.summary.p99_latency_s:.4f}s")
print(f"p99 latency B:      {b.summary.p99_latency_s:.4f}s")
```

---

### 7.10 CLI Stress Testing

```bash
# Default config (ramps to 16 workers, medium payload, 10 s/step)
python -m grdl_te --stress-test

# Custom concurrency and step count
python -m grdl_te --stress-test --stress-concurrency 8 --stress-steps 4

# Short steps for quick smoke test
python -m grdl_te --stress-test --stress-duration 2.0 --size small

# Save reports to a directory
python -m grdl_te --stress-test --report ./reports/

# Persist records to a custom store
python -m grdl_te --stress-test --store-dir ./my_store
```

**Stress CLI options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--stress-test` | off | Enable stress test mode |
| `--stress-concurrency N` | 16 | Maximum worker count for the ramp |
| `--stress-steps N` | 5 | Number of ramp levels (geometric doubling) |
| `--stress-duration S` | 10.0 | Seconds to sustain each concurrency level |
| `--size` | medium | Payload preset (`small`, `medium`, `large`) |
| `--store-dir PATH` | `.benchmarks/` | Where to persist records |
| `--report [PATH]` | print | Print or save stress reports |

---

### 7.11 Complete Stress Test Example

See `stress_testing_example.py` at the repository root for a self-contained, runnable demonstration of all features:

```bash
python stress_testing_example.py
```

The script demonstrates all 11 scenarios in under 60 seconds and writes output (JSON records, text reports, Markdown reports) to `./stress_example_output/`.

---

## Quick Reference

| What you want to do                  | How                                                        |
|--------------------------------------|------------------------------------------------------------|
| Benchmark a single function          | `ComponentBenchmark(name, fn, ...).run()`                  |
| Benchmark a full workflow            | `ActiveBenchmarkRunner(wf, source, ...).run()`             |
| Analyze historical traces            | `PassiveBenchmarkRunner(traces, ...).run()`                |
| Load traces from JSON                | `ForensicTraceReader.from_json_file(path)`                 |
| Load traces from a directory         | `ForensicTraceReader.from_json_directory(dir)`             |
| Load traces from history DB          | `ForensicTraceReader.from_history_db(run_ids)`             |
| Save results to disk                 | `GRDLStore(path).save(record)`                             |
| Auto-save on run                     | Pass `store=` to any runner                                |
| Generate a text report               | `print_report(records)` or `save_report(records, path)`    |
| Generate a Markdown report           | `save_report_md(records, path)`                            |
| Compare records                      | `compare_records(records, labels=...)`                     |
| Run the full CLI suite               | `python -m grdl_te`                                        |
| CLI with report output               | `python -m grdl_te --report ./reports/`                    |
| **Stress test any callable**         | `ComponentStressTester(name, fn).run(config)`              |
| **Stress test a grdl-runtime Workflow** | `WorkflowStressTester(wf).run(config)`                  |
| **Stress test a custom pipeline**    | Subclass `BaseStressTester`, implement `call_once()`       |
| **Custom ramp config**               | `StressTestConfig(max_concurrency=16, ramp_steps=5, ...)`  |
| **Detect failure points**            | `record.failure_points` and `record.summary`               |
| **Save any record type**             | `GRDLStore(path).save(record)` — dispatches automatically  |
| **Load benchmark by ID**             | `GRDLStore(path).load_benchmark(benchmark_id)`             |
| **Load stress by ID**                | `GRDLStore(path).load_stress(stress_test_id)`              |
| **Generate stress report (text)**    | `print_stress_report(record)` / `save_stress_report(...)`  |
| **Generate stress report (md)**      | `save_stress_report_md(record, path)`                      |
| **Unified report dispatch**          | `GRDLStore(path).save_report(record, path, fmt="text")`    |
| **Cross-run comparison**             | `StressTestRecord.from_json(...)`, compare `.summary`      |
| **Link stress to benchmark**         | `ComponentStressTester(..., related_benchmark_id=id)`      |
| **CLI stress test**                  | `python -m grdl_te --stress-test`                          |
| **CLI stress with custom concurrency** | `python -m grdl_te --stress-test --stress-concurrency 8` |
