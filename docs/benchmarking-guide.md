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

## Quick Reference

| What you want to do                  | How                                                        |
|--------------------------------------|------------------------------------------------------------|
| Benchmark a single function          | `ComponentBenchmark(name, fn, ...).run()`                  |
| Benchmark a full workflow            | `ActiveBenchmarkRunner(wf, source, ...).run()`             |
| Analyze historical traces            | `PassiveBenchmarkRunner(traces, ...).run()`                |
| Load traces from JSON                | `ForensicTraceReader.from_json_file(path)`                 |
| Load traces from a directory         | `ForensicTraceReader.from_json_directory(dir)`             |
| Load traces from history DB          | `ForensicTraceReader.from_history_db(run_ids)`             |
| Save results to disk                 | `JSONBenchmarkStore(path).save(record)`                    |
| Auto-save on run                     | Pass `store=` to any runner                                |
| Generate a text report               | `print_report(records)` or `save_report(records, path)`    |
| Generate a Markdown report           | `save_report_md(records, path)`                            |
| Compare records                      | `compare_records(records, labels=...)`                     |
| Run the full CLI suite               | `python -m grdl_te`                                        |
| CLI with report output               | `python -m grdl_te --report ./reports/`                    |
