# GRDL Benchmark Report Possibilities

This document describes every report variation the benchmarking suite can
produce, keyed to the type of benchmark you run. Both output formats — plain
text (`format_report` / `save_report`) and Markdown (`format_report_md` /
`save_report_md`) — emit equivalent information. Examples below use the
Markdown format for readability.

---

## Report anatomy

Every report has four fixed top-level sections:

| Section | Contents |
|---------|----------|
| **Executive Summary** | Topology summary, hardware one-liner, top-5 bottleneck table |
| **Hardware & Configuration** | CPU count, RAM, GPU presence, Python, hostname, capture timestamp, iteration count |
| **Detailed Results** | One subsection per benchmark, sorted slowest-first |
| **Overall Summary** | Aggregate view; single-record shows raw metrics, multi-record shows fastest/slowest/memory extremes |

A **Comparison** section appears between Detailed Results and Overall Summary
only when two or more records are included.

---

## Dimension table

The specific content of each section depends on three axes:

| Axis | Values |
|------|--------|
| **Benchmark type** | `component`, `active` (sequential / parallel / mixed), `passive` |
| **Workflow topology** | `component`, `sequential`, `parallel`, `mixed` |
| **Iteration count** | `N = 1` (single-shot) or `N > 1` (repeated) |

These axes combine into the variations below.

---

## Variation 1 — Single component benchmark, N = 1

**Trigger**: `benchmark_type="component"`, one step, `iterations=1`.

Typical use: microbenchmark a single callable in isolation (a filter, a
normaliser, a chip extractor).

**Step table** (N = 1 — scalar values only):

| # | Step | Wall Time | Latency% | Memory% | Path |
|---|------|-----------|----------|---------|------|
| 0 | `ChipExtractor` | 0.0312s | 100.0% | 100.0% | sequential |

**Executive Summary** bottleneck block points to the only step.

**Overall Summary** (N = 1):

- **Wall Time**: 0.0312s
- **CPU Time**: 0.0280s
- **Peak Memory**: 4.2 MB

No Time Decomposition, no Branch Analysis, no Memory Profile (unless
`peak_overhead_bytes` was instrumented), no Comparison.

---

## Variation 2 — Single component benchmark, N > 1

**Trigger**: `benchmark_type="component"`, one step, `iterations ≥ 2`.

**Step table** (N > 1 — full statistics):

| # | Step | Mean | Median | StdDev | Min | Max | Latency% | Memory% | Path |
|---|------|------|--------|--------|-----|-----|----------|---------|------|
| 0 | `ChipExtractor` | 0.0318s | 0.0314s | 0.0021s | 0.0298s | 0.0351s | 100.0% | 100.0% | sequential |

**Overall Summary** (N > 1):

- **Wall Time**: 0.0318s mean | 0.0298s min | 0.0351s max | 0.0021s stddev
- **CPU Time**: 0.0284s mean | 0.0271s min | 0.0309s max | 0.0018s stddev
- **Peak Memory**: 4.2 MB mean | 4.0 MB min | 4.5 MB max

No Time Decomposition, no Branch Analysis, no Comparison.

---

## Variation 3 — Sequential multi-step workflow, N = 1

**Trigger**: `benchmark_type="active"` or `"passive"`, multiple non-concurrent
steps, `iterations=1`.

Typical use: profile a full GRDL processing pipeline run once against a single
input.

**Step table** (N = 1):

| # | Step | Wall Time | Latency% | Memory% | Path |
|---|------|-----------|----------|---------|------|
| 0 | `GeoTIFFReader` | 0.4210s | 41.8% | 23.0% | sequential |
| 1 | `ChipExtractor` | 0.3100s | 30.8% | 11.0% | sequential |
| 2 | `Normalizer` | 0.2750s | 27.3% | 66.0% | sequential |

No step-count summary line (nothing skipped, no parallel branches).

**Overall Summary** (N = 1):

- **Wall Time**: 1.0060s
- **CPU Time**: 0.8940s
- **Peak Memory**: 182.4 MB

No Time Decomposition, no Branch Analysis. Memory Profile section appears when
`peak_overhead_bytes` is present on any step (see Variation 7).

---

## Variation 4 — Sequential multi-step workflow, N > 1

**Trigger**: `benchmark_type="active"` or `"passive"`, multiple non-concurrent
steps, `iterations ≥ 2`.

**Step table** (N > 1):

| # | Step | Mean | Median | StdDev | Min | Max | Latency% | Memory% | Path |
|---|------|------|--------|--------|-----|-----|----------|---------|------|
| 0 | `GeoTIFFReader` | 0.4210s | 0.4190s | 0.0085s | 0.4100s | 0.4340s | 41.8% | 23.0% | sequential |
| 1 | `ChipExtractor` | 0.3100s | 0.3080s | 0.0062s | 0.3010s | 0.3190s | 30.8% | 11.0% | sequential |
| 2 | `Normalizer` | 0.2750s | 0.2730s | 0.0049s | 0.2690s | 0.2840s | 27.3% | 66.0% | sequential |

**Overall Summary** (N > 1):

- **Wall Time**: 1.0060s mean | 0.9840s min | 1.0290s max | 0.0121s stddev
- **CPU Time**: 0.8940s mean | 0.8770s min | 0.9120s max | 0.0109s stddev
- **Peak Memory**: 182.4 MB mean | 179.2 MB min | 186.1 MB max

---

## Variation 5 — Sequential workflow with skipped steps

**Trigger**: One or more steps have `wall_time_s.max == 0 AND cpu_time_s.max == 0`
(condition guard not met at runtime).

Step-count summary line appears:

*2 ran, 1 skipped / 3 total*

Skipped rows render as `--` across all metric columns:

| # | Step | Wall Time | Latency% | Memory% | Path |
|---|------|-----------|----------|---------|------|
| 0 | `GeoTIFFReader` | 0.4210s | 57.6% | 35.0% | sequential |
| 1 | `NITFReader` | -- | -- | -- | *skipped* |
| 2 | `Normalizer` | 0.3100s | 42.4% | 65.0% | sequential |

Skipped steps do not contribute to Latency% or Memory% totals. The executive
summary bottleneck table excludes skipped steps.

---

## Variation 6 — Parallel (all-concurrent) workflow

**Trigger**: All steps have `concurrent=True`; topology classified as
`WorkflowTopology.PARALLEL`.

Step-count summary line appears:

*3 ran, 3 parallel / 3 total*

**Step table** — critical path step shows full Latency%, non-critical steps
show `--` in Latency% (their wall time is hidden by the critical path and
cannot be meaningfully attributed). All concurrent Memory% values carry a `†`
footnote. All concurrent Mean values carry a `‡` marker.

| # | Step | Wall Time | Latency% | Memory% | Path |
|---|------|-----------|----------|---------|------|
| 0 | `BranchA` | 5.0000s ‡ | 100.0% | 50.0%† | critical |
| 1 | `BranchB` | 3.0000s ‡ | -- | 30.0%† | parallel |
| 2 | `BranchC` | 2.0000s ‡ | -- | 20.0%† | parallel |

*† Memory shared across concurrent steps (tracemalloc is process-wide)*
*‡ Wall time measured under resource contention — not comparable to isolated standalone execution time.*

**Time Decomposition** (appears for parallel/mixed topologies):

| Metric | Value |
|--------|-------|
| Wall Clock (actual elapsed) | 5.0000s |
| Critical Path (longest chain) | 5.0000s |
| Contended Step Sum ‡ | 10.0000s |

*‡ Step times were measured under resource contention — not a valid sequential baseline.*

**Branch Analysis** (appears when `depends_on` graph produces ≥ 2 chains):

| Branch | Steps | Chain Time | Status |
|--------|-------|------------|--------|
| 1 | `BranchA` | 5.0000s | **critical path** |
| 2 | `BranchB` | 3.0000s | idle 2.0000s |
| 3 | `BranchC` | 2.0000s | idle 3.0000s |

---

## Variation 7 — Mixed (sequential + parallel) workflow

**Trigger**: Some steps sequential, some concurrent; topology classified as
`WorkflowTopology.MIXED`.

Example: `Reader → [BranchA ‖ BranchB] → Writer`

Step-count summary line:

*4 ran, 2 parallel / 4 total*

**Step table**:

| # | Step | Wall Time | Latency% | Memory% | Path |
|---|------|-----------|----------|---------|------|
| 0 | `Reader` | 1.0000s | 16.7% | 10.0% | sequential |
| 1 | `BranchA` | 5.0000s ‡ | 83.3% | 50.0%† | critical |
| 2 | `BranchB` | 3.0000s ‡ | -- | 30.0%† | parallel |
| 3 | `Writer` | 0.5000s | included in critical | 10.0% | sequential |

Time Decomposition and Branch Analysis sections appear as in Variation 6.

---

## Variation 8 — Workflow with memory overhead instrumentation

**Trigger**: One or more `StepBenchmarkResult` objects have
`peak_overhead_bytes` set (allocated by grdl-runtime's memory tracer).

A **Memory Profile** subsection appears inside the record detail block after
the step table:

| Step | Peak Overhead | End-of-Step Footprint | Memory% |
|------|-------------|----------------------|---------|
| `GeoTIFFReader` | 12.5 MB | 8.1 MB | 23.0% |
| `Normalizer` | 45.2 MB | N/A | 66.0% |
| **Overall Workflow Peak** | **182.4 MB** | *(high-water mark)* | |

- **Peak Overhead** — transient allocation spike above the pre-step RSS
  baseline (the per-step cost of the operation itself).
- **End-of-Step Footprint** — RSS at the end of the step (data retained after
  the step completes). `N/A` when not instrumented.
- `(concurrent)` annotation on Memory% for steps that ran in parallel.

When `peak_overhead_bytes` is absent on all steps the entire Memory Profile
section is omitted.

---

## Variation 9 — Multi-record comparison (any topology mix)

**Trigger**: Two or more `BenchmarkRecord` objects passed to `format_report_md`
or `format_report`.

Everything from Variations 1–8 applies to each record's individual detail
block. Additionally:

**Comparison section** (between Detailed Results and Overall Summary):

### Workflow Summary

| Workflow | Topology | Wall Time | CPU Time | Peak Memory | Steps |
|----------|----------|-----------|----------|-------------|-------|
| Sequential | sequential | 10.0000s | 8.9000s | 182.4 MB | 3 |
| Optimized | sequential | 5.0000s | 4.5000s | 161.0 MB | 3 |

**Overall Summary** (multi-record):

- **Total Benchmarks**: 2
- **Total Wall Time**: 15.0000s
- **Fastest**: Optimized (5.0000s)
- **Slowest**: Sequential (10.0000s)
- **Least Memory**: Optimized (161.0 MB)
- **Most Memory**: Sequential (182.4 MB)

Records in the Detailed Results block are sorted slowest-first regardless of
the order they were passed in.

---

## What never appears in any variation

The following were removed because they are either statistically invalid at
typical iteration counts or actively misleading:

| Removed | Reason |
|---------|--------|
| **Parallelism Ratio** (`sum_of_steps / wall_clock`) | Step times are measured under resource contention — the ratio overstates speedup by using inflated numerator values |
| **P95 per step** | `np.percentile(arr, 95)` with N ≤ 10 is mathematically the max with a misleading tail-latency label |
| **GPU Memory row** | `gpu_memory_bytes` on `StepBenchmarkResult` is not reliably populated by grdl-runtime; displaying zeros or stale values would mislead |
| **Module Summary** | Required a `module` tag that is not standardised across workflows; replaced by the Workflow Comparison section which works without tags |
| **Statistics at N = 1** | Mean/median/stddev/min/max are all identical at N = 1; only the scalar value is shown |

---

## Annotation legend

| Symbol | Meaning |
|--------|---------|
| `‡` on a Mean value | Wall time measured under resource contention — not a standalone execution time |
| `†` on a Memory% value | Memory shared across concurrent steps; tracemalloc is process-wide |
| `--` in Latency% | Step is non-critical in a parallel DAG — its wall time is hidden by the critical path and cannot be meaningfully attributed |
| *skipped* in Path | Step was not executed (condition guard not met) |
