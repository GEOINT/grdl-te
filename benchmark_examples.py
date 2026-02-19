# -*- coding: utf-8 -*-
"""
Benchmark Example Workflows — stress-test grdl-runtime workflows at scale.

Copies workflow logic from grdl-runtime/examples/ (Sublook Compare,
Standard SAR Display, Multilook Preprocess), instruments each with
``ActiveBenchmarkRunner``, and runs at Small (512x512), Medium (2048x2048),
and Large (4096x4096) to surface per-step bottlenecks and scaling limits.

Results are persisted via ``JSONBenchmarkStore`` to ``.benchmarks/records/``.

Usage
-----
::

    conda activate grdl
    python benchmark_examples.py
    python benchmark_examples.py --iterations 5 --warmup 2
    python benchmark_examples.py --scales small medium
    python benchmark_examples.py --store-dir ./my_benchmarks

Dependencies
------------
grdl
grdl-runtime
grdl-te
numpy

Author
------
Claude Code (Anthropic)

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2026-02-17

Modified
--------
2026-02-17
"""

# Standard library
import argparse
import gc
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Third-party
import numpy as np

# GRDL processors
from grdl.image_processing.intensity import PercentileStretch, ToDecibels
from grdl.image_processing.sar import SublookDecomposition
from grdl.IO.models import SICDDirParam, SICDGrid, SICDMetadata

# grdl-runtime
from grdl_rt import Workflow

# grdl-te benchmarking
from grdl_te.benchmarking import (
    ActiveBenchmarkRunner,
    AggregatedMetrics,
    BenchmarkRecord,
    HardwareSnapshot,
    JSONBenchmarkStore,
    StepBenchmarkResult,
)

# ── Constants ────────────────────────────────────────────────────────────

SCALES: Dict[str, Tuple[int, int]] = {
    "Small": (512, 512),
    "Medium": (2048, 2048),
    "Large": (4096, 4096),
}

DEFAULT_ITERATIONS = 3
DEFAULT_WARMUP = 1


# ── Synthetic Data Generation ────────────────────────────────────────────


def generate_sar_complex64(rows: int, cols: int) -> np.ndarray:
    """Generate synthetic complex64 SAR SLC data.

    Parameters
    ----------
    rows : int
        Number of rows.
    cols : int
        Number of columns.

    Returns
    -------
    np.ndarray
        Complex64 array of shape ``(rows, cols)``.
    """
    rng = np.random.default_rng(42)
    real = rng.standard_normal((rows, cols), dtype=np.float32)
    imag = rng.standard_normal((rows, cols), dtype=np.float32)
    return (real + 1j * imag).astype(np.complex64)


def make_sicd_metadata(rows: int, cols: int) -> SICDMetadata:
    """Create minimal SICDMetadata for SublookDecomposition.

    Parameters match typical Umbra SICD collection parameters used
    in the grdl-runtime example workflows.

    Parameters
    ----------
    rows : int
        Image row count (for metadata consistency).
    cols : int
        Image column count (for metadata consistency).

    Returns
    -------
    SICDMetadata
    """
    dir_param = SICDDirParam(
        ss=0.005,           # sample spacing (meters)
        imp_resp_bw=100.0,  # impulse response bandwidth (cycles/meter)
        k_ctr=0.0,          # center spatial frequency
    )
    grid = SICDGrid(
        row=SICDDirParam(ss=0.005, imp_resp_bw=100.0),
        col=dir_param,
    )
    return SICDMetadata(
        format='SICD',
        rows=rows,
        cols=cols,
        dtype='complex64',
        grid=grid,
    )


# ── Inline Callable (from run_cfar_on_multilook.py) ──────────────────────


def incoherent_multilook(stack: np.ndarray) -> np.ndarray:
    """Average sub-look powers to produce a multilook image.

    Copied from grdl-runtime/examples/run_cfar_on_multilook.py.

    Parameters
    ----------
    stack : np.ndarray
        Complex sub-look stack, shape ``(num_looks, rows, cols)``.

    Returns
    -------
    np.ndarray
        Real-valued multilook power image, shape ``(rows, cols)``.
    """
    return np.mean(np.abs(stack) ** 2, axis=0)


# ── Multi-Scale Benchmark Wrapper ────────────────────────────────────────


class ScaledWorkflowBenchmark:
    """Orchestrate ActiveBenchmarkRunner across multiple input data scales.

    ``ActiveBenchmarkRunner`` runs a single Workflow N times with a fixed
    input.  This wrapper generates appropriately-sized synthetic data for
    each target scale and runs a full benchmark at each, collecting
    records for cross-scale comparison.

    Parameters
    ----------
    name : str
        Display name for this benchmark group.
    workflow : Workflow
        The grdl-runtime Workflow to benchmark.
    data_fn : callable
        ``(rows, cols) -> np.ndarray`` factory for synthetic input data.
    metadata_fn : callable, optional
        ``(rows, cols) -> metadata`` factory.  Forwarded to
        ``Workflow.execute(metadata=...)``.
    iterations : int
        Measurement iterations per scale.
    warmup : int
        Warmup iterations per scale (discarded).
    store : JSONBenchmarkStore, optional
        Persistence store for automatic saving.
    """

    def __init__(
        self,
        name: str,
        workflow: Any,
        data_fn: Callable[[int, int], np.ndarray],
        metadata_fn: Optional[Callable[[int, int], Any]] = None,
        iterations: int = DEFAULT_ITERATIONS,
        warmup: int = DEFAULT_WARMUP,
        store: Optional[JSONBenchmarkStore] = None,
    ) -> None:
        self._name = name
        self._workflow = workflow
        self._data_fn = data_fn
        self._metadata_fn = metadata_fn
        self._iterations = iterations
        self._warmup = warmup
        self._store = store

    def run_scale(
        self,
        scale_name: str,
        rows: int,
        cols: int,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BenchmarkRecord:
        """Run benchmark at a single data scale.

        Parameters
        ----------
        scale_name : str
            Human-readable scale label (e.g. ``"Small"``).
        rows : int
            Array row dimension.
        cols : int
            Array column dimension.
        progress_callback : callable, optional
            Called with ``(current, total)`` after each measurement
            iteration.

        Returns
        -------
        BenchmarkRecord
        """
        data = self._data_fn(rows, cols)

        execute_kwargs: Dict[str, Any] = {}
        if self._metadata_fn is not None:
            execute_kwargs["metadata"] = self._metadata_fn(rows, cols)

        runner = ActiveBenchmarkRunner(
            workflow=self._workflow,
            iterations=self._iterations,
            warmup=self._warmup,
            store=self._store,
            tags={
                "scale": scale_name,
                "dimensions": f"{rows}x{cols}",
                "data_type": "synthetic",
                "workflow": self._name,
            },
        )

        record = runner.run(
            source=data,
            prefer_gpu=False,
            progress_callback=progress_callback,
            **execute_kwargs,
        )

        del data
        gc.collect()

        return record


# ── Workflow Definitions (copied from grdl-runtime/examples/) ────────────


def build_example_workflows() -> Dict[str, Dict[str, Any]]:
    """Build Workflow objects with logic copied from grdl-runtime/examples/.

    These are direct translations of the example scripts, adapted for
    array-mode execution (no file reader or chip strategy) so they can
    run on synthetic data of arbitrary dimensions.

    Returns
    -------
    Dict[str, Dict]
        Mapping of workflow name to configuration dict with keys:
        ``workflow``, ``data_fn``, ``metadata_fn``.
    """
    workflows: Dict[str, Dict[str, Any]] = {}

    # ── Workflow 1: Sublook Compare ──────────────────────────────────
    # Source: grdl-runtime/examples/sublook_compare_workflow.py
    #
    # Original pipeline (file mode):
    #   .reader(SICDReader)
    #   .chip("center", size=5000)
    #   .step(SublookDecomposition, num_looks=3, dimension='azimuth',
    #         overlap=0.0)
    #   .step(ToDecibels)
    #   .step(PercentileStretch, plow=2.0, phigh=98.0)
    #
    # Adapted: removed reader/chip for array mode; metadata supplied
    # externally via execute(metadata=...).
    workflows["Sublook Compare"] = {
        "workflow": (
            Workflow("Sublook Compare", version="1.0.0", modalities=["SAR"])
            .step(SublookDecomposition, num_looks=3,
                  dimension='azimuth', overlap=0.0)
            .step(ToDecibels)
            .step(PercentileStretch, plow=2.0, phigh=98.0)
        ),
        "data_fn": generate_sar_complex64,
        "metadata_fn": make_sicd_metadata,
    }

    # ── Workflow 2: Standard SAR Display ─────────────────────────────
    # Source: grdl-runtime/examples/run_csi_compare.py (Standard SAR)
    #
    # Original pipeline (file mode):
    #   .reader(SICDReader)
    #   .chip("center", size=4096)
    #   .step(ToDecibels, floor_db=-60.0)
    #   .step(PercentileStretch, plow=2.0, phigh=98.0)
    #
    # Adapted: removed reader/chip for array mode; no metadata needed.
    workflows["Standard SAR Display"] = {
        "workflow": (
            Workflow("Standard SAR Display", version="1.0.0",
                     modalities=["SAR"])
            .step(ToDecibels, floor_db=-60.0)
            .step(PercentileStretch, plow=2.0, phigh=98.0)
        ),
        "data_fn": generate_sar_complex64,
        "metadata_fn": None,
    }

    # ── Workflow 3: Multilook Preprocess ─────────────────────────────
    # Source: grdl-runtime/examples/run_cfar_on_multilook.py
    #
    # Original pipeline (file mode):
    #   .reader(SICDReader)
    #   .chip("center", size=4096)
    #   .step(SublookDecomposition, num_looks=9, dimension='azimuth',
    #         overlap=0.3, deweight=True)
    #   .step(incoherent_multilook, name="IncoherentMultilook")
    #
    # Adapted: removed reader/chip for array mode; metadata supplied
    # externally.  incoherent_multilook() copied inline above.
    workflows["Multilook Preprocess"] = {
        "workflow": (
            Workflow("Multilook Preprocess", version="1.0.0",
                     modalities=["SAR"])
            .step(SublookDecomposition, num_looks=9,
                  dimension='azimuth', overlap=0.3, deweight=True)
            .step(incoherent_multilook, name="IncoherentMultilook")
        ),
        "data_fn": generate_sar_complex64,
        "metadata_fn": make_sicd_metadata,
    }

    return workflows


# ── Reporting ────────────────────────────────────────────────────────────


def print_step_breakdown(record: BenchmarkRecord) -> None:
    """Print per-step component analysis table.

    Parameters
    ----------
    record : BenchmarkRecord
        Completed benchmark record with step results.
    """
    # Sum step times for percentage denominator (excludes inter-step
    # overhead like tracemalloc snapshots that inflates pipeline total).
    step_wall_sum = sum(s.wall_time_s.mean for s in record.step_results)

    header = (
        f"    {'Step Name':<28} "
        f"{'Mean Time':>12} "
        f"{'Memory Peak':>14} "
        f"{'% of Total':>12}"
    )
    print(header)
    print(f"    {'-' * 68}")

    for step in record.step_results:
        pct = (
            (step.wall_time_s.mean / step_wall_sum * 100)
            if step_wall_sum > 0
            else 0.0
        )
        mem_mb = step.peak_rss_bytes.mean / (1024 * 1024)
        print(
            f"    {step.processor_name:<28} "
            f"{step.wall_time_s.mean:>10.4f} s "
            f"{mem_mb:>10.2f} MB "
            f"{pct:>10.1f} %"
        )

    total_mem_mb = record.total_peak_rss.mean / (1024 * 1024)
    print(f"    {'-' * 68}")
    print(
        f"    {'TOTAL':<28} "
        f"{step_wall_sum:>10.4f} s "
        f"{total_mem_mb:>10.2f} MB "
        f"{'100.0':>10} %"
    )


def print_scaling_comparison(
    all_results: Dict[str, Dict[str, BenchmarkRecord]],
    scales: Dict[str, Tuple[int, int]],
) -> None:
    """Print throughput scaling comparison across data sizes.

    Parameters
    ----------
    all_results : Dict[str, Dict[str, BenchmarkRecord]]
        Workflow name -> scale name -> benchmark record.
    scales : Dict[str, Tuple[int, int]]
        Scale definitions.
    """
    print(f"\n{'=' * 78}")
    print("SCALING COMPARISON — Throughput (pixels/sec)")
    print(f"{'=' * 78}")

    for wf_name, scale_records in all_results.items():
        print(f"\n  {wf_name}")
        print(
            f"  {'Scale':<10} "
            f"{'Dimensions':>14} "
            f"{'Mean Wall (s)':>14} "
            f"{'Throughput (px/s)':>20}"
        )
        print(f"  {'-' * 60}")

        for scale_name in scales:
            if scale_name not in scale_records:
                continue
            record = scale_records[scale_name]
            rows, cols = scales[scale_name]
            pixels = rows * cols
            mean_time = record.total_wall_time.mean
            throughput = (
                pixels / mean_time if mean_time > 0 else float('inf')
            )
            print(
                f"  {scale_name:<10} "
                f"{f'{rows}x{cols}':>14} "
                f"{mean_time:>14.4f} "
                f"{throughput:>18,.0f}"
            )


def print_bottleneck_analysis(
    all_results: Dict[str, Dict[str, BenchmarkRecord]],
    scales: Dict[str, Tuple[int, int]],
) -> None:
    """Identify per-step bottlenecks and compute cross-scale throughput stats.

    For each workflow step, collects per-scale throughput values and
    aggregates them with ``AggregatedMetrics.from_values()`` to surface
    which steps degrade most as data grows.  The step with the highest
    share of wall time at the largest scale is flagged as the primary
    bottleneck.

    Parameters
    ----------
    all_results : Dict[str, Dict[str, BenchmarkRecord]]
        Workflow name -> scale name -> benchmark record.
    scales : Dict[str, Tuple[int, int]]
        Scale definitions.
    """
    print(f"\n{'=' * 78}")
    print("BOTTLENECK ANALYSIS — Per-Step Scaling")
    print(f"{'=' * 78}")

    scale_names = list(scales.keys())

    for wf_name, scale_records in all_results.items():
        if not scale_records:
            continue

        # Use the first record to enumerate steps
        first_record: BenchmarkRecord = next(iter(scale_records.values()))
        steps: List[StepBenchmarkResult] = first_record.step_results

        # Identify bottleneck at the largest available scale
        largest_scale = scale_names[-1]
        for name in reversed(scale_names):
            if name in scale_records:
                largest_scale = name
                break
        largest_record = scale_records[largest_scale]
        wall_bottleneck: StepBenchmarkResult = max(
            largest_record.step_results,
            key=lambda s: s.wall_time_s.mean,
        )
        mem_bottleneck: StepBenchmarkResult = max(
            largest_record.step_results,
            key=lambda s: s.peak_rss_bytes.mean,
        )

        print(f"\n  {wf_name}")
        print(
            f"  {'Step':<28} "
            f"{'Min px/s':>14} "
            f"{'Max px/s':>14} "
            f"{'Stddev px/s':>14} "
            f"{'Flag':>6}"
        )
        print(f"  {'-' * 78}")

        for step in steps:
            # Collect this step's throughput at each scale
            throughputs: List[float] = []
            for sn in scale_names:
                if sn not in scale_records:
                    continue
                rows, cols = scales[sn]
                pixels = rows * cols
                record = scale_records[sn]
                for sr in record.step_results:
                    if sr.step_index == step.step_index:
                        tp = (
                            pixels / sr.wall_time_s.mean
                            if sr.wall_time_s.mean > 0
                            else 0.0
                        )
                        throughputs.append(tp)

            if not throughputs:
                continue

            agg: AggregatedMetrics = AggregatedMetrics.from_values(throughputs)

            flag = ""
            if step.processor_name == wall_bottleneck.processor_name:
                flag = "SLOW"
            if step.processor_name == mem_bottleneck.processor_name:
                flag = "MEM" if not flag else "SLOW+MEM"

            print(
                f"  {step.processor_name:<28} "
                f"{agg.min:>14,.0f} "
                f"{agg.max:>14,.0f} "
                f"{agg.stddev:>14,.0f} "
                f"{flag:>6}"
            )

        # Summary line
        wall_pct = (
            wall_bottleneck.wall_time_s.mean
            / largest_record.total_wall_time.mean
            * 100
            if largest_record.total_wall_time.mean > 0
            else 0.0
        )
        mem_mb = mem_bottleneck.peak_rss_bytes.mean / (1024 * 1024)
        print(
            f"\n  Wall-time bottleneck: {wall_bottleneck.processor_name} "
            f"({wall_pct:.1f}% at {largest_scale})"
        )
        print(
            f"  Memory bottleneck:   {mem_bottleneck.processor_name} "
            f"({mem_mb:.1f} MB peak at {largest_scale})"
        )


# ── Main ─────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark grdl-runtime example workflows at multiple "
                    "data scales.",
    )
    parser.add_argument(
        "--iterations", "-n", type=int, default=DEFAULT_ITERATIONS,
        help=f"Measurement iterations per scale (default: {DEFAULT_ITERATIONS}).",
    )
    parser.add_argument(
        "--warmup", "-w", type=int, default=DEFAULT_WARMUP,
        help=f"Warmup iterations (discarded) per scale (default: {DEFAULT_WARMUP}).",
    )
    parser.add_argument(
        "--scales", nargs="+", choices=["small", "medium", "large"],
        default=None,
        help="Run only the specified scales (default: all).",
    )
    parser.add_argument(
        "--store-dir", type=Path, default=None,
        help="Benchmark store directory (default: .benchmarks/).",
    )
    return parser.parse_args()


def main() -> None:
    """Run benchmarks for all example workflows at all scales."""
    args = parse_args()

    # Resolve scales
    if args.scales:
        active_scales = {
            k: v for k, v in SCALES.items()
            if k.lower() in args.scales
        }
    else:
        active_scales = SCALES

    iterations = args.iterations
    warmup = args.warmup

    # Banner
    print("=" * 78)
    print("GRDL Workflow Benchmark Suite")
    print("=" * 78)
    scale_strs = ", ".join(
        f"{k} ({v[0]}x{v[1]})" for k, v in active_scales.items()
    )
    print(f"Scales:     {scale_strs}")
    print(f"Iterations: {iterations} (+ {warmup} warmup)")

    store = JSONBenchmarkStore(base_dir=args.store_dir)
    hw = HardwareSnapshot.capture()
    mem_gb = hw.total_memory_bytes / (1024 ** 3)
    gpu_str = "yes" if hw.gpu_available else "no"
    print(f"Hardware:   {hw.cpu_count} CPUs, {mem_gb:.1f} GB RAM, GPU={gpu_str}")

    workflows = build_example_workflows()
    all_results: Dict[str, Dict[str, BenchmarkRecord]] = {}

    for wf_name, wf_config in workflows.items():
        print(f"\n{'=' * 78}")
        print(f"WORKFLOW: {wf_name}")
        print(f"{'=' * 78}")

        bench = ScaledWorkflowBenchmark(
            name=wf_name,
            workflow=wf_config["workflow"],
            data_fn=wf_config["data_fn"],
            metadata_fn=wf_config.get("metadata_fn"),
            iterations=iterations,
            warmup=warmup,
            store=store,
        )

        scale_records: Dict[str, BenchmarkRecord] = {}

        for scale_name, (rows, cols) in active_scales.items():
            print(f"\n  [{scale_name}] {rows}x{cols}")
            print(f"    Generating synthetic complex64 data...")

            def _progress(current: int, total: int) -> None:
                print(f"    Iteration {current}/{total}")

            record = bench.run_scale(
                scale_name, rows, cols,
                progress_callback=_progress,
            )

            scale_records[scale_name] = record

            # Component analysis
            print(f"\n    Component Analysis:")
            print_step_breakdown(record)
            print(f"\n    Record: {record.benchmark_id}")

        all_results[wf_name] = scale_records

    # Cross-scale throughput comparison
    print_scaling_comparison(all_results, active_scales)

    # Per-step bottleneck analysis
    print_bottleneck_analysis(all_results, active_scales)

    # Final summary
    store_path = (
        args.store_dir if args.store_dir else Path.cwd() / ".benchmarks"
    )
    records = store.list_records(limit=1000)
    print(f"\n{'=' * 78}")
    print("BENCHMARK COMPLETE")
    print(f"{'=' * 78}")
    print(f"Store:   {store_path}")
    print(f"Records: {len(records)} total in store")
    print(f"Files:   {store_path / 'records/'}")


if __name__ == "__main__":
    main()
