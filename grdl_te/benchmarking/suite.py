# -*- coding: utf-8 -*-
"""
Component Benchmark Suite — pre-built benchmarks for every GRDL function.

Defines ``ComponentBenchmark`` instances for all public GRDL modules:
IO readers/writers, image processing filters, intensity transforms,
polarimetric decomposition, pipelines, data preparation, geolocation
transforms, coregistration, orthorectification, and SAR processing.

Also runs the comprehensive YAML workflow through
``ActiveBenchmarkRunner`` when grdl-runtime is available.

Dependencies
------------
grdl
numpy
scipy

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
import contextlib
import gc
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Third-party
import numpy as np

# Internal
from grdl_te.benchmarking.component import ComponentBenchmark
from grdl_te.benchmarking.store import JSONBenchmarkStore

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
DEFAULT_ITERATIONS = 10
DEFAULT_WARMUP = 2
DEFAULT_SIZE = "medium"

from grdl_te.benchmarking.source import ARRAY_SIZES

YAML_PATH = Path(__file__).resolve().parents[2] / "workflows" / (
    "comprehensive_benchmark_workflow.yaml"
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _bench(
    name: str,
    fn: Callable[..., Any],
    setup: Optional[Callable[[], Tuple[tuple, dict]]] = None,
    *,
    store: JSONBenchmarkStore,
    iterations: int,
    warmup: int,
    tags: Dict[str, str],
    module: str,
    version: str = "0.0.0",
) -> Optional[Any]:
    """Create and run a single ComponentBenchmark.

    Returns the ``BenchmarkRecord`` on success, ``None`` on failure.
    """
    bench = ComponentBenchmark(
        name=name,
        fn=fn,
        setup=setup,
        iterations=iterations,
        warmup=warmup,
        store=store,
        tags={**tags, "module": module},
        version=version,
    )
    try:
        with _suppress_stdout():
            record = bench.run()
    except Exception as exc:
        print(f"  SKIP  {name:<55s}  {exc}")
        return None

    wall = record.total_wall_time
    mem = record.total_peak_rss
    print(
        f"  OK    {name:<55s}  "
        f"wall={wall.mean:.4f}s (p95={wall.p95:.4f}s)  "
        f"mem={mem.mean / 1024:.0f}KB"
    )
    gc.collect()
    return record


def _section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


@contextlib.contextmanager
def _suppress_stderr():
    """Temporarily silence C-level stderr (e.g. GDAL TXTFMT warnings)."""
    stderr_fd = 2
    old_fd = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, stderr_fd)
        os.close(devnull)
        yield
    finally:
        os.dup2(old_fd, stderr_fd)
        os.close(old_fd)


@contextlib.contextmanager
def _suppress_stdout():
    """Temporarily silence stdout (e.g. verbose image formation prints)."""
    stdout_fd = 1
    old_fd = os.dup(stdout_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, stdout_fd)
        os.close(devnull)
        yield
    finally:
        os.dup2(old_fd, stdout_fd)
        os.close(old_fd)


# ---------------------------------------------------------------------------
# Filter benchmarks
# ---------------------------------------------------------------------------
def run_filter_benchmarks(
    store: JSONBenchmarkStore,
    rows: int,
    cols: int,
    iterations: int,
    warmup: int,
    tags: Dict[str, str],
) -> List:
    """Benchmark all image processing filters.

    Parameters
    ----------
    store : JSONBenchmarkStore
        Persistence backend.
    rows, cols : int
        Array dimensions.
    iterations, warmup : int
        Benchmark repetition counts.
    tags : Dict[str, str]
        User-defined labels applied to every record.

    Returns
    -------
    List[BenchmarkRecord]
    """
    from grdl.image_processing.filters import (
        ComplexLeeFilter,
        GaussianFilter,
        LeeFilter,
        MaxFilter,
        MeanFilter,
        MedianFilter,
        MinFilter,
        PhaseGradientFilter,
        StdDevFilter,
    )

    _section("Image Processing — Filters")
    sz = tags["array_size"]
    results = []
    mod = "image_processing.filters"

    real_img = np.random.rand(rows, cols).astype(np.float32)
    complex_img = (
        np.random.rand(rows, cols) + 1j * np.random.rand(rows, cols)
    ).astype(np.complex64)

    kw = dict(store=store, iterations=iterations, warmup=warmup,
              tags=tags, module=mod)

    # --- Linear filters ---
    for ks in (3, 5, 7):
        filt = MeanFilter(kernel_size=ks)
        r = _bench(f"MeanFilter.apply.k{ks}.{sz}", filt.apply,
                    setup=lambda: ((real_img,), {}), version="1.0.0", **kw)
        if r:
            results.append(r)

    for sigma in (0.5, 1.0, 2.0):
        filt = GaussianFilter(sigma=sigma)
        r = _bench(f"GaussianFilter.apply.s{sigma}.{sz}", filt.apply,
                    setup=lambda: ((real_img,), {}), version="1.0.0", **kw)
        if r:
            results.append(r)

    # --- Rank filters ---
    for ks in (3, 5):
        filt = MedianFilter(kernel_size=ks)
        r = _bench(f"MedianFilter.apply.k{ks}.{sz}", filt.apply,
                    setup=lambda: ((real_img,), {}), version="1.0.0", **kw)
        if r:
            results.append(r)

    filt = MinFilter(kernel_size=3)
    r = _bench(f"MinFilter.apply.k3.{sz}", filt.apply,
                setup=lambda: ((real_img,), {}), version="1.0.0", **kw)
    if r:
        results.append(r)

    filt = MaxFilter(kernel_size=3)
    r = _bench(f"MaxFilter.apply.k3.{sz}", filt.apply,
                setup=lambda: ((real_img,), {}), version="1.0.0", **kw)
    if r:
        results.append(r)

    # --- Statistical ---
    filt = StdDevFilter(kernel_size=5)
    r = _bench(f"StdDevFilter.apply.k5.{sz}", filt.apply,
                setup=lambda: ((real_img,), {}), version="1.0.0", **kw)
    if r:
        results.append(r)

    # --- Speckle filters ---
    for ks in (5, 7):
        filt = LeeFilter(kernel_size=ks)
        r = _bench(f"LeeFilter.apply.k{ks}.{sz}", filt.apply,
                    setup=lambda: ((real_img,), {}), version="1.0.0", **kw)
        if r:
            results.append(r)

    for ks in (5, 7):
        filt = ComplexLeeFilter(kernel_size=ks)
        r = _bench(f"ComplexLeeFilter.apply.k{ks}.{sz}", filt.apply,
                    setup=lambda: ((complex_img,), {}), version="2.0.0", **kw)
        if r:
            results.append(r)

    # --- Phase gradient ---
    for direction in ("row", "col", "magnitude"):
        filt = PhaseGradientFilter(kernel_size=5, direction=direction)
        r = _bench(
            f"PhaseGradientFilter.apply.{direction}.k5.{sz}", filt.apply,
            setup=lambda: ((complex_img,), {}), version="1.0.0", **kw,
        )
        if r:
            results.append(r)

    return results


# ---------------------------------------------------------------------------
# Intensity transform benchmarks
# ---------------------------------------------------------------------------
def run_intensity_benchmarks(
    store: JSONBenchmarkStore,
    rows: int,
    cols: int,
    iterations: int,
    warmup: int,
    tags: Dict[str, str],
) -> List:
    """Benchmark intensity transforms."""
    from grdl.image_processing.intensity import PercentileStretch, ToDecibels

    _section("Image Processing — Intensity Transforms")
    sz = tags["array_size"]
    results = []
    mod = "image_processing.intensity"

    complex_img = (
        np.random.rand(rows, cols) + 1j * np.random.rand(rows, cols)
    ).astype(np.complex64)
    real_img = np.random.rand(rows, cols).astype(np.float32) * 100.0

    kw = dict(store=store, iterations=iterations, warmup=warmup,
              tags=tags, module=mod, version="1.0.0")

    for floor in (-60.0, -40.0):
        xform = ToDecibels(floor_db=floor)
        r = _bench(f"ToDecibels.apply.floor{int(floor)}.{sz}", xform.apply,
                    setup=lambda: ((complex_img,), {}), **kw)
        if r:
            results.append(r)

    for plow, phigh in [(2.0, 98.0), (1.0, 99.0)]:
        xform = PercentileStretch(plow=plow, phigh=phigh)
        r = _bench(f"PercentileStretch.apply.p{plow}-{phigh}.{sz}",
                    xform.apply, setup=lambda: ((real_img,), {}), **kw)
        if r:
            results.append(r)

    return results


# ---------------------------------------------------------------------------
# Decomposition benchmarks
# ---------------------------------------------------------------------------
def run_decomposition_benchmarks(
    store: JSONBenchmarkStore,
    rows: int,
    cols: int,
    iterations: int,
    warmup: int,
    tags: Dict[str, str],
) -> List:
    """Benchmark polarimetric decomposition."""
    from grdl.image_processing.decomposition import PauliDecomposition

    _section("Image Processing — Decomposition")
    sz = tags["array_size"]
    results = []
    mod = "image_processing.decomposition"

    shh = (np.random.rand(rows, cols) + 1j * np.random.rand(rows, cols)).astype(np.complex64)
    shv = (np.random.rand(rows, cols) + 1j * np.random.rand(rows, cols)).astype(np.complex64)
    svh = (np.random.rand(rows, cols) + 1j * np.random.rand(rows, cols)).astype(np.complex64)
    svv = (np.random.rand(rows, cols) + 1j * np.random.rand(rows, cols)).astype(np.complex64)

    decomp = PauliDecomposition()
    kw = dict(store=store, iterations=iterations, warmup=warmup,
              tags=tags, module=mod, version="1.0.0")

    r = _bench(f"PauliDecomposition.decompose.{sz}", decomp.decompose,
                setup=lambda: ((shh, shv, svh, svv), {}), **kw)
    if r:
        results.append(r)

    components = decomp.decompose(shh, shv, svh, svv)
    r = _bench(f"PauliDecomposition.to_rgb.{sz}", decomp.to_rgb,
                setup=lambda: ((components,), {}), **kw)
    if r:
        results.append(r)

    # --- DualPolHAlpha ---
    try:
        from grdl.image_processing.decomposition import DualPolHAlpha

        rng = np.random.default_rng(42)
        co_pol = (
            rng.standard_normal((rows, cols))
            + 1j * rng.standard_normal((rows, cols))
        ).astype(np.complex64)
        cross_pol = (
            rng.standard_normal((rows, cols)) * 0.3
            + 1j * rng.standard_normal((rows, cols)) * 0.3
        ).astype(np.complex64)

        for ws in (7, 11):
            halpha = DualPolHAlpha(window_size=ws)
            _co, _cross = co_pol, cross_pol  # capture for lambda
            r = _bench(
                f"DualPolHAlpha.decompose.w{ws}.{sz}", halpha.decompose,
                setup=lambda _c=_co, _x=_cross: ((_c, _x), {}), **kw,
            )
            if r:
                results.append(r)

        halpha = DualPolHAlpha(window_size=7)
        ha_components = halpha.decompose(co_pol, cross_pol)
        r = _bench(
            f"DualPolHAlpha.to_rgb.{sz}", halpha.to_rgb,
            setup=lambda: ((ha_components,), {}), **kw,
        )
        if r:
            results.append(r)

    except (ImportError, Exception) as exc:
        print(f"  SKIP  DualPolHAlpha benchmarks ({exc})")

    return results


# ---------------------------------------------------------------------------
# Detection benchmarks
# ---------------------------------------------------------------------------
def run_detection_benchmarks(
    store: JSONBenchmarkStore,
    rows: int,
    cols: int,
    iterations: int,
    warmup: int,
    tags: Dict[str, str],
) -> List:
    """Benchmark CFAR detectors and detection data models."""
    _section("Detection — CFAR Detectors")
    sz = tags["array_size"]
    results = []
    mod = "image_processing.detection"

    kw = dict(store=store, iterations=iterations, warmup=warmup,
              tags=tags, module=mod)

    # Synthetic SAR-like magnitude image (Rayleigh-distributed)
    rng = np.random.default_rng(42)
    magnitude = np.abs(
        rng.standard_normal((rows, cols))
        + 1j * rng.standard_normal((rows, cols))
    ).astype(np.float64)
    # Convert to dB for CFAR
    db_img = 20.0 * np.log10(magnitude + 1e-10)

    # Inject 5 bright targets
    peak = db_img.max()
    for r_t, c_t in [(rows // 4, cols // 4), (rows // 2, cols // 2),
                     (rows * 3 // 4, cols * 3 // 4),
                     (rows // 4, cols * 3 // 4),
                     (rows * 3 // 4, cols // 4)]:
        db_img[r_t - 2:r_t + 3, c_t - 2:c_t + 3] = peak + 20.0

    # --- CA-CFAR ---
    try:
        from grdl.image_processing.detection.cfar import CACFARDetector

        for pfa in (1e-3, 1e-4):
            det = CACFARDetector(guard_cells=3, training_cells=12, pfa=pfa)
            _db = db_img
            r = _bench(f"CACFARDetector.detect.pfa{pfa}.{sz}", det.detect,
                       setup=lambda _d=_db: ((_d,), {}), version="1.0.0", **kw)
            if r:
                results.append(r)
    except (ImportError, Exception) as exc:
        print(f"  SKIP  CACFARDetector benchmarks ({exc})")

    # --- GO-CFAR ---
    try:
        from grdl.image_processing.detection.cfar import GOCFARDetector

        det = GOCFARDetector(guard_cells=3, training_cells=12, pfa=1e-3)
        r = _bench(f"GOCFARDetector.detect.{sz}", det.detect,
                   setup=lambda: ((db_img,), {}), version="1.0.0", **kw)
        if r:
            results.append(r)
    except (ImportError, Exception) as exc:
        print(f"  SKIP  GOCFARDetector benchmarks ({exc})")

    # --- SO-CFAR ---
    try:
        from grdl.image_processing.detection.cfar import SOCFARDetector

        det = SOCFARDetector(guard_cells=3, training_cells=12, pfa=1e-3)
        r = _bench(f"SOCFARDetector.detect.{sz}", det.detect,
                   setup=lambda: ((db_img,), {}), version="1.0.0", **kw)
        if r:
            results.append(r)
    except (ImportError, Exception) as exc:
        print(f"  SKIP  SOCFARDetector benchmarks ({exc})")

    # --- OS-CFAR ---
    try:
        from grdl.image_processing.detection.cfar import OSCFARDetector

        det = OSCFARDetector(guard_cells=3, training_cells=12, pfa=1e-3)
        r = _bench(f"OSCFARDetector.detect.{sz}", det.detect,
                   setup=lambda: ((db_img,), {}), version="1.0.0", **kw)
        if r:
            results.append(r)
    except (ImportError, Exception) as exc:
        print(f"  SKIP  OSCFARDetector benchmarks ({exc})")

    # --- DetectionSet construction & serialization ---
    try:
        from shapely.geometry import Point

        from grdl.image_processing.detection import Detection, DetectionSet

        detections = [
            Detection(
                pixel_geometry=Point(c, r_t),
                properties={"snr": float(i)},
                confidence=0.95,
            )
            for i, (r_t, c) in enumerate(
                [(100, 200), (300, 400), (500, 600)]
            )
        ]
        ds = DetectionSet(detections=detections, detector_name="CA-CFAR", detector_version="1.0.0")

        r = _bench(f"DetectionSet.to_geojson.{sz}", ds.to_geojson,
                   version="1.0.0", **kw)
        if r:
            results.append(r)
    except (ImportError, Exception) as exc:
        print(f"  SKIP  DetectionSet benchmarks ({exc})")

    # --- Fields lookup ---
    try:
        from grdl.image_processing.detection import Fields

        r = _bench("Fields.lookup.SNR", lambda: Fields,
                   version="1.0.0", **kw)
        if r:
            results.append(r)
    except (ImportError, Exception) as exc:
        print(f"  SKIP  Fields benchmarks ({exc})")

    return results


# ---------------------------------------------------------------------------
# Pipeline benchmarks
# ---------------------------------------------------------------------------
def run_pipeline_benchmarks(
    store: JSONBenchmarkStore,
    rows: int,
    cols: int,
    iterations: int,
    warmup: int,
    tags: Dict[str, str],
) -> List:
    """Benchmark Pipeline sequential composition."""
    from grdl.image_processing import Pipeline
    from grdl.image_processing.filters import GaussianFilter, MedianFilter
    from grdl.image_processing.intensity import PercentileStretch, ToDecibels

    _section("Image Processing — Pipeline")
    sz = tags["array_size"]
    results = []

    complex_img = (
        np.random.rand(rows, cols) + 1j * np.random.rand(rows, cols)
    ).astype(np.complex64)

    pipe = Pipeline(steps=[
        ToDecibels(floor_db=-60.0),
        GaussianFilter(sigma=1.0),
        MedianFilter(kernel_size=3),
        PercentileStretch(plow=2.0, phigh=98.0),
    ])

    r = _bench(
        f"Pipeline.apply.4step.{sz}", pipe.apply,
        setup=lambda: ((complex_img,), {}),
        store=store, iterations=iterations, warmup=warmup,
        tags=tags, module="image_processing.pipeline", version="1.0.0",
    )
    if r:
        results.append(r)

    return results


# ---------------------------------------------------------------------------
# Data preparation benchmarks
# ---------------------------------------------------------------------------
def run_data_prep_benchmarks(
    store: JSONBenchmarkStore,
    rows: int,
    cols: int,
    iterations: int,
    warmup: int,
    tags: Dict[str, str],
) -> List:
    """Benchmark ChipExtractor, Tiler, and Normalizer."""
    from grdl.data_prep import ChipExtractor, Normalizer, Tiler

    _section("Data Preparation")
    sz = tags["array_size"]
    results = []
    mod = "data_prep"

    real_img = np.random.rand(rows, cols).astype(np.float32)

    kw = dict(store=store, iterations=iterations, warmup=warmup,
              tags=tags, module=mod)

    # --- ChipExtractor ---
    ext = ChipExtractor(nrows=rows, ncols=cols)

    r = _bench(
        f"ChipExtractor.chip_at_point.scalar.{sz}",
        ext.chip_at_point,
        setup=lambda: (
            (rows // 2, cols // 2),
            {"row_width": 256, "col_width": 256},
        ),
        **kw,
    )
    if r:
        results.append(r)

    batch_rows = np.random.randint(0, rows, size=100)
    batch_cols = np.random.randint(0, cols, size=100)
    r = _bench(
        f"ChipExtractor.chip_at_point.batch100.{sz}",
        ext.chip_at_point,
        setup=lambda: (
            (batch_rows, batch_cols),
            {"row_width": 256, "col_width": 256},
        ),
        **kw,
    )
    if r:
        results.append(r)

    r = _bench(
        f"ChipExtractor.chip_positions.256x256.{sz}",
        ext.chip_positions,
        setup=lambda: ((), {"row_width": 256, "col_width": 256}),
        **kw,
    )
    if r:
        results.append(r)

    # --- Tiler ---
    for tile_size in (256, 512):
        tiler = Tiler(nrows=rows, ncols=cols, tile_size=tile_size)
        r = _bench(f"Tiler.tile_positions.t{tile_size}.{sz}",
                    tiler.tile_positions, **kw)
        if r:
            results.append(r)

    tiler = Tiler(nrows=rows, ncols=cols, tile_size=256, stride=128)
    r = _bench(f"Tiler.tile_positions.t256_s128.{sz}",
                tiler.tile_positions, **kw)
    if r:
        results.append(r)

    # --- Normalizer ---
    for method in ("minmax", "zscore", "percentile", "unit_norm"):
        norm = Normalizer(method=method)
        r = _bench(f"Normalizer.normalize.{method}.{sz}", norm.normalize,
                    setup=lambda: ((real_img,), {}), **kw)
        if r:
            results.append(r)

    norm = Normalizer(method="minmax")
    norm.fit(real_img)
    r = _bench(f"Normalizer.transform.minmax.{sz}", norm.transform,
                setup=lambda: ((real_img,), {}), **kw)
    if r:
        results.append(r)

    return results


# ---------------------------------------------------------------------------
# IO benchmarks
# ---------------------------------------------------------------------------
def run_io_benchmarks(
    store: JSONBenchmarkStore,
    rows: int,
    cols: int,
    iterations: int,
    warmup: int,
    tags: Dict[str, str],
) -> List:
    """Benchmark IO readers and writers using temporary files."""
    _section("IO — Readers & Writers")
    sz = tags["array_size"]
    results = []
    mod = "IO"

    real_img = np.random.rand(rows, cols).astype(np.float32)
    uint8_img = (np.random.rand(rows, cols) * 255).astype(np.uint8)

    kw = dict(store=store, iterations=iterations, warmup=warmup,
              tags=tags, module=mod)
    tmpdir = Path(tempfile.mkdtemp(prefix="grdl_bench_"))

    try:
        # --- NumpyWriter (always available) ---
        from grdl.IO.numpy_io import NumpyWriter

        npy_path = tmpdir / "bench.npy"
        r = _bench(f"NumpyWriter.write.{sz}",
                    NumpyWriter(npy_path).write,
                    setup=lambda: ((real_img,), {}), **kw)
        if r:
            results.append(r)

        npz_path = tmpdir / "bench.npz"
        arrays = {"band_0": real_img, "band_1": real_img}
        r = _bench(f"NumpyWriter.write_npz.2band.{sz}",
                    NumpyWriter(npz_path).write_npz,
                    setup=lambda: ((arrays,), {}), **kw)
        if r:
            results.append(r)

        # --- GeoTIFF (requires rasterio) ---
        try:
            from rasterio.transform import from_bounds
            from grdl.IO.geotiff import GeoTIFFReader, GeoTIFFWriter

            _geo = {"crs": "EPSG:4326",
                    "transform": from_bounds(0, 0, 1, 1, cols, rows)}

            tif_path = tmpdir / "bench.tif"
            w = GeoTIFFWriter(tif_path)
            w.write(real_img, geolocation=_geo)
            w.close()

            tif_write_path = tmpdir / "bench_write.tif"
            r = _bench(f"GeoTIFFWriter.write.{sz}",
                        GeoTIFFWriter(tif_write_path).write,
                        setup=lambda: ((real_img,), {"geolocation": _geo}),
                        version="1.0.0", **kw)
            if r:
                results.append(r)

            def _geotiff_read_full():
                reader = GeoTIFFReader(tif_path)
                arr = reader.read_full()
                reader.close()
                return arr

            r = _bench(f"GeoTIFFReader.read_full.{sz}",
                        _geotiff_read_full, version="1.0.0", **kw)
            if r:
                results.append(r)

            r0, c0 = rows // 4, cols // 4
            r1, c1 = rows * 3 // 4, cols * 3 // 4

            def _geotiff_read_chip():
                reader = GeoTIFFReader(tif_path)
                arr = reader.read_chip(r0, r1, c0, c1)
                reader.close()
                return arr

            r = _bench(f"GeoTIFFReader.read_chip.quarter.{sz}",
                        _geotiff_read_chip, version="1.0.0", **kw)
            if r:
                results.append(r)

        except ImportError:
            print("  SKIP  GeoTIFF benchmarks (rasterio not installed)")

        # --- HDF5 (requires h5py) ---
        try:
            from grdl.IO.hdf5 import HDF5Reader, HDF5Writer

            h5_path = tmpdir / "bench.h5"
            w = HDF5Writer(h5_path)
            w.write(real_img)
            w.close()

            h5_write_path = tmpdir / "bench_write.h5"

            def _hdf5_write():
                if h5_write_path.exists():
                    h5_write_path.unlink()
                w = HDF5Writer(h5_write_path)
                w.write(real_img)
                w.close()

            r = _bench(f"HDF5Writer.write.{sz}", _hdf5_write,
                        version="1.0.0", **kw)
            if r:
                results.append(r)

            def _hdf5_read_full():
                reader = HDF5Reader(h5_path)
                arr = reader.read_full()
                reader.close()
                return arr

            r = _bench(f"HDF5Reader.read_full.{sz}",
                        _hdf5_read_full, version="1.0.0", **kw)
            if r:
                results.append(r)

            r0, c0 = rows // 4, cols // 4
            r1, c1 = rows * 3 // 4, cols * 3 // 4

            def _hdf5_read_chip():
                reader = HDF5Reader(h5_path)
                arr = reader.read_chip(r0, r1, c0, c1)
                reader.close()
                return arr

            r = _bench(f"HDF5Reader.read_chip.quarter.{sz}",
                        _hdf5_read_chip, version="1.0.0", **kw)
            if r:
                results.append(r)

        except ImportError:
            print("  SKIP  HDF5 benchmarks (h5py not installed)")

        # --- PNG (requires Pillow or cv2) ---
        try:
            from grdl.IO.png import PngWriter

            png_path = tmpdir / "bench.png"
            r = _bench(f"PngWriter.write.{sz}",
                        PngWriter(png_path).write,
                        setup=lambda: ((uint8_img,), {}), **kw)
            if r:
                results.append(r)

        except ImportError:
            print("  SKIP  PNG benchmarks (Pillow not installed)")

        # --- JPEG2000 (requires rasterio or glymur) ---
        try:
            from grdl.IO.jpeg2000 import JP2Reader

            try:
                import rasterio
                from rasterio.transform import from_bounds

                jp2_path = tmpdir / "bench.jp2"
                transform = from_bounds(0, 0, 1, 1, cols, rows)
                uint16_img = (real_img * 10000).astype(np.uint16)
                with rasterio.open(
                    str(jp2_path), "w", driver="JP2OpenJPEG",
                    height=rows, width=cols, count=1, dtype="uint16",
                    transform=transform,
                ) as dst:
                    dst.write(uint16_img, 1)

                def _jp2_read_full():
                    reader = JP2Reader(jp2_path)
                    arr = reader.read_full()
                    reader.close()
                    return arr

                r = _bench(f"JP2Reader.read_full.{sz}",
                            _jp2_read_full, **kw)
                if r:
                    results.append(r)

            except Exception:
                print("  SKIP  JP2 write failed (JP2 driver unavailable)")

        except ImportError:
            print("  SKIP  JPEG2000 benchmarks (JP2Reader not available)")

        # --- SAR readers (require real data files) ---
        _run_real_data_io(store, rows, cols, iterations, warmup, tags,
                          results)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return results


def _find_data_file(directory: Path, pattern: str) -> Optional[Path]:
    """Find first file matching *pattern* in *directory*.

    Returns ``None`` when the directory is missing or contains no matches.
    """
    if not directory.exists():
        return None
    matches = sorted(directory.glob(pattern))
    return matches[0] if matches else None


def _find_sentinel2_jp2(sentinel2_dir: Path) -> Optional[Path]:
    """Locate a Sentinel-2 B04 JP2, standalone or inside a SAFE archive."""
    # Standalone JP2
    hit = _find_data_file(sentinel2_dir, "T*_B*.jp2")
    if hit:
        return hit
    # Within SAFE structure (README pattern)
    for safe in sorted(sentinel2_dir.glob("S2*.SAFE")):
        jp2s = sorted(safe.glob("**/IMG_DATA/**/*_B04*.jp2"))
        if jp2s:
            return jp2s[0]
    return None


def _run_real_data_io(
    store: JSONBenchmarkStore,
    rows: int,
    cols: int,
    iterations: int,
    warmup: int,
    tags: Dict[str, str],
    results: List,
) -> None:
    """Run IO benchmarks that require real data files.

    Data directory is resolved relative to *this* file so benchmarks work
    regardless of the caller's working directory.  Files are discovered by
    glob patterns defined in the per-dataset README files rather than by
    hard-coded filenames.
    """
    mod = "IO"
    kw = dict(store=store, iterations=iterations, warmup=warmup,
              tags={**tags, "data": "real"}, module=mod)

    # Resolve data/ relative to the grdl-te package root:
    #   suite.py -> benchmarking/ -> grdl_te/ -> grdl-te/ -> data/
    _data_dir = Path(__file__).resolve().parent.parent.parent / "data"

    # ==================================================================
    # Group 1: umbra/ — SICDReader + NITFReader (shared NITF data)
    # ==================================================================
    umbra_dir = _data_dir / "umbra"
    sar_path = (_find_data_file(umbra_dir, "*.nitf")
                or _find_data_file(umbra_dir, "*.ntf"))
    if sar_path:
        try:
            from grdl.IO.sar import SICDReader

            def _sicd_read_full():
                with _suppress_stderr(), SICDReader(sar_path) as reader:
                    return reader.read_full()

            r = _bench("SICDReader.read_full.real_data",
                        _sicd_read_full, **kw)
            if r:
                results.append(r)

            def _sicd_read_chip():
                with _suppress_stderr(), SICDReader(sar_path) as reader:
                    s = reader.get_shape()
                    cx, cy = s[0] // 2, s[1] // 2
                    return reader.read_chip(cx - 512, cx + 512,
                                            cy - 512, cy + 512)

            r = _bench("SICDReader.read_chip.1024x1024.real_data",
                        _sicd_read_chip, **kw)
            if r:
                results.append(r)

        except ImportError:
            print("  SKIP  SICDReader benchmarks (import failed)")

        # NITFReader on the same umbra NITF file
        try:
            from grdl.IO.nitf import NITFReader

            def _nitf_read_full():
                with _suppress_stderr(), NITFReader(sar_path) as reader:
                    return reader.read_full()

            r = _bench("NITFReader.read_full.real_data",
                        _nitf_read_full, **kw)
            if r:
                results.append(r)

        except ImportError:
            print("  SKIP  NITFReader benchmarks (import failed)")
    else:
        print("  SKIP  SICDReader benchmarks (data file not found)")
        print("  SKIP  NITFReader benchmarks (data file not found)")

    # ==================================================================
    # Group 2: viirs/ — VIIRSReader + HDF5Reader (shared HDF5 data)
    # ==================================================================
    viirs_dir = _data_dir / "viirs"
    vpath = (_find_data_file(viirs_dir, "V?P09GA*.h5")
             or _find_data_file(viirs_dir, "V?P09GA*.hdf5"))
    if vpath:
        try:
            from grdl.IO.multispectral import VIIRSReader

            def _viirs_read_full():
                with VIIRSReader(vpath) as reader:
                    return reader.read_full()

            r = _bench("VIIRSReader.read_full.real_data",
                        _viirs_read_full, **kw)
            if r:
                results.append(r)

        except (ImportError, Exception) as exc:
            print(f"  SKIP  VIIRSReader benchmarks ({exc})")

        # HDF5Reader on the same VIIRS HDF5 file
        try:
            from grdl.IO.hdf5 import HDF5Reader

            def _hdf5_read_full_real():
                with HDF5Reader(vpath) as reader:
                    return reader.read_full()

            r = _bench("HDF5Reader.read_full.real_data",
                        _hdf5_read_full_real, **kw)
            if r:
                results.append(r)

        except (ImportError, Exception) as exc:
            print(f"  SKIP  HDF5Reader real data benchmarks ({exc})")
    else:
        print("  SKIP  VIIRSReader benchmarks (data files not found)")
        print("  SKIP  HDF5Reader real data benchmarks (data files not found)")

    # ==================================================================
    # Group 3: sentinel2/ — Sentinel2Reader + JP2Reader (shared JP2 data)
    # ==================================================================
    sentinel2_dir = _data_dir / "sentinel2"
    jp2_path = _find_sentinel2_jp2(sentinel2_dir)
    if jp2_path:
        try:
            from grdl.IO.eo import Sentinel2Reader

            def _sentinel2_read_full():
                with Sentinel2Reader(jp2_path) as reader:
                    return reader.read_full()

            r = _bench("Sentinel2Reader.read_full.real_data",
                        _sentinel2_read_full, **kw)
            if r:
                results.append(r)

        except (ImportError, Exception) as exc:
            print(f"  SKIP  Sentinel2Reader benchmarks ({exc})")

        # JP2Reader on the same Sentinel-2 JP2 file
        try:
            from grdl.IO.jpeg2000 import JP2Reader

            def _jp2_read_full():
                with JP2Reader(jp2_path) as reader:
                    return reader.read_full()

            r = _bench("JP2Reader.read_full.real_data",
                        _jp2_read_full, **kw)
            if r:
                results.append(r)

        except (ImportError, Exception) as exc:
            print(f"  SKIP  JP2Reader benchmarks ({exc})")
    else:
        print("  SKIP  Sentinel2Reader benchmarks (data files not found)")
        print("  SKIP  JP2Reader benchmarks (data files not found)")

    # ==================================================================
    # Group 4: landsat/ — GeoTIFFReader (real GeoTIFF data)
    # ==================================================================
    landsat_dir = _data_dir / "landsat"
    tif_path = _find_data_file(landsat_dir, "LC0[89]*_SR_B*.TIF")
    if tif_path:
        try:
            from grdl.IO.geotiff import GeoTIFFReader

            def _geotiff_read_full_real():
                with GeoTIFFReader(tif_path) as reader:
                    return reader.read_full()

            r = _bench("GeoTIFFReader.read_full.real_data",
                        _geotiff_read_full_real, **kw)
            if r:
                results.append(r)

        except (ImportError, Exception) as exc:
            print(f"  SKIP  GeoTIFFReader real data benchmarks ({exc})")
    else:
        print("  SKIP  GeoTIFFReader real data benchmarks (data files not found)")

    # ==================================================================
    # Group 5: cphd/ — CPHDReader
    # ==================================================================
    cphd_dir = _data_dir / "cphd"
    cphd_path = _find_data_file(cphd_dir, "*.cphd")
    if cphd_path:
        try:
            from grdl.IO.sar import CPHDReader

            def _cphd_read_full():
                with CPHDReader(cphd_path) as reader:
                    return reader.read_full()

            r = _bench("CPHDReader.read_full.real_data",
                        _cphd_read_full, **kw)
            if r:
                results.append(r)

            def _cphd_read_pvp():
                with CPHDReader(cphd_path) as reader:
                    return reader.metadata.pvp

            r = _bench("CPHDReader.read_pvp.real_data",
                        _cphd_read_pvp, **kw)
            if r:
                results.append(r)

        except (ImportError, Exception) as exc:
            print(f"  SKIP  CPHDReader benchmarks ({exc})")
    else:
        print("  SKIP  CPHDReader benchmarks (data files not found)")

    # ==================================================================
    # Group 6: crsd/ — CRSDReader
    # ==================================================================
    crsd_dir = _data_dir / "crsd"
    crsd_path = _find_data_file(crsd_dir, "*.crsd")
    if crsd_path:
        try:
            from grdl.IO.sar import CRSDReader

            def _crsd_read_full():
                with CRSDReader(crsd_path) as reader:
                    return reader.read_full()

            r = _bench("CRSDReader.read_full.real_data",
                        _crsd_read_full, **kw)
            if r:
                results.append(r)

        except (ImportError, Exception) as exc:
            print(f"  SKIP  CRSDReader benchmarks ({exc})")
    else:
        print("  SKIP  CRSDReader benchmarks (data files not found)")

    # ==================================================================
    # Group 7: sidd/ — SIDDReader
    # ==================================================================
    sidd_dir = _data_dir / "sidd"
    sidd_path = (_find_data_file(sidd_dir, "*.nitf")
                 or _find_data_file(sidd_dir, "*.ntf"))
    if sidd_path:
        try:
            from grdl.IO.sar import SIDDReader

            def _sidd_read_full():
                with _suppress_stderr(), SIDDReader(sidd_path) as reader:
                    return reader.read_full()

            r = _bench("SIDDReader.read_full.real_data",
                        _sidd_read_full, **kw)
            if r:
                results.append(r)

        except (ImportError, Exception) as exc:
            print(f"  SKIP  SIDDReader benchmarks ({exc})")
    else:
        print("  SKIP  SIDDReader benchmarks (data files not found)")

    # ==================================================================
    # Group 8: sentinel1/ — Sentinel1SLCReader
    # ==================================================================
    sentinel1_dir = _data_dir / "sentinel1"
    s1_path = _find_data_file(sentinel1_dir, "*.SAFE")
    if s1_path:
        try:
            from grdl.IO.sar import Sentinel1SLCReader

            def _s1_read_full():
                with Sentinel1SLCReader(s1_path) as reader:
                    return reader.read_full()

            r = _bench("Sentinel1SLCReader.read_full.real_data",
                        _s1_read_full, **kw)
            if r:
                results.append(r)

            def _s1_read_chip():
                with Sentinel1SLCReader(s1_path) as reader:
                    s = reader.get_shape()
                    cx, cy = s[0] // 2, s[1] // 2
                    return reader.read_chip(cx - 512, cx + 512,
                                            cy - 512, cy + 512)

            r = _bench("Sentinel1SLCReader.read_chip.1024x1024.real_data",
                        _s1_read_chip, **kw)
            if r:
                results.append(r)

        except (ImportError, Exception) as exc:
            print(f"  SKIP  Sentinel1SLCReader benchmarks ({exc})")
    else:
        print("  SKIP  Sentinel1SLCReader benchmarks (data files not found)")

    # ==================================================================
    # Group 9: aster/ — ASTERReader
    # ==================================================================
    aster_dir = _data_dir / "aster"
    aster_path = _find_data_file(aster_dir, "AST_L1T*.tif")
    if aster_path:
        try:
            from grdl.IO.ir import ASTERReader

            def _aster_read_full():
                with ASTERReader(aster_path) as reader:
                    return reader.read_full()

            r = _bench("ASTERReader.read_full.real_data",
                        _aster_read_full, **kw)
            if r:
                results.append(r)

        except (ImportError, Exception) as exc:
            print(f"  SKIP  ASTERReader benchmarks ({exc})")
    else:
        print("  SKIP  ASTERReader benchmarks (data files not found)")

    # ==================================================================
    # Group 10: biomass/ — BIOMASSL1Reader
    # ==================================================================
    biomass_dir = _data_dir / "biomass"
    biomass_path = _find_data_file(biomass_dir, "BIO_S*")
    if biomass_path:
        try:
            from grdl.IO.sar import BIOMASSL1Reader

            def _biomass_read_full():
                with BIOMASSL1Reader(biomass_path) as reader:
                    return reader.read_full()

            r = _bench("BIOMASSL1Reader.read_full.real_data",
                        _biomass_read_full, **kw)
            if r:
                results.append(r)

        except (ImportError, Exception) as exc:
            print(f"  SKIP  BIOMASSL1Reader benchmarks ({exc})")
    else:
        print("  SKIP  BIOMASSL1Reader benchmarks (data files not found)")

    # ==================================================================
    # Group 11: terrasar/ — TerraSARReader
    # ==================================================================
    terrasar_dir = _data_dir / "terrasar"
    tsx_path = None
    if terrasar_dir.exists():
        for candidate in sorted(terrasar_dir.iterdir()):
            if candidate.is_dir() and (
                candidate.name.startswith("TSX1_")
                or candidate.name.startswith("TDX1_")
            ):
                tsx_path = candidate
                break
        if tsx_path is None:
            xmls = sorted(terrasar_dir.glob("TSX1_SAR__*.xml"))
            if xmls:
                tsx_path = terrasar_dir

    if tsx_path:
        try:
            from grdl.IO.sar import TerraSARReader

            def _tsx_read_full():
                with TerraSARReader(tsx_path) as reader:
                    return reader.read_full()

            r = _bench("TerraSARReader.read_full.real_data",
                        _tsx_read_full, **kw)
            if r:
                results.append(r)

            def _tsx_read_chip():
                with TerraSARReader(tsx_path) as reader:
                    s = reader.get_shape()
                    cx, cy = s[0] // 2, s[1] // 2
                    half = min(512, s[0] // 2, s[1] // 2)
                    return reader.read_chip(cx - half, cx + half,
                                            cy - half, cy + half)

            r = _bench("TerraSARReader.read_chip.1024x1024.real_data",
                        _tsx_read_chip, **kw)
            if r:
                results.append(r)

        except (ImportError, Exception) as exc:
            print(f"  SKIP  TerraSARReader benchmarks ({exc})")
    else:
        print("  SKIP  TerraSARReader benchmarks (data files not found)")

    # ==================================================================
    # Group 12: SICDWriter roundtrip (write to tmpdir, read back)
    # ==================================================================
    if sar_path:
        try:
            import copy
            from grdl.IO.sar import SICDReader, SICDWriter

            with _suppress_stderr(), SICDReader(sar_path) as reader:
                sicd_meta = reader.metadata
                shape = reader.get_shape()
                cx, cy = shape[0] // 2, shape[1] // 2
                half = min(256, shape[0] // 2, shape[1] // 2)
                chip = reader.read_chip(cx - half, cx + half,
                                        cy - half, cy + half)
            # Ensure native complex64 dtype (sarpy expects complex64, not >c8)
            chip = chip.astype(np.complex64, copy=False)

            # Adapt metadata dimensions to match the chip
            cm = copy.deepcopy(sicd_meta)
            cm.rows, cm.cols = chip.shape[0], chip.shape[1]
            cm.image_data.num_rows = chip.shape[0]
            cm.image_data.num_cols = chip.shape[1]
            cm.image_data.first_row = 0
            cm.image_data.first_col = 0
            cm.image_data.scp_pixel.row = chip.shape[0] // 2
            cm.image_data.scp_pixel.col = chip.shape[1] // 2
            if cm.image_data.full_image:
                cm.image_data.full_image.num_rows = chip.shape[0]
                cm.image_data.full_image.num_cols = chip.shape[1]

            tmpdir = Path(tempfile.mkdtemp(prefix="grdl_bench_sicd_"))
            try:
                out_path = tmpdir / "bench_sicd.nitf"

                def _sicd_write():
                    SICDWriter(out_path, metadata=cm).write(chip)

                r = _bench("SICDWriter.write.real_data", _sicd_write, **kw)
                if r:
                    results.append(r)
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

        except (ImportError, Exception) as exc:
            print(f"  SKIP  SICDWriter benchmarks ({exc})")

    # ==================================================================
    # Group 13: NITFWriter (write synthetic array)
    # ==================================================================
    try:
        from grdl.IO.nitf import NITFWriter

        tmpdir = Path(tempfile.mkdtemp(prefix="grdl_bench_nitf_"))
        try:
            nitf_out = tmpdir / "bench_nitf.nitf"
            synthetic = np.random.rand(rows, cols).astype(np.float32)

            def _nitf_write():
                NITFWriter(nitf_out).write(synthetic)

            r = _bench(f"NITFWriter.write.{tags.get('array_size', 'unknown')}",
                        _nitf_write, **kw)
            if r:
                results.append(r)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    except (ImportError, Exception) as exc:
        print(f"  SKIP  NITFWriter benchmarks ({exc})")


# ---------------------------------------------------------------------------
# Geolocation benchmarks
# ---------------------------------------------------------------------------
def run_geolocation_benchmarks(
    store: JSONBenchmarkStore,
    rows: int,
    cols: int,
    iterations: int,
    warmup: int,
    tags: Dict[str, str],
) -> List:
    """Benchmark geolocation transforms."""
    _section("Geolocation")
    sz = tags["array_size"]
    results = []
    mod = "geolocation"

    kw = dict(store=store, iterations=iterations, warmup=warmup,
              tags=tags, module=mod)

    # --- AffineGeolocation (requires rasterio.Affine) ---
    try:
        from rasterio.transform import Affine

        from grdl.geolocation import AffineGeolocation

        transform = Affine(0.00027, 0.0, -118.0,
                           0.0, -0.00027, 34.0)
        geo = AffineGeolocation(
            transform=transform, shape=(rows, cols), crs="EPSG:4326",
        )

        r = _bench(
            f"AffineGeolocation.image_to_latlon.scalar.{sz}",
            geo.image_to_latlon,
            setup=lambda: ((rows // 2, cols // 2), {}), **kw,
        )
        if r:
            results.append(r)

        row_arr = np.random.uniform(0, rows, size=1000)
        col_arr = np.random.uniform(0, cols, size=1000)
        r = _bench(
            f"AffineGeolocation.image_to_latlon.batch1000.{sz}",
            geo.image_to_latlon,
            setup=lambda: ((row_arr, col_arr), {}), **kw,
        )
        if r:
            results.append(r)

        r = _bench(
            f"AffineGeolocation.latlon_to_image.scalar.{sz}",
            geo.latlon_to_image,
            setup=lambda: ((34.0, -118.0), {}), **kw,
        )
        if r:
            results.append(r)

        lat_arr = np.random.uniform(33.5, 34.5, size=1000)
        lon_arr = np.random.uniform(-118.5, -117.5, size=1000)
        r = _bench(
            f"AffineGeolocation.latlon_to_image.batch1000.{sz}",
            geo.latlon_to_image,
            setup=lambda: ((lat_arr, lon_arr), {}), **kw,
        )
        if r:
            results.append(r)

    except ImportError:
        print("  SKIP  AffineGeolocation benchmarks (rasterio not installed)")

    # --- GCPGeolocation (synthetic GCPs) ---
    try:
        from grdl.geolocation import GCPGeolocation

        gcps = []
        for gi in range(5):
            for gj in range(5):
                row = int(gi * (rows - 1) / 4)
                col = int(gj * (cols - 1) / 4)
                lat = 34.0 + gi * 0.1
                lon = -118.0 + gj * 0.1
                gcps.append((lon, lat, 100.0, row, col))

        geo = GCPGeolocation(gcps=gcps, shape=(rows, cols))

        r = _bench(
            f"GCPGeolocation.image_to_latlon.scalar.{sz}",
            geo.image_to_latlon,
            setup=lambda: ((rows // 2, cols // 2), {}), **kw,
        )
        if r:
            results.append(r)

        row_arr = np.random.uniform(0, rows, size=1000)
        col_arr = np.random.uniform(0, cols, size=1000)
        r = _bench(
            f"GCPGeolocation.image_to_latlon.batch1000.{sz}",
            geo.image_to_latlon,
            setup=lambda: ((row_arr, col_arr), {}), **kw,
        )
        if r:
            results.append(r)

        r = _bench(
            f"GCPGeolocation.latlon_to_image.scalar.{sz}",
            geo.latlon_to_image,
            setup=lambda: ((34.2, -117.8), {}), **kw,
        )
        if r:
            results.append(r)

    except (ImportError, Exception) as exc:
        print(f"  SKIP  GCPGeolocation benchmarks ({exc})")

    # --- SICDGeolocation (requires real SICD data) ---
    sar_path = _find_data_file(
        Path(__file__).resolve().parents[2] / "data" / "umbra",
        "*SICD*.nitf",
    )
    if sar_path is not None:
        try:
            from grdl.IO.sar import SICDReader
            from grdl.geolocation import SICDGeolocation

            with SICDReader(sar_path) as reader:
                geo = SICDGeolocation.from_reader(reader)
                shape = reader.get_shape()

            real_kw = dict(store=store, iterations=iterations, warmup=warmup,
                           tags={**tags, "data": "real"}, module=mod)

            r = _bench(
                "SICDGeolocation.image_to_latlon.scalar.real_data",
                geo.image_to_latlon,
                setup=lambda: ((shape[0] // 2, shape[1] // 2), {}),
                **real_kw,
            )
            if r:
                results.append(r)

            row_arr = np.random.uniform(0, shape[0], size=1000)
            col_arr = np.random.uniform(0, shape[1], size=1000)
            r = _bench(
                "SICDGeolocation.image_to_latlon.batch1000.real_data",
                geo.image_to_latlon,
                setup=lambda: ((row_arr, col_arr), {}), **real_kw,
            )
            if r:
                results.append(r)

        except (ImportError, Exception) as exc:
            print(f"  SKIP  SICDGeolocation benchmarks ({exc})")
    else:
        print("  SKIP  SICDGeolocation benchmarks (data file not found)")

    # --- Elevation models ---
    try:
        from grdl.geolocation import ConstantElevation

        elev = ConstantElevation(height=100.0)
        lat_arr = np.random.uniform(33.0, 35.0, size=10000)
        lon_arr = np.random.uniform(-119.0, -117.0, size=10000)

        r = _bench(
            "ConstantElevation.get_elevation.batch10000",
            elev.get_elevation,
            setup=lambda: ((lat_arr, lon_arr), {}), **kw,
        )
        if r:
            results.append(r)

    except (ImportError, Exception) as exc:
        print(f"  SKIP  ConstantElevation benchmarks ({exc})")

    # --- DTEDElevation ---
    _data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    dted_dir = _data_dir / "dted"
    if dted_dir.exists() and list(dted_dir.glob("**/*.dt?")):
        try:
            from grdl.geolocation.elevation import DTEDElevation

            elev = DTEDElevation(str(dted_dir))
            lat_arr = np.random.uniform(33.0, 35.0, size=10000)
            lon_arr = np.random.uniform(-119.0, -117.0, size=10000)

            r = _bench(
                "DTEDElevation.get_elevation.batch10000",
                elev.get_elevation,
                setup=lambda: ((lat_arr, lon_arr), {}),
                **{**kw, "tags": {**tags, "data": "real"}},
            )
            if r:
                results.append(r)

        except (ImportError, Exception) as exc:
            print(f"  SKIP  DTEDElevation benchmarks ({exc})")
    else:
        print("  SKIP  DTEDElevation benchmarks (data files not found)")

    # --- GeoTIFFDEM ---
    dem_path = _find_data_file(_data_dir / "dem", "*.tif")
    if dem_path:
        try:
            from grdl.geolocation.elevation import GeoTIFFDEM

            elev = GeoTIFFDEM(dem_path)
            lat_arr = np.random.uniform(33.0, 35.0, size=10000)
            lon_arr = np.random.uniform(-119.0, -117.0, size=10000)

            r = _bench(
                "GeoTIFFDEM.get_elevation.batch10000",
                elev.get_elevation,
                setup=lambda: ((lat_arr, lon_arr), {}),
                **{**kw, "tags": {**tags, "data": "real"}},
            )
            if r:
                results.append(r)

        except (ImportError, Exception) as exc:
            print(f"  SKIP  GeoTIFFDEM benchmarks ({exc})")
    else:
        print("  SKIP  GeoTIFFDEM benchmarks (data files not found)")

    # --- GeoidCorrection ---
    geoid_path = _find_data_file(_data_dir / "geoid", "*.pgm")
    if geoid_path:
        try:
            from grdl.geolocation.elevation import GeoidCorrection

            geoid = GeoidCorrection(geoid_path)
            lat_arr = np.random.uniform(33.0, 35.0, size=10000)
            lon_arr = np.random.uniform(-119.0, -117.0, size=10000)

            r = _bench(
                "GeoidCorrection.get_undulation.batch10000",
                geoid.get_undulation,
                setup=lambda: ((lat_arr, lon_arr), {}),
                **{**kw, "tags": {**tags, "data": "real"}},
            )
            if r:
                results.append(r)

        except (ImportError, Exception) as exc:
            print(f"  SKIP  GeoidCorrection benchmarks ({exc})")
    else:
        print("  SKIP  GeoidCorrection benchmarks (data files not found)")

    # --- Sentinel1SLCGeolocation ---
    s1_dir = _data_dir / "sentinel1"
    s1_path = _find_data_file(s1_dir, "*.SAFE")
    if s1_path:
        try:
            from grdl.IO.sar import Sentinel1SLCReader
            from grdl.geolocation import Sentinel1SLCGeolocation

            with Sentinel1SLCReader(s1_path) as reader:
                s1_geo = Sentinel1SLCGeolocation.from_reader(reader)
                s1_shape = reader.get_shape()

            real_kw = dict(store=store, iterations=iterations, warmup=warmup,
                           tags={**tags, "data": "real"}, module=mod)

            r = _bench(
                "Sentinel1SLCGeolocation.image_to_latlon.scalar.real_data",
                s1_geo.image_to_latlon,
                setup=lambda: ((s1_shape[0] // 2, s1_shape[1] // 2), {}),
                **real_kw,
            )
            if r:
                results.append(r)

            row_arr = np.random.uniform(0, s1_shape[0], size=1000)
            col_arr = np.random.uniform(0, s1_shape[1], size=1000)
            r = _bench(
                "Sentinel1SLCGeolocation.image_to_latlon.batch1000.real_data",
                s1_geo.image_to_latlon,
                setup=lambda: ((row_arr, col_arr), {}), **real_kw,
            )
            if r:
                results.append(r)

        except (ImportError, Exception) as exc:
            print(f"  SKIP  Sentinel1SLCGeolocation benchmarks ({exc})")
    else:
        print("  SKIP  Sentinel1SLCGeolocation benchmarks (data not found)")

    return results


# ---------------------------------------------------------------------------
# Coregistration benchmarks
# ---------------------------------------------------------------------------
def run_coregistration_benchmarks(
    store: JSONBenchmarkStore,
    rows: int,
    cols: int,
    iterations: int,
    warmup: int,
    tags: Dict[str, str],
) -> List:
    """Benchmark coregistration algorithms."""
    _section("Coregistration")
    sz = tags["array_size"]
    results = []
    mod = "coregistration"

    kw = dict(store=store, iterations=iterations, warmup=warmup,
              tags=tags, module=mod)

    rng = np.random.default_rng(42)
    fixed = rng.random((rows, cols), dtype=np.float32)

    from scipy.ndimage import affine_transform as scipy_affine

    angle = np.radians(2.0)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    offset = np.array([3.0, 3.0])
    moving = scipy_affine(fixed, matrix, offset=offset, order=1)

    pts_fixed = np.array([
        [rows * 0.25, cols * 0.25],
        [rows * 0.25, cols * 0.75],
        [rows * 0.75, cols * 0.25],
        [rows * 0.75, cols * 0.75],
        [rows * 0.5, cols * 0.5],
    ], dtype=np.float64)
    pts_moving = (matrix @ pts_fixed.T).T + offset

    # --- AffineCoRegistration ---
    try:
        from grdl.coregistration import AffineCoRegistration

        coreg = AffineCoRegistration(
            control_points_fixed=pts_fixed,
            control_points_moving=pts_moving,
        )

        r = _bench(f"AffineCoRegistration.estimate.{sz}", coreg.estimate,
                    setup=lambda: ((fixed, moving), {}), **kw)
        if r:
            results.append(r)

        result_obj = coreg.estimate(fixed, moving)
        r = _bench(f"AffineCoRegistration.apply.{sz}", coreg.apply,
                    setup=lambda: ((moving, result_obj), {}), **kw)
        if r:
            results.append(r)

    except (ImportError, Exception) as exc:
        print(f"  SKIP  AffineCoRegistration benchmarks ({exc})")

    # --- FeatureMatchCoRegistration (requires opencv) ---
    try:
        from grdl.coregistration import FeatureMatchCoRegistration

        coreg = FeatureMatchCoRegistration(
            method="orb", max_features=5000,
        )

        r = _bench(f"FeatureMatchCoRegistration.estimate.orb.{sz}",
                    coreg.estimate,
                    setup=lambda: ((fixed, moving), {}), **kw)
        if r:
            results.append(r)

    except ImportError:
        print("  SKIP  FeatureMatchCoRegistration (opencv not installed)")
    except Exception as exc:
        print(f"  SKIP  FeatureMatchCoRegistration ({exc})")

    # --- ProjectiveCoRegistration ---
    try:
        from grdl.coregistration import ProjectiveCoRegistration

        # Need at least 4 control points for projective transform
        pts_fixed_proj = np.array([
            [rows * 0.2, cols * 0.2],
            [rows * 0.2, cols * 0.8],
            [rows * 0.8, cols * 0.2],
            [rows * 0.8, cols * 0.8],
            [rows * 0.5, cols * 0.5],
            [rows * 0.3, cols * 0.7],
        ], dtype=np.float64)
        pts_moving_proj = (matrix @ pts_fixed_proj.T).T + offset

        coreg = ProjectiveCoRegistration(
            control_points_fixed=pts_fixed_proj,
            control_points_moving=pts_moving_proj,
        )

        r = _bench(f"ProjectiveCoRegistration.estimate.{sz}",
                    coreg.estimate,
                    setup=lambda: ((fixed, moving), {}), **kw)
        if r:
            results.append(r)

        result_obj = coreg.estimate(fixed, moving)
        r = _bench(f"ProjectiveCoRegistration.apply.{sz}", coreg.apply,
                    setup=lambda: ((moving, result_obj), {}), **kw)
        if r:
            results.append(r)

    except (ImportError, Exception) as exc:
        print(f"  SKIP  ProjectiveCoRegistration benchmarks ({exc})")

    return results


# ---------------------------------------------------------------------------
# Orthorectification benchmarks
# ---------------------------------------------------------------------------
def run_ortho_benchmarks(
    store: JSONBenchmarkStore,
    rows: int,
    cols: int,
    iterations: int,
    warmup: int,
    tags: Dict[str, str],
) -> List:
    """Benchmark orthorectification."""
    _section("Orthorectification")
    sz = tags["array_size"]
    results = []
    mod = "image_processing.ortho"

    try:
        from rasterio.transform import Affine

        from grdl.geolocation import AffineGeolocation
        from grdl.image_processing.ortho import Orthorectifier, OutputGrid

        real_img = np.random.rand(rows, cols).astype(np.float32)

        transform = Affine(0.00027, 0.0, -118.0,
                           0.0, -0.00027, 34.0)
        geo = AffineGeolocation(
            transform=transform, shape=(rows, cols), crs="EPSG:4326",
        )
        grid = OutputGrid.from_geolocation(
            geo, pixel_size_lat=0.00054, pixel_size_lon=0.00054,
        )
        ortho = Orthorectifier(
            geolocation=geo, output_grid=grid, interpolation="bilinear",
        )

        kw = dict(store=store, iterations=iterations, warmup=warmup,
                  tags=tags, module=mod)

        r = _bench(f"Orthorectifier.compute_mapping.{sz}",
                    ortho.compute_mapping, **kw)
        if r:
            results.append(r)

        r = _bench(f"Orthorectifier.apply.{sz}", ortho.apply,
                    setup=lambda: ((real_img,), {}), **kw)
        if r:
            results.append(r)

    except ImportError:
        print("  SKIP  Orthorectification benchmarks (rasterio not installed)")
    except Exception as exc:
        print(f"  SKIP  Orthorectification benchmarks ({exc})")

    # --- OrthoPipeline (builder-pattern benchmark) ---
    try:
        from rasterio.transform import Affine as RioAffine

        from grdl.geolocation import AffineGeolocation
        from grdl.image_processing.ortho import OrthoPipeline

        kw_ortho = dict(store=store, iterations=iterations, warmup=warmup,
                        tags=tags, module=mod)

        real_img = np.random.rand(rows, cols).astype(np.float32)
        transform = RioAffine(0.00027, 0.0, -118.0,
                              0.0, -0.00027, 34.0)
        geo = AffineGeolocation(
            transform=transform, shape=(rows, cols), crs="EPSG:4326",
        )

        pipeline = (
            OrthoPipeline()
            .with_source_array(real_img)
            .with_geolocation(geo)
            .with_resolution(0.00054, 0.00054)
            .with_interpolation('bilinear')
        )

        r = _bench(f"OrthoPipeline.run.{sz}", pipeline.run,
                    version="1.0.0", **kw_ortho)
        if r:
            results.append(r)

    except (ImportError, Exception) as exc:
        print(f"  SKIP  OrthoPipeline benchmarks ({exc})")

    # --- compute_output_resolution (requires SICD data) ---
    _data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    sar_path = (_find_data_file(_data_dir / "umbra", "*.nitf")
                or _find_data_file(_data_dir / "umbra", "*.ntf"))
    if sar_path:
        try:
            from grdl.IO.sar import SICDReader
            from grdl.geolocation import SICDGeolocation
            from grdl.image_processing.ortho import compute_output_resolution

            with SICDReader(sar_path) as reader:
                sicd_meta = reader.metadata
                sicd_geo = SICDGeolocation.from_reader(reader)

            real_kw = dict(store=store, iterations=iterations, warmup=warmup,
                           tags={**tags, "data": "real"}, module=mod)

            r = _bench(
                "compute_output_resolution.sicd",
                lambda: compute_output_resolution(sicd_meta, sicd_geo),
                version="1.0.0", **real_kw,
            )
            if r:
                results.append(r)

        except (ImportError, Exception) as exc:
            print(f"  SKIP  compute_output_resolution benchmarks ({exc})")
    else:
        print("  SKIP  compute_output_resolution (SICD data not found)")

    return results


# ---------------------------------------------------------------------------
# SAR-specific processing benchmarks
# ---------------------------------------------------------------------------
def run_sar_processing_benchmarks(
    store: JSONBenchmarkStore,
    rows: int,
    cols: int,
    iterations: int,
    warmup: int,
    tags: Dict[str, str],
) -> List:
    """Benchmark SAR image processing (SublookDecomposition)."""
    _section("SAR Image Processing")
    results = []
    mod = "image_processing.sar"

    _data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    umbra_dir = _data_dir / "umbra"
    sar_path = (_find_data_file(umbra_dir, "*.nitf")
                or _find_data_file(umbra_dir, "*.ntf"))
    if not sar_path:
        print("  SKIP  SublookDecomposition benchmarks (SICD data not found)")
        return results

    try:
        from grdl.IO.sar import SICDReader
        from grdl.image_processing.sar import SublookDecomposition

        with SICDReader(sar_path) as reader:
            metadata = reader.metadata
            shape = reader.get_shape()
            cx, cy = shape[0] // 2, shape[1] // 2
            half = min(rows // 2, shape[0] // 2, shape[1] // 2)
            chip = reader.read_chip(cx - half, cx + half,
                                    cy - half, cy + half)

        real_kw = dict(store=store, iterations=iterations, warmup=warmup,
                       tags={**tags, "data": "real"}, module=mod)

        for num_looks in (2, 3):
            decomp = SublookDecomposition(
                metadata=metadata, num_looks=num_looks,
            )
            chip_shape = f"{chip.shape[0]}x{chip.shape[1]}"

            r = _bench(
                f"SublookDecomposition.decompose.{num_looks}looks.{chip_shape}",
                decomp.decompose,
                setup=lambda: ((chip,), {}), **real_kw,
            )
            if r:
                results.append(r)

            looks = decomp.decompose(chip)
            r = _bench(
                f"SublookDecomposition.to_magnitude.{num_looks}looks",
                decomp.to_magnitude,
                setup=lambda: ((looks,), {}), **real_kw,
            )
            if r:
                results.append(r)

            r = _bench(
                f"SublookDecomposition.to_db.{num_looks}looks",
                decomp.to_db,
                setup=lambda: ((looks,), {}), **real_kw,
            )
            if r:
                results.append(r)

    except (ImportError, Exception) as exc:
        print(f"  SKIP  SublookDecomposition benchmarks ({exc})")

    # --- MultilookDecomposition ---
    try:
        from grdl.image_processing.sar import MultilookDecomposition

        if sar_path:
            from grdl.IO.sar import SICDReader as _SICDReader2

            with _SICDReader2(sar_path) as reader:
                _ml_meta = reader.metadata
                _ml_shape = reader.get_shape()
                _cx, _cy = _ml_shape[0] // 2, _ml_shape[1] // 2
                _half = min(rows // 2, _ml_shape[0] // 2, _ml_shape[1] // 2)
                _ml_chip = reader.read_chip(_cx - _half, _cx + _half,
                                            _cy - _half, _cy + _half)

            real_kw2 = dict(store=store, iterations=iterations, warmup=warmup,
                            tags={**tags, "data": "real"}, module=mod)

            for looks_rg, looks_az in [(2, 2), (3, 3)]:
                ml = MultilookDecomposition(
                    metadata=_ml_meta, looks_rg=looks_rg, looks_az=looks_az,
                )
                _c = _ml_chip
                r = _bench(
                    f"MultilookDecomposition.decompose.{looks_rg}x{looks_az}",
                    ml.decompose,
                    setup=lambda _c=_c: ((_c,), {}), **real_kw2,
                )
                if r:
                    results.append(r)

    except (ImportError, Exception) as exc:
        print(f"  SKIP  MultilookDecomposition benchmarks ({exc})")

    # --- CSIProcessor ---
    try:
        from grdl.image_processing.sar import CSIProcessor

        if sar_path:
            from grdl.IO.sar import SICDReader as _SICDReader3

            with _SICDReader3(sar_path) as reader:
                _csi_meta = reader.metadata
                _csi_shape = reader.get_shape()
                _cx, _cy = _csi_shape[0] // 2, _csi_shape[1] // 2
                _half = min(rows // 2, _csi_shape[0] // 2, _csi_shape[1] // 2)
                _csi_chip = reader.read_chip(_cx - _half, _cx + _half,
                                             _cy - _half, _cy + _half)

            csi = CSIProcessor(metadata=_csi_meta)
            _c = _csi_chip
            r = _bench("CSIProcessor.apply", csi.apply,
                        setup=lambda _c=_c: ((_c,), {}),
                        **dict(store=store, iterations=iterations, warmup=warmup,
                               tags={**tags, "data": "real"}, module=mod))
            if r:
                results.append(r)

    except (ImportError, Exception) as exc:
        print(f"  SKIP  CSIProcessor benchmarks ({exc})")

    return results


# ---------------------------------------------------------------------------
# Interpolation benchmarks
# ---------------------------------------------------------------------------
def run_interpolation_benchmarks(
    store: JSONBenchmarkStore,
    rows: int,
    cols: int,
    iterations: int,
    warmup: int,
    tags: Dict[str, str],
) -> List:
    """Benchmark interpolation algorithms (synthetic signals)."""
    _section("Interpolation")
    sz = tags["array_size"]
    results = []
    mod = "interpolation"

    kw = dict(store=store, iterations=iterations, warmup=warmup,
              tags=tags, module=mod)

    rng = np.random.default_rng(42)
    n_samples = rows * cols

    # Synthetic bandlimited signal
    x_old = np.arange(n_samples, dtype=np.float64)
    y_old = (rng.standard_normal(n_samples)
             + 1j * rng.standard_normal(n_samples)).astype(np.complex128)
    x_new = x_old[:-1] + rng.uniform(0.1, 0.9, size=n_samples - 1)

    # --- LanczosInterpolator ---
    try:
        from grdl.interpolation import lanczos_interpolator

        for a in (3, 5):
            interp = lanczos_interpolator(a=a)
            _xo, _yo, _xn = x_old, y_old, x_new
            r = _bench(
                f"LanczosInterpolator.a{a}.{sz}", interp,
                setup=lambda _xo=_xo, _yo=_yo, _xn=_xn: ((_xo, _yo, _xn), {}),
                version="1.0.0", **kw,
            )
            if r:
                results.append(r)

    except (ImportError, Exception) as exc:
        print(f"  SKIP  LanczosInterpolator benchmarks ({exc})")

    # --- KaiserSincInterpolator ---
    try:
        from grdl.interpolation import windowed_sinc_interpolator

        for kl in (8, 16):
            interp = windowed_sinc_interpolator(kernel_length=kl, beta=5.0)
            r = _bench(
                f"KaiserSincInterpolator.kl{kl}.{sz}", interp,
                setup=lambda _xo=x_old, _yo=y_old, _xn=x_new: ((_xo, _yo, _xn), {}),
                version="1.0.0", **kw,
            )
            if r:
                results.append(r)

    except (ImportError, Exception) as exc:
        print(f"  SKIP  KaiserSincInterpolator benchmarks ({exc})")

    # --- LagrangeInterpolator ---
    try:
        from grdl.interpolation import lagrange_interpolator

        for order in (3, 5):
            interp = lagrange_interpolator(order=order)
            r = _bench(
                f"LagrangeInterpolator.order{order}.{sz}", interp,
                setup=lambda _xo=x_old, _yo=y_old, _xn=x_new: ((_xo, _yo, _xn), {}),
                version="1.0.0", **kw,
            )
            if r:
                results.append(r)

    except (ImportError, Exception) as exc:
        print(f"  SKIP  LagrangeInterpolator benchmarks ({exc})")

    # --- FarrowInterpolator ---
    try:
        from grdl.interpolation import farrow_interpolator

        for fo, po in [(4, 3), (8, 5)]:
            interp = farrow_interpolator(filter_order=fo, poly_order=po)
            r = _bench(
                f"FarrowInterpolator.f{fo}_p{po}.{sz}", interp,
                setup=lambda _xo=x_old, _yo=y_old, _xn=x_new: ((_xo, _yo, _xn), {}),
                version="1.0.0", **kw,
            )
            if r:
                results.append(r)

    except (ImportError, Exception) as exc:
        print(f"  SKIP  FarrowInterpolator benchmarks ({exc})")

    # --- PolyphaseInterpolator ---
    try:
        from grdl.interpolation import polyphase_interpolator

        for kl, nph in [(8, 32), (16, 64)]:
            interp = polyphase_interpolator(kernel_length=kl, num_phases=nph)
            r = _bench(
                f"PolyphaseInterpolator.kl{kl}_ph{nph}.{sz}", interp,
                setup=lambda _xo=x_old, _yo=y_old, _xn=x_new: ((_xo, _yo, _xn), {}),
                version="1.0.0", **kw,
            )
            if r:
                results.append(r)

    except (ImportError, Exception) as exc:
        print(f"  SKIP  PolyphaseInterpolator benchmarks ({exc})")

    # --- ThiranDelayFilter ---
    try:
        from grdl.interpolation import thiran_delay

        signal_1d = y_old[:1024].real.astype(np.float64)
        # Thiran stability requires delay >= order - 0.5
        for delay, order in [(0.7, 1), (3.7, 3)]:
            r = _bench(
                f"ThiranDelayFilter.d{delay}_o{order}", thiran_delay,
                setup=lambda _s=signal_1d, _d=delay, _o=order: ((_s, _d, _o), {}),
                version="1.0.0", **kw,
            )
            if r:
                results.append(r)

    except (ImportError, Exception) as exc:
        print(f"  SKIP  ThiranDelayFilter benchmarks ({exc})")

    return results


# ---------------------------------------------------------------------------
# SAR Image Formation benchmarks
# ---------------------------------------------------------------------------
def run_image_formation_benchmarks(
    store: JSONBenchmarkStore,
    rows: int,
    cols: int,
    iterations: int,
    warmup: int,
    tags: Dict[str, str],
) -> List:
    """Benchmark SAR image formation algorithms (requires CPHD data)."""
    _section("SAR Image Formation")
    sz = tags["array_size"]
    results = []
    mod = "image_processing.sar.image_formation"
    kw = dict(store=store, iterations=iterations, warmup=warmup,
              tags=tags, module=mod)

    _data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    cphd_dir = _data_dir / "cphd"
    cphd_path = _find_data_file(cphd_dir, "*.cphd")

    if not cphd_path:
        print("  SKIP  Image formation benchmarks (CPHD data not found)")
        return results

    try:
        from grdl.IO.sar import CPHDReader
        from grdl.image_processing.sar import (
            CollectionGeometry, PolarGrid, PolarFormatAlgorithm,
            SubaperturePartitioner,
        )

        with CPHDReader(cphd_path) as reader:
            metadata = reader.metadata
            phase_data = reader.read_full()

        real_kw = dict(store=store, iterations=iterations, warmup=warmup,
                       tags={**tags, "data": "real"}, module=mod)

        # CollectionGeometry
        r = _bench("CollectionGeometry.init.real_data",
                    lambda: CollectionGeometry(metadata),
                    version="1.0.0", **real_kw)
        if r:
            results.append(r)

        geom = CollectionGeometry(metadata)

        # PolarGrid
        r = _bench("PolarGrid.init.real_data",
                    lambda: PolarGrid(geom),
                    version="1.0.0", **real_kw)
        if r:
            results.append(r)

        grid = PolarGrid(geom)

        # PFA
        _pd = phase_data
        _geom = geom
        r = _bench("PolarFormatAlgorithm.form_image.real_data",
                    lambda _p, geometry, _g=grid: (
                        PolarFormatAlgorithm(grid=_g).form_image(_p, geometry=geometry)
                    ),
                    setup=lambda _p=_pd, _g=_geom: ((_p,), {"geometry": _g}),
                    version="1.0.0", **real_kw)
        if r:
            results.append(r)

        # SubaperturePartitioner
        r = _bench("SubaperturePartitioner.init.real_data",
                    lambda _m=metadata: SubaperturePartitioner(metadata=_m),
                    version="1.0.0", **real_kw)
        if r:
            results.append(r)

        # RDA (may not be compatible with all data)
        try:
            from grdl.image_processing.sar import RangeDopplerAlgorithm

            r = _bench("RangeDopplerAlgorithm.form_image.real_data",
                        lambda _p, geometry, _m=metadata: (
                            RangeDopplerAlgorithm(metadata=_m).form_image(_p, geometry=geometry)
                        ),
                        setup=lambda _p=_pd, _g=_geom: ((_p,), {"geometry": _g}),
                        version="1.0.0", **real_kw)
            if r:
                results.append(r)
        except Exception as exc:
            print(f"  SKIP  RangeDopplerAlgorithm ({exc})")

        # StripmapPFA
        try:
            from grdl.image_processing.sar import StripmapPFA

            r = _bench("StripmapPFA.form_image.real_data",
                        lambda _p, geometry, _m=metadata: (
                            StripmapPFA(metadata=_m).form_image(_p, geometry=geometry)
                        ),
                        setup=lambda _p=_pd, _g=_geom: ((_p,), {"geometry": _g}),
                        version="1.0.0", **real_kw)
            if r:
                results.append(r)
        except Exception as exc:
            print(f"  SKIP  StripmapPFA ({exc})")

        # FastBackProjection
        try:
            from grdl.image_processing.sar import FastBackProjection

            r = _bench("FastBackProjection.form_image.real_data",
                        lambda _p, geometry, _m=metadata: (
                            FastBackProjection(metadata=_m).form_image(_p, geometry=geometry)
                        ),
                        setup=lambda _p=_pd, _g=_geom: ((_p,), {"geometry": _g}),
                        version="1.0.0", **real_kw)
            if r:
                results.append(r)
        except Exception as exc:
            print(f"  SKIP  FastBackProjection ({exc})")

    except (ImportError, Exception) as exc:
        print(f"  SKIP  Image formation benchmarks ({exc})")

    return results


# ---------------------------------------------------------------------------
# Workflow-level benchmark (ActiveBenchmarkRunner)
# ---------------------------------------------------------------------------
def run_workflow_benchmark(
    store: JSONBenchmarkStore,
    iterations: int,
    warmup: int,
    tags: Dict[str, str],
) -> Optional[Any]:
    """Run the comprehensive YAML workflow via ``ActiveBenchmarkRunner``.

    Requires grdl-runtime and SICD data.  Returns a
    ``BenchmarkRecord`` on success, ``None`` otherwise.
    """
    _section("Workflow-Level Benchmark (ActiveBenchmarkRunner)")

    if not YAML_PATH.exists():
        print(f"  SKIP  YAML workflow not found: {YAML_PATH}")
        return None

    _data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    umbra_dir = _data_dir / "umbra"
    sar_path = (_find_data_file(umbra_dir, "*.nitf")
                or _find_data_file(umbra_dir, "*.ntf"))
    if not sar_path:
        print("  SKIP  Workflow benchmark requires SICD data file")
        return None

    try:
        from grdl_rt.api import load_workflow
        from grdl_te.benchmarking import ActiveBenchmarkRunner, BenchmarkSource

        from grdl.IO.sar import SICDReader
        from grdl.data_prep import ChipExtractor
    except ImportError:
        print("  SKIP  Workflow benchmark (grdl-runtime not installed)")
        return None

    try:
        wf = load_workflow(str(YAML_PATH))

        with SICDReader(sar_path) as reader:
            shape = reader.get_shape()
            ext = ChipExtractor(nrows=shape[0], ncols=shape[1])
            region = ext.chip_at_point(
                shape[0] // 2, shape[1] // 2,
                row_width=2048, col_width=2048,
            )
            chip = reader.read_chip(
                region.row_start, region.row_end,
                region.col_start, region.col_end,
            )

        source = BenchmarkSource.from_array(chip)
        runner = ActiveBenchmarkRunner(
            workflow=wf,
            source=source,
            iterations=iterations,
            warmup=warmup,
            store=store,
            tags={**tags, "benchmark_level": "workflow"},
        )

        print(f"  Running workflow '{wf.name}' "
              f"({iterations} iterations, {warmup} warmup)...")
        record = runner.run(
            prefer_gpu=True,
            execution_context={"input_format": "SICD"},
        )

        print(f"  OK    Workflow complete")
        print(f"         wall={record.total_wall_time.mean:.2f}s  "
              f"cpu={record.total_cpu_time.mean:.2f}s  "
              f"steps={len(record.step_results)}")
        for step in record.step_results:
            name = step.processor_name.rsplit(".", 1)[-1]
            print(f"         {name:<35s}  "
                  f"wall={step.wall_time_s.mean:.4f}s  "
                  f"cpu={step.cpu_time_s.mean:.4f}s")

        return record

    except Exception as exc:
        print(f"  FAIL  Workflow benchmark: {exc}")
        return None


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------
def print_summary(all_results: List) -> None:
    """Print a sorted summary of all benchmark results."""
    _section("SUMMARY")

    if not all_results:
        print("  No benchmarks completed.")
        return

    sorted_results = sorted(
        all_results,
        key=lambda r: r.total_wall_time.mean,
        reverse=True,
    )

    print(f"\n  {'Benchmark':<60s}  {'Wall (s)':>10s}  "
          f"{'CPU (s)':>10s}  {'Mem (KB)':>10s}")
    print(f"  {'-' * 60}  {'-' * 10}  {'-' * 10}  {'-' * 10}")

    for record in sorted_results:
        print(
            f"  {record.workflow_name:<60s}  "
            f"{record.total_wall_time.mean:>10.4f}  "
            f"{record.total_cpu_time.mean:>10.4f}  "
            f"{record.total_peak_rss.mean / 1024:>10.0f}"
        )

    module_times: Dict[str, float] = {}
    for record in all_results:
        mod = record.tags.get("module", "unknown")
        module_times[mod] = (module_times.get(mod, 0.0)
                             + record.total_wall_time.mean)

    print(f"\n  Module Totals:")
    print(f"  {'Module':<40s}  {'Total Wall (s)':>14s}")
    print(f"  {'-' * 40}  {'-' * 14}")
    for mod, total in sorted(module_times.items(),
                             key=lambda x: x[1], reverse=True):
        print(f"  {mod:<40s}  {total:>14.2f}")

    print(f"\n  Total benchmarks: {len(all_results)}")
    total_wall = sum(r.total_wall_time.mean for r in all_results)
    print(f"  Total wall time:  {total_wall:.1f}s")


# ---------------------------------------------------------------------------
# Public API: run_suite()
# ---------------------------------------------------------------------------
#: All benchmark groups and their runner functions.
BENCHMARK_GROUPS = {
    "filters": run_filter_benchmarks,
    "intensity": run_intensity_benchmarks,
    "decomposition": run_decomposition_benchmarks,
    "detection": run_detection_benchmarks,
    "pipeline": run_pipeline_benchmarks,
    "data_prep": run_data_prep_benchmarks,
    "io": run_io_benchmarks,
    "geolocation": run_geolocation_benchmarks,
    "coregistration": run_coregistration_benchmarks,
    "ortho": run_ortho_benchmarks,
    "sar": run_sar_processing_benchmarks,
    "interpolation": run_interpolation_benchmarks,
    "image_formation": run_image_formation_benchmarks,
}


def run_suite(
    *,
    size: str = DEFAULT_SIZE,
    iterations: int = DEFAULT_ITERATIONS,
    warmup: int = DEFAULT_WARMUP,
    store_dir: Optional[Path] = None,
    only: Optional[List[str]] = None,
    skip_workflow: bool = False,
) -> List:
    """Run the full GRDL component benchmark suite.

    Parameters
    ----------
    size : str
        Array size preset — ``"small"``, ``"medium"``, or ``"large"``.
    iterations : int
        Number of measurement iterations per benchmark.
    warmup : int
        Number of discarded warmup iterations per benchmark.
    store_dir : Path, optional
        Directory for ``JSONBenchmarkStore``.  Defaults to
        ``<cwd>/.benchmarks/``.
    only : List[str], optional
        Run only the named benchmark groups.  Valid names:
        ``filters``, ``intensity``, ``decomposition``, ``detection``,
        ``pipeline``, ``data_prep``, ``io``, ``geolocation``,
        ``coregistration``, ``ortho``, ``sar``, ``workflow``.
    skip_workflow : bool
        If ``True``, skip the workflow-level benchmark.

    Returns
    -------
    List[BenchmarkRecord]
        All successfully completed benchmark records.
    """
    rows, cols = ARRAY_SIZES[size]
    store = JSONBenchmarkStore(base_dir=store_dir)
    tags = {"array_size": size, "rows": str(rows), "cols": str(cols)}
    run_all = only is None

    print("GRDL Component Benchmark Suite")
    print(f"  Array size:  {size} ({rows} x {cols})")
    print(f"  Iterations:  {iterations}")
    print(f"  Warmup:      {warmup}")
    print(f"  Store:       {store._base_dir}")

    all_results: List = []
    common = dict(store=store, rows=rows, cols=cols,
                  iterations=iterations, warmup=warmup, tags=tags)

    for group_name, runner_fn in BENCHMARK_GROUPS.items():
        if run_all or (only and group_name in only):
            all_results.extend(runner_fn(**common))

    if (run_all or (only and "workflow" in only)) and not skip_workflow:
        wf_result = run_workflow_benchmark(
            store=store, iterations=iterations, warmup=warmup, tags=tags,
        )
        if wf_result:
            all_results.append(wf_result)

    print_summary(all_results)
    return all_results
