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

ARRAY_SIZES = {
    "small": (512, 512),
    "medium": (2048, 2048),
    "large": (4096, 4096),
}

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

    # ------------------------------------------------------------------
    # SICD  (umbra/  — pattern: *.nitf or *.ntf per README)
    # ------------------------------------------------------------------
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
    else:
        print("  SKIP  SICDReader benchmarks (data file not found)")

    # ------------------------------------------------------------------
    # VIIRS  (viirs/  — pattern: V?P09GA*.h5 per README)
    # ------------------------------------------------------------------
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
    else:
        print("  SKIP  VIIRSReader benchmarks (data files not found)")

    # ------------------------------------------------------------------
    # Sentinel-2 JP2  (sentinel2/  — pattern per README)
    # ------------------------------------------------------------------
    sentinel2_dir = _data_dir / "sentinel2"
    jp2_path = _find_sentinel2_jp2(sentinel2_dir)
    if jp2_path:
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
        print("  SKIP  JP2Reader benchmarks (data files not found)")

    # ------------------------------------------------------------------
    # NITF  (any *.nitf / *.ntf across data/ — pattern per README)
    # ------------------------------------------------------------------
    try:
        from grdl.IO.nitf import NITFReader

        nitf_paths = (sorted(_data_dir.glob("**/*.nitf"))
                      + sorted(_data_dir.glob("**/*.ntf")))
        if nitf_paths:
            npath = nitf_paths[0]

            def _nitf_read_full():
                with _suppress_stderr(), NITFReader(npath) as reader:
                    return reader.read_full()

            r = _bench("NITFReader.read_full.real_data",
                        _nitf_read_full, **kw)
            if r:
                results.append(r)
        else:
            print("  SKIP  NITFReader benchmarks (data files not found)")

    except ImportError:
        print("  SKIP  NITFReader benchmarks (import failed)")


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
        from grdl_te.benchmarking import ActiveBenchmarkRunner

        from grdl.IO.sar import SICDReader
        from grdl.data_prep import ChipExtractor

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

        runner = ActiveBenchmarkRunner(
            workflow=wf,
            iterations=iterations,
            warmup=warmup,
            store=store,
            tags={**tags, "benchmark_level": "workflow"},
        )

        print(f"  Running workflow '{wf.name}' "
              f"({iterations} iterations, {warmup} warmup)...")
        record = runner.run(source=chip, prefer_gpu=True)

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

    except ImportError:
        print("  SKIP  Workflow benchmark (grdl-runtime not installed)")
        return None
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
    "pipeline": run_pipeline_benchmarks,
    "data_prep": run_data_prep_benchmarks,
    "io": run_io_benchmarks,
    "geolocation": run_geolocation_benchmarks,
    "coregistration": run_coregistration_benchmarks,
    "ortho": run_ortho_benchmarks,
    "sar": run_sar_processing_benchmarks,
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
        ``filters``, ``intensity``, ``decomposition``, ``pipeline``,
        ``data_prep``, ``io``, ``geolocation``, ``coregistration``,
        ``ortho``, ``sar``, ``workflow``.
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
