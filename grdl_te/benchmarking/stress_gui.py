# -*- coding: utf-8 -*-
"""
Stress Test GUI — interactive NiceGUI dashboard for GRDL component stress testing.

Provides a browser-based interface for GRDL developers to:

  1. Select any importable GRDL component from a categorised catalog.
  2. Configure stress parameters (concurrency, ramp steps, duration,
     payload size, timeout) via form controls.
  3. Run the stress test in a background thread and watch live progress.
  4. Save the resulting ``StressTestRecord`` as JSON.
  5. Load a previous report from a saved JSON path to re-use its config
     and compare it against a new run.
  6. Compare two saved ``StressTestRecord`` JSON files side-by-side with
     a saturation-curve chart, failure analysis table, and delta summary.

Limitation notes (components skipped due to real-data requirements):
  - ``SICDReader``, ``SICDWriter``, ``NITFReader``, ``SIDDReader``,
    ``CPHDReader``, ``CRSDReader``, ``Sentinel1SLCReader``,
    ``BIOMASSL1Reader``, ``TerraSARReader``, ``ASTERReader``,
    ``VIIRSReader``, ``Sentinel2Reader``, ``EONITFReader``:
    All SAR/EO readers that require real sensor data files which cannot be
    synthesised from numpy arrays alone.  Run the CLI stress test
    (``python -m grdl_te --stress-test``) after placing data files in the
    appropriate ``data/<sensor>/`` directory.
  - ``SublookDecomposition``, ``MultilookDecomposition``, ``CSIProcessor``,
    ``CollectionGeometry``, ``PolarGrid``, ``PolarFormatAlgorithm``,
    ``RangeDopplerAlgorithm``, ``StripmapPFA``, ``FastBackProjection``:
    Require SICD/CPHD metadata from real sensor data.
  - ``SICDGeolocation``, ``Sentinel1SLCGeolocation``:
    Require a live reader to build the geolocation model.
  - ``DTEDElevation``, ``GeoTIFFDEM``, ``GeoidCorrection``:
    Require real DEM/geoid data files.
  - Workflow stress testing is intentionally excluded from the GUI because
    workflows are infinitely composable and are better exercised via the
    ``WorkflowStressTester`` or the CLI.

All other publicly importable GRDL components are covered with synthetic
numpy data.

Dependencies
------------
nicegui
numpy
scipy (for coregistration synthetic data)
rasterio (optional — JP2 and GeoTIFF IO)
h5py (optional — HDF5 IO)
shapely (optional — Detection/DetectionSet)

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
2026-04-08
"""

# Standard library
import asyncio
import contextlib
import json
import shutil
import tempfile
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Third-party
import numpy as np
from nicegui import ui

# Internal
from grdl_te.benchmarking._formatting import (
    fmt_bytes as _fmt_bytes,
    fmt_time as _fmt_time,
)
from grdl_te.benchmarking.stress_models import (
    DEFAULT_DURATION_PER_STEP_S,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_RAMP_STEPS,
    DEFAULT_START_CONCURRENCY,
    DEFAULT_TIMEOUT_PER_CALL_S,
    StressTestConfig,
    StressTestRecord,
)
from grdl_te.benchmarking.stress_runner import ComponentStressTester
from grdl_te.benchmarking.stress_store import JSONStressTestStore
from grdl_te.benchmarking.stress_report import format_stress_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _suppress_stderr():
    """Redirect C-level stderr to /dev/null."""
    import os
    fd = 2
    old = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, fd)
        os.close(devnull)
        yield
    finally:
        os.dup2(old, fd)
        os.close(old)


# Root data directory (same as suite.py / conftest.py)
_GUI_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def _find_gui_data_file(directory: Path, pattern: str) -> Optional[Path]:
    """Return first file matching *pattern* in *directory*, or None."""
    if not directory.exists():
        return None
    matches = sorted(directory.glob(pattern))
    return matches[0] if matches else None


def _payload_shape(size: str) -> Tuple[int, int]:
    """Resolve a payload size string to (rows, cols)."""
    from grdl_te.benchmarking.source import ARRAY_SIZES
    if size in ARRAY_SIZES:
        return ARRAY_SIZES[size]
    # Accept ROWSxCOLS
    if "x" in size.lower():
        r, c = size.lower().split("x", 1)
        return int(r), int(c)
    return ARRAY_SIZES["medium"]


# ---------------------------------------------------------------------------
# Component Catalog
# ---------------------------------------------------------------------------
# Each entry is:
#   (display_name, group, factory_fn, note_if_unavailable)
#
# factory_fn() -> (fn, setup_or_None)
#   fn       : the callable to stress
#   setup    : callable(payload: np.ndarray) -> (args, kwargs), or None
#              when None, fn receives payload directly.
#
# If factory_fn raises ImportError or any exception, the component is shown
# as "unavailable" in the UI with a reason.

_CATALOG_ENTRIES: List[Dict[str, Any]] = []


def _reg(
    name: str,
    group: str,
    factory: Callable[[], Tuple[Callable, Optional[Callable]]],
    requires_real_data: bool = False,
    skip_reason: str = "",
) -> None:
    """Register a component in the catalog."""
    _CATALOG_ENTRIES.append(
        {
            "name": name,
            "group": group,
            "factory": factory,
            "requires_real_data": requires_real_data,
            "skip_reason": skip_reason,
        }
    )


# ── Filters ──────────────────────────────────────────────────────────────────

def _mk_mean_filter(ks: int):
    def _factory():
        from grdl.image_processing.filters import MeanFilter
        f = MeanFilter(kernel_size=ks)
        return f.apply, None
    return _factory

def _mk_gaussian_filter(sigma: float):
    def _factory():
        from grdl.image_processing.filters import GaussianFilter
        f = GaussianFilter(sigma=sigma)
        return f.apply, None
    return _factory

def _mk_median_filter(ks: int):
    def _factory():
        from grdl.image_processing.filters import MedianFilter
        f = MedianFilter(kernel_size=ks)
        return f.apply, None
    return _factory

def _mk_lee_filter(ks: int):
    def _factory():
        from grdl.image_processing.filters import LeeFilter
        f = LeeFilter(kernel_size=ks)
        return f.apply, None
    return _factory

def _mk_complex_lee_filter(ks: int):
    def _factory():
        from grdl.image_processing.filters import ComplexLeeFilter
        f = ComplexLeeFilter(kernel_size=ks)
        rng = np.random.default_rng(0)
        def setup(p):
            r, c = p.shape
            arr = (rng.standard_normal((r, c)) + 1j * rng.standard_normal((r, c))).astype(np.complex64)
            return (arr,), {}
        return f.apply, setup
    return _factory

def _mk_phase_gradient_filter(direction: str):
    def _factory():
        from grdl.image_processing.filters import PhaseGradientFilter
        f = PhaseGradientFilter(kernel_size=5, direction=direction)
        rng = np.random.default_rng(0)
        def setup(p):
            r, c = p.shape
            arr = (rng.standard_normal((r, c)) + 1j * rng.standard_normal((r, c))).astype(np.complex64)
            return (arr,), {}
        return f.apply, setup
    return _factory

_reg("MeanFilter.k3",      "Filters", _mk_mean_filter(3))
_reg("MeanFilter.k5",      "Filters", _mk_mean_filter(5))
_reg("MeanFilter.k7",      "Filters", _mk_mean_filter(7))
_reg("GaussianFilter.s0.5","Filters", _mk_gaussian_filter(0.5))
_reg("GaussianFilter.s1.0","Filters", _mk_gaussian_filter(1.0))
_reg("GaussianFilter.s2.0","Filters", _mk_gaussian_filter(2.0))
_reg("MedianFilter.k3",    "Filters", _mk_median_filter(3))
_reg("MedianFilter.k5",    "Filters", _mk_median_filter(5))

def _mk_min_filter():
    def _factory():
        from grdl.image_processing.filters import MinFilter
        f = MinFilter(kernel_size=3)
        return f.apply, None
    return _factory

def _mk_max_filter():
    def _factory():
        from grdl.image_processing.filters import MaxFilter
        f = MaxFilter(kernel_size=3)
        return f.apply, None
    return _factory

def _mk_stddev_filter():
    def _factory():
        from grdl.image_processing.filters import StdDevFilter
        f = StdDevFilter(kernel_size=5)
        return f.apply, None
    return _factory

_reg("MinFilter.k3",      "Filters", _mk_min_filter())
_reg("MaxFilter.k3",      "Filters", _mk_max_filter())
_reg("StdDevFilter.k5",   "Filters", _mk_stddev_filter())
_reg("LeeFilter.k5",      "Filters", _mk_lee_filter(5))
_reg("LeeFilter.k7",      "Filters", _mk_lee_filter(7))
_reg("ComplexLeeFilter.k5","Filters", _mk_complex_lee_filter(5))
_reg("ComplexLeeFilter.k7","Filters", _mk_complex_lee_filter(7))
_reg("PhaseGradientFilter.row",      "Filters", _mk_phase_gradient_filter("row"))
_reg("PhaseGradientFilter.col",      "Filters", _mk_phase_gradient_filter("col"))
_reg("PhaseGradientFilter.magnitude","Filters", _mk_phase_gradient_filter("magnitude"))

# ── Intensity Transforms ─────────────────────────────────────────────────────

def _mk_to_decibels(floor: float):
    def _factory():
        from grdl.image_processing.intensity import ToDecibels
        xf = ToDecibels(floor_db=floor)
        rng = np.random.default_rng(0)
        def setup(p):
            r, c = p.shape
            arr = (rng.standard_normal((r, c)) + 1j * rng.standard_normal((r, c))).astype(np.complex64)
            return (arr,), {}
        return xf.apply, setup
    return _factory

def _mk_percentile_stretch(plow: float, phigh: float):
    def _factory():
        from grdl.image_processing.intensity import PercentileStretch
        xf = PercentileStretch(plow=plow, phigh=phigh)
        rng = np.random.default_rng(0)
        def setup(p):
            r, c = p.shape
            arr = rng.standard_normal((r, c)).astype(np.float32) * 100.0
            return (arr,), {}
        return xf.apply, setup
    return _factory

_reg("ToDecibels.floor-60",        "Intensity", _mk_to_decibels(-60.0))
_reg("ToDecibels.floor-40",        "Intensity", _mk_to_decibels(-40.0))
_reg("PercentileStretch.p2-98",    "Intensity", _mk_percentile_stretch(2.0, 98.0))
_reg("PercentileStretch.p1-99",    "Intensity", _mk_percentile_stretch(1.0, 99.0))

# ── Decomposition ─────────────────────────────────────────────────────────────

def _mk_pauli_decompose():
    def _factory():
        from grdl.image_processing.decomposition import PauliDecomposition
        decomp = PauliDecomposition()
        rng = np.random.default_rng(0)
        def setup(p):
            r, c = p.shape
            def _c(seed_add):
                return (rng.standard_normal((r, c)) + 1j * rng.standard_normal((r, c))).astype(np.complex64)
            return (_c(0), _c(1), _c(2), _c(3)), {}
        return decomp.decompose, setup
    return _factory

def _mk_pauli_to_rgb():
    def _factory():
        from grdl.image_processing.decomposition import PauliDecomposition
        decomp = PauliDecomposition()
        rng = np.random.default_rng(0)
        def setup(p):
            r, c = p.shape
            shh = (rng.standard_normal((r, c)) + 1j * rng.standard_normal((r, c))).astype(np.complex64)
            shv = (rng.standard_normal((r, c)) + 1j * rng.standard_normal((r, c))).astype(np.complex64)
            svh = (rng.standard_normal((r, c)) + 1j * rng.standard_normal((r, c))).astype(np.complex64)
            svv = (rng.standard_normal((r, c)) + 1j * rng.standard_normal((r, c))).astype(np.complex64)
            components = decomp.decompose(shh, shv, svh, svv)
            return (components,), {}
        return decomp.to_rgb, setup
    return _factory

def _mk_dual_pol_halpha(ws: int):
    def _factory():
        from grdl.image_processing.decomposition import DualPolHAlpha
        ha = DualPolHAlpha(window_size=ws)
        rng = np.random.default_rng(0)
        def setup(p):
            r, c = p.shape
            co  = (rng.standard_normal((r, c)) + 1j * rng.standard_normal((r, c))).astype(np.complex64)
            cross = (rng.standard_normal((r, c)) * 0.3 + 1j * rng.standard_normal((r, c)) * 0.3).astype(np.complex64)
            return (co, cross), {}
        return ha.decompose, setup
    return _factory

def _mk_dual_pol_halpha_rgb(ws: int):
    def _factory():
        from grdl.image_processing.decomposition import DualPolHAlpha
        ha = DualPolHAlpha(window_size=ws)
        rng = np.random.default_rng(0)
        def setup(p):
            r, c = p.shape
            co  = (rng.standard_normal((r, c)) + 1j * rng.standard_normal((r, c))).astype(np.complex64)
            cross = (rng.standard_normal((r, c)) * 0.3 + 1j * rng.standard_normal((r, c)) * 0.3).astype(np.complex64)
            components = ha.decompose(co, cross)
            return (components,), {}
        return ha.to_rgb, setup
    return _factory

_reg("PauliDecomposition.decompose",    "Decomposition", _mk_pauli_decompose())
_reg("PauliDecomposition.to_rgb",       "Decomposition", _mk_pauli_to_rgb())
_reg("DualPolHAlpha.decompose.w7",      "Decomposition", _mk_dual_pol_halpha(7))
_reg("DualPolHAlpha.decompose.w11",     "Decomposition", _mk_dual_pol_halpha(11))
_reg("DualPolHAlpha.to_rgb.w7",         "Decomposition", _mk_dual_pol_halpha_rgb(7))

# ── Detection (CFAR) ─────────────────────────────────────────────────────────

def _mk_cfar_db_image():
    """Return a factory-setup helper that builds a dB image."""
    def setup(p):
        rng = np.random.default_rng(42)
        r, c = p.shape
        mag = np.abs(rng.standard_normal((r, c)) + 1j * rng.standard_normal((r, c))).astype(np.float64)
        db = 20.0 * np.log10(mag + 1e-10)
        peak = db.max()
        for rt, ct in [(r // 4, c // 4), (r // 2, c // 2), (r * 3 // 4, c * 3 // 4)]:
            db[rt - 2:rt + 3, ct - 2:ct + 3] = peak + 20.0
        return (db,), {}
    return setup

def _mk_ca_cfar(pfa: float):
    def _factory():
        from grdl.image_processing.detection.cfar import CACFARDetector
        det = CACFARDetector(guard_cells=3, training_cells=12, pfa=pfa)
        return det.detect, _mk_cfar_db_image()
    return _factory

def _mk_go_cfar():
    def _factory():
        from grdl.image_processing.detection.cfar import GOCFARDetector
        det = GOCFARDetector(guard_cells=3, training_cells=12, pfa=1e-3)
        return det.detect, _mk_cfar_db_image()
    return _factory

def _mk_so_cfar():
    def _factory():
        from grdl.image_processing.detection.cfar import SOCFARDetector
        det = SOCFARDetector(guard_cells=3, training_cells=12, pfa=1e-3)
        return det.detect, _mk_cfar_db_image()
    return _factory

def _mk_os_cfar():
    def _factory():
        from grdl.image_processing.detection.cfar import OSCFARDetector
        det = OSCFARDetector(guard_cells=3, training_cells=12, pfa=1e-3)
        return det.detect, _mk_cfar_db_image()
    return _factory

def _mk_detection_set_geojson():
    def _factory():
        from shapely.geometry import Point
        from grdl.image_processing.detection import Detection, DetectionSet
        detections = [
            Detection(pixel_geometry=Point(r * 50, c * 50), properties={"snr": float(i)}, confidence=0.9)
            for i, (r, c) in enumerate([(1, 2), (3, 4), (5, 6)])
        ]
        ds = DetectionSet(detections=detections, detector_name="CA-CFAR", detector_version="1.0.0")
        def run(_payload):
            return ds.to_geojson()
        return run, None
    return _factory

def _mk_transform_pixel_geometry():
    def _factory():
        from shapely.geometry import Point
        from grdl.transforms.detection import transform_pixel_geometry
        from grdl.coregistration.base import RegistrationResult
        import numpy as _np
        matrix = _np.eye(2, 3)
        matrix[0, 2] = 5.0
        matrix[1, 2] = 5.0
        result = RegistrationResult(transform_matrix=matrix, residual_rms=0.0, num_matches=4, inlier_ratio=1.0, metadata={})
        pt = Point(100.0, 50.0)
        def run(_payload):
            return transform_pixel_geometry(pt, result)
        return run, None
    return _factory

def _mk_transform_detection():
    def _factory():
        from shapely.geometry import Point
        from grdl.transforms.detection import transform_detection
        from grdl.coregistration.base import RegistrationResult
        from grdl.image_processing.detection import Detection
        import numpy as _np
        matrix = _np.eye(2, 3)
        matrix[0, 2] = 5.0
        matrix[1, 2] = 5.0
        result = RegistrationResult(transform_matrix=matrix, residual_rms=0.0, num_matches=4, inlier_ratio=1.0, metadata={})
        det = Detection(pixel_geometry=Point(100.0, 50.0), properties={}, confidence=0.9)
        def run(_payload):
            return transform_detection(det, result)
        return run, None
    return _factory

def _mk_transform_detection_set():
    def _factory():
        from shapely.geometry import Point
        from grdl.transforms.detection import transform_detection_set
        from grdl.coregistration.base import RegistrationResult
        from grdl.image_processing.detection import Detection, DetectionSet
        import numpy as _np
        matrix = _np.eye(2, 3)
        matrix[0, 2] = 5.0
        matrix[1, 2] = 5.0
        result = RegistrationResult(transform_matrix=matrix, residual_rms=0.0, num_matches=4, inlier_ratio=1.0, metadata={})
        detections = [Detection(pixel_geometry=Point(float(i * 20), float(i * 10)), properties={}, confidence=0.8) for i in range(5)]
        ds = DetectionSet(detections=detections, detector_name="test", detector_version="0.1.0")
        def run(_payload):
            return transform_detection_set(ds, result)
        return run, None
    return _factory

_reg("CACFARDetector.pfa1e-3",      "Detection", _mk_ca_cfar(1e-3))
_reg("CACFARDetector.pfa1e-4",      "Detection", _mk_ca_cfar(1e-4))
_reg("GOCFARDetector",              "Detection", _mk_go_cfar())
_reg("SOCFARDetector",              "Detection", _mk_so_cfar())
_reg("OSCFARDetector",              "Detection", _mk_os_cfar())
_reg("DetectionSet.to_geojson",     "Detection", _mk_detection_set_geojson())
_reg("transform_pixel_geometry",    "Detection", _mk_transform_pixel_geometry())
_reg("transform_detection",         "Detection", _mk_transform_detection())
_reg("transform_detection_set",     "Detection", _mk_transform_detection_set())

# ── Pipeline ──────────────────────────────────────────────────────────────────

def _mk_pipeline_4step():
    def _factory():
        from grdl.image_processing import Pipeline
        from grdl.image_processing.filters import GaussianFilter, MedianFilter
        from grdl.image_processing.intensity import PercentileStretch, ToDecibels
        pipe = Pipeline(steps=[
            ToDecibels(floor_db=-60.0),
            GaussianFilter(sigma=1.0),
            MedianFilter(kernel_size=3),
            PercentileStretch(plow=2.0, phigh=98.0),
        ])
        rng = np.random.default_rng(0)
        def setup(p):
            r, c = p.shape
            arr = (rng.standard_normal((r, c)) + 1j * rng.standard_normal((r, c))).astype(np.complex64)
            return (arr,), {}
        return pipe.apply, setup
    return _factory

_reg("Pipeline.4step", "Pipeline", _mk_pipeline_4step())

# ── Data Preparation ──────────────────────────────────────────────────────────

def _mk_chip_extractor_scalar():
    def _factory():
        def run(payload):
            from grdl.data_prep import ChipExtractor
            r, c = payload.shape
            ext = ChipExtractor(nrows=r, ncols=c)
            return ext.chip_at_point(r // 2, c // 2, row_width=256, col_width=256)
        return run, None
    return _factory

def _mk_chip_positions():
    def _factory():
        def run(payload):
            from grdl.data_prep import ChipExtractor
            r, c = payload.shape
            ext = ChipExtractor(nrows=r, ncols=c)
            return ext.chip_positions(row_width=256, col_width=256)
        return run, None
    return _factory

def _mk_tiler(ts: int, stride: Optional[int] = None):
    def _factory():
        def run(payload):
            from grdl.data_prep import Tiler
            r, c = payload.shape
            kw = {"tile_size": ts}
            if stride is not None:
                kw["stride"] = stride
            t = Tiler(nrows=r, ncols=c, **kw)
            return t.tile_positions()
        return run, None
    return _factory

def _mk_normalizer(method: str):
    def _factory():
        from grdl.data_prep import Normalizer
        norm = Normalizer(method=method)
        return norm.normalize, None
    return _factory

def _mk_normalizer_transform():
    def _factory():
        from grdl.data_prep import Normalizer
        norm = Normalizer(method="minmax")
        # We need to fit first; do it once here with a small sample
        sample = np.random.rand(256, 256).astype(np.float32)
        norm.fit(sample)
        return norm.transform, None
    return _factory

_reg("ChipExtractor.chip_at_point",      "DataPrep", _mk_chip_extractor_scalar())
_reg("ChipExtractor.chip_positions",     "DataPrep", _mk_chip_positions())
_reg("Tiler.t256",                       "DataPrep", _mk_tiler(256))
_reg("Tiler.t512",                       "DataPrep", _mk_tiler(512))
_reg("Tiler.t256_s128",                  "DataPrep", _mk_tiler(256, 128))
_reg("Normalizer.minmax",                "DataPrep", _mk_normalizer("minmax"))
_reg("Normalizer.zscore",                "DataPrep", _mk_normalizer("zscore"))
_reg("Normalizer.percentile",            "DataPrep", _mk_normalizer("percentile"))
_reg("Normalizer.unit_norm",             "DataPrep", _mk_normalizer("unit_norm"))
_reg("Normalizer.transform.minmax",      "DataPrep", _mk_normalizer_transform())

# ── IO (synthetic data only) ──────────────────────────────────────────────────

def _mk_numpy_writer():
    def _factory():
        from grdl.IO.numpy_io import NumpyWriter
        tmpdir = Path(tempfile.mkdtemp(prefix="grdl_stress_"))
        npy = tmpdir / "bench.npy"
        def run(payload):
            NumpyWriter(npy).write(payload.astype(np.float32))
        def cleanup():
            shutil.rmtree(tmpdir, ignore_errors=True)
        return run, None
    return _factory

def _mk_geotiff_write():
    def _factory():
        from rasterio.transform import from_bounds
        from grdl.IO.geotiff import GeoTIFFWriter
        tmpdir = Path(tempfile.mkdtemp(prefix="grdl_stress_"))
        def run(payload):
            r, c = payload.shape
            geo = {"crs": "EPSG:4326", "transform": from_bounds(0, 0, 1, 1, c, r)}
            out = tmpdir / "bench.tif"
            GeoTIFFWriter(out).write(payload.astype(np.float32), geolocation=geo)
        return run, None
    return _factory

def _mk_geotiff_read():
    def _factory():
        from rasterio.transform import from_bounds
        from grdl.IO.geotiff import GeoTIFFReader, GeoTIFFWriter
        tmpdir = Path(tempfile.mkdtemp(prefix="grdl_stress_"))
        # Pre-write once so reads are consistent
        sample = np.random.rand(512, 512).astype(np.float32)
        geo = {"crs": "EPSG:4326", "transform": from_bounds(0, 0, 1, 1, 512, 512)}
        tif = tmpdir / "bench.tif"
        GeoTIFFWriter(tif).write(sample, geolocation=geo)
        def run(_payload):
            reader = GeoTIFFReader(tif)
            arr = reader.read_full()
            reader.close()
            return arr
        return run, None
    return _factory

def _mk_hdf5_write():
    def _factory():
        from grdl.IO.hdf5 import HDF5Writer
        tmpdir = Path(tempfile.mkdtemp(prefix="grdl_stress_"))
        h5_path = tmpdir / "bench.h5"
        def run(payload):
            if h5_path.exists():
                h5_path.unlink()
            HDF5Writer(h5_path).write(payload.astype(np.float32))
        return run, None
    return _factory

def _mk_hdf5_read():
    def _factory():
        from grdl.IO.hdf5 import HDF5Reader, HDF5Writer
        tmpdir = Path(tempfile.mkdtemp(prefix="grdl_stress_"))
        h5_path = tmpdir / "bench.h5"
        sample = np.random.rand(512, 512).astype(np.float32)
        HDF5Writer(h5_path).write(sample)
        def run(_payload):
            reader = HDF5Reader(h5_path)
            arr = reader.read_full()
            reader.close()
            return arr
        return run, None
    return _factory

def _mk_png_write():
    def _factory():
        from grdl.IO.png import PngWriter
        tmpdir = Path(tempfile.mkdtemp(prefix="grdl_stress_"))
        png_path = tmpdir / "bench.png"
        def run(payload):
            arr = (np.abs(payload[:, :, 0]) * 255).astype(np.uint8) if payload.ndim == 3 else (np.clip(payload, 0, 1) * 255).astype(np.uint8)
            PngWriter(png_path).write(arr)
        return run, None
    return _factory

def _mk_nitf_write():
    def _factory():
        from grdl.IO.nitf import NITFWriter
        tmpdir = Path(tempfile.mkdtemp(prefix="grdl_stress_"))
        nitf_path = tmpdir / "bench.nitf"
        def run(payload):
            NITFWriter(nitf_path).write(payload.astype(np.float32))
        return run, None
    return _factory

_reg("NumpyWriter.write",      "IO",  _mk_numpy_writer())
_reg("GeoTIFFWriter.write",    "IO",  _mk_geotiff_write())
_reg("GeoTIFFReader.read_full","IO",  _mk_geotiff_read())
_reg("HDF5Writer.write",       "IO",  _mk_hdf5_write())
_reg("HDF5Reader.read_full",   "IO",  _mk_hdf5_read())
_reg("PngWriter.write",        "IO",  _mk_png_write())
_reg("NITFWriter.write",       "IO",  _mk_nitf_write())

# ── IO — real-data readers (check file presence at factory time) ──────────────

def _mk_sicd_reader():
    def _factory():
        sar_path = (
            _find_gui_data_file(_GUI_DATA_DIR / "umbra", "*.nitf") or
            _find_gui_data_file(_GUI_DATA_DIR / "umbra", "*.ntf")
        )
        if sar_path is None:
            raise ImportError("No SICD file found in data/umbra/ — download from Umbra Open Data.")
        from grdl.IO.sar import SICDReader as _SR
        def run(_payload):
            with _suppress_stderr(), _SR(sar_path) as reader:
                return reader.read_full()
        return run, None
    return _factory

def _mk_sicd_writer_real():
    def _factory():
        import copy
        sar_path = (
            _find_gui_data_file(_GUI_DATA_DIR / "umbra", "*.nitf") or
            _find_gui_data_file(_GUI_DATA_DIR / "umbra", "*.ntf")
        )
        if sar_path is None:
            raise ImportError("No SICD file found in data/umbra/ — SICDWriter needs real SICD metadata.")
        from grdl.IO.sar import SICDReader as _SR, SICDWriter as _SW
        with _suppress_stderr(), _SR(sar_path) as reader:
            _meta0 = reader.metadata
            shape = reader.get_shape()
            cx, cy = shape[0] // 2, shape[1] // 2
            half = min(256, shape[0] // 2, shape[1] // 2)
            chip = reader.read_chip(cx - half, cx + half, cy - half, cy + half)
        chip = chip.astype(np.complex64, copy=False)
        _cm = copy.deepcopy(_meta0)
        _cm.rows, _cm.cols = chip.shape[0], chip.shape[1]
        _cm.image_data.num_rows = chip.shape[0]
        _cm.image_data.num_cols = chip.shape[1]
        _cm.image_data.first_row = 0
        _cm.image_data.first_col = 0
        _cm.image_data.scp_pixel.row = chip.shape[0] // 2
        _cm.image_data.scp_pixel.col = chip.shape[1] // 2
        if _cm.image_data.full_image:
            _cm.image_data.full_image.num_rows = chip.shape[0]
            _cm.image_data.full_image.num_cols = chip.shape[1]
        tmpdir = Path(tempfile.mkdtemp(prefix="grdl_stress_sicd_"))
        out_path = tmpdir / "bench.nitf"
        def run(_payload):
            _SW(out_path, metadata=_cm).write(chip)
        return run, None
    return _factory

def _mk_nitf_reader_real():
    def _factory():
        sar_path = (
            _find_gui_data_file(_GUI_DATA_DIR / "umbra", "*.nitf") or
            _find_gui_data_file(_GUI_DATA_DIR / "umbra", "*.ntf")
        )
        if sar_path is None:
            raise ImportError("No NITF file found in data/umbra/ — download from Umbra Open Data.")
        from grdl.IO.nitf import NITFReader as _NR
        def run(_payload):
            with _suppress_stderr(), _NR(sar_path) as reader:
                return reader.read_full()
        return run, None
    return _factory

def _mk_sidd_reader():
    def _factory():
        sidd_path = (
            _find_gui_data_file(_GUI_DATA_DIR / "sidd", "*.nitf") or
            _find_gui_data_file(_GUI_DATA_DIR / "sidd", "*.ntf")
        )
        if sidd_path is None:
            raise ImportError("No SIDD file found in data/sidd/ — requires SIDD NITF.")
        from grdl.IO.sar import SIDDReader as _SDR
        def run(_payload):
            with _suppress_stderr(), _SDR(sidd_path) as reader:
                return reader.read_full()
        return run, None
    return _factory

def _mk_cphd_reader():
    def _factory():
        cphd_path = _find_gui_data_file(_GUI_DATA_DIR / "cphd", "*.cphd")
        if cphd_path is None:
            raise ImportError("No CPHD file found in data/cphd/ — requires CPHD sensor data.")
        from grdl.IO.sar import CPHDReader as _CR
        def run(_payload):
            with _CR(cphd_path) as reader:
                return reader.read_full()
        return run, None
    return _factory

def _mk_crsd_reader():
    def _factory():
        crsd_path = _find_gui_data_file(_GUI_DATA_DIR / "crsd", "*.crsd")
        if crsd_path is None:
            raise ImportError("No CRSD file found in data/crsd/ — requires CRSD sensor data.")
        from grdl.IO.sar import CRSDReader as _CRSDr
        def run(_payload):
            with _CRSDr(crsd_path) as reader:
                return reader.read_full()
        return run, None
    return _factory

def _mk_sentinel1_slc_reader():
    def _factory():
        s1_path = _find_gui_data_file(_GUI_DATA_DIR / "sentinel1", "*.SAFE")
        if s1_path is None:
            raise ImportError("No Sentinel-1 SAFE found in data/sentinel1/ — download from Copernicus.")
        from grdl.IO.sar import Sentinel1SLCReader as _S1R
        def run(_payload):
            with _S1R(s1_path) as reader:
                return reader.read_full()
        return run, None
    return _factory

def _mk_viirs_reader():
    def _factory():
        vpath = (
            _find_gui_data_file(_GUI_DATA_DIR / "viirs", "V?P09GA*.h5") or
            _find_gui_data_file(_GUI_DATA_DIR / "viirs", "V?P09GA*.hdf5")
        )
        if vpath is None:
            raise ImportError("No VIIRS VNP09GA file found in data/viirs/ — download from LAADS DAAC.")
        from grdl.IO.multispectral import VIIRSReader as _VR
        def run(_payload):
            with _VR(vpath) as reader:
                return reader.read_full()
        return run, None
    return _factory

def _mk_hdf5_reader_viirs():
    def _factory():
        vpath = (
            _find_gui_data_file(_GUI_DATA_DIR / "viirs", "V?P09GA*.h5") or
            _find_gui_data_file(_GUI_DATA_DIR / "viirs", "V?P09GA*.hdf5")
        )
        if vpath is None:
            raise ImportError("No VIIRS HDF5 found in data/viirs/ — download from LAADS DAAC.")
        from grdl.IO.hdf5 import HDF5Reader as _H5R
        def run(_payload):
            with _H5R(vpath) as reader:
                return reader.read_full()
        return run, None
    return _factory

def _mk_sentinel2_reader():
    def _factory():
        jp2_path = None
        s2_dir = _GUI_DATA_DIR / "sentinel2"
        if s2_dir.exists():
            hits = sorted(s2_dir.glob("T*_B*.jp2"))
            if hits:
                jp2_path = hits[0]
            else:
                for safe in sorted(s2_dir.glob("S2*.SAFE")):
                    jp2s = sorted(safe.glob("**/IMG_DATA/**/*_B04*.jp2"))
                    if jp2s:
                        jp2_path = jp2s[0]
                        break
        if jp2_path is None:
            raise ImportError("No Sentinel-2 JP2 found in data/sentinel2/ — download from Copernicus.")
        from grdl.IO.eo import Sentinel2Reader as _S2R
        def run(_payload):
            with _S2R(jp2_path) as reader:
                return reader.read_full()
        return run, None
    return _factory

def _mk_jp2_reader_real():
    def _factory():
        jp2_path = None
        s2_dir = _GUI_DATA_DIR / "sentinel2"
        if s2_dir.exists():
            hits = sorted(s2_dir.glob("T*_B*.jp2"))
            if hits:
                jp2_path = hits[0]
            else:
                for safe in sorted(s2_dir.glob("S2*.SAFE")):
                    jp2s = sorted(safe.glob("**/IMG_DATA/**/*_B04*.jp2"))
                    if jp2s:
                        jp2_path = jp2s[0]
                        break
        if jp2_path is None:
            raise ImportError("No JPEG2000 file found in data/sentinel2/ — download from Copernicus.")
        from grdl.IO.jpeg2000 import JP2Reader as _JP2R
        def run(_payload):
            with _JP2R(jp2_path) as reader:
                return reader.read_full()
        return run, None
    return _factory

def _mk_aster_reader():
    def _factory():
        aster_path = _find_gui_data_file(_GUI_DATA_DIR / "aster", "AST_L1T*.tif")
        if aster_path is None:
            raise ImportError("No ASTER L1T file found in data/aster/ — download from USGS GLOVIS.")
        from grdl.IO.ir import ASTERReader as _AR
        def run(_payload):
            with _AR(aster_path) as reader:
                return reader.read_full()
        return run, None
    return _factory

def _mk_biomass_reader():
    def _factory():
        bio_path = _find_gui_data_file(_GUI_DATA_DIR / "biomass", "BIO_S*")
        if bio_path is None:
            raise ImportError("No BIOMASS L1 file found in data/biomass/ — requires ESA BIOMASS mission data.")
        from grdl.IO.sar import BIOMASSL1Reader as _BR
        def run(_payload):
            with _BR(bio_path) as reader:
                return reader.read_full()
        return run, None
    return _factory

def _mk_terrasar_reader():
    def _factory():
        tsx_path = None
        ts_dir = _GUI_DATA_DIR / "terrasar"
        if ts_dir.exists():
            for candidate in sorted(ts_dir.iterdir()):
                if candidate.is_dir() and (
                    candidate.name.startswith("TSX1_") or candidate.name.startswith("TDX1_")
                ):
                    tsx_path = candidate
                    break
            if tsx_path is None:
                xmls = sorted(ts_dir.glob("TSX1_SAR__*.xml"))
                if xmls:
                    tsx_path = ts_dir
        if tsx_path is None:
            raise ImportError("No TerraSAR-X product found in data/terrasar/ — requires TSX1_* directory.")
        from grdl.IO.sar import TerraSARReader as _TR
        def run(_payload):
            with _TR(tsx_path) as reader:
                return reader.read_full()
        return run, None
    return _factory

def _mk_eo_nitf_reader():
    def _factory():
        eo_path = (
            _find_gui_data_file(_GUI_DATA_DIR / "eo_nitf", "*.nitf") or
            _find_gui_data_file(_GUI_DATA_DIR / "eo_nitf", "*.ntf")
        )
        if eo_path is None:
            raise ImportError("No EO NITF found in data/eo_nitf/ — requires NITF with RPC/RSM metadata.")
        from grdl.IO.eo import EONITFReader as _EOR
        def run(_payload):
            with _suppress_stderr(), _EOR(eo_path) as reader:
                return reader.read_full()
        return run, None
    return _factory

_reg("SICDReader",                  "IO", _mk_sicd_reader())
_reg("SICDWriter",                  "IO", _mk_sicd_writer_real())
_reg("NITFReader (real data)",      "IO", _mk_nitf_reader_real())
_reg("SIDDReader",                  "IO", _mk_sidd_reader())
_reg("CPHDReader",                  "IO", _mk_cphd_reader())
_reg("CRSDReader",                  "IO", _mk_crsd_reader())
_reg("Sentinel1SLCReader",          "IO", _mk_sentinel1_slc_reader())
_reg("BIOMASSL1Reader",             "IO", _mk_biomass_reader())
_reg("TerraSARReader",              "IO", _mk_terrasar_reader())
_reg("ASTERReader",                 "IO", _mk_aster_reader())
_reg("VIIRSReader",                 "IO", _mk_viirs_reader())
_reg("HDF5Reader (real VIIRS)",     "IO", _mk_hdf5_reader_viirs())
_reg("Sentinel2Reader",             "IO", _mk_sentinel2_reader())
_reg("JP2Reader (real Sentinel-2)", "IO", _mk_jp2_reader_real())
_reg("EONITFReader",                "IO", _mk_eo_nitf_reader())

# ── Geolocation ───────────────────────────────────────────────────────────────

def _mk_affine_geo_image_to_latlon():
    def _factory():
        from rasterio.transform import Affine
        from grdl.geolocation import AffineGeolocation
        transform = Affine(0.00027, 0.0, -118.0, 0.0, -0.00027, 34.0)
        # Shape will be set per-payload; use large enough defaults
        geo = AffineGeolocation(transform=transform, shape=(4096, 4096), crs="EPSG:4326")
        def run(_payload):
            r, c = _payload.shape
            row_arr = np.random.uniform(0, r, size=1000)
            col_arr = np.random.uniform(0, c, size=1000)
            return geo.image_to_latlon(row_arr, col_arr)
        return run, None
    return _factory

def _mk_affine_geo_latlon_to_image():
    def _factory():
        from rasterio.transform import Affine
        from grdl.geolocation import AffineGeolocation
        transform = Affine(0.00027, 0.0, -118.0, 0.0, -0.00027, 34.0)
        geo = AffineGeolocation(transform=transform, shape=(4096, 4096), crs="EPSG:4326")
        rng = np.random.default_rng(0)
        def run(_payload):
            lats = rng.uniform(33.5, 34.5, size=1000)
            lons = rng.uniform(-118.5, -117.5, size=1000)
            return geo.latlon_to_image(lats, lons)
        return run, None
    return _factory

def _mk_gcp_geo_image_to_latlon():
    def _factory():
        from grdl.geolocation import GCPGeolocation
        gcps = []
        for gi in range(5):
            for gj in range(5):
                row = int(gi * 1023 / 4)
                col = int(gj * 1023 / 4)
                gcps.append((-118.0 + gj * 0.1, 34.0 + gi * 0.1, 100.0, row, col))
        geo = GCPGeolocation(gcps=gcps, shape=(1024, 1024))
        rng = np.random.default_rng(0)
        def run(_payload):
            r, c = _payload.shape
            row_arr = rng.uniform(0, min(r, 1024), size=1000)
            col_arr = rng.uniform(0, min(c, 1024), size=1000)
            return geo.image_to_latlon(row_arr, col_arr)
        return run, None
    return _factory

def _mk_constant_elevation():
    def _factory():
        from grdl.geolocation import ConstantElevation
        elev = ConstantElevation(height=100.0)
        rng = np.random.default_rng(0)
        def run(_payload):
            lats = rng.uniform(33.0, 35.0, size=10000)
            lons = rng.uniform(-119.0, -117.0, size=10000)
            return elev.get_elevation(lats, lons)
        return run, None
    return _factory

def _mk_rpc_geolocation():
    def _factory():
        from grdl.IO.models.eo_nitf import RPCCoefficients
        from grdl.geolocation.eo.rpc import RPCGeolocation
        line_num = np.zeros(20); line_num[2] = 1.0
        line_den = np.zeros(20); line_den[0] = 1.0
        samp_num = np.zeros(20); samp_num[1] = 1.0
        samp_den = np.zeros(20); samp_den[0] = 1.0
        rpc = RPCCoefficients(
            line_off=2048.0, samp_off=2048.0, lat_off=37.0, long_off=-122.0,
            height_off=100.0, line_scale=2048.0, samp_scale=2048.0,
            lat_scale=0.05, long_scale=0.05, height_scale=500.0,
            line_num_coef=line_num, line_den_coef=line_den,
            samp_num_coef=samp_num, samp_den_coef=samp_den,
        )
        geo = RPCGeolocation(rpc=rpc, shape=(4096, 4096))
        rng = np.random.default_rng(0)
        def run(_payload):
            r, c = _payload.shape
            row_arr = rng.uniform(0, r, size=200)
            col_arr = rng.uniform(0, c, size=200)
            return geo.image_to_latlon(row_arr, col_arr)
        return run, None
    return _factory

_reg("AffineGeolocation.image_to_latlon", "Geolocation", _mk_affine_geo_image_to_latlon())
_reg("AffineGeolocation.latlon_to_image", "Geolocation", _mk_affine_geo_latlon_to_image())
_reg("GCPGeolocation.image_to_latlon",    "Geolocation", _mk_gcp_geo_image_to_latlon())
_reg("ConstantElevation.get_elevation",   "Geolocation", _mk_constant_elevation())
_reg("RPCGeolocation.image_to_latlon",    "Geolocation", _mk_rpc_geolocation())

# ── Geolocation — real-data (check file presence) ────────────────────────────

def _mk_sicd_geolocation():
    def _factory():
        sar_path = (
            _find_gui_data_file(_GUI_DATA_DIR / "umbra", "*.nitf") or
            _find_gui_data_file(_GUI_DATA_DIR / "umbra", "*.ntf")
        )
        if sar_path is None:
            raise ImportError("No SICD file in data/umbra/ — SICDGeolocation requires a SICD reader.")
        from grdl.IO.sar import SICDReader as _SR
        from grdl.geolocation import SICDGeolocation
        rng = np.random.default_rng(0)
        with _suppress_stderr(), _SR(sar_path) as reader:
            geo = SICDGeolocation.from_reader(reader)
            shape = reader.get_shape()
        def run(_payload):
            row_arr = rng.uniform(0, shape[0], size=500)
            col_arr = rng.uniform(0, shape[1], size=500)
            return geo.image_to_latlon(row_arr, col_arr)
        return run, None
    return _factory

def _mk_sentinel1_slc_geolocation():
    def _factory():
        s1_path = _find_gui_data_file(_GUI_DATA_DIR / "sentinel1", "*.SAFE")
        if s1_path is None:
            raise ImportError("No Sentinel-1 SAFE in data/sentinel1/ — download from Copernicus.")
        from grdl.IO.sar import Sentinel1SLCReader as _S1R
        from grdl.geolocation import Sentinel1SLCGeolocation
        rng = np.random.default_rng(0)
        with _S1R(s1_path) as reader:
            geo = Sentinel1SLCGeolocation.from_reader(reader)
            shape = reader.get_shape()
        def run(_payload):
            row_arr = rng.uniform(0, shape[0], size=500)
            col_arr = rng.uniform(0, shape[1], size=500)
            return geo.image_to_latlon(row_arr, col_arr)
        return run, None
    return _factory

def _mk_dted_elevation():
    def _factory():
        dted_dir = _GUI_DATA_DIR / "dted"
        if not dted_dir.exists() or not list(dted_dir.glob("**/*.dt?")):
            raise ImportError("No DTED tiles found in data/dted/ — download from USGS or NGA.")
        from grdl.geolocation.elevation import DTEDElevation
        elev = DTEDElevation(str(dted_dir))
        rng = np.random.default_rng(0)
        def run(_payload):
            lats = rng.uniform(33.0, 35.0, size=5000)
            lons = rng.uniform(-119.0, -117.0, size=5000)
            return elev.get_elevation(lats, lons)
        return run, None
    return _factory

def _mk_geotiff_dem():
    def _factory():
        dem_path = _find_gui_data_file(_GUI_DATA_DIR / "dem", "*.tif")
        if dem_path is None:
            raise ImportError("No DEM GeoTIFF found in data/dem/ — download from USGS SRTM.")
        from grdl.geolocation.elevation import GeoTIFFDEM
        elev = GeoTIFFDEM(dem_path)
        rng = np.random.default_rng(0)
        def run(_payload):
            lats = rng.uniform(33.0, 35.0, size=5000)
            lons = rng.uniform(-119.0, -117.0, size=5000)
            return elev.get_elevation(lats, lons)
        return run, None
    return _factory

def _mk_geoid_correction():
    def _factory():
        geoid_path = _find_gui_data_file(_GUI_DATA_DIR / "geoid", "*.pgm")
        if geoid_path is None:
            raise ImportError("No geoid .pgm file found in data/geoid/ — requires EGM96/EGM2008 PGM.")
        from grdl.geolocation.elevation import GeoidCorrection
        geoid = GeoidCorrection(geoid_path)
        rng = np.random.default_rng(0)
        def run(_payload):
            lats = rng.uniform(33.0, 35.0, size=5000)
            lons = rng.uniform(-119.0, -117.0, size=5000)
            return geoid.get_undulation(lats, lons)
        return run, None
    return _factory

def _mk_nisar_geolocation():
    def _factory():
        nisar_dir = _GUI_DATA_DIR / "nisar"
        nisar_path = (
            _find_gui_data_file(nisar_dir, "*.h5") or
            _find_gui_data_file(nisar_dir, "*.hdf5")
        )
        if nisar_path is None:
            raise ImportError("No NISAR HDF5 file found in data/nisar/ — download from ASF DAAC.")
        from grdl.IO.sar import NISARReader as _NR
        from grdl.geolocation import NISARGeolocation
        rng = np.random.default_rng(0)
        with _NR(nisar_path) as reader:
            geo = NISARGeolocation.from_reader(reader)
            shape = reader.get_shape()
        def run(_payload):
            row_arr = rng.uniform(0, shape[0], size=500)
            col_arr = rng.uniform(0, shape[1], size=500)
            return geo.image_to_latlon(row_arr, col_arr)
        return run, None
    return _factory

_reg("SICDGeolocation",           "Geolocation", _mk_sicd_geolocation())
_reg("Sentinel1SLCGeolocation",   "Geolocation", _mk_sentinel1_slc_geolocation())
_reg("DTEDElevation",             "Geolocation", _mk_dted_elevation())
_reg("GeoTIFFDEM",                "Geolocation", _mk_geotiff_dem())
_reg("GeoidCorrection",           "Geolocation", _mk_geoid_correction())
_reg("NISARGeolocation",          "Geolocation", _mk_nisar_geolocation())

# ── Coregistration ────────────────────────────────────────────────────────────

def _mk_affine_coreg_estimate():
    def _factory():
        from grdl.coregistration import AffineCoRegistration
        pts_f = np.array([[128, 128], [128, 384], [384, 128], [384, 384], [256, 256]], dtype=np.float64)
        pts_m = pts_f + 3.0
        coreg = AffineCoRegistration(control_points_fixed=pts_f, control_points_moving=pts_m)
        def run(payload):
            r, c = payload.shape
            from scipy.ndimage import affine_transform as _at
            mat = np.eye(2)
            moving = _at(payload, mat, offset=np.array([3.0, 3.0]), order=1)
            return coreg.estimate(payload, moving)
        return run, None
    return _factory

def _mk_affine_coreg_apply():
    def _factory():
        from grdl.coregistration import AffineCoRegistration
        from scipy.ndimage import affine_transform as _at
        pts_f = np.array([[128, 128], [128, 384], [384, 128], [384, 384], [256, 256]], dtype=np.float64)
        pts_m = pts_f + 3.0
        coreg = AffineCoRegistration(control_points_fixed=pts_f, control_points_moving=pts_m)
        # Pre-compute result once
        sample = np.random.rand(512, 512).astype(np.float32)
        moving_s = _at(sample, np.eye(2), offset=np.array([3.0, 3.0]), order=1)
        result_obj = coreg.estimate(sample, moving_s)
        def run(payload):
            r, c = payload.shape
            moving = _at(payload, np.eye(2), offset=np.array([3.0, 3.0]), order=1)
            return coreg.apply(moving, result_obj)
        return run, None
    return _factory

def _mk_feature_match_coreg():
    def _factory():
        from grdl.coregistration import FeatureMatchCoRegistration
        from scipy.ndimage import affine_transform as _at
        coreg = FeatureMatchCoRegistration(method="orb", max_features=5000)
        def run(payload):
            moving = _at(payload, np.eye(2), offset=np.array([3.0, 3.0]), order=1)
            return coreg.estimate(payload, moving)
        return run, None
    return _factory

def _mk_projective_coreg_estimate():
    def _factory():
        from grdl.coregistration import ProjectiveCoRegistration
        from scipy.ndimage import affine_transform as _at
        pts_f = np.array([[128, 128], [128, 384], [384, 128], [384, 384], [256, 256], [192, 256]], dtype=np.float64)
        pts_m = pts_f + 3.0
        coreg = ProjectiveCoRegistration(control_points_fixed=pts_f, control_points_moving=pts_m)
        def run(payload):
            moving = _at(payload, np.eye(2), offset=np.array([3.0, 3.0]), order=1)
            return coreg.estimate(payload, moving)
        return run, None
    return _factory

_reg("AffineCoRegistration.estimate",         "Coregistration", _mk_affine_coreg_estimate())
_reg("AffineCoRegistration.apply",            "Coregistration", _mk_affine_coreg_apply())
_reg("FeatureMatchCoRegistration.estimate",   "Coregistration", _mk_feature_match_coreg())
_reg("ProjectiveCoRegistration.estimate",     "Coregistration", _mk_projective_coreg_estimate())

# ── Orthorectification ────────────────────────────────────────────────────────

def _mk_orthorectifier_compute_mapping():
    def _factory():
        from rasterio.transform import Affine
        from grdl.geolocation import AffineGeolocation
        from grdl.image_processing.ortho import Orthorectifier, OutputGrid
        transform = Affine(0.00027, 0.0, -118.0, 0.0, -0.00027, 34.0)
        geo = AffineGeolocation(transform=transform, shape=(512, 512), crs="EPSG:4326")
        grid = OutputGrid.from_geolocation(geo, pixel_size_lat=0.00054, pixel_size_lon=0.00054)
        ortho = Orthorectifier(geolocation=geo, output_grid=grid, interpolation="bilinear")
        def run(_payload):
            return ortho.compute_mapping()
        return run, None
    return _factory

def _mk_orthorectifier_apply():
    def _factory():
        from rasterio.transform import Affine
        from grdl.geolocation import AffineGeolocation
        from grdl.image_processing.ortho import Orthorectifier, OutputGrid
        transform = Affine(0.00027, 0.0, -118.0, 0.0, -0.00027, 34.0)
        geo = AffineGeolocation(transform=transform, shape=(512, 512), crs="EPSG:4326")
        grid = OutputGrid.from_geolocation(geo, pixel_size_lat=0.00054, pixel_size_lon=0.00054)
        ortho = Orthorectifier(geolocation=geo, output_grid=grid, interpolation="bilinear")
        img = np.random.rand(512, 512).astype(np.float32)
        def run(_payload):
            return ortho.apply(img)
        return run, None
    return _factory

def _mk_ortho_pipeline():
    def _factory():
        from rasterio.transform import Affine
        from grdl.geolocation import AffineGeolocation
        try:
            from grdl.image_processing.ortho import OrthoPipeline
        except ImportError:
            raise ImportError("OrthoPipeline not available in this GRDL version — use Orthorectifier instead")
        transform = Affine(0.00027, 0.0, -118.0, 0.0, -0.00027, 34.0)
        def run(payload):
            # Rebuild each call so shape matches payload
            r, c = payload.shape
            geo = AffineGeolocation(transform=transform, shape=(r, c), crs="EPSG:4326")
            pipeline = (
                OrthoPipeline()
                .with_source_array(payload)
                .with_geolocation(geo)
                .with_resolution(0.00054, 0.00054)
                .with_interpolation("bilinear")
            )
            return pipeline.run()
        return run, None
    return _factory

_reg("Orthorectifier.compute_mapping", "Orthorectification", _mk_orthorectifier_compute_mapping())
_reg("Orthorectifier.apply",           "Orthorectification", _mk_orthorectifier_apply())
_reg("OrthoPipeline.run",              "Orthorectification", _mk_ortho_pipeline())

def _mk_compute_output_resolution():
    def _factory():
        sar_path = (
            _find_gui_data_file(_GUI_DATA_DIR / "umbra", "*.nitf") or
            _find_gui_data_file(_GUI_DATA_DIR / "umbra", "*.ntf")
        )
        if sar_path is None:
            raise ImportError("No SICD file in data/umbra/ — compute_output_resolution needs SICD metadata.")
        from grdl.IO.sar import SICDReader as _SR
        from grdl.image_processing.ortho import compute_output_resolution
        with _suppress_stderr(), _SR(sar_path) as reader:
            _meta = reader.metadata
        def run(_payload):
            return compute_output_resolution(_meta)
        return run, None
    return _factory

_reg("compute_output_resolution", "Orthorectification", _mk_compute_output_resolution())

# ── Interpolation ─────────────────────────────────────────────────────────────

def _mk_lanczos(a: int):
    def _factory():
        from grdl.interpolation import lanczos_interpolator
        interp = lanczos_interpolator(a=a)
        rng = np.random.default_rng(0)
        def setup(p):
            n = p.shape[0] * p.shape[1]
            xo = np.arange(n, dtype=np.float64)
            yo = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex128)
            xn = xo[:-1] + rng.uniform(0.1, 0.9, size=n - 1)
            return (xo, yo, xn), {}
        return interp, setup
    return _factory

def _mk_windowed_sinc(kl: int):
    def _factory():
        from grdl.interpolation import windowed_sinc_interpolator
        interp = windowed_sinc_interpolator(kernel_length=kl, beta=5.0)
        rng = np.random.default_rng(0)
        def setup(p):
            n = p.shape[0] * p.shape[1]
            xo = np.arange(n, dtype=np.float64)
            yo = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex128)
            xn = xo[:-1] + rng.uniform(0.1, 0.9, size=n - 1)
            return (xo, yo, xn), {}
        return interp, setup
    return _factory

def _mk_lagrange(order: int):
    def _factory():
        from grdl.interpolation import lagrange_interpolator
        interp = lagrange_interpolator(order=order)
        rng = np.random.default_rng(0)
        def setup(p):
            n = p.shape[0] * p.shape[1]
            xo = np.arange(n, dtype=np.float64)
            yo = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex128)
            xn = xo[:-1] + rng.uniform(0.1, 0.9, size=n - 1)
            return (xo, yo, xn), {}
        return interp, setup
    return _factory

def _mk_farrow(fo: int, po: int):
    def _factory():
        from grdl.interpolation import farrow_interpolator
        interp = farrow_interpolator(filter_order=fo, poly_order=po)
        rng = np.random.default_rng(0)
        def setup(p):
            n = p.shape[0] * p.shape[1]
            xo = np.arange(n, dtype=np.float64)
            yo = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex128)
            xn = xo[:-1] + rng.uniform(0.1, 0.9, size=n - 1)
            return (xo, yo, xn), {}
        return interp, setup
    return _factory

def _mk_polyphase(kl: int, nph: int):
    def _factory():
        from grdl.interpolation import polyphase_interpolator
        interp = polyphase_interpolator(kernel_length=kl, num_phases=nph)
        rng = np.random.default_rng(0)
        def setup(p):
            n = p.shape[0] * p.shape[1]
            xo = np.arange(n, dtype=np.float64)
            yo = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex128)
            xn = xo[:-1] + rng.uniform(0.1, 0.9, size=n - 1)
            return (xo, yo, xn), {}
        return interp, setup
    return _factory

def _mk_thiran(delay: float, order: int):
    def _factory():
        from grdl.interpolation import thiran_delay
        rng = np.random.default_rng(0)
        def run(_payload):
            signal_1d = rng.standard_normal(1024).astype(np.float64)
            return thiran_delay(signal_1d, delay, order)
        return run, None
    return _factory

_reg("LanczosInterpolator.a3",         "Interpolation", _mk_lanczos(3))
_reg("LanczosInterpolator.a5",         "Interpolation", _mk_lanczos(5))
_reg("KaiserSincInterpolator.kl8",     "Interpolation", _mk_windowed_sinc(8))
_reg("KaiserSincInterpolator.kl16",    "Interpolation", _mk_windowed_sinc(16))
_reg("LagrangeInterpolator.order3",    "Interpolation", _mk_lagrange(3))
_reg("LagrangeInterpolator.order5",    "Interpolation", _mk_lagrange(5))
_reg("FarrowInterpolator.f4_p3",       "Interpolation", _mk_farrow(4, 3))
_reg("FarrowInterpolator.f8_p5",       "Interpolation", _mk_farrow(8, 5))
_reg("PolyphaseInterpolator.kl8_ph32", "Interpolation", _mk_polyphase(8, 32))
_reg("PolyphaseInterpolator.kl16_ph64","Interpolation", _mk_polyphase(16, 64))
_reg("ThiranDelayFilter.d0.7_o1",      "Interpolation", _mk_thiran(0.7, 1))
_reg("ThiranDelayFilter.d3.7_o3",      "Interpolation", _mk_thiran(3.7, 3))

# ── SAR processing — real-data (check file presence) ─────────────────────────

def _mk_sublook_decomposition(num_looks: int):
    def _factory():
        sar_path = (
            _find_gui_data_file(_GUI_DATA_DIR / "umbra", "*.nitf") or
            _find_gui_data_file(_GUI_DATA_DIR / "umbra", "*.ntf")
        )
        if sar_path is None:
            raise ImportError("No SICD file in data/umbra/ — SublookDecomposition needs real SAR chip.")
        from grdl.IO.sar import SICDReader as _SR
        from grdl.image_processing.sar import SublookDecomposition
        with _suppress_stderr(), _SR(sar_path) as reader:
            _meta = reader.metadata
            shape = reader.get_shape()
            cx, cy = shape[0] // 2, shape[1] // 2
            half = min(512, shape[0] // 2, shape[1] // 2)
            chip = reader.read_chip(cx - half, cx + half, cy - half, cy + half)
        decomp = SublookDecomposition(metadata=_meta, num_looks=num_looks)
        def run(_payload):
            return decomp.decompose(chip)
        return run, None
    return _factory

def _mk_multilook_decomposition(looks_rg: int, looks_az: int):
    def _factory():
        sar_path = (
            _find_gui_data_file(_GUI_DATA_DIR / "umbra", "*.nitf") or
            _find_gui_data_file(_GUI_DATA_DIR / "umbra", "*.ntf")
        )
        if sar_path is None:
            raise ImportError("No SICD file in data/umbra/ — MultilookDecomposition needs real SAR chip.")
        from grdl.IO.sar import SICDReader as _SR
        from grdl.image_processing.sar import MultilookDecomposition
        with _suppress_stderr(), _SR(sar_path) as reader:
            _meta = reader.metadata
            shape = reader.get_shape()
            cx, cy = shape[0] // 2, shape[1] // 2
            half = min(512, shape[0] // 2, shape[1] // 2)
            chip = reader.read_chip(cx - half, cx + half, cy - half, cy + half)
        ml = MultilookDecomposition(metadata=_meta, looks_rg=looks_rg, looks_az=looks_az)
        def run(_payload):
            return ml.decompose(chip)
        return run, None
    return _factory

def _mk_csi_processor():
    def _factory():
        sar_path = (
            _find_gui_data_file(_GUI_DATA_DIR / "umbra", "*.nitf") or
            _find_gui_data_file(_GUI_DATA_DIR / "umbra", "*.ntf")
        )
        if sar_path is None:
            raise ImportError("No SICD file in data/umbra/ — CSIProcessor needs real SICD metadata.")
        from grdl.IO.sar import SICDReader as _SR
        from grdl.image_processing.sar import CSIProcessor
        with _suppress_stderr(), _SR(sar_path) as reader:
            _meta = reader.metadata
            shape = reader.get_shape()
            cx, cy = shape[0] // 2, shape[1] // 2
            half = min(512, shape[0] // 2, shape[1] // 2)
            chip = reader.read_chip(cx - half, cx + half, cy - half, cy + half)
        csi = CSIProcessor(metadata=_meta)
        def run(_payload):
            return csi.apply(chip)
        return run, None
    return _factory

def _mk_compute_dominance():
    def _factory():
        sar_path = (
            _find_gui_data_file(_GUI_DATA_DIR / "umbra", "*.nitf") or
            _find_gui_data_file(_GUI_DATA_DIR / "umbra", "*.ntf")
        )
        if sar_path is None:
            raise ImportError("No SICD file in data/umbra/ — compute_dominance needs a sub-look stack.")
        from grdl.IO.sar import SICDReader as _SR
        from grdl.image_processing.sar import SublookDecomposition
        from grdl.image_processing.sar.dominance import compute_dominance
        with _suppress_stderr(), _SR(sar_path) as reader:
            _meta = reader.metadata
            shape = reader.get_shape()
            cx, cy = shape[0] // 2, shape[1] // 2
            half = min(256, shape[0] // 2, shape[1] // 2)
            chip = reader.read_chip(cx - half, cx + half, cy - half, cy + half)
        decomp = SublookDecomposition(metadata=_meta, num_looks=3)
        looks = decomp.decompose(chip)
        sublooks_stack = np.stack(list(looks), axis=0)
        def run(_payload):
            return compute_dominance(sublooks_stack, window_size=7)
        return run, None
    return _factory

def _mk_compute_sublook_entropy():
    def _factory():
        sar_path = (
            _find_gui_data_file(_GUI_DATA_DIR / "umbra", "*.nitf") or
            _find_gui_data_file(_GUI_DATA_DIR / "umbra", "*.ntf")
        )
        if sar_path is None:
            raise ImportError("No SICD file in data/umbra/ — compute_sublook_entropy needs a sub-look stack.")
        from grdl.IO.sar import SICDReader as _SR
        from grdl.image_processing.sar import SublookDecomposition
        from grdl.image_processing.sar.dominance import compute_sublook_entropy
        with _suppress_stderr(), _SR(sar_path) as reader:
            _meta = reader.metadata
            shape = reader.get_shape()
            cx, cy = shape[0] // 2, shape[1] // 2
            half = min(256, shape[0] // 2, shape[1] // 2)
            chip = reader.read_chip(cx - half, cx + half, cy - half, cy + half)
        decomp = SublookDecomposition(metadata=_meta, num_looks=3)
        looks = decomp.decompose(chip)
        sublooks_stack = np.stack(list(looks), axis=0)
        def run(_payload):
            return compute_sublook_entropy(sublooks_stack, window_size=7)
        return run, None
    return _factory

def _mk_collection_geometry():
    def _factory():
        cphd_path = _find_gui_data_file(_GUI_DATA_DIR / "cphd", "*.cphd")
        if cphd_path is None:
            raise ImportError("No CPHD file in data/cphd/ — CollectionGeometry requires CPHD metadata.")
        from grdl.IO.sar import CPHDReader as _CR
        from grdl.image_processing.sar import CollectionGeometry
        with _CR(cphd_path) as reader:
            _meta = reader.metadata
        def run(_payload):
            return CollectionGeometry(_meta)
        return run, None
    return _factory

def _mk_polar_grid():
    def _factory():
        cphd_path = _find_gui_data_file(_GUI_DATA_DIR / "cphd", "*.cphd")
        if cphd_path is None:
            raise ImportError("No CPHD file in data/cphd/ — PolarGrid requires CPHD metadata.")
        from grdl.IO.sar import CPHDReader as _CR
        from grdl.image_processing.sar import CollectionGeometry, PolarGrid
        with _CR(cphd_path) as reader:
            _meta = reader.metadata
        geom = CollectionGeometry(_meta)
        def run(_payload):
            return PolarGrid(geom)
        return run, None
    return _factory

def _mk_subap_partitioner():
    def _factory():
        cphd_path = _find_gui_data_file(_GUI_DATA_DIR / "cphd", "*.cphd")
        if cphd_path is None:
            raise ImportError("No CPHD file in data/cphd/ — SubaperturePartitioner needs CPHD metadata.")
        from grdl.IO.sar import CPHDReader as _CR
        from grdl.image_processing.sar import SubaperturePartitioner
        with _CR(cphd_path) as reader:
            _meta = reader.metadata
        def run(_payload):
            return SubaperturePartitioner(metadata=_meta)
        return run, None
    return _factory

def _mk_polar_format_algorithm():
    def _factory():
        cphd_path = _find_gui_data_file(_GUI_DATA_DIR / "cphd", "*.cphd")
        if cphd_path is None:
            raise ImportError("No CPHD file in data/cphd/ — PolarFormatAlgorithm requires CPHD data.")
        from grdl.IO.sar import CPHDReader as _CR
        from grdl.image_processing.sar import CollectionGeometry, PolarGrid, PolarFormatAlgorithm
        with _CR(cphd_path) as reader:
            _meta = reader.metadata
            phase_data = reader.read_full()
        geom = CollectionGeometry(_meta)
        grid = PolarGrid(geom)
        pfa = PolarFormatAlgorithm(grid=grid)
        def run(_payload):
            return pfa.form_image(phase_data, geometry=geom)
        return run, None
    return _factory

def _mk_nisar_reader():
    def _factory():
        nisar_dir = _GUI_DATA_DIR / "nisar"
        nisar_path = (
            _find_gui_data_file(nisar_dir, "*.h5") or
            _find_gui_data_file(nisar_dir, "*.hdf5")
        )
        if nisar_path is None:
            raise ImportError("No NISAR HDF5 file found in data/nisar/ — download from ASF DAAC.")
        from grdl.IO.sar import NISARReader as _NR
        def run(_payload):
            with _NR(nisar_path) as reader:
                return reader.read_full()
        return run, None
    return _factory

_reg("SublookDecomposition.2looks",   "SAR", _mk_sublook_decomposition(2))
_reg("SublookDecomposition.3looks",   "SAR", _mk_sublook_decomposition(3))
_reg("MultilookDecomposition.2x2",    "SAR", _mk_multilook_decomposition(2, 2))
_reg("MultilookDecomposition.3x3",    "SAR", _mk_multilook_decomposition(3, 3))
_reg("CSIProcessor",                  "SAR", _mk_csi_processor())
_reg("compute_dominance",             "SAR", _mk_compute_dominance())
_reg("compute_sublook_entropy",       "SAR", _mk_compute_sublook_entropy())
_reg("CollectionGeometry",            "SAR", _mk_collection_geometry())
_reg("PolarGrid",                     "SAR", _mk_polar_grid())
_reg("SubaperturePartitioner",        "SAR", _mk_subap_partitioner())
_reg("PolarFormatAlgorithm",          "SAR", _mk_polar_format_algorithm())
_reg("NISARReader",                   "SAR", _mk_nisar_reader())


# ---------------------------------------------------------------------------
# Catalog resolution helpers
# ---------------------------------------------------------------------------

def _build_catalog() -> List[Dict[str, Any]]:
    """Return catalog with availability probed for every entry.

    Returns a list of dicts with keys:
        name, group, available, skip_reason, factory
    """
    out = []
    for entry in _CATALOG_ENTRIES:
        if entry["requires_real_data"]:
            out.append({**entry, "available": False})
            continue
        try:
            fn, setup = entry["factory"]()
            out.append({**entry, "available": True, "_fn": fn, "_setup": setup})
        except ImportError as exc:
            out.append({**entry, "available": False, "skip_reason": str(exc)})
        except Exception as exc:
            out.append({**entry, "available": False, "skip_reason": str(exc)})
    return out


_CATALOG: Optional[List[Dict[str, Any]]] = None


def get_catalog() -> List[Dict[str, Any]]:
    """Return (and cache) the probed component catalog."""
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = _build_catalog()
    return _CATALOG


def get_catalog_groups() -> Dict[str, List[Dict[str, Any]]]:
    """Return catalog entries grouped by group name."""
    cat = get_catalog()
    groups: Dict[str, List] = defaultdict(list)
    for entry in cat:
        groups[entry["group"]].append(entry)
    return dict(groups)


# ---------------------------------------------------------------------------
# Per-level saturation-curve data helper
# ---------------------------------------------------------------------------

def _per_level_stats(record: StressTestRecord) -> List[Dict[str, Any]]:
    """Derive per-concurrency statistics from a stress record."""
    buckets: Dict[int, List] = defaultdict(list)
    for ev in record.events:
        buckets[ev.concurrency_level].append(ev)

    rows = []
    for level in sorted(buckets):
        evs = buckets[level]
        total = len(evs)
        failed = sum(1 for e in evs if not e.success)
        er = failed / total * 100.0 if total > 0 else 0.0
        lats = [e.latency_s for e in evs if e.success]
        p99 = float(np.percentile(lats, 99)) if lats else 0.0
        p50 = float(np.percentile(lats, 50)) if lats else 0.0
        rows.append({
            "concurrency": level,
            "total_calls": total,
            "failed_calls": failed,
            "error_rate_pct": round(er, 1),
            "p50_latency_s": round(p50, 5),
            "p99_latency_s": round(p99, 5),
        })
    return rows


# ---------------------------------------------------------------------------
# Report card renderer
# ---------------------------------------------------------------------------

def _render_metric(label: str, value: str, accent: bool = False) -> None:
    color = "text-cyan-300" if accent else "text-slate-100"
    with ui.column().classes("gap-0 shrink-0"):
        ui.label(label).classes("text-[11px] text-slate-500 uppercase tracking-widest font-medium")
        ui.label(value).classes(f"text-xl font-mono {color} whitespace-nowrap")


def _render_stress_summary_card(record: StressTestRecord) -> None:
    """Render summary metrics for a single StressTestRecord."""
    s = record.summary
    hw = record.hardware

    with ui.card().classes("w-full bg-slate-800/60 shadow-lg ring-1 ring-white/5").props("flat"):
        ui.label("Summary").classes("text-sm font-semibold text-slate-400 uppercase tracking-widest mb-3")

        with ui.row().classes("gap-8 items-end flex-wrap pb-2"):
            _render_metric("Component", record.component_name, accent=True)
            _render_metric("GRDL Version", record.grdl_version)
            _render_metric("Run At", record.created_at[:19])
            _render_metric("Max Sustained", str(s.max_sustained_concurrency), accent=True)
            if s.saturation_concurrency is not None:
                _render_metric("Saturation @", str(s.saturation_concurrency))
            _render_metric("Total Calls", str(s.total_calls))
            _render_metric("Failed", str(s.failed_calls))
            _render_metric("P99 Latency", _fmt_time(s.p99_latency_s), accent=True)
            _render_metric("Memory HWM", _fmt_bytes(s.memory_high_water_mark_bytes))
            if s.first_failure_mode:
                _render_metric("First Failure", s.first_failure_mode)
            else:
                _render_metric("Failures", "None ✓")

        if hw is not None:
            ui.separator().classes("my-2 bg-white/5")
            with ui.row().classes("gap-8 flex-wrap"):
                _render_metric("Host", hw.hostname)
                _render_metric("CPUs", str(hw.cpu_count))
                _render_metric("Memory", _fmt_bytes(hw.total_memory_bytes))


def _render_saturation_chart(record: StressTestRecord) -> None:
    """Render a Plotly saturation-curve chart (latency + optionally error rate vs concurrent workers)."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        ui.label("(Install plotly to see saturation chart)").classes("text-slate-500 text-sm")
        return

    rows_data = _per_level_stats(record)
    if not rows_data:
        return

    levels = [r["concurrency"] for r in rows_data]
    p50s   = [r["p50_latency_s"] for r in rows_data]
    p99s   = [r["p99_latency_s"] for r in rows_data]
    errs   = [r["error_rate_pct"] for r in rows_data]

    # Only add an error-rate subplot when there are actual failures.
    # Plotly autoscales an all-zero bar chart in ways that look misleading.
    has_errors = record.summary.failed_calls > 0
    x_axis_title = "Concurrent workers — number of simultaneous calls issued at once"

    if has_errors:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=("Latency (s)", "Error Rate (%)"))
        num_chart_rows = 2
        chart_height = 360
    else:
        fig = make_subplots(rows=1, cols=1,
                            subplot_titles=("Latency (s)",))
        num_chart_rows = 1
        chart_height = 220

    fig.add_trace(go.Scatter(x=levels, y=p50s, mode="lines+markers", name="P50 latency",
                             line=dict(color="#22d3ee", width=2),
                             marker=dict(size=6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=levels, y=p99s, mode="lines+markers", name="P99 latency",
                             line=dict(color="#fbbf24", width=2, dash="dash"),
                             marker=dict(size=6)), row=1, col=1)

    if has_errors:
        fig.add_trace(go.Bar(x=levels, y=errs, name="Error %",
                             marker_color="#fb7185", opacity=0.7), row=2, col=1)

    if record.summary.saturation_concurrency is not None:
        sat = record.summary.saturation_concurrency
        for row_idx in range(1, num_chart_rows + 1):
            fig.add_vline(x=sat, line=dict(color="#fb7185", dash="dot", width=1.5),
                          annotation_text="saturation point", annotation_font_color="#fb7185",
                          row=row_idx, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        font=dict(family="Inter, sans-serif", color="#cbd5e1", size=11),
        height=chart_height,
        margin=dict(l=10, r=20, t=30, b=30),
        legend=dict(orientation="h", y=1.18, x=0),
    )
    fig.update_xaxes(title_text=x_axis_title, row=num_chart_rows, col=1)
    for ann in fig.layout.annotations:
        ann.font = dict(size=11, color="#94a3b8")

    ui.plotly(fig).classes("w-full")

    if not has_errors:
        ui.label("No call failures at any concurrency level ✓").classes(
            "text-emerald-400 text-xs mt-1"
        )


def _render_failure_table(record: StressTestRecord) -> None:
    """Render a failure-point table if any failures were recorded."""
    if not record.failure_points:
        ui.label("No failures detected ✓").classes("text-emerald-400 text-sm")
        return

    ui.label("Failure Points").classes("text-sm font-semibold text-slate-400 uppercase tracking-widest mt-4 mb-2")
    rows_data = [
        {
            "concurrency": fp.concurrency_level,
            "error_type": fp.error_type,
            "message": fp.error_message[:120],
            "memory": _fmt_bytes(fp.memory_bytes_at_failure),
            "at": fp.first_occurrence_at[:19],
        }
        for fp in record.failure_points
    ]
    cols = [
        {"name": "concurrency", "label": "Concurrency", "field": "concurrency", "align": "left"},
        {"name": "error_type",  "label": "Error Type",  "field": "error_type",  "align": "left"},
        {"name": "message",     "label": "Message",     "field": "message",     "align": "left"},
        {"name": "memory",      "label": "Memory",      "field": "memory",      "align": "right"},
        {"name": "at",          "label": "First At",    "field": "at",          "align": "left"},
    ]
    ui.table(columns=cols, rows=rows_data).props(
        "dark dense flat bordered hide-bottom :rows-per-page-options=\"[0]\""
    ).classes("w-full")


def _render_full_report(record: StressTestRecord) -> None:
    """Render summary card, saturation chart, and failure table for one record."""
    _render_stress_summary_card(record)
    ui.label("Saturation Curve").classes("text-sm font-semibold text-slate-400 uppercase tracking-widest mt-4 mb-2")
    _render_saturation_chart(record)
    _render_failure_table(record)


# ---------------------------------------------------------------------------
# Comparison view
# ---------------------------------------------------------------------------

def _render_comparison_view(rec_a: StressTestRecord, rec_b: StressTestRecord) -> None:
    """Side-by-side comparison of two StressTestRecords."""
    sa, sb = rec_a.summary, rec_b.summary

    ui.label("Comparison").classes("text-lg font-semibold text-slate-200 mt-2 mb-4")

    # Header row: A vs B
    with ui.row().classes("w-full gap-4"):
        with ui.card().classes("flex-1 bg-slate-800/60 ring-1 ring-white/5 p-4").props("flat"):
            ui.badge("Run A").classes("bg-cyan-700 text-slate-100 mb-2")
            ui.label(rec_a.component_name).classes("text-slate-100 font-mono text-sm")
            ui.label(rec_a.created_at[:19]).classes("text-slate-500 text-xs")
        with ui.card().classes("flex-1 bg-slate-800/60 ring-1 ring-white/5 p-4").props("flat"):
            ui.badge("Run B").classes("bg-violet-700 text-slate-100 mb-2")
            ui.label(rec_b.component_name).classes("text-slate-100 font-mono text-sm")
            ui.label(rec_b.created_at[:19]).classes("text-slate-500 text-xs")

    ui.separator().classes("my-4 bg-white/5")

    # Delta table
    def _delta_row(label: str, val_a, val_b, fmt_fn=str, higher_is_better: bool = True):
        if isinstance(val_a, float) and isinstance(val_b, float):
            delta = val_b - val_a
            pct = (delta / val_a * 100.0) if val_a != 0 else 0.0
            if higher_is_better:
                color = "text-emerald-400" if delta >= 0 else "text-red-400"
            else:
                color = "text-emerald-400" if delta <= 0 else "text-red-400"
            delta_str = f"{'+' if delta >= 0 else ''}{fmt_fn(delta)} ({pct:+.1f}%)"
        else:
            delta_str = f"{val_a} → {val_b}"
            color = "text-slate-400"
        return {
            "metric": label,
            "run_a": fmt_fn(val_a) if isinstance(val_a, float) else str(val_a),
            "run_b": fmt_fn(val_b) if isinstance(val_b, float) else str(val_b),
            "delta": delta_str,
            "_color": color,
        }

    delta_rows = [
        _delta_row("Max Sustained Concurrency",
                   float(sa.max_sustained_concurrency), float(sb.max_sustained_concurrency),
                   lambda x: str(int(x)), higher_is_better=True),
        _delta_row("P99 Latency (s)",
                   sa.p99_latency_s, sb.p99_latency_s, _fmt_time, higher_is_better=False),
        _delta_row("Total Calls",
                   float(sa.total_calls), float(sb.total_calls),
                   lambda x: str(int(x)), higher_is_better=True),
        _delta_row("Failed Calls",
                   float(sa.failed_calls), float(sb.failed_calls),
                   lambda x: str(int(x)), higher_is_better=False),
        _delta_row("Memory HWM",
                   float(sa.memory_high_water_mark_bytes), float(sb.memory_high_water_mark_bytes),
                   _fmt_bytes, higher_is_better=False),
        {
            "metric": "Saturation Concurrency",
            "run_a": str(sa.saturation_concurrency) if sa.saturation_concurrency else "—",
            "run_b": str(sb.saturation_concurrency) if sb.saturation_concurrency else "—",
            "delta": "—",
            "_color": "text-slate-400",
        },
        {
            "metric": "First Failure Mode",
            "run_a": sa.first_failure_mode or "None ✓",
            "run_b": sb.first_failure_mode or "None ✓",
            "delta": "—",
            "_color": "text-slate-400",
        },
    ]

    ui.label("Metric Delta").classes("text-sm font-semibold text-slate-400 uppercase tracking-widest mb-2")
    cols = [
        {"name": "metric", "label": "Metric",   "field": "metric", "align": "left"},
        {"name": "run_a",  "label": "Run A",     "field": "run_a",  "align": "right"},
        {"name": "run_b",  "label": "Run B",     "field": "run_b",  "align": "right"},
        {"name": "delta",  "label": "Δ (B − A)", "field": "delta",  "align": "right"},
    ]
    table_rows = [{k: v for k, v in r.items() if k != "_color"} for r in delta_rows]
    ui.table(columns=cols, rows=table_rows).props(
        "dark dense flat bordered hide-bottom :rows-per-page-options=\"[0]\""
    ).classes("w-full max-w-4xl mb-6")

    # Overlaid saturation-curve comparison
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        stats_a = _per_level_stats(rec_a)
        stats_b = _per_level_stats(rec_b)

        ui.label("Saturation Curve Overlay").classes("text-sm font-semibold text-slate-400 uppercase tracking-widest mb-2")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=("P99 Latency (s)", "Error Rate (%)"))

        for stats, label, color, dash in [
            (stats_a, "A", "#22d3ee", "solid"),
            (stats_b, "B", "#a78bfa", "dot"),
        ]:
            lvls = [r["concurrency"] for r in stats]
            p99s = [r["p99_latency_s"] for r in stats]
            errs = [r["error_rate_pct"] for r in stats]
            fig.add_trace(go.Scatter(x=lvls, y=p99s, mode="lines+markers", name=f"P99 {label}",
                                     line=dict(color=color, dash=dash, width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=lvls, y=errs, mode="lines+markers", name=f"Err% {label}",
                                     line=dict(color=color, dash=dash, width=1.5)), row=2, col=1)

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            font=dict(family="Inter, sans-serif", color="#cbd5e1", size=11),
            height=380, margin=dict(l=10, r=20, t=30, b=30),
            legend=dict(orientation="h", y=1.12, x=0),
        )
        fig.update_xaxes(title_text="Concurrency", row=2, col=1)
        for ann in fig.layout.annotations:
            ann.font = dict(size=11, color="#94a3b8")

        ui.plotly(fig).classes("w-full")
    except ImportError:
        pass

    # Per-run detail panels
    ui.separator().classes("my-4 bg-white/5")
    with ui.row().classes("w-full gap-4 flex-wrap"):
        with ui.column().classes("flex-1 min-w-64"):
            ui.badge("Run A Full Detail").classes("bg-cyan-700 text-slate-100 mb-2")
            _render_full_report(rec_a)
        with ui.column().classes("flex-1 min-w-64"):
            ui.badge("Run B Full Detail").classes("bg-violet-700 text-slate-100 mb-2")
            _render_full_report(rec_b)


# ---------------------------------------------------------------------------
# Progress state (shared between background thread and UI)
# ---------------------------------------------------------------------------

class _StressRunState:
    """Mutable state bag shared between the background runner thread and UI."""

    def __init__(self) -> None:
        self.running: bool = False
        self.completed: bool = False
        self.error: Optional[str] = None
        self.log_lines: List[str] = []
        self.record: Optional[StressTestRecord] = None
        self._lock = threading.Lock()

    def append_log(self, line: str) -> None:
        with self._lock:
            self.log_lines.append(line)

    def get_log(self) -> str:
        with self._lock:
            return "\n".join(self.log_lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def launch_stress_gui(
    port: int = 8080,
    store_dir: Optional[Path] = None,
) -> None:
    """Launch the interactive stress test dashboard.

    Parameters
    ----------
    port : int
        Port for the NiceGUI web server.  Default 8080.
    store_dir : Path, optional
        Default directory for saving stress test records.  Defaults to
        ``<cwd>/.benchmarks/``.
    """
    catalog = get_catalog()
    available = [e for e in catalog if e["available"]]
    unavailable = [e for e in catalog if not e["available"]]
    groups = get_catalog_groups()

    default_store = store_dir or (Path.cwd() / ".benchmarks")

    @ui.page("/")
    def _main_page() -> None:   # noqa: C901 — complex but intentionally self-contained
        ui.dark_mode(True)
        ui.add_head_html(
            '<link rel="preconnect" href="https://fonts.googleapis.com">'
            '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
            '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">'
            "<style>"
            "body { background-color: #0f172a !important; font-family: 'Inter', sans-serif; }"
            ".q-table--dark .q-table__bottom, .q-table--dark td, .q-table--dark th,"
            ".q-table--dark thead, .q-table--dark tr { border-color: rgba(255,255,255,0.06) !important; }"
            ".q-tabs__content { overflow-x: auto !important; scroll-behavior: smooth; }"
            ".q-tabs__content::-webkit-scrollbar { height: 3px; }"
            ".q-tabs__content::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 3px; }"
            ".nicegui-log { font-family: 'Courier New', monospace; font-size: 12px; }"
            "</style>"
        )

        # ── Shared page state ────────────────────────────────────────────
        # These are declared in the page closure so each browser tab is isolated.
        _page_state: Dict[str, Any] = {
            "selected_component": available[0]["name"] if available else None,
            "run_state": None,
            "result_record": None,
            "compare_record_a": None,
            "compare_record_b": None,
        }

        # ── Header ───────────────────────────────────────────────────────
        with ui.column().classes("w-full max-w-7xl mx-auto p-6 min-h-screen gap-4"):
            with ui.row().classes("w-full items-center justify-between"):
                ui.label("GRDL Stress Test Runner").classes(
                    "text-xl font-semibold text-slate-200 tracking-wide"
                )
                with ui.row().classes("gap-3 items-center"):
                    ui.badge(f"{len(available)} components").classes("bg-emerald-600 text-white font-semibold")
                    ui.badge(f"{len(unavailable)} require data").classes("bg-slate-500 text-white font-semibold")

            # ── Main tabs ─────────────────────────────────────────────────
            with ui.tabs().classes("w-full").props("dark dense") as tabs:
                tab_run     = ui.tab("Run Stress Test",   icon="play_arrow")
                tab_compare = ui.tab("Compare Reports",   icon="compare_arrows")
                tab_catalog = ui.tab("Component Catalog", icon="list")

            with ui.tab_panels(tabs).classes("w-full bg-slate-900").props("dark"):

                # ============================================================
                # TAB 1: Run Stress Test
                # ============================================================
                with ui.tab_panel(tab_run):
                    with ui.row().classes("w-full gap-6 flex-wrap items-start mt-4"):

                        # ── Left column: component + config ─────────────────
                        with ui.card().classes(
                            "bg-slate-800/60 ring-1 ring-white/5 p-0 w-80 shrink-0"
                        ).props("flat"):
                            ui.label("Configure").classes(
                                "text-sm font-semibold text-slate-400 uppercase tracking-widest px-4 pt-4 pb-2"
                            )

                            # Load previous report button
                            with ui.row().classes("px-4 pb-2 gap-2 items-center"):
                                load_path_input = ui.input(
                                    label="Load previous report path",
                                    placeholder="/path/to/report.json",
                                ).props("dark dense outlined color=cyan").classes("flex-1 text-xs")
                                load_status = ui.label("").classes("text-xs text-slate-500")

                            # Component selector
                            ui.separator().classes("bg-white/5")
                            ui.label("Component").classes("text-xs text-slate-500 uppercase px-4 pt-3 pb-1")

                            group_names = sorted(groups.keys())
                            comp_select = ui.select(
                                options={e["name"]: e["name"] for e in available},
                                value=_page_state["selected_component"],
                                label="Component",
                            ).props("dark dense outlined color=cyan").classes("mx-4 mb-2").style("width: calc(100% - 2rem)")

                            # Group filter
                            group_filter = ui.select(
                                options={"All": "All"} | {g: g for g in group_names},
                                value="All",
                                label="Filter by group",
                            ).props("dark dense outlined color=cyan").classes("mx-4 mb-3").style("width: calc(100% - 2rem)")

                            # Config sliders / inputs
                            ui.separator().classes("bg-white/5")
                            ui.label("Stress Parameters").classes("text-xs text-slate-500 uppercase px-4 pt-3 pb-1")

                            conc_input = ui.number(
                                label="Max Concurrency",
                                value=DEFAULT_MAX_CONCURRENCY, min=1, max=256, precision=0,
                            ).props("dark dense outlined color=cyan").classes("mx-4 mb-2").style("width: calc(100% - 2rem)")

                            steps_input = ui.number(
                                label="Ramp Steps",
                                value=DEFAULT_RAMP_STEPS, min=1, max=20, precision=0,
                            ).props("dark dense outlined color=cyan").classes("mx-4 mb-2").style("width: calc(100% - 2rem)")

                            duration_input = ui.number(
                                label="Step Duration (s)",
                                value=DEFAULT_DURATION_PER_STEP_S, min=1, max=120, precision=1,
                            ).props("dark dense outlined color=cyan").classes("mx-4 mb-2").style("width: calc(100% - 2rem)")

                            timeout_input = ui.number(
                                label="Timeout/Call (s)",
                                value=DEFAULT_TIMEOUT_PER_CALL_S, min=1, max=300, precision=1,
                            ).props("dark dense outlined color=cyan").classes("mx-4 mb-2").style("width: calc(100% - 2rem)")

                            size_select = ui.select(
                                options={"small": "Small (256×256)", "medium": "Medium (1024×1024)", "large": "Large (4096×4096)"},
                                value="medium",
                                label="Payload Size",
                            ).props("dark dense outlined color=cyan").classes("mx-4 mb-2").style("width: calc(100% - 2rem)")

                            start_conc_input = ui.number(
                                label="Start Concurrency",
                                value=DEFAULT_START_CONCURRENCY, min=1, max=64, precision=0,
                            ).props("dark dense outlined color=cyan").classes("mx-4 mb-3").style("width: calc(100% - 2rem)")

                            # Save path
                            ui.separator().classes("bg-white/5")
                            ui.label("Save").classes("text-xs text-slate-500 uppercase px-4 pt-3 pb-1")
                            save_path_input = ui.input(
                                label="Save directory",
                                value=str(default_store),
                            ).props("dark dense outlined color=cyan").classes("mx-4 mb-4").style("width: calc(100% - 2rem)")

                            # Run button
                            run_btn = ui.button("Run Stress Test", icon="play_arrow").props("flat").classes(
                                "w-full bg-cyan-600/20 text-cyan-300 hover:bg-cyan-600/30 mx-0 rounded-none"
                            )

                        # ── Right column: progress + results ─────────────────
                        with ui.column().classes("flex-1 min-w-0 gap-4"):

                            # Progress / log area
                            progress_card = ui.card().classes(
                                "w-full bg-slate-800/60 ring-1 ring-white/5"
                            ).props("flat")
                            with progress_card:
                                ui.label("Status").classes(
                                    "text-sm font-semibold text-slate-400 uppercase tracking-widest mb-2"
                                )
                                progress_label = ui.label("Idle — configure and click Run.").classes(
                                    "text-slate-400 text-sm"
                                )
                                log_el = ui.log(max_lines=120).classes(
                                    "w-full h-36 bg-slate-900/60 rounded text-xs"
                                )

                            # Results placeholder (filled after run)
                            results_container = ui.column().classes("w-full gap-4")

                        # ── Group filter logic ───────────────────────────────
                        def _update_group_filter(e=None) -> None:
                            grp = group_filter.value
                            if grp == "All":
                                opts = {e2["name"]: e2["name"] for e2 in available}
                            else:
                                opts = {e2["name"]: e2["name"] for e2 in available if e2["group"] == grp}
                            comp_select.options = opts
                            if opts:
                                first = next(iter(opts))
                                comp_select.value = first
                                _page_state["selected_component"] = first

                        group_filter.on_value_change(_update_group_filter)
                        comp_select.on_value_change(
                            lambda e: _page_state.update({"selected_component": e.value})
                        )

                        # ── Load previous report logic ────────────────────────
                        def _apply_loaded_report(rec: StressTestRecord) -> None:
                            """Display a loaded record and sync sidebar config fields."""
                            # Pre-fill config sidebar from the loaded record
                            cfg = rec.config
                            conc_input.value = cfg.max_concurrency
                            steps_input.value = cfg.ramp_steps
                            duration_input.value = cfg.duration_per_step_s
                            timeout_input.value = cfg.timeout_per_call_s
                            size_select.value = (
                                cfg.payload_size if cfg.payload_size in ("small", "medium", "large")
                                else "medium"
                            )
                            start_conc_input.value = cfg.start_concurrency

                            # Select matching component if available
                            cname = rec.component_name
                            if cname in {e["name"]: e["name"] for e in available}:
                                comp_select.value = cname
                                _page_state["selected_component"] = cname

                            # Render results
                            _page_state["result_record"] = rec
                            results_container.clear()
                            with results_container:
                                _render_full_report(rec)

                            load_status.text = f"Loaded: {rec.component_name}"
                            load_status.classes(replace="text-xs text-emerald-400")
                            ui.notify(
                                f"Loaded report: {rec.component_name}",
                                type="positive", position="top",
                            )

                        def _load_previous_report() -> None:
                            path_str = load_path_input.value.strip()
                            if not path_str:
                                load_status.text = "Enter a path"
                                return
                            p = Path(path_str).expanduser().resolve()
                            if not p.is_file():
                                load_status.text = "File not found"
                                load_status.classes(replace="text-xs text-red-400")
                                return
                            try:
                                rec = StressTestRecord.from_json(p.read_text(encoding="utf-8"))
                            except Exception as exc:
                                load_status.text = f"Parse error: {exc}"
                                load_status.classes(replace="text-xs text-red-400")
                                return

                            existing = _page_state.get("result_record")
                            if existing is not None:
                                # Ask before replacing currently displayed results
                                with ui.dialog() as confirm_dlg, \
                                     ui.card().classes("bg-slate-800 ring-1 ring-white/10 p-4 gap-3"):
                                    ui.label("Replace current results?").classes(
                                        "text-white font-semibold text-base"
                                    )
                                    ui.label(
                                        f"Currently viewing: {existing.component_name}"
                                    ).classes("text-slate-400 text-sm")
                                    ui.label(
                                        f"Loading: {rec.component_name}  ·  {p.name}"
                                    ).classes("text-slate-400 text-sm")
                                    ui.separator().classes("bg-white/10")
                                    ui.label(
                                        "The current results were auto-saved when the run "
                                        "completed. Any manual edits to the save path have "
                                        "not been applied."
                                    ).classes("text-slate-500 text-xs")
                                    with ui.row().classes("gap-2 mt-1"):
                                        ui.button(
                                            "Load anyway",
                                            on_click=lambda: (
                                                confirm_dlg.close(),
                                                _apply_loaded_report(rec),
                                            ),
                                        ).props("flat").classes(
                                            "bg-cyan-600/20 text-cyan-300 hover:bg-cyan-600/30"
                                        )
                                        ui.button(
                                            "Cancel", on_click=confirm_dlg.close,
                                        ).props("flat").classes("text-slate-400")
                                confirm_dlg.open()
                            else:
                                _apply_loaded_report(rec)

                        load_path_input.on("keydown.enter", lambda: _load_previous_report())
                        ui.button("Load", icon="upload", on_click=_load_previous_report).props(
                            "flat dense"
                        ).classes("text-cyan-400 mx-4 mb-3 -mt-4")

                        # ── Run button logic ──────────────────────────────────
                        def _on_run_click() -> None:   # noqa: C901
                            comp_name = _page_state["selected_component"]
                            if comp_name is None:
                                ui.notify("No component selected.", type="warning", position="top")
                                return
                            if _page_state.get("run_state") and _page_state["run_state"].running:
                                ui.notify("A stress test is already running.", type="warning", position="top")
                                return

                            # Find entry in catalog
                            entry = next((e for e in available if e["name"] == comp_name), None)
                            if entry is None:
                                ui.notify("Component not found.", type="negative", position="top")
                                return

                            # Build config
                            try:
                                config = StressTestConfig(
                                    start_concurrency=int(start_conc_input.value or DEFAULT_START_CONCURRENCY),
                                    max_concurrency=int(conc_input.value or DEFAULT_MAX_CONCURRENCY),
                                    ramp_steps=int(steps_input.value or DEFAULT_RAMP_STEPS),
                                    duration_per_step_s=float(duration_input.value or DEFAULT_DURATION_PER_STEP_S),
                                    payload_size=size_select.value,
                                    timeout_per_call_s=float(timeout_input.value or DEFAULT_TIMEOUT_PER_CALL_S),
                                )
                            except (ValueError, TypeError) as exc:
                                ui.notify(f"Invalid config: {exc}", type="negative", position="top")
                                return

                            # Init state
                            state = _StressRunState()
                            state.running = True
                            _page_state["run_state"] = state

                            run_btn.disable()
                            progress_label.set_text(f"Running {comp_name} …")
                            progress_label.classes(replace="text-slate-300 text-sm")
                            log_el.clear()
                            results_container.clear()

                            # Determine save store
                            save_dir = Path(save_path_input.value.strip() or str(default_store))
                            store_obj = JSONStressTestStore(base_dir=save_dir)

                            def _thread_run():
                                try:
                                    fn, setup = entry["factory"]()
                                    tester = ComponentStressTester(
                                        comp_name, fn, setup=setup, store=store_obj,
                                        tags={"gui": "true", "group": entry["group"]},
                                    )
                                    levels = config.concurrency_levels()
                                    state.append_log(f"Starting: {comp_name}")
                                    state.append_log(
                                        f"Config: concurrency 1→{config.max_concurrency} "
                                        f"({len(levels)} steps), "
                                        f"{config.duration_per_step_s}s/step, "
                                        f"payload={config.payload_size}"
                                    )
                                    state.append_log(f"Concurrency levels: {levels}")
                                    record = tester.run(config)
                                    state.record = record
                                    s = record.summary
                                    state.append_log("")
                                    state.append_log("Done.")
                                    state.append_log(f"  Max sustained concurrency : {s.max_sustained_concurrency}")
                                    state.append_log(f"  Saturation                : {s.saturation_concurrency}")
                                    state.append_log(f"  Total / failed calls      : {s.total_calls} / {s.failed_calls}")
                                    state.append_log(f"  P99 latency               : {_fmt_time(s.p99_latency_s)}")
                                    state.append_log(f"  Memory HWM                : {_fmt_bytes(s.memory_high_water_mark_bytes)}")
                                    state.completed = True
                                except Exception as exc:
                                    state.error = str(exc)
                                    state.append_log(f"ERROR: {exc}")
                                finally:
                                    state.running = False

                            thread = threading.Thread(target=_thread_run, daemon=True)
                            thread.start()

                            # Timer-based polling — runs inside the page client context
                            poll_state = {"last_idx": 0, "timer": None}

                            def _poll_tick() -> None:
                                # Stream new log lines
                                with state._lock:
                                    new_lines = state.log_lines[poll_state["last_idx"]:]
                                    poll_state["last_idx"] = len(state.log_lines)
                                for line in new_lines:
                                    log_el.push(line)

                                if state.running:
                                    return  # keep ticking

                                # Run complete — cancel timer and finalise UI
                                if poll_state["timer"] is not None:
                                    poll_state["timer"].cancel()
                                run_btn.enable()

                                if state.error:
                                    progress_label.set_text(f"Error: {state.error}")
                                    progress_label.classes(replace="text-red-400 text-sm")
                                    ui.notify(
                                        f"Stress test failed: {state.error}",
                                        type="negative", position="top",
                                    )
                                    return

                                record = state.record
                                if record is None:
                                    return

                                _page_state["result_record"] = record
                                progress_label.set_text(
                                    f"Complete — max sustained: "
                                    f"{record.summary.max_sustained_concurrency}"
                                )
                                progress_label.classes(replace="text-emerald-400 text-sm")

                                # Auto-save
                                try:
                                    rid = store_obj.save(record)
                                    saved_path = (
                                        save_dir / "stress" / "records" / f"{rid}.json"
                                    )
                                    log_el.push(f"Saved: {saved_path}")
                                    ui.notify(
                                        f"Report saved: {saved_path.name}",
                                        type="positive", position="top",
                                    )
                                except Exception as exc:
                                    ui.notify(f"Save failed: {exc}", type="warning", position="top")

                                # Render results inline
                                results_container.clear()
                                with results_container:
                                    _render_full_report(record)

                                    with ui.row().classes("gap-2 mt-2 items-start"):
                                        _rec = record  # capture for closure
                                        manual_save_input = ui.input(
                                            label="Save JSON to path",
                                            value=str(
                                                save_dir / "stress" / "records"
                                                / f"{_rec.stress_test_id}.json"
                                            ),
                                        ).props("dark dense outlined color=cyan").classes("w-96 text-xs")

                                        def _manual_save(_r=_rec, _inp=manual_save_input) -> None:
                                            p = Path(_inp.value.strip())
                                            try:
                                                p.parent.mkdir(parents=True, exist_ok=True)
                                                p.write_text(_r.to_json(), encoding="utf-8")
                                                ui.notify(f"Saved to {p}", type="positive", position="top")
                                            except Exception as ex:
                                                ui.notify(f"Save failed: {ex}", type="negative", position="top")

                                        ui.button(
                                            "Save JSON", icon="save", on_click=_manual_save,
                                        ).props("flat").classes(
                                            "bg-cyan-600/20 text-cyan-300 hover:bg-cyan-600/30"
                                        )

                            poll_state["timer"] = ui.timer(0.5, _poll_tick)

                        run_btn.on_click(_on_run_click)

                # ============================================================
                # TAB 2: Compare Reports
                # ============================================================
                with ui.tab_panel(tab_compare):
                    ui.label("Load two saved stress test report JSON files to compare them.").classes(
                        "text-slate-400 text-sm mt-4"
                    )

                    with ui.row().classes("w-full gap-4 flex-wrap mt-4"):
                        # Run A
                        with ui.card().classes("bg-slate-800/60 ring-1 ring-white/5 p-4 flex-1 min-w-64").props("flat"):
                            ui.badge("Run A").classes("bg-cyan-700 text-slate-100 mb-2")
                            input_a = ui.input(
                                label="Path to Report A JSON",
                                placeholder="/path/to/report_a.json",
                            ).props("dark dense outlined color=cyan").classes("w-full")
                            status_a = ui.label("").classes("text-xs mt-1 text-slate-500")

                        # Run B
                        with ui.card().classes("bg-slate-800/60 ring-1 ring-white/5 p-4 flex-1 min-w-64").props("flat"):
                            ui.badge("Run B").classes("bg-violet-700 text-slate-100 mb-2")
                            input_b = ui.input(
                                label="Path to Report B JSON",
                                placeholder="/path/to/report_b.json",
                            ).props("dark dense outlined color=cyan").classes("w-full")
                            status_b = ui.label("").classes("text-xs mt-1 text-slate-500")

                    compare_btn = ui.button("Compare", icon="compare_arrows").props("flat").classes(
                        "mt-4 bg-violet-600/20 text-violet-300 hover:bg-violet-600/30"
                    )
                    compare_out = ui.column().classes("w-full mt-4 gap-4")

                    def _on_compare() -> None:
                        for path_input, status_lbl, attr in [
                            (input_a, status_a, "compare_record_a"),
                            (input_b, status_b, "compare_record_b"),
                        ]:
                            p_str = path_input.value.strip()
                            if not p_str:
                                status_lbl.text = "Enter a path"
                                status_lbl.classes(replace="text-xs mt-1 text-red-400")
                                return
                            p = Path(p_str).expanduser().resolve()
                            if not p.is_file():
                                status_lbl.text = "File not found"
                                status_lbl.classes(replace="text-xs mt-1 text-red-400")
                                return
                            try:
                                rec = StressTestRecord.from_json(p.read_text(encoding="utf-8"))
                                status_lbl.text = f"Loaded: {rec.component_name} ({rec.created_at[:10]})"
                                status_lbl.classes(replace="text-xs mt-1 text-emerald-400")
                                _page_state[attr] = rec
                            except Exception as exc:
                                status_lbl.text = f"Failed: {exc}"
                                status_lbl.classes(replace="text-xs mt-1 text-red-400")
                                return

                        rec_a = _page_state.get("compare_record_a")
                        rec_b = _page_state.get("compare_record_b")
                        if rec_a is None or rec_b is None:
                            return

                        compare_out.clear()
                        with compare_out:
                            _render_comparison_view(rec_a, rec_b)

                    compare_btn.on_click(_on_compare)

                # ============================================================
                # TAB 3: Component Catalog
                # ============================================================
                with ui.tab_panel(tab_catalog):
                    ui.label(
                        "All importable GRDL components and their stress test availability."
                    ).classes("text-slate-400 text-sm mt-4 mb-4")

                    for grp in sorted(groups.keys()):
                        entries = groups[grp]
                        avail_cnt  = sum(1 for e in entries if e["available"])
                        unavail_cnt = len(entries) - avail_cnt

                        with ui.expansion(
                            f"{grp} — {avail_cnt} available, {unavail_cnt} require data",
                        ).props("dark dense header-class='bg-slate-800/60'").classes(
                            "w-full rounded ring-1 ring-white/5 mb-2"
                        ):
                            for entry in entries:
                                with ui.row().classes("items-start gap-3 py-1 border-b border-white/5"):
                                    if entry["available"]:
                                        ui.badge("✓").classes("bg-emerald-700/60 text-emerald-300 shrink-0 mt-0.5")
                                    else:
                                        ui.badge("⊘").classes("bg-slate-700 text-slate-500 shrink-0 mt-0.5")
                                    with ui.column().classes("gap-0"):
                                        ui.label(entry["name"]).classes(
                                            "font-mono text-sm text-slate-200" if entry["available"]
                                            else "font-mono text-sm text-slate-500"
                                        )
                                        if entry.get("skip_reason"):
                                            ui.label(entry["skip_reason"]).classes("text-xs text-slate-500 italic")

    ui.run(reload=False, title="GRDL Stress Test Runner", port=port)
