# -*- coding: utf-8 -*-
"""
GPU Device Verification - CuPy path tests for refactored grdl processors.

Consolidates all GPU (CuPy) validation tests for the grdl image processing
library.  Each test class exercises a single processor with CuPy input and
verifies:

- **Type correctness** — output is a ``cupy.ndarray`` (data stays on-device).
- **Shape consistency** — GPU output shape matches the CPU (numpy) path.
- **Numerical equivalence** — GPU and CPU results agree within tolerance.

SAR processor tests (SublookDecomposition, MultilookDecomposition,
CSIProcessor) require a real Umbra SICD chip via the ``require_umbra_file``
fixture.  ToDecibels uses synthetic data and runs without any data files.

All tests are skipped when CuPy is not installed.

Dependencies
------------
cupy

Author
------
Duane Smalley
170194430+DDSmalls@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-03-09

Modified
--------
2026-03-09
"""

# Standard library
# (none)

# Third-party
import numpy as np
import pytest

try:
    import cupy as cp
    cp.array([1.0])  # verify GPU is actually functional
    _HAS_CUPY = True
except Exception:
    _HAS_CUPY = False

try:
    from grdl.IO.sar import SICDReader
    _HAS_SICD = True
except ImportError:
    _HAS_SICD = False

try:
    from grdl.image_processing.sar import SublookDecomposition
    _HAS_SUBLOOK = True
except ImportError:
    _HAS_SUBLOOK = False

try:
    from grdl.image_processing.sar import MultilookDecomposition
    _HAS_MULTILOOK = True
except ImportError:
    _HAS_MULTILOOK = False

try:
    from grdl.image_processing.sar import CSIProcessor
    _HAS_CSI = True
except ImportError:
    _HAS_CSI = False

try:
    from grdl.image_processing.intensity import ToDecibels
    _HAS_TO_DB = True
except ImportError:
    _HAS_TO_DB = False


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.sar,
    pytest.mark.slow,
    pytest.mark.skipif(not _HAS_CUPY, reason="CuPy not installed"),
]


# =============================================================================
# SublookDecomposition — CuPy GPU path
# =============================================================================


@pytest.mark.skipif(not _HAS_SUBLOOK, reason="SublookDecomposition not available")
@pytest.mark.skipif(not _HAS_SICD, reason="SICDReader not available")
@pytest.mark.requires_data
class TestSublookDecompositionGPU:

    @pytest.fixture(autouse=True)
    def _load_chip(self, require_umbra_file):
        with SICDReader(str(require_umbra_file)) as reader:
            self.metadata = reader.metadata
            shape = reader.get_shape()
            cx, cy = shape[0] // 2, shape[1] // 2
            half = min(256, shape[0] // 2, shape[1] // 2)
            self.chip = reader.read_chip(cx - half, cx + half,
                                         cy - half, cy + half)

    def test_decompose_cupy_input_returns_cupy(self):
        """decompose() with cupy input returns a cupy array."""
        sublook = SublookDecomposition(metadata=self.metadata, num_looks=3)
        result = sublook.decompose(cp.asarray(self.chip))
        assert isinstance(result, cp.ndarray), (
            f"Expected cp.ndarray, got {type(result).__name__}"
        )

    def test_decompose_cupy_shape(self):
        """Output shape matches numpy path for cupy input."""
        sublook = SublookDecomposition(metadata=self.metadata, num_looks=3)
        cpu_result = sublook.decompose(self.chip)
        gpu_result = sublook.decompose(cp.asarray(self.chip))
        assert gpu_result.shape == cpu_result.shape, (
            f"Shape mismatch: GPU {gpu_result.shape} vs CPU {cpu_result.shape}"
        )

    def test_decompose_cupy_matches_numpy(self):
        """CuPy and numpy paths produce numerically identical results."""
        sublook = SublookDecomposition(metadata=self.metadata, num_looks=3)
        cpu_result = sublook.decompose(self.chip)
        gpu_result = cp.asnumpy(sublook.decompose(cp.asarray(self.chip)))
        assert np.allclose(np.abs(cpu_result), np.abs(gpu_result), atol=1e-4), (
            "CuPy and numpy decompose() results differ beyond tolerance"
        )


# =============================================================================
# MultilookDecomposition — CuPy GPU path
# =============================================================================


@pytest.mark.skipif(not _HAS_MULTILOOK, reason="MultilookDecomposition not available")
@pytest.mark.skipif(not _HAS_SICD, reason="SICDReader not available")
@pytest.mark.requires_data
class TestMultilookDecompositionGPU:

    @pytest.fixture(autouse=True)
    def _load_chip(self, require_umbra_file):
        with SICDReader(str(require_umbra_file)) as reader:
            self.metadata = reader.metadata
            shape = reader.get_shape()
            cx, cy = shape[0] // 2, shape[1] // 2
            half = min(256, shape[0] // 2, shape[1] // 2)
            self.chip = reader.read_chip(cx - half, cx + half,
                                         cy - half, cy + half)

    def test_decompose_cupy_input_returns_cupy(self):
        """decompose() with cupy input returns a cupy array."""
        ml = MultilookDecomposition(metadata=self.metadata, looks_rg=2, looks_az=2)
        result = ml.decompose(cp.asarray(self.chip))
        assert isinstance(result, cp.ndarray), (
            f"Expected cp.ndarray, got {type(result).__name__}"
        )

    def test_decompose_cupy_shape(self):
        """Output shape matches numpy path for cupy input."""
        ml = MultilookDecomposition(metadata=self.metadata, looks_rg=2, looks_az=2)
        cpu_result = ml.decompose(self.chip)
        gpu_result = ml.decompose(cp.asarray(self.chip))
        assert gpu_result.shape == cpu_result.shape, (
            f"Shape mismatch: GPU {gpu_result.shape} vs CPU {cpu_result.shape}"
        )

    def test_decompose_cupy_matches_numpy(self):
        """CuPy and numpy paths produce numerically identical results."""
        ml = MultilookDecomposition(metadata=self.metadata, looks_rg=2, looks_az=2)
        cpu_result = ml.decompose(self.chip)
        gpu_result = cp.asnumpy(ml.decompose(cp.asarray(self.chip)))
        assert np.allclose(np.abs(cpu_result), np.abs(gpu_result), atol=1e-4), (
            "CuPy and numpy MultilookDecomposition results differ beyond tolerance"
        )


# =============================================================================
# CSIProcessor — CuPy GPU path
# =============================================================================


@pytest.mark.skipif(not _HAS_CSI, reason="CSIProcessor not available")
@pytest.mark.skipif(not _HAS_SICD, reason="SICDReader not available")
@pytest.mark.requires_data
class TestCSIProcessorGPU:

    @pytest.fixture(autouse=True)
    def _load_chip(self, require_umbra_file):
        with SICDReader(str(require_umbra_file)) as reader:
            self.metadata = reader.metadata
            shape = reader.get_shape()
            cx, cy = shape[0] // 2, shape[1] // 2
            half = min(256, shape[0] // 2, shape[1] // 2)
            self.chip = reader.read_chip(cx - half, cx + half,
                                         cy - half, cy + half)

    def test_csi_cupy_input_returns_cupy(self):
        """apply() with cupy input returns a cupy array."""
        csi = CSIProcessor(metadata=self.metadata)
        result = csi.apply(cp.asarray(self.chip))
        assert isinstance(result, cp.ndarray), (
            f"Expected cp.ndarray, got {type(result).__name__}"
        )

    def test_csi_cupy_matches_numpy(self):
        """CuPy and numpy paths produce numerically equivalent results."""
        csi = CSIProcessor(metadata=self.metadata, normalization="percentile")
        cpu_result = csi.apply(self.chip)
        gpu_result = cp.asnumpy(csi.apply(cp.asarray(self.chip)))
        assert np.allclose(cpu_result, gpu_result, atol=1e-4), (
            "CuPy and numpy CSI apply() results differ beyond tolerance"
        )


# =============================================================================
# ToDecibels — CuPy GPU path (synthetic data, no file fixture required)
# =============================================================================


@pytest.mark.skipif(not _HAS_TO_DB, reason="ToDecibels not available")
class TestToDecibelsGPU:
    """GPU path tests for ToDecibels using synthetic complex data."""

    @pytest.fixture(autouse=True)
    def _make_data(self):
        rng = np.random.default_rng(42)
        real = rng.standard_normal((128, 128)).astype(np.float32)
        imag = rng.standard_normal((128, 128)).astype(np.float32)
        self.image_cpu = (real + 1j * imag).astype(np.complex64)
        self.image_gpu = cp.asarray(self.image_cpu)

    def test_apply_cupy_input_returns_cupy(self):
        """apply() with cupy input returns a cupy array."""
        to_db = ToDecibels()
        result = to_db.apply(self.image_gpu)
        assert isinstance(result, cp.ndarray), (
            f"Expected cp.ndarray, got {type(result).__name__}"
        )

    def test_apply_cupy_shape(self):
        """Output shape matches numpy path for cupy input."""
        to_db = ToDecibels()
        cpu_result = to_db.apply(self.image_cpu)
        gpu_result = to_db.apply(self.image_gpu)
        assert gpu_result.shape == cpu_result.shape, (
            f"Shape mismatch: GPU {gpu_result.shape} vs CPU {cpu_result.shape}"
        )

    def test_apply_cupy_matches_numpy(self):
        """CuPy and numpy paths produce numerically identical dB values."""
        to_db = ToDecibels()
        cpu_result = to_db.apply(self.image_cpu)
        gpu_result = cp.asnumpy(to_db.apply(self.image_gpu))
        assert np.allclose(cpu_result, gpu_result, atol=1e-5), (
            "CuPy and numpy ToDecibels results differ beyond tolerance"
        )
