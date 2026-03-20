# -*- coding: utf-8 -*-
"""
GPU Image Formation Verification - CuPy path tests (Pattern A).

Verifies that SAR image formation algorithms (PFA compress, RDA range
compress, RDA azimuth FFT) correctly dispatch FFT operations to cupy
when a cupy array is passed as input.

Each test class checks:

- **Type correctness** — output is a ``cupy.ndarray`` (stays on-device).
- **Shape consistency** — GPU output shape matches the CPU (numpy) path.
- **Numerical equivalence** — GPU and CPU results agree within tolerance.

All tests use synthetic k-space data (no real CPHD files required) and
are skipped when CuPy is not installed.

Dependencies
------------
cupy
pytest
numpy

Author
------
Ava Courtney
courtney-ava@zai.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-03-10

Modified
--------
2026-03-10
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

# --- grdl imports (guarded) ---

try:
    from grdl.image_processing.sar.image_formation.pfa import PolarFormatAlgorithm
    from grdl.image_processing.sar.image_formation.rda import RangeDopplerAlgorithm
    _HAS_IMAGE_FORMATION = True
except ImportError:
    _HAS_IMAGE_FORMATION = False


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(not _HAS_CUPY, reason="CuPy not installed"),
    pytest.mark.skipif(not _HAS_IMAGE_FORMATION,
                       reason="grdl image formation not available"),
]

# Tolerance for numerical comparisons
_ATOL = 1e-4


# =============================================================================
# Shared synthetic fixtures
# =============================================================================

@pytest.fixture(scope='module')
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope='module')
def kspace_2d(rng):
    """128x128 complex128 synthetic k-space data for PFA/RDA tests."""
    r = rng.standard_normal((128, 128)).astype(np.float64)
    i = rng.standard_normal((128, 128)).astype(np.float64)
    return (r + 1j * i).astype(np.complex128)


# =============================================================================
# PFA compress — 2D FFT dispatch
# =============================================================================


class TestPFACompressGPU:
    """Verify PFA compress() GPU FFT dispatch (Pattern A)."""

    @staticmethod
    def _make_pfa(phase_sgn=-1):
        """Construct PFA without real PolarGrid (compress() doesn't use grid)."""
        return PolarFormatAlgorithm(grid=None, weighting=None, phase_sgn=phase_sgn)

    def test_returns_cupy(self, kspace_2d):
        result = self._make_pfa().compress(cp.asarray(kspace_2d), pad_factor=1.0)
        assert isinstance(result, cp.ndarray), f"Expected cp.ndarray, got {type(result)}"

    def test_shape_matches_cpu(self, kspace_2d):
        pfa = self._make_pfa()
        cpu = pfa.compress(kspace_2d, pad_factor=1.0)
        gpu = pfa.compress(cp.asarray(kspace_2d), pad_factor=1.0)
        assert gpu.shape == cpu.shape

    def test_numerics_match_cpu(self, kspace_2d):
        pfa = self._make_pfa()
        cpu = pfa.compress(kspace_2d, pad_factor=1.0)
        gpu = cp.asnumpy(pfa.compress(cp.asarray(kspace_2d), pad_factor=1.0))
        assert np.allclose(np.abs(cpu), np.abs(gpu), atol=_ATOL), (
            "PFA compress GPU/CPU mismatch"
        )

    def test_padded_shape_matches_cpu(self, kspace_2d):
        pfa = self._make_pfa()
        cpu = pfa.compress(kspace_2d, pad_factor=1.5)
        gpu = pfa.compress(cp.asarray(kspace_2d), pad_factor=1.5)
        assert gpu.shape == cpu.shape

    def test_fft2_path(self, kspace_2d):
        """Verify the fft2 path (phase_sgn=+1)."""
        pfa = self._make_pfa(phase_sgn=1)
        cpu = pfa.compress(kspace_2d, pad_factor=1.0)
        gpu = cp.asnumpy(pfa.compress(cp.asarray(kspace_2d), pad_factor=1.0))
        assert np.allclose(np.abs(cpu), np.abs(gpu), atol=_ATOL), (
            "PFA compress fft2 path GPU/CPU mismatch"
        )


# =============================================================================
# RDA range compress + azimuth FFT — 1D FFT dispatch
# =============================================================================


class TestRDARangeCompressGPU:
    """Verify RDA range_compress() and azimuth_fft() GPU FFT dispatch."""

    @staticmethod
    def _make_rda():
        """Construct minimal RDA bypassing heavy __init__."""
        rda = object.__new__(RangeDopplerAlgorithm)
        rda._range_weight_func = None
        return rda

    def test_range_compress_returns_cupy(self, kspace_2d):
        result = self._make_rda().range_compress(cp.asarray(kspace_2d))
        assert isinstance(result, cp.ndarray)

    def test_range_compress_shape(self, kspace_2d):
        rda = self._make_rda()
        cpu = rda.range_compress(kspace_2d)
        gpu = rda.range_compress(cp.asarray(kspace_2d))
        assert gpu.shape == cpu.shape

    def test_range_compress_numerics(self, kspace_2d):
        rda = self._make_rda()
        cpu = rda.range_compress(kspace_2d)
        gpu = cp.asnumpy(rda.range_compress(cp.asarray(kspace_2d)))
        assert np.allclose(np.abs(cpu), np.abs(gpu), atol=_ATOL), (
            "RDA range_compress GPU/CPU mismatch"
        )

    def test_azimuth_fft_returns_cupy(self, kspace_2d):
        result = self._make_rda().azimuth_fft(cp.asarray(kspace_2d))
        assert isinstance(result, cp.ndarray)

    def test_azimuth_fft_numerics(self, kspace_2d):
        rda = self._make_rda()
        cpu = rda.azimuth_fft(kspace_2d)
        gpu = cp.asnumpy(rda.azimuth_fft(cp.asarray(kspace_2d)))
        assert np.allclose(np.abs(cpu), np.abs(gpu), atol=_ATOL), (
            "RDA azimuth_fft GPU/CPU mismatch"
        )
