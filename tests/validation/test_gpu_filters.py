# -*- coding: utf-8 -*-
"""
GPU Filter Verification - CuPy path tests.

Verifies that all GPU-enabled grdl image filters correctly dispatch to cupy
when a cupy array is passed as input.  Covers linear, rank, statistical,
speckle, and phase gradient filters, plus the BandwiseTransformMixin 3D
GPU path (xp.stack fix in base.py).

Each test class exercises a single filter with synthetic data and checks:

- **Type correctness** — output is a ``cupy.ndarray`` (stays on-device).
- **Shape consistency** — GPU output shape matches the CPU (numpy) path.
- **Numerical equivalence** — GPU and CPU results agree within tolerance.

All tests use synthetic data (no real SAR files required) and are skipped
when CuPy is not installed.

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
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False

# --- grdl imports (guarded) ---

try:
    from grdl.image_processing.filters.linear import GaussianFilter, MeanFilter
    _HAS_LINEAR = True
except ImportError:
    _HAS_LINEAR = False

try:
    from grdl.image_processing.filters.rank import MaxFilter, MedianFilter, MinFilter
    _HAS_RANK = True
except ImportError:
    _HAS_RANK = False

try:
    from grdl.image_processing.filters.statistical import StdDevFilter
    _HAS_STATISTICAL = True
except ImportError:
    _HAS_STATISTICAL = False

try:
    from grdl.image_processing.filters.speckle import (
        ComplexLeeFilter,
        LeeFilter,
    )
    _HAS_SPECKLE = True
except ImportError:
    _HAS_SPECKLE = False

try:
    from grdl.image_processing.filters.phase import PhaseGradientFilter
    _HAS_PHASE = True
except ImportError:
    _HAS_PHASE = False


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(not _HAS_CUPY, reason="CuPy not installed"),
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
def float_2d(rng):
    """512x512 float32 array, values in [0, 1)."""
    return rng.random((512, 512), dtype=np.float32)


@pytest.fixture(scope='module')
def float_2d_positive(rng):
    """512x512 float32 array, strictly positive (SAR intensity-like)."""
    return (rng.random((512, 512), dtype=np.float32) + 0.01).astype(np.float32)


@pytest.fixture(scope='module')
def float_3d(rng):
    """(3, 256, 256) float32 array — multi-band input for BandwiseMixin tests."""
    return rng.random((3, 256, 256), dtype=np.float32)


@pytest.fixture(scope='module')
def complex_2d(rng):
    """512x512 complex128 array."""
    r = rng.standard_normal((512, 512)).astype(np.float64)
    i = rng.standard_normal((512, 512)).astype(np.float64)
    return (r + 1j * i).astype(np.complex128)


# =============================================================================
# Filters — linear
# =============================================================================


@pytest.mark.skipif(not _HAS_LINEAR, reason="grdl.image_processing.filters.linear not available")
class TestMeanFilterGPU:

    def test_returns_cupy(self, float_2d):
        result = MeanFilter(kernel_size=5).apply(cp.asarray(float_2d))
        assert isinstance(result, cp.ndarray), f"Expected cp.ndarray, got {type(result)}"

    def test_shape_matches_cpu(self, float_2d):
        f = MeanFilter(kernel_size=5)
        cpu = f.apply(float_2d)
        gpu = f.apply(cp.asarray(float_2d))
        assert gpu.shape == cpu.shape

    def test_numerics_match_cpu(self, float_2d):
        f = MeanFilter(kernel_size=5)
        cpu = f.apply(float_2d)
        gpu = cp.asnumpy(f.apply(cp.asarray(float_2d)))
        assert np.allclose(cpu, gpu, atol=_ATOL), "MeanFilter GPU/CPU mismatch"


@pytest.mark.skipif(not _HAS_LINEAR, reason="grdl.image_processing.filters.linear not available")
class TestGaussianFilterGPU:

    def test_returns_cupy(self, float_2d):
        result = GaussianFilter(sigma=2.0).apply(cp.asarray(float_2d))
        assert isinstance(result, cp.ndarray)

    def test_shape_matches_cpu(self, float_2d):
        f = GaussianFilter(sigma=2.0)
        assert f.apply(cp.asarray(float_2d)).shape == f.apply(float_2d).shape

    def test_numerics_match_cpu(self, float_2d):
        f = GaussianFilter(sigma=2.0)
        cpu = f.apply(float_2d)
        gpu = cp.asnumpy(f.apply(cp.asarray(float_2d)))
        assert np.allclose(cpu, gpu, atol=_ATOL), "GaussianFilter GPU/CPU mismatch"


# =============================================================================
# Filters — rank
# =============================================================================


@pytest.mark.skipif(not _HAS_RANK, reason="grdl.image_processing.filters.rank not available")
class TestMedianFilterGPU:

    def test_returns_cupy(self, float_2d):
        result = MedianFilter(kernel_size=5).apply(cp.asarray(float_2d))
        assert isinstance(result, cp.ndarray)

    def test_shape_matches_cpu(self, float_2d):
        f = MedianFilter(kernel_size=5)
        assert f.apply(cp.asarray(float_2d)).shape == f.apply(float_2d).shape

    def test_numerics_match_cpu(self, float_2d):
        f = MedianFilter(kernel_size=5)
        cpu = f.apply(float_2d)
        gpu = cp.asnumpy(f.apply(cp.asarray(float_2d)))
        assert np.allclose(cpu, gpu, atol=_ATOL), "MedianFilter GPU/CPU mismatch"


@pytest.mark.skipif(not _HAS_RANK, reason="grdl.image_processing.filters.rank not available")
class TestMinFilterGPU:

    def test_returns_cupy(self, float_2d):
        assert isinstance(MinFilter(kernel_size=3).apply(cp.asarray(float_2d)), cp.ndarray)

    def test_shape_matches_cpu(self, float_2d):
        f = MinFilter(kernel_size=3)
        assert f.apply(cp.asarray(float_2d)).shape == f.apply(float_2d).shape

    def test_numerics_match_cpu(self, float_2d):
        f = MinFilter(kernel_size=3)
        cpu = f.apply(float_2d)
        gpu = cp.asnumpy(f.apply(cp.asarray(float_2d)))
        assert np.allclose(cpu, gpu, atol=_ATOL), "MinFilter GPU/CPU mismatch"


@pytest.mark.skipif(not _HAS_RANK, reason="grdl.image_processing.filters.rank not available")
class TestMaxFilterGPU:

    def test_returns_cupy(self, float_2d):
        assert isinstance(MaxFilter(kernel_size=3).apply(cp.asarray(float_2d)), cp.ndarray)

    def test_shape_matches_cpu(self, float_2d):
        f = MaxFilter(kernel_size=3)
        assert f.apply(cp.asarray(float_2d)).shape == f.apply(float_2d).shape

    def test_numerics_match_cpu(self, float_2d):
        f = MaxFilter(kernel_size=3)
        cpu = f.apply(float_2d)
        gpu = cp.asnumpy(f.apply(cp.asarray(float_2d)))
        assert np.allclose(cpu, gpu, atol=_ATOL), "MaxFilter GPU/CPU mismatch"


# =============================================================================
# BandwiseTransformMixin — 3D GPU path (xp.stack fix in base.py)
# =============================================================================


@pytest.mark.skipif(not _HAS_LINEAR, reason="grdl.image_processing.filters.linear not available")
class TestBandwiseMixin3DGPU:
    """Verify BandwiseTransformMixin.apply() handles 3D cupy arrays (xp.stack fix)."""

    def test_returns_cupy_3d(self, float_3d):
        result = MeanFilter(kernel_size=5).apply(cp.asarray(float_3d))
        assert isinstance(result, cp.ndarray), f"Expected cp.ndarray, got {type(result)}"

    def test_shape_matches_cpu_3d(self, float_3d):
        f = MeanFilter(kernel_size=5)
        cpu = f.apply(float_3d)
        gpu = f.apply(cp.asarray(float_3d))
        assert gpu.shape == cpu.shape == (3, 256, 256)

    def test_numerics_match_cpu_3d(self, float_3d):
        f = MeanFilter(kernel_size=5)
        cpu = f.apply(float_3d)
        gpu = cp.asnumpy(f.apply(cp.asarray(float_3d)))
        assert np.allclose(cpu, gpu, atol=_ATOL), "3D BandwiseMixin GPU/CPU mismatch"


# =============================================================================
# Filters — statistical
# =============================================================================


@pytest.mark.skipif(not _HAS_STATISTICAL, reason="grdl.image_processing.filters.statistical not available")
class TestStdDevFilterGPU:

    def test_returns_cupy(self, float_2d):
        result = StdDevFilter(kernel_size=7).apply(cp.asarray(float_2d))
        assert isinstance(result, cp.ndarray)

    def test_shape_matches_cpu(self, float_2d):
        f = StdDevFilter(kernel_size=7)
        assert f.apply(cp.asarray(float_2d)).shape == f.apply(float_2d).shape

    def test_numerics_match_cpu(self, float_2d):
        f = StdDevFilter(kernel_size=7)
        cpu = f.apply(float_2d)
        gpu = cp.asnumpy(f.apply(cp.asarray(float_2d)))
        assert np.allclose(cpu, gpu, atol=_ATOL), "StdDevFilter GPU/CPU mismatch"


# =============================================================================
# Filters — speckle
# =============================================================================


@pytest.mark.skipif(not _HAS_SPECKLE, reason="grdl.image_processing.filters.speckle not available")
class TestLeeFilterGPU:

    def test_returns_cupy(self, float_2d_positive):
        result = LeeFilter(kernel_size=7).apply(cp.asarray(float_2d_positive))
        assert isinstance(result, cp.ndarray)

    def test_shape_matches_cpu(self, float_2d_positive):
        f = LeeFilter(kernel_size=7)
        cpu = f.apply(float_2d_positive)
        gpu = f.apply(cp.asarray(float_2d_positive))
        assert gpu.shape == cpu.shape

    def test_numerics_match_cpu(self, float_2d_positive):
        f = LeeFilter(kernel_size=7, enl=4.0)
        cpu = f.apply(float_2d_positive)
        gpu = cp.asnumpy(f.apply(cp.asarray(float_2d_positive)))
        assert np.allclose(cpu, gpu, atol=_ATOL), "LeeFilter GPU/CPU mismatch"


@pytest.mark.skipif(not _HAS_SPECKLE, reason="grdl.image_processing.filters.speckle not available")
class TestComplexLeeFilterGPU:

    def test_returns_cupy(self, complex_2d):
        result = ComplexLeeFilter(kernel_size=7).apply(cp.asarray(complex_2d))
        assert isinstance(result, cp.ndarray)

    def test_shape_matches_cpu(self, complex_2d):
        f = ComplexLeeFilter(kernel_size=7)
        cpu = f.apply(complex_2d)
        gpu = f.apply(cp.asarray(complex_2d))
        assert gpu.shape == cpu.shape

    def test_numerics_match_cpu(self, complex_2d):
        f = ComplexLeeFilter(kernel_size=7, enl=4.0)
        cpu = f.apply(complex_2d)
        gpu = cp.asnumpy(f.apply(cp.asarray(complex_2d)))
        assert np.allclose(np.abs(cpu), np.abs(gpu), atol=_ATOL), (
            "ComplexLeeFilter GPU/CPU magnitude mismatch"
        )


# =============================================================================
# Filters — phase gradient
# =============================================================================


@pytest.mark.skipif(not _HAS_PHASE, reason="grdl.image_processing.filters.phase not available")
class TestPhaseGradientFilterGPU:

    def test_magnitude_returns_cupy(self, complex_2d):
        f = PhaseGradientFilter(kernel_size=5, direction='magnitude')
        assert isinstance(f.apply(cp.asarray(complex_2d)), cp.ndarray)

    def test_row_returns_cupy(self, complex_2d):
        f = PhaseGradientFilter(kernel_size=5, direction='row')
        assert isinstance(f.apply(cp.asarray(complex_2d)), cp.ndarray)

    def test_col_returns_cupy(self, complex_2d):
        f = PhaseGradientFilter(kernel_size=5, direction='col')
        assert isinstance(f.apply(cp.asarray(complex_2d)), cp.ndarray)

    def test_shape_matches_cpu(self, complex_2d):
        for direction in ('magnitude', 'row', 'col'):
            f = PhaseGradientFilter(kernel_size=5, direction=direction)
            cpu = f.apply(complex_2d)
            gpu = f.apply(cp.asarray(complex_2d))
            assert gpu.shape == cpu.shape, f"Shape mismatch for direction={direction}"

    def test_numerics_match_cpu(self, complex_2d):
        for direction in ('magnitude', 'row', 'col'):
            f = PhaseGradientFilter(kernel_size=5, direction=direction)
            cpu = f.apply(complex_2d)
            gpu = cp.asnumpy(f.apply(cp.asarray(complex_2d)))
            assert np.allclose(cpu, gpu, atol=_ATOL), (
                f"PhaseGradientFilter GPU/CPU mismatch for direction={direction}"
            )
