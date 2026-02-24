# -*- coding: utf-8 -*-
"""
Spatial Filter Validation - Numerical correctness for all 9 spatial filters.

Tests linear, rank, statistical, speckle, and phase gradient filters against
scipy golden references with synthetic data. All filters are tested for shape
preservation, dtype behavior, bandwise 3D dispatch, golden-reference numerical
accuracy, and edge-case robustness.

- Level 1: Output shape, dtype, bandwise dispatch
- Level 2: Golden reference accuracy, property invariants

Dependencies
------------
pytest
numpy
scipy

Author
------
Claude Code (generated)

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2026-02-24
"""

import pytest
import numpy as np
from scipy.ndimage import (
    gaussian_filter,
    maximum_filter,
    median_filter,
    minimum_filter,
    uniform_filter,
)

try:
    from grdl.image_processing.filters import (
        MeanFilter,
        GaussianFilter,
        MedianFilter,
        MinFilter,
        MaxFilter,
        StdDevFilter,
        LeeFilter,
        ComplexLeeFilter,
        PhaseGradientFilter,
    )
    from grdl.exceptions import ValidationError
    _HAS_FILTERS = True
except ImportError:
    _HAS_FILTERS = False

pytestmark = [
    pytest.mark.filters,
    pytest.mark.skipif(not _HAS_FILTERS, reason="grdl filters not available"),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_2d():
    """64x64 float64 random array for real-valued filter tests."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((64, 64)).astype(np.float64)


@pytest.fixture(scope="module")
def real_3d():
    """3x64x64 float64 random array for bandwise dispatch tests."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((3, 64, 64)).astype(np.float64)


@pytest.fixture(scope="module")
def complex_2d():
    """64x64 complex128 random array for complex filter tests."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal((64, 64))
            + 1j * rng.standard_normal((64, 64))).astype(np.complex128)


@pytest.fixture(scope="module")
def complex_3d():
    """3x64x64 complex128 random array for bandwise complex tests."""
    rng = np.random.default_rng(42)
    c = (rng.standard_normal((3, 64, 64))
         + 1j * rng.standard_normal((3, 64, 64)))
    return c.astype(np.complex128)


@pytest.fixture(scope="module")
def uint8_2d():
    """64x64 uint8 random array for dtype-preservation tests."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (64, 64), dtype=np.uint8)


# ===================================================================
# MeanFilter
# ===================================================================

class TestMeanFilterLevel1:
    """Validate MeanFilter output shape and dtype."""

    def test_mean_output_shape_2d(self, real_2d):
        result = MeanFilter().apply(real_2d)
        assert result.shape == real_2d.shape

    def test_mean_output_shape_3d(self, real_3d):
        result = MeanFilter().apply(real_3d)
        assert result.shape == real_3d.shape

    def test_mean_output_dtype_float32(self, real_2d):
        source = real_2d.astype(np.float32)
        result = MeanFilter().apply(source)
        assert result.dtype == np.float32

    def test_mean_output_dtype_float64(self, real_2d):
        result = MeanFilter().apply(real_2d)
        assert result.dtype == np.float64


class TestMeanFilterLevel2:
    """Validate MeanFilter numerical accuracy against scipy golden reference."""

    def test_mean_golden_reference(self, real_2d):
        result = MeanFilter(kernel_size=3, mode='reflect').apply(real_2d)
        golden = uniform_filter(real_2d.astype(np.float64), size=3,
                                mode='reflect')
        np.testing.assert_allclose(result, golden, rtol=1e-6)

    def test_mean_custom_kernel_size(self, real_2d):
        result = MeanFilter(kernel_size=5, mode='reflect').apply(real_2d)
        golden = uniform_filter(real_2d.astype(np.float64), size=5,
                                mode='reflect')
        np.testing.assert_allclose(result, golden, rtol=1e-6)

    def test_mean_constant_array(self):
        source = np.full((64, 64), 7.5)
        result = MeanFilter().apply(source)
        np.testing.assert_allclose(result, 7.5, atol=1e-12)

    def test_mean_invalid_kernel_size_even(self):
        with pytest.raises((ValidationError, ValueError)):
            MeanFilter(kernel_size=4)


# ===================================================================
# GaussianFilter
# ===================================================================

class TestGaussianFilterLevel1:
    """Validate GaussianFilter output shape and dtype."""

    def test_gaussian_output_shape_2d(self, real_2d):
        result = GaussianFilter().apply(real_2d)
        assert result.shape == real_2d.shape

    def test_gaussian_output_dtype_float32(self, real_2d):
        source = real_2d.astype(np.float32)
        result = GaussianFilter().apply(source)
        assert result.dtype == np.float32

    def test_gaussian_output_dtype_float64(self, real_2d):
        result = GaussianFilter().apply(real_2d)
        assert result.dtype == np.float64


class TestGaussianFilterLevel2:
    """Validate GaussianFilter against scipy golden reference."""

    def test_gaussian_golden_reference(self, real_2d):
        result = GaussianFilter(sigma=1.0, truncate=4.0,
                                mode='reflect').apply(real_2d)
        golden = gaussian_filter(real_2d.astype(np.float64), sigma=1.0,
                                 truncate=4.0, mode='reflect')
        np.testing.assert_allclose(result, golden, rtol=1e-6)

    def test_gaussian_sigma_effect(self, real_2d):
        low = GaussianFilter(sigma=0.5).apply(real_2d)
        high = GaussianFilter(sigma=3.0).apply(real_2d)
        assert np.var(high) < np.var(low), (
            "Higher sigma should produce stronger smoothing (lower variance)"
        )

    def test_gaussian_constant_array(self):
        source = np.full((64, 64), 3.14)
        result = GaussianFilter().apply(source)
        np.testing.assert_allclose(result, 3.14, atol=1e-12)

    def test_gaussian_3d_bandwise(self, real_3d):
        result = GaussianFilter(sigma=1.0).apply(real_3d)
        for b in range(real_3d.shape[0]):
            band_result = GaussianFilter(sigma=1.0).apply(real_3d[b])
            np.testing.assert_allclose(result[b], band_result, rtol=1e-12)


# ===================================================================
# MedianFilter
# ===================================================================

class TestMedianFilterLevel1:
    """Validate MedianFilter output shape, dtype, and bandwise dispatch."""

    def test_median_output_shape(self, real_2d):
        result = MedianFilter().apply(real_2d)
        assert result.shape == real_2d.shape

    def test_median_preserves_dtype_uint8(self, uint8_2d):
        result = MedianFilter().apply(uint8_2d)
        assert result.dtype == np.uint8

    def test_median_preserves_dtype_float64(self, real_2d):
        result = MedianFilter().apply(real_2d)
        assert result.dtype == np.float64

    def test_median_3d_bandwise(self, real_3d):
        result = MedianFilter().apply(real_3d)
        for b in range(real_3d.shape[0]):
            band_result = MedianFilter().apply(real_3d[b])
            np.testing.assert_array_equal(result[b], band_result)


class TestMedianFilterLevel2:
    """Validate MedianFilter against scipy golden reference."""

    def test_median_golden_reference(self, real_2d):
        result = MedianFilter(kernel_size=3, mode='reflect').apply(real_2d)
        golden = median_filter(real_2d, size=3, mode='reflect')
        np.testing.assert_array_equal(result, golden)

    def test_median_removes_salt_pepper(self):
        rng = np.random.default_rng(42)
        source = rng.random((64, 64)).astype(np.float64)
        # Inject salt-and-pepper outliers
        outlier_positions = [(10, 10), (20, 30), (40, 50), (55, 15), (32, 32)]
        for r, c in outlier_positions:
            source[r, c] = 1000.0
        result = MedianFilter(kernel_size=3).apply(source)
        for r, c in outlier_positions:
            assert result[r, c] < 10.0, (
                f"Outlier at ({r},{c}) not removed: {result[r,c]}"
            )

    def test_median_constant_array(self):
        source = np.full((64, 64), 42.0)
        result = MedianFilter().apply(source)
        np.testing.assert_allclose(result, 42.0)


# ===================================================================
# MinFilter
# ===================================================================

class TestMinFilterLevel1:
    """Validate MinFilter output shape and dtype."""

    def test_min_output_shape(self, real_2d):
        result = MinFilter().apply(real_2d)
        assert result.shape == real_2d.shape

    def test_min_preserves_dtype(self, uint8_2d):
        result = MinFilter().apply(uint8_2d)
        assert result.dtype == np.uint8


class TestMinFilterLevel2:
    """Validate MinFilter against scipy golden reference."""

    def test_min_golden_reference(self, real_2d):
        result = MinFilter(kernel_size=3, mode='reflect').apply(real_2d)
        golden = minimum_filter(real_2d, size=3, mode='reflect')
        np.testing.assert_array_equal(result, golden)

    def test_min_output_leq_input(self, real_2d):
        result = MinFilter().apply(real_2d)
        assert np.all(result <= real_2d), (
            "Local minimum must be <= center pixel"
        )

    def test_min_constant_array(self):
        source = np.full((64, 64), 5.0)
        result = MinFilter().apply(source)
        np.testing.assert_allclose(result, 5.0)


# ===================================================================
# MaxFilter
# ===================================================================

class TestMaxFilterLevel1:
    """Validate MaxFilter output shape and dtype."""

    def test_max_output_shape(self, real_2d):
        result = MaxFilter().apply(real_2d)
        assert result.shape == real_2d.shape

    def test_max_preserves_dtype(self, uint8_2d):
        result = MaxFilter().apply(uint8_2d)
        assert result.dtype == np.uint8


class TestMaxFilterLevel2:
    """Validate MaxFilter against scipy golden reference."""

    def test_max_golden_reference(self, real_2d):
        result = MaxFilter(kernel_size=3, mode='reflect').apply(real_2d)
        golden = maximum_filter(real_2d, size=3, mode='reflect')
        np.testing.assert_array_equal(result, golden)

    def test_max_output_geq_input(self, real_2d):
        result = MaxFilter().apply(real_2d)
        assert np.all(result >= real_2d), (
            "Local maximum must be >= center pixel"
        )

    def test_max_constant_array(self):
        source = np.full((64, 64), 5.0)
        result = MaxFilter().apply(source)
        np.testing.assert_allclose(result, 5.0)


# ===================================================================
# StdDevFilter
# ===================================================================

class TestStdDevFilterLevel1:
    """Validate StdDevFilter output shape and dtype."""

    def test_stddev_output_shape(self, real_2d):
        result = StdDevFilter().apply(real_2d)
        assert result.shape == real_2d.shape

    def test_stddev_output_dtype_always_float64(self, uint8_2d):
        result = StdDevFilter().apply(uint8_2d)
        assert result.dtype == np.float64

    def test_stddev_3d_bandwise(self, real_3d):
        result = StdDevFilter().apply(real_3d)
        for b in range(real_3d.shape[0]):
            band_result = StdDevFilter().apply(real_3d[b])
            np.testing.assert_allclose(result[b], band_result, rtol=1e-12)


class TestStdDevFilterLevel2:
    """Validate StdDevFilter using variance decomposition golden reference."""

    def test_stddev_golden_reference(self, real_2d):
        result = StdDevFilter(kernel_size=3, mode='reflect').apply(real_2d)
        x = real_2d.astype(np.float64)
        mean_x = uniform_filter(x, size=3, mode='reflect')
        mean_x2 = uniform_filter(x * x, size=3, mode='reflect')
        variance = mean_x2 - mean_x * mean_x
        np.maximum(variance, 0.0, out=variance)
        golden = np.sqrt(variance)
        np.testing.assert_allclose(result, golden, atol=1e-12)

    def test_stddev_nonnegative(self, real_2d):
        result = StdDevFilter().apply(real_2d)
        assert np.all(result >= 0), "Standard deviation cannot be negative"

    def test_stddev_constant_array_is_zero(self):
        source = np.full((64, 64), 99.0)
        result = StdDevFilter().apply(source)
        np.testing.assert_allclose(result, 0.0, atol=1e-14)

    def test_stddev_known_checkerboard(self):
        """Checkerboard with values 0 and 1: interior 3x3 windows contain
        mixed values, so std > 0 everywhere."""
        source = np.zeros((64, 64), dtype=np.float64)
        source[::2, ::2] = 1.0
        source[1::2, 1::2] = 1.0
        result = StdDevFilter(kernel_size=3).apply(source)
        # Interior pixels should have non-zero std
        assert np.all(result[5:-5, 5:-5] > 0)


# ===================================================================
# LeeFilter
# ===================================================================

@pytest.mark.sar
class TestLeeFilterLevel1:
    """Validate LeeFilter output shape and dtype."""

    def test_lee_output_shape(self, real_2d):
        result = LeeFilter().apply(real_2d)
        assert result.shape == real_2d.shape

    def test_lee_output_dtype_float64(self, real_2d):
        result = LeeFilter().apply(real_2d)
        assert result.dtype == np.float64

    def test_lee_3d_bandwise(self, real_3d):
        # Use absolute values since LeeFilter expects intensity-like data
        source = np.abs(real_3d)
        result = LeeFilter(kernel_size=3).apply(source)
        for b in range(source.shape[0]):
            band_result = LeeFilter(kernel_size=3).apply(source[b])
            np.testing.assert_allclose(result[b], band_result, rtol=1e-12)


@pytest.mark.sar
class TestLeeFilterLevel2:
    """Validate LeeFilter speckle-reduction properties."""

    def test_lee_reduces_variance(self):
        """Speckle filtering must reduce variance of SAR-like data."""
        rng = np.random.default_rng(42)
        real = rng.standard_normal((128, 128))
        imag = rng.standard_normal((128, 128))
        source = np.abs(real + 1j * imag).astype(np.float64)
        result = LeeFilter(kernel_size=7).apply(source)
        assert np.var(result) < np.var(source), (
            "Lee filter must reduce variance (speckle suppression)"
        )

    def test_lee_constant_array_unchanged(self):
        source = np.full((64, 64), 10.0)
        result = LeeFilter(kernel_size=7).apply(source)
        np.testing.assert_allclose(result, 10.0, atol=1e-10)

    def test_lee_auto_enl_estimation(self, real_2d):
        source = np.abs(real_2d) + 0.01
        result = LeeFilter(kernel_size=7, enl=0.0).apply(source)
        assert np.all(np.isfinite(result))

    def test_lee_fixed_enl(self, real_2d):
        source = np.abs(real_2d) + 0.01
        result_auto = LeeFilter(kernel_size=7, enl=0.0).apply(source)
        result_fixed = LeeFilter(kernel_size=7, enl=4.0).apply(source)
        assert not np.allclose(result_auto, result_fixed), (
            "Auto-ENL and fixed-ENL should produce different results"
        )

    def test_lee_homogeneous_region_smoothing(self):
        """On constant + small noise, Lee should smooth heavily (W ~ 0)."""
        rng = np.random.default_rng(42)
        base = 100.0
        noise = rng.standard_normal((128, 128)) * 0.1
        source = base + noise
        result = LeeFilter(kernel_size=7).apply(source)
        assert np.var(result) < 0.5 * np.var(source), (
            f"Lee filter should reduce variance of homogeneous regions: "
            f"input var={np.var(source):.6f}, output var={np.var(result):.6f}"
        )


# ===================================================================
# ComplexLeeFilter
# ===================================================================

@pytest.mark.sar
class TestComplexLeeFilterLevel1:
    """Validate ComplexLeeFilter output shape, dtype, and input validation."""

    def test_complex_lee_output_shape(self, complex_2d):
        result = ComplexLeeFilter(kernel_size=3).apply(complex_2d)
        assert result.shape == complex_2d.shape

    def test_complex_lee_output_dtype_complex128(self, complex_2d):
        result = ComplexLeeFilter(kernel_size=3).apply(complex_2d)
        assert result.dtype == np.complex128

    def test_complex_lee_rejects_real_input(self, real_2d):
        with pytest.raises(ValidationError):
            ComplexLeeFilter(kernel_size=3).apply(real_2d)


@pytest.mark.sar
class TestComplexLeeFilterLevel2:
    """Validate ComplexLeeFilter phase preservation and variance reduction."""

    def test_complex_lee_preserves_mean_phase(self, complex_2d):
        result = ComplexLeeFilter(kernel_size=3).apply(complex_2d)
        input_phase = np.angle(np.mean(complex_2d))
        output_phase = np.angle(np.mean(result))
        phase_diff = np.abs(input_phase - output_phase)
        # Wrap to [0, pi]
        phase_diff = min(phase_diff, 2 * np.pi - phase_diff)
        assert phase_diff < 0.3, (
            f"Mean phase shifted by {phase_diff:.3f} rad (limit 0.3)"
        )

    def test_complex_lee_reduces_intensity_variance(self, complex_2d):
        result = ComplexLeeFilter(kernel_size=7).apply(complex_2d)
        input_var = np.var(np.abs(complex_2d) ** 2)
        output_var = np.var(np.abs(result) ** 2)
        assert output_var < input_var, (
            "Complex Lee filter must reduce intensity variance"
        )

    def test_complex_lee_3d_bandwise(self, complex_3d):
        result = ComplexLeeFilter(kernel_size=3).apply(complex_3d)
        for b in range(complex_3d.shape[0]):
            band_result = ComplexLeeFilter(kernel_size=3).apply(complex_3d[b])
            np.testing.assert_allclose(result[b], band_result, rtol=1e-12)

    def test_complex_lee_auto_enl(self, complex_2d):
        result = ComplexLeeFilter(kernel_size=7, enl=0.0).apply(complex_2d)
        assert np.all(np.isfinite(result.real))
        assert np.all(np.isfinite(result.imag))


# ===================================================================
# PhaseGradientFilter
# ===================================================================

@pytest.mark.sar
class TestPhaseGradientFilterLevel1:
    """Validate PhaseGradientFilter output shape, dtype, and input validation."""

    def test_phase_grad_output_shape(self, complex_2d):
        result = PhaseGradientFilter().apply(complex_2d)
        assert result.shape == complex_2d.shape

    def test_phase_grad_output_dtype_float64(self, complex_2d):
        result = PhaseGradientFilter().apply(complex_2d)
        assert result.dtype == np.float64

    def test_phase_grad_rejects_real_input(self, real_2d):
        with pytest.raises(ValidationError):
            PhaseGradientFilter().apply(real_2d)

    def test_phase_grad_invalid_direction(self):
        with pytest.raises(ValidationError):
            PhaseGradientFilter(direction='diagonal')


@pytest.mark.sar
class TestPhaseGradientFilterLevel2:
    """Validate PhaseGradientFilter numerical properties."""

    def test_phase_grad_magnitude_nonnegative(self, complex_2d):
        result = PhaseGradientFilter(direction='magnitude').apply(complex_2d)
        assert np.all(result >= 0), "Gradient magnitude must be non-negative"

    def test_phase_grad_row_direction_range(self, complex_2d):
        result = PhaseGradientFilter(direction='row').apply(complex_2d)
        assert np.all(result >= -np.pi - 1e-10)
        assert np.all(result <= np.pi + 1e-10)

    def test_phase_grad_col_direction_range(self, complex_2d):
        result = PhaseGradientFilter(direction='col').apply(complex_2d)
        assert np.all(result >= -np.pi - 1e-10)
        assert np.all(result <= np.pi + 1e-10)

    def test_phase_grad_constant_phase_zero_gradient(self):
        """Constant-phase complex array should have zero gradient."""
        z = np.exp(1j * 0.5) * np.ones((64, 64))
        result = PhaseGradientFilter(direction='magnitude').apply(z)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_phase_grad_linear_phase_ramp_row(self):
        """Linear phase ramp in row direction should give known slope."""
        slope = 0.1  # rad/pixel
        rows, cols = 64, 64
        r = np.arange(rows)[:, np.newaxis]
        z = np.exp(1j * slope * r) * np.ones((rows, cols))
        result = PhaseGradientFilter(kernel_size=3,
                                     direction='row').apply(z)
        # Interior pixels (away from edges) should match the slope
        interior = result[5:-5, 5:-5]
        np.testing.assert_allclose(interior, slope, atol=0.05)

    def test_phase_grad_3d_bandwise(self, complex_3d):
        result = PhaseGradientFilter(kernel_size=3).apply(complex_3d)
        for b in range(complex_3d.shape[0]):
            band_result = PhaseGradientFilter(kernel_size=3).apply(
                complex_3d[b])
            np.testing.assert_allclose(result[b], band_result, rtol=1e-12)


# ===================================================================
# Shared Edge Cases
# ===================================================================

class TestFilterEdgeCases:
    """Edge cases shared across filter types."""

    @pytest.mark.parametrize("FilterClass", [
        MeanFilter, GaussianFilter, MedianFilter,
        MinFilter, MaxFilter, StdDevFilter, LeeFilter,
    ])
    def test_filter_single_pixel_real(self, FilterClass):
        source = np.array([[5.0]])
        if FilterClass == LeeFilter:
            filt = FilterClass(kernel_size=3)
        elif FilterClass == GaussianFilter:
            filt = FilterClass()
        else:
            filt = FilterClass(kernel_size=3)
        result = filt.apply(source)
        assert result.shape == (1, 1)
        assert np.all(np.isfinite(result))

    def test_filter_single_pixel_complex_lee(self):
        source = np.array([[1.0 + 2.0j]])
        filt = ComplexLeeFilter(kernel_size=3)
        result = filt.apply(source)
        assert result.shape == (1, 1)
        assert np.all(np.isfinite(np.abs(result)))

    def test_filter_small_array_phase_gradient(self):
        """PhaseGradientFilter needs at least 3x3 for meaningful gradients."""
        source = np.exp(1j * np.ones((3, 3)))
        filt = PhaseGradientFilter(kernel_size=3)
        result = filt.apply(source)
        assert result.shape == (3, 3)
        assert np.all(np.isfinite(result))

    @pytest.mark.parametrize("FilterClass", [
        MeanFilter, GaussianFilter, StdDevFilter,
    ])
    def test_filter_nan_propagation_linear(self, FilterClass):
        """Linear filters backed by uniform_filter propagate NaN."""
        source = np.full((16, 16), np.nan)
        if FilterClass == GaussianFilter:
            filt = FilterClass()
        else:
            filt = FilterClass(kernel_size=3)
        result = filt.apply(source)
        assert np.all(np.isnan(result))

    @pytest.mark.parametrize("FilterClass", [
        MeanFilter, GaussianFilter, MedianFilter,
    ])
    def test_filter_inf_handling(self, FilterClass):
        """Inf input should not raise an unhandled exception."""
        source = np.full((16, 16), np.inf)
        if FilterClass == GaussianFilter:
            filt = FilterClass()
        else:
            filt = FilterClass(kernel_size=3)
        # Should not raise — behavior is defined (though output may be inf/nan)
        result = filt.apply(source)
        assert result.shape == (16, 16)

    @pytest.mark.parametrize("mode", ['reflect', 'constant', 'nearest', 'wrap'])
    def test_filter_all_boundary_modes(self, real_2d, mode):
        result = MeanFilter(kernel_size=3, mode=mode).apply(real_2d)
        assert result.shape == real_2d.shape
        assert np.all(np.isfinite(result))

    def test_filter_invalid_kernel_even(self):
        with pytest.raises((ValidationError, ValueError)):
            MeanFilter(kernel_size=4)

    def test_filter_invalid_kernel_too_small(self):
        with pytest.raises((ValidationError, ValueError)):
            MeanFilter(kernel_size=1)
