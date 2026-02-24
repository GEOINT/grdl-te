# -*- coding: utf-8 -*-
"""
Intensity Transform Validation - ToDecibels and PercentileStretch.

Tests dB conversion and percentile-based contrast stretching against
manual numpy golden references with synthetic data.

- Level 1: Output shape, dtype
- Level 2: Golden reference accuracy, property invariants, edge cases

Dependencies
------------
pytest
numpy

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

try:
    from grdl.image_processing.intensity import ToDecibels, PercentileStretch
    _HAS_INTENSITY = True
except ImportError:
    _HAS_INTENSITY = False

pytestmark = [
    pytest.mark.intensity,
    pytest.mark.skipif(not _HAS_INTENSITY,
                       reason="grdl intensity transforms not available"),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def positive_real_2d():
    """64x64 positive float64 array (no zeros, safe for log10)."""
    rng = np.random.default_rng(42)
    return np.abs(rng.standard_normal((64, 64))).astype(np.float64) + 0.01


@pytest.fixture(scope="module")
def complex_2d():
    """64x64 complex128 random array."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal((64, 64))
            + 1j * rng.standard_normal((64, 64))).astype(np.complex128)


# ===================================================================
# ToDecibels
# ===================================================================

class TestToDecibelsLevel1:
    """Validate ToDecibels output shape and dtype."""

    def test_to_db_output_shape(self, positive_real_2d):
        result = ToDecibels().apply(positive_real_2d)
        assert result.shape == positive_real_2d.shape

    def test_to_db_output_dtype_float64(self, positive_real_2d):
        result = ToDecibels().apply(positive_real_2d)
        assert result.dtype == np.float64

    def test_to_db_3d_input(self):
        rng = np.random.default_rng(42)
        source = np.abs(rng.standard_normal((3, 64, 64))) + 0.01
        result = ToDecibels().apply(source)
        assert result.shape == (3, 64, 64)


class TestToDecibelsLevel2:
    """Validate ToDecibels numerical accuracy."""

    def test_to_db_golden_real(self, positive_real_2d):
        result = ToDecibels(floor_db=-60.0).apply(positive_real_2d)
        eps = np.finfo(np.float64).tiny
        expected = 20.0 * np.log10(np.abs(positive_real_2d) + eps)
        np.maximum(expected, -60.0, out=expected)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_to_db_golden_complex(self, complex_2d):
        result = ToDecibels(floor_db=-60.0).apply(complex_2d)
        eps = np.finfo(np.float64).tiny
        expected = 20.0 * np.log10(np.abs(complex_2d) + eps)
        np.maximum(expected, -60.0, out=expected)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_to_db_floor_clamp(self):
        source = np.full((16, 16), 1e-30)
        result = ToDecibels(floor_db=-40.0).apply(source)
        assert result.min() >= -40.0, (
            f"Floor clamp failed: min dB = {result.min()}"
        )

    def test_to_db_custom_floor(self, positive_real_2d):
        r1 = ToDecibels(floor_db=-60.0).apply(positive_real_2d)
        r2 = ToDecibels(floor_db=-30.0).apply(positive_real_2d)
        assert not np.array_equal(r1, r2), (
            "Different floor_db values should produce different results"
        )

    def test_to_db_zero_input(self):
        source = np.zeros((16, 16))
        result = ToDecibels(floor_db=-60.0).apply(source)
        np.testing.assert_allclose(result, -60.0)

    def test_to_db_known_value_ones(self):
        """|1.0| = 1.0; 20*log10(1.0 + tiny) ~ 0.0."""
        source = np.ones((4, 4))
        result = ToDecibels().apply(source)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)


# ===================================================================
# PercentileStretch
# ===================================================================

class TestPercentileStretchLevel1:
    """Validate PercentileStretch output shape, dtype, and construction."""

    def test_stretch_output_shape(self, positive_real_2d):
        result = PercentileStretch().apply(positive_real_2d)
        assert result.shape == positive_real_2d.shape

    def test_stretch_output_dtype_float32(self, positive_real_2d):
        result = PercentileStretch().apply(positive_real_2d)
        assert result.dtype == np.float32

    def test_stretch_plow_equals_phigh_raises(self):
        with pytest.raises(ValueError):
            PercentileStretch(plow=50.0, phigh=50.0)


class TestPercentileStretchLevel2:
    """Validate PercentileStretch numerical accuracy."""

    def test_stretch_output_range(self, positive_real_2d):
        result = PercentileStretch().apply(positive_real_2d)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_stretch_golden_reference(self, positive_real_2d):
        result = PercentileStretch(plow=2.0, phigh=98.0).apply(
            positive_real_2d)
        vmin = np.percentile(positive_real_2d, 2.0)
        vmax = np.percentile(positive_real_2d, 98.0)
        expected = np.clip(
            (positive_real_2d - vmin) / (vmax - vmin), 0.0, 1.0
        ).astype(np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_stretch_custom_percentiles(self, positive_real_2d):
        r1 = PercentileStretch(plow=1.0, phigh=99.0).apply(positive_real_2d)
        r2 = PercentileStretch(plow=5.0, phigh=95.0).apply(positive_real_2d)
        assert not np.array_equal(r1, r2)

    def test_stretch_constant_array(self):
        source = np.full((16, 16), 5.0)
        result = PercentileStretch().apply(source)
        np.testing.assert_allclose(result, 0.0)

    def test_stretch_3d_input(self):
        rng = np.random.default_rng(42)
        source = rng.random((3, 64, 64)).astype(np.float64)
        result = PercentileStretch().apply(source)
        assert result.shape == (3, 64, 64)

    def test_stretch_known_linear(self):
        """plow=0, phigh=100 on linspace should map to ~[0, 1]."""
        source = np.linspace(0, 100, 101).astype(np.float64)
        result = PercentileStretch(plow=0.0, phigh=100.0).apply(source)
        expected = np.linspace(0, 1, 101).astype(np.float32)
        np.testing.assert_allclose(result, expected, atol=0.02)
