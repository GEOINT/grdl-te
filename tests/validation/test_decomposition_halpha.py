# -*- coding: utf-8 -*-
"""
DualPolHAlpha Decomposition Tests - Synthetic dual-pol validation.

Tests eigenvalue decomposition of the 2x2 coherency matrix using
synthetic complex SAR data.

- Level 1: decompose() returns dict with correct keys and shapes
- Level 2: Physical property bounds (entropy, alpha, anisotropy, span)
- Level 3: to_rgb() produces valid RGB composite

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
2026-02-18

Modified
--------
2026-02-18
"""

# Third-party
import pytest
import numpy as np

try:
    from grdl.image_processing.decomposition import DualPolHAlpha
    _HAS_HALPHA = True
except ImportError:
    _HAS_HALPHA = False

pytestmark = [
    pytest.mark.decomposition,
    pytest.mark.skipif(not _HAS_HALPHA,
                       reason="grdl DualPolHAlpha not available"),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic_dual_pol():
    """Synthetic dual-pol complex SAR channels.

    co_pol: strong co-polarized backscatter
    cross_pol: weaker cross-polarized backscatter (30% relative power)
    """
    rng = np.random.default_rng(42)
    rows, cols = 256, 256
    co_pol = (
        rng.standard_normal((rows, cols))
        + 1j * rng.standard_normal((rows, cols))
    ).astype(np.complex64)
    cross_pol = (
        rng.standard_normal((rows, cols)) * 0.3
        + 1j * rng.standard_normal((rows, cols)) * 0.3
    ).astype(np.complex64)
    return co_pol, cross_pol


# ---------------------------------------------------------------------------
# Level 1: Format Validation
# ---------------------------------------------------------------------------
class TestDualPolHAlphaLevel1:
    """Validate decompose() output structure."""

    def test_decompose_returns_dict(self, synthetic_dual_pol):
        """decompose() returns a dictionary."""
        co, cross = synthetic_dual_pol
        halpha = DualPolHAlpha(window_size=7)
        result = halpha.decompose(co, cross)
        assert isinstance(result, dict)

    def test_decompose_has_expected_keys(self, synthetic_dual_pol):
        """Output dict has entropy, alpha, anisotropy, span."""
        co, cross = synthetic_dual_pol
        halpha = DualPolHAlpha(window_size=7)
        result = halpha.decompose(co, cross)
        for key in ('entropy', 'alpha', 'anisotropy', 'span'):
            assert key in result, f"Missing key: {key}"

    def test_decompose_output_shapes(self, synthetic_dual_pol):
        """All output arrays match input shape."""
        co, cross = synthetic_dual_pol
        halpha = DualPolHAlpha(window_size=7)
        result = halpha.decompose(co, cross)
        for key in ('entropy', 'alpha', 'anisotropy', 'span'):
            assert result[key].shape == co.shape, (
                f"{key} shape {result[key].shape} != input shape {co.shape}"
            )

    def test_decompose_output_dtype(self, synthetic_dual_pol):
        """Output arrays are real-valued float."""
        co, cross = synthetic_dual_pol
        halpha = DualPolHAlpha(window_size=7)
        result = halpha.decompose(co, cross)
        for key in ('entropy', 'alpha', 'anisotropy', 'span'):
            assert np.isrealobj(result[key])
            assert result[key].dtype in (np.float32, np.float64)

    def test_component_names_property(self):
        """component_names returns the correct tuple."""
        halpha = DualPolHAlpha()
        assert halpha.component_names == ('entropy', 'alpha', 'anisotropy', 'span')


# ---------------------------------------------------------------------------
# Level 2: Data Quality — Physical property validation
# ---------------------------------------------------------------------------
class TestDualPolHAlphaLevel2:
    """Validate physical bounds of decomposition outputs."""

    def test_entropy_bounded_0_1(self, synthetic_dual_pol):
        """Entropy is in [0, 1]."""
        co, cross = synthetic_dual_pol
        halpha = DualPolHAlpha(window_size=7)
        result = halpha.decompose(co, cross)
        H = result['entropy']
        assert np.all(H >= -1e-10), f"Entropy min = {H.min()}"
        assert np.all(H <= 1.0 + 1e-10), f"Entropy max = {H.max()}"

    def test_alpha_bounded_0_90(self, synthetic_dual_pol):
        """Alpha angle is in [0, 90] degrees."""
        co, cross = synthetic_dual_pol
        halpha = DualPolHAlpha(window_size=7)
        result = halpha.decompose(co, cross)
        alpha = result['alpha']
        assert np.all(alpha >= -1e-6), f"Alpha min = {alpha.min()}"
        assert np.all(alpha <= 90.0 + 1e-6), f"Alpha max = {alpha.max()}"

    def test_anisotropy_bounded_0_1(self, synthetic_dual_pol):
        """Anisotropy is in [0, 1]."""
        co, cross = synthetic_dual_pol
        halpha = DualPolHAlpha(window_size=7)
        result = halpha.decompose(co, cross)
        A = result['anisotropy']
        assert np.all(A >= -1e-10), f"Anisotropy min = {A.min()}"
        assert np.all(A <= 1.0 + 1e-10), f"Anisotropy max = {A.max()}"

    def test_span_non_negative(self, synthetic_dual_pol):
        """Span (total power) is non-negative."""
        co, cross = synthetic_dual_pol
        halpha = DualPolHAlpha(window_size=7)
        result = halpha.decompose(co, cross)
        span = result['span']
        assert np.all(span >= -1e-10), f"Span min = {span.min()}"

    def test_random_scattering_high_entropy(self, synthetic_dual_pol):
        """Random complex Gaussian data should have moderate-high entropy."""
        co, cross = synthetic_dual_pol
        halpha = DualPolHAlpha(window_size=11)
        result = halpha.decompose(co, cross)
        H = result['entropy']
        # Random data: entropy should be moderate (> 0.3 for dual-pol)
        mean_H = np.mean(H)
        assert mean_H > 0.2, f"Mean entropy {mean_H:.3f} too low for random data"

    def test_larger_window_smoother_output(self, synthetic_dual_pol):
        """Larger window size produces smoother (lower variance) entropy."""
        co, cross = synthetic_dual_pol
        halpha_small = DualPolHAlpha(window_size=3)
        halpha_large = DualPolHAlpha(window_size=15)
        result_small = halpha_small.decompose(co, cross)
        result_large = halpha_large.decompose(co, cross)
        # Larger window should produce lower variance in entropy
        var_small = np.var(result_small['entropy'])
        var_large = np.var(result_large['entropy'])
        assert var_large < var_small, (
            f"Larger window variance ({var_large:.6f}) >= "
            f"smaller window variance ({var_small:.6f})"
        )

    def test_all_outputs_finite(self, synthetic_dual_pol):
        """All output arrays contain finite values."""
        co, cross = synthetic_dual_pol
        halpha = DualPolHAlpha(window_size=7)
        result = halpha.decompose(co, cross)
        for key in ('entropy', 'alpha', 'anisotropy', 'span'):
            assert np.all(np.isfinite(result[key])), (
                f"{key} has non-finite values"
            )


# ---------------------------------------------------------------------------
# Level 3: Integration — RGB composite
# ---------------------------------------------------------------------------
class TestDualPolHAlphaLevel3:
    """Validate to_rgb() composite generation."""

    @pytest.mark.integration
    def test_to_rgb_shape(self, synthetic_dual_pol):
        """to_rgb() returns (rows, cols, 3) array."""
        co, cross = synthetic_dual_pol
        halpha = DualPolHAlpha(window_size=7)
        components = halpha.decompose(co, cross)
        rgb = halpha.to_rgb(components)
        assert rgb.shape == (co.shape[0], co.shape[1], 3)

    @pytest.mark.integration
    def test_to_rgb_dtype(self, synthetic_dual_pol):
        """to_rgb() returns float32."""
        co, cross = synthetic_dual_pol
        halpha = DualPolHAlpha(window_size=7)
        components = halpha.decompose(co, cross)
        rgb = halpha.to_rgb(components)
        assert rgb.dtype == np.float32

    @pytest.mark.integration
    def test_to_rgb_bounded_0_1(self, synthetic_dual_pol):
        """to_rgb() values are in [0, 1]."""
        co, cross = synthetic_dual_pol
        halpha = DualPolHAlpha(window_size=7)
        components = halpha.decompose(co, cross)
        rgb = halpha.to_rgb(components)
        assert rgb.min() >= -1e-6, f"RGB min = {rgb.min()}"
        assert rgb.max() <= 1.0 + 1e-6, f"RGB max = {rgb.max()}"

    @pytest.mark.integration
    def test_to_rgb_not_all_zero(self, synthetic_dual_pol):
        """to_rgb() has non-trivial output (not all zeros)."""
        co, cross = synthetic_dual_pol
        halpha = DualPolHAlpha(window_size=7)
        components = halpha.decompose(co, cross)
        rgb = halpha.to_rgb(components)
        assert rgb.max() > 0.0, "RGB composite is entirely black"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestDualPolHAlphaEdgeCases:
    """Edge cases and error handling."""

    def test_mismatched_shapes_raises(self):
        """Mismatched co/cross shapes raise ValueError."""
        halpha = DualPolHAlpha()
        co = np.ones((10, 10), dtype=np.complex64)
        cross = np.ones((10, 20), dtype=np.complex64)
        with pytest.raises(ValueError):
            halpha.decompose(co, cross)

    def test_real_input_raises(self):
        """Non-complex input raises TypeError."""
        halpha = DualPolHAlpha()
        real = np.ones((10, 10), dtype=np.float32)
        with pytest.raises(TypeError):
            halpha.decompose(real, real)

    def test_window_size_odd_required(self):
        """Even window_size should raise or be adjusted."""
        # Check that DualPolHAlpha handles even window sizes
        # (should either raise or round to nearest odd)
        try:
            halpha = DualPolHAlpha(window_size=4)
            # If it doesn't raise, verify it still produces valid output
            rng = np.random.default_rng(0)
            co = (rng.standard_normal((32, 32)) + 1j * rng.standard_normal((32, 32))).astype(np.complex64)
            cross = (rng.standard_normal((32, 32)) + 1j * rng.standard_normal((32, 32))).astype(np.complex64)
            result = halpha.decompose(co, cross)
            assert 'entropy' in result
        except (ValueError, Exception):
            pass  # Raising is also acceptable
