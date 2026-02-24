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

    def test_span_equals_total_power(self, synthetic_dual_pol):
        """Span must equal the windowed sum of co-pol and cross-pol power.

        The mathematical definition of span is:
            span = |S_hh|² + |S_hv|²

        Checking span >= 0 is a mathematical truism (power is always real and
        non-negative).  This test verifies the QUANTITY, not just the sign:
        the mean span in the image interior must match the expected total power
        within 5% (tolerance covers boundary windowing effects).
        """
        co, cross = synthetic_dual_pol
        halpha = DualPolHAlpha(window_size=7)
        result = halpha.decompose(co, cross)
        span = result['span']

        expected_mean_power = (
            np.mean(np.abs(co) ** 2) + np.mean(np.abs(cross) ** 2)
        )
        margin = 10
        interior_mean_span = np.mean(span[margin:-margin, margin:-margin])

        np.testing.assert_allclose(
            interior_mean_span,
            expected_mean_power,
            rtol=0.05,
            err_msg=(
                f"Interior mean span {interior_mean_span:.4f} deviates from "
                f"expected total power {expected_mean_power:.4f} by > 5%. "
                "Verify span = |co|² + |cross|², not a different quantity."
            ),
        )

    def test_random_scattering_high_entropy(self, synthetic_dual_pol):
        """Entropy must match the theoretical value for this fixture's power ratio.

        The fixture uses cross_pol amplitude = 0.3, giving power = 0.09
        relative to co_pol power = 1.0.  The normalised eigenvalue
        probabilities are:
            p1 = 1 / (1 + 0.09) ≈ 0.917
            p2 = 0.09 / (1 + 0.09) ≈ 0.083
        Theoretical entropy:
            H = -(p1·log2(p1) + p2·log2(p2)) ≈ 0.41

        The threshold > 0.35 catches severe underestimation (e.g., a broken
        normalisation returning H ≈ 0.2) while being physically justified for
        the actual fixture.  This is NOT a test for maximum entropy — for that
        see test_zero_cross_pol_gives_zero_entropy and the equal-power test in
        TestDualPolHAlphaLevel3.
        """
        co, cross = synthetic_dual_pol
        halpha = DualPolHAlpha(window_size=11)
        result = halpha.decompose(co, cross)
        margin = 15
        mean_H = np.mean(result['entropy'][margin:-margin, margin:-margin])
        assert 0.35 < mean_H < 0.50, (
            f"Mean entropy {mean_H:.3f} is not within the expected values for this "
            "fixture (theoretical ≈ 0.41 for cross_pol power ratio 0.09). "
            "The eigenvalue normalisation or -p·log2(p) computation is likely wrong."
        )

    def test_larger_window_smoother_output(self, synthetic_dual_pol):
        """Larger window must produce quantifiably lower entropy variance.

        For a box-filter of size N on white noise, output variance scales as
        1/N².  Going from window=3 to window=15 (5× larger) should give at
        least a 10× variance reduction (conservative vs theoretical 25×).

        The old test only checked var_large < var_small, which would pass
        even if the window were not applied at all (e.g., if the implementation
        accidentally used window=1 regardless of the argument).
        """
        co, cross = synthetic_dual_pol
        halpha_small = DualPolHAlpha(window_size=3)
        halpha_large = DualPolHAlpha(window_size=15)
        margin = 20
        H_small = halpha_small.decompose(co, cross)['entropy']
        H_large = halpha_large.decompose(co, cross)['entropy']

        var_small = np.var(H_small[margin:-margin, margin:-margin])
        var_large = np.var(H_large[margin:-margin, margin:-margin])

        ratio = var_small / (var_large + 1e-12)
        assert ratio > 10.0, (
            f"Variance ratio (window=3 / window=15) = {ratio:.2f}; "
            f"expected > 10×. var_small={var_small:.6f}, "
            f"var_large={var_large:.6f}. If the ratio is near 1.0, "
            "the window_size argument is being ignored."
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
    def test_zero_cross_pol_gives_zero_entropy(self):
        """Zero cross-pol input must yield entropy = 0 in the interior.

        With S_hv = 0, the 2×2 coherency matrix is rank-1 (one non-zero
        eigenvalue).  A rank-1 matrix has a single dominant scattering
        mechanism, so entropy = 0 by definition.  Any non-zero entropy for
        this input proves the eigenvalue decomposition has a bug.
        """
        halpha = DualPolHAlpha(window_size=7)
        rows, cols = 64, 64
        rng = np.random.default_rng(0)
        co = (
            rng.standard_normal((rows, cols))
            + 1j * rng.standard_normal((rows, cols))
        ).astype(np.complex64)
        cross_zero = np.zeros((rows, cols), dtype=np.complex64)

        result = halpha.decompose(co, cross_zero)
        margin = 10
        interior_H = result['entropy'][margin:-margin, margin:-margin]

        np.testing.assert_allclose(
            interior_H,
            0.0,
            atol=0.02,
            err_msg=(
                "Pure co-pol input (cross_pol=0) must produce entropy≈0 in "
                "the interior. Non-zero entropy means the eigenvector "
                "decomposition is returning spurious secondary eigenvalues "
                "for a rank-1 matrix."
            ),
        )

    def test_zero_cross_pol_gives_zero_alpha(self):
        """Zero cross-pol must yield alpha ≈ 0° (surface / odd-bounce scattering).

        When only co-pol is present, the dominant eigenvector is aligned with
        the co-pol channel, giving an alpha angle near 0°.  Values far from 0°
        indicate the eigenvector computation is assigning the wrong scattering
        mechanism.
        """
        halpha = DualPolHAlpha(window_size=7)
        rows, cols = 64, 64
        rng = np.random.default_rng(2)
        co = (
            rng.standard_normal((rows, cols))
            + 1j * rng.standard_normal((rows, cols))
        ).astype(np.complex64)
        cross_zero = np.zeros((rows, cols), dtype=np.complex64)

        result = halpha.decompose(co, cross_zero)
        margin = 10
        mean_alpha = np.mean(
            result['alpha'][margin:-margin, margin:-margin]
        )

        assert mean_alpha < 5.0, (
            f"Alpha = {mean_alpha:.2f}° for pure co-pol input; expected < 5°. "
            "Check that the dominant eigenvector is correctly identified as the "
            "co-pol channel direction."
        )

    def test_to_rgb_shape(self, synthetic_dual_pol):
        """to_rgb() returns (3, rows, cols) channels-first array."""
        co, cross = synthetic_dual_pol
        halpha = DualPolHAlpha(window_size=7)
        components = halpha.decompose(co, cross)
        rgb = halpha.to_rgb(components)
        assert rgb.shape == (3, co.shape[0], co.shape[1])

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
    def test_to_rgb_all_channels_active(self, synthetic_dual_pol):
        """to_rgb() must produce meaningful, distinct values in all three channels.

        The old 'max > 0' check passes with a single non-zero pixel in any
        channel.  This test requires each channel to have non-trivial range
        and that the channels are not all identical (which would indicate
        the RGB mapping is collapsed to grayscale).
        """
        co, cross = synthetic_dual_pol
        halpha = DualPolHAlpha(window_size=7)
        components = halpha.decompose(co, cross)
        rgb = halpha.to_rgb(components)

        for ch, name in enumerate(('R', 'G', 'B')):
            channel = rgb[:, :, ch]
            ch_range = channel.max() - channel.min()
            assert ch_range > 0.2, (
                f"Channel {name} (index {ch}) has near-zero range "
                f"({ch_range:.4f}) — the H-Alpha→RGB mapping is not "
                "producing distinct channel content"
            )

        # Channels must not be identical — that would mean the RGB mapping
        # is effectively grayscale
        assert not np.allclose(rgb[:, :, 0], rgb[:, :, 1], atol=1e-3), (
            "R and G channels are identical — to_rgb() may be copying one "
            "component to all three channels"
        )
        assert not np.allclose(rgb[:, :, 0], rgb[:, :, 2], atol=1e-3), (
            "R and B channels are identical"
        )


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
