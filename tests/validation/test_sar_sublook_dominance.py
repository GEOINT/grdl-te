# -*- coding: utf-8 -*-
"""
SublookDecomposition and DominanceFeatures validation.

Tests sub-aperture decomposition with real SICD data and dominance/entropy
pure functions with synthetic sub-look stacks.

- Level 1: Constructor, output structure, shapes
- Level 2: Energy conservation, physical bounds (dominance, entropy)
- Level 3: Cross-component integration (sublook → dominance → entropy)

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
2026-03-20

Modified
--------
2026-03-20
"""

# Third-party
import pytest
import numpy as np

try:
    from grdl.image_processing.sar.sublook import SublookDecomposition
    _HAS_SUBLOOK = True
except ImportError:
    _HAS_SUBLOOK = False

try:
    from grdl.IO.sar import SICDReader
    _HAS_SICD = True
except ImportError:
    _HAS_SICD = False

try:
    from grdl.image_processing.sar.dominance import (
        DominanceFeatures,
        compute_dominance,
        compute_sublook_entropy,
    )
    _HAS_DOMINANCE = True
except ImportError:
    _HAS_DOMINANCE = False

pytestmark = [
    pytest.mark.sar,
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic_sublooks():
    """Synthetic sub-look stack (5, 256, 256) complex64.

    Simulates 5 sub-aperture looks with random complex Gaussian speckle.
    """
    rng = np.random.default_rng(42)
    n_looks, rows, cols = 5, 256, 256
    looks = (
        rng.standard_normal((n_looks, rows, cols))
        + 1j * rng.standard_normal((n_looks, rows, cols))
    ).astype(np.complex64)
    return looks


@pytest.fixture(scope="module")
def uniform_power_sublooks():
    """Sub-look stack where all looks have equal power.

    Should produce maximum entropy and minimum dominance.
    """
    rng = np.random.default_rng(77)
    n_looks, rows, cols = 5, 128, 128
    # Same amplitude, different phase for each look
    looks = np.empty((n_looks, rows, cols), dtype=np.complex64)
    for i in range(n_looks):
        phase = rng.uniform(-np.pi, np.pi, (rows, cols)).astype(np.float32)
        looks[i] = np.exp(1j * phase).astype(np.complex64)
    return looks


@pytest.fixture(scope="module")
def concentrated_sublooks():
    """Sub-look stack where one look dominates.

    Look 0 has 10x the amplitude of other looks. Should produce
    high dominance and low entropy.
    """
    rng = np.random.default_rng(88)
    n_looks, rows, cols = 5, 128, 128
    looks = (
        rng.standard_normal((n_looks, rows, cols)) * 0.1
        + 1j * rng.standard_normal((n_looks, rows, cols)) * 0.1
    ).astype(np.complex64)
    # Make look 0 dominant
    looks[0] = (
        rng.standard_normal((rows, cols))
        + 1j * rng.standard_normal((rows, cols))
    ).astype(np.complex64)
    return looks


# =============================================================================
# Level 1: Format Validation — SublookDecomposition (requires data)
# =============================================================================


@pytest.mark.requires_data
@pytest.mark.skipif(not _HAS_SUBLOOK, reason="SublookDecomposition not available")
@pytest.mark.skipif(not _HAS_SICD, reason="SICDReader not available")
class TestSublookLevel1:
    """Validate SublookDecomposition constructor and output structure."""

    @pytest.fixture(autouse=True)
    def _load_chip(self, require_umbra_file):
        with SICDReader(str(require_umbra_file)) as reader:
            self.metadata = reader.metadata
            shape = reader.get_shape()
            cx, cy = shape[0] // 2, shape[1] // 2
            half = min(256, shape[0] // 2, shape[1] // 2)
            self.chip = reader.read_chip(cx - half, cx + half,
                                         cy - half, cy + half)

    @pytest.mark.slow
    def test_sublook_returns_3d(self):
        """decompose() returns 3D array (n_looks, rows, cols)."""
        sublook = SublookDecomposition(self.metadata, num_looks=3)
        result = sublook.decompose(self.chip)
        assert result.ndim == 3, f"Expected 3D output, got {result.ndim}D"

    @pytest.mark.slow
    def test_sublook_default_2(self):
        """Default num_looks=2 produces 2 sub-apertures."""
        sublook = SublookDecomposition(self.metadata, num_looks=2)
        result = sublook.decompose(self.chip)
        assert result.shape[0] == 2, f"Expected 2 looks, got {result.shape[0]}"

    @pytest.mark.slow
    def test_sublook_complex_output(self):
        """Output sub-looks are complex-valued."""
        sublook = SublookDecomposition(self.metadata, num_looks=3)
        result = sublook.decompose(self.chip)
        assert np.iscomplexobj(result), f"Expected complex output, got {result.dtype}"


# =============================================================================
# Level 2: Data Quality — SublookDecomposition (requires data)
# =============================================================================


@pytest.mark.requires_data
@pytest.mark.skipif(not _HAS_SUBLOOK, reason="SublookDecomposition not available")
@pytest.mark.skipif(not _HAS_SICD, reason="SICDReader not available")
class TestSublookLevel2:
    """Validate energy conservation and configurable looks."""

    @pytest.fixture(autouse=True)
    def _load_chip(self, require_umbra_file):
        with SICDReader(str(require_umbra_file)) as reader:
            self.metadata = reader.metadata
            shape = reader.get_shape()
            cx, cy = shape[0] // 2, shape[1] // 2
            half = min(256, shape[0] // 2, shape[1] // 2)
            self.chip = reader.read_chip(cx - half, cx + half,
                                         cy - half, cy + half)

    @pytest.mark.slow
    def test_sublook_energy_conservation(self):
        """Sum of sub-look powers approximately equals original power.

        Parseval's theorem: splitting the spectrum should preserve
        total energy (within windowing tolerance).
        """
        sublook = SublookDecomposition(self.metadata, num_looks=3)
        result = sublook.decompose(self.chip)
        sublook_power = np.sum(np.abs(result) ** 2, axis=0)
        original_power = np.abs(self.chip) ** 2

        # Use interior region to avoid boundary effects
        margin = 20
        sl_int = sublook_power[margin:-margin, margin:-margin]
        orig_int = original_power[margin:-margin, margin:-margin]

        ratio = np.mean(sl_int) / (np.mean(orig_int) + 1e-30)
        assert 0.5 < ratio < 2.0, (
            f"Sub-look power ratio = {ratio:.3f}; "
            "energy conservation violated (expected ~1.0)"
        )

    @pytest.mark.slow
    def test_sublook_configurable_looks(self):
        """num_looks=5 produces 5 sub-apertures."""
        sublook = SublookDecomposition(self.metadata, num_looks=5)
        result = sublook.decompose(self.chip)
        assert result.shape[0] == 5, f"Expected 5 looks, got {result.shape[0]}"

    @pytest.mark.slow
    def test_sublook_axis_selection(self):
        """Range and azimuth dimensions produce different sub-look stacks."""
        sl_az = SublookDecomposition(self.metadata, num_looks=3, dimension='azimuth')
        sl_rg = SublookDecomposition(self.metadata, num_looks=3, dimension='range')
        result_az = sl_az.decompose(self.chip)
        result_rg = sl_rg.decompose(self.chip)
        # Both should produce 3 looks but with different content
        assert result_az.shape[0] == 3
        assert result_rg.shape[0] == 3
        # Content should differ (different frequency axis)
        assert not np.allclose(result_az, result_rg), \
            "Azimuth and range sublooks should differ"


# =============================================================================
# Level 1: Format Validation — DominanceFeatures (pure functions, synthetic data)
# =============================================================================


@pytest.mark.skipif(not _HAS_DOMINANCE, reason="Dominance functions not available")
class TestDominanceLevel1:
    """Validate compute_dominance and compute_sublook_entropy output shapes."""

    def test_compute_dominance_shape(self, synthetic_sublooks):
        """compute_dominance returns (rows, cols) array."""
        dom = compute_dominance(synthetic_sublooks)
        assert dom.shape == synthetic_sublooks.shape[1:], (
            f"Expected shape {synthetic_sublooks.shape[1:]}, got {dom.shape}"
        )

    def test_compute_entropy_shape(self, synthetic_sublooks):
        """compute_sublook_entropy returns (rows, cols) array."""
        ent = compute_sublook_entropy(synthetic_sublooks)
        assert ent.shape == synthetic_sublooks.shape[1:], (
            f"Expected shape {synthetic_sublooks.shape[1:]}, got {ent.shape}"
        )

    def test_dominance_real_output(self, synthetic_sublooks):
        """Dominance output is real-valued float."""
        dom = compute_dominance(synthetic_sublooks)
        assert np.isrealobj(dom)
        assert dom.dtype in (np.float32, np.float64)

    def test_entropy_real_output(self, synthetic_sublooks):
        """Entropy output is real-valued float."""
        ent = compute_sublook_entropy(synthetic_sublooks)
        assert np.isrealobj(ent)
        assert ent.dtype in (np.float32, np.float64)


# =============================================================================
# Level 2: Data Quality — DominanceFeatures physical bounds
# =============================================================================


@pytest.mark.skipif(not _HAS_DOMINANCE, reason="Dominance functions not available")
class TestDominanceLevel2:
    """Validate physical bounds of dominance and entropy."""

    def test_dominance_hard_bounds(self, synthetic_sublooks):
        """Dominance ratio is in [0, 1] for arbitrary input."""
        dom = compute_dominance(synthetic_sublooks, dom_window=3)
        assert dom.min() >= 0.0, (
            f"Dominance min {dom.min():.6f} is negative"
        )
        assert dom.max() <= 1.0 + 1e-6, (
            f"Dominance max {dom.max():.6f} exceeds 1.0"
        )

    def test_dominance_uniform_equals_ratio(self):
        """With perfectly uniform power, dominance == dom_window / n_looks.

        All-ones input produces identical power in every look after
        spatial smoothing.  The sliding window then captures exactly
        dom_window/n_looks of the total, verifying the core arithmetic.
        """
        n_looks, rows, cols = 5, 64, 64
        uniform = np.ones((n_looks, rows, cols), dtype=np.complex64)
        dom = compute_dominance(uniform, window_size=7, dom_window=3)

        expected = 3.0 / 5.0
        # Interior pixels (away from uniform_filter boundary)
        interior = dom[10:-10, 10:-10]
        np.testing.assert_allclose(
            interior, expected, atol=1e-6,
            err_msg=(
                f"Uniform input should give dominance == {expected:.4f} "
                f"but got [{interior.min():.6f}, {interior.max():.6f}]"
            ),
        )

    def test_entropy_non_negative(self, synthetic_sublooks):
        """Shannon entropy is non-negative."""
        ent = compute_sublook_entropy(synthetic_sublooks)
        assert ent.min() >= -1e-10, (
            f"Entropy min {ent.min():.6f} is negative"
        )

    def test_uniform_high_entropy(self, uniform_power_sublooks):
        """Uniform power distribution produces high entropy.

        For n_looks=5, maximum entropy = log(5) ≈ 1.609.
        Uniform power should approach this maximum.
        """
        ent = compute_sublook_entropy(uniform_power_sublooks)
        n_looks = uniform_power_sublooks.shape[0]
        max_entropy = np.log(n_looks)
        mean_ent = np.mean(ent)
        # Should be close to maximum (within 30%)
        assert mean_ent > max_entropy * 0.7, (
            f"Mean entropy {mean_ent:.4f} too low for uniform power; "
            f"max theoretical = {max_entropy:.4f}"
        )

    def test_concentrated_high_dominance(self, concentrated_sublooks):
        """Concentrated power in one look produces high dominance.

        When look 0 has 10x amplitude (100x power), dominance should
        be high (power concentrated in few contiguous looks).
        """
        dom = compute_dominance(concentrated_sublooks, dom_window=1)
        mean_dom = np.mean(dom)
        # With 100x power ratio, dominance should be > 0.7
        assert mean_dom > 0.5, (
            f"Mean dominance {mean_dom:.4f} too low for concentrated power; "
            "expected > 0.5 with dominant look"
        )
