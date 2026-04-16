# -*- coding: utf-8 -*-
"""
PauliDecomposition Tests - Synthetic quad-pol validation.

Tests Pauli basis scattering matrix decomposition using synthetic
complex SAR data.

- Level 1: decompose() returns dict with correct keys, shapes, and dtype
- Level 2: Physical formula validation (surface, double_bounce, volume)
- Level 3: to_rgb() produces valid RGB composite, pipeline integration

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
2026-03-20

Modified
--------
2026-03-20
"""

# Third-party
import pytest
import numpy as np

try:
    from grdl.image_processing.decomposition.pauli import PauliDecomposition
    _HAS_PAULI = True
except ImportError:
    _HAS_PAULI = False

pytestmark = [
    pytest.mark.decomposition,
    pytest.mark.skipif(not _HAS_PAULI,
                       reason="grdl PauliDecomposition not available"),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic_quad_pol():
    """Synthetic quad-pol complex SAR channels with monostatic reciprocity.

    Four channels representing the full 2x2 scattering matrix:
    S_HH, S_HV, S_VH, S_VV. Cross-pol channels enforce monostatic
    reciprocity (S_HV == S_VH) at 30% relative power, which is
    the physically correct assumption for monostatic SAR and enables
    exact Pauli span conservation testing.
    """
    rng = np.random.default_rng(42)
    rows, cols = 256, 256
    shh = (
        rng.standard_normal((rows, cols))
        + 1j * rng.standard_normal((rows, cols))
    ).astype(np.complex64)
    svv = (
        rng.standard_normal((rows, cols))
        + 1j * rng.standard_normal((rows, cols))
    ).astype(np.complex64)
    shv = (
        rng.standard_normal((rows, cols)) * 0.3
        + 1j * rng.standard_normal((rows, cols)) * 0.3
    ).astype(np.complex64)
    svh = shv.copy()  # monostatic reciprocity
    return shh, shv, svh, svv


@pytest.fixture(scope="module")
def reciprocal_quad_pol():
    """Quad-pol with monostatic reciprocity: S_HV == S_VH."""
    rng = np.random.default_rng(99)
    rows, cols = 128, 128
    shh = (
        rng.standard_normal((rows, cols))
        + 1j * rng.standard_normal((rows, cols))
    ).astype(np.complex64)
    svv = (
        rng.standard_normal((rows, cols))
        + 1j * rng.standard_normal((rows, cols))
    ).astype(np.complex64)
    shv = (
        rng.standard_normal((rows, cols)) * 0.3
        + 1j * rng.standard_normal((rows, cols)) * 0.3
    ).astype(np.complex64)
    svh = shv.copy()  # reciprocity
    return shh, shv, svh, svv


# ---------------------------------------------------------------------------
# Level 1: Format Validation
# ---------------------------------------------------------------------------
class TestPauliLevel1:
    """Validate decompose() output structure."""

    def test_pauli_output_keys(self, synthetic_quad_pol):
        """decompose() returns dict with surface, double_bounce, volume."""
        shh, shv, svh, svv = synthetic_quad_pol
        pauli = PauliDecomposition()
        result = pauli.decompose(shh, shv, svh, svv)
        assert isinstance(result, dict)
        for key in ('surface', 'double_bounce', 'volume'):
            assert key in result, f"Missing key: {key}"

    def test_pauli_output_shapes(self, synthetic_quad_pol):
        """All output arrays match input spatial shape."""
        shh, shv, svh, svv = synthetic_quad_pol
        pauli = PauliDecomposition()
        result = pauli.decompose(shh, shv, svh, svv)
        for key in ('surface', 'double_bounce', 'volume'):
            assert result[key].shape == shh.shape, (
                f"{key} shape {result[key].shape} != input shape {shh.shape}"
            )

    def test_pauli_output_complex(self, synthetic_quad_pol):
        """Output components are complex-valued."""
        shh, shv, svh, svv = synthetic_quad_pol
        pauli = PauliDecomposition()
        result = pauli.decompose(shh, shv, svh, svv)
        for key in ('surface', 'double_bounce', 'volume'):
            assert np.iscomplexobj(result[key]), (
                f"{key} is not complex: dtype={result[key].dtype}"
            )


# ---------------------------------------------------------------------------
# Level 2: Data Quality — Physical formula validation
# ---------------------------------------------------------------------------
class TestPauliLevel2:
    """Validate Pauli formulas and physical properties."""

    def test_pauli_surface_formula(self, synthetic_quad_pol):
        """surface == (S_HH + S_VV) / sqrt(2)."""
        shh, shv, svh, svv = synthetic_quad_pol
        pauli = PauliDecomposition()
        result = pauli.decompose(shh, shv, svh, svv)

        norm = np.float32(1.0 / np.sqrt(2.0))
        expected = (shh + svv) * norm
        np.testing.assert_allclose(
            result['surface'], expected, rtol=1e-5,
            err_msg="Surface component does not match (S_HH + S_VV) / sqrt(2)"
        )

    def test_pauli_double_bounce_formula(self, synthetic_quad_pol):
        """double_bounce == (S_HH - S_VV) / sqrt(2)."""
        shh, shv, svh, svv = synthetic_quad_pol
        pauli = PauliDecomposition()
        result = pauli.decompose(shh, shv, svh, svv)

        norm = np.float32(1.0 / np.sqrt(2.0))
        expected = (shh - svv) * norm
        np.testing.assert_allclose(
            result['double_bounce'], expected, rtol=1e-5,
            err_msg="Double bounce does not match (S_HH - S_VV) / sqrt(2)"
        )

    def test_pauli_volume_formula(self, synthetic_quad_pol):
        """volume == (S_HV + S_VH) / sqrt(2)."""
        shh, shv, svh, svv = synthetic_quad_pol
        pauli = PauliDecomposition()
        result = pauli.decompose(shh, shv, svh, svv)

        norm = np.float32(1.0 / np.sqrt(2.0))
        expected = (shv + svh) * norm
        np.testing.assert_allclose(
            result['volume'], expected, rtol=1e-5,
            err_msg="Volume component does not match (S_HV + S_VH) / sqrt(2)"
        )

    def test_pauli_span_conservation(self, synthetic_quad_pol):
        """Sum of Pauli component powers equals scattering matrix span.

        Under Pauli basis unitarity:
            |surface|^2 + |double_bounce|^2 + |volume|^2 = Span
        where Span = |S_HH|^2 + |S_VV|^2 + |S_HV|^2 + |S_VH|^2

        This verifies total power conservation, not just a trivial
        non-negativity check.
        """
        shh, shv, svh, svv = synthetic_quad_pol
        pauli = PauliDecomposition()
        result = pauli.decompose(shh, shv, svh, svv)

        pauli_power = (
            np.abs(result['surface']) ** 2
            + np.abs(result['double_bounce']) ** 2
            + np.abs(result['volume']) ** 2
        )
        span = (
            np.abs(shh) ** 2 + np.abs(svv) ** 2
            + np.abs(shv) ** 2 + np.abs(svh) ** 2
        )

        np.testing.assert_allclose(
            np.mean(pauli_power), np.mean(span), rtol=1e-4,
            err_msg="Pauli component power does not conserve scattering matrix span"
        )

    def test_pauli_reciprocity(self, reciprocal_quad_pol):
        """Under reciprocity (S_HV == S_VH), volume = sqrt(2) * S_HV.

        This verifies the mathematical simplification when the
        monostatic reciprocity condition holds.
        """
        shh, shv, svh, svv = reciprocal_quad_pol
        pauli = PauliDecomposition()
        result = pauli.decompose(shh, shv, svh, svv)

        # volume = (S_HV + S_VH) / sqrt(2) = 2*S_HV / sqrt(2) = sqrt(2)*S_HV
        expected = shv * np.float32(np.sqrt(2.0))
        np.testing.assert_allclose(
            result['volume'], expected, rtol=1e-5,
            err_msg="Under reciprocity, volume should equal sqrt(2) * S_HV"
        )


# ---------------------------------------------------------------------------
# Level 3: Integration — RGB composite and pipeline
# ---------------------------------------------------------------------------
class TestPauliLevel3:
    """Validate to_rgb() and pipeline integration."""

    @pytest.mark.integration
    def test_pauli_to_rgb(self, synthetic_quad_pol):
        """to_rgb() returns (3, rows, cols) float32 array in [0, 1]."""
        shh, shv, svh, svv = synthetic_quad_pol
        pauli = PauliDecomposition()
        components = pauli.decompose(shh, shv, svh, svv)
        rgb, _ = pauli.to_rgb(components)

        assert rgb.ndim == 3, f"RGB should be 3D, got {rgb.ndim}D"
        assert rgb.shape[0] == 3, f"First dim should be 3 (RGB), got {rgb.shape[0]}"
        assert rgb.shape[1] == shh.shape[0]
        assert rgb.shape[2] == shh.shape[1]
        assert rgb.dtype == np.float32, f"Expected float32, got {rgb.dtype}"
        assert rgb.min() >= 0.0, f"RGB min = {rgb.min()} < 0"
        assert rgb.max() <= 1.0, f"RGB max = {rgb.max()} > 1"
        # Verify non-trivial content (not all zeros or all ones)
        assert rgb.std() > 0.01, "RGB has near-zero variance — likely degenerate"

    @pytest.mark.integration
    def test_pauli_pipeline_db_rgb(self, synthetic_quad_pol):
        """Decompose with dB representation produces valid RGB."""
        shh, shv, svh, svv = synthetic_quad_pol
        pauli = PauliDecomposition()
        components = pauli.decompose(shh, shv, svh, svv)

        # Test dB representation (default)
        rgb_db, _ = pauli.to_rgb(components, representation='db')
        assert rgb_db.shape[0] == 3
        assert np.isfinite(rgb_db).all(), "dB RGB contains non-finite values"

        # Test magnitude representation
        rgb_mag, _ = pauli.to_rgb(components, representation='magnitude')
        assert rgb_mag.shape[0] == 3
        assert np.isfinite(rgb_mag).all(), "Magnitude RGB contains non-finite values"
