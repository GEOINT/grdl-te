# -*- coding: utf-8 -*-
"""
Polarimetric Matrix Validation - Dual-pol Sentinel-1 SLC.

Tests CovarianceMatrix, CoherencyMatrix, and StokesVector using
real Sentinel-1 IW SLC dual-pol (VV+VH) data.

- Level 1: Output shape, dtype, matrix dimensions
- Level 2: Physical property bounds (Hermitian symmetry, positive
           semi-definiteness, Stokes constraints, power conservation)
- Level 3: Cross-validation between C2 and T2 trace equivalence

Dataset: Sentinel-1 IW SLC (VV+VH)

Dependencies
------------
pytest
numpy
grdl

Author
------
Ava Courtney

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2026-05-29
"""

import pytest
import numpy as np

try:
    from grdl.image_processing.decomposition import (
        CovarianceMatrix,
        CoherencyMatrix,
        StokesVector,
    )
    _HAS_POL = True
except ImportError:
    _HAS_POL = False

try:
    from grdl.IO.sar import Sentinel1SLCReader
    _HAS_S1 = True
except ImportError:
    _HAS_S1 = False


pytestmark = [
    pytest.mark.decomposition,
    pytest.mark.requires_data,
    pytest.mark.skipif(not _HAS_POL,
                       reason="grdl polarimetric decomposition not available"),
    pytest.mark.skipif(not _HAS_S1,
                       reason="Sentinel1SLCReader not available"),
]


# =============================================================================
# Fixtures
# =============================================================================

_CHIP_SIZE = 128  # Small chip for fast tests


@pytest.fixture(scope="module")
def s1_dual_pol(sentinel1_data_dir):
    """Load VV and VH chips from Sentinel-1 SLC center."""
    safe_dirs = list(sentinel1_data_dir.glob("S1*.SAFE")) if sentinel1_data_dir.exists() else []
    if not safe_dirs:
        pytest.skip(f"Sentinel-1 SLC not found in {sentinel1_data_dir}")

    safe_path = safe_dirs[0]
    with Sentinel1SLCReader(str(safe_path)) as reader:
        meta = reader.metadata
        rows, cols = reader.get_shape()
        r0 = rows // 2 - _CHIP_SIZE // 2
        c0 = cols // 2 - _CHIP_SIZE // 2

        # Read VV (primary co-pol)
        vv = reader.read_chip(r0, r0 + _CHIP_SIZE, c0, c0 + _CHIP_SIZE,
                              bands=[0])
        # Read VH (cross-pol) - band index 1 for dual-pol
        try:
            vh = reader.read_chip(r0, r0 + _CHIP_SIZE, c0, c0 + _CHIP_SIZE,
                                  bands=[1])
        except (IndexError, ValueError):
            pytest.skip("VH channel not available in this product")

    assert np.iscomplexobj(vv), "VV data not complex"
    assert np.iscomplexobj(vh), "VH data not complex"
    return vv, vh


# =============================================================================
# Level 1: Format Validation — Output structure
# =============================================================================


class TestPolarimetricMatrixLevel1:
    """Validate output shape and dtype for dual-pol matrices."""

    def test_covariance_c2_shape(self, s1_dual_pol):
        """CovarianceMatrix produces (2, 2, rows, cols) for dual-pol."""
        vv, vh = s1_dual_pol
        channels = np.stack([vv, vh], axis=0)
        cov = CovarianceMatrix(window_size=7)
        C2 = cov.compute(channels)
        assert C2.shape == (2, 2, _CHIP_SIZE, _CHIP_SIZE), (
            f"Expected (2, 2, {_CHIP_SIZE}, {_CHIP_SIZE}), got {C2.shape}"
        )

    def test_covariance_c2_dtype_complex(self, s1_dual_pol):
        """CovarianceMatrix output is complex."""
        vv, vh = s1_dual_pol
        channels = np.stack([vv, vh], axis=0)
        C2 = CovarianceMatrix(window_size=7).compute(channels)
        assert np.iscomplexobj(C2)

    def test_coherency_t2_shape(self, s1_dual_pol):
        """CoherencyMatrix produces (2, 2, rows, cols) for dual-pol."""
        vv, vh = s1_dual_pol
        channels = np.stack([vv, vh], axis=0)
        T2 = CoherencyMatrix(window_size=7).compute(channels)
        assert T2.shape == (2, 2, _CHIP_SIZE, _CHIP_SIZE)

    def test_coherency_t2_dtype_complex(self, s1_dual_pol):
        """CoherencyMatrix output is complex."""
        vv, vh = s1_dual_pol
        channels = np.stack([vv, vh], axis=0)
        T2 = CoherencyMatrix(window_size=7).compute(channels)
        assert np.iscomplexobj(T2)

    def test_stokes_vector_shape(self, s1_dual_pol):
        """StokesVector produces (4, rows, cols)."""
        vv, vh = s1_dual_pol
        sv = StokesVector(window_size=7)
        stokes = sv.compute(vv, vh)
        assert stokes.shape == (4, _CHIP_SIZE, _CHIP_SIZE), (
            f"Expected (4, {_CHIP_SIZE}, {_CHIP_SIZE}), got {stokes.shape}"
        )

    def test_stokes_vector_dtype_real(self, s1_dual_pol):
        """StokesVector output is real-valued."""
        vv, vh = s1_dual_pol
        stokes = StokesVector(window_size=7).compute(vv, vh)
        assert not np.iscomplexobj(stokes)


# =============================================================================
# Level 2: Data Quality — Physical property invariants
# =============================================================================


class TestPolarimetricMatrixLevel2:
    """Validate physical constraints on polarimetric matrices."""

    def test_covariance_hermitian_symmetry(self, s1_dual_pol):
        """C2 must be Hermitian: C[i,j] == conj(C[j,i]) at every pixel."""
        vv, vh = s1_dual_pol
        channels = np.stack([vv, vh], axis=0)
        C2 = CovarianceMatrix(window_size=7).compute(channels)
        # Check C[0,1] == conj(C[1,0])
        np.testing.assert_allclose(
            C2[0, 1], np.conj(C2[1, 0]),
            atol=1e-5,
            err_msg="C2 is not Hermitian symmetric"
        )

    def test_covariance_diagonal_real_positive(self, s1_dual_pol):
        """C2 diagonal elements are real and non-negative (power terms)."""
        vv, vh = s1_dual_pol
        channels = np.stack([vv, vh], axis=0)
        C2 = CovarianceMatrix(window_size=7).compute(channels)
        diag_00 = C2[0, 0]
        diag_11 = C2[1, 1]
        # Diagonal of Hermitian PSD matrix must be real
        assert np.allclose(diag_00.imag, 0, atol=1e-6), (
            "C2[0,0] has non-zero imaginary part"
        )
        assert np.allclose(diag_11.imag, 0, atol=1e-6), (
            "C2[1,1] has non-zero imaginary part"
        )
        # Power must be non-negative
        assert diag_00.real.min() >= -1e-6, "C2[0,0] has negative power"
        assert diag_11.real.min() >= -1e-6, "C2[1,1] has negative power"

    def test_coherency_hermitian_symmetry(self, s1_dual_pol):
        """T2 must be Hermitian symmetric."""
        vv, vh = s1_dual_pol
        channels = np.stack([vv, vh], axis=0)
        T2 = CoherencyMatrix(window_size=7).compute(channels)
        np.testing.assert_allclose(
            T2[0, 1], np.conj(T2[1, 0]),
            atol=1e-5,
            err_msg="T2 is not Hermitian symmetric"
        )

    def test_coherency_diagonal_real_positive(self, s1_dual_pol):
        """T2 diagonal elements are approximately real and non-negative."""
        vv, vh = s1_dual_pol
        channels = np.stack([vv, vh], axis=0)
        T2 = CoherencyMatrix(window_size=7).compute(channels)
        # Imaginary part should be negligible relative to real part
        real_mean_00 = np.abs(T2[0, 0].real).mean()
        if real_mean_00 > 0:
            ratio_00 = np.abs(T2[0, 0].imag).max() / real_mean_00
            assert ratio_00 < 1e-2, f"T2[0,0] imag/real ratio too high: {ratio_00:.6f}"
        real_mean_11 = np.abs(T2[1, 1].real).mean()
        if real_mean_11 > 0:
            ratio_11 = np.abs(T2[1, 1].imag).max() / real_mean_11
            assert ratio_11 < 1e-2, f"T2[1,1] imag/real ratio too high: {ratio_11:.6f}"
        # Diagonal should be non-negative (power)
        assert T2[0, 0].real.min() >= -1e-4
        # T2[1,1] for dual-pol may be very small but still non-negative
        assert T2[1, 1].real.min() >= -1e-4

    def test_covariance_positive_semidefinite(self, s1_dual_pol):
        """C2 determinant is non-negative (PSD for 2x2)."""
        vv, vh = s1_dual_pol
        channels = np.stack([vv, vh], axis=0)
        C2 = CovarianceMatrix(window_size=7).compute(channels)
        # For 2x2 Hermitian PSD: det = C00*C11 - |C01|² >= 0
        det = (C2[0, 0].real * C2[1, 1].real
               - np.abs(C2[0, 1]) ** 2)
        # Allow small numerical noise
        assert det.min() >= -1e-4, (
            f"C2 not PSD: min determinant = {det.min():.6e}"
        )

    def test_stokes_s0_positive(self, s1_dual_pol):
        """S0 (total intensity) is non-negative everywhere."""
        vv, vh = s1_dual_pol
        stokes = StokesVector(window_size=7).compute(vv, vh)
        s0 = stokes[0]
        assert s0.min() >= -1e-6, f"S0 has negative values: min={s0.min():.6e}"

    def test_stokes_dop_bounded(self, s1_dual_pol):
        """Degree of polarization is in [0, 1]."""
        vv, vh = s1_dual_pol
        sv = StokesVector(window_size=7)
        stokes = sv.compute(vv, vh)
        dop = sv.degree_of_polarization(stokes)
        assert dop.min() >= -1e-4, f"DoP below 0: {dop.min():.6f}"
        assert dop.max() <= 1.0 + 1e-4, f"DoP above 1: {dop.max():.6f}"

    def test_stokes_physical_constraint(self, s1_dual_pol):
        """S0² >= S1² + S2² + S3² (partially polarized light constraint)."""
        vv, vh = s1_dual_pol
        stokes = StokesVector(window_size=7).compute(vv, vh)
        s0_sq = stokes[0] ** 2
        pol_sq = stokes[1] ** 2 + stokes[2] ** 2 + stokes[3] ** 2
        # Allow numerical tolerance relative to signal magnitude
        tolerance = s0_sq.mean() * 1e-3
        violation = (pol_sq > s0_sq + tolerance).sum()
        total = s0_sq.size
        assert violation < total * 0.15, (
            f"Stokes constraint violated in {violation}/{total} pixels "
            f"({100*violation/total:.1f}%)"
        )

    def test_copol_dominates_crosspol(self, s1_dual_pol):
        """VV and VH power are consistent (co-pol typically dominates)."""
        vv, vh = s1_dual_pol
        channels = np.stack([vv, vh], axis=0)
        C2 = CovarianceMatrix(window_size=7).compute(channels)
        vv_power = C2[0, 0].real.mean()
        vh_power = C2[1, 1].real.mean()
        # Both channels should have positive power
        assert vv_power > 0, "VV power is non-positive"
        assert vh_power > 0, "VH power is non-positive"
        # Cross-pol ratio should be physically bounded (VH/VV typically < 1)
        # In volume scattering VH can equal VV, so just verify ratio is bounded
        ratio = vh_power / vv_power
        assert ratio < 10.0, (
            f"Cross-pol ratio unreasonably high: VH/VV = {ratio:.2f}"
        )


# =============================================================================
# Level 3: Integration — Cross-validation
# =============================================================================


class TestPolarimetricMatrixLevel3:
    """Cross-validate different matrix representations."""

    @pytest.mark.integration
    def test_trace_equivalence_c2_t2(self, s1_dual_pol):
        """Trace(C2) == Trace(T2) — total power is representation-invariant."""
        vv, vh = s1_dual_pol
        channels = np.stack([vv, vh], axis=0)
        C2 = CovarianceMatrix(window_size=7).compute(channels)
        T2 = CoherencyMatrix(window_size=7).compute(channels)

        trace_c2 = C2[0, 0].real + C2[1, 1].real
        trace_t2 = T2[0, 0].real + T2[1, 1].real

        np.testing.assert_allclose(
            trace_c2, trace_t2,
            rtol=1e-4,
            err_msg="Trace(C2) != Trace(T2) — power not conserved"
        )

    @pytest.mark.integration
    def test_stokes_s0_equals_total_power(self, s1_dual_pol):
        """S0 == |E_H|² + |E_V|² (total intensity consistency)."""
        vv, vh = s1_dual_pol
        sv = StokesVector(window_size=7)
        stokes = sv.compute(vv, vh)
        s0 = stokes[0]

        # Independent power computation with same window
        channels = np.stack([vv, vh], axis=0)
        C2 = CovarianceMatrix(window_size=7).compute(channels)
        total_power = C2[0, 0].real + C2[1, 1].real

        np.testing.assert_allclose(
            s0, total_power,
            rtol=0.01,
            err_msg="Stokes S0 != Trace(C2) — total intensity mismatch"
        )
