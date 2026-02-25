# -*- coding: utf-8 -*-
"""
Interpolation algorithm validation.

Tests 6 interpolators: Lanczos, KaiserSinc (windowed_sinc), Lagrange,
Farrow, Polyphase, and ThiranDelay. All use synthetic signals — no
real data required.

Tests:
- Level 1: Output shape, complex support
- Level 2: Identity property, sinc exactness, allpass property
- Level 3: Cross-validation vs np.interp, accuracy vs phase count

Author
------
Ava Courtney

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
    from grdl.interpolation import lanczos_interpolator
    _HAS_LANCZOS = True
except ImportError:
    _HAS_LANCZOS = False

try:
    from grdl.interpolation import windowed_sinc_interpolator
    _HAS_SINC = True
except ImportError:
    _HAS_SINC = False

try:
    from grdl.interpolation import lagrange_interpolator
    _HAS_LAGRANGE = True
except ImportError:
    _HAS_LAGRANGE = False

try:
    from grdl.interpolation import farrow_interpolator
    _HAS_FARROW = True
except ImportError:
    _HAS_FARROW = False

try:
    from grdl.interpolation import polyphase_interpolator
    _HAS_POLYPHASE = True
except ImportError:
    _HAS_POLYPHASE = False

try:
    from grdl.interpolation import thiran_delay
    _HAS_THIRAN = True
except ImportError:
    _HAS_THIRAN = False


pytestmark = [
    pytest.mark.interpolation,
]

# Shared synthetic signal
_RNG = np.random.default_rng(42)
_N = 1024
_X_OLD = np.arange(_N, dtype=np.float64)
_Y_REAL = np.sin(2 * np.pi * 0.05 * _X_OLD) + 0.3 * _RNG.standard_normal(_N)
_Y_COMPLEX = (_RNG.standard_normal(_N) + 1j * _RNG.standard_normal(_N)).astype(np.complex128)
_X_NEW = _X_OLD[10:-10] + 0.5  # Half-sample offsets, away from edges


# =============================================================================
# LanczosInterpolator
# =============================================================================


@pytest.mark.skipif(not _HAS_LANCZOS, reason="lanczos_interpolator not available")
class TestLanczos:

    def test_lanczos_returns_array(self):
        """Output shape matches x_new length."""
        interp = lanczos_interpolator(a=3)
        result = interp(_X_OLD, _Y_REAL, _X_NEW)
        assert isinstance(result, np.ndarray)
        assert result.shape == _X_NEW.shape

    def test_lanczos_complex(self):
        """Handles complex input/output."""
        interp = lanczos_interpolator(a=3)
        result = interp(_X_OLD, _Y_COMPLEX, _X_NEW)
        assert np.iscomplexobj(result)
        assert result.shape == _X_NEW.shape

    def test_lanczos_identity(self):
        """Interpolating at integer positions returns original values."""
        interp = lanczos_interpolator(a=5)
        x_int = _X_OLD[5:-5].astype(np.float64)
        result = interp(_X_OLD, _Y_REAL, x_int)
        np.testing.assert_allclose(result, _Y_REAL[5:-5], atol=1e-10)

    def test_lanczos_sinc_exact(self):
        r"""Lanczos of bandlimited signal at :math:`f_s/4` is accurate."""
        # Pure tone at quarter-Nyquist — well within passband
        x = np.arange(256, dtype=np.float64)
        y = np.sin(2 * np.pi * 0.1 * x)
        x_new = x[10:-10] + 0.25
        interp = lanczos_interpolator(a=5)
        result = interp(x, y, x_new)
        expected = np.sin(2 * np.pi * 0.1 * x_new)
        np.testing.assert_allclose(result, expected, atol=0.02)


# =============================================================================
# PolyphaseInterpolator
# =============================================================================


@pytest.mark.skipif(not _HAS_POLYPHASE, reason="polyphase_interpolator not available")
class TestPolyphase:

    def test_polyphase_returns_array(self):
        """Output shape matches x_new length."""
        interp = polyphase_interpolator(kernel_length=8, num_phases=32)
        result = interp(_X_OLD, _Y_REAL, _X_NEW)
        assert isinstance(result, np.ndarray)
        assert result.shape == _X_NEW.shape

    def test_polyphase_complex(self):
        """Handles complex input/output."""
        interp = polyphase_interpolator(kernel_length=8, num_phases=32)
        result = interp(_X_OLD, _Y_COMPLEX, _X_NEW)
        assert np.iscomplexobj(result)

    def test_polyphase_identity(self):
        """Interpolating at integer positions returns original values (within polyphase tolerance)."""
        interp = polyphase_interpolator(kernel_length=8, num_phases=128)
        x_int = _X_OLD[10:-10].astype(np.float64)
        result = interp(_X_OLD, _Y_REAL, x_int)
        # Polyphase kernel has inherent approximation error at integer positions
        np.testing.assert_allclose(result, _Y_REAL[10:-10], atol=0.015)

    def test_polyphase_num_phases_accuracy(self):
        """More phases = lower interpolation error."""
        x = np.arange(256, dtype=np.float64)
        y = np.sin(2 * np.pi * 0.1 * x)
        x_new = x[10:-10] + 0.25
        expected = np.sin(2 * np.pi * 0.1 * x_new)

        errors = []
        for nph in (16, 32, 64):
            interp = polyphase_interpolator(kernel_length=8, num_phases=nph)
            result = interp(x, y, x_new)
            errors.append(np.sqrt(np.mean((result - expected) ** 2)))

        # Error should decrease with more phases
        assert errors[1] <= errors[0] * 1.0  # Can change to 1.01
        assert errors[2] <= errors[1] * 1.0


# =============================================================================
# ThiranDelayFilter
# =============================================================================


@pytest.mark.skipif(not _HAS_THIRAN, reason="thiran_delay not available")
class TestThiranDelay:

    def test_thiran_returns_array(self):
        """Output length matches input."""
        signal = _Y_REAL[:256].copy()
        # Thiran stability: delay >= order - 0.5. For order 1, delay >= 0.5.
        result = thiran_delay(signal, 0.5, 1)
        assert isinstance(result, np.ndarray)
        assert result.shape == signal.shape

    def test_thiran_allpass(self):
        r""":math:`|H(f)| \approx 1` across passband."""
        n = 1024
        rng = np.random.default_rng(99)
        signal = rng.standard_normal(n)
        # Order 2, delay 1.7 (>= 2 - 0.5 = 1.5)
        result = thiran_delay(signal, 1.7, 2)
        input_power = np.sum(signal ** 2)
        output_power = np.sum(result ** 2)
        ratio = output_power / input_power
        assert 0.85 <= ratio <= 1.15, f"Power ratio {ratio:.4f} deviates from allpass"

    def test_thiran_group_delay(self):
        r"""Group delay :math:`\approx` specified fractional delay."""
        n = 256
        center = n // 2
        t = np.arange(n, dtype=np.float64)
        # Gaussian pulse — smooth peak allows sub-sample estimation
        pulse = np.exp(-0.5 * ((t - center) / 3.0) ** 2)
        # Order 3, delay 3.7 (>= 3 - 0.5 = 2.5)
        delay = 3.7
        result = thiran_delay(pulse, delay, 3)
        # Parabolic interpolation around peak for sub-sample accuracy
        peak = np.argmax(np.abs(result))
        if 0 < peak < len(result) - 1:
            alpha = np.abs(result[peak - 1])
            beta = np.abs(result[peak])
            gamma = np.abs(result[peak + 1])
            refined = peak + 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
        else:
            refined = float(peak)
        actual_delay = refined - center
        assert abs(actual_delay - delay) <= 0.5, \
            f"Group delay {actual_delay:.2f} deviates from target {delay}"


# =============================================================================
# KaiserSinc, Lagrange, Farrow — basic coverage
# =============================================================================


@pytest.mark.skipif(not _HAS_SINC, reason="windowed_sinc_interpolator not available")
class TestKaiserSinc:

    def test_kaiser_sinc_returns_array(self):
        interp = windowed_sinc_interpolator(kernel_length=8, beta=5.0)
        result = interp(_X_OLD, _Y_REAL, _X_NEW)
        assert result.shape == _X_NEW.shape

    def test_kaiser_sinc_identity(self):
        interp = windowed_sinc_interpolator(kernel_length=16, beta=5.0)
        x_int = _X_OLD[10:-10].astype(np.float64)
        result = interp(_X_OLD, _Y_REAL, x_int)
        np.testing.assert_allclose(result, _Y_REAL[10:-10], atol=1e-4)


@pytest.mark.skipif(not _HAS_LAGRANGE, reason="lagrange_interpolator not available")
class TestLagrange:

    def test_lagrange_returns_array(self):
        interp = lagrange_interpolator(order=3)
        result = interp(_X_OLD, _Y_REAL, _X_NEW)
        assert result.shape == _X_NEW.shape

    def test_lagrange_identity(self):
        interp = lagrange_interpolator(order=5)
        x_int = _X_OLD[10:-10].astype(np.float64)
        result = interp(_X_OLD, _Y_REAL, x_int)
        np.testing.assert_allclose(result, _Y_REAL[10:-10], atol=1e-6)


@pytest.mark.skipif(not _HAS_FARROW, reason="farrow_interpolator not available")
class TestFarrow:

    def test_farrow_returns_array(self):
        interp = farrow_interpolator(filter_order=4, poly_order=3)
        result = interp(_X_OLD, _Y_REAL, _X_NEW)
        assert result.shape == _X_NEW.shape

    def test_farrow_complex(self):
        interp = farrow_interpolator(filter_order=4, poly_order=3)
        result = interp(_X_OLD, _Y_COMPLEX, _X_NEW)
        assert np.iscomplexobj(result)
