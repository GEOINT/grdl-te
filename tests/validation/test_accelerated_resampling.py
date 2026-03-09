# -*- coding: utf-8 -*-
"""
Accelerated Resampling Tests - Multi-backend resample dispatch.

Validates grdl.image_processing.ortho.accelerated:
- Level 1: detect_backend returns a valid backend name; explicit prefer
           values are passed through; auto-detect fallback chain works
           when optional packages are absent (monkeypatched)
- Level 2: resample() produces correct output for scipy backend (identity
           mapping, nodata fill, dtype preservation, multiband, complex)
- Level 3: All available backends agree within 1% RMSE on the same input;
           torch order>1 falls back gracefully; scipy_parallel chunk
           assembles the same result as sequential scipy

All tests use synthetic data only (no real imagery required).

Dependencies
------------
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
2026-03-09

Modified
--------
2026-03-09
"""

# Standard library
from typing import List

# Third-party
import pytest
import numpy as np

# GRDL internal
_NUMBA_AVAILABLE = False
_TORCH_AVAILABLE = False

try:
    import grdl.image_processing.ortho.accelerated as _accel
    from grdl.image_processing.ortho.accelerated import (
        detect_backend,
        resample,
        _NUMBA_AVAILABLE,
        _TORCH_AVAILABLE,
    )
    _HAS_ACCEL = True
except ImportError:
    _HAS_ACCEL = False

pytestmark = [
    pytest.mark.ortho,
    pytest.mark.skipif(not _HAS_ACCEL, reason="grdl.image_processing.ortho.accelerated not available"),
]

# ---------------------------------------------------------------------------
# Known-valid backend names
# ---------------------------------------------------------------------------
_VALID_BACKENDS = frozenset({
    'numba', 'torch_gpu', 'torch', 'scipy_parallel', 'scipy',
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity_maps(h: int, w: int):
    """Row/col maps and valid mask for a trivial identity resampling."""
    out_rows = np.arange(h, dtype=np.float64)
    out_cols = np.arange(w, dtype=np.float64)
    col_map, row_map = np.meshgrid(out_cols, out_rows)
    valid = np.ones((h, w), dtype=bool)
    return row_map, col_map, valid


def _available_backends() -> List[str]:
    """Return backends that can actually run (no ImportError) on this machine."""
    backends = ['scipy']
    if _TORCH_AVAILABLE:
        backends.append('torch')
    if _NUMBA_AVAILABLE:
        backends.append('numba')
    return backends


# ===========================================================================
# Level 1: detect_backend
# ===========================================================================

class TestDetectBackend:
    """Validate backend detection logic."""

    def test_auto_returns_known_backend(self):
        """detect_backend('auto') returns a string from the known-valid set."""
        result = detect_backend('auto')
        assert isinstance(result, str), "detect_backend must return a string"
        assert result in _VALID_BACKENDS, (
            f"detect_backend('auto') returned '{result}', "
            f"which is not in {sorted(_VALID_BACKENDS)}"
        )

    def test_explicit_scipy_passes_through(self):
        """detect_backend(prefer='scipy') always returns 'scipy'."""
        assert detect_backend(prefer='scipy') == 'scipy', (
            "Explicit prefer='scipy' should return 'scipy' regardless of "
            "what is installed"
        )

    def test_explicit_scipy_parallel_passes_through(self):
        """detect_backend(prefer='scipy_parallel') always returns 'scipy_parallel'."""
        assert detect_backend(prefer='scipy_parallel') == 'scipy_parallel', (
            "Explicit prefer='scipy_parallel' should pass through unchanged"
        )

    def test_explicit_numba_passes_through(self):
        """detect_backend(prefer='numba') returns 'numba' (user responsibility)."""
        result = detect_backend(prefer='numba')
        assert result == 'numba', (
            "Explicit prefer value should be returned as-is without validation"
        )

    def test_fallback_to_scipy_parallel_when_no_optional_deps(self, monkeypatch):
        """When numba and torch are both unavailable, auto → 'scipy_parallel'.

        Monkeypatches the module-level availability flags to simulate a
        bare scipy-only environment.  This verifies the fallback chain
        is active without requiring packages to be actually uninstalled.
        """
        monkeypatch.setattr(_accel, '_NUMBA_AVAILABLE', False)
        monkeypatch.setattr(_accel, '_TORCH_AVAILABLE', False)
        result = detect_backend('auto')
        assert result == 'scipy_parallel', (
            f"With no optional backends, detect_backend('auto') should "
            f"return 'scipy_parallel', got '{result}'"
        )

    def test_fallback_prefers_numba_over_torch(self, monkeypatch):
        """When both numba and torch are present, auto selects numba first."""
        monkeypatch.setattr(_accel, '_NUMBA_AVAILABLE', True)
        monkeypatch.setattr(_accel, '_TORCH_AVAILABLE', True)
        result = detect_backend('auto')
        assert result == 'numba', (
            f"With both numba and torch available, expected 'numba' (faster), "
            f"got '{result}'"
        )

    def test_fallback_to_torch_when_no_numba(self, monkeypatch):
        """When only torch is available (no numba), auto selects torch."""
        monkeypatch.setattr(_accel, '_NUMBA_AVAILABLE', False)
        monkeypatch.setattr(_accel, '_TORCH_AVAILABLE', True)
        result = detect_backend('auto')
        assert result in ('torch', 'torch_gpu'), (
            f"With torch available but no numba, expected 'torch' or 'torch_gpu', "
            f"got '{result}'"
        )


# ===========================================================================
# Level 2: resample() correctness — scipy backend
# ===========================================================================

class TestResampleScipy:
    """Validate resample() output with the scipy backend (always available)."""

    def test_identity_mapping_recovers_input(self):
        """Identity mapping produces output identical to input (within float32 precision).

        If row_map[i,j] = i and col_map[i,j] = j, bilinear interpolation at
        integer coordinates should exactly recover the source pixel value.
        """
        rng = np.random.default_rng(7)
        image = rng.random((32, 32), dtype=np.float64)
        row_map, col_map, valid = _identity_maps(32, 32)

        out = resample(image, row_map, col_map, valid,
                       order=1, nodata=0.0, backend='scipy')

        # At integer coordinates bilinear == nearest so values must be exact.
        # Restrict comparison to the inner region to avoid boundary effects.
        np.testing.assert_allclose(
            out[1:-1, 1:-1], image[1:-1, 1:-1], atol=1e-10,
            err_msg="Identity-map resample should recover source pixel values exactly",
        )

    def test_invalid_mask_fills_nodata(self):
        """Pixels with valid_mask=False are set to nodata value.

        Creates a mapping where every pixel is marked invalid and verifies
        that the output is entirely filled with the nodata sentinel.
        """
        image = np.ones((16, 16), dtype=np.float32)
        row_map = np.zeros((8, 8), dtype=np.float64)
        col_map = np.zeros((8, 8), dtype=np.float64)
        valid = np.zeros((8, 8), dtype=bool)  # all invalid

        out = resample(image, row_map, col_map, valid,
                       order=1, nodata=-9999.0, backend='scipy')

        assert np.all(out == -9999.0), (
            "All-invalid mask should produce an output filled entirely with nodata"
        )

    def test_output_shape_matches_maps(self):
        """Output shape equals the shape of row_map/col_map, not the source."""
        image = np.ones((100, 100), dtype=np.float32)
        row_map, col_map, valid = _identity_maps(64, 80)

        out = resample(image, row_map[:64, :80], col_map[:64, :80],
                       valid[:64, :80], backend='scipy')

        assert out.shape == (64, 80), (
            f"Output shape {out.shape} should match map shape (64, 80)"
        )

    def test_dtype_preserved_float32(self):
        """resample preserves float32 dtype in the output."""
        image = np.ones((20, 20), dtype=np.float32)
        row_map, col_map, valid = _identity_maps(20, 20)
        out = resample(image, row_map, col_map, valid, backend='scipy')
        assert out.dtype == np.float32, (
            f"Expected output dtype float32, got {out.dtype}"
        )

    def test_dtype_preserved_float64(self):
        """resample preserves float64 dtype in the output."""
        image = np.ones((20, 20), dtype=np.float64)
        row_map, col_map, valid = _identity_maps(20, 20)
        out = resample(image, row_map, col_map, valid, backend='scipy')
        assert out.dtype == np.float64, (
            f"Expected output dtype float64, got {out.dtype}"
        )

    def test_multiband_input_produces_multiband_output(self):
        """(B, H, W) input returns (B, OH, OW) output."""
        rng = np.random.default_rng(1)
        image = rng.random((3, 32, 32), dtype=np.float32)
        row_map, col_map, valid = _identity_maps(32, 32)

        out = resample(image, row_map, col_map, valid, backend='scipy')

        assert out.ndim == 3, f"Expected 3D output for multiband input, got {out.ndim}D"
        assert out.shape[0] == 3, (
            f"Band count {out.shape[0]} should match input ({image.shape[0]})"
        )
        assert out.shape[1:] == (32, 32), (
            f"Spatial shape {out.shape[1:]} should match map shape (32, 32)"
        )

    def test_complex_input_real_imag_preserved(self):
        """Complex-valued input: real and imag parts are resampled independently.

        An identity-mapped complex image must return the same complex values.
        """
        rng = np.random.default_rng(2)
        real = rng.random((16, 16))
        imag = rng.random((16, 16))
        image = (real + 1j * imag).astype(np.complex64)

        row_map, col_map, valid = _identity_maps(16, 16)
        out = resample(image, row_map, col_map, valid, backend='scipy')

        assert np.iscomplexobj(out), "Complex input must produce complex output"
        assert out.dtype == np.complex64, (
            f"Complex dtype not preserved: got {out.dtype}"
        )
        np.testing.assert_allclose(
            out[1:-1, 1:-1].real, image[1:-1, 1:-1].real, atol=1e-5,
            err_msg="Complex real part changed after identity resample",
        )
        np.testing.assert_allclose(
            out[1:-1, 1:-1].imag, image[1:-1, 1:-1].imag, atol=1e-5,
            err_msg="Complex imaginary part changed after identity resample",
        )

    def test_order_0_nearest_produces_exact_source_values(self):
        """order=0 (nearest-neighbour) output values must exist in the source.

        Nearest-neighbour never blends, so every non-nodata output value
        must be exactly equal to some source pixel value.
        """
        rng = np.random.default_rng(3)
        image = rng.integers(0, 256, (16, 16), dtype=np.uint8).astype(np.float32)
        row_map, col_map, valid = _identity_maps(16, 16)

        out = resample(image, row_map, col_map, valid,
                       order=0, backend='scipy')

        source_values = set(image.ravel().tolist())
        for v in out.ravel()[:100]:
            assert v in source_values, (
                f"Nearest-neighbour output value {v} not found in source pixel set — "
                "nearest is blending values like bilinear"
            )


# ===========================================================================
# Level 3: Cross-backend agreement
# ===========================================================================

class TestCrossBackendAgreement:
    """All available backends must agree with scipy within 1% RMSE."""

    def test_all_available_backends_agree_on_bilinear(self):
        """Every available backend produces bilinear output within 1% RMSE of scipy.

        Builds a non-trivial mapping (sub-pixel shifts) and compares each
        backend's output to the scipy reference.  A large discrepancy means
        the backend implements a different algorithm or has a sign error.
        """
        rng = np.random.default_rng(99)
        h, w = 64, 64
        image = rng.random((h, w), dtype=np.float64)

        # Sub-pixel mapping: shift by 0.4 pixels in both directions
        out_rows = np.arange(h - 2, dtype=np.float64) + 0.4
        out_cols = np.arange(w - 2, dtype=np.float64) + 0.4
        col_map, row_map = np.meshgrid(out_cols, out_rows)
        valid = np.ones_like(row_map, dtype=bool)

        ref = resample(image, row_map, col_map, valid,
                       order=1, nodata=0.0, backend='scipy')
        ref_rms = float(np.sqrt(np.mean(ref ** 2)))
        if ref_rms < 1e-10:
            pytest.skip("Reference image has near-zero RMS (degenerate test data)")

        for backend in _available_backends():
            if backend == 'scipy':
                continue
            out = resample(image, row_map, col_map, valid,
                           order=1, nodata=0.0, backend=backend)
            rmse = float(np.sqrt(np.mean((out - ref) ** 2)))
            relative_rmse = rmse / ref_rms

            assert relative_rmse < 0.01, (
                f"Backend '{backend}' RMSE relative to scipy = {relative_rmse:.4f} "
                f"({rmse:.6f} absolute) — exceeds 1% threshold"
            )

    def test_scipy_parallel_matches_sequential_scipy(self):
        """scipy_parallel assembles the same result as sequential scipy.

        The parallel backend chunks rows across threads; the sequential
        backend processes all rows at once.  Both must produce identical
        output (within float64 rounding).
        """
        rng = np.random.default_rng(55)
        image = rng.random((50, 50), dtype=np.float64)

        out_rows = np.arange(50, dtype=np.float64) + 0.3
        out_cols = np.arange(50, dtype=np.float64) + 0.3
        col_map, row_map = np.meshgrid(out_cols, out_rows)
        valid = (row_map < 49.5) & (col_map < 49.5)

        out_seq = resample(image, row_map, col_map, valid,
                           order=1, nodata=0.0, backend='scipy')
        out_par = resample(image, row_map, col_map, valid,
                           order=1, nodata=0.0, backend='scipy_parallel',
                           num_workers=2)

        np.testing.assert_allclose(
            out_par, out_seq, atol=1e-10,
            err_msg="scipy_parallel and sequential scipy produce different output",
        )

    @pytest.mark.skipif(_TORCH_AVAILABLE, reason="torch is installed; fallback won't fire")
    def test_torch_order2_falls_back_without_crash(self):
        """order=2 with torch backend triggers fallback to scipy (no crash).

        Torch only supports order 0/1.  When torch is requested but absent,
        the dispatch must fall back gracefully rather than raising or
        producing garbage output.  This test only runs when torch is not
        installed so the fallback chain is actually exercised.
        """
        rng = np.random.default_rng(11)
        image = rng.random((20, 20), dtype=np.float64)
        row_map, col_map, valid = _identity_maps(20, 20)

        # Should not raise — fallback chain must handle this gracefully
        out = resample(image, row_map, col_map, valid,
                       order=2, nodata=0.0, backend='torch')

        assert out.shape == (20, 20), (
            "Fallback from torch for order=2 produced wrong output shape"
        )
        assert np.isfinite(out).any(), (
            "Fallback from torch for order=2 produced all-NaN output"
        )
