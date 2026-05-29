# -*- coding: utf-8 -*-
"""
Contrast Operator Validation - Real SAR and EO imagery.

Tests all grdl.contrast operators (MangisDensity, NRLStretch, LinearStretch,
LogStretch, GammaCorrection, SigmoidStretch, HistogramEqualization, CLAHE)
against real Umbra SICD SAR data and Landsat surface reflectance.

- Level 1: Output shape, dtype (float32), value range [0, 1]
- Level 2: Monotonicity, non-degenerate output, complex input handling,
           physical invariant validation
- Level 3: Pipeline composition (stretch → gamma → clip_cast)

Dataset: Umbra SICD (*.nitf), Landsat 8/9 (*.TIF)

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

from pathlib import Path

import pytest
import numpy as np

try:
    from grdl.contrast import (
        MangisDensity,
        NRLStretch,
        LinearStretch,
        LogStretch,
        GammaCorrection,
        SigmoidStretch,
        HistogramEqualization,
        CLAHE,
        clip_cast,
        nan_safe_stats,
    )
    _HAS_CONTRAST = True
except ImportError:
    _HAS_CONTRAST = False

try:
    from grdl.IO.sar import SICDReader
    _HAS_SICD = True
except ImportError:
    _HAS_SICD = False

try:
    from grdl.IO import GeoTIFFReader
    _HAS_GEOTIFF = True
except ImportError:
    _HAS_GEOTIFF = False


pytestmark = [
    pytest.mark.skipif(not _HAS_CONTRAST,
                       reason="grdl.contrast not available"),
    pytest.mark.requires_data,
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def sicd_chip(umbra_data_dir):
    """256x256 complex chip from Umbra SICD center."""
    if not _HAS_SICD:
        pytest.skip("SICDReader not available")
    matches = list(umbra_data_dir.glob("*.nitf")) if umbra_data_dir.exists() else []
    if not matches:
        pytest.skip(f"Umbra SICD not found in {umbra_data_dir}")
    with SICDReader(str(matches[0])) as reader:
        rows, cols = reader.get_shape()
        r0 = rows // 2 - 128
        c0 = cols // 2 - 128
        chip = reader.read_chip(r0, r0 + 256, c0, c0 + 256)
    return chip


@pytest.fixture(scope="module")
def landsat_chip(landsat_data_dir):
    """256x256 real chip from Landsat surface reflectance."""
    if not _HAS_GEOTIFF:
        pytest.skip("GeoTIFFReader not available")
    matches = list(landsat_data_dir.glob("LC0[89]*_SR_B*.TIF")) if landsat_data_dir.exists() else []
    if not matches:
        pytest.skip(f"Landsat data not found in {landsat_data_dir}")
    with GeoTIFFReader(str(matches[0])) as reader:
        rows, cols = reader.get_shape()
        r0 = rows // 2 - 128
        c0 = cols // 2 - 128
        chip = reader.read_chip(r0, r0 + 256, c0, c0 + 256)
    return chip


@pytest.fixture(scope="module")
def sicd_amplitude(sicd_chip):
    """Real-valued SAR amplitude from complex chip."""
    return np.abs(sicd_chip).astype(np.float64)


# =============================================================================
# Level 1: Format Validation — Output contract (shape, dtype, range)
# =============================================================================


class TestContrastLevel1:
    """All operators must produce float32 in [0, 1] with matching shape."""

    @pytest.mark.parametrize("OpClass", [
        MangisDensity, NRLStretch, LinearStretch, LogStretch,
        HistogramEqualization,
    ])
    def test_output_shape_dtype_range_sar(self, sicd_amplitude, OpClass):
        """Operators produce correct shape/dtype/range on SAR amplitude."""
        op = OpClass()
        result = op.apply(sicd_amplitude)
        assert result.shape == sicd_amplitude.shape, (
            f"{OpClass.__name__}: shape mismatch"
        )
        assert result.dtype == np.float32, (
            f"{OpClass.__name__}: expected float32, got {result.dtype}"
        )
        assert result.min() >= 0.0, (
            f"{OpClass.__name__}: output below 0 ({result.min():.6f})"
        )
        assert result.max() <= 1.0, (
            f"{OpClass.__name__}: output above 1 ({result.max():.6f})"
        )

    @pytest.mark.parametrize("OpClass", [
        MangisDensity, NRLStretch, LinearStretch, LogStretch,
        HistogramEqualization,
    ])
    def test_output_shape_dtype_range_eo(self, landsat_chip, OpClass):
        """Operators produce correct shape/dtype/range on EO data."""
        data = landsat_chip.astype(np.float64)
        op = OpClass()
        result = op.apply(data)
        assert result.shape == data.shape
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_complex_input_handled(self, sicd_chip):
        """Complex input is auto-converted to magnitude."""
        op = MangisDensity()
        result = op.apply(sicd_chip)
        assert result.shape == sicd_chip.shape
        assert result.dtype == np.float32
        assert not np.iscomplexobj(result)

    def test_clahe_output_contract(self, sicd_amplitude):
        """CLAHE produces float32 [0, 1]."""
        op = CLAHE(kernel_size=64)
        result = op.apply(sicd_amplitude)
        assert result.shape == sicd_amplitude.shape
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_gamma_output_contract(self, sicd_amplitude):
        """GammaCorrection on pre-normalized input."""
        pre = LinearStretch().apply(sicd_amplitude)
        result = GammaCorrection(gamma=2.2).apply(pre)
        assert result.shape == pre.shape
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_sigmoid_output_contract(self, sicd_amplitude):
        """SigmoidStretch on pre-normalized input."""
        pre = LinearStretch().apply(sicd_amplitude)
        result = SigmoidStretch(center=0.5, slope=10.0).apply(pre)
        assert result.shape == pre.shape
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0


# =============================================================================
# Level 2: Data Quality — Physical invariants
# =============================================================================


class TestContrastLevel2:
    """Validate physical properties and non-degeneracy."""

    def test_mangis_logarithmic_compression(self, sicd_amplitude):
        """MangisDensity compresses dynamic range via log10 mapping.

        The Mangis algorithm: slope * log10(amplitude) + constant, calibrated
        so that c_l = 0.8*mean → dmin/255 and c_h = mmult*c_l → 1.0.

        On real SAR data (Rayleigh-distributed amplitude), naive linear
        stretch concentrates most pixels near 0 because the max is dominated
        by rare bright scatterers. MangisDensity's log compression must
        lift the bulk of the distribution upward relative to linear.
        """
        linear = LinearStretch().apply(sicd_amplitude)
        mangis = MangisDensity().apply(sicd_amplitude)

        # Assertion 1: Log compression lifts the median.
        # Real SAR amplitude is right-skewed (Rayleigh + point targets).
        # LinearStretch pins max to 1.0, so the median ends up very low.
        # MangisDensity's log10 must shift the median significantly higher.
        median_lift = np.median(mangis) - np.median(linear)
        assert median_lift > 0.05, (
            f"MangisDensity did not lift background relative to linear: "
            f"median(mangis)={np.median(mangis):.4f}, "
            f"median(linear)={np.median(linear):.4f}, "
            f"lift={median_lift:.4f}"
        )

        # Assertion 2: MangisDensity reduces the fraction of near-black pixels.
        # In linear space, ~80% of SAR pixels fall below 0.1 (dominated by
        # the clutter floor). After log compression, that fraction must drop
        # substantially — the algorithm's purpose is to make clutter visible.
        frac_dark_linear = (linear < 0.1).sum() / linear.size
        frac_dark_mangis = (mangis < 0.1).sum() / mangis.size
        assert frac_dark_mangis < frac_dark_linear, (
            f"MangisDensity has MORE near-black pixels than linear: "
            f"mangis={frac_dark_mangis:.4f}, linear={frac_dark_linear:.4f}"
        )
        # The reduction should be meaningful (not just 1 pixel difference)
        assert frac_dark_mangis < frac_dark_linear * 0.75, (
            f"Compression insufficient: dark fraction only reduced from "
            f"{frac_dark_linear:.4f} to {frac_dark_mangis:.4f} "
            f"(expected at least 25% reduction)"
        )

    def test_nrl_knee_compression(self, sicd_amplitude):
        """NRLStretch enforces linear-then-log2 knee at the 99th percentile.

        Algorithm structure:
        - x <= p99 (changeover): linear map to [0, knee=0.8]
        - x > p99: log2 compression into [knee=0.8, 1.0]

        The knee is the critical design point: 99% of pixels get 80% of
        the output range (preserving faint detail), while the top 1% bright
        scatterers are compressed into only 20% of the output range.
        """
        nrl = NRLStretch(knee=0.8, percentile=99.0).apply(sicd_amplitude)

        # Identify the input changeover point
        p99_amp = np.percentile(sicd_amplitude, 99)
        below_knee = sicd_amplitude <= p99_amp
        above_knee = sicd_amplitude > p99_amp

        # Assertion 1: Below-changeover pixels map strictly into [0, knee].
        # The linear segment must not exceed 0.8.
        below_output = nrl[below_knee]
        assert below_output.max() <= 0.80 + 1e-4, (
            f"Linear segment exceeds knee: max={below_output.max():.6f}"
        )

        # Assertion 2: Above-changeover pixels map into [knee, 1.0].
        # The log segment must start at or above the knee point.
        above_output = nrl[above_knee]
        assert above_output.min() >= 0.80 - 1e-4, (
            f"Log segment starts below knee: min={above_output.min():.6f}"
        )
        assert above_output.max() <= 1.0 + 1e-6

        # Assertion 3: The fraction of output pixels above the knee
        # should equal the fraction of input pixels above the 99th percentile
        # (i.e., ~1%). This confirms the knee position is calibrated correctly.
        frac_above_knee = (nrl > 0.80).sum() / nrl.size
        assert abs(frac_above_knee - 0.01) < 0.005, (
            f"Knee partition mismatch: {frac_above_knee*100:.2f}% above knee "
            f"(expected ~1%)"
        )

        # Assertion 4: The bright scatterers (top 1%) are compressed.
        # Their input dynamic range spans from p99 to max (potentially huge),
        # but output is confined to [0.8, 1.0] — at most 0.2 span.
        input_bright_range = sicd_amplitude[above_knee].max() - sicd_amplitude[above_knee].min()
        output_bright_span = above_output.max() - above_output.min()
        assert output_bright_span <= 0.20 + 1e-4, (
            f"Log segment exceeds 0.2 span: {output_bright_span:.4f}"
        )

    def test_linear_stretch_uses_full_range(self, sicd_amplitude):
        """LinearStretch maps data to use the full [0, 1] interval."""
        result = LinearStretch().apply(sicd_amplitude)
        assert result.max() >= 0.99, "LinearStretch didn't reach near 1.0"
        assert result.min() <= 0.01, "LinearStretch didn't reach near 0.0"

    def test_log_stretch_compresses_dynamic_range(self, sicd_amplitude):
        """LogStretch compresses the upper end relative to linear."""
        linear = LinearStretch().apply(sicd_amplitude)
        log = LogStretch().apply(sicd_amplitude)
        # Log stretch should shift the median upward (compression of darks)
        assert np.median(log) > np.median(linear), (
            "LogStretch did not compress dynamic range as expected"
        )

    def test_gamma_identity(self, sicd_amplitude):
        """Gamma=1.0 is identity transform (within float32 precision)."""
        pre = LinearStretch().apply(sicd_amplitude)
        result = GammaCorrection(gamma=1.0).apply(pre)
        np.testing.assert_allclose(result, pre, atol=1e-6)

    def test_gamma_brightens_with_high_gamma(self, sicd_amplitude):
        """Gamma > 1 brightens midtones."""
        pre = LinearStretch().apply(sicd_amplitude)
        bright = GammaCorrection(gamma=2.2).apply(pre)
        # Midtone pixels should be brighter (higher value)
        mid_mask = (pre > 0.2) & (pre < 0.8)
        if mid_mask.sum() > 100:
            assert bright[mid_mask].mean() > pre[mid_mask].mean()

    def test_sigmoid_S_shape(self, sicd_amplitude):
        """SigmoidStretch compresses values toward 0 and 1 (S-curve)."""
        pre = LinearStretch().apply(sicd_amplitude)
        sig = SigmoidStretch(center=0.5, slope=10.0).apply(pre)
        # Sigmoid pushes values toward extremes — verify bimodal tendency
        near_zero = (sig < 0.1).sum()
        near_one = (sig > 0.9).sum()
        extremes_frac = (near_zero + near_one) / sig.size
        assert extremes_frac > 0.5, (
            f"Sigmoid S-curve not evident: only {extremes_frac*100:.1f}% at extremes"
        )

    def test_histogram_equalization_uniform_distribution(self, sicd_amplitude):
        """HistogramEqualization produces approximately uniform output."""
        result = HistogramEqualization().apply(sicd_amplitude)
        hist, _ = np.histogram(result[np.isfinite(result)], bins=10,
                               range=(0, 1))
        # No bin should have more than 3x the ideal count
        ideal = hist.sum() / 10
        assert hist.max() < ideal * 3, (
            f"Histogram not approximately uniform: max bin={hist.max()}, "
            f"ideal={ideal:.0f}"
        )

    def test_clahe_clip_limit_constrains_local_histogram(self, sicd_amplitude):
        """CLAHE clip_limit caps the per-tile histogram redistribution.

        CLAHE works by equalizing histograms within local tiles, but clips
        any histogram bin exceeding clip_limit * (tile_pixels / n_bins).
        The clipped counts are redistributed uniformly.

        We verify this by comparing two CLAHE runs:
        - clip_limit=0.01 (aggressive limiting → gentler enhancement)
        - clip_limit=1.0 (no limiting → equivalent to tiled equalization)

        The constrained version must have lower local contrast (measured as
        the max gradient magnitude within tiles) because the clip limit
        prevents any single intensity bin from dominating the CDF.
        """
        kernel_size = 64

        clahe_clipped = CLAHE(kernel_size=kernel_size, clip_limit=0.01).apply(
            sicd_amplitude
        )
        clahe_unclipped = CLAHE(kernel_size=kernel_size, clip_limit=1.0).apply(
            sicd_amplitude
        )

        # Measure local contrast via max absolute difference within each tile.
        # In a clipped CLAHE, the per-tile dynamic range should be smaller
        # because the CDF slope is bounded.
        n_tiles_r = sicd_amplitude.shape[0] // kernel_size
        n_tiles_c = sicd_amplitude.shape[1] // kernel_size

        clipped_tile_ranges = []
        unclipped_tile_ranges = []
        for tr in range(n_tiles_r):
            for tc in range(n_tiles_c):
                r0, r1 = tr * kernel_size, (tr + 1) * kernel_size
                c0, c1 = tc * kernel_size, (tc + 1) * kernel_size
                tile_clip = clahe_clipped[r0:r1, c0:c1]
                tile_unclip = clahe_unclipped[r0:r1, c0:c1]
                clipped_tile_ranges.append(tile_clip.max() - tile_clip.min())
                unclipped_tile_ranges.append(tile_unclip.max() - tile_unclip.min())

        mean_range_clipped = np.mean(clipped_tile_ranges)
        mean_range_unclipped = np.mean(unclipped_tile_ranges)

        # Assertion 1: The unclipped (no limit) version should have higher
        # or equal per-tile dynamic range, since it can push harder.
        assert mean_range_unclipped >= mean_range_clipped * 0.99, (
            f"Unclipped CLAHE has LESS tile range than clipped: "
            f"unclipped={mean_range_unclipped:.4f}, clipped={mean_range_clipped:.4f}"
        )

        # Assertion 2: With a tight clip_limit (0.01), the clipped result
        # should measurably differ — the constraint should actually bite.
        # If clip_limit had no effect, the two outputs would be identical.
        diff = np.abs(clahe_clipped - clahe_unclipped).mean()
        assert diff > 0.01, (
            f"clip_limit=0.01 had no effect vs clip_limit=1.0 (diff={diff:.6f}). "
            f"Either the clip limit is not being enforced or the data lacks "
            f"the dynamic range to trigger it."
        )

        # Assertion 3: Both outputs should be non-degenerate and span [0, 1]
        # to confirm CLAHE is actually running tile-wise equalization.
        assert clahe_clipped.max() > 0.9, "Clipped CLAHE didn't reach near 1.0"
        assert clahe_clipped.min() < 0.1, "Clipped CLAHE didn't reach near 0.0"

    def test_nan_safe_stats_handles_nans(self, sicd_amplitude):
        """nan_safe_stats ignores NaN/Inf values."""
        data = sicd_amplitude.copy()
        data[0, 0] = np.nan
        data[0, 1] = np.inf
        mn, mx, pct = nan_safe_stats(data, percentile=99.0)
        assert np.isfinite(mn)
        assert np.isfinite(mx)
        assert np.isfinite(pct)
        assert mn < mx

    def test_clip_cast_no_overflow(self, sicd_amplitude):
        """clip_cast prevents integer overflow."""
        stretched = LinearStretch().apply(sicd_amplitude) * 300  # exceeds uint8
        result = clip_cast(stretched, dtype='uint8')
        assert result.dtype == np.uint8
        assert result.max() <= 255
        assert result.min() >= 0


# =============================================================================
# Level 3: Integration — Pipeline composition
# =============================================================================


class TestContrastLevel3:
    """End-to-end pipeline: stretch → gamma → clip_cast."""

    @pytest.mark.integration
    def test_sar_display_pipeline(self, sicd_chip):
        """Full SAR display pipeline: complex → MangisDensity → uint8."""
        stretched = MangisDensity().apply(sicd_chip)
        display = clip_cast(stretched * 255.0, dtype='uint8')
        assert display.dtype == np.uint8
        assert display.shape == sicd_chip.shape
        assert display.max() > 0, "Display pipeline produced all-black image"
        assert display.min() < 255, "Display pipeline produced all-white image"

    @pytest.mark.integration
    def test_eo_display_pipeline(self, landsat_chip):
        """Full EO pipeline: PercentileStretch → Gamma → uint8."""
        from grdl.contrast import PercentileStretch
        data = landsat_chip.astype(np.float64)
        stretched = PercentileStretch(plow=2, phigh=98).apply(data)
        gamma = GammaCorrection(gamma=1.5).apply(stretched)
        display = clip_cast(gamma * 255.0, dtype='uint8')
        assert display.dtype == np.uint8
        assert display.shape == data.shape
        assert display.std() > 5, "Pipeline produced low-contrast image"

    @pytest.mark.integration
    def test_nrl_to_sigmoid_chain(self, sicd_amplitude):
        """NRL → Sigmoid composition maintains valid range."""
        nrl = NRLStretch().apply(sicd_amplitude)
        final = SigmoidStretch(center=0.5, slope=8.0).apply(nrl)
        assert final.dtype == np.float32
        assert final.min() >= 0.0
        assert final.max() <= 1.0
        assert final.std() > 0.01
