# -*- coding: utf-8 -*-
"""
NITF Reader Tests - Umbra SICD Validation with GRDL Integration.

Tests grdl.IO.NITFReader with real Umbra SICD files, including:
- Level 1: Format validation (metadata extraction, complex dtype, chip/full reads)
- Level 2: Data quality (NITF tags, complex magnitude/phase, SAR statistics)
- Level 3: Integration (ChipExtractor, Normalizer, Tiler workflows with SAR data)

Dataset: Umbra SAR Spotlight (SICD format in NITF container)

Dependencies
------------
pytest
rasterio
numpy
grdl

Author
------
Duane Smalley
duane.d.smalley@gmail.com

Steven Siebert

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2026-02-11

Modified
--------
2026-02-11
"""

# Standard library
from pathlib import Path

# Third-party
import pytest
import numpy as np

try:
    import rasterio
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False

# GRDL internal
try:
    from grdl.IO.nitf import NITFReader
    from grdl.IO.models import ImageMetadata
    _HAS_GRDL = True
except ImportError:
    _HAS_GRDL = False

try:
    from grdl.data_prep import ChipExtractor, Normalizer, Tiler
    _HAS_DATA_PREP = True
except ImportError:
    _HAS_DATA_PREP = False


pytestmark = [
    pytest.mark.nitf,
    pytest.mark.requires_data,
    pytest.mark.skipif(not _HAS_RASTERIO, reason="rasterio not installed"),
    pytest.mark.skipif(not _HAS_GRDL, reason="grdl not installed"),
]


# =============================================================================
# Level 1: Format Validation
# =============================================================================


@pytest.mark.slow
def test_umbra_metadata(require_umbra_file):
    """Verify NITFReader extracts valid Umbra SICD metadata.

    Validates format string, positive dimensions, and that the NITF
    container is recognized as such.
    """
    with NITFReader(str(require_umbra_file)) as reader:
        metadata = reader.metadata
        assert isinstance(metadata, ImageMetadata)

        # Umbra SICD specific properties
        assert metadata.format == 'NITF'
        assert metadata.rows > 0
        assert metadata.cols > 0

        print(f"Umbra SICD metadata: {metadata.rows}x{metadata.cols}")


@pytest.mark.slow
def test_umbra_complex_dtype(require_umbra_file):
    """Verify NITFReader detects complex SAR data type.

    SICD files contain complex-valued SAR imagery. The dtype must
    be complex64 or complex128.
    """
    with NITFReader(str(require_umbra_file)) as reader:
        dtype = reader.get_dtype()

        # SICD files contain complex-valued SAR imagery
        assert dtype in [np.complex64, np.complex128], \
            f"Expected complex dtype for SICD, got: {dtype}"

        # dtype must agree with metadata
        assert str(dtype) == reader.metadata.dtype

        print(f"Umbra SICD dtype: {dtype} (complex SAR data)")


@pytest.mark.slow
def test_umbra_get_shape(require_umbra_file):
    """Verify get_shape() returns valid positive dimensions consistent with metadata."""
    with NITFReader(str(require_umbra_file)) as reader:
        shape = reader.get_shape()

        assert isinstance(shape, tuple)
        assert len(shape) >= 2

        rows, cols = shape[0], shape[1]
        assert isinstance(rows, int)
        assert isinstance(cols, int)
        assert rows > 0 and cols > 0

        # Shape must agree with metadata
        assert rows == reader.metadata.rows
        assert cols == reader.metadata.cols

        print(f"Umbra SICD shape: {rows} rows x {cols} cols")


@pytest.mark.slow
def test_umbra_read_chip(require_umbra_file):
    """Verify read_chip() returns complex-valued array with correct shape."""
    with NITFReader(str(require_umbra_file)) as reader:
        rows, cols = reader.get_shape()[:2]

        # Read 256x256 chip from center (smaller for complex data)
        row_start = max(0, rows // 2 - 128)
        row_end = min(rows, rows // 2 + 128)
        col_start = max(0, cols // 2 - 128)
        col_end = min(cols, cols // 2 + 128)

        chip = reader.read_chip(row_start, row_end, col_start, col_end)

        assert isinstance(chip, np.ndarray)
        assert chip.ndim == 2  # Single band (complex)
        assert chip.shape[0] == (row_end - row_start)
        assert chip.shape[1] == (col_end - col_start)
        assert chip.dtype in [np.complex64, np.complex128]

        # Complex SAR data should have non-zero content
        assert np.abs(chip).max() > 0, \
            "Center chip has zero magnitude (no SAR signal)"

        print(f"Complex chip shape: {chip.shape}, dtype: {chip.dtype}")


@pytest.mark.slow
def test_umbra_read_full(require_umbra_file):
    """Verify read_full() returns complex array matching reported shape."""
    with NITFReader(str(require_umbra_file)) as reader:
        rows, cols = reader.get_shape()[:2]

        # Complex data is 2x size of real; limit full reads
        if rows * cols > 25_000_000:  # ~25 megapixels for complex
            pytest.skip("Umbra file too large for full read test")

        data = reader.read_full()

        assert isinstance(data, np.ndarray)
        assert data.shape[0] == rows
        assert data.shape[1] == cols
        assert data.dtype in [np.complex64, np.complex128]

        print(f"Full Umbra SICD read: {data.shape}")


@pytest.mark.slow
def test_umbra_context_manager(require_umbra_file):
    """Verify context manager opens and releases NITF resources."""
    reader = NITFReader(str(require_umbra_file))
    assert hasattr(reader, '__enter__')
    assert hasattr(reader, '__exit__')

    with reader:
        shape = reader.get_shape()
        assert len(shape) >= 2


# =============================================================================
# Level 2: Data Quality
# =============================================================================


@pytest.mark.slow
def test_umbra_nitf_tags(require_umbra_file):
    """Verify NITF metadata tags are present and non-empty.

    NITF files carry metadata in tags and TRE (Tagged Record Extension)
    fields. The reader should expose these through metadata extras.
    """
    with NITFReader(str(require_umbra_file)) as reader:
        metadata = reader.metadata

        # The metadata.extras should contain format-specific information
        assert metadata.extras is not None
        assert len(metadata.extras) > 0, \
            "NITF metadata extras should contain format-specific fields"

        print(f"NITF metadata extras keys: {list(metadata.extras.keys())}")

        # If tags are available, validate their structure
        if 'tags' in metadata.extras:
            tags = metadata.extras['tags']
            assert isinstance(tags, dict)
            assert len(tags) > 0, "NITF tags dict should not be empty"
            print(f"NITF has {len(tags)} metadata tags")


@pytest.mark.slow
def test_umbra_complex_magnitude(require_umbra_file):
    """Verify complex SAR magnitude extraction produces valid results.

    Magnitude of complex SAR data must be non-negative and have
    non-zero variance (indicating real signal content).
    """
    with NITFReader(str(require_umbra_file)) as reader:
        rows, cols = reader.get_shape()[:2]

        # Read from center for best signal content
        r0 = max(0, rows // 2 - 128)
        r1 = min(rows, rows // 2 + 128)
        c0 = max(0, cols // 2 - 128)
        c1 = min(cols, cols // 2 + 128)
        chip = reader.read_chip(r0, r1, c0, c1)

        magnitude = np.abs(chip)

        assert isinstance(magnitude, np.ndarray)
        assert magnitude.dtype in [np.float32, np.float64]
        assert magnitude.min() >= 0, "Magnitude must be non-negative"
        assert magnitude.shape == chip.shape
        assert magnitude.max() > 0, "Magnitude max is zero (no signal)"
        assert magnitude.std() > 0, "Magnitude has zero variance"

        print(f"Complex magnitude range: [{magnitude.min():.2f}, "
              f"{magnitude.max():.2f}]")


@pytest.mark.slow
def test_umbra_complex_phase(require_umbra_file):
    """Verify complex SAR phase extraction produces values in [-pi, pi].

    Phase represents relative distance information in SAR imagery.
    All values must fall within the mathematically valid range.
    """
    with NITFReader(str(require_umbra_file)) as reader:
        rows, cols = reader.get_shape()[:2]

        r0 = max(0, rows // 2 - 128)
        r1 = min(rows, rows // 2 + 128)
        c0 = max(0, cols // 2 - 128)
        c1 = min(cols, cols // 2 + 128)
        chip = reader.read_chip(r0, r1, c0, c1)

        phase = np.angle(chip)

        assert isinstance(phase, np.ndarray)
        assert phase.dtype in [np.float32, np.float64]
        assert phase.min() >= -np.pi, \
            f"Phase min {phase.min():.4f} below -pi"
        assert phase.max() <= np.pi, \
            f"Phase max {phase.max():.4f} above pi"
        assert phase.shape == chip.shape

        # Phase should span a reasonable range for real SAR data
        phase_range = phase.max() - phase.min()
        assert phase_range > 0.1, \
            "Phase range is suspiciously narrow for real SAR data"

        print(f"Complex phase range: [{phase.min():.3f}, {phase.max():.3f}] radians")


@pytest.mark.slow
def test_umbra_sar_statistics(require_umbra_file):
    """Verify SAR backscatter statistics are physically plausible.

    SAR magnitude should exhibit positive mean, non-zero variance,
    and a maximum that exceeds the mean (typical for speckle).
    The coefficient of variation (std/mean) for SAR imagery is
    characteristically high due to speckle noise.
    """
    with NITFReader(str(require_umbra_file)) as reader:
        rows, cols = reader.get_shape()[:2]

        r0 = max(0, rows // 2 - 256)
        r1 = min(rows, rows // 2 + 256)
        c0 = max(0, cols // 2 - 256)
        c1 = min(cols, cols // 2 + 256)
        chip = reader.read_chip(r0, r1, c0, c1)

        magnitude = np.abs(chip)

        mean_mag = magnitude.mean()
        std_mag = magnitude.std()
        max_mag = magnitude.max()

        assert mean_mag > 0, "SAR magnitude mean should be positive"
        assert std_mag > 0, "SAR magnitude should have non-zero variance"
        assert max_mag > mean_mag, "SAR max should exceed mean (speckle)"

        # Coefficient of variation: typical for SAR is > 0.1
        cv = std_mag / mean_mag
        assert cv > 0.1, \
            f"Coefficient of variation {cv:.3f} is unusually low for SAR"

        print(f"SAR statistics: mean={mean_mag:.2f}, std={std_mag:.2f}, "
              f"max={max_mag:.2f}, CV={cv:.3f}")


# =============================================================================
# Level 3: Integration with GRDL Utilities
# =============================================================================


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
@pytest.mark.integration
def test_umbra_chip_extractor_integration(require_umbra_file):
    """Validate ChipExtractor partitions Umbra SAR data into uniform chips.

    Verifies chip regions are within bounds and that complex-valued
    chips have correct dimensions and dtype.
    """
    with NITFReader(str(require_umbra_file)) as reader:
        rows, cols = reader.get_shape()[:2]

        extractor = ChipExtractor(nrows=rows, ncols=cols)
        regions = extractor.chip_positions(row_width=128, col_width=128)

        assert len(regions) > 0

        for i, region in enumerate(regions[:5]):
            assert 0 <= region.row_start < region.row_end <= rows
            assert 0 <= region.col_start < region.col_end <= cols

            chip = reader.read_chip(
                region.row_start, region.row_end,
                region.col_start, region.col_end
            )

            assert isinstance(chip, np.ndarray)
            assert chip.ndim == 2
            assert chip.dtype in [np.complex64, np.complex128]
            assert chip.shape[0] == region.row_end - region.row_start
            assert chip.shape[1] == region.col_end - region.col_start

        print(f"ChipExtractor: {len(regions)} SAR chips validated")


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
@pytest.mark.integration
def test_umbra_normalizer_magnitude_integration(require_umbra_file):
    """Validate magnitude extraction and normalization pipeline.

    Extracts magnitude from complex SAR, normalizes with minmax,
    and validates output is finite and bounded in [0, 1].
    """
    with NITFReader(str(require_umbra_file)) as reader:
        rows, cols = reader.get_shape()[:2]
        chip = reader.read_chip(0, min(512, rows), 0, min(512, cols))

        magnitude = np.abs(chip)

        normalizer = Normalizer(method='minmax')
        normalized = normalizer.normalize(magnitude)

        assert isinstance(normalized, np.ndarray)
        assert normalized.dtype == np.float64
        assert np.isfinite(normalized).all()
        assert 0.0 <= normalized.min() <= normalized.max() <= 1.0

        print(f"SAR magnitude normalized: [{normalized.min():.3f}, "
              f"{normalized.max():.3f}]")


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
@pytest.mark.integration
def test_umbra_tiler_integration(require_umbra_file):
    """Validate Tiler creates overlapping tile grid for SAR processing.

    Verifies tile regions are within bounds and complex-valued tiles
    read correctly.
    """
    with NITFReader(str(require_umbra_file)) as reader:
        rows, cols = reader.get_shape()[:2]

        tiler = Tiler(
            nrows=rows,
            ncols=cols,
            tile_size=256,
            stride=128
        )

        tiles = tiler.tile_positions()
        assert len(tiles) > 0

        for i, tile_region in enumerate(tiles[:5]):
            assert 0 <= tile_region.row_start < tile_region.row_end <= rows
            assert 0 <= tile_region.col_start < tile_region.col_end <= cols

            tile = reader.read_chip(
                tile_region.row_start, tile_region.row_end,
                tile_region.col_start, tile_region.col_end
            )

            assert isinstance(tile, np.ndarray)
            assert tile.ndim == 2
            assert tile.dtype in [np.complex64, np.complex128]
            assert tile.shape[0] <= 256
            assert tile.shape[1] <= 256

        print(f"Tiler: {len(tiles)} SAR tiles validated")


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
@pytest.mark.integration
def test_umbra_sar_processing_pipeline(require_umbra_file):
    """Validate complete SAR processing pipeline.

    End-to-end workflow: load complex data, extract chips, convert
    to magnitude, normalize, and validate output properties.
    """
    with NITFReader(str(require_umbra_file)) as reader:
        rows, cols = reader.get_shape()[:2]

        extractor = ChipExtractor(nrows=rows, ncols=cols)
        regions = extractor.chip_positions(row_width=128, col_width=128)

        normalizer = Normalizer(method='zscore')
        normalized_magnitudes = []

        for region in regions[:10]:
            chip = reader.read_chip(
                region.row_start, region.row_end,
                region.col_start, region.col_end
            )

            magnitude = np.abs(chip).astype(np.float32)
            normalized = normalizer.normalize(magnitude)
            normalized_magnitudes.append(normalized)

            assert isinstance(normalized, np.ndarray)
            assert normalized.dtype == np.float64
            assert np.isfinite(normalized).all()

        assert len(normalized_magnitudes) > 0

        # Batch statistics
        all_values = np.concatenate([nm.flatten() for nm in normalized_magnitudes])
        print(f"SAR pipeline: {len(normalized_magnitudes)} chips, "
              f"batch mean={all_values.mean():.3f}, std={all_values.std():.3f}")


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
@pytest.mark.integration
def test_umbra_complex_data_preservation(require_umbra_file):
    """Verify complex data integrity through chip extraction.

    Validates that magnitude and phase can be extracted, and that
    reconstructing complex values from them reproduces the original
    within floating-point precision.
    """
    with NITFReader(str(require_umbra_file)) as reader:
        rows, cols = reader.get_shape()[:2]

        chip = reader.read_chip(
            rows // 4, rows // 4 + 256,
            cols // 4, cols // 4 + 256
        )

        assert chip.dtype in [np.complex64, np.complex128]

        # Extract magnitude and phase
        magnitude = np.abs(chip)
        phase = np.angle(chip)

        # Reconstruct complex from magnitude and phase
        reconstructed = magnitude * np.exp(1j * phase)

        # Should match original within floating-point precision
        # Magnitude/phase round-trip amplifies relative error near zero,
        # so use atol alongside rtol for robustness.
        np.testing.assert_allclose(
            chip.real, reconstructed.real, rtol=1e-2, atol=1e-6)
        np.testing.assert_allclose(
            chip.imag, reconstructed.imag, rtol=1e-2, atol=1e-6)

        print("Complex data integrity verified: magnitude + phase preserved")


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
@pytest.mark.integration
def test_umbra_multi_chip_normalization(require_umbra_file):
    """Validate per-chip percentile normalization for SAR display.

    Each chip is normalized independently using percentile stretch,
    which is the standard approach for SAR display (handles speckle
    outliers better than minmax).
    """
    with NITFReader(str(require_umbra_file)) as reader:
        rows, cols = reader.get_shape()[:2]

        extractor = ChipExtractor(nrows=rows, ncols=cols)
        regions = extractor.chip_positions(row_width=256, col_width=256)

        for i, region in enumerate(regions[:5]):
            chip = reader.read_chip(
                region.row_start, region.row_end,
                region.col_start, region.col_end
            )

            magnitude = np.abs(chip).astype(np.float32)

            # Per-chip percentile normalization
            normalizer = Normalizer(
                method='percentile', percentile_low=5.0, percentile_high=95.0)
            normalized = normalizer.normalize(magnitude)

            assert np.isfinite(normalized).all()
            assert 0.0 <= normalized.min() <= normalized.max() <= 1.0

            print(f"Chip {i}: normalized range [{normalized.min():.3f}, "
                  f"{normalized.max():.3f}]")
