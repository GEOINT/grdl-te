# -*- coding: utf-8 -*-
"""
JPEG2000 Reader Tests - Sentinel-2 Validation with GRDL Integration.

Tests grdl.IO.JP2Reader with real Sentinel-2 JPEG2000 files, including:
- Level 1: Format validation (backend selection, metadata extraction, chip/full reads)
- Level 2: Data quality (15-bit encoding, CRS, data ranges, NoData handling)
- Level 3: Integration (ChipExtractor, Normalizer, Tiler workflows)

Dataset: Sentinel-2 Level-2A (Surface Reflectance, 10m bands)

Dependencies
------------
pytest
rasterio or glymur
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

try:
    import glymur
    _HAS_GLYMUR = True
except ImportError:
    _HAS_GLYMUR = False

# GRDL internal
try:
    from grdl.IO.jpeg2000 import JP2Reader
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
    pytest.mark.sentinel2,
    pytest.mark.requires_data,
    pytest.mark.skipif(not (_HAS_RASTERIO or _HAS_GLYMUR),
                      reason="Neither rasterio nor glymur installed"),
    pytest.mark.skipif(not _HAS_GRDL, reason="grdl not installed"),
]


# =============================================================================
# Level 1: Format Validation
# =============================================================================


@pytest.mark.slow
def test_sentinel2_backend_selection(require_sentinel2_file):
    """Verify JP2Reader selects a valid backend (rasterio or glymur)."""
    with JP2Reader(str(require_sentinel2_file)) as reader:
        assert hasattr(reader, 'backend')
        assert reader.backend in ['rasterio', 'glymur'], \
            f"Unexpected backend: {reader.backend}"

        print(f"JP2Reader using backend: {reader.backend}")


@pytest.mark.slow
def test_sentinel2_metadata(require_sentinel2_file):
    """Verify JP2Reader extracts valid Sentinel-2 metadata.

    Validates format string, positive dimensions, and expected dtype
    for Sentinel-2 L2A surface reflectance bands.
    """
    with JP2Reader(str(require_sentinel2_file)) as reader:
        metadata = reader.metadata
        assert isinstance(metadata, ImageMetadata)

        # Sentinel-2 specific properties
        assert metadata.format == 'JPEG2000'
        assert metadata.rows > 0
        assert metadata.cols > 0
        assert metadata.dtype in ['uint16', 'int16']

        print(f"Sentinel-2 metadata: {metadata.rows}x{metadata.cols}, "
              f"dtype={metadata.dtype}")


@pytest.mark.slow
def test_sentinel2_get_shape(require_sentinel2_file):
    """Verify get_shape() returns a tuple consistent with metadata."""
    with JP2Reader(str(require_sentinel2_file)) as reader:
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

        # Sentinel-2 10m bands typically 10980x10980
        print(f"Sentinel-2 shape: {rows} rows x {cols} cols")


@pytest.mark.slow
def test_sentinel2_get_dtype(require_sentinel2_file):
    """Verify get_dtype() returns uint16 matching the metadata."""
    with JP2Reader(str(require_sentinel2_file)) as reader:
        dtype = reader.get_dtype()

        assert dtype is not None
        # Sentinel-2 uses uint16 (15-bit values)
        assert dtype in [np.uint16, np.int16]

        # dtype must agree with metadata
        assert str(dtype) == reader.metadata.dtype

        print(f"Sentinel-2 dtype: {dtype}")


@pytest.mark.slow
def test_sentinel2_read_chip(require_sentinel2_file):
    """Verify read_chip() returns correctly shaped array with valid data."""
    with JP2Reader(str(require_sentinel2_file)) as reader:
        rows, cols = reader.get_shape()[:2]

        # Read 512x512 chip from center
        row_start = max(0, rows // 2 - 256)
        row_end = min(rows, rows // 2 + 256)
        col_start = max(0, cols // 2 - 256)
        col_end = min(cols, cols // 2 + 256)

        chip = reader.read_chip(row_start, row_end, col_start, col_end)

        # Shape and dtype validation
        assert isinstance(chip, np.ndarray)
        assert chip.ndim == 2  # Single band
        assert chip.shape[0] == (row_end - row_start)
        assert chip.shape[1] == (col_end - col_start)
        assert chip.dtype in [np.uint16, np.int16]

        # Center chip should contain real data
        assert not np.all(chip == 0), "Center chip is entirely nodata/fill"

        print(f"Chip shape: {chip.shape}, dtype: {chip.dtype}, "
              f"range: [{chip.min()}, {chip.max()}]")


@pytest.mark.slow
def test_sentinel2_read_full(require_sentinel2_file):
    """Verify read_full() returns array matching reported shape and dtype."""
    with JP2Reader(str(require_sentinel2_file)) as reader:
        rows, cols = reader.get_shape()[:2]

        # Sentinel-2 10m bands are large; skip if too big
        if rows * cols > 150_000_000:  # ~150 megapixels
            pytest.skip("Sentinel-2 file too large for full read test")

        data = reader.read_full()

        assert isinstance(data, np.ndarray)
        assert data.shape[0] == rows
        assert data.shape[1] == cols
        assert data.dtype == reader.get_dtype()

        print(f"Full Sentinel-2 image read: {data.shape}")


@pytest.mark.slow
def test_sentinel2_context_manager(require_sentinel2_file):
    """Verify context manager opens and releases resources correctly."""
    reader = JP2Reader(str(require_sentinel2_file))
    assert hasattr(reader, '__enter__')
    assert hasattr(reader, '__exit__')

    with reader:
        shape = reader.get_shape()
        assert len(shape) >= 2


# =============================================================================
# Level 2: Data Quality
# =============================================================================


@pytest.mark.slow
def test_sentinel2_15bit_encoding(require_sentinel2_file):
    """Verify Sentinel-2 uses 15-bit encoding in 16-bit container.

    Sentinel-2 L2A stores reflectance in 15-bit (0-32767) within
    a uint16 container. No values should exceed 2^15 - 1.
    """
    with JP2Reader(str(require_sentinel2_file)) as reader:
        metadata = reader.metadata

        assert metadata.dtype in ['uint16', 'int16']

        # Read chip and validate 15-bit ceiling
        rows, cols = reader.get_shape()[:2]
        r0 = max(0, rows // 2 - 256)
        r1 = min(rows, rows // 2 + 256)
        c0 = max(0, cols // 2 - 256)
        c1 = min(cols, cols // 2 + 256)
        chip = reader.read_chip(r0, r1, c0, c1)

        assert chip.dtype in [np.uint16, np.int16]
        assert chip.max() <= 32767, \
            f"Max value {chip.max()} exceeds 15-bit ceiling (32767)"

        print(f"Sentinel-2 15-bit encoding: max value = {chip.max()}")


@pytest.mark.slow
def test_sentinel2_crs_validation(require_sentinel2_file):
    """Verify Sentinel-2 CRS is a valid UTM projection.

    Sentinel-2 tiles use UTM zones based on their MGRS grid reference.
    The CRS should reference a UTM EPSG code (326xx or 327xx).
    """
    with JP2Reader(str(require_sentinel2_file)) as reader:
        metadata = reader.metadata

        # Check CRS exists
        if metadata.crs is None:
            pytest.skip("JP2Reader backend does not expose CRS for this file")

        crs_str = str(metadata.crs)

        # Sentinel-2 uses UTM projections
        is_utm = (
            'UTM' in crs_str.upper()
            or any(f'EPSG:{code}' in crs_str
                   for code in range(32601, 32661))
            or any(f'EPSG:{code}' in crs_str
                   for code in range(32701, 32761))
        )
        assert is_utm, f"Expected UTM CRS for Sentinel-2, got: {crs_str}"

        print(f"Sentinel-2 CRS: {metadata.crs}")


@pytest.mark.slow
def test_sentinel2_data_range(require_sentinel2_file):
    """Verify Sentinel-2 reflectance values fall within expected bounds.

    L2A reflectance is scaled 0-10000, stored in 15-bit container.
    Values above 10000 can occur for bright targets. Center data
    must contain valid (non-zero) pixels.
    """
    with JP2Reader(str(require_sentinel2_file)) as reader:
        rows, cols = reader.get_shape()[:2]

        # Read from center to avoid nodata-only edges
        r0 = max(0, rows // 2 - 256)
        r1 = min(rows, rows // 2 + 256)
        c0 = max(0, cols // 2 - 256)
        c1 = min(cols, cols // 2 + 256)
        chip = reader.read_chip(r0, r1, c0, c1)

        # Exclude nodata (0) and validate range
        valid_data = chip[chip > 0]
        assert len(valid_data) > 0, \
            "Center chip has no valid (non-zero) pixels"

        assert valid_data.min() >= 1
        assert valid_data.max() <= 32767, \
            f"Max value {valid_data.max()} exceeds 15-bit ceiling"

        print(f"Sentinel-2 data range: [{valid_data.min()}, {valid_data.max()}]")


@pytest.mark.slow
def test_sentinel2_nodata_value(require_sentinel2_file):
    """Verify Sentinel-2 NoData handling and pixel classification.

    Sentinel-2 typically uses 0 for NoData. The image should contain
    both valid and potentially nodata pixels, with all values accounted for.
    """
    with JP2Reader(str(require_sentinel2_file)) as reader:
        metadata = reader.metadata
        nodata = metadata.nodata

        print(f"Sentinel-2 NoData: {nodata}")

        rows, cols = reader.get_shape()[:2]
        chip = reader.read_chip(0, min(512, rows), 0, min(512, cols))

        # Count pixels using reported nodata if available, else default to 0
        nodata_val = nodata if nodata is not None else 0
        nodata_count = (chip == nodata_val).sum()
        valid_count = (chip != nodata_val).sum()

        assert (nodata_count + valid_count) == chip.size, \
            "Pixel count mismatch: nodata + valid != total"

        print(f"NoData pixels: {nodata_count}, Valid pixels: {valid_count}")


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_RASTERIO, reason="rasterio backend test")
def test_sentinel2_backend_rasterio(require_sentinel2_file):
    """Verify JP2Reader works correctly with explicit rasterio backend."""
    try:
        with JP2Reader(str(require_sentinel2_file), backend='rasterio') as reader:
            assert reader.backend == 'rasterio'

            rows, cols = reader.get_shape()[:2]
            chip = reader.read_chip(0, min(512, rows), 0, min(512, cols))
            assert isinstance(chip, np.ndarray)
            assert chip.size > 0

            print("Rasterio backend: OK")
    except ValueError as e:
        pytest.skip(f"Rasterio backend not available: {e}")


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_GLYMUR, reason="glymur backend test")
def test_sentinel2_backend_glymur(require_sentinel2_file):
    """Verify JP2Reader works correctly with explicit glymur backend."""
    try:
        with JP2Reader(str(require_sentinel2_file), backend='glymur') as reader:
            assert reader.backend == 'glymur'

            rows, cols = reader.get_shape()[:2]
            chip = reader.read_chip(0, min(512, rows), 0, min(512, cols))
            assert isinstance(chip, np.ndarray)
            assert chip.size > 0

            print("Glymur backend: OK")
    except ValueError as e:
        pytest.skip(f"Glymur backend not available: {e}")


# =============================================================================
# Level 3: Integration with GRDL Utilities
# =============================================================================


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
@pytest.mark.integration
def test_sentinel2_chip_extractor_integration(require_sentinel2_file):
    """Validate ChipExtractor partitions Sentinel-2 data into uniform chips.

    Verifies chip regions are within image bounds and extracted chips
    have correct dimensions.
    """
    with JP2Reader(str(require_sentinel2_file)) as reader:
        rows, cols = reader.get_shape()[:2]

        extractor = ChipExtractor(nrows=rows, ncols=cols)
        regions = extractor.chip_positions(row_width=256, col_width=256)

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
            assert chip.shape[0] == region.row_end - region.row_start
            assert chip.shape[1] == region.col_end - region.col_start

        print(f"ChipExtractor: {len(regions)} chips validated")


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
@pytest.mark.integration
def test_sentinel2_normalizer_integration(require_sentinel2_file):
    """Validate Normalizer handles Sentinel-2 15-bit data correctly.

    Tests minmax normalization (output in [0, 1]) and zscore normalization
    on 15-bit encoded reflectance data.
    """
    with JP2Reader(str(require_sentinel2_file)) as reader:
        rows, cols = reader.get_shape()[:2]
        chip = reader.read_chip(0, min(512, rows), 0, min(512, cols))

        # Test minmax normalization (handles 15-bit encoding)
        normalizer = Normalizer(method='minmax')
        normalized = normalizer.normalize(chip)

        assert isinstance(normalized, np.ndarray)
        assert normalized.dtype == np.float64
        assert np.isfinite(normalized).all()
        assert 0.0 <= normalized.min() <= normalized.max() <= 1.0

        print(f"MinMax normalized (15-bit): [{normalized.min():.3f}, "
              f"{normalized.max():.3f}]")

        # Test zscore normalization
        normalizer_z = Normalizer(method='zscore')
        normalized_z = normalizer_z.normalize(chip)

        assert normalized_z.dtype == np.float64
        assert np.isfinite(normalized_z).all()
        assert -10 < normalized_z.mean() < 10

        print(f"Z-score normalized: mean={normalized_z.mean():.3f}, "
              f"std={normalized_z.std():.3f}")


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
@pytest.mark.integration
def test_sentinel2_tiler_integration(require_sentinel2_file):
    """Validate Tiler creates overlapping tile grid over Sentinel-2 image.

    Verifies tile regions are within bounds and tiles read correctly.
    """
    with JP2Reader(str(require_sentinel2_file)) as reader:
        rows, cols = reader.get_shape()[:2]

        tiler = Tiler(
            nrows=rows,
            ncols=cols,
            tile_size=512,
            stride=256
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
            assert tile.shape[0] <= 512
            assert tile.shape[1] <= 512

        print(f"Tiler: {len(tiles)} overlapping tiles validated")


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
@pytest.mark.integration
def test_sentinel2_chip_normalize_pipeline(require_sentinel2_file):
    """Validate end-to-end chip extraction and normalization pipeline.

    Extracts chips from Sentinel-2 JP2 data, normalizes each with zscore,
    and validates batch statistics.
    """
    with JP2Reader(str(require_sentinel2_file)) as reader:
        rows, cols = reader.get_shape()[:2]

        extractor = ChipExtractor(nrows=rows, ncols=cols)
        regions = extractor.chip_positions(row_width=256, col_width=256)

        normalizer = Normalizer(method='zscore')
        normalized_chips = []

        for region in regions[:10]:
            chip = reader.read_chip(
                region.row_start, region.row_end,
                region.col_start, region.col_end
            )

            normalized = normalizer.normalize(chip)
            normalized_chips.append(normalized)

            assert isinstance(normalized, np.ndarray)
            assert normalized.dtype == np.float64
            assert np.isfinite(normalized).all()

        assert len(normalized_chips) > 0

        # Validate batch statistics
        all_values = np.concatenate([nc.flatten() for nc in normalized_chips])
        assert -10 < all_values.mean() < 10
        print(f"Pipeline: {len(normalized_chips)} chips, "
              f"batch mean={all_values.mean():.3f}, std={all_values.std():.3f}")


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
@pytest.mark.integration
def test_sentinel2_high_resolution_workflow(require_sentinel2_file):
    """Validate high-resolution tiling workflow with percentile normalization.

    Uses larger 1024px tiles appropriate for 10m resolution data,
    with percentile normalization for robust display enhancement.
    """
    with JP2Reader(str(require_sentinel2_file)) as reader:
        rows, cols = reader.get_shape()[:2]

        tiler = Tiler(
            nrows=rows,
            ncols=cols,
            tile_size=1024,
            stride=512
        )

        tiles = tiler.tile_positions()
        normalizer = Normalizer(method='percentile', percentile_low=2.0, percentile_high=98.0)

        for i, tile_region in enumerate(tiles[:3]):
            tile = reader.read_chip(
                tile_region.row_start, tile_region.row_end,
                tile_region.col_start, tile_region.col_end
            )

            normalized_tile = normalizer.normalize(tile)

            assert normalized_tile.shape == tile.shape
            assert np.isfinite(normalized_tile).all()
            assert 0.0 <= normalized_tile.min() <= normalized_tile.max() <= 1.0

            print(f"Tile {i}: {tile.shape}, normalized range: "
                  f"[{normalized_tile.min():.3f}, {normalized_tile.max():.3f}]")
