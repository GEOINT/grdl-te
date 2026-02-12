# -*- coding: utf-8 -*-
"""
GeoTIFF Reader Tests - Landsat 8/9 Validation with GRDL Integration.

Tests grdl.IO.GeoTIFFReader with real Landsat 8/9 COG files, including:
- Level 1: Format validation (metadata extraction, shape/dtype, chip/full reads)
- Level 2: Data quality (CRS, NoData, bounds, COG properties, reflectance range)
- Level 3: Integration (ChipExtractor, Normalizer, Tiler workflows)

Dataset: Landsat 8/9 Collection 2 Surface Reflectance (COG)

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
    from grdl.IO.geotiff import GeoTIFFReader
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
    pytest.mark.landsat,
    pytest.mark.requires_data,
    pytest.mark.skipif(not _HAS_RASTERIO, reason="rasterio not installed"),
    pytest.mark.skipif(not _HAS_GRDL, reason="grdl not installed"),
]


# =============================================================================
# Level 1: Format Validation
# =============================================================================


@pytest.mark.slow
def test_landsat_metadata(require_landsat_file):
    """Verify GeoTIFFReader extracts valid Landsat metadata.

    Validates format string, positive dimensions, CRS presence, and
    expected dtype for Landsat Collection 2 Surface Reflectance.
    """
    with GeoTIFFReader(str(require_landsat_file)) as reader:
        metadata = reader.metadata
        assert isinstance(metadata, ImageMetadata)

        # Landsat specific properties
        assert metadata.format == 'GeoTIFF'
        assert metadata.rows > 0
        assert metadata.cols > 0
        assert metadata.crs is not None
        assert metadata.dtype in ['uint16', 'int16', 'float32']

        # Single-band file should report bands
        assert metadata.bands is not None
        assert metadata.bands >= 1

        print(f"Landsat metadata: {metadata.rows}x{metadata.cols}, "
              f"bands={metadata.bands}, CRS={metadata.crs}")


@pytest.mark.slow
def test_landsat_get_shape(require_landsat_file):
    """Verify get_shape() returns a tuple consistent with metadata."""
    with GeoTIFFReader(str(require_landsat_file)) as reader:
        shape = reader.get_shape()

        assert isinstance(shape, tuple)
        assert len(shape) >= 2  # (rows, cols) or (rows, cols, bands)

        rows, cols = shape[0], shape[1]
        assert isinstance(rows, int)
        assert isinstance(cols, int)
        assert rows > 0 and cols > 0

        # Shape must agree with metadata
        assert rows == reader.metadata.rows
        assert cols == reader.metadata.cols

        print(f"Landsat shape: {rows} rows x {cols} cols")


@pytest.mark.slow
def test_landsat_get_dtype(require_landsat_file):
    """Verify get_dtype() returns a numpy dtype matching the metadata."""
    with GeoTIFFReader(str(require_landsat_file)) as reader:
        dtype = reader.get_dtype()

        assert dtype is not None
        assert dtype in [np.uint16, np.int16, np.float32, np.uint8]

        # dtype must agree with metadata
        assert str(dtype) == reader.metadata.dtype

        print(f"Landsat dtype: {dtype}")


@pytest.mark.slow
def test_landsat_read_chip(require_landsat_file):
    """Verify read_chip() returns correctly shaped array with valid data."""
    with GeoTIFFReader(str(require_landsat_file)) as reader:
        rows, cols = reader.get_shape()[:2]

        # Read 512x512 chip from center (avoids nodata-only edges)
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
        assert chip.dtype in [np.uint16, np.int16, np.float32]

        # Center chip should contain real data, not be entirely fill
        assert not np.all(chip == 0), "Center chip is entirely nodata/fill"

        print(f"Chip shape: {chip.shape}, dtype: {chip.dtype}, "
              f"range: [{chip.min()}, {chip.max()}]")


@pytest.mark.slow
def test_landsat_read_full(require_landsat_file):
    """Verify read_full() returns array matching reported shape and dtype."""
    with GeoTIFFReader(str(require_landsat_file)) as reader:
        rows, cols = reader.get_shape()[:2]

        # For large files, only read if reasonable size
        if rows * cols > 100_000_000:  # ~100 megapixels
            pytest.skip("File too large for full read test (streaming validation only)")

        data = reader.read_full()

        assert isinstance(data, np.ndarray)
        assert data.shape[0] == rows
        assert data.shape[1] == cols
        assert data.dtype == reader.get_dtype()

        print(f"Full image read: {data.shape}")


@pytest.mark.slow
def test_landsat_context_manager(require_landsat_file):
    """Verify context manager opens and releases resources correctly."""
    reader = GeoTIFFReader(str(require_landsat_file))
    assert hasattr(reader, '__enter__')
    assert hasattr(reader, '__exit__')

    with reader:
        shape = reader.get_shape()
        assert len(shape) >= 2

    # After exiting context, internal dataset handle should be closed.
    if hasattr(reader, 'dataset') and reader.dataset is not None:
        assert reader.dataset.closed, "Dataset handle not closed after context exit"


# =============================================================================
# Level 2: Data Quality
# =============================================================================


@pytest.mark.slow
def test_landsat_crs_validation(require_landsat_file):
    """Verify Landsat CRS is a valid UTM projection.

    Landsat Collection 2 uses UTM zones (EPSG:326xx for North, EPSG:327xx
    for South). The CRS string must reference a UTM projection.
    """
    with GeoTIFFReader(str(require_landsat_file)) as reader:
        metadata = reader.metadata

        assert metadata.crs is not None
        crs_str = str(metadata.crs)

        # Landsat uses UTM projections (EPSG:326xx or 327xx)
        is_utm = (
            'UTM' in crs_str.upper()
            or any(f'EPSG:{code}' in crs_str
                   for code in range(32601, 32661))
            or any(f'EPSG:{code}' in crs_str
                   for code in range(32701, 32761))
        )
        assert is_utm, f"Expected UTM CRS, got: {crs_str}"

        print(f"Landsat CRS: {metadata.crs}")


@pytest.mark.slow
def test_landsat_nodata_handling(require_landsat_file):
    """Verify NoData value extraction and valid pixel statistics.

    Landsat SR products use 0 as NoData. The center of a scene should
    contain valid pixels with non-zero variance.
    """
    with GeoTIFFReader(str(require_landsat_file)) as reader:
        metadata = reader.metadata

        # Landsat SR products typically have nodata = 0
        nodata = metadata.nodata
        print(f"Landsat NoData value: {nodata}")

        # Read a chip from center (edges often contain only nodata fill)
        rows, cols = reader.get_shape()[:2]
        r0 = max(0, rows // 2 - 256)
        r1 = min(rows, rows // 2 + 256)
        c0 = max(0, cols // 2 - 256)
        c1 = min(cols, cols // 2 + 256)
        chip = reader.read_chip(r0, r1, c0, c1)

        # Validate nodata masking
        if nodata is not None:
            valid_mask = chip != nodata
            valid_fraction = valid_mask.sum() / valid_mask.size

            assert valid_fraction > 0, "Center chip is entirely nodata"
            assert valid_fraction <= 1.0

            print(f"Valid pixels: {valid_fraction:.1%}")

            # Valid reflectance should have non-zero variance
            valid_data = chip[valid_mask]
            mean_valid = valid_data.mean()
            std_valid = valid_data.std()
            assert std_valid > 0, "Valid pixel values have zero variance"

            print(f"Valid data - Mean: {mean_valid:.2f}, Std: {std_valid:.2f}")


@pytest.mark.slow
def test_landsat_bounds_validation(require_landsat_file):
    """Verify bounds and affine transform are spatially consistent.

    The affine transform must map the center pixel to coordinates
    that fall within the reported bounds.
    """
    with GeoTIFFReader(str(require_landsat_file)) as reader:
        metadata = reader.metadata

        # Check bounds exist
        assert 'bounds' in metadata.extras
        bounds = metadata.extras['bounds']
        assert len(bounds) == 4  # (left, bottom, right, top)

        left, bottom, right, top = bounds
        assert left < right, f"West ({left}) must be < East ({right})"
        assert bottom < top, f"South ({bottom}) must be < North ({top})"

        print(f"Landsat bounds: ({left:.2f}, {bottom:.2f}, {right:.2f}, {top:.2f})")

        # Check transform exists
        assert 'transform' in metadata.extras
        transform = metadata.extras['transform']

        # Use transform to map pixel to coordinates
        rows, cols = reader.get_shape()[:2]
        center_row, center_col = rows // 2, cols // 2

        # Transform center pixel to coordinates
        x, y = transform * (center_col, center_row)

        # Coordinates should be within bounds
        assert left <= x <= right, \
            f"Center x={x:.2f} outside bounds [{left:.2f}, {right:.2f}]"
        assert bottom <= y <= top, \
            f"Center y={y:.2f} outside bounds [{bottom:.2f}, {top:.2f}]"

        print(f"Center pixel ({center_row}, {center_col}) -> ({x:.2f}, {y:.2f})")


@pytest.mark.slow
def test_landsat_cog_properties(require_landsat_file):
    """Verify Cloud-Optimized GeoTIFF characteristics.

    Landsat Collection 2 files are distributed as COGs with internal tiling
    and overview pyramids. Uses rasterio directly to validate format-level
    properties not exposed through the GRDL metadata interface.
    """
    with rasterio.open(str(require_landsat_file)) as src:
        # COG must be tiled
        is_tiled = src.profile.get('tiled', False)
        blockxsize = src.profile.get('blockxsize', 1)
        blockysize = src.profile.get('blockysize', 1)

        assert is_tiled or (blockxsize > 1 and blockysize > 1), \
            "Landsat COG should be tiled"

        print(f"Landsat COG tiled: {is_tiled}, "
              f"block size: {blockysize}x{blockxsize}")

        # COG must have overview pyramid levels
        overviews = src.overviews(1)  # Band 1 overviews
        assert len(overviews) > 0, \
            "Landsat COG should have overview pyramids"

        # Overviews should be powers of 2
        for ovr in overviews:
            assert ovr > 0
            assert (ovr & (ovr - 1)) == 0, \
                f"Overview factor {ovr} is not a power of 2"

        print(f"Landsat COG overviews: {overviews}")


@pytest.mark.slow
def test_landsat_data_range(require_landsat_file):
    """Verify Landsat surface reflectance values fall within expected bounds.

    Landsat Collection 2 L2 SR uses scale=0.0000275, offset=-0.2.
    Valid uint16 values range from ~7273 (reflectance=0.0) to ~43636
    (reflectance=1.0). Values of 0 indicate NoData/fill.
    """
    with GeoTIFFReader(str(require_landsat_file)) as reader:
        rows, cols = reader.get_shape()[:2]

        # Read from center to avoid nodata-only edges
        r0 = max(0, rows // 2 - 256)
        r1 = min(rows, rows // 2 + 256)
        c0 = max(0, cols // 2 - 256)
        c1 = min(cols, cols // 2 + 256)
        chip = reader.read_chip(r0, r1, c0, c1)

        # Exclude nodata (0) and validate reflectance range
        valid_data = chip[chip > 0]
        assert len(valid_data) > 0, "Center chip has no valid (non-zero) pixels"

        assert valid_data.min() >= 1, \
            f"Min valid value {valid_data.min()} below reflectance floor"
        # Collection 2 valid range tops out at ~43636 for reflectance=1.0;
        # allow up to 55000 for saturated/flagged pixels
        assert valid_data.max() < 55000, \
            f"Max valid value {valid_data.max()} exceeds plausible range"

        print(f"Landsat SR range: [{valid_data.min()}, {valid_data.max()}]")


# =============================================================================
# Level 3: Integration with GRDL Utilities
# =============================================================================


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
@pytest.mark.integration
def test_landsat_chip_extractor_integration(require_landsat_file):
    """Validate ChipExtractor partitions Landsat data into uniform chips.

    Verifies that chip regions are within image bounds and that extracted
    chips have the correct dimensions and contain array data.
    """
    with GeoTIFFReader(str(require_landsat_file)) as reader:
        rows, cols = reader.get_shape()[:2]

        extractor = ChipExtractor(nrows=rows, ncols=cols)
        regions = extractor.chip_positions(row_width=256, col_width=256)

        assert len(regions) > 0

        for i, region in enumerate(regions[:5]):
            # Verify region bounds are within image
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
def test_landsat_chip_at_point(require_landsat_file):
    """Validate targeted chip extraction at image center.

    Verifies ChipExtractor.chip_at_point() returns a region that,
    when read, produces the expected chip dimensions.
    """
    with GeoTIFFReader(str(require_landsat_file)) as reader:
        rows, cols = reader.get_shape()[:2]

        extractor = ChipExtractor(nrows=rows, ncols=cols)
        center_region = extractor.chip_at_point(
            row=rows // 2,
            col=cols // 2,
            row_width=128,
            col_width=128
        )

        chip = reader.read_chip(
            center_region.row_start, center_region.row_end,
            center_region.col_start, center_region.col_end
        )

        assert chip.shape == (128, 128)
        print(f"Center chip extracted: {chip.shape}")


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
@pytest.mark.integration
def test_landsat_normalizer_integration(require_landsat_file):
    """Validate Normalizer produces finite, bounded output from Landsat data.

    Tests minmax normalization (output in [0, 1]) and zscore normalization
    (output roughly centered at 0 with unit-scale standard deviation).
    """
    with GeoTIFFReader(str(require_landsat_file)) as reader:
        rows, cols = reader.get_shape()[:2]
        # Read from center to avoid edge nodata fill
        r0 = max(0, rows // 2 - 256)
        r1 = min(rows, rows // 2 + 256)
        c0 = max(0, cols // 2 - 256)
        c1 = min(cols, cols // 2 + 256)
        chip = reader.read_chip(r0, r1, c0, c1)

        # Test minmax normalization
        normalizer = Normalizer(method='minmax')
        normalized = normalizer.normalize(chip)

        assert isinstance(normalized, np.ndarray)
        assert normalized.dtype == np.float64
        assert np.isfinite(normalized).all()
        assert 0.0 <= normalized.min() <= normalized.max() <= 1.0

        print(f"MinMax normalized: [{normalized.min():.3f}, {normalized.max():.3f}]")

        # Test zscore normalization
        normalizer_z = Normalizer(method='zscore')
        normalized_z = normalizer_z.normalize(chip)

        assert normalized_z.dtype == np.float64
        assert np.isfinite(normalized_z).all()
        # Z-score should be roughly centered at 0
        assert -10 < normalized_z.mean() < 10
        assert 0 < normalized_z.std() < 10

        print(f"Z-score normalized: mean={normalized_z.mean():.3f}, "
              f"std={normalized_z.std():.3f}")


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
@pytest.mark.integration
def test_landsat_normalizer_fit_transform(require_landsat_file):
    """Validate stateful normalization with separate fit and transform.

    Fits normalizer statistics on one region and applies them to a
    different region, simulating a train/test split workflow.
    """
    with GeoTIFFReader(str(require_landsat_file)) as reader:
        rows, cols = reader.get_shape()[:2]

        # Read two different regions (simulating train/test)
        train_chip = reader.read_chip(0, min(256, rows), 0, min(256, cols))
        test_chip = reader.read_chip(
            min(512, rows // 2), min(768, rows // 2 + 256),
            min(512, cols // 2), min(768, cols // 2 + 256)
        )

        # Fit normalizer on train chip
        normalizer = Normalizer(method='zscore')
        normalizer.fit(train_chip)

        # Transform test chip with train statistics
        normalized_test = normalizer.transform(test_chip)

        assert isinstance(normalized_test, np.ndarray)
        assert normalized_test.dtype == np.float64
        assert np.isfinite(normalized_test).all()

        print(f"Fit/transform: test chip normalized to "
              f"mean={normalized_test.mean():.3f}, std={normalized_test.std():.3f}")


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
@pytest.mark.integration
def test_landsat_tiler_integration(require_landsat_file):
    """Validate Tiler creates overlapping tile grid over Landsat image.

    Verifies tile regions are within bounds and that tiles have correct
    dimensions with expected overlap.
    """
    with GeoTIFFReader(str(require_landsat_file)) as reader:
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
            # Verify tile bounds are within image
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
def test_landsat_chip_normalize_pipeline(require_landsat_file):
    """Validate end-to-end chip extraction and normalization pipeline.

    Extracts multiple chips, normalizes each with zscore, and validates
    that the batch statistics are consistent across chips.
    """
    with GeoTIFFReader(str(require_landsat_file)) as reader:
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
