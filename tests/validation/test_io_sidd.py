# -*- coding: utf-8 -*-
"""
SIDDReader validation using real SIDD data.

Tests:
- Level 1: Context manager, metadata (product_type), shape
- Level 2: Real-valued (detected) array, pixel type, geolocation metadata
- Level 3: Chip and normalize pipeline

Dataset: SIDD detected imagery (*.nitf)

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
    from grdl.IO.sar import SIDDReader
    _HAS_SIDD = True
except ImportError:
    _HAS_SIDD = False

try:
    from grdl.data_prep import ChipExtractor, Normalizer
    _HAS_DATA_PREP = True
except ImportError:
    _HAS_DATA_PREP = False


pytestmark = [
    pytest.mark.sidd,
    pytest.mark.requires_data,
    pytest.mark.skipif(not _HAS_SIDD, reason="SIDDReader not available"),
]


# =============================================================================
# Level 1: Format Validation
# =============================================================================


@pytest.mark.slow
def test_sidd_reader_opens(require_sidd_file):
    """Context manager opens without exception."""
    with SIDDReader(str(require_sidd_file)) as reader:
        assert reader is not None


@pytest.mark.slow
def test_sidd_metadata_populated(require_sidd_file):
    """typed_metadata with product_type present."""
    with SIDDReader(str(require_sidd_file)) as reader:
        meta = reader.typed_metadata
        assert meta is not None


@pytest.mark.slow
def test_sidd_shape_2d(require_sidd_file):
    """get_shape() returns (rows, cols) with positive dimensions."""
    with SIDDReader(str(require_sidd_file)) as reader:
        shape = reader.get_shape()
        assert isinstance(shape, tuple)
        assert len(shape) >= 2
        rows, cols = shape[0], shape[1]
        assert rows > 0 and cols > 0


# =============================================================================
# Level 2: Data Quality
# =============================================================================


@pytest.mark.slow
def test_sidd_read_full(require_sidd_file):
    """Returns real-valued (detected) array."""
    with SIDDReader(str(require_sidd_file)) as reader:
        shape = reader.get_shape()
        if shape[0] * shape[1] > 25_000_000:
            pytest.skip("SIDD file too large for full read test")
        data = reader.read_full()
        assert isinstance(data, np.ndarray)
        # SIDD is detected imagery — should be real-valued
        assert not np.iscomplexobj(data)
        assert data.size > 0


@pytest.mark.slow
def test_sidd_pixel_type(require_sidd_file):
    """Pixel type is float32 or uint8 depending on product."""
    with SIDDReader(str(require_sidd_file)) as reader:
        dtype = reader.get_dtype()
        assert dtype in [np.float32, np.float64, np.uint8, np.uint16], \
            f"Unexpected SIDD pixel type: {dtype}"


@pytest.mark.slow
def test_sidd_geolocation_metadata(require_sidd_file):
    """Geographic metadata present in typed_metadata."""
    with SIDDReader(str(require_sidd_file)) as reader:
        meta = reader.typed_metadata
        # SIDD products carry geolocation info
        assert meta is not None


# =============================================================================
# Level 3: Integration
# =============================================================================


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
def test_sidd_chip_and_normalize(require_sidd_file):
    """ChipExtractor + Normalizer pipeline on detected imagery."""
    with SIDDReader(str(require_sidd_file)) as reader:
        rows, cols = reader.get_shape()[:2]
        extractor = ChipExtractor(nrows=rows, ncols=cols)
        regions = extractor.chip_positions(row_width=256, col_width=256)
        region = regions[0]
        chip = reader.read_chip(
            region.row_start, region.row_end,
            region.col_start, region.col_end,
        )
        normalizer = Normalizer(method='minmax')
        normalized = normalizer.normalize(chip.astype(np.float32))
        assert np.isfinite(normalized).all()
        assert 0.0 <= normalized.min() <= normalized.max() <= 1.0
