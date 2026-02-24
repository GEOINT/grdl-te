# -*- coding: utf-8 -*-
"""
TerraSARReader validation using real TerraSAR-X/TanDEM-X data.

Tests:
- Level 1: Context manager, TerraSARMetadata, shape, dtype
- Level 2: Chip extraction, finite values, polarizations, metadata fields
- Level 3: open_sar routing, ChipExtractor integration

Dataset: TerraSAR-X SSC/MGD products (TSX1_SAR__* or TDX1_SAR__*)

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
    from grdl.IO.sar import TerraSARReader
    _HAS_TSX = True
except ImportError:
    _HAS_TSX = False

try:
    from grdl.IO.sar import open_sar
    _HAS_OPEN_SAR = True
except ImportError:
    _HAS_OPEN_SAR = False

try:
    from grdl.data_prep import ChipExtractor
    _HAS_DATA_PREP = True
except ImportError:
    _HAS_DATA_PREP = False


pytestmark = [
    pytest.mark.terrasar,
    pytest.mark.requires_data,
    pytest.mark.skipif(not _HAS_TSX, reason="TerraSARReader not available"),
]


# =============================================================================
# Level 1: Format Validation
# =============================================================================


@pytest.mark.slow
def test_tsx_reader_opens(require_terrasar_dir):
    """Context manager opens without exception."""
    with TerraSARReader(require_terrasar_dir) as reader:
        assert reader is not None


@pytest.mark.slow
def test_tsx_metadata_populated(require_terrasar_dir):
    """TerraSARMetadata is not None."""
    with TerraSARReader(require_terrasar_dir) as reader:
        meta = reader.typed_metadata
        assert meta is not None


@pytest.mark.slow
def test_tsx_shape(require_terrasar_dir):
    """get_shape() returns (rows, cols) tuple."""
    with TerraSARReader(require_terrasar_dir) as reader:
        shape = reader.get_shape()
        assert isinstance(shape, tuple)
        assert len(shape) >= 2
        assert shape[0] > 0 and shape[1] > 0


@pytest.mark.slow
def test_tsx_dtype(require_terrasar_dir):
    """get_dtype() returns complex64 (SSC) or float32 (detected)."""
    with TerraSARReader(require_terrasar_dir) as reader:
        dtype = reader.get_dtype()
        assert dtype in [np.complex64, np.complex128, np.float32, np.float64, np.int16]


# =============================================================================
# Level 2: Data Quality
# =============================================================================


@pytest.mark.slow
def test_tsx_read_chip(require_terrasar_dir):
    """Subregion extraction returns correct shape."""
    with TerraSARReader(require_terrasar_dir) as reader:
        rows, cols = reader.get_shape()[:2]
        r0 = max(0, rows // 2 - 256)
        r1 = min(rows, rows // 2 + 256)
        c0 = max(0, cols // 2 - 256)
        c1 = min(cols, cols // 2 + 256)
        chip = reader.read_chip(r0, r1, c0, c1)
        assert isinstance(chip, np.ndarray)
        assert chip.shape[0] == (r1 - r0)
        assert chip.shape[1] == (c1 - c0)


@pytest.mark.slow
def test_tsx_values_finite(require_terrasar_dir):
    """No NaN/Inf in chip data."""
    with TerraSARReader(require_terrasar_dir) as reader:
        rows, cols = reader.get_shape()[:2]
        r0 = max(0, rows // 2 - 128)
        r1 = min(rows, rows // 2 + 128)
        c0 = max(0, cols // 2 - 128)
        c1 = min(cols, cols // 2 + 128)
        chip = reader.read_chip(r0, r1, c0, c1)
        assert np.all(np.isfinite(chip))


@pytest.mark.slow
def test_tsx_available_polarizations(require_terrasar_dir):
    """get_available_polarizations() returns list."""
    with TerraSARReader(require_terrasar_dir) as reader:
        if hasattr(reader, 'get_available_polarizations'):
            pols = reader.get_available_polarizations()
            assert isinstance(pols, (list, tuple))
            assert len(pols) > 0


@pytest.mark.slow
def test_tsx_metadata_fields(require_terrasar_dir):
    """product_info, scene_info, radar_params populated."""
    with TerraSARReader(require_terrasar_dir) as reader:
        meta = reader.typed_metadata
        assert meta is not None
        # TerraSARMetadata has nested dataclasses
        if hasattr(meta, 'product_info'):
            assert meta.product_info is not None
        if hasattr(meta, 'scene_info'):
            assert meta.scene_info is not None
        if hasattr(meta, 'radar_params'):
            assert meta.radar_params is not None


# =============================================================================
# Level 3: Integration
# =============================================================================


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not _HAS_OPEN_SAR, reason="open_sar not available")
def test_tsx_open_sar_routing(require_terrasar_dir):
    """open_sar(tsx_path) returns TerraSARReader instance."""
    reader = open_sar(require_terrasar_dir)
    assert isinstance(reader, TerraSARReader)
    with reader:
        shape = reader.get_shape()
        assert shape[0] > 0


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
def test_tsx_chip_extractor_integration(require_terrasar_dir):
    """ChipExtractor works with TSX shape."""
    with TerraSARReader(require_terrasar_dir) as reader:
        rows, cols = reader.get_shape()[:2]
        extractor = ChipExtractor(nrows=rows, ncols=cols)
        regions = extractor.chip_positions(row_width=256, col_width=256)
        assert len(regions) > 0
        region = regions[0]
        chip = reader.read_chip(
            region.row_start, region.row_end,
            region.col_start, region.col_end,
        )
        assert chip.shape[0] == region.row_end - region.row_start
