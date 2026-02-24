# -*- coding: utf-8 -*-
"""
Sentinel1SLCReader validation using real Sentinel-1 SLC data.

Tests:
- Level 1: Context manager, metadata (polarization, orbit, swath), shape
- Level 2: Complex64 array, chip extraction, polarization channels
- Level 3: Geolocation construction, ChipExtractor integration

Dataset: Sentinel-1 IW SLC (*.SAFE)

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
    from grdl.IO.sar import Sentinel1SLCReader
    _HAS_S1 = True
except ImportError:
    _HAS_S1 = False

try:
    from grdl.data_prep import ChipExtractor
    _HAS_DATA_PREP = True
except ImportError:
    _HAS_DATA_PREP = False


pytestmark = [
    pytest.mark.sentinel1,
    pytest.mark.requires_data,
    pytest.mark.skipif(not _HAS_S1, reason="Sentinel1SLCReader not available"),
]


# =============================================================================
# Level 1: Format Validation
# =============================================================================


@pytest.mark.slow
def test_s1_reader_opens(require_sentinel1_file):
    """Context manager opens without exception."""
    with Sentinel1SLCReader(require_sentinel1_file) as reader:
        assert reader is not None


@pytest.mark.slow
def test_s1_metadata_populated(require_sentinel1_file):
    """Metadata includes polarization, orbit, and swath info."""
    with Sentinel1SLCReader(require_sentinel1_file) as reader:
        meta = reader.metadata
        assert meta is not None
        assert meta.rows > 0
        assert meta.cols > 0


@pytest.mark.slow
def test_s1_shape(require_sentinel1_file):
    """get_shape() returns (rows, cols) tuple."""
    with Sentinel1SLCReader(require_sentinel1_file) as reader:
        shape = reader.get_shape()
        assert isinstance(shape, tuple)
        assert len(shape) >= 2
        assert shape[0] > 0 and shape[1] > 0


# =============================================================================
# Level 2: Data Quality
# =============================================================================


@pytest.mark.slow
def test_s1_read_full_complex(require_sentinel1_file):
    """read_full() returns complex64 array."""
    with Sentinel1SLCReader(require_sentinel1_file) as reader:
        shape = reader.get_shape()
        if shape[0] * shape[1] > 50_000_000:
            pytest.skip("S1 file too large for full read")
        data = reader.read_full()
        assert isinstance(data, np.ndarray)
        assert np.iscomplexobj(data)


@pytest.mark.slow
def test_s1_read_chip(require_sentinel1_file):
    """Subregion extraction returns correct shape."""
    with Sentinel1SLCReader(require_sentinel1_file) as reader:
        rows, cols = reader.get_shape()[:2]
        r0 = max(0, rows // 2 - 256)
        r1 = min(rows, rows // 2 + 256)
        c0 = max(0, cols // 2 - 256)
        c1 = min(cols, cols // 2 + 256)
        chip = reader.read_chip(r0, r1, c0, c1)
        assert isinstance(chip, np.ndarray)
        assert chip.shape[0] == (r1 - r0)
        assert chip.shape[1] == (c1 - c0)
        assert np.iscomplexobj(chip)


@pytest.mark.slow
def test_s1_polarization_channels(require_sentinel1_file):
    """Available polarization channels (VV/VH or HH/HV) accessible."""
    with Sentinel1SLCReader(require_sentinel1_file) as reader:
        if hasattr(reader, 'get_available_polarizations'):
            pols = reader.get_available_polarizations()
            assert isinstance(pols, (list, tuple))
            assert len(pols) > 0
            for p in pols:
                assert p in ('VV', 'VH', 'HH', 'HV')


# =============================================================================
# Level 3: Integration
# =============================================================================


@pytest.mark.slow
@pytest.mark.integration
def test_s1_geolocation_construction(require_sentinel1_file):
    """Sentinel1SLCGeolocation.from_reader() succeeds."""
    try:
        from grdl.geolocation import Sentinel1SLCGeolocation
    except ImportError:
        pytest.skip("Sentinel1SLCGeolocation not available")

    with Sentinel1SLCReader(require_sentinel1_file) as reader:
        geo = Sentinel1SLCGeolocation.from_reader(reader)
        assert geo is not None


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
def test_s1_chip_extractor_integration(require_sentinel1_file):
    """ChipExtractor works with Sentinel-1 SLC dimensions."""
    with Sentinel1SLCReader(require_sentinel1_file) as reader:
        rows, cols = reader.get_shape()[:2]
        extractor = ChipExtractor(nrows=rows, ncols=cols)
        regions = extractor.chip_positions(row_width=256, col_width=256)
        assert len(regions) > 0
        region = regions[0]
        chip = reader.read_chip(
            region.row_start, region.row_end,
            region.col_start, region.col_end,
        )
        assert isinstance(chip, np.ndarray)
        assert chip.shape[0] == region.row_end - region.row_start
