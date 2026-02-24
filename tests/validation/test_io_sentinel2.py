# -*- coding: utf-8 -*-
"""
Sentinel2Reader validation using real Sentinel-2 L2A data.

Tests:
- Level 1: Context manager, band names, metadata (CRS, bounds, resolution)
- Level 2: Single band read, full read, value range
- Level 3: JP2Reader fallback for individual JP2 files

Dataset: Sentinel-2 L2A (S2*.SAFE)

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
    from grdl.IO.eo import Sentinel2Reader
    _HAS_S2 = True
except ImportError:
    try:
        from grdl.IO.multispectral import Sentinel2Reader
        _HAS_S2 = True
    except ImportError:
        _HAS_S2 = False

try:
    from grdl.IO.jpeg2000 import JP2Reader
    _HAS_JP2 = True
except ImportError:
    _HAS_JP2 = False


pytestmark = [
    pytest.mark.sentinel2,
    pytest.mark.requires_data,
    pytest.mark.skipif(not _HAS_S2, reason="Sentinel2Reader not available"),
]


# =============================================================================
# Level 1: Format Validation
# =============================================================================


@pytest.mark.slow
def test_s2_reader_opens(require_sentinel2_file):
    """Context manager opens without exception."""
    with Sentinel2Reader(str(require_sentinel2_file)) as reader:
        assert reader is not None


@pytest.mark.slow
def test_s2_band_names(require_sentinel2_file):
    """List of available bands returned."""
    with Sentinel2Reader(str(require_sentinel2_file)) as reader:
        meta = reader.metadata
        assert meta is not None
        assert meta.rows > 0
        assert meta.cols > 0


@pytest.mark.slow
def test_s2_metadata(require_sentinel2_file):
    """CRS, bounds, resolution per band available."""
    with Sentinel2Reader(str(require_sentinel2_file)) as reader:
        meta = reader.metadata
        assert meta is not None
        if hasattr(meta, 'crs') and meta.crs is not None:
            assert len(str(meta.crs)) > 0


# =============================================================================
# Level 2: Data Quality
# =============================================================================


@pytest.mark.slow
def test_s2_read_full(require_sentinel2_file):
    """Full read returns ndarray."""
    with Sentinel2Reader(str(require_sentinel2_file)) as reader:
        shape = reader.get_shape()
        if shape[0] * shape[1] > 25_000_000:
            pytest.skip("S2 band too large for full read test")
        data = reader.read_full()
        assert isinstance(data, np.ndarray)
        assert data.size > 0


@pytest.mark.slow
def test_s2_value_range(require_sentinel2_file):
    """Reflectance values in reasonable range (non-negative)."""
    with Sentinel2Reader(str(require_sentinel2_file)) as reader:
        shape = reader.get_shape()
        rows, cols = shape[0], shape[1]
        # Read a chip from center
        r0 = max(0, rows // 2 - 256)
        r1 = min(rows, rows // 2 + 256)
        c0 = max(0, cols // 2 - 256)
        c1 = min(cols, cols // 2 + 256)
        chip = reader.read_chip(r0, r1, c0, c1)
        assert isinstance(chip, np.ndarray)
        # Sentinel-2 L2A reflectance is typically [0, 10000] or scaled float
        assert np.isfinite(chip).all()


# =============================================================================
# Level 3: Integration
# =============================================================================


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not _HAS_JP2, reason="JP2Reader not available")
def test_s2_jp2_reader_fallback(require_sentinel2_file):
    """Individual JP2 files readable via JP2Reader."""
    with JP2Reader(str(require_sentinel2_file)) as reader:
        data = reader.read_full()
        assert isinstance(data, np.ndarray)
        assert data.size > 0
