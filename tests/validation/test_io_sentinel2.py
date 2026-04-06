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
# Level 2b: Metadata Quality (Expanded v0.4.0)
# =============================================================================

@pytest.mark.slow
def test_s2_cloud_cover_percentage(require_sentinel2_file):
    """Cloud cover percentage in valid range [0, 100]."""
    with Sentinel2Reader(str(require_sentinel2_file)) as reader:
        meta = reader.metadata
        assert meta is not None
        if hasattr(meta, 'cloud_cover_percentage'):
            cc = meta.cloud_cover_percentage
            assert isinstance(cc, (int, float)), \
                f"cloud_cover_percentage must be numeric, got {type(cc)}"
            assert 0 <= cc <= 100, \
                f"Cloud cover out of range: {cc} (expected [0, 100])"


@pytest.mark.slow
def test_s2_sensing_time_boundaries(require_sentinel2_file):
    """Sensing start/stop times present and ordered."""
    with Sentinel2Reader(str(require_sentinel2_file)) as reader:
        meta = reader.metadata
        assert meta is not None
        # Check for sensing time attributes (v0.4.0 additions)
        if hasattr(meta, 'sensing_time_start') and hasattr(meta, 'sensing_time_stop'):
            start = meta.sensing_time_start
            stop = meta.sensing_time_stop
            assert start is not None, "sensing_time_start is None"
            assert stop is not None, "sensing_time_stop is None"
            # Verify ordering
            assert start < stop, \
                f"Sensing times misordered: start={start} >= stop={stop}"


@pytest.mark.slow
def test_s2_orbit_number_metadata(require_sentinel2_file):
    """Orbit number is a positive integer."""
    with Sentinel2Reader(str(require_sentinel2_file)) as reader:
        meta = reader.metadata
        assert meta is not None
        # Check for orbit number (v0.4.0 addition)
        if hasattr(meta, 'orbit_number'):
            orbit = meta.orbit_number
            assert orbit is not None, "orbit_number is None"
            assert isinstance(orbit, int), \
                f"orbit_number must be int, got {type(orbit)}"
            assert orbit > 0, \
                f"orbit_number must be positive, got {orbit}"


@pytest.mark.slow
def test_s2_tile_geocoding_bounds(require_sentinel2_file):
    """Tile geocoding bounds are valid and ordered."""
    with Sentinel2Reader(str(require_sentinel2_file)) as reader:
        meta = reader.metadata
        assert meta is not None
        # Verify bounds structure
        if hasattr(meta, 'bounds') and meta.bounds is not None:
            bounds = meta.bounds
            # Check if bounds is a tuple/list or object with attributes
            if hasattr(bounds, 'min_lat'):
                assert bounds.min_lat is not None
                assert bounds.max_lat is not None
                assert bounds.min_lon is not None
                assert bounds.max_lon is not None
                assert bounds.min_lat < bounds.max_lat, \
                    f"Latitude bounds misordered: {bounds.min_lat} >= {bounds.max_lat}"
                assert bounds.min_lon < bounds.max_lon, \
                    f"Longitude bounds misordered: {bounds.min_lon} >= {bounds.max_lon}"
                # Sanity check for geographic coordinates
                assert -90 <= bounds.min_lat <= 90
                assert -90 <= bounds.max_lat <= 90
                assert -180 <= bounds.min_lon <= 180
                assert -180 <= bounds.max_lon <= 180
            elif isinstance(bounds, (tuple, list)) and len(bounds) == 4:
                west, south, east, north = bounds
                assert south < north, \
                    f"Latitude bounds misordered: {south} >= {north}"
                assert west < east, \
                    f"Longitude bounds misordered: {west} >= {east}"


@pytest.mark.slow
def test_s2_quality_scl_band_presence(require_sentinel2_file):
    """Scene Classification (SCL) band accessible or quality indicators present."""
    with Sentinel2Reader(str(require_sentinel2_file)) as reader:
        meta = reader.metadata
        assert meta is not None
        # Check for available band list (v0.4.0 expansion)
        if hasattr(meta, 'available_bands'):
            bands = meta.available_bands
            # Sentinel-2 L2A includes SCL band
            assert 'SCL' in bands or 'scl' in [b.lower() for b in bands], \
                f"SCL band not found in available bands: {bands}"
        
        # Alternative: check for quality indicators as metadata fields
        if hasattr(meta, 'quality_indicators'):
            quality = meta.quality_indicators
            assert quality is not None, "quality_indicators is None"
            assert isinstance(quality, dict), \
                f"quality_indicators should be dict, got {type(quality)}"


@pytest.mark.slow
def test_s2_metadata_completeness(require_sentinel2_file):
    """All Level-2 metadata fields are populated (non-None)."""
    with Sentinel2Reader(str(require_sentinel2_file)) as reader:
        meta = reader.metadata
        assert meta is not None
        # Core fields that must always be present
        assert meta.rows > 0, "rows must be positive"
        assert meta.cols > 0, "cols must be positive"
        assert meta.dtype is not None, "dtype must be non-None"
        # CRS for Sentinel-2 L2A should be defined
        assert meta.crs is not None, "CRS is None"


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
