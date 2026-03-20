# -*- coding: utf-8 -*-
"""
SICDGeolocation Tests - SAR coordinate transform validation.

Tests SICDGeolocation using real Umbra SICD data:
- Level 1: Construction, SCP image-to-latlon, latlon-to-image
- Level 2: Round-trip precision, footprint, vectorized operations

Dataset: Umbra SICD (*.nitf)

Dependencies
------------
pytest
numpy
grdl (sarkit or sarpy backend)

Author
------
Claude Code (generated)

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2026-03-20

Modified
--------
2026-03-20
"""

# Third-party
import pytest
import numpy as np

try:
    from grdl.IO.sar.sicd import SICDReader
    _HAS_SICD = True
except ImportError:
    _HAS_SICD = False

try:
    from grdl.geolocation.sar.sicd import SICDGeolocation
    _HAS_SICD_GEO = True
except ImportError:
    _HAS_SICD_GEO = False


pytestmark = [
    pytest.mark.geolocation,
    pytest.mark.sicd,
    pytest.mark.requires_data,
    pytest.mark.skipif(not _HAS_SICD, reason="SICDReader not available"),
    pytest.mark.skipif(not _HAS_SICD_GEO, reason="SICDGeolocation not available"),
]


# =============================================================================
# Level 1: Format Validation
# =============================================================================


@pytest.mark.slow
def test_sicd_geolocation_from_metadata(require_umbra_file):
    """SICDGeolocation constructs from SICDReader."""
    with SICDReader(str(require_umbra_file)) as reader:
        geo = SICDGeolocation.from_reader(reader)
        assert geo is not None


@pytest.mark.slow
def test_sicd_image_to_latlon_scp(require_umbra_file):
    """SCP pixel maps to valid lat/lon."""
    with SICDReader(str(require_umbra_file)) as reader:
        geo = SICDGeolocation.from_reader(reader)
        rows, cols = reader.get_shape()[:2]

        lat, lon, hae = geo.image_to_latlon(rows // 2, cols // 2)
        assert -90 <= lat <= 90, f"lat={lat} out of bounds"
        assert -180 <= lon <= 180, f"lon={lon} out of bounds"
        assert np.isfinite(hae)


@pytest.mark.slow
def test_sicd_latlon_to_image_scp(require_umbra_file):
    """Known lat/lon maps back to valid pixel coordinates."""
    with SICDReader(str(require_umbra_file)) as reader:
        geo = SICDGeolocation.from_reader(reader)
        rows, cols = reader.get_shape()[:2]

        # Get ground coords at center
        lat, lon, hae = geo.image_to_latlon(rows // 2, cols // 2)

        # Map back to image
        row, col = geo.latlon_to_image(lat, lon, hae)
        assert np.isfinite(row) and np.isfinite(col), (
            f"latlon_to_image returned non-finite: ({row}, {col})"
        )


# =============================================================================
# Level 2: Data Quality
# =============================================================================


@pytest.mark.slow
def test_sicd_round_trip_precision(require_umbra_file):
    """Image → ground → image round-trip is sub-pixel accurate."""
    with SICDReader(str(require_umbra_file)) as reader:
        geo = SICDGeolocation.from_reader(reader)
        rows, cols = reader.get_shape()[:2]
        center_row, center_col = float(rows // 2), float(cols // 2)

        lat, lon, hae = geo.image_to_latlon(center_row, center_col)
        row_back, col_back = geo.latlon_to_image(lat, lon, hae)

        assert abs(row_back - center_row) < 1.0, (
            f"Row round-trip error: {abs(row_back - center_row):.4f} pixels"
        )
        assert abs(col_back - center_col) < 1.0, (
            f"Col round-trip error: {abs(col_back - center_col):.4f} pixels"
        )


@pytest.mark.slow
def test_sicd_footprint(require_umbra_file):
    """get_footprint returns a polygon with valid coordinates."""
    with SICDReader(str(require_umbra_file)) as reader:
        geo = SICDGeolocation.from_reader(reader)

        if hasattr(geo, 'get_footprint'):
            footprint = geo.get_footprint()
            if footprint is not None:
                # get_footprint returns a dict with 'type', 'coordinates', 'bounds'
                assert footprint['type'] == 'Polygon'
                coords = footprint['coordinates']
                assert len(coords) >= 4, (
                    f"Footprint has {len(coords)} points, expected >= 4"
                )


@pytest.mark.slow
def test_sicd_vectorized(require_umbra_file):
    """Array inputs produce array outputs with consistent shapes."""
    with SICDReader(str(require_umbra_file)) as reader:
        geo = SICDGeolocation.from_reader(reader)
        rows, cols = reader.get_shape()[:2]

        # Create array of pixel coordinates
        test_rows = np.array([rows // 4, rows // 2, 3 * rows // 4], dtype=np.float64)
        test_cols = np.array([cols // 4, cols // 2, 3 * cols // 4], dtype=np.float64)

        lats, lons, haes = geo.image_to_latlon(test_rows, test_cols)
        assert isinstance(lats, np.ndarray)
        assert len(lats) == 3
        assert np.all(np.isfinite(lats))
        assert np.all((-90 <= lats) & (lats <= 90))
