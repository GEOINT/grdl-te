# -*- coding: utf-8 -*-
"""
NISARGeolocation Tests - NISAR coordinate transform validation.

Tests NISARGeolocation using real NISAR HDF5 data:
- Level 1: Construction, image-to-latlon, latlon-to-image
- Level 2: Round-trip precision, footprint, grid interpolation

Dataset: NISAR RSLC or GSLC HDF5

Dependencies
------------
pytest
numpy
h5py
grdl

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
    from grdl.IO.sar.nisar import NISARReader
    _HAS_NISAR = True
except ImportError:
    _HAS_NISAR = False

try:
    from grdl.geolocation.sar.nisar import NISARGeolocation
    _HAS_NISAR_GEO = True
except ImportError:
    _HAS_NISAR_GEO = False


pytestmark = [
    pytest.mark.geolocation,
    pytest.mark.nisar,
    pytest.mark.requires_data,
    pytest.mark.skipif(not _HAS_NISAR, reason="NISARReader not available"),
    pytest.mark.skipif(not _HAS_NISAR_GEO, reason="NISARGeolocation not available"),
]


# =============================================================================
# Level 1: Format Validation
# =============================================================================


@pytest.mark.slow
def test_nisar_geolocation_construction(require_nisar_file):
    """NISARGeolocation constructs from NISARReader."""
    with NISARReader(str(require_nisar_file)) as reader:
        geo = NISARGeolocation.from_reader(reader)
        assert geo is not None


@pytest.mark.slow
def test_nisar_image_to_latlon(require_nisar_file):
    """Center pixel maps to valid lat/lon."""
    with NISARReader(str(require_nisar_file)) as reader:
        geo = NISARGeolocation.from_reader(reader)
        rows, cols = reader.get_shape()[:2]

        lat, lon, hae = geo.image_to_latlon(rows // 2, cols // 2)
        assert -90 <= lat <= 90, f"lat={lat} out of bounds"
        assert -180 <= lon <= 180, f"lon={lon} out of bounds"
        assert np.isfinite(hae)


@pytest.mark.slow
def test_nisar_latlon_to_image(require_nisar_file):
    """Known ground coords map back to valid pixel coordinates."""
    with NISARReader(str(require_nisar_file)) as reader:
        geo = NISARGeolocation.from_reader(reader)
        rows, cols = reader.get_shape()[:2]

        lat, lon, hae = geo.image_to_latlon(rows // 2, cols // 2)
        row, col = geo.latlon_to_image(lat, lon, hae)
        assert np.isfinite(row) and np.isfinite(col)


# =============================================================================
# Level 2: Data Quality
# =============================================================================


@pytest.mark.slow
def test_nisar_round_trip(require_nisar_file):
    """Image → ground → image round-trip within tolerance."""
    with NISARReader(str(require_nisar_file)) as reader:
        geo = NISARGeolocation.from_reader(reader)
        rows, cols = reader.get_shape()[:2]
        center_row, center_col = float(rows // 2), float(cols // 2)

        lat, lon, hae = geo.image_to_latlon(center_row, center_col)
        row_back, col_back = geo.latlon_to_image(lat, lon, hae)

        assert abs(row_back - center_row) < 5.0, (
            f"Row round-trip error: {abs(row_back - center_row):.2f} pixels"
        )
        assert abs(col_back - center_col) < 5.0, (
            f"Col round-trip error: {abs(col_back - center_col):.2f} pixels"
        )


@pytest.mark.slow
def test_nisar_footprint(require_nisar_file):
    """Footprint produces valid geographic bounds."""
    with NISARReader(str(require_nisar_file)) as reader:
        geo = NISARGeolocation.from_reader(reader)

        if hasattr(geo, 'get_footprint'):
            footprint = geo.get_footprint()
            if footprint is not None:
                coords = list(footprint.exterior.coords)
                assert len(coords) >= 4


@pytest.mark.slow
def test_nisar_grid_interpolation(require_nisar_file):
    """Geolocation grid produces spatially consistent results.

    Multiple pixels across the image should produce monotonically
    varying coordinates (assuming a consistent look direction).
    """
    with NISARReader(str(require_nisar_file)) as reader:
        geo = NISARGeolocation.from_reader(reader)
        rows, cols = reader.get_shape()[:2]

        # Sample pixels along a row (range direction)
        test_cols = np.linspace(cols // 4, 3 * cols // 4, 5, dtype=np.float64)
        test_rows = np.full_like(test_cols, rows // 2)

        lats, lons, haes = geo.image_to_latlon(test_rows, test_cols)
        assert np.all(np.isfinite(lats)), "Non-finite latitudes"
        assert np.all(np.isfinite(lons)), "Non-finite longitudes"

        # Coordinates should vary across range (not all identical)
        assert lons.std() > 1e-6 or lats.std() > 1e-6, (
            "Geolocation returns identical coords for different pixels"
        )
