# -*- coding: utf-8 -*-
"""
Sentinel1SLCGeolocation and NoGeolocation validation.

NoGeolocation tests are synthetic (always run).
Sentinel1SLCGeolocation tests require real Sentinel-1 SLC data.

Tests:
- Level 1: Identity transforms, from_reader construction
- Level 2: Roundtrip consistency, coordinate ranges, corner points
- Level 3: Vectorized batch queries, shape property

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
    from grdl.geolocation import NoGeolocation
    _HAS_NOGEO = True
except ImportError:
    _HAS_NOGEO = False

try:
    from grdl.geolocation import Sentinel1SLCGeolocation
    _HAS_S1_GEO = True
except ImportError:
    _HAS_S1_GEO = False

try:
    from grdl.IO.sar import Sentinel1SLCReader
    _HAS_S1 = True
except ImportError:
    _HAS_S1 = False


pytestmark = [
    pytest.mark.geolocation,
]


# =============================================================================
# NoGeolocation — synthetic, always runs
# =============================================================================


@pytest.mark.skipif(not _HAS_NOGEO, reason="NoGeolocation not available")
class TestNoGeolocation:

    def test_nogeo_constructor(self):
        """Accepts (rows, cols) shape."""
        geo = NoGeolocation(shape=(512, 512))
        assert geo is not None

    def test_nogeo_image_to_latlon_raises(self):
        """image_to_latlon raises NotImplementedError (no geolocation)."""
        geo = NoGeolocation(shape=(512, 512))
        with pytest.raises(NotImplementedError):
            geo.image_to_latlon(100, 200)

    def test_nogeo_latlon_to_image_raises(self):
        """latlon_to_image raises NotImplementedError (no geolocation)."""
        geo = NoGeolocation(shape=(512, 512))
        with pytest.raises(NotImplementedError):
            geo.latlon_to_image(34.0, -118.0)

    def test_nogeo_shape_property(self):
        """.shape returns constructor shape."""
        geo = NoGeolocation(shape=(512, 768))
        assert geo.shape == (512, 768)

    def test_nogeo_footprint_empty(self):
        """get_footprint() returns empty/minimal dict."""
        geo = NoGeolocation(shape=(512, 512))
        fp = geo.get_footprint()
        assert isinstance(fp, dict)


# =============================================================================
# Sentinel1SLCGeolocation — real data, skips if no data
# =============================================================================


@pytest.mark.sentinel1
@pytest.mark.requires_data
@pytest.mark.skipif(not _HAS_S1_GEO, reason="Sentinel1SLCGeolocation not available")
@pytest.mark.skipif(not _HAS_S1, reason="Sentinel1SLCReader not available")
class TestSentinel1SLCGeolocation:

    @pytest.mark.slow
    def test_s1_geo_from_reader(self, require_sentinel1_file):
        """Sentinel1SLCGeolocation.from_reader() succeeds."""
        with Sentinel1SLCReader(require_sentinel1_file) as reader:
            geo = Sentinel1SLCGeolocation.from_reader(reader)
            assert geo is not None

    @pytest.mark.slow
    def test_s1_geo_image_to_latlon(self, require_sentinel1_file):
        """Returns lat in [-90,90], lon in [-180,180]."""
        with Sentinel1SLCReader(require_sentinel1_file) as reader:
            geo = Sentinel1SLCGeolocation.from_reader(reader)
            shape = reader.get_shape()

        lat, lon = geo.image_to_latlon(shape[0] // 2, shape[1] // 2)
        assert -90 <= lat <= 90, f"Latitude {lat} out of range"
        assert -180 <= lon <= 180, f"Longitude {lon} out of range"

    @pytest.mark.slow
    def test_s1_geo_roundtrip(self, require_sentinel1_file):
        """image_to_latlon → latlon_to_image within 1 pixel."""
        with Sentinel1SLCReader(require_sentinel1_file) as reader:
            geo = Sentinel1SLCGeolocation.from_reader(reader)
            shape = reader.get_shape()

        r0, c0 = shape[0] // 2, shape[1] // 2
        lat, lon = geo.image_to_latlon(r0, c0)
        r1, c1 = geo.latlon_to_image(lat, lon)
        assert abs(r1 - r0) <= 1.5, f"Row roundtrip error: {abs(r1-r0)}"
        assert abs(c1 - c0) <= 1.5, f"Col roundtrip error: {abs(c1-c0)}"

    @pytest.mark.slow
    def test_s1_geo_corner_points(self, require_sentinel1_file):
        """All 4 corners produce valid coordinates."""
        with Sentinel1SLCReader(require_sentinel1_file) as reader:
            geo = Sentinel1SLCGeolocation.from_reader(reader)
            shape = reader.get_shape()

        corners = [(0, 0), (0, shape[1]-1), (shape[0]-1, 0), (shape[0]-1, shape[1]-1)]
        for r, c in corners:
            lat, lon = geo.image_to_latlon(r, c)
            assert -90 <= lat <= 90
            assert -180 <= lon <= 180

    @pytest.mark.slow
    @pytest.mark.integration
    def test_s1_geo_batch_1000(self, require_sentinel1_file):
        """Vectorized 1000 points."""
        with Sentinel1SLCReader(require_sentinel1_file) as reader:
            geo = Sentinel1SLCGeolocation.from_reader(reader)
            shape = reader.get_shape()

        rng = np.random.default_rng(42)
        rows = rng.uniform(0, shape[0], size=1000)
        cols = rng.uniform(0, shape[1], size=1000)
        lats, lons = geo.image_to_latlon(rows, cols)
        assert lats.shape == (1000,)
        assert np.all((-90 <= lats) & (lats <= 90))
        assert np.all((-180 <= lons) & (lons <= 180))
