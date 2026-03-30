# -*- coding: utf-8 -*-
"""
UTMGrid and WebMercatorGrid Tests - Output grid validation.

Validates grdl.image_processing.ortho.utm_grid.UTMGrid and
grdl.image_processing.ortho.web_mercator_grid.WebMercatorGrid as
alternatives to ENUGrid for orthorectification output grids:

- Level 1: Construction, property validation (rows, cols, EPSG)
- Level 2: image_to_latlon/latlon_to_image round-trip, CRS accuracy,
           extent validation, sub_grid extraction
- Level 3: from_geolocation factory with synthetic geolocation

All tests use synthetic parameters; no real data files required.

Dependencies
------------
pytest
numpy
pyproj (via grdl)

Author
------
Ava Courtney
courtney-ava@zai.com

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2026-03-30

Modified
--------
2026-03-30
"""

# Standard library
from typing import Tuple, Union

# Third-party
import pytest
import numpy as np

# GRDL internal
try:
    from grdl.image_processing.ortho.utm_grid import UTMGrid
    _HAS_UTM = True
except ImportError:
    _HAS_UTM = False

try:
    from grdl.image_processing.ortho.web_mercator_grid import WebMercatorGrid
    _HAS_WM = True
except ImportError:
    _HAS_WM = False

try:
    from grdl.geolocation.base import Geolocation
    _HAS_GEO = True
except ImportError:
    _HAS_GEO = False


pytestmark = [
    pytest.mark.ortho,
    pytest.mark.geolocation,
]


# ---------------------------------------------------------------------------
# Synthetic geolocation for from_geolocation tests
# ---------------------------------------------------------------------------

if _HAS_GEO:
    class _SimpleAffineGeo(Geolocation):
        """Affine geolocation for testing grid factories."""

        def __init__(
            self, shape: Tuple[int, int],
            origin_lat: float, origin_lon: float,
            dlat: float = 0.001, dlon: float = 0.001,
        ) -> None:
            super().__init__(shape, crs='WGS84')
            self._olat = origin_lat
            self._olon = origin_lon
            self._dlat = dlat
            self._dlon = dlon

        def _image_to_latlon_array(
            self, rows: np.ndarray, cols: np.ndarray,
            height: Union[float, np.ndarray] = 0.0,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            lats = self._olat - rows * self._dlat
            lons = self._olon + cols * self._dlon
            return lats, lons, np.full_like(lats, height)

        def _latlon_to_image_array(
            self, lats: np.ndarray, lons: np.ndarray,
            height: Union[float, np.ndarray] = 0.0,
        ) -> Tuple[np.ndarray, np.ndarray]:
            rows = (self._olat - lats) / self._dlat
            cols = (lons - self._olon) / self._dlon
            return rows, cols


# =============================================================================
# UTMGrid Tests
# =============================================================================

@pytest.mark.skipif(not _HAS_UTM, reason="UTMGrid not available")
class TestUTMGridConstruction:
    """Level 1: UTMGrid construction and property validation."""

    @pytest.fixture
    def utm(self):
        """UTM zone 17N grid over ~1 km area near Washington DC."""
        return UTMGrid(
            zone=17, north=True,
            min_easting=323000.0, max_easting=324000.0,
            min_northing=4305000.0, max_northing=4306000.0,
            pixel_size=10.0,
        )

    def test_constructor_valid(self, utm):
        """UTMGrid instantiates with valid parameters."""
        assert utm is not None

    def test_rows_cols_from_extent(self, utm):
        """rows and cols computed from extent / pixel_size."""
        assert utm.rows == 100, f"Expected 100 rows (1000m / 10m), got {utm.rows}"
        assert utm.cols == 100, f"Expected 100 cols (1000m / 10m), got {utm.cols}"

    def test_epsg_northern_hemisphere(self):
        """EPSG = 32600 + zone for northern hemisphere."""
        grid = UTMGrid(
            zone=33, north=True,
            min_easting=500000.0, max_easting=501000.0,
            min_northing=5500000.0, max_northing=5501000.0,
            pixel_size=10.0,
        )
        assert grid.epsg == 32633, f"Expected EPSG 32633, got {grid.epsg}"

    def test_epsg_southern_hemisphere(self):
        """EPSG = 32700 + zone for southern hemisphere."""
        grid = UTMGrid(
            zone=55, north=False,
            min_easting=300000.0, max_easting=301000.0,
            min_northing=6100000.0, max_northing=6101000.0,
            pixel_size=10.0,
        )
        assert grid.epsg == 32755, f"Expected EPSG 32755, got {grid.epsg}"

    def test_zone_range_valid(self):
        """UTM zones 1-60 are all valid."""
        for z in [1, 30, 60]:
            grid = UTMGrid(
                zone=z, north=True,
                min_easting=400000.0, max_easting=401000.0,
                min_northing=4000000.0, max_northing=4001000.0,
                pixel_size=100.0,
            )
            assert 32601 <= grid.epsg <= 32660


@pytest.mark.skipif(not _HAS_UTM, reason="UTMGrid not available")
class TestUTMGridTransforms:
    """Level 2: Coordinate transform accuracy and round-trip."""

    @pytest.fixture
    def utm(self):
        """UTM zone 17N grid near Washington DC."""
        return UTMGrid(
            zone=17, north=True,
            min_easting=323000.0, max_easting=324000.0,
            min_northing=4305000.0, max_northing=4306000.0,
            pixel_size=10.0,
        )

    def test_image_to_latlon_returns_valid_coords(self, utm):
        """Center pixel maps to valid lat/lon."""
        lat, lon = utm.image_to_latlon(50.0, 50.0)
        assert -90.0 <= float(lat) <= 90.0, f"lat {lat} out of bounds"
        assert -180.0 <= float(lon) <= 180.0, f"lon {lon} out of bounds"

    def test_round_trip_scalar(self, utm):
        """Pixel → latlon → pixel round-trip < 0.01 pixels."""
        row_in, col_in = 25.0, 75.0
        lat, lon = utm.image_to_latlon(row_in, col_in)
        row_out, col_out = utm.latlon_to_image(lat, lon)
        assert float(row_out) == pytest.approx(row_in, abs=0.01), (
            f"UTM round-trip row error: {abs(float(row_out) - row_in):.6f}"
        )
        assert float(col_out) == pytest.approx(col_in, abs=0.01), (
            f"UTM round-trip col error: {abs(float(col_out) - col_in):.6f}"
        )

    def test_round_trip_vectorized(self, utm):
        """Vectorized round-trip across grid corners and center."""
        rows = np.array([0.0, 0.0, 99.0, 99.0, 50.0])
        cols = np.array([0.0, 99.0, 0.0, 99.0, 50.0])
        lats, lons = utm.image_to_latlon(rows, cols)
        rows_rt, cols_rt = utm.latlon_to_image(lats, lons)
        np.testing.assert_allclose(rows_rt, rows, atol=0.01,
            err_msg="UTM vectorized round-trip row error > 0.01 px")
        np.testing.assert_allclose(cols_rt, cols, atol=0.01,
            err_msg="UTM vectorized round-trip col error > 0.01 px")

    def test_row0_is_north_edge(self, utm):
        """Row 0 should have higher latitude than last row (north-up convention)."""
        lat_top, _ = utm.image_to_latlon(0.0, 50.0)
        lat_bot, _ = utm.image_to_latlon(float(utm.rows - 1), 50.0)
        assert float(lat_top) > float(lat_bot), (
            f"Row 0 lat {float(lat_top):.6f}° should be > row {utm.rows-1} lat "
            f"{float(lat_bot):.6f}° (north-up)"
        )

    def test_col0_is_west_edge(self, utm):
        """Col 0 should have lower longitude than last col (east-right convention)."""
        _, lon_left = utm.image_to_latlon(50.0, 0.0)
        _, lon_right = utm.image_to_latlon(50.0, float(utm.cols - 1))
        assert float(lon_left) < float(lon_right), (
            f"Col 0 lon {float(lon_left):.6f}° should be < col {utm.cols-1} lon "
            f"{float(lon_right):.6f}° (east-right)"
        )

    def test_sub_grid_round_trip(self, utm):
        """Sub-grid center pixel maps to same lat/lon as equivalent parent pixel."""
        sub = utm.sub_grid(row_start=25, col_start=25, row_end=75, col_end=75)
        assert sub.rows == 50
        assert sub.cols == 50

        # Sub-grid pixel (0, 0) should equal parent pixel (25, 25)
        lat_sub, lon_sub = sub.image_to_latlon(0.0, 0.0)
        lat_par, lon_par = utm.image_to_latlon(25.0, 25.0)
        assert float(lat_sub) == pytest.approx(float(lat_par), abs=1e-6)
        assert float(lon_sub) == pytest.approx(float(lon_par), abs=1e-6)


@pytest.mark.skipif(not _HAS_UTM, reason="UTMGrid not available")
@pytest.mark.skipif(not _HAS_GEO, reason="Geolocation base not available")
class TestUTMGridFactory:
    """Level 3: from_geolocation factory method."""

    def test_from_geolocation_creates_grid(self):
        """from_geolocation auto-detects UTM zone and builds grid."""
        geo = _SimpleAffineGeo(
            shape=(500, 500),
            origin_lat=38.9, origin_lon=-77.0,
            dlat=0.001, dlon=0.001,
        )
        grid = UTMGrid.from_geolocation(geo, pixel_size_m=10.0)
        assert grid.rows > 0
        assert grid.cols > 0
        # Washington DC is in UTM zone 18N
        assert grid.zone == 18, f"Expected zone 18, got {grid.zone}"
        assert grid.north is True

    def test_from_geolocation_covers_footprint(self):
        """Generated grid covers the geolocation footprint."""
        geo = _SimpleAffineGeo(
            shape=(200, 200),
            origin_lat=34.05, origin_lon=-118.25,
            dlat=0.001, dlon=0.001,
        )
        grid = UTMGrid.from_geolocation(geo, pixel_size_m=30.0, margin_m=100.0)

        # All four image corners should map to valid grid pixels
        for r, c in [(0, 0), (0, 199), (199, 0), (199, 199)]:
            lat, lon, _ = geo.image_to_latlon(float(r), float(c))
            gr, gc = grid.latlon_to_image(lat, lon)
            assert 0 <= float(gr) < grid.rows, (
                f"Image corner ({r},{c}) grid row {float(gr):.1f} outside [0, {grid.rows})"
            )
            assert 0 <= float(gc) < grid.cols, (
                f"Image corner ({r},{c}) grid col {float(gc):.1f} outside [0, {grid.cols})"
            )


# =============================================================================
# WebMercatorGrid Tests
# =============================================================================

@pytest.mark.skipif(not _HAS_WM, reason="WebMercatorGrid not available")
class TestWebMercatorGridConstruction:
    """Level 1: WebMercatorGrid construction and property validation."""

    @pytest.fixture
    def wm(self):
        """WebMercator grid over small area near lat=34, lon=-118 (Los Angeles)."""
        return WebMercatorGrid.from_bounds_latlon(
            min_lat=33.9, max_lat=34.1,
            min_lon=-118.3, max_lon=-118.1,
            pixel_size=100.0,
        )

    def test_constructor_valid(self, wm):
        """WebMercatorGrid instantiates from lat/lon bounds."""
        assert wm is not None

    def test_rows_cols_positive(self, wm):
        """Grid has positive dimensions."""
        assert wm.rows > 0, f"rows should be > 0, got {wm.rows}"
        assert wm.cols > 0, f"cols should be > 0, got {wm.cols}"

    def test_pixel_size_preserved(self, wm):
        """pixel_size attribute matches construction value."""
        assert wm.pixel_size == pytest.approx(100.0)


@pytest.mark.skipif(not _HAS_WM, reason="WebMercatorGrid not available")
class TestWebMercatorGridTransforms:
    """Level 2: Coordinate transform accuracy and EPSG:3857 validation."""

    @pytest.fixture
    def wm(self):
        """WebMercator grid near Los Angeles."""
        return WebMercatorGrid.from_bounds_latlon(
            min_lat=33.9, max_lat=34.1,
            min_lon=-118.3, max_lon=-118.1,
            pixel_size=100.0,
        )

    def test_image_to_latlon_valid(self, wm):
        """Center pixel produces valid lat/lon."""
        lat, lon = wm.image_to_latlon(wm.rows / 2, wm.cols / 2)
        assert -90.0 <= float(lat) <= 90.0
        assert -180.0 <= float(lon) <= 180.0

    def test_center_latlon_approximate(self, wm):
        """Center pixel lat/lon approximately matches input center."""
        lat, lon = wm.image_to_latlon(wm.rows / 2, wm.cols / 2)
        assert float(lat) == pytest.approx(34.0, abs=0.2), (
            f"Center lat {float(lat):.4f}° should be ~34.0°"
        )
        assert float(lon) == pytest.approx(-118.2, abs=0.2), (
            f"Center lon {float(lon):.4f}° should be ~-118.2°"
        )

    def test_round_trip_scalar(self, wm):
        """Pixel → latlon → pixel round-trip < 0.01 pixels."""
        row_in, col_in = 50.0, 75.0
        lat, lon = wm.image_to_latlon(row_in, col_in)
        row_out, col_out = wm.latlon_to_image(lat, lon)
        assert float(row_out) == pytest.approx(row_in, abs=0.01)
        assert float(col_out) == pytest.approx(col_in, abs=0.01)

    def test_round_trip_vectorized(self, wm):
        """Vectorized round-trip across grid corners."""
        rows = np.array([0.0, 0.0, float(wm.rows - 1), float(wm.rows - 1)])
        cols = np.array([0.0, float(wm.cols - 1), 0.0, float(wm.cols - 1)])
        lats, lons = wm.image_to_latlon(rows, cols)
        rows_rt, cols_rt = wm.latlon_to_image(lats, lons)
        np.testing.assert_allclose(rows_rt, rows, atol=0.01,
            err_msg="WebMercator vectorized round-trip row error")
        np.testing.assert_allclose(cols_rt, cols, atol=0.01,
            err_msg="WebMercator vectorized round-trip col error")

    def test_north_up_convention(self, wm):
        """Row 0 has higher latitude than last row."""
        lat_top, _ = wm.image_to_latlon(0.0, float(wm.cols // 2))
        lat_bot, _ = wm.image_to_latlon(float(wm.rows - 1), float(wm.cols // 2))
        assert float(lat_top) > float(lat_bot)

    def test_sub_grid_consistency(self, wm):
        """Sub-grid (0,0) maps to same lat/lon as parent at sub-grid origin."""
        sub = wm.sub_grid(row_start=10, col_start=10, row_end=50, col_end=50)
        lat_sub, lon_sub = sub.image_to_latlon(0.0, 0.0)
        lat_par, lon_par = wm.image_to_latlon(10.0, 10.0)
        assert float(lat_sub) == pytest.approx(float(lat_par), abs=1e-6)
        assert float(lon_sub) == pytest.approx(float(lon_par), abs=1e-6)


@pytest.mark.skipif(not _HAS_WM, reason="WebMercatorGrid not available")
@pytest.mark.skipif(not _HAS_GEO, reason="Geolocation base not available")
class TestWebMercatorGridFactory:
    """Level 3: from_geolocation factory method."""

    def test_from_geolocation_creates_grid(self):
        """from_geolocation builds grid covering image footprint."""
        geo = _SimpleAffineGeo(
            shape=(300, 300),
            origin_lat=51.5, origin_lon=-0.1,  # London
            dlat=0.001, dlon=0.001,
        )
        grid = WebMercatorGrid.from_geolocation(geo, pixel_size=50.0)
        assert grid.rows > 0
        assert grid.cols > 0

        # Center should be near London
        lat, lon = grid.image_to_latlon(grid.rows / 2, grid.cols / 2)
        assert float(lat) == pytest.approx(51.35, abs=0.5)
        assert float(lon) == pytest.approx(0.05, abs=0.5)

    def test_from_geolocation_covers_footprint(self):
        """Generated grid covers all image corners."""
        geo = _SimpleAffineGeo(
            shape=(200, 200),
            origin_lat=48.85, origin_lon=2.35,  # Paris
            dlat=0.001, dlon=0.001,
        )
        grid = WebMercatorGrid.from_geolocation(geo, pixel_size=30.0, margin_m=100.0)

        for r, c in [(0, 0), (0, 199), (199, 0), (199, 199)]:
            lat, lon, _ = geo.image_to_latlon(float(r), float(c))
            gr, gc = grid.latlon_to_image(lat, lon)
            assert 0 <= float(gr) < grid.rows, (
                f"Corner ({r},{c}) grid row {float(gr):.1f} outside bounds"
            )
            assert 0 <= float(gc) < grid.cols, (
                f"Corner ({r},{c}) grid col {float(gc):.1f} outside bounds"
            )
