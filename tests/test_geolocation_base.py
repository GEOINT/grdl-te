# -*- coding: utf-8 -*-
"""
Geolocation Base Class Tests - ABC contract and NoGeolocation validation.

Tests the Geolocation abstract base class dispatch logic (scalar, array,
stacked-array input forms) and the NoGeolocation fallback behavior. Uses a
minimal concrete subclass with known affine math to validate the dispatch
layer independently of real sensor models.

Also tests AffineGeolocation with synthetic transforms (EPSG:4326 and
UTM projections) to validate forward/inverse round-trip precision.

Dependencies
------------
pytest
numpy
grdl

Author
------
Ava Courtney

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2026-02-12

Modified
--------
2026-02-12
"""

# Standard library
from typing import Tuple

# Third-party
import pytest
import numpy as np

# GRDL internal
try:
    from grdl.geolocation.base import Geolocation, NoGeolocation, _is_scalar, _to_array
    _HAS_GRDL_BASE = True
except ImportError:
    _HAS_GRDL_BASE = False

try:
    from grdl.geolocation.eo.affine import AffineGeolocation
    from rasterio.transform import Affine
    import pyproj
    _HAS_AFFINE = True
except ImportError:
    _HAS_AFFINE = False


pytestmark = [
    pytest.mark.geolocation,
    pytest.mark.skipif(not _HAS_GRDL_BASE, reason="grdl.geolocation not installed"),
]


# =============================================================================
# Minimal Concrete Geolocation (for testing ABC dispatch only)
# =============================================================================


class _IdentityGeolocation(Geolocation):
    """Trivial geolocation: pixel (row, col) maps directly to (lat=row, lon=col).

    This exists solely to test the ABC dispatch layer without involving
    real sensor math. Not physically meaningful.
    """

    def __init__(self, shape: Tuple[int, int]):
        super().__init__(shape, crs='WGS84')

    def _image_to_latlon_array(
        self, rows: np.ndarray, cols: np.ndarray, height: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lats = rows.copy()
        lons = cols.copy()
        heights = np.full_like(rows, height)
        return lats, lons, heights

    def _latlon_to_image_array(
        self, lats: np.ndarray, lons: np.ndarray, height: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        rows = lats.copy()
        cols = lons.copy()
        return rows, cols


# =============================================================================
# Private helper tests
# =============================================================================


class TestHelpers:
    """Tests for _is_scalar and _to_array dispatch helpers."""

    def test_is_scalar_int(self):
        assert _is_scalar(5) is True

    def test_is_scalar_float(self):
        assert _is_scalar(3.14) is True

    def test_is_scalar_numpy_int(self):
        assert _is_scalar(np.int64(42)) is True

    def test_is_scalar_numpy_float(self):
        assert _is_scalar(np.float64(1.5)) is True

    def test_is_scalar_0d_array(self):
        assert _is_scalar(np.array(5.0)) is True

    def test_is_scalar_list_false(self):
        assert _is_scalar([1, 2]) is False

    def test_is_scalar_1d_array_false(self):
        assert _is_scalar(np.array([1.0])) is False

    def test_to_array_from_scalar(self):
        result = _to_array(5.0)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        assert result.dtype == np.float64
        assert result[0] == 5.0

    def test_to_array_from_list(self):
        result = _to_array([1.0, 2.0, 3.0])
        assert result.shape == (3,)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_to_array_from_ndarray(self):
        arr = np.array([10.0, 20.0])
        result = _to_array(arr)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, arr)


# =============================================================================
# Geolocation ABC — Scalar/Array/Stacked dispatch
# =============================================================================


class TestGeolocationDispatch:
    """Test Geolocation.image_to_latlon and latlon_to_image dispatch logic."""

    @pytest.fixture
    def geo(self):
        return _IdentityGeolocation(shape=(1000, 2000))

    # -- image_to_latlon --

    def test_scalar_input_returns_tuple_of_floats(self, geo):
        result = geo.image_to_latlon(100.0, 200.0)
        assert isinstance(result, tuple)
        assert len(result) == 3
        lat, lon, h = result
        assert isinstance(lat, float)
        assert isinstance(lon, float)
        assert isinstance(h, float)
        assert lat == pytest.approx(100.0, abs=1e-12)
        assert lon == pytest.approx(200.0, abs=1e-12)

    def test_scalar_int_input(self, geo):
        """Python int inputs should dispatch as scalar."""
        lat, lon, h = geo.image_to_latlon(50, 75)
        assert isinstance(lat, float)
        assert lat == pytest.approx(50.0, abs=1e-12)
        assert lon == pytest.approx(75.0, abs=1e-12)

    def test_array_input_returns_tuple_of_arrays(self, geo):
        rows = np.array([0.0, 100.0, 500.0])
        cols = np.array([0.0, 200.0, 1000.0])
        lats, lons, heights = geo.image_to_latlon(rows, cols)

        assert isinstance(lats, np.ndarray)
        assert isinstance(lons, np.ndarray)
        assert isinstance(heights, np.ndarray)
        np.testing.assert_allclose(lats, rows, atol=1e-12)
        np.testing.assert_allclose(lons, cols, atol=1e-12)

    def test_list_input_returns_arrays(self, geo):
        """Python list inputs should dispatch as array."""
        lats, lons, heights = geo.image_to_latlon([10.0, 20.0], [30.0, 40.0])
        assert isinstance(lats, np.ndarray)
        np.testing.assert_allclose(lats, [10.0, 20.0], atol=1e-12)
        np.testing.assert_allclose(lons, [30.0, 40.0], atol=1e-12)

    def test_stacked_2xN_input_returns_3xN(self, geo):
        points = np.array([[10.0, 20.0, 30.0],
                           [40.0, 50.0, 60.0]])
        result = geo.image_to_latlon(points)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 3)
        np.testing.assert_allclose(result[0], [10.0, 20.0, 30.0], atol=1e-12)
        np.testing.assert_allclose(result[1], [40.0, 50.0, 60.0], atol=1e-12)

    def test_stacked_invalid_shape_raises(self, geo):
        """Non-(2, N) stacked array must raise ValueError."""
        with pytest.raises(ValueError, match="Expected \\(2, N\\)"):
            geo.image_to_latlon(np.array([[1.0, 2.0, 3.0]]))

    def test_height_passthrough(self, geo):
        """Height argument should propagate to output."""
        _, _, h = geo.image_to_latlon(50.0, 50.0, height=100.0)
        assert h == pytest.approx(100.0)

    # -- latlon_to_image --

    def test_latlon_scalar_returns_floats(self, geo):
        row, col = geo.latlon_to_image(10.0, 20.0)
        assert isinstance(row, float)
        assert isinstance(col, float)
        assert row == pytest.approx(10.0, abs=1e-12)
        assert col == pytest.approx(20.0, abs=1e-12)

    def test_latlon_array_returns_arrays(self, geo):
        rows, cols = geo.latlon_to_image(
            np.array([10.0, 20.0]), np.array([30.0, 40.0])
        )
        assert isinstance(rows, np.ndarray)
        np.testing.assert_allclose(rows, [10.0, 20.0], atol=1e-12)
        np.testing.assert_allclose(cols, [30.0, 40.0], atol=1e-12)

    def test_latlon_stacked_returns_2xN(self, geo):
        coords = np.array([[10.0, 20.0], [30.0, 40.0]])
        result = geo.latlon_to_image(coords)
        assert result.shape == (2, 2)

    def test_latlon_stacked_invalid_raises(self, geo):
        with pytest.raises(ValueError, match="Expected \\(2, N\\)"):
            geo.latlon_to_image(np.array([1.0, 2.0, 3.0]))

    # -- Round-trip --

    def test_round_trip_scalar(self, geo):
        """Pixel → latlon → pixel must be identity for identity geolocation."""
        lat, lon, h = geo.image_to_latlon(123.4, 567.8)
        row, col = geo.latlon_to_image(lat, lon)
        assert row == pytest.approx(123.4, abs=1e-10)
        assert col == pytest.approx(567.8, abs=1e-10)

    def test_round_trip_array(self, geo):
        rows = np.array([0.0, 250.0, 500.0, 999.0])
        cols = np.array([0.0, 500.0, 1000.0, 1999.0])

        lats, lons, _ = geo.image_to_latlon(rows, cols)
        rows_back, cols_back = geo.latlon_to_image(lats, lons)

        np.testing.assert_allclose(rows_back, rows, atol=1e-12)
        np.testing.assert_allclose(cols_back, cols, atol=1e-12)


# =============================================================================
# NoGeolocation — Fallback behavior
# =============================================================================


class TestNoGeolocation:
    """Tests for NoGeolocation fallback class."""

    @pytest.fixture
    def no_geo(self):
        return NoGeolocation(shape=(512, 512))

    def test_image_to_latlon_raises(self, no_geo):
        """All forward transform calls must raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="no geolocation"):
            no_geo.image_to_latlon(0.0, 0.0)

    def test_latlon_to_image_raises(self, no_geo):
        """All inverse transform calls must raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="no geolocation"):
            no_geo.latlon_to_image(34.0, -118.0)

    def test_get_footprint_returns_none(self, no_geo):
        """Footprint must return type='None' with null coordinates."""
        result = no_geo.get_footprint()
        assert result['type'] == 'None'
        assert result['coordinates'] is None
        assert result['bounds'] is None

    def test_get_bounds_raises(self, no_geo):
        """get_bounds must raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            no_geo.get_bounds()

    def test_shape_preserved(self, no_geo):
        """Shape attribute must be set correctly."""
        assert no_geo.shape == (512, 512)

    def test_crs_default(self, no_geo):
        """Default CRS should be WGS84."""
        assert no_geo.crs == 'WGS84'


# =============================================================================
# Geolocation — get_footprint() / get_bounds() integration
# =============================================================================


class TestGeolocationFootprint:
    """Test get_footprint() and get_bounds() on concrete identity geolocation."""

    @pytest.fixture
    def geo(self):
        return _IdentityGeolocation(shape=(100, 200))

    def test_footprint_type_polygon(self, geo):
        result = geo.get_footprint()
        assert result['type'] == 'Polygon'

    def test_footprint_has_coordinates(self, geo):
        result = geo.get_footprint()
        assert result['coordinates'] is not None
        assert len(result['coordinates']) > 0

    def test_footprint_bounds_consistent(self, geo):
        """Bounds should span [0, rows-1] x [0, cols-1] for identity geo."""
        result = geo.get_footprint()
        min_lon, min_lat, max_lon, max_lat = result['bounds']

        # Identity geo maps row→lat, col→lon
        assert min_lat == pytest.approx(0.0, abs=1e-9)
        assert max_lat == pytest.approx(99.0, abs=1e-9)
        assert min_lon == pytest.approx(0.0, abs=1e-9)
        assert max_lon == pytest.approx(199.0, abs=1e-9)

    def test_get_bounds_returns_tuple(self, geo):
        bounds = geo.get_bounds()
        assert isinstance(bounds, tuple)
        assert len(bounds) == 4

    def test_get_bounds_ordered(self, geo):
        min_lon, min_lat, max_lon, max_lat = geo.get_bounds()
        assert min_lon <= max_lon
        assert min_lat <= max_lat


# =============================================================================
# AffineGeolocation — Synthetic transform tests
# =============================================================================


@pytest.mark.skipif(not _HAS_AFFINE, reason="rasterio/pyproj not installed")
class TestAffineGeolocationEPSG4326:
    """Test AffineGeolocation with geographic CRS (EPSG:4326).

    This tests the simplest case where the native CRS IS geographic, so
    the pyproj reprojection step is bypassed. Uses a 0.001-degree/pixel
    grid starting at Null Island.
    """

    @pytest.fixture
    def geo_4326(self):
        """AffineGeolocation with EPSG:4326 (0.001°/pixel from origin)."""
        # Affine(a, b, c, d, e, f)
        # x = c + col * a + row * b
        # y = f + col * d + row * e
        # For EPSG:4326: x=lon, y=lat
        # 0.001 deg/pixel in lon (east), -0.001 deg/pixel in lat (south)
        transform = Affine(0.001, 0.0, 0.0,   # lon increases east
                           0.0, -0.001, 1.0)   # lat decreases south from 1.0
        return AffineGeolocation(
            transform=transform,
            shape=(1000, 1000),
            crs='EPSG:4326',
        )

    def test_origin_pixel(self, geo_4326):
        """Pixel (0, 0) maps to upper-left corner of the grid."""
        lat, lon, h = geo_4326.image_to_latlon(0.0, 0.0)
        # y = f + 0*d + 0*e = 1.0 (lat)
        # x = c + 0*a + 0*b = 0.0 (lon)
        assert lat == pytest.approx(1.0, abs=1e-10)
        assert lon == pytest.approx(0.0, abs=1e-10)

    def test_center_pixel(self, geo_4326):
        """Pixel (500, 500) maps to center of the grid."""
        lat, lon, h = geo_4326.image_to_latlon(500.0, 500.0)
        expected_lat = 1.0 + 500 * (-0.001)  # 0.5
        expected_lon = 0.0 + 500 * 0.001      # 0.5
        assert lat == pytest.approx(expected_lat, abs=1e-10)
        assert lon == pytest.approx(expected_lon, abs=1e-10)

    def test_round_trip_scalar(self, geo_4326):
        """Forward then inverse must return original pixel to sub-pixel precision."""
        orig_row, orig_col = 123.456, 789.012
        lat, lon, h = geo_4326.image_to_latlon(orig_row, orig_col)
        row_back, col_back = geo_4326.latlon_to_image(lat, lon)

        assert row_back == pytest.approx(orig_row, abs=1e-12)
        assert col_back == pytest.approx(orig_col, abs=1e-12)

    def test_round_trip_array(self, geo_4326):
        """Round-trip precision for batch of points."""
        rng = np.random.default_rng(42)
        rows = rng.uniform(0, 999, 100)
        cols = rng.uniform(0, 999, 100)

        lats, lons, _ = geo_4326.image_to_latlon(rows, cols)
        rows_back, cols_back = geo_4326.latlon_to_image(lats, lons)

        np.testing.assert_allclose(rows_back, rows, atol=1e-12)
        np.testing.assert_allclose(cols_back, cols, atol=1e-12)

    def test_null_island_pixel(self, geo_4326):
        """Find the pixel corresponding to Null Island (0, 0)."""
        row, col = geo_4326.latlon_to_image(0.0, 0.0)
        # lat=0 → row = (0 - 1.0) / (-0.001) = 1000
        # lon=0 → col = (0 - 0) / 0.001 = 0
        assert row == pytest.approx(1000.0, abs=1e-12)
        assert col == pytest.approx(0.0, abs=1e-12)


@pytest.mark.skipif(not _HAS_AFFINE, reason="rasterio/pyproj not installed")
class TestAffineGeolocationUTM:
    """Test AffineGeolocation with projected CRS (UTM Zone 56S).

    This tests the full pipeline: affine → native CRS (UTM) → WGS84 via
    pyproj. Uses a 10m-pixel-size grid in UTM Zone 56S (Sydney area).
    """

    @pytest.fixture
    def geo_utm(self):
        """AffineGeolocation with EPSG:32756 (UTM 56S, 10m pixels)."""
        # UTM origin at easting=300000, northing=6250000
        transform = Affine(10.0, 0.0, 300000.0,
                           0.0, -10.0, 6250000.0)
        return AffineGeolocation(
            transform=transform,
            shape=(1000, 1000),
            crs='EPSG:32756',
        )

    def test_origin_pixel_in_wgs84(self, geo_utm):
        """Pixel (0, 0) should map to a valid lat/lon in southern Australia."""
        lat, lon, h = geo_utm.image_to_latlon(0.0, 0.0)

        # UTM 56S, easting=300000, northing=6250000 → roughly (-33.9, 150.3)
        assert -40.0 < lat < -30.0, f"Latitude {lat} outside expected range"
        assert 148.0 < lon < 153.0, f"Longitude {lon} outside expected range"

    def test_round_trip_sub_pixel(self, geo_utm):
        """Forward then inverse must agree to < 1e-4 pixel (1 mm ground).

        pyproj UTM-WGS84 round-trip accuracy is sub-millimeter; the affine
        forward/inverse is exact linear algebra.
        """
        orig_row, orig_col = 500.0, 500.0
        lat, lon, h = geo_utm.image_to_latlon(orig_row, orig_col)
        row_back, col_back = geo_utm.latlon_to_image(lat, lon)

        assert row_back == pytest.approx(orig_row, abs=1e-4)
        assert col_back == pytest.approx(orig_col, abs=1e-4)

    def test_round_trip_batch(self, geo_utm):
        """Batch round-trip within 1e-4 pixel (1 mm) across 200 random points.

        pyproj UTM-WGS84 round-trip accuracy is sub-millimeter.
        """
        rng = np.random.default_rng(99)
        rows = rng.uniform(0, 999, 200)
        cols = rng.uniform(0, 999, 200)

        lats, lons, _ = geo_utm.image_to_latlon(rows, cols)
        rows_back, cols_back = geo_utm.latlon_to_image(lats, lons)

        np.testing.assert_allclose(rows_back, rows, atol=1e-4)
        np.testing.assert_allclose(cols_back, cols, atol=1e-4)

    def test_output_crs_is_wgs84(self, geo_utm):
        """Output CRS must always be WGS84 regardless of native CRS."""
        assert geo_utm.crs == 'WGS84'

    def test_native_crs_preserved(self, geo_utm):
        """Native CRS attribute preserves the original UTM string."""
        assert geo_utm.native_crs == 'EPSG:32756'

    def test_footprint_southern_hemisphere(self, geo_utm):
        """Footprint should report negative latitudes for southern hemisphere."""
        footprint = geo_utm.get_footprint()
        assert footprint['type'] == 'Polygon'

        _, _, _, max_lat = footprint['bounds']
        assert max_lat < 0.0, "UTM 56S should produce negative latitudes"


@pytest.mark.skipif(not _HAS_AFFINE, reason="rasterio/pyproj not installed")
class TestAffineGeolocationEdgeCases:
    """Edge-case tests for AffineGeolocation."""

    def test_invalid_transform_type_raises(self):
        """Passing a non-Affine transform must raise TypeError."""
        with pytest.raises(TypeError, match="rasterio.transform.Affine"):
            AffineGeolocation(
                transform=(1.0, 0.0, 0.0, 0.0, -1.0, 100.0),  # tuple, not Affine
                shape=(100, 100),
                crs='EPSG:4326',
            )

    def test_polar_utm_zone(self):
        """UTM Zone 33N (Norway/Arctic) — high latitude geolocation.

        Tests that pyproj handles high-latitude reprojection correctly.
        """
        transform = Affine(10.0, 0.0, 500000.0,
                           0.0, -10.0, 7800000.0)
        geo = AffineGeolocation(transform, (500, 500), 'EPSG:32633')

        lat, lon, h = geo.image_to_latlon(0.0, 0.0)
        # Northing 7800000 in UTM 33N → approximately 70°N
        assert 65.0 < lat < 75.0, f"Expected Arctic latitude, got {lat}"

        # Round-trip
        row_back, col_back = geo.latlon_to_image(lat, lon)
        assert row_back == pytest.approx(0.0, abs=1e-4)
        assert col_back == pytest.approx(0.0, abs=1e-4)

    def test_equatorial_utm(self):
        """UTM Zone 37N (East Africa, equatorial) — near zero latitude."""
        transform = Affine(30.0, 0.0, 200000.0,
                           0.0, -30.0, 100000.0)
        geo = AffineGeolocation(transform, (200, 200), 'EPSG:32637')

        lat, lon, h = geo.image_to_latlon(0.0, 0.0)
        # Northing 100000 in UTM 37N → approximately 0.9°N
        assert -5.0 < lat < 5.0

        row_back, col_back = geo.latlon_to_image(lat, lon)
        assert row_back == pytest.approx(0.0, abs=1e-4)
        assert col_back == pytest.approx(0.0, abs=1e-4)

    def test_stacked_array_dispatch(self):
        """Verify (2, N) stacked-array dispatch works with AffineGeolocation."""
        transform = Affine(0.001, 0.0, -118.0,
                           0.0, -0.001, 35.0)
        geo = AffineGeolocation(transform, (500, 500), 'EPSG:4326')

        points = np.array([[0.0, 100.0, 499.0],
                           [0.0, 100.0, 499.0]])
        result = geo.image_to_latlon(points)

        assert result.shape == (3, 3)
        # First row = lats, second = lons, third = heights
        assert result[0, 0] == pytest.approx(35.0, abs=1e-12)
        assert result[1, 0] == pytest.approx(-118.0, abs=1e-12)

    def test_height_propagation(self):
        """Custom height argument propagates to output."""
        transform = Affine(0.001, 0.0, 0.0, 0.0, -0.001, 0.0)
        geo = AffineGeolocation(transform, (100, 100), 'EPSG:4326')

        _, _, h = geo.image_to_latlon(50.0, 50.0, height=500.0)
        assert h == pytest.approx(500.0)
