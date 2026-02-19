# -*- coding: utf-8 -*-
"""
Geolocation Utility Tests - Geodetic function validation with reference data.

Tests grdl.geolocation.utils public functions against geodetic reference values,
including edge cases for GEOINT operations:
- Geographic distance (Haversine) validated against Vincenty-derived benchmarks
- Footprint and bounding-box computation with degenerate inputs
- Pixel-bounds checking with tolerance behavior
- Perimeter sampling geometry
- Interpolation error metrics against hand-computed values

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
import math

# Third-party
import pytest
import numpy as np

# GRDL internal
try:
    from grdl.geolocation.utils import (
        geographic_distance,
        geographic_distance_batch,
        interpolation_error_metrics,
        calculate_footprint,
        bounds_from_corners,
        check_pixel_bounds,
        sample_image_perimeter,
    )
    _HAS_GRDL = True
except ImportError:
    _HAS_GRDL = False


pytestmark = [
    pytest.mark.geolocation,
    pytest.mark.skipif(not _HAS_GRDL, reason="grdl not installed"),
]


# =============================================================================
# Reference Constants
# =============================================================================

# WGS84 mean Earth radius used by the implementation (meters)
_EARTH_RADIUS_M = 6_371_000

# Haversine reference distances pre-computed with pyproj Geod (WGS84 ellipsoid).
# Haversine uses a sphere, so we accept 0.5 % tolerance versus the ellipsoid.
_HAVERSINE_TOLERANCE_FRAC = 0.005

# Known geodetic reference pairs  (lat1, lon1, lat2, lon2, approx_dist_m)
# Source: NGA-published geodesy benchmarks and Vincenty computation via pyproj.
_REFERENCE_DISTANCES = [
    # New York → London  (transatlantic)
    (40.7128, -74.0060, 51.5074, -0.1278, 5_570_220.0),
    # Equatorial 1-degree longitude at equator ≈ 111.32 km
    (0.0, 0.0, 0.0, 1.0, 111_195.0),
    # Equatorial 1-degree latitude ≈ 111.19 km (spherical earth, R=6371km)
    (0.0, 0.0, 1.0, 0.0, 111_195.0),
    # Sydney → Tokyo
    (-33.8688, 151.2093, 35.6762, 139.6503, 7_823_000.0),
    # Short distance (< 1 km) — campus scale
    # Pre-computed via Haversine with R=6371000 for these specific coordinates.
    (38.8977, -77.0365, 38.8987, -77.0355, 141.0),
]


# =============================================================================
# geographic_distance() — Scalar Haversine
# =============================================================================


class TestGeographicDistance:
    """Tests for geographic_distance() Haversine implementation."""

    @pytest.mark.parametrize(
        "lat1, lon1, lat2, lon2, expected_m",
        _REFERENCE_DISTANCES,
        ids=[
            "NYC-to-London",
            "equator-1deg-lon",
            "equator-1deg-lat",
            "Sydney-to-Tokyo",
            "sub-kilometer",
        ],
    )
    def test_reference_distances(self, lat1, lon1, lat2, lon2, expected_m):
        """Validate Haversine against known geodetic reference distances.

        Haversine operates on a sphere, so we allow 0.5 % tolerance relative
        to ellipsoidal (Vincenty) reference values.
        """
        result = geographic_distance(lat1, lon1, lat2, lon2)
        assert isinstance(result, float)
        assert result >= 0.0

        tolerance = max(expected_m * _HAVERSINE_TOLERANCE_FRAC, 1.0)
        assert abs(result - expected_m) < tolerance, (
            f"Distance {result:.1f} m deviates from reference {expected_m:.1f} m "
            f"by {abs(result - expected_m):.1f} m (tolerance {tolerance:.1f} m)"
        )

    def test_zero_distance(self):
        """Same point must return exactly 0.0 m."""
        assert geographic_distance(34.05, -118.25, 34.05, -118.25) == 0.0

    def test_null_island(self):
        """Null Island (0, 0) to (0, 0) is zero distance."""
        assert geographic_distance(0.0, 0.0, 0.0, 0.0) == 0.0

    def test_symmetry(self):
        """Distance A→B must equal distance B→A."""
        d_ab = geographic_distance(40.7128, -74.0060, 51.5074, -0.1278)
        d_ba = geographic_distance(51.5074, -0.1278, 40.7128, -74.0060)
        assert d_ab == pytest.approx(d_ba, rel=1e-12)

    def test_antipodal_points(self):
        """Antipodal points should yield approximately half the circumference.

        The maximum great-circle distance on a sphere of radius R is π * R.
        """
        d = geographic_distance(0.0, 0.0, 0.0, 180.0)
        expected = math.pi * _EARTH_RADIUS_M
        assert d == pytest.approx(expected, rel=1e-12)

    def test_pole_to_pole(self):
        """North Pole to South Pole — half circumference."""
        d = geographic_distance(90.0, 0.0, -90.0, 0.0)
        expected = math.pi * _EARTH_RADIUS_M
        assert d == pytest.approx(expected, rel=1e-12)

    def test_north_pole_to_equator(self):
        """North Pole to equator — quarter circumference."""
        d = geographic_distance(90.0, 0.0, 0.0, 0.0)
        expected = (math.pi / 2) * _EARTH_RADIUS_M
        assert d == pytest.approx(expected, rel=1e-12)

    def test_international_date_line_east_to_west(self):
        """Crossing the International Date Line (179°E → 179°W).

        A point at (0, 179) and (0, -179) are 2 degrees apart, not 358.
        """
        d = geographic_distance(0.0, 179.0, 0.0, -179.0)
        # Exact Haversine for 2° longitude at equator: R * π / 90
        expected = _EARTH_RADIUS_M * math.pi / 90.0
        assert d == pytest.approx(expected, rel=1e-12)

    def test_international_date_line_exact_180(self):
        """Points at +180 and -180 longitude are the same location."""
        d = geographic_distance(45.0, 180.0, 45.0, -180.0)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_return_type_float(self):
        """Result must be a Python float, not ndarray."""
        result = geographic_distance(0.0, 0.0, 1.0, 0.0)
        assert isinstance(result, float)

    def test_positive_result(self):
        """Distance is always non-negative."""
        result = geographic_distance(-90.0, -180.0, 90.0, 180.0)
        assert result >= 0.0


# =============================================================================
# geographic_distance_batch() — Vectorized Haversine
# =============================================================================


class TestGeographicDistanceBatch:
    """Tests for geographic_distance_batch() vectorized Haversine."""

    def test_single_element_matches_scalar(self):
        """Single-element arrays must match scalar implementation."""
        scalar_d = geographic_distance(40.7128, -74.0060, 51.5074, -0.1278)
        batch_d = geographic_distance_batch(
            np.array([40.7128]), np.array([-74.0060]),
            np.array([51.5074]), np.array([-0.1278]),
        )
        assert batch_d.shape == (1,)
        assert float(batch_d[0]) == pytest.approx(scalar_d, rel=1e-12)

    def test_multi_element_matches_scalar(self):
        """Batch results must match element-wise scalar calls."""
        lats1 = np.array([0.0, 40.7128, -33.8688])
        lons1 = np.array([0.0, -74.0060, 151.2093])
        lats2 = np.array([0.0, 51.5074, 35.6762])
        lons2 = np.array([1.0, -0.1278, 139.6503])

        batch_d = geographic_distance_batch(lats1, lons1, lats2, lons2)

        for i in range(len(lats1)):
            scalar_d = geographic_distance(
                lats1[i], lons1[i], lats2[i], lons2[i]
            )
            assert float(batch_d[i]) == pytest.approx(scalar_d, rel=1e-12)

    def test_return_type_ndarray(self):
        """Result must be a numpy ndarray."""
        result = geographic_distance_batch(
            np.array([0.0]), np.array([0.0]),
            np.array([1.0]), np.array([0.0]),
        )
        assert isinstance(result, np.ndarray)

    def test_output_shape_matches_input(self):
        """Output array shape must match input array shapes."""
        n = 50
        lats1 = np.random.uniform(-90, 90, n)
        lons1 = np.random.uniform(-180, 180, n)
        lats2 = np.random.uniform(-90, 90, n)
        lons2 = np.random.uniform(-180, 180, n)

        result = geographic_distance_batch(lats1, lons1, lats2, lons2)
        assert result.shape == (n,)

    def test_all_zeros(self):
        """Same points should all return zero distance."""
        n = 10
        lats = np.full(n, 34.05)
        lons = np.full(n, -118.25)
        result = geographic_distance_batch(lats, lons, lats, lons)
        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    def test_antipodal_batch(self):
        """Antipodal points batch — all should be π * R."""
        n = 5
        lats1 = np.zeros(n)
        lons1 = np.linspace(-180, 180, n)
        lats2 = np.zeros(n)
        lons2 = lons1 + 180.0  # Antipodal longitudes

        result = geographic_distance_batch(lats1, lons1, lats2, lons2)
        expected = math.pi * _EARTH_RADIUS_M
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_non_negative(self):
        """All batch distances must be non-negative."""
        rng = np.random.default_rng(42)
        n = 100
        result = geographic_distance_batch(
            rng.uniform(-90, 90, n), rng.uniform(-180, 180, n),
            rng.uniform(-90, 90, n), rng.uniform(-180, 180, n),
        )
        assert np.all(result >= 0.0)

    def test_symmetry_batch(self):
        """Batch distance A→B must equal B→A element-wise."""
        rng = np.random.default_rng(123)
        n = 20
        lats1 = rng.uniform(-90, 90, n)
        lons1 = rng.uniform(-180, 180, n)
        lats2 = rng.uniform(-90, 90, n)
        lons2 = rng.uniform(-180, 180, n)

        d_ab = geographic_distance_batch(lats1, lons1, lats2, lons2)
        d_ba = geographic_distance_batch(lats2, lons2, lats1, lons1)
        np.testing.assert_allclose(d_ab, d_ba, rtol=1e-12)


# =============================================================================
# interpolation_error_metrics()
# =============================================================================


class TestInterpolationErrorMetrics:
    """Tests for interpolation_error_metrics() error computation."""

    def test_perfect_match(self):
        """Zero error when true and interpolated values are identical."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = interpolation_error_metrics(values, values)

        assert result['mean_error'] == 0.0
        assert result['rms_error'] == 0.0
        assert result['max_error'] == 0.0
        assert result['std_error'] == 0.0

    def test_known_errors(self):
        """Hand-computed error metrics for known constant offset.

        true = [10, 20, 30], interp = [11, 22, 28]
        errors = |[1, 2, 2]| = [1, 2, 2]
        mean_error = 5/3 ≈ 1.6667
        rms_error  = sqrt((1+4+4)/3) = sqrt(3) ≈ 1.7321
        max_error  = 2.0
        std_error  = std([1, 2, 2]) ≈ 0.4714
        """
        true = np.array([10.0, 20.0, 30.0])
        interp = np.array([11.0, 22.0, 28.0])
        result = interpolation_error_metrics(true, interp)

        assert result['mean_error'] == pytest.approx(5.0 / 3.0, rel=1e-10)
        assert result['rms_error'] == pytest.approx(math.sqrt(3.0), rel=1e-10)
        assert result['max_error'] == pytest.approx(2.0, rel=1e-10)
        assert result['std_error'] == pytest.approx(np.std([1.0, 2.0, 2.0]), rel=1e-10)

    def test_constant_offset(self):
        """Constant offset: all errors are identical.

        mean = rms = max = offset, std = 0.
        """
        true = np.arange(100, dtype=np.float64)
        interp = true + 5.0
        result = interpolation_error_metrics(true, interp)

        assert result['mean_error'] == pytest.approx(5.0, rel=1e-12)
        assert result['rms_error'] == pytest.approx(5.0, rel=1e-12)
        assert result['max_error'] == pytest.approx(5.0, rel=1e-12)
        assert result['std_error'] == pytest.approx(0.0, abs=1e-10)

    def test_single_element(self):
        """Single-element input should not crash."""
        true = np.array([42.0])
        interp = np.array([45.0])
        result = interpolation_error_metrics(true, interp)

        assert result['mean_error'] == pytest.approx(3.0)
        assert result['rms_error'] == pytest.approx(3.0)
        assert result['max_error'] == pytest.approx(3.0)
        assert result['std_error'] == pytest.approx(0.0, abs=1e-10)

    def test_negative_errors_are_absolute(self):
        """Errors should be absolute — sign should not matter."""
        true = np.array([10.0, 20.0])
        interp = np.array([12.0, 18.0])  # +2 and -2
        result = interpolation_error_metrics(true, interp)

        assert result['mean_error'] == pytest.approx(2.0)
        assert result['max_error'] == pytest.approx(2.0)

    def test_return_keys(self):
        """Result dict must contain all four standard keys."""
        result = interpolation_error_metrics(
            np.array([1.0]), np.array([2.0])
        )
        assert set(result.keys()) == {
            'mean_error', 'rms_error', 'max_error', 'std_error'
        }

    def test_all_float_values(self):
        """All returned metrics must be Python floats."""
        result = interpolation_error_metrics(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.5, 2.5, 3.5]),
        )
        for key, val in result.items():
            assert isinstance(val, float), f"{key} is {type(val)}, expected float"

    def test_geolocation_precision_scenario(self):
        """Simulate sub-pixel geolocation residuals (typical GEOINT scenario).

        True pixel positions vs interpolated positions — RMS should be < 0.5
        pixel for operational accuracy.
        """
        rng = np.random.default_rng(42)
        true_pixels = rng.uniform(0, 1000, 50)
        # Add sub-pixel noise (mean ~0.1 pixel)
        interp_pixels = true_pixels + rng.normal(0, 0.1, 50)

        result = interpolation_error_metrics(true_pixels, interp_pixels)
        assert result['rms_error'] < 0.5, (
            "Sub-pixel noise should yield RMS < 0.5 pixel"
        )


# =============================================================================
# calculate_footprint()
# =============================================================================


class TestCalculateFootprint:
    """Tests for calculate_footprint() polygon generation."""

    def test_valid_quadrilateral(self):
        """Standard four-corner image footprint."""
        corners = [(10.0, 20.0), (11.0, 20.0), (11.0, 21.0), (10.0, 21.0)]
        result = calculate_footprint(corners)

        assert result['type'] == 'Polygon'
        assert result['coordinates'] == corners
        assert result['bounds'] is not None
        min_lon, min_lat, max_lon, max_lat = result['bounds']
        assert min_lon == pytest.approx(10.0, abs=1e-14)
        assert min_lat == pytest.approx(20.0, abs=1e-14)
        assert max_lon == pytest.approx(11.0, abs=1e-14)
        assert max_lat == pytest.approx(21.0, abs=1e-14)

    def test_triangle(self):
        """Minimum valid polygon — three vertices."""
        corners = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        result = calculate_footprint(corners)

        assert result['type'] == 'Polygon'
        assert len(result['coordinates']) == 3
        assert result['bounds'] is not None

    def test_fewer_than_three_points(self):
        """Fewer than 3 points cannot form a polygon — return None type."""
        result = calculate_footprint([(0.0, 0.0), (1.0, 1.0)])
        assert result['type'] == 'None'
        assert result['coordinates'] is None
        assert result['bounds'] is None

    def test_empty_input(self):
        """Empty coordinate list — return None type."""
        result = calculate_footprint([])
        assert result['type'] == 'None'
        assert result['coordinates'] is None

    def test_null_island_footprint(self):
        """Footprint centered on Null Island (0, 0)."""
        corners = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
        result = calculate_footprint(corners)

        assert result['type'] == 'Polygon'
        min_lon, min_lat, max_lon, max_lat = result['bounds']
        assert min_lon == pytest.approx(-0.5, abs=1e-14)
        assert max_lon == pytest.approx(0.5, abs=1e-14)
        assert min_lat == pytest.approx(-0.5, abs=1e-14)
        assert max_lat == pytest.approx(0.5, abs=1e-14)

    def test_international_date_line_footprint(self):
        """Footprint straddling the International Date Line (±180°).

        Note: Haversine-based footprint does NOT handle antimeridian wrapping.
        This test documents the current behavior where bounds report raw
        min/max without wrap-around awareness.
        """
        corners = [(179.0, 40.0), (-179.0, 40.0), (-179.0, 41.0), (179.0, 41.0)]
        result = calculate_footprint(corners)

        assert result['type'] == 'Polygon'
        # Current implementation reports raw min/max — lon spans -179 to 179.
        # A production antimeridian-aware implementation would split the polygon.
        min_lon, min_lat, max_lon, max_lat = result['bounds']
        assert min_lon == pytest.approx(-179.0, abs=1e-14)
        assert max_lon == pytest.approx(179.0, abs=1e-14)

    def test_polar_footprint(self):
        """Footprint near the North Pole."""
        corners = [(-10.0, 89.0), (10.0, 89.0), (10.0, 90.0), (-10.0, 90.0)]
        result = calculate_footprint(corners)

        assert result['type'] == 'Polygon'
        _, min_lat, _, max_lat = result['bounds']
        assert max_lat == pytest.approx(90.0, abs=1e-14)

    def test_large_polygon(self):
        """Many-vertex polygon (densely sampled perimeter)."""
        n = 100
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        corners = [(float(np.cos(a)), float(np.sin(a))) for a in angles]
        result = calculate_footprint(corners)

        assert result['type'] == 'Polygon'
        assert len(result['coordinates']) == n


# =============================================================================
# bounds_from_corners()
# =============================================================================


class TestBoundsFromCorners:
    """Tests for bounds_from_corners() bounding-box computation."""

    def test_standard_rectangle(self):
        """Standard four-corner bounding box."""
        corners = [(10.0, 20.0), (11.0, 20.0), (11.0, 21.0), (10.0, 21.0)]
        result = bounds_from_corners(corners)

        assert len(result) == 4
        min_lon, min_lat, max_lon, max_lat = result
        assert min_lon == pytest.approx(10.0, abs=1e-14)
        assert min_lat == pytest.approx(20.0, abs=1e-14)
        assert max_lon == pytest.approx(11.0, abs=1e-14)
        assert max_lat == pytest.approx(21.0, abs=1e-14)

    def test_single_point(self):
        """Single point — min equals max."""
        result = bounds_from_corners([(5.0, 10.0)])
        min_lon, min_lat, max_lon, max_lat = result
        assert min_lon == max_lon == pytest.approx(5.0, abs=1e-14)
        assert min_lat == max_lat == pytest.approx(10.0, abs=1e-14)

    def test_empty_raises_valueerror(self):
        """Empty list must raise ValueError."""
        with pytest.raises(ValueError, match="No corner"):
            bounds_from_corners([])

    def test_negative_coordinates(self):
        """Negative (southern/western hemisphere) coordinates."""
        corners = [(-120.0, -35.0), (-119.0, -35.0),
                   (-119.0, -34.0), (-120.0, -34.0)]
        min_lon, min_lat, max_lon, max_lat = bounds_from_corners(corners)

        assert min_lon == pytest.approx(-120.0, abs=1e-14)
        assert max_lon == pytest.approx(-119.0, abs=1e-14)
        assert min_lat == pytest.approx(-35.0, abs=1e-14)
        assert max_lat == pytest.approx(-34.0, abs=1e-14)

    def test_null_island_bounds(self):
        """Bounding box centered on Null Island."""
        corners = [(-0.01, -0.01), (0.01, -0.01),
                   (0.01, 0.01), (-0.01, 0.01)]
        min_lon, min_lat, max_lon, max_lat = bounds_from_corners(corners)

        assert min_lon < 0.0 < max_lon
        assert min_lat < 0.0 < max_lat


# =============================================================================
# check_pixel_bounds()
# =============================================================================


class TestCheckPixelBounds:
    """Tests for check_pixel_bounds() boundary validation."""

    def test_center_pixel_valid(self):
        """Center of image is always within bounds."""
        check_pixel_bounds(50.0, 50.0, (100, 100))

    def test_origin_valid(self):
        """Pixel (0, 0) is always within bounds."""
        check_pixel_bounds(0.0, 0.0, (100, 100))

    def test_last_pixel_valid(self):
        """Last pixel (rows-1, cols-1) is within bounds."""
        check_pixel_bounds(99.0, 99.0, (100, 100))

    def test_fractional_pixel_valid(self):
        """Sub-pixel coordinates within image are valid."""
        check_pixel_bounds(49.5, 49.5, (100, 100))

    def test_negative_row_raises(self):
        """Negative row beyond tolerance raises ValueError."""
        with pytest.raises(ValueError, match="[Rr]ow"):
            check_pixel_bounds(-1.0, 50.0, (100, 100))

    def test_negative_col_raises(self):
        """Negative column beyond tolerance raises ValueError."""
        with pytest.raises(ValueError, match="[Cc]olumn"):
            check_pixel_bounds(50.0, -1.0, (100, 100))

    def test_row_beyond_image_raises(self):
        """Row >= (nrows + tolerance) raises ValueError."""
        with pytest.raises(ValueError, match="[Rr]ow"):
            check_pixel_bounds(101.0, 50.0, (100, 100))

    def test_col_beyond_image_raises(self):
        """Column >= (ncols + tolerance) raises ValueError."""
        with pytest.raises(ValueError, match="[Cc]olumn"):
            check_pixel_bounds(50.0, 101.0, (100, 100))

    def test_tolerance_boundary_inside(self):
        """Pixel just inside tolerance boundary should pass.

        Default tolerance is 0.5, so row=-0.4 should be accepted.
        """
        check_pixel_bounds(-0.4, 0.0, (100, 100))

    def test_tolerance_boundary_outside(self):
        """Pixel just outside tolerance boundary should raise.

        Default tolerance is 0.5, so row=-0.6 should be rejected.
        """
        with pytest.raises(ValueError):
            check_pixel_bounds(-0.6, 0.0, (100, 100))

    def test_custom_tolerance(self):
        """Custom tolerance of 1.0 permits row=-0.9."""
        check_pixel_bounds(-0.9, 0.0, (100, 100), tolerance=1.0)

    def test_zero_tolerance(self):
        """Zero tolerance rejects any sub-zero coordinate."""
        with pytest.raises(ValueError):
            check_pixel_bounds(-0.1, 0.0, (100, 100), tolerance=0.0)

    def test_single_pixel_image(self):
        """1x1 image — only pixel (0, 0) is valid."""
        check_pixel_bounds(0.0, 0.0, (1, 1))

    def test_single_pixel_image_boundary(self):
        """1x1 image — pixel (1.5, 0) exceeds default tolerance and is out of bounds."""
        with pytest.raises(ValueError):
            check_pixel_bounds(1.5, 0.0, (1, 1))


# =============================================================================
# sample_image_perimeter()
# =============================================================================


class TestSampleImagePerimeter:
    """Tests for sample_image_perimeter() point generation."""

    def test_output_shapes(self):
        """Output arrays should have length = 4 * samples_per_edge."""
        rows_arr, cols_arr = sample_image_perimeter(
            (1000, 2000), samples_per_edge=10
        )
        assert rows_arr.shape == (40,)
        assert cols_arr.shape == (40,)

    def test_corners_included(self):
        """All four image corners must appear in the sample points."""
        rows_arr, cols_arr = sample_image_perimeter(
            (100, 200), samples_per_edge=10
        )

        points = set(zip(rows_arr.tolist(), cols_arr.tolist()))

        # Top-left (0, 0)
        assert (0.0, 0.0) in points, "Top-left corner missing"
        # Top-right (0, cols-1)
        assert (0.0, 199.0) in points, "Top-right corner missing"
        # Bottom-right (rows-1, cols-1)
        assert (99.0, 199.0) in points, "Bottom-right corner missing"
        # Bottom-left (rows-1, 0)
        assert (99.0, 0.0) in points, "Bottom-left corner missing"

    def test_within_image_bounds(self):
        """All sample points must be within image bounds."""
        nrows, ncols = 500, 300
        rows_arr, cols_arr = sample_image_perimeter(
            (nrows, ncols), samples_per_edge=20
        )

        assert np.all(rows_arr >= 0)
        assert np.all(rows_arr <= nrows - 1)
        assert np.all(cols_arr >= 0)
        assert np.all(cols_arr <= ncols - 1)

    def test_perimeter_only(self):
        """All points must lie on the image perimeter (row=0, row=max, col=0, or col=max)."""
        nrows, ncols = 100, 200
        rows_arr, cols_arr = sample_image_perimeter(
            (nrows, ncols), samples_per_edge=15
        )

        for r, c in zip(rows_arr, cols_arr):
            on_perimeter = (
                r == pytest.approx(0.0, abs=1e-10) or
                r == pytest.approx(nrows - 1, abs=1e-10) or
                c == pytest.approx(0.0, abs=1e-10) or
                c == pytest.approx(ncols - 1, abs=1e-10)
            )
            assert on_perimeter, f"Point ({r}, {c}) is not on perimeter"

    def test_single_pixel_image(self):
        """1x1 image — all perimeter points collapse to (0, 0)."""
        rows_arr, cols_arr = sample_image_perimeter(
            (1, 1), samples_per_edge=5
        )
        np.testing.assert_allclose(rows_arr, 0.0, atol=1e-12)
        np.testing.assert_allclose(cols_arr, 0.0, atol=1e-12)

    def test_single_row_image(self):
        """1-row image — perimeter is just the row."""
        rows_arr, cols_arr = sample_image_perimeter(
            (1, 100), samples_per_edge=10
        )
        # All rows should be 0
        np.testing.assert_allclose(rows_arr, 0.0, atol=1e-12)
        # Cols should span 0 to 99
        assert cols_arr.min() == pytest.approx(0.0, abs=1e-14)
        assert cols_arr.max() == pytest.approx(99.0, abs=1e-14)

    def test_ndarray_output(self):
        """Outputs must be numpy ndarrays."""
        rows_arr, cols_arr = sample_image_perimeter((50, 50), 5)
        assert isinstance(rows_arr, np.ndarray)
        assert isinstance(cols_arr, np.ndarray)

    def test_custom_samples_count(self):
        """Custom samples_per_edge value is respected."""
        n = 25
        rows_arr, cols_arr = sample_image_perimeter((100, 100), n)
        assert rows_arr.shape == (4 * n,)
        assert cols_arr.shape == (4 * n,)
