# -*- coding: utf-8 -*-
"""
Affine Geolocation Real-Data Tests - Landsat pixel-to-geographic validation.

Tests grdl.geolocation.eo.affine.AffineGeolocation against real Landsat 8/9
COG files, validating:
- Level 1: Reader-to-geolocation construction, forward/inverse at corners
- Level 2: Sub-pixel round-trip precision, CRS consistency, coordinate sanity
- Level 3: Integration with footprint calculation and geographic distance

Dataset: Landsat 8/9 Collection 2 Surface Reflectance (COG)

Dependencies
------------
pytest
numpy
rasterio
pyproj
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
from pathlib import Path

# Third-party
import pytest
import numpy as np

try:
    import rasterio
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False

try:
    import pyproj
    _HAS_PYPROJ = True
except ImportError:
    _HAS_PYPROJ = False

# GRDL internal
try:
    from grdl.IO.geotiff import GeoTIFFReader
    from grdl.geolocation.eo.affine import AffineGeolocation
    _HAS_GRDL = True
except ImportError:
    _HAS_GRDL = False

try:
    from grdl.geolocation.utils import geographic_distance, geographic_distance_batch
    _HAS_GEO_UTILS = True
except ImportError:
    _HAS_GEO_UTILS = False


pytestmark = [
    pytest.mark.geolocation,
    pytest.mark.landsat,
    pytest.mark.requires_data,
    pytest.mark.skipif(not _HAS_RASTERIO, reason="rasterio not installed"),
    pytest.mark.skipif(not _HAS_PYPROJ, reason="pyproj not installed"),
    pytest.mark.skipif(not _HAS_GRDL, reason="grdl not installed"),
]


# =============================================================================
# Level 1: Geolocation Construction and Basic Forward/Inverse
# =============================================================================


@pytest.mark.slow
def test_affine_from_reader_construction(require_landsat_file):
    """AffineGeolocation.from_reader() constructs without error.

    Validates that the reader's metadata contains the required affine
    transform and CRS, and that the resulting geolocation object has
    a valid shape and WGS84 output CRS.
    """
    with GeoTIFFReader(str(require_landsat_file)) as reader:
        geo = AffineGeolocation.from_reader(reader)

        assert geo.shape[0] > 0 and geo.shape[1] > 0
        assert geo.crs == 'WGS84'
        assert geo.native_crs is not None


@pytest.mark.slow
def test_affine_forward_at_corners(require_landsat_file):
    """Forward geolocation at image corners produces valid WGS84 coordinates.

    All four corner pixels must map to latitudes in [-90, 90] and
    longitudes in [-180, 180]. The coordinates must not be NaN.
    """
    with GeoTIFFReader(str(require_landsat_file)) as reader:
        geo = AffineGeolocation.from_reader(reader)
        nrows, ncols = geo.shape

        corners = [
            (0.0, 0.0),
            (0.0, float(ncols - 1)),
            (float(nrows - 1), 0.0),
            (float(nrows - 1), float(ncols - 1)),
        ]

        for row, col in corners:
            lat, lon, h = geo.image_to_latlon(row, col)

            assert not np.isnan(lat), f"NaN lat at pixel ({row}, {col})"
            assert not np.isnan(lon), f"NaN lon at pixel ({row}, {col})"
            assert -90.0 <= lat <= 90.0, f"Lat {lat} out of range at ({row}, {col})"
            assert -180.0 <= lon <= 180.0, f"Lon {lon} out of range at ({row}, {col})"


@pytest.mark.slow
def test_affine_inverse_at_center(require_landsat_file):
    """Inverse geolocation at image center returns pixel near center.

    Forward-transform the center pixel to lat/lon, then inverse-transform
    back. The result must be within 3.4e-5 pixel (1 mm for 30 m pixels).
    """
    with GeoTIFFReader(str(require_landsat_file)) as reader:
        geo = AffineGeolocation.from_reader(reader)
        nrows, ncols = geo.shape

        center_row = nrows / 2.0
        center_col = ncols / 2.0

        lat, lon, h = geo.image_to_latlon(center_row, center_col)
        row_back, col_back = geo.latlon_to_image(lat, lon)

        assert row_back == pytest.approx(center_row, abs=3.4e-5), (
            f"Row round-trip error: {abs(row_back - center_row):.8f} pixels"
        )
        assert col_back == pytest.approx(center_col, abs=3.4e-5), (
            f"Col round-trip error: {abs(col_back - center_col):.8f} pixels"
        )


# =============================================================================
# Level 2: Precision and Data Quality
# =============================================================================


@pytest.mark.slow
def test_affine_round_trip_all_corners(require_landsat_file):
    """Round-trip pixel-latlon-pixel at all four corners within 3.4e-5 pixel (1 mm).

    Sub-pixel accuracy at image extremes is critical — corner distortion
    indicates affine or CRS transform errors. pyproj round-trip is sub-mm.
    """
    with GeoTIFFReader(str(require_landsat_file)) as reader:
        geo = AffineGeolocation.from_reader(reader)
        nrows, ncols = geo.shape

        test_pixels = np.array([
            [0.0, 0.0],
            [0.0, ncols - 1.0],
            [nrows - 1.0, 0.0],
            [nrows - 1.0, ncols - 1.0],
            [nrows / 2.0, ncols / 2.0],
        ])

        rows = test_pixels[:, 0]
        cols = test_pixels[:, 1]

        lats, lons, _ = geo.image_to_latlon(rows, cols)
        rows_back, cols_back = geo.latlon_to_image(lats, lons)

        max_row_err = np.max(np.abs(rows_back - rows))
        max_col_err = np.max(np.abs(cols_back - cols))

        assert max_row_err < 3.4e-5, f"Max row round-trip error: {max_row_err:.8f} px"
        assert max_col_err < 3.4e-5, f"Max col round-trip error: {max_col_err:.8f} px"


@pytest.mark.slow
def test_affine_round_trip_random_batch(require_landsat_file):
    """Round-trip 500 random interior pixels within 3.4e-5 pixel (1 mm ground).

    Samples the full image interior to detect localized transform errors
    that corner-only tests would miss. pyproj round-trip is sub-mm accurate.
    """
    with GeoTIFFReader(str(require_landsat_file)) as reader:
        geo = AffineGeolocation.from_reader(reader)
        nrows, ncols = geo.shape

        rng = np.random.default_rng(42)
        rows = rng.uniform(10, nrows - 10, 500)
        cols = rng.uniform(10, ncols - 10, 500)

        lats, lons, _ = geo.image_to_latlon(rows, cols)
        rows_back, cols_back = geo.latlon_to_image(lats, lons)

        rms_row = np.sqrt(np.mean((rows_back - rows) ** 2))
        rms_col = np.sqrt(np.mean((cols_back - cols) ** 2))

        assert rms_row < 3.4e-5, f"RMS row error: {rms_row:.8f} px"
        assert rms_col < 3.4e-5, f"RMS col error: {rms_col:.8f} px"


@pytest.mark.slow
def test_affine_crs_consistent_with_reader(require_landsat_file):
    """AffineGeolocation native_crs matches the reader's reported CRS.

    A CRS mismatch between the reader and the geolocation would produce
    coordinates in the wrong projection.
    """
    with GeoTIFFReader(str(require_landsat_file)) as reader:
        geo = AffineGeolocation.from_reader(reader)
        reader_crs = str(reader.metadata.crs)

        # The geolocation's native CRS must match the reader's CRS
        assert geo.native_crs is not None
        assert str(geo.native_crs) == reader_crs, (
            f"CRS mismatch: geolocation reports {geo.native_crs}, "
            f"reader reports {reader_crs}"
        )


@pytest.mark.slow
def test_affine_latitude_ordering(require_landsat_file):
    """Moving down in row should change latitude monotonically.

    For a north-up image, increasing row (moving south in the image)
    should decrease latitude. For south-up, it should increase.
    Either way, the relationship must be monotonic.
    """
    with GeoTIFFReader(str(require_landsat_file)) as reader:
        geo = AffineGeolocation.from_reader(reader)
        nrows, ncols = geo.shape

        center_col = ncols / 2.0
        rows_sample = np.linspace(10, nrows - 10, 20)
        cols_sample = np.full_like(rows_sample, center_col)

        lats, _, _ = geo.image_to_latlon(rows_sample, cols_sample)

        # Check monotonicity (either all increasing or all decreasing)
        diffs = np.diff(lats)
        all_increasing = np.all(diffs > 0)
        all_decreasing = np.all(diffs < 0)

        assert all_increasing or all_decreasing, (
            "Latitude must be monotonic along a column. "
            f"Got {np.sum(diffs > 0)} increasing, {np.sum(diffs < 0)} decreasing."
        )


@pytest.mark.slow
def test_affine_longitude_ordering(require_landsat_file):
    """Moving right in column should change longitude monotonically.

    For a standard east-up orientation, increasing column should
    increase longitude. Must be monotonic either way.
    """
    with GeoTIFFReader(str(require_landsat_file)) as reader:
        geo = AffineGeolocation.from_reader(reader)
        nrows, ncols = geo.shape

        center_row = nrows / 2.0
        cols_sample = np.linspace(10, ncols - 10, 20)
        rows_sample = np.full_like(cols_sample, center_row)

        _, lons, _ = geo.image_to_latlon(rows_sample, cols_sample)

        diffs = np.diff(lons)
        all_increasing = np.all(diffs > 0)
        all_decreasing = np.all(diffs < 0)

        assert all_increasing or all_decreasing, (
            "Longitude must be monotonic along a row."
        )


@pytest.mark.slow
def test_affine_pixel_spacing_consistent(require_landsat_file):
    """Adjacent pixels should map to consistent ground distance.

    For Landsat (30m pixels), adjacent pixels along a row should be
    approximately 30m apart on the ground. We allow 10% tolerance for
    projection distortion.
    """
    with GeoTIFFReader(str(require_landsat_file)) as reader:
        geo = AffineGeolocation.from_reader(reader)
        nrows, ncols = geo.shape

        # Sample 10 adjacent pixel pairs near center
        center_row = nrows / 2.0
        cols = np.array([ncols / 2.0 + i for i in range(11)])
        rows = np.full_like(cols, center_row)

        lats, lons, _ = geo.image_to_latlon(rows, cols)

        if _HAS_GEO_UTILS:
            distances = geographic_distance_batch(
                lats[:-1], lons[:-1], lats[1:], lons[1:]
            )

            # Landsat pixel size is typically 30m
            median_spacing = np.median(distances)
            assert 10.0 < median_spacing < 100.0, (
                f"Pixel spacing {median_spacing:.1f}m outside plausible range "
                "for Landsat (expected ~30m)"
            )

            # All spacings should be similar (< 10% deviation from median)
            max_deviation = np.max(np.abs(distances - median_spacing))
            assert max_deviation < 0.1 * median_spacing, (
                f"Pixel spacing varies by {max_deviation:.1f}m "
                f"(> 10% of median {median_spacing:.1f}m)"
            )


# =============================================================================
# Level 3: Integration with Geolocation Utilities
# =============================================================================


@pytest.mark.slow
@pytest.mark.integration
def test_affine_footprint_bounds_sanity(require_landsat_file):
    """get_footprint() should return a valid polygon with sensible bounds.

    The footprint bounding box should span a reasonable area for a single
    Landsat scene (roughly 170 km × 185 km swath).
    """
    with GeoTIFFReader(str(require_landsat_file)) as reader:
        geo = AffineGeolocation.from_reader(reader)
        footprint = geo.get_footprint()

        assert footprint['type'] == 'Polygon'
        assert footprint['coordinates'] is not None
        assert len(footprint['coordinates']) > 0

        min_lon, min_lat, max_lon, max_lat = footprint['bounds']

        # Bounds should be ordered
        assert min_lon < max_lon
        assert min_lat < max_lat

        # A single Landsat band/file spans < 5 degrees in each direction
        lon_span = max_lon - min_lon
        lat_span = max_lat - min_lat
        assert 0.01 < lon_span < 10.0, f"Longitude span {lon_span}° implausible"
        assert 0.01 < lat_span < 10.0, f"Latitude span {lat_span}° implausible"


@pytest.mark.slow
@pytest.mark.integration
def test_affine_get_bounds_matches_footprint(require_landsat_file):
    """get_bounds() must return the same values as get_footprint()['bounds']."""
    with GeoTIFFReader(str(require_landsat_file)) as reader:
        geo = AffineGeolocation.from_reader(reader)

        bounds = geo.get_bounds()
        footprint_bounds = geo.get_footprint()['bounds']

        assert bounds == footprint_bounds


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not _HAS_GEO_UTILS, reason="grdl.geolocation.utils not available")
def test_affine_footprint_diagonal_distance(require_landsat_file):
    """The diagonal of a Landsat scene footprint should be plausible.

    A single Landsat 8/9 band file covers the full scene extent of
    approximately 170 x 185 km, giving a diagonal of ~251 km. We bound
    at 100-350 km: tight enough to catch a wrong UTM zone or doubled
    pixel spacing, loose enough to accommodate partial scenes.
    """
    with GeoTIFFReader(str(require_landsat_file)) as reader:
        geo = AffineGeolocation.from_reader(reader)
        min_lon, min_lat, max_lon, max_lat = geo.get_bounds()

        diagonal_m = geographic_distance(min_lat, min_lon, max_lat, max_lon)
        diagonal_km = diagonal_m / 1000.0

        assert diagonal_km > 100.0, (
            f"Footprint diagonal {diagonal_km:.1f} km < 100 km — "
            "implausibly small for a Landsat scene"
        )
        assert diagonal_km < 350.0, (
            f"Footprint diagonal {diagonal_km:.1f} km > 350 km — "
            "exceeds plausible Landsat scene extent"
        )


@pytest.mark.slow
@pytest.mark.integration
def test_affine_stacked_array_with_real_data(require_landsat_file):
    """Stacked (2, N) dispatch works with real reader data.

    Exercises the full pathway: from_reader → stacked-array forward →
    separate-array inverse → pixel comparison.
    """
    with GeoTIFFReader(str(require_landsat_file)) as reader:
        geo = AffineGeolocation.from_reader(reader)
        nrows, ncols = geo.shape

        # Stacked (2, 4) — four corner pixels
        points = np.array([
            [0.0, 0.0, nrows - 1.0, nrows - 1.0],
            [0.0, ncols - 1.0, 0.0, ncols - 1.0],
        ])

        result = geo.image_to_latlon(points)
        assert result.shape == (3, 4)

        # All latitudes valid
        assert np.all(np.abs(result[0]) <= 90.0)
        # All longitudes valid
        assert np.all(np.abs(result[1]) <= 180.0)

        # Round-trip via separate arrays
        rows_back, cols_back = geo.latlon_to_image(result[0], result[1])
        np.testing.assert_allclose(rows_back, points[0], atol=3.4e-5)
        np.testing.assert_allclose(cols_back, points[1], atol=3.4e-5)
