# -*- coding: utf-8 -*-
"""
DTEDElevation, GeoTIFFDEM, and GeoidCorrection validation.

Tests:
- Level 1: Constructor, scalar elevation query
- Level 2: Elevation range constraints, known-location checks
- Level 3: Vectorized batch queries (10k points)

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
    from grdl.geolocation.elevation import DTEDElevation
    _HAS_DTED = True
except ImportError:
    _HAS_DTED = False

try:
    from grdl.geolocation.elevation import GeoTIFFDEM
    _HAS_DEM = True
except ImportError:
    _HAS_DEM = False

try:
    from grdl.geolocation.elevation import GeoidCorrection
    _HAS_GEOID = True
except ImportError:
    _HAS_GEOID = False


pytestmark = [
    pytest.mark.elevation,
    pytest.mark.requires_data,
]


def _dted_coverage_center(elev):
    """Return (center_lat, center_lon) from a DTEDElevation's tile index."""
    bounds = elev.coverage_bounds
    if bounds is None:
        pytest.skip("DTED tile index is empty — no tiles found")
    min_lon, min_lat, max_lon, max_lat = bounds
    return (min_lat + max_lat) / 2, (min_lon + max_lon) / 2


def _dted_coverage_range(elev, margin=0.1):
    """Return (lat_lo, lat_hi, lon_lo, lon_hi) inset by *margin* fraction."""
    bounds = elev.coverage_bounds
    if bounds is None:
        pytest.skip("DTED tile index is empty — no tiles found")
    min_lon, min_lat, max_lon, max_lat = bounds
    dlat = (max_lat - min_lat) * margin
    dlon = (max_lon - min_lon) * margin
    return min_lat + dlat, max_lat - dlat, min_lon + dlon, max_lon - dlon


def _geotiff_coverage_center(dem):
    """Return (center_lat, center_lon) from a GeoTIFFDEM's dataset bounds."""
    b = dem._dataset.bounds
    return (b.bottom + b.top) / 2, (b.left + b.right) / 2


def _geotiff_coverage_range(dem, margin=0.1):
    """Return (lat_lo, lat_hi, lon_lo, lon_hi) inset by *margin* fraction."""
    b = dem._dataset.bounds
    dlat = (b.top - b.bottom) * margin
    dlon = (b.right - b.left) * margin
    return b.bottom + dlat, b.top - dlat, b.left + dlon, b.right - dlon


# =============================================================================
# DTEDElevation
# =============================================================================


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DTED, reason="DTEDElevation not available")
def test_dted_constructor(require_dted_dir):
    """Accepts directory path."""
    elev = DTEDElevation(str(require_dted_dir))
    assert elev is not None


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DTED, reason="DTEDElevation not available")
def test_dted_get_elevation_scalar(require_dted_dir):
    """Returns float for single lat/lon."""
    elev = DTEDElevation(str(require_dted_dir))
    lat, lon = _dted_coverage_center(elev)
    val = elev.get_elevation(lat, lon)
    assert isinstance(val, (float, np.floating))


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DTED, reason="DTEDElevation not available")
def test_dted_elevation_range(require_dted_dir):
    """Returns values in [-500, 9000] meters (Dead Sea to Everest)."""
    elev = DTEDElevation(str(require_dted_dir))
    lat, lon = _dted_coverage_center(elev)
    val = elev.get_elevation(lat, lon)
    assert -500 <= val <= 9000, f"Elevation {val}m outside physical range"


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DTED, reason="DTEDElevation not available")
def test_dted_batch_10k(require_dted_dir):
    """10,000-point vectorized query, correct shapes."""
    elev = DTEDElevation(str(require_dted_dir))
    lat_lo, lat_hi, lon_lo, lon_hi = _dted_coverage_range(elev)
    rng = np.random.default_rng(42)
    lat_arr = rng.uniform(lat_lo, lat_hi, size=10000)
    lon_arr = rng.uniform(lon_lo, lon_hi, size=10000)
    result = elev.get_elevation(lat_arr, lon_arr)
    assert isinstance(result, np.ndarray)
    assert result.shape == (10000,)
    assert np.all(np.isfinite(result))


# =============================================================================
# GeoTIFFDEM
# =============================================================================


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DEM, reason="GeoTIFFDEM not available")
def test_geotiff_dem_constructor(require_dem_file):
    """Accepts .tif path."""
    dem = GeoTIFFDEM(str(require_dem_file))
    assert dem is not None


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DEM, reason="GeoTIFFDEM not available")
def test_geotiff_dem_scalar(require_dem_file):
    """Returns float for single query."""
    dem = GeoTIFFDEM(str(require_dem_file))
    lat, lon = _geotiff_coverage_center(dem)
    val = dem.get_elevation(lat, lon)
    assert isinstance(val, (float, np.floating))


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DEM, reason="GeoTIFFDEM not available")
def test_geotiff_dem_range(require_dem_file):
    """Physically reasonable elevation range."""
    dem = GeoTIFFDEM(str(require_dem_file))
    lat, lon = _geotiff_coverage_center(dem)
    val = dem.get_elevation(lat, lon)
    assert -500 <= val <= 9000


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DEM, reason="GeoTIFFDEM not available")
def test_geotiff_dem_finite(require_dem_file):
    """No NaN/Inf in batch results."""
    dem = GeoTIFFDEM(str(require_dem_file))
    lat_lo, lat_hi, lon_lo, lon_hi = _geotiff_coverage_range(dem)
    rng = np.random.default_rng(42)
    lat_arr = rng.uniform(lat_lo, lat_hi, size=100)
    lon_arr = rng.uniform(lon_lo, lon_hi, size=100)
    result = dem.get_elevation(lat_arr, lon_arr)
    assert np.all(np.isfinite(result))


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DEM, reason="GeoTIFFDEM not available")
def test_geotiff_dem_batch_10k(require_dem_file):
    """Vectorized 10k-point query."""
    dem = GeoTIFFDEM(str(require_dem_file))
    lat_lo, lat_hi, lon_lo, lon_hi = _geotiff_coverage_range(dem)
    rng = np.random.default_rng(42)
    lat_arr = rng.uniform(lat_lo, lat_hi, size=10000)
    lon_arr = rng.uniform(lon_lo, lon_hi, size=10000)
    result = dem.get_elevation(lat_arr, lon_arr)
    assert isinstance(result, np.ndarray)
    assert result.shape == (10000,)


# =============================================================================
# GeoidCorrection
# =============================================================================


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_GEOID, reason="GeoidCorrection not available")
def test_geoid_constructor(require_geoid_file):
    """Accepts .pgm path."""
    geoid = GeoidCorrection(str(require_geoid_file))
    assert geoid is not None


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_GEOID, reason="GeoidCorrection not available")
def test_geoid_scalar(require_geoid_file):
    """Returns float undulation."""
    geoid = GeoidCorrection(str(require_geoid_file))
    val = geoid.get_undulation(34.0, -118.0)
    assert isinstance(val, (float, np.floating))


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_GEOID, reason="GeoidCorrection not available")
def test_geoid_range(require_geoid_file):
    """EGM96 undulation in [-110, +90] meters globally."""
    geoid = GeoidCorrection(str(require_geoid_file))
    val = geoid.get_undulation(34.0, -118.0)
    assert -110 <= val <= 90, f"Geoid undulation {val}m outside expected range"


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_GEOID, reason="GeoidCorrection not available")
def test_geoid_equator(require_geoid_file):
    r"""Known undulation at equator/Greenwich :math:`\approx 17 \pm 5` m."""
    geoid = GeoidCorrection(str(require_geoid_file))
    val = geoid.get_undulation(0.0, 0.0)
    assert 10 <= val <= 25, f"Equator geoid {val}m outside expected ~17m"


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_GEOID, reason="GeoidCorrection not available")
def test_geoid_batch_10k(require_geoid_file):
    """Vectorized 10k-point query."""
    geoid = GeoidCorrection(str(require_geoid_file))
    rng = np.random.default_rng(42)
    lat_arr = rng.uniform(-90.0, 90.0, size=10000)
    lon_arr = rng.uniform(-180.0, 180.0, size=10000)
    result = geoid.get_undulation(lat_arr, lon_arr)
    assert isinstance(result, np.ndarray)
    assert result.shape == (10000,)
    assert np.all(np.isfinite(result))
