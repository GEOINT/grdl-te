# -*- coding: utf-8 -*-
"""
Elevation Model Tests - ConstantElevation and ElevationModel ABC validation.

Tests the ConstantElevation concrete implementation and the ElevationModel
abstract base class dispatch logic (scalar, array, stacked-array). These
tests use only synthetic data and require no external DEM files.

For DTEDElevation, GeoTIFFDEM, and GeoidCorrection tests that require real
data files, see the Required Data Assets section in the test priority document.

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

# Third-party
import pytest
import numpy as np

# GRDL internal
try:
    from grdl.geolocation.elevation.base import ElevationModel
    from grdl.geolocation.elevation.constant import ConstantElevation
    _HAS_GRDL = True
except ImportError:
    _HAS_GRDL = False


pytestmark = [
    pytest.mark.geolocation,
    pytest.mark.skipif(not _HAS_GRDL, reason="grdl.geolocation.elevation not installed"),
]


# =============================================================================
# ConstantElevation — Core behavior
# =============================================================================


class TestConstantElevation:
    """Tests for ConstantElevation — fixed-height fallback model."""

    def test_default_height_zero(self):
        """Default height is 0.0 meters."""
        elev = ConstantElevation()
        result = elev.get_elevation(34.05, -118.25)
        assert result == 0.0

    def test_custom_height_scalar(self):
        """Scalar query returns the configured constant height."""
        elev = ConstantElevation(height=100.0)
        result = elev.get_elevation(34.05, -118.25)
        assert isinstance(result, float)
        assert result == pytest.approx(100.0)

    def test_negative_height(self):
        """Negative height (e.g., Dead Sea depression) should work."""
        elev = ConstantElevation(height=-430.0)
        result = elev.get_elevation(31.5, 35.5)
        assert result == pytest.approx(-430.0)

    def test_array_query(self):
        """Array input returns array of constant values."""
        elev = ConstantElevation(height=250.0)
        lats = np.array([34.0, 35.0, 36.0])
        lons = np.array([-118.0, -117.0, -116.0])
        result = elev.get_elevation(lats, lons)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        np.testing.assert_array_almost_equal(result, 250.0)

    def test_stacked_2xN_query(self):
        """Stacked (2, N) input returns (N,) array."""
        elev = ConstantElevation(height=500.0)
        pts = np.array([[34.0, 35.0, 36.0],
                        [-118.0, -117.0, -116.0]])
        result = elev.get_elevation(pts)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        np.testing.assert_array_almost_equal(result, 500.0)

    def test_single_element_array(self):
        """Single-element array input returns array."""
        elev = ConstantElevation(height=42.0)
        result = elev.get_elevation(
            np.array([34.0]), np.array([-118.0])
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        assert result[0] == pytest.approx(42.0)

    def test_null_island_query(self):
        """Query at Null Island (0, 0) returns constant."""
        elev = ConstantElevation(height=0.0)
        result = elev.get_elevation(0.0, 0.0)
        assert result == 0.0

    def test_polar_query(self):
        """Query at North Pole returns constant."""
        elev = ConstantElevation(height=10.0)
        result = elev.get_elevation(90.0, 0.0)
        assert result == pytest.approx(10.0)

    def test_south_pole_query(self):
        """Query at South Pole returns constant."""
        elev = ConstantElevation(height=2835.0)  # Antarctic plateau
        result = elev.get_elevation(-90.0, 0.0)
        assert result == pytest.approx(2835.0)

    def test_international_date_line_query(self):
        """Query near the International Date Line."""
        elev = ConstantElevation(height=0.0)
        result_pos = elev.get_elevation(0.0, 180.0)
        result_neg = elev.get_elevation(0.0, -180.0)
        assert result_pos == result_neg

    def test_large_batch_performance(self):
        """Large batch (10k points) should return correct shape."""
        elev = ConstantElevation(height=1000.0)
        n = 10_000
        lats = np.random.uniform(-90, 90, n)
        lons = np.random.uniform(-180, 180, n)
        result = elev.get_elevation(lats, lons)

        assert result.shape == (n,)
        np.testing.assert_array_almost_equal(result, 1000.0)

    def test_dtype_float64(self):
        """Array output should be float64 for precision."""
        elev = ConstantElevation(height=100.0)
        result = elev.get_elevation(
            np.array([0.0, 0.0]), np.array([0.0, 0.0])
        )
        assert result.dtype == np.float64

    def test_no_geoid_by_default(self):
        """ConstantElevation should have no geoid correction by default."""
        elev = ConstantElevation(height=100.0)
        assert elev._geoid is None

    def test_dem_path_is_none(self):
        """ConstantElevation sets dem_path to None."""
        elev = ConstantElevation(height=100.0)
        assert elev.dem_path is None


# =============================================================================
# ElevationModel ABC — Dispatch contract
# =============================================================================


class _MockElevation(ElevationModel):
    """Mock elevation model that returns lat * 10 for testing dispatch."""

    def _get_elevation_array(
        self, lats: np.ndarray, lons: np.ndarray
    ) -> np.ndarray:
        return lats * 10.0


class TestElevationModelDispatch:
    """Test ElevationModel.get_elevation dispatch logic."""

    @pytest.fixture
    def mock_elev(self):
        return _MockElevation()

    def test_scalar_dispatch(self, mock_elev):
        """Scalar input (lat, lon) returns a Python float."""
        result = mock_elev.get_elevation(5.0, 10.0)
        assert isinstance(result, float)
        assert result == pytest.approx(50.0)

    def test_array_dispatch(self, mock_elev):
        """Separate array inputs return ndarray."""
        result = mock_elev.get_elevation(
            np.array([1.0, 2.0, 3.0]),
            np.array([10.0, 20.0, 30.0]),
        )
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, [10.0, 20.0, 30.0])

    def test_stacked_dispatch(self, mock_elev):
        """Stacked (2, N) input returns (N,) array."""
        pts = np.array([[1.0, 2.0], [10.0, 20.0]])
        result = mock_elev.get_elevation(pts)
        assert result.shape == (2,)
        np.testing.assert_array_almost_equal(result, [10.0, 20.0])

    def test_stacked_invalid_shape_raises(self, mock_elev):
        """Non-(2, N) array must raise ValueError."""
        with pytest.raises(ValueError, match="Expected \\(2, N\\)"):
            mock_elev.get_elevation(np.array([[1.0, 2.0, 3.0]]))

    def test_list_dispatch(self, mock_elev):
        """Python list inputs should dispatch as array."""
        result = mock_elev.get_elevation([1.0, 2.0], [10.0, 20.0])
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, [10.0, 20.0])


# =============================================================================
# ElevationModel — Geoid correction plumbing (mock-based)
# =============================================================================


class TestElevationModelGeoidPlumbing:
    """Verify the geoid correction path in ElevationModel.get_elevation.

    These tests verify the dispatch logic applies geoid undulation when
    configured. They do NOT test GeoidCorrection accuracy — that requires
    a real EGM96 grid file.
    """

    def test_no_geoid_returns_raw_heights(self):
        """Without geoid, get_elevation returns raw _get_elevation_array values."""
        elev = _MockElevation(dem_path=None, geoid_path=None)
        result = elev.get_elevation(5.0, 10.0)
        # MockElevation returns lat * 10 = 50.0
        assert result == pytest.approx(50.0)

    def test_constant_elevation_no_geoid(self):
        """ConstantElevation without geoid returns the constant."""
        elev = ConstantElevation(height=100.0)
        # Scalar
        assert elev.get_elevation(0.0, 0.0) == pytest.approx(100.0)
        # Array
        result = elev.get_elevation(np.zeros(5), np.zeros(5))
        np.testing.assert_array_almost_equal(result, 100.0)
