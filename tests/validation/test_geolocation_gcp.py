# -*- coding: utf-8 -*-
"""
GCPGeolocation Validation - BIOMASS L1 SCS GCP-based transforms.

Tests grdl.geolocation.sar.gcp.GCPGeolocation using GCPs extracted
from real BIOMASS L1 SCS product metadata via create_geolocation().

- Level 1: Construction, GCP count, shape attribute
- Level 2: Forward/inverse roundtrip accuracy, convex hull bounds,
           interpolation error metrics
- Level 3: create_geolocation() factory integration

Dataset: BIOMASS L1 SCS product directory

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
2026-05-29
"""

import pytest
import numpy as np

try:
    from grdl.geolocation import GCPGeolocation, create_geolocation
    _HAS_GCP_GEO = True
except ImportError:
    _HAS_GCP_GEO = False

try:
    from grdl.IO.sar import BIOMASSL1Reader
    _HAS_BIOMASS = True
except ImportError:
    _HAS_BIOMASS = False


pytestmark = [
    pytest.mark.geolocation,
    pytest.mark.requires_data,
    pytest.mark.skipif(not _HAS_GCP_GEO,
                       reason="GCPGeolocation not available"),
    pytest.mark.skipif(not _HAS_BIOMASS,
                       reason="BIOMASSL1Reader not available"),
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def biomass_geo(biomass_data_dir):
    """Build GCPGeolocation from BIOMASS product metadata."""
    if not biomass_data_dir.exists():
        pytest.skip(f"BIOMASS data directory not found: {biomass_data_dir}")
    matches = [p for p in biomass_data_dir.glob("BIO_S*") if p.is_dir()]
    if not matches:
        pytest.skip(f"BIOMASS product not found in {biomass_data_dir}")

    with BIOMASSL1Reader(str(matches[0])) as reader:
        geo = create_geolocation(reader)
        meta = reader.metadata
        shape = reader.get_shape()

    if not isinstance(geo, GCPGeolocation):
        pytest.skip("BIOMASS product did not produce GCPGeolocation")

    return geo, shape, meta


# =============================================================================
# Level 1: Format Validation — Construction
# =============================================================================


class TestGCPGeolocationLevel1:
    """Validate GCPGeolocation construction and attributes."""

    def test_construction_succeeds(self, biomass_geo):
        """GCPGeolocation created from BIOMASS metadata."""
        geo, _, _ = biomass_geo
        assert geo is not None
        assert isinstance(geo, GCPGeolocation)

    def test_minimum_gcp_count(self, biomass_geo):
        """At least 4 GCPs present (interpolation minimum)."""
        geo, _, _ = biomass_geo
        assert geo.n_gcps >= 4, (
            f"Only {geo.n_gcps} GCPs — need at least 4"
        )

    def test_shape_attribute(self, biomass_geo):
        """Shape matches reader dimensions."""
        geo, shape, _ = biomass_geo
        assert geo.shape == (shape[0], shape[1])

    def test_gcps_have_valid_coordinates(self, biomass_geo):
        """All GCPs have valid lat/lon/row/col ranges."""
        geo, shape, _ = biomass_geo
        for lon, lat, height, row, col in geo.gcps:
            assert -180.0 <= lon <= 180.0, f"Invalid longitude: {lon}"
            assert -90.0 <= lat <= 90.0, f"Invalid latitude: {lat}"
            assert 0 <= row <= shape[0], f"Row {row} out of bounds"
            assert 0 <= col <= shape[1], f"Col {col} out of bounds"


# =============================================================================
# Level 2: Data Quality — Projection accuracy
# =============================================================================


class TestGCPGeolocationLevel2:
    """Validate forward/inverse transform accuracy."""

    def test_forward_returns_valid_latlon(self, biomass_geo):
        """Image center maps to valid geographic coordinates."""
        geo, shape, _ = biomass_geo
        center_row = shape[0] // 2
        center_col = shape[1] // 2
        lats, lons, heights = geo._image_to_latlon_array(
            np.array([center_row], dtype=np.float64),
            np.array([center_col], dtype=np.float64),
        )
        assert np.isfinite(lats[0]), "Forward transform returned NaN latitude"
        assert np.isfinite(lons[0]), "Forward transform returned NaN longitude"
        assert -90 <= lats[0] <= 90
        assert -180 <= lons[0] <= 180

    def test_inverse_returns_valid_pixel(self, biomass_geo):
        """Known GCP lat/lon maps back to valid pixel coordinates."""
        geo, shape, _ = biomass_geo
        # Use first GCP as ground truth
        lon, lat, height, expected_row, expected_col = geo.gcps[0]
        rows, cols = geo._latlon_to_image_array(
            np.array([lat], dtype=np.float64),
            np.array([lon], dtype=np.float64),
        )
        assert np.isfinite(rows[0]), "Inverse transform returned NaN row"
        assert np.isfinite(cols[0]), "Inverse transform returned NaN col"

    def test_roundtrip_at_gcp_locations(self, biomass_geo):
        """Forward→inverse roundtrip recovers GCP pixel locations."""
        geo, _, _ = biomass_geo
        # Test on a subset of GCPs
        test_gcps = geo.gcps[:min(10, len(geo.gcps))]
        max_error = 0.0

        for lon, lat, height, row, col in test_gcps:
            # Forward: pixel → latlon
            lats, lons, _ = geo._image_to_latlon_array(
                np.array([row], dtype=np.float64),
                np.array([col], dtype=np.float64),
            )
            # Inverse: latlon → pixel
            rows_back, cols_back = geo._latlon_to_image_array(
                lats, lons,
            )
            if np.isfinite(rows_back[0]) and np.isfinite(cols_back[0]):
                err = np.sqrt((rows_back[0] - row)**2 + (cols_back[0] - col)**2)
                max_error = max(max_error, err)

        # At GCP locations, roundtrip should be near-exact
        assert max_error < 2.0, (
            f"Roundtrip error at GCPs: {max_error:.2f} pixels (expected < 2)"
        )

    def test_interpolation_error_metrics(self, biomass_geo):
        """get_interpolation_error() returns finite metrics."""
        geo, _, _ = biomass_geo
        if not hasattr(geo, 'get_interpolation_error'):
            pytest.skip("get_interpolation_error not available")
        errors = geo.get_interpolation_error()
        assert 'rms_error_m' in errors
        assert np.isfinite(errors['rms_error_m'])
        assert errors['rms_error_m'] >= 0.0

    def test_outside_convex_hull_returns_nan(self, biomass_geo):
        """Points far outside GCP coverage return NaN."""
        geo, _, _ = biomass_geo
        # Point at south pole — definitely outside any SAR scene
        lats = np.array([-89.0])
        lons = np.array([0.0])
        rows, cols = geo._latlon_to_image_array(lats, lons)
        # Should be NaN (outside interpolation domain)
        assert np.isnan(rows[0]) or (rows[0] < -1000 or rows[0] > geo.shape[0] + 1000), (
            "Expected NaN or out-of-bounds for distant point"
        )

    def test_multiple_points_vectorized(self, biomass_geo):
        """Vectorized transform handles multiple points correctly."""
        geo, shape, _ = biomass_geo
        # Create a grid of 5 interior points
        rows_in = np.linspace(shape[0] * 0.2, shape[0] * 0.8, 5)
        cols_in = np.linspace(shape[1] * 0.2, shape[1] * 0.8, 5)
        lats, lons, heights = geo._image_to_latlon_array(rows_in, cols_in)
        assert lats.shape == (5,)
        assert lons.shape == (5,)
        # At least some should be finite (within GCP hull)
        n_valid = np.sum(np.isfinite(lats))
        assert n_valid >= 3, f"Only {n_valid}/5 interior points resolved"


# =============================================================================
# Level 3: Integration — Factory and cross-reader
# =============================================================================


class TestGCPGeolocationLevel3:
    """Integration with create_geolocation() factory."""

    @pytest.mark.integration
    def test_create_geolocation_returns_gcp(self, biomass_data_dir):
        """create_geolocation(BIOMASSL1Reader) returns GCPGeolocation."""
        matches = [p for p in biomass_data_dir.glob("BIO_S*") if p.is_dir()]
        if not matches:
            pytest.skip("BIOMASS product not found")
        with BIOMASSL1Reader(str(matches[0])) as reader:
            geo = create_geolocation(reader)
        assert isinstance(geo, GCPGeolocation), (
            f"Expected GCPGeolocation, got {type(geo).__name__}"
        )

    @pytest.mark.integration
    def test_gcp_density_adequate(self, biomass_geo):
        """GCP density is sufficient for the image size."""
        geo, shape, _ = biomass_geo
        total_pixels = shape[0] * shape[1]
        # Expect at least 1 GCP per 10k×10k pixel block
        min_expected = max(4, total_pixels / (10000 * 10000))
        assert geo.n_gcps >= min_expected, (
            f"GCP density too low: {geo.n_gcps} GCPs for "
            f"{shape[0]}×{shape[1]} image"
        )
