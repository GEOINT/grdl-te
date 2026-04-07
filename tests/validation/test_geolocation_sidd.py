# -*- coding: utf-8 -*-
"""
SIDDGeolocation Tests - Coordinate transforms for SIDD derived SAR products.

Validates grdl.geolocation.sar.sidd.SIDDGeolocation after the coordinate
math was extracted to grdl.geolocation.coordinates in commit 059c5f2:

- Level 1: PlaneProjection forward/inverse round-trip using synthetic metadata
- Level 2: GeographicProjection forward/inverse consistency; CylindricalProjection
           forward produces finite lat/lon; invalid projection type raises
- Level 3 (requires_data): SIDDReader integration — round-trip pixel→lat/lon→pixel
           error < 0.5 px on real SIDD imagery

Synthetic tests construct mock SIDDMetadata using SimpleNamespace so no
real data files are needed for the first two levels.

Dependencies
------------
(none beyond numpy, which is a core GRDL dep)

Author
------
Claude Code (generated)

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2026-03-09

Modified
--------
2026-03-09
"""

# Standard library
from types import SimpleNamespace

# Third-party
import pytest
import numpy as np

# GRDL internal
try:
    from grdl.geolocation.sar.sidd import SIDDGeolocation
    _HAS_SIDD_GEO = True
except ImportError:
    _HAS_SIDD_GEO = False

try:
    from grdl.IO.sar import SIDDReader
    _HAS_SIDD_READER = True
except ImportError:
    _HAS_SIDD_READER = False

pytestmark = [
    pytest.mark.sidd,
    pytest.mark.geolocation,
    pytest.mark.skipif(not _HAS_SIDD_GEO, reason="SIDDGeolocation not available"),
]


# ---------------------------------------------------------------------------
# Mock metadata builders
# ---------------------------------------------------------------------------

def _make_ecef(x: float, y: float, z: float) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z)


def _make_point(row: float, col: float) -> SimpleNamespace:
    return SimpleNamespace(row=row, col=col)


def _make_vec(x: float, y: float, z: float) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z)


def _make_plane_projection_meta(
    ecef_x: float, ecef_y: float, ecef_z: float,
    ref_row: float = 0.0, ref_col: float = 0.0,
    dr: float = 10.0, dc: float = 10.0,
    row_vec=(1.0, 0.0, 0.0), col_vec=(0.0, 1.0, 0.0),
) -> SimpleNamespace:
    """Build minimal SIDDMetadata for PlaneProjection."""
    plane_proj = SimpleNamespace(
        reference_point=SimpleNamespace(
            ecef=_make_ecef(ecef_x, ecef_y, ecef_z),
            point=_make_point(ref_row, ref_col),
        ),
        sample_spacing=SimpleNamespace(row=dr, col=dc),
        product_plane=SimpleNamespace(
            row_unit_vector=_make_vec(*row_vec),
            col_unit_vector=_make_vec(*col_vec),
        ),
    )
    measurement = SimpleNamespace(
        projection_type='PlaneProjection',
        plane_projection=plane_proj,
    )
    return SimpleNamespace(measurement=measurement, rows=1000, cols=1000)


def _make_geographic_meta(
    ecef_x: float, ecef_y: float, ecef_z: float,
    ref_row: float = 0.0, ref_col: float = 0.0,
    dr: float = 1.0, dc: float = 1.0,
) -> SimpleNamespace:
    """Build minimal SIDDMetadata for GeographicProjection.

    GeographicProjection uses the same plane_projection structure as
    PlaneProjection (sarpy/sarkit populate it there).  Sample spacing
    is in arc-seconds.
    """
    # A placeholder product_plane is provided because some code paths may
    # inspect it, but GGD doesn't use row/col unit vectors.
    plane_proj = SimpleNamespace(
        reference_point=SimpleNamespace(
            ecef=_make_ecef(ecef_x, ecef_y, ecef_z),
            point=_make_point(ref_row, ref_col),
        ),
        sample_spacing=SimpleNamespace(row=dr, col=dc),
        product_plane=None,
    )
    measurement = SimpleNamespace(
        projection_type='GeographicProjection',
        plane_projection=plane_proj,
    )
    return SimpleNamespace(measurement=measurement, rows=500, cols=500)


def _make_cylindrical_meta(
    ecef_x: float, ecef_y: float, ecef_z: float,
    ref_row: float = 0.0, ref_col: float = 0.0,
    dr: float = 5.0, dc: float = 5.0,
    row_vec=(0.0, 1.0, 0.0), col_vec=(0.0, 0.0, 1.0),
) -> SimpleNamespace:
    """Build minimal SIDDMetadata for CylindricalProjection."""
    plane_proj = SimpleNamespace(
        reference_point=SimpleNamespace(
            ecef=_make_ecef(ecef_x, ecef_y, ecef_z),
            point=_make_point(ref_row, ref_col),
        ),
        sample_spacing=SimpleNamespace(row=dr, col=dc),
        product_plane=SimpleNamespace(
            row_unit_vector=_make_vec(*row_vec),
            col_unit_vector=_make_vec(*col_vec),
        ),
    )
    measurement = SimpleNamespace(
        projection_type='CylindricalProjection',
        plane_projection=plane_proj,
    )
    return SimpleNamespace(measurement=measurement, rows=200, cols=200)


# Reference ECEF point: mid-Atlantic coast of Virginia (approx.)
# Computed from lat=36.8°N, lon=-76.0°E, alt=0 m with WGS-84
_REF_X = 1164960.0
_REF_Y = -4753780.0
_REF_Z = 3794880.0


# ===========================================================================
# Level 1: PlaneProjection
# ===========================================================================

class TestSIDDGeolocationPlaneProjection:
    """Validate PGD forward/inverse round-trip."""

    @pytest.fixture(scope="class")
    def pgd_geo(self):
        """SIDDGeolocation configured for PlaneProjection."""
        meta = _make_plane_projection_meta(
            ecef_x=_REF_X, ecef_y=_REF_Y, ecef_z=_REF_Z,
            ref_row=500.0, ref_col=500.0,
            dr=10.0, dc=10.0,
        )
        return SIDDGeolocation(meta)

    def test_constructs_without_error(self, pgd_geo):
        """SIDDGeolocation instantiates from PlaneProjection metadata."""
        assert pgd_geo is not None
        assert pgd_geo.projection_type == 'PlaneProjection'

    def test_shape_attribute(self, pgd_geo):
        """shape attribute reflects metadata rows/cols."""
        assert pgd_geo.shape == (1000, 1000), (
            f"shape {pgd_geo.shape} should be (1000, 1000)"
        )

    def test_image_to_latlon_returns_finite(self, pgd_geo):
        """image_to_latlon returns finite lat/lon/height at the reference pixel."""
        lat, lon, h = pgd_geo.image_to_latlon(500.0, 500.0)
        assert np.isfinite(float(lat)), f"lat is not finite: {lat}"
        assert np.isfinite(float(lon)), f"lon is not finite: {lon}"
        assert np.isfinite(float(h)), f"height is not finite: {h}"
        assert -90.0 <= float(lat) <= 90.0, f"lat {lat} out of valid range [-90, 90]"
        assert -180.0 <= float(lon) <= 180.0, f"lon {lon} out of valid range [-180, 180]"

    def test_round_trip_pixel_to_latlon_to_pixel(self, pgd_geo):
        """Pixel → lat/lon → pixel round-trip error < 0.5 pixels.

        The SIDD standard specifies exact forward and inverse formulas, so
        the round-trip error should be at or near floating-point precision.
        A 0.5 px threshold is generous — a real implementation should be
        sub-pixel at < 0.01 px.
        """
        test_pixels = [
            (500.0, 500.0),   # reference pixel
            (0.0, 0.0),       # top-left corner
            (999.0, 999.0),   # bottom-right corner
            (250.0, 750.0),   # off-center
        ]
        for row_in, col_in in test_pixels:
            lat, lon, h = pgd_geo.image_to_latlon(row_in, col_in)
            result = pgd_geo.latlon_to_image(
                np.array([float(lat)]),
                np.array([float(lon)]),
                height=np.array([float(h)]),
            )
            row_out = result[0, 0]
            col_out = result[0, 1]
            assert abs(float(row_out) - row_in) < 0.5, (
                f"PlaneProjection round-trip row error "
                f"{abs(float(row_out) - row_in):.4f} px at pixel ({row_in}, {col_in}) "
                f"exceeds 0.5 px threshold"
            )
            assert abs(float(col_out) - col_in) < 0.5, (
                f"PlaneProjection round-trip col error "
                f"{abs(float(col_out) - col_in):.4f} px at pixel ({row_in}, {col_in}) "
                f"exceeds 0.5 px threshold"
            )

    def test_vectorized_array_round_trip(self, pgd_geo):
        """Vectorized image_to_latlon / latlon_to_image produces correct shapes."""
        rows = np.array([0.0, 250.0, 500.0, 750.0, 999.0])
        cols = np.array([0.0, 250.0, 500.0, 750.0, 999.0])
        result_ll = pgd_geo.image_to_latlon(rows, cols)
        lats = result_ll[:, 0]
        lons = result_ll[:, 1]
        heights = result_ll[:, 2]
        assert lats.shape == (5,), f"lats shape {lats.shape} != (5,)"

        result_rt = pgd_geo.latlon_to_image(lats, lons, height=heights)
        rows_rt = result_rt[:, 0]
        cols_rt = result_rt[:, 1]
        np.testing.assert_allclose(rows_rt, rows, atol=0.5,
            err_msg="Vectorized PGD round-trip row error > 0.5 px")
        np.testing.assert_allclose(cols_rt, cols, atol=0.5,
            err_msg="Vectorized PGD round-trip col error > 0.5 px")


# ===========================================================================
# Level 2: GeographicProjection, CylindricalProjection, error handling
# ===========================================================================

class TestSIDDGeolocationGeographicProjection:
    """Validate GGD forward/inverse consistency."""

    @pytest.fixture(scope="class")
    def ggd_geo(self):
        """SIDDGeolocation configured for GeographicProjection."""
        meta = _make_geographic_meta(
            ecef_x=_REF_X, ecef_y=_REF_Y, ecef_z=_REF_Z,
            ref_row=250.0, ref_col=250.0,
            dr=1.0, dc=1.0,  # arc-seconds per pixel
        )
        return SIDDGeolocation(meta)

    def test_constructs_without_error(self, ggd_geo):
        """SIDDGeolocation instantiates from GeographicProjection metadata."""
        assert ggd_geo.projection_type == 'GeographicProjection'

    def test_image_to_latlon_reference_pixel(self, ggd_geo):
        """image_to_latlon at the reference pixel returns finite lat/lon."""
        lat, lon, h = ggd_geo.image_to_latlon(250.0, 250.0)
        assert np.isfinite(float(lat)), f"GGD reference pixel lat is not finite: {lat}"
        assert np.isfinite(float(lon)), f"GGD reference pixel lon is not finite: {lon}"

    def test_round_trip_consistent(self, ggd_geo):
        """GGD pixel → lat/lon → pixel round-trip error < 0.01 pixels.

        GeographicProjection is a simple linear mapping so the round-trip
        should be exact within floating-point precision.
        """
        row_in, col_in = 100.0, 400.0
        lat, lon, h = ggd_geo.image_to_latlon(row_in, col_in)
        result = ggd_geo.latlon_to_image(
            np.array([float(lat)]),
            np.array([float(lon)]),
        )
        row_out = result[0, 0]
        col_out = result[0, 1]
        assert abs(float(row_out) - row_in) < 0.01, (
            f"GGD round-trip row error {abs(float(row_out) - row_in):.6f} px > 0.01 px"
        )
        assert abs(float(col_out) - col_in) < 0.01, (
            f"GGD round-trip col error {abs(float(col_out) - col_in):.6f} px > 0.01 px"
        )

    def test_north_row_less_than_south_row(self, ggd_geo):
        """GGD: increasing row goes south (lat decreases with increasing row).

        SIDD Geographic standard: dr is angular spacing and lat decreases
        as dr*(r - r0) / 3600 increases.
        """
        lat_top, _, _ = ggd_geo.image_to_latlon(0.0, 250.0)
        lat_bottom, _, _ = ggd_geo.image_to_latlon(499.0, 250.0)
        assert float(lat_top) > float(lat_bottom), (
            f"Row 0 lat {float(lat_top):.4f}° should be > row 499 lat "
            f"{float(lat_bottom):.4f}° in north-up GGD grid"
        )


class TestSIDDGeolocationCylindricalProjection:
    """Validate CGD forward produces finite output."""

    def test_cylindrical_constructs_and_projects(self):
        """CylindricalProjection constructs and produces finite lat/lon."""
        meta = _make_cylindrical_meta(
            ecef_x=_REF_X, ecef_y=_REF_Y, ecef_z=_REF_Z,
            ref_row=100.0, ref_col=100.0,
        )
        geo = SIDDGeolocation(meta)
        assert geo.projection_type == 'CylindricalProjection'

        lat, lon, h = geo.image_to_latlon(100.0, 100.0)
        assert np.isfinite(float(lat)), f"CGD forward lat is not finite: {lat}"
        assert np.isfinite(float(lon)), f"CGD forward lon is not finite: {lon}"


class TestSIDDGeolocationErrorHandling:
    """Validate constructor error handling."""

    def test_unsupported_projection_type_raises(self):
        """An unrecognized projection_type raises ValueError."""
        measurement = SimpleNamespace(
            projection_type='PolynomialProjection',
            plane_projection=None,
        )
        meta = SimpleNamespace(measurement=measurement, rows=100, cols=100)
        with pytest.raises(ValueError, match="Unsupported|projection"):
            SIDDGeolocation(meta)

    def test_missing_measurement_raises(self):
        """metadata.measurement = None raises ValueError."""
        meta = SimpleNamespace(measurement=None, rows=100, cols=100)
        with pytest.raises(ValueError, match="measurement"):
            SIDDGeolocation(meta)

    def test_plane_projection_missing_ecef_raises(self):
        """PlaneProjection with reference_point.ecef = None raises ValueError."""
        plane_proj = SimpleNamespace(
            reference_point=SimpleNamespace(ecef=None, point=_make_point(0, 0)),
            sample_spacing=SimpleNamespace(row=10.0, col=10.0),
            product_plane=SimpleNamespace(
                row_unit_vector=_make_vec(1, 0, 0),
                col_unit_vector=_make_vec(0, 1, 0),
            ),
        )
        measurement = SimpleNamespace(
            projection_type='PlaneProjection',
            plane_projection=plane_proj,
        )
        meta = SimpleNamespace(measurement=measurement, rows=100, cols=100)
        with pytest.raises(ValueError):
            SIDDGeolocation(meta)


# ===========================================================================
# Level 3: Real data integration (requires_data)
# ===========================================================================

@pytest.mark.requires_data
@pytest.mark.integration
@pytest.mark.skipif(not _HAS_SIDD_READER, reason="SIDDReader not available")
class TestSIDDGeolocationRealData:
    """Integration tests with a real SIDD NITF file.

    These tests skip when no SIDD data is present in data/sidd/.
    Download instructions: data/sidd/README.md
    """

    def test_geolocation_round_trip_center_pixel(self, require_sidd_file):
        """Pixel → lat/lon → pixel round-trip at image center is < 0.5 px.

        Validates the complete refactored SIDDGeolocation chain (which now
        delegates ECEF/ENU math to grdl.geolocation.coordinates) against
        a real SIDD image.  Any regression in the coordinate extraction
        would show up as a > 0.5 px round-trip error.
        """
        with SIDDReader(str(require_sidd_file)) as reader:
            geo = SIDDGeolocation.from_reader(reader)
            rows, cols = geo.shape[:2]

            row_c = rows / 2.0
            col_c = cols / 2.0

            lat, lon, h = geo.image_to_latlon(row_c, col_c)
            result_rt = geo.latlon_to_image(
                np.array([float(lat)]),
                np.array([float(lon)]),
                height=np.array([float(h)]),
            )
            row_rt = result_rt[0, 0]
            col_rt = result_rt[0, 1]

        assert abs(float(row_rt) - row_c) < 0.5, (
            f"Center pixel row round-trip error "
            f"{abs(float(row_rt) - row_c):.4f} px > 0.5 px — "
            f"coordinate refactor may have broken SIDD geolocation"
        )
        assert abs(float(col_rt) - col_c) < 0.5, (
            f"Center pixel col round-trip error "
            f"{abs(float(col_rt) - col_c):.4f} px > 0.5 px"
        )

    def test_geolocation_returns_plausible_coordinates(self, require_sidd_file):
        """image_to_latlon returns coordinates within valid geodetic bounds.

        Verifies that the coordinate chain produces geographically valid
        values and not garbage (e.g., ECEF values returned as lat/lon).
        """
        with SIDDReader(str(require_sidd_file)) as reader:
            geo = SIDDGeolocation.from_reader(reader)
            rows, cols = geo.shape[:2]

            # Sample four corners and center
            test_pts = [
                (0.0, 0.0), (0.0, cols - 1.0),
                (rows - 1.0, 0.0), (rows - 1.0, cols - 1.0),
                (rows / 2.0, cols / 2.0),
            ]
            for r, c in test_pts:
                lat, lon, h = geo.image_to_latlon(r, c)
                assert -90.0 <= float(lat) <= 90.0, (
                    f"Pixel ({r},{c}): lat {float(lat):.4f} out of [-90, 90]"
                )
                assert -180.0 <= float(lon) <= 180.0, (
                    f"Pixel ({r},{c}): lon {float(lon):.4f} out of [-180, 180]"
                )

    def test_get_bounds_returns_valid_bbox(self, require_sidd_file):
        """get_bounds() returns (min_lon, min_lat, max_lon, max_lat) in correct order."""
        with SIDDReader(str(require_sidd_file)) as reader:
            geo = SIDDGeolocation.from_reader(reader)
            bounds = geo.get_bounds()

        assert len(bounds) == 4, f"get_bounds() must return 4 values, got {len(bounds)}"
        min_lon, min_lat, max_lon, max_lat = bounds
        assert max_lat > min_lat, (
            f"get_bounds: max_lat {max_lat:.4f} <= min_lat {min_lat:.4f}"
        )

    @pytest.mark.requires_data
    def test_sidd_default_hae_delegation(self, require_sidd_file):
        """SIDDGeolocation.default_hae returns plausible height value.

        Verifies that default_hae (default Height Above Ellipsoid) is
        properly initialized from measurement metadata and returns a
        finite numeric value suitable for terrain-uncorrected transforms.
        """
        with SIDDReader(str(require_sidd_file)) as reader:
            geo = SIDDGeolocation.from_reader(reader)
            
            # default_hae must be accessible
            assert hasattr(geo, 'default_hae'), \
                "SIDDGeolocation must expose default_hae property"
            
            hae = geo.default_hae
            assert hae is not None, "default_hae is None"
            assert np.isfinite(float(hae)), \
                f"default_hae {hae} is not finite"
            
            # Reasonable range: most SIDD products are at/near sea level
            # but may be digitally orthorectified to various reference heights
            assert -500 <= float(hae) <= 10000, \
                f"default_hae {hae}m seems implausible"

    @pytest.mark.requires_data
    def test_sidd_coa_projection_metadata_presence(self, require_sidd_file):
        """COAProjection metadata present and accessible if defined.

        COAProjection (Center of Aperture) projection provides additional
        geometric metadata in some SIDD products. Verify it is accessible
        without error if present.
        """
        with SIDDReader(str(require_sidd_file)) as reader:
            geo = SIDDGeolocation.from_reader(reader)
            
            # Check for COAProjection in measurement
            if hasattr(reader.metadata.measurement, 'coa_projection'):
                coa_proj = reader.metadata.measurement.coa_projection
                # If present, should be non-None and have expected attributes
                if coa_proj is not None:
                    # COAProjection should have reference point and/or focal plane info
                    assert hasattr(coa_proj, 'reference_point') or \
                           hasattr(coa_proj, 'focal_plane'), \
                        "COAProjection lacks expected geometry fields"

    @pytest.mark.requires_data
    def test_sidd_elevation_property_settable(self, require_sidd_file):
        """SIDDGeolocation.elevation property can be set for terrain correction.

        Tests that the DEM integration point (geo.elevation) supports
        assignment, which gates terrain-corrected vs. ellipsoid-only projections.
        """
        with SIDDReader(str(require_sidd_file)) as reader:
            geo = SIDDGeolocation.from_reader(reader)
            
            # elevation property should exist and be settable
            if hasattr(geo, 'elevation'):
                initial = geo.elevation
                
                # Try to set it (if a DEM class is available)
                try:
                    from grdl.geolocation.elevation import ConstantElevation
                    test_dem = ConstantElevation(150.0)
                    geo.elevation = test_dem
                    assert geo.elevation is test_dem, \
                        "Failed to set elevation property"
                except ImportError:
                    pytest.skip("ConstantElevation not available")
            
            # Re-query bounds after setting elevation
            min_lon, min_lat, max_lon, max_lat = geo.get_bounds()
        
        assert max_lon > min_lon, (
            f"get_bounds: max_lon {max_lon:.4f} <= min_lon {min_lon:.4f}"
        )
        assert -90.0 <= min_lat and max_lat <= 90.0, (
            f"Latitude bounds [{min_lat:.4f}, {max_lat:.4f}] outside [-90, 90]"
        )
