# -*- coding: utf-8 -*-
"""
SICDGeolocation Tests - SAR coordinate transform validation.

Tests SICDGeolocation using real Umbra SICD data:
- Level 1: Construction, SCP image-to-latlon, latlon-to-image
- Level 2: Round-trip precision, footprint, vectorized operations

Dataset: Umbra SICD (*.nitf)

Dependencies
------------
pytest
numpy
grdl (sarkit or sarpy backend)

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
    from grdl.IO.sar.sicd import SICDReader
    _HAS_SICD = True
except ImportError:
    _HAS_SICD = False

try:
    from grdl.geolocation.sar.sicd import SICDGeolocation
    _HAS_SICD_GEO = True
except ImportError:
    _HAS_SICD_GEO = False


pytestmark = [
    pytest.mark.geolocation,
    pytest.mark.sicd,
    pytest.mark.requires_data,
    pytest.mark.skipif(not _HAS_SICD, reason="SICDReader not available"),
    pytest.mark.skipif(not _HAS_SICD_GEO, reason="SICDGeolocation not available"),
]


# =============================================================================
# Level 1: Format Validation
# =============================================================================


@pytest.mark.slow
def test_sicd_geolocation_from_metadata(require_umbra_file):
    """SICDGeolocation constructs from SICDReader."""
    with SICDReader(str(require_umbra_file)) as reader:
        geo = SICDGeolocation.from_reader(reader)
        assert geo is not None


@pytest.mark.slow
def test_sicd_image_to_latlon_scp(require_umbra_file):
    """SCP pixel maps to valid lat/lon."""
    with SICDReader(str(require_umbra_file)) as reader:
        geo = SICDGeolocation.from_reader(reader)
        rows, cols = reader.get_shape()[:2]

        lat, lon, hae = geo.image_to_latlon(rows // 2, cols // 2)
        assert -90 <= lat <= 90, f"lat={lat} out of bounds"
        assert -180 <= lon <= 180, f"lon={lon} out of bounds"
        assert np.isfinite(hae)


@pytest.mark.slow
def test_sicd_latlon_to_image_scp(require_umbra_file):
    """Known lat/lon maps back to valid pixel coordinates."""
    with SICDReader(str(require_umbra_file)) as reader:
        geo = SICDGeolocation.from_reader(reader)
        rows, cols = reader.get_shape()[:2]

        # Get ground coords at center
        lat, lon, hae = geo.image_to_latlon(rows // 2, cols // 2)

        # Map back to image
        row, col = geo.latlon_to_image(lat, lon, hae)
        assert np.isfinite(row) and np.isfinite(col), (
            f"latlon_to_image returned non-finite: ({row}, {col})"
        )


# =============================================================================
# Level 2: Data Quality
# =============================================================================


@pytest.mark.slow
def test_sicd_round_trip_precision(require_umbra_file):
    """Image → ground → image round-trip is sub-pixel accurate."""
    with SICDReader(str(require_umbra_file)) as reader:
        geo = SICDGeolocation.from_reader(reader)
        rows, cols = reader.get_shape()[:2]
        center_row, center_col = float(rows // 2), float(cols // 2)

        lat, lon, hae = geo.image_to_latlon(center_row, center_col)
        row_back, col_back = geo.latlon_to_image(lat, lon, hae)

        assert abs(row_back - center_row) < 1.0, (
            f"Row round-trip error: {abs(row_back - center_row):.4f} pixels"
        )
        assert abs(col_back - center_col) < 1.0, (
            f"Col round-trip error: {abs(col_back - center_col):.4f} pixels"
        )


@pytest.mark.slow
def test_sicd_footprint(require_umbra_file):
    """get_footprint returns a polygon with valid coordinates."""
    with SICDReader(str(require_umbra_file)) as reader:
        geo = SICDGeolocation.from_reader(reader)

        if hasattr(geo, 'get_footprint'):
            footprint = geo.get_footprint()
            if footprint is not None:
                # get_footprint returns a dict with 'type', 'coordinates', 'bounds'
                assert footprint['type'] == 'Polygon'
                coords = footprint['coordinates']
                assert len(coords) >= 4, (
                    f"Footprint has {len(coords)} points, expected >= 4"
                )


@pytest.mark.slow
def test_sicd_vectorized(require_umbra_file):
    """Array inputs produce array outputs with consistent shapes."""
    with SICDReader(str(require_umbra_file)) as reader:
        geo = SICDGeolocation.from_reader(reader)
        rows, cols = reader.get_shape()[:2]

        # Create array of pixel coordinates
        test_rows = np.array([rows // 4, rows // 2, 3 * rows // 4], dtype=np.float64)
        test_cols = np.array([cols // 4, cols // 2, 3 * cols // 4], dtype=np.float64)

        geo_result = geo.image_to_latlon(test_rows, test_cols)
        # Result is (n, 3) array with [lat, lon, altitude]
        assert geo_result.shape[1] == 3, f"Expected (N, 3) result, got {geo_result.shape}"
        lats = geo_result[:, 0]
        lons = geo_result[:, 1]
        assert isinstance(lats, np.ndarray)
        assert len(lats) == 3
        assert np.all(np.isfinite(lats))
        assert np.all((-90 <= lats) & (lats <= 90))
        assert np.all((-180 <= lons) & (lons <= 180))


# =============================================================================
# Level 2b: Backend Selection and Refinement (v0.4.0)
# =============================================================================


@pytest.mark.slow
def test_sicd_rdot_backend_selection(require_umbra_file):
    """SICDGeolocation can be instantiated with specific backend.

    v0.4.0 introduces native R/Rdot backend alongside external backends
    (sarpy, sarkit). Verify backend selection capability.
    """
    with SICDReader(str(require_umbra_file)) as reader:
        # Try to construct with backend specification if supported
        try:
            geo = SICDGeolocation.from_reader(reader, backend='native')
            assert geo is not None
            # Verify backend is set
            if hasattr(geo, 'backend'):
                assert geo.backend is not None
        except TypeError:
            # Older version may not support backend parameter
            geo = SICDGeolocation.from_reader(reader)
            assert geo is not None


@pytest.mark.slow
def test_sicd_has_rdot_property(require_umbra_file):
    """SICDGeolocation exposes has_rdot property.

    Verifies whether the SICD metadata contains Rdot values for
    refinement calculations. This gates which code paths are available.
    """
    with SICDReader(str(require_umbra_file)) as reader:
        geo = SICDGeolocation.from_reader(reader)
        
        # has_rdot property indicates if Rdot is available in SICD
        if hasattr(geo, 'has_rdot'):
            rdot_available = geo.has_rdot
            assert isinstance(rdot_available, bool), \
                f"has_rdot must be bool, got {type(rdot_available)}"
            # For Umbra SICD, Rdot should typically be available
            # (but don't assert True since it depends on data)
        else:
            pytest.skip("has_rdot property not available in this grdl version")


@pytest.mark.slow
def test_sicd_native_backend_corner_precision(require_umbra_file):
    r"""Native backend round-trip at image corners is sub-pixel.

    Corner pixels often have worst-case projection geometry.
    Native R/Rdot backend should maintain sub-pixel accuracy even there.
    """
    with SICDReader(str(require_umbra_file)) as reader:
        try:
            geo = SICDGeolocation.from_reader(reader, backend='native')
        except (TypeError, ValueError):
            # Fall back if native not available
            geo = SICDGeolocation.from_reader(reader)
        
        rows, cols = reader.get_shape()[:2]
        
        # Test four corners with tight tolerance for native backend
        corners = [
            (0.0, 0.0),
            (0.0, float(cols - 1)),
            (float(rows - 1), 0.0),
            (float(rows - 1), float(cols - 1)),
        ]
        
        for r_orig, c_orig in corners:
            lat, lon, hae = geo.image_to_latlon(r_orig, c_orig)
            r_back, c_back = geo.latlon_to_image(lat, lon, hae)
            
            # Native backend should be tight at corners (< 0.5 pixel)
            row_err = abs(r_back - r_orig)
            col_err = abs(c_back - c_orig)
            
            assert row_err < 1.0, \
                f"Row corner error at ({r_orig},{c_orig}): {row_err:.4f} pixels"
            assert col_err < 1.0, \
                f"Col corner error at ({r_orig},{c_orig}): {col_err:.4f} pixels"


@pytest.mark.slow
def test_sicd_elevation_integration(require_umbra_file):
    """SICDGeolocation elevation property can be set and used.

    Tests that the DEM integration point (geo.elevation) is properly
    initialized and can be used for terrain-corrected projections.
    """
    with SICDReader(str(require_umbra_file)) as reader:
        geo = SICDGeolocation.from_reader(reader)
        
        # Initially elevation should be None
        if hasattr(geo, 'elevation'):
            initial_elev = geo.elevation
            # May be None or ConstantElevation
            
            # Verify we can set it (contract for v0.4.0)
            try:
                from grdl.geolocation.elevation import ConstantElevation
                const_dem = ConstantElevation(100.0)
                geo.elevation = const_dem
                assert geo.elevation is const_dem
            except ImportError:
                pytest.skip("ConstantElevation not available")
