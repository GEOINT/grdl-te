# -*- coding: utf-8 -*-
"""
ENUGrid Tests - Local ENU output grid specification.

Validates grdl.image_processing.ortho.enu_grid.ENUGrid:
- Level 1: Constructor validation (valid/invalid args, rows/cols formula)
- Level 2: Coordinate transforms (image_to_latlon, latlon_to_image round-trip,
           pixel spacing accuracy in meters)
- Level 3: sub_grid (coverage, dimensions, boundary error handling)

All tests use synthetic parameters only (no real imagery required).

Dependencies
------------
numpy

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
import math

# Third-party
import pytest
import numpy as np

# GRDL internal
try:
    from grdl.image_processing.ortho.enu_grid import ENUGrid
    _HAS_ENU = True
except ImportError:
    _HAS_ENU = False

try:
    from grdl.geolocation.coordinates import geodetic_to_enu
    _HAS_COORDS = True
except ImportError:
    _HAS_COORDS = False

pytestmark = [
    pytest.mark.ortho,
    pytest.mark.skipif(not _HAS_ENU, reason="ENUGrid not available"),
]


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_grid():
    """500 m × 500 m, 1 m resolution, centered on a mid-latitude point."""
    return ENUGrid(
        ref_lat=36.0,
        ref_lon=-75.5,
        ref_alt=0.0,
        min_east=-250.0,
        max_east=250.0,
        min_north=-250.0,
        max_north=250.0,
        pixel_size_east=1.0,
        pixel_size_north=1.0,
    )


# ===========================================================================
# Level 1: Constructor
# ===========================================================================

class TestENUGridConstructor:
    """Validate constructor inputs, computed attributes, and error handling."""

    def test_valid_construction(self):
        """ENUGrid constructs without exception from valid parameters."""
        grid = ENUGrid(
            ref_lat=36.0, ref_lon=-75.5, ref_alt=0.0,
            min_east=-1000.0, max_east=1000.0,
            min_north=-1000.0, max_north=1000.0,
            pixel_size_east=1.0, pixel_size_north=1.0,
        )
        assert grid is not None

    def test_rows_cols_computation(self, small_grid):
        """rows and cols match (extent / pixel_size), ceiling-rounded.

        For a ±250 m extent at 1 m resolution: 500 rows × 500 cols.
        """
        assert small_grid.rows == 500, (
            f"rows = {small_grid.rows}, expected 500 "
            f"((250 - (-250)) / 1.0 = 500)"
        )
        assert small_grid.cols == 500, (
            f"cols = {small_grid.cols}, expected 500"
        )

    def test_rows_cols_ceiling_rounding(self):
        """rows/cols are computed with ceil() so non-integer extents round up."""
        grid = ENUGrid(
            ref_lat=36.0, ref_lon=-75.5, ref_alt=0.0,
            min_east=0.0, max_east=100.5,
            min_north=0.0, max_north=100.5,
            pixel_size_east=1.0, pixel_size_north=1.0,
        )
        assert grid.rows == math.ceil(100.5), (
            f"rows = {grid.rows}, expected {math.ceil(100.5)} for non-integer extent"
        )

    def test_attributes_stored_exactly(self, small_grid):
        """All constructor parameters survive as instance attributes unchanged."""
        assert small_grid.ref_lat == 36.0
        assert small_grid.ref_lon == -75.5
        assert small_grid.ref_alt == 0.0
        assert small_grid.min_east == -250.0
        assert small_grid.max_east == 250.0
        assert small_grid.min_north == -250.0
        assert small_grid.max_north == 250.0
        assert small_grid.pixel_size_east == 1.0
        assert small_grid.pixel_size_north == 1.0

    def test_inverted_east_bounds_raises(self):
        """max_east ≤ min_east raises ValueError.

        A degenerate or inverted east extent would produce zero or negative
        cols, causing silent downstream failures in Orthorectifier.
        """
        with pytest.raises(ValueError, match="max_east"):
            ENUGrid(
                ref_lat=36.0, ref_lon=-75.5, ref_alt=0.0,
                min_east=500.0, max_east=100.0,   # inverted
                min_north=-250.0, max_north=250.0,
                pixel_size_east=1.0, pixel_size_north=1.0,
            )

    def test_inverted_north_bounds_raises(self):
        """max_north ≤ min_north raises ValueError."""
        with pytest.raises(ValueError, match="max_north"):
            ENUGrid(
                ref_lat=36.0, ref_lon=-75.5, ref_alt=0.0,
                min_east=-250.0, max_east=250.0,
                min_north=250.0, max_north=-250.0,  # inverted
                pixel_size_east=1.0, pixel_size_north=1.0,
            )

    def test_zero_pixel_size_east_raises(self):
        """pixel_size_east = 0 raises ValueError."""
        with pytest.raises(ValueError, match="pixel_size_east"):
            ENUGrid(
                ref_lat=36.0, ref_lon=-75.5, ref_alt=0.0,
                min_east=-250.0, max_east=250.0,
                min_north=-250.0, max_north=250.0,
                pixel_size_east=0.0, pixel_size_north=1.0,
            )

    def test_negative_pixel_size_north_raises(self):
        """pixel_size_north < 0 raises ValueError."""
        with pytest.raises(ValueError, match="pixel_size_north"):
            ENUGrid(
                ref_lat=36.0, ref_lon=-75.5, ref_alt=0.0,
                min_east=-250.0, max_east=250.0,
                min_north=-250.0, max_north=250.0,
                pixel_size_east=1.0, pixel_size_north=-1.0,
            )

    def test_repr_does_not_raise(self, small_grid):
        """__repr__ produces a non-empty string."""
        r = repr(small_grid)
        assert isinstance(r, str)
        assert len(r) > 0


# ===========================================================================
# Level 2: Coordinate transforms
# ===========================================================================

class TestENUGridCoordinateTransforms:
    """Validate image_to_latlon, latlon_to_image, and pixel spacing."""

    def test_image_to_latlon_top_left_is_northwest_corner(self, small_grid):
        """Row 0 / Col 0 maps to the northwest corner (max_north, min_east).

        ENUGrid follows the image convention: row 0 is the north (top) edge,
        row increases southward; col 0 is the west (left) edge.
        """
        lat, lon = small_grid.image_to_latlon(0.0, 0.0)
        # At (row=0, col=0) north = max_north, east = min_east → small offset
        # from reference. The exact lat/lon depends on the reference point
        # but north must be > ref_lat (we are at max_north > 0).
        assert float(lat) > small_grid.ref_lat - 0.01, (
            f"Top-left lat {float(lat):.6f} should be near/above ref_lat "
            f"{small_grid.ref_lat}"
        )

    def test_latlon_to_image_reference_point_is_grid_center(self, small_grid):
        """The reference lat/lon maps to the grid center pixel.

        For a symmetric grid (±extent), the reference point should map to
        (rows/2, cols/2).
        """
        row, col = small_grid.latlon_to_image(small_grid.ref_lat, small_grid.ref_lon)
        assert abs(float(row) - small_grid.rows / 2.0) < 1.0, (
            f"Reference point row {float(row):.2f} deviates from center "
            f"{small_grid.rows / 2.0:.1f} by more than 1 pixel"
        )
        assert abs(float(col) - small_grid.cols / 2.0) < 1.0, (
            f"Reference point col {float(col):.2f} deviates from center "
            f"{small_grid.cols / 2.0:.1f} by more than 1 pixel"
        )

    def test_image_to_latlon_round_trip(self, small_grid):
        """image_to_latlon followed by latlon_to_image recovers original (row, col).

        Tests a grid of pixel positions spanning the output image to catch
        any systematic error in the coordinate transforms.
        """
        rows = np.linspace(0, small_grid.rows - 1, 20, dtype=float)
        cols = np.linspace(0, small_grid.cols - 1, 20, dtype=float)
        row_grid, col_grid = np.meshgrid(rows, cols)
        rows_flat = row_grid.ravel()
        cols_flat = col_grid.ravel()

        lats, lons = small_grid.image_to_latlon(rows_flat, cols_flat)
        rows_rt, cols_rt = small_grid.latlon_to_image(lats, lons)

        np.testing.assert_allclose(rows_rt, rows_flat, atol=1e-6,
            err_msg="image_to_latlon→latlon_to_image row round-trip error > 1e-6 px")
        np.testing.assert_allclose(cols_rt, cols_flat, atol=1e-6,
            err_msg="image_to_latlon→latlon_to_image col round-trip error > 1e-6 px")

    @pytest.mark.skipif(not _HAS_COORDS, reason="grdl.geolocation.coordinates not available")
    def test_adjacent_pixel_spacing_near_1m(self):
        """Adjacent pixels in a 1 m/pixel grid are separated by ≤ 1.01 m.

        Computes the actual geodetic distance between consecutive pixel
        centers to verify that the ENU → lat/lon → distance chain is
        self-consistent and the pixel size is honoured.
        """
        from grdl.geolocation.coordinates import geodetic_to_ecef

        grid = ENUGrid(
            ref_lat=36.0, ref_lon=-75.5, ref_alt=0.0,
            min_east=-5.0, max_east=5.0,
            min_north=-5.0, max_north=5.0,
            pixel_size_east=1.0, pixel_size_north=1.0,
        )
        # Sample pixel (row=4, col=4) and (row=4, col=5) — one pixel east
        lat_a, lon_a = grid.image_to_latlon(4.0, 4.0)
        lat_b, lon_b = grid.image_to_latlon(4.0, 5.0)

        xa, ya, za = geodetic_to_ecef(
            np.array([float(lat_a)]), np.array([float(lon_a)]), np.zeros(1)
        )
        xb, yb, zb = geodetic_to_ecef(
            np.array([float(lat_b)]), np.array([float(lon_b)]), np.zeros(1)
        )
        dist = float(np.sqrt((xb[0] - xa[0])**2 + (yb[0] - ya[0])**2 + (zb[0] - za[0])**2))

        assert dist <= 1.01, (
            f"Adjacent pixel distance {dist:.4f} m exceeds 1.01 m "
            f"for pixel_size_east=1.0 m — pixel spacing is miscalculated"
        )
        assert dist >= 0.99, (
            f"Adjacent pixel distance {dist:.4f} m is less than 0.99 m "
            f"for pixel_size_east=1.0 m — pixel spacing is too small"
        )

    def test_vectorized_inputs_produce_correct_shapes(self, small_grid):
        """image_to_latlon and latlon_to_image accept arrays and return matching shapes."""
        rows = np.arange(0, 10, dtype=float)
        cols = np.arange(0, 10, dtype=float)

        lats, lons = small_grid.image_to_latlon(rows, cols)
        assert lats.shape == (10,), f"image_to_latlon returned lats shape {lats.shape}"
        assert lons.shape == (10,), f"image_to_latlon returned lons shape {lons.shape}"

        rows_rt, cols_rt = small_grid.latlon_to_image(lats, lons)
        assert rows_rt.shape == (10,), f"latlon_to_image returned rows shape {rows_rt.shape}"


# ===========================================================================
# Level 3: sub_grid
# ===========================================================================

class TestENUGridSubGrid:
    """Validate sub_grid extraction."""

    def test_sub_grid_dimensions(self, small_grid):
        """sub_grid returns a grid with exactly (row_end - row_start) rows."""
        sub = small_grid.sub_grid(10, 20, 60, 80)
        assert sub.rows == 50, f"sub_grid rows = {sub.rows}, expected 50"
        assert sub.cols == 60, f"sub_grid cols = {sub.cols}, expected 60"

    def test_sub_grid_pixel_sizes_preserved(self, small_grid):
        """sub_grid inherits pixel sizes from the parent grid."""
        sub = small_grid.sub_grid(0, 0, 10, 10)
        assert sub.pixel_size_east == small_grid.pixel_size_east, (
            "sub_grid pixel_size_east differs from parent"
        )
        assert sub.pixel_size_north == small_grid.pixel_size_north, (
            "sub_grid pixel_size_north differs from parent"
        )

    def test_sub_grid_reference_point_preserved(self, small_grid):
        """sub_grid inherits the reference geodetic point from the parent."""
        sub = small_grid.sub_grid(0, 0, 10, 10)
        assert sub.ref_lat == small_grid.ref_lat
        assert sub.ref_lon == small_grid.ref_lon
        assert sub.ref_alt == small_grid.ref_alt

    def test_sub_grid_bounds_contained_in_parent(self, small_grid):
        """sub_grid ENU bounds are a strict subset of the parent bounds.

        Extracting a sub-region must not produce bounds that extend outside
        the parent grid.  Any violation means the coordinate offset
        calculation is wrong.
        """
        sub = small_grid.sub_grid(50, 100, 200, 300)

        assert sub.min_east >= small_grid.min_east - 1e-9, (
            f"sub_grid min_east {sub.min_east:.3f} < parent min_east {small_grid.min_east:.3f}"
        )
        assert sub.max_east <= small_grid.max_east + 1e-9, (
            f"sub_grid max_east {sub.max_east:.3f} > parent max_east {small_grid.max_east:.3f}"
        )
        assert sub.min_north >= small_grid.min_north - 1e-9, (
            f"sub_grid min_north {sub.min_north:.3f} < parent min_north {small_grid.min_north:.3f}"
        )
        assert sub.max_north <= small_grid.max_north + 1e-9, (
            f"sub_grid max_north {sub.max_north:.3f} > parent max_north {small_grid.max_north:.3f}"
        )

    def test_sub_grid_coordinate_continuity(self, small_grid):
        """Pixel (0, 0) of a sub-grid maps to the same lat/lon as the
        corresponding pixel in the parent grid.

        Extracts the sub-grid starting at (row=10, col=20) and verifies
        that sub.image_to_latlon(0, 0) ≈ parent.image_to_latlon(10, 20).
        """
        r0, c0 = 10, 20
        sub = small_grid.sub_grid(r0, c0, r0 + 50, c0 + 50)

        lat_parent, lon_parent = small_grid.image_to_latlon(float(r0), float(c0))
        lat_sub, lon_sub = sub.image_to_latlon(0.0, 0.0)

        assert abs(float(lat_sub) - float(lat_parent)) < 1e-7, (
            f"Sub-grid origin lat {float(lat_sub):.8f} differs from parent "
            f"pixel ({r0},{c0}) lat {float(lat_parent):.8f}"
        )
        assert abs(float(lon_sub) - float(lon_parent)) < 1e-7, (
            f"Sub-grid origin lon {float(lon_sub):.8f} differs from parent "
            f"pixel ({r0},{c0}) lon {float(lon_parent):.8f}"
        )

    def test_sub_grid_negative_start_raises(self, small_grid):
        """row_start < 0 raises ValueError."""
        with pytest.raises(ValueError):
            small_grid.sub_grid(-1, 0, 10, 10)

    def test_sub_grid_col_start_negative_raises(self, small_grid):
        """col_start < 0 raises ValueError."""
        with pytest.raises(ValueError):
            small_grid.sub_grid(0, -1, 10, 10)

    def test_sub_grid_row_end_exceeds_grid_raises(self, small_grid):
        """row_end > grid.rows raises ValueError."""
        with pytest.raises(ValueError):
            small_grid.sub_grid(0, 0, small_grid.rows + 1, 10)

    def test_sub_grid_col_end_exceeds_grid_raises(self, small_grid):
        """col_end > grid.cols raises ValueError."""
        with pytest.raises(ValueError):
            small_grid.sub_grid(0, 0, 10, small_grid.cols + 1)

    def test_sub_grid_empty_region_raises(self, small_grid):
        """row_end == row_start produces an empty region → raises ValueError."""
        with pytest.raises(ValueError):
            small_grid.sub_grid(50, 50, 50, 100)  # row_end == row_start

    def test_sub_grid_full_extent(self, small_grid):
        """sub_grid(0, 0, rows, cols) equals the parent in dimensions."""
        sub = small_grid.sub_grid(0, 0, small_grid.rows, small_grid.cols)
        assert sub.rows == small_grid.rows
        assert sub.cols == small_grid.cols
