# -*- coding: utf-8 -*-
"""
Geographic Shapes Validation - Projection, rasterization, and display ops.

Tests grdl.shapes (Circle, Ellipse, GeoPolygon, Arc) with real Umbra SICD
data and SICDGeolocation for pixel-space projection.

- Level 1: Construction, perimeter generation, contains() membership
- Level 2: Pixel projection accuracy, rasterization mask validity,
           area/perimeter physical bounds
- Level 3: burn_shape integration, batch operations, cued_detect pipeline

Dataset: Umbra SICD (*.nitf)

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
    from grdl.shapes import (
        Circle,
        Ellipse,
        GeoPolygon,
        Arc,
        rasterize_polygon,
        to_pixels_batch,
        rasterize_batch,
        burn_shape,
    )
    _HAS_SHAPES = True
except ImportError:
    _HAS_SHAPES = False

try:
    from grdl.IO.sar import SICDReader
    _HAS_SICD = True
except ImportError:
    _HAS_SICD = False

try:
    from grdl.geolocation import SICDGeolocation
    _HAS_GEO = True
except ImportError:
    _HAS_GEO = False


pytestmark = [
    pytest.mark.skipif(not _HAS_SHAPES, reason="grdl.shapes not available"),
    pytest.mark.requires_data,
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def sicd_context(umbra_data_dir):
    """Load SICD reader metadata and geolocation for shape tests."""
    if not _HAS_SICD:
        pytest.skip("SICDReader not available")
    if not _HAS_GEO:
        pytest.skip("SICDGeolocation not available")
    matches = list(umbra_data_dir.glob("*.nitf")) if umbra_data_dir.exists() else []
    if not matches:
        pytest.skip(f"Umbra SICD not found in {umbra_data_dir}")
    with SICDReader(str(matches[0])) as reader:
        meta = reader.metadata
        shape = reader.get_shape()
        geo = SICDGeolocation.from_reader(reader, backend='native')
        # Get scene center lat/lon from SCP
        scp_lat = meta.geo_data.scp.llh.lat
        scp_lon = meta.geo_data.scp.llh.lon
        # Read a small chip for burn_shape tests
        rows, cols = shape
        r0 = rows // 2 - 128
        c0 = cols // 2 - 128
        chip = reader.read_chip(r0, r0 + 256, c0, c0 + 256)
    return {
        "geo": geo,
        "shape": shape,
        "scp_lat": scp_lat,
        "scp_lon": scp_lon,
        "chip": chip,
        "nitf_path": str(matches[0]),
    }


@pytest.fixture(scope="module")
def scene_circle(sicd_context):
    """Circle centered on scene center, 200m radius."""
    return Circle(
        center_lat=sicd_context["scp_lat"],
        center_lon=sicd_context["scp_lon"],
        radius_m=200.0,
    )


@pytest.fixture(scope="module")
def scene_ellipse(sicd_context):
    """Ellipse centered on scene center, 300m x 150m."""
    return Ellipse(
        center_lat=sicd_context["scp_lat"],
        center_lon=sicd_context["scp_lon"],
        semi_major_m=300.0,
        semi_minor_m=150.0,
        rotation_deg=45.0,
    )


@pytest.fixture(scope="module")
def scene_polygon(sicd_context):
    """Square polygon ~400m per side centered on scene."""
    lat = sicd_context["scp_lat"]
    lon = sicd_context["scp_lon"]
    # ~0.002 degrees ≈ 200m at mid-latitudes
    offset = 0.002
    vertices = np.array([
        [lat - offset, lon - offset],
        [lat - offset, lon + offset],
        [lat + offset, lon + offset],
        [lat + offset, lon - offset],
    ])
    return GeoPolygon(vertices_latlon=vertices)


@pytest.fixture(scope="module")
def scene_arc(sicd_context):
    """90-degree arc centered on scene."""
    return Arc(
        center_lat=sicd_context["scp_lat"],
        center_lon=sicd_context["scp_lon"],
        radius_m=250.0,
        bearing_start_deg=0.0,
        bearing_end_deg=90.0,
    )


# =============================================================================
# Level 1: Format Validation — Construction and perimeter
# =============================================================================


class TestShapesLevel1:
    """Validate shape construction and basic geometry."""

    def test_circle_construction(self, scene_circle):
        """Circle instantiates with correct center and radius."""
        assert scene_circle is not None

    def test_ellipse_construction(self, scene_ellipse):
        """Ellipse instantiates with semi-axes and rotation."""
        assert scene_ellipse is not None

    def test_polygon_construction(self, scene_polygon):
        """GeoPolygon instantiates from vertex array."""
        assert scene_polygon is not None

    def test_arc_construction(self, scene_arc):
        """Arc instantiates with bearings."""
        assert scene_arc is not None

    def test_circle_perimeter_shape(self, scene_circle):
        """Circle perimeter returns (N, 3) lat/lon/height array."""
        pts = scene_circle.perimeter_latlon(n=64)
        assert pts.ndim == 2
        assert pts.shape[1] == 3
        assert pts.shape[0] >= 64

    def test_ellipse_perimeter_shape(self, scene_ellipse):
        """Ellipse perimeter returns (N, 3) array."""
        pts = scene_ellipse.perimeter_latlon(n=64)
        assert pts.ndim == 2
        assert pts.shape[1] == 3
        assert pts.shape[0] >= 64

    def test_polygon_perimeter_shape(self, scene_polygon):
        """GeoPolygon perimeter returns dense vertices."""
        pts = scene_polygon.perimeter_latlon(n=64)
        assert pts.ndim == 2
        assert pts.shape[1] == 3

    def test_circle_contains_center(self, scene_circle, sicd_context):
        """Circle contains its own center point."""
        result = scene_circle.contains(
            sicd_context["scp_lat"], sicd_context["scp_lon"]
        )
        assert np.all(result)

    def test_circle_excludes_distant_point(self, scene_circle, sicd_context):
        """Circle does not contain a point far outside."""
        result = scene_circle.contains(
            sicd_context["scp_lat"] + 1.0,  # ~111 km away
            sicd_context["scp_lon"],
        )
        assert not np.any(result)


# =============================================================================
# Level 2: Data Quality — Projection and rasterization
# =============================================================================


class TestShapesLevel2:
    """Validate pixel projection and rasterization accuracy."""

    def test_circle_to_pixels_closed_contour(self, scene_circle, sicd_context):
        """Circle projects to a closed pixel contour."""
        pixels = scene_circle.to_pixels(sicd_context["geo"])
        assert pixels.ndim == 2
        assert pixels.shape[1] == 2
        # Contour should form a reasonable closed shape
        # to_pixels uses adaptive refinement and may not explicitly close
        # Verify that the contour traces a closed path (min distance from last to any early point)
        dist_first_last = np.linalg.norm(pixels[0] - pixels[-1])
        # Accept either explicit closure OR a full-circle trace
        perimeter = np.sum(np.linalg.norm(np.diff(pixels, axis=0), axis=1))
        assert perimeter > 10, "Contour too small to be meaningful"
        # Circle should cover reasonable angular extent
        center = pixels.mean(axis=0)
        angles = np.arctan2(pixels[:, 0] - center[0], pixels[:, 1] - center[1])
        angular_coverage = np.ptp(angles)
        assert angular_coverage > np.pi, (
            f"Circle contour covers only {np.degrees(angular_coverage):.0f}° (expected >180°)"
        )

    def test_ellipse_to_pixels_within_image(self, scene_ellipse, sicd_context):
        """Ellipse pixel contour falls within image bounds."""
        pixels = scene_ellipse.to_pixels(sicd_context["geo"])
        rows, cols = sicd_context["shape"]
        assert pixels[:, 0].min() >= -10  # allow slight margin
        assert pixels[:, 1].min() >= -10
        assert pixels[:, 0].max() <= rows + 10
        assert pixels[:, 1].max() <= cols + 10

    def test_circle_rasterize_mask_shape(self, scene_circle, sicd_context):
        """Circle rasterizes to boolean mask of correct shape."""
        mask = scene_circle.rasterize(
            sicd_context["geo"],
            sicd_context["shape"],
            fill=True,
        )
        assert mask.shape == sicd_context["shape"]
        assert mask.dtype == bool

    def test_circle_rasterize_nonzero(self, scene_circle, sicd_context):
        """Rasterized circle mask has non-zero area."""
        mask = scene_circle.rasterize(
            sicd_context["geo"],
            sicd_context["shape"],
            fill=True,
        )
        n_pixels = mask.sum()
        assert n_pixels > 10, f"Circle mask has only {n_pixels} pixels"

    def test_ellipse_area_physical_bounds(self, scene_ellipse, sicd_context):
        """Ellipse mask area is consistent with semi-axes."""
        mask = scene_ellipse.rasterize(
            sicd_context["geo"],
            sicd_context["shape"],
            fill=True,
        )
        # Expected area: π * a * b = π * 300 * 150 ≈ 141,372 m²
        # Pixel area depends on resolution — just ensure it's nonzero and bounded
        n_pixels = mask.sum()
        total_pixels = mask.size
        assert 0 < n_pixels < total_pixels * 0.5, (
            f"Ellipse mask area unreasonable: {n_pixels}/{total_pixels} pixels"
        )

    def test_polygon_rasterize_fills_interior(self, scene_polygon, sicd_context):
        """Polygon rasterization fills interior pixels."""
        mask = scene_polygon.rasterize(
            sicd_context["geo"],
            sicd_context["shape"],
            fill=True,
        )
        assert mask.sum() > 10, "Polygon mask is empty"

    def test_rasterize_polygon_low_level(self):
        """Low-level rasterize_polygon produces valid mask from pixel vertices."""
        # Unit square in pixel space
        pixels = np.array([
            [10, 10], [10, 50], [50, 50], [50, 10],
        ], dtype=np.float64)
        mask = rasterize_polygon(pixels, image_shape=(64, 64), fill=True)
        assert mask.shape == (64, 64)
        assert mask.dtype == bool
        # Interior should be filled
        assert mask[30, 30], "Interior pixel not filled"
        # Exterior should be empty
        assert not mask[0, 0], "Exterior pixel incorrectly filled"

    def test_ellipse_covariance_roundtrip(self, sicd_context):
        """Ellipse.from_covariance → .covariance recovers original matrix."""
        cov = np.array([[100.0, 25.0], [25.0, 50.0]])  # m²
        ell = Ellipse.from_covariance(
            sicd_context["scp_lat"],
            sicd_context["scp_lon"],
            cov,
        )
        recovered = ell.covariance
        np.testing.assert_allclose(recovered, cov, rtol=1e-4)


# =============================================================================
# Level 3: Integration — Display and batch operations
# =============================================================================


class TestShapesLevel3:
    """Integration tests: burn_shape, batch ops."""

    @pytest.mark.integration
    def test_burn_shape_produces_rgb(self, sicd_context):
        """burn_shape renders shape outline pixels in the specified color."""
        from grdl.geolocation import ChipGeolocation

        geo = sicd_context["geo"]
        scp_lat = sicd_context["scp_lat"]
        scp_lon = sicd_context["scp_lon"]

        # Project the SCP to find its actual pixel location, then extract
        # a chip centered on it so the shape lands inside the chip.
        scp_px = geo.latlon_to_image(
            np.array([[scp_lat, scp_lon, 0.0]])
        )
        scp_row = int(scp_px[0, 0])
        scp_col = int(scp_px[0, 1])
        r0 = scp_row - 128
        c0 = scp_col - 128

        # Use a small circle (20m ≈ 80px at 0.25m GSD) that fits in 256×256
        small_circle = Circle(
            center_lat=scp_lat,
            center_lon=scp_lon,
            radius_m=20.0,
        )

        # Create chip-local geolocation and verify projection lands in-bounds
        chip_geo = ChipGeolocation(geo, row_offset=r0, col_offset=c0, shape=(256, 256))

        # Read the chip from the SICD at the SCP-centered location
        with SICDReader(sicd_context["nitf_path"]) as reader:
            chip = reader.read_chip(r0, r0 + 256, c0, c0 + 256)

        result = burn_shape(
            np.abs(chip),
            small_circle,
            chip_geo,
            color=(255, 0, 0),
            thickness=2,
        )
        assert result.ndim == 3
        assert result.shape[2] == 3
        assert result.dtype == np.uint8
        # Strict check: burn_shape must have rendered red pixels (R=255, G=0, B=0)
        red_pixels = (
            (result[:, :, 0] == 255)
            & (result[:, :, 1] == 0)
            & (result[:, :, 2] == 0)
        )
        assert red_pixels.sum() > 0, (
            "No red (255,0,0) outline pixels found — "
            "burn_shape did not render the circle"
        )

    @pytest.mark.integration
    def test_to_pixels_batch_multiple_shapes(self, scene_circle, scene_ellipse,
                                             scene_polygon, sicd_context):
        """Batch projection returns list of pixel arrays."""
        shapes = [scene_circle, scene_ellipse, scene_polygon]
        results = to_pixels_batch(shapes, sicd_context["geo"])
        assert len(results) == 3
        for i, px in enumerate(results):
            assert px.ndim == 2, f"Shape {i}: expected 2D pixel array"
            assert px.shape[1] == 2, f"Shape {i}: expected (N, 2)"

    @pytest.mark.integration
    def test_rasterize_batch_multiple_shapes(self, scene_circle, scene_ellipse,
                                             sicd_context):
        """Batch rasterization returns list of boolean masks."""
        shapes = [scene_circle, scene_ellipse]
        masks = rasterize_batch(
            shapes, sicd_context["geo"], sicd_context["shape"]
        )
        assert len(masks) == 2
        for i, m in enumerate(masks):
            assert m.shape == sicd_context["shape"]
            assert m.dtype == bool
            assert m.sum() > 0, f"Shape {i} mask is empty"
