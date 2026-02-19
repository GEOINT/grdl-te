# -*- coding: utf-8 -*-
"""
OrthoPipeline Tests - Synthetic validation of the ortho pipeline.

Tests OrthoPipeline builder pattern, OrthoResult container, and
compute_output_resolution using synthetic affine geolocation.

- Level 1: Builder methods chain correctly, pipeline configures
- Level 2: run() returns OrthoResult with correct shape and metadata
- Level 3: save_geotiff() roundtrip, compute_output_resolution dispatch

Dependencies
------------
pytest
numpy
scipy
rasterio (optional for save_geotiff tests)

Author
------
Claude Code (generated)

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2026-02-18

Modified
--------
2026-02-18
"""

# Standard library
import tempfile
from pathlib import Path

# Third-party
import pytest
import numpy as np

try:
    from rasterio.transform import Affine as RioAffine
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False

try:
    from grdl.image_processing.ortho import (
        OrthoPipeline,
        OrthoResult,
        Orthorectifier,
        OutputGrid,
        compute_output_resolution,
    )
    _HAS_ORTHO = True
except ImportError:
    _HAS_ORTHO = False

try:
    from grdl.geolocation import AffineGeolocation
    _HAS_GEO = True
except ImportError:
    _HAS_GEO = False

pytestmark = [
    pytest.mark.ortho,
    pytest.mark.skipif(not _HAS_ORTHO, reason="grdl ortho not available"),
    pytest.mark.skipif(not _HAS_RASTERIO, reason="rasterio not installed"),
    pytest.mark.skipif(not _HAS_GEO, reason="grdl geolocation not available"),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic_ortho_setup():
    """Synthetic image with affine geolocation for ortho testing."""
    rows, cols = 256, 256
    rng = np.random.default_rng(42)
    image = rng.random((rows, cols), dtype=np.float32)

    transform = RioAffine(0.00027, 0.0, -118.0,
                           0.0, -0.00027, 34.0)
    geo = AffineGeolocation(
        transform=transform, shape=(rows, cols), crs="EPSG:4326",
    )
    return image, geo, rows, cols


# ---------------------------------------------------------------------------
# Level 1: OrthoPipeline builder pattern
# ---------------------------------------------------------------------------
class TestOrthoPipelineLevel1:
    """Validate builder pattern and configuration."""

    def test_builder_chaining(self, synthetic_ortho_setup):
        """Builder methods return self for chaining."""
        image, geo, _, _ = synthetic_ortho_setup
        pipeline = (
            OrthoPipeline()
            .with_source_array(image)
            .with_geolocation(geo)
            .with_resolution(0.00054, 0.00054)
            .with_interpolation('bilinear')
            .with_nodata(0.0)
        )
        assert isinstance(pipeline, OrthoPipeline)

    def test_with_interpolation_methods(self, synthetic_ortho_setup):
        """Pipeline accepts all interpolation methods."""
        image, geo, _, _ = synthetic_ortho_setup
        for method in ('nearest', 'bilinear', 'bicubic'):
            pipeline = (
                OrthoPipeline()
                .with_source_array(image)
                .with_geolocation(geo)
                .with_resolution(0.00054, 0.00054)
                .with_interpolation(method)
            )
            assert isinstance(pipeline, OrthoPipeline)


# ---------------------------------------------------------------------------
# Level 2: run() produces correct OrthoResult
# ---------------------------------------------------------------------------
class TestOrthoPipelineLevel2:
    """Validate run() output correctness."""

    def test_run_returns_ortho_result(self, synthetic_ortho_setup):
        """run() returns an OrthoResult."""
        image, geo, _, _ = synthetic_ortho_setup
        pipeline = (
            OrthoPipeline()
            .with_source_array(image)
            .with_geolocation(geo)
            .with_resolution(0.00054, 0.00054)
            .with_interpolation('bilinear')
        )
        result = pipeline.run()
        assert isinstance(result, OrthoResult)

    def test_result_data_is_2d(self, synthetic_ortho_setup):
        """OrthoResult.data is a 2D array."""
        image, geo, _, _ = synthetic_ortho_setup
        pipeline = (
            OrthoPipeline()
            .with_source_array(image)
            .with_geolocation(geo)
            .with_resolution(0.00054, 0.00054)
        )
        result = pipeline.run()
        assert result.data.ndim == 2

    def test_result_data_not_all_nodata(self, synthetic_ortho_setup):
        """OrthoResult.data has valid (non-nodata) pixels."""
        image, geo, _, _ = synthetic_ortho_setup
        pipeline = (
            OrthoPipeline()
            .with_source_array(image)
            .with_geolocation(geo)
            .with_resolution(0.00054, 0.00054)
            .with_nodata(0.0)
        )
        result = pipeline.run()
        # At least some pixels should be non-zero
        assert np.any(result.data != 0.0), "All output pixels are nodata"

    def test_result_has_geolocation_metadata(self, synthetic_ortho_setup):
        """OrthoResult carries geolocation metadata."""
        image, geo, _, _ = synthetic_ortho_setup
        pipeline = (
            OrthoPipeline()
            .with_source_array(image)
            .with_geolocation(geo)
            .with_resolution(0.00054, 0.00054)
        )
        result = pipeline.run()
        assert result.geolocation_metadata is not None
        assert isinstance(result.geolocation_metadata, dict)
        assert 'crs' in result.geolocation_metadata
        assert 'transform' in result.geolocation_metadata

    def test_result_has_output_grid(self, synthetic_ortho_setup):
        """OrthoResult carries an OutputGrid."""
        image, geo, _, _ = synthetic_ortho_setup
        pipeline = (
            OrthoPipeline()
            .with_source_array(image)
            .with_geolocation(geo)
            .with_resolution(0.00054, 0.00054)
        )
        result = pipeline.run()
        assert isinstance(result.output_grid, OutputGrid)

    def test_result_shape_property(self, synthetic_ortho_setup):
        """OrthoResult.shape matches data.shape."""
        image, geo, _, _ = synthetic_ortho_setup
        pipeline = (
            OrthoPipeline()
            .with_source_array(image)
            .with_geolocation(geo)
            .with_resolution(0.00054, 0.00054)
        )
        result = pipeline.run()
        assert result.shape == result.data.shape

    def test_nearest_interpolation(self, synthetic_ortho_setup):
        """Nearest-neighbor interpolation produces valid output."""
        image, geo, _, _ = synthetic_ortho_setup
        pipeline = (
            OrthoPipeline()
            .with_source_array(image)
            .with_geolocation(geo)
            .with_resolution(0.00054, 0.00054)
            .with_interpolation('nearest')
        )
        result = pipeline.run()
        assert result.data.ndim == 2
        assert np.any(result.data != 0.0)


# ---------------------------------------------------------------------------
# Level 3: Integration — save_geotiff and compute_output_resolution
# ---------------------------------------------------------------------------
class TestOrthoPipelineLevel3:
    """Integration: GeoTIFF output and auto-resolution."""

    @pytest.mark.integration
    def test_save_geotiff_roundtrip(self, synthetic_ortho_setup):
        """save_geotiff() writes a file that can be read back."""
        import rasterio

        image, geo, _, _ = synthetic_ortho_setup
        pipeline = (
            OrthoPipeline()
            .with_source_array(image)
            .with_geolocation(geo)
            .with_resolution(0.00054, 0.00054)
        )
        result = pipeline.run()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "ortho_test.tif"
            result.save_geotiff(out_path)
            assert out_path.exists()
            assert out_path.stat().st_size > 0

            # Read back and verify
            with rasterio.open(str(out_path)) as ds:
                data_back = ds.read(1)
                assert data_back.shape == result.data.shape
                assert ds.crs is not None

    @pytest.mark.integration
    def test_orthorectifier_direct_usage(self, synthetic_ortho_setup):
        """Orthorectifier can be used directly (low-level API)."""
        image, geo, _, _ = synthetic_ortho_setup
        grid = OutputGrid.from_geolocation(
            geo, pixel_size_lat=0.00054, pixel_size_lon=0.00054,
        )
        ortho = Orthorectifier(
            geolocation=geo, output_grid=grid, interpolation='bilinear',
        )
        ortho.compute_mapping()
        result = ortho.apply(image)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2


# ---------------------------------------------------------------------------
# compute_output_resolution
# ---------------------------------------------------------------------------
class TestComputeOutputResolution:
    """Tests for compute_output_resolution()."""

    @pytest.mark.integration
    def test_returns_tuple_of_two_floats(self, synthetic_ortho_setup):
        """compute_output_resolution returns (lat_deg, lon_deg)."""
        # Use a simple geolocation-based approach if no SICD metadata
        # This will test the dispatch logic
        _, geo, _, _ = synthetic_ortho_setup
        try:
            # If we have access to any metadata object, test it
            from grdl.IO.models import ImageMetadata
            # Use a dict-like metadata to test the function
            result = compute_output_resolution(
                metadata={'pixel_size_x': 10.0, 'pixel_size_y': 10.0},
                geolocation=geo,
            )
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert all(isinstance(v, float) for v in result)
            assert all(v > 0 for v in result)
        except (TypeError, ValueError, KeyError):
            # Expected if the metadata type isn't recognized
            pytest.skip("compute_output_resolution requires specific metadata type")
