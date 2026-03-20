# -*- coding: utf-8 -*-
"""
OrthoBuilder Tests - Synthetic validation of the ortho pipeline.

Tests OrthoBuilder builder pattern, OrthoResult container, and
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
        OrthoBuilder,
        OrthoResult,
        Orthorectifier,
        OutputGrid,
        compute_output_resolution,
    )
    _HAS_ORTHO = True
except ImportError:
    _HAS_ORTHO = False

try:
    from grdl.image_processing.ortho.enu_grid import ENUGrid
    _HAS_ENU = True
except ImportError:
    _HAS_ENU = False

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
# Level 1: OrthoBuilder builder pattern
# ---------------------------------------------------------------------------
class TestOrthoBuilderLevel1:
    """Validate builder pattern and configuration."""

    def test_builder_chaining(self, synthetic_ortho_setup):
        """Builder methods return self for chaining."""
        image, geo, _, _ = synthetic_ortho_setup
        pipeline = (
            OrthoBuilder()
            .with_source_array(image)
            .with_geolocation(geo)
            .with_resolution(0.00054, 0.00054)
            .with_interpolation('bilinear')
            .with_nodata(0.0)
        )
        assert isinstance(pipeline, OrthoBuilder)

    def test_with_interpolation_methods(self, synthetic_ortho_setup):
        """Pipeline accepts all interpolation methods."""
        image, geo, _, _ = synthetic_ortho_setup
        for method in ('nearest', 'bilinear', 'bicubic'):
            pipeline = (
                OrthoBuilder()
                .with_source_array(image)
                .with_geolocation(geo)
                .with_resolution(0.00054, 0.00054)
                .with_interpolation(method)
            )
            assert isinstance(pipeline, OrthoBuilder)


# ---------------------------------------------------------------------------
# Level 2: run() produces correct OrthoResult
# ---------------------------------------------------------------------------
class TestOrthoBuilderLevel2:
    """Validate run() output correctness."""

    def test_run_returns_ortho_result(self, synthetic_ortho_setup):
        """run() returns an OrthoResult."""
        image, geo, _, _ = synthetic_ortho_setup
        pipeline = (
            OrthoBuilder()
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
            OrthoBuilder()
            .with_source_array(image)
            .with_geolocation(geo)
            .with_resolution(0.00054, 0.00054)
        )
        result = pipeline.run()
        assert result.data.ndim == 2

    def test_result_data_value_range_conserved(self, synthetic_ortho_setup):
        """Bilinear interpolation must not extrapolate beyond input value range.

        Bilinear interpolation is a convex combination of neighbours, so every
        output value must lie within [min(input), max(input)].  An output below
        the input minimum or above the input maximum proves the interpolator is
        performing extrapolation or using wrong weights.
        """
        image, geo, _, _ = synthetic_ortho_setup
        pipeline = (
            OrthoBuilder()
            .with_source_array(image)
            .with_geolocation(geo)
            .with_resolution(0.00054, 0.00054)
            .with_nodata(0.0)
            .with_interpolation('bilinear')
        )
        result = pipeline.run()
        valid = result.data[result.data != 0.0]
        assert len(valid) > 0, "All output pixels are nodata"
        assert valid.min() >= image.min() - 1e-5, (
            f"Interpolated min {valid.min():.6f} is below input min "
            f"{image.min():.6f} — bilinear is extrapolating"
        )
        assert valid.max() <= image.max() + 1e-5, (
            f"Interpolated max {valid.max():.6f} is above input max "
            f"{image.max():.6f} — bilinear is extrapolating"
        )

    def test_result_has_geolocation_metadata(self, synthetic_ortho_setup):
        """OrthoResult carries geolocation metadata with non-null values.

        Checks key presence AND that the values are non-null.  A dict of
        {'crs': None, 'transform': None} satisfies key-existence tests but
        is useless for downstream GIS consumers.
        """
        image, geo, _, _ = synthetic_ortho_setup
        pipeline = (
            OrthoBuilder()
            .with_source_array(image)
            .with_geolocation(geo)
            .with_resolution(0.00054, 0.00054)
        )
        result = pipeline.run()
        assert isinstance(result.geolocation_metadata, dict)
        assert result.geolocation_metadata.get('crs') is not None, (
            "geolocation_metadata['crs'] must not be None"
        )
        assert result.geolocation_metadata.get('transform') is not None, (
            "geolocation_metadata['transform'] must not be None"
        )
        # CRS must reference WGS84 / EPSG:4326.  The library may store this as
        # the string 'WGS84', 'EPSG:4326', or an authority object — accept any.
        crs_str = str(result.geolocation_metadata['crs']).upper()
        assert 'WGS84' in crs_str or '4326' in crs_str, (
            f"Expected WGS84 / EPSG:4326 in CRS, got: "
            f"{result.geolocation_metadata['crs']!r}"
        )

    def test_result_has_output_grid(self, synthetic_ortho_setup):
        """OrthoResult carries an OutputGrid."""
        image, geo, _, _ = synthetic_ortho_setup
        pipeline = (
            OrthoBuilder()
            .with_source_array(image)
            .with_geolocation(geo)
            .with_resolution(0.00054, 0.00054)
        )
        result = pipeline.run()
        assert isinstance(result.output_grid, OutputGrid)

    def test_result_shape_reflects_resolution(self, synthetic_ortho_setup):
        """Coarser pixel size produces a smaller output array.

        The original tautology 'result.shape == result.data.shape' proved
        nothing — shape is always equal to itself.  This test verifies that
        the pipeline actually uses the requested resolution: doubling the
        pixel size must approximately halve the pixel count in each dimension.
        """
        image, geo, _, _ = synthetic_ortho_setup
        fine = (
            OrthoBuilder()
            .with_source_array(image)
            .with_geolocation(geo)
            .with_resolution(0.00027, 0.00027)   # 1× input pixel size
        ).run()
        coarse = (
            OrthoBuilder()
            .with_source_array(image)
            .with_geolocation(geo)
            .with_resolution(0.00054, 0.00054)   # 2× coarser
        ).run()
        assert coarse.data.shape[0] < fine.data.shape[0], (
            "Coarser resolution must produce fewer output rows"
        )
        assert coarse.data.shape[1] < fine.data.shape[1], (
            "Coarser resolution must produce fewer output columns"
        )

    def test_nearest_interpolation(self, synthetic_ortho_setup):
        """Nearest-neighbor produces values that are a strict subset of input values.

        Nearest-neighbour never blends pixels, so every non-nodata output
        value must be exactly present in the input array.  Bilinear output
        can contain values between input values, so the two methods must
        produce different results — if they are pixel-identical, one of them
        is not actually using its stated algorithm.
        """
        image, geo, _, _ = synthetic_ortho_setup
        nearest_result = (
            OrthoBuilder()
            .with_source_array(image)
            .with_geolocation(geo)
            .with_resolution(0.00054, 0.00054)
            .with_nodata(0.0)
            .with_interpolation('nearest')
        ).run()
        bilinear_result = (
            OrthoBuilder()
            .with_source_array(image)
            .with_geolocation(geo)
            .with_resolution(0.00054, 0.00054)
            .with_nodata(0.0)
            .with_interpolation('bilinear')
        ).run()

        # Nearest values must all exist in the input pixel set
        valid_nearest = nearest_result.data[nearest_result.data != 0.0]
        input_values = set(image.flatten().tolist())
        for v in valid_nearest.flatten()[:500]:   # sample first 500 to keep fast
            assert v in input_values, (
                f"Nearest-neighbour output value {v} not found in input — "
                "nearest is blending pixels like bilinear"
            )

        # The two methods must not produce bit-identical results
        assert not np.array_equal(nearest_result.data, bilinear_result.data), (
            "Nearest and bilinear interpolation produced identical output — "
            "one of the methods is not using its stated algorithm"
        )


# ---------------------------------------------------------------------------
# Level 3: Integration — save_geotiff and compute_output_resolution
# ---------------------------------------------------------------------------
class TestOrthoBuilderLevel3:
    """Integration: GeoTIFF output and auto-resolution."""

    @pytest.mark.integration
    def test_save_geotiff_roundtrip(self, synthetic_ortho_setup):
        """save_geotiff() writes a file that can be read back."""
        import rasterio

        image, geo, _, _ = synthetic_ortho_setup
        pipeline = (
            OrthoBuilder()
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

                # Pixel values must survive the GeoTIFF write/read roundtrip.
                # shape-only checks miss silent precision loss or channel swaps.
                np.testing.assert_allclose(
                    data_back.astype(np.float32),
                    result.data.astype(np.float32),
                    rtol=1e-4,
                    err_msg=(
                        "GeoTIFF roundtrip: pixel values differ from in-memory "
                        "result — check dtype promotion or nodata masking in "
                        "save_geotiff()"
                    ),
                )

                # CRS must be written and match the source (EPSG:4326)
                assert ds.crs is not None, "CRS was not written to the GeoTIFF"
                assert '4326' in ds.crs.to_string(), (
                    f"Expected EPSG:4326 in output CRS, got: {ds.crs}"
                )

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
    """Tests for compute_output_resolution().

    compute_output_resolution dispatches by metadata content:
      1. isinstance(SICDMetadata)             -> _resolution_from_sicd()
      2. dict with 'range_pixel_spacing'      -> _resolution_from_biomass()
      3. dict with 'transform' + 'resolution' -> _resolution_from_geotiff()
      4. anything else                        -> raises ValueError
    Tests below exercise paths 2, 3, and 4 without requiring real sensor data.
    """

    def test_biomass_dict_returns_degrees(self, synthetic_ortho_setup):
        """BIOMASS-style dict (range/azimuth spacing in metres) returns (lat_deg, lon_deg).

        The fixture geolocation is centred near lat=34°N.  For 10 m pixel
        spacing the expected latitude resolution is 10 / 111_320 ≈ 9e-5°.
        The test accepts a generous ±5× band to allow for implementation
        differences in the metres-to-degrees conversion, while still
        catching pathological values (e.g., metres returned directly as
        'degrees', or a divide-by-zero returning inf/nan).
        """
        _, geo, _, _ = synthetic_ortho_setup
        result = compute_output_resolution(
            metadata={
                'range_pixel_spacing': 10.0,
                'azimuth_pixel_spacing': 10.0,
            },
            geolocation=geo,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result)
        assert all(np.isfinite(v) and v > 0 for v in result)
        # 10 m at lat≈34° ≈ 9e-5°; accept 5e-5° to 5e-4° as sanity band
        assert 5e-5 < result[0] < 5e-4, (
            f"Latitude pixel size {result[0]:.2e}° is outside the expected "
            "range for a 10 m spacing — check the metres-to-degrees formula."
        )

    def test_geotiff_geographic_crs_returns_resolution_unchanged(self):
        """GeoTIFF dict with geographic CRS (EPSG:4326) returns pixel size as-is.

        When the CRS is geographic the resolution is already in degrees and
        must be passed through without any metres-to-degrees conversion.
        """
        resolution_deg = (0.00027, 0.00027)
        result = compute_output_resolution(
            metadata={
                'transform': True,          # non-None triggers GeoTIFF dispatch
                'resolution': resolution_deg,
                'crs': 'EPSG:4326',
            },
        )
        assert result[0] == pytest.approx(resolution_deg[0], rel=1e-5), (
            f"Geographic GeoTIFF lat resolution {result[0]:.6f}° != "
            f"input {resolution_deg[0]:.6f}° — resolution is being converted "
            "despite the CRS already being geographic."
        )
        assert result[1] == pytest.approx(resolution_deg[1], rel=1e-5)

    def test_scale_factor_doubles_pixel_sizes(self, synthetic_ortho_setup):
        """scale_factor=2.0 must exactly double both output pixel sizes."""
        _, geo, _, _ = synthetic_ortho_setup
        meta = {'range_pixel_spacing': 10.0, 'azimuth_pixel_spacing': 10.0}
        base = compute_output_resolution(metadata=meta, geolocation=geo)
        scaled = compute_output_resolution(metadata=meta, geolocation=geo, scale_factor=2.0)
        assert scaled[0] == pytest.approx(base[0] * 2.0, rel=1e-6), (
            "scale_factor=2.0 did not double the latitude pixel size"
        )
        assert scaled[1] == pytest.approx(base[1] * 2.0, rel=1e-6), (
            "scale_factor=2.0 did not double the longitude pixel size"
        )

    def test_unrecognized_metadata_raises_value_error(self):
        """Metadata with no recognised dispatch key raises ValueError.

        A dict with arbitrary keys (not 'range_pixel_spacing', 'transform',
        or an SICDMetadata instance) must raise ValueError, not silently
        return a wrong value or skip.
        """
        with pytest.raises(ValueError):
            compute_output_resolution(
                metadata={'pixel_size_x': 10.0, 'pixel_size_y': 10.0},
            )


# ---------------------------------------------------------------------------
# ENUGrid integration
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _HAS_ENU, reason="ENUGrid not available")
class TestENUGridIntegration:
    """Validate ENUGrid as a drop-in replacement for OutputGrid in
    Orthorectifier and OrthoBuilder.

    All tests use the same synthetic AffineGeolocation fixture as the
    existing ortho tests (no real imagery required).
    """

    def test_orthorectifier_accepts_enu_grid(self, synthetic_ortho_setup):
        """Orthorectifier constructs and runs with an ENUGrid output specification.

        ENUGrid is documented as a drop-in for OutputGrid.  If construction
        raises or apply() fails, the polymorphic contract is broken.
        """
        image, geo, _, _ = synthetic_ortho_setup
        enu_grid = ENUGrid.from_geolocation(
            geo, pixel_size_m=10.0, ref_alt=0.0,
        )
        ortho = Orthorectifier(
            geolocation=geo, output_grid=enu_grid, interpolation='bilinear',
        )
        ortho.compute_mapping()
        result = ortho.apply(image)

        assert isinstance(result, np.ndarray), (
            "Orthorectifier.apply() with ENUGrid must return ndarray"
        )
        assert result.shape == (enu_grid.rows, enu_grid.cols), (
            f"Output shape {result.shape} must match ENUGrid dimensions "
            f"({enu_grid.rows}, {enu_grid.cols})"
        )

    def test_orthorectifier_enu_output_has_valid_pixels(self, synthetic_ortho_setup):
        """ENUGrid orthorectification produces at least some non-nodata pixels.

        An all-nodata output means the mapping failed: the grid bounds
        do not overlap the source image footprint.
        """
        image, geo, _, _ = synthetic_ortho_setup
        enu_grid = ENUGrid.from_geolocation(
            geo, pixel_size_m=10.0, ref_alt=0.0,
        )
        ortho = Orthorectifier(
            geolocation=geo, output_grid=enu_grid, interpolation='bilinear',
        )
        result = ortho.apply(image)

        valid_count = np.sum(result != 0.0)
        assert valid_count > 0, (
            "ENUGrid orthorectification returned all-nodata output — "
            "grid bounds do not overlap the source image footprint"
        )

    def test_ortho_pipeline_with_enu_builder(self, synthetic_ortho_setup):
        """OrthoBuilder.with_enu_grid() executes and returns OrthoResult."""
        image, geo, _, _ = synthetic_ortho_setup
        result = (
            OrthoBuilder()
            .with_source_array(image)
            .with_geolocation(geo)
            .with_enu_grid(pixel_size_m=10.0)
            .run()
        )
        assert isinstance(result, OrthoResult), (
            "OrthoBuilder.with_enu_grid().run() must return OrthoResult"
        )

    def test_enu_result_is_enu_flag(self, synthetic_ortho_setup):
        """OrthoResult.is_enu is True when the pipeline used with_enu_grid()."""
        image, geo, _, _ = synthetic_ortho_setup
        result = (
            OrthoBuilder()
            .with_source_array(image)
            .with_geolocation(geo)
            .with_enu_grid(pixel_size_m=10.0)
            .run()
        )
        assert result.is_enu is True, (
            "OrthoResult.is_enu should be True when output grid is ENUGrid"
        )

    def test_standard_pipeline_is_not_enu(self, synthetic_ortho_setup):
        """OrthoResult.is_enu is False for a standard geographic OutputGrid pipeline."""
        image, geo, _, _ = synthetic_ortho_setup
        result = (
            OrthoBuilder()
            .with_source_array(image)
            .with_geolocation(geo)
            .with_resolution(0.00054, 0.00054)
            .run()
        )
        assert result.is_enu is False, (
            "OrthoResult.is_enu should be False for standard geographic OutputGrid"
        )

    def test_enu_result_pixel_size_meters(self, synthetic_ortho_setup):
        """OrthoResult.pixel_size_meters returns the configured pixel size."""
        image, geo, _, _ = synthetic_ortho_setup
        result = (
            OrthoBuilder()
            .with_source_array(image)
            .with_geolocation(geo)
            .with_enu_grid(pixel_size_m=15.0)
            .run()
        )
        sizes = result.pixel_size_meters
        assert sizes is not None, "pixel_size_meters must not be None for ENU result"
        assert len(sizes) == 2, "pixel_size_meters must be a 2-tuple (east, north)"
        assert sizes[0] == pytest.approx(15.0, rel=1e-6), (
            f"pixel_size_meters east = {sizes[0]:.4f} m, expected 15.0 m"
        )
        assert sizes[1] == pytest.approx(15.0, rel=1e-6), (
            f"pixel_size_meters north = {sizes[1]:.4f} m, expected 15.0 m"
        )

    def test_enu_result_bounds_meters_are_finite(self, synthetic_ortho_setup):
        """OrthoResult.bounds_meters returns finite (min_east, min_north, max_east, max_north)."""
        image, geo, _, _ = synthetic_ortho_setup
        result = (
            OrthoBuilder()
            .with_source_array(image)
            .with_geolocation(geo)
            .with_enu_grid(pixel_size_m=10.0)
            .run()
        )
        bounds = result.bounds_meters
        assert bounds is not None, "bounds_meters must not be None for ENU result"
        assert len(bounds) == 4, "bounds_meters must be a 4-tuple"
        assert all(
            isinstance(v, float) and np.isfinite(v) for v in bounds
        ), f"bounds_meters contains non-finite values: {bounds}"
        min_east, min_north, max_east, max_north = bounds
        assert max_east > min_east, "bounds_meters: max_east must exceed min_east"
        assert max_north > min_north, "bounds_meters: max_north must exceed min_north"

    def test_enu_geolocation_metadata_crs_key(self, synthetic_ortho_setup):
        """geolocation_metadata['crs'] is 'ENU' for an ENU output."""
        image, geo, _, _ = synthetic_ortho_setup
        result = (
            OrthoBuilder()
            .with_source_array(image)
            .with_geolocation(geo)
            .with_enu_grid(pixel_size_m=10.0)
            .run()
        )
        meta = result.geolocation_metadata
        assert 'crs' in meta, "geolocation_metadata must contain 'crs' key"
        assert meta['crs'] == 'ENU', (
            f"geolocation_metadata['crs'] = {meta['crs']!r}, expected 'ENU'"
        )

    def test_enu_geolocation_metadata_transform_key(self, synthetic_ortho_setup):
        """geolocation_metadata['transform'] is a 6-tuple for ENU output."""
        image, geo, _, _ = synthetic_ortho_setup
        result = (
            OrthoBuilder()
            .with_source_array(image)
            .with_geolocation(geo)
            .with_enu_grid(pixel_size_m=10.0)
            .run()
        )
        meta = result.geolocation_metadata
        assert 'transform' in meta, "geolocation_metadata must contain 'transform' key"
        t = meta['transform']
        assert len(t) == 6, (
            f"geolocation_metadata['transform'] must be a 6-tuple, got length {len(t)}"
        )
        assert all(np.isfinite(v) for v in t), (
            f"geolocation_metadata['transform'] contains non-finite values: {t}"
        )

    def test_with_enu_grid_then_with_output_grid_uses_explicit_grid(self, synthetic_ortho_setup):
        """with_output_grid() called after with_enu_grid() overrides ENU params.

        The pipeline should honour the last explicit grid specification.
        """
        image, geo, _, _ = synthetic_ortho_setup
        explicit_grid = OutputGrid.from_geolocation(
            geo, pixel_size_lat=0.00054, pixel_size_lon=0.00054,
        )
        result = (
            OrthoBuilder()
            .with_source_array(image)
            .with_geolocation(geo)
            .with_enu_grid(pixel_size_m=10.0)    # set ENU
            .with_output_grid(explicit_grid)       # override with geographic grid
            .run()
        )
        # with_output_grid takes priority → result must not be ENU
        assert result.is_enu is False, (
            "with_output_grid() should override with_enu_grid() when called last; "
            "expected is_enu=False"
        )

    @pytest.mark.integration
    def test_parallel_compute_mapping_produces_valid_pixels(self, synthetic_ortho_setup):
        """Parallel compute_mapping (>1M pixel grids) covers valid image region.

        Creates an ENUGrid large enough to exceed the 1M pixel threshold
        (_PARALLEL_THRESHOLD = 1_000_000) so that _compute_mapping_parallel
        is exercised.  Verifies valid pixels > 0 and output shape is correct.
        """
        image, geo, _, _ = synthetic_ortho_setup
        # Force a large-enough ENUGrid: 1024 x 1024 = 1M pixels exactly.
        # Use a coarser pixel size to keep this small in extent.
        enu_grid = ENUGrid.from_geolocation(
            geo, pixel_size_m=2.5, ref_alt=0.0,
        )
        # Only run if the grid is large enough to trigger parallel path
        if enu_grid.rows * enu_grid.cols < 1_000_000:
            pytest.skip(
                f"Generated ENUGrid ({enu_grid.rows}×{enu_grid.cols}) is too small "
                "to trigger parallel compute_mapping; increase pixel_size_m or extent"
            )

        ortho = Orthorectifier(
            geolocation=geo, output_grid=enu_grid, interpolation='nearest',
        )
        src_rows, src_cols, valid_mask = ortho.compute_mapping()

        assert src_rows.shape == (enu_grid.rows, enu_grid.cols), (
            f"Parallel compute_mapping source_rows shape {src_rows.shape} != "
            f"expected ({enu_grid.rows}, {enu_grid.cols})"
        )
        assert valid_mask.sum() > 0, (
            "Parallel compute_mapping produced zero valid pixels"
        )
