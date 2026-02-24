# -*- coding: utf-8 -*-
"""
Data Preparation Validation - ChipExtractor, Tiler, and Normalizer.

Tests chip region computation, tiled grid layout, and intensity
normalization using purely synthetic data. No IO dependencies.

- Level 1: Construction, property access, type validation
- Level 2: Numerical correctness, coverage invariants, edge cases

Dependencies
------------
pytest
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
2026-02-24
"""

import pytest
import numpy as np

try:
    from grdl.data_prep import ChipExtractor, Tiler, Normalizer
    from grdl.data_prep.base import ChipRegion
    _HAS_DATA_PREP = True
except ImportError:
    _HAS_DATA_PREP = False

pytestmark = [
    pytest.mark.data_prep,
    pytest.mark.skipif(not _HAS_DATA_PREP,
                       reason="grdl data_prep not available"),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def normalizer_data():
    """1D float64 array of 1000 elements for normalizer tests."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(1000).astype(np.float64)


@pytest.fixture(scope="module")
def normalizer_2d():
    """64x64 float64 array for 2D normalizer tests."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((64, 64)).astype(np.float64)


# ===================================================================
# ChipExtractor Level 1
# ===================================================================

class TestChipExtractorLevel1:
    """Validate ChipExtractor construction and property access."""

    def test_chip_extractor_init(self):
        ext = ChipExtractor(nrows=100, ncols=200)
        assert ext is not None

    def test_chip_extractor_shape_property(self):
        ext = ChipExtractor(nrows=100, ncols=200)
        assert ext.shape == (100, 200)

    def test_chip_extractor_nrows_ncols(self):
        ext = ChipExtractor(nrows=100, ncols=200)
        assert ext.nrows == 100
        assert ext.ncols == 200

    def test_chip_extractor_invalid_nrows_type(self):
        with pytest.raises(TypeError):
            ChipExtractor(nrows=10.5, ncols=10)

    def test_chip_extractor_invalid_nrows_value(self):
        with pytest.raises(ValueError):
            ChipExtractor(nrows=-1, ncols=10)

    def test_chip_at_point_scalar_return_type(self):
        ext = ChipExtractor(nrows=100, ncols=100)
        result = ext.chip_at_point(50, 50, row_width=20, col_width=20)
        assert isinstance(result, ChipRegion)


# ===================================================================
# ChipExtractor Level 2
# ===================================================================

class TestChipExtractorLevel2:
    """Validate ChipExtractor numerical correctness and edge cases."""

    def test_chip_at_point_center(self):
        ext = ChipExtractor(nrows=100, ncols=100)
        region = ext.chip_at_point(50, 50, row_width=20, col_width=20)
        assert region.row_start == 40
        assert region.col_start == 40
        assert region.row_end == 60
        assert region.col_end == 60

    def test_chip_at_point_edge_snap_top(self):
        """Near top edge: chip snaps inward to maintain full dimensions."""
        ext = ChipExtractor(nrows=100, ncols=100)
        region = ext.chip_at_point(5, 50, row_width=20, col_width=20)
        assert region.row_start == 0
        assert region.row_end == 20
        assert region.row_end - region.row_start == 20

    def test_chip_at_point_edge_snap_bottom(self):
        """Near bottom edge: chip snaps inward to maintain full dimensions."""
        ext = ChipExtractor(nrows=100, ncols=100)
        region = ext.chip_at_point(95, 50, row_width=20, col_width=20)
        assert region.row_start == 80
        assert region.row_end == 100
        assert region.row_end - region.row_start == 20

    def test_chip_at_point_batch(self):
        ext = ChipExtractor(nrows=100, ncols=100)
        result = ext.chip_at_point([50, 5], [50, 5],
                                   row_width=20, col_width=20)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(r, ChipRegion) for r in result)

    def test_chip_at_point_out_of_bounds(self):
        ext = ChipExtractor(nrows=100, ncols=100)
        with pytest.raises(ValueError):
            ext.chip_at_point(200, 50, row_width=20, col_width=20)

    def test_chip_at_point_invalid_width_type(self):
        ext = ChipExtractor(nrows=100, ncols=100)
        with pytest.raises(TypeError):
            ext.chip_at_point(50, 50, row_width=10.0, col_width=10)

    def test_chip_positions_count(self):
        ext = ChipExtractor(nrows=100, ncols=100)
        regions = ext.chip_positions(row_width=50, col_width=50)
        assert len(regions) == 4

    def test_chip_positions_full_coverage(self):
        """All chip regions must cover every pixel of the image."""
        ext = ChipExtractor(nrows=100, ncols=100)
        regions = ext.chip_positions(row_width=40, col_width=40)
        covered = np.zeros((100, 100), dtype=bool)
        for reg in regions:
            covered[reg.row_start:reg.row_end,
                    reg.col_start:reg.col_end] = True
        assert np.all(covered), (
            f"Coverage gap: {np.sum(~covered)} pixels uncovered"
        )


# ===================================================================
# Tiler Level 1
# ===================================================================

class TestTilerLevel1:
    """Validate Tiler construction and property access."""

    def test_tiler_init(self):
        tiler = Tiler(nrows=100, ncols=100, tile_size=32)
        assert tiler is not None

    def test_tiler_properties(self):
        tiler = Tiler(nrows=100, ncols=100, tile_size=32)
        assert tiler.tile_size == (32, 32)
        assert tiler.stride == (32, 32)

    def test_tiler_default_stride_equals_tile_size(self):
        tiler = Tiler(nrows=100, ncols=100, tile_size=32)
        assert tiler.stride == tiler.tile_size

    def test_tiler_stride_exceeds_tile_raises(self):
        with pytest.raises(ValueError):
            Tiler(nrows=100, ncols=100, tile_size=32, stride=64)

    def test_tiler_tuple_tile_size(self):
        tiler = Tiler(nrows=100, ncols=100, tile_size=(32, 64))
        assert tiler.tile_size == (32, 64)


# ===================================================================
# Tiler Level 2
# ===================================================================

class TestTilerLevel2:
    """Validate Tiler grid layout and coverage."""

    def test_tiler_no_overlap_count(self):
        tiler = Tiler(nrows=100, ncols=100, tile_size=50)
        regions = tiler.tile_positions()
        assert len(regions) == 4

    def test_tiler_overlap_count(self):
        tiler = Tiler(nrows=100, ncols=100, tile_size=64, stride=32)
        regions = tiler.tile_positions()
        # row: 1 + ceil((100-64)/32) = 1 + 2 = 3
        # col: same = 3. Total = 9.
        assert len(regions) == 9

    def test_tiler_full_coverage(self):
        """Union of tiles must cover every pixel."""
        tiler = Tiler(nrows=100, ncols=100, tile_size=32, stride=16)
        regions = tiler.tile_positions()
        covered = np.zeros((100, 100), dtype=bool)
        for reg in regions:
            covered[reg.row_start:reg.row_end,
                    reg.col_start:reg.col_end] = True
        assert np.all(covered)

    def test_tiler_tile_dimensions_consistent(self):
        """All tiles should have the requested tile_size dimensions."""
        tiler = Tiler(nrows=200, ncols=200, tile_size=64, stride=32)
        regions = tiler.tile_positions()
        for reg in regions:
            assert reg.row_end - reg.row_start == 64
            assert reg.col_end - reg.col_start == 64

    def test_tiler_image_smaller_than_tile(self):
        """Image smaller than tile → single tile covering full image."""
        tiler = Tiler(nrows=30, ncols=30, tile_size=64)
        regions = tiler.tile_positions()
        assert len(regions) == 1
        assert regions[0] == ChipRegion(0, 0, 30, 30)

    def test_tiler_invalid_tile_size_zero(self):
        with pytest.raises(ValueError):
            Tiler(nrows=100, ncols=100, tile_size=0)

    def test_tiler_repr(self):
        tiler = Tiler(nrows=100, ncols=100, tile_size=32)
        r = repr(tiler)
        assert 'Tiler' in r
        assert '100' in r


# ===================================================================
# Normalizer Level 1
# ===================================================================

class TestNormalizerLevel1:
    """Validate Normalizer construction and error handling."""

    def test_normalizer_invalid_method(self):
        with pytest.raises(ValueError):
            Normalizer(method='invalid')

    def test_normalizer_invalid_percentile_low_ge_high(self):
        with pytest.raises(ValueError):
            Normalizer(percentile_low=60.0, percentile_high=40.0)

    def test_normalizer_not_ndarray_raises(self):
        norm = Normalizer()
        with pytest.raises(TypeError):
            norm.normalize([1, 2, 3])

    def test_normalizer_transform_before_fit_raises(self, normalizer_data):
        norm = Normalizer()
        with pytest.raises(RuntimeError):
            norm.transform(normalizer_data)

    def test_normalizer_fit_returns_self(self, normalizer_data):
        norm = Normalizer()
        result = norm.fit(normalizer_data)
        assert result is norm

    def test_normalizer_is_fitted_property(self, normalizer_data):
        norm = Normalizer()
        assert not norm.is_fitted
        norm.fit(normalizer_data)
        assert norm.is_fitted


# ===================================================================
# Normalizer Level 2
# ===================================================================

class TestNormalizerLevel2:
    """Validate Normalizer numerical correctness across all methods."""

    def test_normalize_minmax_range(self, normalizer_data):
        result = Normalizer(method='minmax').normalize(normalizer_data)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_normalize_minmax_golden(self, normalizer_data):
        result = Normalizer(method='minmax').normalize(normalizer_data)
        x = normalizer_data.astype(np.float64)
        expected = (x - x.min()) / (x.max() - x.min())
        np.testing.assert_allclose(result, expected)

    def test_normalize_zscore_golden(self, normalizer_data):
        result = Normalizer(method='zscore').normalize(normalizer_data)
        x = normalizer_data.astype(np.float64)
        expected = (x - x.mean()) / x.std()
        np.testing.assert_allclose(result, expected)

    def test_normalize_zscore_zero_mean_unit_var(self, normalizer_data):
        result = Normalizer(method='zscore').normalize(normalizer_data)
        np.testing.assert_allclose(result.mean(), 0.0, atol=1e-10)
        np.testing.assert_allclose(result.std(), 1.0, atol=1e-10)

    def test_normalize_percentile_range(self, normalizer_data):
        result = Normalizer(method='percentile').normalize(normalizer_data)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_normalize_unit_norm(self, normalizer_data):
        result = Normalizer(method='unit_norm').normalize(normalizer_data)
        np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-10)

    def test_normalize_output_dtype_float64(self, normalizer_data):
        for method in ('minmax', 'zscore', 'percentile', 'unit_norm'):
            result = Normalizer(method=method).normalize(normalizer_data)
            assert result.dtype == np.float64, (
                f"Method '{method}' should output float64, got {result.dtype}"
            )

    def test_normalize_constant_array_returns_zeros(self):
        source = np.full((16,), 5.0)
        result = Normalizer(method='minmax').normalize(source)
        np.testing.assert_allclose(result, 0.0)

    def test_fit_transform_equals_normalize(self, normalizer_data):
        norm1 = Normalizer(method='minmax')
        result1 = norm1.normalize(normalizer_data)
        norm2 = Normalizer(method='minmax')
        result2 = norm2.fit_transform(normalizer_data)
        np.testing.assert_allclose(result1, result2)

    def test_fit_then_transform_different_data(self):
        """Transform with fitted params should use training statistics."""
        train = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0,
                          60.0, 70.0, 80.0, 90.0, 100.0])
        test = np.array([5.0, 15.0, 25.0])
        norm = Normalizer(method='minmax')
        norm.fit(train)
        result = norm.transform(test)
        # Training min=0, max=100, so test values mapped to 0.05, 0.15, 0.25
        expected = np.array([0.05, 0.15, 0.25])
        np.testing.assert_allclose(result, expected, atol=1e-10)
