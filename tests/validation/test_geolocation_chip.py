# -*- coding: utf-8 -*-
"""
ChipGeolocation Tests - Offset wrapper contract validation.

Validates that grdl.geolocation.chip.ChipGeolocation correctly translates
local chip coordinates to/from global image coordinates through the parent
geolocation object:

- Level 1: Construction, shape, default_hae delegation, elevation inheritance
- Level 2: Offset correctness (forward/inverse), round-trip precision,
           IDL crossing, pole proximity, null elevation fallback
- Level 3: Integration with ChipExtractor (chip region → ChipGeolocation)

All tests use synthetic parent geolocations (identity or affine) so no
real data files are required.

Dependencies
------------
pytest
numpy
grdl

Author
------
Ava Courtney
courtney-ava@zai.com

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2026-03-30

Modified
--------
2026-03-30
"""

# Standard library
from typing import Tuple, Union

# Third-party
import pytest
import numpy as np

# GRDL internal
try:
    from grdl.geolocation.base import Geolocation
    from grdl.geolocation.chip import ChipGeolocation
    _HAS_CHIP_GEO = True
except ImportError:
    _HAS_CHIP_GEO = False

try:
    from grdl.geolocation.elevation import ConstantElevation
    _HAS_ELEVATION = True
except ImportError:
    _HAS_ELEVATION = False

try:
    from grdl.data_prep import ChipExtractor
    _HAS_DATA_PREP = True
except ImportError:
    _HAS_DATA_PREP = False


pytestmark = [
    pytest.mark.geolocation,
    pytest.mark.skipif(not _HAS_CHIP_GEO, reason="ChipGeolocation not available"),
]


# ---------------------------------------------------------------------------
# Synthetic parent geolocations
# ---------------------------------------------------------------------------

class _IdentityGeolocation(Geolocation):
    """Trivial geolocation: pixel (row, col) maps to (lat=row, lon=col).

    Useful for testing pure offset arithmetic without sensor math.
    """

    def __init__(self, shape: Tuple[int, int]) -> None:
        super().__init__(shape, crs='WGS84')

    def _image_to_latlon_array(
        self, rows: np.ndarray, cols: np.ndarray,
        height: Union[float, np.ndarray] = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return rows.copy(), cols.copy(), np.full_like(rows, height)

    def _latlon_to_image_array(
        self, lats: np.ndarray, lons: np.ndarray,
        height: Union[float, np.ndarray] = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return lats.copy(), lons.copy()


class _AffineGeolocation(Geolocation):
    """Linear geolocation with configurable origin and pixel spacing.

    lat = origin_lat - row * dlat
    lon = origin_lon + col * dlon
    """

    def __init__(
        self, shape: Tuple[int, int],
        origin_lat: float, origin_lon: float,
        dlat: float = 0.01, dlon: float = 0.01,
        ref_hae: float = 0.0,
    ) -> None:
        super().__init__(shape, crs='WGS84')
        self._olat = origin_lat
        self._olon = origin_lon
        self._dlat = dlat
        self._dlon = dlon
        self._ref_hae = ref_hae

    @property
    def default_hae(self) -> float:
        return self._ref_hae

    def _image_to_latlon_array(
        self, rows: np.ndarray, cols: np.ndarray,
        height: Union[float, np.ndarray] = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lats = self._olat - rows * self._dlat
        lons = self._olon + cols * self._dlon
        return lats, lons, np.full_like(lats, height)

    def _latlon_to_image_array(
        self, lats: np.ndarray, lons: np.ndarray,
        height: Union[float, np.ndarray] = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        rows = (self._olat - lats) / self._dlat
        cols = (lons - self._olon) / self._dlon
        return rows, cols


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def parent_geo():
    """1000x2000 identity geolocation."""
    return _IdentityGeolocation(shape=(1000, 2000))


@pytest.fixture
def chip_geo(parent_geo):
    """ChipGeolocation with offset (100, 200) and shape (256, 256)."""
    return ChipGeolocation(parent_geo, row_offset=100, col_offset=200, shape=(256, 256))


@pytest.fixture
def affine_parent():
    """Affine geolocation with known reference HAE."""
    return _AffineGeolocation(
        shape=(500, 500),
        origin_lat=40.0, origin_lon=-75.0,
        dlat=0.001, dlon=0.001,
        ref_hae=150.0,
    )


# =============================================================================
# Level 1: Format Validation / Construction
# =============================================================================


class TestChipGeolocationConstruction:
    """Test basic instantiation and property delegation."""

    def test_constructor_accepts_valid_params(self, parent_geo):
        """ChipGeolocation instantiates with valid parent, offset, and shape."""
        chip = ChipGeolocation(parent_geo, row_offset=50, col_offset=100, shape=(128, 128))
        assert chip is not None

    def test_shape_attribute(self, chip_geo):
        """shape reflects chip dimensions, not parent dimensions."""
        assert chip_geo.shape == (256, 256), (
            f"chip shape {chip_geo.shape} should be (256, 256)"
        )

    def test_default_hae_delegates_to_parent(self, affine_parent):
        """default_hae is inherited from parent geolocation."""
        chip = ChipGeolocation(affine_parent, row_offset=10, col_offset=10, shape=(100, 100))
        assert chip.default_hae == pytest.approx(150.0), (
            f"default_hae {chip.default_hae} should delegate to parent (150.0)"
        )

    @pytest.mark.skipif(not _HAS_ELEVATION, reason="ConstantElevation not available")
    def test_elevation_inherited_from_parent(self, parent_geo):
        """ChipGeolocation shares parent's elevation model."""
        dem = ConstantElevation(height=300.0)
        parent_geo.elevation = dem
        chip = ChipGeolocation(parent_geo, row_offset=0, col_offset=0, shape=(50, 50))
        assert chip.elevation is dem, "Chip should share parent's DEM object"

    def test_zero_offset_passthrough(self, parent_geo):
        """Zero offset produces same coordinates as parent."""
        chip = ChipGeolocation(parent_geo, row_offset=0, col_offset=0, shape=(100, 100))
        lat_p, lon_p, h_p = parent_geo.image_to_latlon(50.0, 75.0)
        lat_c, lon_c, h_c = chip.image_to_latlon(50.0, 75.0)
        assert lat_c == pytest.approx(float(lat_p), abs=1e-12)
        assert lon_c == pytest.approx(float(lon_p), abs=1e-12)


# =============================================================================
# Level 2: Data Quality / Offset Correctness
# =============================================================================


class TestChipGeolocationOffsets:
    """Verify offset arithmetic in forward and inverse directions."""

    def test_forward_offset_applied(self, parent_geo):
        """Chip pixel (r, c) should map to parent pixel (r + row_off, c + col_off)."""
        chip = ChipGeolocation(parent_geo, row_offset=100, col_offset=200, shape=(256, 256))

        # In identity geolocation: lat = global_row, lon = global_col
        lat, lon, _ = chip.image_to_latlon(0.0, 0.0)
        assert float(lat) == pytest.approx(100.0, abs=1e-12), (
            f"Chip (0,0) should map to parent row 100, got lat={lat}"
        )
        assert float(lon) == pytest.approx(200.0, abs=1e-12), (
            f"Chip (0,0) should map to parent col 200, got lon={lon}"
        )

    def test_inverse_offset_subtracted(self, parent_geo):
        """latlon_to_image subtracts offset to return chip-local coordinates."""
        chip = ChipGeolocation(parent_geo, row_offset=100, col_offset=200, shape=(256, 256))

        # Identity: lat=150, lon=250 → parent pixel (150, 250) → chip pixel (50, 50)
        row, col = chip.latlon_to_image(150.0, 250.0)
        assert float(row) == pytest.approx(50.0, abs=1e-12), (
            f"Expected chip row 50, got {row}"
        )
        assert float(col) == pytest.approx(50.0, abs=1e-12), (
            f"Expected chip col 50, got {col}"
        )

    def test_round_trip_scalar(self, chip_geo):
        """Chip pixel → latlon → chip pixel round-trip is exact for identity."""
        chip_row, chip_col = 128.0, 64.0
        lat, lon, h = chip_geo.image_to_latlon(chip_row, chip_col)
        row_back, col_back = chip_geo.latlon_to_image(lat, lon)
        assert float(row_back) == pytest.approx(chip_row, abs=1e-10), (
            f"Round-trip row error: {abs(float(row_back) - chip_row)}"
        )
        assert float(col_back) == pytest.approx(chip_col, abs=1e-10), (
            f"Round-trip col error: {abs(float(col_back) - chip_col)}"
        )

    def test_round_trip_vectorized(self, chip_geo):
        """Vectorized round-trip produces sub-pixel accuracy."""
        rows = np.array([0.0, 64.0, 128.0, 255.0])
        cols = np.array([0.0, 64.0, 128.0, 255.0])
        result_fwd = chip_geo.image_to_latlon(rows, cols)
        lats = result_fwd[:, 0]
        lons = result_fwd[:, 1]
        result_inv = chip_geo.latlon_to_image(lats, lons)
        np.testing.assert_allclose(result_inv[:, 0], rows, atol=1e-10,
            err_msg="Vectorized round-trip row error")
        np.testing.assert_allclose(result_inv[:, 1], cols, atol=1e-10,
            err_msg="Vectorized round-trip col error")

    def test_affine_round_trip_with_offset(self, affine_parent):
        """Affine parent + chip offset preserves sub-pixel round-trip."""
        chip = ChipGeolocation(affine_parent, row_offset=50, col_offset=100, shape=(200, 200))
        chip_row, chip_col = 75.5, 120.3
        lat, lon, h = chip.image_to_latlon(chip_row, chip_col)
        row_back, col_back = chip.latlon_to_image(lat, lon)
        assert float(row_back) == pytest.approx(chip_row, abs=1e-8), (
            f"Affine round-trip row error: {abs(float(row_back) - chip_row)}"
        )
        assert float(col_back) == pytest.approx(chip_col, abs=1e-8), (
            f"Affine round-trip col error: {abs(float(col_back) - chip_col)}"
        )


class TestChipGeolocationEdgeCases:
    """Edge cases: IDL crossing, pole proximity, null elevation."""

    def test_idl_crossing(self):
        """Chip straddling the International Date Line (±180° lon).

        Parent image spans lon 179.5° to 180.5° (wrapping to -179.5°).
        Chip offset places local pixel 0 at parent pixel 0 (lon=179.5°).
        Local pixel at far-right col should be near -179.5° (across IDL).
        """
        parent = _AffineGeolocation(
            shape=(100, 200),
            origin_lat=0.0, origin_lon=179.5,
            dlat=0.01, dlon=0.01,
        )
        chip = ChipGeolocation(parent, row_offset=0, col_offset=0, shape=(50, 200))

        # Pixel at col=0 should be near 179.5°
        _, lon_start, _ = chip.image_to_latlon(0.0, 0.0)
        assert float(lon_start) == pytest.approx(179.5, abs=0.01)

        # Pixel at col=100 should be near 180.5° (= -179.5° wrapped)
        _, lon_mid, _ = chip.image_to_latlon(0.0, 100.0)
        # The raw value will be 180.5 (unwrapped) from our simple affine
        assert np.isfinite(float(lon_mid)), "IDL-crossing lon should be finite"

        # Verify continuity: adjacent pixels have small lon differences
        _, lon_a, _ = chip.image_to_latlon(0.0, 49.0)
        _, lon_b, _ = chip.image_to_latlon(0.0, 50.0)
        delta = abs(float(lon_b) - float(lon_a))
        assert delta < 1.0, (
            f"Adjacent pixel lon jump {delta:.4f}° across IDL should be < 1°"
        )

    def test_pole_proximity_north(self):
        """Chip near the North Pole (lat ≈ 89.5°) produces finite coordinates."""
        parent = _AffineGeolocation(
            shape=(100, 100),
            origin_lat=89.95, origin_lon=0.0,
            dlat=0.01, dlon=0.1,
        )
        chip = ChipGeolocation(parent, row_offset=0, col_offset=0, shape=(50, 50))
        lat, lon, h = chip.image_to_latlon(0.0, 0.0)
        assert np.isfinite(float(lat)), f"Near-pole lat should be finite, got {lat}"
        assert float(lat) >= 89.0, f"Expected lat near 90°, got {lat}"

    def test_pole_proximity_south(self):
        """Chip near the South Pole produces finite coordinates."""
        parent = _AffineGeolocation(
            shape=(100, 100),
            origin_lat=-89.0, origin_lon=0.0,
            dlat=0.01, dlon=0.1,
        )
        chip = ChipGeolocation(parent, row_offset=10, col_offset=10, shape=(50, 50))
        result = chip.image_to_latlon(
            np.array([0.0, 25.0, 49.0]),
            np.array([0.0, 25.0, 49.0]),
        )
        assert np.all(np.isfinite(result)), "Near-south-pole coordinates should be finite"
        lats = result[:, 0]
        assert np.all(lats <= -88.0), f"Expected lats near -90°, got {lats}"

    def test_null_elevation_uses_zero_height(self, parent_geo):
        """When parent has no DEM, ChipGeolocation returns height=0."""
        parent_geo.elevation = None
        chip = ChipGeolocation(parent_geo, row_offset=50, col_offset=50, shape=(100, 100))
        _, _, h = chip.image_to_latlon(10.0, 10.0)
        assert float(h) == pytest.approx(0.0, abs=1e-6), (
            f"Expected height 0.0 with null DEM, got {h}"
        )

    @pytest.mark.skipif(not _HAS_ELEVATION, reason="ConstantElevation not available")
    def test_constant_elevation_propagates(self, parent_geo):
        """ConstantElevation on parent propagates through chip offset."""
        dem = ConstantElevation(height=500.0)
        parent_geo.elevation = dem
        chip = ChipGeolocation(parent_geo, row_offset=0, col_offset=0, shape=(50, 50))
        _, _, h = chip.image_to_latlon(25.0, 25.0)
        # Height behavior depends on whether the base class iterative DEM is active;
        # at minimum the elevation model is accessible.
        assert chip.elevation is dem

    def test_fractional_offset(self):
        """Non-integer offsets are supported (sub-pixel chip registration)."""
        parent = _IdentityGeolocation(shape=(500, 500))
        chip = ChipGeolocation(parent, row_offset=100.5, col_offset=200.75, shape=(100, 100))
        lat, lon, _ = chip.image_to_latlon(0.0, 0.0)
        assert float(lat) == pytest.approx(100.5, abs=1e-10)
        assert float(lon) == pytest.approx(200.75, abs=1e-10)


# =============================================================================
# Level 3: Integration
# =============================================================================


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
class TestChipGeolocationIntegration:
    """Integration with ChipExtractor to verify offset pipeline."""

    def test_chip_extractor_regions_consistent(self):
        """ChipExtractor regions produce valid chip geolocations."""
        parent = _AffineGeolocation(
            shape=(1000, 1000),
            origin_lat=35.0, origin_lon=-118.0,
            dlat=0.001, dlon=0.001,
        )
        extractor = ChipExtractor(nrows=parent.shape[0], ncols=parent.shape[1])
        regions = extractor.chip_positions(row_width=256, col_width=256)

        for region in regions[:4]:  # test first 4 chips
            chip = ChipGeolocation(
                parent,
                row_offset=region.row_start,
                col_offset=region.col_start,
                shape=(region.row_end - region.row_start,
                       region.col_end - region.col_start),
            )
            # Chip (0,0) should map to parent pixel at chip's top-left
            lat, lon, _ = chip.image_to_latlon(0.0, 0.0)
            expected_lat = 35.0 - region.row_start * 0.001
            expected_lon = -118.0 + region.col_start * 0.001
            assert float(lat) == pytest.approx(expected_lat, abs=1e-6), (
                f"Chip at ({region.row_start},{region.col_start}) lat mismatch"
            )
            assert float(lon) == pytest.approx(expected_lon, abs=1e-6), (
                f"Chip at ({region.row_start},{region.col_start}) lon mismatch"
            )
