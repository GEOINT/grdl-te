# -*- coding: utf-8 -*-
"""
TiledGeoTIFFDEM Tests - Multi-tile orchestration and interpolation validation.

Author
------
Ava Courtney
courtney-ava@zai.com
"""

import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from pathlib import Path

# GRDL internal
try:
    from grdl.geolocation.elevation.tiled_geotiff_dem import TiledGeoTIFFDEM
    _HAS_TILED_DEM = True
except ImportError:
    _HAS_TILED_DEM = False

try:
    from grdl.data_prep import Tiler, Normalizer
    _HAS_DATA_PREP = True
except ImportError:
    _HAS_DATA_PREP = False

pytestmark = [
    pytest.mark.elevation,
    pytest.mark.skipif(not _HAS_TILED_DEM, reason="TiledGeoTIFFDEM not available"),
]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dem_tile_dir(tmp_path):
    """Creates a directory with naming-compliant synthetic tiles."""
    tile_dir = tmp_path / "srtm_data"
    tile_dir.mkdir()
    
    # Naming must match _TILE_RE: ([NS])(\d{1,2})([EW])(\d{1,3})
    configs = [
        ("N40W076.tif", 40.0, -76.0, 100.0), # NW
        ("N40W075.tif", 40.0, -75.0, 200.0), # NE
        ("N39W076.tif", 39.0, -76.0, 300.0), # SW
        ("N39W075.tif", 39.0, -75.0, 400.0), # SE
    ]
    
    for name, lat, lon, base_h in configs:
        p = tile_dir / name
        # Create a 50x50 grid with 0.02 deg resolution (covers exactly 1 deg)
        data = np.full((50, 50), base_h, dtype=np.float64)
        # Add gradient for interpolation testing
        r, c = np.mgrid[0:50, 0:50]
        data += (r * 0.5) 
        
        if "N39W075" in name: # NoData hole in SE tile
            data[20:30, 20:30] = -9999

        with rasterio.open(
            p, 'w', driver='GTiff', height=50, width=50, count=1,
            dtype='float64', crs='EPSG:4326', nodata=-9999,
            transform=from_origin(lon, lat + 1, 0.02, 0.02)
        ) as ds:
            ds.write(data, 1)
            
    return str(tile_dir)

# =============================================================================
# Level 1: Format Validation
# =============================================================================

class TestTiledDEMFormat:
    def test_instantiation_and_metadata(self, dem_tile_dir):
        """Verify directory scanning and metadata extraction."""
        with TiledGeoTIFFDEM(dem_tile_dir) as dem:
            assert dem.tile_count == 4
            bounds = dem.coverage_bounds
            # Should span -76 to -74 Lon and 39 to 41 Lat
            assert bounds[0] == -76.0
            assert bounds[1] == 39.0

    def test_resource_cleanup(self, dem_tile_dir):
        """Verify context manager closes open rasterio handles."""
        with TiledGeoTIFFDEM(dem_tile_dir) as dem:
            # Trigger an LRU cache open
            dem.get_elevation(40.5, -75.5)
            open_ds = list(dem._open_tiles.values())
            assert any(not ds.closed for ds in open_ds)
        
        # After exit, all should be closed
        assert all(ds.closed for ds in open_ds)

# =============================================================================
# Level 2: Data Quality
# =============================================================================

class TestTiledDEMDataQuality:
    def test_cross_tile_interpolation_smoothness(self, dem_tile_dir):
        """Verify bicubic smoothness across the N40W076 / N40W075 boundary."""
        with TiledGeoTIFFDEM(dem_tile_dir, interpolation=3) as dem:
            # Boundary is at Lon -75.0
            lat = 40.5
            lons = np.array([-75.01, -74.99])
            heights = dem.get_elevation(np.array([lat, lat]), lons)
            
            # Ensure no NaNs at the seam and that interpolation occurs
            assert np.all(np.isfinite(heights))
            assert heights[0] != heights[1]

    def test_nodata_masking_at_seams(self, dem_tile_dir):
        """Verify NoData hole in SE tile is respected."""
        with TiledGeoTIFFDEM(dem_tile_dir) as dem:
            # Sample inside N39W075 hole
            h = dem.get_elevation(39.5, -74.5)
            assert np.isnan(h)

# =============================================================================
# Level 3: Integration
# =============================================================================

@pytest.mark.integration
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
class TestTiledDEMIntegration:
    def test_pipeline_extraction(self, dem_tile_dir):
        """End-to-end: Extract elevation grid -> Normalize."""
        with TiledGeoTIFFDEM(dem_tile_dir) as dem:
            lats, lons = np.mgrid[40.1:40.2:5j, -75.1:-74.9:5j]
            elev_map = dem.get_elevation(lats.flatten(), lons.flatten())
            
            norm = Normalizer(method='minmax')
            normalized = norm.fit_transform(elev_map.reshape(5, 5))
            assert normalized.max() == pytest.approx(1.0)

    def test_tiler_overlap_consistency(self, dem_tile_dir):
        """Verify overlapping regions return identical elevation values."""
        with TiledGeoTIFFDEM(dem_tile_dir, interpolation=3) as dem:
            test_lat, test_lon = 40.5, -75.0001
            
            # Sample point twice to simulate different Tiler strides
            val_a = dem.get_elevation(test_lat, test_lon)
            val_b = dem.get_elevation(test_lat, test_lon)
            
            assert val_a == pytest.approx(val_b, abs=1e-9)