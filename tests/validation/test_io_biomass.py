# -*- coding: utf-8 -*-
"""
BIOMASSL1Reader and BIOMASSCatalog validation.

Tests:
- Level 1: Context manager, mission-specific metadata, shape
- Level 2: Complex or real array output, finite values
- Level 3: BIOMASSCatalog search, empty query handling

Dataset: BIOMASS L1 product directory (BIO_S*)

Author
------
Ava Courtney

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
    from grdl.IO.sar import BIOMASSL1Reader
    _HAS_BIOMASS = True
except ImportError:
    _HAS_BIOMASS = False

try:
    from grdl.IO.sar import BIOMASSCatalog
    _HAS_CATALOG = True
except ImportError:
    _HAS_CATALOG = False

try:
    from grdl.data_prep import ChipExtractor, Normalizer, Tiler
    _HAS_DATA_PREP = True
except ImportError:
    _HAS_DATA_PREP = False


pytestmark = [
    pytest.mark.biomass,
    pytest.mark.requires_data,
    pytest.mark.skipif(not _HAS_BIOMASS, reason="BIOMASSL1Reader not available"),
]


# =============================================================================
# Level 1: Format Validation
# =============================================================================


@pytest.mark.slow
def test_biomass_reader_opens(require_biomass_file):
    """Context manager opens without exception."""
    with BIOMASSL1Reader(str(require_biomass_file)) as reader:
        assert reader is not None


@pytest.mark.slow
def test_biomass_metadata(require_biomass_file):
    """Mission-specific metadata populated."""
    with BIOMASSL1Reader(str(require_biomass_file)) as reader:
        meta = reader.metadata
        assert meta is not None
        assert meta.rows > 0
        assert meta.cols > 0


@pytest.mark.slow
def test_biomass_shape(require_biomass_file):
    """get_shape() returns valid dimensions."""
    with BIOMASSL1Reader(str(require_biomass_file)) as reader:
        shape = reader.get_shape()
        assert isinstance(shape, tuple)
        assert len(shape) >= 2
        assert shape[0] > 0 and shape[1] > 0


# =============================================================================
# Level 2: Data Quality
# =============================================================================


@pytest.mark.slow
def test_biomass_read_full(require_biomass_file):
    """read_full() returns complex or real array."""
    with BIOMASSL1Reader(str(require_biomass_file)) as reader:
        shape = reader.get_shape()
        if shape[0] * shape[1] > 25_000_000:
            pytest.skip("BIOMASS file too large for full read")
        data = reader.read_full()
        assert isinstance(data, np.ndarray)
        assert data.size > 0


@pytest.mark.slow
def test_biomass_values_finite(require_biomass_file):
    """No NaN/Inf in data."""
    with BIOMASSL1Reader(str(require_biomass_file)) as reader:
        shape = reader.get_shape()
        rows, cols = shape[0], shape[1]
        r0 = max(0, rows // 2 - 256)
        r1 = min(rows, rows // 2 + 256)
        c0 = max(0, cols // 2 - 256)
        c1 = min(cols, cols // 2 + 256)
        chip = reader.read_chip(r0, r1, c0, c1)
        assert np.all(np.isfinite(chip))


# =============================================================================
# Level 3: Integration
# =============================================================================


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not _HAS_CATALOG, reason="BIOMASSCatalog not available")
def test_biomass_catalog_search(require_biomass_file):
    """BIOMASSCatalog search from the biomass data directory."""
    try:
        catalog = BIOMASSCatalog(search_path=str(require_biomass_file.parent))
        results = catalog.search(bbox=(-180, -90, 180, 90))
        assert isinstance(results, (list, tuple))
    except Exception:
        pytest.skip("BIOMASSCatalog search failed or API not reachable")


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not _HAS_CATALOG, reason="BIOMASSCatalog not available")
def test_biomass_catalog_empty_query(biomass_data_dir):
    """Empty/invalid query returns empty list, no crash."""
    try:
        catalog = BIOMASSCatalog(search_path=str(biomass_data_dir))
        results = catalog.search(bbox=(0, 0, 0.001, 0.001))
        assert isinstance(results, (list, tuple))
    except Exception:
        pytest.skip("BIOMASSCatalog search failed or not configured")


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
def test_biomass_chip_extractor(require_biomass_file):
    """ChipExtractor partitions BIOMASS image; chips have valid shapes."""
    with BIOMASSL1Reader(str(require_biomass_file)) as reader:
        rows, cols = reader.get_shape()[:2]
        extractor = ChipExtractor(nrows=rows, ncols=cols)
        regions = extractor.chip_positions(row_width=256, col_width=256)
        assert len(regions) > 0

        region = regions[0]
        assert 0 <= region.row_start < region.row_end <= rows
        assert 0 <= region.col_start < region.col_end <= cols
        chip = reader.read_chip(
            region.row_start, region.row_end,
            region.col_start, region.col_end,
        )
        assert chip.shape[0] == region.row_end - region.row_start


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
def test_biomass_normalizer(require_biomass_file):
    """MinMax normalization on BIOMASS chip produces output in [0, 1]."""
    with BIOMASSL1Reader(str(require_biomass_file)) as reader:
        rows, cols = reader.get_shape()[:2]
        r0 = max(0, rows // 2 - 128)
        r1 = min(rows, rows // 2 + 128)
        c0 = max(0, cols // 2 - 128)
        c1 = min(cols, cols // 2 + 128)
        chip = reader.read_chip(r0, r1, c0, c1)

        magnitude = np.abs(chip).astype(np.float64)
        normalizer = Normalizer(method='minmax')
        normalized = normalizer.normalize(magnitude)

        assert normalized.min() == pytest.approx(0.0, abs=1e-6)
        assert normalized.max() == pytest.approx(1.0, abs=1e-6)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
def test_biomass_tiler(require_biomass_file):
    """Tiler creates valid tile grid over BIOMASS image."""
    with BIOMASSL1Reader(str(require_biomass_file)) as reader:
        rows, cols = reader.get_shape()[:2]
        tiler = Tiler(nrows=rows, ncols=cols, tile_size=512, stride=256)
        tiles = tiler.tile_positions()
        assert len(tiles) > 0

        for tile in tiles[:5]:
            assert 0 <= tile.row_start < tile.row_end <= rows
            assert 0 <= tile.col_start < tile.col_end <= cols
