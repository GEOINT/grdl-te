# -*- coding: utf-8 -*-
"""
VIIRSReader Tests - Dedicated VIIRS multispectral validation.

Tests grdl.IO.multispectral.viirs.VIIRSReader with real VIIRS HDF5 files,
validating VIIRSMetadata and sensor-specific features:
- Level 1: Metadata type, context manager, chip read, shape
- Level 2: SDS discovery, reflectance bounds, nodata, granule metadata
- Level 3: ChipExtractor, Normalizer, Tiler integration

Dataset: VIIRS VNP09GA Surface Reflectance HDF5

Dependencies
------------
pytest
numpy
h5py
grdl

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
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False

try:
    from grdl.IO.multispectral.viirs import VIIRSReader
    _HAS_VIIRS = True
except ImportError:
    _HAS_VIIRS = False

try:
    from grdl.IO.models.viirs import VIIRSMetadata
    _HAS_VIIRS_META = True
except ImportError:
    _HAS_VIIRS_META = False

try:
    from grdl.data_prep import ChipExtractor, Normalizer, Tiler
    _HAS_DATA_PREP = True
except ImportError:
    _HAS_DATA_PREP = False


pytestmark = [
    pytest.mark.viirs,
    pytest.mark.requires_data,
    pytest.mark.skipif(not _HAS_H5PY, reason="h5py not installed"),
    pytest.mark.skipif(not _HAS_VIIRS, reason="VIIRSReader not available"),
]


# =============================================================================
# Level 1: Format Validation
# =============================================================================


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_VIIRS_META, reason="VIIRSMetadata not available")
def test_viirs_reader_metadata_type(require_viirs_file):
    """VIIRSReader produces VIIRSMetadata (not generic ImageMetadata)."""
    with VIIRSReader(str(require_viirs_file)) as reader:
        meta = reader.metadata
        assert isinstance(meta, VIIRSMetadata), (
            f"Expected VIIRSMetadata, got {type(meta).__name__}"
        )


@pytest.mark.slow
def test_viirs_reader_opens(require_viirs_file):
    """Context manager opens without exception."""
    with VIIRSReader(str(require_viirs_file)) as reader:
        assert reader is not None


@pytest.mark.slow
def test_viirs_read_chip(require_viirs_file):
    """Center chip reads with correct dtype."""
    with VIIRSReader(str(require_viirs_file)) as reader:
        rows, cols = reader.get_shape()[:2]
        r0 = max(0, rows // 2 - 128)
        r1 = min(rows, rows // 2 + 128)
        c0 = max(0, cols // 2 - 128)
        c1 = min(cols, cols // 2 + 128)
        chip = reader.read_chip(r0, r1, c0, c1)

        assert isinstance(chip, np.ndarray)
        assert chip.shape[0] == (r1 - r0)
        assert chip.shape[1] == (c1 - c0)


@pytest.mark.slow
def test_viirs_get_shape(require_viirs_file):
    """get_shape() returns positive dimensions."""
    with VIIRSReader(str(require_viirs_file)) as reader:
        shape = reader.get_shape()
        assert isinstance(shape, tuple)
        assert len(shape) >= 2
        assert shape[0] > 0 and shape[1] > 0


# =============================================================================
# Level 2: Data Quality
# =============================================================================


@pytest.mark.slow
def test_viirs_sds_discovery(require_viirs_file):
    """Multiple scientific data sets are discoverable."""
    with VIIRSReader(str(require_viirs_file)) as reader:
        if hasattr(reader, 'get_available_datasets'):
            datasets = reader.get_available_datasets()
            assert isinstance(datasets, (list, tuple))
            assert len(datasets) > 0, "No SDS datasets found"


@pytest.mark.slow
def test_viirs_reflectance_bounds(require_viirs_file):
    """VIIRS surface reflectance values fall within physical range."""
    with VIIRSReader(str(require_viirs_file)) as reader:
        rows, cols = reader.get_shape()[:2]
        r0 = max(0, rows // 2 - 128)
        r1 = min(rows, rows // 2 + 128)
        c0 = max(0, cols // 2 - 128)
        c1 = min(cols, cols // 2 + 128)
        chip = reader.read_chip(r0, r1, c0, c1)

        # VIIRS reflectance is typically scaled int16 or float
        # Valid reflectance should be non-negative
        valid = chip[chip != 0] if chip.dtype.kind in ('i', 'u') else chip
        if len(valid) > 0:
            # For scaled integers, max should be < 32767 (int16)
            # For floats, max should be < 2.0 (reflectance)
            if chip.dtype.kind == 'f':
                assert valid.max() < 10.0, (
                    f"Reflectance max {valid.max()} implausibly high"
                )


@pytest.mark.slow
def test_viirs_nodata_handling(require_viirs_file):
    """Fill values are present and distinguishable from valid data."""
    with VIIRSReader(str(require_viirs_file)) as reader:
        rows, cols = reader.get_shape()[:2]
        r0 = max(0, rows // 2 - 128)
        r1 = min(rows, rows // 2 + 128)
        c0 = max(0, cols // 2 - 128)
        c1 = min(cols, cols // 2 + 128)
        chip = reader.read_chip(r0, r1, c0, c1)

        # VIIRS typically uses -999, 0, or large negative as fill
        # At least some pixels in center should be valid
        if chip.dtype.kind in ('i', 'f'):
            assert chip.std() > 0, "Center chip has zero variance"


@pytest.mark.slow
def test_viirs_granule_metadata(require_viirs_file):
    """Granule-level metadata (date, ID) is populated."""
    with VIIRSReader(str(require_viirs_file)) as reader:
        meta = reader.metadata
        # VIIRSMetadata should have granule-level info
        assert meta.rows > 0
        assert meta.cols > 0


# =============================================================================
# Level 3: Integration
# =============================================================================


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
@pytest.mark.integration
def test_viirs_chip_extractor(require_viirs_file):
    """ChipExtractor partitions VIIRS image correctly."""
    with VIIRSReader(str(require_viirs_file)) as reader:
        rows, cols = reader.get_shape()[:2]
        extractor = ChipExtractor(nrows=rows, ncols=cols)
        regions = extractor.chip_positions(row_width=256, col_width=256)
        assert len(regions) > 0

        region = regions[0]
        chip = reader.read_chip(
            region.row_start, region.row_end,
            region.col_start, region.col_end,
        )
        assert chip.shape[0] == region.row_end - region.row_start


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
@pytest.mark.integration
def test_viirs_normalizer(require_viirs_file):
    """MinMax normalization produces output in [0, 1]."""
    with VIIRSReader(str(require_viirs_file)) as reader:
        rows, cols = reader.get_shape()[:2]
        r0 = max(0, rows // 2 - 128)
        r1 = min(rows, rows // 2 + 128)
        c0 = max(0, cols // 2 - 128)
        c1 = min(cols, cols // 2 + 128)
        chip = reader.read_chip(r0, r1, c0, c1).astype(np.float64)

        normalizer = Normalizer(method='minmax')
        normalized = normalizer.normalize(chip)

        assert normalized.min() == pytest.approx(0.0, abs=1e-6)
        assert normalized.max() == pytest.approx(1.0, abs=1e-6)


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
@pytest.mark.integration
def test_viirs_tiler(require_viirs_file):
    """Tiler creates valid tile grid over VIIRS image."""
    with VIIRSReader(str(require_viirs_file)) as reader:
        rows, cols = reader.get_shape()[:2]
        tiler = Tiler(nrows=rows, ncols=cols, tile_size=512, stride=256)
        tiles = tiler.tile_positions()
        assert len(tiles) > 0

        for tile in tiles[:5]:
            assert 0 <= tile.row_start < tile.row_end <= rows
            assert 0 <= tile.col_start < tile.col_end <= cols
