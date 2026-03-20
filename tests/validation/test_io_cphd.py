# -*- coding: utf-8 -*-
"""
CPHDReader validation using real CPHD phase history data.

Tests:
- Level 1: Context manager, metadata, shape
- Level 2: Complex array output, PVP population, finite values
- Level 3: Metadata to CollectionGeometry integration

Dataset: CPHD phase history files (*.cphd)

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
    from grdl.IO.sar import CPHDReader
    _HAS_CPHD = True
except ImportError:
    _HAS_CPHD = False


try:
    from grdl.data_prep import ChipExtractor, Normalizer
    _HAS_DATA_PREP = True
except ImportError:
    _HAS_DATA_PREP = False


pytestmark = [
    pytest.mark.cphd,
    pytest.mark.requires_data,
    pytest.mark.skipif(not _HAS_CPHD, reason="CPHDReader not available"),
]


# =============================================================================
# Level 1: Format Validation
# =============================================================================


@pytest.mark.slow
def test_cphd_reader_opens(require_cphd_file):
    """Context manager opens without exception."""
    with CPHDReader(require_cphd_file) as reader:
        assert reader is not None


@pytest.mark.slow
def test_cphd_metadata_populated(require_cphd_file):
    """typed_metadata is not None."""
    with CPHDReader(require_cphd_file) as reader:
        assert reader.metadata is not None


@pytest.mark.slow
def test_cphd_shape(require_cphd_file):
    """get_shape() returns (n_vectors, n_samples) tuple."""
    with CPHDReader(require_cphd_file) as reader:
        shape = reader.get_shape()
        assert isinstance(shape, tuple)
        assert len(shape) >= 2
        assert all(isinstance(d, int) and d > 0 for d in shape)


# =============================================================================
# Level 2: Data Quality
# =============================================================================


@pytest.mark.slow
def test_cphd_read_full_complex(require_cphd_file):
    """read_full() returns complex array."""
    with CPHDReader(require_cphd_file) as reader:
        data = reader.read_full()
        assert isinstance(data, np.ndarray)
        assert np.iscomplexobj(data)
        assert data.size > 0


@pytest.mark.slow
def test_cphd_pvp_populated(require_cphd_file):
    """read_pvp() returns structured array with fields."""
    with CPHDReader(require_cphd_file) as reader:
        pvp = reader.metadata.pvp
        assert pvp is not None
        # PVP should have core fields populated
        assert pvp.tx_time is not None or pvp.tx_pos is not None


@pytest.mark.slow
def test_cphd_values_finite(require_cphd_file):
    """No NaN/Inf in phase data."""
    with CPHDReader(require_cphd_file) as reader:
        data = reader.read_full()
        assert np.all(np.isfinite(data)), "CPHD data contains NaN or Inf"


# =============================================================================
# Level 3: Integration
# =============================================================================


@pytest.mark.slow
@pytest.mark.integration
def test_cphd_metadata_to_collection_geometry(require_cphd_file):
    """typed_metadata → CollectionGeometry construction."""
    try:
        from grdl.image_processing.sar import CollectionGeometry
    except ImportError:
        pytest.skip("CollectionGeometry not available")

    with CPHDReader(require_cphd_file) as reader:
        meta = reader.metadata
        geom = CollectionGeometry(meta)
        assert geom is not None
        assert hasattr(geom, 'graz_ang') or hasattr(geom, 'azim_ang')


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
def test_cphd_chip_extractor(require_cphd_file):
    """ChipExtractor partitions CPHD phase history data."""
    with CPHDReader(require_cphd_file) as reader:
        shape = reader.get_shape()
        rows, cols = shape[0], shape[1]
        extractor = ChipExtractor(nrows=rows, ncols=cols)
        regions = extractor.chip_positions(row_width=128, col_width=128)
        assert len(regions) > 0

        region = regions[0]
        assert 0 <= region.row_start < region.row_end <= rows
        assert 0 <= region.col_start < region.col_end <= cols


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
def test_cphd_normalizer(require_cphd_file):
    """MinMax normalization on CPHD magnitude produces [0, 1]."""
    with CPHDReader(require_cphd_file) as reader:
        data = reader.read_full()
        magnitude = np.abs(data).astype(np.float64)

        normalizer = Normalizer(method='minmax')
        normalized = normalizer.normalize(magnitude)

        assert normalized.min() == pytest.approx(0.0, abs=1e-6)
        assert normalized.max() == pytest.approx(1.0, abs=1e-6)
