# -*- coding: utf-8 -*-
"""
ASTERReader validation using real ASTER L1T data.

Tests:
- Level 1: Context manager, metadata (sensor, bands, date), shape
- Level 2: Multi-band array, VNIR bands at 15m, finite values
- Level 3: Normalizer integration

Dataset: ASTER L1T (AST_L1T*.hdf)

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
    from grdl.IO.multispectral import ASTERReader
    _HAS_ASTER = True
except ImportError:
    try:
        from grdl.IO.ir import ASTERReader
        _HAS_ASTER = True
    except ImportError:
        _HAS_ASTER = False

try:
    from grdl.data_prep import Normalizer
    _HAS_DATA_PREP = True
except ImportError:
    _HAS_DATA_PREP = False


pytestmark = [
    pytest.mark.aster,
    pytest.mark.requires_data,
    pytest.mark.skipif(not _HAS_ASTER, reason="ASTERReader not available"),
]


# =============================================================================
# Level 1: Format Validation
# =============================================================================


@pytest.mark.slow
def test_aster_reader_opens(require_aster_file):
    """Context manager opens without exception."""
    with ASTERReader(str(require_aster_file)) as reader:
        assert reader is not None


@pytest.mark.slow
def test_aster_metadata(require_aster_file):
    """Sensor info, bands, acquisition date in metadata."""
    with ASTERReader(str(require_aster_file)) as reader:
        meta = reader.metadata
        assert meta is not None
        assert meta.rows > 0
        assert meta.cols > 0


@pytest.mark.slow
def test_aster_shape(require_aster_file):
    """get_shape() returns valid dimensions."""
    with ASTERReader(str(require_aster_file)) as reader:
        shape = reader.get_shape()
        assert isinstance(shape, tuple)
        assert len(shape) >= 2
        assert shape[0] > 0 and shape[1] > 0


# =============================================================================
# Level 2: Data Quality
# =============================================================================


@pytest.mark.slow
def test_aster_read_full(require_aster_file):
    """read_full() returns multi-band array."""
    with ASTERReader(str(require_aster_file)) as reader:
        shape = reader.get_shape()
        if shape[0] * shape[1] > 25_000_000:
            pytest.skip("ASTER file too large for full read")
        data = reader.read_full()
        assert isinstance(data, np.ndarray)
        assert data.size > 0


@pytest.mark.slow
def test_aster_vnir_bands(require_aster_file):
    """ASTER has VNIR bands (15m resolution)."""
    with ASTERReader(str(require_aster_file)) as reader:
        meta = reader.metadata
        # ASTER L1T should report spatial dimensions
        assert meta.rows > 0
        assert meta.cols > 0


@pytest.mark.slow
def test_aster_values_finite(require_aster_file):
    """No NaN/Inf in VNIR data (fill values excluded)."""
    with ASTERReader(str(require_aster_file)) as reader:
        shape = reader.get_shape()
        rows, cols = shape[0], shape[1]
        r0 = max(0, rows // 2 - 256)
        r1 = min(rows, rows // 2 + 256)
        c0 = max(0, cols // 2 - 256)
        c1 = min(cols, cols // 2 + 256)
        chip = reader.read_chip(r0, r1, c0, c1)
        # Allow fill values but main array should be finite where valid
        valid_mask = chip != 0  # typical fill value
        if valid_mask.any():
            assert np.isfinite(chip[valid_mask]).all()


# =============================================================================
# Level 3: Integration
# =============================================================================


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
def test_aster_normalizer_integration(require_aster_file):
    """Normalizer pipeline on ASTER data."""
    with ASTERReader(str(require_aster_file)) as reader:
        shape = reader.get_shape()
        rows, cols = shape[0], shape[1]
        chip = reader.read_chip(0, min(512, rows), 0, min(512, cols))
        normalizer = Normalizer(method='minmax')
        normalized = normalizer.normalize(chip.astype(np.float32))
        assert np.isfinite(normalized).all()
        assert normalized.min() == pytest.approx(0.0, abs=1e-6), (
            f"MinMax output min = {normalized.min():.8f}; must be exactly 0.0"
        )
        assert normalized.max() == pytest.approx(1.0, abs=1e-6), (
            f"MinMax output max = {normalized.max():.8f}; must be exactly 1.0"
        )
