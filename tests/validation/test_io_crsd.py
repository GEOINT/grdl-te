# -*- coding: utf-8 -*-
"""
CRSDReader validation using real CRSD data.

Tests:
- Level 1: Context manager, metadata, shape
- Level 2: Complex array output, finite values
- Level 3: Normalizer integration

Dataset: CRSD files (*.crsd)

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
    from grdl.IO.sar import CRSDReader
    _HAS_CRSD = True
except ImportError:
    _HAS_CRSD = False

try:
    from grdl.data_prep import Normalizer
    _HAS_DATA_PREP = True
except ImportError:
    _HAS_DATA_PREP = False


pytestmark = [
    pytest.mark.crsd,
    pytest.mark.requires_data,
    pytest.mark.skipif(not _HAS_CRSD, reason="CRSDReader not available"),
]


# =============================================================================
# Level 1: Format Validation
# =============================================================================


@pytest.mark.slow
def test_crsd_reader_opens(require_crsd_file):
    """Context manager opens without exception."""
    with CRSDReader(require_crsd_file) as reader:
        assert reader is not None


@pytest.mark.slow
def test_crsd_metadata_populated(require_crsd_file):
    """typed_metadata is not None."""
    with CRSDReader(require_crsd_file) as reader:
        assert reader.metadata is not None


@pytest.mark.slow
def test_crsd_shape(require_crsd_file):
    """get_shape() returns valid tuple."""
    with CRSDReader(require_crsd_file) as reader:
        shape = reader.get_shape()
        assert isinstance(shape, tuple)
        assert len(shape) >= 2
        assert all(isinstance(d, int) and d > 0 for d in shape)


# =============================================================================
# Level 2: Data Quality
# =============================================================================


@pytest.mark.slow
def test_crsd_read_full_complex(require_crsd_file):
    """read_full() returns complex array."""
    with CRSDReader(require_crsd_file) as reader:
        data = reader.read_full()
        assert isinstance(data, np.ndarray)
        # CRSD may return structured array with real/imag fields; convert to complex if needed
        if data.dtype.names and 'real' in data.dtype.names:
            data = data['real'].astype(np.complex128) + 1j * data['imag'].astype(np.complex128)
        assert np.iscomplexobj(data), f"Expected complex data, got {data.dtype}"
        assert data.size > 0


@pytest.mark.slow
def test_crsd_values_finite(require_crsd_file):
    """No NaN/Inf in data."""
    with CRSDReader(require_crsd_file) as reader:
        data = reader.read_full()
        # Handle structured arrays (real/imag fields)
        if data.dtype.names and 'real' in data.dtype.names:
            real_vals = data['real'].astype(np.float32)
            imag_vals = data['imag'].astype(np.float32)
            assert np.all(np.isfinite(real_vals)) and np.all(np.isfinite(imag_vals)), \
                "CRSD data contains NaN or Inf"
        else:
            assert np.all(np.isfinite(data)), "CRSD data contains NaN or Inf"


# =============================================================================
# Level 3: Integration
# =============================================================================


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
def test_crsd_integration_with_normalizer(require_crsd_file):
    """Normalizer works with CRSD magnitude data."""
    with CRSDReader(require_crsd_file) as reader:
        data = reader.read_full()
        # Handle structured array (real/imag fields from CRSD reader)
        if data.dtype.names and 'real' in data.dtype.names:
            complex_data = data['real'] + 1j * data['imag']
            magnitude = np.abs(complex_data).astype(np.float32)
        else:
            magnitude = np.abs(data).astype(np.float32)
        normalizer = Normalizer(method='minmax')
        normalized = normalizer.normalize(magnitude)
        assert np.isfinite(normalized).all()
        assert normalized.min() == pytest.approx(0.0, abs=1e-6), (
            f"MinMax output min = {normalized.min():.8f}; must be exactly 0.0"
        )
        assert normalized.max() == pytest.approx(1.0, abs=1e-6), (
            f"MinMax output max = {normalized.max():.8f}; must be exactly 1.0"
        )
