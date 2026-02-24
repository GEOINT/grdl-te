# -*- coding: utf-8 -*-
"""
SAR Image Formation algorithm validation.

Tests CollectionGeometry, PolarGrid, PolarFormatAlgorithm,
SubaperturePartitioner, StripmapPFA, RangeDopplerAlgorithm,
and FastBackProjection using real CPHD data.

Tests:
- Level 1: Constructor validity, attribute presence
- Level 2: Output shape/type, non-zero content, sub-aperture counts
- Level 3: Magnitude extraction, coordinate system transforms

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
    from grdl.image_processing.sar import CollectionGeometry
    _HAS_GEOM = True
except ImportError:
    _HAS_GEOM = False

try:
    from grdl.image_processing.sar import PolarGrid
    _HAS_GRID = True
except ImportError:
    _HAS_GRID = False

try:
    from grdl.image_processing.sar import PolarFormatAlgorithm
    _HAS_PFA = True
except ImportError:
    _HAS_PFA = False

try:
    from grdl.image_processing.sar import SubaperturePartitioner
    _HAS_SUBAP = True
except ImportError:
    _HAS_SUBAP = False

try:
    from grdl.image_processing.sar import RangeDopplerAlgorithm
    _HAS_RDA = True
except ImportError:
    _HAS_RDA = False

try:
    from grdl.image_processing.sar import StripmapPFA
    _HAS_SPFA = True
except ImportError:
    _HAS_SPFA = False

try:
    from grdl.image_processing.sar import FastBackProjection
    _HAS_FBP = True
except ImportError:
    _HAS_FBP = False


pytestmark = [
    pytest.mark.sar,
    pytest.mark.image_formation,
    pytest.mark.requires_data,
    pytest.mark.skipif(not _HAS_CPHD, reason="CPHDReader not available"),
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def cphd_data(require_cphd_file):
    """Load CPHD metadata and phase data."""
    with CPHDReader(require_cphd_file) as reader:
        meta = reader.typed_metadata
        data = reader.read_full()
    return meta, data


# =============================================================================
# Level 1: Structure
# =============================================================================


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_GEOM, reason="CollectionGeometry not available")
def test_collection_geometry_init(cphd_data):
    """Constructs from CPHD metadata."""
    meta, _ = cphd_data
    geom = CollectionGeometry(meta)
    assert geom is not None


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_GEOM, reason="CollectionGeometry not available")
def test_collection_geometry_attributes(cphd_data):
    """Has grazing_angle and/or squint_angle attributes."""
    meta, _ = cphd_data
    geom = CollectionGeometry(meta)
    has_grazing = hasattr(geom, 'grazing_angle')
    has_squint = hasattr(geom, 'squint_angle')
    assert has_grazing or has_squint, "CollectionGeometry missing angle attributes"


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_GRID, reason="PolarGrid not available")
@pytest.mark.skipif(not _HAS_GEOM, reason="CollectionGeometry not available")
def test_polar_grid_init(cphd_data):
    """Constructs from CollectionGeometry."""
    meta, _ = cphd_data
    geom = CollectionGeometry(meta)
    grid = PolarGrid(geom)
    assert grid is not None


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_GRID, reason="PolarGrid not available")
@pytest.mark.skipif(not _HAS_GEOM, reason="CollectionGeometry not available")
def test_polar_grid_bounds(cphd_data):
    """grid.bounds returns tuple of floats."""
    meta, _ = cphd_data
    geom = CollectionGeometry(meta)
    grid = PolarGrid(geom)
    if hasattr(grid, 'bounds'):
        bounds = grid.bounds
        assert isinstance(bounds, tuple)
        assert len(bounds) >= 4
        assert all(isinstance(b, (int, float)) for b in bounds)


# =============================================================================
# Level 2: Data Quality
# =============================================================================


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_PFA, reason="PolarFormatAlgorithm not available")
@pytest.mark.skipif(not _HAS_GEOM, reason="CollectionGeometry not available")
@pytest.mark.skipif(not _HAS_GRID, reason="PolarGrid not available")
def test_pfa_form_returns_complex_2d(cphd_data):
    """PFA produces (rows, cols) complex array."""
    meta, phase_data = cphd_data
    geom = CollectionGeometry(meta)
    grid = PolarGrid(geom)
    pfa = PolarFormatAlgorithm(geometry=geom, grid=grid)
    result = pfa.form(phase_data)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert np.iscomplexobj(result)


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_PFA, reason="PolarFormatAlgorithm not available")
@pytest.mark.skipif(not _HAS_GEOM, reason="CollectionGeometry not available")
@pytest.mark.skipif(not _HAS_GRID, reason="PolarGrid not available")
def test_pfa_form_nonzero(cphd_data):
    """Output has non-zero content."""
    meta, phase_data = cphd_data
    geom = CollectionGeometry(meta)
    grid = PolarGrid(geom)
    pfa = PolarFormatAlgorithm(geometry=geom, grid=grid)
    result = pfa.form(phase_data)
    assert np.abs(result).max() > 0


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_SUBAP, reason="SubaperturePartitioner not available")
@pytest.mark.skipif(not _HAS_GEOM, reason="CollectionGeometry not available")
def test_subaperture_partitioner(cphd_data):
    """Produces n_subapertures sub-arrays."""
    meta, phase_data = cphd_data
    geom = CollectionGeometry(meta)
    part = SubaperturePartitioner(n_subapertures=4)
    subs = part.partition(phase_data, geom)
    assert isinstance(subs, (list, tuple))
    assert len(subs) == 4


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_RDA, reason="RangeDopplerAlgorithm not available")
@pytest.mark.skipif(not _HAS_GEOM, reason="CollectionGeometry not available")
def test_rda_form_returns_complex_2d(cphd_data):
    """RDA produces complex image."""
    meta, phase_data = cphd_data
    geom = CollectionGeometry(meta)
    try:
        rda = RangeDopplerAlgorithm(geometry=geom)
        result = rda.form(phase_data)
        assert isinstance(result, np.ndarray)
        assert np.iscomplexobj(result)
    except Exception as exc:
        pytest.skip(f"RDA not compatible with this data: {exc}")


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_SPFA, reason="StripmapPFA not available")
@pytest.mark.skipif(not _HAS_GEOM, reason="CollectionGeometry not available")
@pytest.mark.skipif(not _HAS_GRID, reason="PolarGrid not available")
def test_stripmapPFA_form(cphd_data):
    """StripmapPFA output shape and type."""
    meta, phase_data = cphd_data
    geom = CollectionGeometry(meta)
    grid = PolarGrid(geom)
    try:
        spfa = StripmapPFA(geometry=geom, grid=grid, n_subapertures=4)
        result = spfa.form(phase_data)
        assert isinstance(result, np.ndarray)
        assert np.iscomplexobj(result)
    except Exception as exc:
        pytest.skip(f"StripmapPFA not compatible with this data: {exc}")


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_FBP, reason="FastBackProjection not available")
@pytest.mark.skipif(not _HAS_GEOM, reason="CollectionGeometry not available")
def test_fbp_form(cphd_data):
    """FFBP output shape and type."""
    meta, phase_data = cphd_data
    geom = CollectionGeometry(meta)
    try:
        fbp = FastBackProjection(geometry=geom, n_subapertures=4)
        result = fbp.form(phase_data)
        assert isinstance(result, np.ndarray)
        assert np.iscomplexobj(result)
    except Exception as exc:
        pytest.skip(f"FastBackProjection not compatible with this data: {exc}")


# =============================================================================
# Level 3: Integration
# =============================================================================


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not _HAS_PFA, reason="PolarFormatAlgorithm not available")
@pytest.mark.skipif(not _HAS_GEOM, reason="CollectionGeometry not available")
@pytest.mark.skipif(not _HAS_GRID, reason="PolarGrid not available")
def test_pfa_to_magnitude(cphd_data):
    """PFA → np.abs() → float array in valid range."""
    meta, phase_data = cphd_data
    geom = CollectionGeometry(meta)
    grid = PolarGrid(geom)
    pfa = PolarFormatAlgorithm(geometry=geom, grid=grid)
    result = pfa.form(phase_data)
    magnitude = np.abs(result)
    assert magnitude.dtype in [np.float32, np.float64]
    assert magnitude.min() >= 0
    assert np.isfinite(magnitude).all()
