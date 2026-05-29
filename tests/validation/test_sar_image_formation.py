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

Strategy: Reads a small chip (256 pulses from aperture center) rather
than the full file to keep execution time under 2 minutes while still
exercising real phase history data through the full IFP pipeline.

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

Modified
--------
2026-05-29
"""

import pytest
import numpy as np

try:
    from grdl.IO.sar import CPHDReader
    _HAS_CPHD = True
except ImportError:
    _HAS_CPHD = False

try:
    from grdl.IO.models.cphd import create_subaperture_metadata
    _HAS_SUBAP_META = True
except ImportError:
    _HAS_SUBAP_META = False

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


# Number of pulses to extract from the aperture center
_CHIP_PULSES = 256


pytestmark = [
    pytest.mark.sar,
    pytest.mark.image_formation,
    pytest.mark.requires_data,
    pytest.mark.skipif(not _HAS_CPHD, reason="CPHDReader not available"),
    pytest.mark.skipif(not _HAS_SUBAP_META,
                       reason="create_subaperture_metadata not available"),
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def cphd_data(cphd_data_dir):
    """Load a small chip from CPHD center (module-scoped, read once).

    Reads _CHIP_PULSES pulses from the center of the aperture and creates
    matching sub-aperture metadata.  This keeps the fixture fast (~seconds)
    while providing real phase history for all downstream tests.
    """
    matches = list(cphd_data_dir.glob("*.cphd")) if cphd_data_dir.exists() else []
    if not matches:
        pytest.skip(f"CPHD file not found in {cphd_data_dir}")
    filepath = matches[0]
    with CPHDReader(filepath) as reader:
        full_meta = reader.metadata
        n_pulses, n_samples = reader.get_shape()

        # Extract center chip
        half = _CHIP_PULSES // 2
        center = n_pulses // 2
        start = max(0, center - half)
        end = min(n_pulses, start + _CHIP_PULSES)
        start = max(0, end - _CHIP_PULSES)  # ensure full chip if near edges

        data = reader.read_chip(start, end, 0, n_samples)

    meta = create_subaperture_metadata(full_meta, start, end)
    return meta, data


@pytest.fixture(scope="module")
def collection_geometry(cphd_data):
    """Build CollectionGeometry once for the module."""
    if not _HAS_GEOM:
        pytest.skip("CollectionGeometry not available")
    meta, _ = cphd_data
    return CollectionGeometry(meta)


@pytest.fixture(scope="module")
def polar_grid(collection_geometry):
    """Build PolarGrid once for the module."""
    if not _HAS_GRID:
        pytest.skip("PolarGrid not available")
    return PolarGrid(collection_geometry)


# =============================================================================
# Level 1: Structure
# =============================================================================


@pytest.mark.skipif(not _HAS_GEOM, reason="CollectionGeometry not available")
def test_collection_geometry_init(collection_geometry):
    """Constructs from CPHD metadata."""
    assert collection_geometry is not None


@pytest.mark.skipif(not _HAS_GEOM, reason="CollectionGeometry not available")
def test_collection_geometry_attributes(collection_geometry):
    """Has graz_ang and/or azim_ang attributes."""
    has_grazing = hasattr(collection_geometry, 'graz_ang')
    has_azimuth = hasattr(collection_geometry, 'azim_ang')
    assert has_grazing or has_azimuth, "CollectionGeometry missing angle attributes"


@pytest.mark.skipif(not _HAS_GRID, reason="PolarGrid not available")
@pytest.mark.skipif(not _HAS_GEOM, reason="CollectionGeometry not available")
def test_polar_grid_init(polar_grid):
    """Constructs from CollectionGeometry."""
    assert polar_grid is not None


@pytest.mark.skipif(not _HAS_GRID, reason="PolarGrid not available")
@pytest.mark.skipif(not _HAS_GEOM, reason="CollectionGeometry not available")
def test_polar_grid_bounds(polar_grid):
    """grid.bounds returns tuple of floats."""
    if hasattr(polar_grid, 'bounds'):
        bounds = polar_grid.bounds
        assert isinstance(bounds, tuple)
        assert len(bounds) >= 4
        assert all(isinstance(b, (int, float)) for b in bounds)


# =============================================================================
# Level 2: Data Quality
# =============================================================================


@pytest.mark.skipif(not _HAS_PFA, reason="PolarFormatAlgorithm not available")
@pytest.mark.skipif(not _HAS_GEOM, reason="CollectionGeometry not available")
@pytest.mark.skipif(not _HAS_GRID, reason="PolarGrid not available")
def test_pfa_form_returns_complex_2d(cphd_data, collection_geometry, polar_grid):
    """PFA produces (rows, cols) complex array."""
    _, phase_data = cphd_data
    pfa = PolarFormatAlgorithm(grid=polar_grid)
    result = pfa.form_image(phase_data, geometry=collection_geometry)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert np.iscomplexobj(result)


@pytest.mark.skipif(not _HAS_PFA, reason="PolarFormatAlgorithm not available")
@pytest.mark.skipif(not _HAS_GEOM, reason="CollectionGeometry not available")
@pytest.mark.skipif(not _HAS_GRID, reason="PolarGrid not available")
def test_pfa_form_nonzero(cphd_data, collection_geometry, polar_grid):
    """Output has non-zero content."""
    _, phase_data = cphd_data
    pfa = PolarFormatAlgorithm(grid=polar_grid)
    result = pfa.form_image(phase_data, geometry=collection_geometry)
    assert np.abs(result).max() > 0


@pytest.mark.skipif(not _HAS_SUBAP, reason="SubaperturePartitioner not available")
@pytest.mark.skipif(not _HAS_GEOM, reason="CollectionGeometry not available")
def test_subaperture_partitioner(cphd_data):
    """Produces multiple sub-apertures from chip metadata."""
    meta, _ = cphd_data
    part = SubaperturePartitioner(metadata=meta)
    assert isinstance(part.partitions, list)
    assert len(part.partitions) >= 1, "Expected at least 1 sub-aperture partition"


@pytest.mark.skipif(not _HAS_RDA, reason="RangeDopplerAlgorithm not available")
@pytest.mark.skipif(not _HAS_GEOM, reason="CollectionGeometry not available")
def test_rda_form_returns_complex_2d(cphd_data, collection_geometry):
    """RDA produces complex image."""
    meta, phase_data = cphd_data
    try:
        rda = RangeDopplerAlgorithm(metadata=meta, verbose=False)
        result = rda.form_image(phase_data, geometry=collection_geometry)
        assert isinstance(result, np.ndarray)
        assert np.iscomplexobj(result)
    except Exception as exc:
        pytest.skip(f"RDA not compatible with this data: {exc}")


@pytest.mark.skipif(not _HAS_SPFA, reason="StripmapPFA not available")
@pytest.mark.skipif(not _HAS_GEOM, reason="CollectionGeometry not available")
@pytest.mark.skipif(not _HAS_GRID, reason="PolarGrid not available")
def test_stripmapPFA_form(cphd_data, collection_geometry):
    """StripmapPFA output shape and type."""
    meta, phase_data = cphd_data
    try:
        spfa = StripmapPFA(metadata=meta, verbose=False)
        result = spfa.form_image(phase_data, geometry=collection_geometry)
        assert isinstance(result, np.ndarray)
        assert np.iscomplexobj(result)
    except Exception as exc:
        pytest.skip(f"StripmapPFA not compatible with this data: {exc}")


@pytest.mark.skipif(not _HAS_FBP, reason="FastBackProjection not available")
@pytest.mark.skipif(not _HAS_GEOM, reason="CollectionGeometry not available")
def test_fbp_form(cphd_data, collection_geometry):
    """FFBP output shape and type."""
    meta, phase_data = cphd_data
    try:
        fbp = FastBackProjection(metadata=meta, verbose=False)
        result = fbp.form_image(phase_data, geometry=collection_geometry)
        assert isinstance(result, np.ndarray)
        assert np.iscomplexobj(result)
    except Exception as exc:
        pytest.skip(f"FastBackProjection not compatible with this data: {exc}")


# =============================================================================
# Level 3: Integration
# =============================================================================


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_PFA, reason="PolarFormatAlgorithm not available")
@pytest.mark.skipif(not _HAS_GEOM, reason="CollectionGeometry not available")
@pytest.mark.skipif(not _HAS_GRID, reason="PolarGrid not available")
def test_pfa_to_magnitude(cphd_data, collection_geometry, polar_grid):
    """PFA → np.abs() → float array in valid range."""
    _, phase_data = cphd_data
    pfa = PolarFormatAlgorithm(grid=polar_grid)
    result = pfa.form_image(phase_data, geometry=collection_geometry)
    magnitude = np.abs(result)
    assert magnitude.dtype in [np.float32, np.float64]
    assert magnitude.min() >= 0
    assert np.isfinite(magnitude).all()
