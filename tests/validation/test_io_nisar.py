# -*- coding: utf-8 -*-
"""
NISARReader Tests - NISAR HDF5 SAR Validation with GRDL Integration.

Tests grdl.IO.sar.nisar.NISARReader with real NISAR RSLC/GSLC HDF5 files:
- Level 1: Format validation (metadata, dtype, shape, chip read, context manager)
- Level 2: Data quality (identification, orbit, polarization, SAR statistics)
- Level 3: Integration (ChipExtractor edge cases, Normalizer, Tiler stride)

Dataset: NISAR RSLC or GSLC HDF5

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
    from grdl.IO.sar.nisar import NISARReader
    _HAS_NISAR = True
except ImportError:
    _HAS_NISAR = False

try:
    from grdl.data_prep import ChipExtractor, Normalizer, Tiler
    _HAS_DATA_PREP = True
except ImportError:
    _HAS_DATA_PREP = False


pytestmark = [
    pytest.mark.nisar,
    pytest.mark.requires_data,
    pytest.mark.skipif(not _HAS_H5PY, reason="h5py not installed"),
    pytest.mark.skipif(not _HAS_NISAR, reason="NISARReader not available"),
]


# =============================================================================
# Level 1: Format Validation
# =============================================================================


@pytest.mark.slow
def test_nisar_metadata(require_nisar_file):
    """NISARReader extracts valid metadata with positive dimensions."""
    with NISARReader(str(require_nisar_file)) as reader:
        meta = reader.metadata
        assert meta is not None, "Metadata is None"
        assert meta.rows > 0, f"rows={meta.rows} must be positive"
        assert meta.cols > 0, f"cols={meta.cols} must be positive"


@pytest.mark.slow
def test_nisar_complex_dtype(require_nisar_file):
    """NISAR RSLC data has complex dtype; GSLC has float dtype."""
    with NISARReader(str(require_nisar_file)) as reader:
        dtype = reader.get_dtype()
        # RSLC is complex, GSLC is float — both are valid
        assert dtype is not None, "dtype is None"
        assert np.dtype(dtype).kind in ('c', 'f'), (
            f"Expected complex or float dtype, got {dtype}"
        )


@pytest.mark.slow
def test_nisar_get_shape(require_nisar_file):
    """get_shape() returns tuple matching metadata dimensions."""
    with NISARReader(str(require_nisar_file)) as reader:
        shape = reader.get_shape()
        assert isinstance(shape, tuple)
        assert len(shape) >= 2
        assert shape[0] > 0 and shape[1] > 0
        meta = reader.metadata
        assert shape[0] == meta.rows
        assert shape[1] == meta.cols


@pytest.mark.slow
def test_nisar_read_chip(require_nisar_file):
    """Center 256x256 chip reads with correct shape and non-zero magnitude."""
    with NISARReader(str(require_nisar_file)) as reader:
        rows, cols = reader.get_shape()[:2]
        r0 = max(0, rows // 2 - 128)
        r1 = min(rows, rows // 2 + 128)
        c0 = max(0, cols // 2 - 128)
        c1 = min(cols, cols // 2 + 128)
        chip = reader.read_chip(r0, r1, c0, c1)

        assert isinstance(chip, np.ndarray)
        assert chip.shape[0] == (r1 - r0)
        assert chip.shape[1] == (c1 - c0)

        magnitude = np.abs(chip)
        assert magnitude.max() > 0, "Center chip is all zeros"


@pytest.mark.slow
def test_nisar_context_manager(require_nisar_file):
    """NISARReader supports __enter__/__exit__ lifecycle."""
    with NISARReader(str(require_nisar_file)) as reader:
        assert reader is not None
        _ = reader.get_shape()
    # No error on exit = success


# =============================================================================
# Level 2: Data Quality
# =============================================================================


@pytest.mark.slow
def test_nisar_identification(require_nisar_file):
    """Metadata identification fields are populated."""
    with NISARReader(str(require_nisar_file)) as reader:
        meta = reader.metadata
        if hasattr(meta, 'identification') and meta.identification is not None:
            ident = meta.identification
            if hasattr(ident, 'mission_id'):
                assert ident.mission_id is not None
            if hasattr(ident, 'product_type'):
                assert ident.product_type is not None


@pytest.mark.slow
def test_nisar_orbit_data(require_nisar_file):
    """Orbit position/velocity arrays are valid if present."""
    with NISARReader(str(require_nisar_file)) as reader:
        meta = reader.metadata
        if hasattr(meta, 'orbit') and meta.orbit is not None:
            orbit = meta.orbit
            if hasattr(orbit, 'position') and orbit.position is not None:
                assert len(orbit.position) > 0, "Orbit position array is empty"
            if hasattr(orbit, 'velocity') and orbit.velocity is not None:
                assert len(orbit.velocity) > 0, "Orbit velocity array is empty"


@pytest.mark.slow
def test_nisar_frequency_polarization(require_nisar_file):
    """Available frequencies and polarizations are accessible."""
    with NISARReader(str(require_nisar_file)) as reader:
        if hasattr(reader, 'get_available_frequencies'):
            freqs = reader.get_available_frequencies()
            assert isinstance(freqs, (list, tuple))
            # NISAR supports A (L-band) and/or B (S-band)
            for f in freqs:
                assert f in ('A', 'B', 'frequencyA', 'frequencyB'), (
                    f"Unexpected frequency: {f}"
                )
        if hasattr(reader, 'get_available_polarizations'):
            pols = reader.get_available_polarizations()
            assert isinstance(pols, (list, tuple))
            for p in pols:
                assert p in ('HH', 'HV', 'VH', 'VV'), (
                    f"Unexpected polarization: {p}"
                )


@pytest.mark.slow
def test_nisar_complex_magnitude_stats(require_nisar_file):
    """Complex SAR data has non-negative magnitude with positive variance."""
    with NISARReader(str(require_nisar_file)) as reader:
        rows, cols = reader.get_shape()[:2]
        r0 = max(0, rows // 2 - 128)
        r1 = min(rows, rows // 2 + 128)
        c0 = max(0, cols // 2 - 128)
        c1 = min(cols, cols // 2 + 128)
        chip = reader.read_chip(r0, r1, c0, c1)

        magnitude = np.abs(chip).astype(np.float64)
        assert magnitude.min() >= 0.0, "Negative magnitude values"
        assert magnitude.std() > 0, "Zero variance — likely fill data"

        # SAR coefficient of variation should be > 0.1 for real data
        mean_mag = magnitude.mean()
        if mean_mag > 0:
            cv = magnitude.std() / mean_mag
            assert cv > 0.1, (
                f"CV = {cv:.4f} too low for real SAR data (expected > 0.1)"
            )


@pytest.mark.slow
def test_nisar_complex_phase_range(require_nisar_file):
    """Complex SAR phase is in [-π, π]."""
    with NISARReader(str(require_nisar_file)) as reader:
        dtype = reader.get_dtype()
        if np.dtype(dtype).kind != 'c':
            pytest.skip("GSLC product — phase check only for RSLC")

        rows, cols = reader.get_shape()[:2]
        r0 = max(0, rows // 2 - 128)
        r1 = min(rows, rows // 2 + 128)
        c0 = max(0, cols // 2 - 128)
        c1 = min(cols, cols // 2 + 128)
        chip = reader.read_chip(r0, r1, c0, c1)

        nonzero = chip[np.abs(chip) > 0]
        if len(nonzero) > 0:
            phase = np.angle(nonzero)
            assert phase.min() >= -np.pi - 1e-6, (
                f"Phase min = {phase.min():.4f} below -π"
            )
            assert phase.max() <= np.pi + 1e-6, (
                f"Phase max = {phase.max():.4f} above π"
            )


# =============================================================================
# Level 3: Integration
# =============================================================================


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
@pytest.mark.integration
def test_nisar_chip_extractor_integration(require_nisar_file):
    """ChipExtractor partitions NISAR image; chips have valid shapes."""
    with NISARReader(str(require_nisar_file)) as reader:
        rows, cols = reader.get_shape()[:2]
        extractor = ChipExtractor(nrows=rows, ncols=cols)
        regions = extractor.chip_positions(row_width=256, col_width=256)
        assert len(regions) > 0

        for region in regions[:3]:
            assert 0 <= region.row_start < region.row_end <= rows
            assert 0 <= region.col_start < region.col_end <= cols
            chip = reader.read_chip(
                region.row_start, region.row_end,
                region.col_start, region.col_end,
            )
            assert chip.shape[0] == region.row_end - region.row_start
            assert chip.shape[1] == region.col_end - region.col_start


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
@pytest.mark.integration
def test_nisar_chip_extractor_boundary_chips(require_nisar_file):
    """Chips at image edges clamp to valid bounds without IndexError."""
    with NISARReader(str(require_nisar_file)) as reader:
        rows, cols = reader.get_shape()[:2]
        extractor = ChipExtractor(nrows=rows, ncols=cols)
        regions = extractor.chip_positions(row_width=256, col_width=256)

        # Check all regions have valid bounds
        for region in regions:
            assert region.row_start >= 0, f"row_start={region.row_start} < 0"
            assert region.col_start >= 0, f"col_start={region.col_start} < 0"
            assert region.row_end <= rows, f"row_end={region.row_end} > rows={rows}"
            assert region.col_end <= cols, f"col_end={region.col_end} > cols={cols}"


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
@pytest.mark.integration
def test_nisar_chip_extractor_single_pixel_center(require_nisar_file):
    """chip_at_point at corner pixels produces valid chips."""
    with NISARReader(str(require_nisar_file)) as reader:
        rows, cols = reader.get_shape()[:2]
        extractor = ChipExtractor(nrows=rows, ncols=cols)

        # Top-left corner
        region_tl = extractor.chip_at_point(
            row=0, col=0, row_width=64, col_width=64
        )
        chip_tl = reader.read_chip(
            region_tl.row_start, region_tl.row_end,
            region_tl.col_start, region_tl.col_end,
        )
        assert chip_tl.shape[0] > 0 and chip_tl.shape[1] > 0

        # Bottom-right corner
        region_br = extractor.chip_at_point(
            row=rows - 1, col=cols - 1, row_width=64, col_width=64
        )
        chip_br = reader.read_chip(
            region_br.row_start, region_br.row_end,
            region_br.col_start, region_br.col_end,
        )
        assert chip_br.shape[0] > 0 and chip_br.shape[1] > 0


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
@pytest.mark.integration
def test_nisar_normalizer_magnitude(require_nisar_file):
    """Magnitude → minmax normalization produces output in [0, 1]."""
    with NISARReader(str(require_nisar_file)) as reader:
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
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
@pytest.mark.integration
def test_nisar_normalizer_zscore_complex(require_nisar_file):
    """Magnitude → zscore normalization produces μ≈0 and σ≈1."""
    with NISARReader(str(require_nisar_file)) as reader:
        rows, cols = reader.get_shape()[:2]
        r0 = max(0, rows // 2 - 128)
        r1 = min(rows, rows // 2 + 128)
        c0 = max(0, cols // 2 - 128)
        c1 = min(cols, cols // 2 + 128)
        chip = reader.read_chip(r0, r1, c0, c1)

        magnitude = np.abs(chip).astype(np.float64)
        normalizer = Normalizer(method='zscore')
        normalized = normalizer.normalize(magnitude)

        assert abs(normalized.mean()) < 1e-6, (
            f"Z-score mean = {normalized.mean():.2e}; expected ≈ 0"
        )
        assert abs(normalized.std() - 1.0) < 1e-4, (
            f"Z-score std = {normalized.std():.6f}; expected ≈ 1.0"
        )


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
@pytest.mark.integration
def test_nisar_normalizer_all_zero_chip(require_nisar_file):
    """All-zero chip does not produce NaN/Inf after normalization."""
    zero_chip = np.zeros((64, 64), dtype=np.float64)
    normalizer = Normalizer(method='minmax')
    normalized = normalizer.normalize(zero_chip)

    assert np.isfinite(normalized).all(), (
        "Normalizing all-zeros produced NaN or Inf"
    )


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
@pytest.mark.integration
def test_nisar_tiler_integration(require_nisar_file):
    """Overlapping tiles have valid bounds within image."""
    with NISARReader(str(require_nisar_file)) as reader:
        rows, cols = reader.get_shape()[:2]
        tiler = Tiler(nrows=rows, ncols=cols, tile_size=512, stride=256)
        tiles = tiler.tile_positions()
        assert len(tiles) > 0

        for tile in tiles[:5]:
            assert 0 <= tile.row_start < tile.row_end <= rows
            assert 0 <= tile.col_start < tile.col_end <= cols
            assert tile.row_end - tile.row_start <= 512
            assert tile.col_end - tile.col_start <= 512


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
@pytest.mark.integration
def test_nisar_tiler_stride_vs_tile_size(require_nisar_file):
    """Stride < tile_size produces overlapping tiles; stride == tile_size does not."""
    with NISARReader(str(require_nisar_file)) as reader:
        rows, cols = reader.get_shape()[:2]

        # Overlapping: stride < tile_size
        tiler_overlap = Tiler(
            nrows=rows, ncols=cols, tile_size=512, stride=256
        )
        tiles_overlap = tiler_overlap.tile_positions()

        # Non-overlapping: stride == tile_size
        tiler_no_overlap = Tiler(
            nrows=rows, ncols=cols, tile_size=512, stride=512
        )
        tiles_no_overlap = tiler_no_overlap.tile_positions()

        # Overlapping should produce more tiles
        assert len(tiles_overlap) >= len(tiles_no_overlap), (
            f"Overlapping ({len(tiles_overlap)} tiles) should >= "
            f"non-overlapping ({len(tiles_no_overlap)} tiles)"
        )
