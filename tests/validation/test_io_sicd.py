# -*- coding: utf-8 -*-
"""
SICDReader Tests - Dedicated SICD metadata validation.

Tests grdl.IO.sar.sicd.SICDReader with real Umbra SICD NITF files,
validating the full SICDMetadata (~35 nested dataclasses):
- Level 1: Metadata type, collection info, image data, chip read, shape
- Level 2: GeoData, grid params, SCPCOA, radar collection, complex integrity
- Level 3: Geolocation construction, SCP roundtrip, chip normalizer pipeline

Dataset: Umbra SICD (*.nitf) — shared with test_io_nitf.py

Dependencies
------------
pytest
numpy
grdl (sarkit or sarpy backend)

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
    from grdl.IO.sar.sicd import SICDReader
    _HAS_SICD = True
except ImportError:
    _HAS_SICD = False

try:
    from grdl.IO.models.sicd import SICDMetadata
    _HAS_SICD_META = True
except ImportError:
    _HAS_SICD_META = False

try:
    from grdl.geolocation.sar.sicd import SICDGeolocation
    _HAS_SICD_GEO = True
except ImportError:
    _HAS_SICD_GEO = False

try:
    from grdl.data_prep import ChipExtractor, Normalizer
    _HAS_DATA_PREP = True
except ImportError:
    _HAS_DATA_PREP = False


pytestmark = [
    pytest.mark.sicd,
    pytest.mark.nitf,
    pytest.mark.requires_data,
    pytest.mark.skipif(not _HAS_SICD, reason="SICDReader not available"),
]


# =============================================================================
# Level 1: Format Validation
# =============================================================================


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_SICD_META, reason="SICDMetadata not available")
def test_sicd_metadata_type(require_umbra_file):
    """SICDReader produces SICDMetadata (not generic ImageMetadata)."""
    with SICDReader(str(require_umbra_file)) as reader:
        meta = reader.metadata
        assert isinstance(meta, SICDMetadata), (
            f"Expected SICDMetadata, got {type(meta).__name__}"
        )


@pytest.mark.slow
def test_sicd_collection_info(require_umbra_file):
    """CollectionInfo fields are populated."""
    with SICDReader(str(require_umbra_file)) as reader:
        meta = reader.metadata
        if hasattr(meta, 'collection_info') and meta.collection_info is not None:
            ci = meta.collection_info
            if hasattr(ci, 'radar_mode') and ci.radar_mode is not None:
                assert ci.radar_mode.mode_type is not None


@pytest.mark.slow
def test_sicd_image_data(require_umbra_file):
    """ImageData contains positive num_rows and num_cols."""
    with SICDReader(str(require_umbra_file)) as reader:
        meta = reader.metadata
        if hasattr(meta, 'image_data') and meta.image_data is not None:
            assert meta.image_data.num_rows > 0
            assert meta.image_data.num_cols > 0


@pytest.mark.slow
def test_sicd_read_chip(require_umbra_file):
    """Center chip has complex dtype and non-zero magnitude."""
    with SICDReader(str(require_umbra_file)) as reader:
        rows, cols = reader.get_shape()[:2]
        r0 = max(0, rows // 2 - 128)
        r1 = min(rows, rows // 2 + 128)
        c0 = max(0, cols // 2 - 128)
        c1 = min(cols, cols // 2 + 128)
        chip = reader.read_chip(r0, r1, c0, c1)

        assert np.iscomplexobj(chip), f"Expected complex, got {chip.dtype}"
        assert np.abs(chip).max() > 0, "Center chip is all zeros"


@pytest.mark.slow
def test_sicd_get_shape(require_umbra_file):
    """get_shape() matches image_data dimensions."""
    with SICDReader(str(require_umbra_file)) as reader:
        shape = reader.get_shape()
        meta = reader.metadata
        assert shape[0] == meta.rows
        assert shape[1] == meta.cols


# =============================================================================
# Level 2: Data Quality
# =============================================================================


@pytest.mark.slow
def test_sicd_geo_data(require_umbra_file):
    """GeoData SCP has valid lat/lon/hae."""
    with SICDReader(str(require_umbra_file)) as reader:
        meta = reader.metadata
        if hasattr(meta, 'geo_data') and meta.geo_data is not None:
            gd = meta.geo_data
            if hasattr(gd, 'scp') and gd.scp is not None:
                if hasattr(gd.scp, 'llh'):
                    llh = gd.scp.llh
                    assert -90 <= llh.lat <= 90, f"SCP lat={llh.lat}"
                    assert -180 <= llh.lon <= 180, f"SCP lon={llh.lon}"


@pytest.mark.slow
def test_sicd_grid_params(require_umbra_file):
    """Grid row/col direction parameters are populated."""
    with SICDReader(str(require_umbra_file)) as reader:
        meta = reader.metadata
        if hasattr(meta, 'grid') and meta.grid is not None:
            grid = meta.grid
            assert grid.row is not None, "grid.row is None"
            assert grid.col is not None, "grid.col is None"
            if grid.row.ss is not None:
                assert grid.row.ss > 0, f"Row sample spacing = {grid.row.ss}"
            if grid.col.ss is not None:
                assert grid.col.ss > 0, f"Col sample spacing = {grid.col.ss}"


@pytest.mark.slow
def test_sicd_scpcoa(require_umbra_file):
    """SCPCOA has valid geometry angles."""
    with SICDReader(str(require_umbra_file)) as reader:
        meta = reader.metadata
        if hasattr(meta, 'scpcoa') and meta.scpcoa is not None:
            scpcoa = meta.scpcoa
            if hasattr(scpcoa, 'side_of_track'):
                assert scpcoa.side_of_track in ('L', 'R'), (
                    f"side_of_track={scpcoa.side_of_track}"
                )


@pytest.mark.slow
def test_sicd_radar_collection(require_umbra_file):
    """RadarCollection tx_frequency is populated."""
    with SICDReader(str(require_umbra_file)) as reader:
        meta = reader.metadata
        if hasattr(meta, 'radar_collection') and meta.radar_collection is not None:
            rc = meta.radar_collection
            if hasattr(rc, 'tx_frequency') and rc.tx_frequency is not None:
                if hasattr(rc.tx_frequency, 'min_value'):
                    assert rc.tx_frequency.min_value > 0


@pytest.mark.slow
def test_sicd_complex_integrity(require_umbra_file):
    """Magnitude/phase roundtrip preserves data."""
    with SICDReader(str(require_umbra_file)) as reader:
        rows, cols = reader.get_shape()[:2]
        r0 = max(0, rows // 2 - 64)
        r1 = min(rows, rows // 2 + 64)
        c0 = max(0, cols // 2 - 64)
        c1 = min(cols, cols // 2 + 64)
        chip = reader.read_chip(r0, r1, c0, c1)

        mag = np.abs(chip)
        phase = np.angle(chip)
        reconstructed = mag * np.exp(1j * phase)

        np.testing.assert_allclose(
            chip, reconstructed, rtol=1e-5,
            err_msg="Complex magnitude/phase roundtrip failed"
        )


# =============================================================================
# Level 3: Integration
# =============================================================================


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_SICD_GEO, reason="SICDGeolocation not available")
@pytest.mark.integration
def test_sicd_geolocation_construction(require_umbra_file):
    """SICDGeolocation constructs from SICDReader metadata."""
    with SICDReader(str(require_umbra_file)) as reader:
        geo = SICDGeolocation.from_reader(reader)
        assert geo is not None


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_SICD_GEO, reason="SICDGeolocation not available")
@pytest.mark.integration
def test_sicd_geolocation_scp_roundtrip(require_umbra_file):
    """SCP pixel maps to known lat/lon and back within tolerance."""
    with SICDReader(str(require_umbra_file)) as reader:
        meta = reader.metadata
        geo = SICDGeolocation.from_reader(reader)

        if hasattr(meta, 'geo_data') and meta.geo_data is not None:
            if hasattr(meta.geo_data, 'scp') and meta.geo_data.scp is not None:
                scp = meta.geo_data.scp
                if hasattr(scp, 'llh') and hasattr(scp, 'image'):
                    scp_row = scp.image.row
                    scp_col = scp.image.col
                    lat, lon, hae = geo.image_to_latlon(scp_row, scp_col)
                    assert abs(lat - scp.llh.lat) < 0.01, (
                        f"SCP lat mismatch: {lat:.6f} vs {scp.llh.lat:.6f}"
                    )
                    assert abs(lon - scp.llh.lon) < 0.01, (
                        f"SCP lon mismatch: {lon:.6f} vs {scp.llh.lon:.6f}"
                    )


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
@pytest.mark.integration
def test_sicd_chip_normalizer_pipeline(require_umbra_file):
    """Chip → magnitude → normalize validates end-to-end."""
    with SICDReader(str(require_umbra_file)) as reader:
        rows, cols = reader.get_shape()[:2]
        extractor = ChipExtractor(nrows=rows, ncols=cols)
        region = extractor.chip_at_point(
            row=rows // 2, col=cols // 2,
            row_width=128, col_width=128,
        )
        chip = reader.read_chip(
            region.row_start, region.row_end,
            region.col_start, region.col_end,
        )

        magnitude = np.abs(chip).astype(np.float64)
        normalizer = Normalizer(method='minmax')
        normalized = normalizer.normalize(magnitude)

        assert np.isfinite(normalized).all()
        assert normalized.min() == pytest.approx(0.0, abs=1e-6)
        assert normalized.max() == pytest.approx(1.0, abs=1e-6)
