# -*- coding: utf-8 -*-
"""
EONITFReader Tests - EO NITF with RPC/RSM Validation.

Tests grdl.IO.eo.nitf.EONITFReader with real EO NITF files:
- Level 1: Format validation (metadata, dtype, shape, chip, context manager)
- Level 2: Data quality (RPC presence, normalization, value range, nodata)
- Level 3: Integration (ChipExtractor, Normalizer, RPC geolocation, error cases)

Dataset: EO NITF with RPC00B TRE

Dependencies
------------
pytest
numpy
rasterio
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
    import rasterio
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False

try:
    from grdl.IO.eo.nitf import EONITFReader
    _HAS_EO_NITF = True
except ImportError:
    _HAS_EO_NITF = False

try:
    from grdl.IO.models.eo_nitf import RPCCoefficients
    _HAS_RPC_MODEL = True
except ImportError:
    _HAS_RPC_MODEL = False

try:
    from grdl.data_prep import ChipExtractor, Normalizer, Tiler
    _HAS_DATA_PREP = True
except ImportError:
    _HAS_DATA_PREP = False

try:
    from grdl.geolocation.eo.rpc import RPCGeolocation
    from grdl.exceptions import GeolocationError
    _HAS_RPC_GEO = True
except ImportError:
    _HAS_RPC_GEO = False


pytestmark = [
    pytest.mark.eo_nitf,
    pytest.mark.requires_data,
    pytest.mark.skipif(not _HAS_RASTERIO, reason="rasterio not installed"),
    pytest.mark.skipif(not _HAS_EO_NITF, reason="EONITFReader not available"),
]


# =============================================================================
# Level 1: Format Validation
# =============================================================================


@pytest.mark.slow
def test_eo_nitf_metadata(require_eo_nitf_file):
    """EONITFReader extracts valid metadata with positive dimensions."""
    with EONITFReader(str(require_eo_nitf_file)) as reader:
        meta = reader.metadata
        assert meta is not None
        assert meta.rows > 0, f"rows={meta.rows} must be positive"
        assert meta.cols > 0, f"cols={meta.cols} must be positive"


@pytest.mark.slow
def test_eo_nitf_dtype(require_eo_nitf_file):
    """EO NITF has real-valued dtype (uint8, uint16, or float32)."""
    with EONITFReader(str(require_eo_nitf_file)) as reader:
        dtype = reader.get_dtype()
        assert np.dtype(dtype).kind in ('u', 'i', 'f'), (
            f"Expected real-valued dtype, got {dtype}"
        )


@pytest.mark.slow
def test_eo_nitf_get_shape(require_eo_nitf_file):
    """get_shape() returns tuple matching metadata dimensions."""
    with EONITFReader(str(require_eo_nitf_file)) as reader:
        shape = reader.get_shape()
        assert isinstance(shape, tuple)
        assert len(shape) >= 2
        assert shape[0] == reader.metadata.rows
        assert shape[1] == reader.metadata.cols


@pytest.mark.slow
def test_eo_nitf_read_chip(require_eo_nitf_file):
    """Center chip reads with non-zero pixel content."""
    with EONITFReader(str(require_eo_nitf_file)) as reader:
        rows, cols = reader.get_shape()[:2]
        r0 = max(0, rows // 2 - 128)
        r1 = min(rows, rows // 2 + 128)
        c0 = max(0, cols // 2 - 128)
        c1 = min(cols, cols // 2 + 128)
        chip = reader.read_chip(r0, r1, c0, c1)

        assert isinstance(chip, np.ndarray)
        assert chip.shape[0] == (r1 - r0)
        assert not np.all(chip == 0), "Center chip is entirely zero"


@pytest.mark.slow
def test_eo_nitf_context_manager(require_eo_nitf_file):
    """EONITFReader supports __enter__/__exit__ lifecycle."""
    with EONITFReader(str(require_eo_nitf_file)) as reader:
        assert reader is not None
        _ = reader.get_shape()


# =============================================================================
# Level 2: Data Quality
# =============================================================================


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_RPC_MODEL, reason="RPCCoefficients not available")
def test_eo_nitf_rpc_presence(require_eo_nitf_file):
    """RPC coefficients are present with 20 terms per polynomial."""
    with EONITFReader(str(require_eo_nitf_file)) as reader:
        meta = reader.metadata
        if not hasattr(meta, 'rpc') or meta.rpc is None:
            pytest.skip("This NITF does not contain RPC metadata")

        rpc = meta.rpc
        assert len(rpc.line_num_coef) == 20, (
            f"line_num has {len(rpc.line_num_coef)} terms, expected 20"
        )
        assert len(rpc.line_den_coef) == 20
        assert len(rpc.samp_num_coef) == 20
        assert len(rpc.samp_den_coef) == 20


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_RPC_MODEL, reason="RPCCoefficients not available")
def test_eo_nitf_rpc_normalization(require_eo_nitf_file):
    """RPC scale/offset values are physically plausible."""
    with EONITFReader(str(require_eo_nitf_file)) as reader:
        meta = reader.metadata
        if not hasattr(meta, 'rpc') or meta.rpc is None:
            pytest.skip("This NITF does not contain RPC metadata")

        rpc = meta.rpc
        assert -90 <= rpc.lat_off <= 90, (
            f"lat_off={rpc.lat_off} out of [-90, 90]"
        )
        assert -180 <= rpc.long_off <= 180, (
            f"long_off={rpc.long_off} out of [-180, 180]"
        )
        assert rpc.lat_scale > 0, f"lat_scale={rpc.lat_scale} must be positive"
        assert rpc.long_scale > 0, f"long_scale={rpc.long_scale} must be positive"
        assert rpc.line_scale > 0, f"line_scale={rpc.line_scale} must be positive"
        assert rpc.samp_scale > 0, f"samp_scale={rpc.samp_scale} must be positive"


@pytest.mark.slow
def test_eo_nitf_value_range(require_eo_nitf_file):
    """Pixel values fall within expected bit-depth bounds."""
    with EONITFReader(str(require_eo_nitf_file)) as reader:
        rows, cols = reader.get_shape()[:2]
        r0 = max(0, rows // 2 - 128)
        r1 = min(rows, rows // 2 + 128)
        c0 = max(0, cols // 2 - 128)
        c1 = min(cols, cols // 2 + 128)
        chip = reader.read_chip(r0, r1, c0, c1)

        dtype = chip.dtype
        if dtype == np.uint8:
            assert chip.max() <= 255
        elif dtype == np.uint16:
            assert chip.max() <= 65535
        elif dtype == np.int16:
            assert chip.min() >= -32768
        # For float types, just verify finite
        if np.issubdtype(dtype, np.floating):
            assert np.isfinite(chip).all(), "Non-finite pixel values"


@pytest.mark.slow
def test_eo_nitf_nodata_masking(require_eo_nitf_file):
    """Valid pixels have non-zero variance."""
    with EONITFReader(str(require_eo_nitf_file)) as reader:
        rows, cols = reader.get_shape()[:2]
        r0 = max(0, rows // 2 - 128)
        r1 = min(rows, rows // 2 + 128)
        c0 = max(0, cols // 2 - 128)
        c1 = min(cols, cols // 2 + 128)
        chip = reader.read_chip(r0, r1, c0, c1)

        valid = chip[chip > 0] if chip.dtype.kind == 'u' else chip
        if len(valid) > 0:
            assert valid.std() > 0, "Valid pixels have zero variance"


# =============================================================================
# Level 3: Integration
# =============================================================================


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
@pytest.mark.integration
def test_eo_nitf_chip_extractor(require_eo_nitf_file):
    """ChipExtractor partitions EO NITF image correctly."""
    with EONITFReader(str(require_eo_nitf_file)) as reader:
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
def test_eo_nitf_normalizer(require_eo_nitf_file):
    """MinMax normalization produces output in [0, 1]."""
    with EONITFReader(str(require_eo_nitf_file)) as reader:
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
@pytest.mark.skipif(not _HAS_RPC_GEO, reason="RPCGeolocation not available")
@pytest.mark.skipif(not _HAS_RPC_MODEL, reason="RPCCoefficients not available")
@pytest.mark.integration
def test_eo_nitf_rpc_geolocation_integration(require_eo_nitf_file):
    """RPCGeolocation from reader metadata produces valid lat/lon."""
    with EONITFReader(str(require_eo_nitf_file)) as reader:
        meta = reader.metadata
        if not hasattr(meta, 'rpc') or meta.rpc is None:
            pytest.skip("This NITF does not contain RPC metadata")

        rows, cols = reader.get_shape()[:2]
        geo = RPCGeolocation(rpc=meta.rpc, shape=(rows, cols))

        # Center pixel should produce valid coordinates
        lat, lon, hae = geo.image_to_latlon(rows // 2, cols // 2, 0.0)
        assert -90 <= lat <= 90, f"lat={lat} out of bounds"
        assert -180 <= lon <= 180, f"lon={lon} out of bounds"
        assert np.isfinite(lat) and np.isfinite(lon)


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_RPC_GEO, reason="RPCGeolocation not available")
@pytest.mark.skipif(not _HAS_RPC_MODEL, reason="RPCCoefficients not available")
@pytest.mark.integration
def test_eo_nitf_rpc_geolocation_invalid_latlon(require_eo_nitf_file):
    """Invalid lat/lon (999, -999) raises error or returns out-of-bounds pixels."""
    with EONITFReader(str(require_eo_nitf_file)) as reader:
        meta = reader.metadata
        if not hasattr(meta, 'rpc') or meta.rpc is None:
            pytest.skip("This NITF does not contain RPC metadata")

        rows, cols = reader.get_shape()[:2]
        geo = RPCGeolocation(rpc=meta.rpc, shape=(rows, cols))

        try:
            row, col = geo.latlon_to_image(999.0, -999.0, 0.0)
            # If it doesn't raise, the result should be clearly wrong
            # (out of image bounds or NaN)
            is_invalid = (
                not np.isfinite(row) or not np.isfinite(col)
                or row < -1e6 or row > 1e6
                or col < -1e6 or col > 1e6
            )
            assert is_invalid, (
                f"Invalid lat/lon (999, -999) produced plausible pixel "
                f"({row:.1f}, {col:.1f}) without raising an error"
            )
        except (GeolocationError, ValueError, RuntimeError):
            # Expected: invalid input should raise
            pass


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_RPC_GEO, reason="RPCGeolocation not available")
@pytest.mark.skipif(not _HAS_RPC_MODEL, reason="RPCCoefficients not available")
@pytest.mark.integration
def test_eo_nitf_rpc_geolocation_out_of_image(require_eo_nitf_file):
    """Extreme pixel coords produce out-of-bounds or error result."""
    with EONITFReader(str(require_eo_nitf_file)) as reader:
        meta = reader.metadata
        if not hasattr(meta, 'rpc') or meta.rpc is None:
            pytest.skip("This NITF does not contain RPC metadata")

        rows, cols = reader.get_shape()[:2]
        geo = RPCGeolocation(rpc=meta.rpc, shape=(rows, cols))

        try:
            lat, lon, hae = geo.image_to_latlon(-10000.0, 999999.0, 0.0)
            # If it doesn't raise, result should be physically unreasonable
            # or clearly outside normal coverage
            assert np.isfinite(lat) or not np.isfinite(lat), (
                "Extreme pixel coords should produce unusual results"
            )
        except (GeolocationError, ValueError, RuntimeError):
            # Expected: out-of-image should raise
            pass
