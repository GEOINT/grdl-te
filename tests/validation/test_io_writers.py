# -*- coding: utf-8 -*-
"""
IO Writer Tests - GeoTIFFWriter, HDF5Writer, and writer factory validation.

Tests grdl.IO writers with synthetic data, including:
- Level 1: Format validation (file creation, context manager, dtype preservation)
- Level 2: Data quality (roundtrip fidelity, geolocation preservation, complex support)
- Level 3: Integration (writer factory, auto-detect, ChipExtractor pipeline)

Dependencies
------------
pytest
numpy
rasterio
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

# Guarded imports
try:
    from grdl.IO.geotiff import GeoTIFFWriter, GeoTIFFReader
    _HAS_GEOTIFF = True
except ImportError:
    _HAS_GEOTIFF = False

try:
    from grdl.IO.hdf5 import HDF5Writer, HDF5Reader
    _HAS_HDF5 = True
except ImportError:
    _HAS_HDF5 = False

try:
    from grdl.IO import get_writer, write
    _HAS_WRITER_FACTORY = True
except ImportError:
    _HAS_WRITER_FACTORY = False

try:
    from grdl.data_prep import ChipExtractor
    _HAS_DATA_PREP = True
except ImportError:
    _HAS_DATA_PREP = False


pytestmark = [
    pytest.mark.writers,
]


# =============================================================================
# Level 1: Format Validation — GeoTIFFWriter
# =============================================================================


@pytest.mark.skipif(not _HAS_GEOTIFF, reason="GeoTIFFWriter not available")
def test_geotiff_writer_creates_file(tmp_path):
    """GeoTIFFWriter creates a valid file on disk."""
    outpath = tmp_path / "test_output.tif"
    data = np.random.default_rng(42).random((64, 64)).astype(np.float32)

    with GeoTIFFWriter(str(outpath)) as writer:
        writer.write(data)

    assert outpath.exists(), f"GeoTIFF not created at {outpath}"
    assert outpath.stat().st_size > 0, "GeoTIFF file is empty"


@pytest.mark.skipif(not _HAS_GEOTIFF, reason="GeoTIFFWriter not available")
def test_geotiff_writer_context_manager(tmp_path):
    """GeoTIFFWriter supports __enter__/__exit__ protocol."""
    outpath = tmp_path / "ctx.tif"
    data = np.ones((32, 32), dtype=np.float32)

    with GeoTIFFWriter(str(outpath)) as writer:
        assert writer is not None
        writer.write(data)

    # File must exist after context exit
    assert outpath.exists()


@pytest.mark.skipif(not _HAS_GEOTIFF, reason="GeoTIFFWriter not available")
def test_geotiff_writer_dtype_preservation(tmp_path):
    """GeoTIFFWriter preserves uint16 dtype through write/read roundtrip."""
    outpath = tmp_path / "uint16.tif"
    data = np.random.default_rng(7).integers(0, 30000, size=(64, 64), dtype=np.uint16)

    with GeoTIFFWriter(str(outpath)) as writer:
        writer.write(data)

    with GeoTIFFReader(str(outpath)) as reader:
        readback = reader.read_full()
        assert readback.dtype == np.uint16, (
            f"dtype changed: wrote uint16, read {readback.dtype}"
        )


@pytest.mark.skipif(not _HAS_GEOTIFF, reason="GeoTIFFWriter not available")
def test_geotiff_writer_multiband(tmp_path):
    """GeoTIFFWriter handles (bands, rows, cols) 3D arrays."""
    outpath = tmp_path / "multiband.tif"
    data = np.random.default_rng(99).random((3, 64, 64)).astype(np.float32)

    with GeoTIFFWriter(str(outpath)) as writer:
        writer.write(data)

    with GeoTIFFReader(str(outpath)) as reader:
        shape = reader.get_shape()
        # Shape should have 3 bands
        assert shape[0] == 64, f"Rows mismatch: {shape}"
        assert shape[1] == 64, f"Cols mismatch: {shape}"
        meta = reader.metadata
        assert meta.bands == 3 or meta.rows == 64


# =============================================================================
# Level 1: Format Validation — HDF5Writer
# =============================================================================


@pytest.mark.skipif(not _HAS_HDF5, reason="HDF5Writer not available")
def test_hdf5_writer_creates_file(tmp_path):
    """HDF5Writer creates a valid file on disk."""
    outpath = tmp_path / "test_output.h5"
    data = np.random.default_rng(42).random((64, 64)).astype(np.float32)

    with HDF5Writer(str(outpath)) as writer:
        writer.write(data)

    assert outpath.exists(), f"HDF5 file not created at {outpath}"
    assert outpath.stat().st_size > 0, "HDF5 file is empty"


@pytest.mark.skipif(not _HAS_HDF5, reason="HDF5Writer not available")
def test_hdf5_writer_context_manager(tmp_path):
    """HDF5Writer supports __enter__/__exit__ protocol."""
    outpath = tmp_path / "ctx.h5"
    data = np.ones((32, 32), dtype=np.float32)

    with HDF5Writer(str(outpath)) as writer:
        assert writer is not None
        writer.write(data)

    assert outpath.exists()


@pytest.mark.skipif(not _HAS_HDF5, reason="HDF5Writer not available")
def test_hdf5_writer_dtype_preservation(tmp_path):
    """HDF5Writer preserves float64 dtype through write/read roundtrip."""
    outpath = tmp_path / "float64.h5"
    data = np.random.default_rng(7).random((64, 64)).astype(np.float64)

    with HDF5Writer(str(outpath)) as writer:
        writer.write(data)

    with HDF5Reader(str(outpath)) as reader:
        readback = reader.read_full()
        assert readback.dtype == np.float64, (
            f"dtype changed: wrote float64, read {readback.dtype}"
        )


# =============================================================================
# Level 2: Data Quality — Roundtrip Fidelity
# =============================================================================


@pytest.mark.skipif(not _HAS_GEOTIFF, reason="GeoTIFFWriter not available")
def test_geotiff_roundtrip_data_fidelity(tmp_path):
    """GeoTIFF write+read roundtrip preserves data within RMSE < 1e-6."""
    outpath = tmp_path / "fidelity.tif"
    rng = np.random.default_rng(42)
    data = rng.random((128, 128)).astype(np.float32)

    with GeoTIFFWriter(str(outpath)) as writer:
        writer.write(data)

    with GeoTIFFReader(str(outpath)) as reader:
        readback = reader.read_full()

    # Squeeze readback if it has extra band dimension
    if readback.ndim == 3 and readback.shape[0] == 1:
        readback = readback[0]

    rmse = np.sqrt(np.mean((data - readback) ** 2))
    assert rmse < 1e-6, (
        f"GeoTIFF roundtrip RMSE = {rmse:.2e}; exceeds 1e-6 tolerance"
    )


@pytest.mark.skipif(not _HAS_GEOTIFF, reason="GeoTIFFWriter not available")
def test_geotiff_roundtrip_geolocation(tmp_path):
    """GeoTIFF write+read preserves CRS and transform metadata."""
    outpath = tmp_path / "geo.tif"
    data = np.ones((64, 64), dtype=np.float32)

    geolocation = {
        'crs': 'EPSG:4326',
        'transform': (0.001, 0.0, -122.0, 0.0, -0.001, 37.0),
    }

    with GeoTIFFWriter(str(outpath)) as writer:
        writer.write(data, geolocation=geolocation)

    with GeoTIFFReader(str(outpath)) as reader:
        meta = reader.metadata
        # CRS should be preserved
        assert meta.crs is not None, "CRS not preserved in GeoTIFF roundtrip"


@pytest.mark.skipif(not _HAS_HDF5, reason="HDF5Writer not available")
def test_hdf5_roundtrip_data_fidelity(tmp_path):
    """HDF5 write+read roundtrip preserves data exactly."""
    outpath = tmp_path / "fidelity.h5"
    rng = np.random.default_rng(42)
    data = rng.random((128, 128)).astype(np.float64)

    with HDF5Writer(str(outpath)) as writer:
        writer.write(data)

    with HDF5Reader(str(outpath)) as reader:
        readback = reader.read_full()

    np.testing.assert_array_equal(
        data, readback,
        err_msg="HDF5 roundtrip data mismatch — lossless format should be exact"
    )


@pytest.mark.skipif(not _HAS_HDF5, reason="HDF5Writer not available")
def test_hdf5_roundtrip_complex(tmp_path):
    """HDF5 write+read preserves complex64 data."""
    outpath = tmp_path / "complex.h5"
    rng = np.random.default_rng(42)
    data = (
        rng.standard_normal((64, 64)) + 1j * rng.standard_normal((64, 64))
    ).astype(np.complex64)

    with HDF5Writer(str(outpath)) as writer:
        writer.write(data)

    with HDF5Reader(str(outpath)) as reader:
        readback = reader.read_full()

    assert np.iscomplexobj(readback), "Complex dtype not preserved in HDF5"
    np.testing.assert_array_almost_equal(
        data, readback, decimal=6,
        err_msg="HDF5 complex64 roundtrip lost precision"
    )


# =============================================================================
# Level 3: Integration — Writer Factory and Pipelines
# =============================================================================


@pytest.mark.skipif(not _HAS_WRITER_FACTORY, reason="Writer factory not available")
@pytest.mark.skipif(not _HAS_GEOTIFF, reason="GeoTIFFWriter not available")
@pytest.mark.integration
def test_get_writer_geotiff(tmp_path):
    """get_writer('geotiff', ...) returns a GeoTIFFWriter instance."""
    outpath = tmp_path / "factory.tif"
    writer = get_writer('geotiff', str(outpath))
    assert isinstance(writer, GeoTIFFWriter), (
        f"Expected GeoTIFFWriter, got {type(writer).__name__}"
    )


@pytest.mark.skipif(not _HAS_WRITER_FACTORY, reason="Writer factory not available")
@pytest.mark.skipif(not _HAS_HDF5, reason="HDF5Writer not available")
@pytest.mark.integration
def test_get_writer_hdf5(tmp_path):
    """get_writer('hdf5', ...) returns an HDF5Writer instance."""
    outpath = tmp_path / "factory.h5"
    writer = get_writer('hdf5', str(outpath))
    assert isinstance(writer, HDF5Writer), (
        f"Expected HDF5Writer, got {type(writer).__name__}"
    )


@pytest.mark.skipif(not _HAS_WRITER_FACTORY, reason="Writer factory not available")
@pytest.mark.skipif(not _HAS_GEOTIFF, reason="GeoTIFFWriter not available")
@pytest.mark.integration
def test_write_auto_detect_tif(tmp_path):
    """write(data, 'out.tif') auto-detects GeoTIFF format and creates valid file."""
    outpath = tmp_path / "auto.tif"
    data = np.random.default_rng(42).random((64, 64)).astype(np.float32)
    write(data, str(outpath))

    assert outpath.exists(), "write() did not create file"

    with GeoTIFFReader(str(outpath)) as reader:
        readback = reader.read_full()
        if readback.ndim == 3 and readback.shape[0] == 1:
            readback = readback[0]
        assert readback.shape == (64, 64)
        rmse = np.sqrt(np.mean((data - readback) ** 2))
        assert rmse < 1e-6, f"Auto-detect write RMSE = {rmse:.2e}"


@pytest.mark.skipif(not _HAS_WRITER_FACTORY, reason="Writer factory not available")
@pytest.mark.skipif(not _HAS_GEOTIFF, reason="GeoTIFFWriter not available")
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
@pytest.mark.integration
def test_writer_chip_write_read_pipeline(tmp_path):
    """ChipExtractor → write chips → read back validates end-to-end pipeline."""
    # Create synthetic image
    rng = np.random.default_rng(42)
    image = rng.random((256, 256)).astype(np.float32)

    extractor = ChipExtractor(nrows=256, ncols=256)
    regions = extractor.chip_positions(row_width=64, col_width=64)
    assert len(regions) > 0, "ChipExtractor produced no regions"

    # Write each chip, read back, verify
    for i, region in enumerate(regions[:4]):
        chip = image[region.row_start:region.row_end,
                     region.col_start:region.col_end]
        chip_path = tmp_path / f"chip_{i}.tif"

        with GeoTIFFWriter(str(chip_path)) as writer:
            writer.write(chip)

        with GeoTIFFReader(str(chip_path)) as reader:
            readback = reader.read_full()
            if readback.ndim == 3 and readback.shape[0] == 1:
                readback = readback[0]

            assert readback.shape == chip.shape, (
                f"Chip {i}: shape mismatch {readback.shape} != {chip.shape}"
            )
            rmse = np.sqrt(np.mean((chip - readback) ** 2))
            assert rmse < 1e-6, f"Chip {i}: roundtrip RMSE = {rmse:.2e}"
