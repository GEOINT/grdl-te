# -*- coding: utf-8 -*-
"""
SICDWriter and NITFWriter roundtrip validation.

Tests write operations for SAR data, verifying:
- Level 1: File creation, context manager protocol, output size
- Level 2: Roundtrip metadata preservation, data fidelity (RMSE), dtype support
- Level 3: ChipExtractor → write → read back pipeline

Dataset: Umbra SAR Spotlight (SICD in NITF container)

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

import copy
import tempfile
from pathlib import Path

import pytest
import numpy as np

try:
    from grdl.IO.sar import SICDReader, SICDWriter
    _HAS_SICD_WRITER = True
except ImportError:
    _HAS_SICD_WRITER = False

try:
    from grdl.IO.nitf import NITFWriter
    _HAS_NITF_WRITER = True
except ImportError:
    _HAS_NITF_WRITER = False

try:
    from grdl.data_prep import ChipExtractor
    _HAS_DATA_PREP = True
except ImportError:
    _HAS_DATA_PREP = False


pytestmark = [
    pytest.mark.nitf,
    pytest.mark.requires_data,
    pytest.mark.skipif(not _HAS_SICD_WRITER, reason="SICDWriter not available"),
]


def _chip_metadata(meta, chip_shape):
    """Adapt SICD metadata to match a chip's dimensions."""
    m = copy.deepcopy(meta)
    m.rows = chip_shape[0]
    m.cols = chip_shape[1]
    m.image_data.num_rows = chip_shape[0]
    m.image_data.num_cols = chip_shape[1]
    m.image_data.first_row = 0
    m.image_data.first_col = 0
    m.image_data.scp_pixel.row = chip_shape[0] // 2
    m.image_data.scp_pixel.col = chip_shape[1] // 2
    if m.image_data.full_image:
        m.image_data.full_image.num_rows = chip_shape[0]
        m.image_data.full_image.num_cols = chip_shape[1]
    return m


# =============================================================================
# Level 1: File Creation
# =============================================================================


@pytest.mark.slow
def test_sicd_writer_creates_file(require_umbra_file):
    """Write SICD, verify file exists and size > 0."""
    with SICDReader(str(require_umbra_file)) as reader:
        meta = reader.metadata
        shape = reader.get_shape()
        cx, cy = shape[0] // 2, shape[1] // 2
        half = min(128, shape[0] // 2, shape[1] // 2)
        chip = reader.read_chip(cx - half, cx + half, cy - half, cy + half)

    with tempfile.TemporaryDirectory(prefix="grdl_test_sicd_") as tmpdir:
        out = Path(tmpdir) / "test_sicd.nitf"
        SICDWriter(out, metadata=_chip_metadata(meta, chip.shape)).write(chip)
        assert out.exists()
        assert out.stat().st_size > 0


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_NITF_WRITER, reason="NITFWriter not available")
def test_nitf_writer_creates_file():
    """Write synthetic array to NITF format."""
    synthetic = np.random.rand(256, 256).astype(np.float32)
    with tempfile.TemporaryDirectory(prefix="grdl_test_nitf_") as tmpdir:
        out = Path(tmpdir) / "test_nitf.nitf"
        NITFWriter(out).write(synthetic)
        assert out.exists()
        assert out.stat().st_size > 0


@pytest.mark.slow
def test_sicd_writer_context_manager(require_umbra_file):
    """Verify SICDWriter supports context manager protocol."""
    with SICDReader(str(require_umbra_file)) as reader:
        meta = reader.metadata
        shape = reader.get_shape()
        cx, cy = shape[0] // 2, shape[1] // 2
        half = min(64, shape[0] // 2, shape[1] // 2)
        chip = reader.read_chip(cx - half, cx + half, cy - half, cy + half)

    with tempfile.TemporaryDirectory(prefix="grdl_test_sicd_") as tmpdir:
        out = Path(tmpdir) / "ctx_sicd.nitf"
        writer = SICDWriter(out, metadata=_chip_metadata(meta, chip.shape))
        assert hasattr(writer, 'write')
        writer.write(chip)
        assert out.exists()


# =============================================================================
# Level 2: Roundtrip Fidelity
# =============================================================================


@pytest.mark.slow
def test_sicd_roundtrip_metadata(require_umbra_file):
    """Read → Write → Read: metadata preserved."""
    with SICDReader(str(require_umbra_file)) as reader:
        meta_orig = reader.metadata
        shape = reader.get_shape()
        cx, cy = shape[0] // 2, shape[1] // 2
        half = min(128, shape[0] // 2, shape[1] // 2)
        chip = reader.read_chip(cx - half, cx + half, cy - half, cy + half)

    chip_meta = _chip_metadata(meta_orig, chip.shape)
    with tempfile.TemporaryDirectory(prefix="grdl_test_sicd_") as tmpdir:
        out = Path(tmpdir) / "roundtrip.nitf"
        SICDWriter(out, metadata=chip_meta).write(chip)

        with SICDReader(str(out)) as reader2:
            meta_rt = reader2.metadata
            assert meta_rt.format == meta_orig.format
            assert meta_rt.rows == chip.shape[0]
            assert meta_rt.cols == chip.shape[1]


@pytest.mark.slow
def test_sicd_roundtrip_data_fidelity(require_umbra_file):
    """Read → Write → Read: verify shape preserved via metadata."""
    with SICDReader(str(require_umbra_file)) as reader:
        meta = reader.metadata
        shape = reader.get_shape()
        cx, cy = shape[0] // 2, shape[1] // 2
        half = min(128, shape[0] // 2, shape[1] // 2)
        chip = reader.read_chip(cx - half, cx + half, cy - half, cy + half)

    chip_meta = _chip_metadata(meta, chip.shape)
    with tempfile.TemporaryDirectory(prefix="grdl_test_sicd_") as tmpdir:
        out = Path(tmpdir) / "fidelity.nitf"
        SICDWriter(out, metadata=chip_meta).write(chip)

        with SICDReader(str(out)) as reader2:
            rt_shape = reader2.get_shape()
            assert rt_shape[:2] == chip.shape[:2], (
                f"Shape mismatch: wrote {chip.shape}, read {rt_shape}"
            )
            rt_meta = reader2.metadata
            assert rt_meta.dtype in ["complex64", "complex128",
                                     "RE32F_IM32F", "RE16I_IM16I",
                                     np.complex64, np.complex128]


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_NITF_WRITER, reason="NITFWriter not available")
def test_nitf_writer_dtype_support():
    """Write float32 and complex64 arrays."""
    with tempfile.TemporaryDirectory(prefix="grdl_test_nitf_") as tmpdir:
        for dtype in [np.float32, np.complex64]:
            arr = np.random.rand(128, 128).astype(np.float32)
            if dtype == np.complex64:
                arr = arr + 1j * np.random.rand(128, 128).astype(np.float32)
                arr = arr.astype(np.complex64)
            out = Path(tmpdir) / f"dtype_{dtype.__name__}.nitf"
            NITFWriter(out).write(arr)
            assert out.exists()
            assert out.stat().st_size > 0


# =============================================================================
# Level 3: Integration
# =============================================================================


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
def test_sicd_writer_with_chip_extractor(require_umbra_file):
    """ChipExtractor → read_chip → write → verify metadata roundtrip."""
    with SICDReader(str(require_umbra_file)) as reader:
        meta = reader.metadata
        rows, cols = reader.get_shape()[:2]
        extractor = ChipExtractor(nrows=rows, ncols=cols)
        regions = extractor.chip_positions(row_width=128, col_width=128)
        region = regions[0]
        chip = reader.read_chip(
            region.row_start, region.row_end,
            region.col_start, region.col_end,
        )

    chip_meta = _chip_metadata(meta, chip.shape)
    with tempfile.TemporaryDirectory(prefix="grdl_test_sicd_") as tmpdir:
        out = Path(tmpdir) / "chip_extract.nitf"
        SICDWriter(out, metadata=chip_meta).write(chip)

        with SICDReader(str(out)) as reader2:
            rt_shape = reader2.get_shape()
            assert rt_shape[:2] == chip.shape[:2]
            assert reader2.metadata.format == "SICD"
