# -*- coding: utf-8 -*-
"""
NumPy and PNG Writer Validation - File creation, roundtrip, and edge cases.

Tests NumpyWriter (.npy/.npz with JSON sidecar) and PngWriter (grayscale
and RGB via Pillow). All tests use tempfile.TemporaryDirectory for
filesystem isolation.

- Level 1: File creation, context manager, write_chip raises
- Level 2: Roundtrip data fidelity, sidecar content, dtype support

Dependencies
------------
pytest
numpy
Pillow (optional, for PngWriter tests)

Author
------
Claude Code (generated)

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2026-02-24
"""

import json
import tempfile
from pathlib import Path

import pytest
import numpy as np

try:
    from grdl.IO.numpy_io import NumpyWriter
    _HAS_NUMPY_WRITER = True
except ImportError:
    _HAS_NUMPY_WRITER = False

try:
    from grdl.IO.png import PngWriter
    _HAS_PNG_WRITER = True
except ImportError:
    _HAS_PNG_WRITER = False

try:
    from PIL import Image as PILImage
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    from grdl.IO.models import ImageMetadata
    _HAS_METADATA = True
except ImportError:
    _HAS_METADATA = False


# ===================================================================
# NumpyWriter Level 1
# ===================================================================

@pytest.mark.skipif(not _HAS_NUMPY_WRITER,
                    reason="grdl NumpyWriter not available")
class TestNumpyWriterLevel1:
    """Validate NumpyWriter file creation and context manager."""

    def test_numpy_write_creates_file(self):
        data = np.random.rand(32, 32).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npy"
            writer = NumpyWriter(path)
            writer.write(data)
            assert path.exists()

    def test_numpy_write_sidecar_created(self):
        data = np.random.rand(32, 32).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npy"
            writer = NumpyWriter(path)
            writer.write(data)
            sidecar = Path(str(path) + '.json')
            assert sidecar.exists()

    def test_numpy_write_context_manager(self):
        data = np.random.rand(32, 32).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npy"
            with NumpyWriter(path) as writer:
                writer.write(data)
            assert path.exists()

    def test_numpy_write_npz_creates_file(self):
        arr1 = np.random.rand(16, 16).astype(np.float32)
        arr2 = np.random.rand(16, 16).astype(np.float64)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npz"
            writer = NumpyWriter(path)
            writer.write_npz({'array_a': arr1, 'array_b': arr2})
            # np.savez appends .npz if not present; check the actual file
            actual = Path(str(path))
            assert actual.exists() or Path(str(path) + '.npz').exists()

    def test_numpy_write_chip_raises(self):
        data = np.random.rand(32, 32).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npy"
            writer = NumpyWriter(path)
            with pytest.raises(NotImplementedError):
                writer.write_chip(data, 0, 0)


# ===================================================================
# NumpyWriter Level 2
# ===================================================================

@pytest.mark.skipif(not _HAS_NUMPY_WRITER,
                    reason="grdl NumpyWriter not available")
class TestNumpyWriterLevel2:
    """Validate NumpyWriter data roundtrip fidelity."""

    def test_numpy_write_roundtrip_float32(self):
        data = np.random.rand(32, 32).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npy"
            NumpyWriter(path).write(data)
            loaded = np.load(str(path))
            np.testing.assert_allclose(loaded, data)

    def test_numpy_write_roundtrip_float64(self):
        data = np.random.rand(32, 32).astype(np.float64)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npy"
            NumpyWriter(path).write(data)
            loaded = np.load(str(path))
            np.testing.assert_allclose(loaded, data)

    def test_numpy_write_roundtrip_complex64(self):
        data = (np.random.rand(32, 32)
                + 1j * np.random.rand(32, 32)).astype(np.complex64)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npy"
            NumpyWriter(path).write(data)
            loaded = np.load(str(path))
            np.testing.assert_allclose(loaded, data)

    def test_numpy_write_roundtrip_dtype_preserved(self):
        for dtype in (np.float32, np.float64, np.int16, np.uint8):
            data = np.ones((8, 8), dtype=dtype)
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "test.npy"
                NumpyWriter(path).write(data)
                loaded = np.load(str(path))
                assert loaded.dtype == dtype, (
                    f"Expected {dtype}, got {loaded.dtype}"
                )

    def test_numpy_write_sidecar_content(self):
        data = np.random.rand(32, 32).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npy"
            NumpyWriter(path).write(data)
            sidecar_path = Path(str(path) + '.json')
            with open(sidecar_path) as f:
                sidecar = json.load(f)
            assert sidecar['shape'] == [32, 32]
            assert sidecar['dtype'] == 'float32'

    def test_numpy_write_npz_roundtrip(self):
        arr1 = np.arange(10, dtype=np.float32)
        arr2 = np.arange(20, dtype=np.float64)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test"
            NumpyWriter(path).write_npz({'a': arr1, 'b': arr2})
            # np.savez may add .npz extension
            npz_path = Path(str(path) + '.npz') if not path.exists() else path
            loaded = np.load(str(npz_path))
            np.testing.assert_allclose(loaded['a'], arr1)
            np.testing.assert_allclose(loaded['b'], arr2)

    @pytest.mark.skipif(not _HAS_METADATA,
                        reason="grdl ImageMetadata not available")
    def test_numpy_write_with_metadata(self):
        data = np.random.rand(32, 32).astype(np.float32)
        meta = ImageMetadata(
            format='numpy', rows=32, cols=32, dtype='float32')
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npy"
            NumpyWriter(path, metadata=meta).write(data)
            sidecar_path = Path(str(path) + '.json')
            with open(sidecar_path) as f:
                sidecar = json.load(f)
            assert 'format' in sidecar


# ===================================================================
# PngWriter Level 1
# ===================================================================

@pytest.mark.skipif(not _HAS_PNG_WRITER,
                    reason="grdl PngWriter not available")
class TestPngWriterLevel1:
    """Validate PngWriter file creation and error handling."""

    def test_png_write_creates_file(self):
        data = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.png"
            PngWriter(path).write(data)
            assert path.exists()

    def test_png_write_context_manager(self):
        data = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.png"
            with PngWriter(path) as writer:
                writer.write(data)
            assert path.exists()

    def test_png_write_chip_raises(self):
        data = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.png"
            writer = PngWriter(path)
            with pytest.raises(NotImplementedError):
                writer.write_chip(data, 0, 0)

    def test_png_write_invalid_shape_4channel(self):
        data = np.random.randint(0, 256, (64, 64, 4), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.png"
            writer = PngWriter(path)
            with pytest.raises(ValueError):
                writer.write(data)

    def test_png_write_invalid_shape_1d(self):
        data = np.random.randint(0, 256, (64,), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.png"
            writer = PngWriter(path)
            with pytest.raises(ValueError):
                writer.write(data)


# ===================================================================
# PngWriter Level 2
# ===================================================================

@pytest.mark.skipif(not _HAS_PNG_WRITER or not _HAS_PIL,
                    reason="grdl PngWriter or Pillow not available")
class TestPngWriterLevel2:
    """Validate PngWriter data roundtrip via PIL readback."""

    def test_png_write_grayscale_roundtrip(self):
        data = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.png"
            PngWriter(path).write(data)
            img = PILImage.open(str(path))
            assert img.mode == 'L'
            loaded = np.array(img)
            np.testing.assert_array_equal(loaded, data)

    def test_png_write_rgb_roundtrip(self):
        data = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.png"
            PngWriter(path).write(data)
            img = PILImage.open(str(path))
            assert img.mode == 'RGB'
            loaded = np.array(img)
            np.testing.assert_array_equal(loaded, data)

    def test_png_write_float_auto_normalize_warns(self):
        data = np.random.rand(16, 16).astype(np.float64)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.png"
            with pytest.warns(UserWarning, match="auto-normalized"):
                PngWriter(path).write(data)
            assert path.exists()

    def test_png_write_size_matches(self):
        data = np.random.randint(0, 256, (48, 64), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.png"
            PngWriter(path).write(data)
            img = PILImage.open(str(path))
            # PIL size is (width, height) = (cols, rows)
            assert img.size == (64, 48)

    def test_png_write_constant_float_all_zeros(self):
        """Constant float array: dmax-dmin == 0, so output is all zeros."""
        data = np.full((16, 16), 5.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.png"
            with pytest.warns(UserWarning):
                PngWriter(path).write(data)
            img = PILImage.open(str(path))
            loaded = np.array(img)
            np.testing.assert_array_equal(loaded, 0)
