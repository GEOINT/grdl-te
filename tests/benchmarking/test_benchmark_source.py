# -*- coding: utf-8 -*-
"""
Tests for BenchmarkSource.

Validates construction via classmethod factories, lazy resolution,
caching, read-only protection, and description/shape_hint properties.

Author
------
Ava Courtney

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2026-02-18

Modified
--------
2026-02-18
"""

# Standard library
from pathlib import Path

# Third-party
import numpy as np
import pytest

# Internal
from grdl_te.benchmarking.source import ARRAY_SIZES, BenchmarkSource


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestBenchmarkSourceConstruction:
    """Tests for BenchmarkSource classmethod factories."""

    def test_direct_init_raises(self):
        """Direct __init__ raises TypeError."""
        with pytest.raises(TypeError, match="Use BenchmarkSource"):
            BenchmarkSource()

    def test_synthetic_default(self):
        """Default synthetic creates a medium float32 source."""
        src = BenchmarkSource.synthetic()
        assert src._mode == "synthetic"
        assert src._rows == 2048
        assert src._cols == 2048
        assert src._dtype == np.float32

    def test_synthetic_preset_small(self):
        """'small' preset resolves to 512x512."""
        src = BenchmarkSource.synthetic("small")
        assert src._rows == 512
        assert src._cols == 512

    def test_synthetic_preset_large(self):
        """'large' preset resolves to 4096x4096."""
        src = BenchmarkSource.synthetic("large")
        assert src._rows == 4096
        assert src._cols == 4096

    def test_synthetic_explicit_dims(self):
        """Explicit (rows, cols) tuple is accepted."""
        src = BenchmarkSource.synthetic((100, 200))
        assert src._rows == 100
        assert src._cols == 200

    def test_synthetic_invalid_preset_raises(self):
        """Unknown preset raises ValueError."""
        with pytest.raises(ValueError, match="Unknown preset"):
            BenchmarkSource.synthetic("xlarge")

    def test_from_file_str(self, tmp_path):
        """String path is converted to Path."""
        f = tmp_path / "image.nitf"
        f.write_bytes(b"\x00")
        src = BenchmarkSource.from_file(str(f))
        assert src._mode == "file"
        assert src._path == f

    def test_from_file_path(self, tmp_path):
        """Path object is stored directly."""
        f = tmp_path / "test.tif"
        f.write_bytes(b"\x00")
        src = BenchmarkSource.from_file(f)
        assert src._path == f

    def test_from_file_missing_raises(self):
        """Non-existent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            BenchmarkSource.from_file("/nonexistent/image.nitf")

    def test_from_array(self):
        """Array source stores the array."""
        arr = np.zeros((32, 32), dtype=np.float64)
        src = BenchmarkSource.from_array(arr)
        assert src._mode == "array"
        assert src._rows == 32
        assert src._cols == 32
        assert src._dtype == np.float64


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

class TestBenchmarkSourceResolve:
    """Tests for BenchmarkSource.resolve()."""

    def test_synthetic_resolves_to_ndarray(self):
        """Synthetic source resolves to an ndarray of correct shape."""
        src = BenchmarkSource.synthetic("small")
        arr = src.resolve()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (512, 512)
        assert arr.dtype == np.float32

    def test_synthetic_complex_dtype(self):
        """Complex dtype generates complex data."""
        src = BenchmarkSource.synthetic((64, 64), dtype=np.complex64)
        arr = src.resolve()
        assert arr.dtype == np.complex64
        assert np.iscomplexobj(arr)

    def test_synthetic_float64_dtype(self):
        """float64 dtype generates float64 data."""
        src = BenchmarkSource.synthetic((32, 32), dtype=np.float64)
        arr = src.resolve()
        assert arr.dtype == np.float64

    def test_file_resolves_to_path(self, tmp_path):
        """File source resolves to a Path object."""
        f = tmp_path / "image.nitf"
        f.write_bytes(b"\x00")
        src = BenchmarkSource.from_file(f)
        result = src.resolve()
        assert isinstance(result, Path)
        assert result == f

    def test_array_resolves_to_view(self):
        """Array source resolves to a read-only view."""
        arr = np.ones((16, 16), dtype=np.float32)
        src = BenchmarkSource.from_array(arr)
        result = src.resolve()
        assert isinstance(result, np.ndarray)
        assert result.shape == (16, 16)
        np.testing.assert_array_equal(result, arr)

    def test_resolve_caches_synthetic(self):
        """Repeated resolve() returns the exact same object."""
        src = BenchmarkSource.synthetic((64, 64))
        first = src.resolve()
        second = src.resolve()
        assert first is second

    def test_resolve_caches_array(self):
        """Repeated resolve() on array source returns same object."""
        arr = np.zeros((8, 8))
        src = BenchmarkSource.from_array(arr)
        first = src.resolve()
        second = src.resolve()
        assert first is second

    def test_resolved_array_is_read_only(self):
        """Resolved synthetic array is not writeable."""
        src = BenchmarkSource.synthetic((32, 32))
        arr = src.resolve()
        assert not arr.flags.writeable

    def test_resolved_from_array_is_read_only(self):
        """Resolved from_array view is not writeable."""
        arr = np.zeros((16, 16))
        src = BenchmarkSource.from_array(arr)
        result = src.resolve()
        assert not result.flags.writeable

    def test_custom_generator(self):
        """Custom generator is used for data creation."""
        def gen(rows, cols):
            return np.ones((rows, cols), dtype=np.float32)

        src = BenchmarkSource.synthetic((32, 32), generator=gen)
        arr = src.resolve()
        np.testing.assert_array_equal(arr, np.ones((32, 32), dtype=np.float32))

    def test_seed_reproducibility(self):
        """Same seed produces identical arrays."""
        src1 = BenchmarkSource.synthetic((64, 64), seed=123)
        src2 = BenchmarkSource.synthetic((64, 64), seed=123)
        np.testing.assert_array_equal(src1.resolve(), src2.resolve())

    def test_different_seeds_differ(self):
        """Different seeds produce different arrays."""
        src1 = BenchmarkSource.synthetic((64, 64), seed=1)
        src2 = BenchmarkSource.synthetic((64, 64), seed=2)
        assert not np.array_equal(src1.resolve(), src2.resolve())


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestBenchmarkSourceProperties:
    """Tests for description and shape_hint properties."""

    def test_description_synthetic_preset(self):
        """Description includes preset name and dimensions."""
        src = BenchmarkSource.synthetic("medium")
        desc = src.description
        assert "synthetic" in desc
        assert "medium" in desc
        assert "2048x2048" in desc
        assert "float32" in desc

    def test_description_synthetic_custom_dims(self):
        """Description shows 'custom' for explicit dims."""
        src = BenchmarkSource.synthetic((100, 200))
        desc = src.description
        assert "synthetic" in desc
        assert "custom" in desc
        assert "100x200" in desc

    def test_description_file(self, tmp_path):
        """Description includes filename."""
        f = tmp_path / "test.nitf"
        f.write_bytes(b"\x00")
        src = BenchmarkSource.from_file(f)
        desc = src.description
        assert "file" in desc
        assert "test.nitf" in desc

    def test_description_array(self):
        """Description includes array dims and dtype."""
        arr = np.zeros((64, 64), dtype=np.complex64)
        src = BenchmarkSource.from_array(arr)
        desc = src.description
        assert "array" in desc
        assert "64x64" in desc
        assert "complex64" in desc

    def test_shape_hint_preset(self):
        """shape_hint returns (rows, cols) for preset."""
        src = BenchmarkSource.synthetic("small")
        assert src.shape_hint == (512, 512)

    def test_shape_hint_explicit_dims(self):
        """shape_hint returns (rows, cols) for explicit dims."""
        src = BenchmarkSource.synthetic((100, 200))
        assert src.shape_hint == (100, 200)

    def test_shape_hint_array(self):
        """shape_hint returns (rows, cols) for array."""
        arr = np.zeros((32, 64))
        src = BenchmarkSource.from_array(arr)
        assert src.shape_hint == (32, 64)

    def test_shape_hint_file_is_none(self, tmp_path):
        """shape_hint is None for file sources."""
        f = tmp_path / "test.nitf"
        f.write_bytes(b"\x00")
        src = BenchmarkSource.from_file(f)
        assert src.shape_hint is None


# ---------------------------------------------------------------------------
# ARRAY_SIZES constant
# ---------------------------------------------------------------------------

class TestArraySizes:
    """Tests for the ARRAY_SIZES constant."""

    def test_contains_small(self):
        """'small' preset exists."""
        assert "small" in ARRAY_SIZES
        assert ARRAY_SIZES["small"] == (512, 512)

    def test_contains_medium(self):
        """'medium' preset exists."""
        assert "medium" in ARRAY_SIZES
        assert ARRAY_SIZES["medium"] == (2048, 2048)

    def test_contains_large(self):
        """'large' preset exists."""
        assert "large" in ARRAY_SIZES
        assert ARRAY_SIZES["large"] == (4096, 4096)
