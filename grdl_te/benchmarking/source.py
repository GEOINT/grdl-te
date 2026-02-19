# -*- coding: utf-8 -*-
"""
Benchmark Source — unified data source for benchmark runners.

Provides ``BenchmarkSource`` with classmethod factories for creating
benchmark input data from size presets, explicit dimensions, file paths,
or existing arrays.  Data is generated lazily on first ``resolve()``
call and cached for reuse across warmup and measurement iterations.

Dependencies
------------
numpy

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
from typing import Any, Callable, Dict, Optional, Tuple, Union

# Third-party
import numpy as np

__all__ = ["ARRAY_SIZES", "BenchmarkSource"]

ARRAY_SIZES: Dict[str, Tuple[int, int]] = {
    "small": (512, 512),
    "medium": (2048, 2048),
    "large": (4096, 4096),
}
"""Canonical benchmark array size presets."""


def _default_generator(
    rows: int,
    cols: int,
    dtype: np.dtype,
    seed: int,
) -> np.ndarray:
    """Generate synthetic benchmark data with reproducible randomness.

    Parameters
    ----------
    rows : int
        Number of rows.
    cols : int
        Number of columns.
    dtype : np.dtype
        Target dtype.  Complex dtypes generate real + imaginary parts.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
    """
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.complexfloating):
        float_dtype = np.float32 if dtype == np.complex64 else np.float64
        real = rng.standard_normal((rows, cols), dtype=float_dtype)
        imag = rng.standard_normal((rows, cols), dtype=float_dtype)
        return (real + 1j * imag).astype(dtype)
    return rng.standard_normal((rows, cols)).astype(dtype)


class BenchmarkSource:
    """Benchmark data source with generate-once semantics.

    Wraps the different ways benchmark input data can be specified
    and resolves them lazily to a value that ``Workflow.execute()``
    accepts.  The resolved value is cached so that warmup and
    measurement iterations all operate on identical, read-only data.

    Use the classmethod factories to construct instances:

    - :meth:`synthetic` — size preset or explicit ``(rows, cols)``
    - :meth:`from_file` — file path for real-data benchmarking
    - :meth:`from_array` — existing ``np.ndarray``

    Examples
    --------
    >>> source = BenchmarkSource.synthetic("medium")
    >>> source.resolve().shape
    (2048, 2048)

    >>> source = BenchmarkSource.synthetic((1024, 512), dtype=np.complex64)
    >>> source.resolve().dtype
    dtype('complex64')

    >>> source = BenchmarkSource.from_file("image.nitf")
    >>> source.resolve()
    PosixPath('image.nitf')
    """

    def __init__(self) -> None:
        raise TypeError(
            "Use BenchmarkSource.synthetic(), .from_file(), or "
            ".from_array() to create instances."
        )

    @classmethod
    def synthetic(
        cls,
        size: Union[str, Tuple[int, int]] = "medium",
        *,
        dtype: Any = np.float32,
        seed: int = 42,
        generator: Optional[Callable[[int, int], np.ndarray]] = None,
    ) -> 'BenchmarkSource':
        """Create a synthetic data source.

        Parameters
        ----------
        size : str or Tuple[int, int]
            Size preset name (``"small"``, ``"medium"``, ``"large"``)
            or explicit ``(rows, cols)`` tuple.
        dtype : numpy dtype
            Data type for generation.  Default ``np.float32``.
            Ignored when *generator* is provided.
        seed : int
            Random seed for reproducibility.  Default 42.
            Ignored when *generator* is provided.
        generator : callable, optional
            Custom ``(rows, cols) -> np.ndarray`` factory.  Overrides
            the default random generator.

        Returns
        -------
        BenchmarkSource

        Raises
        ------
        ValueError
            If *size* is a string not in :data:`ARRAY_SIZES`.
        """
        if isinstance(size, str):
            if size not in ARRAY_SIZES:
                raise ValueError(
                    f"Unknown preset {size!r}. "
                    f"Choose from: {', '.join(ARRAY_SIZES)}"
                )
            rows, cols = ARRAY_SIZES[size]
            preset_name = size
        else:
            rows, cols = size
            preset_name = None

        inst = object.__new__(cls)
        inst._mode = "synthetic"
        inst._rows = rows
        inst._cols = cols
        inst._dtype = np.dtype(dtype)
        inst._seed = seed
        inst._generator = generator
        inst._preset_name = preset_name
        inst._path = None
        inst._array = None
        inst._resolved = None
        return inst

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'BenchmarkSource':
        """Create a file-based data source.

        The file path is passed directly to ``Workflow.execute()``,
        which handles reading via its configured reader.

        Parameters
        ----------
        path : str or Path
            Path to the image file.  Must exist.

        Returns
        -------
        BenchmarkSource

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        """
        resolved_path = Path(path)
        if not resolved_path.exists():
            raise FileNotFoundError(
                f"Benchmark source file not found: {resolved_path}"
            )
        inst = object.__new__(cls)
        inst._mode = "file"
        inst._path = resolved_path
        inst._rows = None
        inst._cols = None
        inst._dtype = None
        inst._seed = None
        inst._generator = None
        inst._preset_name = None
        inst._array = None
        inst._resolved = inst._path
        return inst

    @classmethod
    def from_array(cls, array: np.ndarray) -> 'BenchmarkSource':
        """Create a source from an existing array.

        The array is made read-only to prevent mutation across
        benchmark iterations.

        Parameters
        ----------
        array : np.ndarray
            Input array.

        Returns
        -------
        BenchmarkSource
        """
        inst = object.__new__(cls)
        inst._mode = "array"
        inst._rows = array.shape[0]
        inst._cols = array.shape[1] if array.ndim > 1 else 1
        inst._dtype = array.dtype
        inst._seed = None
        inst._generator = None
        inst._preset_name = None
        inst._path = None
        inst._array = array
        # Make read-only to prevent mutation across iterations
        view = array.view()
        view.flags.writeable = False
        inst._resolved = view
        return inst

    def resolve(self) -> Union[np.ndarray, Path]:
        """Return the benchmark data, generating it if needed.

        For synthetic sources, data is generated on first call and
        cached.  The returned array is read-only to prevent accidental
        mutation across benchmark iterations.

        For file sources, returns the ``Path`` object.

        Returns
        -------
        np.ndarray or Path
        """
        if self._resolved is not None:
            return self._resolved

        # Synthetic generation
        if self._generator is not None:
            arr = self._generator(self._rows, self._cols)
        else:
            arr = _default_generator(
                self._rows, self._cols, self._dtype, self._seed,
            )

        # Freeze to prevent mutation across iterations
        arr.flags.writeable = False
        self._resolved = arr
        return arr

    @property
    def description(self) -> str:
        """Human-readable source description for tagging.

        Returns
        -------
        str
            E.g. ``"synthetic/medium/2048x2048/float32"``,
            ``"file/image.nitf"``, ``"array/1024x1024/complex64"``.
        """
        if self._mode == "file":
            return f"file/{self._path.name}"
        dims = f"{self._rows}x{self._cols}"
        dtype_str = str(self._dtype) if self._dtype is not None else "unknown"
        if self._mode == "array":
            return f"array/{dims}/{dtype_str}"
        # synthetic
        label = self._preset_name or "custom"
        return f"synthetic/{label}/{dims}/{dtype_str}"

    @property
    def shape_hint(self) -> Optional[Tuple[int, int]]:
        """Array dimensions if known.

        Returns
        -------
        Tuple[int, int] or None
            ``(rows, cols)`` for synthetic and array sources,
            ``None`` for file sources.
        """
        if self._rows is not None and self._cols is not None:
            return (self._rows, self._cols)
        return None
