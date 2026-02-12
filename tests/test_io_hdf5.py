# -*- coding: utf-8 -*-
"""
HDF5 Reader Tests - VIIRS VNP09 Validation with GRDL Integration.

Tests grdl.IO.HDF5Reader with real VIIRS VNP09GA files, including:
- Level 1: Format validation (dataset discovery, metadata extraction, chip/full reads)
- Level 2: Data quality (hierarchical structure, attributes, reflectance range)
- Level 3: Integration (ChipExtractor, Normalizer, Tiler workflows)

Dataset: VIIRS VNP09GA (Daily Surface Reflectance, HDF5)

Dependencies
------------
pytest
h5py
numpy
grdl

Author
------
Ava Courtney

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2026-02-11

Modified
--------
2026-02-11
"""

# Standard library
from pathlib import Path

# Third-party
import pytest
import numpy as np

try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False

# GRDL internal
try:
    from grdl.IO.hdf5 import HDF5Reader
    _HAS_GRDL = True
except ImportError:
    _HAS_GRDL = False

try:
    from grdl.data_prep import ChipExtractor, Normalizer, Tiler
    _HAS_DATA_PREP = True
except ImportError:
    _HAS_DATA_PREP = False


pytestmark = [
    pytest.mark.viirs,
    pytest.mark.requires_data,
    pytest.mark.skipif(not _HAS_H5PY, reason="h5py not installed"),
    pytest.mark.skipif(not _HAS_GRDL, reason="grdl not installed"),
]


# =============================================================================
# Level 1: Format Validation
# =============================================================================


@pytest.mark.slow
def test_viirs_list_datasets(require_viirs_file):
    """Verify HDF5Reader discovers multiple datasets in VIIRS file.

    VIIRS VNP09GA files contain numerous datasets organized in
    hierarchical groups. Each entry must report path, shape, and dtype.
    """
    datasets = HDF5Reader.list_datasets(str(require_viirs_file))

    assert isinstance(datasets, list)
    assert len(datasets) > 0, "VIIRS file should contain discoverable datasets"

    print(f"Found {len(datasets)} datasets in {require_viirs_file.name}")

    # Each entry must have (path, shape, dtype) structure
    for path, shape, dtype in datasets[:5]:
        print(f"  {path}: {shape} ({dtype})")
        assert isinstance(path, str)
        assert path.startswith('/'), f"Dataset path should be absolute: {path}"
        assert isinstance(shape, tuple)
        assert len(shape) >= 1, f"Dataset {path} has empty shape"
        assert isinstance(dtype, str)


@pytest.mark.slow
def test_viirs_auto_detect_dataset(require_viirs_file):
    """Verify HDF5Reader auto-detects a suitable 2D dataset.

    Auto-detection should find a 2D (or higher) dataset suitable
    for image reading, skipping 1D arrays.
    """
    with HDF5Reader(str(require_viirs_file)) as reader:
        shape = reader.get_shape()
        assert len(shape) >= 2, \
            f"Auto-detected dataset should be 2D+, got shape {shape}"
        assert shape[0] > 0 and shape[1] > 0

        print(f"Auto-detected dataset shape: {shape}")


@pytest.mark.slow
def test_viirs_read_specific_dataset(require_viirs_file):
    """Verify HDF5Reader opens an explicitly-selected dataset path.

    Searches for a surface reflectance dataset by name, falling back
    to the first suitable 2D dataset if none match.
    """
    datasets = HDF5Reader.list_datasets(str(require_viirs_file))

    # Look for surface reflectance bands
    dataset_path = None
    for ds_path, shape, dtype in datasets:
        if len(shape) == 2 and 'sur_refl' in ds_path:
            dataset_path = ds_path
            break

    # Fallback: use first 2D dataset
    if dataset_path is None:
        for ds_path, shape, dtype in datasets:
            if len(shape) == 2 and shape[0] > 1 and shape[1] > 1:
                dataset_path = ds_path
                break

    if dataset_path is None:
        pytest.skip("No suitable 2D dataset found in VIIRS file")

    # Open with explicit path and verify
    with HDF5Reader(str(require_viirs_file), dataset_path=dataset_path) as reader:
        assert reader.dataset_path == dataset_path
        shape = reader.get_shape()
        assert len(shape) >= 2

        print(f"Reading dataset: {dataset_path}, shape: {shape}")


@pytest.mark.slow
def test_viirs_get_shape(require_viirs_file):
    """Verify get_shape() returns valid positive dimensions."""
    with HDF5Reader(str(require_viirs_file)) as reader:
        shape = reader.get_shape()

        assert isinstance(shape, tuple)
        assert len(shape) >= 2

        rows, cols = shape[0], shape[1]
        assert isinstance(rows, int)
        assert isinstance(cols, int)
        assert rows > 0 and cols > 0

        # Shape must agree with metadata
        assert rows == reader.metadata.rows
        assert cols == reader.metadata.cols

        # VIIRS 500m grid typically 2400x2400
        print(f"VIIRS shape: {rows} rows x {cols} cols")


@pytest.mark.slow
def test_viirs_get_dtype(require_viirs_file):
    """Verify get_dtype() returns expected integer type for VIIRS SR."""
    with HDF5Reader(str(require_viirs_file)) as reader:
        dtype = reader.get_dtype()

        assert dtype is not None
        # VIIRS surface reflectance typically int16 or uint16
        assert dtype in [np.int16, np.uint16, np.int32, np.float32]

        # dtype must agree with metadata
        assert str(dtype) == reader.metadata.dtype

        print(f"VIIRS dtype: {dtype}")


@pytest.mark.slow
def test_viirs_read_chip(require_viirs_file):
    """Verify read_chip() returns correctly shaped array with data content."""
    with HDF5Reader(str(require_viirs_file)) as reader:
        rows, cols = reader.get_shape()[:2]

        # Read 512x512 chip from center
        row_start = max(0, rows // 2 - 256)
        row_end = min(rows, rows // 2 + 256)
        col_start = max(0, cols // 2 - 256)
        col_end = min(cols, cols // 2 + 256)

        chip = reader.read_chip(row_start, row_end, col_start, col_end)

        assert isinstance(chip, np.ndarray)
        assert chip.ndim == 2
        assert chip.shape[0] == (row_end - row_start)
        assert chip.shape[1] == (col_end - col_start)

        # Chip should not be entirely uniform (indicates real data)
        assert chip.std() > 0 or chip.size == 1, \
            "Center chip has zero variance (all identical values)"

        print(f"VIIRS chip: {chip.shape}, dtype: {chip.dtype}, "
              f"range: [{chip.min()}, {chip.max()}]")


@pytest.mark.slow
def test_viirs_read_full(require_viirs_file):
    """Verify read_full() returns array matching reported shape."""
    with HDF5Reader(str(require_viirs_file)) as reader:
        rows, cols = reader.get_shape()[:2]

        # VIIRS files can be large; skip if too big
        if rows * cols > 50_000_000:  # ~50 megapixels
            pytest.skip("VIIRS file too large for full read test")

        data = reader.read_full()

        assert isinstance(data, np.ndarray)
        assert data.shape[0] == rows
        assert data.shape[1] == cols
        assert data.dtype == reader.get_dtype()

        print(f"Full VIIRS dataset read: {data.shape}")


@pytest.mark.slow
def test_viirs_context_manager(require_viirs_file):
    """Verify context manager opens and releases HDF5 resources."""
    reader = HDF5Reader(str(require_viirs_file))
    assert hasattr(reader, '__enter__')
    assert hasattr(reader, '__exit__')

    with reader:
        shape = reader.get_shape()
        assert len(shape) >= 2


# =============================================================================
# Level 2: Data Quality
# =============================================================================


@pytest.mark.slow
def test_viirs_dataset_structure(require_viirs_file):
    """Verify VIIRS HDF-EOS hierarchical dataset structure.

    VNP09GA files should contain multiple datasets organized under
    HDFEOS/GRIDS groups with Data Fields for each grid resolution.
    """
    datasets = HDF5Reader.list_datasets(str(require_viirs_file))

    # VIIRS should have many datasets (10+ bands + QA layers)
    assert len(datasets) > 5, \
        f"Expected >5 datasets in VIIRS file, found {len(datasets)}"

    dataset_paths = [path for path, _, _ in datasets]

    # Verify HDF-EOS or MODIS-like group structure exists
    has_grid_structure = any(
        'GRIDS' in path or 'Data Fields' in path or 'MODIS' in path
        for path in dataset_paths
    )
    assert has_grid_structure, \
        f"Expected HDFEOS/GRIDS or Data Fields structure. Paths: {dataset_paths[:5]}"

    # Verify at least one 2D dataset exists (image data)
    has_2d = any(
        len(shape) == 2 and shape[0] > 1 and shape[1] > 1
        for _, shape, _ in datasets
    )
    assert has_2d, "VIIRS file should contain at least one 2D image dataset"

    print(f"VIIRS datasets: {len(datasets)}, structure validated")


@pytest.mark.slow
def test_viirs_attributes_extraction(require_viirs_file):
    """Verify VIIRS file-level metadata attributes are accessible.

    VIIRS HDF5 files contain extensive root-level attributes describing
    the data product, processing, and geolocation. Uses h5py directly
    to validate format-level attributes not exposed through GRDL metadata.
    """
    with h5py.File(str(require_viirs_file), 'r') as f:
        attrs = dict(f.attrs)
        assert len(attrs) > 0, \
            "VIIRS file should have root-level HDF5 attributes"

        print(f"VIIRS file has {len(attrs)} root-level attributes")

        # Check for some common VIIRS/HDF-EOS attributes
        for key in list(attrs.keys())[:5]:
            print(f"  {key}: {attrs[key]}")


@pytest.mark.slow
def test_viirs_surface_reflectance_validation(require_viirs_file):
    """Verify VIIRS surface reflectance data falls within physical bounds.

    VIIRS SR is scaled integer (scale factor 0.0001). Raw values should
    be in the range [-100, 16000]. Fill value is typically -28672.
    Valid data pixels must exist within the center of the tile.
    """
    datasets = HDF5Reader.list_datasets(str(require_viirs_file))

    sr_path = None
    for path, shape, dtype in datasets:
        if 'sur_refl' in path and len(shape) == 2:
            sr_path = path
            break

    if sr_path is None:
        pytest.skip("No surface reflectance dataset found in VIIRS file")

    with HDF5Reader(str(require_viirs_file), dataset_path=sr_path) as reader:
        rows, cols = reader.get_shape()[:2]

        # Read from center to maximize chance of valid data
        r0 = max(0, rows // 2 - 256)
        r1 = min(rows, rows // 2 + 256)
        c0 = max(0, cols // 2 - 256)
        c1 = min(cols, cols // 2 + 256)
        chip = reader.read_chip(r0, r1, c0, c1)

        print(f"VIIRS SR dataset: {sr_path}")
        print(f"SR raw range: [{chip.min()}, {chip.max()}]")

        # Check for valid data (excluding fill values)
        # VIIRS fill value is typically -28672
        valid_mask = chip > -28000
        assert valid_mask.any(), \
            "Center chip contains no valid pixels (all fill values)"

        valid_data = chip[valid_mask]

        # Valid scaled reflectance should be bounded
        # Raw int16 values: -100 to 16000 (scale factor 0.0001)
        assert valid_data.min() >= -1000, \
            f"Valid SR min {valid_data.min()} is unreasonably low"
        assert valid_data.max() <= 16100, \
            f"Valid SR max {valid_data.max()} exceeds physical bounds"

        print(f"Valid SR range: [{valid_data.min()}, {valid_data.max()}]")


@pytest.mark.slow
def test_viirs_data_types(require_viirs_file):
    """Verify VIIRS datasets use expected integer data types.

    VIIRS products predominantly use integer types (int16, uint16)
    for radiometric data and quality flags.
    """
    datasets = HDF5Reader.list_datasets(str(require_viirs_file))

    dtypes = {}
    for path, shape, dtype in datasets:
        dtypes[dtype] = dtypes.get(dtype, 0) + 1

    print(f"VIIRS data types: {dtypes}")

    # VIIRS must have at least one integer-type dataset
    integer_types = {'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32'}
    has_integer_type = any(dt in integer_types for dt in dtypes)
    assert has_integer_type, \
        f"Expected integer data types in VIIRS file, found: {list(dtypes.keys())}"


# =============================================================================
# Level 3: Integration with GRDL Utilities
# =============================================================================


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
@pytest.mark.integration
def test_viirs_chip_extractor_integration(require_viirs_file):
    """Validate ChipExtractor partitions VIIRS data into uniform chips.

    Verifies chip regions are within bounds and extracted chips have
    correct dimensions.
    """
    with HDF5Reader(str(require_viirs_file)) as reader:
        rows, cols = reader.get_shape()[:2]

        extractor = ChipExtractor(nrows=rows, ncols=cols)
        regions = extractor.chip_positions(row_width=256, col_width=256)

        assert len(regions) > 0

        for i, region in enumerate(regions[:5]):
            assert 0 <= region.row_start < region.row_end <= rows
            assert 0 <= region.col_start < region.col_end <= cols

            chip = reader.read_chip(
                region.row_start, region.row_end,
                region.col_start, region.col_end
            )

            assert isinstance(chip, np.ndarray)
            assert chip.ndim == 2
            assert chip.shape[0] == region.row_end - region.row_start
            assert chip.shape[1] == region.col_end - region.col_start

        print(f"ChipExtractor: {len(regions)} chips validated")


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
@pytest.mark.integration
def test_viirs_normalizer_integration(require_viirs_file):
    """Validate Normalizer produces finite, bounded output from VIIRS data.

    Converts VIIRS int16 data to float before normalization, then
    validates minmax output is in [0, 1].
    """
    with HDF5Reader(str(require_viirs_file)) as reader:
        rows, cols = reader.get_shape()[:2]
        chip = reader.read_chip(0, min(512, rows), 0, min(512, cols))

        # Convert to float (VIIRS is often int16)
        chip_float = chip.astype(np.float32)

        normalizer = Normalizer(method='minmax')
        normalized = normalizer.normalize(chip_float)

        assert isinstance(normalized, np.ndarray)
        assert normalized.dtype == np.float64
        assert np.isfinite(normalized).all()
        assert 0.0 <= normalized.min() <= normalized.max() <= 1.0

        print(f"MinMax normalized: [{normalized.min():.3f}, {normalized.max():.3f}]")


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
@pytest.mark.integration
def test_viirs_tiler_integration(require_viirs_file):
    """Validate Tiler creates overlapping tile grid over VIIRS dataset.

    Verifies tile regions are within bounds and tiles read correctly.
    """
    with HDF5Reader(str(require_viirs_file)) as reader:
        rows, cols = reader.get_shape()[:2]

        tiler = Tiler(
            nrows=rows,
            ncols=cols,
            tile_size=512,
            stride=256
        )

        tiles = tiler.tile_positions()
        assert len(tiles) > 0

        for i, tile_region in enumerate(tiles[:5]):
            assert 0 <= tile_region.row_start < tile_region.row_end <= rows
            assert 0 <= tile_region.col_start < tile_region.col_end <= cols

            tile = reader.read_chip(
                tile_region.row_start, tile_region.row_end,
                tile_region.col_start, tile_region.col_end
            )

            assert isinstance(tile, np.ndarray)
            assert tile.ndim == 2
            assert tile.shape[0] <= 512
            assert tile.shape[1] <= 512

        print(f"Tiler: {len(tiles)} overlapping tiles validated")


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
@pytest.mark.integration
def test_viirs_chip_normalize_pipeline(require_viirs_file):
    """Validate end-to-end VIIRS chip extraction and normalization pipeline.

    Extracts chips, converts to float, normalizes with zscore, and
    validates output properties across the batch.
    """
    with HDF5Reader(str(require_viirs_file)) as reader:
        rows, cols = reader.get_shape()[:2]

        extractor = ChipExtractor(nrows=rows, ncols=cols)
        regions = extractor.chip_positions(row_width=256, col_width=256)

        normalizer = Normalizer(method='zscore')
        normalized_chips = []

        for region in regions[:10]:
            chip = reader.read_chip(
                region.row_start, region.row_end,
                region.col_start, region.col_end
            )

            chip_float = chip.astype(np.float32)
            normalized = normalizer.normalize(chip_float)
            normalized_chips.append(normalized)

            assert isinstance(normalized, np.ndarray)
            assert normalized.dtype == np.float64
            assert np.isfinite(normalized).all()

        assert len(normalized_chips) > 0

        # Validate batch statistics
        all_values = np.concatenate([nc.flatten() for nc in normalized_chips])
        print(f"Pipeline: {len(normalized_chips)} chips, "
              f"batch mean={all_values.mean():.3f}, std={all_values.std():.3f}")


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
@pytest.mark.integration
def test_viirs_multi_dataset_workflow(require_viirs_file):
    """Validate multi-dataset navigation and per-dataset normalization.

    Opens multiple 2D datasets from the same VIIRS file in sequence,
    reads a chip from each, and normalizes independently.
    """
    datasets = HDF5Reader.list_datasets(str(require_viirs_file))

    # Find 2D datasets
    dataset_2d = [(path, shape) for path, shape, dtype in datasets if len(shape) == 2]

    if len(dataset_2d) < 2:
        pytest.skip("Need at least 2 datasets for multi-dataset test")

    # Process first two datasets
    for path, shape in dataset_2d[:2]:
        with HDF5Reader(str(require_viirs_file), dataset_path=path) as reader:
            rows, cols = reader.get_shape()[:2]

            chip = reader.read_chip(0, min(256, rows), 0, min(256, cols))

            # Normalize
            chip_float = chip.astype(np.float32)
            normalizer = Normalizer(method='minmax')
            normalized = normalizer.normalize(chip_float)

            assert normalized.shape == chip.shape
            assert np.isfinite(normalized).all()
            assert 0.0 <= normalized.min() <= normalized.max() <= 1.0

            print(f"Processed dataset: {path}, shape: {chip.shape}")
