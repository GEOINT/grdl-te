# GRDL-TE Development Guide

## Project Overview

GRDL-TE (GEOINT Rapid Development Library - Testing & Evaluation) is a high-fidelity validation suite that tests GRDL's IO readers against real-world remote sensing data from operational satellite platforms.

This is a **validation suite, not a library**. It produces no reusable code -- it validates that GRDL works correctly with production satellite data.

## Relationship to GRDL

GRDL-TE is a **dependency consumer** of GRDL. It imports and exercises GRDL's public API:

| GRDL Module | What GRDL-TE Tests |
|-------------|-------------------|
| `grdl.IO.GeoTIFFReader` | Landsat 8/9 COG files |
| `grdl.IO.HDF5Reader` | VIIRS VNP09GA granules |
| `grdl.IO.JP2Reader` | Sentinel-2 L2A JPEG2000 |
| `grdl.IO.NITFReader` | Umbra SICD complex SAR |
| `grdl.data_prep.ChipExtractor` | Region extraction integration |
| `grdl.data_prep.Normalizer` | Normalization integration |
| `grdl.data_prep.Tiler` | Tile grid integration |

GRDL-TE does not modify, extend, or wrap GRDL code. It only calls the public API and asserts correctness.

## Development Environment

```bash
conda activate grdl
```

## Design Principles

These principles are non-negotiable. Every test must satisfy all four:

| Principle | Rule |
|-----------|------|
| **No Pass by Default** | A test only passes when data is present AND code functions correctly. |
| **Missing Data = Skipped** | If the required data file is absent, the test is skipped with a download instruction. |
| **Broken Code = Failed** | If data is present but the reader or utility malfunctions, the test fails hard. |
| **Meaningful Assertions** | Every test validates a specific physical property (reflectance bounds, CRS projection, complex integrity). No tests that only check "did it load." |

## Test Architecture

All test files follow a strict **3-level validation structure**:

### Level 1: Format Validation
- Reader instantiation with context manager
- Metadata extraction (format, rows, cols, dtype, CRS, bands)
- Shape and dtype consistency with metadata
- Chip reads from image center (validates real data, not fill values)
- Full image reads (with size guards for large files)
- Resource cleanup verification

### Level 2: Data Quality
- **CRS/projection validation** (UTM zone ranges, EPSG codes, geospatial bounds)
- **NoData handling** (masking, valid pixel statistics, variance confirmation)
- **Value range bounds** (reflectance scales, bit-depth encoding ceilings, SAR magnitude ranges)
- **Format-specific features**:
  - GeoTIFF: COG tiling, internal overviews (pyramid levels)
  - HDF5: EOS-Grid structure, multi-SDS discovery
  - JP2: 15-bit encoding in 16-bit container, high-order bit usage
  - NITF: Complex magnitude/phase separation, SAR speckle statistics

### Level 3: Integration
- **ChipExtractor**: Uniform chip partitioning with bounds validation
- **Normalizer**: MinMax, Z-score, and percentile normalization workflows
- **Tiler**: Overlapping tile grids with stride control
- **Pipelines**: End-to-end workflows (chip -> normalize -> validate batch statistics)

Level 3 tests require `grdl.data_prep` to be importable. They are marked `integration` for selective execution.

## Data Management

### Single-Source-of-Truth Strategy

Each reader requires **exactly one representative file**. One file per reader. No multi-file test matrices.

| Reader | Directory | File Pattern | Source |
|--------|-----------|-------------|--------|
| GeoTIFFReader | `data/landsat/` | `LC0[89]*_SR_B*.TIF` | [USGS EarthExplorer](https://earthexplorer.usgs.gov) |
| HDF5Reader | `data/viirs/` | `V?P09GA*.h5` | [LAADS DAAC](https://ladsweb.modaps.eosdis.nasa.gov) |
| JP2Reader | `data/sentinel2/` | `S2*.SAFE/.../R10m/*_B04_10m.jp2` | [Copernicus Data Space](https://dataspace.copernicus.eu) |
| NITFReader | `data/umbra/` | `*.nitf` | [Umbra Open Data (AWS S3)](https://umbra-open-data-catalog.s3.amazonaws.com/index.html) |

### Data Rules

- Actual data files are **never committed** to git (`.gitignore`'d).
- Each `data/<dataset>/` directory contains a `README.md` with download instructions. These READMEs **are** committed.
- The `require_data_file()` fixture in `conftest.py` handles missing data gracefully -- tests skip with a message pointing to the relevant README.
- Do not add multiple files per reader unless testing a fundamentally different code path.

## Directory Structure

```
grdl-te/
├── data/                           # Real-world data files (not in git)
│   ├── README.md                   # Data strategy documentation
│   ├── landsat/
│   │   └── README.md               # Landsat download instructions
│   ├── viirs/
│   │   └── README.md               # VIIRS download instructions
│   ├── sentinel2/
│   │   └── README.md               # Sentinel-2 download instructions
│   └── umbra/
│       └── README.md               # Umbra download instructions
├── tests/
│   ├── conftest.py                 # Shared fixtures + graceful skip logic
│   ├── test_io_geotiff.py          # GeoTIFFReader (Landsat)
│   ├── test_io_hdf5.py             # HDF5Reader (VIIRS)
│   ├── test_io_jpeg2000.py         # JP2Reader (Sentinel-2)
│   └── test_io_nitf.py             # NITFReader (Umbra SICD)
├── pyproject.toml                  # Package config + pytest markers
├── LICENSE                         # MIT License
└── README.md                       # Project overview
```

## Test File Conventions

### File Naming

- Test files: `test_io_<format>.py` (one file per GRDL IO reader)
- Each test file is independently runnable: `pytest tests/test_io_geotiff.py -v`

### Module-Level Structure

Every test file follows this structure:

```python
# -*- coding: utf-8 -*-
"""
<Reader> Tests - <Dataset> Validation with GRDL Integration.

Tests grdl.IO.<Reader> with real <dataset> files, including:
- Level 1: Format validation (metadata extraction, shape/dtype, chip/full reads)
- Level 2: Data quality (CRS, NoData, bounds, format-specific features)
- Level 3: Integration (ChipExtractor, Normalizer, Tiler workflows)

Dataset: <Dataset description>

Dependencies
------------
pytest
<format-specific dependency>
numpy
grdl

Author
------
<Author name and email>

License
-------
MIT License
See LICENSE file for full text.

Created
-------
<YYYY-MM-DD>

Modified
--------
<YYYY-MM-DD>
"""

# Standard library
from pathlib import Path

# Third-party
import pytest
import numpy as np

# Guarded imports for optional dependencies
try:
    import <format_lib>
    _HAS_FORMAT = True
except ImportError:
    _HAS_FORMAT = False

try:
    from grdl.IO.<module> import <Reader>
    from grdl.IO.models import ImageMetadata
    _HAS_GRDL = True
except ImportError:
    _HAS_GRDL = False

try:
    from grdl.data_prep import ChipExtractor, Normalizer, Tiler
    _HAS_DATA_PREP = True
except ImportError:
    _HAS_DATA_PREP = False

# Module-level markers
pytestmark = [
    pytest.mark.<dataset>,
    pytest.mark.requires_data,
    pytest.mark.skipif(not _HAS_FORMAT, reason="<lib> not installed"),
    pytest.mark.skipif(not _HAS_GRDL, reason="grdl not installed"),
]


# =============================================================================
# Level 1: Format Validation
# =============================================================================

def test_<reader>_metadata(require_<dataset>_file):
    ...

# =============================================================================
# Level 2: Data Quality
# =============================================================================

def test_<reader>_crs_validation(require_<dataset>_file):
    ...

# =============================================================================
# Level 3: Integration
# =============================================================================

@pytest.mark.integration
@pytest.mark.skipif(not _HAS_DATA_PREP, reason="grdl.data_prep not available")
def test_<reader>_chip_extractor(require_<dataset>_file):
    ...
```

### Import Guards

GRDL-TE tests must handle the case where GRDL or format-specific dependencies are not installed. Use `try`/`except` guards at the module level and `pytest.mark.skipif` to skip the entire file when dependencies are missing.

### Pytest Markers

| Marker | Purpose |
|--------|---------|
| `landsat` | Landsat 8/9 tests (GeoTIFFReader) |
| `viirs` | VIIRS VNP09GA tests (HDF5Reader) |
| `sentinel2` | Sentinel-2 tests (JP2Reader) |
| `nitf` | Umbra SICD tests (NITFReader) |
| `requires_data` | Test requires real data files in `data/` |
| `slow` | Long-running test (large file reads, full pipelines) |
| `integration` | Level 3 tests (ChipExtractor, Normalizer, Tiler workflows) |

All markers are registered in `pyproject.toml`. Add new markers there before using them.

### Fixture Conventions

- **Session-scoped directory fixtures**: `landsat_data_dir`, `viirs_data_dir`, etc. return `Path` objects.
- **Data file fixtures**: `require_landsat_file`, `require_viirs_file`, etc. return a `Path` to the first matching file or skip the test.
- Use `require_data_file(directory, pattern)` from `conftest.py` for all data file lookups. Never write raw `Path.glob()` in tests.

## File Header Standard

Same as GRDL. Every `.py` file must begin with the standardized header:

```python
# -*- coding: utf-8 -*-
"""
<Module title -- short description.>

<Extended description.>

Dependencies
------------
<Non-numpy third-party packages>

Author
------
<Author name and email>

License
-------
MIT License
See LICENSE file for full text.

Created
-------
<YYYY-MM-DD>

Modified
--------
<YYYY-MM-DD>
"""
```

Retrieve author name and email from the local OS user profile, git config, or prompt the user. Do not hardcode a default.

## Python Standards

Follow GRDL's standards (see `grdl/CLAUDE.md`):

- **PEP 8** -- naming (`snake_case` files/functions, `PascalCase` classes, `UPPER_SNAKE_CASE` constants)
- **PEP 257 / NumPy-style** -- docstrings on all public functions and fixtures
- **PEP 484** -- type hints on all public function signatures
- **Imports** -- three groups separated by blank lines: standard library, third-party, GRDL internal

### What Not to Do

- Do not write ad-hoc GRDL wrappers. Call the public API directly.
- Do not test GRDL internals. Test observable behavior through public methods.
- Do not add tests that only check "did it load." Every assertion must validate a physical or structural property.
- Do not create utility modules. `conftest.py` is the only shared infrastructure.

## Writing Assertions

### Good Assertions (Validate Physical Properties)

```python
# Reflectance bounds for Landsat Surface Reflectance
assert chip.min() >= 0, "Negative reflectance values"
assert chip.max() <= 10000, "Exceeds SR scale factor ceiling"

# CRS validation
assert 32601 <= epsg <= 32660, f"UTM zone out of range: EPSG {epsg}"

# Complex SAR integrity
magnitude = np.abs(chip)
assert magnitude.std() > 0, "Zero variance -- likely fill data"
```

### Bad Assertions (Trivial or Meaningless)

```python
# Don't: just check it loaded
assert reader is not None

# Don't: check type without validating content
assert isinstance(metadata, dict)

# Don't: assert trivially true properties
assert chip.shape[0] > 0
```

## Running Tests

```bash
# Full suite (missing data files skip cleanly)
pytest tests/ -v

# Specific reader
pytest tests/test_io_geotiff.py -v

# By marker
pytest tests/ -m landsat
pytest tests/ -m integration
pytest tests/ -m "nitf and not slow"

# Skip all data-dependent tests
pytest tests/ -m "not requires_data"

# Parallel execution (requires pytest-xdist)
pytest tests/ -n auto --benchmark-disable
```

## Dependency Management

### Source of Truth: `pyproject.toml`

**`pyproject.toml` is the single source of truth** for all dependencies. All package metadata and dependencies are defined here.

### Keeping Files in Sync

Two files must be kept synchronized:

| File | Purpose | How to Update |
|------|---------|---------------|
| `pyproject.toml` | **Source of truth** — package metadata, all dependencies | Edit directly; this is the authoritative definition |
| `requirements.txt` (if it exists) | Development convenience — pinned versions for reproducible environments | `pip freeze > requirements.txt` after updating dependencies in `pyproject.toml` and installing |

**Note:** GRDL-TE is a **validation suite, not a published library**, so there is no `.github/workflows/publish.yml` or PyPI versioning requirement.

**Workflow:**
1. Update dependencies in `pyproject.toml` (add new packages, change versions, create/rename extras)
2. Install dependencies: `pip install -e ".[all,dev]"` (or appropriate extras for your work)
3. If `requirements.txt` exists in this project, regenerate it: `pip freeze > requirements.txt`
4. Commit both files

## Dependencies

**Required:**

| Package | Purpose |
|---------|---------|
| `grdl` | The library under test |
| `pytest>=7.0` | Test framework with marker support |
| `pytest-cov>=4.0` | Coverage reporting |
| `numpy>=1.21` | Array operations in assertions |
| `h5py>=3.0` | HDF5 format support |
| `rasterio>=1.3` | GeoTIFF/NITF format support |

**Optional (dev):**

| Package | Purpose |
|---------|---------|
| `pytest-benchmark>=4.0` | Performance benchmarks |
| `pytest-xdist>=3.0` | Parallel test execution |

## Adding a New Reader Test -- Checklist

1. **Select ONE representative dataset** -- prioritize open data, production quality, and global accessibility.
2. **Create** `data/<dataset>/README.md` with download instructions, file format specs, and licensing info.
3. **Add fixtures** to `tests/conftest.py`:
   - Session-scoped directory fixture (`<dataset>_data_dir`)
   - Data file fixture (`require_<dataset>_file`) using `require_data_file()`
4. **Create test file** `tests/test_io_<reader>.py` with the 3-level structure:
   - Level 1: Format validation (5+ tests)
   - Level 2: Data quality (5+ tests)
   - Level 3: Integration with `grdl.data_prep` (3+ tests)
5. **Register markers** in `pyproject.toml` under `[tool.pytest.ini_options]`.
6. **Add module-level `pytestmark`** with dataset marker, `requires_data`, and dependency skip conditions.
7. **Update** `README.md` data manifesto table and test counts.
8. **Verify** the test file runs cleanly with data present (all pass) and without data (all skip).

## Git Practices

Same as GRDL:

- Commit messages: imperative mood, one line, under 72 characters. Body if needed.
- One logical change per commit. Do not mix unrelated changes.
- Branch naming: `<type>/<short-description>` (e.g. `test/aster-reader`, `fix/sentinel2-safe-discovery`)