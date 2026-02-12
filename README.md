# GRDL-TE: High-Fidelity Evaluation Suite

Validates GRDL IO readers against real-world remote sensing data from operational satellite platforms. Every test evaluates a specific transformation, extraction, or integration -- if data is present it must pass; if data is missing it skips with a clear message.

## Design Principles

| Principle | Rule |
|-----------|------|
| **No Pass by Default** | A test only passes when the data is present AND the code functions correctly. |
| **Missing Data = Skipped** | If the required data file is absent, the test is skipped with a download instruction. |
| **Broken Code = Failed** | If data is present but the reader or utility malfunctions, the test fails hard. |
| **Meaningful Assertions** | Every test validates a specific property (reflectance bounds, CRS projection, complex integrity). No tests that only check "did it load." |

## Repository Structure

```
grdl-te/
├── data/                        # Real-world data files (not committed to git)
│   ├── README.md                # Data strategy documentation
│   ├── landsat/                 # Landsat 8/9 COG (GeoTIFFReader)
│   │   └── README.md            # Download instructions
│   ├── viirs/                   # VIIRS VNP09GA HDF5 (HDF5Reader)
│   │   └── README.md
│   ├── sentinel2/               # Sentinel-2 JP2 (JP2Reader)
│   │   └── README.md
│   └── umbra/                   # Umbra SICD NITF (NITFReader)
│       └── README.md
├── tests/
│   ├── conftest.py              # Shared fixtures + graceful skip logic
│   ├── test_io_geotiff.py       # GeoTIFFReader (Landsat) -- 15 tests
│   ├── test_io_hdf5.py          # HDF5Reader (VIIRS) -- 15 tests
│   ├── test_io_jpeg2000.py      # JP2Reader (Sentinel-2) -- 16 tests
│   └── test_io_nitf.py          # NITFReader (Umbra SICD) -- 15 tests
├── pyproject.toml               # Package configuration + markers
├── LICENSE                      # MIT License
└── README.md                    # This file
```

## Test Architecture

All test files follow a **3-level validation structure**:

### Level 1: Format Validation
- Metadata extraction (format, rows, cols, dtype, CRS, bands)
- Shape and dtype consistency with metadata
- Chip reads from image center (validates data exists, not just fill)
- Full image reads (with size guards)
- Context manager resource cleanup

### Level 2: Data Quality
- **CRS/projection validation** (UTM zone ranges for Landsat/Sentinel-2)
- **NoData handling** (masking, valid pixel statistics, variance)
- **Value range bounds** (reflectance scales, 15-bit encoding ceilings)
- **Format-specific features** (COG tiling/overviews, HDF-EOS structure, complex magnitude/phase, SAR speckle statistics)

### Level 3: Integration
- **ChipExtractor**: Uniform chip partitioning with bounds validation
- **Normalizer**: MinMax, Z-score, and percentile normalization
- **Tiler**: Overlapping tile grids with stride control
- **Pipelines**: End-to-end workflows (chip -> normalize -> validate batch)

## Data Manifesto

Each reader requires **exactly one file**. Tests skip gracefully if the file is missing.

| Reader | Directory | File Pattern | Format | Size | Source |
|--------|-----------|-------------|--------|------|--------|
| **GeoTIFFReader** | `data/landsat/` | `LC0[89]*_SR_B*.TIF` | Cloud-Optimized GeoTIFF (uint16) | 50-150 MB | [USGS EarthExplorer](https://earthexplorer.usgs.gov) |
| **HDF5Reader** | `data/viirs/` | `V?P09GA*.h5` | HDF5/HDF-EOS5 (int16) | 100-300 MB | [LAADS DAAC](https://ladsweb.modaps.eosdis.nasa.gov) |
| **JP2Reader** | `data/sentinel2/` | `S2*.SAFE/.../R10m/*_B04_10m.jp2` | JPEG2000 15-bit (uint16) | 100-200 MB | [Copernicus Data Space](https://dataspace.copernicus.eu) |
| **NITFReader** | `data/umbra/` | `*.nitf` | NITF + SICD XML (complex64) | 50-500 MB | [Umbra Open Data (AWS S3)](https://umbra-open-data-catalog.s3.amazonaws.com/index.html) |

### Download Quick Start

**Landsat** -- Any single Landsat 8 or 9 Collection 2 Surface Reflectance band file:
```
data/landsat/LC09_L2SP_001028_20260210_20260211_02_T2_SR_B4.TIF
```
Source: [USGS EarthExplorer](https://earthexplorer.usgs.gov) (free account required)

**VIIRS** -- Any VNP09GA daily surface reflectance granule:
```
data/viirs/VNP09GA.A2026004.h17v14.002.2026005131408.h5
```
Source: [LAADS DAAC](https://ladsweb.modaps.eosdis.nasa.gov) (NASA Earthdata login required)

**Sentinel-2** -- A complete SAFE archive (or standalone B04 JP2):
```
data/sentinel2/S2B_MSIL2A_20260204T170409_N0512_R069_T15RTP_20260204T223135.SAFE/
```
Source: [Copernicus Data Space](https://dataspace.copernicus.eu) (free account required)

**Umbra** -- Any Umbra SICD spotlight NITF:
```
data/umbra/2025-10-25-20-00-44_UMBRA-10_SICD.nitf
```
Source: [Umbra Open Data (S3)](https://umbra-open-data-catalog.s3.amazonaws.com/index.html) (no login, CC BY 4.0)

Each `data/<dataset>/README.md` contains detailed download instructions and file format specifications.

## Running Tests

```bash
# Activate environment
conda activate grdl

# Run all tests (missing data files skip cleanly)
pytest tests/ -v

# Run tests for a specific reader
pytest tests/test_io_geotiff.py -v      # Landsat
pytest tests/test_io_hdf5.py -v         # VIIRS
pytest tests/test_io_jpeg2000.py -v     # Sentinel-2
pytest tests/test_io_nitf.py -v         # Umbra

# Run by marker
pytest tests/ -m landsat                # All Landsat tests
pytest tests/ -m viirs                  # All VIIRS tests
pytest tests/ -m integration            # Only Level 3 integration tests
pytest tests/ -m "nitf and not slow"    # NITF tests, skip slow ones

# Skip all data-dependent tests
pytest tests/ -m "not requires_data"
```

### Expected Output

**With all data present:**
```
tests/test_io_geotiff.py    15 passed
tests/test_io_hdf5.py       15 passed
tests/test_io_jpeg2000.py   16 passed
tests/test_io_nitf.py       15 passed
```

**With no data present:**
```
tests/test_io_geotiff.py    15 skipped
tests/test_io_hdf5.py       15 skipped
tests/test_io_jpeg2000.py   16 skipped
tests/test_io_nitf.py       15 skipped
```

Each skip message points to the exact file pattern and README with download instructions.

## Test Markers

| Marker | Purpose |
|--------|---------|
| `landsat` | Landsat 8/9 tests (GeoTIFFReader) |
| `viirs` | VIIRS VNP09GA tests (HDF5Reader) |
| `sentinel2` | Sentinel-2 tests (JP2Reader) |
| `nitf` | Umbra SICD tests (NITFReader) |
| `requires_data` | Test requires real data files in `data/` |
| `slow` | Long-running test (large file reads, full pipelines) |
| `integration` | Level 3 tests (ChipExtractor, Normalizer, Tiler workflows) |

## Adding a New Reader Test

1. **Select ONE representative dataset** -- prioritize open data, production quality
2. **Create** `data/<dataset>/README.md` with download instructions
3. **Add fixture** to `tests/conftest.py` using `require_data_file()`
4. **Create test file** `tests/test_io_<reader>.py` with 3-level structure
5. **Register markers** in `pyproject.toml`

## License

MIT License -- see [LICENSE](LICENSE) for details.
