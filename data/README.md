# GRDL-TE Test Data - Single-Source-of-Truth Strategy

This repository tests GRDL IO readers with **one representative dataset per reader**.

## Dataset Matrix

### IO Readers/Writers

| Reader | Platform / Standard | Directory | Format | Status |
|--------|-------------------|-----------|--------|--------|
| **GeoTIFFReader** | Landsat 8/9 | `landsat/` | COG | Available |
| **HDF5Reader** | VIIRS VNP09 | `viirs/` | HDF5 | Available |
| **JP2Reader** | Sentinel-2 | `sentinel2/` | JPEG2000 | Available |
| **SICDReader** | Umbra SICD | `umbra/` | NITF+SICD | Available |
| **NITFReader** | Umbra SICD | `umbra/` | NITF | Available |
| **CPHDReader** | Umbra/Capella | `cphd/` | CPHD | Copy from SAR store |
| **CRSDReader** | NGA CRSD | `crsd/` | CRSD | Copy from SAR store |
| **SIDDReader** | NGA SIDD | `sidd/` | NITF+SIDD | Needs acquisition |
| **Sentinel1SLCReader** | Sentinel-1A/C | `sentinel1/` | SAFE | Copy from SAR store |
| **Sentinel2Reader** | Sentinel-2 | `sentinel2/` | SAFE+JP2 | Available |
| **TerraSARReader** | TerraSAR-X/TDX | `terrasar/` | COSAR/XML | Copy from SAR store |
| **ASTERReader** | Terra/ASTER | `aster/` | HDF-EOS | Needs acquisition |
| **BIOMASSL1Reader** | ESA BIOMASS | `biomass/` | GeoTIFF | Needs acquisition |
| **SICDWriter** | (roundtrip) | `umbra/` | NITF+SICD | Available |
| **NITFWriter** | (synthetic) | — | NITF | N/A (synthetic) |

### Elevation & Geolocation

| Component | Data Source | Directory | Format | Status |
|-----------|-----------|-----------|--------|--------|
| **DTEDElevation** | SRTM/USGS | `dted/` | DTED (.dt1/.dt2) | Needs acquisition |
| **GeoTIFFDEM** | Copernicus/SRTM | `dem/` | GeoTIFF | Needs acquisition |
| **GeoidCorrection** | NGA EGM96 | `geoid/` | PGM | Needs acquisition |
| **Sentinel1SLCGeolocation** | Sentinel-1 | `sentinel1/` | (paired with reader) | Copy from SAR store |

## Why These Datasets?

**Landsat 8/9** - GeoTIFF validation
- Native Cloud-Optimized GeoTIFF (COG) format
- Tests: COG tiling, overviews, CRS, affine transforms

**VIIRS VNP09** - HDF5 validation
- Native HDF5 format (no HDF4 conversion needed)
- Tests: Dataset discovery, multi-SDS handling

**Sentinel-2** - JPEG2000 + multispectral validation
- Highest-resolution free multispectral (10m)
- Also validates Sentinel2Reader (SAFE archive parsing)

**Umbra SICD** - NITF + SICD + SAR validation
- NITF with embedded SICD XML metadata
- Also used for SICDWriter roundtrip and SICDGeolocation

**CPHD** - Phase history validation
- Required input for SAR image formation algorithms (PFA, RDA, FFBP)
- Umbra and Capella sources available

**CRSD** - Radar signal validation
- NGA compensated radar signal standard

**Sentinel-1 SLC** - Copernicus SAR validation
- Most-used SAR dataset globally
- Tests: SAFE parsing, burst timing, dual-pol channels

**TerraSAR-X** - Commercial X-band SAR validation
- COSAR binary format (unique to TSX/TDX)
- Tests: Binary row-by-row reader, auto-detection via `open_sar()`

**ASTER L1T** - Thermal IR validation
- HDF-EOS format (different from plain HDF5)

**BIOMASS L1** - P-band SAR validation
- ESA's newest SAR mission (launched 2025)

**DTED / DEM / Geoid** - Elevation model validation
- DTED: NGA/USGS tile-based elevation
- GeoTIFF DEM: Copernicus/SRTM raster elevation
- EGM96 geoid: MSL-to-HAE conversion grid

## Directory Structure

```
data/
├── README.md             # This file
├── landsat/              # GeoTIFFReader — Landsat COG
├── viirs/                # VIIRSReader / HDF5Reader — VIIRS HDF5
├── sentinel2/            # JP2Reader / Sentinel2Reader — Sentinel-2
├── umbra/                # SICDReader / NITFReader — Umbra SICD NITF
├── cphd/                 # CPHDReader — CPHD phase history
├── crsd/                 # CRSDReader — CRSD radar signal
├── sidd/                 # SIDDReader — SIDD derived data
├── sentinel1/            # Sentinel1SLCReader — Sentinel-1 SLC SAFE
├── terrasar/             # TerraSARReader — TerraSAR-X COSAR/GeoTIFF
├── aster/                # ASTERReader — ASTER L1T HDF-EOS
├── biomass/              # BIOMASSL1Reader — BIOMASS L1 GeoTIFF
├── dted/                 # DTEDElevation — DTED tiles
├── dem/                  # GeoTIFFDEM — Copernicus/SRTM DEM
└── geoid/                # GeoidCorrection — EGM96 geoid grid
```

## Data Handling

**Not committed to repository**: Actual data files are `.gitignore`'d. Each directory contains only:
- `README.md` with download/copy instructions
- Placeholder `.gitkeep` (optional)

**Graceful degradation**: Tests skip when data files are missing, with clear messages pointing to READMEs.

**Minimal dataset**: One representative file per reader is sufficient. No need to download entire scenes or time series.

## Quick Start

1. **Choose a reader to test** (e.g., CPHDReader)
2. **Navigate to directory** (e.g., `data/cphd/`)
3. **Read README.md** for download/copy instructions
4. **Place data file in directory**
5. **Run tests**: `pytest tests/validation/ -v -m cphd`

## Integration Testing

Tests go beyond simple `reader.load()` validation to verify data works with GRDL utilities:

**Phase 1 (Complete)**: Filters, Detection, Decomposition, Ortho, CoRegistration
- CFAR detection, DualPolHAlpha, OrthoPipeline, ProjectiveCoRegistration

**Phase 2 (Current)**: IO Readers/Writers, SAR Processing, Elevation
- Real-data IO benchmarks, MultilookDecomposition, CSI, elevation models

**Phase 3 (Planned)**: SAR Image Formation, Interpolation
- PFA, RDA, FFBP (require CPHD data), Interpolator benchmarks

## Running Tests

```bash
# Install grdl-te
pip install -e .

# Run all tests (skips if data missing)
pytest tests/ -v

# Run specific reader
pytest tests/validation/ -v -m cphd

# Run only tests with data available
pytest tests/ -m requires_data -v

# Run benchmark suite (all groups)
python -c "from grdl_te.benchmarking.suite import run_suite; run_suite(size='small')"

# Skip slow tests
pytest tests/ -m "not slow" -v
```

## Adding New Datasets

If you want to test a different platform:

1. Place file in appropriate format directory
2. Update fixture patterns in `tests/validation/conftest.py` if needed
3. Tests should auto-discover via glob patterns

**Note**: The single-source-of-truth approach means we prioritize depth over breadth. Additional datasets are welcome for experimentation but not required for CI/validation.

## License

Test data usage follows respective data provider licenses:
- **Landsat**: Public domain (USGS)
- **VIIRS**: Public domain (NASA/NOAA)
- **Sentinel-1/2**: Free and open (Copernicus)
- **Umbra**: Open data (CC BY 4.0)
- **Capella**: Contact provider
- **TerraSAR-X**: DLR scientific use license
- **ASTER**: Public domain (NASA)
- **BIOMASS**: Free and open (Copernicus)
- **SRTM/DTED**: Public domain (USGS/NGA)
- **EGM96**: Public domain (NGA)

See individual READMEs for specific licensing terms.
