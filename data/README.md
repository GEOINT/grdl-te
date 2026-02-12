# GRDL-TE Test Data - Single-Source-of-Truth Strategy

This repository tests GRDL IO readers with **one representative dataset per reader**.

## Dataset Matrix

| Reader | Platform | Directory | Format | Download |
|--------|----------|-----------|--------|----------|
| **GeoTIFFReader** | Landsat 8/9 | `landsat/` | COG | [AWS Open Data](https://registry.opendata.aws/landsat-8/) |
| **HDF5Reader** | VIIRS VNP09 | `viirs/` | HDF5 | [LAADS DAAC](https://ladsweb.modaps.eosdis.nasa.gov) |
| **JP2Reader** | Sentinel-2 | `sentinel2/` | JPEG2000 | [AWS](https://registry.opendata.aws/sentinel-2/) / [Copernicus](https://dataspace.copernicus.eu) |
| **NITFReader** | Umbra SICD | `umbra/` | NITF+SICD | [AWS Open Data](https://registry.opendata.aws/umbra-open-data/) |

## Why These Datasets?

**Landsat 8/9** - GeoTIFF validation
- Workhorse of civilian Earth observation
- Native Cloud-Optimized GeoTIFF (COG) format
- Free, globally accessible, 40+ year archive
- Tests: COG tiling, overviews, CRS, affine transforms

**VIIRS VNP09** - HDF5 validation
- MODIS successor with improved calibration
- Native HDF5 format (no HDF4 conversion needed)
- Hierarchical dataset navigation tests
- Tests: Dataset discovery, multi-SDS handling

**Sentinel-2** - JPEG2000 validation
- Highest-resolution free multispectral (10m)
- Tests 15-bit encoding in 16-bit container
- Global coverage, active mission
- Tests: JP2 compression, non-standard bit depths

**Umbra SICD** - NITF validation
- Open SAR data from commercial provider
- NITF with embedded SICD XML metadata
- Complex-valued SAR imagery
- Tests: NITF TRE parsing, complex dtypes, SAR metadata

## Directory Structure

```
data/
├── README.md           # This file
├── landsat/
│   ├── README.md       # Landsat download instructions
│   └── LC08_*_SR_B*.TIF
├── viirs/
│   ├── README.md       # VIIRS download instructions
│   └── VNP09GA.A*.h5
├── sentinel2/
│   ├── README.md       # Sentinel-2 download instructions
│   └── T*_B04.jp2 or S2*.SAFE/
└── umbra/
    ├── README.md       # Umbra download instructions
    └── *.nitf
```

## Data Handling

**Not committed to repository**: Actual data files are `.gitignore`'d. Each directory contains only:
- `README.md` with download instructions
- Placeholder `.gitkeep` (optional)

**Graceful degradation**: Tests skip when data files are missing, with clear messages pointing to READMEs.

**Minimal dataset**: One representative file per reader is sufficient. No need to download entire scenes or time series.

## Quick Start

1. **Choose a reader to test** (e.g., GeoTIFF)
2. **Navigate to directory** (e.g., `data/landsat/`)
3. **Read README.md** for download instructions
4. **Download one file** (e.g., single Landsat band)
5. **Place in directory**
6. **Run tests**: `pytest tests/test_io_geotiff.py -v`

## Integration Testing

Tests go beyond simple `reader.load()` validation to verify data works with GRDL utilities:

**Phase 1 (Current)**: Data Preparation
- `ChipExtractor` - Region extraction for ML pipelines
- `Normalizer` - Intensity normalization
- `Tiler` - Overlapping tile grids

**Phase 2 (Future)**: Processing & Display
- `ToDecibels`, `PercentileStretch` - Display enhancement
- `Geolocation` - Coordinate transforms

**Phase 3 (Future)**: Advanced SAR
- `SublookDecomposition` - SAR analysis
- `FeatureMatchCoRegistration` - Multi-temporal alignment

## Running Tests

```bash
# Install grdl-te
pip install -e .

# Run all tests (skips if data missing)
pytest tests/ -v

# Run specific reader
pytest tests/test_io_geotiff.py -v

# Run only tests with data available
pytest tests/ -m requires_data -v

# Skip slow tests
pytest tests/ -m "not slow" -v
```

## Adding New Datasets

If you want to test a different platform:

1. Place file in appropriate format directory
2. Update fixture patterns in `tests/conftest.py` if needed
3. Tests should auto-discover via glob patterns

**Note**: The single-source-of-truth approach means we prioritize depth over breadth. Additional datasets are welcome for experimentation but not required for CI/validation.

## License

Test data usage follows respective data provider licenses:
- **Landsat**: Public domain (USGS)
- **VIIRS**: Public domain (NASA/NOAA)
- **Sentinel-2**: Free and open (Copernicus)
- **Umbra**: Open data (CC BY 4.0)

See individual READMEs for specific licensing terms.