# Umbra SICD Test Data - NITFReader Validation

## Dataset

**Platform**: Umbra SAR
**Product**: Spotlight SAR imagery (SICD format)
**Format**: NITF 2.1 with embedded SICD XML metadata
**Resolution**: ~25 cm (spotlight mode)
**File Size**: ~50-500 MB
**Data Type**: complex64 (complex-valued SAR)

## Required File

One file is sufficient for testing:
- **Pattern**: `*.nitf` or `*.ntf`
- **Structure**: NITF with SICD XML in TRE (Tagged Record Extension)
- **Geometry**: Slant-plane (not orthorectified)

## Download

### Option 1: AWS S3 (No Sign-in Required)

```bash
# List available collections
aws s3 ls s3://umbra-open-data-catalog/ --no-sign-request

# Download a SICD file (replace with actual path)
aws s3 cp s3://umbra-open-data-catalog/sar-data/tasks/[TASK_ID]/[FILE].nitf \
    data/umbra/ --no-sign-request
```

Browse: https://registry.opendata.aws/umbra-open-data/

### Option 2: Web Interface

1. Visit: https://umbra-open-data-catalog.s3.amazonaws.com/index.html
2. Browse available collections (urban areas, events, demonstrations)
3. Download any `.nitf` file

## File Properties

Typical Umbra SICD:
- **Dimensions**: ~4096 × 4096 pixels (varies by collect)
- **Data type**: complex64 (magnitude + phase)
- **Frequency**: X-band (~9.6 GHz)
- **Polarization**: Single-pol (VV or HH)
- **Orbit**: Sun-synchronous LEO (~500 km altitude)

## Data Characteristics

**Complex-valued SAR**:
- Each pixel: `value = magnitude × exp(i × phase)`
- Magnitude: backscatter intensity
- Phase: relative distance information

**Collect Modes**:
- **Spotlight** (recommended): Highest resolution (~25 cm), smaller scenes
- **Stripmap**: Lower resolution (~50 cm), larger coverage

## SICD Metadata

SICD XML embedded in NITF contains:
- Collection info (time, platform, sensor)
- Image data (dimensions, pixel type)
- GeoData (scene center point, earth model)
- Grid (image plane definition)
- Radar collection (waveform, frequency, polarization)

Access via SICDReader: `from grdl.IO import SICDReader`

## Notes

- **License**: CC BY 4.0 (attribution required)
- **File naming**: Varies by collection, typically `UMBRA_[DATE]_[LOCATION]_[MODE].nitf`
- **Geometry**: Slant-plane requires geolocation model for ground projection

## Data Attribution

```
Data provided by Umbra Lab Inc.
Source: Umbra Open Data Program
License: CC BY 4.0
```

## References

- Umbra Open Data: https://registry.opendata.aws/umbra-open-data/
- SICD Standard: https://github.com/ngageoint/six-library
- Umbra Docs: https://docs.umbra.space/
