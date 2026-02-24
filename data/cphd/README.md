# CPHD Test Data - CPHDReader Validation

## Dataset

**Standard**: NGA Compensated Phase History Data (CPHD)
**Format**: CPHD 1.0 / 1.1 (binary with XML header)
**Data Type**: complex64 (phase history vectors)
**File Size**: ~100 MB - 2 GB (varies by collection)

## Required File

One file is sufficient for testing:
- **Pattern**: `*.cphd`
- **Structure**: Binary phase history with Per-Vector Parameters (PVP)
- **Content**: Raw SAR phase data before image formation

## You Already Have This Data

Copy any `.cphd` file from your SAR data collection:

```bash
# Recommended: Umbra CPHD (compact, well-formed)
cp /path/to/sar/from_duane/CPHD/2024-11-18-18-04-39_UMBRA-08_CPHD.cphd data/cphd/

# Alternative: Capella CPHD
cp /path/to/sar/cphd/Capella/2025/CAPELLA_C13_SP_CPHD_HH_20250826023518_20250826023527.cphd data/cphd/
```

One file is enough. Multiple files from different vendors (Umbra, Capella) provide broader coverage.

## File Properties

Typical CPHD file:
- **Phase vectors**: N vectors x M samples (complex64)
- **PVP fields**: Per-vector timing, position, velocity
- **Metadata**: XML header with collection geometry, waveform parameters
- **Frequency**: X-band (Umbra ~9.6 GHz, Capella ~9.65 GHz)
- **Polarization**: Single-pol (HH or VV typically)

## Data Characteristics

**Phase History Data**:
- Pre-image-formation radar returns
- Each vector corresponds to one transmitted pulse
- Required input for SAR Image Formation algorithms (PFA, RDA, FFBP)
- PVP contains precise platform position/velocity per pulse

**Relationship to Image Formation**:
- CPHDReader output feeds directly into `CollectionGeometry` and `PolarFormatAlgorithm`
- Image formation benchmarks depend on CPHD data being available

## CPHD Metadata

XML header contains:
- CollectionID (collector, platform, mode)
- Global (domain type, PVP types, signal arrays)
- Channel (parameters per receive channel)
- PVP (per-vector parameter definitions)
- SupportArray (optional auxiliary data)

Access via CPHDReader: `from grdl.IO.sar import CPHDReader`

## Notes

- **CPHD 1.0 vs 1.1**: Reader supports both versions
- **Phase-critical**: Data integrity is essential — do not truncate or compress
- **Large files**: Some collects exceed 1 GB; a small spotlight collect (~100 MB) is preferred for benchmarking
- **License**: Varies by provider. Umbra: CC BY 4.0. Capella: contact provider.

## References

- NGA CPHD Standard: https://nsgreg.nga.mil/doc/view?i=5381
- Umbra Open Data: https://registry.opendata.aws/umbra-open-data/
- Capella Open Data: https://www.capellaspace.com/community/open-data/
