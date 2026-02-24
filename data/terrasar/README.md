# TerraSAR-X / TanDEM-X Test Data - TerraSARReader Validation

## Dataset

**Platform**: TerraSAR-X (TSX1) / TanDEM-X (TDX1) (DLR/Airbus)
**Product**: SSC (Single-look Slant-range Complex) or MGD/GEC/EEC (detected)
**Format**: COSAR binary (SSC) or GeoTIFF (detected products), with XML annotation
**Resolution**: ~1 m (Spotlight), ~3 m (StripMap), ~16 m (ScanSAR)
**Data Type**: complex32 (SSC/COSAR) or float32 (detected GeoTIFF)

## Required File

One TSX product directory is sufficient for testing:
- **Pattern**: `TSX1_SAR__*` or `TDX1_SAR__*` (product directory)
- **Structure**: Directory with XML annotation, COSAR binary or GeoTIFF imagery
- **Preferred**: SSC product (tests the COSAR binary reader path)

## You Already Have This Data

Copy a TerraSAR-X product directory from your collection:

```bash
cp -r /path/to/sar/jasonphd/TSX/<product_directory> data/terrasar/
```

## File Structure

```
TSX1_SAR__SSC______SM_S_SRA_20100101T120000_20100101T120001/
├── TSX1_SAR__SSC_*.xml              # Main annotation XML
├── IMAGEDATA/
│   └── IMAGE_*.cos                   # COSAR binary (complex SLC)
├── ANNOTATION/
│   ├── GEOREF.xml                    # Geolocation grid
│   └── CALDATA/
│       └── CALIBRATION.xml           # Radiometric calibration
└── PREVIEW/
    └── BROWSE.tif                    # Quick-look
```

## File Properties

Typical TerraSAR-X SSC (Spotlight):
- **Dimensions**: ~10000 x 10000 pixels (varies by scene)
- **Data type**: complex32 (COSAR binary format, interleaved I/Q int16)
- **Frequency**: X-band (9.65 GHz)
- **Polarization**: Single-pol (HH or VV) or Dual-pol (HH/VV)
- **Orbit**: Sun-synchronous, 514 km altitude, 11-day repeat

## Data Characteristics

**COSAR Binary Format**:
- TerraSAR-X proprietary binary format for SSC data
- Row-by-row storage with per-row headers
- Complex int16 I/Q pairs
- Performance-critical: `_read_cosar_chip()` uses per-row seek-and-read

**Detected Products (MGD/GEC/EEC)**:
- Standard GeoTIFF format
- Georeferenced (GEC) or map-projected (EEC)
- Read through the GeoTIFF backend path in TerraSARReader

**Metadata**:
- Rich XML annotation with orbit state vectors, Doppler info, calibration
- Geolocation grid points for terrain correction
- 9 metadata dataclasses in `grdl.IO.models.terrasar`

## Notes

- **SSC preferred**: The COSAR binary path is the unique/complex code path to benchmark
- **Product naming**: Directories start with `TSX1_` or `TDX1_` followed by mode identifiers
- **DLR license**: TerraSAR-X data is available for scientific use via DLR proposal process
- **`open_sar()` routing**: `open_sar()` auto-detects TSX/TDX products by directory naming convention

Access via TerraSARReader: `from grdl.IO.sar import TerraSARReader, open_terrasar`

## References

- DLR TerraSAR-X: https://www.dlr.de/hr/en/desktopdefault.aspx/tabid-2317/
- Airbus TerraSAR-X: https://www.intelligence-airbusds.com/imagery/constellation/terrasar-x/
- Science Service System: https://sss.terrasar-x.dlr.de/
