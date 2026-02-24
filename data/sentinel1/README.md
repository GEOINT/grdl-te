# Sentinel-1 SLC Test Data - Sentinel1SLCReader Validation

## Dataset

**Platform**: Sentinel-1A / Sentinel-1C (ESA Copernicus)
**Product**: IW SLC (Interferometric Wide Swath, Single Look Complex)
**Format**: SAFE archive (ZIP or directory) with TIFF measurement data
**Resolution**: ~5 m (range) x ~20 m (azimuth)
**File Size**: ~4-8 GB per IW SLC product
**Data Type**: complex (int16 I/Q pairs in TIFF)

## Required File

One SAFE archive is sufficient for testing:
- **Pattern**: `S1?_IW_SLC__*.SAFE` (directory) or `*.SAFE.zip`
- **Structure**: SAFE format with annotation XML, calibration, measurement TIFFs
- **Polarization**: Dual-pol VV+VH (most common) or HH+HV

## You Already Have This Data

Copy any Sentinel-1 SLC SAFE directory from your collection:

```bash
# Recommended: Recent S1A IW SLC (compact, well-formed)
cp -r /path/to/sar/from_duane/SICD/ships/S1A_IW_SLC__1SDV_20260201T142455_20260201T142522_063027_07E8C0_81A5.SAFE data/sentinel1/

# Alternative: InSAR pair (if you want burst timing tests)
cp -r /path/to/sar/from_duane/SLC/insar_pairs/pair_001_RON57_ASC_11d/S1A_IW_SLC__1SDV_20241203T141658_20241203T141725_056829_06FA36_5E2F.SAFE data/sentinel1/
```

One SAFE archive is enough. For InSAR testing, a pair is useful.

## File Structure

```
S1A_IW_SLC__1SDV_20260201T142455_*.SAFE/
├── manifest.safe                    # Product manifest
├── annotation/
│   ├── s1a-iw1-slc-vv-*.xml       # Swath 1, VV annotation
│   ├── s1a-iw1-slc-vh-*.xml       # Swath 1, VH annotation
│   ├── s1a-iw2-slc-vv-*.xml       # Swath 2, VV
│   ├── ... (3 swaths x 2 pols)
│   ├── calibration/                 # Radiometric calibration LUTs
│   └── rfi/                         # RFI annotations (newer products)
├── measurement/
│   ├── s1a-iw1-slc-vv-*.tiff      # Complex SLC data (int16 I/Q)
│   ├── s1a-iw1-slc-vh-*.tiff
│   └── ... (3 swaths x 2 pols)
├── preview/
│   └── quick-look.png
└── support/
    └── *.xsd                        # XML schemas
```

## File Properties

Typical IW SLC product:
- **Swaths**: 3 (IW1, IW2, IW3) covering ~250 km
- **Bursts**: 9-12 per swath
- **Burst dimensions**: ~1500 (range) x ~22000 (azimuth) pixels
- **Polarization**: Dual-pol VV+VH (S1SDV) or Single-pol VV (S1SSV)
- **Frequency**: C-band (5.405 GHz)
- **Orbit**: Sun-synchronous, 693 km altitude, 12-day repeat

## Data Characteristics

**Complex SLC**:
- In-phase (I) and Quadrature (Q) stored as int16 pairs
- Zero-Doppler focused, slant-range geometry
- Burst-mode acquisition (TOPS technique)

**Paired Testing**:
- Sentinel1SLCReader should be tested together with Sentinel1SLCGeolocation
- Burst timing model validation requires annotation XML parsing

## Notes

- **Large files**: IW SLC products are 4-8 GB. A single swath TIFF is ~500 MB.
- **SAFE format**: Standardized ESA archive format (not a single file)
- **S1A vs S1C**: Both use identical format. S1C replaced S1B in 2024.
- **License**: Free and open (Copernicus data policy)

Access via Sentinel1SLCReader: `from grdl.IO.sar import Sentinel1SLCReader`

## References

- Copernicus Data Space: https://dataspace.copernicus.eu
- Sentinel-1 User Guide: https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-1-sar
- ASF DAAC (Alaska mirror): https://search.asf.alaska.edu
