# Sentinel-2 Test Data - JP2Reader Validation

## Dataset

**Platform**: Sentinel-2A/2B/2C
**Product**: Level-2A (Surface Reflectance) or Level-1C (TOA Reflectance)
**Format**: JPEG2000 with 15-bit encoding
**Recommended Band**: B04 (Red, 10m resolution)
**File Size**: ~100-200 MB per 10m band
**Data Type**: uint16 (15-bit: 0-32767)

## Required File

One file is sufficient for testing:
- **Pattern**: `T*_*_B04*.jp2` (standalone) or `S2*.SAFE/GRANULE/*/IMG_DATA/R10m/*_B04_10m.jp2`
- **CRS**: UTM (EPSG:326XX or EPSG:327XX)
- **NoData**: 0

## Download

### Option 1: Copernicus Data Space (Native SAFE/JP2)

1. Visit: https://dataspace.copernicus.eu
2. Register for free account
3. Search: Product type "S2MSI2A" (Level-2A), cloud cover < 20%
4. Download: Standalone band `T*_B04_10m.jp2` OR full SAFE archive
5. If SAFE: Extract `GRANULE/*/IMG_DATA/R10m/*_B04_10m.jp2`

### Option 2: ESA Scihub Archive

1. Visit: https://scihub.copernicus.eu/dhus/#/home
2. Register (free)
3. Search: "Sentinel-2"
4. Download SAFE archive
5. Extract: `S2*.SAFE/GRANULE/*/IMG_DATA/R10m/*_B04_10m.jp2`

## File Structure

### Standalone JP2
```
T10SEG_20240115T184719_B04_10m.jp2
```

### Within SAFE Archive
```
S2A_MSIL2A_20240115T184719_N0510_R070_T10SEG_20240115T201234.SAFE/
└── GRANULE/
    └── L2A_T10SEG_A044567_20240115T184719/
        └── IMG_DATA/
            └── R10m/
                └── T10SEG_20240115T184719_B04_10m.jp2  ← Use this
```

## File Properties

- **Dimensions**: 10980 × 10980 pixels
- **Scene size**: ~100km × 100km (MGRS tile)
- **Resolution**: 10m
- **Projection**: UTM
- **15-bit encoding**: Values 0-32767 (not standard 16-bit 0-65535)
- **Reflectance range**: Typically 0-10000 (× 10000 scale factor)

## 15-bit Encoding

Sentinel-2 uses **15-bit unsigned integers** in 16-bit containers:
- Max value: 32767 (2^15 - 1)
- Encoding challenge for some JPEG2000 decoders
- glymur backend recommended for correct decoding

## Band Options

Any 10m band works for testing:

| Band | Wavelength | Resolution | Notes |
|------|-----------|------------|-------|
| B02 | 490 nm (Blue) | 10m | Water, atmosphere |
| B03 | 560 nm (Green) | 10m | Vegetation, water |
| **B04** | **665 nm (Red)** | **10m** | **Recommended** |
| B08 | 842 nm (NIR) | 10m | Vegetation (high values) |

## Notes

- **Tile system**: MGRS (Military Grid Reference System), 100km × 100km
- **Three resolutions**: 10m, 20m, 60m bands
- **SAFE format**: Standard Archive Format for Europe (directory structure)
- **Sentinel-2C**: Launched Sep 2024, replaced 2A operationally Jan 2025

## References

- Sentinel-2 MSI: https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-2
- Copernicus Data Space: https://dataspace.copernicus.eu
- SAFE Format: https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/data-formats
