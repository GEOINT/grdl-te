# VIIRS VNP09 Test Data - HDF5Reader Validation

## Dataset

**Platform**: VIIRS/NPP (Suomi National Polar-orbiting Partnership)
**Product**: VNP09GA (Daily L2G Surface Reflectance)
**Format**: HDF5 (HDF-EOS5)
**Resolution**: 500m (selected bands)
**File Size**: ~100-300 MB
**Data Type**: int16 (scaled reflectance)

## Required File

One file is sufficient for testing:
- **Pattern**: `VNP09GA.A*.h5` or `VNP09GA.A*.hdf5`
- **Structure**: Hierarchical HDF5 with multiple Science Data Sets (SDS)
- **Recommended dataset**: `/HDFEOS/GRIDS/VIIRS_Grid_500m_2D/Data Fields/SurfReflect_I1_1` (Red, 500m)

## Download

### Option 1: LAADS DAAC (NASA Earthdata Account Required)

1. Visit: https://ladsweb.modaps.eosdis.nasa.gov
2. Register: https://urs.earthdata.nasa.gov/users/new (free)
3. Search: Product "VNP09GA", Collection 2, any recent date
4. Download single granule HDF5 file

## File Structure

Typical VIIRS HDF5 hierarchy:
```
VNP09GA.A2024015.h09v05.002.2024017041234.h5
└── HDFEOS/
    └── GRIDS/
        └── VIIRS_Grid_500m_2D/
            └── Data Fields/
                ├── SurfReflect_I1_1      # Red (600-680 nm), 500m
                ├── SurfReflect_I2_1      # NIR (846-885 nm), 500m
                ├── SurfReflect_I3_1      # Blue (478-498 nm), 500m
                └── ... (10+ datasets)
```

## File Properties

- **Dimensions**: 2400 × 2400 pixels (500m grid)
- **Swath size**: 1200km × 1200km
- **Projection**: Sinusoidal (same as MODIS)
- **Tile system**: hXXvYY format (e.g., h09v05)
- **Fill value**: -28672
- **Scale factor**: 0.0001 (multiply to get reflectance 0.0-1.0)

## Notes

- **VIIRS/NPP** (Suomi NPP) or **VIIRS/NOAA20** (VJ109GA) both work
- **Collection 2**: Version 002 (current, HDF5 format)
- **HDF5 native**: No conversion needed, h5py reads directly
- File naming: `VNP09GA.AYYYYDDD.hHHvVV.VVV.YYYYDDDHHMMSS.h5`
  - YYYYDDD = Year and day of year
  - hHHvVV = Tile coordinates
- VIIRS is MODIS's successor with improved calibration

## References

- VIIRS Surface Reflectance: https://lpdaac.usgs.gov/products/vnp09gav002/
- LAADS DAAC: https://ladsweb.modaps.eosdis.nasa.gov
- VIIRS Overview: https://www.earthdata.nasa.gov/sensors/viirs
