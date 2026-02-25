# Geoid Test Data - GeoidCorrection Validation

## Dataset

**Standard**: EGM96 (Earth Gravitational Model 1996)
**Format**: PGM (Portable GrayMap) binary grid
**Resolution**: 15 arc-minutes (~28 km)
**Data Type**: float (geoid undulation in meters)
**File Size**: ~2.1 MB

## Required File

One file:
- **Filename**: `egm96-15.pgm` (15 arc-minute EGM96 grid)
- **Alternatives**: `egm96-5.pgm` (5 arc-minute, ~18 MB) for higher precision
- **Content**: Global geoid undulation grid (MSL - WGS84 ellipsoid separation)

## How to Acquire

### Option 1: NGA Direct Download

1. Visit: https://earth-info.nga.mil/index.php?dir=wgs84&action=wgs84#tab_egm96
2. Download: EGM96 grid file
3. Convert to PGM format if needed

### Option 2: PROJ Data Grid

If you have PROJ installed:
```bash
# PROJ may already have the geoid grid
projinfo --searchpaths
# Look for egm96_15.gtx or similar in the PROJ data directory
```

## File Properties

EGM96-15 PGM grid:
- **Coverage**: Global (90N to 90S, 0E to 360E)
- **Grid spacing**: 15 arc-minutes (0.25 degrees)
- **Grid size**: 721 x 1441 cells
- **Values**: Geoid undulation in meters (range: approximately -110 to +90 m)
- **Format**: PGM binary (P5) with metadata header

## Data Characteristics

**Geoid Undulation**:
- Difference between MSL (Mean Sea Level) and WGS84 ellipsoid height
- `Height_MSL = Height_HAE - Geoid_Undulation`
- Essential for converting between ellipsoidal (GPS/SAR) and orthometric (map) heights

**Known Values for Validation**:
- Equator/Greenwich (0N, 0E): ~17 m undulation
- Indian Ocean minimum: ~-106 m
- Papua New Guinea maximum: ~+85 m

**Testing Focus**:
- PGM file parsing
- Bilinear interpolation between grid nodes
- Batch vectorized queries (10,000+ lat/lon points)
- HAE-to-MSL and MSL-to-HAE conversions
- Integration with elevation models (DTED, GeoTIFFDEM)

## Notes

- **EGM96 vs EGM2008**: EGM96 (15 arc-min) is simpler and sufficient for most SAR applications. EGM2008 (2.5 arc-min) is more precise but larger.
- **Small file**: At ~2 MB, this is the smallest test data requirement.
- **Always needed**: Any workflow converting between HAE and MSL needs geoid correction.
- **License**: Public domain (NGA)

Access via GeoidCorrection: `from grdl.geolocation.elevation import GeoidCorrection`

## References

- EGM96 Model: https://earth-info.nga.mil/index.php?dir=wgs84&action=wgs84#tab_egm96
- GeographicLib Geoids: https://geographiclib.sourceforge.io/C++/doc/geoid.html
- NGA WGS84/EGM: https://nsgreg.nga.mil
