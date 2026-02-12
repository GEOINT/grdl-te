# Landsat 8/9 Test Data - GeoTIFFReader Validation

## Dataset

**Platform**: Landsat 8/9 Collection 2
**Product**: Level-2 Surface Reflectance
**Format**: Cloud-Optimized GeoTIFF (COG)
**Recommended Band**: B4 (Red, 30m resolution)
**File Size**: ~50-150 MB per band
**Data Type**: uint16 (surface reflectance scaled 0-10000)

## Required File

One file is sufficient for testing:
- **Pattern**: `LC08_*_SR_B4.TIF` or `LC09_*_SR_B4.TIF`
- **CRS**: UTM (EPSG:326XX or EPSG:327XX)
- **NoData**: 0

## Download
 
### Option 1: USGS EarthExplorer (Free Account Required)

1. Visit: https://earthexplorer.usgs.gov
2. Register for free EROS account
3. Search: "Landsat 8-9 OLI/TIRS C2 L2"
4. Filter: Cloud cover < 20%, any recent date
5. Download individual band (B4)

## File Properties

Typical Landsat COG:
- **Dimensions**: ~7811 × 7681 pixels
- **Scene size**: ~30km × 30km
- **Projection**: UTM
- **COG Features**: Internal tiling, overviews (pyramid levels: 2, 4, 8, 16)
- **NoData value**: 0

## Notes

- **Collection 2**: Improved geolocation and radiometry (use this, not Collection 1)
- **Level-2**: Atmospherically corrected surface reflectance
- **Landsat 8 vs 9**: Either works (L9 launched Sep 2021, identical sensors)
- Any band works for testing; B4 (red) recommended for good SNR

## References

- Landsat Collection 2: https://www.usgs.gov/landsat-missions/landsat-collection-2
- COG Specification: https://www.cogeo.org/
