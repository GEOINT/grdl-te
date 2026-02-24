# DEM Test Data - GeoTIFFDEM Validation

## Dataset

**Format**: GeoTIFF Digital Elevation Model
**Sources**: Copernicus DEM, SRTM, ALOS, NASADEM, or any elevation GeoTIFF
**Resolution**: 30 m (1 arc-second) recommended
**Data Type**: float32 or int16 (elevation in meters)
**File Size**: ~25-100 MB per tile

## Required File

One GeoTIFF DEM file is sufficient for testing:
- **Pattern**: `*.tif`
- **Structure**: Single-band GeoTIFF with elevation values
- **CRS**: EPSG:4326 (geographic) or any projected CRS
- **Recommended**: Copernicus 30m GLO-30 or SRTM 1-arc-second

## How to Acquire

### Option 1: Copernicus DEM (Free, Best Quality)

1. Visit: https://dataspace.copernicus.eu
2. Search: "Copernicus DEM" GLO-30
3. Select tile covering your test area
4. Download: GeoTIFF format

Or via AWS Open Data (no sign-in):
```bash
# Copernicus DEM 30m tiles on AWS
aws s3 ls s3://copernicus-dem-30m/ --no-sign-request
aws s3 cp s3://copernicus-dem-30m/Copernicus_DSM_COG_10_N33_00_W118_00_DEM/Copernicus_DSM_COG_10_N33_00_W118_00_DEM.tif \
    data/dem/ --no-sign-request
```

### Option 2: SRTM via OpenTopography (Free)

1. Visit: https://portal.opentopography.org/dataCatalog
2. Select: SRTM GL1 (30m) or GL3 (90m)
3. Draw bounding box over area of interest
4. Download: GeoTIFF format

### Option 3: USGS 3DEP (US Only, High Resolution)

1. Visit: https://apps.nationalmap.gov/downloader/
2. Select: Elevation Products (3DEP)
3. Download: 1/3 arc-second (~10 m) GeoTIFF

### Option 4: GDAL Virtual (Quick Synthetic Test)

```bash
# Create a small synthetic DEM for basic testing
gdal_create -outsize 1201 1201 -bands 1 -ot Float32 \
    -a_srs EPSG:4326 -a_ullr -118.0 34.0 -117.0 33.0 \
    data/dem/synthetic_dem.tif
```

## File Properties

Typical DEM GeoTIFF:
- **Coverage**: 1 degree x 1 degree (varies by source)
- **Pixel type**: Float32 (meters above reference)
- **NoData**: -32768, -9999, or NaN (varies by source)
- **CRS**: EPSG:4326 (WGS84 geographic) most common
- **Reference**: EGM2008 geoid (Copernicus), EGM96 (SRTM)

## Data Characteristics

**GeoTIFF Elevation**:
- Standard raster format readable by GDAL/rasterio
- GeoTIFFDEM performs bilinear interpolation for sub-pixel queries
- Supports any CRS (auto-reprojected to WGS84 lat/lon queries)

**Testing Focus**:
- GeoTIFF-based elevation queries vs DTED tile-based queries
- Batch vectorized lat/lon elevation extraction
- NoData handling at ocean/void areas
- Integration with ortho pipeline

## Notes

- **Copernicus DEM preferred**: Best global quality, consistent void-filling
- **Vertical reference**: Check whether heights are MSL (geoid) or HAE (ellipsoid)
- **One tile is enough**: Single 1-degree tile sufficient for benchmark/validation
- **License**: Copernicus DEM is free (Copernicus data policy). SRTM is public domain.

Access via GeoTIFFDEM: `from grdl.geolocation.elevation import GeoTIFFDEM`

## References

- Copernicus DEM: https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model
- AWS Copernicus DEM: https://registry.opendata.aws/copernicus-dem/
- OpenTopography: https://portal.opentopography.org
- USGS 3DEP: https://www.usgs.gov/3d-elevation-program
