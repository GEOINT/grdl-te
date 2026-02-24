# DTED Test Data - DTEDElevation Validation

## Dataset

**Standard**: Digital Terrain Elevation Data (DTED)
**Format**: DTED Level 0/1/2 (`.dt0`, `.dt1`, `.dt2`)
**Resolution**: ~900 m (Level 0), ~90 m (Level 1), ~30 m (Level 2)
**Data Type**: int16 (elevation in meters, MSL)
**File Size**: ~1-25 MB per tile

## Required Files

At least one DTED tile covering a known area:
- **Pattern**: `*.dt0`, `*.dt1`, or `*.dt2`
- **Structure**: 1-degree x 1-degree tiles in DTED directory hierarchy
- **Directory layout**: `dted/w118/n33.dt1` (lon directory / lat file)

## How to Acquire

### Option 1: USGS EarthExplorer (Free)

1. Visit: https://earthexplorer.usgs.gov
2. Register: Free account
3. Search: "SRTM" or "DTED" in Data Sets tab
4. Select area of interest
5. Download: DTED Level 1 or Level 2 tiles

### Option 2: OpenTopography (Free, No Account for SRTM)

1. Visit: https://portal.opentopography.org/dataCatalog
2. Select: SRTM GL1 or GL3
3. Draw bounding box
4. Download: DTED format option

### Option 3: NGA/DoD (Restricted)

DTED Level 2 globally is available through NGA for authorized users:
- Contact NGA data distribution
- DTED Level 0/1 are unclassified and freely available

### Option 4: Convert from SRTM HGT

```bash
# SRTM .hgt files can be converted to DTED using GDAL
gdal_translate -of DTED N33W118.hgt dted/w118/n33.dt1
```

## File Properties

DTED tile specification:
- **Coverage**: 1 degree latitude x 1 degree longitude per tile
- **Grid**: Regular lat/lon grid (not projected)
- **Level 0**: 120 x 120 posts (~900 m spacing)
- **Level 1**: 1201 x 1201 posts (~90 m spacing)
- **Level 2**: 3601 x 3601 posts (~30 m spacing)
- **Elevation**: Signed int16, meters above Mean Sea Level (MSL)
- **Void value**: -32767

## Directory Structure

DTED uses a hierarchical directory convention:
```
dted/
├── w118/
│   ├── n33.dt1    # Tile covering 33N-34N, 118W-117W
│   └── n34.dt1
├── w117/
│   └── n33.dt1
└── ...
```

## Data Characteristics

**Elevation Model**:
- Heights referenced to EGM96 geoid (MSL), not WGS84 ellipsoid (HAE)
- SAR orthorectification pipelines use DTED for terrain correction
- DTEDElevation performs bilinear interpolation between grid posts

**Testing Focus**:
- DTED tile lookup by lat/lon
- Multi-tile boundary handling
- Vectorized elevation queries (batch lat/lon arrays)
- Integration with ortho pipeline DEM input

## Notes

- **MSL vs HAE**: DTED elevations are Mean Sea Level. Use GeoidCorrection to convert to HAE if needed.
- **Minimum tiles**: 1 tile is enough for unit testing; 4 adjacent tiles test boundary handling.
- **SRTM equivalence**: SRTM 1-arc-second data is functionally equivalent to DTED Level 2.
- **License**: SRTM/DTED Level 0-1 are public domain. Level 2 may have restrictions outside US.

Access via DTEDElevation: `from grdl.geolocation.elevation import DTEDElevation`

## References

- USGS EarthExplorer: https://earthexplorer.usgs.gov
- OpenTopography: https://portal.opentopography.org
- SRTM Data: https://www2.jpl.nasa.gov/srtm/
- DTED Specification: MIL-PRF-89020B
