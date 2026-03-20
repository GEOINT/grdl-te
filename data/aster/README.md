# ASTER Test Data - ASTERReader Validation

## Dataset

**Platform**: Terra (NASA EOS)
**Sensor**: ASTER (Advanced Spaceborne Thermal Emission and Reflection Radiometer)
**Product**: AST_L1T (Registered Radiance at the Sensor)
**Format**: GeoTIFF
**Resolution**: 15 m (VNIR), 30 m (SWIR), 90 m (TIR)
**File Size**: ~100-300 MB

## Required File

One file is sufficient for testing:
- **Pattern**: `AST_L1T*.tif`
- **Structure**: GeoTIFF with multi-resolution band groups
- **Recommended**: Any cloud-free AST_L1T scene

## How to Acquire

### Option 1: NASA Earthdata (Free Account Required)

1. Register: https://urs.earthdata.nasa.gov/users/new (free)
2. Visit: https://search.earthdata.nasa.gov
3. Search: "AST_L1T" in the search bar
4. Filter: Select a recent, cloud-free scene
5. Download: Single GeoTIFF file

### Option 2: LPDAAC Data Pool

1. Visit: https://e4ftl01.cr.usgs.gov/ASTT/AST_L1T.003/
2. Browse by date
3. Download any `.tif` file

### Option 3: AppEEARS (Subset/Extract)

1. Visit: https://appeears.earthdatacloud.nasa.gov
2. Create area sample request for ASTER L1T
3. Download extracted subset

## File Properties

Typical AST_L1T file:
- **VNIR bands**: Band 1 (520-600 nm), Band 2 (630-690 nm), Band 3N (760-860 nm) at 15 m
- **SWIR bands**: Bands 4-9 at 30 m (note: SWIR failed in 2008)
- **TIR bands**: Bands 10-14 at 90 m
- **Projection**: UTM (terrain-corrected, orthorectified)
- **Data type**: uint8 (VNIR), uint8 (SWIR), uint16 (TIR)

## Data Characteristics

**Multi-Resolution Thermal IR**:
- Primary use case: thermal/IR analysis with VNIR context
- GeoTIFF format read via rasterio
- Band groups organized by subsystem (VNIR, SWIR, TIR)

**Testing Focus**:
- GeoTIFF format parsing
- Multi-resolution band handling
- Thermal band data types and scaling

## Notes

- **SWIR bands**: Non-functional since April 2008 (detector cryocooler failure). VNIR and TIR still operational.
- **Format change**: NASA transitioned AST_L1T from HDF-EOS to GeoTIFF. Current downloads are `.tif`.
- **End of life approaching**: Terra orbit is degrading; collect data while available.
- **License**: Public domain (NASA)

Access via ASTERReader: `from grdl.IO.ir import ASTERReader`

## References

- ASTER User Guide: https://asterweb.jpl.nasa.gov/
- LPDAAC ASTER Products: https://lpdaac.usgs.gov/products/ast_l1tv003/
- NASA Earthdata Search: https://search.earthdata.nasa.gov
