# SIDD Test Data - SIDDReader Validation

## Dataset

**Standard**: NGA Sensor Independent Derived Data (SIDD)
**Format**: NITF 2.1 with embedded SIDD XML metadata
**Data Type**: uint8 or float32 (detected/derived SAR product)
**File Size**: ~10-500 MB

## Required File

One file is sufficient for testing:
- **Pattern**: `*.nitf` or `*.ntf`
- **Structure**: NITF with SIDD XML in TRE (Tagged Record Extension)
- **Content**: Detected (non-complex) SAR-derived imagery

## How to Acquire

SIDD files are derived products created from SICD data. Options:

### Option 1: Generate from SICD

Use the SIX library to convert an existing SICD to SIDD:

```bash
# Using NGA's six-library tools
cphd_to_sidd --input data/umbra/*.nitf --output data/sidd/derived.nitf
```

### Option 2: NGA Sample Data

Check NGA's six-library test data:
- https://github.com/ngageoint/six-library (test data in releases)

### Option 3: SarPy Generation

```python
# Using sarpy to create a SIDD from SICD
import sarpy.io.complex as sicd_io
import sarpy.processing.sidd as sidd_proc
# ... convert SICD to SIDD
```

## File Properties

Typical SIDD file:
- **Dimensions**: Varies (same scene extent as source SICD)
- **Data type**: uint8 (8-bit detected) or float32 (floating point)
- **Content**: Detected magnitude, not complex-valued
- **Metadata**: SIDD XML with product creation info, geographic data, display parameters

## Data Characteristics

**Derived Data**:
- Non-complex (magnitude only, no phase)
- Orthorectified or projected to output plane
- May include exploitation-ready enhancements
- Paired with SICD in NGA production workflows

**Relationship to SICD**:
- SIDD is the "display-ready" product derived from SICD
- Testing SIDDReader alongside SICDReader validates the full NGA SAR data chain

## Notes

- **SIDD vs SICD**: SICD = complex (phase-preserving), SIDD = detected (display-ready)
- **NITF container**: Same container format as SICD, different XML metadata schema
- **License**: Depends on source data provider

Access via SIDDReader: `from grdl.IO.sar import SIDDReader`

## References

- NGA SIDD Standard: https://nsgreg.nga.mil/doc/view?i=5382
- NGA six-library: https://github.com/ngageoint/six-library
- SarPy: https://github.com/ngageoint/sarpy
