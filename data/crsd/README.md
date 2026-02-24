# CRSD Test Data - CRSDReader Validation

## Dataset

**Standard**: NGA Compensated Radar Signal Data (CRSD)
**Format**: CRSD 1.0 (binary with XML header)
**Data Type**: complex64 (radar signal data)
**File Size**: ~100 MB - 2 GB

## Required File

One file is sufficient for testing:
- **Pattern**: `*.crsd`
- **Structure**: Binary radar signal with XML metadata header
- **Content**: Compensated radar signal data (related to CPHD)

## You Already Have This Data

Copy any `.crsd` file from your SAR data collection:

```bash
cp /path/to/sar/raw_crsd/*.crsd data/crsd/
```

## File Properties

- **Signal data**: N vectors x M samples (complex64)
- **Metadata**: XML header with collection parameters
- **Relationship**: CRSD is a companion standard to CPHD, focused on radar signal representation

## Data Characteristics

**Radar Signal Data**:
- Compensated (motion-corrected) radar returns
- Similar structure to CPHD but with different compensation model
- Used in advanced SAR processing workflows

## Notes

- **Phase-critical**: Do not truncate or compress
- **NGA standard**: Part of the NGA SAR data standards family (SICD, SIDD, CPHD, CRSD)

Access via CRSDReader: `from grdl.IO.sar import CRSDReader`

## References

- NGA CRSD Standard: https://nsgreg.nga.mil
- NGA SAR Standards: https://github.com/ngageoint/six-library
