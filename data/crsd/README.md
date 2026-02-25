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

## How to Acquire

CRSD is an NGA standard without public open-data repositories. Options:

### Option 1: Existing SAR Data Collection

If you already have CRSD files, copy one here:

```bash
cp /path/to/sar/raw_crsd/*.crsd data/crsd/
```

### Option 2: NGA / Government Sources

1. Visit the NGA Standards Registry: https://nsgreg.nga.mil
2. CRSD sample files may be available alongside the standard specification
3. Requires appropriate access credentials

### Option 3: SAR Vendor Request

Some SAR data providers (Umbra, Capella, etc.) may supply CRSD-formatted
data on request, though CPHD is far more common.

### Option 4: Generate with sarkit

If you have CPHD data, sarkit may support writing CRSD files programmatically.
Check sarkit documentation for CRSD writer capabilities.

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
