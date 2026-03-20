# NISAR Test Data - NISARReader Validation

## Dataset

**Platform**: NISAR (NASA-ISRO Synthetic Aperture Radar)
**Product**: RSLC (Range-Doppler Single-Look Complex) or GSLC (Geocoded SLC)
**Format**: HDF5
**Frequency**: L-band (1.257 GHz) and/or S-band (3.2 GHz)
**File Size**: ~500 MB - 5 GB
**Data Type**: complex64 (RSLC) or float32 (GSLC)

## Required File

One file is sufficient for testing:
- **Pattern**: `NISAR*.h5`
- **Structure**: HDF5 with science/LSAR (or SSAR) group hierarchy
- **Geometry**: Slant-range (RSLC) or geocoded (GSLC)

## Download

### Option 1: NASA ASF DAAC (Recommended)

```bash
# Requires NASA Earthdata login
# Create account: https://urs.earthdata.nasa.gov/users/new

# Search and download via ASF Vertex:
# https://search.asf.alaska.edu/#/?dataset=NISAR

# Or use asf_search Python package:
pip install asf_search
python -c "
import asf_search as asf
results = asf.search(
    platform=['NISAR'],
    processingLevel=['RSLC'],
    maxResults=1
)
results[0].download('data/nisar/')
"
```

### Option 2: NASA Earthdata Search

1. Visit: https://search.earthdata.nasa.gov/
2. Search for "NISAR RSLC" or "NISAR GSLC"
3. Select a product and download the HDF5 file

### Option 3: Sample Data (Simulated)

NASA provides simulated NISAR data for development:
- https://nisar.jpl.nasa.gov/data/sample-data/

## File Properties

Typical NISAR RSLC:
- **Dimensions**: varies by swath/burst (~10000 × 5000 pixels typical)
- **Data type**: complex64 (I/Q samples)
- **Frequency bands**: L-band (primary), S-band (secondary)
- **Polarizations**: HH, HV, VH, VV (quad-pol) or HH+HV (dual-pol)
- **Orbit**: Sun-synchronous LEO (~747 km altitude)

## HDF5 Structure

```
/
├── science/
│   └── LSAR/  (or SSAR for S-band)
│       ├── RSLC/
│       │   ├── swaths/
│       │   │   └── frequencyA/  (or frequencyB)
│       │   │       ├── HH/          # Complex SLC data
│       │   │       ├── HV/
│       │   │       └── listOfPolarizations
│       │   └── metadata/
│       │       ├── orbit/
│       │       ├── attitude/
│       │       └── processingInformation/
│       └── identification/
│           ├── missionId
│           ├── productType
│           └── lookDirection
```

Access via NISARReader: `from grdl.IO.sar.nisar import NISARReader`

## Notes

- **License**: NASA Open Data (free for research and commercial use)
- **File naming**: `NISAR_L1_PR_RSLC_[ORBIT]_[DATE]_[FREQ]_[POL]_v[VER].h5`
- **Geometry**: RSLC is in radar coordinates; use NISARGeolocation for ground projection

## References

- NISAR Mission: https://nisar.jpl.nasa.gov/
- ASF DAAC: https://asf.alaska.edu/
- NISAR Product Spec: https://nisar.jpl.nasa.gov/data/
