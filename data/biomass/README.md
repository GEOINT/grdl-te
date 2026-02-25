# BIOMASS Test Data - BIOMASSL1Reader Validation

## Dataset

**Platform**: BIOMASS (ESA Earth Explorer)
**Product**: L1 SCS (Single-look Complex Slant-range)
**Format**: GeoTIFF pair (magnitude + phase) with XML annotation
**Frequency**: P-band (435 MHz) - lowest frequency spaceborne SAR
**Resolution**: ~25 m (range) x ~6 m (azimuth)
**File Size**: ~100-500 MB per product

## Required Data

One complete product **directory** is required (not a single file). The reader
needs both the magnitude and phase TIFFs to reconstruct complex SLC data:

```
data/biomass/
└── BIO_S*_<product_dir>/
    ├── annotation/
    │   └── *_annot.xml          # Product metadata
    └── measurement/
        ├── *_abs.tiff           # Magnitude component
        └── *_phase.tiff         # Phase component (radians)
```

- **Pattern**: `BIO_S*` product directory (any swath: S1, S2, S3, etc.)
- **Both TIFFs required**: The reader reconstructs complex values as `magnitude * exp(1j * phase)`
- **Content**: P-band SAR imagery (forest biomass mapping)

## How to Acquire

### Option 1: ESA Open Access Hub

BIOMASS launched in April 2025. Data is available through ESA:

1. Visit: https://dataspace.copernicus.eu
2. Search: "BIOMASS" mission
3. Filter: L1 SCS product type
4. Download: Single product (extract the full directory)

### Option 2: ESA BIOMASS Data Hub

1. Visit: https://biomass.esa.int (mission-specific portal)
2. Register for data access
3. Browse available L1 products

### Option 3: ESA Science Hub API

```bash
# Using OData API (requires ESA Copernicus account)
curl -u "username:password" \
  "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?\$filter=Collection/Name eq 'BIOMASS'"
```

## File Properties

Typical BIOMASS L1 SCS:
- **Frequency**: P-band (435 MHz, ~69 cm wavelength)
- **Polarization**: Full quad-pol (HH, HV, VH, VV)
- **Orbit**: Sun-synchronous, dawn-dusk, 666 km altitude
- **Repeat cycle**: 25 days (tomographic phase: 7-day sub-cycle)

## Data Characteristics

**P-band SAR**:
- Penetrates forest canopy to measure biomass
- Longest wavelength spaceborne SAR ever
- Unique scattering mechanisms compared to X/C/L-band
- Ideal for forest structure and above-ground biomass estimation

**Testing Focus**:
- GeoTIFF-based SAR product parsing
- BIOMASS-specific metadata model
- Integration with BIOMASSCatalog for discovery

## Notes

- **New mission**: BIOMASS launched April 2025; data availability is growing
- **Niche but growing**: Only P-band SAR in orbit; unique science capability
- **License**: Free and open (Copernicus data policy)
- **BIOMASSCatalog**: Discovery/download utility, not core processing (low benchmark priority)

Access via BIOMASSL1Reader: `from grdl.IO.sar import BIOMASSL1Reader`

## References

- ESA BIOMASS Mission: https://www.esa.int/Applications/Observing_the_Earth/FutureEO/Biomass
- Copernicus Data Space: https://dataspace.copernicus.eu
- BIOMASS User Guide: https://sentinels.copernicus.eu/web/sentinel/missions/biomass
