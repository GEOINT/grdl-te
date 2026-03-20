# EO NITF Test Data - EONITFReader Validation

## Dataset

**Platform**: Various electro-optical (EO) satellites
**Product**: Ortho-ready or raw imagery with RPC/RSM geolocation
**Format**: NITF 2.1 with RPC00B TRE or RSM TREs
**File Size**: ~50-500 MB
**Data Type**: uint8 or uint16 (real-valued panchromatic or multispectral)

## Required File

One file is sufficient for testing:
- **Pattern**: `*.ntf` or `*.nitf`
- **Structure**: NITF with RPC00B Tagged Record Extension (preferred) or RSM TREs
- **Geometry**: Requires RPC or RSM coefficients for geolocation

## Download

### Option 1: NGA Sample Data

```bash
# NGA provides open NITF samples with RPC metadata:
# https://github.com/ngageoint/six-library (check test data)
# https://gwg.nga.mil/misb/faq.html (NITF samples)
```

### Option 2: SpaceNet (AWS Open Data)

```bash
# SpaceNet datasets include NITF imagery with RPC metadata
# https://registry.opendata.aws/spacenet/

aws s3 ls s3://spacenet-dataset/ --no-sign-request
# Download a WorldView NITF file with RPC
```

### Option 3: DigitalGlobe/Maxar Open Data

WorldView or GeoEye imagery from disaster response:
- https://www.maxar.com/open-data
- Select an event, download NITF products

## File Properties

Typical EO NITF with RPC:
- **Dimensions**: varies (~5000-30000 × 5000-30000 pixels)
- **Data type**: uint16 (11-bit or 16-bit radiometric depth)
- **Bands**: 1 (panchromatic) or 4-8 (multispectral)
- **Resolution**: 0.3-2.0 m (commercial high-resolution)

## RPC00B Structure

The Rational Polynomial Coefficients (RPC00B) TRE contains:
- 20 numerator coefficients (line/sample)
- 20 denominator coefficients (line/sample)
- Normalization offsets and scales for lat, lon, height, line, sample
- Maps image (line, sample) to geographic (lat, lon, height)

## Notes

- **License**: Varies by source; SpaceNet data is CC BY-SA 4.0
- **File naming**: Varies by provider
- **Key feature**: RPC or RSM geolocation model embedded in NITF TREs

Access via EONITFReader: `from grdl.IO.eo.nitf import EONITFReader`

## References

- NITF Standard: https://gwg.nga.mil/ntb/baseline/docs/2500c/
- RPC00B TRE: https://gwg.nga.mil/misb/docs/tre/
- SpaceNet: https://spacenet.ai/
