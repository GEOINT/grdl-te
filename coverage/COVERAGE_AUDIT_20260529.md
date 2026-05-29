# GRDL-TE Coverage Audit Report

**Generated:** 2026-05-29  
**GRDL version tested against:** latest (post-v0.4.0)  
**GRDL-TE version:** 0.4.0

---

## Executive Summary

| Metric | Result |
|--------|--------|
| Total tests (excl. IFP) | 1306 passed, 37 skipped, 0 failures |
| IFP tests (optimized) | 11 passed in 5:25 (chip-based, 256 pulses) |
| GPU tests | 11 skipped (no GPU on this host) |
| **grdl-te is API-compatible** | ✅ All tests pass against current `grdl` |

---

## Phase 1: Triage Results

**No failures found.** All 1306 non-IFP tests pass cleanly against the current `grdl` codebase. The `grdl-te` test suite is fully compatible with the latest `grdl` API.

---

## Phase 2: IFP Test Optimization

The `test_sar_image_formation.py` was refactored from loading the full 3.2GB CPHD file to extracting a 256-pulse chip from the aperture center using `CPHDReader.read_chip()` + `create_subaperture_metadata()`.

| Algorithm | Status | Notes |
|-----------|--------|-------|
| PFA (PolarFormatAlgorithm) | ✅ PASS | Fast (~seconds) |
| RDA (RangeDopplerAlgorithm) | ✅ PASS | Fast (~seconds) |
| StripmapPFA | ✅ PASS | Moderate |
| FastBackProjection (FFBP) | ✅ PASS | Slowest (~4 min on 256 pulses) |
| CollectionGeometry | ✅ PASS | Instant |
| PolarGrid | ✅ PASS | Instant |
| SubaperturePartitioner | ✅ PASS | Instant |

**Total IFP runtime: 5:25** (was previously untestable in CI due to full-file processing).

---

## Phase 3: Coverage Gap Analysis

### Components WITH real-data test coverage in grdl-te

| Module | Component | Test File |
|--------|-----------|-----------|
| `grdl.IO` | GeoTIFFReader | test_io_geotiff.py |
| `grdl.IO` | HDF5Reader | test_io_hdf5.py |
| `grdl.IO` | JP2Reader | test_io_jpeg2000.py |
| `grdl.IO` | NITFReader | test_io_nitf.py |
| `grdl.IO` | SICDReader | test_io_sicd.py |
| `grdl.IO` | CPHDReader | test_io_cphd.py |
| `grdl.IO` | CRSDReader | test_io_crsd.py |
| `grdl.IO` | SIDDReader | test_io_sidd.py |
| `grdl.IO` | BIOMASSL1Reader | test_io_biomass.py |
| `grdl.IO` | Sentinel1SLCReader | test_io_sentinel1.py |
| `grdl.IO` | Sentinel2Reader | test_io_sentinel2.py |
| `grdl.IO` | TerraSARReader | test_io_terrasar.py |
| `grdl.IO` | NISARReader | test_io_nisar.py |
| `grdl.IO` | EONITFReader | test_io_eo_nitf.py |
| `grdl.IO` | ASTERReader | test_io_aster.py |
| `grdl.IO` | VIIRSReader | test_io_viirs.py |
| `grdl.IO` | GeoTIFFWriter | test_io_writers.py |
| `grdl.IO` | HDF5Writer | test_io_writers.py |
| `grdl.IO` | NumpyWriter / PngWriter | test_io_numpy_png.py |
| `grdl.IO` | SICDWriter / SIDDWriter | test_io_sar_writers.py |
| `grdl.geolocation` | AffineGeolocation | test_geolocation_affine_real.py |
| `grdl.geolocation` | SICDGeolocation | test_geolocation_sicd.py |
| `grdl.geolocation` | SIDDGeolocation | test_geolocation_sidd.py |
| `grdl.geolocation` | NISARGeolocation | test_geolocation_nisar.py |
| `grdl.geolocation` | Sentinel1SLCGeolocation | test_geolocation_sentinel1.py |
| `grdl.geolocation` | RPCGeolocation | test_geolocation_rpc_rsm.py |
| `grdl.geolocation` | RSMGeolocation | test_geolocation_rpc_rsm.py |
| `grdl.geolocation` | ChipGeolocation | test_geolocation_chip.py |
| `grdl.geolocation` | coordinate utilities | test_geolocation_base.py, test_coordinate_utils.py |
| `grdl.geolocation.elevation` | DTEDElevation, GeoTIFFDEM, TiledGeoTIFFDEM | test_elevation_models.py, test_elevation_tiled_geotiff.py |
| `grdl.image_processing` | Pipeline | test_pipeline.py |
| `grdl.image_processing` | PauliDecomposition | test_decomposition_pauli.py |
| `grdl.image_processing` | DualPolHAlpha | test_decomposition_halpha.py |
| `grdl.image_processing` | SublookDecomposition | test_sar_sublook_dominance.py |
| `grdl.image_processing` | MultilookDecomposition | test_sar_multilook.py |
| `grdl.image_processing` | CSIProcessor | test_sar_multilook.py |
| `grdl.image_processing` | Dominance/Entropy | test_sar_sublook_dominance.py |
| `grdl.image_processing` | CFAR detectors (CA/GO/SO/OS) | test_detection_cfar.py |
| `grdl.image_processing` | Detection models | test_detection_models.py |
| `grdl.image_processing` | Filters (Mean/Gauss/Median/etc.) | test_filters.py |
| `grdl.image_processing` | ToDecibels, PercentileStretch | test_intensity.py |
| `grdl.image_processing` | Orthorectifier / OrthoBuilder | test_ortho_pipeline.py |
| `grdl.image_processing` | ENUGrid | test_enu_grid.py |
| `grdl.image_processing` | Image Formation (PFA/RDA/FFBP/StripmapPFA) | test_sar_image_formation.py |
| `grdl.image_processing.ortho` | resample / accelerated | test_accelerated_resampling.py |
| `grdl.coregistration` | AffineCoRegistration | test_coregistration_affine.py |
| `grdl.coregistration` | ProjectiveCoRegistration | test_coregistration_projective.py |
| `grdl.coregistration` | FeatureMatchCoRegistration | test_coregistration_feature_match.py |
| `grdl.data_prep` | ChipExtractor, Tiler, Normalizer | test_data_prep.py |
| `grdl.transforms` | detection geometry transforms | test_transforms_detection.py |
| `grdl.interpolation` | All interpolators | test_interpolation.py |
| `grdl.discovery` | MetadataScanner, LocalCatalog, PluginRegistry | test_discovery.py |

---

### Components WITHOUT real-data test coverage (GAPS)

| # | Module | Component | Priority | Notes |
|---|--------|-----------|----------|-------|
| 1 | `grdl.contrast` | MangisDensity | HIGH | Core SAR display, used everywhere |
| 2 | `grdl.contrast` | NRLStretch | HIGH | Navy standard SAR remap |
| 3 | `grdl.contrast` | LinearStretch | MEDIUM | Generic display stretch |
| 4 | `grdl.contrast` | LogStretch | MEDIUM | Common SAR stretch |
| 5 | `grdl.contrast` | GammaCorrection | MEDIUM | Standard gamma |
| 6 | `grdl.contrast` | SigmoidStretch | LOW | Less common |
| 7 | `grdl.contrast` | HistogramEqualization | MEDIUM | Standard EO enhancement |
| 8 | `grdl.contrast` | CLAHE | MEDIUM | Adaptive local contrast |
| 9 | `grdl.IO.gmti` | STANAG4607Reader | HIGH | No GMTI data exists |
| 10 | `grdl.IO.gmti` | STANAG4607Writer | HIGH | No GMTI data exists |
| 11 | `grdl.IO.sar` | Sentinel1L0Reader | MEDIUM | No L0 RAW data exists |
| 12 | `grdl.IO.sar` | SICDCollectionReader | MEDIUM | Need multi-pol SICD set |
| 13 | `grdl.shapes` | Circle, Ellipse, GeoPolygon, Arc | HIGH | Entire module untested |
| 14 | `grdl.shapes` | convolve_ellipses, combine_evidence | MEDIUM | Error propagation |
| 15 | `grdl.shapes` | overlay_shape, burn_shape | MEDIUM | Display integration |
| 16 | `grdl.shapes` | cued_detect | HIGH | Detection cueing |
| 17 | `grdl.IO.catalog` | Sentinel1SLCCatalog | LOW | Network-dependent |
| 18 | `grdl.IO.catalog` | TerraSARCatalog | LOW | Network-dependent |
| 19 | `grdl.IO.catalog` | NISARCatalog | LOW | Network-dependent |
| 20 | `grdl.image_processing.decomposition` | CovarianceMatrix | MEDIUM | Polarimetric matrix |
| 21 | `grdl.image_processing.decomposition` | CoherencyMatrix | MEDIUM | Polarimetric matrix |
| 22 | `grdl.image_processing.decomposition` | StokesVector | LOW | Stokes analysis |
| 23 | `grdl.image_processing.decomposition` | KennaughMatrix | LOW | Kennaugh scattering |
| 24 | `grdl.geolocation.sar` | GCPGeolocation | MEDIUM | GCP-based geolocation |
| 25 | `grdl.discovery` | DataSynthesizer | LOW | Test data generation |
| 26 | `grdl.discovery` | compute_beam_footprint | MEDIUM | SAR beam geometry |

---

## Phase 4: Data Assessment

### Data Available vs. Required for Gap Coverage

| Gap # | Component | Data Required | Existing Data Sufficient? | Action |
|--------|-----------|---------------|---------------------------|--------|
| 1-8 | `grdl.contrast` (all 8) | Any imagery (SAR or EO) | ✅ YES — Umbra SICD, Landsat, Sentinel-2 all present | Write test using existing data |
| 9-10 | STANAG4607Reader/Writer | STANAG 4607 `.4607` GMTI file | ❌ NO — No GMTI data in repo | **Procure:** Need a real or declassified STANAG 4607 GMTI recording |
| 11 | Sentinel1L0Reader | Sentinel-1 Level-0 RAW SAFE product (`.SAFE` with `S1x_xx_RAW__0S*`) | ❌ NO — Only SLC exists | **Procure:** Download from ESA Copernicus Data Space (free, ~1 GB per segment) |
| 12 | SICDCollectionReader | Multi-polarization SICD set (HH+HV+VH+VV `.nitf` from same collect) | ❌ NO — Only single-pol Umbra SICD | **Procure:** Need quad-pol SICD (e.g., from Sandia or NGA) |
| 13-16 | `grdl.shapes` | Any imagery with geolocation (for pixel↔geo transforms) | ✅ YES — Can use Umbra SICD with SICDGeolocation | Write test using existing data |
| 17-19 | IO Catalogs (S1, TSX, NISAR) | Network access + credentials | ⚠️ PARTIAL — Requires live API tokens | Consider mock/VCR tests for CI |
| 20-21 | CovarianceMatrix / CoherencyMatrix | Quad-pol SAR data (HH, HV, VH, VV) | ❌ NO — No quad-pol data | **Procure:** Need quad-pol SLC (ALOS PALSAR-2, RADARSAT-2, or TerraSAR quad-pol) |
| 22-23 | StokesVector / KennaughMatrix | Quad-pol SAR data | ❌ NO — Same as above | Same quad-pol source as #20-21 |
| 24 | GCPGeolocation | Imagery with GCPs (BIOMASS uses this internally) | ✅ YES — BIOMASS data present | Write test using existing BIOMASS data |
| 25 | DataSynthesizer | None (generates synthetic data) | ✅ YES — No external data needed | Write test with no data dependency |
| 26 | compute_beam_footprint | SAR metadata with orbit/attitude | ✅ YES — Any SICD/CPHD/S1 metadata | Write test using existing data |

---

## Summary: New Tests Needed

### Tests writable NOW (existing data sufficient)

| Test File to Create | Components Tested | Data Source |
|---------------------|-------------------|-------------|
| `test_contrast.py` | MangisDensity, NRLStretch, LinearStretch, LogStretch, GammaCorrection, SigmoidStretch, HistogramEqualization, CLAHE | Umbra SICD chip + Landsat chip |
| `test_shapes.py` | Circle, Ellipse, GeoPolygon, Arc, overlay_shape, burn_shape, cued_detect | Umbra SICD + SICDGeolocation |
| `test_polarimetric_matrix.py` | CovarianceMatrix, CoherencyMatrix (synthetic dual-pol from S1 VV+VH) | Sentinel-1 SLC (VV+VH) |
| `test_geolocation_gcp.py` | GCPGeolocation | BIOMASS data |
| `test_discovery_advanced.py` | DataSynthesizer, compute_beam_footprint | CPHD/SICD metadata (no data file needed for synthesizer) |

### Tests requiring NEW data procurement

| Test File to Create | Components Tested | Data Needed | Source |
|---------------------|-------------------|-------------|--------|
| `test_io_gmti.py` | STANAG4607Reader, STANAG4607Writer | STANAG 4607 `.4607` file | Classified/synthetic — check if GRDL has a test generator |
| `test_io_sentinel1_l0.py` | Sentinel1L0Reader | S1 Level-0 RAW `.SAFE` | [ESA CDSE](https://dataspace.copernicus.eu) (free, ~1GB) |
| `test_io_sicd_collection.py` | SICDCollectionReader, open_sicd_collection | Quad-pol SICD set (4 NITF files) | Sandia / NGA open data |
| `test_decomposition_quadpol.py` | CovarianceMatrix, CoherencyMatrix, StokesVector, KennaughMatrix (full validation) | Quad-pol SLC | ALOS PALSAR-2 / RADARSAT-2 |

### Tests deferred (network-dependent, consider mocking)

| Test File | Components | Notes |
|-----------|------------|-------|
| `test_catalog_network.py` | Sentinel1SLCCatalog, TerraSARCatalog, NISARCatalog | Requires live API tokens; best tested via VCR/cassette recordings |

---

## Recommendations

1. **Immediate wins:** Write `test_contrast.py`, `test_shapes.py`, and `test_geolocation_gcp.py` — all can be created today with zero new data.
2. **Quick procurements:** Sentinel-1 L0 data is freely available and small enough for CI.
3. **Blocking gaps:** STANAG 4607 and quad-pol SICD require specialized data sources. Check if `grdl` itself has synthetic GMTI generation capabilities that could produce test fixtures.
4. **Catalog tests:** Use `pytest-recording` or similar VCR library to record API interactions for deterministic CI replay.
