# Benchmark Coverage Gap Analysis

**Generated:** 2026-02-18
**grdl version:** Post-pull (latest)
**Audit scope:** `grdl.image_processing`, `grdl.IO`, `grdl.data_prep`, `grdl.geolocation`, `grdl.coregistration`
**Benchmark sources:** `grdl_te/benchmarking/suite.py`, `workflows/comprehensive_benchmark_workflow.yaml`

---

## Coverage Summary

| Domain | Total | Benchmarked | Gaps | Coverage |
|--------|:-----:|:-----------:|:----:|:--------:|
| Filters | 9 | 9 | 0 | 100% |
| Intensity Transforms | 2 | 2 | 0 | 100% |
| Decomposition | 3 | 1 | 2 | 33% |
| Detection (CFAR + models) | 6 | 0 | 6 | 0% |
| SAR Processing (Multilook/CSI) | 2 | 0 | 2 | 0% |
| SAR Image Formation | 7 | 0 | 7 | 0% |
| Orthorectification | 5 | 2 | 3 | 40% |
| Pipeline | 1 | 1 | 0 | 100% |
| Data Prep | 3 | 3 | 0 | 100% |
| IO Readers/Writers | 20 | 10 | 10 | 50% |
| Geolocation | 5 | 3 | 2 | 60% |
| Elevation | 4 | 1 | 3 | 25% |
| CoRegistration | 3 | 2 | 1 | 67% |
| **Total** | **70** | **34** | **36** | **49%** |

---

## Currently Benchmarked (No Action Needed)

| Component | Domain | suite.py | YAML |
|-----------|--------|:--------:|:----:|
| MeanFilter | Filters | Y | Y |
| GaussianFilter | Filters | Y | Y |
| MedianFilter | Filters | Y | Y |
| MinFilter | Filters | Y | Y |
| MaxFilter | Filters | Y | Y |
| StdDevFilter | Filters | Y | Y |
| LeeFilter | Filters | Y | Y |
| ComplexLeeFilter | Filters | Y | Y |
| PhaseGradientFilter | Filters | Y | Y |
| ToDecibels | Intensity | Y | Y |
| PercentileStretch | Intensity | Y | Y |
| PauliDecomposition | Decomposition | Y | - |
| Pipeline | Pipeline | Y | - |
| Orthorectifier | Ortho | Y | - |
| OutputGrid | Ortho | Y | - |
| SublookDecomposition | SAR | Y | - |
| ChipExtractor | Data Prep | Y | - |
| Tiler | Data Prep | Y | - |
| Normalizer | Data Prep | Y | - |
| GeoTIFFReader | IO | Y | - |
| GeoTIFFWriter | IO | Y | - |
| HDF5Reader | IO | Y | - |
| HDF5Writer | IO | Y | - |
| JP2Reader | IO | Y | Y |
| NITFReader | IO | Y | Y |
| NumpyWriter | IO | Y | - |
| PngWriter | IO | Y | - |
| SICDReader | IO (SAR) | Y | Y |
| VIIRSReader | IO (Multi) | Y | Y |
| AffineGeolocation | Geolocation | Y | - |
| GCPGeolocation | Geolocation | Y | - |
| SICDGeolocation | Geolocation | Y | - |
| ConstantElevation | Elevation | Y | - |
| AffineCoRegistration | CoReg | Y | - |
| FeatureMatchCoRegistration | CoReg | Y | - |

---

## Gap Detail

### IO Readers/Writers (10 gaps)

| Component | File | Priority | Rationale |
|-----------|------|:--------:|-----------|
| CPHDReader | `grdl/IO/sar/cphd.py` | **High** | NGA Phase History standard; complex PVP vectors, phase-critical |
| CRSDReader | `grdl/IO/sar/crsd.py` | **High** | NGA Radar Signal standard; phase-critical format |
| SIDDReader | `grdl/IO/sar/sidd.py` | **High** | NGA Derived Data; paired with SICD in production |
| SICDWriter | `grdl/IO/sar/sicd_writer.py` | **High** | Roundtrip validation with SICDReader is critical |
| NITFWriter | `grdl/IO/nitf.py` | **High** | NITF is primary delivery format |
| Sentinel1SLCReader | `grdl/IO/sar/sentinel1_slc.py` | **High** | Most-used SAR dataset; complex SAFE parsing |
| Sentinel2Reader | `grdl/IO/eo/sentinel2.py` | **High** | Primary EO multispectral source |
| ASTERReader | `grdl/IO/ir/aster.py` | **Med** | Thermal IR; HDF-EOS format |
| BIOMASSL1Reader | `grdl/IO/sar/biomass.py` | **Med** | ESA BIOMASS L1 SCS; niche but growing |
| BIOMASSCatalog | `grdl/IO/sar/biomass_catalog.py` | **Low** | Discovery/download utility, not core processing |

### Detection (6 gaps)

| Component | File | Priority | Rationale |
|-----------|------|:--------:|-----------|
| CACFARDetector | `grdl/image_processing/detection/cfar/ca_cfar.py` | **High** | Primary SAR target detection algorithm |
| GOCFARDetector | `grdl/image_processing/detection/cfar/go_cfar.py` | **High** | Clutter-edge CFAR variant |
| SOCFARDetector | `grdl/image_processing/detection/cfar/so_cfar.py` | **Med** | Complementary to GO-CFAR |
| OSCFARDetector | `grdl/image_processing/detection/cfar/os_cfar.py` | **Med** | Robust outlier CFAR variant |
| Detection / DetectionSet | `grdl/image_processing/detection/models.py` | **Med** | Output containers for all detectors |
| FieldDefinition / Fields | `grdl/image_processing/detection/fields.py` | **Low** | Data dictionary accessors |

### Decomposition (2 gaps)

| Component | File | Priority | Rationale |
|-----------|------|:--------:|-----------|
| DualPolHAlpha | `grdl/image_processing/decomposition/dual_pol_halpha.py` | **High** | Eigenvalue decomposition; numerically sensitive |
| MultilookDecomposition | `grdl/image_processing/sar/multilook.py` | **High** | Fundamental SAR multi-look step |

### SAR Processing (2 gaps)

| Component | File | Priority | Rationale |
|-----------|------|:--------:|-----------|
| CSIProcessor | `grdl/image_processing/sar/csi.py` | **Med** | Coherent Shape Index; color sub-aperture visualization |
| MultilookDecomposition | `grdl/image_processing/sar/multilook.py` | **High** | Listed above under Decomposition |

### SAR Image Formation (7 gaps)

| Component | File | Priority | Rationale |
|-----------|------|:--------:|-----------|
| PolarFormatAlgorithm | `grdl/image_processing/sar/image_formation/pfa.py` | **High** | Core spotlight SAR formation algorithm |
| RangeDopplerAlgorithm | `grdl/image_processing/sar/image_formation/rda.py` | **High** | Canonical SAR formation algorithm |
| StripmapPFA | `grdl/image_processing/sar/image_formation/stripmap_pfa.py` | **High** | Extends PFA to wide-area collection |
| FastBackProjection | `grdl/image_processing/sar/image_formation/ffbp.py` | **Med** | FFT-accelerated backprojection; compute-intensive |
| CollectionGeometry | `grdl/image_processing/sar/image_formation/geometry.py` | **Med** | Input data class for all IFA algorithms |
| PolarGrid | `grdl/image_processing/sar/image_formation/polar_grid.py` | **Med** | Grid spec required by PFA/StripmapPFA |
| SubaperturePartitioner | `grdl/image_processing/sar/image_formation/subaperture.py` | **Med** | Aperture partitioning for FFBP |

### Orthorectification (3 gaps)

| Component | File | Priority | Rationale |
|-----------|------|:--------:|-----------|
| OrthoPipeline | `grdl/image_processing/ortho/ortho_pipeline.py` | **High** | **NEW** (2026-02-17) — recommended entry point for ortho |
| OrthoResult | `grdl/image_processing/ortho/ortho_pipeline.py` | **Med** | **NEW** — output container with save_geotiff() |
| compute_output_resolution | `grdl/image_processing/ortho/resolution.py` | **Med** | **NEW** — auto-resolution from SICD/BIOMASS metadata |

### Geolocation (2 gaps)

| Component | File | Priority | Rationale |
|-----------|------|:--------:|-----------|
| Sentinel1SLCGeolocation | `grdl/geolocation/sar/sentinel1_slc.py` | **High** | Paired with Sentinel1SLCReader; burst timing model |
| NoGeolocation | `grdl/geolocation/base.py` | **Low** | Trivial identity fallback |

### Elevation (3 gaps)

| Component | File | Priority | Rationale |
|-----------|------|:--------:|-----------|
| DTEDElevation | `grdl/geolocation/elevation/dted.py` | **High** | DTED tile lookup; SAR ortho pipelines |
| GeoTIFFDEM | `grdl/geolocation/elevation/geotiff_dem.py` | **High** | Primary DEM source for non-DTED workflows |
| GeoidCorrection | `grdl/geolocation/elevation/geoid.py` | **Med** | EGM96 geoid; affects HAE-to-MSL conversions |

### CoRegistration (1 gap)

| Component | File | Priority | Rationale |
|-----------|------|:--------:|-----------|
| ProjectiveCoRegistration | `grdl/coregistration/projective.py` | **Med** | Homography for oblique-view image pairs |

---

## Integration Order

### Phase 1 — High Priority (18 components)

Core algorithmic and IO components with high complexity or mission criticality.

**1a. CFAR Detection Pipeline**
- `CACFARDetector` — primary detection algorithm
- `GOCFARDetector` — clutter-edge variant
- `Detection` / `DetectionSet` — output models (required by all detectors)

**1b. SAR Image Formation**
- `PolarFormatAlgorithm` — spotlight SAR
- `RangeDopplerAlgorithm` — canonical stripmap SAR
- `StripmapPFA` — wide-area PFA variant

**1c. Decomposition & SAR Processing**
- `DualPolHAlpha` — dual-pol eigenvalue decomposition
- `MultilookDecomposition` — fundamental multi-look step

**1d. NGA IO Roundtrips**
- `SICDWriter` — roundtrip with SICDReader
- `NITFWriter` — delivery format write path
- `CPHDReader` — phase history data
- `CRSDReader` — radar signal data
- `SIDDReader` — derived data product

**1e. Sensor-Specific IO + Geolocation**
- `Sentinel1SLCReader` + `Sentinel1SLCGeolocation` (test as pair)
- `Sentinel2Reader`

**1f. Elevation Models**
- `DTEDElevation`
- `GeoTIFFDEM`

### Phase 2 — Medium Priority (14 components)

Niche formats, supporting classes, secondary algorithms, and new ortho pipeline.

**2a. New Ortho Pipeline**
- `OrthoPipeline` — new builder-pattern orchestrator
- `OrthoResult` — output container
- `compute_output_resolution` — auto-resolution function

**2b. Remaining CFAR**
- `SOCFARDetector`
- `OSCFARDetector`

**2c. SAR Image Formation Support**
- `FastBackProjection`
- `CollectionGeometry`
- `PolarGrid`
- `SubaperturePartitioner`

**2d. Remaining IO & Utilities**
- `ASTERReader`
- `BIOMASSL1Reader`
- `CSIProcessor`
- `GeoidCorrection`
- `ProjectiveCoRegistration`

### Phase 3 — Low Priority (4 components)

Discovery utilities, field accessors, trivial fallbacks.

- `BIOMASSCatalog` — download catalog, not core processing
- `FieldDefinition` / `Fields` — data dictionary accessors
- `NoGeolocation` — identity fallback

---

## Notes

- **New since last audit:** `OrthoPipeline`, `OrthoResult`, `compute_output_resolution` were added on 2026-02-17 and have zero benchmark coverage.
- **Paired testing recommended:** Sentinel1SLCReader should be benchmarked together with Sentinel1SLCGeolocation; SICDWriter should include a roundtrip test reading back with SICDReader.
- **Real data dependencies:** CPHDReader, CRSDReader, SIDDReader, Sentinel1SLCReader, BIOMASSL1Reader, and ASTERReader all require real satellite data files. Plan benchmark definitions to skip gracefully when data is absent (consistent with existing suite.py patterns).
- **Image Formation algorithms** require CPHD phase history data — benchmark coverage depends on CPHDReader being implemented first.
