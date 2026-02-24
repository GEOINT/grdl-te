# Benchmark Coverage Analysis

**Updated:** 2026-02-24
**grdl version:** Post-pull (latest, includes TerraSAR-X reader + Interpolation module)
**Audit scope:** `grdl.image_processing`, `grdl.IO`, `grdl.data_prep`, `grdl.geolocation`, `grdl.coregistration`, `grdl.interpolation`
**Benchmark sources:** `grdl_te/benchmarking/suite.py` (13 groups, ~103 `_bench()` calls), `workflows/comprehensive_benchmark_workflow.yaml` (v2.0.0, 28 steps)

---

## Coverage Summary

| Domain | Total | Benchmarked | Gaps | Coverage |
|--------|:-----:|:-----------:|:----:|:--------:|
| Filters | 9 | 9 | 0 | 100% |
| Intensity Transforms | 2 | 2 | 0 | 100% |
| Decomposition | 3 | 3 | 0 | 100% |
| Detection (CFAR + models) | 6 | 6 | 0 | 100% |
| SAR Processing (Multilook/CSI) | 2 | 2 | 0 | 100% |
| SAR Image Formation | 7 | 7 | 0 | 100% |
| Orthorectification | 5 | 5 | 0 | 100% |
| Pipeline | 1 | 1 | 0 | 100% |
| Data Prep | 3 | 3 | 0 | 100% |
| IO Readers/Writers | 22 | 22 | 0 | 100% |
| Geolocation | 5 | 5 | 0 | 100% |
| Elevation | 4 | 4 | 0 | 100% |
| CoRegistration | 3 | 3 | 0 | 100% |
| Interpolation | 6 | 6 | 0 | 100% |
| **Total** | **78** | **78** | **0** | **100%** |

> **Note:** "Benchmarked" means the component has at least one entry in `suite.py`.
> The "Validation Test" column shows the dedicated `tests/validation/` file that
> exercises correctness assertions. All 78 components now have dedicated validation tests.

---

## Complete Benchmark Inventory

### Filters (9/9)

| Component | suite.py | YAML | Validation Test |
|-----------|:--------:|:----:|-----------------|
| MeanFilter | Y | Y | test_filters.py |
| GaussianFilter | Y | Y | test_filters.py |
| MedianFilter | Y | Y | test_filters.py |
| MinFilter | Y | Y | test_filters.py |
| MaxFilter | Y | Y | test_filters.py |
| StdDevFilter | Y | Y | test_filters.py |
| LeeFilter | Y | Y | test_filters.py |
| ComplexLeeFilter | Y | Y | test_filters.py |
| PhaseGradientFilter | Y | Y | test_filters.py |

### Intensity Transforms (2/2)

| Component | suite.py | YAML | Validation Test |
|-----------|:--------:|:----:|-----------------|
| ToDecibels | Y | Y | test_intensity.py |
| PercentileStretch | Y | Y | test_intensity.py |

### Decomposition (3/3)

| Component | suite.py | YAML | Validation Test |
|-----------|:--------:|:----:|-----------------|
| PauliDecomposition | Y | - | test_decomposition_halpha.py |
| DualPolHAlpha | Y | Y | test_decomposition_halpha.py |
| SublookDecomposition | Y | - | test_sar_multilook.py |

### Detection (6/6)

| Component | suite.py | YAML | Validation Test |
|-----------|:--------:|:----:|-----------------|
| CACFARDetector | Y | Y | test_detection_cfar.py |
| GOCFARDetector | Y | Y | test_detection_cfar.py |
| SOCFARDetector | Y | Y | test_detection_cfar.py |
| OSCFARDetector | Y | Y | test_detection_cfar.py |
| Detection / DetectionSet | Y | - | test_detection_models.py |
| Fields / FieldDefinition | Y | - | test_detection_models.py |

### SAR Processing (2/2)

| Component | suite.py | YAML | Validation Test |
|-----------|:--------:|:----:|-----------------|
| MultilookDecomposition | Y | Y | test_sar_multilook.py |
| CSIProcessor | Y | Y | test_sar_multilook.py |

### SAR Image Formation (7/7)

| Component | suite.py | YAML | Validation Test |
|-----------|:--------:|:----:|-----------------|
| CollectionGeometry | Y | Y | test_sar_image_formation.py |
| PolarGrid | Y | Y | test_sar_image_formation.py |
| PolarFormatAlgorithm | Y | Y | test_sar_image_formation.py |
| RangeDopplerAlgorithm | Y | Y | test_sar_image_formation.py |
| StripmapPFA | Y | Y | test_sar_image_formation.py |
| FastBackProjection | Y | Y | test_sar_image_formation.py |
| SubaperturePartitioner | Y | Y | test_sar_image_formation.py |

### Orthorectification (5/5)

| Component | suite.py | YAML | Validation Test |
|-----------|:--------:|:----:|-----------------|
| Orthorectifier | Y | - | test_ortho_pipeline.py |
| OutputGrid | Y | - | test_ortho_pipeline.py |
| OrthoPipeline | Y | Y | test_ortho_pipeline.py |
| OrthoResult | Y | - | test_ortho_pipeline.py |
| compute_output_resolution | Y | - | test_ortho_pipeline.py |

### Pipeline (1/1)

| Component | suite.py | YAML | Validation Test |
|-----------|:--------:|:----:|-----------------|
| Pipeline | Y | - | test_pipeline.py |

### Data Prep (3/3)

| Component | suite.py | YAML | Validation Test |
|-----------|:--------:|:----:|-----------------|
| ChipExtractor | Y | - | test_data_prep.py |
| Tiler | Y | - | test_data_prep.py |
| Normalizer | Y | - | test_data_prep.py |

### IO Readers/Writers (22/22)

| Component | suite.py | YAML | Validation Test |
|-----------|:--------:|:----:|-----------------|
| GeoTIFFReader | Y | - | test_io_geotiff.py |
| GeoTIFFWriter | Y | - | test_io_geotiff.py |
| HDF5Reader | Y | - | test_io_hdf5.py |
| HDF5Writer | Y | - | test_io_hdf5.py |
| JP2Reader | Y | Y | test_io_jpeg2000.py |
| NITFReader | Y | Y | test_io_nitf.py |
| NITFWriter | Y | - | test_io_sar_writers.py |
| NumpyWriter | Y | - | test_io_numpy_png.py |
| PngWriter | Y | - | test_io_numpy_png.py |
| SICDReader | Y | Y | test_io_nitf.py |
| SICDWriter | Y | - | test_io_sar_writers.py |
| Sentinel1SLCReader | Y | - | test_io_sentinel1.py |
| Sentinel2Reader | Y | - | test_io_sentinel2.py |
| VIIRSReader | Y | Y | test_io_hdf5.py |
| CPHDReader | Y | - | test_io_cphd.py |
| CRSDReader | Y | - | test_io_crsd.py |
| SIDDReader | Y | - | test_io_sidd.py |
| ASTERReader | Y | - | test_io_aster.py |
| BIOMASSL1Reader | Y | - | test_io_biomass.py |
| BIOMASSCatalog | Y | - | test_io_biomass.py |
| TerraSARReader | Y | - | test_io_terrasar.py |
| TerraSARMetadata | Y | - | test_io_terrasar.py |

### Geolocation (5/5)

| Component | suite.py | YAML | Validation Test |
|-----------|:--------:|:----:|-----------------|
| AffineGeolocation | Y | - | test_geolocation_base.py, test_geolocation_affine_real.py |
| GCPGeolocation | Y | - | test_geolocation_base.py |
| SICDGeolocation | Y | - | test_geolocation_base.py |
| NoGeolocation | Y | - | test_geolocation_base.py, test_geolocation_sentinel1.py |
| Sentinel1SLCGeolocation | Y | - | test_geolocation_sentinel1.py |

### Elevation (4/4)

| Component | suite.py | YAML | Validation Test |
|-----------|:--------:|:----:|-----------------|
| ConstantElevation | Y | - | test_geolocation_elevation.py |
| DTEDElevation | Y | - | test_elevation_models.py |
| GeoTIFFDEM | Y | - | test_elevation_models.py |
| GeoidCorrection | Y | - | test_elevation_models.py |

### CoRegistration (3/3)

| Component | suite.py | YAML | Validation Test |
|-----------|:--------:|:----:|-----------------|
| AffineCoRegistration | Y | - | test_coregistration_projective.py |
| FeatureMatchCoRegistration | Y | - | test_coregistration_projective.py |
| ProjectiveCoRegistration | Y | - | test_coregistration_projective.py |

### Interpolation (6/6)

| Component | suite.py | YAML | Validation Test |
|-----------|:--------:|:----:|-----------------|
| LanczosInterpolator | Y | - | test_interpolation.py |
| KaiserSincInterpolator | Y | - | test_interpolation.py |
| LagrangeInterpolator | Y | - | test_interpolation.py |
| FarrowInterpolator | Y | - | test_interpolation.py |
| PolyphaseInterpolator | Y | - | test_interpolation.py |
| ThiranDelayFilter | Y | - | test_interpolation.py |

---

## Test File Inventory

### Validation Tests (`tests/validation/` — 32 files)

| File | Components Covered |
|------|--------------------|
| test_coregistration_projective.py | AffineCoRegistration, FeatureMatchCoRegistration, ProjectiveCoRegistration |
| test_decomposition_halpha.py | PauliDecomposition, DualPolHAlpha |
| test_data_prep.py | ChipExtractor, Tiler, Normalizer |
| test_detection_cfar.py | CACFARDetector, GOCFARDetector, SOCFARDetector, OSCFARDetector |
| test_detection_models.py | Detection, DetectionSet, FieldDefinition, Fields |
| test_elevation_models.py | DTEDElevation, GeoTIFFDEM, GeoidCorrection |
| test_filters.py | MeanFilter, GaussianFilter, MedianFilter, MinFilter, MaxFilter, StdDevFilter, LeeFilter, ComplexLeeFilter, PhaseGradientFilter |
| test_geolocation_affine_real.py | AffineGeolocation (real Landsat data) |
| test_geolocation_base.py | AffineGeolocation, GCPGeolocation, SICDGeolocation, NoGeolocation |
| test_geolocation_elevation.py | ConstantElevation, ElevationModel ABC |
| test_geolocation_sentinel1.py | Sentinel1SLCGeolocation, NoGeolocation |
| test_geolocation_utils.py | Haversine, footprints, bounds, pixel checks |
| test_intensity.py | ToDecibels, PercentileStretch |
| test_interpolation.py | Lanczos, KaiserSinc, Lagrange, Farrow, Polyphase, ThiranDelay |
| test_io_aster.py | ASTERReader |
| test_io_biomass.py | BIOMASSL1Reader, BIOMASSCatalog |
| test_io_cphd.py | CPHDReader |
| test_io_crsd.py | CRSDReader |
| test_io_geotiff.py | GeoTIFFReader, GeoTIFFWriter |
| test_io_hdf5.py | HDF5Reader, HDF5Writer, VIIRSReader |
| test_io_jpeg2000.py | JP2Reader |
| test_io_nitf.py | NITFReader, SICDReader |
| test_io_numpy_png.py | NumpyWriter, PngWriter |
| test_io_sar_writers.py | SICDWriter, NITFWriter |
| test_io_sentinel1.py | Sentinel1SLCReader |
| test_io_sentinel2.py | Sentinel2Reader |
| test_io_sidd.py | SIDDReader |
| test_io_terrasar.py | TerraSARReader, TerraSARMetadata |
| test_ortho_pipeline.py | Orthorectifier, OutputGrid, OrthoPipeline, OrthoResult, compute_output_resolution |
| test_pipeline.py | Pipeline |
| test_sar_image_formation.py | CollectionGeometry, PolarGrid, PFA, RDA, StripmapPFA, FFBP, SubaperturePartitioner |
| test_sar_multilook.py | MultilookDecomposition, CSIProcessor, SublookDecomposition |

### Benchmarking Tests (`tests/benchmarking/` — 6 files)

| File | Purpose |
|------|---------|
| test_active_runner.py | ActiveBenchmarkRunner (workflow-based) |
| test_benchmark_models.py | AggregatedMetrics, HardwareSnapshot, StepBenchmarkResult, BenchmarkRecord |
| test_benchmark_report.py | Report formatting and generation |
| test_benchmark_source.py | BenchmarkSource construction and resolution |
| test_benchmark_store.py | JSONBenchmarkStore persistence |
| test_component_benchmark.py | ComponentBenchmark (single-function timing) |

---

## Test Data Inventory

| Data Type | Directory | Source | Status |
|-----------|-----------|--------|:------:|
| SICD NITF | `data/umbra/` | Umbra Open Data | Available |
| Landsat COG | `data/landsat/` | USGS | Available |
| VIIRS HDF5 | `data/viirs/` | NASA LAADS | Available |
| Sentinel-2 JP2 | `data/sentinel2/` | Copernicus | Available |
| CPHD | `data/cphd/` | SAR store | Copy needed |
| CRSD | `data/crsd/` | SAR store | Copy needed |
| Sentinel-1 SLC | `data/sentinel1/` | SAR store | Copy needed |
| TerraSAR-X | `data/terrasar/` | SAR store | Copy needed |
| SIDD | `data/sidd/` | NGA samples / generate | Acquire needed |
| ASTER L1T | `data/aster/` | NASA Earthdata | Acquire needed |
| BIOMASS L1 | `data/biomass/` | ESA Copernicus | Acquire needed |
| DTED tiles | `data/dted/` | USGS / OpenTopography | Acquire needed |
| GeoTIFF DEM | `data/dem/` | AWS Copernicus DEM | Acquire needed |
| EGM96 Geoid | `data/geoid/` | GeographicLib | Acquire needed |

All tests skip gracefully when data is absent via `pytest.skip` / `require_data_file()` patterns.
All data directories contain a `README.md` with download/acquisition instructions.

---

## Implementation History

- **Phase 1 (2026-02-19):** 47/78 components benchmarked — CACFARDetector, GOCFARDetector, SOCFARDetector, OSCFARDetector, Detection/DetectionSet, Fields/FieldDefinition, DualPolHAlpha, OrthoPipeline, OrthoResult, compute_output_resolution, ProjectiveCoRegistration, NoGeolocation.
- **Phase 2 (2026-02-24):** Real-data IO readers/writers + conftest fixtures — CPHDReader, CRSDReader, SIDDReader, Sentinel1SLCReader, Sentinel2Reader, ASTERReader, BIOMASSL1Reader, BIOMASSCatalog, SICDWriter, NITFWriter. MultilookDecomposition, CSIProcessor. DTEDElevation, GeoTIFFDEM, GeoidCorrection. Sentinel1SLCGeolocation.
- **Phase 2b (2026-02-24):** Interpolation benchmarks — LanczosInterpolator, KaiserSincInterpolator, LagrangeInterpolator, FarrowInterpolator, PolyphaseInterpolator, ThiranDelayFilter. New `run_interpolation_benchmarks()` group in suite.py.
- **Phase 2c (2026-02-24):** TerraSAR-X benchmarks — TerraSARReader, TerraSARMetadata.
- **Phase 3 (2026-02-24):** SAR image formation — CollectionGeometry, PolarGrid, PolarFormatAlgorithm, SubaperturePartitioner, RangeDopplerAlgorithm, StripmapPFA, FastBackProjection. New `run_image_formation_benchmarks()` group. YAML workflow updated to v2.0.0 (28 steps).
- **Phase 4 (2026-02-24):** Verification — all markers in pyproject.toml, all conftest.py fixtures, 13 benchmark groups total.
- **Phase 5 (2026-02-24):** Validation test suite for 16 previously uncovered components — test_filters.py (85 tests: 9 filters with scipy golden refs, dtype contracts, 3D bandwise, edge cases), test_intensity.py (18 tests: ToDecibels, PercentileStretch), test_pipeline.py (10 tests: composition, progress, nested), test_data_prep.py (42 tests: ChipExtractor, Tiler, Normalizer), test_io_numpy_png.py (22 tests: NumpyWriter, PngWriter roundtrip). 4 new pytest markers added (filters, intensity, pipeline, data_prep).

**Final state:** 78/78 components benchmarked. 78/78 with dedicated validation tests. 13 benchmark groups in suite.py. 28-step YAML workflow. 32 validation test files + 6 benchmarking test files.

---

