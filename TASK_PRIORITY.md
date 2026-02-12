# GRDL Test & Evaluation — Task Priority Document

**Version:** 1.0
**Date:** 2026-02-12
**Classification:** UNCLASSIFIED
**Author:** Senior Lead GEOINT Developer / SDET

---

## Executive Summary

The GRDL repository currently has **41 unit-test files** containing **~872 test functions** in the core library, plus **4 real-data validation files** (68 tests) in GRDL-TE. Coverage is strong for base-format readers (GeoTIFF, HDF5, JP2, NITF), data preparation (ChipExtractor, Tiler, Normalizer), and image processing transforms. However, **critical gaps exist** in geolocation utilities (Haversine distance, footprint calculation, interpolation error metrics), elevation models (DTED, GeoTIFF DEM, EGM96 geoid), SAR-specific readers (SICD, CPHD, CRSD, SIDD via sarpy/sarkit), and the detection-transform bridge — all of which are mission-critical functions where a single floating-point error can cascade into meter-scale geolocation failures. The GRDL-TE repository currently validates only IO readers against real data and does not yet exercise geolocation, image processing, or coregistration against production satellite imagery.

---

## Priority Matrix — GRDL Core Library (`grdl/`)

### Must Have — Mission-Critical (Spatial Calculations, Coordinate Transforms, Data Ingestion)

| Function / Class | Module | Category | Risk Level | Existing Tests | Logic for Priority |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `geographic_distance()` | `geolocation.utils` | Geometry | **CRITICAL** | **None** | Haversine implementation is the single source of truth for all distance computations. A radian-conversion bug or Earth-radius constant error silently corrupts every downstream proximity check. Must validate against known geodetic benchmarks (e.g., Vincenty reference values) to sub-meter precision. |
| `geographic_distance_batch()` | `geolocation.utils` | Geometry | **CRITICAL** | **None** | Vectorized Haversine for batch operations. Must confirm numerical equivalence with scalar version across edge cases: antipodal points, zero distance, pole-to-equator, and ±180° longitude wrap. |
| `interpolation_error_metrics()` | `geolocation.utils` | Geometry | **CRITICAL** | **None** | Computes RMS/mean/max/std geolocation error. These metrics are the acceptance criteria for geolocation accuracy — if RMS computation is wrong, an out-of-spec product passes QA. Must validate against hand-computed reference values. |
| `calculate_footprint()` | `geolocation.utils` | Geometry | **HIGH** | **None** | Generates GeoJSON polygon footprints from corner coordinates. An incorrect footprint leads to wrong spatial index queries, missed collections, and false negatives in area-of-interest searches. Must test degenerate cases (collinear points, < 3 vertices, antimeridian crossing). |
| `bounds_from_corners()` | `geolocation.utils` | Geometry | **HIGH** | **None** | Bounding-box computation from corner coordinates. Incorrect bounds cause geolocation clipping, missed intersections, and tiling errors. Must test single-point input, collinear points, and antimeridian wraparound. |
| `check_pixel_bounds()` | `geolocation.utils` | Geometry | **HIGH** | **None** | Pixel-coordinate boundary validation with tolerance. A bug here silently allows out-of-bounds array indexing or incorrectly rejects valid edge pixels. Must test exact boundary, tolerance edge, and negative coordinates. |
| `sample_image_perimeter()` | `geolocation.utils` | Geometry | **HIGH** | **None** | Generates perimeter sample points for footprint estimation. Incorrect sampling produces distorted footprints. Must verify corner inclusion, uniform spacing, and degenerate shapes (1-row, 1-col images). |
| `Geolocation.image_to_latlon()` | `geolocation.base` | Coordinate Transform | **CRITICAL** | Indirect via subclass tests | ABC contract for pixel→geographic conversion. The abstract interface's scalar/array dispatch logic must be verified through comprehensive subclass testing. |
| `Geolocation.latlon_to_image()` | `geolocation.base` | Coordinate Transform | **CRITICAL** | Indirect via subclass tests | ABC contract for geographic→pixel conversion. Round-trip consistency (image→latlon→image) must hold to sub-pixel precision. |
| `GCPGeolocation` | `geolocation.sar.gcp` | Coordinate Transform | **CRITICAL** | 12 tests | Delaunay-interpolated GCP geolocation for SAR (BIOMASS). Existing tests should be expanded to cover: degenerate triangulations, extrapolation beyond convex hull, dense/sparse GCP grids, and round-trip precision validation. |
| `SICDGeolocation` | `geolocation.sar.sicd` | Coordinate Transform | **CRITICAL** | 20 tests | SICD-native geolocation via sarpy/sarkit. Tests exist but must verify sub-pixel round-trip accuracy and agreement with NGA-published SICD test vectors if available. |
| `AffineGeolocation` | `geolocation.eo.affine` | Coordinate Transform | **CRITICAL** | 27 tests | Affine + pyproj geolocation for geocoded rasters. Well-tested; ensure CRS edge cases (polar stereographic, cross-UTM-zone) and sub-pixel precision are covered. |
| `DTEDElevation` | `geolocation.elevation` | Elevation / DEM | **CRITICAL** | **Indirect (22 total for elevation module)** | DTED tile-based terrain lookup. Incorrect elevation directly corrupts image-to-ground geolocation for non-nadir sensors. Must test: tile boundary interpolation, missing tile fallback, DTED levels 0/1/2, and known-elevation reference points. |
| `GeoTIFFDEM` | `geolocation.elevation` | Elevation / DEM | **CRITICAL** | **Indirect** | GeoTIFF DEM reader. Same geolocation risk as DTED. Must test: out-of-bounds queries, NoData handling, interpolation at pixel edges, and CRS mismatch between query and DEM. |
| `GeoidCorrection` | `geolocation.elevation` | Elevation / DEM | **CRITICAL** | **Indirect** | EGM96 geoid undulation lookup (MSL→HAE). A geoid error of even 1 meter propagates directly into height-above-ellipsoid, corrupting all 3D ground projections. Must validate against published EGM96 test points. |
| `ConstantElevation` | `geolocation.elevation` | Elevation / DEM | **MEDIUM** | **Indirect** | Fixed-height fallback. Simple but must verify it returns the constant for any query and handles array inputs correctly. |
| `Orthorectifier` | `image_processing.ortho` | Spatial Transform | **CRITICAL** | 26 tests | Reprojects imagery from native acquisition geometry to geographic grid. Geolocation accuracy of the output product depends entirely on this. Must verify pixel-level agreement with reference ortho products. |
| `OutputGrid` | `image_processing.ortho` | Spatial Transform | **HIGH** | Tested with Orthorectifier | Defines the geographic grid specification. Incorrect grid bounds or pixel size leads to misregistered output products. |
| `PauliDecomposition` | `image_processing.decomposition` | SAR Processing | **HIGH** | 35 tests (module) | Quad-pol Pauli decomposition. Tests exist; ensure complex-valued edge cases (zero channels, single-pol fallback) are covered. |
| `SublookDecomposition` | `image_processing.sar` | SAR Processing | **HIGH** | 27 tests | Complex SAR sub-aperture splitting. Spectral leakage or incorrect bandwidth partitioning corrupts coherent change detection. |
| `ToDecibels` | `image_processing.intensity` | Radiometric | **HIGH** | 20 tests (module) | Linear→dB conversion. A log10 domain error silently compresses dynamic range. Must test: zero input (should handle gracefully), negative input, and known reference values. |
| `PercentileStretch` | `image_processing.intensity` | Radiometric | **MEDIUM** | 20 tests (module) | Percentile-based contrast stretch. Important for visualization but not mission-critical for analytics. |
| `Pipeline` | `image_processing.pipeline` | Orchestration | **HIGH** | **Limited** | Sequential composition of transforms. Must verify: ordering guarantees, error propagation, empty pipeline, single-step pipeline, and dtype preservation across steps. |
| `SICDReader` | `IO.sar` | Data Ingestion | **CRITICAL** | 6 tests (SAR module) | SICD complex SAR reader. Complex-valued data integrity is paramount — phase corruption invalidates all coherent processing. Must test: complex dtype preservation, magnitude/phase separation, and metadata SICD structure compliance. |
| `CPHDReader` | `IO.sar` | Data Ingestion | **HIGH** | 6 tests (SAR module) | Compensated Phase History Data reader. Limited testing; phase history integrity is critical for SAR image formation. |
| `CRSDReader` | `IO.sar` | Data Ingestion | **HIGH** | 6 tests (SAR module) | CRSD reader. Similar risk profile to CPHD. |
| `SIDDReader` | `IO.sar` | Data Ingestion | **HIGH** | 6 tests (SAR module) | SIDD derived data reader. Metadata fidelity (exploitation parameters) is critical for downstream exploitation tools. |
| `BIOMASSL1Reader` | `IO.sar` | Data Ingestion | **HIGH** | 10 tests | ESA BIOMASS L1 SAR reader. Tests exist; ensure GCP extraction and complex data integrity are validated. |
| `GeoTIFFReader` | `IO.geotiff` | Data Ingestion | **HIGH** | 14 + 17 (TE) | Well-tested in both grdl and grdl-te. Maintain coverage; add edge cases for multi-band, tiled COG, and unusual dtypes. |
| `HDF5Reader` | `IO.hdf5` | Data Ingestion | **HIGH** | 25 + 17 (TE) | Well-tested. Ensure SDS discovery, group hierarchy navigation, and attribute extraction remain covered. |
| `JP2Reader` | `IO.jpeg2000` | Data Ingestion | **HIGH** | 22 + 18 (TE) | Well-tested. Ensure 15-bit-in-16-bit container handling and resolution level reads are validated. |
| `NITFReader` | `IO.nitf` | Data Ingestion | **HIGH** | 10 + 16 (TE) | Tested in both repos. Ensure TRE (Tagged Record Extension) parsing and multi-image-segment support are validated. |
| `ASTERReader` | `IO.ir` | Data Ingestion | **HIGH** | 21 tests | ASTER thermal and GDEM reader. Tests exist; verify temperature/radiance conversion accuracy. |
| `VIIRSReader` | `IO.multispectral` | Data Ingestion | **HIGH** | 27 tests | VIIRS reader. Tests exist; verify reflectance scale factor application and QA flag interpretation. |
| `Sentinel2Reader` | `IO.eo` | Data Ingestion | **HIGH** | 21 tests | Sentinel-2 L1C/L2A reader. Tests exist; verify SAFE directory traversal and band resolution mapping. |
| `ChipExtractor` | `data_prep` | Data Preparation | **CRITICAL** | 36 tests | Point-centered chip region computation. Incorrect chip bounds lead to spatial misalignment in ML training data. Well-tested — maintain. |
| `Tiler` | `data_prep` | Data Preparation | **HIGH** | 24 tests | Stride-based tile grid computation. Must ensure: complete coverage without gaps, correct overlap calculation, and boundary tile handling. |
| `Normalizer` | `data_prep` | Data Preparation | **HIGH** | 35 tests | Intensity normalization (minmax, zscore, percentile). Incorrect normalization corrupts ML model inputs. Well-tested — maintain. |
| `AffineCoRegistration` | `coregistration` | Image Alignment | **CRITICAL** | 35 tests (module) | Affine transform estimation from control points. Registration errors propagate into all multi-temporal analysis. |
| `ProjectiveCoRegistration` | `coregistration` | Image Alignment | **CRITICAL** | 35 tests (module) | Homography estimation. Same risk as affine; additionally must handle near-degenerate point configurations. |
| `FeatureMatchCoRegistration` | `coregistration` | Image Alignment | **HIGH** | 20 tests | Automated feature-based registration. Tests exist; ensure robustness to low-texture scenes and outlier matches. |
| `open_image()` | `IO` | Data Ingestion | **HIGH** | **Limited** | Universal file opener with format auto-detection. A format-sniffing bug silently opens a NITF as GeoTIFF (wrong metadata model). Must test each format path and unknown-extension fallback. |
| `open_sar()` | `IO.sar` | Data Ingestion | **HIGH** | **Limited** | SAR format auto-detection. Must correctly distinguish SICD/CPHD/CRSD/SIDD. |
| `get_writer()` / `write()` | `IO` | Data Egress | **HIGH** | 13 + 8 + 8 + 12 + 12 + 11 | Writer factory and convenience function. Tests exist across multiple writer test files. Maintain coverage. |
| `transform_detection_set()` | `transforms` | Vector Transform | **HIGH** | 36 tests | Applies spatial transforms to detection geometries. Tests exist. Ensure transform composition and edge cases (empty set, identity transform) are covered. |
| `transform_detection()` | `transforms` | Vector Transform | **HIGH** | 36 tests (module) | Single-detection transform. Tests exist. |
| `transform_pixel_geometry()` | `transforms` | Vector Transform | **HIGH** | 36 tests (module) | Shapely geometry transform. Tests exist; ensure polygon, multipoint, and degenerate geometries are handled. |

### Should Have — Important (File I/O Wrappers, API Convenience, Filtering)

| Function / Class | Module | Category | Risk Level | Existing Tests | Logic for Priority |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `ImageReader` (ABC) | `IO.base` | Interface Contract | **MEDIUM** | Indirect via all readers | Abstract interface. Tested through concrete implementations. Should have explicit ABC contract tests (e.g., context manager protocol, required method signatures). |
| `ImageWriter` (ABC) | `IO.base` | Interface Contract | **MEDIUM** | Indirect via all writers | Same as ImageReader — verify ABC contract enforcement. |
| `ImageMetadata` | `IO.models` | Metadata | **MEDIUM** | 68 tests | Base metadata model. Well-tested; maintain. |
| `SICDMetadata` | `IO.models.sicd` | Metadata | **MEDIUM** | 68 tests (module) | SICD metadata container with 40+ component classes. Model tests exist; should verify round-trip serialization and NGA-compliant field validation. |
| `SIDDMetadata` | `IO.models.sidd` | Metadata | **MEDIUM** | 68 tests (module) | SIDD metadata container. Similar to SICD — verify field completeness. |
| `BIOMASSMetadata` | `IO.models.biomass` | Metadata | **MEDIUM** | Covered via BIOMASS reader tests | ESA BIOMASS metadata. Verify required fields for GCP geolocation extraction. |
| `VIIRSMetadata` | `IO.models.viirs` | Metadata | **MEDIUM** | Covered via VIIRS reader tests | VIIRS metadata. Verify band mapping and scale factors. |
| `ASTERMetadata` | `IO.models.aster` | Metadata | **MEDIUM** | Covered via ASTER reader tests | ASTER metadata. Verify thermal band calibration coefficients. |
| `Sentinel2Metadata` | `IO.models.sentinel2` | Metadata | **MEDIUM** | Covered via Sentinel-2 reader tests | Sentinel-2 metadata. Verify SAFE structure parsing. |
| `BIOMASSCatalog` | `IO.sar` | Catalog / Discovery | **MEDIUM** | **None** | BIOMASS product discovery and download. Network-dependent; should have mock-based tests for catalog parsing and credential flow. |
| `load_credentials()` | `IO.sar` | Authentication | **MEDIUM** | **None** | Credential loading for BIOMASS catalog. Should validate credential format and error handling without hitting real endpoints. |
| `open_eo()` | `IO.eo` | Convenience | **MEDIUM** | **Limited** | EO format auto-detection. Lower risk than `open_sar` since EO formats are simpler. |
| `open_ir()` | `IO.ir` | Convenience | **MEDIUM** | **Limited** | IR format auto-detection. Single-format (ASTER) currently. |
| `open_multispectral()` | `IO.multispectral` | Convenience | **MEDIUM** | **Limited** | Multispectral format auto-detection. Single-format (VIIRS) currently. |
| `open_biomass()` | `IO.sar` | Convenience | **MEDIUM** | **Limited** | BIOMASS-specific opener. Thin wrapper; lower risk. |
| `GeoTIFFWriter` | `IO.geotiff` | Data Egress | **MEDIUM** | 12 tests | Well-tested. Maintain; verify CRS and transform preservation on write. |
| `HDF5Writer` | `IO.hdf5` | Data Egress | **MEDIUM** | 11 tests | Well-tested. Maintain; verify attribute roundtrip. |
| `NITFWriter` | `IO.nitf` | Data Egress | **MEDIUM** | 12 tests | Well-tested. Maintain; verify TRE writing. |
| `NumpyWriter` | `IO.numpy_io` | Data Egress | **LOW** | 8 tests | Well-tested. Simple format. |
| `PngWriter` | `IO.png` | Data Egress | **LOW** | 8 tests | Well-tested. Non-geospatial format. |
| `Detection` / `DetectionSet` | `image_processing.detection` | Data Model | **MEDIUM** | 53 tests | Detection output model with shapely geometry. Well-tested. |
| `FieldDefinition` / `Fields` | `image_processing.detection` | Data Dictionary | **MEDIUM** | 53 tests (module) | Standardized field names and definitions. Tested via detection module. |
| `ImageProcessor` (ABC) | `image_processing.base` | Interface Contract | **MEDIUM** | Indirect via subclasses | Abstract base. Should have explicit contract tests. |
| `ImageTransform` (ABC) | `image_processing.base` | Interface Contract | **MEDIUM** | Indirect via subclasses | Abstract base. Should have explicit contract tests. |
| `ImageDetector` (ABC) | `image_processing.base` | Interface Contract | **MEDIUM** | Indirect via subclasses | Abstract base. Should have explicit contract tests. |
| `CoRegistration` (ABC) | `coregistration.base` | Interface Contract | **MEDIUM** | Indirect via subclasses | Abstract base. Should have explicit contract tests. |
| `RegistrationResult` | `coregistration` | Data Model | **MEDIUM** | 35 tests (module) | Result container. Tested via coregistration module. |
| `processor_version()` | `image_processing.versioning` | Decorator | **MEDIUM** | 30 tests | Version decorator. Well-tested. |
| `processor_tags()` | `image_processing.versioning` | Decorator | **MEDIUM** | 30 tests | Metadata decorator. Well-tested. |
| `collect_param_specs()` | `image_processing.versioning` | Introspection | **MEDIUM** | 66 tests (tunable module) | Parameter introspection. Well-tested. |
| `DetectionInputSpec` | `image_processing.versioning` | Decorator | **MEDIUM** | 30 tests (module) | Detection input declaration. Tested via versioning module. |
| `NoGeolocation` | `geolocation.base` | Fallback | **MEDIUM** | **None** | Identity fallback when no geolocation exists. Simple but should confirm it raises appropriate warnings/errors rather than silently returning garbage coordinates. |

### Could Have — Low Priority (Internal Helpers, String Formatting, Enums)

| Function / Class | Module | Category | Risk Level | Existing Tests | Logic for Priority |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `ImageModality` | `vocabulary` | Enum | **LOW** | 10 tests | Enumeration. Well-tested. |
| `ProcessorCategory` | `vocabulary` | Enum | **LOW** | 10 tests | Enumeration. Well-tested. |
| `DetectionType` | `vocabulary` | Enum | **LOW** | 10 tests | Enumeration. Well-tested. |
| `SegmentationType` | `vocabulary` | Enum | **LOW** | 10 tests | Enumeration. Well-tested. |
| `ExecutionPhase` | `vocabulary` | Enum | **LOW** | 10 tests | Enumeration. Well-tested. |
| `GpuCapability` | `vocabulary` | Enum | **LOW** | 10 tests | Enumeration. Well-tested. |
| `OutputFormat` | `vocabulary` | Enum | **LOW** | 10 tests | Enumeration. Well-tested. |
| `GrdlError` | `exceptions` | Exception | **LOW** | **None** | Base exception class. Trivial to test but low risk — Python exception semantics are well-defined. |
| `ValidationError` | `exceptions` | Exception | **LOW** | **None** | Exception subclass. Trivial. |
| `ProcessorError` | `exceptions` | Exception | **LOW** | **None** | Exception subclass. Trivial. |
| `DependencyError` | `exceptions` | Exception | **LOW** | **None** | Exception subclass. Trivial. |
| `GeolocationError` | `exceptions` | Exception | **LOW** | **None** | Exception subclass. Trivial. |
| `Range` | `image_processing.params` | Constraint Marker | **LOW** | 66 tests (tunable) | Annotation marker. Tested via tunable parameter system. |
| `Options` | `image_processing.params` | Constraint Marker | **LOW** | 66 tests (tunable) | Annotation marker. Tested via tunable parameter system. |
| `Desc` | `image_processing.params` | Constraint Marker | **LOW** | 66 tests (tunable) | Annotation marker. Tested via tunable parameter system. |
| `ParamSpec` | `image_processing.params` | Introspection | **LOW** | 66 tests (tunable) | Data class. Tested via tunable parameter system. |
| `ChipBase` (ABC) | `data_prep.base` | Interface Contract | **LOW** | Indirect via subclasses | Abstract base. Tested through ChipExtractor and Tiler. |
| `ChipRegion` | `data_prep.base` | NamedTuple | **LOW** | Indirect | Simple named tuple. No logic to test. |
| `ElevationModel` (ABC) | `geolocation.elevation.base` | Interface Contract | **LOW** | Indirect | Abstract base. Tested through concrete implementations. |
| `BandwiseTransformMixin` | `image_processing.base` | Mixin | **LOW** | Indirect via subclasses | Auto-apply mixin. Tested through transforms that use it. |
| `globalprocessor()` | `image_processing.versioning` | Decorator | **LOW** | 13 tests | Global-pass decorator. Tested. |
| `XYZ`, `LatLon`, `LatLonHAE`, `RowCol` | `IO.models.common` | Data Primitives | **LOW** | 68 tests (models) | Simple dataclasses. Well-tested. |
| `Poly1D`, `Poly2D`, `XYZPoly` | `IO.models.common` | Data Primitives | **LOW** | 68 tests (models) | Polynomial representations. Well-tested. |
| `CatalogInterface` (ABC) | `IO.base` | Interface Contract | **LOW** | **None** | Abstract interface for artifact catalogs. Low risk — only GRDK uses it. |
| `PolarimetricDecomposition` (ABC) | `image_processing.decomposition` | Interface Contract | **LOW** | Indirect | Abstract base. Tested via PauliDecomposition. |

### Won't Have (This Cycle) — GRDK GUI Components

| Function / Class | Module | Category | Risk Level | Existing Tests | Logic for Priority |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `normalize_array()` | `grdk.viewers.image_canvas` | Visualization | **LOW** | 19 tests | Well-tested; visualization-only. |
| `ImageCanvas` | `grdk.viewers.image_canvas` | GUI Widget | **LOW** | 19 tests | Qt widget; tested with synthetic data. |
| `build_param_controls()` | `grdk.widgets._param_controls` | GUI Builder | **LOW** | 11 tests | Dynamic UI builder; tested. |
| `OWImageLoader` through `OWPublisher` | `grdk.widgets.geodev` | GUI Workflows | **LOW** | 2 tests | Orange3 widgets; require GUI testing infrastructure. |
| `OWCatalogBrowser` through `OWUpdateMonitor` | `grdk.widgets.admin` | GUI Admin | **LOW** | 0 tests | Orange3 widgets; administrative functions. |
| All `grdk.core.*` classes | `grdk.core` | Business Logic | **MEDIUM** | 0 tests | Important but lower priority than GRDL core. Should be tested in a separate GRDK T&E effort. |
| All `grdk.catalog.*` classes | `grdk.catalog` | Catalog Logic | **MEDIUM** | 0 tests | SQLite catalog system. Important but GRDK-scoped. |

---

## Priority Matrix — GRDL-TE Expansion (Real-Data Validation)

The following GRDL modules currently have **zero real-data validation** in GRDL-TE and represent the highest-priority expansion targets:

| Target Module | Priority | Proposed Dataset | Rationale |
| :--- | :--- | :--- | :--- |
| `geolocation.sar.gcp` | **Must Have** | BIOMASS L1 (existing umbra data) | GCP geolocation is untested against real satellite data. Round-trip pixel→latlon→pixel accuracy with known ground truth is essential. |
| `geolocation.sar.sicd` | **Must Have** | Umbra SICD NITF (existing) | SICD geolocation is untested in GRDL-TE. Must validate image_to_latlon against metadata-embedded SCP (Scene Center Point). |
| `geolocation.eo.affine` | **Must Have** | Landsat COG (existing) | Affine geolocation against real GeoTIFF with known CRS. Verify pixel→latlon round-trip against USGS-published corner coordinates. |
| `geolocation.elevation` | **Must Have** | DTED / SRTM reference tiles | Elevation lookup accuracy directly affects geolocation. Need reference DEM tiles with known elevations at survey benchmarks. |
| `image_processing.ortho` | **Should Have** | BIOMASS L1 + reference ortho | Orthorectification output should be validated against a reference ortho product (e.g., Landsat) by measuring coregistration residuals. |
| `coregistration` | **Should Have** | Multi-temporal Landsat pair | Register two Landsat scenes from different dates and validate sub-pixel alignment against known tie points. |
| `IO.sar.sicd` (SICD-specific) | **Should Have** | Umbra SICD (existing) | Dedicated SICD reader tests beyond what NITFReader covers — complex dtype, phase integrity, SICD metadata completeness. |
| `data_prep` (with real data) | **Should Have** | All existing datasets | ChipExtractor/Tiler/Normalizer are tested in GRDL-TE Level 3 but should be expanded with more aggressive edge cases (1-pixel chips, whole-image chips, extreme normalization ranges). |

---

## Gap Analysis Summary

### Untested Public API Functions (Zero Direct Tests)

| # | Function | Module | Risk |
| :--- | :--- | :--- | :--- |
| 1 | `geographic_distance()` | `geolocation.utils` | CRITICAL |
| 2 | `geographic_distance_batch()` | `geolocation.utils` | CRITICAL |
| 3 | `interpolation_error_metrics()` | `geolocation.utils` | CRITICAL |
| 4 | `calculate_footprint()` | `geolocation.utils` | HIGH |
| 5 | `bounds_from_corners()` | `geolocation.utils` | HIGH |
| 6 | `check_pixel_bounds()` | `geolocation.utils` | HIGH |
| 7 | `sample_image_perimeter()` | `geolocation.utils` | HIGH |
| 8 | `NoGeolocation` | `geolocation.base` | MEDIUM |
| 9 | `BIOMASSCatalog` | `IO.sar` | MEDIUM |
| 10 | `load_credentials()` | `IO.sar` | MEDIUM |
| 11 | `open_image()` | `IO` | HIGH |
| 12 | `Pipeline` (direct) | `image_processing.pipeline` | HIGH |

### Modules with Indirect-Only Coverage (Tested via Integration, Not Isolated)

| Module | Covered By | Gap |
| :--- | :--- | :--- |
| `geolocation.elevation.dted` | `test_geolocation_elevation.py` | Need isolated unit tests for tile boundary cases |
| `geolocation.elevation.geotiff_dem` | `test_geolocation_elevation.py` | Need isolated unit tests for CRS mismatch handling |
| `geolocation.elevation.geoid` | `test_geolocation_elevation.py` | Need validation against published EGM96 reference points |
| `IO.sar.sicd` | `test_io_sar_readers.py` (6 tests) | Need dedicated SICD tests with complex data validation |
| `IO.sar.cphd` | `test_io_sar_readers.py` (6 tests) | Need dedicated CPHD phase history tests |
| `IO.sar.crsd` | `test_io_sar_readers.py` (6 tests) | Need dedicated CRSD tests |
| `IO.sar.sidd` | `test_io_sar_readers.py` (6 tests) | Need dedicated SIDD exploitation parameter tests |

---

## Recommended Implementation Order

**Sprint 1 — Geolocation Utilities (Highest Risk, Zero Coverage)**
1. `test_geolocation_utils.py` — All 7 utility functions with geodetic reference values
2. `test_geolocation_nogeolocation.py` — NoGeolocation fallback behavior

**Sprint 2 — Elevation Model Isolation**
3. `test_elevation_dted_isolated.py` — DTED tile boundary, interpolation, missing tiles
4. `test_elevation_geotiff_dem_isolated.py` — CRS mismatch, NoData, edge interpolation
5. `test_elevation_geoid_isolated.py` — EGM96 validation against published values

**Sprint 3 — SAR Reader Hardening**
6. `test_io_sicd_dedicated.py` — Complex dtype, phase, SICD metadata structure
7. `test_io_cphd_dedicated.py` — Phase history integrity
8. `test_io_sidd_dedicated.py` — Exploitation parameter completeness

**Sprint 4 — Pipeline & Convenience API**
9. `test_pipeline_isolated.py` — Composition, ordering, error propagation
10. `test_io_open_image.py` — Format auto-detection for all supported formats
11. `test_io_open_sar.py` — SAR format discrimination

**Sprint 5 — GRDL-TE Expansion (Real-Data Geolocation)**
12. `grdl-te/tests/test_geolocation_affine.py` — Landsat pixel→latlon round-trip
13. `grdl-te/tests/test_geolocation_sicd.py` — SICD geolocation with Umbra data
14. `grdl-te/tests/test_ortho_biomass.py` — Orthorectification validation

---

## GEOINT-Specific Edge Cases to Prioritize

| Edge Case | Affected Functions | Why It Matters |
| :--- | :--- | :--- |
| **Antimeridian crossing (±180° lon)** | `calculate_footprint`, `bounds_from_corners`, `geographic_distance` | Pacific theater imagery regularly crosses the antimeridian. Incorrect polygon splitting causes spatial indexing failures. |
| **Polar regions (lat > 85°)** | `geographic_distance`, `AffineGeolocation` | Haversine accuracy degrades near poles. UTM zones are undefined above 84°N. |
| **Date-line-adjacent UTM zones** | `AffineGeolocation`, `GeoTIFFReader` CRS handling | Zone 1 (180°W–174°W) and Zone 60 (174°E–180°E) create wrap-around edge cases. |
| **Sub-pixel precision loss** | `GCPGeolocation`, `SICDGeolocation`, `AffineGeolocation` | Round-trip (image→latlon→image) must hold to <0.5 pixel or geolocation accuracy claims are invalid. |
| **Complex SAR zero-fill regions** | `SICDReader`, `SublookDecomposition`, `PauliDecomposition` | SAR images contain zero-fill padding. Processing these regions as data corrupts statistics and detections. |
| **EGM96 geoid at ocean surface** | `GeoidCorrection` | Ocean areas have HAE ≈ geoid undulation. A sign error in the correction produces >100m height errors. |
| **DTED void cells (ocean/missing data)** | `DTEDElevation` | DTED tiles over ocean or unmapped areas contain NoData (-32767). Must not return this as a valid elevation. |
| **Float32 precision in WGS84** | All geolocation functions | Float32 has ~7 decimal digits. At the equator, 1e-7 degrees ≈ 1.1cm. Float32 truncation at high precision can lose centimeter-level accuracy. Ensure Float64 is used for all lat/lon operations. |

---

*End of document. This priority matrix should be reviewed and updated as tests are implemented and new modules are added to GRDL.*
