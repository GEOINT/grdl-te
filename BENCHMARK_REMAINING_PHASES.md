# GRDL-TE Benchmark Coverage — Phases 2–5

> **Status: ALL PHASES COMPLETE** (2026-02-24)
>
> All 78 components are benchmarked in `suite.py` (13 groups, ~103 `_bench()` calls).
> The YAML workflow is v2.0.0 with 28 steps. 27 validation test files and 6
> benchmarking test files are in place. All conftest.py fixtures and pyproject.toml
> markers have been added.

---

## Phase Summary

| Phase | Status | Date | Deliverables |
|-------|:------:|------|--------------|
| 1 | COMPLETE | 2026-02-19 | 15 synthetic-data components, 5 validation test files, `run_detection_benchmarks()` group |
| 2 | COMPLETE | 2026-02-24 | 16 real-data IO/geolocation/elevation components, 10 validation test files, conftest.py fixtures |
| 2b | COMPLETE | 2026-02-24 | 6 interpolators, `run_interpolation_benchmarks()` group, test_interpolation.py |
| 2c | COMPLETE | 2026-02-24 | TerraSARReader + TerraSARMetadata, test_io_terrasar.py |
| 3 | COMPLETE | 2026-02-24 | 7 SAR image formation, `run_image_formation_benchmarks()` group, YAML v2.0.0 (28 steps) |
| 4 | COMPLETE | 2026-02-24 | Verification, pyproject.toml markers, all fixtures confirmed |

---

## Phase 1: Synthetic-Data Components (COMPLETE)

Delivered 15 components using synthetic data only (no real data files required):

- **Detection** (6): CACFARDetector, GOCFARDetector, SOCFARDetector, OSCFARDetector, Detection/DetectionSet, Fields/FieldDefinition
- **Decomposition** (1): DualPolHAlpha added to `run_decomposition_benchmarks()`
- **Ortho** (3): OrthoPipeline, OrthoResult, compute_output_resolution added to `run_ortho_benchmarks()`
- **CoRegistration** (1): ProjectiveCoRegistration added to `run_coregistration_benchmarks()`
- **Geolocation** (1): NoGeolocation added to `run_geolocation_benchmarks()`
- **Infrastructure**: `run_detection_benchmarks()` new group, BENCHMARK_GROUPS grew from 10 to 11, 7 new pytest markers

**Validation test files created:**
- test_detection_cfar.py, test_detection_models.py
- test_decomposition_halpha.py
- test_ortho_pipeline.py
- test_coregistration_projective.py

---

## Phase 2: Real-Data IO Readers/Writers + conftest Fixtures (COMPLETE)

### 2.1 conftest.py Fixtures

All session-scoped data directory fixtures and `require_*` file fixtures added to
`tests/validation/conftest.py`:

- Data directories: `cphd_data_dir`, `crsd_data_dir`, `sidd_data_dir`, `sentinel1_data_dir`, `aster_data_dir`, `biomass_data_dir`, `dted_data_dir`, `dem_data_dir`, `geoid_data_dir`, `terrasar_data_dir`
- File fixtures: `require_cphd_file`, `require_crsd_file`, `require_sidd_file`, `require_sentinel1_file`, `require_aster_file`, `require_biomass_file`, `require_dted_dir`, `require_dem_file`, `require_geoid_file`, `require_terrasar_dir`

### 2.2 IO Reader/Writer Test Files

All follow the 3-level validation pattern (L1: structure, L2: data quality, L3: integration):

| Test File | Components | Data Source |
|-----------|-----------|-------------|
| test_io_sar_writers.py | SICDWriter, NITFWriter | Umbra SICD |
| test_io_cphd.py | CPHDReader | `data/cphd/*.cphd` |
| test_io_crsd.py | CRSDReader | `data/crsd/*.crsd` |
| test_io_sidd.py | SIDDReader | `data/sidd/*.nitf` |
| test_io_sentinel1.py | Sentinel1SLCReader | `data/sentinel1/*.SAFE` |
| test_io_sentinel2.py | Sentinel2Reader | `data/sentinel2/S2*.SAFE` |
| test_io_aster.py | ASTERReader | `data/aster/AST_L1T*.hdf` |
| test_io_biomass.py | BIOMASSL1Reader, BIOMASSCatalog | `data/biomass/BIO_S2_*.tif` |

### 2.3 suite.py Extensions

`_run_real_data_io()` extended with benchmark entries for: CPHDReader (read_full, read_pvp), CRSDReader, SIDDReader, Sentinel1SLCReader (read_full, read_chip), ASTERReader, BIOMASSL1Reader, SICDWriter (roundtrip), NITFWriter (synthetic).

### 2.4 SAR Processing

MultilookDecomposition (2x2, 3x3) and CSIProcessor added to `run_sar_processing_benchmarks()`.

**Validation:** test_sar_multilook.py

### 2.5 Elevation Models

DTEDElevation, GeoTIFFDEM, and GeoidCorrection added to `run_geolocation_benchmarks()` with real-data batch benchmarks (10,000 points).

**Validation:** test_elevation_models.py

### 2.6 Geolocation

Sentinel1SLCGeolocation added to `run_geolocation_benchmarks()`.

**Validation:** test_geolocation_sentinel1.py (also covers NoGeolocation)

### 2.7 pyproject.toml Markers Added

`cphd`, `crsd`, `sidd`, `sentinel1`, `sentinel2`, `aster`, `biomass`, `elevation`, `geolocation`, `terrasar`, `interpolation`, `image_formation`

---

## Phase 2b: Interpolation Benchmarks (COMPLETE)

All 6 interpolators in `grdl.interpolation` benchmarked. Synthetic data only (no real data needed).

### Delivered

- New group `run_interpolation_benchmarks()` in suite.py (12 individual benchmarks)
- Registered as `"interpolation"` in BENCHMARK_GROUPS
- Each interpolator tested at 2 parameter configurations

| Interpolator | Benchmark Variants |
|--------------|--------------------|
| LanczosInterpolator | a=3, a=5 |
| KaiserSincInterpolator | kl=8, kl=16 |
| LagrangeInterpolator | order=3, order=5 |
| FarrowInterpolator | f4_p3, f8_p5 |
| PolyphaseInterpolator | kl=8/ph=32, kl=16/ph=64 |
| ThiranDelayFilter | d=0.7/o=1, d=3.7/o=3 |

**Validation:** test_interpolation.py

---

## Phase 2c: TerraSAR-X Benchmarks (COMPLETE)

TerraSARReader and TerraSARMetadata added to `_run_real_data_io()` in suite.py.

- Benchmarks: `TerraSARReader.read_full.real_data`, `TerraSARReader.read_chip.1024x1024.real_data`
- Product directory auto-detection: `TSX1_*` / `TDX1_*` prefixes and annotation XMLs

**Validation:** test_io_terrasar.py

---

## Phase 3: SAR Image Formation + YAML Workflow (COMPLETE)

### 3.1 Image Formation Benchmarks

New group `run_image_formation_benchmarks()` in suite.py (7 individual benchmarks). Requires CPHD data.

| Component | Benchmark |
|-----------|-----------|
| CollectionGeometry | `.init` from CPHD metadata |
| PolarGrid | `.init` from CollectionGeometry |
| PolarFormatAlgorithm | `.form` on phase data |
| SubaperturePartitioner | `.partition` (4 sub-apertures) |
| RangeDopplerAlgorithm | `.form` (optional, data-dependent) |
| StripmapPFA | `.form` (optional) |
| FastBackProjection | `.form` (optional) |

Registered as `"image_formation"` in BENCHMARK_GROUPS.

**Validation:** test_sar_image_formation.py

### 3.2 YAML Workflow (v2.0.0)

`workflows/comprehensive_benchmark_workflow.yaml` updated to 28 steps:

| Stage | Steps | Components |
|-------|:-----:|------------|
| 1: Complex-domain speckle | 1 | ComplexLeeFilter |
| 2a: Phase gradient (fan-out) | 3 | PhaseGradientFilter (row, col, mag) |
| 2b: Amplitude conversion | 1 | ToDecibels |
| 3: Real-valued filters | 6 | LeeFilter, MeanFilter, GaussianFilter, MedianFilter, MinFilter, MaxFilter |
| 3: Statistical texture | 1 | StdDevFilter |
| 3b: Polarimetric decomposition | 1 | DualPolHAlpha |
| 3c: SAR sub-aperture | 2 | MultilookDecomposition, CSIProcessor |
| 4: CFAR detection | 4 | CACFARDetector, GOCFARDetector, SOCFARDetector, OSCFARDetector |
| 5: Contrast stretch | 1 | PercentileStretch |
| 6: Ortho pipeline | 1 | OrthoPipeline |
| Image formation branch | 7 | CollectionGeometry, PolarGrid, PFA, RDA, StripmapPFA, FFBP, SubaperturePartitioner |

---

## Phase 4: Verification & Gap Closure (COMPLETE)

### Verification Commands

```bash
# Synthetic tests (no data required)
pytest tests/validation/ -v -m "not requires_data" --tb=short

# Full validation suite (real-data tests skip when data absent)
pytest tests/validation/ -v --tb=short

# Benchmark suite smoke test
python -c "
from grdl_te.benchmarking.suite import run_suite
run_suite(size='small', only=['detection', 'interpolation'])
"

# Full benchmark suite
python -c "
from grdl_te.benchmarking.suite import run_suite
run_suite(size='small')
"

# Image formation (requires CPHD data)
python -c "
from grdl_te.benchmarking.suite import run_suite
run_suite(size='small', only=['image_formation'])
"

# YAML workflow validation
python -c "
from grdl_rt.api import load_workflow
wf = load_workflow('workflows/comprehensive_benchmark_workflow.yaml')
print(f'{len(wf.steps)} steps loaded')
for s in wf.steps:
    print(f'  {s.id}: {s.processor} (depends_on={s.depends_on})')
"
```

### Final Counts

| Metric | Count |
|--------|------:|
| Components benchmarked | 78 |
| Benchmark groups in suite.py | 13 |
| Individual `_bench()` calls | ~103 |
| YAML workflow steps | 28 |
| Validation test files | 27 |
| Benchmarking test files | 6 |
| conftest.py fixtures | 22 (11 data dirs + 11 require_* files) |
| pyproject.toml markers | 23 |

---

## Implementation Notes

### Important API Details

1. **SICDWriter**: `SICDWriter(filepath, metadata=SICDMetadata)` then `.write(data)` — metadata goes in constructor, NOT in write()
2. **CSIProcessor**: Uses `.apply(source)` NOT `.process()`. Default `normalization='none'` returns unnormalized float64; use `normalization='percentile'` for [0,1] range
3. **MultilookDecomposition.decompose()**: Returns `(num_looks, rows, cols)` — first axis is num_looks, NOT spatially reduced
4. **NoGeolocation**: Raises `NotImplementedError` on all transforms — it is NOT an identity transform
5. **RegistrationResult**: Uses `transform_matrix` (not `transform`) and `residual_rms` (not `rms_error`)
6. **`list_fields()`**: Returns `List[FieldDefinition]` objects — compare via `.name` attribute, not string equality
7. **DualPolHAlpha** `component_names`: `('entropy', 'alpha', 'anisotropy', 'span')`
8. **OrthoPipeline** builder: `.with_source_array()`, `.with_geolocation()`, `.with_resolution(lat, lon)`, `.with_interpolation(str)`, `.with_nodata(float)`, `.with_elevation(ElevationModel)`, `.run()` returns `OrthoResult`
9. **OrthoResult**: `.data`, `.shape`, `.geolocation_metadata` (dict with 'crs', 'transform'), `.output_grid`, `.save_geotiff(path)`
10. **CFAR detectors**: `guard_cells=3, training_cells=12, pfa=1e-3, min_pixels=9, assumption='gaussian'`
11. **Lambda capture in benchmarks**: Always use `setup=lambda _x=var: ((_x,), {})` to avoid closure issues
12. **Interpolator factories**: `lanczos_interpolator(a=3)`, `polyphase_interpolator(kernel_length, num_phases)`, etc. — return callable `Interpolator` instances with `(x_old, y_old, x_new) -> y_new`
13. **ThiranDelayFilter**: Requires `delay >= order - 0.5` (e.g., order=3 needs delay >= 2.5)
14. **TerraSARReader**: Constructor `(filepath, polarization='HH')`. Product auto-detection via `open_sar()` checks for `TSX1_`/`TDX1_` prefixes or annotation XMLs
15. **BIOMASSCatalog**: Requires `search_path` positional arg

### Skip Patterns

**Real-data validation tests:**
```python
try:
    from grdl.IO.sar import SomeReader
    _HAS_READER = True
except ImportError:
    _HAS_READER = False

pytestmark = [
    pytest.mark.some_marker,
    pytest.mark.skipif(not _HAS_READER, reason="SomeReader not available"),
]
# Data fixture uses require_data_file() from conftest.py
```

**Benchmark entries in suite.py:**
```python
try:
    from grdl.some_module import SomeClass
    # ... benchmark code ...
except (ImportError, Exception) as exc:
    print(f"  SKIP  SomeClass benchmarks ({exc})")
```

---

## Data Availability

| Data | Status | Notes |
|------|:------:|-------|
| Umbra SICD NITF | Available | `data/umbra/` |
| Landsat 8/9 COG | Available | `data/landsat/` |
| VIIRS VNP09GA | Available | `data/viirs/` |
| Sentinel-2 L2A | Available | `data/sentinel2/` |
| CPHD | Copy needed | SAR data store → `data/cphd/` |
| CRSD | Copy needed | SAR data store → `data/crsd/` |
| Sentinel-1 SLC | Copy needed | SAR data store → `data/sentinel1/` |
| TerraSAR-X | Copy needed | SAR data store → `data/terrasar/` |
| SIDD | Acquire needed | NGA samples / generate |
| ASTER L1T | Acquire needed | NASA Earthdata |
| BIOMASS L1 | Acquire needed | ESA Copernicus |
| DTED tiles | Acquire needed | USGS / OpenTopography |
| GeoTIFF DEM | Acquire needed | AWS Copernicus DEM |
| EGM96 Geoid | Acquire needed | GeographicLib |

All `data/*/` directories contain a `README.md` with acquisition instructions.
All tests and benchmarks skip gracefully when data is absent.
