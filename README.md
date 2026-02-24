# GRDL-TE: Testing, Evaluation & Benchmarking

GRDL-TE is the validation and benchmarking suite for the [GRDL](../grdl/) (GEOINT Rapid Development Library). It serves two purposes:

1. **Validation** — tests GRDL's public API against real-world satellite data with 3-level validation (format, quality, integration).
2. **Benchmarking** — profiles GRDL workflows and individual components, aggregates metrics across runs, and persists results for regression detection and cross-hardware comparison.

GRDL-TE is a *consumer* of GRDL — it only imports the public API. It never modifies GRDL internals.

## Design Principles

| Principle | Rule |
|-----------|------|
| **No Pass by Default** | A test only passes when the data is present AND the code functions correctly. |
| **Missing Data = Skipped** | If the required data file is absent, the test is skipped with a download instruction. |
| **Broken Code = Failed** | If data is present but the reader or utility malfunctions, the test fails hard. |
| **Meaningful Assertions** | Every test validates a specific property (reflectance bounds, CRS projection, complex integrity). No tests that only check "did it load." |

## Repository Structure

```
grdl-te/
├── grdl_te/                          # Installable package (pip install -e .)
│   ├── __init__.py                   # Public API re-exports
│   ├── __main__.py                   # CLI entry point (python -m grdl_te)
│   └── benchmarking/
│       ├── __init__.py               # Subpackage exports + lazy ActiveBenchmarkRunner
│       ├── models.py                 # HardwareSnapshot, AggregatedMetrics, BenchmarkRecord
│       ├── base.py                   # BenchmarkRunner, BenchmarkStore ABCs
│       ├── store.py                  # JSONBenchmarkStore (file-per-record persistence)
│       ├── source.py                 # BenchmarkSource (synthetic/file/array data factory)
│       ├── component.py              # ComponentBenchmark (single-function profiling)
│       ├── active.py                 # ActiveBenchmarkRunner (workflow N-run aggregation)
│       ├── suite.py                  # run_suite() orchestration + CLI benchmark groups
│       └── report.py                 # format_report, print_report, save_report
├── tests/
│   ├── validation/                   # 27 test files — GRDL API validation
│   │   ├── conftest.py               # Shared fixtures + graceful skip logic
│   │   ├── test_io_geotiff.py        # GeoTIFFReader — Landsat 8/9 COG
│   │   ├── test_io_hdf5.py           # HDF5Reader — VIIRS VNP09GA
│   │   ├── test_io_jpeg2000.py       # JP2Reader — Sentinel-2 Level-2A
│   │   ├── test_io_nitf.py           # NITFReader — Umbra SICD
│   │   ├── test_io_cphd.py           # CPHDReader — CPHD phase history
│   │   ├── test_io_crsd.py           # CRSDReader — CRSD format
│   │   ├── test_io_sidd.py           # SIDDReader — SIDD detected imagery
│   │   ├── test_io_sentinel1.py      # Sentinel1Reader — Sentinel-1 SLC
│   │   ├── test_io_sentinel2.py      # Sentinel2Reader — Sentinel-2 multi-band
│   │   ├── test_io_aster.py          # ASTERReader — ASTER L1T HDF
│   │   ├── test_io_biomass.py        # BIOMASSReader — BIOMASS L1 GeoTIFF
│   │   ├── test_io_terrasar.py       # TerraSARReader — TerraSAR-X/TanDEM-X
│   │   ├── test_io_sar_writers.py    # SAR writer round-trip tests
│   │   ├── test_geolocation_base.py  # Geolocation ABC contract + NoGeolocation
│   │   ├── test_geolocation_affine_real.py  # Affine transforms, UTM/WGS84
│   │   ├── test_geolocation_utils.py        # Coordinate transforms, geodetic calcs
│   │   ├── test_geolocation_elevation.py    # DEM-based elevation models
│   │   ├── test_geolocation_sentinel1.py    # Sentinel-1 geolocation
│   │   ├── test_elevation_models.py         # Elevation model validation
│   │   ├── test_detection_cfar.py           # CFAR detector algorithms
│   │   ├── test_detection_models.py         # Detection data models
│   │   ├── test_decomposition_halpha.py     # Polarimetric H/Alpha decomposition
│   │   ├── test_coregistration_projective.py  # Projective coregistration
│   │   ├── test_ortho_pipeline.py           # Orthorectification pipeline
│   │   ├── test_sar_image_formation.py      # SAR image formation
│   │   ├── test_sar_multilook.py            # Multilook processing
│   │   └── test_interpolation.py            # Interpolation algorithms
│   └── benchmarking/                 # 6 test files — benchmark infrastructure
│       ├── test_benchmark_models.py  # Dataclass serialization tests
│       ├── test_benchmark_store.py   # JSON store persistence tests
│       ├── test_benchmark_source.py  # BenchmarkSource data generation tests
│       ├── test_benchmark_report.py  # Report formatting tests
│       ├── test_active_runner.py     # Active runner iteration/aggregation tests
│       └── test_component_benchmark.py  # Component profiling tests
├── data/                             # Real-world data files (git-ignored)
│   ├── README.md                     # Data strategy documentation
│   ├── landsat/                      # Landsat 8/9 COG
│   ├── viirs/                        # VIIRS VNP09GA HDF5
│   ├── sentinel2/                    # Sentinel-2 JPEG2000
│   ├── umbra/                        # Umbra SICD NITF
│   ├── cphd/                         # CPHD phase history
│   ├── crsd/                         # CRSD format
│   ├── sidd/                         # SIDD detected imagery
│   ├── sentinel1/                    # Sentinel-1 SLC
│   ├── aster/                        # ASTER L1T HDF
│   ├── biomass/                      # BIOMASS L1 GeoTIFF
│   ├── terrasar/                     # TerraSAR-X/TanDEM-X
│   ├── dted/                         # DTED elevation tiles
│   ├── dem/                          # GeoTIFF DEM
│   └── geoid/                        # EGM96 geoid model
├── workflows/
│   └── comprehensive_benchmark_workflow.yaml  # Example SAR processing pipeline
├── .benchmarks/                      # Benchmark result storage (git-ignored)
│   ├── index.json                    # Lightweight index for fast filtering
│   └── records/                      # One JSON file per benchmark run
├── benchmark_examples.py             # Example active workflow benchmarking script
├── pyproject.toml                    # Package configuration + pytest markers
├── CLAUDE.md                         # Development guide and standards
├── LICENSE                           # MIT License
└── README.md
```

## Setup

### Environment

GRDL-TE shares the `grdl` conda environment with all GRDX repositories:

```bash
conda activate grdl
```

### Installation

```bash
# Core — models, store, component benchmarks, all tests
pip install -e .

# With workflow benchmarking (requires grdl-runtime)
pip install -e ".[benchmarking]"

# With dev tools (pytest-benchmark, pytest-xdist)
pip install -e ".[dev]"
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `grdl` | latest | Library under test |
| `pytest` | >=7.0 | Test framework |
| `pytest-cov` | >=4.0 | Coverage reporting |
| `numpy` | >=1.21 | Array operations |
| `h5py` | >=3.0 | HDF5 format support |
| `rasterio` | >=1.3 | GeoTIFF/NITF support (via GDAL) |

**Optional:**

| Package | Install extra | Purpose |
|---------|--------------|---------|
| `grdl-runtime` | `benchmarking` | Active workflow benchmarking |
| `pytest-benchmark` | `dev` | Benchmark comparison |
| `pytest-xdist` | `dev` | Parallel test execution (`-n auto`) |

## Test Data

Each IO reader requires **exactly one representative file**. Data files are git-ignored — only the README with download instructions is committed for each dataset.

### Core Readers

| Reader | Directory | File Pattern | Format | Size | Source |
|--------|-----------|-------------|--------|------|--------|
| **GeoTIFFReader** | `data/landsat/` | `LC0[89]*_SR_B*.TIF` | Cloud-Optimized GeoTIFF (uint16) | 50-150 MB | [USGS EarthExplorer](https://earthexplorer.usgs.gov) |
| **HDF5Reader** | `data/viirs/` | `V?P09GA*.h5` | HDF5/HDF-EOS5 (int16) | 100-300 MB | [LAADS DAAC](https://ladsweb.modaps.eosdis.nasa.gov) |
| **JP2Reader** | `data/sentinel2/` | `S2*.SAFE/.../R10m/*_B04_10m.jp2` | JPEG2000 15-bit (uint16) | 100-200 MB | [Copernicus Data Space](https://dataspace.copernicus.eu) |
| **NITFReader** | `data/umbra/` | `*.nitf` | NITF + SICD XML (complex64) | 50-500 MB | [Umbra Open Data (AWS S3)](https://umbra-open-data-catalog.s3.amazonaws.com/index.html) |

### SAR Formats

| Reader | Directory | File Pattern | Format |
|--------|-----------|-------------|--------|
| **CPHDReader** | `data/cphd/` | `*.cphd` | CPHD phase history |
| **CRSDReader** | `data/crsd/` | `*.crsd` | CRSD format |
| **SIDDReader** | `data/sidd/` | `*.nitf` | SIDD detected imagery |
| **TerraSARReader** | `data/terrasar/` | `TSX1_*/TDX1_*` | TerraSAR-X/TanDEM-X product |

### Additional Sensors

| Reader | Directory | File Pattern | Format |
|--------|-----------|-------------|--------|
| **Sentinel1Reader** | `data/sentinel1/` | `*.SAFE` | Sentinel-1 SLC |
| **ASTERReader** | `data/aster/` | `AST_L1T*.hdf` | ASTER L1T HDF |
| **BIOMASSReader** | `data/biomass/` | `BIO_S2_*.tif` | BIOMASS L1 GeoTIFF |

### Elevation Data

| Dataset | Directory | File Pattern |
|---------|-----------|-------------|
| **DTED** | `data/dted/` | `**/*.dt?` |
| **DEM** | `data/dem/` | `*.tif` |
| **Geoid** | `data/geoid/` | `*.pgm` |

Each `data/<dataset>/README.md` contains detailed download instructions, expected file properties, and format specifications.

## Test Architecture

**552 tests** across **33 test files** in two directories: `tests/validation/` (27 files) and `tests/benchmarking/` (6 files).

### 3-Level Validation

All IO test files follow a three-level validation structure:

**Level 1 — Format Validation**
- Reader instantiation via context manager
- Metadata extraction (format, rows, cols, dtype, CRS, bands)
- Shape and dtype consistency with metadata
- Chip reads from image center (validates real data, not just fill)
- Full image reads (with size guards for large files)
- Resource cleanup on context manager exit

**Level 2 — Data Quality**
- CRS/projection validation (UTM zone ranges, sinusoidal, etc.)
- NoData handling (masking, valid pixel statistics, variance)
- Value range bounds (reflectance scales, 15-bit encoding ceilings)
- Format-specific features (COG tiling/overviews, HDF-EOS structure, complex magnitude/phase, SAR speckle statistics)

**Level 3 — Integration** (marked `@pytest.mark.integration`)
- **ChipExtractor**: uniform chip partitioning with bounds validation
- **Normalizer**: MinMax, Z-score, and percentile normalization
- **Tiler**: overlapping tile grids with stride control
- **Pipelines**: end-to-end workflows (chip -> normalize -> validate batch)

### IO Reader Tests

| Test File | Reader | Data Source |
|-----------|--------|------------|
| `test_io_geotiff.py` | GeoTIFFReader | Landsat 8/9 COG |
| `test_io_hdf5.py` | HDF5Reader | VIIRS VNP09GA |
| `test_io_jpeg2000.py` | JP2Reader | Sentinel-2 Level-2A |
| `test_io_nitf.py` | NITFReader | Umbra SICD |
| `test_io_cphd.py` | CPHDReader | CPHD phase history |
| `test_io_crsd.py` | CRSDReader | CRSD format |
| `test_io_sidd.py` | SIDDReader | SIDD detected imagery |
| `test_io_sentinel1.py` | Sentinel1Reader | Sentinel-1 SLC |
| `test_io_sentinel2.py` | Sentinel2Reader | Sentinel-2 multi-band |
| `test_io_aster.py` | ASTERReader | ASTER L1T HDF |
| `test_io_biomass.py` | BIOMASSReader | BIOMASS L1 |
| `test_io_terrasar.py` | TerraSARReader | TerraSAR-X/TanDEM-X |
| `test_io_sar_writers.py` | SAR writers | Round-trip write/read |

### Geolocation Tests

| Test File | Coverage |
|-----------|----------|
| `test_geolocation_base.py` | Geolocation ABC contract, `NoGeolocation` fallback, scalar/array dispatch |
| `test_geolocation_affine_real.py` | Affine transforms with real CRS data, UTM/WGS84 forward-inverse round-trips |
| `test_geolocation_utils.py` | Coordinate transformation utilities, geodetic calculations |
| `test_geolocation_elevation.py` | DEM-based elevation models, 3D geolocation |
| `test_geolocation_sentinel1.py` | Sentinel-1 orbit-based geolocation |
| `test_elevation_models.py` | DTED, DEM, and geoid elevation model validation |

### Processing Tests

| Test File | Coverage |
|-----------|----------|
| `test_detection_cfar.py` | CFAR (Constant False Alarm Rate) detector algorithms |
| `test_detection_models.py` | Detection data model serialization and validation |
| `test_decomposition_halpha.py` | Polarimetric H/Alpha decomposition |
| `test_coregistration_projective.py` | Projective image coregistration |
| `test_ortho_pipeline.py` | Orthorectification pipeline end-to-end |
| `test_sar_image_formation.py` | SAR image formation algorithms |
| `test_sar_multilook.py` | Multilook processing and spatial averaging |
| `test_interpolation.py` | Interpolation algorithms (polyphase, Thiran delay) |

### Benchmarking Tests

Benchmarking tests validate the profiling infrastructure itself (no real data required):

| Test File | Coverage |
|-----------|----------|
| `test_benchmark_models.py` | `HardwareSnapshot`, `AggregatedMetrics`, `StepBenchmarkResult`, `BenchmarkRecord` serialization |
| `test_benchmark_store.py` | `JSONBenchmarkStore` save/load/query and index consistency |
| `test_benchmark_source.py` | `BenchmarkSource` synthetic/file/array data generation and lazy resolution |
| `test_benchmark_report.py` | Report formatting, terminal printing, file saving |
| `test_active_runner.py` | `ActiveBenchmarkRunner` iteration counting, warmup exclusion, per-step aggregation |
| `test_component_benchmark.py` | `ComponentBenchmark` timing, memory measurement, pytest integration |

### Graceful Skip Behavior

Tests skip cleanly when data is absent — they never fail due to missing files:

**With data present:**
```
tests/validation/test_io_geotiff.py    15 passed
tests/validation/test_io_hdf5.py       15 passed
tests/validation/test_io_jpeg2000.py   16 passed
tests/validation/test_io_nitf.py       15 passed
```

**With data absent:**
```
tests/validation/test_io_geotiff.py    15 skipped
tests/validation/test_io_hdf5.py       15 skipped
tests/validation/test_io_jpeg2000.py   16 skipped
tests/validation/test_io_nitf.py       15 skipped
```

Each skip message includes the expected file pattern and a path to the README with download instructions.

## Running Tests

```bash
conda activate grdl

# Full suite (missing data files skip cleanly)
pytest

# Specific reader
pytest tests/validation/test_io_geotiff.py -v        # Landsat
pytest tests/validation/test_io_hdf5.py -v            # VIIRS
pytest tests/validation/test_io_jpeg2000.py -v        # Sentinel-2
pytest tests/validation/test_io_nitf.py -v            # Umbra
pytest tests/validation/test_io_cphd.py -v            # CPHD
pytest tests/validation/test_io_sentinel1.py -v       # Sentinel-1
pytest tests/validation/test_io_terrasar.py -v        # TerraSAR-X

# Geolocation tests
pytest tests/validation/test_geolocation_base.py tests/validation/test_geolocation_utils.py -v
pytest tests/validation/test_geolocation_affine_real.py -v
pytest tests/validation/test_geolocation_elevation.py -v

# Processing tests
pytest tests/validation/test_detection_cfar.py -v
pytest tests/validation/test_decomposition_halpha.py -v
pytest tests/validation/test_sar_image_formation.py -v

# Benchmarking infrastructure tests
pytest tests/benchmarking/ -v

# By marker
pytest -m landsat                     # All Landsat tests
pytest -m viirs                       # All VIIRS tests
pytest -m geolocation                 # All geolocation tests
pytest -m integration                 # Only Level 3 integration tests
pytest -m "nitf and not slow"         # NITF tests, skip slow ones
pytest -m benchmark                   # Benchmarking infrastructure tests
pytest -m sar                         # SAR processing tests
pytest -m detection                   # Detection algorithm tests
pytest -m decomposition               # Polarimetric decomposition tests
pytest -m interpolation               # Interpolation tests

# Skip all data-dependent tests
pytest -m "not requires_data"
```

### Test Markers

| Marker | Purpose |
|--------|---------|
| `landsat` | Landsat 8/9 tests (GeoTIFFReader) |
| `viirs` | VIIRS VNP09GA tests (HDF5Reader) |
| `sentinel2` | Sentinel-2 tests (JP2Reader) |
| `nitf` | Umbra SICD tests (NITFReader) |
| `cphd` | CPHD format tests |
| `crsd` | CRSD format tests |
| `sidd` | SIDD format tests |
| `sentinel1` | Sentinel-1 SLC tests |
| `aster` | ASTER L1T tests |
| `biomass` | BIOMASS L1 tests |
| `terrasar` | TerraSAR-X/TanDEM-X tests |
| `geolocation` | Geolocation utility and coordinate transform tests |
| `elevation` | Elevation model tests |
| `requires_data` | Test requires real data files in `data/` |
| `slow` | Long-running test (large file reads, full pipelines) |
| `integration` | Level 3 tests (ChipExtractor, Normalizer, Tiler workflows) |
| `benchmark` | Performance benchmark tests |
| `sar` | SAR-specific processing tests |
| `image_formation` | SAR image formation tests |
| `detection` | Detection model tests |
| `cfar` | CFAR detector tests |
| `decomposition` | Polarimetric decomposition tests |
| `ortho` | Orthorectification tests |
| `coregistration` | CoRegistration tests |
| `interpolation` | Interpolation algorithm tests |

## Benchmarking

The `grdl_te` package provides infrastructure for profiling GRDL workflows and individual components.

### CLI Benchmark Suite

Run the full benchmark suite from the command line:

```bash
python -m grdl_te                              # medium arrays, 10 iterations
python -m grdl_te --size small -n 5            # quick run
python -m grdl_te --size large -n 20           # thorough run
python -m grdl_te --only filters intensity     # specific benchmark groups
python -m grdl_te --skip-workflow              # component benchmarks only
python -m grdl_te --store-dir ./results        # custom output directory
python -m grdl_te --report                     # print report to terminal
python -m grdl_te --report ./reports/          # save report to directory
python -m grdl_te --report ./my_report.txt     # save report to file
```

**Array size presets:**

| Preset | Dimensions |
|--------|-----------|
| `small` | 512 x 512 |
| `medium` | 2048 x 2048 |
| `large` | 4096 x 4096 |

**Benchmark groups (13):**

| Group | Coverage |
|-------|----------|
| `filters` | Image processing filters (Mean, Gaussian, Median, Lee, ComplexLee, PhaseGradient) |
| `intensity` | Intensity transforms (ToDecibels, PercentileStretch) |
| `sar` | SAR-specific processing (SublookDecomposition, multilook) |
| `decomposition` | Polarimetric decomposition (DualPolHAlpha) |
| `detection` | Detection algorithms (CFAR detector) |
| `ortho` | Orthorectification pipelines |
| `coregistration` | Image coregistration |
| `io` | IO readers/writers |
| `multilook` | Multilook processing |
| `interpolation` | Interpolation algorithms |
| `pipelines` | End-to-end workflow benchmarks |
| `data_prep` | Data preparation (ChipExtractor, Normalizer, Tiler) |
| `geolocation` | Geolocation transformations |

### Active Workflow Benchmarking

Run a grdl-runtime `Workflow` N times, aggregate per-step metrics, and persist results:

```python
from grdl_rt import Workflow
from grdl_rt.api import load_workflow
from grdl_te.benchmarking import ActiveBenchmarkRunner, BenchmarkSource, JSONBenchmarkStore

store = JSONBenchmarkStore()

# ==== Pass a declared wf ====
wf = (
    Workflow("SAR Pipeline", modalities=["SAR"])
    .reader(SICDReader)
    .step(SublookDecomposition, num_looks=3)
    .step(ToDecibels)
)
runner = ActiveBenchmarkRunner(wf, iterations=10, warmup=2, store=store)
record = runner.run(source="image.nitf", prefer_gpu=True)

# ==== Load in a yaml workflow ====
wf = load_workflow("path/to/my_workflow.yaml")
source = BenchmarkSource.synthetic("medium")

runner = ActiveBenchmarkRunner(
    workflow=wf, source=source, iterations=5, warmup=1, store=store,
)
record = runner.run()

# record.total_wall_time.mean, .stddev, .p95
# record.step_results[0].wall_time_s.mean
# record.hardware.cpu_count, .gpu_devices
```

### Component Benchmarking

Profile individual GRDL functions outside of a workflow context:

```python
from grdl.data_prep import Normalizer
from grdl_te.benchmarking import ComponentBenchmark

image = np.random.rand(4096, 4096).astype(np.float32)
norm = Normalizer(method='minmax')

bench = ComponentBenchmark(
    name="Normalizer.minmax.4k",
    fn=norm.normalize,
    setup=lambda: ((image,), {}),
    iterations=20,
    warmup=3,
)
record = bench.run()
```

### Benchmark Data Sources

The `BenchmarkSource` class provides a unified interface for benchmark input data:

```python
from grdl_te.benchmarking import BenchmarkSource

# Synthetic data (lazy generation with caching)
source = BenchmarkSource.synthetic("medium")   # 2048x2048
source = BenchmarkSource.synthetic("small")    # 512x512
source = BenchmarkSource.synthetic("large")    # 4096x4096

# Real data file
source = BenchmarkSource.from_file("path/to/image.nitf")

# Existing array
source = BenchmarkSource.from_array(my_array)
```

### Result Storage

Results are stored as JSON files in `.benchmarks/`:

```
.benchmarks/
  index.json              # lightweight index for fast filtering
  records/
    <uuid>.json           # full BenchmarkRecord per run
```

Each record captures hardware state (`HardwareSnapshot`), per-step timing/memory aggregations (`StepBenchmarkResult`), and raw per-iteration metrics for lossless analysis.

### Key Types

| Type | Purpose |
|------|---------|
| `HardwareSnapshot` | Frozen machine state (CPU, RAM, GPU, platform) at benchmark time |
| `AggregatedMetrics` | Statistics (min, max, mean, median, stddev, p95) across N runs |
| `StepBenchmarkResult` | Per-step aggregated wall time, CPU time, memory, GPU usage |
| `BenchmarkRecord` | Complete benchmark result — the atomic unit of persistence |
| `BenchmarkSource` | Unified data source factory (synthetic, file, array) with lazy generation |
| `ActiveBenchmarkRunner` | Runs a Workflow N times with warmup and aggregates metrics |
| `ComponentBenchmark` | Profiles a single callable with timing and tracemalloc |
| `JSONBenchmarkStore` | File-per-record JSON persistence with index |
| `as_pytest_benchmark` | Helper to integrate ComponentBenchmark with pytest-benchmark |
| `run_suite` | Orchestrates multiple benchmark groups from config or CLI |
| `format_report` | Format benchmark results as structured text |
| `print_report` | Print formatted report to terminal |
| `save_report` | Save formatted report to file or directory |

### Example Workflow

The `workflows/` directory contains example grdl-runtime workflow definitions for benchmarking. `comprehensive_benchmark_workflow.yaml` defines a multi-stage SAR processing pipeline (complex speckle filtering, phase gradient analysis, amplitude conversion, rank/linear/statistical filters, and contrast stretching).

`benchmark_examples.py` in the repository root demonstrates active workflow benchmarking with `ActiveBenchmarkRunner` at multiple scales (small, medium, large).

### Future Phases

- **Passive Monitoring** — `ExecutionHook` that captures metrics from production workflows
- **Forensic Analysis** — load a completed workflow result, substitute step metrics to simulate "what if" scenarios (e.g., GPU vs CPU)
- **Comparison & Regression Detection** — compare benchmark records across runs with configurable thresholds
- **Cross-Hardware Prediction** — collect results from different machines, predict performance on new hardware

## Adding a New Reader Test

1. **Select ONE representative dataset** — prioritize open data, production quality
2. **Create** `data/<dataset>/README.md` with download instructions and file format specifications
3. **Add fixtures** to `tests/validation/conftest.py` using `require_data_file()`
4. **Create test file** `tests/validation/test_io_<reader>.py` with 3-level structure (format, quality, integration)
5. **Register markers** in `pyproject.toml`

## Dependency Management

### Source of Truth: `pyproject.toml`

All dependencies are defined in `pyproject.toml`. Keep these files synchronized:

- **`pyproject.toml`** — source of truth for versions and dependencies
- **`requirements.txt`** — regenerate with `pip freeze > requirements.txt` after updating `pyproject.toml`

**Note:** GRDL-TE is a **validation suite, not a published library**, so there is no `.github/workflows/publish.yml` or PyPI versioning requirement.

### Updating Dependencies

1. Update dependencies in `pyproject.toml` (add new packages, change versions, create/rename extras)
2. Install dependencies: `pip install -e ".[all,dev]"` (or appropriate extras for your work)
3. If `requirements.txt` exists, regenerate it: `pip freeze > requirements.txt`
4. Commit both files

See [CLAUDE.md](CLAUDE.md#dependency-management) for detailed dependency management guidelines.

## License

MIT License — see [LICENSE](LICENSE) for details.
