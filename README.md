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
│       ├── component.py              # ComponentBenchmark (single-function profiling)
│       ├── active.py                 # ActiveBenchmarkRunner (workflow N-run aggregation)
│       └── suite.py                  # run_suite() orchestration + CLI benchmark groups
├── tests/
│   ├── conftest.py                   # Shared fixtures + graceful skip logic
│   ├── test_io_geotiff.py            # GeoTIFFReader — Landsat 8/9 COG
│   ├── test_io_hdf5.py              # HDF5Reader — VIIRS VNP09GA
│   ├── test_io_jpeg2000.py           # JP2Reader — Sentinel-2 Level-2A
│   ├── test_io_nitf.py              # NITFReader — Umbra SICD
│   ├── test_geolocation_base.py      # Geolocation ABC contract + NoGeolocation fallback
│   ├── test_geolocation_affine_real.py # Affine transforms, UTM/WGS84 round-trip
│   ├── test_geolocation_utils.py     # Coordinate transforms, geodetic calculations
│   ├── test_geolocation_elevation.py # DEM-based elevation models
│   ├── test_benchmark_models.py      # Benchmark dataclass tests
│   ├── test_benchmark_store.py       # JSON store persistence tests
│   ├── test_active_runner.py         # Active benchmark runner tests
│   └── test_component_benchmark.py   # Component profiling tests
├── data/                             # Real-world data files (git-ignored)
│   ├── README.md                     # Data strategy documentation
│   ├── landsat/README.md             # Landsat 8/9 download instructions
│   ├── viirs/README.md               # VIIRS VNP09GA download instructions
│   ├── sentinel2/README.md           # Sentinel-2 download instructions
│   └── umbra/README.md               # Umbra SICD download instructions
├── workflows/
│   └── comprehensive_benchmark_workflow.yaml  # Example SAR processing pipeline
├── .benchmarks/                      # Benchmark result storage (git-ignored)
│   ├── index.json                    # Lightweight index for fast filtering
│   └── records/                      # One JSON file per benchmark run
├── pyproject.toml                    # Package configuration + pytest markers
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

| Reader | Directory | File Pattern | Format | Size | Source |
|--------|-----------|-------------|--------|------|--------|
| **GeoTIFFReader** | `data/landsat/` | `LC0[89]*_SR_B*.TIF` | Cloud-Optimized GeoTIFF (uint16) | 50-150 MB | [USGS EarthExplorer](https://earthexplorer.usgs.gov) |
| **HDF5Reader** | `data/viirs/` | `V?P09GA*.h5` | HDF5/HDF-EOS5 (int16) | 100-300 MB | [LAADS DAAC](https://ladsweb.modaps.eosdis.nasa.gov) |
| **JP2Reader** | `data/sentinel2/` | `S2*.SAFE/.../R10m/*_B04_10m.jp2` | JPEG2000 15-bit (uint16) | 100-200 MB | [Copernicus Data Space](https://dataspace.copernicus.eu) |
| **NITFReader** | `data/umbra/` | `*.nitf` | NITF + SICD XML (complex64) | 50-500 MB | [Umbra Open Data (AWS S3)](https://umbra-open-data-catalog.s3.amazonaws.com/index.html) |

### Download Quick Start

**Landsat** — any single Landsat 8 or 9 Collection 2 Surface Reflectance band file:
```
data/landsat/LC09_L2SP_001028_20260210_20260211_02_T2_SR_B4.TIF
```
Source: [USGS EarthExplorer](https://earthexplorer.usgs.gov) (free account required)

**VIIRS** — any VNP09GA daily surface reflectance granule:
```
data/viirs/VNP09GA.A2026004.h17v14.002.2026005131408.h5
```
Source: [LAADS DAAC](https://ladsweb.modaps.eosdis.nasa.gov) (NASA Earthdata login required)

**Sentinel-2** — a complete SAFE archive (or standalone B04 JP2):
```
data/sentinel2/S2B_MSIL2A_20260204T170409_N0512_R069_T15RTP_20260204T223135.SAFE/
```
Source: [Copernicus Data Space](https://dataspace.copernicus.eu) (free account required)

**Umbra** — any Umbra SICD spotlight NITF:
```
data/umbra/2025-10-25-20-00-44_UMBRA-10_SICD.nitf
```
Source: [Umbra Open Data (S3)](https://umbra-open-data-catalog.s3.amazonaws.com/index.html) (no login, CC BY 4.0)

Each `data/<dataset>/README.md` contains detailed download instructions, expected file properties, and format specifications.

## Test Architecture

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

### Geolocation Tests

Geolocation tests validate GRDL's coordinate transformation and elevation systems:

| Test File | Coverage |
|-----------|----------|
| `test_geolocation_base.py` | Geolocation ABC contract, `NoGeolocation` fallback, scalar/array dispatch |
| `test_geolocation_affine_real.py` | Affine transforms with real CRS data, UTM/WGS84 forward-inverse round-trips |
| `test_geolocation_utils.py` | Coordinate transformation utilities, geodetic calculations |
| `test_geolocation_elevation.py` | DEM-based elevation models, 3D geolocation |

### Benchmarking Tests

Benchmarking tests validate the profiling infrastructure itself (no real data required):

| Test File | Coverage |
|-----------|----------|
| `test_benchmark_models.py` | `HardwareSnapshot`, `AggregatedMetrics`, `StepBenchmarkResult`, `BenchmarkRecord` serialization |
| `test_benchmark_store.py` | `JSONBenchmarkStore` save/load/query and index consistency |
| `test_active_runner.py` | `ActiveBenchmarkRunner` iteration counting, warmup exclusion, per-step aggregation |
| `test_component_benchmark.py` | `ComponentBenchmark` timing, memory measurement, pytest integration |

### Graceful Skip Behavior

Tests skip cleanly when data is absent — they never fail due to missing files:

**With all data present:**
```
tests/test_io_geotiff.py    15 passed
tests/test_io_hdf5.py       15 passed
tests/test_io_jpeg2000.py   16 passed
tests/test_io_nitf.py       15 passed
```

**With no data present:**
```
tests/test_io_geotiff.py    15 skipped
tests/test_io_hdf5.py       15 skipped
tests/test_io_jpeg2000.py   16 skipped
tests/test_io_nitf.py       15 skipped
```

Each skip message includes the expected file pattern and a path to the README with download instructions.

## Running Tests

```bash
conda activate grdl

# Full suite (missing data files skip cleanly)
pytest tests/ -v

# Specific reader
pytest tests/test_io_geotiff.py -v        # Landsat
pytest tests/test_io_hdf5.py -v           # VIIRS
pytest tests/test_io_jpeg2000.py -v       # Sentinel-2
pytest tests/test_io_nitf.py -v           # Umbra

# Geolocation tests
pytest tests/test_geolocation_base.py tests/test_geolocation_utils.py -v
pytest tests/test_geolocation_affine_real.py -v
pytest tests/test_geolocation_elevation.py -v

# Benchmarking infrastructure tests
pytest tests/test_benchmark_models.py tests/test_benchmark_store.py -v
pytest tests/test_active_runner.py tests/test_component_benchmark.py -v

# By marker
pytest tests/ -m landsat                  # All Landsat tests
pytest tests/ -m viirs                    # All VIIRS tests
pytest tests/ -m geolocation              # All geolocation tests
pytest tests/ -m integration              # Only Level 3 integration tests
pytest tests/ -m "nitf and not slow"      # NITF tests, skip slow ones
pytest tests/ -m benchmark                # Benchmarking infrastructure tests

# Skip all data-dependent tests
pytest tests/ -m "not requires_data"
```

### Test Markers

| Marker | Purpose |
|--------|---------|
| `landsat` | Landsat 8/9 tests (GeoTIFFReader) |
| `viirs` | VIIRS VNP09GA tests (HDF5Reader) |
| `sentinel2` | Sentinel-2 tests (JP2Reader) |
| `nitf` | Umbra SICD tests (NITFReader) |
| `geolocation` | Geolocation utility and coordinate transform tests |
| `requires_data` | Test requires real data files in `data/` |
| `slow` | Long-running test (large file reads, full pipelines) |
| `integration` | Level 3 tests (ChipExtractor, Normalizer, Tiler workflows) |
| `benchmark` | Performance benchmark tests |

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
```

**Array size presets:**

| Preset | Dimensions |
|--------|-----------|
| `small` | 512 x 512 |
| `medium` | 2048 x 2048 |
| `large` | 4096 x 4096 |

### Active Workflow Benchmarking

Run a grdl-runtime `Workflow` N times, aggregate per-step metrics, and persist results:

```python
from grdl_rt import Workflow
from grdl_te.benchmarking import ActiveBenchmarkRunner, JSONBenchmarkStore

wf = (
    Workflow("SAR Pipeline", modalities=["SAR"])
    .reader(SICDReader)
    .step(SublookDecomposition, num_looks=3)
    .step(ToDecibels)
)

store = JSONBenchmarkStore()
runner = ActiveBenchmarkRunner(wf, iterations=10, warmup=2, store=store)
record = runner.run(source="image.nitf", prefer_gpu=True)

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
| `ActiveBenchmarkRunner` | Runs a Workflow N times with warmup and aggregates metrics |
| `ComponentBenchmark` | Profiles a single callable with timing and tracemalloc |
| `JSONBenchmarkStore` | File-per-record JSON persistence with index |
| `as_pytest_benchmark` | Helper to integrate ComponentBenchmark with pytest-benchmark |
| `run_suite` | Orchestrates multiple benchmark groups from config or CLI |

### Example Workflow

The `workflows/` directory contains example grdl-runtime workflow definitions for benchmarking. `comprehensive_benchmark_workflow.yaml` defines a multi-stage SAR processing pipeline (complex speckle filtering, phase gradient analysis, amplitude conversion, rank/linear/statistical filters, and contrast stretching).

### Future Phases

- **Passive Monitoring** — `ExecutionHook` that captures metrics from production workflows
- **Forensic Analysis** — load a completed workflow result, substitute step metrics to simulate "what if" scenarios (e.g., GPU vs CPU)
- **Comparison & Regression Detection** — compare benchmark records across runs with configurable thresholds
- **Cross-Hardware Prediction** — collect results from different machines, predict performance on new hardware

## Adding a New Reader Test

1. **Select ONE representative dataset** — prioritize open data, production quality
2. **Create** `data/<dataset>/README.md` with download instructions and file format specifications
3. **Add fixtures** to `tests/conftest.py` using `require_data_file()`
4. **Create test file** `tests/test_io_<reader>.py` with 3-level structure (format, quality, integration)
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
