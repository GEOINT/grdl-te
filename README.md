# GRDL-TE: Testing, Evaluation & Benchmarking

GRDL-TE serves two purposes:

1. **Validation** — tests GRDL's public API against real-world satellite data with 3-level validation (format, quality, integration).
2. **Benchmarking** — profiles GRDL workflows and individual components, aggregates metrics across runs, and persists results for regression detection and cross-hardware comparison.

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
├── grdl_te/                     # Benchmarking package (pip install -e .)
│   └── benchmarking/
│       ├── models.py            # HardwareSnapshot, AggregatedMetrics, BenchmarkRecord
│       ├── base.py              # BenchmarkRunner, BenchmarkStore ABCs
│       ├── store.py             # JSONBenchmarkStore (file-per-record persistence)
│       ├── active.py            # ActiveBenchmarkRunner (workflow N-run aggregation)
│       └── component.py         # ComponentBenchmark (single-function profiling)
├── data/                        # Real-world data files (not committed to git)
│   ├── README.md                # Data strategy documentation
│   ├── landsat/                 # Landsat 8/9 COG (GeoTIFFReader)
│   │   └── README.md            # Download instructions
│   ├── viirs/                   # VIIRS VNP09GA HDF5 (HDF5Reader)
│   │   └── README.md
│   ├── sentinel2/               # Sentinel-2 JP2 (JP2Reader)
│   │   └── README.md
│   └── umbra/                   # Umbra SICD NITF (NITFReader)
│       └── README.md
├── tests/
│   ├── conftest.py              # Shared fixtures + graceful skip logic
│   ├── test_io_geotiff.py       # GeoTIFFReader (Landsat)
│   ├── test_io_hdf5.py          # HDF5Reader (VIIRS)
│   ├── test_io_jpeg2000.py      # JP2Reader (Sentinel-2)
│   ├── test_io_nitf.py          # NITFReader (Umbra SICD)
│   ├── test_benchmark_models.py # Benchmark dataclass tests
│   ├── test_benchmark_store.py  # JSON store tests
│   ├── test_active_runner.py    # Active benchmark runner tests
│   └── test_component_benchmark.py # Component benchmark tests
├── pyproject.toml               # Package configuration + markers
├── LICENSE                      # MIT License
└── README.md                    # This file
```

## Test Architecture

All test files follow a **3-level validation structure**:

### Level 1: Format Validation
- Metadata extraction (format, rows, cols, dtype, CRS, bands)
- Shape and dtype consistency with metadata
- Chip reads from image center (validates data exists, not just fill)
- Full image reads (with size guards)
- Context manager resource cleanup

### Level 2: Data Quality
- **CRS/projection validation** (UTM zone ranges for Landsat/Sentinel-2)
- **NoData handling** (masking, valid pixel statistics, variance)
- **Value range bounds** (reflectance scales, 15-bit encoding ceilings)
- **Format-specific features** (COG tiling/overviews, HDF-EOS structure, complex magnitude/phase, SAR speckle statistics)

### Level 3: Integration
- **ChipExtractor**: Uniform chip partitioning with bounds validation
- **Normalizer**: MinMax, Z-score, and percentile normalization
- **Tiler**: Overlapping tile grids with stride control
- **Pipelines**: End-to-end workflows (chip -> normalize -> validate batch)

## Data Manifesto

Each reader requires **exactly one file**. Tests skip gracefully if the file is missing.

| Reader | Directory | File Pattern | Format | Size | Source |
|--------|-----------|-------------|--------|------|--------|
| **GeoTIFFReader** | `data/landsat/` | `LC0[89]*_SR_B*.TIF` | Cloud-Optimized GeoTIFF (uint16) | 50-150 MB | [USGS EarthExplorer](https://earthexplorer.usgs.gov) |
| **HDF5Reader** | `data/viirs/` | `V?P09GA*.h5` | HDF5/HDF-EOS5 (int16) | 100-300 MB | [LAADS DAAC](https://ladsweb.modaps.eosdis.nasa.gov) |
| **JP2Reader** | `data/sentinel2/` | `S2*.SAFE/.../R10m/*_B04_10m.jp2` | JPEG2000 15-bit (uint16) | 100-200 MB | [Copernicus Data Space](https://dataspace.copernicus.eu) |
| **NITFReader** | `data/umbra/` | `*.nitf` | NITF + SICD XML (complex64) | 50-500 MB | [Umbra Open Data (AWS S3)](https://umbra-open-data-catalog.s3.amazonaws.com/index.html) |

### Download Quick Start

**Landsat** -- Any single Landsat 8 or 9 Collection 2 Surface Reflectance band file:
```
data/landsat/LC09_L2SP_001028_20260210_20260211_02_T2_SR_B4.TIF
```
Source: [USGS EarthExplorer](https://earthexplorer.usgs.gov) (free account required)

**VIIRS** -- Any VNP09GA daily surface reflectance granule:
```
data/viirs/VNP09GA.A2026004.h17v14.002.2026005131408.h5
```
Source: [LAADS DAAC](https://ladsweb.modaps.eosdis.nasa.gov) (NASA Earthdata login required)

**Sentinel-2** -- A complete SAFE archive (or standalone B04 JP2):
```
data/sentinel2/S2B_MSIL2A_20260204T170409_N0512_R069_T15RTP_20260204T223135.SAFE/
```
Source: [Copernicus Data Space](https://dataspace.copernicus.eu) (free account required)

**Umbra** -- Any Umbra SICD spotlight NITF:
```
data/umbra/2025-10-25-20-00-44_UMBRA-10_SICD.nitf
```
Source: [Umbra Open Data (S3)](https://umbra-open-data-catalog.s3.amazonaws.com/index.html) (no login, CC BY 4.0)

Each `data/<dataset>/README.md` contains detailed download instructions and file format specifications.

## Benchmarking

The `grdl_te` package provides infrastructure for profiling GRDL workflows and individual components. It consumes grdl-runtime types (`Workflow`, `WorkflowResult`, `StepMetrics`) directly.

### Installation

```bash
pip install -e .                          # core (models, store, component benchmarks)
pip install -e ".[benchmarking]"          # + grdl-runtime (active workflow benchmarks)
```

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

### Future Phases

- **Passive Monitoring** — `ExecutionHook` that captures metrics from production workflows
- **Forensic Analysis** — load a completed workflow result, substitute step metrics to simulate "what if" scenarios (e.g., GPU vs CPU)
- **Comparison & Regression Detection** — compare benchmark records across runs with configurable thresholds
- **Cross-Hardware Prediction** — collect results from different machines, predict performance on new hardware

## Running Tests

```bash
# Activate environment
conda activate grdl

# Run all tests (missing data files skip cleanly)
pytest tests/ -v

# Run tests for a specific reader
pytest tests/test_io_geotiff.py -v      # Landsat
pytest tests/test_io_hdf5.py -v         # VIIRS
pytest tests/test_io_jpeg2000.py -v     # Sentinel-2
pytest tests/test_io_nitf.py -v         # Umbra

# Run by marker
pytest tests/ -m landsat                # All Landsat tests
pytest tests/ -m viirs                  # All VIIRS tests
pytest tests/ -m integration            # Only Level 3 integration tests
pytest tests/ -m "nitf and not slow"    # NITF tests, skip slow ones

# Skip all data-dependent tests
pytest tests/ -m "not requires_data"

# Run benchmarking tests
pytest tests/test_benchmark_models.py tests/test_benchmark_store.py -v
pytest tests/test_active_runner.py tests/test_component_benchmark.py -v
```

### Expected Output

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

Each skip message points to the exact file pattern and README with download instructions.

## Test Markers

| Marker | Purpose |
|--------|---------|
| `landsat` | Landsat 8/9 tests (GeoTIFFReader) |
| `viirs` | VIIRS VNP09GA tests (HDF5Reader) |
| `sentinel2` | Sentinel-2 tests (JP2Reader) |
| `nitf` | Umbra SICD tests (NITFReader) |
| `requires_data` | Test requires real data files in `data/` |
| `slow` | Long-running test (large file reads, full pipelines) |
| `integration` | Level 3 tests (ChipExtractor, Normalizer, Tiler workflows) |
| `benchmark` | Performance benchmark tests |

## Adding a New Reader Test

1. **Select ONE representative dataset** -- prioritize open data, production quality
2. **Create** `data/<dataset>/README.md` with download instructions
3. **Add fixture** to `tests/conftest.py` using `require_data_file()`
4. **Create test file** `tests/test_io_<reader>.py` with 3-level structure
5. **Register markers** in `pyproject.toml`

## License

MIT License -- see [LICENSE](LICENSE) for details.
