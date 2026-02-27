# GRDL-TE: Testing, Evaluation & Benchmarking

GRDL-TE is the validation and benchmarking suite for the [GRDL](../grdl/) (GEOINT Rapid Development Library). It serves two purposes:

1. **Validation** — tests GRDL's public API against real-world satellite data with 3-level validation (format, quality, integration).
2. **Benchmarking** — profiles GRDL workflows and individual components, aggregates metrics across runs, and persists results for regression detection and cross-hardware comparison.

GRDL-TE is a *consumer* of GRDL — it only imports the public API. It never modifies GRDL internals.

## Architecture

```
grdx/
├── grdl/             # Core library — readers, filters, transforms, geolocation
├── grdl-runtime/     # Workflow execution engine (DAG orchestration, YAML pipelines)
├── grdk/             # GUI toolkit (Orange3 widgets, napari viewers)
└── grdl-te/          # This package — validation tests + benchmark profiling
```

| Layer | Package | Role |
|-------|---------|------|
| **Library** | `grdl` | Modular building blocks for GEOINT image processing |
| **Runtime** | `grdl-runtime` | Headless workflow executor, YAML pipeline loader |
| **T&E** | `grdl-te` | Correctness validation and performance profiling against `grdl` |

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

## Validation Suite

Three-level validation against real-world satellite data (552 tests, 38 test files):

| Level | Scope | Examples |
|-------|-------|---------|
| **L1 — Format** | Reader instantiation, metadata, shape/dtype, chip reads, resource cleanup | SICD complex64 dtype, GeoTIFF COG tiling |
| **L2 — Quality** | CRS projection, value ranges, NoData masking, format-specific features | UTM zone validation, 15-bit reflectance ceilings, SAR speckle statistics |
| **L3 — Integration** | Multi-component pipelines (chip, normalize, tile, detect) | ChipExtractor → Normalizer → batch validation |

Tests skip gracefully when data is absent (`pytest.skip` with download instructions). Present data produces pass/fail — never a false pass.

Each `data/<dataset>/README.md` contains download instructions, expected file properties, and format specifications.

### Running Tests

```bash
conda activate grdl

# Full suite (missing data files skip cleanly)
pytest

# Specific reader
pytest tests/validation/test_io_geotiff.py -v        # Landsat
pytest tests/validation/test_io_nitf.py -v            # Umbra SICD
pytest tests/validation/test_io_sentinel1.py -v       # Sentinel-1

# Geolocation tests
pytest tests/validation/test_geolocation_base.py tests/validation/test_geolocation_utils.py -v
pytest tests/validation/test_geolocation_affine_real.py -v

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
| `filters` | Mean, Gaussian, Median, Min, Max, StdDev, Lee, ComplexLee, PhaseGradient |
| `intensity` | ToDecibels, PercentileStretch |
| `decomposition` | Pauli, DualPolHAlpha, SublookDecomposition |
| `detection` | CA-CFAR, GO-CFAR, SO-CFAR, OS-CFAR, DetectionSet, Fields |
| `sar` | MultilookDecomposition, CSIProcessor |
| `image_formation` | CollectionGeometry, PolarGrid, PFA, RDA, StripmapPFA, FFBP, SubaperturePartitioner |
| `ortho` | Orthorectifier, OutputGrid, OrthoPipeline, compute_output_resolution |
| `coregistration` | Affine, FeatureMatch, Projective |
| `io` | 22 readers/writers (GeoTIFF, HDF5, NITF, JP2, SICD, CPHD, CRSD, SIDD, Sentinel-1/2, ASTER, BIOMASS, TerraSAR-X) |
| `geolocation` | Affine, GCP, SICD, Sentinel-1 SLC, NoGeolocation |
| `interpolation` | Lanczos, KaiserSinc, Lagrange, Farrow, Polyphase, ThiranDelay |
| `data_prep` | ChipExtractor, Tiler, Normalizer |
| `pipeline` | Sequential Pipeline composition |

### Active Workflow Benchmarking

Run a `grdl-runtime` Workflow N times, aggregate per-step metrics, and persist results:

```python
from grdl_rt import Workflow
from grdl_rt.api import load_workflow
from grdl_te.benchmarking import ActiveBenchmarkRunner, BenchmarkSource, JSONBenchmarkStore

store = JSONBenchmarkStore()

# ==== Pass a declared workflow ====
wf = (
    Workflow("SAR Pipeline", modalities=["SAR"])
    .reader(SICDReader)
    .step(SublookDecomposition, num_looks=3)
    .step(ToDecibels)
)
runner = ActiveBenchmarkRunner(wf, iterations=10, warmup=2, store=store)
record = runner.run(source="image.nitf", prefer_gpu=True)

# ==== Load a YAML workflow ====
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

Each `BenchmarkRecord` captures the `HardwareSnapshot` (CPU, RAM, GPU, platform), per-step `AggregatedMetrics` (min, max, mean, median, stddev, p95), and raw per-iteration measurements for lossless post-hoc analysis.

### Example Workflow

The `workflows/` directory contains example `grdl-runtime` workflow definitions. `comprehensive_benchmark_workflow.yaml` defines a 28-step SAR processing pipeline covering complex speckle filtering, phase gradient analysis, amplitude conversion, CFAR detection, and conditional orthorectification.

`benchmark_examples.py` demonstrates active workflow benchmarking with `ActiveBenchmarkRunner` at multiple array scales.

## Project Status

**Component coverage: 78/78 (100%)**

All public GRDL components have both a dedicated benchmark in `suite.py` and a correctness validation test in `tests/validation/`. See [BENCHMARK_COVERAGE_GAPS.md](BENCHMARK_COVERAGE_GAPS.md) for the full inventory.

| Metric | Value |
|--------|-------|
| Benchmarked components | 78/78 |
| Benchmark groups | 13 |
| Validation test files | 32 |
| Benchmark infrastructure tests | 6 |
| YAML workflow steps | 28 |
| Array size presets | small (512), medium (2048), large (4096) |

### Active Development

- **Passive Monitoring** — `ExecutionHook` for capturing metrics from production workflows
- **Regression Detection** — cross-run comparison with configurable thresholds
- **Cross-Hardware Prediction** — collect results from different machines, predict performance on new hardware

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
