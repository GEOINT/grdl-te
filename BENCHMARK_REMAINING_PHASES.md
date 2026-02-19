# GRDL-TE Benchmark Coverage — Phases 2–4

> **Context**: Phase 1 is complete. It delivered 15 synthetic-data components:
> - 5 validation test files (81 tests, 80 pass, 1 skip)
> - `run_detection_benchmarks()` new group in suite.py
> - DualPolHAlpha added to `run_decomposition_benchmarks()`
> - OrthoPipeline + `compute_output_resolution` added to `run_ortho_benchmarks()`
> - ProjectiveCoRegistration added to `run_coregistration_benchmarks()`
> - NoGeolocation added to `run_geolocation_benchmarks()`
> - BENCHMARK_GROUPS now has 11 entries (was 10)
> - 7 new pytest markers in pyproject.toml
>
> **What remains**: 21 components across IO readers/writers, SAR image formation,
> elevation models, SAR sub-aperture processing, and YAML workflow integration.

---

## Phase 2: Real-Data IO Readers/Writers + conftest Fixtures

### 2.1 Update `tests/validation/conftest.py`

Add session-scoped data directory fixtures and `require_*` file fixtures so all
new IO test files can reference them consistently.

```python
# After existing sentinel2_data_dir fixture:

@pytest.fixture(scope="session")
def cphd_data_dir(data_dir):
    return data_dir / "cphd"

@pytest.fixture(scope="session")
def crsd_data_dir(data_dir):
    return data_dir / "crsd"

@pytest.fixture(scope="session")
def sidd_data_dir(data_dir):
    return data_dir / "sidd"

@pytest.fixture(scope="session")
def sentinel1_data_dir(data_dir):
    return data_dir / "sentinel1"

@pytest.fixture(scope="session")
def aster_data_dir(data_dir):
    return data_dir / "aster"

@pytest.fixture(scope="session")
def biomass_data_dir(data_dir):
    return data_dir / "biomass"

@pytest.fixture(scope="session")
def dted_data_dir(data_dir):
    return data_dir / "dted"

@pytest.fixture(scope="session")
def dem_data_dir(data_dir):
    return data_dir / "dem"

@pytest.fixture(scope="session")
def geoid_data_dir(data_dir):
    return data_dir / "geoid"

# --- Data file fixtures ---
@pytest.fixture
def require_cphd_file(cphd_data_dir):
    return require_data_file(cphd_data_dir, "*.cphd")

@pytest.fixture
def require_crsd_file(crsd_data_dir):
    return require_data_file(crsd_data_dir, "*.crsd")

@pytest.fixture
def require_sidd_file(sidd_data_dir):
    return require_data_file(sidd_data_dir, "*.nitf")

@pytest.fixture
def require_sentinel1_file(sentinel1_data_dir):
    return require_data_file(sentinel1_data_dir, "*.SAFE")

@pytest.fixture
def require_aster_file(aster_data_dir):
    return require_data_file(aster_data_dir, "AST_L1T*.hdf")

@pytest.fixture
def require_biomass_file(biomass_data_dir):
    return require_data_file(biomass_data_dir, "BIO_S2_*.tif")

@pytest.fixture
def require_dted_dir(dted_data_dir):
    if not dted_data_dir.exists() or not list(dted_data_dir.glob("**/*.dt?")):
        pytest.skip(f"DTED data not found in {dted_data_dir}")
    return dted_data_dir

@pytest.fixture
def require_dem_file(dem_data_dir):
    return require_data_file(dem_data_dir, "*.tif")

@pytest.fixture
def require_geoid_file(geoid_data_dir):
    return require_data_file(geoid_data_dir, "*.pgm")
```

### 2.2 New Test Files — IO Readers/Writers

Each file follows the existing 3-level pattern (see `test_io_geotiff.py`):
- Level 1: metadata, shape, dtype, context manager protocol
- Level 2: Format-specific properties (CRS, NoData, value ranges, band counts)
- Level 3: Integration with ChipExtractor/Normalizer

#### `tests/validation/test_io_sar_writers.py` — SICDWriter + NITFWriter

```python
"""SICDWriter and NITFWriter roundtrip validation."""

# Imports:
from grdl.IO.sar import SICDReader, SICDWriter
from grdl.IO.nitf import NITFWriter

# Markers: pytest.mark.io, requires_data (umbra SICD NITF)
# Fixture: require_umbra_file (existing)

# Tests:
# Level 1:
#   test_sicd_writer_creates_file — Write SICD, verify file exists and size > 0
#   test_nitf_writer_creates_file — Write array to NITF format
#   test_sicd_writer_context_manager — with SICDWriter(...) as w: w.write(...)

# Level 2:
#   test_sicd_roundtrip_metadata — Read → Write → Read: metadata preserved
#   test_sicd_roundtrip_data_fidelity — Read → Write → Read: RMSE < 1e-4
#   test_nitf_writer_dtype_support — Write float32 and complex64

# Level 3:
#   test_sicd_writer_with_chip_extractor — ChipExtractor → read_chip → write → read back
```

#### `tests/validation/test_io_cphd.py` — CPHDReader

```python
"""CPHDReader validation using real CPHD phase history data."""

# Import: from grdl.IO.sar import CPHDReader
# Markers: pytest.mark.cphd, requires_data
# Fixture: require_cphd_file
# Data pattern: data/cphd/*.cphd

# Tests:
# Level 1:
#   test_cphd_reader_opens — Context manager, no exception
#   test_cphd_metadata_populated — typed_metadata is not None
#   test_cphd_shape — get_shape() returns (n_vectors, n_samples) tuple

# Level 2:
#   test_cphd_read_full_complex — read_full() returns complex array
#   test_cphd_pvp_populated — read_pvp() returns structured array with fields
#   test_cphd_values_finite — No NaN/Inf in phase data

# Level 3:
#   test_cphd_metadata_to_collection_geometry — typed_metadata → CollectionGeometry
```

#### `tests/validation/test_io_crsd.py` — CRSDReader

```python
"""CRSDReader validation using real CRSD data."""

# Import: from grdl.IO.sar import CRSDReader
# Markers: pytest.mark.crsd, requires_data
# Fixture: require_crsd_file
# Data pattern: data/crsd/*.crsd

# Tests:
# Level 1:
#   test_crsd_reader_opens
#   test_crsd_metadata_populated
#   test_crsd_shape

# Level 2:
#   test_crsd_read_full_complex
#   test_crsd_values_finite

# Level 3:
#   test_crsd_integration_with_normalizer
```

#### `tests/validation/test_io_sidd.py` — SIDDReader

```python
"""SIDDReader validation using real SIDD data."""

# Import: from grdl.IO.sar import SIDDReader
# Markers: pytest.mark.sidd, requires_data
# Fixture: require_sidd_file
# Data pattern: data/sidd/*.nitf

# Tests:
# Level 1:
#   test_sidd_reader_opens
#   test_sidd_metadata_populated — typed_metadata with product_type
#   test_sidd_shape_2d

# Level 2:
#   test_sidd_read_full — Returns real-valued (detected) array
#   test_sidd_pixel_type — float32 or uint8 depending on product
#   test_sidd_geolocation_metadata — Geographic metadata present

# Level 3:
#   test_sidd_chip_and_normalize
```

#### `tests/validation/test_io_sentinel1.py` — Sentinel1SLCReader

```python
"""Sentinel1SLCReader validation using real Sentinel-1 SLC data."""

# Import: from grdl.IO.sar import Sentinel1SLCReader
# Markers: pytest.mark.sentinel1, requires_data
# Fixture: require_sentinel1_file
# Data pattern: data/sentinel1/*.SAFE

# Tests:
# Level 1:
#   test_s1_reader_opens — Context manager
#   test_s1_metadata_populated — polarization, orbit, swath info
#   test_s1_shape — (rows, cols) tuple

# Level 2:
#   test_s1_read_full_complex — Complex64 array
#   test_s1_read_chip — Subregion extraction
#   test_s1_polarization_channels — VV/VH or HH/HV accessible

# Level 3:
#   test_s1_geolocation_construction — Sentinel1SLCGeolocation.from_reader()
#   test_s1_chip_extractor_integration
```

#### `tests/validation/test_io_sentinel2.py` — Sentinel2Reader (if not already covered)

```python
"""Sentinel2Reader validation using real Sentinel-2 L2A data."""

# Import: from grdl.IO.multispectral import Sentinel2Reader
# Markers: pytest.mark.sentinel2, requires_data
# Fixture: require_sentinel2_file (existing or extend)
# Data pattern: data/sentinel2/S2*.SAFE

# Tests:
# Level 1:
#   test_s2_reader_opens
#   test_s2_band_names — List of available bands (B01-B12, B8A)
#   test_s2_metadata — CRS, bounds, resolution per band

# Level 2:
#   test_s2_read_band — Single band as ndarray with expected shape
#   test_s2_read_full — All bands stacked
#   test_s2_value_range — Reflectance values in reasonable range (0-10000)

# Level 3:
#   test_s2_jp2_reader_fallback — Individual JP2 files via JP2Reader
```

#### `tests/validation/test_io_aster.py` — ASTERReader

```python
"""ASTERReader validation using real ASTER L1T data."""

# Import: from grdl.IO.multispectral import ASTERReader
# Markers: pytest.mark.aster, requires_data
# Fixture: require_aster_file
# Data pattern: data/aster/AST_L1T*.hdf

# Tests:
# Level 1:
#   test_aster_reader_opens
#   test_aster_metadata — sensor info, bands, acquisition date
#   test_aster_shape

# Level 2:
#   test_aster_read_full — Multi-band array
#   test_aster_vnir_bands — 3 VNIR bands at 15m resolution
#   test_aster_values_finite

# Level 3:
#   test_aster_normalizer_integration
```

#### `tests/validation/test_io_biomass.py` — BIOMASSL1Reader + BIOMASSCatalog

```python
"""BIOMASSL1Reader and BIOMASSCatalog validation."""

# Imports:
# from grdl.IO.sar import BIOMASSL1Reader, BIOMASSCatalog
# Markers: pytest.mark.biomass, requires_data
# Fixture: require_biomass_file
# Data pattern: data/biomass/BIO_S2_*.tif

# Tests:
# Level 1:
#   test_biomass_reader_opens
#   test_biomass_metadata — Mission-specific metadata
#   test_biomass_shape

# Level 2:
#   test_biomass_read_full — Complex or real array
#   test_biomass_values_finite

# Level 3:
#   test_biomass_catalog_search — BIOMASSCatalog.search() returns results
#   test_biomass_catalog_empty_query — Returns empty list, no crash
```

### 2.3 Extend `_run_real_data_io()` in suite.py

After the existing NITF block, add benchmark entries for each new reader:

```python
    # ------------------------------------------------------------------
    # CPHD  (cphd/ — pattern: *.cphd)
    # ------------------------------------------------------------------
    cphd_dir = _data_dir / "cphd"
    cphd_path = _find_data_file(cphd_dir, "*.cphd")
    if cphd_path:
        try:
            from grdl.IO.sar import CPHDReader

            def _cphd_read_full():
                with CPHDReader(cphd_path) as reader:
                    return reader.read_full()

            r = _bench("CPHDReader.read_full.real_data",
                        _cphd_read_full, **kw)
            if r:
                results.append(r)

            def _cphd_read_pvp():
                with CPHDReader(cphd_path) as reader:
                    return reader.read_pvp()

            r = _bench("CPHDReader.read_pvp.real_data",
                        _cphd_read_pvp, **kw)
            if r:
                results.append(r)

        except (ImportError, Exception) as exc:
            print(f"  SKIP  CPHDReader benchmarks ({exc})")
    else:
        print("  SKIP  CPHDReader benchmarks (data files not found)")

    # ------------------------------------------------------------------
    # CRSD  (crsd/ — pattern: *.crsd)
    # ------------------------------------------------------------------
    crsd_dir = _data_dir / "crsd"
    crsd_path = _find_data_file(crsd_dir, "*.crsd")
    if crsd_path:
        try:
            from grdl.IO.sar import CRSDReader

            def _crsd_read_full():
                with CRSDReader(crsd_path) as reader:
                    return reader.read_full()

            r = _bench("CRSDReader.read_full.real_data",
                        _crsd_read_full, **kw)
            if r:
                results.append(r)

        except (ImportError, Exception) as exc:
            print(f"  SKIP  CRSDReader benchmarks ({exc})")
    else:
        print("  SKIP  CRSDReader benchmarks (data files not found)")

    # ------------------------------------------------------------------
    # SIDD  (sidd/ — pattern: *.nitf)
    # ------------------------------------------------------------------
    sidd_dir = _data_dir / "sidd"
    sidd_path = _find_data_file(sidd_dir, "*.nitf")
    if sidd_path:
        try:
            from grdl.IO.sar import SIDDReader

            def _sidd_read_full():
                with _suppress_stderr(), SIDDReader(sidd_path) as reader:
                    return reader.read_full()

            r = _bench("SIDDReader.read_full.real_data",
                        _sidd_read_full, **kw)
            if r:
                results.append(r)

        except (ImportError, Exception) as exc:
            print(f"  SKIP  SIDDReader benchmarks ({exc})")
    else:
        print("  SKIP  SIDDReader benchmarks (data files not found)")

    # ------------------------------------------------------------------
    # Sentinel-1 SLC  (sentinel1/ — pattern: *.SAFE)
    # ------------------------------------------------------------------
    sentinel1_dir = _data_dir / "sentinel1"
    s1_path = _find_data_file(sentinel1_dir, "*.SAFE")
    if s1_path:
        try:
            from grdl.IO.sar import Sentinel1SLCReader

            def _s1_read_full():
                with Sentinel1SLCReader(s1_path) as reader:
                    return reader.read_full()

            r = _bench("Sentinel1SLCReader.read_full.real_data",
                        _s1_read_full, **kw)
            if r:
                results.append(r)

            def _s1_read_chip():
                with Sentinel1SLCReader(s1_path) as reader:
                    s = reader.get_shape()
                    cx, cy = s[0] // 2, s[1] // 2
                    return reader.read_chip(cx - 512, cx + 512,
                                            cy - 512, cy + 512)

            r = _bench("Sentinel1SLCReader.read_chip.1024x1024.real_data",
                        _s1_read_chip, **kw)
            if r:
                results.append(r)

        except (ImportError, Exception) as exc:
            print(f"  SKIP  Sentinel1SLCReader benchmarks ({exc})")
    else:
        print("  SKIP  Sentinel1SLCReader benchmarks (data files not found)")

    # ------------------------------------------------------------------
    # ASTER  (aster/ — pattern: AST_L1T*.hdf)
    # ------------------------------------------------------------------
    aster_dir = _data_dir / "aster"
    aster_path = _find_data_file(aster_dir, "AST_L1T*.hdf")
    if aster_path:
        try:
            from grdl.IO.multispectral import ASTERReader

            def _aster_read_full():
                with ASTERReader(aster_path) as reader:
                    return reader.read_full()

            r = _bench("ASTERReader.read_full.real_data",
                        _aster_read_full, **kw)
            if r:
                results.append(r)

        except (ImportError, Exception) as exc:
            print(f"  SKIP  ASTERReader benchmarks ({exc})")
    else:
        print("  SKIP  ASTERReader benchmarks (data files not found)")

    # ------------------------------------------------------------------
    # BIOMASS  (biomass/ — pattern: BIO_S2_*.tif)
    # ------------------------------------------------------------------
    biomass_dir = _data_dir / "biomass"
    biomass_path = _find_data_file(biomass_dir, "BIO_S2_*.tif")
    if biomass_path:
        try:
            from grdl.IO.sar import BIOMASSL1Reader

            def _biomass_read_full():
                with BIOMASSL1Reader(biomass_path) as reader:
                    return reader.read_full()

            r = _bench("BIOMASSL1Reader.read_full.real_data",
                        _biomass_read_full, **kw)
            if r:
                results.append(r)

        except (ImportError, Exception) as exc:
            print(f"  SKIP  BIOMASSL1Reader benchmarks ({exc})")
    else:
        print("  SKIP  BIOMASSL1Reader benchmarks (data files not found)")

    # ------------------------------------------------------------------
    # SICDWriter roundtrip  (write to tmpdir, read back)
    # ------------------------------------------------------------------
    if sar_path:
        try:
            from grdl.IO.sar import SICDReader, SICDWriter

            with _suppress_stderr(), SICDReader(sar_path) as reader:
                sicd_meta = reader.metadata
                shape = reader.get_shape()
                cx, cy = shape[0] // 2, shape[1] // 2
                half = min(256, shape[0] // 2, shape[1] // 2)
                chip = reader.read_chip(cx - half, cx + half,
                                        cy - half, cy + half)

            import tempfile
            tmpdir = Path(tempfile.mkdtemp(prefix="grdl_bench_sicd_"))
            try:
                out_path = tmpdir / "bench_sicd.nitf"

                def _sicd_write():
                    SICDWriter(out_path).write(chip, metadata=sicd_meta)

                r = _bench("SICDWriter.write.real_data", _sicd_write, **kw)
                if r:
                    results.append(r)
            finally:
                import shutil
                shutil.rmtree(tmpdir, ignore_errors=True)

        except (ImportError, Exception) as exc:
            print(f"  SKIP  SICDWriter benchmarks ({exc})")

    # ------------------------------------------------------------------
    # NITFWriter (write synthetic array)
    # ------------------------------------------------------------------
    try:
        from grdl.IO.nitf import NITFWriter

        import tempfile
        tmpdir = Path(tempfile.mkdtemp(prefix="grdl_bench_nitf_"))
        try:
            nitf_out = tmpdir / "bench_nitf.nitf"
            synthetic = np.random.rand(rows, cols).astype(np.float32)

            def _nitf_write():
                NITFWriter(nitf_out).write(synthetic)

            r = _bench(f"NITFWriter.write.{tags.get('array_size', 'unknown')}",
                        _nitf_write, **kw)
            if r:
                results.append(r)
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    except (ImportError, Exception) as exc:
        print(f"  SKIP  NITFWriter benchmarks ({exc})")
```

### 2.4 SAR Sub-Aperture Processing — Extend `run_sar_processing_benchmarks()`

After the existing `SublookDecomposition` block, add:

```python
    # --- MultilookDecomposition ---
    try:
        from grdl.image_processing.sar import MultilookDecomposition

        if sar_path:
            for looks_rg, looks_az in [(2, 2), (3, 3)]:
                ml = MultilookDecomposition(
                    metadata=metadata, looks_rg=looks_rg, looks_az=looks_az,
                )
                _chip = chip
                r = _bench(
                    f"MultilookDecomposition.decompose.{looks_rg}x{looks_az}",
                    ml.decompose,
                    setup=lambda _c=_chip: ((_c,), {}), **real_kw,
                )
                if r:
                    results.append(r)

    except (ImportError, Exception) as exc:
        print(f"  SKIP  MultilookDecomposition benchmarks ({exc})")

    # --- CSIProcessor ---
    try:
        from grdl.image_processing.sar import CSIProcessor

        if sar_path:
            csi = CSIProcessor(metadata=metadata)
            _chip = chip
            r = _bench("CSIProcessor.process", csi.process,
                        setup=lambda _c=_chip: ((_c,), {}), **real_kw)
            if r:
                results.append(r)

    except (ImportError, Exception) as exc:
        print(f"  SKIP  CSIProcessor benchmarks ({exc})")
```

### 2.5 Elevation Models — Extend `run_geolocation_benchmarks()`

After the NoGeolocation block (added in Phase 1), before `return results`:

```python
    # --- DTEDElevation ---
    _data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    dted_dir = _data_dir / "dted"
    if dted_dir.exists() and list(dted_dir.glob("**/*.dt?")):
        try:
            from grdl.geolocation.elevation import DTEDElevation

            elev = DTEDElevation(str(dted_dir))
            lat_arr = np.random.uniform(33.0, 35.0, size=10000)
            lon_arr = np.random.uniform(-119.0, -117.0, size=10000)

            r = _bench(
                "DTEDElevation.get_elevation.batch10000",
                elev.get_elevation,
                setup=lambda: ((lat_arr, lon_arr), {}),
                **{**kw, "tags": {**tags, "data": "real"}},
            )
            if r:
                results.append(r)

        except (ImportError, Exception) as exc:
            print(f"  SKIP  DTEDElevation benchmarks ({exc})")
    else:
        print("  SKIP  DTEDElevation benchmarks (data files not found)")

    # --- GeoTIFFDEM ---
    dem_path = _find_data_file(_data_dir / "dem", "*.tif")
    if dem_path:
        try:
            from grdl.geolocation.elevation import GeoTIFFDEM

            elev = GeoTIFFDEM(dem_path)
            lat_arr = np.random.uniform(33.0, 35.0, size=10000)
            lon_arr = np.random.uniform(-119.0, -117.0, size=10000)

            r = _bench(
                "GeoTIFFDEM.get_elevation.batch10000",
                elev.get_elevation,
                setup=lambda: ((lat_arr, lon_arr), {}),
                **{**kw, "tags": {**tags, "data": "real"}},
            )
            if r:
                results.append(r)

        except (ImportError, Exception) as exc:
            print(f"  SKIP  GeoTIFFDEM benchmarks ({exc})")
    else:
        print("  SKIP  GeoTIFFDEM benchmarks (data files not found)")

    # --- GeoidCorrection ---
    geoid_path = _find_data_file(_data_dir / "geoid", "*.pgm")
    if geoid_path:
        try:
            from grdl.geolocation.elevation import GeoidCorrection

            geoid = GeoidCorrection(geoid_path)
            lat_arr = np.random.uniform(33.0, 35.0, size=10000)
            lon_arr = np.random.uniform(-119.0, -117.0, size=10000)

            r = _bench(
                "GeoidCorrection.get_undulation.batch10000",
                geoid.get_undulation,
                setup=lambda: ((lat_arr, lon_arr), {}),
                **{**kw, "tags": {**tags, "data": "real"}},
            )
            if r:
                results.append(r)

        except (ImportError, Exception) as exc:
            print(f"  SKIP  GeoidCorrection benchmarks ({exc})")
    else:
        print("  SKIP  GeoidCorrection benchmarks (data files not found)")
```

### 2.6 New Test Files — Elevation + Geolocation

#### `tests/validation/test_elevation_models.py`

```python
"""DTEDElevation, GeoTIFFDEM, and GeoidCorrection validation."""

# Imports:
# from grdl.geolocation.elevation import DTEDElevation, GeoTIFFDEM, GeoidCorrection
# Markers: pytest.mark.elevation, requires_data
# Fixtures: require_dted_dir, require_dem_file, require_geoid_file

# DTEDElevation Tests:
# Level 1:
#   test_dted_constructor — Accepts directory path
#   test_dted_get_elevation_scalar — Returns float for single lat/lon
# Level 2:
#   test_dted_elevation_range — Returns values in [-500, 9000] meters
#   test_dted_known_location — Check Mt. Whitney or similar known elevation
# Level 3:
#   test_dted_batch_10k — 10,000-point vectorized query, correct shapes

# GeoTIFFDEM Tests:
# Level 1:
#   test_geotiff_dem_constructor — Accepts .tif path
#   test_geotiff_dem_scalar — Returns float
# Level 2:
#   test_geotiff_dem_range — Physically reasonable elevation range
#   test_geotiff_dem_finite — No NaN/Inf
# Level 3:
#   test_geotiff_dem_batch_10k — Vectorized query

# GeoidCorrection Tests:
# Level 1:
#   test_geoid_constructor — Accepts .pgm path
#   test_geoid_scalar — Returns float undulation
# Level 2:
#   test_geoid_range — EGM96 undulation in [-110, +90] meters globally
#   test_geoid_equator — Known undulation at equator/Greenwich ≈ 17m
# Level 3:
#   test_geoid_batch_10k — Vectorized query
```

#### `tests/validation/test_geolocation_sentinel1.py`

```python
"""Sentinel1SLCGeolocation and NoGeolocation validation."""

# Imports:
# from grdl.geolocation import Sentinel1SLCGeolocation, NoGeolocation
# from grdl.IO.sar import Sentinel1SLCReader
# Markers: pytest.mark.sentinel1, pytest.mark.geolocation

# NoGeolocation (synthetic — always runs):
# Level 1:
#   test_nogeo_image_to_latlon — Returns (row, col) unchanged
#   test_nogeo_latlon_to_image — Returns (lat, lon) unchanged
# Level 2:
#   test_nogeo_roundtrip — image_to_latlon → latlon_to_image = identity
#   test_nogeo_batch — Vectorized 1000 points
# Level 3:
#   test_nogeo_shape_property — .shape returns constructor shape

# Sentinel1SLCGeolocation (real data — skips if no data):
# Level 1:
#   test_s1_geo_from_reader — Sentinel1SLCGeolocation.from_reader() succeeds
#   test_s1_geo_image_to_latlon — Returns lat in [-90,90], lon in [-180,180]
# Level 2:
#   test_s1_geo_roundtrip — image_to_latlon → latlon_to_image within 1 pixel
#   test_s1_geo_corner_points — All 4 corners produce valid coordinates
# Level 3:
#   test_s1_geo_batch_1000 — Vectorized 1000 points
```

### 2.7 pyproject.toml — Additional Markers

```toml
"cphd: CPHD format tests",
"crsd: CRSD format tests",
"sidd: SIDD format tests",
"sentinel1: Sentinel-1 SLC tests",
"sentinel2: Sentinel-2 L2A tests",
"aster: ASTER L1T tests",
"biomass: BIOMASS L1 tests",
"elevation: Elevation model tests",
"geolocation: Geolocation transform tests",
```

---

## Phase 3: SAR Image Formation + YAML Workflow

### 3.1 New Function: `run_image_formation_benchmarks()` in suite.py

```python
def run_image_formation_benchmarks(
    store: JSONBenchmarkStore,
    rows: int,
    cols: int,
    iterations: int,
    warmup: int,
    tags: Dict[str, str],
) -> List:
    """Benchmark SAR image formation algorithms."""
    _section("SAR Image Formation")
    sz = tags["array_size"]
    results = []
    mod = "image_processing.sar.image_formation"
    kw = dict(store=store, iterations=iterations, warmup=warmup,
              tags=tags, module=mod)

    _data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    cphd_dir = _data_dir / "cphd"
    cphd_path = _find_data_file(cphd_dir, "*.cphd")

    if not cphd_path:
        print("  SKIP  Image formation benchmarks (CPHD data not found)")
        return results

    try:
        from grdl.IO.sar import CPHDReader
        from grdl.image_processing.sar import (
            CollectionGeometry, PolarGrid, PolarFormatAlgorithm,
            SubaperturePartitioner, StripmapPFA, RangeDopplerAlgorithm,
            FastBackProjection,
        )

        with CPHDReader(cphd_path) as reader:
            metadata = reader.typed_metadata
            phase_data = reader.read_full()

        real_kw = dict(store=store, iterations=iterations, warmup=warmup,
                       tags={**tags, "data": "real"}, module=mod)

        # CollectionGeometry
        r = _bench("CollectionGeometry.init.real_data",
                    lambda: CollectionGeometry(metadata),
                    version="1.0.0", **real_kw)
        if r:
            results.append(r)

        geom = CollectionGeometry(metadata)

        # PolarGrid
        r = _bench("PolarGrid.init.real_data",
                    lambda: PolarGrid(geom),
                    version="1.0.0", **real_kw)
        if r:
            results.append(r)

        grid = PolarGrid(geom)

        # PFA
        pfa = PolarFormatAlgorithm(geometry=geom, grid=grid)
        _pd = phase_data
        r = _bench("PolarFormatAlgorithm.form.real_data", pfa.form,
                    setup=lambda _p=_pd: ((_p,), {}),
                    version="1.0.0", **real_kw)
        if r:
            results.append(r)

        # SubaperturePartitioner
        part = SubaperturePartitioner(n_subapertures=4)
        r = _bench("SubaperturePartitioner.partition.real_data",
                    part.partition,
                    setup=lambda _p=_pd, _g=geom: ((_p, _g), {}),
                    version="1.0.0", **real_kw)
        if r:
            results.append(r)

        # RDA (may not be compatible with all data)
        try:
            rda = RangeDopplerAlgorithm(geometry=geom)
            r = _bench("RangeDopplerAlgorithm.form.real_data", rda.form,
                        setup=lambda _p=_pd: ((_p,), {}),
                        version="1.0.0", **real_kw)
            if r:
                results.append(r)
        except Exception as exc:
            print(f"  SKIP  RangeDopplerAlgorithm ({exc})")

        # StripmapPFA
        try:
            spfa = StripmapPFA(geometry=geom, grid=grid, n_subapertures=4)
            r = _bench("StripmapPFA.form.real_data", spfa.form,
                        setup=lambda _p=_pd: ((_p,), {}),
                        version="1.0.0", **real_kw)
            if r:
                results.append(r)
        except Exception as exc:
            print(f"  SKIP  StripmapPFA ({exc})")

        # FastBackProjection
        try:
            fbp = FastBackProjection(geometry=geom, n_subapertures=4)
            r = _bench("FastBackProjection.form.real_data", fbp.form,
                        setup=lambda _p=_pd: ((_p,), {}),
                        version="1.0.0", **real_kw)
            if r:
                results.append(r)
        except Exception as exc:
            print(f"  SKIP  FastBackProjection ({exc})")

    except (ImportError, Exception) as exc:
        print(f"  SKIP  Image formation benchmarks ({exc})")

    return results
```

Register in BENCHMARK_GROUPS:
```python
BENCHMARK_GROUPS = {
    ...
    "image_formation": run_image_formation_benchmarks,
}
```

### 3.2 New Test File: `tests/validation/test_sar_image_formation.py`

```python
"""SAR Image Formation algorithm validation."""

# Imports:
# from grdl.image_processing.sar import (
#     CollectionGeometry, PolarGrid, PolarFormatAlgorithm,
#     SubaperturePartitioner, StripmapPFA, RangeDopplerAlgorithm,
#     FastBackProjection,
# )
# Markers: pytest.mark.sar, pytest.mark.image_formation, requires_data (CPHD)
# Fixture: require_cphd_file

# Level 1 (structure):
#   test_collection_geometry_init — Constructs from CPHD metadata
#   test_collection_geometry_attributes — Has .grazing_angle, .squint_angle, etc.
#   test_polar_grid_init — Constructs from CollectionGeometry
#   test_polar_grid_bounds — grid.bounds returns 4-tuple of floats

# Level 2 (data quality):
#   test_pfa_form_returns_complex_2d — PFA produces (rows, cols) complex array
#   test_pfa_form_nonzero — Output has non-zero content
#   test_subaperture_partitioner — Produces n_subapertures sub-arrays
#   test_rda_form_returns_complex_2d — RDA produces complex image
#   test_stripmapPFA_form — StripmapPFA output shape and type
#   test_fbp_form — FFBP output shape and type

# Level 3 (integration):
#   test_pfa_to_magnitude — PFA → np.abs() → float array in valid range
#   test_collection_geometry_coordinate_systems — .ecef_to_arp, .arp_to_slant valid
```

### 3.3 New Test File: `tests/validation/test_sar_multilook.py`

```python
"""MultilookDecomposition and CSIProcessor validation."""

# Imports:
# from grdl.image_processing.sar import MultilookDecomposition, CSIProcessor
# from grdl.IO.sar import SICDReader
# Markers: pytest.mark.sar, requires_data (SICD)
# Fixture: require_umbra_file (existing)

# MultilookDecomposition:
# Level 1:
#   test_multilook_init — Constructs with metadata + looks parameters
#   test_multilook_decompose_returns_array — decompose() returns ndarray
# Level 2:
#   test_multilook_output_shape — Output shape = input / (looks_rg * looks_az)
#   test_multilook_reduces_speckle — Variance of output < variance of input
# Level 3:
#   test_multilook_2x2_vs_3x3 — 3x3 produces smaller output than 2x2

# CSIProcessor:
# Level 1:
#   test_csi_init — Constructs with SICD metadata
#   test_csi_process_returns_array — .process() returns ndarray
# Level 2:
#   test_csi_output_3channel — Output shape is (rows, cols, 3)
#   test_csi_output_real — Output is real-valued float
# Level 3:
#   test_csi_output_bounded — Values in [0, 1] after normalization
```

### 3.4 YAML Workflow Updates

Modify `workflows/comprehensive_benchmark_workflow.yaml`:

**Header update:**
```yaml
schema_version: "2.0"
name: comprehensive_benchmark_workflow
version: "2.0.0"
description: >-
  Full GRDL processing pipeline: IO → filtering → intensity →
  decomposition → SAR processing → detection → orthorectification.
```

**New steps to add after existing steps:**

```yaml
  # ── Stage 3b: Polarimetric decomposition ──────────────────────────────
  - processor: grdl.image_processing.decomposition.DualPolHAlpha
    version: "1.0.0"
    id: dual_pol_halpha
    depends_on: [despeckle]
    phase: global_processing
    params:
      window_size: 7

  # ── Stage 3c: SAR sub-aperture processing ─────────────────────────────
  - processor: grdl.image_processing.sar.MultilookDecomposition
    version: "1.0.0"
    id: multilook_2x2
    depends_on: [despeckle]
    phase: global_processing
    params:
      looks_rg: 2
      looks_az: 2

  - processor: grdl.image_processing.sar.CSIProcessor
    version: "0.2.0"
    id: csi_rgb
    depends_on: [despeckle]
    phase: global_processing
    params:
      overlap: 0.1
      deweight: true
      normalize: true

  # ── Stage 4: CFAR detection on dB branch ──────────────────────────────
  - processor: grdl.image_processing.detection.cfar.CACFARDetector
    version: "1.0.0"
    id: cfar_ca
    depends_on: [to_db]
    phase: global_processing
    params:
      guard_cells: 3
      training_cells: 12
      pfa: 0.001

  - processor: grdl.image_processing.detection.cfar.GOCFARDetector
    version: "1.0.0"
    id: cfar_go
    depends_on: [to_db]
    phase: global_processing
    params:
      guard_cells: 3
      training_cells: 12
      pfa: 0.001

  - processor: grdl.image_processing.detection.cfar.SOCFARDetector
    version: "1.0.0"
    id: cfar_so
    depends_on: [to_db]
    phase: global_processing
    params:
      guard_cells: 3
      training_cells: 12
      pfa: 0.001

  - processor: grdl.image_processing.detection.cfar.OSCFARDetector
    version: "1.0.0"
    id: cfar_os
    depends_on: [to_db]
    phase: global_processing
    params:
      guard_cells: 3
      training_cells: 12
      pfa: 0.001

  # ── Stage 5: Ortho pipeline (finalization) ────────────────────────────
  - processor: grdl.image_processing.ortho.OrthoPipeline
    version: "1.0.0"
    id: ortho_pipeline
    depends_on: [percentile_stretch]
    phase: finalization
    params:
      interpolation: bilinear

  # ── Image Formation branch (CPHD input, conditional) ─────────────────
  - processor: grdl.image_processing.sar.image_formation.CollectionGeometry
    version: "1.0.0"
    id: collection_geometry
    phase: global_processing
    params: {}

  - processor: grdl.image_processing.sar.image_formation.PolarGrid
    version: "1.0.0"
    id: polar_grid
    depends_on: [collection_geometry]
    phase: global_processing
    params: {}

  - processor: grdl.image_processing.sar.image_formation.PolarFormatAlgorithm
    version: "1.0.0"
    id: pfa_form
    depends_on: [polar_grid]
    phase: global_processing
    params: {}

  - processor: grdl.image_processing.sar.image_formation.RangeDopplerAlgorithm
    version: "1.0.0"
    id: rda_form
    depends_on: [collection_geometry]
    phase: global_processing
    params: {}

  - processor: grdl.image_processing.sar.image_formation.StripmapPFA
    version: "1.0.0"
    id: stripmap_pfa
    depends_on: [polar_grid]
    phase: global_processing
    params:
      n_subapertures: 4

  - processor: grdl.image_processing.sar.image_formation.FastBackProjection
    version: "1.0.0"
    id: ffbp
    depends_on: [collection_geometry]
    phase: global_processing
    params:
      n_subapertures: 4

  - processor: grdl.image_processing.sar.image_formation.SubaperturePartitioner
    version: "1.0.0"
    id: subaperture_partition
    depends_on: [collection_geometry]
    phase: global_processing
    params:
      n_subapertures: 4
```

---

## Phase 4: Verification & Gap Closure

### 4.1 Run All Synthetic Tests (No Data Required)

```bash
pytest tests/validation/ -v -m "not requires_data" --tb=short
```

Expected: All Phase 1 + NoGeolocation tests pass. Real-data tests skip cleanly.

### 4.2 Run Full Validation Suite

```bash
pytest tests/validation/ -v --tb=short
```

Expected: Real-data tests skip with clear messages when data files are absent.
When data is present, all tests pass.

### 4.3 Benchmark Suite Smoke Test

```bash
# Quick verification — small arrays, detection group only
python -c "
from grdl_te.benchmarking.suite import run_suite
run_suite(size='small', only=['detection'])
"

# Full suite
python -c "
from grdl_te.benchmarking.suite import run_suite
run_suite(size='small')
"

# Image formation (requires CPHD data)
python -c "
from grdl_te.benchmarking.suite import run_suite
run_suite(size='small', only=['image_formation'])
"
```

### 4.4 YAML Workflow Validation

```bash
python -c "
from grdl_rt.api import load_workflow
wf = load_workflow('workflows/comprehensive_benchmark_workflow.yaml')
print(f'{len(wf.steps)} steps loaded')
for s in wf.steps:
    print(f'  {s.id}: {s.processor} (depends_on={s.depends_on})')
"
```

### 4.5 Update Coverage Gap Document

Update `BENCHMARK_COVERAGE_GAPS.md` to mark all 36 components as covered:

```
Total Components: 70
Benchmarked:      70 (100% coverage)
Missing:          0
```

---

## Component Summary by Phase

| Phase | Components | Data Type | Files |
|-------|-----------|-----------|-------|
| 1 (Done) | CACFARDetector, GOCFARDetector, SOCFARDetector, OSCFARDetector, Detection, DetectionSet, Fields, FieldDefinition, DualPolHAlpha, OrthoPipeline, OrthoResult, compute_output_resolution, ProjectiveCoRegistration, NoGeolocation, DATA_DICTIONARY | Synthetic | 5 test files, suite.py edits |
| 2 | CPHDReader, CRSDReader, SIDDReader, Sentinel1SLCReader, Sentinel2Reader, ASTERReader, BIOMASSL1Reader, BIOMASSCatalog, SICDWriter, NITFWriter, MultilookDecomposition, CSIProcessor, DTEDElevation, GeoTIFFDEM, GeoidCorrection, Sentinel1SLCGeolocation | Real data | 10 test files, conftest.py, suite.py edits |
| 3 | CollectionGeometry, PolarGrid, PolarFormatAlgorithm, SubaperturePartitioner, StripmapPFA, RangeDopplerAlgorithm, FastBackProjection | Real data (CPHD) | 2 test files, suite.py new group, YAML workflow |
| 4 | Verification only | — | Gap doc update |

---

## Implementation Notes

### Important API Details Discovered in Phase 1

1. **RegistrationResult** uses `transform_matrix` (not `transform`) and `residual_rms` (not `rms_error`)
2. **`list_fields()`** returns `List[FieldDefinition]` objects — compare via `.name` attribute, not string equality
3. **DualPolHAlpha** `component_names` property returns `('entropy', 'alpha', 'anisotropy', 'span')`
4. **OrthoPipeline** builder: `.with_source_array()`, `.with_geolocation()`, `.with_resolution(lat, lon)`, `.with_interpolation(str)`, `.with_nodata(float)`, `.with_elevation(ElevationModel)`, `.run()` → `OrthoResult`
5. **OrthoResult** has `.data`, `.shape`, `.geolocation_metadata` (dict with 'crs', 'transform'), `.output_grid` (OutputGrid), `.save_geotiff(path)`
6. **CFAR detectors** constructor: `guard_cells=3, training_cells=12, pfa=1e-3, min_pixels=9, assumption='gaussian'`
7. **Lambda capture in benchmarks**: Always use `setup=lambda _x=var: ((_x,), {})` to avoid closure issues with mutable variables

### Skip Pattern for Real-Data Tests

All real-data tests must follow this pattern:
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
# If data is absent, test skips with a message linking to README
```

### Benchmark Skip Pattern

All benchmark entries in suite.py follow:
```python
try:
    from grdl.some_module import SomeClass
    # ... benchmark code ...
except (ImportError, Exception) as exc:
    print(f"  SKIP  SomeClass benchmarks ({exc})")
```

This ensures the suite always completes, even with missing optional dependencies or data.
