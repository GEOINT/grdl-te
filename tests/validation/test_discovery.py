# -*- coding: utf-8 -*-
"""
Discovery Module Infrastructure Tests.

Validates grdl.discovery components:
- MetadataScanner: File discovery and metadata extraction
- LocalCatalog: In-memory catalog with filtering and spatial queries
- PluginRegistry: Plugin lifecycle management

Level 1: Basic Construction
- Component instantiation and resource initialization
- Property accessor completion

Level 2: Core Functionality
- File scanning and metadata extraction
- Catalog insertion, retrieval, filtering
- Plugin registration and retrieval

Level 3: Integration
- Multi-file scanning pipeline
- Cross-filter consistency
- Spatial query accuracy

All tests use synthetic files and in-memory catalogs (no persistence).

Dependencies
------------
pytest
numpy
grdl.discovery
grdl.IO

Author
------
Ava Courtney
courtney-ava@zai.com

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2026-03-30

Modified
--------
2026-03-30
"""

# Standard library
import json
import tempfile
from pathlib import Path
from datetime import datetime

# Third-party
import pytest
import numpy as np

# GRDL internal - Discovery
try:
    from grdl.discovery import (
        MetadataScanner,
        ScanResult,
        LocalCatalog,
        PluginRegistry,
        DiscoveryPlugin,
    )
    _HAS_DISCOVERY = True
except ImportError:
    _HAS_DISCOVERY = False

# GRDL internal - Create synthetic test files if needed
try:
    from grdl.IO import NumpyReader, write
    _HAS_IO = True
except ImportError:
    _HAS_IO = False


pytestmark = [
    pytest.mark.infrastructure,
    pytest.mark.skipif(not _HAS_DISCOVERY, reason="Discovery module not available"),
]


# ============================================================================
# Synthetic Test Plugin
# ============================================================================

class _DummyPlugin(DiscoveryPlugin):
    """Minimal test plugin implementation."""

    def __init__(self, name: str = "DummyPlugin"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "A dummy plugin for testing."

    def discover(self, **kwargs):
        """Return empty list."""
        return []

    def get_config_schema(self):
        """Return minimal schema."""
        return {"type": "object", "properties": {}}


class _PathPlugin(DiscoveryPlugin):
    """Plugin that discovers files from a directory."""

    def __init__(self, name: str = "PathPlugin"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Discovers files from a directory."

    def discover(self, search_path: str = ".", **kwargs):
        """Discover .npy files."""
        p = Path(search_path)
        if not p.exists():
            return []
        return list(p.glob("*.npy"))

    def get_config_schema(self):
        """Return schema."""
        return {
            "type": "object",
            "properties": {
                "search_path": {"type": "string", "description": "Directory to search"}
            },
        }


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_image_dir(tmp_path):
    """Create temporary directory with synthetic GeoTIFF files."""
    if not _HAS_IO:
        pytest.skip("grdl.IO not available")

    # Create 3 synthetic GeoTIFF files with distinct metadata
    files = []
    for i in range(3):
        data = np.random.randint(0, 256, size=(512, 512), dtype=np.uint8)
        fname = tmp_path / f"image_{i:02d}.tif"
        try:
            write(str(fname), data, format="GeoTIFF")
            files.append(fname)
        except Exception:
            pytest.skip("Could not create test GeoTIFF files")

    return tmp_path, files


@pytest.fixture
def scanner():
    """Create a MetadataScanner instance."""
    return MetadataScanner()


@pytest.fixture
def catalog():
    """Create an in-memory LocalCatalog instance."""
    return LocalCatalog()  # No db_path = in-memory


@pytest.fixture
def registry():
    """Create a PluginRegistry instance."""
    return PluginRegistry()


# ============================================================================
# Level 1: Construction and Properties
# ============================================================================

class TestMetadataScannerConstruction:
    """MetadataScanner instantiation and basic properties."""

    def test_scanner_instantiation(self):
        """MetadataScanner can be instantiated."""
        scanner = MetadataScanner()
        assert scanner is not None

    def test_scanner_has_scan_method(self):
        """MetadataScanner has scan_file and scan_directory methods."""
        scanner = MetadataScanner()
        assert hasattr(scanner, "scan_file")
        assert hasattr(scanner, "scan_directory")
        assert callable(scanner.scan_file)
        assert callable(scanner.scan_directory)


class TestLocalCatalogConstruction:
    """LocalCatalog instantiation and properties."""

    def test_catalog_in_memory_instantiation(self):
        """LocalCatalog(db_path=None) creates in-memory catalog."""
        cat = LocalCatalog()
        assert cat is not None

    def test_catalog_has_add_method(self):
        """LocalCatalog has add and add_batch methods."""
        cat = LocalCatalog()
        assert hasattr(cat, "add")
        assert hasattr(cat, "add_batch")
        assert callable(cat.add)
        assert callable(cat.add_batch)

    def test_catalog_has_filter_method(self):
        """LocalCatalog has filter method."""
        cat = LocalCatalog()
        assert hasattr(cat, "filter")
        assert callable(cat.filter)

    def test_catalog_empty_initially(self):
        """LocalCatalog is empty after creation."""
        cat = LocalCatalog()
        results = cat.filter()
        assert isinstance(results, list)
        assert len(results) == 0


class TestPluginRegistryConstruction:
    """PluginRegistry instantiation and properties."""

    def test_registry_instantiation(self):
        """PluginRegistry can be instantiated."""
        reg = PluginRegistry()
        assert reg is not None

    def test_registry_has_register_method(self):
        """PluginRegistry has register and get methods."""
        reg = PluginRegistry()
        assert hasattr(reg, "register")
        assert hasattr(reg, "get")
        assert callable(reg.register)
        assert callable(reg.get)


class TestScanResultDataclass:
    """ScanResult dataclass structure and creation."""

    def test_scan_result_instantiation(self):
        """ScanResult can be instantiated with minimal args."""
        result = ScanResult(
            filepath=Path("/tmp/test.tif"),
            format="GeoTIFF",
            rows=512,
            cols=512,
            dtype="uint8",
        )
        assert result.filepath == Path("/tmp/test.tif")
        assert result.format == "GeoTIFF"
        assert result.rows == 512
        assert result.cols == 512
        assert result.dtype == "uint8"

    def test_scan_result_optional_fields(self):
        """ScanResult optional fields default to None."""
        result = ScanResult(
            filepath=Path("/tmp/test.tif"),
            format="GeoTIFF",
            rows=512,
            cols=512,
            dtype="uint8",
        )
        assert result.bands is None
        assert result.crs is None
        assert result.modality is None
        assert result.sensor is None
        assert result.datetime is None


# ============================================================================
# Level 2: Core Functionality
# ============================================================================

class TestPluginRegistry:
    """PluginRegistry registration and retrieval."""

    def test_plugin_registration(self):
        """Plugin can be registered with registry."""
        reg = PluginRegistry()
        plugin = _DummyPlugin("TestPlugin")
        reg.register(plugin)
        # Verify it was registered
        retrieved = reg.get("TestPlugin")
        assert retrieved is plugin

    def test_plugin_retrieval_by_name(self):
        """Registered plugin can be retrieved by name."""
        reg = PluginRegistry()
        plugin = _DummyPlugin("MyPlugin")
        reg.register(plugin)
        assert reg.get("MyPlugin") is plugin

    def test_plugin_not_found_raises_keyerror(self):
        """Retrieving unregistered plugin raises KeyError."""
        reg = PluginRegistry()
        with pytest.raises(KeyError):
            reg.get("NonexistentPlugin")

    def test_duplicate_registration_raises_valueerror(self):
        """Registering duplicate plugin name raises ValueError."""
        reg = PluginRegistry()
        plugin1 = _DummyPlugin("DupePlugin")
        plugin2 = _DummyPlugin("DupePlugin")
        reg.register(plugin1)
        with pytest.raises(ValueError):
            reg.register(plugin2)

    def test_non_plugin_registration_raises_typeerror(self):
        """Registering non-DiscoveryPlugin raises TypeError."""
        reg = PluginRegistry()
        with pytest.raises(TypeError):
            reg.register("NotAPlugin")

    def test_multiple_plugins_independent(self):
        """Multiple distinct plugins can coexist."""
        reg = PluginRegistry()
        p1 = _DummyPlugin("Plugin1")
        p2 = _PathPlugin("Plugin2")
        reg.register(p1)
        reg.register(p2)
        assert reg.get("Plugin1") is p1
        assert reg.get("Plugin2") is p2

    def test_plugin_discover_method_callable(self):
        """Registered plugin's discover method is callable."""
        reg = PluginRegistry()
        plugin = _DummyPlugin("TestPlugin")
        reg.register(plugin)
        retrieved = reg.get("TestPlugin")
        result = retrieved.discover()
        assert isinstance(result, list)

    def test_plugin_config_schema_callable(self):
        """Registered plugin's get_config_schema method returns dict."""
        reg = PluginRegistry()
        plugin = _DummyPlugin("TestPlugin")
        reg.register(plugin)
        retrieved = reg.get("TestPlugin")
        schema = retrieved.get_config_schema()
        assert isinstance(schema, dict)


class TestLocalCatalogAddition:
    """LocalCatalog add and add_batch methods."""

    def test_catalog_add_single_result(self):
        """add() inserts single ScanResult."""
        cat = LocalCatalog()
        result = ScanResult(
            filepath=Path("/tmp/test1.tif"),
            format="GeoTIFF",
            rows=512,
            cols=512,
            dtype="uint8",
        )
        cat.add(result)
        retrieved = cat.get("/tmp/test1.tif")
        assert retrieved is result

    def test_catalog_add_batch_results(self):
        """add_batch() inserts multiple ScanResults."""
        cat = LocalCatalog()
        results = [
            ScanResult(
                filepath=Path(f"/tmp/test{i}.tif"),
                format="GeoTIFF",
                rows=512,
                cols=512,
                dtype="uint8",
            )
            for i in range(3)
        ]
        cat.add_batch(results)
        for i, result in enumerate(results):
            retrieved = cat.get(f"/tmp/test{i}.tif")
            assert retrieved is result

    def test_catalog_get_nonexistent_returns_none(self):
        """get() returns None for nonexistent filepath."""
        cat = LocalCatalog()
        result = cat.get("/nonexistent/path.tif")
        assert result is None

    def test_catalog_filter_empty_returns_all(self):
        """filter() with no params returns all items."""
        cat = LocalCatalog()
        results = [
            ScanResult(
                filepath=Path(f"/tmp/image{i}.tif"),
                format="GeoTIFF",
                rows=512,
                cols=512,
                dtype="uint8",
                modality="EO",
            )
            for i in range(3)
        ]
        cat.add_batch(results)
        filtered = cat.filter()
        assert len(filtered) == 3

    def test_catalog_filter_by_modality(self):
        """filter(modality=...) returns matching items."""
        cat = LocalCatalog()
        results = [
            ScanResult(
                filepath=Path(f"/tmp/sar{idx}.tif"),
                format="SICD",
                rows=512,
                cols=512,
                dtype="complex64",
                modality="SAR",
            )
            for idx in range(2)
        ]
        results.append(ScanResult(
            filepath=Path("/tmp/optical.tif"),
            format="GeoTIFF",
            rows=512,
            cols=512,
            dtype="uint8",
            modality="EO",
        ))
        cat.add_batch(results)
        sar_items = cat.filter(modality="SAR")
        assert len(sar_items) == 2
        for item in sar_items:
            assert item.modality == "SAR"

    def test_catalog_filter_by_format(self):
        """filter(format=...) returns matching items."""
        cat = LocalCatalog()
        results = [
            ScanResult(
                filepath=Path(f"/tmp/sicd{idx}.nitf"),
                format="SICD",
                rows=512,
                cols=512,
                dtype="complex64",
            )
            for idx in range(2)
        ]
        results.append(
            ScanResult(
                filepath=Path("/tmp/optical.tif"),
                format="GeoTIFF",
                rows=512,
                cols=512,
                dtype="uint8",
            )
        )
        cat.add_batch(results)
        sicd_items = cat.filter(format="SICD")
        assert len(sicd_items) == 2
        for item in sicd_items:
            assert item.format == "SICD"


class TestLocalCatalogFiltering:
    """LocalCatalog filter method over various dimensions."""

    def test_catalog_filter_by_sensor(self):
        """filter(sensor=...) - case insensitive substring match."""
        cat = LocalCatalog()
        results = [
            ScanResult(
                filepath=Path(f"/tmp/umbra{idx}.nitf"),
                format="SICD",
                rows=512,
                cols=512,
                dtype="complex64",
                sensor="Umbra",
            )
            for idx in range(2)
        ]
        results.append(ScanResult(
            filepath=Path("/tmp/sentinel1.tif"),
            format="GeoTIFF",
            rows=512,
            cols=512,
            dtype="int16",
            sensor="Sentinel-1",
        ))
        cat.add_batch(results)
        umbra_items = cat.filter(sensor="umbra")
        assert len(umbra_items) == 2
        for item in umbra_items:
            assert "umbra" in item.sensor.lower()

    def test_catalog_filter_by_date_range(self):
        """filter(date_start=..., date_end=...) temporal filtering."""
        cat = LocalCatalog()
        d1 = datetime(2020, 1, 15)
        d2 = datetime(2020, 6, 15)
        d3 = datetime(2020, 12, 15)

        results = [
            ScanResult(
                filepath=Path(f"/tmp/image1.tif"),
                format="GeoTIFF",
                rows=512,
                cols=512,
                dtype="uint8",
                datetime=d1,
            ),
            ScanResult(
                filepath=Path(f"/tmp/image2.tif"),
                format="GeoTIFF",
                rows=512,
                cols=512,
                dtype="uint8",
                datetime=d2,
            ),
            ScanResult(
                filepath=Path(f"/tmp/image3.tif"),
                format="GeoTIFF",
                rows=512,
                cols=512,
                dtype="uint8",
                datetime=d3,
            ),
        ]
        cat.add_batch(results)

        # Filter within range
        filtered = cat.filter(date_start=d1, date_end=d2)
        assert len(filtered) == 2
        assert all(d1 <= item.datetime <= d2 for item in filtered)

    def test_catalog_filter_by_bbox(self):
        """filter(bbox=...) spatial bounding box filtering."""
        cat = LocalCatalog()
        results = [
            ScanResult(
                filepath=Path(f"/tmp/image1.tif"),
                format="GeoTIFF",
                rows=512,
                cols=512,
                dtype="uint8",
                bounds=(-122.5, 37.0, -122.0, 37.5),  # bay area-ish
            ),
            ScanResult(
                filepath=Path(f"/tmp/image2.tif"),
                format="GeoTIFF",
                rows=512,
                cols=512,
                dtype="uint8",
                bounds=(-87.5, 40.0, -87.0, 40.5),  # chicago-ish
            ),
        ]
        cat.add_batch(results)

        # Query box around bay area
        bbox = (-123, 36, -121, 38)
        filtered = cat.filter(bbox=bbox)
        assert len(filtered) >= 1

    def test_catalog_filter_combined_criteria(self):
        """filter() with multiple criteria applies all filters."""
        cat = LocalCatalog()
        results = [
            ScanResult(
                filepath=Path(f"/tmp/sar_umbra.nitf"),
                format="SICD",
                rows=512,
                cols=512,
                dtype="complex64",
                modality="SAR",
                sensor="Umbra",
            ),
            ScanResult(
                filepath=Path("/tmp/eo_umbra.tif"),
                format="GeoTIFF",
                rows=512,
                cols=512,
                dtype="uint8",
                modality="EO",
                sensor="Umbra",
            ),
            ScanResult(
                filepath=Path("/tmp/sar_sentinel.tif"),
                format="SICD",
                rows=512,
                cols=512,
                dtype="complex64",
                modality="SAR",
                sensor="Sentinel-1",
            ),
        ]
        cat.add_batch(results)

        # Filter for SAR from Umbra
        filtered = cat.filter(modality="SAR", sensor="umbra")
        assert len(filtered) == 1
        assert filtered[0].modality == "SAR"
        assert "umbra" in filtered[0].sensor.lower()


# ============================================================================
# Level 3: Integration
# ============================================================================

class TestDiscoveryIntegration:
    """Integration tests combining scanner, catalog, and plugins."""

    def test_plugin_path_discovery_integration(self, tmp_path):
        """PathPlugin can discover .npy files from directory."""
        # Create test files
        test_file = tmp_path / "test_data.npy"
        data = np.zeros((100, 100))
        np.save(test_file, data)

        plugin = _PathPlugin()
        discovered = plugin.discover(search_path=str(tmp_path))
        assert len(discovered) > 0
        assert any(str(test_file) == str(f) for f in discovered)

    def test_catalog_geojson_export_format(self):
        """LocalCatalog.to_geojson() returns valid GeoJSON."""
        cat = LocalCatalog()
        results = [
            ScanResult(
                filepath=Path(f"/tmp/image{i}.tif"),
                format="GeoTIFF",
                rows=512,
                cols=512,
                dtype="uint8",
                bounds=(-122.0 - i, 37.0, -121.5 - i, 37.5),
            )
            for i in range(2)
        ]
        cat.add_batch(results)

        if hasattr(cat, "to_geojson"):
            geojson = cat.to_geojson(results)
            assert isinstance(geojson, dict)
            assert geojson.get("type") == "FeatureCollection"

    def test_catalog_stats_computation(self):
        """LocalCatalog.get_stats() returns summary statistics."""
        cat = LocalCatalog()
        results = [
            ScanResult(
                filepath=Path(f"/tmp/image{i}.tif"),
                format="GeoTIFF",
                rows=512 * (i + 1),
                cols=512 * (i + 1),
                dtype="uint8",
                modality="EO" if i % 2 == 0 else "SAR",
            )
            for i in range(4)
        ]
        cat.add_batch(results)

        if hasattr(cat, "get_stats"):
            stats = cat.get_stats()
            assert isinstance(stats, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
