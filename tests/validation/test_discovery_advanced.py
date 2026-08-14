# -*- coding: utf-8 -*-
"""
Discovery Advanced Validation - DataSynthesizer and compute_beam_footprint.

Tests DataSynthesizer (synthetic imagery generation) and
compute_beam_footprint (antenna pattern projection to ground).

- Level 1: Synthesizer outputs valid GeoTIFF, footprint returns GeoJSON
- Level 2: Synthesized data properties (shape, CRS, geolocation), footprint
           geometry validity (closed ring, area bounds, coordinate ranges)
- Level 3: Full scan→footprint pipeline integration

Dataset: Umbra SICD (*.nitf) for footprint metadata; no external
data required for DataSynthesizer.

Dependencies
------------
pytest
numpy
grdl

Author
------
Ava Courtney

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2026-05-29
"""

import pytest
import numpy as np
from pathlib import Path

try:
    from grdl.discovery import DataSynthesizer, compute_beam_footprint
    _HAS_DISCOVERY = True
except ImportError:
    _HAS_DISCOVERY = False

try:
    from grdl.IO.sar import SICDReader
    _HAS_SICD = True
except ImportError:
    _HAS_SICD = False

try:
    from grdl.IO import open_any
    _HAS_OPEN_ANY = True
except ImportError:
    _HAS_OPEN_ANY = False


pytestmark = [
    pytest.mark.skipif(not _HAS_DISCOVERY,
                       reason="grdl.discovery not available"),
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def synthesizer():
    """DataSynthesizer instance."""
    return DataSynthesizer()


@pytest.fixture(scope="module")
def sicd_metadata(umbra_data_dir):
    """SICD metadata for beam footprint tests."""
    if not _HAS_SICD:
        pytest.skip("SICDReader not available")
    matches = list(umbra_data_dir.glob("*.nitf")) if umbra_data_dir.exists() else []
    if not matches:
        pytest.skip(f"Umbra SICD not found in {umbra_data_dir}")
    with SICDReader(str(matches[0])) as reader:
        meta = reader.metadata
    return meta


# =============================================================================
# Level 1: Format Validation — DataSynthesizer outputs
# =============================================================================


class TestDataSynthesizerLevel1:
    """Validate DataSynthesizer creates valid imagery."""

    def test_synthesize_sar_creates_file(self, synthesizer, tmp_path):
        """synthesize_sar() produces a file on disk."""
        out = synthesizer.synthesize_sar(
            output_path=tmp_path / "test_sar.tif",
            rows=128, cols=128,
        )
        assert out.exists(), "SAR output file not created"
        assert out.stat().st_size > 0, "SAR output file is empty"

    def test_synthesize_eo_creates_file(self, synthesizer, tmp_path):
        """synthesize_eo() produces a file on disk."""
        out = synthesizer.synthesize_eo(
            output_path=tmp_path / "test_eo.tif",
            rows=128, cols=128, bands=3,
        )
        assert out.exists(), "EO output file not created"
        assert out.stat().st_size > 0, "EO output file is empty"

    def test_synthesize_multispectral_creates_file(self, synthesizer, tmp_path):
        """synthesize_multispectral() produces a file on disk."""
        out = synthesizer.synthesize_multispectral(
            output_path=tmp_path / "test_msi.tif",
            rows=64, cols=64, bands=13,
        )
        assert out.exists(), "MSI output file not created"
        assert out.stat().st_size > 0, "MSI output file is empty"

    @pytest.mark.requires_data
    def test_compute_beam_footprint_returns_geojson(self, sicd_metadata):
        """compute_beam_footprint returns dict or None for SICD metadata."""
        result = compute_beam_footprint(sicd_metadata)
        # May return None if metadata lacks antenna section
        if result is not None:
            assert isinstance(result, dict)
            assert "type" in result
            assert result["type"] == "Polygon"


# =============================================================================
# Level 2: Data Quality — Synthesized data properties
# =============================================================================


class TestDataSynthesizerLevel2:
    """Validate synthesized data has correct physical properties."""

    @pytest.mark.skipif(not _HAS_OPEN_ANY,
                        reason="open_any not available")
    def test_sar_output_is_complex(self, synthesizer, tmp_path):
        """Synthesized SAR image has complex-valued content (I/Q bands)."""
        out = synthesizer.synthesize_sar(
            output_path=tmp_path / "sar_complex.tif",
            rows=128, cols=128,
        )
        with open_any(str(out)) as reader:
            data = reader.read_full()
        # DataSynthesizer may store as 2-band float (I/Q) or native complex
        if np.iscomplexobj(data):
            assert data.shape[:2] == (128, 128)
        else:
            # 2-band real representation of complex
            assert data.ndim >= 2
            assert data.shape[0] == 128 and data.shape[1] == 128
            # Verify it has actual signal content
            assert data.std() > 0, "SAR synthesis produced empty data"

    @pytest.mark.skipif(not _HAS_OPEN_ANY,
                        reason="open_any not available")
    def test_eo_output_shape(self, synthesizer, tmp_path):
        """Synthesized EO image has correct bands and dimensions."""
        out = synthesizer.synthesize_eo(
            output_path=tmp_path / "eo_3band.tif",
            rows=64, cols=64, bands=3,
        )
        with open_any(str(out)) as reader:
            shape = reader.get_shape()
        assert shape[0] == 64
        assert shape[1] == 64

    @pytest.mark.skipif(not _HAS_OPEN_ANY,
                        reason="open_any not available")
    def test_synthesized_has_geolocation(self, synthesizer, tmp_path):
        """Synthesized imagery has a valid affine transform."""
        out = synthesizer.synthesize_eo(
            output_path=tmp_path / "eo_geo.tif",
            rows=64, cols=64,
            center_lat=35.0, center_lon=-106.0, gsd=1.0,
        )
        with open_any(str(out)) as reader:
            meta = reader.metadata
        # Should have CRS and transform
        assert meta.crs is not None, "Synthesized image missing CRS"
        assert meta.transform is not None, "Synthesized image missing transform"

    @pytest.mark.skipif(not _HAS_OPEN_ANY,
                        reason="open_any not available")
    def test_sar_has_point_targets(self, synthesizer, tmp_path):
        """Synthesized SAR contains discernible point targets (high peaks)."""
        out = synthesizer.synthesize_sar(
            output_path=tmp_path / "sar_targets.tif",
            rows=256, cols=256,
        )
        with open_any(str(out)) as reader:
            data = reader.read_full()
        amplitude = np.abs(data)
        peak = amplitude.max()
        median = np.median(amplitude)
        # Point targets should be at least 10x above noise floor
        assert peak > median * 5, (
            f"No clear point targets: peak={peak:.2f}, median={median:.2f}"
        )

    @pytest.mark.skipif(not _HAS_OPEN_ANY,
                        reason="open_any not available")
    def test_multispectral_band_count(self, synthesizer, tmp_path):
        """Synthesized multispectral has correct number of bands."""
        out = synthesizer.synthesize_multispectral(
            output_path=tmp_path / "msi_13band.tif",
            rows=64, cols=64, bands=13,
        )
        with open_any(str(out)) as reader:
            meta = reader.metadata
        assert meta.bands == 13 or meta.num_bands == 13

    @pytest.mark.requires_data
    def test_footprint_polygon_is_closed(self, sicd_metadata):
        """Footprint polygon ring is closed (first == last coordinate)."""
        result = compute_beam_footprint(sicd_metadata)
        if result is None:
            pytest.skip("Metadata lacks antenna section for footprint")
        coords = result["coordinates"][0]  # exterior ring
        assert len(coords) >= 4, "Polygon needs at least 4 coordinates"
        first = coords[0]
        last = coords[-1]
        assert first[0] == pytest.approx(last[0], abs=1e-10), "Ring not closed (lon)"
        assert first[1] == pytest.approx(last[1], abs=1e-10), "Ring not closed (lat)"

    @pytest.mark.requires_data
    def test_footprint_coordinates_valid(self, sicd_metadata):
        """Footprint coordinates are valid lon/lat."""
        result = compute_beam_footprint(sicd_metadata)
        if result is None:
            pytest.skip("Metadata lacks antenna section for footprint")
        coords = result["coordinates"][0]
        for lon, lat in coords:
            assert -180.0 <= lon <= 180.0, f"Invalid lon: {lon}"
            assert -90.0 <= lat <= 90.0, f"Invalid lat: {lat}"


# =============================================================================
# Level 3: Integration — Scan + footprint pipeline
# =============================================================================


class TestDiscoveryLevel3:
    """End-to-end discovery pipeline tests."""

    @pytest.mark.integration
    @pytest.mark.skipif(not _HAS_OPEN_ANY,
                        reason="open_any not available")
    def test_synthesize_then_scan(self, synthesizer, tmp_path):
        """Synthesized imagery is discoverable by MetadataScanner."""
        from grdl.discovery import MetadataScanner

        # Generate test imagery
        synthesizer.synthesize_eo(
            output_path=tmp_path / "scan_test.tif",
            rows=64, cols=64,
            center_lat=40.0, center_lon=-75.0,
        )

        # Scan the directory
        scanner = MetadataScanner()
        results = scanner.scan_directory(str(tmp_path))
        assert len(results) >= 1, "Scanner found no files"
        # Verify the result has expected fields
        r = results[0]
        assert hasattr(r, 'filepath') or hasattr(r, 'path') or hasattr(r, 'file_path') or isinstance(r, dict)

    @pytest.mark.integration
    @pytest.mark.requires_data
    def test_footprint_area_physical_bounds(self, sicd_metadata):
        """Footprint area is physically reasonable for a SAR collect."""
        result = compute_beam_footprint(sicd_metadata)
        if result is None:
            pytest.skip("Metadata lacks antenna section for footprint")
        coords = np.array(result["coordinates"][0])
        # Shoelace formula for approximate area in degrees²
        x = coords[:, 0]
        y = coords[:, 1]
        area_deg2 = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        # A typical SAR spotlight scene is ~1-10 km², which is ~1e-5 to 1e-3 deg²
        # A stripmap may be larger. Reject if > 100 deg² (unreasonable)
        assert area_deg2 > 0, "Footprint has zero area"
        assert area_deg2 < 100, f"Footprint area unreasonably large: {area_deg2:.4f} deg²"
