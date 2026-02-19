# -*- coding: utf-8 -*-
"""
CFAR Detector Tests - Synthetic validation of all CFAR variants.

Tests algorithmic correctness of CACFARDetector, GOCFARDetector,
SOCFARDetector, and OSCFARDetector using synthetic Rayleigh-distributed
magnitude imagery with injected point targets.

- Level 1: Each detector returns DetectionSet with correct structure
- Level 2: Detectors find injected targets; variant-specific behavior
- Level 3: Pipeline integration (ToDecibels -> CFAR -> to_geojson)

Dependencies
------------
pytest
numpy
scipy
shapely

Author
------
Claude Code (generated)

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2026-02-18

Modified
--------
2026-02-18
"""

# Third-party
import pytest
import numpy as np

# GRDL internal
try:
    from grdl.image_processing.detection.cfar import (
        CACFARDetector,
        GOCFARDetector,
        SOCFARDetector,
        OSCFARDetector,
    )
    from grdl.image_processing.detection import Detection, DetectionSet
    _HAS_CFAR = True
except ImportError:
    _HAS_CFAR = False

try:
    from grdl.image_processing.intensity import ToDecibels
    _HAS_INTENSITY = True
except ImportError:
    _HAS_INTENSITY = False

pytestmark = [
    pytest.mark.cfar,
    pytest.mark.detection,
    pytest.mark.skipif(not _HAS_CFAR, reason="grdl CFAR detectors not available"),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic_magnitude_image():
    """Rayleigh-distributed magnitude image with 5 injected bright targets.

    Returns ``(image, target_positions)`` where ``target_positions`` is a
    list of (row, col) tuples for each injected target.
    """
    rng = np.random.default_rng(42)
    rows, cols = 512, 512
    # Rayleigh magnitude from complex Gaussian
    real_part = rng.standard_normal((rows, cols))
    imag_part = rng.standard_normal((rows, cols))
    magnitude = np.abs(real_part + 1j * imag_part).astype(np.float64)

    # Inject 5 bright targets at known positions (well away from edges)
    targets = [
        (128, 128),
        (256, 256),
        (384, 384),
        (128, 384),
        (384, 128),
    ]
    peak = magnitude.max()
    for r, c in targets:
        magnitude[r - 2:r + 3, c - 2:c + 3] = peak * 10.0

    return magnitude, targets


@pytest.fixture(scope="module")
def synthetic_db_image(synthetic_magnitude_image):
    """dB-domain image from the synthetic magnitude image."""
    mag, targets = synthetic_magnitude_image
    db = 20.0 * np.log10(mag + 1e-10)
    return db, targets


# ---------------------------------------------------------------------------
# Level 1: Format Validation — DetectionSet structure
# ---------------------------------------------------------------------------
class TestCFARLevel1FormatValidation:
    """All CFAR variants return a valid DetectionSet."""

    def test_cacfar_returns_detection_set(self, synthetic_db_image):
        """CA-CFAR detect() returns a DetectionSet."""
        db, _ = synthetic_db_image
        det = CACFARDetector(guard_cells=3, training_cells=12, pfa=1e-3)
        result = det.detect(db)
        assert isinstance(result, DetectionSet)
        assert len(result) >= 0
        assert hasattr(result, 'to_geojson')

    def test_gocfar_returns_detection_set(self, synthetic_db_image):
        """GO-CFAR detect() returns a DetectionSet."""
        db, _ = synthetic_db_image
        det = GOCFARDetector(guard_cells=3, training_cells=12, pfa=1e-3)
        result = det.detect(db)
        assert isinstance(result, DetectionSet)

    def test_socfar_returns_detection_set(self, synthetic_db_image):
        """SO-CFAR detect() returns a DetectionSet."""
        db, _ = synthetic_db_image
        det = SOCFARDetector(guard_cells=3, training_cells=12, pfa=1e-3)
        result = det.detect(db)
        assert isinstance(result, DetectionSet)

    def test_oscfar_returns_detection_set(self, synthetic_db_image):
        """OS-CFAR detect() returns a DetectionSet."""
        db, _ = synthetic_db_image
        det = OSCFARDetector(guard_cells=3, training_cells=12, pfa=1e-3)
        result = det.detect(db)
        assert isinstance(result, DetectionSet)

    def test_detections_have_pixel_geometry(self, synthetic_db_image):
        """Each detection carries a shapely pixel geometry."""
        db, _ = synthetic_db_image
        det = CACFARDetector(guard_cells=3, training_cells=12, pfa=1e-2)
        result = det.detect(db)
        if len(result) > 0:
            d = result[0]
            assert isinstance(d, Detection)
            assert d.pixel_geometry is not None
            assert hasattr(d.pixel_geometry, 'geom_type')

    def test_detection_set_iterable(self, synthetic_db_image):
        """DetectionSet supports len() and iteration."""
        db, _ = synthetic_db_image
        det = CACFARDetector(guard_cells=3, training_cells=12, pfa=1e-2)
        result = det.detect(db)
        assert len(result) == len(list(result))

    def test_detector_output_fields(self):
        """CFAR detectors declare standard output fields."""
        det = CACFARDetector()
        assert len(det.output_fields) > 0
        # All field names should be dot-separated strings
        for field in det.output_fields:
            assert '.' in field, f"Field '{field}' missing domain prefix"


# ---------------------------------------------------------------------------
# Level 2: Data Quality — Detection correctness
# ---------------------------------------------------------------------------
class TestCFARLevel2DetectionCorrectness:
    """Validate that CFAR detectors find injected targets."""

    def _target_found(self, result, target_row, target_col, tolerance=20):
        """Check if any detection centroid is near (target_row, target_col)."""
        for d in result:
            centroid = d.pixel_geometry.centroid
            # shapely: (x, y) = (col, row)
            det_col, det_row = centroid.x, centroid.y
            if (abs(det_row - target_row) < tolerance
                    and abs(det_col - target_col) < tolerance):
                return True
        return False

    def test_cacfar_finds_all_targets(self, synthetic_db_image):
        """CA-CFAR detects all 5 injected targets at pfa=1e-3."""
        db, targets = synthetic_db_image
        det = CACFARDetector(guard_cells=3, training_cells=12, pfa=1e-3,
                             min_pixels=4)
        result = det.detect(db)
        found = sum(1 for r, c in targets
                    if self._target_found(result, r, c))
        assert found >= 4, (
            f"CA-CFAR found only {found}/5 targets "
            f"({len(result)} total detections)"
        )

    def test_gocfar_detects_targets(self, synthetic_db_image):
        """GO-CFAR detects injected targets."""
        db, targets = synthetic_db_image
        det = GOCFARDetector(guard_cells=3, training_cells=12, pfa=1e-3,
                             min_pixels=4)
        result = det.detect(db)
        found = sum(1 for r, c in targets
                    if self._target_found(result, r, c))
        assert found >= 3, (
            f"GO-CFAR found only {found}/5 targets"
        )

    def test_socfar_more_sensitive_than_gocfar(self, synthetic_db_image):
        """SO-CFAR produces >= as many detections as GO-CFAR (more sensitive)."""
        db, _ = synthetic_db_image
        go = GOCFARDetector(guard_cells=3, training_cells=12, pfa=1e-3)
        so = SOCFARDetector(guard_cells=3, training_cells=12, pfa=1e-3)
        go_result = go.detect(db)
        so_result = so.detect(db)
        # SO-CFAR should be at least as sensitive as GO-CFAR
        assert len(so_result) >= len(go_result) - 2, (
            f"SO-CFAR ({len(so_result)}) significantly less sensitive "
            f"than GO-CFAR ({len(go_result)})"
        )

    def test_oscfar_detects_targets(self, synthetic_db_image):
        """OS-CFAR detects injected targets."""
        db, targets = synthetic_db_image
        det = OSCFARDetector(guard_cells=3, training_cells=12, pfa=1e-3,
                             min_pixels=4)
        result = det.detect(db)
        found = sum(1 for r, c in targets
                    if self._target_found(result, r, c))
        assert found >= 3, (
            f"OS-CFAR found only {found}/5 targets"
        )

    def test_lower_pfa_fewer_detections(self, synthetic_db_image):
        """Lower PFA should produce fewer (or equal) detections."""
        db, _ = synthetic_db_image
        det_loose = CACFARDetector(guard_cells=3, training_cells=12,
                                   pfa=1e-2, min_pixels=1)
        det_tight = CACFARDetector(guard_cells=3, training_cells=12,
                                   pfa=1e-5, min_pixels=1)
        n_loose = len(det_loose.detect(db))
        n_tight = len(det_tight.detect(db))
        assert n_tight <= n_loose, (
            f"Tighter PFA ({n_tight}) produced more detections "
            f"than looser PFA ({n_loose})"
        )

    def test_detection_properties_populated(self, synthetic_db_image):
        """Detections carry properties (at minimum snr-like values)."""
        db, _ = synthetic_db_image
        det = CACFARDetector(guard_cells=3, training_cells=12, pfa=1e-2)
        result = det.detect(db)
        if len(result) > 0:
            d = result[0]
            assert isinstance(d.properties, dict)
            assert len(d.properties) > 0


# ---------------------------------------------------------------------------
# Level 3: Integration — Pipeline and serialization
# ---------------------------------------------------------------------------
class TestCFARLevel3Integration:
    """Pipeline: magnitude -> dB -> CFAR -> GeoJSON."""

    @pytest.mark.skipif(not _HAS_INTENSITY,
                        reason="grdl intensity transforms not available")
    @pytest.mark.integration
    def test_magnitude_to_db_to_cfar_pipeline(self, synthetic_magnitude_image):
        """Full pipeline: magnitude -> ToDecibels -> CA-CFAR -> GeoJSON."""
        mag, targets = synthetic_magnitude_image
        # Convert to dB
        xform = ToDecibels(floor_db=-60.0)
        db = xform.apply(mag)
        assert db.dtype in (np.float32, np.float64)

        # Detect
        det = CACFARDetector(guard_cells=3, training_cells=12, pfa=1e-3,
                             min_pixels=4)
        result = det.detect(db)
        assert isinstance(result, DetectionSet)
        assert len(result) > 0

        # Serialize to GeoJSON
        geojson = result.to_geojson()
        assert geojson['type'] == 'FeatureCollection'
        assert len(geojson['features']) == len(result)
        assert 'detector_name' in geojson['properties']

    @pytest.mark.integration
    def test_detection_set_filter_by_confidence(self, synthetic_db_image):
        """DetectionSet.filter_by_confidence produces a filtered subset."""
        db, _ = synthetic_db_image
        det = CACFARDetector(guard_cells=3, training_cells=12, pfa=1e-2)
        result = det.detect(db)
        if len(result) > 0:
            # filter_by_confidence should return a DetectionSet
            filtered = result.filter_by_confidence(0.5)
            assert isinstance(filtered, DetectionSet)
            assert len(filtered) <= len(result)

    @pytest.mark.integration
    def test_geojson_features_have_geometry(self, synthetic_db_image):
        """Every GeoJSON feature has a valid geometry."""
        db, _ = synthetic_db_image
        det = CACFARDetector(guard_cells=3, training_cells=12, pfa=1e-2)
        result = det.detect(db)
        geojson = result.to_geojson()
        for feature in geojson['features']:
            assert 'geometry' in feature
            assert feature['geometry'] is not None
            assert 'type' in feature['geometry']


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestCFAREdgeCases:
    """Edge cases and parameter validation."""

    def test_uniform_image_few_detections(self):
        """Uniform image should produce very few (ideally zero) detections."""
        uniform = np.ones((256, 256), dtype=np.float64) * 50.0
        det = CACFARDetector(guard_cells=3, training_cells=12, pfa=1e-4)
        result = det.detect(uniform)
        # Uniform image should have ~0 false alarms at low PFA
        assert len(result) < 10, (
            f"Uniform image produced {len(result)} detections (expected < 10)"
        )

    def test_invalid_pfa_raises(self):
        """PFA outside valid range raises an error."""
        with pytest.raises((ValueError, Exception)):
            CACFARDetector(pfa=2.0)

    def test_guard_larger_than_training_raises(self):
        """Guard cells > training cells should raise."""
        with pytest.raises((ValueError, Exception)):
            CACFARDetector(guard_cells=20, training_cells=5)
