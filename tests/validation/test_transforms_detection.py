# -*- coding: utf-8 -*-
"""
Detection Geometry Transform Tests - Vector coordinate transforms.

Tests transform_pixel_geometry, transform_detection, and
transform_detection_set using synthetic detections and known transforms.

- Level 1: Output types and structure
- Level 2: Known translation verification, attribute preservation

Dependencies
------------
pytest
numpy
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
2026-03-20

Modified
--------
2026-03-20
"""

# Third-party
import pytest
import numpy as np

try:
    from shapely.geometry import Point, box
    _HAS_SHAPELY = True
except ImportError:
    _HAS_SHAPELY = False

try:
    from grdl.transforms.detection import (
        transform_pixel_geometry,
        transform_detection,
        transform_detection_set,
    )
    _HAS_TRANSFORMS = True
except ImportError:
    _HAS_TRANSFORMS = False

try:
    from grdl.coregistration.base import RegistrationResult
    _HAS_COREG = True
except ImportError:
    _HAS_COREG = False

try:
    from grdl.image_processing.detection.models import Detection, DetectionSet
    _HAS_DETECTION = True
except ImportError:
    _HAS_DETECTION = False

pytestmark = [
    pytest.mark.transforms,
    pytest.mark.detection,
    pytest.mark.skipif(not _HAS_SHAPELY, reason="shapely not installed"),
    pytest.mark.skipif(not _HAS_TRANSFORMS, reason="grdl.transforms not available"),
    pytest.mark.skipif(not _HAS_COREG, reason="grdl.coregistration not available"),
    pytest.mark.skipif(not _HAS_DETECTION, reason="grdl.detection models not available"),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def translation_result():
    """RegistrationResult with a pure translation in (row, col) space.

    Matrix is (2, 3) operating on (row, col) coordinates:
        row_out = row_in + 10
        col_out = col_in + 5

    For shapely Point(x=col, y=row), this means:
        Point(col, row) → Point(col+5, row+10)
    """
    matrix = np.array([
        [1.0, 0.0, 10.0],  # row_out = row + 10
        [0.0, 1.0, 5.0],   # col_out = col + 5
    ], dtype=np.float64)
    return RegistrationResult(
        transform_matrix=matrix,
        residual_rms=0.0,
        num_matches=10,
        inlier_ratio=1.0,
        metadata={},
    )


@pytest.fixture(scope="module")
def sample_detection():
    """A Detection with a Point geometry and attributes."""
    geom = Point(100.0, 50.0)  # (col=100, row=50)
    return Detection(
        pixel_geometry=geom,
        properties={'label': 'vehicle', 'score': 0.95},
        confidence=0.95,
    )


@pytest.fixture(scope="module")
def sample_detection_set(sample_detection):
    """A DetectionSet with 3 detections."""
    detections = []
    for i in range(3):
        geom = Point(100.0 + i * 20, 50.0 + i * 10)
        det = Detection(
            pixel_geometry=geom,
            properties={'label': 'vehicle', 'index': i},
            confidence=0.8 + i * 0.05,
        )
        detections.append(det)
    return DetectionSet(
        detections=detections,
        detector_name='test_detector',
        detector_version='0.1.0',
    )


# =============================================================================
# Level 1: Format Validation
# =============================================================================


class TestTransformsLevel1:
    """Validate output types and structure."""

    def test_transform_pixel_geometry_point(self, translation_result):
        """transform_pixel_geometry on a Point returns a Point."""
        pt = Point(100.0, 50.0)
        result = transform_pixel_geometry(pt, translation_result)
        assert result.geom_type == 'Point', (
            f"Expected Point, got {result.geom_type}"
        )

    def test_transform_pixel_geometry_polygon(self, translation_result):
        """transform_pixel_geometry on a box Polygon returns a Polygon."""
        bbox = box(10, 20, 50, 60)  # (minx, miny, maxx, maxy)
        result = transform_pixel_geometry(bbox, translation_result)
        assert result.geom_type == 'Polygon', (
            f"Expected Polygon, got {result.geom_type}"
        )

    def test_transform_detection_returns_detection(
        self, sample_detection, translation_result
    ):
        """transform_detection returns a Detection instance."""
        result = transform_detection(sample_detection, translation_result)
        assert isinstance(result, Detection), (
            f"Expected Detection, got {type(result).__name__}"
        )

    def test_transform_detection_set_returns_set(
        self, sample_detection_set, translation_result
    ):
        """transform_detection_set returns a DetectionSet."""
        result = transform_detection_set(sample_detection_set, translation_result)
        assert isinstance(result, DetectionSet), (
            f"Expected DetectionSet, got {type(result).__name__}"
        )
        assert len(result) == len(sample_detection_set), (
            f"Detection count changed: {len(result)} != {len(sample_detection_set)}"
        )


# =============================================================================
# Level 2: Data Quality — Correct transforms
# =============================================================================


class TestTransformsLevel2:
    """Validate correct coordinate transformation."""

    def test_transform_known_translation(self, translation_result):
        """Translation in (row, col) space applied correctly to Point.

        Input Point(x=100, y=50) means col=100, row=50.
        Matrix adds 10 to row, 5 to col → Point(105, 60).
        """
        pt = Point(100.0, 50.0)
        result = transform_pixel_geometry(pt, translation_result)
        # row=50+10=60, col=100+5=105 → Point(x=105, y=60)
        assert abs(result.x - 105.0) < 0.1, (
            f"Translated col: x={result.x:.2f}, expected 105.0"
        )
        assert abs(result.y - 60.0) < 0.1, (
            f"Translated row: y={result.y:.2f}, expected 60.0"
        )

    def test_transform_preserves_attributes(
        self, sample_detection, translation_result
    ):
        """transform_detection preserves properties and confidence."""
        result = transform_detection(sample_detection, translation_result)
        assert result.properties['label'] == 'vehicle', (
            "label attribute not preserved"
        )
        assert result.properties['score'] == 0.95, (
            "score attribute not preserved"
        )
        assert result.confidence == 0.95, (
            f"confidence changed: {result.confidence} != 0.95"
        )

    def test_transform_bbox_refit(self, translation_result):
        """bbox_mode='refit' produces correctly translated bounding box.

        box(10, 20, 50, 60) = col∈[10,50], row∈[20,60].
        After row+10, col+5 → col∈[15,55], row∈[30,70].
        """
        bbox = box(10, 20, 50, 60)
        result = transform_pixel_geometry(
            bbox, translation_result, bbox_mode='refit'
        )
        bounds = result.bounds  # (minx, miny, maxx, maxy)
        assert abs(bounds[0] - 15.0) < 0.1, f"mincol: minx={bounds[0]}, expected 15"
        assert abs(bounds[1] - 30.0) < 0.1, f"minrow: miny={bounds[1]}, expected 30"
        assert abs(bounds[2] - 55.0) < 0.1, f"maxcol: maxx={bounds[2]}, expected 55"
        assert abs(bounds[3] - 70.0) < 0.1, f"maxrow: maxy={bounds[3]}, expected 70"
