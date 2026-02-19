# -*- coding: utf-8 -*-
"""
Detection Data Model Tests - Detection, DetectionSet, Fields, FieldDefinition.

Validates the detection data models using synthetic in-memory objects.
No real data or heavy processing required.

Dependencies
------------
pytest
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

try:
    from shapely.geometry import Point, box
    _HAS_SHAPELY = True
except ImportError:
    _HAS_SHAPELY = False

try:
    from grdl.image_processing.detection import (
        Detection,
        DetectionSet,
        FieldDefinition,
        Fields,
        DATA_DICTIONARY,
        lookup_field,
        is_dictionary_field,
        list_fields,
    )
    _HAS_DETECTION = True
except ImportError:
    _HAS_DETECTION = False

pytestmark = [
    pytest.mark.detection,
    pytest.mark.skipif(not _HAS_DETECTION,
                       reason="grdl detection models not available"),
    pytest.mark.skipif(not _HAS_SHAPELY,
                       reason="shapely not installed"),
]


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
class TestDetection:
    """Tests for the Detection data model."""

    def test_construction_with_point(self):
        """Detection can be constructed with a shapely Point."""
        d = Detection(
            pixel_geometry=Point(100, 200),
            properties={"snr": 15.3},
            confidence=0.95,
        )
        assert d.pixel_geometry.geom_type == 'Point'
        assert d.properties['snr'] == 15.3
        assert d.confidence == 0.95
        assert d.geo_geometry is None

    def test_construction_with_bbox(self):
        """Detection can be constructed with a bounding box."""
        d = Detection(
            pixel_geometry=box(10, 20, 50, 80),
            properties={"label": "vehicle"},
        )
        assert d.pixel_geometry.geom_type == 'Polygon'
        assert d.confidence is None

    def test_to_geojson_feature(self):
        """to_geojson_feature() produces valid GeoJSON Feature dict."""
        d = Detection(
            pixel_geometry=Point(100, 200),
            properties={"snr": 15.3},
            confidence=0.95,
        )
        feature = d.to_geojson_feature()
        assert feature['type'] == 'Feature'
        assert 'geometry' in feature
        assert feature['geometry']['type'] == 'Point'
        assert 'properties' in feature
        assert feature['properties']['snr'] == 15.3
        assert feature['properties']['confidence'] == 0.95

    def test_to_geojson_uses_geo_geometry_when_set(self):
        """to_geojson_feature() prefers geo_geometry over pixel_geometry."""
        d = Detection(
            pixel_geometry=Point(100, 200),
            properties={},
            geo_geometry=Point(-118.25, 34.05),
        )
        feature = d.to_geojson_feature()
        coords = feature['geometry']['coordinates']
        assert abs(coords[0] - (-118.25)) < 1e-6
        assert abs(coords[1] - 34.05) < 1e-6

    def test_repr(self):
        """Detection has a readable repr."""
        d = Detection(
            pixel_geometry=Point(100, 200),
            properties={},
            confidence=0.9,
        )
        r = repr(d)
        assert 'Point' in r
        assert '0.9' in r


# ---------------------------------------------------------------------------
# DetectionSet
# ---------------------------------------------------------------------------
class TestDetectionSet:
    """Tests for the DetectionSet collection."""

    def _make_detections(self, n=5):
        return [
            Detection(
                pixel_geometry=Point(i * 100, i * 50),
                properties={"snr": float(i + 10)},
                confidence=0.5 + 0.1 * i,
            )
            for i in range(n)
        ]

    def test_length(self):
        """len(DetectionSet) matches number of detections."""
        ds = DetectionSet(
            detections=self._make_detections(3),
            detector_name="TestDetector",
            detector_version="1.0.0",
        )
        assert len(ds) == 3

    def test_iteration(self):
        """DetectionSet is iterable."""
        dets = self._make_detections(4)
        ds = DetectionSet(
            detections=dets,
            detector_name="TestDetector",
            detector_version="1.0.0",
        )
        collected = list(ds)
        assert len(collected) == 4
        assert all(isinstance(d, Detection) for d in collected)

    def test_indexing(self):
        """DetectionSet supports integer indexing."""
        dets = self._make_detections(3)
        ds = DetectionSet(
            detections=dets,
            detector_name="TestDetector",
            detector_version="1.0.0",
        )
        assert ds[0] is dets[0]
        assert ds[2] is dets[2]

    def test_to_geojson(self):
        """to_geojson() returns valid GeoJSON FeatureCollection."""
        ds = DetectionSet(
            detections=self._make_detections(3),
            detector_name="CA-CFAR",
            detector_version="1.0.0",
        )
        geojson = ds.to_geojson()
        assert geojson['type'] == 'FeatureCollection'
        assert len(geojson['features']) == 3
        assert geojson['properties']['detector_name'] == 'CA-CFAR'
        assert geojson['properties']['detector_version'] == '1.0.0'

    def test_filter_by_confidence(self):
        """filter_by_confidence returns subset above threshold."""
        dets = self._make_detections(5)
        # confidences: 0.5, 0.6, 0.7, 0.8, 0.9
        ds = DetectionSet(
            detections=dets,
            detector_name="Test",
            detector_version="1.0.0",
        )
        filtered = ds.filter_by_confidence(0.75)
        assert isinstance(filtered, DetectionSet)
        assert len(filtered) == 2  # 0.8 and 0.9

    def test_empty_detection_set(self):
        """Empty DetectionSet is valid."""
        ds = DetectionSet(
            detections=[],
            detector_name="Empty",
            detector_version="0.0.0",
        )
        assert len(ds) == 0
        geojson = ds.to_geojson()
        assert geojson['type'] == 'FeatureCollection'
        assert len(geojson['features']) == 0

    def test_repr(self):
        """DetectionSet has a readable repr."""
        ds = DetectionSet(
            detections=self._make_detections(3),
            detector_name="CA-CFAR",
            detector_version="1.0.0",
        )
        r = repr(ds)
        assert 'CA-CFAR' in r
        assert '3' in r


# ---------------------------------------------------------------------------
# Fields and FieldDefinition
# ---------------------------------------------------------------------------
class TestFieldsAndDictionary:
    """Tests for Fields, FieldDefinition, and DATA_DICTIONARY."""

    def test_field_definition_construction(self):
        """FieldDefinition can be constructed and has expected attributes."""
        fd = FieldDefinition(
            name='sar.sigma0',
            dtype='float',
            description='Sigma naught',
            units='dB',
        )
        assert fd.name == 'sar.sigma0'
        assert fd.dtype == 'float'
        assert fd.units == 'dB'
        assert fd.domain == 'sar'

    def test_field_definition_domain_parsing(self):
        """Domain is extracted from dotted name."""
        fd = FieldDefinition(
            name='physical.area',
            dtype='float',
            description='Physical area',
            units='m^2',
        )
        assert fd.domain == 'physical'

    def test_data_dictionary_populated(self):
        """DATA_DICTIONARY contains entries."""
        assert isinstance(DATA_DICTIONARY, dict)
        assert len(DATA_DICTIONARY) > 20

    def test_lookup_field(self):
        """lookup_field returns a FieldDefinition for known fields."""
        fd = lookup_field('sar.sigma0')
        assert isinstance(fd, FieldDefinition)
        assert fd.name == 'sar.sigma0'

    def test_is_dictionary_field(self):
        """is_dictionary_field correctly identifies known fields."""
        assert is_dictionary_field('sar.sigma0') is True
        assert is_dictionary_field('nonexistent.field') is False

    def test_list_fields(self):
        """list_fields returns all field names."""
        fields = list_fields()
        assert isinstance(fields, list)
        assert len(fields) > 20
        field_names = [f.name for f in fields]
        assert 'sar.sigma0' in field_names

    def test_fields_accessor_class(self):
        """Fields provides IDE-friendly attribute access."""
        # Test that Fields has domain sub-attributes
        assert hasattr(Fields, 'sar')
        assert hasattr(Fields, 'physical')
        assert hasattr(Fields, 'identity')

    def test_field_definition_repr(self):
        """FieldDefinition repr is readable."""
        fd = FieldDefinition(
            name='physical.area',
            dtype='float',
            description='Physical area',
            units='m^2',
        )
        r = repr(fd)
        assert 'physical.area' in r
        assert 'float' in r
