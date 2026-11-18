# -*- coding: utf-8 -*-
"""
Spatial Truth Association & Optimal Bipartite Matching.

Associates predicted detections (from ``DetectionSet``, GeoJSON features, or
coordinate tuples) against labeled ground-truth targets using Hungarian optimal
assignment (``scipy.optimize.linear_sum_assignment``) within a spatial match gate.

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

_EARTH_RADIUS_M = 6371000.0
_UNMATCHED_COST = 1e12


@dataclass
class SpatialRecord:
    """Normalized spatial item (either prediction or ground-truth target)."""

    index: int
    coord_y: float  # latitude (wgs84) or row (pixel)
    coord_x: float  # longitude (wgs84) or col (pixel)
    confidence: float = 1.0
    label: str = ""
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class MatchPair:
    """A matched prediction ↔ ground-truth pair."""

    prediction: SpatialRecord
    truth: SpatialRecord
    distance: float  # meters (wgs84) or pixels (pixel)


@dataclass
class AssociationResult:
    """Output of bipartite spatial matching."""

    matches: list[MatchPair]
    unmatched_predictions: list[SpatialRecord]
    unmatched_truth: list[SpatialRecord]
    coordinate_system: str

    @property
    def tp(self) -> int:
        return len(self.matches)

    @property
    def fp(self) -> int:
        return len(self.unmatched_predictions)

    @property
    def fn(self) -> int:
        return len(self.unmatched_truth)


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in meters between two WGS84 points."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * _EARTH_RADIUS_M * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def _centroid_from_geojson_geometry(geom: dict[str, Any]) -> tuple[float, float]:
    """Extract (coord_x, coord_y) centroid from Point or Polygon GeoJSON geometry."""
    gtype = geom.get("type", "")
    coords = geom.get("coordinates", [])
    if gtype == "Point":
        return float(coords[0]), float(coords[1])
    if gtype == "Polygon" and coords:
        ring = np.asarray(coords[0], dtype=np.float64)
        return float(ring[:, 0].mean()), float(ring[:, 1].mean())
    if gtype == "MultiPolygon" and coords and coords[0]:
        ring = np.asarray(coords[0][0], dtype=np.float64)
        return float(ring[:, 0].mean()), float(ring[:, 1].mean())
    raise ValueError(f"Unsupported GeoJSON geometry type for point matching: {gtype!r}")


def extract_truth_records(
    ground_truth: dict[str, Any],
    *,
    label_property: str = "class",
) -> list[SpatialRecord]:
    """Parse a GeoJSON FeatureCollection into ``SpatialRecord`` instances."""
    features = ground_truth.get("features") or []
    records: list[SpatialRecord] = []
    for idx, feat in enumerate(features):
        geom = feat.get("geometry")
        if not geom:
            continue
        props = dict(feat.get("properties") or {})
        x, y = _centroid_from_geojson_geometry(geom)
        records.append(
            SpatialRecord(
                index=idx,
                coord_y=y,
                coord_x=x,
                confidence=1.0,
                label=str(props.get(label_property, "")),
                properties=props,
            )
        )
    return records


def extract_prediction_records(
    predictions: Any,
    *,
    coordinate_system: str = "wgs84",
    geolocation: Any | None = None,
) -> list[SpatialRecord]:
    """Normalize a ``DetectionSet``, GeoJSON dict, or sequence of dicts/tuples."""
    records: list[SpatialRecord] = []

    # Case 1: GeoJSON FeatureCollection
    if isinstance(predictions, dict) and predictions.get("type") == "FeatureCollection":
        for idx, feat in enumerate(predictions.get("features") or []):
            props = dict(feat.get("properties") or {})
            x, y = _centroid_from_geojson_geometry(feat["geometry"])
            records.append(
                SpatialRecord(
                    index=idx,
                    coord_y=y,
                    coord_x=x,
                    confidence=float(props.get("confidence", 1.0)),
                    label=str(props.get("label", props.get("class", ""))),
                    properties=props,
                )
            )
        return records

    # Case 2: GRDL DetectionSet (duck-typed via .detections or iterable)
    items = getattr(predictions, "detections", predictions)
    if items is None:
        return records

    for idx, det in enumerate(items):
        if isinstance(det, dict):
            props = dict(det)
            conf = float(det.get("confidence", 1.0))
            label = str(det.get("label", det.get("class", "")))
            if coordinate_system == "wgs84" and "lat" in det and "lon" in det:
                y, x = float(det["lat"]), float(det["lon"])
            elif "row" in det and "col" in det:
                y, x = float(det["row"]), float(det["col"])
                if coordinate_system == "wgs84" and geolocation is not None:
                    lat, lon, _ = geolocation.image_to_latlon(y, x)
                    y, x = float(lat), float(lon)
            else:
                raise ValueError(f"Cannot extract coordinates from prediction dict: {det}")
            records.append(
                SpatialRecord(index=idx, coord_y=y, coord_x=x, confidence=conf, label=label, properties=props)
            )
            continue

        if isinstance(det, (tuple, list)):
            y, x = float(det[0]), float(det[1])
            conf = float(det[2]) if len(det) > 2 else 1.0
            if coordinate_system == "wgs84" and geolocation is not None:
                lat, lon, _ = geolocation.image_to_latlon(y, x)
                y, x = float(lat), float(lon)
            records.append(SpatialRecord(index=idx, coord_y=y, coord_x=x, confidence=conf))
            continue

        # GRDL Detection object
        conf = float(getattr(det, "confidence", 1.0) or 1.0)
        props = dict(getattr(det, "properties", {}) or {})
        label = str(props.get("label", props.get("class", "")))
        geo_geom = getattr(det, "geo_geometry", None)
        pix_geom = getattr(det, "pixel_geometry", None)

        if coordinate_system == "wgs84" and geo_geom is not None:
            c = geo_geom.centroid
            # geo_geometry in shapely uses (lon, lat) = (x, y)
            x, y = float(c.x), float(c.y)
        elif pix_geom is not None:
            c = pix_geom.centroid
            row, col = float(c.y), float(c.x)
            if coordinate_system == "wgs84" and geolocation is not None:
                lat, lon, _ = geolocation.image_to_latlon(row, col)
                y, x = float(lat), float(lon)
            else:
                y, x = row, col
        else:
            raise ValueError(f"Detection {det!r} lacks usable geometry for {coordinate_system!r}")

        records.append(
            SpatialRecord(index=idx, coord_y=y, coord_x=x, confidence=conf, label=label, properties=props)
        )

    return records


def compute_pairwise_distance(
    preds: list[SpatialRecord],
    truths: list[SpatialRecord],
    *,
    coordinate_system: str = "wgs84",
) -> np.ndarray:
    """Compute ``(len(preds), len(truths))`` distance matrix."""
    n_p = len(preds)
    n_t = len(truths)
    dist = np.zeros((n_p, n_t), dtype=np.float64)
    if coordinate_system == "wgs84":
        for i, p in enumerate(preds):
            for j, t in enumerate(truths):
                dist[i, j] = haversine_distance_m(p.coord_y, p.coord_x, t.coord_y, t.coord_x)
    else:
        p_xy = np.array([[p.coord_y, p.coord_x] for p in preds], dtype=np.float64)
        t_xy = np.array([[t.coord_y, t.coord_x] for t in truths], dtype=np.float64)
        diff = p_xy[:, None, :] - t_xy[None, :, :]
        dist = np.hypot(diff[..., 0], diff[..., 1])
    return dist


def associate_detections(
    predictions: list[SpatialRecord],
    truths: list[SpatialRecord],
    *,
    max_distance: float,
    coordinate_system: str = "wgs84",
) -> AssociationResult:
    """Associate predictions to ground truth via optimal Hungarian matching."""
    if not predictions or not truths:
        return AssociationResult(
            matches=[],
            unmatched_predictions=list(predictions),
            unmatched_truth=list(truths),
            coordinate_system=coordinate_system,
        )

    dist = compute_pairwise_distance(predictions, truths, coordinate_system=coordinate_system)
    cost = np.where(dist <= max_distance, dist, _UNMATCHED_COST)
    row_ind, col_ind = linear_sum_assignment(cost)

    matches: list[MatchPair] = []
    matched_p: set[int] = set()
    matched_t: set[int] = set()

    for r, c in zip(row_ind, col_ind):
        if dist[r, c] <= max_distance:
            matches.append(
                MatchPair(prediction=predictions[r], truth=truths[c], distance=float(dist[r, c]))
            )
            matched_p.add(r)
            matched_t.add(c)

    unmatched_preds = [p for i, p in enumerate(predictions) if i not in matched_p]
    unmatched_truth = [t for j, t in enumerate(truths) if j not in matched_t]

    return AssociationResult(
        matches=matches,
        unmatched_predictions=unmatched_preds,
        unmatched_truth=unmatched_truth,
        coordinate_system=coordinate_system,
    )

