# -*- coding: utf-8 -*-
"""
Geolocation Fit-for-Purpose Evaluator.

Measures sensor geolocation accuracy against surveyed Ground Control Points
(GCPs): horizontal RMSE, CE50, CE90, directional North/East bias, and
vertical LE90 / RMSE.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from grdl_te.evaluation.association import haversine_distance_m
from grdl_te.evaluation.evaluators.base import (
    EvaluationResult,
    FitForPurposeEvaluator,
    check_acceptance_thresholds,
)
from grdl_te.evaluation.manifest import EvaluationTask, ScenarioManifest

_METERS_PER_DEG_LAT = 111320.0


class GeolocationEvaluator(FitForPurposeEvaluator):
    """Evaluates sensor geolocation models or predicted coordinates vs. surveyed GCPs."""

    task = EvaluationTask.GEOLOCATION

    def evaluate(
        self,
        prediction: Any,
        ground_truth: dict[str, Any],
        manifest: ScenarioManifest,
        *,
        candidate_name: str = "candidate",
        **context: Any,
    ) -> EvaluationResult:
        features = ground_truth.get("features") or []
        if not features:
            raise ValueError(f"Scenario '{manifest.scenario_id}' ground truth has no GCP features")

        gcp_rows: list[float] = []
        gcp_cols: list[float] = []
        true_lats: list[float] = []
        true_lons: list[float] = []
        true_haes: list[float] = []

        for feat in features:
            props = feat.get("properties") or {}
            coords = feat["geometry"]["coordinates"]
            true_lons.append(float(coords[0]))
            true_lats.append(float(coords[1]))
            true_haes.append(float(props.get("hae", coords[2] if len(coords) > 2 else 0.0)))
            gcp_rows.append(float(props["pixel_row"]))
            gcp_cols.append(float(props["pixel_col"]))

        pred_lats, pred_lons, pred_haes = self._resolve_predictions(prediction, gcp_rows, gcp_cols)

        horiz_errors_m: list[float] = []
        north_biases_m: list[float] = []
        east_biases_m: list[float] = []
        vert_errors_m: list[float] = []

        for p_lat, p_lon, p_hae, t_lat, t_lon, t_hae in zip(
            pred_lats, pred_lons, pred_haes, true_lats, true_lons, true_haes
        ):
            horiz_errors_m.append(haversine_distance_m(p_lat, p_lon, t_lat, t_lon))
            d_north = (p_lat - t_lat) * _METERS_PER_DEG_LAT
            d_east = (p_lon - t_lon) * _METERS_PER_DEG_LAT * math.cos(math.radians(t_lat))
            north_biases_m.append(d_north)
            east_biases_m.append(d_east)
            vert_errors_m.append(abs(p_hae - t_hae))

        herr = np.asarray(horiz_errors_m, dtype=np.float64)
        verr = np.asarray(vert_errors_m, dtype=np.float64)

        metrics: dict[str, float] = {
            "num_gcps": float(len(features)),
            "horizontal_rmse_m": float(np.sqrt(np.mean(herr ** 2))),
            "ce50_m": float(np.percentile(herr, 50)),
            "ce90_m": float(np.percentile(herr, 90)),
            "max_horizontal_error_m": float(np.max(herr)),
            "mean_north_bias_m": float(np.mean(north_biases_m)),
            "mean_east_bias_m": float(np.mean(east_biases_m)),
            "vertical_rmse_m": float(np.sqrt(np.mean(verr ** 2))),
            "le90_m": float(np.percentile(verr, 90)),
        }

        passed, failures = check_acceptance_thresholds(metrics, manifest.acceptance_thresholds)

        return EvaluationResult(
            task=self.task,
            scenario_id=manifest.scenario_id,
            candidate_name=candidate_name,
            metrics=metrics,
            details={"horizontal_errors_m": horiz_errors_m, "vertical_errors_m": vert_errors_m},
            passed=passed,
            threshold_failures=failures,
        )

    @staticmethod
    def _resolve_predictions(
        prediction: Any,
        rows: list[float],
        cols: list[float],
    ) -> tuple[list[float], list[float], list[float]]:
        if hasattr(prediction, "image_to_latlon"):
            lats: list[float] = []
            lons: list[float] = []
            haes: list[float] = []
            for r, c in zip(rows, cols):
                lat, lon, hae = prediction.image_to_latlon(r, c)
                lats.append(float(lat))
                lons.append(float(lon))
                haes.append(float(hae))
            return lats, lons, haes

        if isinstance(prediction, dict) and "lat" in prediction and "lon" in prediction:
            haes = prediction.get("hae") or [0.0] * len(prediction["lat"])
            return [float(x) for x in prediction["lat"]], [float(x) for x in prediction["lon"]], [float(x) for x in haes]

        if isinstance(prediction, (list, tuple)):
            lats = [float(p[0]) for p in prediction]
            lons = [float(p[1]) for p in prediction]
            haes = [float(p[2]) if len(p) > 2 else 0.0 for p in prediction]
            return lats, lons, haes

        raise TypeError(
            "GeolocationEvaluator expects a Geolocation instance (image_to_latlon), "
            "a dict with 'lat'/'lon', or a sequence of (lat, lon[, hae]) tuples."
        )

