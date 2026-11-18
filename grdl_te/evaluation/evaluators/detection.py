# -*- coding: utf-8 -*-
"""
Detection Fit-for-Purpose Evaluator.

Computes operational detection figures of merit:
- Probability of Detection ($P_d$ / Recall)
- Precision and $F_1$ score
- False Alarm Rate per $\\text{km}^2$ ($\\text{FAR}/\\text{km}^2$) & empirical $P_{fa}$
- Spatial Localization Error (Mean, $\\text{CEP}_{50}$, $\\text{CEP}_{90}$)
- Confidence-sweep ROC curve and AUC
"""

from __future__ import annotations

from typing import Any

import numpy as np

from grdl_te.evaluation.association import (
    associate_detections,
    extract_prediction_records,
    extract_truth_records,
)
from grdl_te.evaluation.evaluators.base import (
    EvaluationResult,
    FitForPurposeEvaluator,
    check_acceptance_thresholds,
)
from grdl_te.evaluation.manifest import EvaluationTask, ScenarioManifest


class DetectionEvaluator(FitForPurposeEvaluator):
    """Quantitative detection evaluator against labeled point/polygon targets."""

    task = EvaluationTask.DETECTION

    def evaluate(
        self,
        prediction: Any,
        ground_truth: dict[str, Any],
        manifest: ScenarioManifest,
        *,
        candidate_name: str = "candidate",
        geolocation: Any | None = None,
        **context: Any,
    ) -> EvaluationResult:
        gt_spec = manifest.ground_truth
        coord_sys = gt_spec.coordinate_system
        max_dist = gt_spec.match_radius_m if coord_sys == "wgs84" else gt_spec.match_radius_px

        truths = extract_truth_records(ground_truth, label_property=gt_spec.label_property)
        preds = extract_prediction_records(
            prediction,
            coordinate_system=coord_sys,
            geolocation=geolocation,
        )

        assoc = associate_detections(
            preds,
            truths,
            max_distance=max_dist,
            coordinate_system=coord_sys,
        )

        tp, fp, fn = assoc.tp, assoc.fp, assoc.fn
        n_truth = len(truths)
        pd = tp / n_truth if n_truth > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2.0 * precision * pd / (precision + pd)) if (precision + pd) > 0 else 0.0

        metrics: dict[str, float] = {
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
            "pd": float(pd),
            "precision": float(precision),
            "f1": float(f1),
        }

        area_km2 = manifest.dataset.area_km2
        if area_km2 and area_km2 > 0:
            metrics["far_per_km2"] = float(fp / area_km2)

        if manifest.dataset.image_shape:
            n_pixels = manifest.dataset.image_shape[0] * manifest.dataset.image_shape[1]
            if n_pixels > 0:
                metrics["pfa_empirical"] = float(fp / n_pixels)

        distances = np.array([m.distance for m in assoc.matches], dtype=np.float64)
        suffix = "m" if coord_sys == "wgs84" else "px"
        if len(distances) > 0:
            metrics[f"mean_loc_error_{suffix}"] = float(distances.mean())
            metrics[f"cep50_{suffix}"] = float(np.percentile(distances, 50))
            metrics[f"cep90_{suffix}"] = float(np.percentile(distances, 90))
        else:
            metrics[f"mean_loc_error_{suffix}"] = float("nan")
            metrics[f"cep50_{suffix}"] = float("nan")
            metrics[f"cep90_{suffix}"] = float("nan")

        roc_points = self._compute_roc_sweep(preds, truths, max_dist, coord_sys, area_km2)
        if len(roc_points) >= 2:
            # Compute trapezoidal AUC over normalized FP rate [0, 1] vs Pd
            xs = np.array([pt["fp_norm"] for pt in roc_points], dtype=np.float64)
            ys = np.array([pt["pd"] for pt in roc_points], dtype=np.float64)
            order = np.argsort(xs)
            metrics["roc_auc"] = float(np.trapezoid(ys[order], xs[order]))

        passed, failures = check_acceptance_thresholds(metrics, manifest.acceptance_thresholds)

        return EvaluationResult(
            task=self.task,
            scenario_id=manifest.scenario_id,
            candidate_name=candidate_name,
            metrics=metrics,
            details={
                "coordinate_system": coord_sys,
                "match_gate": max_dist,
                "num_truth": n_truth,
                "num_predictions": len(preds),
                "roc_curve": roc_points,
            },
            passed=passed,
            threshold_failures=failures,
        )

    @staticmethod
    def _compute_roc_sweep(
        preds: list[Any],
        truths: list[Any],
        max_dist: float,
        coord_sys: str,
        area_km2: float | None,
    ) -> list[dict[str, float]]:
        if not preds or not truths:
            return []
        confs = sorted({p.confidence for p in preds}, reverse=True)
        if len(confs) < 2:
            return []

        max_fp = max(1, len(preds))
        points: list[dict[str, float]] = [{"threshold": confs[0] + 1e-6, "pd": 0.0, "fp": 0.0, "fp_norm": 0.0}]
        for thresh in confs:
            subset = [p for p in preds if p.confidence >= thresh]
            res = associate_detections(subset, truths, max_distance=max_dist, coordinate_system=coord_sys)
            pd = res.tp / len(truths)
            pt = {
                "threshold": float(thresh),
                "pd": float(pd),
                "fp": float(res.fp),
                "fp_norm": float(res.fp / max_fp),
            }
            if area_km2 and area_km2 > 0:
                pt["far_per_km2"] = float(res.fp / area_km2)
            points.append(pt)
        points.append({"threshold": 0.0, "pd": points[-1]["pd"], "fp": float(max_fp), "fp_norm": 1.0})
        return points

