# -*- coding: utf-8 -*-
"""
Radiometric & Speckle Quality Fit-for-Purpose Evaluator.

Evaluates processed imagery against labeled homogeneous clutter, edge, target,
and background ROIs in pixel coordinates:
- Equivalent Number of Looks (ENL) over homogeneous regions
- Speckle Suppression Index (SSI) relative to raw input
- Mean radiometric preservation bias
- Edge Preservation Index (EPI) across labeled edge transects/boxes
- Contrast-to-Noise Ratio (CNR) between target and clutter ROIs
"""

from __future__ import annotations

from typing import Any

import numpy as np

from grdl_te.evaluation.evaluators.base import (
    EvaluationResult,
    FitForPurposeEvaluator,
    check_acceptance_thresholds,
)
from grdl_te.evaluation.manifest import EvaluationTask, ScenarioManifest


def _polygon_mask(shape: tuple[int, int], ring: list[list[float]]) -> np.ndarray:
    """Vectorized point-in-polygon rasterizer in pixel (x=col, y=row) coordinates."""
    h, w = shape
    xs = np.array([pt[0] for pt in ring], dtype=np.float64)
    ys = np.array([pt[1] for pt in ring], dtype=np.float64)

    min_c = max(0, int(np.floor(xs.min())))
    max_c = min(w - 1, int(np.ceil(xs.max())))
    min_r = max(0, int(np.floor(ys.min())))
    max_r = min(h - 1, int(np.ceil(ys.max())))

    mask = np.zeros(shape, dtype=bool)
    if max_c < min_c or max_r < min_r:
        return mask

    grid_r, grid_c = np.mgrid[min_r : max_r + 1, min_c : max_c + 1]
    px = grid_c.ravel() + 0.5
    py = grid_r.ravel() + 0.5

    inside = np.zeros(px.shape, dtype=bool)
    n = len(xs)
    j = n - 1
    for i in range(n):
        yi, yj = ys[i], ys[j]
        xi, xj = xs[i], xs[j]
        cond = (yi > py) != (yj > py)
        x_intersect = (xj - xi) * (py - yi) / ((yj - yi) + 1e-15) + xi
        inside ^= cond & (px < x_intersect)
        j = i

    mask[min_r : max_r + 1, min_c : max_c + 1] = inside.reshape(grid_r.shape)
    return mask


class RadiometricEvaluator(FitForPurposeEvaluator):
    """Evaluates speckle suppression, radiometric preservation, and edge fidelity."""

    task = EvaluationTask.RADIOMETRIC_QUALITY

    def evaluate(
        self,
        prediction: Any,
        ground_truth: dict[str, Any],
        manifest: ScenarioManifest,
        *,
        candidate_name: str = "candidate",
        raw_input: np.ndarray | None = None,
        **context: Any,
    ) -> EvaluationResult:
        out = np.abs(np.asarray(prediction, dtype=np.float64))
        if out.ndim == 3:
            out = out[0]
        raw = np.abs(np.asarray(raw_input, dtype=np.float64)) if raw_input is not None else None
        if raw is not None and raw.ndim == 3:
            raw = raw[0]

        features = ground_truth.get("features") or []
        homogeneous_masks: list[np.ndarray] = []
        edge_masks: list[np.ndarray] = []
        target_masks: list[np.ndarray] = []
        clutter_masks: list[np.ndarray] = []

        for feat in features:
            props = feat.get("properties") or {}
            roi_type = str(props.get("roi_type", props.get("class", "homogeneous"))).lower()
            geom = feat["geometry"]
            if geom["type"] != "Polygon":
                continue
            m = _polygon_mask(out.shape, geom["coordinates"][0])
            if not np.any(m):
                continue
            if roi_type == "homogeneous":
                homogeneous_masks.append(m)
            elif roi_type == "edge":
                edge_masks.append(m)
            elif roi_type == "target":
                target_masks.append(m)
            elif roi_type == "clutter":
                clutter_masks.append(m)

        metrics: dict[str, float] = {}

        if homogeneous_masks:
            enls: list[float] = []
            ssis: list[float] = []
            biases: list[float] = []
            for m in homogeneous_masks:
                vals = out[m]
                mu = float(vals.mean())
                var = float(vals.var())
                enls.append((mu ** 2 / var) if var > 1e-12 else float("inf"))
                if raw is not None:
                    rvals = raw[m]
                    r_mu = float(rvals.mean())
                    r_std = float(rvals.std())
                    if r_mu > 1e-12:
                        biases.append(abs(mu - r_mu) / r_mu)
                        cv_in = r_std / r_mu
                        cv_out = float(vals.std()) / max(mu, 1e-12)
                        if cv_in > 1e-12:
                            ssis.append(cv_out / cv_in)
            metrics["enl"] = float(np.mean(enls))
            if ssis:
                metrics["ssi"] = float(np.mean(ssis))
            if biases:
                metrics["mean_preservation_bias"] = float(np.mean(biases))

        if edge_masks and raw is not None:
            epis: list[float] = []
            gy_o, gx_o = np.gradient(out)
            gy_i, gx_i = np.gradient(raw)
            grad_out = np.hypot(gx_o, gy_o)
            grad_in = np.hypot(gx_i, gy_i)
            for m in edge_masks:
                denom = float(grad_in[m].sum())
                if denom > 1e-12:
                    epis.append(float(grad_out[m].sum() / denom))
            if epis:
                metrics["epi"] = float(np.mean(epis))

        if target_masks and clutter_masks:
            t_vals = np.concatenate([out[m] for m in target_masks])
            c_vals = np.concatenate([out[m] for m in clutter_masks])
            c_std = float(c_vals.std())
            if c_std > 1e-12:
                cnr = abs(float(t_vals.mean()) - float(c_vals.mean())) / c_std
                metrics["cnr_db"] = float(20.0 * np.log10(max(cnr, 1e-12)))

        passed, failures = check_acceptance_thresholds(metrics, manifest.acceptance_thresholds)

        return EvaluationResult(
            task=self.task,
            scenario_id=manifest.scenario_id,
            candidate_name=candidate_name,
            metrics=metrics,
            details={
                "num_homogeneous_rois": len(homogeneous_masks),
                "num_edge_rois": len(edge_masks),
                "num_target_rois": len(target_masks),
                "num_clutter_rois": len(clutter_masks),
            },
            passed=passed,
            threshold_failures=failures,
        )

