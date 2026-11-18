# -*- coding: utf-8 -*-
"""
Segmentation & Land-Cover Classification Fit-for-Purpose Evaluator.

Evaluates pixel-level classification masks (or thresholded spectral index maps)
against labeled GeoJSON polygon truth masks or reference label arrays:
- Overall pixel accuracy
- Mean Intersection-over-Union (IoU / Jaccard)
- Dice / F1 score
- Cohen's Kappa
- Per-class IoU & confusion matrix
"""

from __future__ import annotations

from typing import Any

import numpy as np

from grdl_te.evaluation.evaluators.base import (
    EvaluationResult,
    FitForPurposeEvaluator,
    check_acceptance_thresholds,
)
from grdl_te.evaluation.evaluators.radiometric import _polygon_mask
from grdl_te.evaluation.manifest import EvaluationTask, ScenarioManifest


class SegmentationEvaluator(FitForPurposeEvaluator):
    """Evaluates pixel masks against labeled polygon or raster ground truth."""

    task = EvaluationTask.SEGMENTATION

    def evaluate(
        self,
        prediction: Any,
        ground_truth: dict[str, Any],
        manifest: ScenarioManifest,
        *,
        candidate_name: str = "candidate",
        class_map: dict[str, int] | None = None,
        threshold: float | None = None,
        **context: Any,
    ) -> EvaluationResult:
        pred_arr = np.asarray(prediction)
        if pred_arr.ndim == 3:
            pred_arr = pred_arr[0]
        if threshold is not None:
            pred_mask = (pred_arr >= threshold).astype(np.int64)
        else:
            pred_mask = pred_arr.astype(np.int64)

        truth_mask = self._build_truth_mask(
            pred_mask.shape,
            ground_truth,
            label_property=manifest.ground_truth.label_property,
            class_map=class_map,
        )

        classes = sorted(set(np.unique(truth_mask).tolist()) | set(np.unique(pred_mask).tolist()))
        n_cls = len(classes)
        cls_to_idx = {c: i for i, c in enumerate(classes)}
        conf_mat = np.zeros((n_cls, n_cls), dtype=np.int64)
        for t_cls in classes:
            t_bool = truth_mask == t_cls
            for p_cls in classes:
                conf_mat[cls_to_idx[t_cls], cls_to_idx[p_cls]] = int(
                    np.sum(t_bool & (pred_mask == p_cls))
                )

        total = float(conf_mat.sum())
        overall_acc = float(np.trace(conf_mat) / total) if total > 0 else 0.0

        ious: dict[str, float] = {}
        dices: list[float] = []
        for c in classes:
            idx = cls_to_idx[c]
            tp = float(conf_mat[idx, idx])
            fp = float(conf_mat[:, idx].sum() - tp)
            fn = float(conf_mat[idx, :].sum() - tp)
            denom = tp + fp + fn
            iou = tp / denom if denom > 0 else 1.0
            ious[str(c)] = iou
            dice_denom = 2.0 * tp + fp + fn
            dices.append((2.0 * tp / dice_denom) if dice_denom > 0 else 1.0)

        # Positive-class IoU if binary (0, 1), plus mean IoU across all present classes
        mean_iou = float(np.mean(list(ious.values())))
        mean_dice = float(np.mean(dices))

        # Cohen's Kappa
        row_marg = conf_mat.sum(axis=1).astype(np.float64)
        col_marg = conf_mat.sum(axis=0).astype(np.float64)
        pe = float(np.sum(row_marg * col_marg) / (total ** 2)) if total > 0 else 0.0
        kappa = ((overall_acc - pe) / (1.0 - pe)) if (1.0 - pe) > 1e-12 else 1.0

        metrics: dict[str, float] = {
            "overall_accuracy": overall_acc,
            "mean_iou": mean_iou,
            "dice_f1": mean_dice,
            "cohens_kappa": float(kappa),
        }
        if "1" in ious:
            metrics["foreground_iou"] = ious["1"]

        passed, failures = check_acceptance_thresholds(metrics, manifest.acceptance_thresholds)

        return EvaluationResult(
            task=self.task,
            scenario_id=manifest.scenario_id,
            candidate_name=candidate_name,
            metrics=metrics,
            details={
                "classes": classes,
                "per_class_iou": ious,
                "confusion_matrix": conf_mat.tolist(),
            },
            passed=passed,
            threshold_failures=failures,
        )

    @staticmethod
    def _build_truth_mask(
        shape: tuple[int, int],
        ground_truth: dict[str, Any],
        *,
        label_property: str,
        class_map: dict[str, int] | None,
    ) -> np.ndarray:
        if "mask" in ground_truth:
            return np.asarray(ground_truth["mask"], dtype=np.int64)

        mask = np.zeros(shape, dtype=np.int64)
        inferred_map: dict[str, int] = dict(class_map or {})
        next_id = max(inferred_map.values(), default=0) + 1

        for feat in ground_truth.get("features") or []:
            geom = feat.get("geometry") or {}
            if geom.get("type") != "Polygon":
                continue
            props = feat.get("properties") or {}
            if "class_id" in props:
                cid = int(props["class_id"])
            else:
                label = str(props.get(label_property, "foreground"))
                if label not in inferred_map:
                    inferred_map[label] = next_id
                    next_id += 1
                cid = inferred_map[label]
            poly_mask = _polygon_mask(shape, geom["coordinates"][0])
            mask[poly_mask] = cid

        return mask

