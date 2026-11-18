# -*- coding: utf-8 -*-
"""Fit-for-Purpose task evaluators."""

from grdl_te.evaluation.evaluators.base import (
    EvaluationResult,
    FitForPurposeEvaluator,
    check_acceptance_thresholds,
)
from grdl_te.evaluation.evaluators.detection import DetectionEvaluator
from grdl_te.evaluation.evaluators.geolocation import GeolocationEvaluator
from grdl_te.evaluation.evaluators.radiometric import RadiometricEvaluator
from grdl_te.evaluation.evaluators.segmentation import SegmentationEvaluator

__all__ = [
    "EvaluationResult",
    "FitForPurposeEvaluator",
    "check_acceptance_thresholds",
    "DetectionEvaluator",
    "GeolocationEvaluator",
    "RadiometricEvaluator",
    "SegmentationEvaluator",
]

