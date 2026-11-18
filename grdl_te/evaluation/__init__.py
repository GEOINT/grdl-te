# -*- coding: utf-8 -*-
"""
GRDL-TE Fit-for-Purpose Evaluation Framework.

Provides scenario manifests indexed by modality, region, and detectable
signature type, spatial truth association, task-specific quantitative
evaluators, and apples-to-apples comparison scorecards.
"""

from grdl_te.evaluation.association import (
    AssociationResult,
    MatchPair,
    SpatialRecord,
    associate_detections,
    haversine_distance_m,
)
from grdl_te.evaluation.evaluators import (
    DetectionEvaluator,
    EvaluationResult,
    FitForPurposeEvaluator,
    GeolocationEvaluator,
    RadiometricEvaluator,
    SegmentationEvaluator,
)
from grdl_te.evaluation.manifest import (
    EvaluationTask,
    GroundTruthSpec,
    ScenarioManifest,
    ScenarioRegistry,
    SensorDatasetSpec,
)
from grdl_te.evaluation.runner import EvaluationCandidate, EvaluationRunner
from grdl_te.evaluation.scorecard import CandidateScore, EvaluationScorecard

__all__ = [
    "EvaluationTask",
    "GroundTruthSpec",
    "SensorDatasetSpec",
    "ScenarioManifest",
    "ScenarioRegistry",
    "SpatialRecord",
    "MatchPair",
    "AssociationResult",
    "associate_detections",
    "haversine_distance_m",
    "EvaluationResult",
    "FitForPurposeEvaluator",
    "DetectionEvaluator",
    "GeolocationEvaluator",
    "RadiometricEvaluator",
    "SegmentationEvaluator",
    "EvaluationCandidate",
    "EvaluationRunner",
    "CandidateScore",
    "EvaluationScorecard",
]

