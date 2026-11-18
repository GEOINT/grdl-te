# -*- coding: utf-8 -*-
"""
Base types for Fit-for-Purpose quantitative evaluators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from grdl_te.evaluation.manifest import EvaluationTask, ScenarioManifest


@dataclass
class EvaluationResult:
    """Quantitative effectiveness output from a Fit-for-Purpose evaluator."""

    task: EvaluationTask
    scenario_id: str
    candidate_name: str
    metrics: dict[str, float]
    details: dict[str, Any] = field(default_factory=dict)
    passed: bool | None = None
    threshold_failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.value,
            "scenario_id": self.scenario_id,
            "candidate_name": self.candidate_name,
            "metrics": dict(self.metrics),
            "details": dict(self.details),
            "passed": self.passed,
            "threshold_failures": list(self.threshold_failures),
        }


def check_acceptance_thresholds(
    metrics: dict[str, float],
    thresholds: dict[str, float],
) -> tuple[bool | None, list[str]]:
    """Evaluate scenario acceptance thresholds against computed metrics.

    Threshold keys follow ``min_<metric>`` or ``max_<metric>`` conventions
    (e.g. ``min_pd: 0.9``, ``max_far_per_km2: 0.5``, ``max_ce90_m: 10.0``).
    Exact metric name keys default to minimum required value.
    """
    if not thresholds:
        return None, []

    failures: list[str] = []
    for key, limit in thresholds.items():
        if key.startswith("min_"):
            metric_name = key[4:]
            val = metrics.get(metric_name)
            if val is None or val < limit:
                failures.append(f"{metric_name}={val} < min {limit}")
        elif key.startswith("max_"):
            metric_name = key[4:]
            val = metrics.get(metric_name)
            if val is None or val > limit:
                failures.append(f"{metric_name}={val} > max {limit}")
        else:
            val = metrics.get(key)
            if val is None or val < limit:
                failures.append(f"{key}={val} < min {limit}")

    return len(failures) == 0, failures


class FitForPurposeEvaluator(ABC):
    """Abstract base class for scenario-driven quantitative evaluators."""

    task: EvaluationTask

    @abstractmethod
    def evaluate(
        self,
        prediction: Any,
        ground_truth: dict[str, Any],
        manifest: ScenarioManifest,
        *,
        candidate_name: str = "candidate",
        **context: Any,
    ) -> EvaluationResult:
        """Score *prediction* against *ground_truth* for *manifest*."""
