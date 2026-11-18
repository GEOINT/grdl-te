# -*- coding: utf-8 -*-
"""
Apples-to-Apples Fit-for-Purpose Evaluation Runner.

Executes one or more candidate components (``ImageProcessor`` instances,
``grdl_rt.Workflow`` definitions, or callables) against a ``ScenarioManifest``
and its labeled ground-truth dataset, recording both effectiveness metrics and
execution performance.
"""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from grdl_te.evaluation.evaluators import (
    DetectionEvaluator,
    FitForPurposeEvaluator,
    GeolocationEvaluator,
    RadiometricEvaluator,
    SegmentationEvaluator,
)
from grdl_te.evaluation.manifest import EvaluationTask, ScenarioManifest
from grdl_te.evaluation.scorecard import CandidateScore, EvaluationScorecard

_EVALUATOR_BY_TASK: dict[EvaluationTask, type[FitForPurposeEvaluator]] = {
    EvaluationTask.DETECTION: DetectionEvaluator,
    EvaluationTask.GEOLOCATION: GeolocationEvaluator,
    EvaluationTask.RADIOMETRIC_QUALITY: RadiometricEvaluator,
    EvaluationTask.SEGMENTATION: SegmentationEvaluator,
}


def _rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        return 0.0


@dataclass
class EvaluationCandidate:
    """A component, workflow, or callable to evaluate on a scenario."""

    name: str
    target: Any
    kwargs: dict[str, Any] = field(default_factory=dict)
    postprocess: Callable[[Any], Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class EvaluationRunner:
    """Runs apples-to-apples Fit-for-Purpose + Performance evaluations."""

    def __init__(
        self,
        evaluators: dict[EvaluationTask, FitForPurposeEvaluator] | None = None,
    ) -> None:
        self._evaluators: dict[EvaluationTask, FitForPurposeEvaluator] = {
            task: cls() for task, cls in _EVALUATOR_BY_TASK.items()
        }
        if evaluators:
            self._evaluators.update(evaluators)

    def evaluate_scenario(
        self,
        manifest: ScenarioManifest,
        candidates: list[EvaluationCandidate],
        *,
        source_data: Any | None = None,
        metadata: Any | None = None,
        geolocation: Any | None = None,
        evaluator_context: dict[str, Any] | None = None,
    ) -> EvaluationScorecard:
        """Execute all *candidates* on *manifest* and return an ``EvaluationScorecard``."""
        ground_truth = manifest.load_ground_truth()
        evaluator = self._evaluators[manifest.evaluation_task]

        if source_data is None and manifest.dataset.path is not None:
            source_data, loaded_meta = self._load_dataset(manifest)
            if metadata is None:
                metadata = loaded_meta

        ctx: dict[str, Any] = dict(evaluator_context or {})
        if geolocation is not None:
            ctx.setdefault("geolocation", geolocation)
        if manifest.evaluation_task == EvaluationTask.RADIOMETRIC_QUALITY and source_data is not None:
            ctx.setdefault("raw_input", source_data)

        scores: list[CandidateScore] = []
        for cand in candidates:
            rss_before = _rss_mb()
            t0 = time.perf_counter()
            raw_pred = self._invoke_candidate(
                cand, source_data=source_data, metadata=metadata, geolocation=geolocation
            )
            elapsed = time.perf_counter() - t0
            rss_after = _rss_mb()

            pred = cand.postprocess(raw_pred) if cand.postprocess is not None else raw_pred
            eval_res = evaluator.evaluate(
                pred,
                ground_truth,
                manifest,
                candidate_name=cand.name,
                **ctx,
            )
            scores.append(
                CandidateScore(
                    candidate_name=cand.name,
                    evaluation=eval_res,
                    wall_time_s=elapsed,
                    memory_delta_mb=max(0.0, rss_after - rss_before),
                    metadata=dict(cand.metadata),
                )
            )

        return EvaluationScorecard(scenario=manifest, scores=scores)

    @staticmethod
    def _invoke_candidate(
        cand: EvaluationCandidate,
        *,
        source_data: Any,
        metadata: Any,
        geolocation: Any,
    ) -> Any:
        target = cand.target
        kw = dict(cand.kwargs)

        # 1. ImageDetector (.detect)
        if hasattr(target, "detect") and callable(target.detect):
            if geolocation is not None and "geolocation" not in kw:
                kw["geolocation"] = geolocation
            return target.detect(source_data, **kw)

        # 2. ImageProcessor (.execute)
        if hasattr(target, "execute") and callable(target.execute) and metadata is not None:
            out, _ = target.execute(metadata, source_data, **kw)
            return out

        # 3. ImageTransform (.apply)
        if hasattr(target, "apply") and callable(target.apply):
            return target.apply(source_data, **kw)

        # 4. grdl_rt.Workflow (.execute)
        if hasattr(target, "execute") and callable(target.execute):
            result = target.execute(source_data, **kw)
            return getattr(result, "result", result)

        # 5. Geolocation object directly
        if hasattr(target, "image_to_latlon") and source_data is None:
            return target

        # 6. Plain callable
        if callable(target):
            return target(source_data, **kw) if source_data is not None else target(**kw)

        return target

    @staticmethod
    def _load_dataset(manifest: ScenarioManifest) -> tuple[Any, Any]:
        ds = manifest.dataset
        if ds.path is None or not ds.path.exists():
            raise FileNotFoundError(
                f"Scenario '{manifest.scenario_id}' dataset not found: {ds.path}"
            )
        if ds.reader_class:
            mod_path, _, cls_name = ds.reader_class.rpartition(".")
            reader_cls = getattr(importlib.import_module(mod_path), cls_name)
            with reader_cls(str(ds.path)) as reader:
                return reader.read_full(), getattr(reader, "metadata", None)

        from grdl.IO.generic import open_any

        with open_any(str(ds.path)) as reader:
            return reader.read_full(), getattr(reader, "metadata", None)

