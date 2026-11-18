# -*- coding: utf-8 -*-
"""
Scenario Manifest & Ground Truth Taxonomy for Fit-for-Purpose Evaluation.

Defines structured scenario specifications indexed by imagery modality,
geographic/clutter region, target/signature phenomenology, and evaluation
task, paired with labeled ground-truth datasets.

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class EvaluationTask(str, Enum):
    """Supported Fit-for-Purpose evaluation task families."""

    DETECTION = "detection"
    GEOLOCATION = "geolocation"
    RADIOMETRIC_QUALITY = "radiometric_quality"
    SEGMENTATION = "segmentation"


@dataclass
class GroundTruthSpec:
    """Specification for a scenario's labeled ground-truth dataset."""

    path: Path
    format: str = "geojson"
    coordinate_system: str = "wgs84"  # 'wgs84' or 'pixel'
    match_radius_m: float = 25.0
    match_radius_px: float = 5.0
    min_iou: float = 0.5
    label_property: str = "class"
    confidence_property: str = "confidence"

    @classmethod
    def from_dict(cls, data: dict[str, Any], base_dir: Path | None = None) -> GroundTruthSpec:
        raw_path = Path(data["path"])
        resolved = (base_dir / raw_path) if (base_dir and not raw_path.is_absolute()) else raw_path
        return cls(
            path=resolved,
            format=str(data.get("format", "geojson")),
            coordinate_system=str(data.get("coordinate_system", "wgs84")),
            match_radius_m=float(data.get("match_radius_m", 25.0)),
            match_radius_px=float(data.get("match_radius_px", 5.0)),
            min_iou=float(data.get("min_iou", 0.5)),
            label_property=str(data.get("label_property", "class")),
            confidence_property=str(data.get("confidence_property", "confidence")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "format": self.format,
            "coordinate_system": self.coordinate_system,
            "match_radius_m": self.match_radius_m,
            "match_radius_px": self.match_radius_px,
            "min_iou": self.min_iou,
            "label_property": self.label_property,
            "confidence_property": self.confidence_property,
        }


@dataclass
class SensorDatasetSpec:
    """Specification for the operational sensor imagery in a scenario."""

    path: Path | None = None
    reader_class: str | None = None
    collection_params: dict[str, Any] = field(default_factory=dict)
    area_km2: float | None = None
    pixel_spacing_m: tuple[float, float] | None = None
    image_shape: tuple[int, int] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], base_dir: Path | None = None) -> SensorDatasetSpec:
        raw_path = data.get("path")
        resolved: Path | None = None
        if raw_path:
            p = Path(raw_path)
            resolved = (base_dir / p) if (base_dir and not p.is_absolute()) else p
        spacing = data.get("pixel_spacing_m")
        shape = data.get("image_shape")
        return cls(
            path=resolved,
            reader_class=data.get("reader_class"),
            collection_params=dict(data.get("collection_params") or {}),
            area_km2=float(data["area_km2"]) if data.get("area_km2") is not None else None,
            pixel_spacing_m=(float(spacing[0]), float(spacing[1])) if spacing else None,
            image_shape=(int(shape[0]), int(shape[1])) if shape else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path) if self.path is not None else None,
            "reader_class": self.reader_class,
            "collection_params": dict(self.collection_params),
            "area_km2": self.area_km2,
            "pixel_spacing_m": list(self.pixel_spacing_m) if self.pixel_spacing_m else None,
            "image_shape": list(self.image_shape) if self.image_shape else None,
        }


@dataclass
class ScenarioManifest:
    """Complete operational scenario definition pairing imagery and ground truth."""

    scenario_id: str
    name: str
    description: str
    modality: str
    region_type: str
    signature_type: str
    evaluation_task: EvaluationTask
    dataset: SensorDatasetSpec
    ground_truth: GroundTruthSpec
    tags: list[str] = field(default_factory=list)
    acceptance_thresholds: dict[str, float] = field(default_factory=dict)
    base_dir: Path | None = None

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> ScenarioManifest:
        path = Path(yaml_path).resolve()
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        base_dir = path.parent
        task_val = str(data["evaluation_task"]).lower()
        return cls(
            scenario_id=str(data["scenario_id"]),
            name=str(data.get("name", data["scenario_id"])),
            description=str(data.get("description", "")),
            modality=str(data["modality"]).upper(),
            region_type=str(data["region_type"]).lower(),
            signature_type=str(data["signature_type"]).lower(),
            evaluation_task=EvaluationTask(task_val),
            dataset=SensorDatasetSpec.from_dict(data.get("dataset") or {}, base_dir=base_dir),
            ground_truth=GroundTruthSpec.from_dict(data["ground_truth"], base_dir=base_dir),
            tags=[str(t) for t in (data.get("tags") or [])],
            acceptance_thresholds={
                str(k): float(v) for k, v in (data.get("acceptance_thresholds") or {}).items()
            },
            base_dir=base_dir,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "description": self.description,
            "modality": self.modality,
            "region_type": self.region_type,
            "signature_type": self.signature_type,
            "evaluation_task": self.evaluation_task.value,
            "dataset": self.dataset.to_dict(),
            "ground_truth": self.ground_truth.to_dict(),
            "tags": list(self.tags),
            "acceptance_thresholds": dict(self.acceptance_thresholds),
        }

    def load_ground_truth(self) -> dict[str, Any]:
        """Load the scenario's GeoJSON or JSON ground-truth payload."""
        if not self.ground_truth.path.exists():
            raise FileNotFoundError(
                f"Ground-truth file not found for scenario '{self.scenario_id}': "
                f"{self.ground_truth.path}"
            )
        with open(self.ground_truth.path, encoding="utf-8") as f:
            return json.load(f)


class ScenarioRegistry:
    """Discovers and filters scenario manifests under a directory tree."""

    def __init__(self, root_dir: str | Path | None = None) -> None:
        if root_dir is None:
            root_dir = Path(__file__).resolve().parents[2] / "scenarios"
        self.root_dir = Path(root_dir)
        self._scenarios: dict[str, ScenarioManifest] = {}
        self.reload()

    def reload(self) -> None:
        self._scenarios.clear()
        if not self.root_dir.exists():
            return
        for yaml_file in sorted(self.root_dir.rglob("scenario.yaml")):
            manifest = ScenarioManifest.from_yaml(yaml_file)
            self._scenarios[manifest.scenario_id] = manifest

    def get(self, scenario_id: str) -> ScenarioManifest:
        if scenario_id not in self._scenarios:
            raise KeyError(f"Unknown scenario '{scenario_id}'. Available: {sorted(self._scenarios)}")
        return self._scenarios[scenario_id]

    def list_all(self) -> list[ScenarioManifest]:
        return [self._scenarios[k] for k in sorted(self._scenarios)]

    def filter(
        self,
        *,
        modality: str | None = None,
        region_type: str | None = None,
        signature_type: str | None = None,
        evaluation_task: EvaluationTask | str | None = None,
    ) -> list[ScenarioManifest]:
        task_filter = (
            EvaluationTask(evaluation_task.lower())
            if isinstance(evaluation_task, str)
            else evaluation_task
        )
        results: list[ScenarioManifest] = []
        for s in self.list_all():
            if modality and s.modality != modality.upper():
                continue
            if region_type and s.region_type != region_type.lower():
                continue
            if signature_type and s.signature_type != signature_type.lower():
                continue
            if task_filter and s.evaluation_task != task_filter:
                continue
            results.append(s)
        return results

