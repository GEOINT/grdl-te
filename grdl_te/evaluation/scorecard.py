# -*- coding: utf-8 -*-
"""
Evaluation Scorecard — Unified Fit-for-Purpose + Performance comparison report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from grdl_te.evaluation.evaluators.base import EvaluationResult
from grdl_te.evaluation.manifest import ScenarioManifest


@dataclass
class CandidateScore:
    """Combined quantitative effectiveness and performance metrics for one candidate."""

    candidate_name: str
    evaluation: EvaluationResult
    wall_time_s: float
    memory_delta_mb: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_name": self.candidate_name,
            "wall_time_s": round(self.wall_time_s, 6),
            "memory_delta_mb": round(self.memory_delta_mb, 3),
            "evaluation": self.evaluation.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass
class EvaluationScorecard:
    """Apples-to-apples comparison scorecard across candidates for one scenario."""

    scenario: ScenarioManifest
    scores: list[CandidateScore]
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "scenario": self.scenario.to_dict(),
            "candidates": [s.to_dict() for s in self.scores],
        }

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        payload = json.dumps(self.to_dict(), indent=indent)
        if path is not None:
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(payload, encoding="utf-8")
        return payload

    def to_markdown(self) -> str:
        """Render an apples-to-apples Markdown comparison table."""
        lines = [
            f"# Fit-for-Purpose Evaluation Scorecard: {self.scenario.name}",
            "",
            f"- **Scenario ID**: `{self.scenario.scenario_id}`",
            f"- **Modality**: `{self.scenario.modality}` | "
            f"**Region**: `{self.scenario.region_type}` | "
            f"**Signature**: `{self.scenario.signature_type}`",
            f"- **Task**: `{self.scenario.evaluation_task.value}`",
            "",
        ]
        if not self.scores:
            lines.append("_No candidates evaluated._")
            return "\n".join(lines)

        metric_keys: list[str] = []
        seen: set[str] = set()
        for s in self.scores:
            for k in s.evaluation.metrics:
                if k not in seen:
                    seen.add(k)
                    metric_keys.append(k)

        headers = ["Candidate", "Pass", "Wall Time (s)", "Mem Delta (MB)"] + metric_keys
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for s in self.scores:
            pass_str = (
                "PASS"
                if s.evaluation.passed is True
                else ("FAIL" if s.evaluation.passed is False else "N/A")
            )
            row = [
                s.candidate_name,
                pass_str,
                f"{s.wall_time_s:.4f}",
                f"{s.memory_delta_mb:.2f}",
            ]
            for k in metric_keys:
                val = s.evaluation.metrics.get(k)
                if val is None:
                    row.append("-")
                elif isinstance(val, float):
                    row.append(f"{val:.4f}")
                else:
                    row.append(str(val))
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

