# -*- coding: utf-8 -*-
"""
Unit & Integration Tests for grdl_te.evaluation (Fit-for-Purpose Framework).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from grdl_te.evaluation import (
    DetectionEvaluator,
    EvaluationCandidate,
    EvaluationRunner,
    EvaluationTask,
    GeolocationEvaluator,
    RadiometricEvaluator,
    ScenarioRegistry,
    SegmentationEvaluator,
)

pytestmark = pytest.mark.evaluation


class TestScenarioRegistry:
    def test_loads_curated_scenarios_and_filters_by_taxonomy(self):
        registry = ScenarioRegistry()
        all_scenarios = registry.list_all()
        assert len(all_scenarios) >= 4

        sar_det = registry.filter(
            modality="SAR",
            region_type="littoral",
            signature_type="maritime_vessel",
            evaluation_task=EvaluationTask.DETECTION,
        )
        assert len(sar_det) == 1
        assert sar_det[0].scenario_id == "sar_maritime_vessel_detection"

        msi_seg = registry.filter(modality="MSI", evaluation_task="segmentation")
        assert len(msi_seg) == 1
        assert msi_seg[0].scenario_id == "eo_msi_vegetation_water_segmentation"


class TestDetectionEvaluator:
    def test_detection_metrics_and_roc_auc(self):
        registry = ScenarioRegistry()
        manifest = registry.get("sar_maritime_vessel_detection")
        gt = manifest.load_ground_truth()

        # 3 true positives (within 35m), 1 false positive, 1 missed target (FN)
        preds = [
            {"lat": 36.02005, "lon": -75.78005, "confidence": 0.95},
            {"lat": 36.02805, "lon": -75.76505, "confidence": 0.85},
            {"lat": 36.01505, "lon": -75.75205, "confidence": 0.75},
            {"lat": 36.05000, "lon": -75.70000, "confidence": 0.30},  # False alarm
        ]

        evaluator = DetectionEvaluator()
        res = evaluator.evaluate(preds, gt, manifest, candidate_name="cfar_test")

        assert res.metrics["tp"] == 3.0
        assert res.metrics["fp"] == 1.0
        assert res.metrics["fn"] == 1.0
        assert res.metrics["pd"] == pytest.approx(0.75)
        assert res.metrics["precision"] == pytest.approx(0.75)
        assert res.metrics["far_per_km2"] == pytest.approx(1.0 / 25.0)
        assert res.metrics["cep90_m"] < 25.0
        assert "roc_auc" in res.metrics
        # Threshold min_pd is 0.85 -> fails acceptance
        assert res.passed is False


class TestGeolocationEvaluator:
    def test_geolocation_ce90_le90_and_bias(self):
        registry = ScenarioRegistry()
        manifest = registry.get("sar_geolocation_calibration_array")
        gt = manifest.load_ground_truth()

        class _MockGeo:
            def image_to_latlon(self, row: float, col: float):
                # Exact match for first 2, ~1.1m north offset on all
                lookup = {
                    (250.0, 300.0): (36.01001, -75.75000, 12.2),
                    (750.0, 800.0): (36.02001, -75.74000, 14.3),
                    (1250.0, 1300.0): (36.01501, -75.73000, 11.4),
                }
                return lookup[(row, col)]

        res = GeolocationEvaluator().evaluate(_MockGeo(), gt, manifest, candidate_name="sicd_geo")
        assert res.metrics["num_gcps"] == 3.0
        assert res.metrics["ce90_m"] < 5.0
        assert res.metrics["le90_m"] < 5.0
        assert res.metrics["mean_north_bias_m"] > 0.0
        assert res.passed is True


class TestApplesToApplesSarFilters:
    def test_compare_grdl_sar_speckle_filters(self, tmp_path):
        from grdl_sar.filters import BoxcarFilter, FrostFilter, KuanFilter, LeeFilter

        registry = ScenarioRegistry()
        manifest = registry.get("sar_speckle_homogeneous_clutter")

        # Synthetic 256x256 SAR intensity image with two intensity steps & multiplicative speckle
        rng = np.random.default_rng(123)
        clean = np.full((256, 256), 100.0, dtype=np.float64)
        clean[:, 128:] = 300.0  # step edge along col 128 (inside edge ROI [118..138])
        speckle = rng.gamma(shape=1.0, scale=1.0, size=(256, 256))
        sar_img = (clean * speckle).astype(np.float32)

        runner = EvaluationRunner()
        scorecard = runner.evaluate_scenario(
            manifest,
            candidates=[
                EvaluationCandidate("LeeFilter(k=7)", LeeFilter(kernel_size=7)),
                EvaluationCandidate("FrostFilter(w=7)", FrostFilter(window_size=7, damping_factor=2.0)),
                EvaluationCandidate("KuanFilter(w=7)", KuanFilter(window_size=7, num_looks=1)),
                EvaluationCandidate("BoxcarFilter(w=7)", BoxcarFilter(window_size=7)),
            ],
            source_data=sar_img,
        )

        assert len(scorecard.scores) == 4
        by_name = {s.candidate_name: s for s in scorecard.scores}
        # All filters must improve ENL (> 1.0) and reduce speckle (SSI < 1.0)
        for score in scorecard.scores:
            assert score.evaluation.metrics["enl"] > 1.0
            assert score.evaluation.metrics["ssi"] < 1.0
            assert score.wall_time_s >= 0.0

        # Boxcar / Lee / Kuan achieve strong ENL (> 3.0); Frost(damping=2.0) preserves higher EPI
        assert by_name["BoxcarFilter(w=7)"].evaluation.metrics["enl"] > 10.0
        assert by_name["LeeFilter(k=7)"].evaluation.metrics["enl"] > 3.0
        assert by_name["KuanFilter(w=7)"].evaluation.metrics["enl"] > 3.0
        assert by_name["FrostFilter(w=7)"].evaluation.metrics["epi"] > by_name["BoxcarFilter(w=7)"].evaluation.metrics["epi"]

        md = scorecard.to_markdown()
        assert "LeeFilter(k=7)" in md
        assert "BoxcarFilter(w=7)" in md
        out_json = tmp_path / "scorecard.json"
        scorecard.to_json(out_json)
        assert out_json.exists()


class TestApplesToApplesEoSegmentation:
    def test_compare_grdl_eo_ndvi_segmentation(self):
        from grdl_eo.spectral import SpectralIndexTransform

        registry = ScenarioRegistry()
        manifest = registry.get("eo_msi_vegetation_water_segmentation")

        # 6-band MSI cube (128x128); vegetation polygon is cols [16..64], rows [16..96]
        cube = np.full((6, 128, 128), 0.2, dtype=np.float32)
        cube[2] = 0.5  # Red background
        cube[3] = 0.2  # NIR background -> NDVI < 0
        # Vegetated patch: high NIR, low Red -> NDVI = (0.8 - 0.1)/(0.8 + 0.1) = +0.777
        cube[2, 16:96, 16:64] = 0.1
        cube[3, 16:96, 16:64] = 0.8

        runner = EvaluationRunner()
        scorecard = runner.evaluate_scenario(
            manifest,
            candidates=[
                EvaluationCandidate(
                    "NDVI>=0.3",
                    SpectralIndexTransform(index_type="NDVI", nir_idx=3, red_idx=2),
                    postprocess=lambda ndvi: (ndvi >= 0.3).astype(np.int64),
                )
            ],
            source_data=cube,
        )

        res = scorecard.scores[0].evaluation
        assert res.metrics["mean_iou"] >= 0.99
        assert res.metrics["overall_accuracy"] >= 0.99
        assert res.metrics["cohens_kappa"] >= 0.99
        assert res.passed is True
