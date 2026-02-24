# -*- coding: utf-8 -*-
"""
Pipeline Validation - Sequential composition of ImageTransform steps.

Tests Pipeline construction, sequential execution, progress callback,
and multi-step integration with real filter and intensity transforms.

- Level 1: Construction validation, properties
- Level 2: Sequential composition correctness, progress callback
- Level 3: Full pipeline integration with filters + intensity

Dependencies
------------
pytest
numpy

Author
------
Claude Code (generated)

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2026-02-24
"""

import pytest
import numpy as np

try:
    from grdl.image_processing.pipeline import Pipeline
    from grdl.image_processing.base import ImageTransform
    _HAS_PIPELINE = True
except ImportError:
    _HAS_PIPELINE = False

try:
    from grdl.image_processing.filters import (
        MeanFilter,
        GaussianFilter,
        MedianFilter,
    )
    from grdl.image_processing.intensity import ToDecibels, PercentileStretch
    _HAS_COMPONENTS = True
except ImportError:
    _HAS_COMPONENTS = False

pytestmark = [
    pytest.mark.pipeline,
    pytest.mark.skipif(not _HAS_PIPELINE,
                       reason="grdl Pipeline not available"),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_2d():
    """32x32 positive float64 array for pipeline tests."""
    rng = np.random.default_rng(42)
    return np.abs(rng.standard_normal((32, 32))).astype(np.float64) + 0.01


# ===================================================================
# Pipeline Level 1 — Construction and properties
# ===================================================================

@pytest.mark.skipif(not _HAS_COMPONENTS,
                    reason="grdl filter/intensity components not available")
class TestPipelineLevel1:
    """Validate Pipeline construction, properties, and error handling."""

    def test_pipeline_empty_raises(self):
        with pytest.raises(ValueError):
            Pipeline([])

    def test_pipeline_non_transform_raises(self):
        with pytest.raises(TypeError):
            Pipeline(["not_a_transform"])

    def test_pipeline_len(self):
        pipe = Pipeline([MeanFilter()])
        assert len(pipe) == 1

    def test_pipeline_steps_property(self):
        step = MeanFilter()
        pipe = Pipeline([step])
        steps = pipe.steps
        assert len(steps) == 1
        assert isinstance(steps[0], MeanFilter)
        # steps is a shallow copy
        assert steps is not pipe._steps

    def test_pipeline_repr(self):
        pipe = Pipeline([MeanFilter()])
        assert 'MeanFilter' in repr(pipe)


# ===================================================================
# Pipeline Level 2 — Composition correctness
# ===================================================================

@pytest.mark.skipif(not _HAS_COMPONENTS,
                    reason="grdl filter/intensity components not available")
class TestPipelineLevel2:
    """Validate Pipeline sequential composition against direct calls."""

    def test_pipeline_single_step_matches_direct(self, real_2d):
        pipe = Pipeline([MeanFilter()])
        pipe_result = pipe.apply(real_2d)
        direct_result = MeanFilter().apply(real_2d)
        np.testing.assert_allclose(pipe_result, direct_result)

    def test_pipeline_two_step_composition(self, real_2d):
        g = GaussianFilter(sigma=1.0)
        m = MeanFilter()
        pipe = Pipeline([g, m])
        pipe_result = pipe.apply(real_2d)
        sequential_result = m.apply(g.apply(real_2d))
        np.testing.assert_allclose(pipe_result, sequential_result)

    def test_pipeline_progress_callback(self, real_2d):
        progress_values = []

        def callback(fraction):
            progress_values.append(fraction)

        pipe = Pipeline([MeanFilter(), GaussianFilter()])
        pipe.apply(real_2d, progress_callback=callback)
        assert len(progress_values) > 0
        # Progress should be monotonically non-decreasing
        for i in range(1, len(progress_values)):
            assert progress_values[i] >= progress_values[i - 1]
        # Final value should be 1.0
        np.testing.assert_allclose(progress_values[-1], 1.0)


# ===================================================================
# Pipeline Level 3 — Integration
# ===================================================================

@pytest.mark.integration
@pytest.mark.skipif(not _HAS_COMPONENTS,
                    reason="grdl filter/intensity components not available")
class TestPipelineLevel3:
    """Full pipeline integration tests."""

    def test_pipeline_four_step_integration(self, real_2d):
        pipe = Pipeline([
            ToDecibels(),
            GaussianFilter(sigma=1.0),
            MedianFilter(),
            PercentileStretch(),
        ])
        result = pipe.apply(real_2d)
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        assert result.shape == real_2d.shape

    def test_pipeline_nested(self, real_2d):
        """Pipeline is an ImageTransform, so it can be used as a step."""
        inner = Pipeline([MeanFilter()])
        outer = Pipeline([inner, MeanFilter()])
        result = outer.apply(real_2d)
        assert result.shape == real_2d.shape
        assert np.all(np.isfinite(result))
