# -*- coding: utf-8 -*-
"""
GPU CFAR Detector Verification - CuPy path tests.

Verifies that CFAR helpers (_annular_stats, _quadrant_means) and detectors
(CA-CFAR, GO-CFAR, SO-CFAR) correctly dispatch to cupy when a cupy array
is passed as input.  Also covers dominance/entropy feature extraction.

Each test class checks:

- **Type correctness** — output is a ``cupy.ndarray`` (stays on-device).
- **Shape consistency** — GPU output shape matches the CPU (numpy) path.
- **Numerical equivalence** — GPU and CPU results agree within tolerance.
- **Detection fidelity** — GPU and CPU detections have identical geometry,
  confidence, and properties (not just matching counts).

All tests use synthetic data (no real SAR files required) and are skipped
when CuPy is not installed.

Dependencies
------------
cupy
pytest
numpy

Author
------
Ava Courtney
courtney-ava@zai.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-03-10

Modified
--------
2026-03-10
"""

# Standard library
# (none)

# Third-party
import numpy as np
import pytest

try:
    import cupy as cp
    cp.array([1.0])  # verify GPU is actually functional
    _HAS_CUPY = True
except Exception:
    _HAS_CUPY = False

# --- grdl imports (guarded) ---

try:
    from grdl.image_processing.detection.cfar._base import (
        _annular_stats,
        _quadrant_means,
    )
    from grdl.image_processing.detection.cfar.ca_cfar import CACFARDetector
    from grdl.image_processing.detection.cfar.go_cfar import GOCFARDetector
    from grdl.image_processing.detection.cfar.so_cfar import SOCFARDetector
    _HAS_CFAR = True
except ImportError:
    _HAS_CFAR = False

try:
    from grdl.image_processing.sar.dominance import (
        compute_dominance,
        compute_sublook_entropy,
    )
    _HAS_DOMINANCE = True
except ImportError:
    _HAS_DOMINANCE = False


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(not _HAS_CUPY, reason="CuPy not installed"),
]

# Tolerance for numerical comparisons
_ATOL = 1e-4


# =============================================================================
# Shared synthetic fixtures
# =============================================================================

@pytest.fixture(scope='module')
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope='module')
def cfar_image(rng):
    """256x256 float64 dB-domain image with a few bright targets."""
    bg = rng.standard_normal((256, 256)).astype(np.float64) * 2.0 + 5.0
    # Insert a few bright pixels
    bg[64, 64] = 30.0
    bg[128, 200] = 35.0
    return bg


@pytest.fixture(scope='module')
def sublooks_stack(rng):
    """(5, 256, 256) complex128 sub-aperture stack."""
    r = rng.standard_normal((5, 256, 256))
    i = rng.standard_normal((5, 256, 256))
    return (r + 1j * i).astype(np.complex128)


# =============================================================================
# CFAR helpers
# =============================================================================


@pytest.mark.skipif(not _HAS_CFAR, reason="grdl CFAR not available")
class TestAnnularStatsGPU:

    def test_returns_cupy(self, cfar_image):
        img_gpu = cp.asarray(cfar_image)
        mean, std = _annular_stats(img_gpu, guard_cells=3, training_cells=12)
        assert isinstance(mean, cp.ndarray)
        assert isinstance(std, cp.ndarray)

    def test_shape_matches_cpu(self, cfar_image):
        img_gpu = cp.asarray(cfar_image)
        mean_cpu, std_cpu = _annular_stats(cfar_image, guard_cells=3, training_cells=12)
        mean_gpu, std_gpu = _annular_stats(img_gpu, guard_cells=3, training_cells=12)
        assert mean_gpu.shape == mean_cpu.shape
        assert std_gpu.shape == std_cpu.shape

    def test_numerics_match_cpu(self, cfar_image):
        img_gpu = cp.asarray(cfar_image)
        mean_cpu, std_cpu = _annular_stats(cfar_image, guard_cells=3, training_cells=12)
        mean_gpu, std_gpu = _annular_stats(img_gpu, guard_cells=3, training_cells=12)
        assert np.allclose(mean_cpu, cp.asnumpy(mean_gpu), atol=_ATOL), (
            "_annular_stats bg_mean GPU/CPU mismatch"
        )
        assert np.allclose(std_cpu, cp.asnumpy(std_gpu), atol=_ATOL), (
            "_annular_stats bg_std GPU/CPU mismatch"
        )


@pytest.mark.skipif(not _HAS_CFAR, reason="grdl CFAR not available")
class TestQuadrantMeansGPU:

    def test_returns_list_of_cupy(self, cfar_image):
        img_gpu = cp.asarray(cfar_image)
        quads = _quadrant_means(img_gpu, guard_cells=3, training_cells=12)
        assert len(quads) == 4
        for q in quads:
            assert isinstance(q, cp.ndarray)

    def test_shape_matches_cpu(self, cfar_image):
        img_gpu = cp.asarray(cfar_image)
        quads_cpu = _quadrant_means(cfar_image, guard_cells=3, training_cells=12)
        quads_gpu = _quadrant_means(img_gpu, guard_cells=3, training_cells=12)
        for qc, qg in zip(quads_cpu, quads_gpu):
            assert qg.shape == qc.shape

    def test_numerics_match_cpu(self, cfar_image):
        img_gpu = cp.asarray(cfar_image)
        quads_cpu = _quadrant_means(cfar_image, guard_cells=3, training_cells=12)
        quads_gpu = _quadrant_means(img_gpu, guard_cells=3, training_cells=12)
        for i, (qc, qg) in enumerate(zip(quads_cpu, quads_gpu)):
            assert np.allclose(qc, cp.asnumpy(qg), atol=_ATOL), (
                f"_quadrant_means Q{i} GPU/CPU mismatch"
            )


# =============================================================================
# CFAR detectors (full pipeline)
# =============================================================================


def _assert_detections_match(cpu_ds, gpu_ds, label="CFAR"):
    """Assert two DetectionSets have identical detections (geometry + properties)."""
    assert len(cpu_ds.detections) == len(gpu_ds.detections), (
        f"{label} count mismatch: CPU={len(cpu_ds)}, GPU={len(gpu_ds)}"
    )
    for i, (cd, gd) in enumerate(zip(cpu_ds.detections, gpu_ds.detections)):
        assert cd.pixel_geometry.bounds == gd.pixel_geometry.bounds, (
            f"{label} detection {i} bbox mismatch: "
            f"CPU={cd.pixel_geometry.bounds}, GPU={gd.pixel_geometry.bounds}"
        )
        if cd.confidence is not None:
            assert abs(cd.confidence - gd.confidence) < 1e-6, (
                f"{label} detection {i} confidence mismatch: "
                f"CPU={cd.confidence}, GPU={gd.confidence}"
            )
        for key in cd.properties:
            cv = cd.properties[key]
            gv = gd.properties[key]
            if isinstance(cv, float):
                assert abs(cv - gv) < 1e-2, (
                    f"{label} detection {i} property '{key}' mismatch: "
                    f"CPU={cv}, GPU={gv}"
                )


@pytest.mark.skipif(not _HAS_CFAR, reason="grdl CFAR not available")
class TestCACFARDetectorGPU:

    def test_detect_with_cupy_returns_detectionset(self, cfar_image):
        """detect() with cupy input completes and returns a DetectionSet."""
        from grdl.image_processing.detection.models import DetectionSet
        detector = CACFARDetector(guard_cells=3, training_cells=12, pfa=1e-3)
        result = detector.detect(cp.asarray(cfar_image))
        assert isinstance(result, DetectionSet)

    def test_detect_gpu_matches_cpu_count(self, cfar_image):
        """GPU and CPU paths produce the same number of detections."""
        detector = CACFARDetector(guard_cells=3, training_cells=12, pfa=1e-3)
        cpu_result = detector.detect(cfar_image)
        gpu_result = detector.detect(cp.asarray(cfar_image))
        assert len(gpu_result.detections) == len(cpu_result.detections), (
            f"CA-CFAR detection count mismatch: GPU={len(gpu_result.detections)}, "
            f"CPU={len(cpu_result.detections)}"
        )

    def test_detect_gpu_matches_cpu_details(self, cfar_image):
        """GPU and CPU detections have identical geometry and properties."""
        detector = CACFARDetector(guard_cells=3, training_cells=12, pfa=1e-3)
        cpu_result = detector.detect(cfar_image)
        gpu_result = detector.detect(cp.asarray(cfar_image))
        _assert_detections_match(cpu_result, gpu_result, label="CA-CFAR")


@pytest.mark.skipif(not _HAS_CFAR, reason="grdl CFAR not available")
class TestGOCFARDetectorGPU:

    def test_detect_with_cupy_returns_detectionset(self, cfar_image):
        from grdl.image_processing.detection.models import DetectionSet
        detector = GOCFARDetector(guard_cells=3, training_cells=12, pfa=1e-3)
        result = detector.detect(cp.asarray(cfar_image))
        assert isinstance(result, DetectionSet)

    def test_detect_gpu_matches_cpu_count(self, cfar_image):
        detector = GOCFARDetector(guard_cells=3, training_cells=12, pfa=1e-3)
        cpu_result = detector.detect(cfar_image)
        gpu_result = detector.detect(cp.asarray(cfar_image))
        assert len(gpu_result.detections) == len(cpu_result.detections), (
            f"GO-CFAR detection count mismatch: GPU={len(gpu_result.detections)}, "
            f"CPU={len(cpu_result.detections)}"
        )

    def test_detect_gpu_matches_cpu_details(self, cfar_image):
        """GPU and CPU detections have identical geometry and properties."""
        detector = GOCFARDetector(guard_cells=3, training_cells=12, pfa=1e-3)
        cpu_result = detector.detect(cfar_image)
        gpu_result = detector.detect(cp.asarray(cfar_image))
        _assert_detections_match(cpu_result, gpu_result, label="GO-CFAR")


@pytest.mark.skipif(not _HAS_CFAR, reason="grdl CFAR not available")
class TestSOCFARDetectorGPU:

    def test_detect_with_cupy_returns_detectionset(self, cfar_image):
        from grdl.image_processing.detection.models import DetectionSet
        detector = SOCFARDetector(guard_cells=3, training_cells=12, pfa=1e-3)
        result = detector.detect(cp.asarray(cfar_image))
        assert isinstance(result, DetectionSet)

    def test_detect_gpu_matches_cpu_count(self, cfar_image):
        detector = SOCFARDetector(guard_cells=3, training_cells=12, pfa=1e-3)
        cpu_result = detector.detect(cfar_image)
        gpu_result = detector.detect(cp.asarray(cfar_image))
        assert len(gpu_result.detections) == len(cpu_result.detections), (
            f"SO-CFAR detection count mismatch: GPU={len(gpu_result.detections)}, "
            f"CPU={len(cpu_result.detections)}"
        )

    def test_detect_gpu_matches_cpu_details(self, cfar_image):
        """GPU and CPU detections have identical geometry and properties."""
        detector = SOCFARDetector(guard_cells=3, training_cells=12, pfa=1e-3)
        cpu_result = detector.detect(cfar_image)
        gpu_result = detector.detect(cp.asarray(cfar_image))
        _assert_detections_match(cpu_result, gpu_result, label="SO-CFAR")


# =============================================================================
# Dominance / entropy
# =============================================================================


@pytest.mark.skipif(not _HAS_DOMINANCE, reason="grdl.image_processing.sar.dominance not available")
class TestComputeDominanceGPU:

    def test_returns_cupy(self, sublooks_stack):
        result = compute_dominance(cp.asarray(sublooks_stack), window_size=7)
        assert isinstance(result, cp.ndarray)

    def test_shape_matches_cpu(self, sublooks_stack):
        cpu = compute_dominance(sublooks_stack, window_size=7)
        gpu = compute_dominance(cp.asarray(sublooks_stack), window_size=7)
        assert gpu.shape == cpu.shape

    def test_numerics_match_cpu(self, sublooks_stack):
        cpu = compute_dominance(sublooks_stack, window_size=7)
        gpu = cp.asnumpy(compute_dominance(cp.asarray(sublooks_stack), window_size=7))
        assert np.allclose(cpu, gpu, atol=_ATOL), "compute_dominance GPU/CPU mismatch"


@pytest.mark.skipif(not _HAS_DOMINANCE, reason="grdl.image_processing.sar.dominance not available")
class TestComputeSublookEntropyGPU:

    def test_returns_cupy(self, sublooks_stack):
        result = compute_sublook_entropy(cp.asarray(sublooks_stack), window_size=7)
        assert isinstance(result, cp.ndarray)

    def test_shape_matches_cpu(self, sublooks_stack):
        cpu = compute_sublook_entropy(sublooks_stack, window_size=7)
        gpu = compute_sublook_entropy(cp.asarray(sublooks_stack), window_size=7)
        assert gpu.shape == cpu.shape

    def test_numerics_match_cpu(self, sublooks_stack):
        cpu = compute_sublook_entropy(sublooks_stack, window_size=7)
        gpu = cp.asnumpy(
            compute_sublook_entropy(cp.asarray(sublooks_stack), window_size=7)
        )
        assert np.allclose(cpu, gpu, atol=_ATOL), "compute_sublook_entropy GPU/CPU mismatch"
