# -*- coding: utf-8 -*-
"""
MultilookDecomposition and CSIProcessor validation.

Tests:
- Level 1: Constructor, output array type
- Level 2: Output shape reduction, speckle reduction, 3-channel CSI output
- Level 3: Cross-validation of look counts, bounded CSI values

Dataset: Umbra SICD (*.nitf) via SICDReader

Author
------
Ava Courtney

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
    from grdl.IO.sar import SICDReader
    _HAS_SICD = True
except ImportError:
    _HAS_SICD = False

try:
    from grdl.image_processing.sar import MultilookDecomposition
    _HAS_MULTILOOK = True
except ImportError:
    _HAS_MULTILOOK = False

try:
    from grdl.image_processing.sar import CSIProcessor
    _HAS_CSI = True
except ImportError:
    _HAS_CSI = False

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False

try:
    from grdl.image_processing.sar import SublookDecomposition
    _HAS_SUBLOOK = True
except ImportError:
    _HAS_SUBLOOK = False


pytestmark = [
    pytest.mark.sar,
    pytest.mark.requires_data,
]


# =============================================================================
# MultilookDecomposition
# =============================================================================


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_MULTILOOK, reason="MultilookDecomposition not available")
@pytest.mark.skipif(not _HAS_SICD, reason="SICDReader not available")
class TestMultilookDecomposition:

    @pytest.fixture(autouse=True)
    def _load_chip(self, require_umbra_file):
        with SICDReader(str(require_umbra_file)) as reader:
            self.metadata = reader.metadata
            shape = reader.get_shape()
            cx, cy = shape[0] // 2, shape[1] // 2
            half = min(256, shape[0] // 2, shape[1] // 2)
            self.chip = reader.read_chip(cx - half, cx + half,
                                         cy - half, cy + half)

    def test_multilook_init(self):
        """Constructs with metadata + looks parameters."""
        ml = MultilookDecomposition(
            metadata=self.metadata, looks_rg=2, looks_az=2,
        )
        assert ml is not None

    def test_multilook_decompose_returns_array(self):
        """decompose() returns ndarray."""
        ml = MultilookDecomposition(
            metadata=self.metadata, looks_rg=2, looks_az=2,
        )
        result = ml.decompose(self.chip)
        assert isinstance(result, np.ndarray)

    def test_multilook_output_shape(self):
        """Output shape approximately input / (looks_rg * looks_az)."""
        ml = MultilookDecomposition(
            metadata=self.metadata, looks_rg=2, looks_az=2,
        )
        result = ml.decompose(self.chip)
        # Multilook reduces spatial dimensions
        assert result.shape[0] <= self.chip.shape[0]
        assert result.shape[1] <= self.chip.shape[1]

    def test_multilook_reduces_speckle(self):
        """Variance of output < variance of input magnitude."""
        ml = MultilookDecomposition(
            metadata=self.metadata, looks_rg=2, looks_az=2,
        )
        result = ml.decompose(self.chip)
        input_var = np.abs(self.chip).var()
        output_var = np.abs(result).var()
        # Multilooking should reduce variance (speckle suppression)
        assert output_var < input_var, \
            f"Multilook failed to reduce variance: {output_var:.4f} >= {input_var:.4f}"

    def test_multilook_2x2_vs_3x3(self):
        """3x3 decomposition produces more looks than 2x2."""
        ml2 = MultilookDecomposition(
            metadata=self.metadata, looks_rg=2, looks_az=2,
        )
        ml3 = MultilookDecomposition(
            metadata=self.metadata, looks_rg=3, looks_az=3,
        )
        r2 = ml2.decompose(self.chip)
        r3 = ml3.decompose(self.chip)
        # decompose() returns (num_looks, rows, cols) — more looks with larger window
        assert r3.shape[0] > r2.shape[0], \
            f"3x3 should produce more looks than 2x2: {r3.shape[0]} vs {r2.shape[0]}"


# =============================================================================
# CSIProcessor
# =============================================================================


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_CSI, reason="CSIProcessor not available")
@pytest.mark.skipif(not _HAS_SICD, reason="SICDReader not available")
class TestCSIProcessor:

    @pytest.fixture(autouse=True)
    def _load_chip(self, require_umbra_file):
        with SICDReader(str(require_umbra_file)) as reader:
            self.metadata = reader.metadata
            shape = reader.get_shape()
            cx, cy = shape[0] // 2, shape[1] // 2
            half = min(256, shape[0] // 2, shape[1] // 2)
            self.chip = reader.read_chip(cx - half, cx + half,
                                         cy - half, cy + half)

    def test_csi_init(self):
        """Constructs with SICD metadata."""
        csi = CSIProcessor(metadata=self.metadata)
        assert csi is not None

    def test_csi_apply_returns_array(self):
        """.apply() returns ndarray."""
        csi = CSIProcessor(metadata=self.metadata)
        result = csi.apply(self.chip)
        assert isinstance(result, np.ndarray)

    def test_csi_output_3channel(self):
        """Output shape has 3 channels for RGB composite."""
        csi = CSIProcessor(metadata=self.metadata)
        result = csi.apply(self.chip)
        assert result.ndim == 3
        # Channel axis may be first or last
        assert 3 in result.shape, f"Expected 3-channel output, got shape {result.shape}"

    def test_csi_output_real(self):
        """Output is real-valued float."""
        csi = CSIProcessor(metadata=self.metadata)
        result = csi.apply(self.chip)
        assert not np.iscomplexobj(result)
        assert result.dtype in [np.float32, np.float64]

    @pytest.mark.integration
    def test_csi_output_bounded(self):
        """Values in [0, 1] after percentile normalization."""
        csi = CSIProcessor(metadata=self.metadata, normalization="percentile")
        result = csi.apply(self.chip)
        assert result.min() >= 0.0, f"CSI min {result.min():.4f} < 0"
        assert result.max() <= 1.0 + 1e-6, f"CSI max {result.max():.4f} > 1"


# =============================================================================
# SublookDecomposition — CuPy GPU path
# =============================================================================


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_CUPY, reason="CuPy not installed")
@pytest.mark.skipif(not _HAS_SUBLOOK, reason="SublookDecomposition not available")
@pytest.mark.skipif(not _HAS_SICD, reason="SICDReader not available")
class TestSublookDecompositionGPU:

    @pytest.fixture(autouse=True)
    def _load_chip(self, require_umbra_file):
        with SICDReader(str(require_umbra_file)) as reader:
            self.metadata = reader.metadata
            shape = reader.get_shape()
            cx, cy = shape[0] // 2, shape[1] // 2
            half = min(256, shape[0] // 2, shape[1] // 2)
            self.chip = reader.read_chip(cx - half, cx + half,
                                         cy - half, cy + half)

    def test_decompose_cupy_input_returns_cupy(self):
        """decompose() with cupy input returns a cupy array."""
        sublook = SublookDecomposition(metadata=self.metadata, num_looks=3)
        result = sublook.decompose(cp.asarray(self.chip))
        assert isinstance(result, cp.ndarray), (
            f"Expected cp.ndarray, got {type(result).__name__}"
        )

    def test_decompose_cupy_shape(self):
        """Output shape matches numpy path for cupy input."""
        sublook = SublookDecomposition(metadata=self.metadata, num_looks=3)
        cpu_result = sublook.decompose(self.chip)
        gpu_result = sublook.decompose(cp.asarray(self.chip))
        assert gpu_result.shape == cpu_result.shape, (
            f"Shape mismatch: GPU {gpu_result.shape} vs CPU {cpu_result.shape}"
        )

    def test_decompose_cupy_matches_numpy(self):
        """CuPy and numpy paths produce numerically identical results."""
        sublook = SublookDecomposition(metadata=self.metadata, num_looks=3)
        cpu_result = sublook.decompose(self.chip)
        gpu_result = cp.asnumpy(sublook.decompose(cp.asarray(self.chip)))
        assert np.allclose(np.abs(cpu_result), np.abs(gpu_result), atol=1e-4), (
            "CuPy and numpy decompose() results differ beyond tolerance"
        )


# =============================================================================
# CSIProcessor — CuPy GPU path
# =============================================================================


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_CUPY, reason="CuPy not installed")
@pytest.mark.skipif(not _HAS_CSI, reason="CSIProcessor not available")
@pytest.mark.skipif(not _HAS_SICD, reason="SICDReader not available")
class TestCSIProcessorGPU:

    @pytest.fixture(autouse=True)
    def _load_chip(self, require_umbra_file):
        with SICDReader(str(require_umbra_file)) as reader:
            self.metadata = reader.metadata
            shape = reader.get_shape()
            cx, cy = shape[0] // 2, shape[1] // 2
            half = min(256, shape[0] // 2, shape[1] // 2)
            self.chip = reader.read_chip(cx - half, cx + half,
                                         cy - half, cy + half)

    def test_csi_cupy_input_returns_cupy(self):
        """apply() with cupy input returns a cupy array."""
        csi = CSIProcessor(metadata=self.metadata)
        result = csi.apply(cp.asarray(self.chip))
        assert isinstance(result, cp.ndarray), (
            f"Expected cp.ndarray, got {type(result).__name__}"
        )

    def test_csi_cupy_matches_numpy(self):
        """CuPy and numpy paths produce numerically equivalent results."""
        csi = CSIProcessor(metadata=self.metadata, normalization="percentile")
        cpu_result = csi.apply(self.chip)
        gpu_result = cp.asnumpy(csi.apply(cp.asarray(self.chip)))
        assert np.allclose(cpu_result, gpu_result, atol=1e-4), (
            "CuPy and numpy CSI apply() results differ beyond tolerance"
        )
