# -*- coding: utf-8 -*-
"""
ProjectiveCoRegistration Tests - Synthetic homography validation.

Tests ProjectiveCoRegistration with synthetically warped image pairs.

- Level 1: estimate() returns RegistrationResult with correct structure
- Level 2: Transform matrix close to known homography
- Level 3: apply() reduces alignment error

Dependencies
------------
pytest
numpy
scipy

Author
------
Claude Code (generated)

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2026-02-18

Modified
--------
2026-02-18
"""

# Third-party
import pytest
import numpy as np

try:
    from scipy.ndimage import affine_transform as scipy_affine
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    from grdl.coregistration import (
        ProjectiveCoRegistration,
        RegistrationResult,
    )
    _HAS_PROJECTIVE = True
except ImportError:
    _HAS_PROJECTIVE = False

pytestmark = [
    pytest.mark.coregistration,
    pytest.mark.skipif(not _HAS_PROJECTIVE,
                       reason="grdl ProjectiveCoRegistration not available"),
    pytest.mark.skipif(not _HAS_SCIPY,
                       reason="scipy not installed"),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic_image_pair():
    """Synthetic fixed/moving image pair with known affine warp.

    Returns (fixed, moving, pts_fixed, pts_moving, true_matrix, true_offset).
    """
    rng = np.random.default_rng(42)
    rows, cols = 256, 256
    fixed = rng.random((rows, cols), dtype=np.float32)

    # Small rotation + translation (projective ≈ affine for small angles)
    angle = np.radians(2.0)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    offset = np.array([3.0, 3.0])
    moving = scipy_affine(fixed, matrix, offset=offset, order=1)

    # Control points (well-distributed across the image)
    pts_fixed = np.array([
        [rows * 0.2, cols * 0.2],
        [rows * 0.2, cols * 0.8],
        [rows * 0.8, cols * 0.2],
        [rows * 0.8, cols * 0.8],
        [rows * 0.5, cols * 0.5],
        [rows * 0.3, cols * 0.7],
    ], dtype=np.float64)
    pts_moving = (matrix @ pts_fixed.T).T + offset

    return fixed, moving, pts_fixed, pts_moving, matrix, offset


# ---------------------------------------------------------------------------
# Level 1: Format Validation
# ---------------------------------------------------------------------------
class TestProjectiveLevel1:
    """Validate estimate() output structure."""

    def test_estimate_returns_result(self, synthetic_image_pair):
        """estimate() returns a RegistrationResult."""
        fixed, moving, pts_f, pts_m, _, _ = synthetic_image_pair
        coreg = ProjectiveCoRegistration(
            control_points_fixed=pts_f,
            control_points_moving=pts_m,
        )
        result = coreg.estimate(fixed, moving)
        assert isinstance(result, RegistrationResult)

    def test_result_has_transform(self, synthetic_image_pair):
        """RegistrationResult contains a transform matrix."""
        fixed, moving, pts_f, pts_m, _, _ = synthetic_image_pair
        coreg = ProjectiveCoRegistration(
            control_points_fixed=pts_f,
            control_points_moving=pts_m,
        )
        result = coreg.estimate(fixed, moving)
        assert result.transform_matrix is not None
        assert isinstance(result.transform_matrix, np.ndarray)

    def test_result_transform_is_3x3(self, synthetic_image_pair):
        """Projective transform matrix is 3x3."""
        fixed, moving, pts_f, pts_m, _, _ = synthetic_image_pair
        coreg = ProjectiveCoRegistration(
            control_points_fixed=pts_f,
            control_points_moving=pts_m,
        )
        result = coreg.estimate(fixed, moving)
        assert result.transform_matrix.shape == (3, 3)


# ---------------------------------------------------------------------------
# Level 2: Data Quality — Transform accuracy
# ---------------------------------------------------------------------------
class TestProjectiveLevel2:
    """Validate transform estimation accuracy."""

    def test_transform_close_to_identity_for_small_warp(self, synthetic_image_pair):
        """Recovered transform should be close to the known transformation."""
        fixed, moving, pts_f, pts_m, true_matrix, true_offset = synthetic_image_pair
        coreg = ProjectiveCoRegistration(
            control_points_fixed=pts_f,
            control_points_moving=pts_m,
        )
        result = coreg.estimate(fixed, moving)

        # The top-left 2x2 of the 3x3 homography should be close to true_matrix
        H = result.transform_matrix
        recovered_2x2 = H[:2, :2] / H[2, 2] if abs(H[2, 2]) > 1e-10 else H[:2, :2]
        # Allow tolerance for the estimation
        diff = np.abs(recovered_2x2 - true_matrix)
        assert np.all(diff < 0.5), (
            f"Transform 2x2 block differs from truth by up to {diff.max():.4f}"
        )

    def test_rms_residual_small(self, synthetic_image_pair):
        """RMS residual of control points should be small."""
        fixed, moving, pts_f, pts_m, _, _ = synthetic_image_pair
        coreg = ProjectiveCoRegistration(
            control_points_fixed=pts_f,
            control_points_moving=pts_m,
        )
        result = coreg.estimate(fixed, moving)
        assert result.residual_rms < 5.0, (
            f"RMS error {result.residual_rms:.4f} too large"
        )


# ---------------------------------------------------------------------------
# Level 3: Integration — apply()
# ---------------------------------------------------------------------------
class TestProjectiveLevel3:
    """Validate apply() produces aligned output."""

    @pytest.mark.integration
    def test_apply_returns_array(self, synthetic_image_pair):
        """apply() returns a numpy array."""
        fixed, moving, pts_f, pts_m, _, _ = synthetic_image_pair
        coreg = ProjectiveCoRegistration(
            control_points_fixed=pts_f,
            control_points_moving=pts_m,
        )
        result = coreg.estimate(fixed, moving)
        aligned = coreg.apply(moving, result)
        assert isinstance(aligned, np.ndarray)
        assert aligned.shape == fixed.shape

    @pytest.mark.integration
    def test_apply_reduces_error(self, synthetic_image_pair):
        """Aligned image should be closer to fixed than the original moving."""
        fixed, moving, pts_f, pts_m, _, _ = synthetic_image_pair
        coreg = ProjectiveCoRegistration(
            control_points_fixed=pts_f,
            control_points_moving=pts_m,
        )
        result = coreg.estimate(fixed, moving)
        aligned = coreg.apply(moving, result)

        # Compare center region to avoid border effects
        margin = 30
        center = slice(margin, -margin), slice(margin, -margin)
        error_before = np.mean((fixed[center] - moving[center]) ** 2)
        error_after = np.mean((fixed[center] - aligned[center]) ** 2)
        assert error_after < error_before, (
            f"Alignment did not improve: before={error_before:.6f}, "
            f"after={error_after:.6f}"
        )

    @pytest.mark.integration
    def test_apply_preserves_dtype(self, synthetic_image_pair):
        """apply() preserves the input array dtype."""
        fixed, moving, pts_f, pts_m, _, _ = synthetic_image_pair
        coreg = ProjectiveCoRegistration(
            control_points_fixed=pts_f,
            control_points_moving=pts_m,
        )
        result = coreg.estimate(fixed, moving)
        aligned = coreg.apply(moving, result)
        assert aligned.dtype == moving.dtype


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestProjectiveEdgeCases:
    """Edge cases and error handling."""

    def test_minimum_4_points(self):
        """Projective requires at least 4 control points."""
        pts_3 = np.array([[10, 10], [10, 90], [90, 10]], dtype=np.float64)
        with pytest.raises((ValueError, Exception)):
            ProjectiveCoRegistration(
                control_points_fixed=pts_3,
                control_points_moving=pts_3,
            )

    def test_exactly_4_points_works(self):
        """Projective works with exactly 4 control points."""
        pts_f = np.array([
            [10, 10], [10, 90], [90, 10], [90, 90]
        ], dtype=np.float64)
        pts_m = pts_f + 2.0
        coreg = ProjectiveCoRegistration(
            control_points_fixed=pts_f,
            control_points_moving=pts_m,
        )
        fixed = np.random.rand(100, 100).astype(np.float32)
        moving = np.random.rand(100, 100).astype(np.float32)
        result = coreg.estimate(fixed, moving)
        assert isinstance(result, RegistrationResult)
