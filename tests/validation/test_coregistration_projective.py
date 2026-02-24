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
    # scipy.ndimage.affine_transform uses inverse mapping:
    #   moving[r, c] = fixed[matrix @ [r, c] + offset]
    # So the moving coordinate for fixed point p_f is:
    #   p_m = matrix^{-1} @ (p_f - offset) = matrix.T @ (p_f - offset)
    pts_moving = (matrix.T @ (pts_fixed - offset).T).T

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
        """Recovered rotation must match the known 2° rotation to within 0.05.

        ProjectiveCoRegistration returns H in moving→fixed convention:
            fixed_coord ≈ H @ homogeneous(moving_coord)
        scipy.ndimage.affine_transform uses the same moving→fixed convention
        (moving[r,c] = fixed[matrix @ [r,c] + offset]), so true_matrix IS
        the moving→fixed transform.  H[:2,:2] must agree with true_matrix to
        within 0.005 ≈ 0.3° angular tolerence under noiseless conditions.
        """
        fixed, moving, pts_f, pts_m, true_matrix, true_offset = synthetic_image_pair
        coreg = ProjectiveCoRegistration(
            control_points_fixed=pts_f,
            control_points_moving=pts_m,
        )
        result = coreg.estimate(fixed, moving)

        # H is already moving→fixed; no inversion needed
        H = result.transform_matrix
        scale = H[2, 2] if abs(H[2, 2]) > 1e-10 else 1.0
        recovered_2x2 = H[:2, :2] / scale

        diff = np.abs(recovered_2x2 - true_matrix)
        assert np.all(diff < 0.005), (
            f"H[:2,:2] differs from true_matrix by up to {diff.max():.4f} "
            "(tolerance 0.005 ≈ 0.3° angular error). With noiseless control "
            "points this should be near-zero. Check the homography solver."
        )

    def test_translation_recovery(self, synthetic_image_pair):
        """H must reproduce the known translation offset of (3.0, 3.0) pixels.

        H is in moving→fixed convention.  The fixture encodes the offset as
        the additive term in fixed_coord = matrix @ moving_coord + offset,
        so H[0,2] ≈ offset[0] (row offset) and H[1,2] ≈ offset[1] (col
        offset).  No inversion is needed; the translation lives directly in H.
        Tolerance is 0.5 px — noiseless control points should achieve < 0.1 px.
        """
        fixed, moving, pts_f, pts_m, _, true_offset = synthetic_image_pair
        coreg = ProjectiveCoRegistration(
            control_points_fixed=pts_f,
            control_points_moving=pts_m,
        )
        result = coreg.estimate(fixed, moving)

        # H is moving→fixed; translation is directly in H[:2, 2]
        H = result.transform_matrix
        scale = H[2, 2] if abs(H[2, 2]) > 1e-10 else 1.0
        recovered_row_offset = H[0, 2] / scale   # row offset (true_offset[0])
        recovered_col_offset = H[1, 2] / scale   # col offset (true_offset[1])

        assert abs(recovered_row_offset - true_offset[0]) < 0.5, (
            f"Row translation {recovered_row_offset:.4f} != "
            f"true {true_offset[0]:.4f} (tolerance 0.5 px)"
        )
        assert abs(recovered_col_offset - true_offset[1]) < 0.5, (
            f"Col translation {recovered_col_offset:.4f} != "
            f"true {true_offset[1]:.4f} (tolerance 0.5 px)"
        )

    def test_rms_residual_small(self, synthetic_image_pair):
        """RMS control-point residual must be sub-pixel for noiseless inputs.

        The control points are derived analytically from the true transform
        (pts_moving = matrix @ pts_fixed + offset) with zero added noise.
        Sub-pixel residual (< 0.5 px) is the correct expectation here.

        The old threshold of 5.0 pixels (≈ 2% of image width) would pass
        even for a poorly-conditioned solve that introduced significant error.
        """
        fixed, moving, pts_f, pts_m, _, _ = synthetic_image_pair
        coreg = ProjectiveCoRegistration(
            control_points_fixed=pts_f,
            control_points_moving=pts_m,
        )
        result = coreg.estimate(fixed, moving)
        assert result.residual_rms < 0.5, (
            f"RMS residual {result.residual_rms:.4f} px exceeds 0.5 px. "
            "With analytically-perfect (noiseless) control points the solver "
            "should achieve near-zero residual. Check the DLT or SVD solve."
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
        """Alignment must reduce MSE by at least 50% in the image interior.

        The fixture warps the fixed image by only 2° rotation + 3px translation
        so a correct alignment should come close to perfect reconstruction.
        'Any improvement' (error_after < error_before) is too weak — a 0.1%
        MSE reduction satisfies it trivially.  We require at least 50% reduction
        to confirm the alignment is actually working, not just accidentally
        marginally better.
        """
        fixed, moving, pts_f, pts_m, _, _ = synthetic_image_pair
        coreg = ProjectiveCoRegistration(
            control_points_fixed=pts_f,
            control_points_moving=pts_m,
        )
        result = coreg.estimate(fixed, moving)
        aligned = coreg.apply(moving, result)

        margin = 30
        center = slice(margin, -margin), slice(margin, -margin)
        error_before = np.mean((fixed[center] - moving[center]) ** 2)
        error_after = np.mean((fixed[center] - aligned[center]) ** 2)

        assert error_after < error_before * 0.5, (
            f"Alignment reduced MSE by only "
            f"{(1 - error_after / error_before) * 100:.1f}% "
            f"(before={error_before:.6f}, after={error_after:.6f}). "
            "Expected > 50% reduction for a 2° warp with perfect control points."
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
