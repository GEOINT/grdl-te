# -*- coding: utf-8 -*-
"""
AffineCoRegistration Tests - Synthetic control point validation.

Tests affine co-registration with known synthetic transforms.

- Level 1: Constructor, result type, matrix shape
- Level 2: Transform recovery, residuals, alignment quality

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
2026-03-20

Modified
--------
2026-03-20
"""

# Third-party
import pytest
import numpy as np

try:
    from grdl.coregistration.affine import AffineCoRegistration
    from grdl.coregistration.base import RegistrationResult
    _HAS_AFFINE = True
except ImportError:
    _HAS_AFFINE = False

pytestmark = [
    pytest.mark.coregistration,
    pytest.mark.skipif(not _HAS_AFFINE,
                       reason="AffineCoRegistration not available"),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def known_affine_transform():
    """Known affine transform: 2px translation + small rotation.

    Returns (fixed_pts, moving_pts, ground_truth_matrix).
    """
    rng = np.random.default_rng(42)
    n_pts = 20

    # Generate fixed control points spread across a 256x256 image
    fixed_pts = rng.uniform(20, 236, size=(n_pts, 2))

    # Known transform: 2px translation + 2-degree rotation
    theta = np.radians(2.0)
    tx, ty = 2.0, -1.5
    gt_matrix = np.array([
        [np.cos(theta), -np.sin(theta), tx],
        [np.sin(theta),  np.cos(theta), ty],
    ], dtype=np.float64)

    # Apply transform to get moving points
    ones = np.ones((n_pts, 1))
    fixed_h = np.hstack([fixed_pts, ones])
    moving_pts = (gt_matrix @ fixed_h.T).T

    return fixed_pts, moving_pts, gt_matrix


@pytest.fixture(scope="module")
def synthetic_images():
    """Simple synthetic 256x256 images with features."""
    rng = np.random.default_rng(42)
    fixed = rng.random((256, 256)).astype(np.float32)
    moving = rng.random((256, 256)).astype(np.float32)
    return fixed, moving


# =============================================================================
# Level 1: Format Validation
# =============================================================================


class TestAffineLevel1:
    """Validate constructor and result structure."""

    def test_affine_returns_result(self, known_affine_transform, synthetic_images):
        """estimate() returns a RegistrationResult."""
        fixed_pts, moving_pts, _ = known_affine_transform
        fixed_img, moving_img = synthetic_images

        coreg = AffineCoRegistration(fixed_pts, moving_pts)
        result = coreg.estimate(fixed_img, moving_img)
        assert isinstance(result, RegistrationResult), (
            f"Expected RegistrationResult, got {type(result).__name__}"
        )

    def test_affine_matrix_shape(self, known_affine_transform, synthetic_images):
        """Result transform_matrix is a 2x3 matrix."""
        fixed_pts, moving_pts, _ = known_affine_transform
        fixed_img, moving_img = synthetic_images

        coreg = AffineCoRegistration(fixed_pts, moving_pts)
        result = coreg.estimate(fixed_img, moving_img)
        assert result.transform_matrix.shape == (2, 3), (
            f"Expected (2, 3) matrix, got {result.transform_matrix.shape}"
        )

    def test_affine_minimum_3_points(self, synthetic_images):
        """Works with exactly 3 non-collinear control points."""
        fixed_pts = np.array([[10.0, 10.0], [10.0, 90.0], [90.0, 50.0]])
        moving_pts = np.array([[12.0, 11.0], [12.0, 91.0], [92.0, 51.0]])
        fixed_img, moving_img = synthetic_images

        coreg = AffineCoRegistration(fixed_pts, moving_pts)
        result = coreg.estimate(fixed_img, moving_img)
        assert result.transform_matrix.shape == (2, 3)

    def test_affine_rejects_too_few_points(self):
        """Raises ValueError with fewer than 3 control points."""
        fixed_pts = np.array([[10.0, 10.0], [90.0, 90.0]])
        moving_pts = np.array([[12.0, 11.0], [92.0, 91.0]])

        with pytest.raises(ValueError, match="at least 3"):
            AffineCoRegistration(fixed_pts, moving_pts)


# =============================================================================
# Level 2: Data Quality — Transform recovery
# =============================================================================


class TestAffineLevel2:
    """Validate transform recovery and alignment quality."""

    def test_affine_recovers_known_transform(self, known_affine_transform,
                                              synthetic_images):
        """Estimated matrix is close to ground truth for clean control points."""
        fixed_pts, moving_pts, gt_matrix = known_affine_transform
        fixed_img, moving_img = synthetic_images

        coreg = AffineCoRegistration(fixed_pts, moving_pts)
        result = coreg.estimate(fixed_img, moving_img)

        np.testing.assert_allclose(
            result.transform_matrix, gt_matrix, atol=0.1,
            err_msg="Estimated affine matrix deviates from ground truth"
        )

    def test_affine_residuals_low(self, known_affine_transform, synthetic_images):
        """RMS residual is below 0.5 pixels for clean control points."""
        fixed_pts, moving_pts, _ = known_affine_transform
        fixed_img, moving_img = synthetic_images

        coreg = AffineCoRegistration(fixed_pts, moving_pts)
        result = coreg.estimate(fixed_img, moving_img)

        assert result.residual_rms < 0.5, (
            f"RMS residual {result.residual_rms:.4f} exceeds 0.5 pixel threshold"
        )

    def test_affine_apply_reduces_error(self, known_affine_transform):
        """Applying the warp reduces alignment error."""
        fixed_pts, moving_pts, gt_matrix = known_affine_transform

        # Create a simple test image with a shifted version
        rng = np.random.default_rng(42)
        fixed_img = np.zeros((256, 256), dtype=np.float32)
        # Place bright features at control point locations
        for pt in fixed_pts[:5]:
            r, c = int(pt[0]), int(pt[1])
            fixed_img[max(0, r-2):min(256, r+3), max(0, c-2):min(256, c+3)] = 1.0

        moving_img = np.zeros((256, 256), dtype=np.float32)
        for pt in moving_pts[:5]:
            r, c = int(pt[0]), int(pt[1])
            moving_img[max(0, r-2):min(256, r+3), max(0, c-2):min(256, c+3)] = 1.0

        coreg = AffineCoRegistration(fixed_pts, moving_pts)
        result = coreg.estimate(fixed_img, moving_img)
        aligned = coreg.apply(moving_img, result)

        # Error should decrease after alignment
        error_before = np.mean(np.abs(fixed_img - moving_img))
        error_after = np.mean(np.abs(fixed_img - aligned))
        assert error_after <= error_before, (
            f"Alignment did not reduce error: before={error_before:.4f}, "
            f"after={error_after:.4f}"
        )
