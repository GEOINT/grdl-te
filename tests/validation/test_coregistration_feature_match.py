# -*- coding: utf-8 -*-
"""
FeatureMatchCoRegistration Tests - Automated feature-based registration.

Tests feature detection and matching using synthetic images with
known geometric transforms.

- Level 1: Constructor, result type, keypoint detection
- Level 2: Translation detection, residuals, RANSAC robustness

Dependencies
------------
pytest
numpy
opencv-python-headless

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
    from grdl.coregistration.feature_match import FeatureMatchCoRegistration
    from grdl.coregistration.base import RegistrationResult
    _HAS_FEATURE_MATCH = True
except ImportError:
    _HAS_FEATURE_MATCH = False

pytestmark = [
    pytest.mark.coregistration,
    pytest.mark.skipif(not _HAS_FEATURE_MATCH,
                       reason="FeatureMatchCoRegistration not available "
                              "(requires opencv-python-headless)"),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def feature_rich_pair():
    """Synthetic image pair with rich features and known 5px shift.

    Creates a random texture with high-contrast blobs, then shifts
    it by (5, 3) pixels to create a moving image.
    """
    rng = np.random.default_rng(42)
    base = np.zeros((256, 256), dtype=np.uint8)

    # Create random bright blobs as features
    for _ in range(40):
        r = rng.integers(30, 226)
        c = rng.integers(30, 226)
        radius = rng.integers(5, 15)
        brightness = rng.integers(180, 255)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr * dr + dc * dc <= radius * radius:
                    base[r + dr, c + dc] = brightness

    # Add some texture noise
    base = base.astype(np.float32)
    base += rng.standard_normal((256, 256)).astype(np.float32) * 10
    base = np.clip(base, 0, 255).astype(np.uint8)

    fixed = base.copy()

    # Shift moving image by (5, 3) pixels
    moving = np.zeros_like(base)
    moving[5:, 3:] = base[:-5, :-3]

    return fixed, moving


# =============================================================================
# Level 1: Format Validation
# =============================================================================


class TestFeatureMatchLevel1:
    """Validate constructor and result structure."""

    def test_feature_match_returns_result(self, feature_rich_pair):
        """estimate() returns a RegistrationResult."""
        fixed, moving = feature_rich_pair
        coreg = FeatureMatchCoRegistration(method='orb', max_features=2000)
        result = coreg.estimate(fixed, moving)
        assert isinstance(result, RegistrationResult), (
            f"Expected RegistrationResult, got {type(result).__name__}"
        )

    def test_feature_match_detects_keypoints(self, feature_rich_pair):
        """Estimation succeeds with detected keypoints (num_matches > 0)."""
        fixed, moving = feature_rich_pair
        coreg = FeatureMatchCoRegistration(method='orb', max_features=2000)
        result = coreg.estimate(fixed, moving)
        assert result.num_matches > 0, (
            "No feature matches found between synthetic images"
        )

    def test_feature_match_descriptor_type(self, feature_rich_pair):
        """ORB descriptor method works without error."""
        fixed, moving = feature_rich_pair
        coreg = FeatureMatchCoRegistration(method='orb')
        result = coreg.estimate(fixed, moving)
        # Metadata records the full method name
        assert result.metadata.get('method') == 'feature_match_orb'


# =============================================================================
# Level 2: Data Quality — Transform recovery
# =============================================================================


class TestFeatureMatchLevel2:
    """Validate transform detection and robustness."""

    def test_feature_match_synthetic_shift(self, feature_rich_pair):
        """Detects known 5px translation within tolerance.

        The moving image is shifted down by 5 rows and right by 3 cols
        relative to fixed (moving[5:, 3:] = fixed[:-5, :-3]). The
        transform maps moving → fixed in (row, col) space, so the
        estimated translation should be approximately (-5, -3).
        """
        fixed, moving = feature_rich_pair
        coreg = FeatureMatchCoRegistration(
            method='orb', max_features=3000, transform_type='affine'
        )
        result = coreg.estimate(fixed, moving)
        matrix = result.transform_matrix

        # Matrix is (2, 3) in (row, col) space: [[a b ty], [c d tx]]
        ty = matrix[0, 2]  # row translation (moving → fixed)
        tx = matrix[1, 2]  # col translation (moving → fixed)

        # Moving→fixed undoes the (5, 3) shift → expect ≈ (-5, -3)
        assert abs(ty - (-5.0)) < 3.0, (
            f"Row translation {ty:.2f} not close to expected -5.0"
        )
        assert abs(tx - (-3.0)) < 3.0, (
            f"Col translation {tx:.2f} not close to expected -3.0"
        )

    def test_feature_match_residuals(self, feature_rich_pair):
        """RMS residual is below 5 pixels for synthetic shift."""
        fixed, moving = feature_rich_pair
        coreg = FeatureMatchCoRegistration(method='orb', max_features=2000)
        result = coreg.estimate(fixed, moving)
        assert result.residual_rms < 5.0, (
            f"RMS residual {result.residual_rms:.4f} exceeds 5.0 pixel threshold"
        )

    def test_feature_match_ransac_outlier_rejection(self, feature_rich_pair):
        """RANSAC achieves reasonable inlier ratio (> 0.3)."""
        fixed, moving = feature_rich_pair
        coreg = FeatureMatchCoRegistration(
            method='orb', max_features=2000, ransac_threshold=5.0
        )
        result = coreg.estimate(fixed, moving)
        assert result.inlier_ratio > 0.3, (
            f"Inlier ratio {result.inlier_ratio:.3f} too low; "
            "RANSAC may be failing to find consensus"
        )
