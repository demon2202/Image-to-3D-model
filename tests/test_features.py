"""
Tests for Phase 1: Feature Detection and Matching.

Run with: pytest tests/test_features.py -v
"""

import os
import sys
import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phase1_features.detector import FeatureDetector, FeatureData
from src.phase1_features.matcher import FeatureMatcher, MatchResult


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_images():
    """
    Create two synthetic images with a known homography between them
    so we can verify correct matching.
    """
    np.random.seed(42)

    # Create a textured image (checkerboard + noise)
    img1 = np.zeros((480, 640, 3), dtype=np.uint8)

    # Checkerboard pattern
    block = 40
    for y in range(0, 480, block * 2):
        for x in range(0, 640, block * 2):
            img1[y:y + block, x:x + block] = 200
            img1[y + block:y + 2 * block, x + block:x + 2 * block] = 200

    # Add some circles (distinct keypoints)
    for i in range(20):
        cx = np.random.randint(50, 590)
        cy = np.random.randint(50, 430)
        color = (np.random.randint(100, 255),) * 3
        cv2.circle(img1, (cx, cy), 15, color, -1)

    # Add noise
    noise = np.random.randint(0, 30, img1.shape, dtype=np.uint8)
    img1 = np.clip(img1.astype(np.int32) + noise, 0, 255).astype(np.uint8)

    # Create img2 by applying a known transformation
    # Small rotation + translation
    h, w = img1.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), 8.0, 0.95)
    M[0, 2] += 25   # translate x
    M[1, 2] += 15   # translate y
    img2 = cv2.warpAffine(img1, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    return img1, img2, M


@pytest.fixture(scope="module")
def blank_image():
    """A uniform (featureless) image — should produce no keypoints."""
    return np.ones((240, 320, 3), dtype=np.uint8) * 128


@pytest.fixture(scope="module")
def sift_detector():
    return FeatureDetector(method="sift", max_keypoints=1000)


@pytest.fixture(scope="module")
def orb_detector():
    return FeatureDetector(method="orb", max_keypoints=1000)


@pytest.fixture(scope="module")
def flann_matcher():
    return FeatureMatcher(
        method="flann",
        ratio_threshold=0.75,
        ransac_threshold=3.0,
        min_matches=10,
        descriptor_type="sift"
    )


# ─────────────────────────────────────────────────────────────
# FeatureDetector Tests
# ─────────────────────────────────────────────────────────────

class TestFeatureDetector:

    def test_sift_detector_creation(self):
        """SIFT detector initializes without error."""
        detector = FeatureDetector(method="sift", max_keypoints=500)
        assert detector is not None
        assert detector.method == "sift"

    def test_orb_detector_creation(self):
        """ORB detector initializes without error."""
        detector = FeatureDetector(method="orb", max_keypoints=500)
        assert detector is not None
        assert detector.method == "orb"

    def test_invalid_method_raises(self):
        """Invalid detector method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown detector"):
            FeatureDetector(method="surf")   # SURF not free

    def test_sift_detects_features(self, sift_detector, synthetic_images):
        """SIFT finds keypoints in a textured image."""
        img1, _, _ = synthetic_images
        features = sift_detector.detect(img1)

        assert isinstance(features, FeatureData)
        assert features.num_features > 0
        assert features.descriptors is not None
        assert features.descriptors.shape[1] == 128    # SIFT = 128-dim

    def test_orb_detects_features(self, orb_detector, synthetic_images):
        """ORB finds keypoints in a textured image."""
        img1, _, _ = synthetic_images
        features = orb_detector.detect(img1)

        assert features.num_features > 0
        assert features.descriptors.shape[1] == 32     # ORB = 32-dim

    def test_blank_image_few_features(self, sift_detector, blank_image):
        """Blank image produces very few or no keypoints."""
        features = sift_detector.detect(blank_image)
        assert features.num_features < 10

    def test_max_keypoints_respected(self, synthetic_images):
        """Detector respects max_keypoints limit."""
        img1, _, _ = synthetic_images
        max_kp = 50
        detector = FeatureDetector(method="sift", max_keypoints=max_kp)
        features = detector.detect(img1)
        assert features.num_features <= max_kp

    def test_grayscale_input(self, sift_detector, synthetic_images):
        """Detector handles grayscale (single-channel) input."""
        img1, _, _ = synthetic_images
        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        features = sift_detector.detect(gray)
        assert features.num_features > 0

    def test_get_points_shape(self, sift_detector, synthetic_images):
        """get_points() returns Nx2 array."""
        img1, _, _ = synthetic_images
        features = sift_detector.detect(img1)
        pts = features.get_points()
        assert pts.shape == (features.num_features, 2)
        assert pts.dtype == np.float64

    def test_detect_batch(self, sift_detector, synthetic_images):
        """detect_batch processes list of images."""
        img1, img2, _ = synthetic_images
        results = sift_detector.detect_batch([img1, img2])
        assert len(results) == 2
        assert all(isinstance(r, FeatureData) for r in results)
        assert all(r.num_features > 0 for r in results)

    def test_image_shape_stored(self, sift_detector, synthetic_images):
        """FeatureData stores correct image shape."""
        img1, _, _ = synthetic_images
        features = sift_detector.detect(img1)
        assert features.image_shape == (img1.shape[0], img1.shape[1])


# ─────────────────────────────────────────────────────────────
# FeatureMatcher Tests
# ─────────────────────────────────────────────────────────────

class TestFeatureMatcher:

    def test_flann_matcher_creation(self):
        """FLANN matcher initializes without error."""
        matcher = FeatureMatcher(method="flann", descriptor_type="sift")
        assert matcher is not None

    def test_bf_matcher_creation(self):
        """BFMatcher initializes without error."""
        matcher = FeatureMatcher(method="bf", descriptor_type="sift")
        assert matcher is not None

    def test_invalid_matcher_raises(self):
        """Invalid matcher name raises ValueError."""
        with pytest.raises(ValueError):
            FeatureMatcher(method="unknown_matcher")

    def test_matches_found_between_similar_images(
        self, sift_detector, flann_matcher, synthetic_images
    ):
        """Matcher finds inlier matches between similar images."""
        img1, img2, _ = synthetic_images
        feat1 = sift_detector.detect(img1)
        feat2 = sift_detector.detect(img2)

        result = flann_matcher.match(feat1, feat2, geometric_verify=True)

        assert result is not None
        assert isinstance(result, MatchResult)
        assert result.num_inliers > 10
        assert result.num_inliers <= result.num_matches

    def test_no_match_on_blank_images(
        self, sift_detector, flann_matcher, blank_image
    ):
        """Matcher returns None when images have no features."""
        feat1 = sift_detector.detect(blank_image)
        feat2 = sift_detector.detect(blank_image)
        result = flann_matcher.match(feat1, feat2)
        # Either None or very few matches — both acceptable
        if result is not None:
            assert result.num_inliers < 5

    def test_inlier_ratio_reasonable(
        self, sift_detector, flann_matcher, synthetic_images
    ):
        """Inlier ratio should be > 0.3 for similar image pairs."""
        img1, img2, _ = synthetic_images
        feat1 = sift_detector.detect(img1)
        feat2 = sift_detector.detect(img2)
        result = flann_matcher.match(feat1, feat2, geometric_verify=True)

        assert result is not None
        assert result.inlier_ratio > 0.3

    def test_fundamental_matrix_estimated(
        self, sift_detector, flann_matcher, synthetic_images
    ):
        """Matcher estimates fundamental matrix during geometric verification."""
        img1, img2, _ = synthetic_images
        feat1 = sift_detector.detect(img1)
        feat2 = sift_detector.detect(img2)
        result = flann_matcher.match(feat1, feat2, geometric_verify=True)

        assert result is not None
        assert result.fundamental_matrix is not None
        assert result.fundamental_matrix.shape == (3, 3)

    def test_get_inlier_points_shape(
        self, sift_detector, flann_matcher, synthetic_images
    ):
        """get_inlier_points() returns correctly shaped arrays."""
        img1, img2, _ = synthetic_images
        feat1 = sift_detector.detect(img1)
        feat2 = sift_detector.detect(img2)
        result = flann_matcher.match(feat1, feat2, geometric_verify=True)

        assert result is not None
        pts1, pts2 = result.get_inlier_points()

        assert pts1.shape[1] == 2
        assert pts2.shape[1] == 2
        assert len(pts1) == len(pts2)
        assert len(pts1) == result.num_inliers

    def test_match_all_pairs(self, sift_detector, flann_matcher, synthetic_images):
        """match_all_pairs processes multiple image pairs."""
        img1, img2, _ = synthetic_images
        features = sift_detector.detect_batch([img1, img2])
        pairs = flann_matcher.match_all_pairs(features)

        assert (0, 1) in pairs
        assert pairs[(0, 1)].num_inliers > 0

    def test_no_geometric_verify(
        self, sift_detector, flann_matcher, synthetic_images
    ):
        """Matching without geometric verification still works."""
        img1, img2, _ = synthetic_images
        feat1 = sift_detector.detect(img1)
        feat2 = sift_detector.detect(img2)
        result = flann_matcher.match(feat1, feat2, geometric_verify=False)

        assert result is not None
        # Without RANSAC, no inlier mask
        assert result.num_matches > 0

    def test_orb_bf_matching(self, synthetic_images):
        """ORB features work with BFMatcher."""
        img1, img2, _ = synthetic_images
        detector = FeatureDetector(method="orb", max_keypoints=500)
        matcher  = FeatureMatcher(
            method="bf",
            descriptor_type="orb",
            min_matches=10
        )

        feat1 = detector.detect(img1)
        feat2 = detector.detect(img2)
        result = matcher.match(feat1, feat2)

        assert result is not None
        assert result.num_matches > 0