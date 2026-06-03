"""
Feature matching with FLANN/BFMatcher, Lowe's ratio test, and RANSAC geometric verification.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
from .detector import FeatureData


@dataclass
class MatchResult:
    """Container for matching results between two images."""
    matches: List[cv2.DMatch]
    inlier_mask: Optional[np.ndarray]    # Boolean mask after RANSAC
    pts1: np.ndarray                      # Matched points in image 1 (Nx2)
    pts2: np.ndarray                      # Matched points in image 2 (Nx2)
    fundamental_matrix: Optional[np.ndarray]
    
    @property
    def num_matches(self) -> int:
        return len(self.matches)
    
    @property
    def num_inliers(self) -> int:
        if self.inlier_mask is None:
            return self.num_matches
        return int(np.sum(self.inlier_mask))
    
    @property
    def inlier_ratio(self) -> float:
        if self.num_matches == 0:
            return 0.0
        return self.num_inliers / self.num_matches
    
    def get_inlier_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return only inlier point correspondences."""
        if self.inlier_mask is None:
            return self.pts1, self.pts2
        mask = self.inlier_mask.ravel().astype(bool)
        return self.pts1[mask], self.pts2[mask]


class FeatureMatcher:
    """
    Robust feature matcher with ratio test and geometric verification.
    """
    
    def __init__(
        self,
        method: str = "flann",
        ratio_threshold: float = 0.75,
        ransac_threshold: float = 1.0,
        min_matches: int = 30,
        descriptor_type: str = "sift"
    ):
        self.ratio_threshold = ratio_threshold
        self.ransac_threshold = ransac_threshold
        self.min_matches = min_matches
        self.descriptor_type = descriptor_type.lower()
        self._matcher = self._create_matcher(method.lower())
    
    def _create_matcher(self, method: str):
        if method == "flann":
            if self.descriptor_type == "sift":
                # SIFT uses float descriptors
                index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
                search_params = dict(checks=50)
            else:
                # ORB uses binary descriptors
                index_params = dict(
                    algorithm=6,  # FLANN_INDEX_LSH
                    table_number=6,
                    key_size=12,
                    multi_probe_level=1
                )
                search_params = dict(checks=50)
            return cv2.FlannBasedMatcher(index_params, search_params)
        elif method == "bf":
            if self.descriptor_type == "sift":
                return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            else:
                return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            raise ValueError(f"Unknown matcher: {method}")
    
    def match(
        self,
        feat1: FeatureData,
        feat2: FeatureData,
        geometric_verify: bool = True
    ) -> Optional[MatchResult]:
        """
        Match features between two images with ratio test and optional RANSAC.
        
        Args:
            feat1: Features from image 1
            feat2: Features from image 2
            geometric_verify: Whether to apply RANSAC geometric verification
            
        Returns:
            MatchResult or None if insufficient matches
        """
        if feat1.descriptors.size == 0 or feat2.descriptors.size == 0:
            return None
        
        # Ensure correct dtype for FLANN
        desc1 = feat1.descriptors.astype(np.float32) if self.descriptor_type == "sift" else feat1.descriptors
        desc2 = feat2.descriptors.astype(np.float32) if self.descriptor_type == "sift" else feat2.descriptors
        
        # KNN matching with k=2 for ratio test
        try:
            raw_matches = self._matcher.knnMatch(desc1, desc2, k=2)
        except cv2.error:
            return None
        
        # Lowe's ratio test
        good_matches = []
        for match_pair in raw_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < self.min_matches:
            return None
        
        # Extract matched point coordinates
        pts1 = np.array([feat1.keypoints[m.queryIdx].pt for m in good_matches], dtype=np.float64)
        pts2 = np.array([feat2.keypoints[m.trainIdx].pt for m in good_matches], dtype=np.float64)
        
        # Geometric verification with RANSAC
        F = None
        inlier_mask = None
        
        if geometric_verify and len(good_matches) >= 8:
            F, inlier_mask = cv2.findFundamentalMat(
                pts1, pts2,
                method=cv2.FM_RANSAC,
                ransacReprojThreshold=self.ransac_threshold,
                confidence=0.999
            )
            
            if inlier_mask is not None:
                num_inliers = np.sum(inlier_mask)
                if num_inliers < self.min_matches:
                    return None
        
        return MatchResult(
            matches=good_matches,
            inlier_mask=inlier_mask,
            pts1=pts1,
            pts2=pts2,
            fundamental_matrix=F
        )
    
    def match_all_pairs(
        self,
        features: List[FeatureData],
        image_names: Optional[List[str]] = None
    ) -> dict:
        """
        Match all image pairs. Returns dict keyed by (i, j) tuples.
        """
        n = len(features)
        pair_matches = {}
        
        for i in range(n):
            for j in range(i + 1, n):
                result = self.match(features[i], features[j])
                if result is not None:
                    pair_matches[(i, j)] = result
                    if image_names:
                        print(f"  {image_names[i]} <-> {image_names[j]}: "
                              f"{result.num_inliers}/{result.num_matches} inliers "
                              f"({result.inlier_ratio:.1%})")
        
        return pair_matches