"""
Visualization utilities for feature matches.
"""

import cv2
import numpy as np
from typing import Optional
from .detector import FeatureData
from .matcher import MatchResult


def draw_matches(
    img1: np.ndarray,
    img2: np.ndarray,
    feat1: FeatureData,
    feat2: FeatureData,
    match_result: MatchResult,
    max_display: int = 100,
    show_inliers_only: bool = True,
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Draw feature matches between two images.
    """
    if show_inliers_only and match_result.inlier_mask is not None:
        mask = match_result.inlier_mask.ravel().astype(bool)
        display_matches = [m for m, valid in zip(match_result.matches, mask) if valid]
    else:
        display_matches = match_result.matches
    
    # Subsample for display
    if len(display_matches) > max_display:
        indices = np.random.choice(len(display_matches), max_display, replace=False)
        display_matches = [display_matches[i] for i in indices]
    
    result = cv2.drawMatches(
        img1, feat1.keypoints,
        img2, feat2.keypoints,
        display_matches, None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    if output_path:
        cv2.imwrite(output_path, result)
    
    return result


def draw_keypoints(
    image: np.ndarray,
    features: FeatureData,
    output_path: Optional[str] = None
) -> np.ndarray:
    """Draw detected keypoints on an image."""
    result = cv2.drawKeypoints(
        image, features.keypoints, None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    if output_path:
        cv2.imwrite(output_path, result)
    
    return result