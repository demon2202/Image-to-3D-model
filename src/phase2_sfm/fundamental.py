"""
Fundamental and Essential matrix estimation.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def estimate_fundamental_matrix(
    pts1: np.ndarray,
    pts2: np.ndarray,
    method: int = cv2.FM_RANSAC,
    ransac_thresh: float = 1.0,
    confidence: float = 0.999
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Estimate fundamental matrix from point correspondences.
    
    Returns:
        (F, inlier_mask) or (None, None) if estimation fails.
    """
    if len(pts1) < 8:
        return None, None
    
    F, mask = cv2.findFundamentalMat(pts1, pts2, method, ransac_thresh, confidence)
    
    if F is None or F.shape != (3, 3):
        return None, None
    
    return F, mask


def estimate_essential_matrix(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    ransac_thresh: float = 1.0,
    confidence: float = 0.999
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Estimate essential matrix from point correspondences and intrinsics.
    
    Args:
        pts1, pts2: Nx2 point correspondences (pixel coordinates)
        K: 3x3 camera intrinsic matrix
        
    Returns:
        (E, inlier_mask) or (None, None)
    """
    if len(pts1) < 5:
        return None, None
    
    E, mask = cv2.findEssentialMat(
        pts1, pts2, K,
        method=cv2.RANSAC,
        prob=confidence,
        threshold=ransac_thresh
    )
    
    if E is None or E.shape != (3, 3):
        return None, None
    
    return E, mask


def decompose_essential_matrix(
    E: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose essential matrix into rotation and translation.
    Uses cheirality check to select the correct solution.
    
    Returns:
        (R, t, mask) — R is 3x3, t is 3x1, mask indicates valid points
    """
    num_inliers, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    return R, t, mask