"""
3D point triangulation from two or more views.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def triangulate_points(
    P1: np.ndarray,
    P2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray
) -> np.ndarray:
    """
    Triangulate 3D points from two projection matrices and 2D correspondences.
    
    Args:
        P1: 3x4 projection matrix for camera 1
        P2: 3x4 projection matrix for camera 2
        pts1: Nx2 points in image 1
        pts2: Nx2 points in image 2
    
    Returns:
        Nx3 array of 3D points
    """
    pts1_h = pts1.T.astype(np.float64)  # 2xN
    pts2_h = pts2.T.astype(np.float64)  # 2xN
    
    points_4d = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)  # 4xN
    
    # Convert from homogeneous
    points_3d = points_4d[:3] / points_4d[3:4]  # 3xN
    
    return points_3d.T  # Nx3


def filter_triangulated_points(
    points_3d: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    reprojection_threshold: float = 4.0,
    min_angle_deg: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter triangulated points by reprojection error and triangulation angle.
    
    Returns:
        (filtered_points_3d, valid_mask)
    """
    n = len(points_3d)
    valid = np.ones(n, dtype=bool)
    
    # 1. Check points are in front of both cameras
    # Camera 1
    R1 = P1[:3, :3]
    t1 = P1[:3, 3]
    cam1_z = (R1[2:3] @ points_3d.T + t1[2]).ravel()
    valid &= cam1_z > 0
    
    # Camera 2
    R2 = P2[:3, :3]
    t2 = P2[:3, 3]
    cam2_z = (R2[2:3] @ points_3d.T + t2[2]).ravel()
    valid &= cam2_z > 0
    
    # 2. Reprojection error
    pts_h = np.hstack([points_3d, np.ones((n, 1))])
    
    proj1 = (P1 @ pts_h.T).T
    proj1 = proj1[:, :2] / proj1[:, 2:3]
    err1 = np.linalg.norm(proj1 - pts1, axis=1)
    
    proj2 = (P2 @ pts_h.T).T
    proj2 = proj2[:, :2] / proj2[:, 2:3]
    err2 = np.linalg.norm(proj2 - pts2, axis=1)
    
    valid &= err1 < reprojection_threshold
    valid &= err2 < reprojection_threshold
    
    # 3. Triangulation angle check
    # Camera centers
    C1 = -np.linalg.inv(P1[:3, :3]) @ P1[:3, 3]
    C2 = -np.linalg.inv(P2[:3, :3]) @ P2[:3, 3]
    
    for i in range(n):
        if not valid[i]:
            continue
        v1 = points_3d[i] - C1
        v2 = points_3d[i] - C2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.degrees(np.arccos(cos_angle))
        if angle < min_angle_deg:
            valid[i] = False
    
    return points_3d[valid], valid


def compute_reprojection_error(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray
) -> np.ndarray:
    """
    Compute per-point reprojection error.
    
    Returns:
        Array of reprojection errors (pixels)
    """
    rvec, _ = cv2.Rodrigues(R)
    projected, _ = cv2.projectPoints(
        points_3d.reshape(-1, 1, 3),
        rvec, t, K,
        distCoeffs=None
    )
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(projected - points_2d, axis=1)
    return errors