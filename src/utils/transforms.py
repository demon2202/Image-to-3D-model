"""
Coordinate transformation utilities for 3D vision.
"""

import numpy as np
from typing import Tuple


# ─────────────────────────────────────────────
# Basic rotations
# ─────────────────────────────────────────────

def rotation_matrix_x(angle_deg: float) -> np.ndarray:
    """3x3 rotation matrix around X-axis."""
    a = np.radians(angle_deg)
    return np.array([
        [1,       0,        0],
        [0, np.cos(a), -np.sin(a)],
        [0, np.sin(a),  np.cos(a)]
    ], dtype=np.float64)


def rotation_matrix_y(angle_deg: float) -> np.ndarray:
    """3x3 rotation matrix around Y-axis."""
    a = np.radians(angle_deg)
    return np.array([
        [ np.cos(a), 0, np.sin(a)],
        [       0,   1,        0 ],
        [-np.sin(a), 0, np.cos(a)]
    ], dtype=np.float64)


def rotation_matrix_z(angle_deg: float) -> np.ndarray:
    """3x3 rotation matrix around Z-axis."""
    a = np.radians(angle_deg)
    return np.array([
        [np.cos(a), -np.sin(a), 0],
        [np.sin(a),  np.cos(a), 0],
        [       0,          0,  1]
    ], dtype=np.float64)


# ─────────────────────────────────────────────
# World ↔ Camera transforms
# ─────────────────────────────────────────────

def world_to_camera(
    points_world: np.ndarray,
    R: np.ndarray,
    t: np.ndarray
) -> np.ndarray:
    """
    Transform Nx3 world points to camera coordinates.
    x_cam = R @ x_world + t
    """
    t = t.reshape(3, 1)
    return (R @ points_world.T + t).T


def camera_to_world(
    points_cam: np.ndarray,
    R: np.ndarray,
    t: np.ndarray
) -> np.ndarray:
    """
    Transform Nx3 camera points to world coordinates.
    x_world = R^T @ (x_cam - t)
    """
    t = t.reshape(3)
    return (R.T @ (points_cam - t).T).T


def c2w_to_Rt(c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert 4x4 camera-to-world matrix to (R, t) world-to-camera convention.
    """
    R = c2w[:3, :3].T
    t = -c2w[:3, :3].T @ c2w[:3, 3]
    return R, t.reshape(3, 1)


def Rt_to_c2w(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Convert (R, t) world-to-camera to 4x4 camera-to-world matrix.
    """
    c2w = np.eye(4)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t.ravel()
    return c2w


# ─────────────────────────────────────────────
# Point normalization
# ─────────────────────────────────────────────

def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize a set of 2D or 3D points to have zero mean and unit RMS distance.
    Used in DLT / 8-point algorithm for numerical stability.

    Returns:
        (normalized_points, T) where T is the 3x3 or 4x4 normalization matrix
    """
    n, d = points.shape
    centroid = points.mean(axis=0)
    centered = points - centroid

    if d == 2:
        avg_dist = np.sqrt((centered ** 2).sum(axis=1)).mean()
        scale = np.sqrt(2) / (avg_dist + 1e-8)
        T = np.array([
            [scale,     0, -scale * centroid[0]],
            [    0, scale, -scale * centroid[1]],
            [    0,     0,                   1 ]
        ], dtype=np.float64)
    else:  # 3D
        avg_dist = np.sqrt((centered ** 2).sum(axis=1)).mean()
        scale = np.sqrt(3) / (avg_dist + 1e-8)
        T = np.array([
            [scale,     0,     0, -scale * centroid[0]],
            [    0, scale,     0, -scale * centroid[1]],
            [    0,     0, scale, -scale * centroid[2]],
            [    0,     0,     0,                   1 ]
        ], dtype=np.float64)

    pts_h = np.hstack([points, np.ones((n, 1))])
    pts_norm = (T @ pts_h.T).T[:, :d]
    return pts_norm, T


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """
    3x3 skew-symmetric matrix for vector v.
    Used in cross-product: [v]x @ u = v × u
    """
    return np.array([
        [    0, -v[2],  v[1]],
        [ v[2],     0, -v[0]],
        [-v[1],  v[0],     0]
    ], dtype=np.float64)


def homogeneous(points: np.ndarray) -> np.ndarray:
    """Add homogeneous coordinate: (N, D) -> (N, D+1)."""
    return np.hstack([points, np.ones((len(points), 1))])


def dehomogeneous(points: np.ndarray) -> np.ndarray:
    """Remove homogeneous coordinate (divide by last): (N, D+1) -> (N, D)."""
    return points[:, :-1] / points[:, -1:]


# ─────────────────────────────────────────────
# Coordinate system conversions
# ─────────────────────────────────────────────

def opencv_to_opengl(R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert camera pose from OpenCV convention (Y down, Z forward)
    to OpenGL convention (Y up, Z backward). Used for NeRF compatibility.
    """
    flip = np.diag([1, -1, -1])
    R_gl = flip @ R
    t_gl = flip @ t.ravel()
    return R_gl, t_gl.reshape(3, 1)


def opengl_to_opencv(R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse of opencv_to_opengl."""
    return opencv_to_opengl(R, t)   # flip is its own inverse


def nerf_to_opencv(c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert NeRF's camera-to-world (OpenGL convention) to OpenCV (R, t).
    """
    # Flip Y and Z axes
    flip = np.diag([1, -1, -1, 1])
    c2w_cv = c2w @ flip
    R = c2w_cv[:3, :3].T
    t = -c2w_cv[:3, :3].T @ c2w_cv[:3, 3]
    return R, t.reshape(3, 1)


def compute_scene_scale(camera_centers: np.ndarray) -> float:
    """
    Estimate scene scale from camera centers as the mean pairwise distance.
    Useful for normalizing scene before NeRF training.
    """
    if len(camera_centers) < 2:
        return 1.0
    dists = []
    for i in range(len(camera_centers)):
        for j in range(i + 1, len(camera_centers)):
            dists.append(np.linalg.norm(camera_centers[i] - camera_centers[j]))
    return float(np.mean(dists))