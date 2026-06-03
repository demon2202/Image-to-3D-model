"""
Camera intrinsic/extrinsic utility classes and functions.
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class Camera:
    """
    Represents a camera with intrinsics and optional extrinsics.

    Convention:
        - Extrinsics (R, t) map world -> camera: x_cam = R @ x_world + t
        - c2w (camera-to-world 4x4) is the inverse: x_world = R^T @ (x_cam - t)
    """
    # Intrinsics
    fx: float = 800.0
    fy: float = 800.0
    cx: float = 320.0
    cy: float = 240.0
    width: int = 640
    height: int = 480

    # Distortion coefficients [k1, k2, p1, p2, k3]
    dist_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(5))

    # Extrinsics (world-to-camera)
    R: Optional[np.ndarray] = None   # 3x3 rotation
    t: Optional[np.ndarray] = None   # 3x1 translation

    def __post_init__(self):
        if self.dist_coeffs is None:
            self.dist_coeffs = np.zeros(5)
        if self.R is None:
            self.R = np.eye(3)
        if self.t is None:
            self.t = np.zeros((3, 1))

    @property
    def K(self) -> np.ndarray:
        """3x3 intrinsic matrix."""
        return np.array([
            [self.fx,    0,    self.cx],
            [0,       self.fy, self.cy],
            [0,          0,       1  ]
        ], dtype=np.float64)

    @property
    def P(self) -> np.ndarray:
        """3x4 projection matrix P = K @ [R | t]."""
        return self.K @ np.hstack([self.R, self.t.reshape(3, 1)])

    @property
    def c2w(self) -> np.ndarray:
        """4x4 camera-to-world matrix."""
        mat = np.eye(4)
        mat[:3, :3] = self.R.T
        mat[:3, 3] = (-self.R.T @ self.t.ravel())
        return mat

    @property
    def center(self) -> np.ndarray:
        """Camera center in world coordinates (3,)."""
        return (-self.R.T @ self.t.ravel())

    @classmethod
    def from_K(
        cls,
        K: np.ndarray,
        width: int,
        height: int,
        R: Optional[np.ndarray] = None,
        t: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None
    ) -> "Camera":
        """Construct Camera from intrinsic matrix."""
        return cls(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            width=width,
            height=height,
            dist_coeffs=dist_coeffs if dist_coeffs is not None else np.zeros(5),
            R=R if R is not None else np.eye(3),
            t=t.reshape(3, 1) if t is not None else np.zeros((3, 1))
        )

    @classmethod
    def from_fov(
        cls,
        fov_x_deg: float,
        width: int,
        height: int
    ) -> "Camera":
        """Construct Camera from horizontal field of view."""
        fx = (width / 2) / np.tan(np.radians(fov_x_deg / 2))
        fy = fx
        return cls(
            fx=fx, fy=fy,
            cx=width / 2, cy=height / 2,
            width=width, height=height
        )

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Project Nx3 world points to Nx2 image coordinates.
        Includes distortion if dist_coeffs are set.
        """
        rvec, _ = cv2.Rodrigues(self.R)
        projected, _ = cv2.projectPoints(
            points_3d.reshape(-1, 1, 3).astype(np.float64),
            rvec,
            self.t.astype(np.float64),
            self.K,
            self.dist_coeffs
        )
        return projected.reshape(-1, 2)

    def backproject(self, points_2d: np.ndarray, depth: float = 1.0) -> np.ndarray:
        """
        Back-project Nx2 image points to Nx3 camera-space rays
        at the given depth.
        """
        pts = points_2d.reshape(-1, 2).astype(np.float64)
        # Undistort first
        undist = cv2.undistortPoints(
            pts.reshape(-1, 1, 2), self.K, self.dist_coeffs, P=self.K
        ).reshape(-1, 2)

        x = (undist[:, 0] - self.cx) / self.fx
        y = (undist[:, 1] - self.cy) / self.fy
        z = np.ones(len(x))
        rays = np.stack([x, y, z], axis=1)

        # Normalize and scale by depth
        rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
        return rays * depth

    def is_point_in_frustum(
        self,
        point_world: np.ndarray,
        near: float = 0.1,
        far: float = 100.0
    ) -> bool:
        """Check if a world point is inside this camera's view frustum."""
        # Transform to camera space
        p_cam = self.R @ point_world + self.t.ravel()
        z = p_cam[2]

        if z < near or z > far:
            return False

        # Project and check image bounds
        x_img = self.fx * p_cam[0] / z + self.cx
        y_img = self.fy * p_cam[1] / z + self.cy

        return (0 <= x_img < self.width) and (0 <= y_img < self.height)

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """Remove lens distortion from an image."""
        return cv2.undistort(image, self.K, self.dist_coeffs)

    def __repr__(self) -> str:
        return (f"Camera(fx={self.fx:.1f}, fy={self.fy:.1f}, "
                f"cx={self.cx:.1f}, cy={self.cy:.1f}, "
                f"{self.width}x{self.height})")


def build_projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build 3x4 projection matrix P = K @ [R | t]."""
    t = t.reshape(3, 1)
    return K @ np.hstack([R, t])


def camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Compute camera center C = -R^T @ t in world coordinates."""
    return (-R.T @ t.ravel())


def decompose_projection(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose 3x4 projection matrix into K, R, t using RQ decomposition.

    Returns:
        (K, R, t)
    """
    M = P[:3, :3]
    # RQ decomposition via QR on transposed
    Q, R = np.linalg.qr(np.linalg.inv(M).T)
    K_inv_T = R.T
    R_mat = Q.T

    K = np.linalg.inv(K_inv_T)
    # Normalize K so K[2,2] = 1
    K = K / K[2, 2]

    # Ensure positive diagonal in K
    sign_diag = np.diag(np.sign(np.diag(K)))
    K = K @ sign_diag
    R_mat = sign_diag @ R_mat

    t_vec = np.linalg.inv(K) @ P[:, 3]

    return K, R_mat, t_vec


def estimate_intrinsics_from_fov(
    image_width: int,
    image_height: int,
    fov_horizontal_deg: float = 60.0
) -> np.ndarray:
    """
    Estimate camera intrinsic matrix from known/assumed FOV.
    Useful when calibration data is not available.
    """
    fx = (image_width / 2.0) / np.tan(np.radians(fov_horizontal_deg / 2.0))
    fy = fx  # Assume square pixels
    cx = image_width / 2.0
    cy = image_height / 2.0

    return np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float64)


def interpolate_poses(
    pose1: Tuple[np.ndarray, np.ndarray],
    pose2: Tuple[np.ndarray, np.ndarray],
    alpha: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Linearly interpolate between two camera poses.

    Args:
        pose1, pose2: (R, t) tuples
        alpha: interpolation factor [0, 1]

    Returns:
        Interpolated (R, t)
    """
    from scipy.spatial.transform import Rotation, Slerp

    R1, t1 = pose1
    R2, t2 = pose2

    # SLERP for rotation
    rots = Rotation.from_matrix([R1, R2])
    slerp = Slerp([0, 1], rots)
    R_interp = slerp(alpha).as_matrix()

    # Linear interpolation for translation
    t_interp = (1 - alpha) * t1.ravel() + alpha * t2.ravel()

    return R_interp, t_interp.reshape(3, 1)