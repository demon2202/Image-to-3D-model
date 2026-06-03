"""
Dense stereo matching for depth estimation.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class DenseStereo:
    """
    Dense depth estimation using Semi-Global Block Matching (SGBM).
    """
    
    def __init__(
        self,
        num_disparities: int = 128,
        block_size: int = 5,
        min_disparity: int = 0
    ):
        self.num_disparities = num_disparities
        self.block_size = block_size
        self.min_disparity = min_disparity
        
        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=min_disparity,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * 3 * block_size ** 2,
            P2=32 * 3 * block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
    
    def compute_disparity(
        self,
        img_left: np.ndarray,
        img_right: np.ndarray
    ) -> np.ndarray:
        """Compute disparity map from rectified stereo pair."""
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY) if len(img_left.shape) == 3 else img_left
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY) if len(img_right.shape) == 3 else img_right
        
        disparity = self.sgbm.compute(gray_left, gray_right).astype(np.float32) / 16.0
        
        # Invalidate negative disparities
        disparity[disparity < 0] = 0
        
        return disparity
    
    def disparity_to_depth(
        self,
        disparity: np.ndarray,
        baseline: float,
        focal_length: float
    ) -> np.ndarray:
        """Convert disparity to depth map. depth = baseline * focal / disparity."""
        depth = np.zeros_like(disparity)
        valid = disparity > 0
        depth[valid] = baseline * focal_length / disparity[valid]
        return depth
    
    def depth_to_pointcloud(
        self,
        depth: np.ndarray,
        K: np.ndarray,
        color_image: Optional[np.ndarray] = None,
        R: np.ndarray = np.eye(3),
        t: np.ndarray = np.zeros((3, 1)),
        max_depth: float = 100.0
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Convert depth map to 3D point cloud.
        
        Returns:
            (Nx3 points, Nx3 colors or None)
        """
        h, w = depth.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        # Create mesh grid
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # Valid depth mask
        valid = (depth > 0) & (depth < max_depth)
        
        # Back-project to 3D
        z = depth[valid]
        x = (u[valid] - cx) * z / fx
        y = (v[valid] - cy) * z / fy
        
        points_cam = np.stack([x, y, z], axis=1)  # Nx3
        
        # Transform to world coordinates
        points_world = (R.T @ (points_cam.T - t.reshape(3, 1))).T
        
        # Colors
        colors = None
        if color_image is not None:
            if len(color_image.shape) == 3:
                colors = color_image[valid][:, ::-1].astype(np.float64) / 255.0
            else:
                gray = color_image[valid].astype(np.float64) / 255.0
                colors = np.stack([gray, gray, gray], axis=1)
        
        return points_world, colors


def rectify_stereo_pair(
    img1: np.ndarray,
    img2: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Rectify a stereo pair given relative pose.
    
    Returns:
        (rectified_img1, rectified_img2, Q_matrix, baseline)
    """
    h, w = img1.shape[:2]
    dist_coeffs = np.zeros(5)
    
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K, dist_coeffs, K, dist_coeffs,
        (w, h), R, t,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0
    )
    
    map1x, map1y = cv2.initUndistortRectifyMap(K, dist_coeffs, R1, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K, dist_coeffs, R2, P2, (w, h), cv2.CV_32FC1)
    
    rect1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
    rect2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
    
    baseline = np.linalg.norm(t)
    
    return rect1, rect2, Q, baseline