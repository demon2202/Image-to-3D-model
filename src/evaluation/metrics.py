"""
Unified evaluation metrics module.
"""

import numpy as np
from typing import Dict, List, Tuple


def sfm_reprojection_error(
    points_3d: np.ndarray,
    observations: List[Tuple[int, int, np.ndarray]],
    camera_poses: Dict[int, Tuple[np.ndarray, np.ndarray]],
    K: np.ndarray
) -> Dict[str, float]:
    """
    Comprehensive SfM evaluation.
    """
    import cv2
    
    errors = []
    per_camera_errors = {}
    
    for cam_idx, pt_idx, pt_2d in observations:
        if cam_idx not in camera_poses or pt_idx >= len(points_3d):
            continue
        
        R, t = camera_poses[cam_idx]
        rvec, _ = cv2.Rodrigues(R)
        
        projected, _ = cv2.projectPoints(
            points_3d[pt_idx:pt_idx+1].reshape(1, 1, 3).astype(np.float64),
            rvec, t.astype(np.float64), K, None
        )
        
        err = np.linalg.norm(projected.reshape(2) - pt_2d)
        errors.append(err)
        
        if cam_idx not in per_camera_errors:
            per_camera_errors[cam_idx] = []
        per_camera_errors[cam_idx].append(err)
    
    return {
        'mean_error': np.mean(errors) if errors else float('inf'),
        'median_error': np.median(errors) if errors else float('inf'),
        'max_error': np.max(errors) if errors else float('inf'),
        'std_error': np.std(errors) if errors else float('inf'),
        'num_observations': len(errors),
        'per_camera_mean': {
            k: np.mean(v) for k, v in per_camera_errors.items()
        }
    }


def nerf_image_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """
    Compute all NeRF image quality metrics.
    """
    from skimage.metrics import structural_similarity
    
    mse = np.mean((pred - target) ** 2)
    psnr = -10 * np.log10(mse) if mse > 0 else float('inf')
    ssim = structural_similarity(pred, target, channel_axis=2, data_range=1.0)
    
    # LPIPS requires a separate model; placeholder
    return {
        'psnr': psnr,
        'ssim': ssim,
        'mse': mse,
    }