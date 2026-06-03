"""
Evaluation — Metrics and Comparisons
"""

from .metrics import sfm_reprojection_error, nerf_image_metrics
from .compare import (
    compare_images,
    compare_rgb_gt_pred,
    compare_point_clouds,
    compare_camera_trajectories,
    plot_training_curves,
    generate_comparison_report,
)

__all__ = [
    "sfm_reprojection_error",
    "nerf_image_metrics",
    "compare_images",
    "compare_rgb_gt_pred",
    "compare_point_clouds",
    "compare_camera_trajectories",
    "plot_training_curves",
    "generate_comparison_report",
]