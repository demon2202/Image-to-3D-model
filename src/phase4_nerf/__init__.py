"""
Phase 4 — Neural Radiance Fields (NeRF)
"""

from .model   import NeRFModel, NeRFSmall, PositionalEncoding
from .render  import sample_along_rays, volume_render, render_rays
from .dataset import NeRFSyntheticDataset, SfMDataset, get_rays
from .train   import NeRFTrainer
from .evaluate import compute_psnr, compute_ssim, render_full_image
from .utils   import (
    positional_encoding_dim,
    encode_position,
    normalize_scene,
    generate_360_path,
    generate_spiral_path,
    interpolate_poses_slerp,
    compute_near_far_from_poses,
    CheckpointManager,
    visualize_rays,
    visualize_depth_map,
    save_render_grid,
    print_model_summary,
)

__all__ = [
    # Model
    "NeRFModel",
    "NeRFSmall",
    "PositionalEncoding",
    # Rendering
    "sample_along_rays",
    "volume_render",
    "render_rays",
    # Dataset
    "NeRFSyntheticDataset",
    "SfMDataset",
    "get_rays",
    # Training
    "NeRFTrainer",
    # Evaluation
    "compute_psnr",
    "compute_ssim",
    "render_full_image",
    # Utils
    "positional_encoding_dim",
    "encode_position",
    "normalize_scene",
    "generate_360_path",
    "generate_spiral_path",
    "interpolate_poses_slerp",
    "compute_near_far_from_poses",
    "CheckpointManager",
    "visualize_rays",
    "visualize_depth_map",
    "save_render_grid",
    "print_model_summary",
]