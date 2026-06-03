"""
NeRF utility functions.

Covers:
  - Positional encoding helpers
  - Ray generation helpers
  - Scene normalization (NeRF++ style)
  - Spherical harmonics for view-dependent color
  - Camera path generation for video rendering
  - Checkpoint management
  - Debug visualization helpers
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Callable


# ─────────────────────────────────────────────────────────────
# Positional Encoding Utilities
# ─────────────────────────────────────────────────────────────

def positional_encoding_dim(num_freqs: int, include_input: bool = True, input_dim: int = 3) -> int:
    """
    Compute output dimension of positional encoding.

    For input_dim=3 (xyz):
        output = 3 + 3 * num_freqs * 2  (if include_input)
        output = 3 * num_freqs * 2      (if not include_input)
    """
    enc_dim = input_dim * num_freqs * 2
    if include_input:
        enc_dim += input_dim
    return enc_dim


def encode_position(
    x: torch.Tensor,
    num_freqs: int = 10,
    include_input: bool = True,
    log_sampling: bool = True
) -> torch.Tensor:
    """
    Standalone positional encoding function (functional form).

    Args:
        x          : (..., D) input tensor
        num_freqs  : number of frequency bands L
        include_input: prepend original x to output
        log_sampling : use 2^[0..L-1] frequencies (original NeRF paper)
                       vs linear sampling

    Returns:
        (..., encoded_dim) tensor
    """
    if log_sampling:
        freqs = 2.0 ** torch.linspace(
            0, num_freqs - 1, num_freqs,
            device=x.device, dtype=x.dtype
        )
    else:
        freqs = torch.linspace(
            1.0, 2 ** (num_freqs - 1), num_freqs,
            device=x.device, dtype=x.dtype
        )

    parts = []
    if include_input:
        parts.append(x)

    for freq in freqs:
        parts.append(torch.sin(freq * x))
        parts.append(torch.cos(freq * x))

    return torch.cat(parts, dim=-1)


def integrated_positional_encoding(
    means: torch.Tensor,
    covs: torch.Tensor,
    num_freqs: int = 10
) -> torch.Tensor:
    """
    Integrated Positional Encoding (Mip-NeRF style).
    Accounts for the footprint of a conical frustum instead of a single ray.

    Args:
        means : (..., 3) mean positions of Gaussian samples
        covs  : (..., 3) diagonal covariance (variance per axis)
        num_freqs: number of frequency bands

    Returns:
        (..., 6*num_freqs) encoded tensor (sin + cos, variance-weighted)
    """
    freqs = (2.0 ** torch.arange(num_freqs, device=means.device, dtype=means.dtype))

    # Scale means and apply sin/cos
    means_enc = means[..., None] * freqs                  # (..., 3, L)
    covs_enc  = covs[..., None]  * (freqs ** 2)          # (..., 3, L)

    # Expected value of sin/cos under Gaussian:
    # E[sin(omega*x)] = sin(omega*mu) * exp(-omega^2 * sigma^2 / 2)
    decay = torch.exp(-0.5 * covs_enc)
    sin_enc = torch.sin(means_enc) * decay
    cos_enc = torch.cos(means_enc) * decay

    encoded = torch.cat([
        sin_enc.flatten(-2),   # (..., 3*L)
        cos_enc.flatten(-2)    # (..., 3*L)
    ], dim=-1)

    return encoded


# ─────────────────────────────────────────────────────────────
# Scene Normalization
# ─────────────────────────────────────────────────────────────

def normalize_scene(
    poses: np.ndarray,       # (N, 4, 4) c2w matrices
    points_3d: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, dict]:
    """
    Normalize scene so that camera centers lie on the unit sphere.
    This is the standard normalization for NeRF training.

    Algorithm (LLFF / NeRF paper convention):
      1. Compute centroid of camera centers
      2. Compute scale = max distance from centroid
      3. Translate and scale all poses

    Args:
        poses     : (N, 4, 4) camera-to-world matrices
        points_3d : optional (M, 3) sparse 3D points to transform

    Returns:
        (normalized_poses, transform_dict) where transform_dict contains
        the normalization parameters to undo the transform
    """
    # Camera centers from c2w matrices
    centers = poses[:, :3, 3]    # (N, 3)

    centroid = centers.mean(axis=0)
    centered = centers - centroid

    # Scale: 90th percentile distance to be robust to outliers
    dists  = np.linalg.norm(centered, axis=1)
    scale  = float(np.percentile(dists, 90))
    if scale < 1e-8:
        scale = 1.0

    # Apply normalization to poses
    norm_poses = poses.copy()
    norm_poses[:, :3, 3] = (centers - centroid) / scale

    transform = {
        "centroid": centroid.tolist(),
        "scale": scale
    }

    # Transform 3D points if provided
    if points_3d is not None:
        norm_pts = (points_3d - centroid) / scale
        return norm_poses, transform, norm_pts

    return norm_poses, transform


def denormalize_scene(
    poses: np.ndarray,
    transform: dict
) -> np.ndarray:
    """
    Undo scene normalization.
    """
    centroid = np.array(transform["centroid"])
    scale    = float(transform["scale"])

    denorm_poses = poses.copy()
    denorm_poses[:, :3, 3] = poses[:, :3, 3] * scale + centroid
    return denorm_poses


def compute_near_far_from_poses(
    poses: np.ndarray,
    points_3d: Optional[np.ndarray] = None,
    percentile_near: float = 0.1,
    percentile_far: float = 99.9
) -> Tuple[float, float]:
    """
    Automatically compute near and far plane values from camera
    positions and optionally the sparse 3D point cloud.

    Args:
        poses         : (N, 4, 4) camera-to-world matrices
        points_3d     : optional (M, 3) sparse 3D points
        percentile_near/far: percentile of depth values to use

    Returns:
        (near, far) plane distances
    """
    centers = poses[:, :3, 3]

    if points_3d is not None and len(points_3d) > 0:
        # Compute depth of each 3D point from each camera
        depths = []
        for i in range(len(poses)):
            c2w = poses[i]
            w2c = np.linalg.inv(c2w)
            pts_cam = (w2c[:3, :3] @ points_3d.T + w2c[:3, 3:4]).T
            z = pts_cam[:, 2]
            depths.extend(z[z > 0].tolist())

        if depths:
            near = max(0.01, float(np.percentile(depths, percentile_near)))
            far  = float(np.percentile(depths, percentile_far))
            return near, far

    # Fallback: use camera spacing
    dists = np.linalg.norm(centers - centers.mean(axis=0), axis=1)
    near = max(0.01, float(dists.min()) * 0.1)
    far  = float(dists.max()) * 5.0

    return near, far


# ─────────────────────────────────────────────────────────────
# Camera Path Generation
# ─────────────────────────────────────────────────────────────

def generate_spiral_path(
    poses: np.ndarray,
    n_frames: int = 120,
    n_rotations: float = 2.0,
    z_variation: float = 0.5,
    z_phase: float = 0.0,
    rads_scale: float = 0.75
) -> np.ndarray:
    """
    Generate a smooth spiral camera path for video rendering.
    Based on the NeRF LLFF spiral path generation.

    Args:
        poses      : (N, 4, 4) training camera poses
        n_frames   : number of frames to generate
        n_rotations: number of full rotations around scene
        z_variation: up-down oscillation amplitude
        rads_scale : fraction of scene radius to use for spiral

    Returns:
        path_poses: (n_frames, 4, 4) camera-to-world matrices
    """
    # Compute average pose
    centers = poses[:, :3, 3]
    centroid = centers.mean(axis=0)

    # Compute scene bounding box
    dists = np.linalg.norm(centers - centroid, axis=1)
    radius = float(np.percentile(dists, 75)) * rads_scale

    # Average rotation (look-at direction)
    avg_rot = np.mean(poses[:, :3, :3], axis=0)
    # Orthogonalize via SVD
    U, _, Vt = np.linalg.svd(avg_rot)
    avg_rot = U @ Vt

    path_poses = []

    for i in range(n_frames):
        t = i / n_frames

        # Spiral in camera coordinate frame
        theta = t * 2 * np.pi * n_rotations
        phi   = z_variation * np.sin(t * 2 * np.pi + z_phase)

        # Position on spiral
        offset = np.array([
            radius * np.cos(theta),
            radius * np.sin(theta),
            radius * phi * 0.5
        ])

        # Rotate offset to world frame
        cam_center = centroid + avg_rot @ offset

        # Look-at: camera points toward scene centroid
        forward = centroid - cam_center
        forward = forward / (np.linalg.norm(forward) + 1e-8)

        # Up vector (avoid gimbal lock)
        up_ref = avg_rot[:, 1]   # use average camera up
        right  = np.cross(forward, up_ref)
        right  = right / (np.linalg.norm(right) + 1e-8)
        up     = np.cross(right, forward)

        # Build c2w matrix
        c2w = np.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -forward   # OpenGL: -Z is forward
        c2w[:3, 3] = cam_center

        path_poses.append(c2w)

    return np.stack(path_poses, axis=0)


def generate_360_path(
    radius: float = 4.0,
    height: float = 0.5,
    n_frames: int = 120,
    center: np.ndarray = None
) -> np.ndarray:
    """
    Generate a simple 360° horizontal camera orbit.
    Useful for object-centric scenes (NeRF Synthetic).

    Returns:
        (n_frames, 4, 4) c2w matrices
    """
    if center is None:
        center = np.zeros(3)

    poses = []
    for i in range(n_frames):
        theta = 2 * np.pi * i / n_frames

        cam_x = center[0] + radius * np.cos(theta)
        cam_y = center[1] + height
        cam_z = center[2] + radius * np.sin(theta)
        cam_pos = np.array([cam_x, cam_y, cam_z])

        # Look at center
        forward = center - cam_pos
        forward = forward / (np.linalg.norm(forward) + 1e-8)

        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-6:
            up = np.array([0.0, 0.0, 1.0])
            right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up    = np.cross(right, forward)

        c2w = np.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -forward
        c2w[:3, 3] = cam_pos

        poses.append(c2w)

    return np.stack(poses, axis=0)


def interpolate_poses_slerp(
    poses: np.ndarray,
    n_frames: int
) -> np.ndarray:
    """
    Smooth interpolation between a sequence of poses using SLERP
    for rotation and per-axis linear interpolation for translation.

    Args:
        poses   : (N, 4, 4) key-frame camera-to-world matrices
        n_frames: total number of frames in the output path

    Returns:
        (n_frames, 4, 4) interpolated poses
    """
    from scipy.spatial.transform import Rotation, Slerp

    N = len(poses)

    # Edge cases
    if N == 0:
        raise ValueError("poses must contain at least one pose")
    if N == 1:
        return np.stack([poses[0]] * n_frames)

    # Key-frame times uniformly spaced in [0, 1]
    key_times   = np.linspace(0.0, 1.0, N)
    query_times = np.linspace(0.0, 1.0, n_frames)

    # ── Rotation: SLERP ──────────────────────────────────────────────────────
    rotations = Rotation.from_matrix(poses[:, :3, :3])  # (N,)
    slerp_fn  = Slerp(key_times, rotations)

    # ── Translation: per-axis np.interp (each axis is 1-D) ───────────────────
    translations = poses[:, :3, 3]          # (N, 3)

    interp_poses = []
    for t in query_times:
        R = slerp_fn(t).as_matrix()         # (3, 3)

        # Interpolate each of X, Y, Z independently
        tx = float(np.interp(t, key_times, translations[:, 0]))
        ty = float(np.interp(t, key_times, translations[:, 1]))
        tz = float(np.interp(t, key_times, translations[:, 2]))

        c2w = np.eye(4)
        c2w[:3, :3] = R
        c2w[:3,  3] = [tx, ty, tz]
        interp_poses.append(c2w)

    return np.stack(interp_poses, axis=0)   # (n_frames, 4, 4)


# ─────────────────────────────────────────────────────────────
# Checkpoint Management
# ─────────────────────────────────────────────────────────────

class CheckpointManager:
    """
    Manages NeRF model checkpoints: saving, loading, pruning old ones.
    """

    def __init__(self, checkpoint_dir: str, keep_last: int = 5):
        self.checkpoint_dir = checkpoint_dir
        self.keep_last = keep_last
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(
        self,
        iteration: int,
        model_coarse: nn.Module,
        model_fine: Optional[nn.Module],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        metrics: dict,
        config: dict
    ) -> str:
        """Save checkpoint and prune old ones."""
        path = os.path.join(
            self.checkpoint_dir,
            f"ckpt_{iteration:07d}.pt"
        )

        state = {
            "iteration": iteration,
            "model_coarse": model_coarse.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
        }

        if model_fine is not None:
            state["model_fine"] = model_fine.state_dict()

        if scheduler is not None:
            state["scheduler"] = scheduler.state_dict()

        torch.save(state, path)
        print(f"  Saved checkpoint: {path}")

        # Prune old checkpoints
        self._prune()

        # Save "latest" symlink info
        with open(os.path.join(self.checkpoint_dir, "latest.txt"), "w") as f:
            f.write(path)

        return path

    def load(
        self,
        path: Optional[str] = None,
        device: str = "cpu"
    ) -> dict:
        """
        Load checkpoint from path.
        If path is None, loads the latest checkpoint.
        """
        if path is None:
            latest_file = os.path.join(self.checkpoint_dir, "latest.txt")
            if not os.path.exists(latest_file):
                raise FileNotFoundError(
                    f"No checkpoints found in {self.checkpoint_dir}"
                )
            with open(latest_file) as f:
                path = f.read().strip()

        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        state = torch.load(path, map_location=device)
        print(f"  Loaded checkpoint: {path} (iter={state.get('iteration', '?')})")
        return state

    def _prune(self):
        """Keep only the last N checkpoints."""
        ckpts = sorted(
            [f for f in os.listdir(self.checkpoint_dir) if f.startswith("ckpt_") and f.endswith(".pt")]
        )
        while len(ckpts) > self.keep_last:
            old = os.path.join(self.checkpoint_dir, ckpts.pop(0))
            os.remove(old)

    def list_checkpoints(self) -> List[str]:
        """Return all checkpoint paths sorted by iteration."""
        ckpts = sorted(
            [
                os.path.join(self.checkpoint_dir, f)
                for f in os.listdir(self.checkpoint_dir)
                if f.startswith("ckpt_") and f.endswith(".pt")
            ]
        )
        return ckpts


# ─────────────────────────────────────────────────────────────
# Debug / Visualization Utilities
# ─────────────────────────────────────────────────────────────

def visualize_rays(
    rays_o: np.ndarray,
    rays_d: np.ndarray,
    n_samples: int = 64,
    near: float = 2.0,
    far: float = 6.0,
    max_rays: int = 20,
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize a set of rays in 3D to debug ray generation.

    Args:
        rays_o   : (N, 3) ray origins
        rays_d   : (N, 3) ray directions
        n_samples: number of sample points along each ray to show
        near/far : depth bounds
        max_rays : maximum number of rays to plot

    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')

    indices = np.random.choice(len(rays_o), min(max_rays, len(rays_o)), replace=False)
    t_vals  = np.linspace(near, far, n_samples)

    for idx in indices:
        o = rays_o[idx]
        d = rays_d[idx]
        pts = o[None] + t_vals[:, None] * d[None]   # (n_samples, 3)

        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'b-', alpha=0.3, linewidth=0.5)
        ax.scatter(*o, color='red', s=10, zorder=5)

    ax.set_title(f"Ray Visualization ({min(max_rays, len(rays_o))} rays)")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=120)
        print(f"Saved ray visualization: {output_path}")

    return fig


def visualize_depth_map(
    depth: np.ndarray,
    title: str = "Depth Map",
    output_path: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
) -> plt.Figure:
    """Visualize a depth map with colorbar."""
    valid = depth > 0
    if not valid.any():
        print("Warning: depth map has no valid pixels")
        return plt.figure()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Colormap visualization
    v_min = vmin if vmin is not None else float(depth[valid].min())
    v_max = vmax if vmax is not None else float(depth[valid].max())

    disp = depth.copy()
    disp[~valid] = np.nan

    im = axes[0].imshow(disp, cmap='plasma', vmin=v_min, vmax=v_max)
    axes[0].set_title(f"{title}\nRange: [{v_min:.2f}, {v_max:.2f}]")
    plt.colorbar(im, ax=axes[0], label="Depth (world units)")

    # Histogram of valid depths
    axes[1].hist(depth[valid].ravel(), bins=50, color='steelblue', edgecolor='none')
    axes[1].set_title("Depth Histogram (valid pixels)")
    axes[1].set_xlabel("Depth"); axes[1].set_ylabel("Count")
    axes[1].axvline(depth[valid].mean(), color='red', linestyle='--',
                    label=f"Mean={depth[valid].mean():.2f}")
    axes[1].legend()

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=150)

    return fig


def save_render_grid(
    images: List[np.ndarray],
    titles: Optional[List[str]] = None,
    n_cols: int = 4,
    output_path: str = "outputs/renders/grid.png"
) -> plt.Figure:
    """
    Save a grid of rendered images for quick inspection.

    Args:
        images   : list of (H, W, 3) float [0,1] images
        titles   : optional list of titles
        n_cols   : number of columns in grid
        output_path: output file path
    """
    n = len(images)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
        axes = [[ax] for ax in axes] if n_cols > 1 else [[axes]]

    img_idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            ax = axes[row][col] if n_rows > 1 else axes[col]
            if img_idx < n:
                ax.imshow(np.clip(images[img_idx], 0, 1))
                if titles and img_idx < len(titles):
                    ax.set_title(titles[img_idx], fontsize=9)
            ax.axis("off")
            img_idx += 1

    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved render grid: {output_path}")

    return fig


def print_model_summary(model: nn.Module, input_shapes: List[tuple]):
    """
    Print a summary of model layers and parameter counts.
    """
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*50}")
    print(f"Model: {model.__class__.__name__}")
    print(f"{'='*50}")
    print(f"{'Layer':<30} {'Output Shape':<20} {'Params':>10}")
    print(f"{'-'*62}")

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:   # leaf modules only
            param_count = sum(p.numel() for p in module.parameters())
            print(f"{name:<30} {str(type(module).__name__):<20} {param_count:>10,}")

    print(f"{'-'*62}")
    print(f"{'Total parameters':<50} {total_params:>10,}")
    print(f"{'Trainable parameters':<50} {trainable_params:>10,}")
    print(f"{'Non-trainable parameters':<50} {total_params - trainable_params:>10,}")
    print(f"{'='*50}\n")