"""
Side-by-side comparison and qualitative evaluation tools.

Compares:
  - Sparse vs. dense point clouds
  - Ground truth vs. estimated camera poses
  - NeRF rendered images vs. ground truth photos
  - Multiple NeRF checkpoints over training
"""

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")           # non-interactive backend for servers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from typing import List, Tuple, Dict, Optional
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# Image-level comparisons
# ─────────────────────────────────────────────────────────────

def compare_images(
    images: List[np.ndarray],
    titles: List[str],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = None,
    cmap: str = None,
    main_title: str = ""
) -> plt.Figure:
    """
    Create a side-by-side comparison of multiple images.

    Args:
        images      : List of (H, W, 3) or (H, W) images in [0,1] or [0,255]
        titles      : Labels for each image
        output_path : If given, save figure to this path
        figsize     : Override figure size
        cmap        : Matplotlib colormap (None for RGB, 'gray' for grayscale)
        main_title  : Super-title for the whole figure

    Returns:
        matplotlib Figure object
    """
    n = len(images)
    if figsize is None:
        figsize = (5 * n, 5)

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, img, title in zip(axes, images, titles):
        display = img.copy()

        # Normalize float images
        if display.dtype in (np.float32, np.float64):
            display = np.clip(display, 0, 1)
        else:
            display = display.astype(np.float32) / 255.0

        if cmap is None and len(display.shape) == 3:
            ax.imshow(display)
        elif len(display.shape) == 2:
            ax.imshow(display, cmap='gray', vmin=0, vmax=1)
        else:
            ax.imshow(display, cmap=cmap)

        ax.set_title(title, fontsize=11)
        ax.axis("off")

    if main_title:
        fig.suptitle(main_title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison: {output_path}")

    return fig


def compare_rgb_gt_pred(
    gt_image: np.ndarray,
    pred_image: np.ndarray,
    output_path: Optional[str] = None,
    label_gt: str = "Ground Truth",
    label_pred: str = "NeRF Prediction"
) -> plt.Figure:
    """
    Three-panel comparison: GT | Prediction | Error map.
    """
    # Ensure float [0,1]
    gt   = gt_image.astype(np.float32)
    pred = pred_image.astype(np.float32)
    if gt.max() > 1.5:
        gt /= 255.0
    if pred.max() > 1.5:
        pred /= 255.0

    gt   = np.clip(gt,   0, 1)
    pred = np.clip(pred, 0, 1)

    # Per-pixel L2 error
    error = np.sqrt(((gt - pred) ** 2).sum(axis=-1))
    error_norm = error / (error.max() + 1e-8)

    # Metrics
    mse  = np.mean((gt - pred) ** 2)
    psnr = -10 * np.log10(mse) if mse > 0 else float('inf')
    from skimage.metrics import structural_similarity
    ssim = structural_similarity(gt, pred, channel_axis=2, data_range=1.0)

    fig = plt.figure(figsize=(15, 5))
    gs  = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])

    ax_gt   = fig.add_subplot(gs[0])
    ax_pred = fig.add_subplot(gs[1])
    ax_err  = fig.add_subplot(gs[2])
    ax_cbar = fig.add_subplot(gs[3])

    ax_gt.imshow(gt);   ax_gt.set_title(label_gt);   ax_gt.axis("off")
    ax_pred.imshow(pred); ax_pred.set_title(label_pred); ax_pred.axis("off")

    im = ax_err.imshow(error_norm, cmap='hot', vmin=0, vmax=1)
    ax_err.set_title(f"Error Map\nPSNR={psnr:.2f}dB | SSIM={ssim:.4f}")
    ax_err.axis("off")

    plt.colorbar(im, cax=ax_cbar, label="Normalized Error")
    fig.suptitle("NeRF Rendering Quality Comparison", fontsize=13, fontweight='bold')
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved GT vs Pred: {output_path}")

    return fig


def make_video_comparison(
    gt_dir: str,
    pred_dir: str,
    output_path: str = "outputs/renders/comparison.mp4",
    fps: int = 10
):
    """
    Create a side-by-side GT | Prediction video.
    """
    import imageio.v3 as iio

    gt_files   = sorted(Path(gt_dir).glob("*.png"))
    pred_files = sorted(Path(pred_dir).glob("*.png"))

    if not gt_files or not pred_files:
        print("No PNG files found for video comparison.")
        return

    frames = []
    for gf, pf in zip(gt_files, pred_files):
        gt_img   = np.array(iio.imread(str(gf)))[..., :3]
        pred_img = np.array(iio.imread(str(pf)))[..., :3]

        # Resize to same size
        h = min(gt_img.shape[0], pred_img.shape[0])
        w = min(gt_img.shape[1], pred_img.shape[1])
        gt_img   = cv2.resize(gt_img,   (w, h))
        pred_img = cv2.resize(pred_img, (w, h))

        side_by_side = np.hstack([gt_img, pred_img])
        frames.append(side_by_side)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    iio.imwrite(output_path, frames, fps=fps)
    print(f"Saved comparison video: {output_path}")


# ─────────────────────────────────────────────────────────────
# 3D Point Cloud Comparison
# ─────────────────────────────────────────────────────────────

def compare_point_clouds(
    sparse_points: np.ndarray,
    dense_points: np.ndarray,
    sparse_colors: Optional[np.ndarray] = None,
    dense_colors: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    max_display: int = 50_000
) -> plt.Figure:
    """
    Compare sparse SfM vs dense MVS point clouds in a 2x2 subplot grid.
    Views: Top (XZ), Side (YZ), Front (XY), 3D perspective.
    """
    def _subsample(pts, clr, n):
        if len(pts) > n:
            idx = np.random.choice(len(pts), n, replace=False)
            return pts[idx], (clr[idx] if clr is not None else None)
        return pts, clr

    sp, sc = _subsample(sparse_points, sparse_colors, max_display // 2)
    dp, dc = _subsample(dense_points,  dense_colors,  max_display)

    sc_rgb = sc if sc is not None else np.ones((len(sp), 3)) * [0.2, 0.6, 1.0]
    dc_rgb = dc if dc is not None else np.ones((len(dp), 3)) * [0.8, 0.4, 0.1]

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        f"Point Cloud Comparison\n"
        f"Sparse: {len(sparse_points):,} pts  |  Dense: {len(dense_points):,} pts",
        fontsize=13, fontweight='bold'
    )

    views = [
        ("Top  (X-Z)",  (0, 2), (0, 2)),    # (axis_x, axis_y, col_x, col_y)
        ("Side (Y-Z)",  (1, 2), (1, 2)),
        ("Front(X-Y)",  (0, 1), (0, 1)),
    ]

    for plot_i, (view_name, (ax_x, ax_y), _) in enumerate(views):
        ax = fig.add_subplot(2, 2, plot_i + 1)

        labels = ["X", "Y", "Z"]
        ax.scatter(sp[:, ax_x], sp[:, ax_y], c=sc_rgb, s=1,   alpha=0.5, label=f"Sparse ({len(sp):,})")
        ax.scatter(dp[:, ax_x], dp[:, ax_y], c=dc_rgb, s=0.3, alpha=0.3, label=f"Dense ({len(dp):,})")

        ax.set_xlabel(labels[ax_x])
        ax.set_ylabel(labels[ax_y])
        ax.set_title(view_name)
        ax.legend(markerscale=5, fontsize=8)
        ax.set_aspect('equal', 'box')

    # 3D subplot
    ax3d = fig.add_subplot(2, 2, 4, projection='3d')
    ax3d.scatter(sp[:, 0], sp[:, 1], sp[:, 2], c=sc_rgb, s=1,   alpha=0.5)
    ax3d.scatter(dp[:, 0], dp[:, 1], dp[:, 2], c=dc_rgb, s=0.2, alpha=0.2)
    ax3d.set_title("3D Perspective")
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved point cloud comparison: {output_path}")

    return fig


# ─────────────────────────────────────────────────────────────
# Camera Pose Comparison
# ─────────────────────────────────────────────────────────────

def compare_camera_trajectories(
    poses_estimated: List[Tuple[np.ndarray, np.ndarray]],
    poses_gt: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    labels: Tuple[str, str] = ("Estimated", "Ground Truth"),
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot estimated vs ground truth camera trajectories in 3D.
    """
    centers_est = np.array([-R.T @ t.ravel() for R, t in poses_estimated])

    fig = plt.figure(figsize=(14, 6))

    # 3D view
    ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    ax3d.plot(centers_est[:, 0], centers_est[:, 1], centers_est[:, 2],
              'b.-', linewidth=1.5, markersize=6, label=labels[0])

    if poses_gt is not None:
        centers_gt = np.array([-R.T @ t.ravel() for R, t in poses_gt])
        ax3d.plot(centers_gt[:, 0], centers_gt[:, 1], centers_gt[:, 2],
                  'r.-', linewidth=1.5, markersize=6, label=labels[1])

    # Mark start and end
    ax3d.scatter(*centers_est[0],  color='green', s=100, marker='^', label='Start', zorder=5)
    ax3d.scatter(*centers_est[-1], color='red',   s=100, marker='s', label='End',   zorder=5)

    ax3d.set_title("Camera Trajectory (3D)")
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    ax3d.legend(fontsize=8)

    # Top-down view
    ax2d = fig.add_subplot(1, 2, 2)
    ax2d.plot(centers_est[:, 0], centers_est[:, 2], 'b.-', label=labels[0])

    if poses_gt is not None:
        ax2d.plot(centers_gt[:, 0], centers_gt[:, 2], 'r.-', label=labels[1])

    ax2d.scatter(centers_est[0, 0],  centers_est[0, 2],  color='green', s=80, marker='^', zorder=5)
    ax2d.scatter(centers_est[-1, 0], centers_est[-1, 2], color='red',   s=80, marker='s', zorder=5)
    ax2d.set_title("Top View (X-Z plane)")
    ax2d.set_xlabel("X"); ax2d.set_ylabel("Z")
    ax2d.set_aspect('equal', 'box')
    ax2d.legend()
    ax2d.grid(True, alpha=0.3)

    fig.suptitle("Camera Trajectory Comparison", fontsize=13, fontweight='bold')
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory comparison: {output_path}")

    return fig


# ─────────────────────────────────────────────────────────────
# Training Curve Visualization
# ─────────────────────────────────────────────────────────────

def plot_training_curves(
    train_losses: List[float],
    val_psnrs: List[float],
    val_interval: int = 5000,
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot NeRF training loss and validation PSNR curves.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    iterations = np.arange(1, len(train_losses) + 1)
    ax1.semilogy(iterations, train_losses, color='steelblue', linewidth=0.8, alpha=0.7)

    # Smoothed version (running average)
    if len(train_losses) > 100:
        window = min(500, len(train_losses) // 10)
        smoothed = np.convolve(train_losses, np.ones(window) / window, mode='valid')
        ax1.semilogy(
            np.arange(window // 2, window // 2 + len(smoothed)),
            smoothed, color='red', linewidth=2, label=f"Smooth (w={window})"
        )
        ax1.legend()

    ax1.set_title("Training Loss (MSE)", fontsize=12)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("MSE Loss (log scale)")
    ax1.grid(True, alpha=0.3, which='both')

    # PSNR curve
    if val_psnrs:
        val_iters = np.arange(len(val_psnrs)) * val_interval + val_interval
        ax2.plot(val_iters, val_psnrs, 'g.-', linewidth=2, markersize=8, label="Val PSNR")
        ax2.axhline(y=max(val_psnrs), color='r', linestyle='--', alpha=0.5,
                    label=f"Best: {max(val_psnrs):.2f} dB")
        ax2.set_title("Validation PSNR", fontsize=12)
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("PSNR (dB)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=max(0, min(val_psnrs) - 2))

    fig.suptitle("NeRF Training Progress", fontsize=13, fontweight='bold')
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves: {output_path}")

    return fig


def generate_comparison_report(
    output_dir: str,
    sfm_results: dict,
    nerf_results: Optional[dict] = None
):
    """
    Generate a comprehensive HTML report with all comparison figures.
    """
    report_dir = os.path.join(output_dir, "report")
    os.makedirs(report_dir, exist_ok=True)

    html_parts = [
        "<!DOCTYPE html>",
        "<html><head>",
        "<title>3D Reconstruction Report</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; max-width: 1200px; margin: auto; padding: 20px; }",
        "h1 { color: #2c3e50; } h2 { color: #3498db; }",
        "table { border-collapse: collapse; width: 100%; }",
        "td, th { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "th { background-color: #3498db; color: white; }",
        "tr:nth-child(even) { background-color: #f2f2f2; }",
        "img { max-width: 100%; border-radius: 4px; }",
        ".metric { font-size: 1.5em; font-weight: bold; color: #2ecc71; }",
        "</style>",
        "</head><body>",
        "<h1>3D Reconstruction Pipeline Report</h1>",
    ]

    # SfM section
    html_parts.append("<h2>Phase 2: Structure-from-Motion Results</h2>")
    html_parts.append("<table>")
    for k, v in sfm_results.items():
        html_parts.append(f"<tr><th>{k}</th><td class='metric'>{v}</td></tr>")
    html_parts.append("</table>")

    # NeRF section
    if nerf_results:
        html_parts.append("<h2>Phase 4: NeRF Rendering Results</h2>")
        html_parts.append("<table>")
        for k, v in nerf_results.items():
            html_parts.append(f"<tr><th>{k}</th><td class='metric'>{v}</td></tr>")
        html_parts.append("</table>")

    html_parts.append("</body></html>")

    report_path = os.path.join(report_dir, "index.html")
    with open(report_path, "w") as f:
        f.write("\n".join(html_parts))

    print(f"Report saved: {report_path}")
    return report_path