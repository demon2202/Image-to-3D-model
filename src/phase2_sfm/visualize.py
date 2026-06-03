"""
3D visualization of point cloud and camera poses.
Drop-in replacement for the Open3D version — uses only matplotlib + numpy.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (registers 3d projection)
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from typing import Dict, Tuple, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _frustum_segments(
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    image_size: Tuple[int, int] = (640, 480),
    scale: float = 0.3,
) -> np.ndarray:
    """
    Return (N, 2, 3) array of line segments for one camera frustum,
    expressed in world coordinates.
    """
    w, h = image_size
    fx, fy = K[0, 0], K[1, 1]

    half_w = w / 2 * scale / fx
    half_h = h / 2 * scale / fy
    d = scale

    # 5 points: apex + 4 corners (in camera space)
    corners_cam = np.array([
        [0,       0,       0],   # 0 – apex
        [-half_w, -half_h, d],   # 1 – top-left
        [ half_w, -half_h, d],   # 2 – top-right
        [ half_w,  half_h, d],   # 3 – bottom-right
        [-half_w,  half_h, d],   # 4 – bottom-left
    ])

    # Camera → world:  Xw = Rᵀ (Xc − t)
    t_col = t.reshape(3, 1)
    corners_world = (R.T @ (corners_cam.T - t_col)).T   # (5, 3)

    edges = [(0,1),(0,2),(0,3),(0,4),(1,2),(2,3),(3,4),(4,1)]
    segs = np.array([[corners_world[a], corners_world[b]] for a, b in edges])
    return segs                                          # (8, 2, 3)


def _remove_outliers(pts: np.ndarray, z_thresh: float = 3.0) -> np.ndarray:
    """Boolean mask: True for inlier points (within z_thresh std-devs of mean)."""
    if len(pts) == 0:
        return np.array([], dtype=bool)
    mu = np.median(pts, axis=0)
    sd = np.std(pts, axis=0) + 1e-9
    return np.all(np.abs(pts - mu) < z_thresh * sd, axis=1)


# ──────────────────────────────────────────────────────────────────────────────
# Public API  (mirrors the original open3d-based interface)
# ──────────────────────────────────────────────────────────────────────────────

def create_camera_frustum(
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    image_size: Tuple[int, int] = (640, 480),
    scale: float = 0.3,
    color: List[float] = [1, 0, 0],
) -> dict:
    """
    Return a dict describing a camera frustum for use with visualize_reconstruction.

    Kept for API compatibility; the returned object is opaque — just pass it to
    visualize_reconstruction via the camera_poses argument instead.
    """
    segs = _frustum_segments(R, t, K, image_size, scale)
    return {"segments": segs, "color": tuple(color)}


def visualize_reconstruction(
    points_3d: np.ndarray,
    colors: np.ndarray,
    camera_poses: Dict[int, Tuple[np.ndarray, np.ndarray]],
    K: np.ndarray,
    window_name: str = "SfM Reconstruction",
    # ── extra knobs ──────────────────────────────────────────────────────────
    frustum_scale: float = 0.2,
    point_size: float = 1.5,
    filter_outliers: bool = True,
    save_path: Optional[str] = None,
    show: bool = True,
    image_size: Tuple[int, int] = (640, 480),
    figsize: Tuple[int, int] = (14, 9),
    bg_color: str = "#0d0d0d",
    camera_color: str = "#00e5ff",
    axis_color: str = "#444444",
):
    """
    Visualize a Structure-from-Motion reconstruction with matplotlib.

    Parameters
    ----------
    points_3d      : (N, 3) float array of reconstructed 3-D points.
    colors         : (N, 3) float array of RGB colours in [0, 1].
    camera_poses   : dict mapping image index → (R, t) where
                     R is (3,3) rotation and t is (3,) or (3,1) translation
                     such that  x_cam = R @ x_world + t.
    K              : (3,3) intrinsic matrix.
    window_name    : figure window title.
    frustum_scale  : size of the camera frustum glyphs in scene units.
    point_size     : matplotlib scatter point size.
    filter_outliers: remove extreme outlier points before plotting.
    save_path      : if given, save the figure to this path (png/pdf/svg …).
    show           : call plt.show() at the end.
    image_size     : (width, height) used to compute frustum corners.
    figsize        : figure size in inches.
    bg_color       : background colour (any matplotlib colour string).
    camera_color   : frustum wireframe colour.
    axis_color     : 3-D axis spine / tick colour.
    """

    fig = plt.figure(figsize=figsize, facecolor=bg_color)
    fig.canvas.manager.set_window_title(window_name) if hasattr(
        fig.canvas, "manager"
    ) else None

    ax: Axes3D = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(bg_color)

    # ── style axes ──────────────────────────────────────────────────────────
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor(axis_color)
    for spine in ax.spines.values():
        spine.set_color(axis_color)
    ax.tick_params(colors=axis_color, labelsize=7)
    ax.xaxis.label.set_color(axis_color)
    ax.yaxis.label.set_color(axis_color)
    ax.zaxis.label.set_color(axis_color)
    ax.set_xlabel("X", labelpad=4)
    ax.set_ylabel("Y", labelpad=4)
    ax.set_zlabel("Z", labelpad=4)
    ax.set_title(window_name, color="#cccccc", fontsize=11, pad=10)

    # ── point cloud ─────────────────────────────────────────────────────────
    scene_pts = np.empty((0, 3))
    if len(points_3d) > 0:
        pts = np.asarray(points_3d)
        cols = np.asarray(colors)

        mask = _remove_outliers(pts) if filter_outliers else np.ones(len(pts), bool)
        pts, cols = pts[mask], cols[mask]

        if len(pts) > 0:
            scene_pts = pts
            # Clamp colors to [0, 1] in case of floating-point drift
            cols = np.clip(cols, 0, 1)
            ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                c=cols, s=point_size, linewidths=0,
                alpha=0.85, depthshade=True,
                label=f"Points ({len(pts):,})",
            )

    # ── camera frustums ──────────────────────────────────────────────────────
    all_cam_centers = []
    all_segs = []

    for idx, (R, t) in camera_poses.items():
        R = np.asarray(R)
        t = np.asarray(t).ravel()

        segs = _frustum_segments(R, t, K, image_size, frustum_scale)
        all_segs.append(segs)

        C = -R.T @ t          # camera centre in world coords
        all_cam_centers.append(C)

    if all_segs:
        combined = np.concatenate(all_segs, axis=0)   # (M*8, 2, 3)
        lc = Line3DCollection(
            combined,
            colors=camera_color,
            linewidths=0.8,
            alpha=0.9,
            label=f"Cameras ({len(camera_poses)})",
        )
        ax.add_collection3d(lc)

        cam_arr = np.array(all_cam_centers)
        ax.scatter(
            cam_arr[:, 0], cam_arr[:, 1], cam_arr[:, 2],
            c=camera_color, s=18, zorder=5, linewidths=0,
        )

    # ── coordinate axes (origin cross) ──────────────────────────────────────
    # Compute a sensible axis length from the scene scale
    all_pts_for_scale = scene_pts if len(scene_pts) else (
        np.array(all_cam_centers) if all_cam_centers else np.zeros((1, 3))
    )
    scene_scale = np.ptp(all_pts_for_scale, axis=0).max()
    axis_len = max(scene_scale * 0.08, 0.01)

    origin = np.zeros(3)
    for vec, col in zip(np.eye(3), ["#ff4444", "#44ff44", "#4499ff"]):
        ax.plot(
            [origin[0], vec[0] * axis_len],
            [origin[1], vec[1] * axis_len],
            [origin[2], vec[2] * axis_len],
            color=col, linewidth=2,
        )

    # ── equal-aspect bounding box trick ─────────────────────────────────────
    if len(scene_pts) > 0:
        lo, hi = scene_pts.min(axis=0), scene_pts.max(axis=0)
        centre = (lo + hi) / 2
        half = max((hi - lo).max() / 2, 0.1)
        ax.set_xlim(centre[0] - half, centre[0] + half)
        ax.set_ylim(centre[1] - half, centre[1] + half)
        ax.set_zlim(centre[2] - half, centre[2] + half)

    ax.legend(
        facecolor="#1a1a1a", edgecolor="#555555",
        labelcolor="#cccccc", fontsize=8, loc="upper left",
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=bg_color)
        print(f"[visualize] saved → {save_path}")

    if show:
        plt.show()

    return fig, ax