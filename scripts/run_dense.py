"""
Run dense reconstruction (MVS) on SfM output.

Pipeline:
  1. Load SfM camera poses + sparse point cloud
  2. For each consecutive pair: rectify stereo → compute SGBM disparity → depth map
  3. Back-project depth to 3D point clouds
  4. Fuse all depth maps using TSDF (via trimesh voxel grid)
  5. Export dense PLY and optional mesh

Usage:
    python scripts/run_dense.py [sfm_output_dir] [image_dir]

    sfm_output_dir : directory containing camera_poses.json + point_cloud.ply
                     (default: outputs/sparse)
    image_dir      : directory with original images
                     (default: data/raw)
"""

import sys
import os
import json
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phase3_dense.stereo_matching import DenseStereo, rectify_stereo_pair
from src.phase3_dense.mesh import (
    build_dense_point_cloud,
    poisson_surface_reconstruction,
    clean_mesh,
    save_mesh
)
from src.utils.io_utils import load_image, export_ply, load_images_from_dir
from src.utils.camera import build_projection_matrix


# ─────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Dense MVS Reconstruction")
    parser.add_argument(
        "--sfm_dir", default="outputs/sparse",
        help="Directory with camera_poses.json (SfM output)"
    )
    parser.add_argument(
        "--image_dir", default="data/raw",
        help="Directory with original images"
    )
    parser.add_argument(
        "--output_dir", default="outputs/dense",
        help="Where to save dense outputs"
    )
    parser.add_argument(
        "--max_pairs", type=int, default=None,
        help="Maximum number of stereo pairs to process (None = all)"
    )
    parser.add_argument(
        "--num_disparities", type=int, default=128,
        help="SGBM number of disparities (must be divisible by 16)"
    )
    parser.add_argument(
        "--block_size", type=int, default=5,
        help="SGBM matching block size (odd number)"
    )
    parser.add_argument(
        "--max_depth", type=float, default=50.0,
        help="Maximum depth value to keep (meters)"
    )
    parser.add_argument(
        "--voxel_size", type=float, default=0.02,
        help="Voxel size for point cloud downsampling"
    )
    parser.add_argument(
        "--run_mesh", action="store_true",
        help="Also run Poisson surface reconstruction"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Show 3D visualization after reconstruction"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def load_sfm_output(sfm_dir: str):
    """
    Load camera poses and intrinsics from SfM output JSON.

    Returns:
        K, poses_dict {img_name -> (R, t)}, image_order [names]
    """
    poses_path = os.path.join(sfm_dir, "camera_poses.json")
    if not os.path.exists(poses_path):
        raise FileNotFoundError(
            f"camera_poses.json not found in {sfm_dir}. Run SfM first."
        )

    with open(poses_path) as f:
        data = json.load(f)

    K = np.array(data["K"], dtype=np.float64)
    poses_by_name = {}
    index_to_name = {}

    for idx_str, info in data["poses"].items():
        R = np.array(info["R"], dtype=np.float64)
        t = np.array(info["t"], dtype=np.float64).reshape(3, 1)
        name = info["image"]
        poses_by_name[name] = (R, t)
        index_to_name[int(idx_str)] = name

    # Return in sorted index order
    image_order = [index_to_name[i] for i in sorted(index_to_name)]
    return K, poses_by_name, image_order


def compute_stereo_pair_score(
    R1: np.ndarray, t1: np.ndarray,
    R2: np.ndarray, t2: np.ndarray
) -> float:
    """
    Score a stereo pair by baseline and angle.
    Higher is better for dense reconstruction.
    Good pairs have: moderate baseline, small rotation angle.
    """
    c1 = -R1.T @ t1.ravel()
    c2 = -R2.T @ t2.ravel()
    baseline = np.linalg.norm(c1 - c2)

    # Relative rotation angle
    R_rel = R1 @ R2.T
    trace = np.clip((np.trace(R_rel) - 1) / 2.0, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(trace))

    # Prefer: baseline > 0.05 (not too close), angle < 30 deg
    if baseline < 1e-4:
        return 0.0
    if angle_deg > 45.0:
        return 0.0

    score = baseline / (1.0 + angle_deg / 10.0)
    return float(score)


def select_stereo_pairs(
    image_order: list,
    poses_by_name: dict,
    max_pairs: int = None,
    min_baseline: float = 0.01,
    window_size: int = 5
) -> list:
    """
    Select good stereo pairs from registered cameras.
    Uses a sliding window to avoid comparing all O(N^2) pairs.

    Returns:
        List of (name_i, name_j, score) tuples, sorted by score descending
    """
    pairs = []
    n = len(image_order)

    for i in range(n):
        name_i = image_order[i]
        if name_i not in poses_by_name:
            continue
        R1, t1 = poses_by_name[name_i]

        for j in range(i + 1, min(i + window_size + 1, n)):
            name_j = image_order[j]
            if name_j not in poses_by_name:
                continue
            R2, t2 = poses_by_name[name_j]

            score = compute_stereo_pair_score(R1, t1, R2, t2)
            if score > 0:
                pairs.append((name_i, name_j, score))

    pairs.sort(key=lambda x: x[2], reverse=True)

    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    return pairs


def depth_map_from_stereo(
    img1: np.ndarray,
    img2: np.ndarray,
    R1: np.ndarray, t1: np.ndarray,
    R2: np.ndarray, t2: np.ndarray,
    K: np.ndarray,
    stereo: DenseStereo,
    max_depth: float = 50.0
) -> np.ndarray:
    """
    Compute a depth map for img1 using img2 as the second stereo view.

    Returns:
        depth: (H, W) float32 depth map in world units
    """
    # Relative pose: R2 * X + t2 = R1 * X + t1 + R_rel * (X - C1)
    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1

    # Rectify
    rect1, rect2, Q, baseline = rectify_stereo_pair(img1, img2, K, R_rel, t_rel)

    # Disparity
    disparity = stereo.compute_disparity(rect1, rect2)

    # Convert to depth
    focal = K[0, 0]
    depth = stereo.disparity_to_depth(disparity, float(np.linalg.norm(t_rel)), focal)

    # Clip
    depth[depth > max_depth] = 0.0
    depth[depth < 0.01] = 0.0

    return depth.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 60)
    print("Dense MVS Reconstruction")
    print("=" * 60)

    # Load SfM output
    print(f"\n[1/5] Loading SfM output from: {args.sfm_dir}")
    K, poses_by_name, image_order = load_sfm_output(args.sfm_dir)
    print(f"  Loaded {len(poses_by_name)} camera poses")
    print(f"  K =\n{K}")

    # Load images
    print(f"\n[2/5] Loading images from: {args.image_dir}")
    images_dict = {}
    for name in image_order:
        fpath = os.path.join(args.image_dir, name)
        if os.path.exists(fpath):
            img = load_image(fpath, color_mode="bgr")
            images_dict[name] = img
        else:
            print(f"  Warning: image not found: {name}")

    print(f"  Loaded {len(images_dict)} images")

    # Select stereo pairs
    print(f"\n[3/5] Selecting stereo pairs...")
    pairs = select_stereo_pairs(
        [n for n in image_order if n in images_dict and n in poses_by_name],
        poses_by_name,
        max_pairs=args.max_pairs,
        window_size=5
    )
    print(f"  Selected {len(pairs)} stereo pairs")

    if not pairs:
        print("ERROR: No valid stereo pairs found.")
        print("Possible causes:")
        print("  - Too few registered cameras")
        print("  - All camera baselines are too small")
        print("  - Camera rotation between views is too large")
        sys.exit(1)

    # Initialize stereo matcher
    stereo = DenseStereo(
        num_disparities=args.num_disparities,
        block_size=args.block_size
    )

    # Process each pair
    print(f"\n[4/5] Computing depth maps for {len(pairs)} pairs...")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "depth_maps"), exist_ok=True)

    all_points  = []
    all_colors  = []
    success_count = 0

    for pair_idx, (name1, name2, score) in enumerate(tqdm(pairs, desc="Depth maps")):
        img1 = images_dict.get(name1)
        img2 = images_dict.get(name2)

        if img1 is None or img2 is None:
            continue

        R1, t1 = poses_by_name[name1]
        R2, t2 = poses_by_name[name2]

        try:
            depth = depth_map_from_stereo(
                img1, img2, R1, t1, R2, t2,
                K, stereo, max_depth=args.max_depth
            )
        except Exception as e:
            tqdm.write(f"  Pair {name1}<->{name2} failed: {e}")
            continue

        # Save depth visualization
        if pair_idx < 20:   # Save first 20 for inspection
            depth_vis = depth.copy()
            valid = depth_vis > 0
            if valid.any():
                d_min = depth_vis[valid].min()
                d_max = depth_vis[valid].max()
                depth_vis[valid] = (depth_vis[valid] - d_min) / (d_max - d_min + 1e-8)
            depth_u8 = (depth_vis * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_PLASMA)
            cv2.imwrite(
                os.path.join(args.output_dir, "depth_maps",
                             f"depth_{pair_idx:04d}.png"),
                depth_color
            )

        # Back-project to 3D in the frame of camera 1
        pts, colors = stereo.depth_to_pointcloud(
            depth, K, color_image=img1,
            R=R1, t=t1,
            max_depth=args.max_depth
        )

        if len(pts) > 100:
            all_points.append(pts)
            all_colors.append(colors if colors is not None
                              else np.ones((len(pts), 3)) * 0.7)
            success_count += 1

    print(f"  Successfully processed {success_count}/{len(pairs)} pairs")

    if not all_points:
        print("ERROR: No depth maps generated. Check image quality and camera baselines.")
        sys.exit(1)

    # Merge all point clouds
    print(f"\n[5/5] Merging and saving dense point cloud...")
    points_merged = np.concatenate(all_points, axis=0)
    colors_merged = np.concatenate(all_colors, axis=0)
    print(f"  Total points before filtering: {len(points_merged):,}")

    # Remove statistical outliers using distance filtering
    points_clean, colors_clean = statistical_outlier_removal(
        points_merged, colors_merged, k=20, std_multiplier=2.0
    )
    print(f"  Points after outlier removal: {len(points_clean):,}")

    # Voxel downsample
    if args.voxel_size > 0:
        points_ds, colors_ds = voxel_downsample(
            points_clean, colors_clean, voxel_size=args.voxel_size
        )
        print(f"  Points after voxel downsampling ({args.voxel_size}m): {len(points_ds):,}")
    else:
        points_ds, colors_ds = points_clean, colors_clean

    # Save dense PLY
    dense_ply = os.path.join(args.output_dir, "dense_point_cloud.ply")
    export_ply(dense_ply, points_ds, colors_ds)
    print(f"  Saved dense PLY: {dense_ply}")

    # Also save sparse for comparison
    sparse_ply_src = os.path.join(args.sfm_dir, "point_cloud.ply")
    if os.path.exists(sparse_ply_src):
        import shutil
        shutil.copy(sparse_ply_src, os.path.join(args.output_dir, "sparse_point_cloud.ply"))

    # Surface reconstruction
    if args.run_mesh:
        print("\n  Running Poisson surface reconstruction...")
        mesh = poisson_surface_reconstruction(points_ds, depth=8)
        if mesh is not None:
            mesh_clean = clean_mesh(mesh)
            mesh_path  = os.path.join(args.output_dir, "mesh.ply")
            save_mesh(mesh_clean, mesh_path)
            print(f"  Saved mesh: {mesh_path}")
        else:
            print("  Poisson reconstruction failed (not enough points/normals).")

    # Visualization
    if args.visualize:
        visualize_dense(points_ds, colors_ds, args.output_dir)

    print("\n" + "=" * 60)
    print("Dense reconstruction complete!")
    print(f"  Dense point cloud : {dense_ply}")
    print(f"  Depth maps        : {os.path.join(args.output_dir, 'depth_maps/')}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────
# Post-processing
# ─────────────────────────────────────────────────────────────

def statistical_outlier_removal(
    points: np.ndarray,
    colors: np.ndarray,
    k: int = 20,
    std_multiplier: float = 2.0
) -> tuple:
    """
    Remove outlier points whose mean distance to k nearest neighbors
    is more than std_multiplier standard deviations from the global mean.
    """
    from scipy.spatial import KDTree

    if len(points) < k + 1:
        return points, colors

    tree = KDTree(points)
    dists, _ = tree.query(points, k=k + 1)   # includes self at dist=0
    mean_dists = dists[:, 1:].mean(axis=1)   # exclude self

    global_mean = mean_dists.mean()
    global_std  = mean_dists.std()
    threshold   = global_mean + std_multiplier * global_std

    mask = mean_dists < threshold
    return points[mask], colors[mask]


def voxel_downsample(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float
) -> tuple:
    """
    Downsample point cloud using voxel grid.
    Each voxel retains the centroid of all points within it.
    """
    if voxel_size <= 0 or len(points) == 0:
        return points, colors

    # Compute voxel indices
    min_pt = points.min(axis=0)
    indices = np.floor((points - min_pt) / voxel_size).astype(np.int64)

    # Unique voxels
    voxel_keys = indices[:, 0] * 1_000_003 + indices[:, 1] * 1009 + indices[:, 2]
    unique_keys, inv = np.unique(voxel_keys, return_inverse=True)

    ds_points = np.zeros((len(unique_keys), 3))
    ds_colors = np.zeros((len(unique_keys), 3))

    # Accumulate
    np.add.at(ds_points, inv, points)
    np.add.at(ds_colors, inv, colors)

    counts = np.bincount(inv).astype(float)
    ds_points /= counts[:, None]
    ds_colors /= counts[:, None]

    return ds_points, ds_colors


def visualize_dense(
    points: np.ndarray,
    colors: np.ndarray,
    output_dir: str
):
    """Visualize dense point cloud using matplotlib (pyvista optional)."""
    try:
        import pyvista as pv
        cloud = pv.PolyData(points.astype(np.float32))
        if colors is not None:
            rgb = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
            cloud["RGB"] = rgb
        plotter = pv.Plotter(window_size=(1280, 720))
        plotter.add_points(cloud, rgb=True if colors is not None else False,
                           point_size=2)
        plotter.add_axes()
        plotter.show(title="Dense Point Cloud")
    except ImportError:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(12, 8))
        ax  = fig.add_subplot(111, projection='3d')

        # Subsample for display
        n = min(50_000, len(points))
        idx = np.random.choice(len(points), n, replace=False)
        pts = points[idx]
        clr = colors[idx] if colors is not None else np.ones((n, 3)) * 0.5

        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=clr, s=0.5, alpha=0.6)
        ax.set_title(f"Dense Point Cloud ({len(points):,} pts)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "dense_viz.png"), dpi=150)
        plt.show()
        print(f"  Saved visualization: {output_dir}/dense_viz.png")


if __name__ == "__main__":
    main()