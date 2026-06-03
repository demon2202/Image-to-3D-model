"""
PatchMatch Multi-View Stereo (MVS) depth estimation.

PatchMatch MVS is the algorithm behind COLMAP's dense reconstruction.
This is a simplified but fully functional Python implementation.

Algorithm:
  1. Random initialization of depth + normal hypotheses per pixel
  2. Spatial propagation: propagate good hypotheses to neighbors
  3. Random refinement: perturb and test hypotheses
  4. Repeat for N iterations
  5. Aggregate depth from multiple source views

Reference: Bleyer et al. "PatchMatch Stereo" BMVC 2011
           Schönberger et al. "Pixelwise View Selection for Unstructured MVS" ECCV 2016
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field


@dataclass
class PatchMatchConfig:
    """Configuration for PatchMatch MVS."""
    patch_radius: int = 5          # Half-size of matching patch
    num_iterations: int = 5        # Number of PatchMatch iterations
    num_random_trials: int = 6     # Random hypotheses per refinement step
    depth_min: float = 0.1         # Minimum depth
    depth_max: float = 100.0       # Maximum depth
    normal_cone_angle: float = 60.0  # Max angle for normal perturbation (deg)
    ncc_threshold: float = 0.5     # Minimum NCC score to keep hypothesis
    depth_geometric_consistency: float = 0.01  # Relative depth consistency threshold
    use_geometric_consistency: bool = True
    num_source_views: int = 4      # How many source views to use


# ─────────────────────────────────────────────────────────────
# Patch Matching Cost
# ─────────────────────────────────────────────────────────────

class PatchCostComputer:
    """
    Computes photometric matching costs between patches across views.
    Uses Normalized Cross-Correlation (NCC) for robustness to
    illumination differences.
    """

    def __init__(self, patch_radius: int = 5):
        self.patch_radius = patch_radius
        self.patch_size   = 2 * patch_radius + 1

    def extract_patch(
        self,
        image: np.ndarray,
        x: int,
        y: int
    ) -> Optional[np.ndarray]:
        """
        Extract a (patch_size x patch_size) patch centered at (x, y).
        Returns None if out of bounds.
        """
        r = self.patch_radius
        h, w = image.shape[:2]

        if x - r < 0 or x + r >= w or y - r < 0 or y + r >= h:
            return None

        patch = image[y - r:y + r + 1, x - r:x + r + 1]
        return patch.astype(np.float32)

    def ncc(self, patch1: np.ndarray, patch2: np.ndarray) -> float:
        """
        Normalized Cross-Correlation between two patches.
        Returns value in [-1, 1]. 1.0 = perfect match.
        """
        p1 = patch1.ravel().astype(np.float64)
        p2 = patch2.ravel().astype(np.float64)

        mu1, mu2 = p1.mean(), p2.mean()
        std1 = p1.std() + 1e-8
        std2 = p2.std() + 1e-8

        return float(np.mean((p1 - mu1) * (p2 - mu2)) / (std1 * std2))

    def compute_multi_view_cost(
        self,
        ref_image: np.ndarray,
        src_images: List[np.ndarray],
        ref_x: int,
        ref_y: int,
        src_coords: List[Tuple[float, float]]
    ) -> float:
        """
        Average NCC cost across multiple source views.
        Uses best N-1 views to handle occlusion (robust aggregation).

        Returns:
            mean NCC score (higher = better match)
        """
        ref_patch = self.extract_patch(ref_image, ref_x, ref_y)
        if ref_patch is None:
            return -1.0

        scores = []
        for src_img, (sx, sy) in zip(src_images, src_coords):
            sx_i, sy_i = int(round(sx)), int(round(sy))
            src_patch = self.extract_patch(src_img, sx_i, sy_i)
            if src_patch is None:
                continue

            # Handle grayscale
            if ref_patch.shape != src_patch.shape:
                continue

            score = self.ncc(ref_patch, src_patch)
            scores.append(score)

        if not scores:
            return -1.0

        # Robust: drop worst score if we have enough views
        scores.sort(reverse=True)
        if len(scores) > 2:
            scores = scores[:-1]

        return float(np.mean(scores))


# ─────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────

def project_point_to_image(
    point_3d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray
) -> Optional[Tuple[float, float]]:
    """
    Project a 3D world point into image coordinates.
    Returns None if point is behind camera.
    """
    p_cam = R @ point_3d + t.ravel()
    if p_cam[2] <= 0:
        return None

    p_img = K @ p_cam
    return float(p_img[0] / p_img[2]), float(p_img[1] / p_img[2])


def backproject_pixel(
    x: float, y: float,
    depth: float,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray
) -> np.ndarray:
    """
    Back-project a pixel (x, y) at given depth to world 3D point.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Camera-space point
    p_cam = np.array([
        (x - cx) * depth / fx,
        (y - cy) * depth / fy,
        depth
    ], dtype=np.float64)

    # World-space point
    p_world = R.T @ (p_cam - t.ravel())
    return p_world


def compute_homography_warp(
    ref_K: np.ndarray, src_K: np.ndarray,
    R_rel: np.ndarray, t_rel: np.ndarray,
    normal: np.ndarray, depth: float
) -> np.ndarray:
    """
    Compute homography H that warps the reference image patch
    to the source image given a depth/normal hypothesis.

    H = src_K @ (R_rel - t_rel @ n^T / d) @ ref_K^{-1}
    """
    n = normal.reshape(3, 1)
    d = max(depth, 1e-4)

    H = src_K @ (R_rel - (t_rel.reshape(3, 1) @ n.T) / d) @ np.linalg.inv(ref_K)
    return H


# ─────────────────────────────────────────────────────────────
# PatchMatch Core
# ─────────────────────────────────────────────────────────────

class PatchMatchDepthEstimator:
    """
    PatchMatch MVS depth estimator for a single reference image.

    Usage:
        estimator = PatchMatchDepthEstimator(config)
        depth_map = estimator.estimate(
            ref_image, src_images, ref_pose, src_poses, K
        )
    """

    def __init__(self, config: Optional[PatchMatchConfig] = None):
        self.config = config or PatchMatchConfig()
        self.cost_computer = PatchCostComputer(self.config.patch_radius)

    def estimate(
        self,
        ref_image: np.ndarray,
        src_images: List[np.ndarray],
        ref_pose: Tuple[np.ndarray, np.ndarray],   # (R, t)
        src_poses: List[Tuple[np.ndarray, np.ndarray]],
        K: np.ndarray,
        initial_depth: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate depth map for ref_image using PatchMatch.

        Args:
            ref_image  : (H, W) or (H, W, 3) reference image
            src_images : list of source images
            ref_pose   : (R_ref, t_ref) reference camera pose
            src_poses  : list of (R_src, t_src) source camera poses
            K          : 3x3 intrinsic matrix (same for all cameras)
            initial_depth: (H, W) initial depth map (random if None)

        Returns:
            (depth_map, cost_map) — (H, W) float arrays
        """
        cfg = self.config

        # Convert to grayscale for matching
        if len(ref_image.shape) == 3:
            ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        else:
            ref_gray = ref_image.astype(np.float32) / 255.0

        src_grays = []
        for s in src_images[:cfg.num_source_views]:
            if len(s.shape) == 3:
                src_grays.append(cv2.cvtColor(s, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0)
            else:
                src_grays.append(s.astype(np.float32) / 255.0)

        H, W = ref_gray.shape
        R_ref, t_ref = ref_pose

        # Precompute relative poses
        rel_poses = []
        for R_src, t_src in src_poses[:cfg.num_source_views]:
            R_rel = R_src @ R_ref.T
            t_rel = t_src - R_rel @ t_ref
            rel_poses.append((R_rel, t_rel.ravel()))

        # ── Initialize depth and normal hypotheses ──
        if initial_depth is not None:
            depth_map = initial_depth.copy().astype(np.float32)
        else:
            # Random log-uniform depth initialization
            log_min = np.log(cfg.depth_min)
            log_max = np.log(cfg.depth_max)
            depth_map = np.exp(
                np.random.uniform(log_min, log_max, (H, W))
            ).astype(np.float32)

        # Random unit normals
        normal_map = np.random.randn(H, W, 3).astype(np.float32)
        norms = np.linalg.norm(normal_map, axis=2, keepdims=True) + 1e-8
        normal_map /= norms

        # Cost map (lower is BETTER here — we negate NCC)
        cost_map = np.full((H, W), 1.0, dtype=np.float32)

        # ── Initialize costs ──
        print(f"  PatchMatch init: {W}x{H} image, "
              f"{len(src_grays)} source views, "
              f"{cfg.num_iterations} iterations")

        cost_map = self._compute_full_cost(
            ref_gray, src_grays, depth_map, normal_map,
            K, rel_poses, H, W
        )

        # ── Main PatchMatch iterations ──
        for iteration in range(cfg.num_iterations):
            # Alternate scan direction each iteration
            if iteration % 2 == 0:
                xs = range(1, W - 1)
                ys = range(1, H - 1)
            else:
                xs = range(W - 2, 0, -1)
                ys = range(H - 2, 0, -1)

            improved = 0

            for y in ys:
                for x in xs:
                    improved += self._process_pixel(
                        x, y,
                        ref_gray, src_grays,
                        depth_map, normal_map, cost_map,
                        K, rel_poses, H, W
                    )

            print(f"  Iter {iteration+1}/{cfg.num_iterations}: "
                  f"improved {improved} pixels, "
                  f"mean cost={cost_map[1:-1, 1:-1].mean():.4f}")

        # ── Depth refinement pass ──
        depth_map = self._refinement_pass(
            ref_gray, src_grays,
            depth_map, normal_map, cost_map,
            K, rel_poses, H, W
        )

        # ── Geometric consistency filter ──
        if cfg.use_geometric_consistency and len(src_poses) > 0:
            depth_map = self._geometric_consistency_filter(
                depth_map, src_poses, ref_pose, K
            )

        return depth_map, cost_map

    def _compute_pixel_cost(
        self,
        x: int, y: int,
        depth: float, normal: np.ndarray,
        ref_gray: np.ndarray,
        src_grays: List[np.ndarray],
        K: np.ndarray,
        rel_poses: List[Tuple[np.ndarray, np.ndarray]]
    ) -> float:
        """Compute multi-view NCC cost for a single pixel hypothesis."""
        src_coords = []

        for (R_rel, t_rel) in rel_poses:
            # Warp reference pixel to source via homography
            H_mat = compute_homography_warp(K, K, R_rel, t_rel, normal, depth)
            ref_pt = np.array([x, y, 1.0])
            src_pt = H_mat @ ref_pt
            if abs(src_pt[2]) < 1e-8:
                src_coords.append((-1, -1))
            else:
                src_coords.append((src_pt[0] / src_pt[2], src_pt[1] / src_pt[2]))

        ncc_score = self.cost_computer.compute_multi_view_cost(
            ref_gray, src_grays, x, y, src_coords
        )

        return 1.0 - ncc_score   # Convert to cost (lower = better)

    def _compute_full_cost(
        self,
        ref_gray, src_grays,
        depth_map, normal_map,
        K, rel_poses, H, W
    ) -> np.ndarray:
        """Compute cost for all pixels (used at initialization)."""
        cost_map = np.ones((H, W), dtype=np.float32)

        # Process sparse grid for speed
        step = max(1, self.config.patch_radius)
        for y in range(self.config.patch_radius, H - self.config.patch_radius, step):
            for x in range(self.config.patch_radius, W - self.config.patch_radius, step):
                cost = self._compute_pixel_cost(
                    x, y,
                    float(depth_map[y, x]),
                    normal_map[y, x],
                    ref_gray, src_grays, K, rel_poses
                )
                cost_map[y, x] = cost

        return cost_map

    def _process_pixel(
        self,
        x: int, y: int,
        ref_gray, src_grays,
        depth_map, normal_map, cost_map,
        K, rel_poses, H, W
    ) -> int:
        """
        Process one pixel: spatial propagation + random refinement.
        Returns 1 if pixel was improved, 0 otherwise.
        """
        cfg = self.config
        current_cost = cost_map[y, x]
        best_depth  = float(depth_map[y, x])
        best_normal = normal_map[y, x].copy()

        improved = 0

        # ── Spatial propagation from 4 neighbors ──
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        for nx, ny in neighbors:
            if not (0 <= nx < W and 0 <= ny < H):
                continue

            cand_depth  = float(depth_map[ny, nx])
            cand_normal = normal_map[ny, nx]

            cand_cost = self._compute_pixel_cost(
                x, y, cand_depth, cand_normal,
                ref_gray, src_grays, K, rel_poses
            )

            if cand_cost < current_cost:
                current_cost = cand_cost
                best_depth   = cand_depth
                best_normal  = cand_normal
                improved     = 1

        # ── Random refinement ──
        depth_range = (cfg.depth_max - cfg.depth_min) / 2.0
        for trial in range(cfg.num_random_trials):
            # Exponentially decreasing perturbation
            scale = depth_range * (0.5 ** trial)

            # Perturb depth
            new_depth = best_depth + np.random.uniform(-scale, scale)
            new_depth = float(np.clip(new_depth, cfg.depth_min, cfg.depth_max))

            # Perturb normal (small rotation)
            angle_scale = cfg.normal_cone_angle * (0.5 ** trial)
            perturb = np.random.randn(3).astype(np.float32)
            perturb = perturb / (np.linalg.norm(perturb) + 1e-8)
            perturb *= np.tan(np.radians(angle_scale))
            new_normal = best_normal + perturb
            new_normal = new_normal / (np.linalg.norm(new_normal) + 1e-8)

            cand_cost = self._compute_pixel_cost(
                x, y, new_depth, new_normal,
                ref_gray, src_grays, K, rel_poses
            )

            if cand_cost < current_cost:
                current_cost = cand_cost
                best_depth   = new_depth
                best_normal  = new_normal
                improved     = 1

        # Update maps
        depth_map[y, x]   = best_depth
        normal_map[y, x]  = best_normal
        cost_map[y, x]    = current_cost

        return improved

    def _refinement_pass(
        self,
        ref_gray, src_grays,
        depth_map, normal_map, cost_map,
        K, rel_poses, H, W
    ) -> np.ndarray:
        """Final depth refinement: bilateral filter to smooth while preserving edges."""
        # Use depth confidence (NCC-based) as guidance
        confidence = np.clip(1.0 - cost_map, 0, 1)

        # Weighted median filter
        depth_refined = cv2.bilateralFilter(
            depth_map,
            d=9,
            sigmaColor=0.5,
            sigmaSpace=5.0
        )

        # Where cost is high (unreliable), fallback to original
        mask_unreliable = confidence < 0.3
        depth_refined[mask_unreliable] = 0.0

        return depth_refined

    def _geometric_consistency_filter(
        self,
        depth_map: np.ndarray,
        src_poses: List[Tuple[np.ndarray, np.ndarray]],
        ref_pose: Tuple[np.ndarray, np.ndarray],
        K: np.ndarray,
        max_inconsistency: float = 0.05
    ) -> np.ndarray:
        """
        Remove depth estimates that are inconsistent across views.
        A depth estimate is consistent if projecting it into another
        view and back gives a similar depth value.
        """
        cfg = self.config
        H, W = depth_map.shape
        R_ref, t_ref = ref_pose
        depth_clean = depth_map.copy()

        for R_src, t_src in src_poses[:2]:   # Check 2 source views for speed
            consistency_count = np.zeros((H, W), dtype=np.int32)

            for y in range(H):
                for x in range(W):
                    d = float(depth_map[y, x])
                    if d <= 0:
                        continue

                    # Back-project reference pixel to 3D
                    p3d = backproject_pixel(x, y, d, K, R_ref, t_ref)

                    # Project to source view
                    src_coord = project_point_to_image(p3d, K, R_src, t_src)
                    if src_coord is None:
                        continue

                    sx, sy = src_coord
                    si, sj = int(round(sx)), int(round(sy))

                    if not (0 <= si < W and 0 <= sj < H):
                        continue

                    # Depth at source pixel
                    d_src = float(depth_map[sj, si])
                    if d_src <= 0:
                        continue

                    # Back-project source pixel and re-project to reference
                    p3d_src  = backproject_pixel(si, sj, d_src, K, R_src, t_src)
                    ref_coord = project_point_to_image(p3d_src, K, R_ref, t_ref)

                    if ref_coord is None:
                        continue

                    # Reprojection consistency check
                    rx, ry = ref_coord
                    reproject_dist = np.sqrt((rx - x) ** 2 + (ry - y) ** 2)

                    if reproject_dist < 2.0:   # within 2 pixels
                        consistency_count[y, x] += 1

            # Zero out inconsistent depth estimates
            depth_clean[consistency_count == 0] = 0.0

        return depth_clean


# ─────────────────────────────────────────────────────────────
# Multi-view PatchMatch Orchestrator
# ─────────────────────────────────────────────────────────────

class MultiViewPatchMatch:
    """
    Runs PatchMatch depth estimation for all registered cameras
    and fuses the results.
    """

    def __init__(self, config: Optional[PatchMatchConfig] = None):
        self.config    = config or PatchMatchConfig()
        self.estimator = PatchMatchDepthEstimator(self.config)

    def run(
        self,
        images: List[np.ndarray],
        poses: List[Tuple[np.ndarray, np.ndarray]],
        K: np.ndarray,
        output_dir: Optional[str] = None
    ) -> List[np.ndarray]:
        """
        Estimate depth for each image using all others as source views.

        Args:
            images : list of (H, W, 3) images
            poses  : list of (R, t) poses
            K      : intrinsic matrix

        Returns:
            depth_maps: list of (H, W) depth maps (one per image)
        """
        n = len(images)
        depth_maps = []

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        for ref_idx in range(n):
            print(f"\nPatchMatch: Reference view {ref_idx + 1}/{n}")

            # Select best source views
            src_indices = self._select_source_views(
                ref_idx, poses, max_views=self.config.num_source_views
            )

            src_imgs  = [images[i]  for i in src_indices]
            src_poses = [poses[i]   for i in src_indices]

            # Downsample for speed
            ref_img = self._downsample(images[ref_idx])
            src_imgs_ds = [self._downsample(s) for s in src_imgs]

            h_ds, w_ds = ref_img.shape[:2]
            K_ds = K.copy()
            K_ds[0] *= w_ds / images[ref_idx].shape[1]
            K_ds[1] *= h_ds / images[ref_idx].shape[0]

            depth, cost = self.estimator.estimate(
                ref_img, src_imgs_ds,
                poses[ref_idx], src_poses,
                K_ds
            )

            # Upsample depth to original resolution
            orig_h, orig_w = images[ref_idx].shape[:2]
            depth_full = cv2.resize(
                depth, (orig_w, orig_h),
                interpolation=cv2.INTER_LINEAR
            )
            depth_full *= (orig_h / h_ds)   # scale depth proportionally

            depth_maps.append(depth_full)

            if output_dir:
                # Save depth visualization
                valid = depth_full > 0
                if valid.any():
                    d_norm = (depth_full - depth_full[valid].min())
                    d_norm /= (depth_full[valid].max() + 1e-8)
                    d_vis = (d_norm * 255).astype(np.uint8)
                    d_colored = cv2.applyColorMap(d_vis, cv2.COLORMAP_INFERNO)
                    cv2.imwrite(
                        os.path.join(output_dir, f"depth_pm_{ref_idx:04d}.png"),
                        d_colored
                    )

        return depth_maps

    def _select_source_views(
        self,
        ref_idx: int,
        poses: List[Tuple[np.ndarray, np.ndarray]],
        max_views: int = 4
    ) -> List[int]:
        """Select source views with good baseline relative to reference."""
        ref_R, ref_t = poses[ref_idx]
        ref_center = -ref_R.T @ ref_t.ravel()

        baselines = []
        for i, (R, t) in enumerate(poses):
            if i == ref_idx:
                continue
            center = -R.T @ t.ravel()
            baseline = np.linalg.norm(center - ref_center)

            # Relative rotation
            R_rel  = R @ ref_R.T
            trace  = np.clip((np.trace(R_rel) - 1) / 2.0, -1.0, 1.0)
            angle  = np.degrees(np.arccos(trace))

            # Score: good baseline, small rotation
            score = 0.0
            if 0.01 < baseline < 10.0 and angle < 60.0:
                score = baseline / (1.0 + angle / 20.0)

            baselines.append((i, score))

        baselines.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in baselines[:max_views]]

    def _downsample(
        self,
        image: np.ndarray,
        max_dim: int = 640
    ) -> np.ndarray:
        """Downscale image so max(H,W) <= max_dim."""
        h, w = image.shape[:2]
        if max(h, w) <= max_dim:
            return image
        scale  = max_dim / max(h, w)
        new_w  = int(w * scale)
        new_h  = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)