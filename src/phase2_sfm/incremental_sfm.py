"""
Complete Incremental Structure-from-Motion pipeline.
"""

import cv2
import numpy as np
import os
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

from ..phase1_features.detector import FeatureDetector, FeatureData
from ..phase1_features.matcher import FeatureMatcher, MatchResult
from .fundamental import estimate_essential_matrix, decompose_essential_matrix
from .triangulate import triangulate_points, filter_triangulated_points, compute_reprojection_error
from .bundle_adjust import run_bundle_adjustment


class IncrementalSfM:
    """
    Incremental Structure-from-Motion pipeline.
    
    Algorithm:
    1. Detect features in all images
    2. Match all pairs
    3. Initialize with best pair
    4. Iteratively add cameras via PnP + triangulate new points
    5. Bundle adjustment at each step
    """
    
    def __init__(self, K: np.ndarray, config: dict = None):
        """
        Args:
            K: 3x3 camera intrinsic matrix
            config: Configuration dictionary
        """
        self.K = K.astype(np.float64)
        self.config = config or {}
        
        # Feature detection / matching
        self.detector = FeatureDetector(
            method=self.config.get('detector', 'sift'),
            max_keypoints=self.config.get('max_keypoints', 8000)
        )
        self.matcher = FeatureMatcher(
            method=self.config.get('matcher', 'flann'),
            ratio_threshold=self.config.get('ratio_threshold', 0.75),
            ransac_threshold=self.config.get('ransac_threshold', 1.0),
            min_matches=self.config.get('min_matches', 30),
            descriptor_type=self.config.get('detector', 'sift')
        )
        
        # State
        self.images: List[np.ndarray] = []
        self.image_names: List[str] = []
        self.features: List[FeatureData] = []
        self.pair_matches: Dict[Tuple[int, int], MatchResult] = {}
        
        # Reconstruction
        self.registered_cameras: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}  # img_idx -> (R, t)
        self.points_3d: List[np.ndarray] = []        # List of 3D points
        self.point_colors: List[np.ndarray] = []      # RGB colors
        self.observations: List[Tuple[int, int, np.ndarray]] = []  # (cam_idx, pt_idx, 2d)
        
        # Track which 2D features correspond to which 3D points
        # key: (img_idx, feat_idx) -> point_3d_idx
        self.feature_to_3d: Dict[Tuple[int, int], int] = {}
    
    def load_images(self, image_dir: str, max_dim: int = 1024):
        """Load images from directory."""
        valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        files = sorted([
            f for f in os.listdir(image_dir)
            if os.path.splitext(f)[1].lower() in valid_ext
        ])
        
        print(f"Loading {len(files)} images from {image_dir}")
        
        for fname in tqdm(files):
            img = cv2.imread(os.path.join(image_dir, fname))
            if img is None:
                continue
            
            # Resize if needed
            h, w = img.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                img = cv2.resize(img, None, fx=scale, fy=scale)
            
            self.images.append(img)
            self.image_names.append(fname)
        
        print(f"Loaded {len(self.images)} images")
    
    def detect_features(self):
        """Detect features in all loaded images."""
        print("Detecting features...")
        self.features = []
        for i, img in enumerate(tqdm(self.images)):
            feat = self.detector.detect(img)
            self.features.append(feat)
            print(f"  {self.image_names[i]}: {feat.num_features} features")
    
    def match_features(self):
        """Match features between all image pairs."""
        print("Matching features...")
        self.pair_matches = self.matcher.match_all_pairs(
            self.features, self.image_names
        )
        print(f"Found {len(self.pair_matches)} valid image pairs")
    
    def _select_initial_pair(self) -> Tuple[int, int]:
        """Select best initial pair based on number of inliers and baseline."""
        best_pair = None
        best_score = 0
        
        for (i, j), match_result in self.pair_matches.items():
            n_inliers = match_result.num_inliers
            
            # Estimate essential matrix and check decomposition
            pts1_in, pts2_in = match_result.get_inlier_points()
            E, mask = estimate_essential_matrix(pts1_in, pts2_in, self.K)
            
            if E is None:
                continue
            
            R, t, pose_mask = decompose_essential_matrix(E, pts1_in, pts2_in, self.K)
            
            # Score: number of valid points after cheirality
            n_valid = int(np.sum(pose_mask > 0))
            
            # Prefer pairs with good baseline (not too small translation)
            score = n_valid
            
            if score > best_score:
                best_score = score
                best_pair = (i, j)
        
        if best_pair is None:
            raise RuntimeError("No valid initial pair found!")
        
        print(f"Selected initial pair: {self.image_names[best_pair[0]]} <-> "
              f"{self.image_names[best_pair[1]]} (score={best_score})")
        return best_pair
    
    def _initialize_reconstruction(self, idx1: int, idx2: int):
        """Initialize reconstruction from the first image pair."""
        match_result = self.pair_matches.get((idx1, idx2)) or self.pair_matches.get((idx2, idx1))
        
        if match_result is None:
            raise RuntimeError(f"No matches between images {idx1} and {idx2}")
        
        # If the pair is stored as (idx2, idx1), swap
        if (idx2, idx1) in self.pair_matches:
            pts1, pts2 = match_result.pts2, match_result.pts1
            inlier_mask = match_result.inlier_mask
            swapped = True
        else:
            pts1, pts2 = match_result.pts1, match_result.pts2
            inlier_mask = match_result.inlier_mask
            swapped = False
        
        # Track indices of original matches
        match_indices = np.arange(len(match_result.matches))
        
        # Use only inliers
        if inlier_mask is not None:
            mask_bool = inlier_mask.ravel().astype(bool)
            pts1 = pts1[mask_bool]
            pts2 = pts2[mask_bool]
            match_indices = match_indices[mask_bool]
        
        # Essential matrix
        E, e_mask = estimate_essential_matrix(pts1, pts2, self.K)
        if E is None:
            raise RuntimeError("Essential matrix estimation failed")
        
        if e_mask is not None:
            mask_bool = e_mask.ravel().astype(bool)
            pts1 = pts1[mask_bool]
            pts2 = pts2[mask_bool]
            match_indices = match_indices[mask_bool]
        
        # Recover pose
        R, t, pose_mask = decompose_essential_matrix(E, pts1, pts2, self.K)
        
        # Camera 1 at origin
        R1 = np.eye(3)
        t1 = np.zeros((3, 1))
        
        # Camera 2
        R2 = R
        t2 = t
        
        self.registered_cameras[idx1] = (R1, t1)
        self.registered_cameras[idx2] = (R2, t2)
        
        # Projection matrices
        P1 = self.K @ np.hstack([R1, t1])
        P2 = self.K @ np.hstack([R2, t2])
        
        # Triangulate
        points_3d = triangulate_points(P1, P2, pts1, pts2)
        
        # Filter
        filtered_points, valid_mask = filter_triangulated_points(
            points_3d, P1, P2, pts1, pts2
        )
        
        # Store points and observations
        pts1_valid = pts1[valid_mask]
        pts2_valid = pts2[valid_mask]
        final_match_indices = match_indices[valid_mask]
        
        for k in range(len(filtered_points)):
            pt_idx = len(self.points_3d)
            self.points_3d.append(filtered_points[k])
            
            # Get color from first image
            x, y = int(pts1_valid[k, 0]), int(pts1_valid[k, 1])
            h, w = self.images[idx1].shape[:2]
            if 0 <= x < w and 0 <= y < h:
                color = self.images[idx1][y, x, ::-1]  # BGR -> RGB
            else:
                color = np.array([128, 128, 128], dtype=np.uint8)
            self.point_colors.append(color)
            
            self.observations.append((idx1, pt_idx, pts1_valid[k]))
            self.observations.append((idx2, pt_idx, pts2_valid[k]))
            
            # Populate feature_to_3d mapping
            m_idx = final_match_indices[k]
            m = match_result.matches[m_idx]
            feat1_idx = m.trainIdx if swapped else m.queryIdx
            feat2_idx = m.queryIdx if swapped else m.trainIdx
            self.feature_to_3d[(idx1, feat1_idx)] = pt_idx
            self.feature_to_3d[(idx2, feat2_idx)] = pt_idx
        
        print(f"Initialized with {len(filtered_points)} 3D points")
    
    def _register_next_image(self, img_idx: int) -> bool:
        """
        Register a new image using PnP from existing 3D-2D correspondences.
        """
        # Find 3D-2D correspondences
        pts_3d = []
        pts_2d = []
        
        for registered_idx in self.registered_cameras:
            pair = (min(registered_idx, img_idx), max(registered_idx, img_idx))
            match_result = self.pair_matches.get(pair)
            
            if match_result is None:
                continue
            
            swapped = (pair[0] == img_idx)
            
            matches = match_result.matches
            mask_bool = np.ones(len(matches), dtype=bool)
            if match_result.inlier_mask is not None:
                mask_bool = match_result.inlier_mask.ravel().astype(bool)
            
            pts_new_all = match_result.pts1 if swapped else match_result.pts2
            
            for k, (m, inlier) in enumerate(zip(matches, mask_bool)):
                if not inlier:
                    continue
                
                feat_new_idx = m.queryIdx if swapped else m.trainIdx
                feat_old_idx = m.trainIdx if swapped else m.queryIdx
                
                pt_3d_idx = self.feature_to_3d.get((registered_idx, feat_old_idx))
                if pt_3d_idx is not None:
                    pts_3d.append(self.points_3d[pt_3d_idx])
                    pts_2d.append(pts_new_all[k])
                    
                    # Also link the new feature to the 3D point (so subsequent cameras can use it)
                    self.feature_to_3d[(img_idx, feat_new_idx)] = pt_3d_idx
        
        if len(pts_3d) < 6:
            return False
        
        pts_3d = np.array(pts_3d, dtype=np.float64)
        pts_2d = np.array(pts_2d, dtype=np.float64)
        
        # Solve PnP
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d.reshape(-1, 1, 3),
            pts_2d.reshape(-1, 1, 2),
            self.K, None,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=4.0,
            confidence=0.999,
            iterationsCount=1000
        )
        
        if not success or inliers is None or len(inliers) < 6:
            return False
        
        R, _ = cv2.Rodrigues(rvec)
        t = tvec
        
        self.registered_cameras[img_idx] = (R, t)
        
        print(f"  Registered {self.image_names[img_idx]} with {len(inliers)} PnP inliers")
        
        # Triangulate new points with all registered cameras
        new_points = 0
        for registered_idx in list(self.registered_cameras.keys()):
            if registered_idx == img_idx:
                continue
            
            pair = (min(registered_idx, img_idx), max(registered_idx, img_idx))
            match_result = self.pair_matches.get(pair)
            
            if match_result is None:
                continue
            
            swapped = (pair[0] == img_idx)
            
            matches = match_result.matches
            mask_bool = np.ones(len(matches), dtype=bool)
            if match_result.inlier_mask is not None:
                mask_bool = match_result.inlier_mask.ravel().astype(bool)
            
            if swapped:
                pts_new = match_result.pts1
                pts_old = match_result.pts2
            else:
                pts_new = match_result.pts2
                pts_old = match_result.pts1
            
            # Save the indices of the inliers
            inlier_indices = np.where(mask_bool)[0]
            
            pts_new_in = pts_new[mask_bool]
            pts_old_in = pts_old[mask_bool]
            
            R_old, t_old = self.registered_cameras[registered_idx]
            R_new, t_new = self.registered_cameras[img_idx]
            
            P_old = self.K @ np.hstack([R_old, t_old])
            P_new = self.K @ np.hstack([R_new, t_new])
            
            tri_pts = triangulate_points(P_old, P_new, pts_old_in, pts_new_in)
            filtered, valid = filter_triangulated_points(
                tri_pts, P_old, P_new, pts_old_in, pts_new_in
            )
            
            pts_old_valid = pts_old_in[valid]
            pts_new_valid = pts_new_in[valid]
            success_indices = inlier_indices[valid]
            
            for k in range(len(filtered)):
                pt_idx = len(self.points_3d)
                self.points_3d.append(filtered[k])
                
                x, y = int(pts_new_valid[k, 0]), int(pts_new_valid[k, 1])
                h, w = self.images[img_idx].shape[:2]
                if 0 <= x < w and 0 <= y < h:
                    color = self.images[img_idx][y, x, ::-1]
                else:
                    color = np.array([128, 128, 128], dtype=np.uint8)
                self.point_colors.append(color)
                
                self.observations.append((registered_idx, pt_idx, pts_old_valid[k]))
                self.observations.append((img_idx, pt_idx, pts_new_valid[k]))
                
                # Populate feature_to_3d mapping
                m = matches[success_indices[k]]
                feat_new_idx = m.queryIdx if swapped else m.trainIdx
                feat_old_idx = m.trainIdx if swapped else m.queryIdx
                self.feature_to_3d[(registered_idx, feat_old_idx)] = pt_idx
                self.feature_to_3d[(img_idx, feat_new_idx)] = pt_idx
                
                new_points += 1
        
        print(f"  Triangulated {new_points} new points. Total: {len(self.points_3d)}")
        return True
    
    def reconstruct(self, bundle_adjust_interval: int = 3):
        """
        Run the full incremental SfM pipeline.
        """
        # Step 1: Detect and match features (if not already done)
        if not self.features:
            self.detect_features()
        if not self.pair_matches:
            self.match_features()
        
        # Step 2: Initialize
        idx1, idx2 = self._select_initial_pair()
        self._initialize_reconstruction(idx1, idx2)
        
        # Step 3: Incrementally add cameras
        registered_set = {idx1, idx2}
        remaining = set(range(len(self.images))) - registered_set
        
        iteration = 0
        while remaining:
            # Find the unregistered image with most matches to registered images
            best_img = None
            best_match_count = 0
            
            for img_idx in remaining:
                count = 0
                for reg_idx in registered_set:
                    pair = (min(reg_idx, img_idx), max(reg_idx, img_idx))
                    if pair in self.pair_matches:
                        count += self.pair_matches[pair].num_inliers
                if count > best_match_count:
                    best_match_count = count
                    best_img = img_idx
            
            if best_img is None or best_match_count < 30:
                print(f"Cannot register remaining {len(remaining)} images")
                break
            
            success = self._register_next_image(best_img)
            
            if success:
                registered_set.add(best_img)
                iteration += 1
                
                # Periodic bundle adjustment
                if iteration % bundle_adjust_interval == 0 and len(self.points_3d) > 0:
                    self._run_bundle_adjustment()
            
            remaining.discard(best_img)
        
        # Final bundle adjustment
        if len(self.points_3d) > 0:
            print("\nRunning final bundle adjustment...")
            self._run_bundle_adjustment()
        
        print(f"\nReconstruction complete!")
        print(f"  Registered cameras: {len(self.registered_cameras)}/{len(self.images)}")
        print(f"  3D points: {len(self.points_3d)}")
    
    def _run_bundle_adjustment(self):
        """Run bundle adjustment on current reconstruction."""
        if len(self.registered_cameras) < 2 or len(self.points_3d) < 10:
            return
        
        # Map camera indices to sequential indices
        cam_indices = sorted(self.registered_cameras.keys())
        cam_map = {idx: i for i, idx in enumerate(cam_indices)}
        
        poses = [self.registered_cameras[idx] for idx in cam_indices]
        pts_3d = np.array(self.points_3d)
        
        # Remap observations
        mapped_obs = []
        for cam_idx, pt_idx, pt_2d in self.observations:
            if cam_idx in cam_map and pt_idx < len(self.points_3d):
                mapped_obs.append((cam_map[cam_idx], pt_idx, pt_2d))
        
        if len(mapped_obs) < 10:
            return
        
        try:
            optimized_poses, optimized_points = run_bundle_adjustment(
                poses, pts_3d, mapped_obs, self.K,
                fix_first_camera=True,
                max_iterations=30,
                verbose=True
            )
            
            # Update state
            for i, idx in enumerate(cam_indices):
                self.registered_cameras[idx] = optimized_poses[i]
            
            self.points_3d = [optimized_points[i] for i in range(len(optimized_points))]
            
        except Exception as e:
            print(f"Bundle adjustment failed: {e}")
    
    def get_point_cloud(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (Nx3 points, Nx3 colors) as numpy arrays."""
        points = np.array(self.points_3d)
        colors = np.array(self.point_colors).astype(np.float64) / 255.0
        return points, colors
    
    def get_camera_poses(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Return registered camera poses."""
        return self.registered_cameras
    
    def compute_mean_reprojection_error(self) -> float:
        """Compute mean reprojection error across all observations."""
        errors = []
        pts_3d = np.array(self.points_3d)
        
        for cam_idx, pt_idx, pt_2d in self.observations:
            if cam_idx not in self.registered_cameras or pt_idx >= len(pts_3d):
                continue
            R, t = self.registered_cameras[cam_idx]
            err = compute_reprojection_error(
                pts_3d[pt_idx:pt_idx+1], pt_2d.reshape(1, 2), self.K, R, t
            )
            errors.extend(err)
        
        if errors:
            return np.mean(errors)
        return float('inf')
    
    def save_point_cloud(self, filepath: str):
        """Export point cloud as PLY file."""
        points, colors = self.get_point_cloud()
        
        with open(filepath, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            for i in range(len(points)):
                x, y, z = points[i]
                r, g, b = (colors[i] * 255).astype(np.uint8)
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
        
        print(f"Saved point cloud to {filepath}")