"""
Camera pose estimation utilities.
Covers: PnP solving, pose graph, relative pose, absolute pose refinement.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass, field


@dataclass
class PoseEstimationResult:
    """
    Container for a single pose estimation result.
    """
    success: bool
    R: Optional[np.ndarray] = None          # 3x3 rotation matrix
    t: Optional[np.ndarray] = None          # 3x1 translation vector
    inlier_mask: Optional[np.ndarray] = None
    num_inliers: int = 0
    reprojection_error: float = float('inf')
    method: str = ""

    @property
    def camera_center(self) -> Optional[np.ndarray]:
        """Camera center in world coordinates."""
        if self.R is None or self.t is None:
            return None
        return (-self.R.T @ self.t.ravel())

    @property
    def c2w(self) -> Optional[np.ndarray]:
        """4x4 camera-to-world matrix."""
        if self.R is None or self.t is None:
            return None
        mat = np.eye(4)
        mat[:3, :3] = self.R.T
        mat[:3, 3]  = self.camera_center
        return mat

    def __repr__(self) -> str:
        return (f"PoseResult(success={self.success}, "
                f"inliers={self.num_inliers}, "
                f"repr_err={self.reprojection_error:.3f}px, "
                f"method={self.method})")


# ─────────────────────────────────────────────────────────────
# PnP — Perspective-n-Point Pose Estimation
# ─────────────────────────────────────────────────────────────

def solve_pnp(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    K: np.ndarray,
    dist_coeffs: Optional[np.ndarray] = None,
    method: int = cv2.SOLVEPNP_ITERATIVE,
    use_ransac: bool = True,
    ransac_threshold: float = 4.0,
    ransac_confidence: float = 0.999,
    max_iterations: int = 1000,
    initial_R: Optional[np.ndarray] = None,
    initial_t: Optional[np.ndarray] = None
) -> PoseEstimationResult:
    """
    Solve Perspective-n-Point to estimate camera pose from 3D-2D correspondences.

    Args:
        points_3d : (N, 3) world coordinates
        points_2d : (N, 2) image coordinates
        K         : 3x3 camera intrinsic matrix
        dist_coeffs: distortion coefficients (None = no distortion)
        method    : PnP solver (SOLVEPNP_ITERATIVE, SOLVEPNP_EPNP, SOLVEPNP_AP3P)
        use_ransac: whether to use RANSAC for outlier rejection
        ransac_threshold: reprojection error threshold in pixels
        initial_R, initial_t: initial guess for iterative refinement

    Returns:
        PoseEstimationResult
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5)

    pts3d = points_3d.reshape(-1, 1, 3).astype(np.float64)
    pts2d = points_2d.reshape(-1, 1, 2).astype(np.float64)

    # Need at least 4 points (3 for minimal P3P, 6+ recommended)
    if len(points_3d) < 4:
        return PoseEstimationResult(success=False, method="pnp_insufficient_points")

    # Initial guess via Rodrigues
    rvec_init = None
    tvec_init = None
    if initial_R is not None and initial_t is not None:
        rvec_init, _ = cv2.Rodrigues(initial_R)
        tvec_init = initial_t.reshape(3, 1).astype(np.float64)

    if use_ransac:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d, pts2d, K, dist_coeffs,
            rvec=rvec_init, tvec=tvec_init,
            useExtrinsicGuess=(rvec_init is not None),
            iterationsCount=max_iterations,
            reprojectionError=ransac_threshold,
            confidence=ransac_confidence,
            flags=method
        )
    else:
        inliers = None
        success = cv2.solvePnP(
            pts3d, pts2d, K, dist_coeffs,
            rvec=rvec_init, tvec=tvec_init,
            useExtrinsicGuess=(rvec_init is not None),
            flags=method
        )
        if success:
            _, rvec, tvec = success  # unpack
        else:
            return PoseEstimationResult(success=False, method="pnp_failed")

    if not success:
        return PoseEstimationResult(success=False, method="pnp_ransac_failed")

    # Refine with LM on inliers only
    if inliers is not None and len(inliers) >= 6:
        inlier_idx = inliers.ravel()
        _, rvec, tvec = cv2.solvePnP(
            pts3d[inlier_idx], pts2d[inlier_idx],
            K, dist_coeffs,
            rvec=rvec, tvec=tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)

    # Compute reprojection error on inliers
    mask = inliers.ravel() if inliers is not None else np.arange(len(points_3d))
    proj_pts, _ = cv2.projectPoints(
        pts3d[mask], rvec, tvec, K, dist_coeffs
    )
    proj_pts = proj_pts.reshape(-1, 2)
    orig_pts = pts2d[mask].reshape(-1, 2)
    repr_err = float(np.mean(np.linalg.norm(proj_pts - orig_pts, axis=1)))

    inlier_mask_full = np.zeros(len(points_3d), dtype=bool)
    if inliers is not None:
        inlier_mask_full[inliers.ravel()] = True
    else:
        inlier_mask_full[:] = True

    return PoseEstimationResult(
        success=True,
        R=R,
        t=t,
        inlier_mask=inlier_mask_full,
        num_inliers=int(inlier_mask_full.sum()),
        reprojection_error=repr_err,
        method="pnp_ransac" if use_ransac else "pnp"
    )


def solve_pnp_multimethod(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    K: np.ndarray,
    dist_coeffs: Optional[np.ndarray] = None
) -> PoseEstimationResult:
    """
    Try multiple PnP methods and return the best result (lowest reprojection error).
    Robust fallback for difficult scenes.
    """
    methods = [
        (cv2.SOLVEPNP_EPNP,      "EPnP"),
        (cv2.SOLVEPNP_ITERATIVE, "Iterative"),
        (cv2.SOLVEPNP_AP3P,      "AP3P"),
    ]

    best: Optional[PoseEstimationResult] = None

    for flag, name in methods:
        result = solve_pnp(
            points_3d, points_2d, K,
            dist_coeffs=dist_coeffs,
            method=flag,
            use_ransac=True
        )
        result.method = name

        if result.success:
            if best is None or result.reprojection_error < best.reprojection_error:
                best = result

    return best if best is not None else PoseEstimationResult(
        success=False, method="all_methods_failed"
    )


# ─────────────────────────────────────────────────────────────
# Relative Pose from Essential Matrix
# ─────────────────────────────────────────────────────────────

def recover_relative_pose(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    method: int = cv2.RANSAC,
    threshold: float = 1.0,
    confidence: float = 0.999
) -> PoseEstimationResult:
    """
    Recover relative camera pose from 2D-2D correspondences via Essential matrix.

    Image 1 is assumed to be at the origin (R=I, t=0).
    Returns the pose of image 2 relative to image 1.

    Args:
        pts1, pts2: Nx2 matched pixel coordinates
        K         : 3x3 intrinsic matrix (same for both cameras assumed)

    Returns:
        PoseEstimationResult for camera 2
    """
    if len(pts1) < 5:
        return PoseEstimationResult(success=False, method="essential_too_few_points")

    # Estimate Essential matrix
    E, e_mask = cv2.findEssentialMat(
        pts1.astype(np.float64),
        pts2.astype(np.float64),
        K,
        method=method,
        prob=confidence,
        threshold=threshold
    )

    if E is None or E.shape != (3, 3):
        return PoseEstimationResult(success=False, method="essential_failed")

    # Apply E mask
    if e_mask is not None:
        mask_bool = e_mask.ravel().astype(bool)
        pts1_in = pts1[mask_bool]
        pts2_in = pts2[mask_bool]
    else:
        pts1_in, pts2_in = pts1, pts2

    # Recover pose via cheirality
    n_inliers, R, t, pose_mask = cv2.recoverPose(
        E,
        pts1_in.astype(np.float64),
        pts2_in.astype(np.float64),
        K
    )

    # Compute reprojection error
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])

    pts1_T = pts1_in.T.astype(np.float64)
    pts2_T = pts2_in.T.astype(np.float64)
    pts4d  = cv2.triangulatePoints(P1, P2, pts1_T, pts2_T)
    pts3d  = (pts4d[:3] / pts4d[3:]).T

    proj1, _ = cv2.projectPoints(pts3d, np.zeros(3), np.zeros(3), K, None)
    proj2, _ = cv2.projectPoints(pts3d, cv2.Rodrigues(R)[0], t,   K, None)

    err1 = np.linalg.norm(proj1.reshape(-1, 2) - pts1_in, axis=1)
    err2 = np.linalg.norm(proj2.reshape(-1, 2) - pts2_in, axis=1)
    repr_err = float(np.mean(np.concatenate([err1, err2])))

    return PoseEstimationResult(
        success=True,
        R=R,
        t=t,
        inlier_mask=pose_mask.ravel().astype(bool) if pose_mask is not None else None,
        num_inliers=int(n_inliers),
        reprojection_error=repr_err,
        method="essential_5pt"
    )


# ─────────────────────────────────────────────────────────────
# Pose Graph
# ─────────────────────────────────────────────────────────────

@dataclass
class PoseGraphNode:
    """Node in the pose graph representing a registered camera."""
    image_index: int
    R: np.ndarray
    t: np.ndarray
    is_fixed: bool = False          # Fixed = anchor / not optimized

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t.ravel()


@dataclass
class PoseGraphEdge:
    """Edge between two pose graph nodes — relative pose constraint."""
    src_idx: int
    dst_idx: int
    R_rel: np.ndarray       # Relative rotation:  R_dst = R_rel @ R_src
    t_rel: np.ndarray       # Relative translation
    weight: float = 1.0     # Confidence / number of inliers
    num_inliers: int = 0


class PoseGraph:
    """
    Pose graph for managing camera poses in SfM.
    Supports incremental addition, cycle detection, and
    retrieval of spanning tree for initialization order.
    """

    def __init__(self):
        self.nodes: Dict[int, PoseGraphNode] = {}
        self.edges: List[PoseGraphEdge] = []
        self._adjacency: Dict[int, List[int]] = {}

    def add_node(
        self,
        image_index: int,
        R: np.ndarray,
        t: np.ndarray,
        is_fixed: bool = False
    ):
        self.nodes[image_index] = PoseGraphNode(image_index, R, t, is_fixed)
        if image_index not in self._adjacency:
            self._adjacency[image_index] = []

    def add_edge(
        self,
        src_idx: int,
        dst_idx: int,
        R_rel: np.ndarray,
        t_rel: np.ndarray,
        weight: float = 1.0,
        num_inliers: int = 0
    ):
        edge = PoseGraphEdge(src_idx, dst_idx, R_rel, t_rel, weight, num_inliers)
        self.edges.append(edge)
        self._adjacency.setdefault(src_idx, []).append(dst_idx)
        self._adjacency.setdefault(dst_idx, []).append(src_idx)

    def has_node(self, image_index: int) -> bool:
        return image_index in self.nodes

    def get_pose(self, image_index: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if image_index not in self.nodes:
            return None
        node = self.nodes[image_index]
        return node.R, node.t

    def update_pose(self, image_index: int, R: np.ndarray, t: np.ndarray):
        if image_index in self.nodes:
            self.nodes[image_index].R = R
            self.nodes[image_index].t = t

    def get_best_next_image(
        self,
        registered: set,
        all_indices: set
    ) -> Optional[int]:
        """
        Find the unregistered image with most edges to registered images.
        Used to determine the order of image registration in SfM.
        """
        unregistered = all_indices - registered
        if not unregistered:
            return None

        best_img = None
        best_count = 0

        for img_idx in unregistered:
            count = sum(
                1 for neighbor in self._adjacency.get(img_idx, [])
                if neighbor in registered
            )
            if count > best_count:
                best_count = count
                best_img = img_idx

        return best_img if best_count > 0 else None

    def get_camera_trajectory(self) -> np.ndarray:
        """
        Return camera centers ordered by image index as (N, 3) array.
        """
        sorted_nodes = sorted(self.nodes.values(), key=lambda n: n.image_index)
        return np.array([n.center for n in sorted_nodes])

    def compute_baseline(self, idx1: int, idx2: int) -> float:
        """Compute baseline (distance) between two cameras."""
        if idx1 not in self.nodes or idx2 not in self.nodes:
            return 0.0
        c1 = self.nodes[idx1].center
        c2 = self.nodes[idx2].center
        return float(np.linalg.norm(c1 - c2))

    def to_dict(self) -> dict:
        """Serialize pose graph for saving/loading."""
        return {
            "nodes": {
                str(k): {
                    "image_index": v.image_index,
                    "R": v.R.tolist(),
                    "t": v.t.tolist(),
                    "is_fixed": v.is_fixed
                }
                for k, v in self.nodes.items()
            },
            "edges": [
                {
                    "src": e.src_idx,
                    "dst": e.dst_idx,
                    "R_rel": e.R_rel.tolist(),
                    "t_rel": e.t_rel.tolist(),
                    "weight": e.weight,
                    "num_inliers": e.num_inliers
                }
                for e in self.edges
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PoseGraph":
        pg = cls()
        for k, v in data["nodes"].items():
            pg.add_node(
                int(k),
                np.array(v["R"]),
                np.array(v["t"]),
                v["is_fixed"]
            )
        for e in data["edges"]:
            pg.add_edge(
                e["src"], e["dst"],
                np.array(e["R_rel"]),
                np.array(e["t_rel"]),
                e["weight"],
                e["num_inliers"]
            )
        return pg


# ─────────────────────────────────────────────────────────────
# Pose Refinement
# ─────────────────────────────────────────────────────────────

def refine_pose(
    R_init: np.ndarray,
    t_init: np.ndarray,
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    K: np.ndarray,
    dist_coeffs: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Refine a camera pose using Levenberg-Marquardt minimization
    of reprojection error.

    Args:
        R_init  : Initial 3x3 rotation matrix
        t_init  : Initial 3x1 translation
        points_3d: (N, 3) world points
        points_2d: (N, 2) observed image points
        K       : 3x3 intrinsics

    Returns:
        (R_refined, t_refined, mean_reprojection_error)
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5)

    rvec_init, _ = cv2.Rodrigues(R_init)

    success, rvec, tvec = cv2.solvePnP(
        points_3d.reshape(-1, 1, 3).astype(np.float64),
        points_2d.reshape(-1, 1, 2).astype(np.float64),
        K, dist_coeffs,
        rvec=rvec_init, tvec=t_init.reshape(3, 1),
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    R_ref, _ = cv2.Rodrigues(rvec)
    t_ref = tvec.reshape(3, 1)

    # Compute final reprojection error
    proj, _ = cv2.projectPoints(
        points_3d.reshape(-1, 1, 3),
        rvec, tvec, K, dist_coeffs
    )
    err = float(np.mean(np.linalg.norm(
        proj.reshape(-1, 2) - points_2d, axis=1
    )))

    return R_ref, t_ref, err


def compute_pose_difference(
    R1: np.ndarray, t1: np.ndarray,
    R2: np.ndarray, t2: np.ndarray
) -> Tuple[float, float]:
    """
    Compute rotation angle difference (degrees) and translation distance
    between two camera poses. Useful for evaluating SfM accuracy.

    Returns:
        (rotation_diff_deg, translation_diff)
    """
    # Relative rotation
    R_rel = R1 @ R2.T
    # Angle from rotation matrix
    trace = np.clip((np.trace(R_rel) - 1) / 2.0, -1.0, 1.0)
    angle_deg = float(np.degrees(np.arccos(trace)))

    # Camera centers
    c1 = -R1.T @ t1.ravel()
    c2 = -R2.T @ t2.ravel()
    trans_diff = float(np.linalg.norm(c1 - c2))

    return angle_deg, trans_diff


def align_poses_umeyama(
    poses_estimated: List[Tuple[np.ndarray, np.ndarray]],
    poses_ground_truth: List[Tuple[np.ndarray, np.ndarray]]
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Align estimated camera trajectory to ground truth using
    Umeyama similarity transform (scale + rotation + translation).

    Args:
        poses_estimated  : List of (R, t) — estimated
        poses_ground_truth: List of (R, t) — ground truth

    Returns:
        (R_align, scale, t_align) — similarity transform that maps
        estimated centers to ground truth centers
    """
    assert len(poses_estimated) == len(poses_ground_truth)

    # Extract camera centers
    centers_est = np.array([-R.T @ t.ravel() for R, t in poses_estimated])
    centers_gt  = np.array([-R.T @ t.ravel() for R, t in poses_ground_truth])

    n = len(centers_est)

    # Means
    mu_est = centers_est.mean(axis=0)
    mu_gt  = centers_gt.mean(axis=0)

    # Centered
    A = centers_est - mu_est
    B = centers_gt  - mu_gt

    # Covariance
    sigma2_est = np.mean(np.sum(A ** 2, axis=1))
    H = (B.T @ A) / n

    # SVD
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(U @ Vt)
    D = np.diag([1, 1, d])

    R_align = U @ D @ Vt
    scale   = float(np.sum(S * np.diag(D)) / sigma2_est)
    t_align = mu_gt - scale * R_align @ mu_est

    return R_align, scale, t_align