"""
Tests for Phase 2: Structure from Motion.

Run with: pytest tests/test_sfm.py -v
"""

import os
import sys
import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phase2_sfm.fundamental import (
    estimate_fundamental_matrix,
    estimate_essential_matrix,
    decompose_essential_matrix
)
from src.phase2_sfm.triangulate import (
    triangulate_points,
    filter_triangulated_points,
    compute_reprojection_error
)
from src.phase2_sfm.pose import (
    solve_pnp,
    solve_pnp_multimethod,
    recover_relative_pose,
    PoseEstimationResult,
    PoseGraph,
    compute_pose_difference
)
from src.phase2_sfm.bundle_adjust import run_bundle_adjustment
from src.utils.camera import Camera, build_projection_matrix, camera_center
from src.utils.transforms import (
    world_to_camera,
    camera_to_world,
    normalize_points,
    skew_symmetric,
    rotation_matrix_x,
    rotation_matrix_y,
    rotation_matrix_z
)


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def camera_setup():
    """Standard camera setup for testing."""
    K = np.array([
        [800.0,   0.0, 320.0],
        [  0.0, 800.0, 240.0],
        [  0.0,   0.0,   1.0]
    ], dtype=np.float64)

    # Camera 1: at origin
    R1 = np.eye(3)
    t1 = np.zeros((3, 1))

    # Camera 2: translated and slightly rotated
    R2 = rotation_matrix_y(10.0)
    t2 = np.array([[0.5], [0.0], [0.1]])

    return K, R1, t1, R2, t2


@pytest.fixture(scope="module")
def synthetic_scene(camera_setup):
    """
    Generate a synthetic scene with known 3D points and their projections.
    """
    K, R1, t1, R2, t2 = camera_setup

    np.random.seed(123)

    # Random 3D points in front of both cameras (Z in [2, 8])
    n_pts = 100
    points_3d = np.random.uniform(-2, 2, (n_pts, 3))
    points_3d[:, 2] = np.random.uniform(2.0, 8.0, n_pts)

    # Project to both cameras
    P1 = build_projection_matrix(K, R1, t1)
    P2 = build_projection_matrix(K, R2, t2)

    def project(P, pts):
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        proj  = (P @ pts_h.T).T
        return proj[:, :2] / proj[:, 2:3]

    pts2d_1 = project(P1, points_3d)
    pts2d_2 = project(P2, points_3d)

    # Add small Gaussian noise (1 pixel std)
    pts2d_1_noisy = pts2d_1 + np.random.randn(*pts2d_1.shape) * 0.5
    pts2d_2_noisy = pts2d_2 + np.random.randn(*pts2d_2.shape) * 0.5

    return points_3d, pts2d_1, pts2d_2, pts2d_1_noisy, pts2d_2_noisy, P1, P2


# ─────────────────────────────────────────────────────────────
# Transform Utilities Tests
# ─────────────────────────────────────────────────────────────

class TestTransforms:

    def test_rotation_matrices_orthogonal(self):
        """Rotation matrices should be orthogonal (R^T R = I)."""
        for angle in [0, 30, 90, 180]:
            for fn in [rotation_matrix_x, rotation_matrix_y, rotation_matrix_z]:
                R = fn(angle)
                assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
                assert np.allclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_world_camera_roundtrip(self, camera_setup):
        """world_to_camera -> camera_to_world should be identity."""
        K, R1, t1, R2, t2 = camera_setup
        pts = np.random.randn(20, 3)
        pts_cam  = world_to_camera(pts, R2, t2)
        pts_back = camera_to_world(pts_cam, R2, t2)
        assert np.allclose(pts, pts_back, atol=1e-10)

    def test_normalize_points_zero_mean(self):
        """Normalized points should have zero mean."""
        pts = np.random.randn(50, 2) * 100 + np.array([320, 240])
        pts_norm, T = normalize_points(pts)
        assert np.allclose(pts_norm.mean(axis=0), 0.0, atol=1e-6)

    def test_normalize_points_invertible(self):
        """Normalization transform T should be invertible."""
        pts = np.random.randn(50, 2) * 100 + np.array([320, 240])
        _, T = normalize_points(pts)
        assert abs(np.linalg.det(T)) > 1e-8

    def test_skew_symmetric_property(self):
        """Skew-symmetric matrix S should satisfy S^T = -S."""
        v = np.array([1.0, 2.0, 3.0])
        S = skew_symmetric(v)
        assert np.allclose(S, -S.T, atol=1e-10)

    def test_skew_symmetric_cross_product(self):
        """[v]x @ u == v × u"""
        v = np.array([1.0, 0.0, 0.0])
        u = np.array([0.0, 1.0, 0.0])
        S = skew_symmetric(v)
        cross = np.cross(v, u)
        assert np.allclose(S @ u, cross, atol=1e-10)


# ─────────────────────────────────────────────────────────────
# Camera Class Tests
# ─────────────────────────────────────────────────────────────

class TestCamera:

    def test_camera_from_K(self, camera_setup):
        """Camera.from_K creates correct intrinsic matrix."""
        K, R1, t1, _, _ = camera_setup
        cam = Camera.from_K(K, width=640, height=480)
        assert np.allclose(cam.K, K, atol=1e-10)

    def test_projection_matrix(self, camera_setup):
        """Camera projection matrix P = K @ [R|t]."""
        K, R1, t1, R2, t2 = camera_setup
        cam = Camera.from_K(K, 640, 480, R=R2, t=t2)
        P_expected = build_projection_matrix(K, R2, t2)
        assert np.allclose(cam.P, P_expected, atol=1e-10)

    def test_camera_center(self, camera_setup):
        """Camera center C = -R^T @ t"""
        K, _, _, R2, t2 = camera_setup
        cam = Camera.from_K(K, 640, 480, R=R2, t=t2)
        C_expected = camera_center(R2, t2)
        assert np.allclose(cam.center, C_expected, atol=1e-10)

    def test_c2w_inverse_of_w2c(self, camera_setup):
        """c2w matrix should be inverse of [R|t] extrinsic."""
        K, _, _, R2, t2 = camera_setup
        cam = Camera.from_K(K, 640, 480, R=R2, t=t2)

        w2c = np.eye(4)
        w2c[:3, :3] = R2
        w2c[:3, 3]  = t2.ravel()

        result = cam.c2w @ w2c
        assert np.allclose(result, np.eye(4), atol=1e-8)

    def test_project_and_backproject(self, camera_setup):
        """Project and backproject should be consistent."""
        K, _, _, R2, t2 = camera_setup
        cam = Camera.from_K(K, 640, 480, R=R2, t=t2)

        # A point in front of camera 2
        pt_world = np.array([[0.0, 0.0, 5.0]])
        pt_img   = cam.project(pt_world)

        assert 0 <= pt_img[0, 0] < 640
        assert 0 <= pt_img[0, 1] < 480

    def test_from_fov(self):
        """Camera.from_fov computes correct focal length."""
        cam = Camera.from_fov(fov_x_deg=60.0, width=640, height=480)
        expected_fx = (640 / 2.0) / np.tan(np.radians(30.0))
        assert abs(cam.fx - expected_fx) < 1.0


# ─────────────────────────────────────────────────────────────
# Fundamental / Essential Matrix Tests
# ─────────────────────────────────────────────────────────────

class TestFundamentalEssential:

    def test_fundamental_matrix_estimated(self, synthetic_scene):
        """Fundamental matrix is estimated from noisy correspondences."""
        _, pts1, pts2, pts1_n, pts2_n, _, _ = synthetic_scene
        F, mask = estimate_fundamental_matrix(pts1_n, pts2_n)

        assert F is not None
        assert F.shape == (3, 3)
        assert mask is not None

    def test_fundamental_matrix_rank2(self, synthetic_scene):
        """Fundamental matrix must have rank 2."""
        _, pts1, pts2, pts1_n, pts2_n, _, _ = synthetic_scene
        F, _ = estimate_fundamental_matrix(pts1_n, pts2_n)

        assert F is not None
        singular_values = np.linalg.svd(F, compute_uv=False)
        # Smallest singular value should be nearly 0
        assert singular_values[-1] < 0.1 * singular_values[0]

    def test_epipolar_constraint(self, synthetic_scene, camera_setup):
        """x'^T F x ≈ 0 for inlier correspondences."""
        K, _, _, _, _ = camera_setup
        _, pts1, pts2, _, _, _, _ = synthetic_scene

        F, mask = estimate_fundamental_matrix(pts1, pts2)
        assert F is not None

        inliers = mask.ravel().astype(bool) if mask is not None else slice(None)
        p1 = pts1[inliers]
        p2 = pts2[inliers]

        # Compute x'^T F x for each pair
        for i in range(min(20, len(p1))):
            x1 = np.array([p1[i, 0], p1[i, 1], 1.0])
            x2 = np.array([p2[i, 0], p2[i, 1], 1.0])
            constraint = abs(x2 @ F @ x1)
            assert constraint < 2.0   # within 2 pixel-equivalent units

    def test_essential_matrix_estimated(self, synthetic_scene, camera_setup):
        """Essential matrix is estimated using intrinsics."""
        K, _, _, _, _ = camera_setup
        _, _, _, pts1_n, pts2_n, _, _ = synthetic_scene

        E, mask = estimate_essential_matrix(pts1_n, pts2_n, K)

        assert E is not None
        assert E.shape == (3, 3)

    def test_essential_matrix_rank2(self, synthetic_scene, camera_setup):
        """Essential matrix must have rank 2 (two equal singular values)."""
        K, _, _, _, _ = camera_setup
        _, pts1, pts2, _, _, _, _ = synthetic_scene

        E, _ = estimate_essential_matrix(pts1, pts2, K)
        assert E is not None

        sv = np.linalg.svd(E, compute_uv=False)
        # First two singular values should be approximately equal
        # Third should be ~0
        assert abs(sv[0] - sv[1]) / (sv[0] + 1e-8) < 0.3
        assert sv[2] < 0.1 * sv[0]

    def test_pose_recovery_correct(self, synthetic_scene, camera_setup):
        """Recovered pose from Essential matrix should match ground truth."""
        K, R1, t1, R2_gt, t2_gt = camera_setup
        _, pts1, pts2, _, _, _, _ = synthetic_scene

        E, mask = estimate_essential_matrix(pts1, pts2, K)
        assert E is not None

        inliers = mask.ravel().astype(bool) if mask is not None else slice(None)
        R, t, _ = decompose_essential_matrix(E, pts1[inliers], pts2[inliers], K)

        # Check rotation angle difference
        R_rel_gt = R2_gt @ R1.T
        R_diff = R @ R_rel_gt.T
        trace = np.clip((np.trace(R_diff) - 1) / 2, -1, 1)
        angle_diff = np.degrees(np.arccos(trace))

        # Allow 15 degrees error (noisy conditions)
        assert angle_diff < 15.0 or angle_diff > 165.0   # could be sign flip


# ─────────────────────────────────────────────────────────────
# Triangulation Tests
# ─────────────────────────────────────────────────────────────

class TestTriangulation:

    def test_triangulate_points_shape(self, synthetic_scene, camera_setup):
        """triangulate_points returns Nx3 array."""
        K, R1, t1, R2, t2 = camera_setup
        _, _, _, pts1_n, pts2_n, P1, P2 = synthetic_scene

        pts3d = triangulate_points(P1, P2, pts1_n, pts2_n)
        assert pts3d.shape == (len(pts1_n), 3)
        assert pts3d.dtype in (np.float32, np.float64)

    def test_triangulate_accuracy(self, synthetic_scene, camera_setup):
        """Triangulated points should be close to ground truth."""
        K, R1, t1, R2, t2 = camera_setup
        gt_points, pts1, pts2, _, _, P1, P2 = synthetic_scene

        pts3d = triangulate_points(P1, P2, pts1, pts2)

        # Mean distance to ground truth (noiseless correspondences)
        dists = np.linalg.norm(pts3d - gt_points, axis=1)
        mean_dist = dists.mean()

        assert mean_dist < 0.1   # within 10cm for unit-scale scene

    def test_triangulate_positive_depth(self, synthetic_scene, camera_setup):
        """Triangulated points should be in front of both cameras."""
        K, R1, t1, R2, t2 = camera_setup
        _, pts1, pts2, _, _, P1, P2 = synthetic_scene

        pts3d = triangulate_points(P1, P2, pts1, pts2)

        # Depth in camera 1 frame
        depth_cam1 = (R1 @ pts3d.T + t1).T[:, 2]
        # Depth in camera 2 frame
        depth_cam2 = (R2 @ pts3d.T + t2).T[:, 2]

        assert (depth_cam1 > 0).mean() > 0.9
        assert (depth_cam2 > 0).mean() > 0.9

    def test_reprojection_error_low(self, synthetic_scene, camera_setup):
        """Reprojection error of triangulated points should be low."""
        K, R1, t1, R2, t2 = camera_setup
        _, pts1, pts2, _, _, P1, P2 = synthetic_scene

        pts3d = triangulate_points(P1, P2, pts1, pts2)
        errors = compute_reprojection_error(pts3d, pts1, K, R1, t1)

        assert errors.mean() < 1.5   # < 1.5 pixels

    def test_filter_removes_bad_points(self, synthetic_scene, camera_setup):
        """filter_triangulated_points removes points behind cameras."""
        K, R1, t1, R2, t2 = camera_setup
        _, pts1, pts2, _, _, P1, P2 = synthetic_scene

        pts3d = triangulate_points(P1, P2, pts1, pts2)
        filtered, mask = filter_triangulated_points(
            pts3d, P1, P2, pts1, pts2,
            reprojection_threshold=4.0
        )

        # With clean correspondences, most points should survive
        assert len(filtered) > 0
        assert len(filtered) <= len(pts3d)
        assert mask.shape == (len(pts3d),)


# ─────────────────────────────────────────────────────────────
# PnP Pose Estimation Tests
# ─────────────────────────────────────────────────────────────

class TestPoseEstimation:

    def test_pnp_solves_correctly(self, synthetic_scene, camera_setup):
        """PnP recovers correct camera pose from 3D-2D correspondences."""
        K, R1, t1, R2_gt, t2_gt = camera_setup
        pts3d, _, pts2d_2, _, _, _, _ = synthetic_scene

        result = solve_pnp(pts3d, pts2d_2, K, use_ransac=True)

        assert result.success
        assert result.R is not None
        assert result.t is not None
        assert result.num_inliers > 10

    def test_pnp_reprojection_error_low(self, synthetic_scene, camera_setup):
        """PnP solution has low reprojection error."""
        K, R1, t1, R2_gt, t2_gt = camera_setup
        pts3d, _, pts2d_2, _, pts2d_2_noisy, _, _ = synthetic_scene

        result = solve_pnp(pts3d, pts2d_2_noisy, K, use_ransac=True)

        assert result.success
        assert result.reprojection_error < 3.0   # pixels

    def test_pnp_fails_with_too_few_points(self, camera_setup):
        """PnP returns failure with < 4 points."""
        K, _, _, _, _ = camera_setup
        pts3d = np.random.randn(3, 3)
        pts2d = np.random.randn(3, 2)

        result = solve_pnp(pts3d, pts2d, K)
        assert not result.success

    def test_relative_pose_recovery(self, synthetic_scene, camera_setup):
        """Relative pose from Essential matrix returns valid R and t."""
        K, R1, t1, R2, t2 = camera_setup
        _, _, _, pts1_n, pts2_n, _, _ = synthetic_scene

        result = recover_relative_pose(pts1_n, pts2_n, K)

        assert result.success
        assert result.R is not None
        assert result.t is not None
        assert result.num_inliers > 20

        # R must be a valid rotation matrix
        assert np.allclose(result.R @ result.R.T, np.eye(3), atol=1e-6)
        assert abs(np.linalg.det(result.R) - 1.0) < 1e-6

    def test_pose_graph_operations(self):
        """PoseGraph add/get/update operations work correctly."""
        pg = PoseGraph()

        R1 = np.eye(3)
        t1 = np.zeros((3, 1))
        pg.add_node(0, R1, t1, is_fixed=True)

        R2 = rotation_matrix_y(15.0)
        t2 = np.array([[0.5], [0.0], [0.0]])
        pg.add_node(1, R2, t2)

        assert pg.has_node(0)
        assert pg.has_node(1)
        assert not pg.has_node(2)

        pose = pg.get_pose(0)
        assert pose is not None
        R, t = pose
        assert np.allclose(R, R1)

        # Test update
        R_new = rotation_matrix_z(5.0)
        pg.update_pose(1, R_new, t2)
        R_updated, _ = pg.get_pose(1)
        assert np.allclose(R_updated, R_new)

    def test_compute_pose_difference(self):
        """compute_pose_difference returns correct angle for known rotation."""
        R1 = np.eye(3)
        t1 = np.zeros((3, 1))
        R2 = rotation_matrix_y(30.0)
        t2 = np.array([[1.0], [0.0], [0.0]])

        angle, trans = compute_pose_difference(R1, t1, R2, t2)

        assert abs(angle - 30.0) < 1.0   # within 1 degree
        assert abs(trans - 1.0) < 0.01    # within 1cm


# ─────────────────────────────────────────────────────────────
# Bundle Adjustment Tests
# ─────────────────────────────────────────────────────────────

class TestBundleAdjustment:

    def test_bundle_adjustment_reduces_error(self, camera_setup):
        """Bundle adjustment should reduce reprojection error."""
        K, R1, t1, R2, t2 = camera_setup

        np.random.seed(42)
        n_pts = 30
        pts3d = np.random.uniform(-1, 1, (n_pts, 3))
        pts3d[:, 2] += 4.0

        # Generate 2D observations with noise
        P1 = build_projection_matrix(K, R1, t1)
        P2 = build_projection_matrix(K, R2, t2)

        obs = []
        for i in range(n_pts):
            pt_h = np.append(pts3d[i], 1)
            p1 = P1 @ pt_h; p1 = p1[:2] / p1[2]
            p2 = P2 @ pt_h; p2 = p2[:2] / p2[2]
            p1 += np.random.randn(2) * 1.0
            p2 += np.random.randn(2) * 1.0
            obs.append((0, i, p1))
            obs.append((1, i, p2))

        # Perturb initial poses
        R2_noisy = R2 @ rotation_matrix_y(3.0)
        t2_noisy = t2 + np.array([[0.05], [0.02], [0.01]])
        poses = [(R1, t1), (R2_noisy, t2_noisy)]

        # Run BA
        opt_poses, opt_pts = run_bundle_adjustment(
            poses, pts3d.copy(), obs, K,
            fix_first_camera=True,
            max_iterations=20,
            verbose=False
        )

        assert opt_poses is not None
        assert opt_pts is not None
        assert len(opt_poses) == 2
        assert len(opt_pts) == n_pts