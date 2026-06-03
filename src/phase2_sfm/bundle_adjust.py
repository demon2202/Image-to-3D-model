"""
Bundle Adjustment using scipy's least_squares (Levenberg-Marquardt / TRF).

Fixed for scipy >= 1.16 compatibility:
  - No longer uses equal lower/upper bounds to fix parameters
  - Instead excludes fixed camera parameters from the optimization vector
    and reconstructs the full parameter vector inside the residual function
"""

import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from typing import Tuple, List, Optional


# ─────────────────────────────────────────────────────────────
# Rodrigues helpers
# ─────────────────────────────────────────────────────────────

def rodrigues_to_matrix(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    return R


def matrix_to_rodrigues(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(R)
    return rvec.ravel()


# ─────────────────────────────────────────────────────────────
# Projection
# ─────────────────────────────────────────────────────────────

def project_point(
    point_3d: np.ndarray,
    camera_params: np.ndarray,
    K: np.ndarray
) -> np.ndarray:
    """Project a 3D point using camera parameters [rx, ry, rz, tx, ty, tz]."""
    rvec = camera_params[:3]
    tvec = camera_params[3:6]
    R    = rodrigues_to_matrix(rvec)
    p_cam  = R @ point_3d + tvec
    p_proj = K @ p_cam
    return p_proj[:2] / (p_proj[2] + 1e-10)


# ─────────────────────────────────────────────────────────────
# Residuals  (works on the FULL parameter vector)
# ─────────────────────────────────────────────────────────────

def _full_residuals(
    params: np.ndarray,
    n_cameras: int,
    n_points: int,
    camera_indices: np.ndarray,
    point_indices: np.ndarray,
    points_2d: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Compute 2*n_obs residuals from the FULL parameter vector."""
    camera_params = params[: n_cameras * 6].reshape(n_cameras, 6)
    points_3d     = params[n_cameras * 6 :].reshape(n_points, 3)

    residuals = np.empty(len(camera_indices) * 2, dtype=np.float64)

    for i, (cam_idx, pt_idx) in enumerate(zip(camera_indices, point_indices)):
        proj = project_point(points_3d[pt_idx], camera_params[cam_idx], K)
        residuals[2 * i    ] = proj[0] - points_2d[i, 0]
        residuals[2 * i + 1] = proj[1] - points_2d[i, 1]

    return residuals


# ─────────────────────────────────────────────────────────────
# Residuals  (works on the REDUCED parameter vector, camera-0 fixed)
# ─────────────────────────────────────────────────────────────

def _reduced_residuals(
    free_params: np.ndarray,
    fixed_cam_params: np.ndarray,       # shape (6,) — camera 0
    n_cameras: int,
    n_points: int,
    camera_indices: np.ndarray,
    point_indices: np.ndarray,
    points_2d: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """
    Compute residuals when camera-0 is held fixed.
    free_params = [(n_cameras-1)*6 camera params] + [n_points*3 point params]
    """
    n_free_cams = n_cameras - 1
    free_cam_block = free_params[: n_free_cams * 6].reshape(n_free_cams, 6)
    points_3d      = free_params[n_free_cams * 6 :].reshape(n_points, 3)

    # Rebuild full camera_params: insert fixed cam at index 0
    camera_params = np.vstack([fixed_cam_params[None, :], free_cam_block])

    residuals = np.empty(len(camera_indices) * 2, dtype=np.float64)

    for i, (cam_idx, pt_idx) in enumerate(zip(camera_indices, point_indices)):
        proj = project_point(points_3d[pt_idx], camera_params[cam_idx], K)
        residuals[2 * i    ] = proj[0] - points_2d[i, 0]
        residuals[2 * i + 1] = proj[1] - points_2d[i, 1]

    return residuals


# ─────────────────────────────────────────────────────────────
# Sparsity matrix builder
# ─────────────────────────────────────────────────────────────

def _build_sparsity(
    n_cameras: int,
    n_points: int,
    camera_indices: np.ndarray,
    point_indices: np.ndarray,
    fix_first: bool,
) -> lil_matrix:
    """
    Build Jacobian sparsity for the (possibly reduced) parameter vector.
    """
    n_obs = len(camera_indices)
    m     = n_obs * 2                            # number of residuals

    n_cam_params = (n_cameras - 1 if fix_first else n_cameras) * 6
    n_pt_params  = n_points * 3
    n            = n_cam_params + n_pt_params

    A = lil_matrix((m, n), dtype=int)

    # Map original camera index → column offset in reduced vector
    # Camera 0 is fixed → removed; camera k (k>0) → slot k-1
    def _cam_col(cam_idx: int) -> Optional[int]:
        if fix_first:
            if cam_idx == 0:
                return None          # fixed — no column
            return (cam_idx - 1) * 6
        return cam_idx * 6

    for i, (cam_idx, pt_idx) in enumerate(zip(camera_indices, point_indices)):
        col_cam = _cam_col(cam_idx)
        col_pt  = n_cam_params + pt_idx * 3

        if col_cam is not None:
            A[2 * i : 2 * i + 2, col_cam : col_cam + 6] = 1

        A[2 * i : 2 * i + 2, col_pt : col_pt + 3] = 1

    return A


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def run_bundle_adjustment(
    camera_poses: List[Tuple[np.ndarray, np.ndarray]],
    points_3d: np.ndarray,
    observations: List[Tuple[int, int, np.ndarray]],
    K: np.ndarray,
    fix_first_camera: bool = True,
    max_iterations: int = 50,
    verbose: bool = True,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """
    Run sparse bundle adjustment.

    Args:
        camera_poses  : list of (R 3×3, t 3×1) — one per camera
        points_3d     : (M, 3) float64 3-D points
        observations  : list of (camera_index, point_index, 2d_point np.ndarray)
        K             : 3×3 camera intrinsic matrix
        fix_first_camera : keep camera 0 frozen (recommended — removes gauge freedom)
        max_iterations   : maximum optimiser evaluations
        verbose          : print before/after reprojection error

    Returns:
        (optimized_poses, optimized_points_3d)
    """
    n_cameras = len(camera_poses)
    n_points  = len(points_3d)

    # ── Pack full camera parameter block ──────────────────────
    camera_params = np.zeros((n_cameras, 6), dtype=np.float64)
    for i, (R, t) in enumerate(camera_poses):
        camera_params[i, :3] = matrix_to_rodrigues(R)
        camera_params[i, 3:] = t.ravel()

    # ── Unpack observation lists ──────────────────────────────
    camera_indices = np.array([obs[0] for obs in observations], dtype=np.int32)
    point_indices  = np.array([obs[1] for obs in observations], dtype=np.int32)
    points_2d      = np.array([obs[2] for obs in observations], dtype=np.float64)

    if points_2d.ndim == 1:
        points_2d = points_2d.reshape(-1, 2)

    # ── Build initial parameter vector (free params only) ─────
    if fix_first_camera and n_cameras > 1:
        fixed_cam  = camera_params[0].copy()        # (6,) — frozen
        free_cams  = camera_params[1:].ravel()       # (n_cameras-1)*6
        x0         = np.concatenate([free_cams, points_3d.ravel()])

        sparsity = _build_sparsity(
            n_cameras, n_points, camera_indices, point_indices, fix_first=True
        )

        # Diagnostic: initial residual
        if verbose:
            r0 = _reduced_residuals(
                x0, fixed_cam, n_cameras, n_points,
                camera_indices, point_indices, points_2d, K
            )
            print(f"  BA initial  mean |residual|: {np.mean(np.abs(r0)):.4f} px")

        result = least_squares(
            _reduced_residuals,
            x0,
            jac_sparsity=sparsity,
            verbose=2 if verbose else 0,
            x_scale="jac",
            ftol=1e-6,
            xtol=1e-6,
            gtol=1e-8,
            method="trf",
            max_nfev=max_iterations,
            args=(
                fixed_cam, n_cameras, n_points,
                camera_indices, point_indices, points_2d, K,
            ),
        )

        if verbose:
            print(f"  BA final    mean |residual|: {np.mean(np.abs(result.fun)):.4f} px")

        # ── Unpack result ─────────────────────────────────────
        n_free = n_cameras - 1
        free_cam_out = result.x[: n_free * 6].reshape(n_free, 6)
        pts_out      = result.x[n_free * 6 :].reshape(n_points, 3)

        # Rebuild full camera_params (insert fixed camera 0)
        full_cam_out = np.vstack([fixed_cam[None, :], free_cam_out])

    else:
        # No fixing — optimise all cameras
        x0 = np.concatenate([camera_params.ravel(), points_3d.ravel()])

        sparsity = _build_sparsity(
            n_cameras, n_points, camera_indices, point_indices, fix_first=False
        )

        if verbose:
            r0 = _full_residuals(
                x0, n_cameras, n_points,
                camera_indices, point_indices, points_2d, K
            )
            print(f"  BA initial  mean |residual|: {np.mean(np.abs(r0)):.4f} px")

        result = least_squares(
            _full_residuals,
            x0,
            jac_sparsity=sparsity,
            verbose=2 if verbose else 0,
            x_scale="jac",
            ftol=1e-6,
            xtol=1e-6,
            gtol=1e-8,
            method="trf",
            max_nfev=max_iterations,
            args=(
                n_cameras, n_points,
                camera_indices, point_indices, points_2d, K,
            ),
        )

        if verbose:
            print(f"  BA final    mean |residual|: {np.mean(np.abs(result.fun)):.4f} px")

        full_cam_out = result.x[: n_cameras * 6].reshape(n_cameras, 6)
        pts_out      = result.x[n_cameras * 6 :].reshape(n_points, 3)

    # ── Convert back to (R, t) pairs ─────────────────────────
    optimized_poses: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_cameras):
        R_opt = rodrigues_to_matrix(full_cam_out[i, :3])
        t_opt = full_cam_out[i, 3:].reshape(3, 1)
        optimized_poses.append((R_opt, t_opt))

    return optimized_poses, pts_out