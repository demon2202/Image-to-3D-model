"""
Phase 3 — Dense Reconstruction (MVS)
"""

from .stereo_matching import DenseStereo, rectify_stereo_pair
from .mesh            import (
    estimate_normals_pca,
    poisson_surface_reconstruction,
    clean_mesh,
    decimate_mesh,
    smooth_mesh,
    save_mesh,
    load_mesh,
    compute_mesh_quality,
    build_dense_point_cloud,
)
from .patch_match     import PatchMatchDepthEstimator, MultiViewPatchMatch, PatchMatchConfig

__all__ = [
    "DenseStereo",
    "rectify_stereo_pair",
    "estimate_normals_pca",
    "poisson_surface_reconstruction",
    "clean_mesh",
    "decimate_mesh",
    "smooth_mesh",
    "save_mesh",
    "load_mesh",
    "compute_mesh_quality",
    "build_dense_point_cloud",
    "PatchMatchDepthEstimator",
    "MultiViewPatchMatch",
    "PatchMatchConfig",
]