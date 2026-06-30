import os
import numpy as np
import trimesh
from scipy.spatial import KDTree
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh
from typing import Optional, Tuple, List
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────
# Normal Estimation
# ─────────────────────────────────────────────────────────────

def estimate_normals_pca(
    points: np.ndarray,
    k_neighbors: int = 20,
    view_point: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Estimate surface normals via PCA on local point neighborhoods.

    For each point:
      1. Find k nearest neighbors
      2. Compute local covariance matrix
      3. Normal = eigenvector with smallest eigenvalue

    Args:
        points      : (N, 3) point cloud
        k_neighbors : number of nearest neighbors to use
        view_point  : (3,) camera viewpoint for consistent normal orientation
                      If None, normals are oriented toward centroid

    Returns:
        normals: (N, 3) unit normal vectors
    """
    n = len(points)
    normals = np.zeros((n, 3), dtype=np.float64)

    tree = KDTree(points)
    # +1 because query returns the point itself at distance 0
    dists, indices = tree.query(points, k=min(k_neighbors + 1, n))

    for i in range(n):
        neighbors = points[indices[i]]           # (k+1, 3)
        centroid  = neighbors.mean(axis=0)
        centered  = neighbors - centroid

        # Covariance matrix
        cov = (centered.T @ centered) / len(neighbors)

        # Smallest eigenvector = normal direction
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        normal = eigenvectors[:, 0]              # smallest eigenvalue

        normals[i] = normal

    # Orient normals consistently
    if view_point is not None:
        vp = np.array(view_point, dtype=np.float64)
        for i in range(n):
            if np.dot(normals[i], vp - points[i]) < 0:
                normals[i] = -normals[i]
    else:
        # Orient toward centroid of entire cloud
        centroid = points.mean(axis=0)
        for i in range(n):
            if np.dot(normals[i], centroid - points[i]) < 0:
                normals[i] = -normals[i]

    # Normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / (norms + 1e-8)

    return normals


def propagate_normal_orientation(
    points: np.ndarray,
    normals: np.ndarray,
    k_neighbors: int = 15
) -> np.ndarray:
    """
    Propagate consistent normal orientation using minimum spanning tree.
    Implements Hoppe et al. 1992 normal consistency propagation.

    Args:
        points  : (N, 3) point positions
        normals : (N, 3) initial (possibly inconsistent) normals

    Returns:
        normals_consistent: (N, 3) consistently oriented normals
    """
    n = len(points)
    if n < 2:
        return normals

    tree = KDTree(points)
    _, indices = tree.query(points, k=min(k_neighbors + 1, n))

    # Build Riemannian graph with edge weights = |1 - |ni . nj||
    # Higher weight = more inconsistent, we want minimum weight MST
    adj = lil_matrix((n, n), dtype=np.float64)

    for i in range(n):
        for j_idx in range(1, len(indices[i])):
            j = indices[i][j_idx]
            cost = 1.0 - abs(np.dot(normals[i], normals[j]))
            adj[i, j] = cost
            adj[j, i] = cost

    # Compute Minimum Spanning Tree (forest) using SciPy
    from scipy.sparse.csgraph import minimum_spanning_tree
    mst = minimum_spanning_tree(adj.tocsr())

    # Make MST undirected for traversal
    mst_undirected = mst + mst.T
    indptr = mst_undirected.indptr
    indices_mst = mst_undirected.indices

    # Traverse the MST using BFS to consistently orient normals
    from collections import deque
    visited = np.zeros(n, dtype=bool)
    normals_out = normals.copy()

    for root in range(n):
        if not visited[root]:
            q = deque([root])
            visited[root] = True
            while q:
                i = q.popleft()
                start = indptr[i]
                end = indptr[i+1]
                nbrs = indices_mst[start:end]
                for j in nbrs:
                    if not visited[j]:
                        visited[j] = True
                        if np.dot(normals_out[i], normals_out[j]) < 0:
                            normals_out[j] = -normals_out[j]
                        q.append(j)

    # Normalize
    norms = np.linalg.norm(normals_out, axis=1, keepdims=True)
    return normals_out / (norms + 1e-8)


# ─────────────────────────────────────────────────────────────
# Surface Reconstruction
# ─────────────────────────────────────────────────────────────

def poisson_surface_reconstruction(
    points: np.ndarray,
    normals: Optional[np.ndarray] = None,
    depth: int = 8,
    k_neighbors: int = 20,
    trim_threshold: float = 5.0,
    min_points: int = 1000
) -> Optional[trimesh.Trimesh]:
    """
    Poisson surface reconstruction.

    Uses trimesh's interface to reconstruct a watertight mesh
    from an oriented point cloud.

    Args:
        points        : (N, 3) point cloud
        normals       : (N, 3) surface normals. Estimated if None.
        depth         : octree depth — higher = finer detail but slower
                        (6=coarse, 8=default, 10=fine, 12=very fine)
        trim_threshold: remove low-density mesh regions
        min_points    : minimum points required to attempt reconstruction

    Returns:
        trimesh.Trimesh or None if reconstruction fails
    """
    if len(points) < min_points:
        print(f"  Poisson: too few points ({len(points)} < {min_points})")
        return None

    # Estimate normals if not provided
    if normals is None:
        print(f"  Estimating normals for {len(points):,} points...")
        normals = estimate_normals_pca(points, k_neighbors=k_neighbors)
        normals = propagate_normal_orientation(points, normals, k_neighbors=15)
        print("  Normal estimation complete")

    # Try trimesh's poisson reconstruction
    try:
        pcd = trimesh.PointCloud(vertices=points)
        pcd.vertices_normal = normals   # type: ignore

        # trimesh uses pysdf or open3d internally; try direct approach
        mesh = _poisson_via_scipy(points, normals, depth)
        return mesh

    except Exception as e:
        print(f"  Poisson via trimesh failed: {e}")

    # Fallback: convex hull (rough but always works)
    print("  Falling back to convex hull reconstruction...")
    try:
        hull = trimesh.convex.convex_hull(points)
        return hull
    except Exception as e2:
        print(f"  Convex hull also failed: {e2}")
        return None


def _poisson_via_scipy(
    points: np.ndarray,
    normals: np.ndarray,
    depth: int = 8
) -> trimesh.Trimesh:
    """
    Approximate Poisson reconstruction using marching cubes on
    a signed distance field. Production systems should use
    CGAL or Open3D's implementation.
    """
    from scipy.ndimage import gaussian_filter

    # Compute bounding box with padding
    pad = 0.1
    mins = points.min(axis=0) - pad
    maxs = points.max(axis=0) + pad

    # Grid resolution based on depth
    res = 2 ** min(depth, 9)     # cap at 512^3 to avoid OOM
    grid_size = min(res, 128)    # further cap for scipy

    # Create voxel grid
    x = np.linspace(mins[0], maxs[0], grid_size)
    y = np.linspace(mins[1], maxs[1], grid_size)
    z = np.linspace(mins[2], maxs[2], grid_size)

    voxel_size = (maxs - mins) / grid_size

    # Splat points onto grid with their normal-weighted SDF
    sdf = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    weights = np.zeros_like(sdf)

    tree = KDTree(points)

    # For each grid point, interpolate SDF from nearby point cloud points
    # This is a simplified implicit surface — not true Poisson
    gx, gy, gz = np.meshgrid(x, y, z, indexing='ij')
    grid_pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    # Process in chunks to avoid memory issues
    chunk = 50_000
    sdf_flat = np.zeros(len(grid_pts), dtype=np.float32)

    for start in range(0, len(grid_pts), chunk):
        end  = min(start + chunk, len(grid_pts))
        gp   = grid_pts[start:end]
        dists, idxs = tree.query(gp, k=min(10, len(points)))

        for ci in range(len(gp)):
            wsum  = 0.0
            sdsum = 0.0
            for ki in range(len(idxs[ci])):
                pt_idx = idxs[ci][ki]
                d = dists[ci][ki]
                if d < 1e-8:
                    continue
                w = np.exp(-d * d / (2 * voxel_size[0] ** 2))
                diff = gp[ci] - points[pt_idx]
                sd = np.dot(diff, normals[pt_idx])
                sdsum += w * sd
                wsum  += w

            sdf_flat[start + ci] = sdsum / (wsum + 1e-8)

    sdf = sdf_flat.reshape(grid_size, grid_size, grid_size)

    # Smooth the SDF
    sdf = gaussian_filter(sdf, sigma=1.0)

    # Marching cubes at iso-level 0
    try:
        from skimage.measure import marching_cubes
        verts_idx, faces, vertex_normals, _ = marching_cubes(
            sdf, level=0.0, spacing=tuple(voxel_size)
        )
        # Shift vertices to world coordinates
        verts_idx += mins

        mesh = trimesh.Trimesh(
            vertices=verts_idx,
            faces=faces,
            vertex_normals=vertex_normals,
            process=True
        )
        return mesh

    except Exception as e:
        raise RuntimeError(f"Marching cubes failed: {e}")


def ball_pivoting_reconstruction(
    points: np.ndarray,
    normals: Optional[np.ndarray] = None,
    radii: Optional[List[float]] = None
) -> Optional[trimesh.Trimesh]:
    """
    Ball-Pivoting Algorithm (BPA) surface reconstruction.
    Good for evenly sampled point clouds without holes.

    Args:
        points : (N, 3) point cloud
        normals: (N, 3) normals (estimated if None)
        radii  : list of ball radii to try (auto-computed if None)

    Returns:
        trimesh.Trimesh or None
    """
    if normals is None:
        normals = estimate_normals_pca(points, k_neighbors=20)

    if radii is None:
        # Auto-estimate: use 2-5x mean nearest neighbor distance
        tree = KDTree(points)
        dists, _ = tree.query(points, k=2)
        avg_dist = dists[:, 1].mean()
        radii = [avg_dist * r for r in [1.5, 2.5, 4.0]]

    try:
        pcd = trimesh.PointCloud(vertices=points.astype(np.float32))
        # BPA via trimesh
        mesh = trimesh.reconstruction.ball_pivoting(
            pcd,
            radii=radii
        )
        return mesh
    except Exception as e:
        print(f"  Ball pivoting failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# Mesh Cleaning and Post-processing
# ─────────────────────────────────────────────────────────────

def clean_mesh(
    mesh: trimesh.Trimesh,
    remove_small_components: bool = True,
    min_component_ratio: float = 0.01,
    remove_duplicate_faces: bool = True,
    remove_degenerate_faces: bool = True,
    fill_holes: bool = False
) -> trimesh.Trimesh:
    """
    Clean a reconstructed mesh by removing artifacts.

    Args:
        mesh                    : Input trimesh mesh
        remove_small_components : Remove disconnected components smaller than
                                  min_component_ratio * largest_component
        min_component_ratio     : Threshold for component removal
        remove_duplicate_faces  : Remove duplicated triangles
        remove_degenerate_faces : Remove zero-area triangles
        fill_holes              : Attempt to fill mesh holes

    Returns:
        Cleaned trimesh.Trimesh
    """
    print(f"  Cleaning mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")

    # Remove duplicate/degenerate faces
    if remove_duplicate_faces or remove_degenerate_faces:
        mesh = mesh.process(validate=True)

    # Remove small disconnected components
    if remove_small_components:
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            sizes = [len(c.faces) for c in components]
            max_size = max(sizes)
            threshold = max_size * min_component_ratio

            kept = [c for c in components if len(c.faces) >= threshold]
            if kept:
                mesh = trimesh.util.concatenate(kept)
                print(f"  Kept {len(kept)}/{len(components)} components "
                      f"({len(mesh.faces):,} faces remaining)")

    # Fill holes (experimental)
    if fill_holes and not mesh.is_watertight:
        try:
            trimesh.repair.fill_holes(mesh)
        except Exception:
            pass

    # Final clean pass
    mesh.remove_unreferenced_vertices()
    # remove_duplicate_faces() was removed from trimesh's public API;
    # rebuild the mesh with process=True which handles deduplication internally.
    mesh = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        process=True,
        validate=True
    )

    print(f"  Clean mesh: {len(mesh.vertices):,} vertices, "
          f"{len(mesh.faces):,} faces, "
          f"watertight={mesh.is_watertight}")

    return mesh


def decimate_mesh(
    mesh: trimesh.Trimesh,
    target_faces: int = 50_000,
    method: str = "quadric"
) -> trimesh.Trimesh:
    """
    Reduce mesh face count while preserving shape.

    Args:
        mesh        : Input mesh
        target_faces: Target number of faces
        method      : "quadric" (QEM) or "vertex_clustering"

    Returns:
        Decimated trimesh.Trimesh
    """
    if len(mesh.faces) <= target_faces:
        print(f"  Mesh already has {len(mesh.faces):,} faces "
              f"(<= target {target_faces:,}), skipping decimation.")
        return mesh

    print(f"  Decimating {len(mesh.faces):,} -> {target_faces:,} faces...")

    try:
        # trimesh uses simplify_quadric_decimation
        decimated = mesh.simplify_quadric_decimation(target_faces)
        print(f"  Decimation result: {len(decimated.faces):,} faces")
        return decimated
    except Exception as e:
        print(f"  Quadric decimation failed: {e}. Using vertex clustering...")

        # Fallback: vertex clustering (voxel-based)
        voxel_size = mesh.scale / (target_faces ** (1 / 3)) * 2
        try:
            clustered = mesh.voxelized(pitch=voxel_size).marching_cubes
            return clustered
        except Exception as e2:
            print(f"  Vertex clustering also failed: {e2}. Returning original.")
            return mesh


def smooth_mesh(
    mesh: trimesh.Trimesh,
    iterations: int = 5,
    lamb: float = 0.5
) -> trimesh.Trimesh:
    """
    Laplacian mesh smoothing.

    Args:
        mesh      : Input mesh
        iterations: Number of smoothing iterations
        lamb      : Smoothing factor [0, 1]. Higher = more smooth.

    Returns:
        Smoothed trimesh.Trimesh
    """
    vertices = mesh.vertices.copy().astype(np.float64)
    faces = mesh.faces

    # Build adjacency list
    adj: List[set] = [set() for _ in range(len(vertices))]
    for f in faces:
        for i in range(3):
            for j in range(3):
                if i != j:
                    adj[f[i]].add(f[j])

    for _ in range(iterations):
        new_verts = vertices.copy()
        for i in range(len(vertices)):
            neighbors = list(adj[i])
            if not neighbors:
                continue
            centroid = vertices[neighbors].mean(axis=0)
            new_verts[i] = (1 - lamb) * vertices[i] + lamb * centroid
        vertices = new_verts

    smoothed = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False
    )
    smoothed.compute_vertex_normals()
    return smoothed


def compute_mesh_quality(mesh: trimesh.Trimesh) -> dict:
    """
    Compute mesh quality metrics.

    Returns dict with:
        - num_vertices, num_faces
        - is_watertight
        - surface_area
        - volume (if watertight)
        - euler_number
        - avg_edge_length
        - min/max face area
    """
    face_areas = mesh.area_faces
    edges = mesh.edges_unique_length

    quality = {
        "num_vertices": int(len(mesh.vertices)),
        "num_faces": int(len(mesh.faces)),
        "num_edges": int(len(mesh.edges_unique)),
        "is_watertight": bool(mesh.is_watertight),
        "is_winding_consistent": bool(mesh.is_winding_consistent),
        "surface_area": float(mesh.area),
        "euler_number": int(mesh.euler_number),
        "avg_edge_length": float(edges.mean()) if len(edges) > 0 else 0.0,
        "min_face_area": float(face_areas.min()) if len(face_areas) > 0 else 0.0,
        "max_face_area": float(face_areas.max()) if len(face_areas) > 0 else 0.0,
        "degenerate_faces": int((face_areas < 1e-10).sum()),
    }

    if mesh.is_watertight:
        quality["volume"] = float(abs(mesh.volume))

    return quality


# ─────────────────────────────────────────────────────────────
# Point Cloud → Mesh Helper
# ─────────────────────────────────────────────────────────────

def build_dense_point_cloud(
    depth_maps: List[np.ndarray],
    images: List[np.ndarray],
    poses: List[Tuple[np.ndarray, np.ndarray]],
    K: np.ndarray,
    max_depth: float = 50.0,
    voxel_size: float = 0.02
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge multiple depth maps + images into a unified colored point cloud.

    Args:
        depth_maps : list of (H, W) float depth maps
        images     : list of (H, W, 3) BGR images
        poses      : list of (R, t) camera poses
        K          : 3x3 intrinsic matrix
        max_depth  : clip depth at this value
        voxel_size : downsampling voxel size

    Returns:
        (points_merged, colors_merged) as numpy arrays
    """
    from src.phase3_dense.stereo_matching import DenseStereo

    stereo_helper = DenseStereo()
    all_points = []
    all_colors = []

    for depth, img, (R, t) in zip(depth_maps, images, poses):
        pts, clr = stereo_helper.depth_to_pointcloud(
            depth, K,
            color_image=img,
            R=R, t=t,
            max_depth=max_depth
        )
        if len(pts) > 0:
            all_points.append(pts)
            all_colors.append(clr if clr is not None else np.ones((len(pts), 3)) * 0.6)

    if not all_points:
        return np.zeros((0, 3)), np.zeros((0, 3))

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)

    # Voxel downsample
    if voxel_size > 0 and len(points) > 0:
        min_pt = points.min(axis=0)
        indices = np.floor((points - min_pt) / voxel_size).astype(np.int64)
        keys = indices[:, 0] * 1_000_003 + indices[:, 1] * 1009 + indices[:, 2]
        unique_keys, inv = np.unique(keys, return_inverse=True)

        ds_pts = np.zeros((len(unique_keys), 3))
        ds_clr = np.zeros((len(unique_keys), 3))
        counts = np.bincount(inv).astype(float)

        np.add.at(ds_pts, inv, points)
        np.add.at(ds_clr, inv, colors)

        ds_pts /= counts[:, None]
        ds_clr /= counts[:, None]
        return ds_pts, np.clip(ds_clr, 0, 1)

    return points, np.clip(colors, 0, 1)


# ─────────────────────────────────────────────────────────────
# Save / Load
# ─────────────────────────────────────────────────────────────

def save_mesh(
    mesh: trimesh.Trimesh,
    filepath: str,
    include_color: bool = True
) -> None:
    """
    Save a trimesh mesh to PLY or OBJ format.

    Args:
        mesh      : trimesh.Trimesh object
        filepath  : output path (.ply or .obj)
        include_color: whether to include vertex colors if available
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    ext = os.path.splitext(filepath)[1].lower()

    if ext not in ('.ply', '.obj', '.stl', '.glb'):
        raise ValueError(f"Unsupported mesh format: {ext}")

    mesh.export(filepath)
    size_mb = os.path.getsize(filepath) / 1024 / 1024
    print(f"Saved mesh: {filepath} ({len(mesh.vertices):,} verts, "
          f"{len(mesh.faces):,} faces, {size_mb:.1f} MB)")


def load_mesh(filepath: str) -> trimesh.Trimesh:
    """Load a mesh from PLY, OBJ, or STL file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Mesh file not found: {filepath}")
    return trimesh.load(filepath, force='mesh')


def export_mesh_stats(mesh: trimesh.Trimesh, output_path: str) -> dict:
    """Compute quality metrics and save as JSON."""
    import json
    quality = compute_mesh_quality(mesh)
    with open(output_path, 'w') as f:
        json.dump(quality, f, indent=2)
    print(f"Mesh stats saved: {output_path}")
    return quality