"""
I/O utilities — image loading, PLY/OBJ export, file management.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from PIL import Image


VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def load_image(
    filepath: str,
    color_mode: str = "bgr",
    max_dim: Optional[int] = None
) -> np.ndarray:
    """
    Load an image from disk.

    Args:
        filepath: Path to image file
        color_mode: "bgr" (OpenCV default), "rgb", or "gray"
        max_dim: If set, resize so max(H,W) <= max_dim

    Returns:
        Image as numpy array (H, W, C) or (H, W) for gray
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Image not found: {filepath}")

    img = cv2.imread(filepath)
    if img is None:
        raise ValueError(f"Failed to load image: {filepath}")

    if max_dim is not None:
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if color_mode == "rgb":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif color_mode == "gray":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif color_mode != "bgr":
        raise ValueError(f"Unknown color_mode: {color_mode}")

    return img


def save_image(image: np.ndarray, filepath: str, color_mode: str = "bgr") -> None:
    """
    Save a numpy array as an image.

    Args:
        image: (H, W, C) or (H, W) array. Float images assumed [0,1]; uint8 [0,255]
        filepath: Output path
        color_mode: "bgr" (default), "rgb" (will convert to BGR for saving)
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    # Convert float to uint8
    if image.dtype in (np.float32, np.float64):
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    if color_mode == "rgb" and len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(filepath, image)


def load_images_from_dir(
    directory: str,
    max_dim: Optional[int] = None,
    color_mode: str = "bgr",
    sort: bool = True
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load all valid images from a directory.

    Returns:
        (images, filenames) where filenames are basename strings
    """
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Not a directory: {directory}")

    files = [
        f for f in os.listdir(directory)
        if Path(f).suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]

    if sort:
        files = sorted(files)

    images = []
    valid_files = []

    for fname in files:
        fpath = os.path.join(directory, fname)
        try:
            img = load_image(fpath, color_mode=color_mode, max_dim=max_dim)
            images.append(img)
            valid_files.append(fname)
        except (FileNotFoundError, ValueError) as e:
            print(f"  Warning: Skipping {fname} — {e}")

    return images, valid_files


def export_ply(
    filepath: str,
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    faces: Optional[np.ndarray] = None,
    binary: bool = False
) -> None:
    """
    Export a point cloud or mesh to PLY format.

    Args:
        filepath: Output .ply path
        points: (N, 3) float array of XYZ coordinates
        colors: (N, 3) float [0,1] or uint8 [0,255] RGB colors (optional)
        normals: (N, 3) float normal vectors (optional)
        faces: (M, 3) int triangle face indices (optional, for mesh)
        binary: If True, write binary PLY (smaller, faster); else ASCII
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    n_pts = len(points)
    n_faces = len(faces) if faces is not None else 0

    has_colors = colors is not None and len(colors) == n_pts
    has_normals = normals is not None and len(normals) == n_pts

    # Convert colors to uint8
    colors_u8 = None
    if has_colors:
        c = np.array(colors)
        if c.dtype in (np.float32, np.float64):
            colors_u8 = (np.clip(c, 0, 1) * 255).astype(np.uint8)
        else:
            colors_u8 = c.astype(np.uint8)

    with open(filepath, 'w', encoding='ascii') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_pts}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if has_normals:
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
        if has_colors:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        if n_faces > 0:
            f.write(f"element face {n_faces}\n")
            f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        # Vertex data
        for i in range(n_pts):
            x, y, z = points[i]
            line = f"{x:.6f} {y:.6f} {z:.6f}"
            if has_normals:
                nx, ny, nz = normals[i]
                line += f" {nx:.6f} {ny:.6f} {nz:.6f}"
            if has_colors:
                r, g, b = colors_u8[i]
                line += f" {r} {g} {b}"
            f.write(line + "\n")

        # Face data
        if n_faces > 0:
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    print(f"Saved PLY: {filepath} ({n_pts} points, {n_faces} faces)")


def export_obj(
    filepath: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: Optional[np.ndarray] = None,
    texture_coords: Optional[np.ndarray] = None,
    material_name: Optional[str] = None
) -> None:
    """
    Export a mesh to Wavefront OBJ format.

    Args:
        filepath: Output .obj path
        vertices: (N, 3) float vertex positions
        faces: (M, 3) int face indices (0-based)
        normals: (N, 3) vertex normals (optional)
        texture_coords: (N, 2) UV coordinates (optional)
        material_name: Name for .mtl reference (optional)
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    with open(filepath, 'w') as f:
        f.write(f"# OBJ file generated by 3D Reconstruction Pipeline\n")
        f.write(f"# {len(vertices)} vertices, {len(faces)} faces\n\n")

        if material_name:
            mtl_path = filepath.replace('.obj', '.mtl')
            f.write(f"mtllib {os.path.basename(mtl_path)}\n")
            f.write(f"usemtl {material_name}\n\n")

        # Vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # Texture coordinates
        if texture_coords is not None:
            for uv in texture_coords:
                f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

        # Normals
        if normals is not None:
            for n in normals:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

        # Faces (1-based indices in OBJ)
        f.write("\n")
        for face in faces:
            i0, i1, i2 = face[0] + 1, face[1] + 1, face[2] + 1
            if texture_coords is not None and normals is not None:
                f.write(f"f {i0}/{i0}/{i0} {i1}/{i1}/{i1} {i2}/{i2}/{i2}\n")
            elif normals is not None:
                f.write(f"f {i0}//{i0} {i1}//{i1} {i2}//{i2}\n")
            else:
                f.write(f"f {i0} {i1} {i2}\n")

    print(f"Saved OBJ: {filepath} ({len(vertices)} vertices, {len(faces)} faces)")


def load_ply(filepath: str) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load a PLY file and return (points, colors, normals).
    Simple ASCII PLY reader — for binary use trimesh.
    """
    import trimesh
    mesh_or_pcd = trimesh.load(filepath)

    if isinstance(mesh_or_pcd, trimesh.PointCloud):
        points = np.array(mesh_or_pcd.vertices)
        colors = np.array(mesh_or_pcd.colors)[:, :3] / 255.0 if mesh_or_pcd.colors is not None else None
        return points, colors, None

    elif isinstance(mesh_or_pcd, trimesh.Trimesh):
        points = np.array(mesh_or_pcd.vertices)
        colors = None
        normals = np.array(mesh_or_pcd.vertex_normals)
        return points, colors, normals

    else:
        raise ValueError(f"Unknown PLY content type: {type(mesh_or_pcd)}")


def images_to_video(
    image_paths: List[str],
    output_path: str,
    fps: int = 24
) -> None:
    """Combine a list of images into an MP4 video."""
    import imageio.v3 as iio

    frames = []
    for p in image_paths:
        img = iio.imread(p)
        if img.shape[-1] == 4:      # Drop alpha if present
            img = img[:, :, :3]
        frames.append(img)

    iio.imwrite(output_path, frames, fps=fps, codec='libx264')
    print(f"Saved video: {output_path}")