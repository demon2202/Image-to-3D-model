"""
Generate synthetic multi-view images of a 3D scene for pipeline testing.

Creates:
  - data/raw/          : 30 rendered images (640x480)
  - data/raw/gt_poses.json : ground truth camera poses
  - data/nerf_synthetic/synthetic_scene/ : NeRF-compatible format

Usage:
    python scripts/generate_synthetic_data.py
    python scripts/generate_synthetic_data.py --n_views 30 --scene lego
"""

import sys
import os
import json
import argparse
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic multi-view dataset")
    p.add_argument("--n_views",  type=int,   default=30,
                   help="Number of views to render (default: 30)")
    p.add_argument("--width",    type=int,   default=640,
                   help="Image width  (default: 640)")
    p.add_argument("--height",   type=int,   default=480,
                   help="Image height (default: 480)")
    p.add_argument("--scene",    type=str,   default="colored_boxes",
                   choices=["colored_boxes", "spheres", "checkerboard"],
                   help="Scene type to render")
    p.add_argument("--output_raw",  default="data/raw",
                   help="Output directory for raw images (SfM input)")
    p.add_argument("--output_nerf", default="data/nerf_synthetic/synthetic_scene",
                   help="Output directory for NeRF-format dataset")
    p.add_argument("--radius",   type=float, default=3.5,
                   help="Camera orbit radius")
    p.add_argument("--noise",    type=float, default=0.0,
                   help="Image noise std (0 = clean)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# Scene definitions
# ─────────────────────────────────────────────────────────────

def make_scene_colored_boxes(H: int, W: int) -> list:
    """
    Returns list of (center, size, color) for colored boxes.
    """
    objects = [
        # (center_xyz, half_size, BGR_color)
        (np.array([ 0.0,  0.0,  0.0]), 0.5,  (50,  100, 220)),   # blue box
        (np.array([ 1.2,  0.3,  0.4]), 0.35, (50,  200,  50)),    # green box
        (np.array([-1.0,  0.2, -0.3]), 0.4,  (220,  80,  50)),    # red box
        (np.array([ 0.2, -0.6,  0.8]), 0.3,  (200, 180,  40)),    # yellow box
        (np.array([-0.5,  0.6, -0.6]), 0.25, (180,  50, 180)),    # purple box
        # Ground plane represented as a flat box
        (np.array([ 0.0, -0.8,  0.0]), np.array([3.0, 0.1, 3.0]), (120, 120, 120)),
    ]
    return objects


def make_scene_spheres() -> list:
    """Returns list of (center, radius, color) for spheres."""
    return [
        (np.array([ 0.0,  0.0,  0.0]), 0.6,  (50,  100, 220)),
        (np.array([ 1.3,  0.3,  0.3]), 0.4,  (50,  200,  50)),
        (np.array([-1.1,  0.2, -0.4]), 0.45, (220,  80,  50)),
        (np.array([ 0.3, -0.5,  0.9]), 0.35, (200, 180,  40)),
        (np.array([-0.6,  0.7, -0.5]), 0.3,  (180,  50, 180)),
    ]


# ─────────────────────────────────────────────────────────────
# Software ray-caster
# ─────────────────────────────────────────────────────────────

def ray_box_intersect(
    ray_o: np.ndarray,
    ray_d: np.ndarray,
    box_center: np.ndarray,
    box_half: np.ndarray
) -> float:
    """
    Ray-AABB intersection. Returns t (distance) or -1 if no hit.
    """
    box_min = box_center - box_half
    box_max = box_center + box_half

    inv_d = np.where(np.abs(ray_d) > 1e-10, 1.0 / ray_d, np.sign(ray_d) * 1e10)

    t1 = (box_min - ray_o) * inv_d
    t2 = (box_max - ray_o) * inv_d

    t_near = np.maximum.reduce([np.minimum(t1, t2)])
    t_far  = np.minimum.reduce([np.maximum(t1, t2)])

    t_near_max = max(t1[0] if ray_d[0] >= 0 else t2[0],
                     t1[1] if ray_d[1] >= 0 else t2[1],
                     t1[2] if ray_d[2] >= 0 else t2[2])

    t_far_min  = min(t2[0] if ray_d[0] >= 0 else t1[0],
                     t2[1] if ray_d[1] >= 0 else t1[1],
                     t2[2] if ray_d[2] >= 0 else t1[2])

    if t_far_min < 0 or t_near_max > t_far_min:
        return -1.0
    t = t_near_max if t_near_max > 0 else t_far_min
    return float(t) if t > 0 else -1.0


def ray_sphere_intersect(
    ray_o: np.ndarray,
    ray_d: np.ndarray,
    center: np.ndarray,
    radius: float
) -> float:
    """
    Ray-sphere intersection. Returns t or -1.
    """
    oc = ray_o - center
    a  = np.dot(ray_d, ray_d)
    b  = 2.0 * np.dot(oc, ray_d)
    c  = np.dot(oc, oc) - radius * radius
    disc = b * b - 4 * a * c

    if disc < 0:
        return -1.0

    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)

    if t1 > 0:
        return float(t1)
    if t2 > 0:
        return float(t2)
    return -1.0


def compute_box_normal(hit_point: np.ndarray, box_center: np.ndarray) -> np.ndarray:
    """Compute outward normal for a point on an AABB surface."""
    local  = hit_point - box_center
    abs_l  = np.abs(local)
    axis   = np.argmax(abs_l)
    normal = np.zeros(3)
    normal[axis] = np.sign(local[axis])
    return normal


def shade(
    normal: np.ndarray,
    base_color: tuple,
    light_dir: np.ndarray = None
) -> np.ndarray:
    """
    Simple Lambertian + ambient shading.
    Returns BGR uint8 pixel color.
    """
    if light_dir is None:
        light_dir = np.array([0.5, 1.0, 0.7])
        light_dir = light_dir / np.linalg.norm(light_dir)

    diffuse  = max(0.0, np.dot(normal, light_dir))
    ambient  = 0.25
    shade_f  = ambient + (1.0 - ambient) * diffuse

    bc = np.array(base_color, dtype=np.float32)
    return np.clip(bc * shade_f, 0, 255).astype(np.uint8)


def render_image(
    c2w: np.ndarray,
    K: np.ndarray,
    H: int,
    W: int,
    scene_type: str = "colored_boxes",
    noise_std: float = 0.0
) -> np.ndarray:
    """
    Software ray-caster: renders one image given camera pose.

    Args:
        c2w       : (4, 4) camera-to-world matrix
        K         : (3, 3) intrinsics
        H, W      : image dimensions
        scene_type: which scene to render
        noise_std : Gaussian noise std (0 = clean)

    Returns:
        (H, W, 3) uint8 BGR image
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Background sky gradient
    image = np.zeros((H, W, 3), dtype=np.uint8)

    # Light direction (world space)
    light_dir = np.array([0.6, 1.0, 0.5], dtype=np.float64)
    light_dir /= np.linalg.norm(light_dir)

    # Camera origin in world space
    cam_origin = c2w[:3, 3]

    # Load scene objects
    if scene_type == "colored_boxes":
        objects = make_scene_colored_boxes(H, W)
        obj_type = "box"
    elif scene_type == "spheres":
        objects = make_scene_spheres()
        obj_type = "sphere"
    else:
        objects = make_scene_colored_boxes(H, W)
        obj_type = "box"

    for py in range(H):
        # Sky gradient background
        sky_frac = py / H
        sky_top    = np.array([135, 200, 235], dtype=np.float32)  # light blue BGR
        sky_bottom = np.array([220, 240, 255], dtype=np.float32)  # near white
        sky_color  = (sky_top * (1 - sky_frac) + sky_bottom * sky_frac).astype(np.uint8)
        image[py, :] = sky_color

        for px in range(W):
            # Ray direction in camera space → world space
            ray_d_cam = np.array([
                (px - cx) / fx,
                (py - cy) / fy,
                1.0
            ], dtype=np.float64)
            ray_d_world = c2w[:3, :3] @ ray_d_cam
            ray_d_world /= np.linalg.norm(ray_d_world)

            # Test intersection with all objects
            best_t     = np.inf
            best_color = None
            best_normal = None

            for obj in objects:
                if obj_type == "box":
                    center, half_size, color = obj
                    if isinstance(half_size, float):
                        half = np.array([half_size, half_size, half_size])
                    else:
                        half = np.array(half_size)

                    t = ray_box_intersect(cam_origin, ray_d_world, center, half)
                    if 0 < t < best_t:
                        best_t      = t
                        hit_point   = cam_origin + t * ray_d_world
                        best_normal = compute_box_normal(hit_point, center)
                        best_color  = color

                elif obj_type == "sphere":
                    center, radius, color = obj
                    t = ray_sphere_intersect(cam_origin, ray_d_world, center, radius)
                    if 0 < t < best_t:
                        best_t      = t
                        hit_point   = cam_origin + t * ray_d_world
                        best_normal = (hit_point - center) / radius
                        best_color  = color

            if best_color is not None:
                pixel = shade(best_normal, best_color, light_dir)
                image[py, px] = pixel

    # Add noise
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return image


# ─────────────────────────────────────────────────────────────
# Camera path generation
# ─────────────────────────────────────────────────────────────

def generate_camera_orbit(
    n_views: int,
    radius: float = 3.5,
    elevation_deg: float = 25.0,
    n_elevation_levels: int = 2
) -> list:
    """
    Generate orbital camera poses looking at the origin.
    Uses multiple elevation levels for better 3D coverage.

    Returns:
        list of (R, t, c2w) — n_views camera poses
    """
    poses = []

    # Distribute views across elevation levels
    views_per_level = n_views // n_elevation_levels
    elevations = np.linspace(-elevation_deg / 2, elevation_deg, n_elevation_levels)

    for elev_deg in elevations:
        for i in range(views_per_level):
            azimuth = 2 * np.pi * i / views_per_level
            elev    = np.radians(elev_deg)

            # Camera position
            x = radius * np.cos(elev) * np.cos(azimuth)
            y = radius * np.sin(elev)
            z = radius * np.cos(elev) * np.sin(azimuth)
            cam_pos = np.array([x, y, z])

            # Look-at: point toward origin
            forward = -cam_pos / np.linalg.norm(cam_pos)

            # Up vector
            up_ref = np.array([0.0, 1.0, 0.0])
            right  = np.cross(forward, up_ref)
            if np.linalg.norm(right) < 1e-6:
                up_ref = np.array([0.0, 0.0, 1.0])
                right  = np.cross(forward, up_ref)
            right /= np.linalg.norm(right)
            up     = np.cross(right, forward)

            # c2w matrix (OpenCV convention: Y down, Z forward)
            # Columns: right, down, forward
            c2w = np.eye(4)
            c2w[:3, 0] =  right
            c2w[:3, 1] = -up        # Y points down in OpenCV
            c2w[:3, 2] =  forward
            c2w[:3, 3] =  cam_pos

            # world-to-camera R, t
            R = c2w[:3, :3].T
            t = -c2w[:3, :3].T @ cam_pos

            poses.append({
                "R": R,
                "t": t,
                "c2w": c2w,
                "cam_pos": cam_pos,
                "azimuth_deg": float(np.degrees(azimuth)),
                "elevation_deg": float(elev_deg),
            })

    # Fill any remaining views
    while len(poses) < n_views:
        poses.append(poses[len(poses) % len(poses)])

    return poses[:n_views]


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 60)
    print("Synthetic Dataset Generator")
    print("=" * 60)
    print(f"  Scene    : {args.scene}")
    print(f"  Views    : {args.n_views}")
    print(f"  Size     : {args.width}x{args.height}")
    print(f"  Raw out  : {args.output_raw}")
    print(f"  NeRF out : {args.output_nerf}")
    print()

    # Create output directories
    os.makedirs(args.output_raw,  exist_ok=True)
    os.makedirs(args.output_nerf, exist_ok=True)

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(args.output_nerf, split), exist_ok=True)

    # Camera intrinsics
    focal = max(args.width, args.height) * 1.1
    K = np.array([
        [focal,     0, args.width  / 2],
        [0,     focal, args.height / 2],
        [0,         0,               1]
    ], dtype=np.float64)

    print(f"  Focal length : {focal:.1f} px")
    print(f"  Intrinsics K :\n{K}\n")

    # Generate camera poses
    print(f"Generating {args.n_views} camera poses...")
    poses = generate_camera_orbit(
        n_views=args.n_views,
        radius=args.radius,
        elevation_deg=30.0,
        n_elevation_levels=2
    )

    # ── Render and save images ────────────────────────────────
    print(f"\nRendering {args.n_views} images (this may take 1-2 minutes)...")

    gt_poses   = {}
    transforms = {"camera_angle_x": 2 * np.arctan(args.width / (2 * focal)),
                  "frames": []}

    for i, pose_info in enumerate(poses):
        print(f"  Rendering view {i+1:3d}/{args.n_views}...", end="\r")

        image = render_image(
            c2w=pose_info["c2w"],
            K=K,
            H=args.height,
            W=args.width,
            scene_type=args.scene,
            noise_std=args.noise
        )

        # ── Save to data/raw/ ─────────────────────────────────
        raw_fname = f"image_{i:04d}.jpg"
        raw_path  = os.path.join(args.output_raw, raw_fname)
        cv2.imwrite(raw_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # ── Save ground truth poses ───────────────────────────
        gt_poses[i] = {
            "R": pose_info["R"].tolist(),
            "t": pose_info["t"].tolist(),
            "image": raw_fname,
            "azimuth_deg":   pose_info["azimuth_deg"],
            "elevation_deg": pose_info["elevation_deg"],
        }

        # ── Save to NeRF format ───────────────────────────────
        # Determine split
        if i < int(args.n_views * 0.7):
            split = "train"
        elif i < int(args.n_views * 0.85):
            split = "val"
        else:
            split = "test"

        nerf_fname = f"r_{i:04d}"
        nerf_img_path = os.path.join(args.output_nerf, split, nerf_fname + ".png")

        # Convert BGR → RGB for NeRF format, save as PNG with alpha
        img_rgb  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_rgba = np.dstack([img_rgb, np.ones((args.height, args.width), np.uint8) * 255])
        cv2.imwrite(nerf_img_path, cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGRA))

        # c2w in NeRF/OpenGL convention (flip Y and Z)
        c2w_nerf = pose_info["c2w"].copy()
        c2w_nerf[:3, 1] *= -1   # flip Y
        c2w_nerf[:3, 2] *= -1   # flip Z

        transforms["frames"].append({
            "file_path": f"./{split}/{nerf_fname}",
            "rotation": 0.012566,
            "transform_matrix": c2w_nerf.tolist()
        })

    print(f"\n  Rendered {args.n_views} images ✓")

    # ── Save ground truth poses JSON ──────────────────────────
    gt_data = {
        "K": K.tolist(),
        "image_width":  args.width,
        "image_height": args.height,
        "focal_length": focal,
        "scene": args.scene,
        "num_views": args.n_views,
        "poses": gt_poses
    }

    gt_path = os.path.join(args.output_raw, "gt_poses.json")
    with open(gt_path, "w") as f:
        json.dump(gt_data, f, indent=2)
    print(f"  Saved GT poses: {gt_path}")

    # ── Save NeRF transforms JSON (split by train/val/test) ───
    # Split frames
    all_frames = transforms["frames"]
    train_frames = [fr for fr in all_frames if "/train/" in fr["file_path"]]
    val_frames   = [fr for fr in all_frames if "/val/"   in fr["file_path"]]
    test_frames  = [fr for fr in all_frames if "/test/"  in fr["file_path"]]

    for split_name, frames in [("train", train_frames),
                                ("val",   val_frames),
                                ("test",  test_frames)]:
        split_transforms = {
            "camera_angle_x": transforms["camera_angle_x"],
            "frames": frames
        }
        out = os.path.join(args.output_nerf, f"transforms_{split_name}.json")
        with open(out, "w") as f:
            json.dump(split_transforms, f, indent=2)
        print(f"  Saved NeRF {split_name}: {len(frames)} frames → {out}")

    # ── Final summary ─────────────────────────────────────────
    print()
    print("=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)
    print()
    print("Files created:")
    print(f"  {args.output_raw}/")
    print(f"    image_0000.jpg ... image_{args.n_views-1:04d}.jpg")
    print(f"    gt_poses.json  (ground truth camera poses)")
    print()
    print(f"  {args.output_nerf}/")
    print(f"    transforms_train.json  ({len(train_frames)} frames)")
    print(f"    transforms_val.json    ({len(val_frames)} frames)")
    print(f"    transforms_test.json   ({len(test_frames)} frames)")
    print(f"    train/  val/  test/    (rendered PNG images)")
    print()
    print("Next steps:")
    print()
    print("  1. Run SfM:")
    print(f"     python scripts/run_sfm.py {args.output_raw}")
    print()
    print("  2. Run Dense:")
    print(f"     python scripts/run_dense.py --sfm_dir outputs/sparse --image_dir {args.output_raw}")
    print()
    print("  3. Train NeRF:")
    print(f"     python scripts/train_nerf.py {args.output_nerf}")
    print()
    print("  4. Render novel views:")
    print(f"     python scripts/render_views.py outputs/nerf_checkpoints/synthetic_scene/ckpt_0050000.pt {args.output_nerf}")


if __name__ == "__main__":
    main()