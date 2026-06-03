"""
Run the complete SfM pipeline.
"""

import sys
import os
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phase2_sfm.incremental_sfm import IncrementalSfM
from src.phase2_sfm.visualize import visualize_reconstruction


def main():
    # Load config
    config_path = "config/default.yaml"
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Camera intrinsics — adjust for your dataset
    # For NeRF Synthetic (800x800 images): focal ≈ 1111
    # For a typical phone camera at 1024px width: focal ≈ 800
    image_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    
    # Try to detect image size for intrinsics
    import cv2
    sample_images = [f for f in os.listdir(image_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not sample_images:
        print(f"No images found in {image_dir}")
        return
    
    sample = cv2.imread(os.path.join(image_dir, sample_images[0]))
    h, w = sample.shape[:2]
    
    # Approximate intrinsics (adjust if you have calibration data)
    focal = max(h, w) * 1.2  # Rough estimate
    K = np.array([
        [focal, 0, w / 2],
        [0, focal, h / 2],
        [0, 0, 1]
    ], dtype=np.float64)
    
    print(f"Image size: {w}x{h}")
    print(f"Estimated focal length: {focal:.1f}")
    print(f"Intrinsics:\n{K}")
    
    # Run SfM
    sfm = IncrementalSfM(K, config={
    'detector':         'sift',
    'max_keypoints':    12000,     # more features = better matching
    'matcher':          'flann',
    'ratio_threshold':  0.80,      # slightly looser for outdoor scenes
    'ransac_threshold': 2.0,       # looser RANSAC for noisy data
    'min_matches':      20,        # accept pairs with fewer matches
})
    sfm.load_images(image_dir, max_dim=1024)
    sfm.reconstruct(bundle_adjust_interval=3)
    
    # Results
    mean_error = sfm.compute_mean_reprojection_error()
    print(f"\nMean reprojection error: {mean_error:.2f} pixels")
    
    # Save outputs
    os.makedirs("outputs/sparse", exist_ok=True)
    sfm.save_point_cloud("outputs/sparse/point_cloud.ply")
    
    # Save camera poses for NeRF
    poses_data = {}
    for idx, (R, t) in sfm.registered_cameras.items():
        poses_data[idx] = {
            'R': R.tolist(),
            't': t.tolist(),
            'image': sfm.image_names[idx]
        }
    
    import json
    with open("outputs/sparse/camera_poses.json", 'w') as f:
        json.dump({
            'K': K.tolist(),
            'poses': poses_data
        }, f, indent=2)
    
    print("Saved camera poses to outputs/sparse/camera_poses.json")
    
    # Visualize
    points, colors = sfm.get_point_cloud()
    if len(points) > 0:
        visualize_reconstruction(
            points, colors,
            sfm.registered_cameras,
            K,
            window_name="SfM Result"
        )


if __name__ == "__main__":
    main()