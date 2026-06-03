"""
NeRF dataset — ray generation from posed images.
Supports both NeRF Synthetic (Blender) format and custom SfM output.
"""

import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Tuple, Optional, Dict
import cv2


def get_rays(
    H: int, W: int, K: np.ndarray, c2w: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate rays for every pixel in an image.
    
    Args:
        H, W: Image height and width
        K: 3x3 intrinsic matrix
        c2w: 4x4 camera-to-world matrix
        
    Returns:
        (rays_o, rays_d) each of shape (H, W, 3)
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float64),
        np.arange(H, dtype=np.float64),
        indexing='xy'
    )
    
    # Direction in camera coordinates
    dirs = np.stack([
        (i - cx) / fx,
        (j - cy) / fy,  # Note: some conventions negate y
        np.ones_like(i)
    ], axis=-1)  # (H, W, 3)
    
    # Rotate to world coordinates
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], axis=-1)  # (H, W, 3)
    
    # Normalize direction
    rays_d = rays_d / (np.linalg.norm(rays_d, axis=-1, keepdims=True) + 1e-8)
    
    # Origin is the camera position
    rays_o = np.broadcast_to(c2w[:3, 3], rays_d.shape).copy()
    
    return rays_o, rays_d


class NeRFSyntheticDataset(Dataset):
    """
    Dataset for NeRF Synthetic (Blender) format.
    
    Expected directory structure:
        scene_dir/
        ├── transforms_train.json
        ├── transforms_val.json
        ├── transforms_test.json
        ├── train/
        │   ├── r_0.png
        │   ├── r_1.png
        │   └── ...
        ├── val/
        └── test/
    """
    
    def __init__(
        self,
        scene_dir: str,
        split: str = "train",
        img_scale: float = 1.0,
        white_background: bool = True
    ):
        super().__init__()
        
        self.scene_dir = scene_dir
        self.split = split
        self.img_scale = img_scale
        self.white_background = white_background
        
        # Load transforms
        with open(os.path.join(scene_dir, f"transforms_{split}.json"), 'r') as f:
            meta = json.load(f)
        
        # Parse camera info
        self.camera_angle_x = meta["camera_angle_x"]
        
        # Load images and poses
        self.images = []
        self.poses = []
        self.image_paths = []
        
        for frame in meta["frames"]:
            filepath = os.path.join(scene_dir, frame["file_path"])
            if not filepath.endswith(".png"):
                filepath += ".png"
            
            # Handle relative paths
            if not os.path.exists(filepath):
                filepath = os.path.join(scene_dir, os.path.basename(frame["file_path"]))
                if not filepath.endswith(".png"):
                    filepath += ".png"
            
            img = Image.open(filepath)
            
            # Resize
            if self.img_scale != 1.0:
                new_w = int(img.width * self.img_scale)
                new_h = int(img.height * self.img_scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)
            
            img = np.array(img, dtype=np.float32) / 255.0
            
            # Handle alpha channel
            if img.shape[-1] == 4:
                if white_background:
                    img = img[..., :3] * img[..., 3:4] + (1 - img[..., 3:4])
                else:
                    img = img[..., :3] * img[..., 3:4]
            
            self.images.append(img)
            self.poses.append(np.array(frame["transform_matrix"], dtype=np.float64))
            self.image_paths.append(filepath)
        
        self.images = np.stack(self.images)  # (N, H, W, 3)
        self.poses = np.stack(self.poses)    # (N, 4, 4)
        
        # Compute intrinsics
        self.H, self.W = self.images.shape[1:3]
        self.focal = 0.5 * self.W / np.tan(0.5 * self.camera_angle_x)
        
        self.K = np.array([
            [self.focal, 0, self.W / 2],
            [0, self.focal, self.H / 2],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Precompute all rays
        self._precompute_rays()
        
        print(f"Loaded {len(self.images)} {split} images ({self.H}x{self.W}), focal={self.focal:.1f}")
    
    def _precompute_rays(self):
        """Precompute rays for all images."""
        all_rays_o = []
        all_rays_d = []
        all_rgb = []
        
        for i in range(len(self.images)):
            rays_o, rays_d = get_rays(self.H, self.W, self.K, self.poses[i])
            all_rays_o.append(rays_o.reshape(-1, 3))
            all_rays_d.append(rays_d.reshape(-1, 3))
            all_rgb.append(self.images[i].reshape(-1, 3))
        
        self.all_rays_o = torch.FloatTensor(np.concatenate(all_rays_o, axis=0))
        self.all_rays_d = torch.FloatTensor(np.concatenate(all_rays_d, axis=0))
        self.all_rgb = torch.FloatTensor(np.concatenate(all_rgb, axis=0))
    
    def __len__(self) -> int:
        return len(self.all_rays_o)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return {
            'rays_o': self.all_rays_o[idx],
            'rays_d': self.all_rays_d[idx],
            'rgb': self.all_rgb[idx]
        }
    
    def get_image_rays(self, img_idx: int) -> Dict[str, torch.Tensor]:
        """Get all rays for a single image."""
        rays_o, rays_d = get_rays(self.H, self.W, self.K, self.poses[img_idx])
        return {
            'rays_o': torch.FloatTensor(rays_o),       # (H, W, 3)
            'rays_d': torch.FloatTensor(rays_d),       # (H, W, 3)
            'rgb': torch.FloatTensor(self.images[img_idx])  # (H, W, 3)
        }


class SfMDataset(Dataset):
    """
    Dataset that uses camera poses from our SfM pipeline.
    Converts from SfM (R, t world-to-camera) to NeRF convention (c2w).
    """
    
    def __init__(
        self,
        image_dir: str,
        camera_poses: Dict[int, Tuple[np.ndarray, np.ndarray]],
        K: np.ndarray,
        image_names: list,
        img_scale: float = 1.0
    ):
        super().__init__()
        
        self.K = K.copy()
        if img_scale != 1.0:
            self.K[0] *= img_scale
            self.K[1] *= img_scale
        
        self.images = []
        self.poses = []
        
        sorted_indices = sorted(camera_poses.keys())
        
        for idx in sorted_indices:
            R, t = camera_poses[idx]
            
            # Convert world-to-camera (R, t) to camera-to-world 4x4
            c2w = np.eye(4)
            c2w[:3, :3] = R.T
            c2w[:3, 3] = (-R.T @ t).ravel()
            
            # Load image
            img_path = os.path.join(image_dir, image_names[idx])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if img_scale != 1.0:
                new_w = int(img.shape[1] * img_scale)
                new_h = int(img.shape[0] * img_scale)
                img = cv2.resize(img, (new_w, new_h))
            
            self.images.append(img.astype(np.float32) / 255.0)
            self.poses.append(c2w)
        
        self.images = np.stack(self.images)
        self.poses = np.stack(self.poses)
        self.H, self.W = self.images.shape[1:3]
        
        self._precompute_rays()
    
    def _precompute_rays(self):
        all_rays_o, all_rays_d, all_rgb = [], [], []
        for i in range(len(self.images)):
            rays_o, rays_d = get_rays(self.H, self.W, self.K, self.poses[i])
            all_rays_o.append(rays_o.reshape(-1, 3))
            all_rays_d.append(rays_d.reshape(-1, 3))
            all_rgb.append(self.images[i].reshape(-1, 3))
        
        self.all_rays_o = torch.FloatTensor(np.concatenate(all_rays_o))
        self.all_rays_d = torch.FloatTensor(np.concatenate(all_rays_d))
        self.all_rgb = torch.FloatTensor(np.concatenate(all_rgb))
    
    def __len__(self):
        return len(self.all_rays_o)
    
    def __getitem__(self, idx):
        return {
            'rays_o': self.all_rays_o[idx],
            'rays_d': self.all_rays_d[idx],
            'rgb': self.all_rgb[idx]
        }