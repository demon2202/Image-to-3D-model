"""
Render novel views from a trained NeRF model.
"""

import sys
import os
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phase4_nerf.model import NeRFModel
from src.phase4_nerf.dataset import NeRFSyntheticDataset, get_rays
from src.phase4_nerf.evaluate import render_full_image


def generate_spiral_path(
    center: np.ndarray,
    radius: float,
    height_range: float,
    num_frames: int,
    focal: float,
    H: int, W: int
) -> list:
    """Generate a spiral camera path for video rendering."""
    poses = []
    
    for i in range(num_frames):
        theta = 2 * np.pi * i / num_frames
        
        # Camera position
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        z = center[2] + height_range * np.sin(2 * theta)
        
        cam_pos = np.array([x, y, z])
        
        # Look at center
        forward = center - cam_pos
        forward = forward / np.linalg.norm(forward)
        
        # Up direction
        up = np.array([0, 0, 1.0])
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        # Camera-to-world matrix
        c2w = np.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -forward  # OpenGL convention
        c2w[:3, 3] = cam_pos
        
        poses.append(c2w)
    
    return poses


def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/nerf_checkpoints/lego/checkpoint_050000.pt"
    scene_dir = sys.argv[2] if len(sys.argv) > 2 else "data/nerf_synthetic/lego"
    output_dir = "outputs/renders"
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    
    # Load model
    model_coarse = NeRFModel(
        pos_enc_dims=config.get('pos_enc_dims', 10),
        dir_enc_dims=config.get('dir_enc_dims', 4),
        hidden_dim=config.get('hidden_dim', 256)
    ).to(device)
    
    model_fine = NeRFModel(
        pos_enc_dims=config.get('pos_enc_dims', 10),
        dir_enc_dims=config.get('dir_enc_dims', 4),
        hidden_dim=config.get('hidden_dim', 256)
    ).to(device)
    
    model_coarse.load_state_dict(checkpoint['model_coarse'])
    model_fine.load_state_dict(checkpoint['model_fine'])
    model_coarse.eval()
    model_fine.eval()
    
    # Load dataset for intrinsics
    dataset = NeRFSyntheticDataset(scene_dir, split="test", img_scale=config.get('img_scale', 0.5))
    H, W = dataset.H, dataset.W
    K = dataset.K
    near = config.get('near', 2.0)
    far = config.get('far', 6.0)
    
    print(f"Rendering {len(dataset.poses)} test views at {W}x{H}")
    
    # Render test views
    psnrs = []
    ssims = []
    
    for i in range(min(10, len(dataset.poses))):
        print(f"  Rendering view {i+1}...")
        
        rays_o, rays_d = get_rays(H, W, K, dataset.poses[i])
        rays_o = torch.FloatTensor(rays_o).to(device)
        rays_d = torch.FloatTensor(rays_d).to(device)
        
        rgb_pred, depth_pred = render_full_image(
            model_coarse, model_fine,
            rays_o, rays_d, H, W,
            near, far,
            num_coarse=config.get('num_coarse_samples', 64),
            num_fine=config.get('num_fine_samples', 128),
            chunk_size=config.get('chunk_size', 32768)
        )
        
        # Save rendered image
        img = (np.clip(rgb_pred, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(output_dir, f"render_{i:03d}.png"))
        
        # Save depth
        depth_vis = (depth_pred - depth_pred.min()) / (depth_pred.max() - depth_pred.min() + 1e-8)
        depth_img = (depth_vis * 255).astype(np.uint8)
        Image.fromarray(depth_img).save(os.path.join(output_dir, f"depth_{i:03d}.png"))
        
        # Compute metrics against ground truth
        from src.phase4_nerf.evaluate import compute_psnr, compute_ssim
        gt = dataset.images[i]
        psnr = compute_psnr(rgb_pred, gt)
        ssim = compute_ssim(rgb_pred, gt)
        psnrs.append(psnr)
        ssims.append(ssim)
        print(f"    PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")
    
    print(f"\nAverage PSNR: {np.mean(psnrs):.2f} dB")
    print(f"Average SSIM: {np.mean(ssims):.4f}")
    print(f"Renders saved to {output_dir}/")


if __name__ == "__main__":
    main()