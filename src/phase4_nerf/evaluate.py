"""
Evaluation metrics: PSNR and SSIM, plus full image rendering.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from skimage.metrics import structural_similarity as ssim_fn

from .render import render_rays


def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.
    
    Args:
        pred: Predicted image (H, W, 3) in [0, 1]
        target: Ground truth image (H, W, 3) in [0, 1]
    
    Returns:
        PSNR in dB
    """
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return -10.0 * np.log10(mse)


def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute Structural Similarity Index.
    """
    return ssim_fn(pred, target, channel_axis=2, data_range=1.0)


@torch.no_grad()
def render_full_image(
    model_coarse: nn.Module,
    model_fine: Optional[nn.Module],
    rays_o: torch.Tensor,       # (H, W, 3)
    rays_d: torch.Tensor,       # (H, W, 3)
    H: int,
    W: int,
    near: float,
    far: float,
    num_coarse: int = 64,
    num_fine: int = 128,
    chunk_size: int = 32768
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render a full image by processing rays in chunks.
    
    Returns:
        (rgb_image, depth_image) as numpy arrays of shape (H, W, 3) and (H, W)
    """
    model_coarse.eval()
    if model_fine is not None:
        model_fine.eval()
    
    rays_o_flat = rays_o.reshape(-1, 3)
    rays_d_flat = rays_d.reshape(-1, 3)
    
    n_rays = rays_o_flat.shape[0]
    rgb_chunks = []
    depth_chunks = []
    
    for i in range(0, n_rays, chunk_size):
        chunk_o = rays_o_flat[i:i+chunk_size]
        chunk_d = rays_d_flat[i:i+chunk_size]
        
        result = render_rays(
            model=model_coarse,
            rays_o=chunk_o,
            rays_d=chunk_d,
            near=near,
            far=far,
            num_coarse=num_coarse,
            num_fine=num_fine,
            model_fine=model_fine,
            perturb=False,
            white_background=True
        )
        
        key = 'rgb_fine' if 'rgb_fine' in result else 'rgb_coarse'
        depth_key = 'depth_fine' if 'depth_fine' in result else 'depth_coarse'
        
        rgb_chunks.append(result[key].cpu())
        depth_chunks.append(result[depth_key].cpu())
    
    rgb = torch.cat(rgb_chunks, dim=0).reshape(H, W, 3).numpy()
    depth = torch.cat(depth_chunks, dim=0).reshape(H, W).numpy()
    
    return rgb, depth