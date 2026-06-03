"""
Volume rendering for NeRF.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Callable


def sample_along_rays(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    num_samples: int,
    perturb: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample points along rays.
    
    Args:
        rays_o: (N, 3) ray origins
        rays_d: (N, 3) ray directions
        near, far: near and far bounds
        num_samples: number of samples per ray
        perturb: whether to add random perturbation to sample locations
        
    Returns:
        pts: (N, num_samples, 3) 3D sample points
        z_vals: (N, num_samples) depth values
    """
    N = rays_o.shape[0]
    
    # Uniform spacing
    t_vals = torch.linspace(0.0, 1.0, num_samples, device=rays_o.device)
    z_vals = near * (1.0 - t_vals) + far * t_vals  # (num_samples,)
    z_vals = z_vals.expand(N, num_samples).clone()
    
    # Stratified sampling
    if perturb:
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., :1], mids], dim=-1)
        t_rand = torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * t_rand
    
    # Compute 3D points
    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # (N, S, 3)
    
    return pts, z_vals


def sample_pdf(
    bins: torch.Tensor,
    weights: torch.Tensor,
    num_samples: int,
    det: bool = False
) -> torch.Tensor:
    """
    Hierarchical sampling based on PDF from coarse network weights.

    Args:
        bins       : (N, S-1) bin midpoints between coarse z_vals
        weights    : (N, S-2) weights from coarse volume rendering (inner weights)
        num_samples: number of fine samples to draw
        det        : if True use deterministic (uniform) sampling; else random

    Returns:
        samples: (N, num_samples) new depth values drawn from the weight PDF
    """
    N = weights.shape[0]

    # ── 1. Build normalised PDF / CDF ────────────────────────────────────────
    # Guard against zero weights
    weights = weights + 1e-5                                      # (N, S-2)
    pdf     = weights / weights.sum(dim=-1, keepdim=True)         # (N, S-2)
    cdf     = torch.cumsum(pdf, dim=-1)                           # (N, S-2)
    # Prepend 0 so CDF starts at 0
    cdf     = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # (N, S-1)

    # ── 2. Draw uniform samples u ∈ [0, 1) ───────────────────────────────────
    if det:
        u = torch.linspace(0.0, 1.0, num_samples, device=bins.device, dtype=bins.dtype)
        u = u.unsqueeze(0).expand(N, num_samples)                 # (N, num_samples)
    else:
        u = torch.rand(N, num_samples, device=bins.device, dtype=bins.dtype)

    u = u.contiguous()

    # ── 3. Invert CDF via binary search ──────────────────────────────────────
    # cdf shape: (N, S-1)   bins shape: (N, S-1)
    # For each u find the bin index where cdf first exceeds u
    inds = torch.searchsorted(cdf.detach(), u, right=True)       # (N, num_samples)

    # Clamp so gather indices are valid
    below = (inds - 1).clamp(min=0)                              # (N, num_samples)
    above = inds.clamp(max=cdf.shape[-1] - 1)                    # (N, num_samples)

    # ── 4. Gather CDF and bin values at below / above ─────────────────────────
    cdf_below = torch.gather(cdf,  dim=-1, index=below)          # (N, num_samples)
    cdf_above = torch.gather(cdf,  dim=-1, index=above)          # (N, num_samples)
    bin_below = torch.gather(bins, dim=-1, index=below)          # (N, num_samples)
    bin_above = torch.gather(bins, dim=-1, index=above)          # (N, num_samples)

    # ── 5. Linear interpolation inside each bin ───────────────────────────────
    denom = cdf_above - cdf_below                                # (N, num_samples)
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)

    t       = (u - cdf_below) / denom                           # (N, num_samples)
    samples = bin_below + t * (bin_above - bin_below)            # (N, num_samples)

    return samples


def volume_render(
    rgb: torch.Tensor,
    sigma: torch.Tensor,
    z_vals: torch.Tensor,
    rays_d: torch.Tensor,
    white_background: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Classic NeRF volume rendering equation.
    
    C(r) = sum_i T_i * (1 - exp(-sigma_i * delta_i)) * c_i
    where T_i = exp(-sum_{j<i} sigma_j * delta_j)
    
    Args:
        rgb: (N, S, 3) color at each sample
        sigma: (N, S) density at each sample
        z_vals: (N, S) depth values
        rays_d: (N, 3) ray directions
        
    Returns:
        rgb_map: (N, 3) rendered pixel colors
        depth_map: (N,) expected depth
        acc_map: (N,) accumulated opacity
        weights: (N, S) sample weights
    """
    # Compute distances between samples
    dists = z_vals[..., 1:] - z_vals[..., :-1]  # (N, S-1)
    # Last distance is infinity (or a large number)
    dists = torch.cat([dists, torch.tensor([1e10], device=dists.device).expand(dists[..., :1].shape)], dim=-1)
    
    # Multiply by ray direction magnitude
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    
    # Alpha = 1 - exp(-sigma * delta)
    alpha = 1.0 - torch.exp(-sigma * dists)  # (N, S)
    
    # Transmittance T_i = prod_{j<i} (1 - alpha_j)
    # Use cumulative product with exclusive operation
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1
    )[..., :-1]  # (N, S)
    
    # Weights w_i = T_i * alpha_i
    weights = transmittance * alpha  # (N, S)
    
    # Rendered color
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # (N, 3)
    
    # Depth map
    depth_map = torch.sum(weights * z_vals, dim=-1)  # (N,)
    
    # Accumulated opacity
    acc_map = torch.sum(weights, dim=-1)  # (N,)
    
    # White background
    if white_background:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])
    
    return rgb_map, depth_map, acc_map, weights


def run_model_chunked(
    model: nn.Module,
    pts_flat: torch.Tensor,
    dirs_flat: torch.Tensor,
    chunk: int = 32768
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run the NeRF MLP in chunks to avoid OOM on large point batches.

    With batch_size=4096, num_coarse=64, num_fine=128 the fine pass
    produces 4096*(64+128) = 786 432 points — far too large for a 4 GB
    GPU in one shot.  Processing them in slices of `chunk` keeps peak
    activation memory proportional to chunk, not to the full tensor.

    Args:
        model    : NeRF MLP
        pts_flat : (M, 3) all 3-D sample positions (M = N_rays * N_samples)
        dirs_flat: (M, 3) corresponding view directions
        chunk    : max points per forward pass (tune down if still OOM)

    Returns:
        rgb   : (M, 3)
        sigma : (M,)
    """
    rgb_chunks, sigma_chunks = [], []
    for i in range(0, pts_flat.shape[0], chunk):
        rgb_c, sigma_c = model(pts_flat[i:i + chunk], dirs_flat[i:i + chunk])
        rgb_chunks.append(rgb_c)
        sigma_chunks.append(sigma_c)
    return torch.cat(rgb_chunks, dim=0), torch.cat(sigma_chunks, dim=0)


def render_rays(
    model: nn.Module,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    num_coarse: int = 64,
    num_fine: int = 128,
    model_fine: Optional[nn.Module] = None,
    perturb: bool = True,
    white_background: bool = True,
    chunk: int = 32768
) -> dict:
    """
    Full rendering pipeline for a batch of rays.

    Args:
        model: Coarse NeRF model
        rays_o: (N, 3) ray origins
        rays_d: (N, 3) ray directions
        near, far: scene bounds
        num_coarse: coarse samples per ray
        num_fine: fine samples per ray (hierarchical)
        model_fine: Fine NeRF model (if None, use coarse model)
        chunk: MLP chunk size — reduce if still OOM (default 32768)

    Returns:
        Dictionary with 'rgb_coarse', 'rgb_fine', 'depth_coarse', 'depth_fine', etc.
    """
    result = {}

    # ----- Coarse Sampling -----
    pts_coarse, z_vals_coarse = sample_along_rays(
        rays_o, rays_d, near, far, num_coarse, perturb
    )

    # Expand ray directions for each sample
    dirs_coarse = rays_d[:, None, :].expand_as(pts_coarse)

    # Query coarse model — chunked to avoid OOM
    pts_flat  = pts_coarse.reshape(-1, 3)
    dirs_flat = dirs_coarse.reshape(-1, 3)

    rgb_coarse_flat, sigma_coarse_flat = run_model_chunked(model, pts_flat, dirs_flat, chunk)

    rgb_coarse   = rgb_coarse_flat.reshape(pts_coarse.shape)        # (N, S, 3)
    sigma_coarse = sigma_coarse_flat.reshape(pts_coarse.shape[:-1]) # (N, S)

    # Free intermediates before fine pass
    del pts_flat, dirs_flat, rgb_coarse_flat, sigma_coarse_flat

    # Volume render coarse
    rgb_map_c, depth_map_c, acc_map_c, weights_c = volume_render(
        rgb_coarse, sigma_coarse, z_vals_coarse, rays_d, white_background
    )

    result['rgb_coarse']   = rgb_map_c
    result['depth_coarse'] = depth_map_c
    result['acc_coarse']   = acc_map_c

    # ----- Fine (Hierarchical) Sampling -----
    if num_fine > 0:
        # Sample from PDF defined by coarse weights
        z_vals_mid = 0.5 * (z_vals_coarse[..., 1:] + z_vals_coarse[..., :-1])
        z_samples  = sample_pdf(
            z_vals_mid,
            weights_c[..., 1:-1],
            num_fine,
            det=not perturb
        )
        z_samples = z_samples.detach()

        # Free coarse activations — weights_c is no longer needed
        del rgb_coarse, sigma_coarse, weights_c

        # Combine coarse and fine z values
        z_vals_fine, _ = torch.sort(
            torch.cat([z_vals_coarse, z_samples], dim=-1), dim=-1
        )

        pts_fine  = rays_o[:, None, :] + rays_d[:, None, :] * z_vals_fine[..., :, None]
        dirs_fine = rays_d[:, None, :].expand_as(pts_fine)

        fine_model = model_fine if model_fine is not None else model

        # Fine model — chunked
        rgb_f_flat, sigma_f_flat = run_model_chunked(
            fine_model,
            pts_fine.reshape(-1, 3),
            dirs_fine.reshape(-1, 3),
            chunk
        )

        rgb_f   = rgb_f_flat.reshape(pts_fine.shape)
        sigma_f = sigma_f_flat.reshape(pts_fine.shape[:-1])

        del pts_fine, dirs_fine, rgb_f_flat, sigma_f_flat

        rgb_map_f, depth_map_f, acc_map_f, _ = volume_render(
            rgb_f, sigma_f, z_vals_fine, rays_d, white_background
        )

        result['rgb_fine']   = rgb_map_f
        result['depth_fine'] = depth_map_f
        result['acc_fine']   = acc_map_f

    return result