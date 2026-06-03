"""
Tests for Phase 4: NeRF Model, Rendering, and Dataset.

Run with: pytest tests/test_nerf.py -v
"""

import os
import sys
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phase4_nerf.model import NeRFModel, NeRFSmall, PositionalEncoding
from src.phase4_nerf.render import (
    sample_along_rays,
    volume_render,
    render_rays
)
from src.phase4_nerf.utils import (
    positional_encoding_dim,
    encode_position,
    normalize_scene,
    generate_360_path,
    interpolate_poses_slerp,
    compute_near_far_from_poses
)
from src.phase4_nerf.dataset import get_rays
from src.utils.transforms import rotation_matrix_y


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def device():
    return torch.device("cpu")


@pytest.fixture(scope="module")
def small_nerf(device):
    return NeRFSmall(pos_enc_dims=4, dir_enc_dims=2, hidden_dim=32).to(device)


@pytest.fixture(scope="module")
def full_nerf(device):
    return NeRFModel(
        pos_enc_dims=6,
        dir_enc_dims=2,
        hidden_dim=64,
        num_layers=4,
        skip_layer=2
    ).to(device)


@pytest.fixture(scope="module")
def sample_rays(device):
    N = 64
    rays_o = torch.randn(N, 3)
    rays_d = torch.randn(N, 3)
    rays_d = rays_d / (rays_d.norm(dim=-1, keepdim=True) + 1e-8)
    return rays_o.to(device), rays_d.to(device)


@pytest.fixture(scope="module")
def sample_poses():
    """Simple circle of 8 camera poses."""
    poses = []
    for i in range(8):
        angle = i * 45.0
        R = rotation_matrix_y(angle)
        t = np.array([2 * np.cos(np.radians(angle)),
                      0.0,
                      2 * np.sin(np.radians(angle))]).reshape(3, 1)
        # c2w
        c2w = np.eye(4)
        c2w[:3, :3] = R.T
        c2w[:3, 3]  = -R.T @ t.ravel()
        poses.append(c2w)
    return np.stack(poses)


# ─────────────────────────────────────────────────────────────
# Positional Encoding Tests
# ─────────────────────────────────────────────────────────────

class TestPositionalEncoding:

    def test_output_dim_correct(self):
        """Output dimension matches expected formula."""
        for L in [4, 6, 10]:
            enc = PositionalEncoding(num_freqs=L, include_input=True)
            x = torch.randn(10, 3)
            out = enc(x)
            expected = 3 + 3 * L * 2    # input + sin/cos per freq per dim
            assert out.shape[-1] == expected, f"L={L}: {out.shape[-1]} != {expected}"

    def test_output_dim_no_input(self):
        """Output dimension without input concatenation."""
        L = 6
        enc = PositionalEncoding(num_freqs=L, include_input=False)
        x = torch.randn(10, 3)
        out = enc(x)
        expected = 3 * L * 2
        assert out.shape[-1] == expected

    def test_encode_position_functional(self):
        """Functional positional encoding matches module output."""
        x = torch.randn(5, 3)
        L = 6
        enc_module = PositionalEncoding(num_freqs=L)
        enc_fn     = encode_position(x, num_freqs=L)
        enc_mod    = enc_module(x)

        assert enc_fn.shape == enc_mod.shape
        assert torch.allclose(enc_fn, enc_mod, atol=1e-6)

    def test_encoding_not_all_zero(self):
        """Positional encoding of any non-zero input should not be all zeros."""
        enc = PositionalEncoding(num_freqs=6)
        x   = torch.ones(4, 3)
        out = enc(x)
        assert not torch.all(out == 0)

    def test_positional_encoding_dim_helper(self):
        """positional_encoding_dim utility function is correct."""
        for L in [4, 6, 10]:
            dim = positional_encoding_dim(L, include_input=True, input_dim=3)
            enc = PositionalEncoding(num_freqs=L, include_input=True)
            x   = torch.randn(1, 3)
            out = enc(x)
            assert dim == out.shape[-1]

    def test_encoding_continuity(self):
        """Nearby positions should have similar encodings."""
        enc = PositionalEncoding(num_freqs=6)
        x1  = torch.tensor([[0.0, 0.0, 0.0]])
        x2  = torch.tensor([[0.001, 0.001, 0.001]])
        out1 = enc(x1)
        out2 = enc(x2)
        diff = (out1 - out2).abs().max().item()
        assert diff < 0.5   # small input difference → small encoding difference


# ─────────────────────────────────────────────────────────────
# NeRF Model Tests
# ─────────────────────────────────────────────────────────────

class TestNeRFModel:

    def test_small_nerf_forward_shape(self, small_nerf, device):
        """NeRFSmall forward pass returns correct shapes."""
        N = 32
        pos  = torch.randn(N, 3).to(device)
        dirs = torch.randn(N, 3).to(device)
        dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)

        rgb, sigma = small_nerf(pos, dirs)

        assert rgb.shape   == (N, 3)
        assert sigma.shape == (N,)

    def test_full_nerf_forward_shape(self, full_nerf, device):
        """Full NeRFModel forward pass returns correct shapes."""
        N = 16
        pos  = torch.randn(N, 3).to(device)
        dirs = torch.randn(N, 3).to(device)
        dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)

        rgb, sigma = full_nerf(pos, dirs)

        assert rgb.shape   == (N, 3)
        assert sigma.shape == (N,)

    def test_rgb_in_range(self, small_nerf, device):
        """RGB output should be in [0, 1] (sigmoid activation)."""
        pos  = torch.randn(100, 3).to(device)
        dirs = torch.randn(100, 3).to(device)
        dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)

        rgb, _ = small_nerf(pos, dirs)

        assert rgb.min() >= 0.0 - 1e-6
        assert rgb.max() <= 1.0 + 1e-6

    def test_sigma_non_negative(self, small_nerf, device):
        """Density sigma should be >= 0 (ReLU activation)."""
        pos  = torch.randn(100, 3).to(device)
        dirs = torch.randn(100, 3).to(device)

        _, sigma = small_nerf(pos, dirs)

        assert (sigma >= 0).all()

    def test_view_dependence(self, small_nerf, device):
        """Same position, different directions should give different RGB."""
        pos  = torch.zeros(1, 3).to(device)
        dir1 = torch.tensor([[1.0, 0.0, 0.0]]).to(device)
        dir2 = torch.tensor([[0.0, 1.0, 0.0]]).to(device)

        rgb1, sigma1 = small_nerf(pos, dir1)
        rgb2, sigma2 = small_nerf(pos, dir2)

        # RGB should differ (view-dependent)
        assert not torch.allclose(rgb1, rgb2)
        # Sigma should be the same (view-independent density)
        assert torch.allclose(sigma1, sigma2, atol=1e-5)

    def test_model_gradients_flow(self, small_nerf, device):
        """Gradients should flow through the entire model."""
        pos  = torch.randn(10, 3, requires_grad=True).to(device)
        dirs = torch.randn(10, 3).to(device)

        rgb, sigma = small_nerf(pos, dirs)
        loss = rgb.mean() + sigma.mean()
        loss.backward()

        assert pos.grad is not None
        assert pos.grad.abs().sum() > 0

    def test_batch_size_flexibility(self, small_nerf, device):
        """Model handles different batch sizes."""
        for N in [1, 8, 64, 256, 1024]:
            pos  = torch.randn(N, 3).to(device)
            dirs = torch.randn(N, 3).to(device)
            rgb, sigma = small_nerf(pos, dirs)
            assert rgb.shape == (N, 3)
            assert sigma.shape == (N,)


# ─────────────────────────────────────────────────────────────
# Ray Sampling Tests
# ─────────────────────────────────────────────────────────────

class TestRaySampling:

    def test_sample_along_rays_shape(self, sample_rays, device):
        """sample_along_rays returns correct shapes."""
        rays_o, rays_d = sample_rays
        N = len(rays_o)
        S = 32

        pts, z_vals = sample_along_rays(rays_o, rays_d, near=2.0, far=6.0, num_samples=S)

        assert pts.shape    == (N, S, 3)
        assert z_vals.shape == (N, S)

    def test_z_vals_in_near_far_range(self, sample_rays, device):
        """Sample depths should be within [near, far]."""
        rays_o, rays_d = sample_rays
        near, far = 1.0, 10.0

        _, z_vals = sample_along_rays(rays_o, rays_d, near=near, far=far, num_samples=64)

        assert z_vals.min() >= near - 0.1
        assert z_vals.max() <= far  + 0.1

    def test_z_vals_monotonic(self, sample_rays, device):
        """Sample depths should be monotonically increasing."""
        rays_o, rays_d = sample_rays

        _, z_vals = sample_along_rays(
            rays_o, rays_d, near=2.0, far=6.0,
            num_samples=64, perturb=False
        )

        diffs = z_vals[:, 1:] - z_vals[:, :-1]
        assert (diffs >= 0).all()

    def test_points_on_ray(self, sample_rays, device):
        """Sampled points should lie on the corresponding rays."""
        rays_o, rays_d = sample_rays
        N, S = 8, 16

        r_o = rays_o[:N]
        r_d = rays_d[:N]

        pts, z_vals = sample_along_rays(r_o, r_d, near=2.0, far=6.0, num_samples=S, perturb=False)

        # pts[i, j] should equal rays_o[i] + z_vals[i, j] * rays_d[i]
        expected = r_o[:, None, :] + r_d[:, None, :] * z_vals[:, :, None]
        assert torch.allclose(pts, expected, atol=1e-5)


# ─────────────────────────────────────────────────────────────
# Volume Rendering Tests
# ─────────────────────────────────────────────────────────────

class TestVolumeRendering:

    def test_volume_render_output_shapes(self, device):
        """volume_render returns correctly shaped tensors."""
        N, S = 16, 32
        rgb    = torch.rand(N, S, 3).to(device)
        sigma  = torch.rand(N, S).to(device)
        z_vals = torch.linspace(2, 6, S).expand(N, S).to(device)
        rays_d = torch.randn(N, 3).to(device)

        rgb_map, depth_map, acc_map, weights = volume_render(
            rgb, sigma, z_vals, rays_d, white_background=False
        )

        assert rgb_map.shape   == (N, 3)
        assert depth_map.shape == (N,)
        assert acc_map.shape   == (N,)
        assert weights.shape   == (N, S)

    def test_rgb_output_range(self, device):
        """Rendered RGB should be in [0, 1] range."""
        N, S = 8, 16
        rgb    = torch.rand(N, S, 3).to(device)
        sigma  = torch.rand(N, S).to(device) * 5.0   # moderate density
        z_vals = torch.linspace(2, 6, S).expand(N, S).to(device)
        rays_d = torch.zeros(N, 3).to(device)
        rays_d[:, 2] = 1.0

        rgb_map, _, _, _ = volume_render(rgb, sigma, z_vals, rays_d, white_background=True)

        assert rgb_map.min() >= -1e-5
        assert rgb_map.max() <=  1.0 + 1e-5

    def test_weights_sum_to_at_most_one(self, device):
        """Rendering weights should sum to at most 1 per ray."""
        N, S = 8, 32
        rgb    = torch.rand(N, S, 3).to(device)
        sigma  = torch.rand(N, S).to(device)
        z_vals = torch.linspace(1, 10, S).expand(N, S).to(device)
        rays_d = torch.zeros(N, 3).to(device)
        rays_d[:, 2] = 1.0

        _, _, _, weights = volume_render(rgb, sigma, z_vals, rays_d)

        weight_sums = weights.sum(dim=-1)
        assert (weight_sums <= 1.0 + 1e-4).all()
        assert (weight_sums >= 0.0 - 1e-4).all()

    def test_empty_scene_white_background(self, device):
        """With zero density, rendered color should be white (white background)."""
        N, S = 4, 16
        rgb    = torch.rand(N, S, 3).to(device)
        sigma  = torch.zeros(N, S).to(device)   # empty scene
        z_vals = torch.linspace(2, 6, S).expand(N, S).to(device)
        rays_d = torch.zeros(N, 3).to(device)
        rays_d[:, 2] = 1.0

        rgb_map, _, acc_map, _ = volume_render(
            rgb, sigma, z_vals, rays_d, white_background=True
        )

        # All white = 1.0
        assert torch.allclose(rgb_map, torch.ones_like(rgb_map), atol=1e-5)
        # Accumulated opacity = 0
        assert torch.allclose(acc_map, torch.zeros_like(acc_map), atol=1e-5)

    def test_full_render_rays_pipeline(self, small_nerf, sample_rays, device):
        """render_rays should complete without error."""
        rays_o, rays_d = sample_rays
        rays_o = rays_o[:8]
        rays_d = rays_d[:8]

        result = render_rays(
            model=small_nerf,
            rays_o=rays_o,
            rays_d=rays_d,
            near=2.0,
            far=6.0,
            num_coarse=16,
            num_fine=16,
            model_fine=small_nerf,
            perturb=False,
            white_background=True
        )

        assert "rgb_coarse" in result
        assert "rgb_fine"   in result
        assert result["rgb_coarse"].shape == (8, 3)
        assert result["rgb_fine"].shape   == (8, 3)


# ─────────────────────────────────────────────────────────────
# NeRF Utility Tests
# ─────────────────────────────────────────────────────────────

class TestNeRFUtils:

    def test_normalize_scene_centers_at_origin(self, sample_poses):
        """After normalization, camera centers should be near unit sphere."""
        norm_poses, transform = normalize_scene(sample_poses)
        centers = norm_poses[:, :3, 3]
        dists   = np.linalg.norm(centers, axis=1)
        # 90th percentile should be ~1.0 after normalization
        assert abs(np.percentile(dists, 90) - 1.0) < 0.2

    def test_normalize_scene_returns_transform(self, sample_poses):
        """normalize_scene returns transform dict with centroid and scale."""
        _, transform = normalize_scene(sample_poses)
        assert "centroid" in transform
        assert "scale"    in transform
        assert transform["scale"] > 0

    def test_generate_360_path_shape(self):
        """generate_360_path returns (n_frames, 4, 4) array."""
        n = 60
        path = generate_360_path(radius=4.0, n_frames=n)
        assert path.shape == (n, 4, 4)

    def test_generate_360_path_valid_rotations(self):
        """Each pose in path should have valid rotation matrix."""
        path = generate_360_path(n_frames=24)
        for i in range(len(path)):
            R = path[i, :3, :3]
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
            assert abs(np.linalg.det(R) - 1.0) < 1e-6

    def test_interpolate_poses_slerp_shape(self, sample_poses):
        """SLERP interpolation returns correct number of frames."""
        n = 60
        path = interpolate_poses_slerp(sample_poses, n_frames=n)
        assert path.shape == (n, 4, 4)

    def test_interpolate_poses_endpoints_preserved(self, sample_poses):
        """SLERP path should start and end close to the first/last input pose."""
        n = 60
        path = interpolate_poses_slerp(sample_poses, n_frames=n)

        # Translation at start and end
        t_start = path[0,  :3, 3]
        t_end   = path[-1, :3, 3]
        t_first = sample_poses[0,  :3, 3]
        t_last  = sample_poses[-1, :3, 3]

        assert np.linalg.norm(t_start - t_first) < 0.1
        assert np.linalg.norm(t_end   - t_last)  < 0.1

    def test_get_rays_shape(self):
        """get_rays returns correct shapes for each image."""
        H, W = 32, 48
        K = np.array([
            [50.0, 0,   W / 2],
            [0,   50.0, H / 2],
            [0,    0,   1    ]
        ])
        c2w = np.eye(4)
        rays_o, rays_d = get_rays(H, W, K, c2w)

        assert rays_o.shape == (H, W, 3)
        assert rays_d.shape == (H, W, 3)

    def test_get_rays_directions_normalized(self):
        """Ray directions should be unit vectors."""
        H, W = 16, 24
        K = np.array([[30, 0, 12], [0, 30, 8], [0, 0, 1]], dtype=np.float64)
        c2w = np.eye(4)

        _, rays_d = get_rays(H, W, K, c2w)
        norms = np.linalg.norm(rays_d.reshape(-1, 3), axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_compute_near_far_from_poses(self, sample_poses):
        """near/far should be positive and far > near."""
        near, far = compute_near_far_from_poses(sample_poses)
        assert near > 0
        assert far > near
        assert far < 1000   # sanity check