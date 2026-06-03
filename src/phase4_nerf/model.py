"""
NeRF model architecture — Vanilla NeRF with positional encoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class PositionalEncoding(nn.Module):
    """
    Positional encoding: γ(p) = [sin(2^0 π p), cos(2^0 π p), ..., sin(2^{L-1} π p), cos(2^{L-1} π p)]
    """
    
    def __init__(self, num_freqs: int, include_input: bool = True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        
        # Precompute frequency bands
        freq_bands = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
        self.register_buffer('freq_bands', freq_bands)
    
    @property
    def output_dim(self) -> int:
        d = self.num_freqs * 2  # sin + cos for each frequency
        if self.include_input:
            d += 1  # per input dimension, but we handle this in forward
        return d
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., D) input tensor
            
        Returns:
            (..., D * output_dim) encoded tensor
        """
        out = []
        if self.include_input:
            out.append(x)
        
        for freq in self.freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        
        return torch.cat(out, dim=-1)


class NeRFModel(nn.Module):
    """
    Vanilla NeRF MLP.
    
    Architecture:
    - 8-layer MLP for density + feature
    - Skip connection at layer 4
    - Separate 1-layer head for RGB (conditioned on view direction)
    """
    
    def __init__(
        self,
        pos_enc_dims: int = 10,
        dir_enc_dims: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 8,
        skip_layer: int = 4
    ):
        super().__init__()
        
        self.pos_encoding = PositionalEncoding(pos_enc_dims)
        self.dir_encoding = PositionalEncoding(dir_enc_dims)
        
        # Input dimensions
        self.pos_input_dim = 3 + 3 * pos_enc_dims * 2  # xyz + encoded
        self.dir_input_dim = 3 + 3 * dir_enc_dims * 2  # dir + encoded
        
        self.skip_layer = skip_layer
        
        # Build density network
        self.density_layers = nn.ModuleList()
        
        # First layer
        self.density_layers.append(nn.Linear(self.pos_input_dim, hidden_dim))
        
        for i in range(1, num_layers):
            if i == skip_layer:
                # Skip connection: concatenate input
                self.density_layers.append(nn.Linear(hidden_dim + self.pos_input_dim, hidden_dim))
            else:
                self.density_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Density output (sigma)
        self.density_out = nn.Linear(hidden_dim, 1)
        
        # Feature vector for color
        self.feature_layer = nn.Linear(hidden_dim, hidden_dim)
        
        # Color network (conditioned on view direction)
        self.color_layer1 = nn.Linear(hidden_dim + self.dir_input_dim, hidden_dim // 2)
        self.color_out = nn.Linear(hidden_dim // 2, 3)
    
    def forward(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            positions: (..., 3) 3D positions
            directions: (..., 3) view directions (normalized)
            
        Returns:
            (rgb, sigma) — rgb in [0,1], sigma >= 0
        """
        # Encode inputs
        pos_encoded = self.pos_encoding(positions)  # (..., pos_input_dim)
        dir_encoded = self.dir_encoding(directions)  # (..., dir_input_dim)
        
        # Density network
        h = pos_encoded
        for i, layer in enumerate(self.density_layers):
            if i == self.skip_layer:
                h = torch.cat([h, pos_encoded], dim=-1)
            h = layer(h)
            h = F.relu(h)
        
        # Density
        sigma = F.relu(self.density_out(h))  # (..., 1) — non-negative
        
        # Feature
        feature = self.feature_layer(h)  # (..., hidden_dim)
        
        # Color (conditioned on direction)
        h_color = torch.cat([feature, dir_encoded], dim=-1)
        h_color = F.relu(self.color_layer1(h_color))
        rgb = torch.sigmoid(self.color_out(h_color))  # (..., 3) in [0, 1]
        
        return rgb, sigma.squeeze(-1)


class NeRFSmall(nn.Module):
    """
    Smaller NeRF for faster training / testing. 4 layers, 128 hidden dim.
    """
    
    def __init__(self, pos_enc_dims: int = 6, dir_enc_dims: int = 2, hidden_dim: int = 128):
        super().__init__()
        
        self.pos_encoding = PositionalEncoding(pos_enc_dims)
        self.dir_encoding = PositionalEncoding(dir_enc_dims)
        
        pos_dim = 3 + 3 * pos_enc_dims * 2
        dir_dim = 3 + 3 * dir_enc_dims * 2
        
        self.net = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        
        self.density_out = nn.Linear(hidden_dim, 1)
        self.feature_out = nn.Linear(hidden_dim, hidden_dim)
        
        self.color_net = nn.Sequential(
            nn.Linear(hidden_dim + dir_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3), nn.Sigmoid()
        )
    
    def forward(self, positions, directions):
        pos_enc = self.pos_encoding(positions)
        dir_enc = self.dir_encoding(directions)
        
        h = self.net(pos_enc)
        sigma = F.relu(self.density_out(h)).squeeze(-1)
        feature = self.feature_out(h)
        rgb = self.color_net(torch.cat([feature, dir_enc], dim=-1))
        
        return rgb, sigma