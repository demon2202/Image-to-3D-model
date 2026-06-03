"""
NeRF training loop with logging and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from tqdm import tqdm
from typing import Optional, Dict

from .model import NeRFModel, NeRFSmall
from .render import render_rays
from .dataset import NeRFSyntheticDataset
from .evaluate import compute_psnr, compute_ssim, render_full_image


class NeRFTrainer:
    """
    Complete NeRF training pipeline.
    """
    
    def __init__(
        self,
        scene_dir: str,
        output_dir: str = "outputs/nerf_checkpoints",
        config: dict = None
    ):
        self.scene_dir = scene_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Config
        self.batch_size = self.config.get('batch_size', 1024)       # 4096 → 1024 for 4 GB GPU
        self.lr = self.config.get('learning_rate', 5e-4)
        self.num_iterations = self.config.get('num_iterations', 50000)
        self.num_coarse = self.config.get('num_coarse_samples', 64)
        self.num_fine = self.config.get('num_fine_samples', 128)
        self.near = self.config.get('near', 2.0)
        self.far = self.config.get('far', 6.0)
        self.log_every = self.config.get('log_every', 100)
        self.val_every = self.config.get('val_every', 5000)
        self.save_every = self.config.get('save_every', 10000)
        self.chunk_size = self.config.get('chunk_size', 16384)       # MLP chunk: tune down if OOM
        self.use_amp    = self.config.get('use_amp', True) and torch.cuda.is_available()
        
        # Dataset
        img_scale = self.config.get('img_scale', 0.5)  # Downscale for faster training
        self.train_dataset = NeRFSyntheticDataset(scene_dir, split="train", img_scale=img_scale)
        self.val_dataset = NeRFSyntheticDataset(scene_dir, split="val", img_scale=img_scale)
        
        # Models
        use_small = self.config.get('use_small_model', False)
        if use_small:
            self.model_coarse = NeRFSmall().to(self.device)
            self.model_fine = NeRFSmall().to(self.device)
        else:
            self.model_coarse = NeRFModel(
                pos_enc_dims=self.config.get('pos_enc_dims', 10),
                dir_enc_dims=self.config.get('dir_enc_dims', 4),
                hidden_dim=self.config.get('hidden_dim', 256)
            ).to(self.device)
            self.model_fine = NeRFModel(
                pos_enc_dims=self.config.get('pos_enc_dims', 10),
                dir_enc_dims=self.config.get('dir_enc_dims', 4),
                hidden_dim=self.config.get('hidden_dim', 256)
            ).to(self.device)
        
        # Optimizer
        params = list(self.model_coarse.parameters()) + list(self.model_fine.parameters())
        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999))

        # AMP — halves memory usage with negligible quality loss
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config.get('lr_decay_steps', 50000),
            gamma=self.config.get('lr_decay_rate', 0.5)
        )
        
        # Logging
        self.train_losses = []
        self.val_psnrs = []
        
        print(f"Coarse model params: {sum(p.numel() for p in self.model_coarse.parameters()):,}")
        print(f"Fine model params:   {sum(p.numel() for p in self.model_fine.parameters()):,}")
    
    def train(self):
        """Main training loop."""
        print(f"\nStarting training for {self.num_iterations} iterations")
        print(f"Batch size: {self.batch_size}, Near: {self.near}, Far: {self.far}")
        
        n_rays = len(self.train_dataset)
        
        self.model_coarse.train()
        self.model_fine.train()
        
        start_time = time.time()
        
        for iteration in range(1, self.num_iterations + 1):
            # Random ray batch
            indices = torch.randint(0, n_rays, (self.batch_size,))
            batch = {
                'rays_o': self.train_dataset.all_rays_o[indices].to(self.device),
                'rays_d': self.train_dataset.all_rays_d[indices].to(self.device),
                'rgb': self.train_dataset.all_rgb[indices].to(self.device)
            }
            
            # Forward pass — wrapped in AMP autocast (noop if use_amp=False)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                result = render_rays(
                    model=self.model_coarse,
                    rays_o=batch['rays_o'],
                    rays_d=batch['rays_d'],
                    near=self.near,
                    far=self.far,
                    num_coarse=self.num_coarse,
                    num_fine=self.num_fine,
                    model_fine=self.model_fine,
                    perturb=True,
                    white_background=True,
                    chunk=self.chunk_size
                )

                # Loss: MSE on both coarse and fine
                loss_coarse = nn.functional.mse_loss(result['rgb_coarse'], batch['rgb'])

                if 'rgb_fine' in result:
                    loss_fine = nn.functional.mse_loss(result['rgb_fine'], batch['rgb'])
                    loss = loss_coarse + loss_fine
                else:
                    loss = loss_coarse

            # Backward — AMP-aware
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            # Free CUDA cache periodically to avoid fragmentation
            if iteration % 100 == 0:
                torch.cuda.empty_cache()
            
            self.train_losses.append(loss.item())
            
            # Logging
            if iteration % self.log_every == 0:
                elapsed = time.time() - start_time
                psnr_train = -10 * np.log10(loss.item())
                lr = self.optimizer.param_groups[0]['lr']
                
                print(f"Iter {iteration:6d}/{self.num_iterations} | "
                      f"Loss: {loss.item():.6f} | "
                      f"PSNR: {psnr_train:.2f} dB | "
                      f"LR: {lr:.2e} | "
                      f"Time: {elapsed:.1f}s")
            
            # Validation
            if iteration % self.val_every == 0:
                self._validate(iteration)
            
            # Checkpoint
            if iteration % self.save_every == 0:
                self._save_checkpoint(iteration)
        
        # Final save
        self._save_checkpoint(self.num_iterations)
        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time:.1f}s ({total_time/60:.1f} min)")
    
    @torch.no_grad()
    def _validate(self, iteration: int):
        """Run validation on a single image."""
        self.model_coarse.eval()
        self.model_fine.eval()
        
        # Render one validation image
        img_idx = 0
        val_data = self.val_dataset.get_image_rays(img_idx)
        
        rgb_pred, depth_pred = render_full_image(
            model_coarse=self.model_coarse,
            model_fine=self.model_fine,
            rays_o=val_data['rays_o'].to(self.device),
            rays_d=val_data['rays_d'].to(self.device),
            H=self.val_dataset.H,
            W=self.val_dataset.W,
            near=self.near,
            far=self.far,
            num_coarse=self.num_coarse,
            num_fine=self.num_fine,
            chunk_size=self.chunk_size
        )
        
        rgb_gt = val_data['rgb'].numpy()
        
        psnr = compute_psnr(rgb_pred, rgb_gt)
        ssim = compute_ssim(rgb_pred, rgb_gt)
        self.val_psnrs.append(psnr)
        
        print(f"  [Val @ iter {iteration}] PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")
        
        # Save rendered image
        from PIL import Image
        img = (np.clip(rgb_pred, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img).save(
            os.path.join(self.output_dir, f"val_{iteration:06d}.png")
        )
        
        self.model_coarse.train()
        self.model_fine.train()
    
    def _save_checkpoint(self, iteration: int):
        """Save model checkpoint."""
        path = os.path.join(self.output_dir, f"checkpoint_{iteration:06d}.pt")
        torch.save({
            'iteration': iteration,
            'model_coarse': self.model_coarse.state_dict(),
            'model_fine': self.model_fine.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            'train_losses': self.train_losses,
            'val_psnrs': self.val_psnrs,
            'config': self.config,
        }, path)
        print(f"  Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model_coarse.load_state_dict(checkpoint['model_coarse'])
        self.model_fine.load_state_dict(checkpoint['model_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scaler' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_psnrs    = checkpoint.get('val_psnrs', [])
        print(f"Loaded checkpoint from iteration {checkpoint['iteration']}")