"""
Train NeRF on a scene.
"""

import sys
import os
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phase4_nerf.train import NeRFTrainer


def main():
    scene_dir = sys.argv[1] if len(sys.argv) > 1 else "data/nerf_synthetic/lego"
    
    if not os.path.exists(scene_dir):
        print(f"Scene directory not found: {scene_dir}")
        print("Please download the dataset first: python scripts/download_data.py")
        return
    
    # Load config
    config_path = "config/default.yaml"
    if os.path.exists(config_path):
        with open(config_path) as f:
            full_config = yaml.safe_load(f)
        config = full_config.get('nerf', {})
    else:
        config = {}
    
    # Override for faster training (comment out for full quality)
    config.update({
        'num_iterations': 5000,      # Full quality: 200000
        'batch_size': 4096,
        'img_scale': 0.5,             # Half resolution for speed
        'num_coarse_samples': 64,
        'num_fine_samples': 128,
        'near': 2.0,
        'far': 6.0,
        'learning_rate': 5e-4,
        'log_every': 100,
        'val_every': 2500,
        'save_every': 10000,
        'use_small_model': False,      # True for quick test
    })
    
    output_dir = os.path.join("outputs/nerf_checkpoints", os.path.basename(scene_dir))
    
    trainer = NeRFTrainer(
        scene_dir=scene_dir,
        output_dir=output_dir,
        config=config
    )
    
    # Optional: resume from checkpoint
    # trainer.load_checkpoint("outputs/nerf_checkpoints/lego/checkpoint_010000.pt")
    
    trainer.train()


if __name__ == "__main__":
    main()