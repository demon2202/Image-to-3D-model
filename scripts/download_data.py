"""
Download NeRF Synthetic dataset.
"""

import os
import sys
import urllib.request
import zipfile
import tarfile
from pathlib import Path


DATASETS = {
    "nerf_synthetic": {
        "url": "https://drive.google.com/uc?export=download&id=18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG",
        "filename": "nerf_synthetic.zip",
        "description": "NeRF Synthetic (Blender) dataset - 8 scenes"
    }
}


def download_from_kaggle():
    """Download using kaggle CLI."""
    print("Attempting Kaggle download...")
    print("\nOption 1: Download from Kaggle")
    print("  pip install kaggle")
    print("  kaggle datasets download -d sauravmaheshkar/nerf-dataset")
    print("  unzip nerf-dataset.zip -d data/")
    print("\nOption 2: Direct download from Kaggle web:")
    print("  https://www.kaggle.com/datasets/sauravmaheshkar/nerf-dataset")
    print("  https://www.kaggle.com/datasets/nguyenhung1903/nerf-synthetic-dataset")
    print("\nOption 3: Google Drive (original):")
    print("  https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1")


def setup_data_directory():
    """Create data directory structure."""
    dirs = [
        "data/raw",
        "data/processed",
        "data/nerf_synthetic",
        "outputs/sparse",
        "outputs/dense",
        "outputs/nerf_checkpoints",
        "outputs/renders",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Created: {d}")


if __name__ == "__main__":
    print("=" * 60)
    print("3D Reconstruction Pipeline - Data Setup")
    print("=" * 60)
    
    setup_data_directory()
    
    print("\n--- Dataset Download Instructions ---\n")
    download_from_kaggle()
    
    print("\n\nAfter downloading, place the scene folder (e.g., 'lego') in:")
    print("  data/nerf_synthetic/lego/")
    print("  Expected contents:")
    print("    transforms_train.json")
    print("    transforms_val.json")
    print("    transforms_test.json")
    print("    train/  (100 images)")
    print("    val/    (100 images)")
    print("    test/   (200 images)")