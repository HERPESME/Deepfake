#!/usr/bin/env python3
"""
Dataset download script for deepfake detection project.
Downloads and organizes datasets from various sources.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import requests
import zipfile
import tarfile
from tqdm import tqdm
import json


class DatasetDownloader:
    """Download and organize deepfake datasets."""
    
    def __init__(self, data_root: str = "data/raw"):
        """
        Initialize dataset downloader.
        
        Args:
            data_root: Root directory for raw data
        """
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
    
    def download_faceforensics(self, compression: str = "c23"):
        """
        Download FaceForensics++ dataset.
        Note: Requires manual download from Google Form.
        
        Args:
            compression: Compression level ("c23" or "c40")
        """
        print("FaceForensics++ Dataset Download")
        print("=" * 50)
        print("FaceForensics++ requires manual download from:")
        print("https://github.com/ondyari/FaceForensics")
        print("\nSteps:")
        print("1. Fill out the Google Form to request access")
        print("2. Download the dataset manually")
        print("3. Extract to data/raw/faceforensics/")
        print(f"4. Ensure compression level {compression} is available")
        
        # Check if dataset exists
        dataset_path = self.data_root / "faceforensics"
        if dataset_path.exists():
            print(f"\nDataset directory exists: {dataset_path}")
            print("Please ensure the dataset is properly organized.")
        else:
            print(f"\nPlease create directory: {dataset_path}")
            print("And download the dataset there.")
    
    def download_celebd(self):
        """Download Celeb-DF dataset."""
        print("Celeb-DF Dataset Download")
        print("=" * 50)
        print("Celeb-DF dataset information:")
        print("Paper: https://arxiv.org/abs/1909.12962")
        print("Download: https://github.com/yuezunli/celeb-deepfakeforensics")
        print("\nManual download required:")
        print("1. Visit the GitHub repository")
        print("2. Follow the download instructions")
        print("3. Extract to data/raw/celebd/")
        
        dataset_path = self.data_root / "celebd"
        if dataset_path.exists():
            print(f"\nDataset directory exists: {dataset_path}")
        else:
            print(f"\nPlease create directory: {dataset_path}")
    
    def download_dfdc(self):
        """Download DFDC dataset."""
        print("DFDC Dataset Download")
        print("=" * 50)
        print("DeepFake Detection Challenge (DFDC) dataset:")
        print("Kaggle: https://www.kaggle.com/c/deepfake-detection-challenge")
        print("\nDownload instructions:")
        print("1. Create a Kaggle account")
        print("2. Join the competition")
        print("3. Download the dataset using Kaggle API:")
        print("   kaggle competitions download -c deepfake-detection-challenge")
        print("4. Extract to data/raw/dfdc/")
        
        dataset_path = self.data_root / "dfdc"
        if dataset_path.exists():
            print(f"\nDataset directory exists: {dataset_path}")
        else:
            print(f"\nPlease create directory: {dataset_path}")
    
    def download_wilddeepfake(self):
        """Download WildDeepfake dataset."""
        print("WildDeepfake Dataset Download")
        print("=" * 50)
        print("WildDeepfake dataset:")
        print("Paper: https://arxiv.org/abs/2002.00191")
        print("Download: https://github.com/deepfake-database/wilddeepfake")
        print("\nManual download required:")
        print("1. Visit the GitHub repository")
        print("2. Follow the download instructions")
        print("3. Extract to data/raw/wilddeepfake/")
        
        dataset_path = self.data_root / "wilddeepfake"
        if dataset_path.exists():
            print(f"\nDataset directory exists: {dataset_path}")
        else:
            print(f"\nPlease create directory: {dataset_path}")
    
    def setup_kaggle_api(self):
        """Setup Kaggle API for dataset downloads."""
        print("Kaggle API Setup")
        print("=" * 50)
        print("To download datasets from Kaggle:")
        print("1. Install Kaggle API: pip install kaggle")
        print("2. Get API credentials from https://www.kaggle.com/account")
        print("3. Download kaggle.json and place it in ~/.kaggle/")
        print("4. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
        
        # Check if kaggle is installed
        try:
            import kaggle
            print("\n✓ Kaggle API is installed")
        except ImportError:
            print("\n✗ Kaggle API not installed. Run: pip install kaggle")
    
    def create_dataset_info(self):
        """Create dataset information file."""
        dataset_info = {
            "faceforensics": {
                "name": "FaceForensics++",
                "description": "1000 original video sequences manipulated by 4 methods",
                "methods": ["DeepFakes", "Face2Face", "FaceSwap", "NeuralTextures"],
                "compression": ["c23", "c40"],
                "download_url": "https://github.com/ondyari/FaceForensics",
                "paper": "https://arxiv.org/abs/1901.08971"
            },
            "celebd": {
                "name": "Celeb-DF (v2)",
                "description": "590 real videos + 5,639 deepfake videos",
                "quality": "High-quality modern deepfakes",
                "download_url": "https://github.com/yuezunli/celeb-deepfakeforensics",
                "paper": "https://arxiv.org/abs/1909.12962"
            },
            "dfdc": {
                "name": "DeepFake Detection Challenge",
                "description": "100,000+ video clips from 3,426 actors",
                "methods": "Multiple face-swap techniques",
                "download_url": "https://www.kaggle.com/c/deepfake-detection-challenge",
                "paper": "https://arxiv.org/abs/2006.07397"
            },
            "wilddeepfake": {
                "name": "WildDeepfake",
                "description": "7,314 face sequences from 707 internet-sourced videos",
                "challenge": "Most realistic test set",
                "download_url": "https://github.com/deepfake-database/wilddeepfake",
                "paper": "https://arxiv.org/abs/2002.00191"
            }
        }
        
        info_path = self.data_root / "dataset_info.json"
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"Dataset information saved to: {info_path}")
    
    def check_dataset_structure(self):
        """Check if datasets are properly organized."""
        print("Dataset Structure Check")
        print("=" * 50)
        
        expected_structure = {
            "faceforensics": [
                "original_sequences/youtube/c23",
                "manipulated_sequences/Deepfakes/c23",
                "manipulated_sequences/Face2Face/c23",
                "manipulated_sequences/FaceSwap/c23",
                "manipulated_sequences/NeuralTextures/c23"
            ],
            "celebd": [
                "Celeb-real",
                "Celeb-synthesis"
            ],
            "dfdc": [
                "train_sample_videos",
                "test_videos"
            ],
            "wilddeepfake": [
                "real",
                "fake"
            ]
        }
        
        for dataset, structure in expected_structure.items():
            dataset_path = self.data_root / dataset
            print(f"\n{dataset.upper()}:")
            
            if not dataset_path.exists():
                print(f"  ✗ Directory not found: {dataset_path}")
                continue
            
            print(f"  ✓ Directory exists: {dataset_path}")
            
            for subdir in structure:
                subdir_path = dataset_path / subdir
                if subdir_path.exists():
                    print(f"    ✓ {subdir}")
                else:
                    print(f"    ✗ {subdir} (missing)")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Download deepfake datasets")
    parser.add_argument("--dataset", 
                       choices=["faceforensics", "celebd", "dfdc", "wilddeepfake", "all"],
                       default="all",
                       help="Dataset to download")
    parser.add_argument("--data_root", default="data/raw", help="Root directory for data")
    parser.add_argument("--setup_kaggle", action="store_true", help="Setup Kaggle API")
    parser.add_argument("--check_structure", action="store_true", help="Check dataset structure")
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.data_root)
    
    if args.setup_kaggle:
        downloader.setup_kaggle_api()
        return
    
    if args.check_structure:
        downloader.check_dataset_structure()
        return
    
    # Create dataset info
    downloader.create_dataset_info()
    
    # Download datasets
    if args.dataset == "all":
        downloader.download_faceforensics()
        downloader.download_celebd()
        downloader.download_dfdc()
        downloader.download_wilddeepfake()
    elif args.dataset == "faceforensics":
        downloader.download_faceforensics()
    elif args.dataset == "celebd":
        downloader.download_celebd()
    elif args.dataset == "dfdc":
        downloader.download_dfdc()
    elif args.dataset == "wilddeepfake":
        downloader.download_wilddeepfake()
    
    print("\n" + "=" * 50)
    print("Dataset download information completed!")
    print("Please follow the instructions above to download the datasets manually.")
    print("After downloading, run: python scripts/download_datasets.py --check_structure")


if __name__ == "__main__":
    main()
