"""
Data loading utilities for deepfake detection.
Supports PyTorch DataLoader with various augmentation strategies.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DeepfakeDataset(Dataset):
    """PyTorch Dataset for deepfake detection."""
    
    def __init__(self, 
                 image_paths: List[str],
                 labels: List[int],
                 transform: Optional[transforms.Compose] = None,
                 use_albumentations: bool = False,
                 albumentations_transform: Optional[A.Compose] = None):
        """
        Initialize dataset.
        
        Args:
            image_paths: List of image file paths
            labels: List of labels (0 for real, 1 for fake)
            transform: PyTorch transforms
            use_albumentations: Whether to use albumentations
            albumentations_transform: Albumentations transform pipeline
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.use_albumentations = use_albumentations
        self.albumentations_transform = albumentations_transform
        
        assert len(image_paths) == len(labels), "Number of images and labels must match"
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get item by index."""
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            if self.use_albumentations:
                image = np.array(Image.open(image_path).convert('RGB'))
                if self.albumentations_transform:
                    transformed = self.albumentations_transform(image=image)
                    image = transformed['image']
                else:
                    image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            else:
                image = Image.open(image_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                else:
                    # Default transform
                    image = transforms.ToTensor()(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy image
            image = torch.zeros(3, 224, 224)
        
        return image, label


class DataLoaderFactory:
    """Factory class for creating data loaders."""
    
    @staticmethod
    def get_transforms(split: str = "train", 
                      image_size: int = 224,
                      use_albumentations: bool = True) -> transforms.Compose:
        """
        Get transforms for different splits.
        
        Args:
            split: Data split ("train", "val", "test")
            image_size: Target image size
            use_albumentations: Whether to use albumentations
            
        Returns:
            Transform pipeline
        """
        if use_albumentations:
            return DataLoaderFactory._get_albumentations_transforms(split, image_size)
        else:
            return DataLoaderFactory._get_torchvision_transforms(split, image_size)
    
    @staticmethod
    def _get_albumentations_transforms(split: str, image_size: int) -> A.Compose:
        """Get albumentations transforms."""
        if split == "train":
            return A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.RandomGamma(p=0.3),
                A.HueSaturationValue(p=0.3),
                A.GaussNoise(p=0.2),
                A.Blur(blur_limit=3, p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
    
    @staticmethod
    def _get_torchvision_transforms(split: str, image_size: int) -> transforms.Compose:
        """Get torchvision transforms."""
        if split == "train":
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    @staticmethod
    def create_dataloaders(data_root: str,
                          batch_size: int = 32,
                          num_workers: int = 4,
                          image_size: int = 224,
                          use_albumentations: bool = True,
                          splits_dir: Optional[str] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test data loaders.
        
        Args:
            data_root: Root directory of processed data
            batch_size: Batch size
            num_workers: Number of worker processes
            image_size: Image size
            use_albumentations: Whether to use albumentations
            splits_dir: Directory containing split files
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        data_root = Path(data_root)
        
        if splits_dir is None:
            # Try multiple possible locations
            possible_splits = [
                Path(data_root) / "splits",
                Path(data_root).parent / "splits",
                Path(data_root).parent.parent / "splits"
            ]
            splits_dir = None
            for possible in possible_splits:
                if possible.exists():
                    splits_dir = possible
                    break
            if splits_dir is None:
                splits_dir = Path(data_root) / "splits"
        else:
            splits_dir = Path(splits_dir)
        
        loaders = {}
        
        for split in ["train", "val", "test"]:
            split_file = splits_dir / f"{split}.txt"
            
            if not split_file.exists():
                print(f"Split file {split_file} not found. Creating random splits...")
                DataLoaderFactory._create_random_splits(data_root, splits_dir)
            
            # Load file paths and labels
            image_paths, labels = DataLoaderFactory._load_split_data(split_file, data_root)
            
            # Create transforms
            if use_albumentations:
                transform = DataLoaderFactory._get_albumentations_transforms(split, image_size)
                dataset = DeepfakeDataset(image_paths, labels, 
                                       use_albumentations=True, 
                                       albumentations_transform=transform)
            else:
                transform = DataLoaderFactory._get_torchvision_transforms(split, image_size)
                dataset = DeepfakeDataset(image_paths, labels, transform=transform)
            
            # Create data loader
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=num_workers,
                pin_memory=True,
                drop_last=(split == "train")
            )
            
            loaders[split] = loader
        
        return loaders["train"], loaders["val"], loaders["test"]
    
    @staticmethod
    def _load_split_data(split_file: Path, data_root: Optional[Path] = None) -> Tuple[List[str], List[int]]:
        """Load image paths and labels from split file."""
        image_paths = []
        labels = []
        
        # Get data_root from split_file location if not provided
        if data_root is None:
            # If split_file is in a splits subdirectory, get parent
            if split_file.parent.name == "splits":
                data_root = split_file.parent.parent
            else:
                data_root = split_file.parent
        
        with open(split_file, 'r') as f:
            for line in f:
                image_path = line.strip()
                if not image_path:
                    continue
                
                # Resolve path relative to data_root
                full_path = (data_root / image_path).resolve()
                if not full_path.exists():
                    # Try alternate path resolution
                    alt_path = Path(image_path).resolve()
                    if alt_path.exists():
                        full_path = alt_path
                    else:
                        print(f"Warning: Image not found: {full_path} (from {image_path})")
                        continue
                
                image_paths.append(str(full_path))
                
                # Determine label from path
                if "real" in str(image_path).lower() or "real" in str(full_path).lower():
                    labels.append(0)
                elif "fake" in str(image_path).lower() or "fake" in str(full_path).lower():
                    labels.append(1)
                else:
                    # Try to infer from parent directory
                    parent_dir = Path(image_path).parent.name
                    if parent_dir == "real":
                        labels.append(0)
                    elif parent_dir == "fake":
                        labels.append(1)
                    else:
                        # Check full path parent
                        full_parent = full_path.parent.name
                        if full_parent == "real":
                            labels.append(0)
                        elif full_parent == "fake":
                            labels.append(1)
                        else:
                            print(f"Warning: Could not determine label for {image_path}")
                            labels.append(0)  # Default to real
        
        return image_paths, labels
    
    @staticmethod
    def _create_random_splits(data_root: Path, splits_dir: Path):
        """Create random train/val/test splits."""
        splits_dir.mkdir(exist_ok=True)
        
        all_files = []
        for label_dir in ["real", "fake"]:
            label_path = data_root / label_dir
            if label_path.exists():
                files = list(label_path.glob("*.jpg")) + list(label_path.glob("*.png"))
                all_files.extend(files)
        
        # Shuffle files
        np.random.seed(42)
        np.random.shuffle(all_files)
        
        # Split files
        n_files = len(all_files)
        n_train = int(n_files * 0.8)
        n_val = int(n_files * 0.1)
        
        train_files = all_files[:n_train]
        val_files = all_files[n_train:n_train + n_val]
        test_files = all_files[n_train + n_val:]
        
        # Save splits
        for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            split_file = splits_dir / f"{split_name}.txt"
            with open(split_file, 'w') as f:
                for file_path in files:
                    f.write(f"{file_path}\n")


class CrossDatasetLoader:
    """Data loader for cross-dataset evaluation."""
    
    @staticmethod
    def create_cross_dataset_loaders(train_dataset: str,
                                   test_dataset: str,
                                   data_root: str,
                                   batch_size: int = 32,
                                   image_size: int = 224) -> Tuple[DataLoader, DataLoader]:
        """
        Create data loaders for cross-dataset evaluation.
        
        Args:
            train_dataset: Name of training dataset
            test_dataset: Name of test dataset
            data_root: Root directory containing all datasets
            batch_size: Batch size
            image_size: Image size
            
        Returns:
            Tuple of (train_loader, test_loader)
        """
        data_root = Path(data_root)
        
        # Load training data
        train_data_dir = data_root / train_dataset / "processed"
        train_loader, _, _ = DataLoaderFactory.create_dataloaders(
            str(train_data_dir), batch_size=batch_size, image_size=image_size
        )
        
        # Load test data
        test_data_dir = data_root / test_dataset / "processed"
        _, _, test_loader = DataLoaderFactory.create_dataloaders(
            str(test_data_dir), batch_size=batch_size, image_size=image_size
        )
        
        return train_loader, test_loader


def main():
    """Test data loading functionality."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test data loading")
    parser.add_argument("--data_root", required=True, help="Root directory of processed data")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    
    args = parser.parse_args()
    
    # Create data loaders
    train_loader, val_loader, test_loader = DataLoaderFactory.create_dataloaders(
        args.data_root, args.batch_size, image_size=args.image_size
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test loading a batch
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}, Labels: {labels}")
        break


if __name__ == "__main__":
    main()
