#!/usr/bin/env python3
"""
Simple sample data creator that doesn't require numpy.
Creates basic directory structure and placeholder files.
"""

import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw
import json


def create_simple_face(size=(224, 224), label="real"):
    """Create a simple colored square as placeholder."""
    # Create different colors for real vs fake
    if label == "real":
        color = (200, 150, 120)  # Skin tone
    else:
        color = (220, 160, 130)  # Slightly different
    
    img = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(img)
    
    # Draw simple face-like shapes
    center_x, center_y = size[0] // 2, size[1] // 2
    
    # Face circle
    face_size = 100
    draw.ellipse(
        [center_x - face_size//2, center_y - face_size//2,
         center_x + face_size//2, center_y + face_size//2],
        fill=color,
        outline=(100, 80, 70)
    )
    
    # Eyes
    eye_y = center_y - 30
    draw.ellipse([center_x - 30, eye_y - 10, center_x - 15, eye_y + 10], fill="black")
    draw.ellipse([center_x + 15, eye_y - 10, center_x + 30, eye_y + 10], fill="black")
    
    # Mouth
    mouth_y = center_y + 30
    draw.arc([center_x - 20, mouth_y, center_x + 20, mouth_y + 20], 0, 180, fill="black", width=3)
    
    return img


def create_sample_dataset_simple(
    output_dir="data/processed/sample", 
    num_train=20, 
    num_val=10, 
    num_test=10
):
    """Create sample dataset with minimal dependencies."""
    
    output_path = Path(output_dir)
    
    print("=" * 80)
    print("CREATING SAMPLE DATASET (Simple Version)")
    print("=" * 80)
    print(f"Output: {output_path}")
    print()
    
    # Create directory structure
    splits = {
        "train": num_train, 
        "val": num_val, 
        "test": num_test
    }
    
    labels = ["real", "fake"]
    
    for split, num_images in splits.items():
        for label in labels:
            split_dir = output_path / split / label
            split_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created: {split_dir}")
            
            # Create images
            print(f"  Creating {num_images} {label} images...")
            for i in range(num_images):
                img = create_simple_face(label=label)
                img_filename = f"{label}_{i:04d}.jpg"
                img_path = split_dir / img_filename
                img.save(img_path, quality=95)
            
            print(f"  ✅ Created {num_images} images")
    
    # Create splits file structure
    splits_dir = output_path.parent / "splits"
    splits_dir.mkdir(exist_ok=True)
    
    for split in splits.keys():
        split_file = splits_dir / f"{split}.txt"
        with open(split_file, 'w') as f:
            for label in labels:
                label_dir = output_path / split / label
                for img_file in sorted(label_dir.glob("*.jpg")):
                    f.write(f"{img_file}\n")
    
    # Create metadata
    metadata = {
        "dataset": "sample",
        "num_classes": 2,
        "classes": ["real", "fake"],
        "counts": {
            split: {
                label: len(list((output_path / split / label).glob("*.jpg")))
                for label in labels
            }
            for split in splits.keys()
        }
    }
    
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print()
    print("=" * 80)
    print("✅ SAMPLE DATASET CREATED!")
    print("=" * 80)
    print()
    print(f"📊 Summary:")
    for split, counts in metadata["counts"].items():
        total = sum(counts.values())
        print(f"  {split}: {total} images ({counts['real']} real, {counts['fake']} fake)")
    print()
    print("Next: Install dependencies and train:")
    print("  pip install -r requirements.txt")
    print("  python main.py train --model xception --dataset sample --epochs 5")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create simple sample dataset")
    parser.add_argument("--output_dir", default="data/processed/sample")
    parser.add_argument("--train", type=int, default=20)
    parser.add_argument("--val", type=int, default=10)
    parser.add_argument("--test", type=int, default=10)
    
    args = parser.parse_args()
    
    try:
        create_sample_dataset_simple(
            output_dir=args.output_dir,
            num_train=args.train,
            num_val=args.val,
            num_test=args.test
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
