#!/usr/bin/env python3
"""
Create sample dataset for quick testing without downloading full datasets.
Generates synthetic face images for testing the pipeline.
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
from tqdm import tqdm


def create_synthetic_face(size=(224, 224), label="real", idx=0):
    """
    Create a synthetic face image for testing.

    Args:
        size: Image size (width, height)
        label: 'real' or 'fake'
        idx: Image index for variation

    Returns:
        PIL Image
    """
    # Create base image with skin tone
    if label == "real":
        # Real faces: more natural colors
        base_color = (
            int(220 + np.random.randint(-20, 20)),
            int(180 + np.random.randint(-20, 20)),
            int(160 + np.random.randint(-20, 20)),
        )
    else:
        # Fake faces: slightly off colors (simulating artifacts)
        base_color = (
            int(230 + np.random.randint(-30, 30)),
            int(190 + np.random.randint(-30, 30)),
            int(170 + np.random.randint(-30, 30)),
        )

    img = Image.new("RGB", size, base_color)
    draw = ImageDraw.Draw(img)

    # Draw simple face features
    center_x, center_y = size[0] // 2, size[1] // 2

    # Face oval
    face_bbox = [center_x - 70, center_y - 90, center_x + 70, center_y + 90]
    draw.ellipse(face_bbox, fill=base_color, outline=(100, 80, 70), width=2)

    # Eyes
    eye_y = center_y - 20
    # Left eye
    draw.ellipse(
        [center_x - 40, eye_y - 10, center_x - 20, eye_y + 10],
        fill="white",
        outline="black",
        width=1,
    )
    draw.ellipse([center_x - 35, eye_y - 5, center_x - 25, eye_y + 5], fill="black")
    # Right eye
    draw.ellipse(
        [center_x + 20, eye_y - 10, center_x + 40, eye_y + 10],
        fill="white",
        outline="black",
        width=1,
    )
    draw.ellipse([center_x + 25, eye_y - 5, center_x + 35, eye_y + 5], fill="black")

    # Nose
    nose_points = [
        (center_x, center_y - 10),
        (center_x - 10, center_y + 10),
        (center_x + 10, center_y + 10),
    ]
    draw.polygon(nose_points, fill=None, outline=(100, 80, 70))

    # Mouth
    mouth_bbox = [center_x - 30, center_y + 30, center_x + 30, center_y + 50]
    draw.arc(mouth_bbox, 0, 180, fill="black", width=2)

    # Add artifacts for fake images
    if label == "fake":
        # Add some noise/artifacts
        noise_layer = np.array(img)
        noise = np.random.randint(-20, 20, noise_layer.shape, dtype=np.int16)
        noise_layer = np.clip(noise_layer + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(noise_layer)

        # Add some color inconsistencies
        draw = ImageDraw.Draw(img)
        for _ in range(5):
            x = np.random.randint(0, size[0])
            y = np.random.randint(0, size[1])
            w = np.random.randint(5, 15)
            draw.ellipse(
                [x, y, x + w, y + w],
                fill=(
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                ),
            )

    return img


def create_sample_dataset(
    output_dir="data/processed/sample", num_train=100, num_val=30, num_test=20
):
    """
    Create a complete sample dataset with train/val/test splits.

    Args:
        output_dir: Output directory for sample dataset
        num_train: Number of training images per class
        num_val: Number of validation images per class
        num_test: Number of test images per class
    """
    output_path = Path(output_dir)

    print("=" * 80)
    print("CREATING SAMPLE DATASET FOR TESTING")
    print("=" * 80)
    print(f"Output directory: {output_path}")
    print(f"Train: {num_train} real + {num_train} fake")
    print(f"Val: {num_val} real + {num_val} fake")
    print(f"Test: {num_test} real + {num_test} fake")
    print()

    # Create directory structure
    splits = {"train": num_train, "val": num_val, "test": num_test}

    labels = ["real", "fake"]

    for split, num_images in splits.items():
        for label in labels:
            split_dir = output_path / split / label
            split_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created: {split_dir}")

    print()

    # Generate images
    metadata = {
        "dataset": "sample",
        "num_classes": 2,
        "classes": ["real", "fake"],
        "splits": {},
    }

    for split, num_images in splits.items():
        print(f"🎨 Generating {split} images...")
        split_metadata = {"real": [], "fake": []}

        for label in labels:
            split_dir = output_path / split / label

            for i in tqdm(range(num_images), desc=f"  {label}"):
                # Create synthetic image
                img = create_synthetic_face(label=label, idx=i)

                # Save image
                img_filename = f"{label}_{i:04d}.jpg"
                img_path = split_dir / img_filename
                img.save(img_path, quality=95)

                # Add to metadata
                split_metadata[label].append(
                    {
                        "filename": img_filename,
                        "path": str(img_path.relative_to(output_path)),
                        "label": label,
                        "label_id": 1 if label == "fake" else 0,
                    }
                )

        metadata["splits"][split] = split_metadata
        print(f"  ✅ Created {num_images * 2} images")

    # Save metadata
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print()
    print(f"📄 Saved metadata: {metadata_path}")

    # Create labels file for each split
    for split in splits.keys():
        labels_file = output_path / split / "labels.txt"
        with open(labels_file, "w") as f:
            for label_id, label_name in enumerate(["real", "fake"]):
                f.write(f"{label_id}\t{label_name}\n")

    print()
    print("=" * 80)
    print("✅ SAMPLE DATASET CREATED SUCCESSFULLY!")
    print("=" * 80)
    print()
    print("You can now train with:")
    print(
        "  python main.py train --model xception --dataset sample --data_root data/processed --epochs 5"
    )
    print()
    print("Or test data loading:")
    print(
        "  python src/data/dataloader.py --data_root data/processed/sample --batch_size 4"
    )


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Create sample dataset for testing")
    parser.add_argument(
        "--output_dir",
        default="data/processed/sample",
        help="Output directory for sample dataset",
    )
    parser.add_argument(
        "--train", type=int, default=100, help="Number of training images per class"
    )
    parser.add_argument(
        "--val", type=int, default=30, help="Number of validation images per class"
    )
    parser.add_argument(
        "--test", type=int, default=20, help="Number of test images per class"
    )

    args = parser.parse_args()

    try:
        create_sample_dataset(
            output_dir=args.output_dir,
            num_train=args.train,
            num_val=args.val,
            num_test=args.test,
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
