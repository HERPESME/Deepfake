#!/usr/bin/env python3
"""
Phase 6: Grad-CAM Visualization + Failure Analysis for all 7 models.
Generates side-by-side heatmap comparisons and systematic failure analysis.
"""

import sys
import json
import random
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from src.data.dataloader import DataLoaderFactory
from src.models.ensemble_models import load_model_from_path
from src.explainability.gradcam import GradCAM

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Model configs ──
MODEL_CONFIGS = {
    "efficientnet_b0": {
        "arch": "efficientnet_b0",
        "path": "experiments/effb0_celebd_full/final_model.pth",
    },
    "xception": {
        "arch": "xception",
        "path": "experiments/xception_celebd/checkpoint_epoch_10.pth",
    },
    "resnet50": {
        "arch": "resnet50",
        "path": "experiments/resnet50_celebd_optimized/final_model.pth",
    },
    "vit": {
        "arch": "vit",
        "path": "experiments/vit_celebd_optimized/checkpoint_epoch_15.pth",
    },
    "hybrid": {
        "arch": "hybrid_cnn_transformer",
        "path": "experiments/hybrid_celebd/final_model.pth",
    },
    "multiscale": {
        "arch": "multiscale",
        "path": "experiments/multiscale_celebd/final_model.pth",
    },
    "frequency_aware": {
        "arch": "frequency_aware",
        "path": "experiments/freq_aware_celebd/final_model.pth",
    },
}


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def find_target_layer(model, model_name):
    """Find the best target layer for Grad-CAM given a model."""
    target_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layers.append(name)

    if not target_layers:
        return None

    # For frequency_aware, pick the spatial branch's last conv
    if model_name == "frequency_aware":
        spatial_layers = [n for n in target_layers if "spatial_branch" in n]
        if spatial_layers:
            return spatial_layers[-1]

    # For hybrid, pick the backbone's last conv
    if model_name == "hybrid":
        backbone_layers = [n for n in target_layers if "backbone" in n or "spatial" in n]
        if backbone_layers:
            return backbone_layers[-1]

    # Default: last conv layer
    return target_layers[-1]


def preprocess_image(image_rgb):
    """Preprocess image for model input (same as ExplainabilityAnalyzer)."""
    image_resized = cv2.resize(image_rgb, (224, 224))
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_normalized = (image_normalized - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0).float()
    return image_tensor


def generate_gradcam_overlay(model, image_rgb, target_layer_name, device):
    """Generate Grad-CAM overlay for a single image on a single model."""
    image_tensor = preprocess_image(image_rgb).to(device)

    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1).item()
        conf = F.softmax(output, dim=1)[0].max().item()

    # Generate Grad-CAM
    try:
        gradcam = GradCAM(model, [target_layer_name])
        cam = gradcam.generate_cam(image_tensor, target_class=pred)
        gradcam.cleanup()

        # Ensure 2D
        if cam.ndim > 2:
            cam = cam.squeeze()
        if cam.ndim != 2:
            cam = cam.reshape(cam.shape[-2:]) if cam.ndim > 2 else np.zeros((7, 7))

        # Resize to image size
        cam_resized = cv2.resize(cam.astype(np.float32), (image_rgb.shape[1], image_rgb.shape[0]))

        # Normalize
        cam_min, cam_max = cam_resized.min(), cam_resized.max()
        if cam_max > cam_min:
            cam_resized = (cam_resized - cam_min) / (cam_max - cam_min + 1e-8)
        else:
            cam_resized = np.zeros_like(cam_resized)

        # Apply colormap
        heatmap = cm.jet(cam_resized)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)

        # Overlay
        overlay = cv2.addWeighted(image_rgb, 0.6, heatmap, 0.4, 0)

    except Exception as e:
        logger.warning(f"Grad-CAM failed for layer {target_layer_name}: {e}")
        overlay = image_rgb.copy()
        heatmap = np.zeros_like(image_rgb)

    return overlay, pred, conf


def select_sample_images(data_root, dataset_name, num_per_class=5):
    """Select sample images from a dataset using splits/test.txt."""
    base = Path(data_root) / dataset_name
    split_file = base / "splits" / "test.txt"

    real_images = []
    fake_images = []

    if split_file.exists():
        with open(split_file) as f:
            for line in f:
                rel_path = line.strip()
                if not rel_path:
                    continue
                full_path = base / rel_path
                if not full_path.exists():
                    continue
                if rel_path.startswith("real/"):
                    real_images.append(full_path)
                elif rel_path.startswith("fake/"):
                    fake_images.append(full_path)
    else:
        # Fallback: scan real/ and fake/ directories directly
        real_dir = base / "real"
        fake_dir = base / "fake"
        real_images = sorted(list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.png")))
        fake_images = sorted(list(fake_dir.glob("*.jpg")) + list(fake_dir.glob("*.png")))

    random.seed(42)
    selected_real = random.sample(real_images, min(num_per_class, len(real_images)))
    selected_fake = random.sample(fake_images, min(num_per_class, len(fake_images)))

    logger.info(f"  {dataset_name}: {len(real_images)} real, {len(fake_images)} fake in test split")

    # Return (path, label) tuples — label 0=real, 1=fake
    return [(str(p), 0) for p in selected_real] + [(str(p), 1) for p in selected_fake]


def create_comparison_grid(image_rgb, model_overlays, model_names, true_label, save_path):
    """Create a comparison grid: original + 7 model Grad-CAMs."""
    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models + 1, figsize=(3.5 * (n_models + 1), 4))

    label_str = "REAL" if true_label == 0 else "FAKE"

    # Original image
    axes[0].imshow(image_rgb)
    axes[0].set_title(f"Original\n({label_str})", fontsize=9, fontweight='bold')
    axes[0].axis('off')

    # Model overlays
    for i, (name, (overlay, pred, conf)) in enumerate(zip(model_names, model_overlays)):
        axes[i + 1].imshow(overlay)
        pred_str = "REAL" if pred == 0 else "FAKE"
        correct = "✓" if pred == true_label else "✗"
        color = "green" if pred == true_label else "red"
        axes[i + 1].set_title(f"{name}\n{pred_str} {conf:.0%} {correct}", fontsize=8, color=color)
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    device = get_device()
    logger.info(f"Device: {device}")

    output_dir = Path("experiments/phase6_gradcam")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "celebd_comparisons").mkdir(exist_ok=True)
    (output_dir / "ff_comparisons").mkdir(exist_ok=True)
    (output_dir / "failure_analysis").mkdir(exist_ok=True)

    # ── 1. Select sample images ──
    logger.info("Selecting sample images...")
    celebd_images = select_sample_images("data/processed", "celebd", num_per_class=5)
    ff_images = select_sample_images("data/processed", "faceforensics", num_per_class=5)
    logger.info(f"Selected {len(celebd_images)} CelebDF + {len(ff_images)} FF++ images")

    # ── 2. Load all models + find target layers ──
    logger.info("Loading models and discovering target layers...")
    models = {}
    target_layers = {}
    for name, cfg in MODEL_CONFIGS.items():
        model = load_model_from_path(cfg["arch"], cfg["path"], device)
        model.eval()
        layer = find_target_layer(model, name)
        if layer is None:
            logger.warning(f"No target layer found for {name}, skipping Grad-CAM")
        else:
            logger.info(f"  {name}: target_layer = {layer}")
        models[name] = model
        target_layers[name] = layer

    model_names = list(MODEL_CONFIGS.keys())
    all_predictions = {}

    # ── 3. Generate Grad-CAM grids ──
    for dataset_label, images, subdir in [
        ("celebd", celebd_images, "celebd_comparisons"),
        ("faceforensics", ff_images, "ff_comparisons"),
    ]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Generating Grad-CAM grids for {dataset_label}")
        logger.info(f"{'='*60}")

        for img_idx, (img_path, true_label) in enumerate(images):
            image_bgr = cv2.imread(img_path)
            if image_bgr is None:
                logger.warning(f"Could not read {img_path}, skipping")
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_rgb = cv2.resize(image_rgb, (224, 224))

            model_overlays = []
            for mname in model_names:
                layer = target_layers[mname]
                if layer is None:
                    # No Grad-CAM possible, just show prediction
                    image_tensor = preprocess_image(image_rgb).to(device)
                    models[mname].eval()
                    with torch.no_grad():
                        output = models[mname](image_tensor)
                        pred = torch.argmax(output, dim=1).item()
                        conf = F.softmax(output, dim=1)[0].max().item()
                    model_overlays.append((image_rgb.copy(), pred, conf))
                else:
                    overlay, pred, conf = generate_gradcam_overlay(
                        models[mname], image_rgb, layer, device
                    )
                    model_overlays.append((overlay, pred, conf))

                # Track predictions
                key = f"{dataset_label}_{img_idx}"
                if key not in all_predictions:
                    all_predictions[key] = {
                        "image_path": img_path,
                        "true_label": true_label,
                        "dataset": dataset_label,
                        "predictions": {},
                    }
                all_predictions[key]["predictions"][mname] = {
                    "pred": pred,
                    "confidence": conf,
                    "correct": pred == true_label,
                }

            # Save grid
            label_str = "real" if true_label == 0 else "fake"
            grid_path = output_dir / subdir / f"{label_str}_{img_idx}.png"
            create_comparison_grid(image_rgb, model_overlays, model_names, true_label, grid_path)
            logger.info(f"  Saved grid: {grid_path.name}")

    # ── 4. Failure analysis ──
    logger.info(f"\n{'='*60}")
    logger.info("FAILURE ANALYSIS")
    logger.info(f"{'='*60}")

    # Per-model accuracy breakdown
    model_stats = {name: {"celebd_correct": 0, "celebd_total": 0,
                          "ff_correct": 0, "ff_total": 0} for name in model_names}

    disagreement_cases = []

    for key, info in all_predictions.items():
        ds = info["dataset"]
        preds = info["predictions"]
        pred_values = [p["pred"] for p in preds.values()]

        for mname, p in preds.items():
            if ds == "celebd":
                model_stats[mname]["celebd_total"] += 1
                if p["correct"]:
                    model_stats[mname]["celebd_correct"] += 1
            else:
                model_stats[mname]["ff_total"] += 1
                if p["correct"]:
                    model_stats[mname]["ff_correct"] += 1

        # Check for disagreement
        if len(set(pred_values)) > 1:
            disagreement_cases.append(info)

    # Print accuracy summary
    logger.info(f"\n{'Model':<20} {'CelebDF':>10} {'FF++':>10}")
    logger.info("-" * 42)
    for name in model_names:
        s = model_stats[name]
        cd_acc = s["celebd_correct"] / max(s["celebd_total"], 1)
        ff_acc = s["ff_correct"] / max(s["ff_total"], 1)
        logger.info(f"{name:<20} {cd_acc:>9.0%} {ff_acc:>9.0%}")

    logger.info(f"\nDisagreement cases (models disagree): {len(disagreement_cases)}/{len(all_predictions)}")

    # Find hardest images (most models got wrong)
    hardest = []
    for key, info in all_predictions.items():
        n_wrong = sum(1 for p in info["predictions"].values() if not p["correct"])
        if n_wrong > 0:
            hardest.append((n_wrong, info))
    hardest.sort(key=lambda x: -x[0])

    if hardest:
        logger.info(f"\nHardest images (most models wrong):")
        for n_wrong, info in hardest[:5]:
            wrong_models = [m for m, p in info["predictions"].items() if not p["correct"]]
            logger.info(f"  {Path(info['image_path']).name} ({info['dataset']}, "
                        f"true={'REAL' if info['true_label']==0 else 'FAKE'}): "
                        f"{n_wrong}/{len(model_names)} wrong — {', '.join(wrong_models)}")

    # ── 5. Save summary ──
    summary = {
        "model_stats": model_stats,
        "all_predictions": all_predictions,
        "disagreement_count": len(disagreement_cases),
        "hardest_images": [
            {"image": info["image_path"], "dataset": info["dataset"],
             "true_label": info["true_label"], "num_wrong": n_wrong,
             "wrong_models": [m for m, p in info["predictions"].items() if not p["correct"]]}
            for n_wrong, info in hardest[:10]
        ],
    }

    with open(output_dir / "phase6_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\nAll Phase 6 results saved to {output_dir}/")


if __name__ == "__main__":
    main()
