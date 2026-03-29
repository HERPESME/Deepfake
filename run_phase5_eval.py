#!/usr/bin/env python3
"""
Phase 5: Comprehensive evaluation of all models.
1. Cross-dataset evaluation (each individual model on FaceForensics++)
2. Adversarial robustness testing (FGSM + PGD at multiple epsilons)
"""

import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from src.data.dataloader import DataLoaderFactory
from src.models.ensemble_models import load_model_from_path
from src.evaluation.adversarial import evaluate_adversarial_robustness

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# All models to evaluate
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


def run_cross_dataset_evaluation(device):
    """Evaluate each model individually on both Celeb-DF and FaceForensics++."""
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    import torch.nn.functional as F
    
    logger.info("=" * 70)
    logger.info("CROSS-DATASET EVALUATION (Individual Models)")
    logger.info("=" * 70)
    
    # Load test sets
    _, _, test_celebd = DataLoaderFactory.create_dataloaders(
        "data/processed/celebd", batch_size=16, image_size=224, num_workers=0
    )
    _, _, test_ff = DataLoaderFactory.create_dataloaders(
        "data/processed/faceforensics", batch_size=16, image_size=224, num_workers=0
    )
    
    results = {}
    
    for model_name, cfg in MODEL_CONFIGS.items():
        logger.info(f"\n--- Evaluating: {model_name} ---")
        model = load_model_from_path(cfg["arch"], cfg["path"], device)
        model.eval()
        
        model_results = {}
        
        for dataset_name, loader in [("celebd", test_celebd), ("faceforensics", test_ff)]:
            all_preds, all_labels, all_probs = [], [], []
            
            with torch.no_grad():
                for images, labels in loader:
                    images = images.to(device)
                    outputs = model(images)
                    probs = F.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs, dim=1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.numpy())
                    all_probs.extend(probs[:, 1].cpu().numpy())
            
            import numpy as np
            y_true = np.array(all_labels)
            y_pred = np.array(all_preds)
            y_prob = np.array(all_probs)
            
            acc = float(accuracy_score(y_true, y_pred))
            auc = float(roc_auc_score(y_true, y_prob))
            f1 = float(f1_score(y_true, y_pred, average="weighted"))
            
            model_results[dataset_name] = {"accuracy": acc, "auc": auc, "f1": f1}
            logger.info(f"  {dataset_name}: Acc={acc:.4f}, AUC={auc:.4f}, F1={f1:.4f}")
        
        results[model_name] = model_results
        del model
        if device == "mps":
            torch.mps.empty_cache()
    
    # Summary table
    logger.info(f"\n{'='*80}")
    logger.info("CROSS-DATASET SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"{'Model':<20} {'CelebDF Acc':>12} {'CelebDF AUC':>12} {'FF++ Acc':>12} {'FF++ AUC':>12} {'Gap':>8}")
    logger.info("-" * 80)
    for name, r in results.items():
        gap = r["celebd"]["auc"] - r["faceforensics"]["auc"]
        logger.info(
            f"{name:<20} {r['celebd']['accuracy']:>11.4f} {r['celebd']['auc']:>11.4f} "
            f"{r['faceforensics']['accuracy']:>11.4f} {r['faceforensics']['auc']:>11.4f} "
            f"{gap:>7.4f}"
        )
    
    return results


def run_adversarial_evaluation(device):
    """Run FGSM and PGD attacks on all models."""
    logger.info(f"\n{'='*70}")
    logger.info("ADVERSARIAL ROBUSTNESS TESTING")
    logger.info(f"{'='*70}")
    
    _, _, test_celebd = DataLoaderFactory.create_dataloaders(
        "data/processed/celebd", batch_size=16, image_size=224, num_workers=0
    )
    
    epsilons = [0.0, 0.01, 0.02, 0.05, 0.1]
    results = {}
    
    for model_name, cfg in MODEL_CONFIGS.items():
        logger.info(f"\n--- Adversarial testing: {model_name} ---")
        model = load_model_from_path(cfg["arch"], cfg["path"], device)
        
        adv_results = evaluate_adversarial_robustness(
            model, test_celebd, epsilons=epsilons,
            attacks=["fgsm", "pgd"], device=device, max_batches=30
        )
        results[model_name] = adv_results
        
        del model
        if device == "mps":
            torch.mps.empty_cache()
    
    # Summary tables
    for attack in ["fgsm", "pgd"]:
        logger.info(f"\n{'='*80}")
        logger.info(f"ADVERSARIAL ROBUSTNESS: {attack.upper()}")
        logger.info(f"{'='*80}")
        header = f"{'Model':<20}" + "".join(f" {'ε='+str(e):>10}" for e in epsilons)
        logger.info(header)
        logger.info("-" * 80)
        for name, r in results.items():
            vals = "".join(f" {r[attack][str(e)]['accuracy']:>9.4f}" for e in epsilons)
            logger.info(f"{name:<20}{vals}")
    
    return results


def main():
    device = get_device()
    logger.info(f"Device: {device}")
    
    output_dir = Path("experiments/phase5_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Cross-dataset evaluation
    cross_results = run_cross_dataset_evaluation(device)
    with open(output_dir / "cross_dataset_results.json", "w") as f:
        json.dump(cross_results, f, indent=2)
    
    # 2. Adversarial robustness
    adv_results = run_adversarial_evaluation(device)
    with open(output_dir / "adversarial_results.json", "w") as f:
        json.dump(adv_results, f, indent=2)
    
    logger.info(f"\nAll Phase 5 results saved to {output_dir}/")


if __name__ == "__main__":
    main()
