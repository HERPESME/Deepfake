#!/usr/bin/env python3
"""
Run ensemble evaluation on Celeb-DF test set and FaceForensics++ cross-dataset.
Evaluates multiple ensemble configurations with hard voting, soft voting, and stacking.
"""

import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from src.data.dataloader import DataLoaderFactory
from src.models.ensemble_models import EnsembleDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_ensemble_evaluation():
    device = get_device()
    logger.info(f"Device: {device}")
    
    # Model checkpoint paths
    MODEL_CONFIGS = {
        "efficientnet_b0": {
            "arch": "efficientnet_b0",
            "path": "experiments/effb0_celebd_full/final_model.pth",
            "weight": 1.0,
        },
        "xception": {
            "arch": "xception",
            "path": "experiments/xception_celebd/checkpoint_epoch_10.pth",
            "weight": 1.0,
        },
        "resnet50": {
            "arch": "resnet50",
            "path": "experiments/resnet50_celebd_optimized/final_model.pth",
            "weight": 1.0,
        },
        "vit": {
            "arch": "vit",
            "path": "experiments/vit_celebd_optimized/checkpoint_epoch_15.pth",
            "weight": 0.8,  # Slightly lower due to lower in-distribution accuracy
        },
        "hybrid": {
            "arch": "hybrid_cnn_transformer",
            "path": "experiments/hybrid_celebd/final_model.pth",
            "weight": 1.0,
        },
        "multiscale": {
            "arch": "multiscale",
            "path": "experiments/multiscale_celebd/final_model.pth",
            "weight": 1.0,
        },
    }
    
    # Ensemble configurations to test
    ENSEMBLE_CONFIGS = {
        "cnn_only": ["efficientnet_b0", "xception", "resnet50"],
        "full_6model": ["efficientnet_b0", "xception", "resnet50", "vit", "hybrid", "multiscale"],
        "best_pair_eff_vit": ["efficientnet_b0", "vit"],
        "cnn_plus_vit": ["efficientnet_b0", "xception", "resnet50", "vit"],
        "new_models": ["hybrid", "multiscale"],
        "all_best": ["efficientnet_b0", "xception", "hybrid", "multiscale", "vit"],
    }
    
    # Load data
    logger.info("Loading Celeb-DF test data...")
    _, val_loader_celebd, test_loader_celebd = DataLoaderFactory.create_dataloaders(
        "data/processed/celebd", batch_size=16, image_size=224, num_workers=0
    )
    
    logger.info("Loading FaceForensics++ test data...")
    _, _, test_loader_ff = DataLoaderFactory.create_dataloaders(
        "data/processed/faceforensics", batch_size=16, image_size=224, num_workers=0
    )
    
    all_results = {}
    
    for ensemble_name, model_names in ENSEMBLE_CONFIGS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating ensemble: {ensemble_name}")
        logger.info(f"Models: {model_names}")
        logger.info(f"{'='*60}")
        
        # Build ensemble
        ensemble = EnsembleDetector(device=device)
        for name in model_names:
            cfg = MODEL_CONFIGS[name]
            ensemble.load_and_add_model(name, cfg["arch"], cfg["path"], cfg["weight"])
        
        ensemble_results = {"models": model_names}
        
        # Evaluate with hard voting and soft voting on Celeb-DF
        for method in ["hard_vote", "soft_vote"]:
            logger.info(f"\n--- {method} on Celeb-DF ---")
            results = ensemble.evaluate_on_dataloader(test_loader_celebd, method=method)
            ensemble_results[f"celebd_{method}"] = results
            logger.info(f"  Accuracy: {results['accuracy']:.4f}")
            logger.info(f"  AUC: {results['auc']:.4f}")
            logger.info(f"  F1: {results['f1_score']:.4f}")
            
            # Cross-dataset on FaceForensics++
            logger.info(f"\n--- {method} on FaceForensics++ ---")
            results_ff = ensemble.evaluate_on_dataloader(test_loader_ff, method=method)
            ensemble_results[f"ff_{method}"] = results_ff
            logger.info(f"  Accuracy: {results_ff['accuracy']:.4f}")
            logger.info(f"  AUC: {results_ff['auc']:.4f}")
            logger.info(f"  F1: {results_ff['f1_score']:.4f}")
        
        # Train and evaluate stacking (use validation set for meta-training)
        logger.info(f"\n--- Training stacking meta-classifier ---")
        ensemble.train_stacking(val_loader_celebd)
        
        logger.info(f"\n--- stacked on Celeb-DF ---")
        results_stack = ensemble.evaluate_on_dataloader(test_loader_celebd, method="stacked")
        ensemble_results["celebd_stacked"] = results_stack
        logger.info(f"  Accuracy: {results_stack['accuracy']:.4f}")
        logger.info(f"  AUC: {results_stack['auc']:.4f}")
        logger.info(f"  F1: {results_stack['f1_score']:.4f}")
        
        logger.info(f"\n--- stacked on FaceForensics++ ---")
        results_stack_ff = ensemble.evaluate_on_dataloader(test_loader_ff, method="stacked")
        ensemble_results["ff_stacked"] = results_stack_ff
        logger.info(f"  Accuracy: {results_stack_ff['accuracy']:.4f}")
        logger.info(f"  AUC: {results_stack_ff['auc']:.4f}")
        logger.info(f"  F1: {results_stack_ff['f1_score']:.4f}")
        
        all_results[ensemble_name] = ensemble_results
        
        # Free memory
        del ensemble
        torch.mps.empty_cache() if device == "mps" else None
    
    # Save all results
    output_dir = Path("experiments/ensemble_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "all_ensemble_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Print summary table
    logger.info(f"\n{'='*80}")
    logger.info("ENSEMBLE RESULTS SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"{'Ensemble':<20} {'Method':<12} {'CelebDF Acc':>12} {'CelebDF AUC':>12} {'FF++ Acc':>12} {'FF++ AUC':>12}")
    logger.info("-" * 80)
    
    for ensemble_name, results in all_results.items():
        for method in ["soft_vote", "hard_vote", "stacked"]:
            celebd_key = f"celebd_{method}"
            ff_key = f"ff_{method}"
            if celebd_key in results and ff_key in results:
                logger.info(
                    f"{ensemble_name:<20} {method:<12} "
                    f"{results[celebd_key]['accuracy']:>11.4f} "
                    f"{results[celebd_key]['auc']:>11.4f} "
                    f"{results[ff_key]['accuracy']:>11.4f} "
                    f"{results[ff_key]['auc']:>11.4f}"
                )
    
    logger.info(f"\nResults saved to {output_dir / 'all_ensemble_results.json'}")


if __name__ == "__main__":
    run_ensemble_evaluation()
