#!/usr/bin/env python3
"""
Main entry point for the Deepfake Detection Project.
Provides unified interface for training, evaluation, and inference.
"""

import os
import sys
import argparse
import torch
from pathlib import Path
import logging
from typing import Optional, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config import ConfigLoader
from src.data.dataloader import DataLoaderFactory
from src.models.baseline_models import create_model as create_baseline_model
from src.models.advanced_models import create_advanced_model
from src.training.trainer import DeepfakeTrainer, CrossDatasetTrainer
from src.evaluation.metrics import ModelEvaluator
from src.reporting.report_generator import ReportGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_model(args):
    """Train a deepfake detection model."""
    logger.info("=" * 80)
    logger.info("DEEPFAKE DETECTION - TRAINING MODE")
    logger.info("=" * 80)

    # Load configuration
    if args.config:
        config = ConfigLoader.load_config(args.config)
    else:
        config = ConfigLoader.create_default_config()

    # Override config with command line arguments
    if args.model:
        config.model.name = args.model
    if args.dataset:
        config.data.dataset = args.dataset
    if args.data_root:
        config.data.data_root = args.data_root
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr

    logger.info(f"Model: {config.model.name}")
    logger.info(f"Dataset: {config.data.dataset}")
    logger.info(f"Data root: {config.data.data_root}")
    logger.info(f"Epochs: {config.training.epochs}")
    logger.info(f"Batch size: {config.data.batch_size}")
    logger.info(f"Learning rate: {config.training.learning_rate}")

    # Setup device
    if args.cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Device: {device}")

    # Create data loaders
    logger.info("Loading data...")
    data_path = Path(config.data.data_root) / config.data.dataset

    try:
        train_loader, val_loader, test_loader = DataLoaderFactory.create_dataloaders(
            str(data_path),
            batch_size=config.data.batch_size,
            image_size=config.data.image_size,
            num_workers=config.data.num_workers,
            use_albumentations=config.data.use_albumentations,
        )
        logger.info(f"Train samples: {len(train_loader.dataset)}")
        logger.info(f"Val samples: {len(val_loader.dataset)}")
        logger.info(f"Test samples: {len(test_loader.dataset)}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        logger.error(
            "Please ensure data is preprocessed and available at the specified path."
        )
        sys.exit(1)

    # Create model
    logger.info(f"Creating model: {config.model.name}")
    try:
        if config.model.name in [
            "xception",
            "efficientnet_b0",
            "efficientnet_b4",
            "vit",
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
        ]:
            # ViT uses 'dropout' parameter, not 'dropout_rate'
            if config.model.name == "vit":
                model = create_baseline_model(
                    config.model.name,
                    num_classes=config.model.num_classes,
                    pretrained=config.model.pretrained,
                    dropout=config.model.dropout_rate,  # ViT uses 'dropout' not 'dropout_rate'
                )
            else:
                model = create_baseline_model(
                    config.model.name,
                    num_classes=config.model.num_classes,
                    pretrained=config.model.pretrained,
                    dropout_rate=config.model.dropout_rate,
                )
        else:
            model = create_advanced_model(
                config.model.name,
                num_classes=config.model.num_classes,
                pretrained=config.model.pretrained,
            )
        logger.info(f"Model created successfully")
        logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        sys.exit(1)

    # Create trainer
    experiment_name = (
        args.experiment_name or f"{config.model.name}_{config.data.dataset}"
    )
    # Get mixed precision setting from config or args
    mixed_precision = getattr(args, 'mixed_precision', None)
    if mixed_precision is None:
        mixed_precision = config.advanced.mixed_precision if hasattr(config, 'advanced') and hasattr(config.advanced, 'mixed_precision') else False
    
    trainer = DeepfakeTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        experiment_name=experiment_name,
        use_wandb=config.logging.use_wandb,
        wandb_project=config.logging.wandb_project,
        mixed_precision=mixed_precision,
    )

    # Train model
    logger.info("Starting training...")
    try:
        results = trainer.train(
            epochs=config.training.epochs,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            optimizer_type=config.training.optimizer,
            scheduler_type=config.training.scheduler,
            early_stopping_patience=config.training.early_stopping_patience,
            save_frequency=config.training.save_frequency,
        )

        logger.info("Training completed!")
        logger.info(f"Best validation AUC: {results['best_val_score']:.4f}")

        # Generate report if requested
        if config.output.generate_report:
            logger.info("Generating report...")
            try:
                report_gen = ReportGenerator(output_dir="reports")
                experiment_data = {
                    "metrics": results.get("final_results", {}),
                    "training_history": results.get("history", {})
                }
                report_paths = report_gen.generate_experiment_report(
                    experiment_data=experiment_data,
                    model_name=config.model.name,
                    dataset_name=config.data.dataset,
                    output_format="both"
                )
                logger.info(f"Report saved to: {report_paths}")
            except Exception as e:
                logger.warning(f"Could not generate report: {e}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise


def evaluate_model(args):
    """Evaluate a trained model."""
    logger.info("=" * 80)
    logger.info("DEEPFAKE DETECTION - EVALUATION MODE")
    logger.info("=" * 80)

    if not args.model_path:
        logger.error("Model path is required for evaluation mode")
        sys.exit(1)

    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)

    # Load configuration
    if args.config:
        config = ConfigLoader.load_config(args.config)
    else:
        config = ConfigLoader.create_default_config()

    # Override with CLI args
    if args.dataset:
        config.data.dataset = args.dataset
    if args.data_root:
        config.data.data_root = args.data_root

    # Setup device
    if args.cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Device: {device}")

    # Load data
    logger.info(f"Loading test data from: {config.data.dataset}")
    data_path = Path(config.data.data_root) / config.data.dataset

    try:
        _, _, test_loader = DataLoaderFactory.create_dataloaders(
            str(data_path),
            batch_size=args.batch_size or 32,
            image_size=config.data.image_size,
        )
        logger.info(f"Test samples: {len(test_loader.dataset)}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)

    # Load model
    logger.info(f"Loading model from: {model_path}")
    try:
        # Determine model type from filename or config
        model_name = args.model or config.model.name

        if model_name in [
            "xception",
            "efficientnet_b0",
            "efficientnet_b4",
            "vit",
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
        ]:
            model = create_baseline_model(model_name, num_classes=2, pretrained=False)
        else:
            model = create_advanced_model(model_name, num_classes=2, pretrained=False)

        # Load weights (use weights_only=False for PyTorch 2.6+ compatibility)
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        except TypeError:
            # Fallback for older PyTorch versions
            checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(device)
        model.eval()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)

    # Evaluate
    logger.info("Evaluating model...")
    evaluator = ModelEvaluator(output_dir="reports/evaluation")

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    import numpy as np

    results = evaluator.evaluate_model(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
        model_name=model_name,
        dataset_name=config.data.dataset,
    )

    # Extract metrics from nested structure
    metrics = results.get('metrics', {})
    
    logger.info("Evaluation Results:")
    logger.info(f"  Accuracy: {metrics.get('accuracy', 0.0):.4f}")
    logger.info(f"  AUC: {metrics.get('auc', metrics.get('roc_auc', 0.0)):.4f}")
    logger.info(f"  Precision: {metrics.get('precision', 0.0):.4f}")
    logger.info(f"  Recall: {metrics.get('recall', 0.0):.4f}")
    logger.info(f"  F1-Score: {metrics.get('f1_score', metrics.get('f1', 0.0)):.4f}")
    if 'visualization_dir' in results:
        logger.info(f"Results saved to: {results['visualization_dir']}")
    logger.info(f"Full results: {list(results.keys())}")


def cross_dataset_evaluation(args):
    """Perform cross-dataset evaluation."""
    logger.info("=" * 80)
    logger.info("DEEPFAKE DETECTION - CROSS-DATASET EVALUATION")
    logger.info("=" * 80)

    # Load configuration
    if args.config:
        config = ConfigLoader.load_config(args.config)
    else:
        config = ConfigLoader.create_default_config()

    # Setup device
    if args.cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Create model
    model_name = args.model or config.model.name
    if model_name in [
        "xception",
        "efficientnet_b0",
        "efficientnet_b4",
        "vit",
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
    ]:
        model = create_baseline_model(
            model_name, num_classes=2, pretrained=config.model.pretrained
        )
    else:
        model = create_advanced_model(
            model_name, num_classes=2, pretrained=config.model.pretrained
        )

    # Create cross-dataset trainer
    train_dataset = args.train_dataset or config.cross_dataset.train_dataset
    test_datasets = args.test_datasets or config.cross_dataset.test_datasets

    trainer = CrossDatasetTrainer(device=device)

    # Train and evaluate
    logger.info(f"Training on: {train_dataset}")
    logger.info(f"Testing on: {test_datasets}")

    results = trainer.train_and_evaluate(
        model_name=model_name,
        train_dataset=train_dataset,
        test_datasets=test_datasets,
        data_root=config.data.data_root,
        epochs=config.training.epochs,
        learning_rate=config.training.learning_rate,
        batch_size=config.data.batch_size
    )

    logger.info("Cross-dataset evaluation completed!")
    logger.info(f"Results saved to: experiments/cross_dataset_{train_dataset}")


def inference_single(args):
    """Run inference on a single image."""
    logger.info("=" * 80)
    logger.info("DEEPFAKE DETECTION - INFERENCE MODE")
    logger.info("=" * 80)

    if not args.model_path:
        logger.error("Model path is required for inference")
        sys.exit(1)

    if not args.image_path:
        logger.error("Image path is required for inference")
        sys.exit(1)

    from PIL import Image
    from torchvision import transforms
    import numpy as np

    # Setup device
    if args.cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Load model
    model_name = args.model or "xception"
    if model_name in [
        "xception",
        "efficientnet_b0",
        "efficientnet_b4",
        "vit",
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
    ]:
        model = create_baseline_model(model_name, num_classes=2, pretrained=False)
    else:
        model = create_advanced_model(model_name, num_classes=2, pretrained=False)

    # Load checkpoint (use weights_only=False for PyTorch 2.6+ compatibility)
    try:
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions
        checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # Load and preprocess image
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(args.image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(output, dim=1)

    fake_prob = probs[0, 1].item()
    real_prob = probs[0, 0].item()
    prediction = "FAKE" if pred[0] == 1 else "REAL"

    logger.info(f"Image: {args.image_path}")
    logger.info(f"Prediction: {prediction}")
    logger.info(f"Real probability: {real_prob:.4f}")
    logger.info(f"Fake probability: {fake_prob:.4f}")

    if args.explainability:
        logger.info("Generating explainability visualizations...")
        from src.explainability.gradcam import ExplainabilityAnalyzer

        analyzer = ExplainabilityAnalyzer(model, output_dir="reports/explainability")
        # Pass model name for filename differentiation
        analysis = analyzer.analyze_image(args.image_path, label=pred[0].item(), model_name=model_name)
        logger.info(f"Visualizations saved to: reports/explainability")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Deepfake Detection Project - State-of-the-Art Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train XceptionNet on FaceForensics++
  python main.py train --model xception --dataset faceforensics --epochs 50

  # Evaluate a trained model
  python main.py evaluate --model_path experiments/model.pth --dataset celebd

  # Cross-dataset evaluation
  python main.py cross-dataset --model xception --train-dataset faceforensics

  # Single image inference
  python main.py inference --model_path experiments/model.pth --image_path test.jpg
        """,
    )

    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")

    # Training mode
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--config", help="Path to config file")
    train_parser.add_argument("--model", help="Model name")
    train_parser.add_argument("--dataset", help="Dataset name")
    train_parser.add_argument("--data_root", help="Data root directory")
    train_parser.add_argument("--epochs", type=int, help="Number of epochs")
    train_parser.add_argument("--batch_size", type=int, help="Batch size")
    train_parser.add_argument("--lr", type=float, help="Learning rate")
    train_parser.add_argument("--cpu", action="store_true", help="Use CPU only")
    train_parser.add_argument("--experiment_name", help="Experiment name")
    train_parser.add_argument("--mixed_precision", action="store_true", default=None, help="Enable mixed precision training (30-50% speedup)")
    train_parser.add_argument("--no_mixed_precision", action="store_false", dest="mixed_precision", help="Disable mixed precision training")

    # Evaluation mode
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument(
        "--model_path", required=True, help="Path to trained model"
    )
    eval_parser.add_argument("--model", help="Model architecture name")
    eval_parser.add_argument("--dataset", help="Dataset to evaluate on")
    eval_parser.add_argument("--data_root", help="Data root directory")
    eval_parser.add_argument("--batch_size", type=int, help="Batch size")
    eval_parser.add_argument("--config", help="Path to config file")
    eval_parser.add_argument("--cpu", action="store_true", help="Use CPU only")

    # Cross-dataset mode
    cross_parser = subparsers.add_parser(
        "cross-dataset", help="Cross-dataset evaluation"
    )
    cross_parser.add_argument("--config", help="Path to config file")
    cross_parser.add_argument("--model", help="Model name")
    cross_parser.add_argument("--train_dataset", help="Training dataset")
    cross_parser.add_argument("--test_datasets", nargs="+", help="Test datasets")
    cross_parser.add_argument("--cpu", action="store_true", help="Use CPU only")

    # Inference mode
    inference_parser = subparsers.add_parser(
        "inference", help="Run inference on single image"
    )
    inference_parser.add_argument(
        "--model_path", required=True, help="Path to trained model"
    )
    inference_parser.add_argument("--image_path", required=True, help="Path to image")
    inference_parser.add_argument("--model", help="Model architecture name")
    inference_parser.add_argument("--cpu", action="store_true", help="Use CPU only")
    inference_parser.add_argument(
        "--explainability",
        action="store_true",
        help="Generate explainability visualizations",
    )

    args = parser.parse_args()

    if not args.mode:
        parser.print_help()
        sys.exit(1)

    # Route to appropriate function
    if args.mode == "train":
        train_model(args)
    elif args.mode == "evaluate":
        evaluate_model(args)
    elif args.mode == "cross-dataset":
        cross_dataset_evaluation(args)
    elif args.mode == "inference":
        inference_single(args)


if __name__ == "__main__":
    main()
