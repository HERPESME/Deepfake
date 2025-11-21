"""
Training pipeline for deepfake detection models.
Supports various training strategies including cross-dataset evaluation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from tqdm import tqdm
import json
import wandb
from datetime import datetime
import os

from ..data.dataloader import DataLoaderFactory, CrossDatasetLoader
from ..models.baseline_models import create_model as create_baseline_model
from ..models.advanced_models import create_advanced_model
from ..evaluation.metrics import MetricsCalculator
from torch.utils.data import DataLoader
from tqdm import tqdm


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score: float, model: nn.Module) -> bool:
        """
        Check if training should stop early.
        
        Args:
            val_score: Current validation score
            model: Model to potentially save weights
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            self.save_checkpoint(model)
        
        return False
    
    def save_checkpoint(self, model: nn.Module):
        """Save model checkpoint."""
        self.best_weights = model.state_dict().copy()


class LearningRateScheduler:
    """Learning rate scheduler utility."""
    
    @staticmethod
    def create_scheduler(optimizer: optim.Optimizer, 
                        scheduler_type: str = "cosine",
                        **kwargs) -> optim.lr_scheduler._LRScheduler:
        """
        Create learning rate scheduler.
        
        Args:
            optimizer: Optimizer
            scheduler_type: Type of scheduler
            **kwargs: Scheduler-specific arguments
            
        Returns:
            Learning rate scheduler
        """
        if scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=kwargs.get("T_max", 100)
            )
        elif scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                optimizer, step_size=kwargs.get("step_size", 30), 
                gamma=kwargs.get("gamma", 0.1)
            )
        elif scheduler_type == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", patience=kwargs.get("patience", 5),
                factor=kwargs.get("factor", 0.5)
            )
        elif scheduler_type == "warmup_cosine":
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=kwargs.get("T_0", 10), T_mult=kwargs.get("T_mult", 2)
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


class DeepfakeTrainer:
    """Main trainer class for deepfake detection models."""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: Optional[DataLoader] = None,
                 device: str = "cuda",
                 experiment_name: Optional[str] = None,
                 use_wandb: bool = False,
                 wandb_project: str = "deepfake-detection",
                 mixed_precision: bool = False):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader (optional)
            device: Device to use for training
            experiment_name: Name of the experiment
            use_wandb: Whether to use Weights & Biases logging
            wandb_project: W&B project name
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.mixed_precision = mixed_precision
        
        # Initialize mixed precision scaler if enabled
        if mixed_precision and device in ["cuda", "mps"]:
            self.scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
            self.logger = logging.getLogger(__name__)
            self.logger.info("Mixed precision training enabled (30-50% speedup expected)")
        else:
            self.scaler = None
            if mixed_precision:
                self.logger = logging.getLogger(__name__)
                self.logger.warning("Mixed precision requested but not available on CPU. Disabling.")
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator()
        
        # Setup logging
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(__name__)
        
        # Setup W&B
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                project=wandb_project,
                name=self.experiment_name,
                config={
                    "model": model.__class__.__name__,
                    "device": device,
                    "train_samples": len(train_loader.dataset),
                    "val_samples": len(val_loader.dataset),
                }
            )
        
        # Create output directory
        self.output_dir = Path("experiments") / self.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_val_score = 0.0
        self.train_history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_auc": []
        }
    
    def train(self,
              epochs: int = 100,
              learning_rate: float = 1e-4,
              weight_decay: float = 1e-4,
              optimizer_type: str = "adam",
              scheduler_type: str = "cosine",
              early_stopping_patience: int = 10,
              save_frequency: int = 5,
              **kwargs) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate
            weight_decay: Weight decay
            optimizer_type: Type of optimizer
            scheduler_type: Type of learning rate scheduler
            early_stopping_patience: Patience for early stopping
            save_frequency: Frequency to save checkpoints
            **kwargs: Additional training arguments
            
        Returns:
            Training history and results
        """
        # Setup optimizer
        optimizer = self._create_optimizer(optimizer_type, learning_rate, weight_decay)
        
        # Setup scheduler
        scheduler = LearningRateScheduler.create_scheduler(
            optimizer, scheduler_type, **kwargs
        )
        
        # Setup early stopping
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        self.logger.info(f"Starting training for {epochs} epochs")
        self.logger.info(f"Model: {self.model.__class__.__name__}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Optimizer: {optimizer_type}, LR: {learning_rate}")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self._train_epoch(optimizer, criterion)
            
            # Validation phase
            val_metrics = self._validate_epoch(criterion)
            
            # Update scheduler
            if scheduler_type == "plateau":
                scheduler.step(val_metrics["auc"])
            else:
                scheduler.step()
            
            # Log metrics
            self._log_metrics(train_metrics, val_metrics, epoch)
            
            # Update history
            self.train_history["train_loss"].append(train_metrics["loss"])
            self.train_history["train_acc"].append(train_metrics["accuracy"])
            self.train_history["val_loss"].append(val_metrics["loss"])
            self.train_history["val_acc"].append(val_metrics["accuracy"])
            self.train_history["val_auc"].append(val_metrics["auc"])
            
            # Save checkpoint
            if epoch % save_frequency == 0:
                self._save_checkpoint(epoch, optimizer, scheduler)
            
            # Early stopping
            if early_stopping(val_metrics["auc"], self.model):
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Final evaluation
        final_results = self._final_evaluation()
        
        # Save final model and results
        self._save_final_results(final_results)
        
        return {
            "history": self.train_history,
            "final_results": final_results,
            "best_val_score": self.best_val_score
        }
    
    def _create_optimizer(self, optimizer_type: str, lr: float, weight_decay: float) -> optim.Optimizer:
        """Create optimizer."""
        if optimizer_type.lower() == "adam":
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == "adamw":
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == "sgd":
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
    def _train_epoch(self, optimizer: optim.Optimizer, criterion: nn.Module) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} - Training")
        
        # Use autocast for mixed precision if enabled
        use_amp = self.mixed_precision and self.device in ["cuda", "mps"]
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision if enabled
            if use_amp and self.device == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            elif use_amp and self.device == "mps":
                # MPS doesn't support autocast, but we can still use float16
                with torch.autocast(device_type="cpu", dtype=torch.float16):
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            else:
                # Standard training without mixed precision
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{np.mean(np.array(all_predictions) == np.array(all_labels)):.4f}'
            })
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(
            np.array(all_labels), np.array(all_predictions)
        )
        metrics["loss"] = total_loss / len(self.train_loader)
        
        return metrics
    
    def _validate_epoch(self, criterion: nn.Module) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        # Use autocast for mixed precision if enabled
        use_amp = self.mixed_precision and self.device in ["cuda", "mps"]
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch} - Validation")
            
            for images, labels in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass with mixed precision if enabled
                if use_amp and self.device == "cuda":
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = criterion(outputs, labels)
                elif use_amp and self.device == "mps":
                    with torch.autocast(device_type="cpu", dtype=torch.float16):
                        outputs = self.model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                
                # Update metrics
                total_loss += loss.item()
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Fake class probability
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{np.mean(np.array(all_predictions) == np.array(all_labels)):.4f}'
                })
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(
            np.array(all_labels), np.array(all_predictions), np.array(all_probabilities)
        )
        metrics["loss"] = total_loss / len(self.val_loader)
        
        # Update best score
        if metrics["auc"] > self.best_val_score:
            self.best_val_score = metrics["auc"]
        
        return metrics
    
    def _log_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float], epoch: int):
        """Log metrics to console and W&B."""
        self.logger.info(
            f"Epoch {epoch}: "
            f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
            f"Val AUC: {val_metrics['auc']:.4f}"
        )
        
        if self.use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_auc": val_metrics["auc"],
                "learning_rate": self._get_current_lr()
            })
    
    def _get_current_lr(self) -> float:
        """Get current learning rate."""
        for param_group in self.model.parameters():
            if hasattr(param_group, 'lr'):
                return param_group['lr']
        return 0.0
    
    def _save_checkpoint(self, epoch: int, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_score': self.best_val_score,
            'train_history': self.train_history
        }
        
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
    
    def _final_evaluation(self) -> Dict[str, Any]:
        """Perform final evaluation on test set."""
        if self.test_loader is None:
            return {}
        
        self.logger.info("Performing final evaluation on test set")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Final Evaluation"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
        
        # Calculate final metrics
        final_metrics = self.metrics_calculator.calculate_metrics(
            np.array(all_labels), np.array(all_predictions), np.array(all_probabilities)
        )
        
        return final_metrics
    
    def _save_final_results(self, final_results: Dict[str, Any]):
        """Save final results and model."""
        # Save final model
        final_model_path = self.output_dir / "final_model.pth"
        torch.save(self.model.state_dict(), final_model_path)
        
        # Save training history
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        
        # Save final results
        if final_results:
            results_path = self.output_dir / "final_results.json"
            with open(results_path, 'w') as f:
                json.dump(final_results, f, indent=2)
        
        self.logger.info(f"Results saved to {self.output_dir}")


class CrossDatasetTrainer:
    """Trainer for cross-dataset evaluation."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.results = {}
    
    def train_and_evaluate(self,
                          model_name: str,
                          train_dataset: str,
                          test_datasets: List[str],
                          data_root: str,
                          **training_kwargs) -> Dict[str, Any]:
        """
        Train on one dataset and evaluate on multiple test datasets.
        
        Args:
            model_name: Name of the model
            train_dataset: Training dataset name
            test_datasets: List of test dataset names
            data_root: Root directory containing datasets
            **training_kwargs: Training arguments
            
        Returns:
            Cross-dataset evaluation results
        """
        results = {}
        
        # Create model
        if model_name in ["xception", "efficientnet_b0", "efficientnet_b4", "vit", "resnet50"]:
            model = create_baseline_model(model_name, num_classes=2)
        else:
            model = create_advanced_model(model_name, num_classes=2)
        
        # Create training data loader
        train_data_dir = Path(data_root) / train_dataset / "processed"
        train_loader, val_loader, _ = DataLoaderFactory.create_dataloaders(
            str(train_data_dir), batch_size=training_kwargs.get("batch_size", 32)
        )
        
        # Train model
        trainer = DeepfakeTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=None,
            device=self.device,
            experiment_name=f"{model_name}_{train_dataset}_cross_dataset",
            use_wandb=False
        )
        
        training_results = trainer.train(**training_kwargs)
        results["training"] = training_results
        
        # Evaluate on test datasets
        test_results = {}
        for test_dataset in test_datasets:
            test_data_dir = Path(data_root) / test_dataset / "processed"
            _, _, test_loader = DataLoaderFactory.create_dataloaders(
                str(test_data_dir), batch_size=training_kwargs.get("batch_size", 32)
            )
            
            # Evaluate trained model
            test_metrics = self._evaluate_model(model, test_loader, test_dataset)
            test_results[test_dataset] = test_metrics
        
        results["cross_dataset_evaluation"] = test_results
        return results
    
    def _evaluate_model(self, model: nn.Module, test_loader: DataLoader, dataset_name: str) -> Dict[str, float]:
        """Evaluate model on test dataset."""
        model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        metrics_calculator = MetricsCalculator()
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Evaluating on {dataset_name}"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
        
        metrics = metrics_calculator.calculate_metrics(
            np.array(all_labels), np.array(all_predictions), np.array(all_probabilities)
        )
        
        return metrics


def main():
    """Test training pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train deepfake detection model")
    parser.add_argument("--model", default="xception", help="Model name")
    parser.add_argument("--dataset", default="faceforensics", help="Dataset name")
    parser.add_argument("--data_root", required=True, help="Root directory of processed data")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    
    args = parser.parse_args()
    
    # Create data loaders
    data_dir = Path(args.data_root) / args.dataset / "processed"
    train_loader, val_loader, test_loader = DataLoaderFactory.create_dataloaders(
        str(data_dir), batch_size=args.batch_size
    )
    
    # Create model
    if args.model in ["xception", "efficientnet_b0", "efficientnet_b4", "vit", "resnet50"]:
        model = create_baseline_model(args.model, num_classes=2)
    else:
        model = create_advanced_model(args.model, num_classes=2)
    
    # Create trainer
    trainer = DeepfakeTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=args.device,
        use_wandb=args.use_wandb
    )
    
    # Train model
    results = trainer.train(
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size
    )
    
    print("Training completed!")
    print(f"Best validation AUC: {results['best_val_score']:.4f}")


if __name__ == "__main__":
    main()
