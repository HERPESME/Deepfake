"""
Ensemble models for deepfake detection.
Combines multiple pre-trained models using various fusion strategies:
- Hard Voting (majority vote)
- Soft Voting (weighted probability averaging)  
- Learned Stacking (meta-classifier on model outputs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import logging
import json
import pickle

logger = logging.getLogger(__name__)


def load_model_from_path(model_name: str, model_path: str, device: str = "cpu"):
    """
    Load a trained model from checkpoint or final_model.pth.
    
    Args:
        model_name: Architecture name (e.g., 'efficientnet_b0', 'vit', 'xception')
        model_path: Path to .pth file
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    from src.models.baseline_models import create_model as create_baseline_model
    from src.models.advanced_models import create_advanced_model
    
    # Create model architecture
    baseline_models = [
        "xception", "efficientnet_b0", "efficientnet_b4", "vit",
        "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
    ]
    
    if model_name in baseline_models:
        if model_name == "vit":
            model = create_baseline_model(model_name, num_classes=2, pretrained=False, dropout=0.1)
        else:
            model = create_baseline_model(model_name, num_classes=2, pretrained=False)
    else:
        model = create_advanced_model(model_name, num_classes=2, pretrained=False)
    
    # Load weights
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    logger.info(f"Loaded {model_name} from {model_path}")
    return model


class EnsembleDetector:
    """
    Ensemble deepfake detector combining multiple pre-trained models.
    Supports hard voting, soft voting, and learned stacking.
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.models: Dict[str, nn.Module] = {}
        self.model_weights: Dict[str, float] = {}
        self.stacking_meta_model = None
        self.stacking_scaler = None
    
    def add_model(self, name: str, model: nn.Module, weight: float = 1.0):
        """Add a pre-trained model to the ensemble."""
        self.models[name] = model
        self.model_weights[name] = weight
        logger.info(f"Added model '{name}' with weight {weight}")
    
    def load_and_add_model(self, name: str, arch_name: str, model_path: str, weight: float = 1.0):
        """Load a model from path and add to ensemble."""
        model = load_model_from_path(arch_name, model_path, self.device)
        self.add_model(name, model, weight)
    
    def _get_all_outputs(self, images: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get predictions from all models.
        
        Returns:
            Dict mapping model name -> (probabilities, predictions)
        """
        outputs = {}
        with torch.no_grad():
            for name, model in self.models.items():
                logits = model(images)
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                outputs[name] = (probs, preds)
        return outputs
    
    def hard_vote(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hard voting: majority vote across all models.
        
        Returns:
            (predictions, confidence) where confidence is the vote fraction
        """
        outputs = self._get_all_outputs(images)
        batch_size = images.shape[0]
        
        # Stack all predictions: (num_models, batch_size)
        all_preds = torch.stack([outputs[name][1] for name in self.models], dim=0)
        
        # Majority vote
        votes_fake = all_preds.sum(dim=0).float()  # Count of "fake" votes
        num_models = len(self.models)
        predictions = (votes_fake > num_models / 2).long()
        confidence = votes_fake / num_models  # Fraction voting "fake"
        
        return predictions, confidence
    
    def soft_vote(self, images: torch.Tensor, use_weights: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Soft voting: (weighted) average of probability outputs.
        
        Returns:
            (predictions, fake_probability)
        """
        outputs = self._get_all_outputs(images)
        
        # Weighted average of probabilities
        weighted_probs = torch.zeros_like(outputs[list(self.models.keys())[0]][0])
        total_weight = 0.0
        
        for name in self.models:
            w = self.model_weights[name] if use_weights else 1.0
            weighted_probs += w * outputs[name][0]
            total_weight += w
        
        weighted_probs /= total_weight
        predictions = torch.argmax(weighted_probs, dim=1)
        fake_probs = weighted_probs[:, 1]
        
        return predictions, fake_probs
    
    def train_stacking(self, dataloader, labels_list=None):
        """
        Train a stacking meta-classifier on model outputs.
        Uses logistic regression on concatenated model probabilities.
        
        Args:
            dataloader: DataLoader with (images, labels)
        """
        logger.info("Training stacking meta-classifier...")
        
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                
                # Get probabilities from each model
                batch_features = []
                for name, model in self.models.items():
                    logits = model(images)
                    probs = F.softmax(logits, dim=1)
                    batch_features.append(probs[:, 1].cpu().numpy())  # fake probability
                
                # Stack: (batch_size, num_models)
                batch_features = np.stack(batch_features, axis=1)
                all_features.append(batch_features)
                all_labels.append(labels.numpy())
        
        X = np.concatenate(all_features, axis=0)
        y = np.concatenate(all_labels, axis=0)
        
        logger.info(f"Stacking features shape: {X.shape}, labels shape: {y.shape}")
        
        # Normalize features
        self.stacking_scaler = StandardScaler()
        X_scaled = self.stacking_scaler.fit_transform(X)
        
        # Train logistic regression meta-classifier
        self.stacking_meta_model = LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        )
        self.stacking_meta_model.fit(X_scaled, y)
        
        train_acc = self.stacking_meta_model.score(X_scaled, y)
        logger.info(f"Stacking meta-classifier training accuracy: {train_acc:.4f}")
    
    def stacked_predict(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stacking prediction using trained meta-classifier.
        
        Returns:
            (predictions, fake_probability)
        """
        if self.stacking_meta_model is None:
            raise RuntimeError("Stacking meta-classifier not trained. Call train_stacking() first.")
        
        with torch.no_grad():
            batch_features = []
            for name, model in self.models.items():
                logits = model(images)
                probs = F.softmax(logits, dim=1)
                batch_features.append(probs[:, 1].cpu().numpy())
            
            X = np.stack(batch_features, axis=1)
            X_scaled = self.stacking_scaler.transform(X)
            
            predictions = self.stacking_meta_model.predict(X_scaled)
            fake_probs = self.stacking_meta_model.predict_proba(X_scaled)[:, 1]
        
        return (
            torch.tensor(predictions, dtype=torch.long),
            torch.tensor(fake_probs, dtype=torch.float)
        )
    
    def evaluate_on_dataloader(self, dataloader, method: str = "soft_vote") -> Dict:
        """
        Evaluate ensemble on a dataloader using the specified method.
        
        Args:
            dataloader: DataLoader with (images, labels)
            method: 'hard_vote', 'soft_vote', or 'stacked'
            
        Returns:
            Dictionary with evaluation metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score, confusion_matrix
        )
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        for images, labels in dataloader:
            images = images.to(self.device)
            
            if method == "hard_vote":
                preds, probs = self.hard_vote(images)
            elif method == "soft_vote":
                preds, probs = self.soft_vote(images)
            elif method == "stacked":
                preds, probs = self.stacked_predict(images)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
        
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)
        
        results = {
            "method": method,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="weighted")),
            "recall": float(recall_score(y_true, y_pred, average="weighted")),
            "f1_score": float(f1_score(y_true, y_pred, average="weighted")),
            "auc": float(roc_auc_score(y_true, y_prob)),
            "average_precision": float(average_precision_score(y_true, y_prob)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "num_models": len(self.models),
            "model_names": list(self.models.keys()),
        }
        
        return results
    
    def save_stacking_model(self, path: str):
        """Save the stacking meta-classifier to disk."""
        save_data = {
            "meta_model": self.stacking_meta_model,
            "scaler": self.stacking_scaler,
            "model_weights": self.model_weights,
        }
        with open(path, "wb") as f:
            pickle.dump(save_data, f)
        logger.info(f"Stacking model saved to {path}")
    
    def load_stacking_model(self, path: str):
        """Load a stacking meta-classifier from disk."""
        with open(path, "rb") as f:
            save_data = pickle.load(f)
        self.stacking_meta_model = save_data["meta_model"]
        self.stacking_scaler = save_data["scaler"]
        self.model_weights = save_data.get("model_weights", self.model_weights)
        logger.info(f"Stacking model loaded from {path}")
