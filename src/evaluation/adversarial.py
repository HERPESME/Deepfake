"""
Adversarial robustness testing for deepfake detection models.
Implements FGSM and PGD attacks to evaluate model robustness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def fgsm_attack(model: nn.Module, images: torch.Tensor, labels: torch.Tensor,
                epsilon: float, device: str = "cpu") -> torch.Tensor:
    """
    Fast Gradient Sign Method (FGSM) attack.
    Single-step adversarial perturbation along the gradient direction.
    
    Args:
        model: Target model
        images: Input images (B, C, H, W)
        labels: True labels (B,)
        epsilon: Perturbation magnitude
        device: Device
        
    Returns:
        Adversarial images (B, C, H, W)
    """
    images = images.clone().detach().to(device).requires_grad_(True)
    labels = labels.to(device)
    
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()
    
    # Create perturbation
    perturbed = images + epsilon * images.grad.sign()
    # Clamp to valid image range
    perturbed = torch.clamp(perturbed, 0, 1)
    
    return perturbed.detach()


def pgd_attack(model: nn.Module, images: torch.Tensor, labels: torch.Tensor,
               epsilon: float, alpha: float = None, num_steps: int = 10,
               device: str = "cpu") -> torch.Tensor:
    """
    Projected Gradient Descent (PGD) attack.
    Multi-step iterative adversarial perturbation.
    
    Args:
        model: Target model
        images: Input images (B, C, H, W)
        labels: True labels (B,)
        epsilon: Maximum perturbation magnitude
        alpha: Step size (default: epsilon/4)
        num_steps: Number of PGD steps
        device: Device
        
    Returns:
        Adversarial images (B, C, H, W)
    """
    if alpha is None:
        alpha = epsilon / 4.0
    
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    
    # Start from random point within epsilon ball
    perturbed = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
    perturbed = torch.clamp(perturbed, 0, 1).detach()
    
    for _ in range(num_steps):
        perturbed.requires_grad_(True)
        outputs = model(perturbed)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        
        # Step in gradient direction
        adv_images = perturbed + alpha * perturbed.grad.sign()
        # Project back to epsilon ball around original
        delta = torch.clamp(adv_images - images, -epsilon, epsilon)
        perturbed = torch.clamp(images + delta, 0, 1).detach()
    
    return perturbed


def evaluate_adversarial_robustness(
    model: nn.Module,
    dataloader,
    epsilons: List[float] = [0.0, 0.01, 0.02, 0.05, 0.1],
    attacks: List[str] = ["fgsm", "pgd"],
    device: str = "cpu",
    max_batches: int = None,
) -> Dict:
    """
    Evaluate model robustness against adversarial attacks at multiple epsilon values.
    
    Args:
        model: Target model in eval mode  
        dataloader: Test dataloader
        epsilons: List of perturbation magnitudes
        attacks: List of attack types
        device: Device
        max_batches: Limit number of batches (for speed)
        
    Returns:
        Dictionary with results for each attack/epsilon combination
    """
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    model.eval()
    results = {}
    
    for attack_name in attacks:
        results[attack_name] = {}
        
        for eps in epsilons:
            all_preds = []
            all_labels = []
            all_probs = []
            
            for batch_idx, (images, labels) in enumerate(dataloader):
                if max_batches and batch_idx >= max_batches:
                    break
                
                images = images.to(device)
                labels_dev = labels.to(device)
                
                if eps == 0.0:
                    # Clean evaluation
                    with torch.no_grad():
                        outputs = model(images)
                else:
                    # Generate adversarial examples
                    model.eval()  # Keep in eval mode for BN/dropout
                    
                    if attack_name == "fgsm":
                        adv_images = fgsm_attack(model, images, labels_dev, eps, device)
                    elif attack_name == "pgd":
                        adv_images = pgd_attack(model, images, labels_dev, eps, device=device)
                    else:
                        raise ValueError(f"Unknown attack: {attack_name}")
                    
                    with torch.no_grad():
                        outputs = model(adv_images)
                
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
            
            y_true = np.array(all_labels)
            y_pred = np.array(all_preds)
            y_prob = np.array(all_probs)
            
            acc = float(accuracy_score(y_true, y_pred))
            try:
                auc = float(roc_auc_score(y_true, y_prob))
            except ValueError:
                auc = 0.0
            
            results[attack_name][str(eps)] = {
                "accuracy": acc,
                "auc": auc,
                "num_samples": len(y_true),
            }
            
            logger.info(f"  {attack_name} ε={eps:.3f}: Acc={acc:.4f}, AUC={auc:.4f}")
    
    return results
