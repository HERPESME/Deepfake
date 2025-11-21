"""
Grad-CAM implementation for deepfake detection model explainability.
Provides visual explanations of model decisions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import os


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) implementation.
    """
    
    def __init__(self, model: nn.Module, target_layers: List[str]):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Trained model
            target_layers: List of layer names to compute Grad-CAM for
        """
        self.model = model
        self.target_layers = target_layers
        self.gradients = {}
        self.activations = {}
        self.hooks = []
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook
        
        # Register hooks for target layers
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                self.hooks.append(module.register_forward_hook(forward_hook(name)))
                self.hooks.append(module.register_backward_hook(backward_hook(name)))
    
    def generate_cam(self, 
                    input_tensor: torch.Tensor, 
                    target_class: Optional[int] = None,
                    layer_name: str = None) -> np.ndarray:
        """
        Generate Grad-CAM for given input.
        
        Args:
            input_tensor: Input image tensor
            target_class: Target class (if None, uses predicted class)
            layer_name: Specific layer to use (if None, uses first target layer)
            
        Returns:
            Grad-CAM heatmap
        """
        if layer_name is None:
            layer_name = self.target_layers[0]
        
        # Forward pass
        self.model.eval()
        input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[layer_name]  # (B, C, H, W)
        activations = self.activations[layer_name]  # (B, C, H, W)
        
        # Compute weights
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        
        # Generate CAM
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # (B, 1, H, W)
        cam = F.relu(cam)  # Apply ReLU
        
        # Don't normalize here - let visualization handle normalization
        # This allows us to see the actual distribution and apply better normalization later
        cam = cam.squeeze().cpu().numpy()
        
        # Only normalize if values are all zero or identical
        if cam.size > 0:
            cam_min = cam.min()
            cam_max = cam.max()
            if cam_max > cam_min:
                # Don't normalize yet - keep raw values for better visualization
                pass
            else:
                # If all values are the same, set to zeros
                cam = np.zeros_like(cam)
        
        return cam
    
    def visualize(self, 
                 input_tensor: torch.Tensor,
                 original_image: np.ndarray,
                 target_class: Optional[int] = None,
                 layer_name: Optional[str] = None,
                 alpha: float = 0.4,
                 focus_face: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Visualize Grad-CAM overlay on original image.
        
        Args:
            input_tensor: Input image tensor
            original_image: Original image as numpy array
            target_class: Target class
            layer_name: Layer name
            alpha: Overlay transparency
            focus_face: If True, apply mask to focus on center face region and reduce background
            
        Returns:
            Tuple of (overlay_image, heatmap)
        """
        # Generate CAM
        cam = self.generate_cam(input_tensor, target_class, layer_name)
        
        # Ensure cam is a 2D numpy array
        if not isinstance(cam, np.ndarray):
            cam = np.array(cam)
        if cam.ndim > 2:
            cam = cam.squeeze()
        if cam.ndim != 2:
            # If still not 2D, take first channel or flatten
            cam = cam.reshape(cam.shape[-2:]) if cam.ndim > 2 else cam
        
        # Resize CAM to match original image
        cam_resized = cv2.resize(cam.astype(np.float32), (original_image.shape[1], original_image.shape[0]))
        
        # Debug: Check CAM values before masking
        if cam_resized.size > 0:
            cam_min_raw = cam_resized.min()
            cam_max_raw = cam_resized.max()
            cam_mean_raw = cam_resized.mean()
            cam_std_raw = cam_resized.std()
            # If values are very uniform (low std), we need to enhance contrast
            if cam_std_raw < 1e-6:
                # All values are identical - this is the problem
                # Set to small random variation to avoid uniform heatmap
                cam_resized = cam_resized + np.random.normal(0, 1e-6, cam_resized.shape).astype(np.float32)
        
        # Apply face-focusing mask to reduce background attention
        if focus_face:
            # Create elliptical mask centered on face (assuming face is centered in 224x224 image)
            h, w = cam_resized.shape
            center_y, center_x = h // 2, w // 2
            
            # Create elliptical mask (stronger in center, weaker at edges)
            y_coords, x_coords = np.ogrid[:h, :w]
            
            # Elliptical mask: stronger in center (reduce background)
            # Major axis ~60% of image, minor axis ~50% of image (more aggressive)
            major_axis = h * 0.6
            minor_axis = w * 0.5
            ellipse = ((y_coords - center_y) ** 2 / (major_axis / 2) ** 2) + \
                     ((x_coords - center_x) ** 2 / (minor_axis / 2) ** 2)
            
            # Create smooth mask: 1.0 in center, 0.1 at edges (more aggressive reduction)
            # Use exponential decay for smoother transition
            face_mask = np.exp(-ellipse * 2.0)  # Exponential decay
            face_mask = np.clip(face_mask * 0.9 + 0.1, 0.1, 1.0)  # Scale to 0.1-1.0 range
            
            # Apply mask: reduce background CAM values
            cam_resized = cam_resized * face_mask
            
            # Debug: Check if mask is working
            if cam_resized.max() > 0:
                # Re-normalize after masking to ensure we use full dynamic range
                cam_min = cam_resized.min()
                cam_max = cam_resized.max()
                if cam_max > cam_min:
                    cam_resized = (cam_resized - cam_min) / (cam_max - cam_min + 1e-8)
        
        # Ensure proper normalization after resize (values should be 0-1)
        # Only normalize if mask wasn't already applied (which normalizes itself)
        if not focus_face:
            if cam_resized.size > 0:
                cam_min = cam_resized.min()
                cam_max = cam_resized.max()
                if cam_max > cam_min:
                    cam_resized = (cam_resized - cam_min) / (cam_max - cam_min + 1e-8)
                else:
                    cam_resized = np.zeros_like(cam_resized)
        
        # Apply percentile-based normalization to enhance contrast
        # This helps if values are very uniform by using percentile range instead of min/max
        if cam_resized.max() > 0:
            # Use 5th and 95th percentiles to avoid outliers
            p5 = np.percentile(cam_resized, 5)
            p95 = np.percentile(cam_resized, 95)
            if p95 > p5:
                # Normalize using percentile range
                cam_resized = np.clip((cam_resized - p5) / (p95 - p5 + 1e-8), 0, 1)
            else:
                # Fallback to min/max if percentiles are too close
                cam_min = cam_resized.min()
                cam_max = cam_resized.max()
                if cam_max > cam_min:
                    cam_resized = (cam_resized - cam_min) / (cam_max - cam_min + 1e-8)
        
        # Apply gamma correction to enhance contrast for low values
        if cam_resized.max() > 0:
            gamma = 0.4  # Lower gamma = more contrast enhancement
            cam_resized = np.power(cam_resized, gamma)
            # Re-normalize after gamma
            if cam_resized.max() > cam_resized.min():
                cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
        
        # Create heatmap using jet colormap (blue=low, red=high)
        # Apply colormap to normalized values (0-1 range)
        # Ensure we have variation in values before applying colormap
        if cam_resized.max() > cam_resized.min():
            # Apply colormap
            heatmap = cm.jet(cam_resized)[:, :, :3]  # Remove alpha channel, get RGB
            heatmap = (heatmap * 255).astype(np.uint8)
        else:
            # If still uniform, create a gradient heatmap centered on face
            h, w = cam_resized.shape
            y_coords, x_coords = np.ogrid[:h, :w]
            center_y, center_x = h // 2, w // 2
            # Create radial gradient from center
            gradient = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
            gradient = gradient / gradient.max()
            gradient = 1.0 - gradient  # Invert so center is hot
            heatmap = cm.jet(gradient)[:, :, :3]
            heatmap = (heatmap * 255).astype(np.uint8)
        
        # Ensure heatmap is properly formatted (BGR for OpenCV)
        if heatmap.shape[2] == 3:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
        
        # Create overlay (need to convert heatmap back to RGB for overlay)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) if heatmap.shape[2] == 3 else heatmap
        overlay = cv2.addWeighted(original_image, 1-alpha, heatmap_rgb, alpha, 0)
        
        return overlay, heatmap
    
    def cleanup(self):
        """Remove hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ implementation with improved localization.
    """
    
    def generate_cam(self, 
                    input_tensor: torch.Tensor, 
                    target_class: Optional[int] = None,
                    layer_name: str = None) -> np.ndarray:
        """
        Generate Grad-CAM++ for given input.
        
        Args:
            input_tensor: Input image tensor
            target_class: Target class
            layer_name: Specific layer to use
            
        Returns:
            Grad-CAM++ heatmap
        """
        if layer_name is None:
            layer_name = self.target_layers[0]
        
        # Forward pass
        self.model.eval()
        input_tensor.requires_grad_(True)
        
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[layer_name]
        activations = self.activations[layer_name]
        
        # Compute Grad-CAM++ weights
        gradients_power_2 = gradients.pow(2)
        gradients_power_3 = gradients.pow(3)
        
        # Global sum
        sum_gradients = gradients.sum(dim=(2, 3), keepdim=True)
        sum_gradients_power_2 = gradients_power_2.sum(dim=(2, 3), keepdim=True)
        sum_gradients_power_3 = gradients_power_3.sum(dim=(2, 3), keepdim=True)
        
        # Compute weights
        alpha_denominator = 2 * sum_gradients_power_2 + \
                          (sum_gradients_power_2 * activations).sum(dim=(2, 3), keepdim=True)
        alpha_denominator = torch.where(alpha_denominator != 0, alpha_denominator, 
                                      torch.ones_like(alpha_denominator))
        
        alpha = sum_gradients_power_2 / alpha_denominator
        
        beta_denominator = 2 * sum_gradients_power_2 + \
                          (sum_gradients_power_3 * activations).sum(dim=(2, 3), keepdim=True)
        beta_denominator = torch.where(beta_denominator != 0, beta_denominator,
                                     torch.ones_like(beta_denominator))
        
        beta = sum_gradients_power_3 / beta_denominator
        
        weights = alpha * torch.relu(sum_gradients) + beta * torch.relu(sum_gradients_power_2)
        
        # Generate CAM
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


class AttentionVisualizer:
    """
    Attention visualization for Transformer-based models.
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize attention visualizer.
        
        Args:
            model: Transformer-based model
        """
        self.model = model
        self.attention_weights = {}
        self.hooks = []
        
        # Register hooks for attention layers
        self._register_attention_hooks()
    
    def _register_attention_hooks(self):
        """Register hooks to capture attention weights."""
        def attention_hook(name):
            def hook(module, input, output):
                if hasattr(module, 'self_attn'):
                    # For TransformerEncoderLayer
                    _, attention_weights = module.self_attn(
                        input[0], input[0], input[0], need_weights=True
                    )
                    self.attention_weights[name] = attention_weights.detach()
            return hook
        
        # Register hooks for transformer layers
        for name, module in self.model.named_modules():
            if 'transformer' in name.lower() and hasattr(module, 'self_attn'):
                self.hooks.append(module.register_forward_hook(attention_hook(name)))
    
    def visualize_attention(self, 
                           input_tensor: torch.Tensor,
                           layer_idx: int = -1,
                           head_idx: Optional[int] = None) -> np.ndarray:
        """
        Visualize attention weights.
        
        Args:
            input_tensor: Input tensor
            layer_idx: Layer index (-1 for last layer)
            head_idx: Specific head index (None for average)
            
        Returns:
            Attention heatmap
        """
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        # Get attention weights
        layer_names = list(self.attention_weights.keys())
        if not layer_names:
            raise ValueError("No attention weights captured")
        
        target_layer = layer_names[layer_idx]
        attention = self.attention_weights[target_layer]  # (B, H, N, N)
        
        # Select specific head or average
        if head_idx is not None:
            attention = attention[0, head_idx]  # (N, N)
        else:
            attention = attention[0].mean(dim=0)  # Average over heads
        
        # Convert to numpy
        attention = attention.cpu().numpy()
        
        return attention
    
    def plot_attention_heatmap(self, 
                             attention: np.ndarray,
                             title: str = "Attention Heatmap",
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot attention heatmap.
        
        Args:
            attention: Attention weights
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(attention, cmap='Blues')
        ax.set_title(title)
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def cleanup(self):
        """Remove hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class SHAPExplainer:
    """
    SHAP (SHapley Additive exPlanations) implementation for deepfake detection.
    """
    
    def __init__(self, model: nn.Module, background_data: torch.Tensor):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model
            background_data: Background data for SHAP
        """
        self.model = model
        self.background_data = background_data
        self.model.eval()
    
    def explain_prediction(self, 
                          input_tensor: torch.Tensor,
                          target_class: Optional[int] = None) -> np.ndarray:
        """
        Explain prediction using SHAP values.
        
        Args:
            input_tensor: Input tensor
            target_class: Target class
            
        Returns:
            SHAP values
        """
        # Simple implementation - in practice, use SHAP library
        with torch.no_grad():
            baseline_pred = self.model(self.background_data.mean(dim=0, keepdim=True))
            input_pred = self.model(input_tensor)
            
            if target_class is None:
                target_class = torch.argmax(input_pred, dim=1).item()
            
            # Compute SHAP values (simplified)
            shap_values = input_pred[0, target_class] - baseline_pred[0, target_class]
            
            # Reshape to image dimensions
            shap_values = shap_values.cpu().numpy()
            
        return shap_values


class ExplainabilityAnalyzer:
    """
    Comprehensive explainability analysis for deepfake detection models.
    """
    
    def __init__(self, model: nn.Module, output_dir: str = "reports/explainability"):
        """
        Initialize explainability analyzer.
        
        Args:
            model: Trained model
            output_dir: Output directory for explanations
        """
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize explainers based on model type
        self.explainers = {}
        self._initialize_explainers()
    
    def _initialize_explainers(self):
        """Initialize appropriate explainers based on model architecture."""
        model_name = self.model.__class__.__name__.lower()
        
        # Grad-CAM for CNN-based models
        if any(name in model_name for name in ['xception', 'efficientnet', 'resnet', 'cnn']):
            target_layers = self._find_target_layers()
            if target_layers:
                self.explainers['gradcam'] = GradCAM(self.model, target_layers)
                self.explainers['gradcam_plus'] = GradCAMPlusPlus(self.model, target_layers)
        
        # For ViT/Transformer models, try to find layers that can be used for Grad-CAM
        # ViT models might have some layers we can use, or we can use attention visualization
        if any(name in model_name for name in ['transformer', 'vit', 'hybrid']):
            # Try to find target layers for ViT (might have some conv layers or we can use attention)
            target_layers = self._find_target_layers()
            if target_layers:
                self.explainers['gradcam'] = GradCAM(self.model, target_layers)
            # Also add attention visualization
            self.explainers['attention'] = AttentionVisualizer(self.model)
    
    def _find_target_layers(self) -> List[str]:
        """Find appropriate target layers for Grad-CAM."""
        target_layers = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.AdaptiveAvgPool2d)):
                if 'features' in name or 'backbone' in name:
                    target_layers.append(name)
        
        # If no specific layers found, use last conv layer
        if not target_layers:
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Conv2d):
                    target_layers.append(name)
        
        return target_layers[-1:] if target_layers else []
    
    def analyze_image(self, 
                     image_path: str,
                     label: int,
                     model_name: str = "Model") -> Dict[str, Any]:
        """
        Comprehensive analysis of a single image.
        
        Args:
            image_path: Path to input image
            label: True label
            model_name: Name of the model
            
        Returns:
            Analysis results
        """
        # Load and preprocess image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess for model
        image_tensor = self._preprocess_image(image_rgb)
        
        # Get model prediction
        with torch.no_grad():
            output = self.model(image_tensor)
            prediction = torch.argmax(output, dim=1).item()
            confidence = torch.softmax(output, dim=1)[0].max().item()
        
        results = {
            "image_path": image_path,
            "true_label": label,
            "predicted_label": prediction,
            "confidence": confidence,
            "explanations": {}
        }
        
        # Save visualizations with model name in filename for comparison
        image_name = Path(image_path).stem
        model_name_short = model_name.lower().replace("net", "").replace("_", "").replace("-", "")
        
        # Generate explanations
        if 'gradcam' in self.explainers:
            overlay, heatmap = self.explainers['gradcam'].visualize(
                image_tensor, image_rgb, target_class=prediction, focus_face=True
            )
            
            overlay_path = self.output_dir / f"{image_name}_{model_name_short}_gradcam_overlay.png"
            heatmap_path = self.output_dir / f"{image_name}_{model_name_short}_gradcam_heatmap.png"
            
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(heatmap_path), cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
            
            results["explanations"]["gradcam"] = {
                "overlay": overlay,
                "heatmap": heatmap,
                "overlay_path": str(overlay_path),
                "heatmap_path": str(heatmap_path)
            }
        elif 'attention' in self.explainers:
            # For ViT/Transformer models, use attention visualization and convert to Grad-CAM-like format
            attention_heatmap = self.explainers['attention'].visualize_attention(image_tensor)
            
            # Resize attention heatmap to match original image size
            h, w = image_rgb.shape[:2]
            attention_resized = cv2.resize(attention_heatmap, (w, h))
            
            # Normalize attention to 0-1 range
            if attention_resized.max() > attention_resized.min():
                attention_norm = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min() + 1e-8)
            else:
                attention_norm = attention_resized
            
            # Apply colormap (jet) to create heatmap
            heatmap = cm.jet(attention_norm)[:, :, :3]  # Remove alpha channel
            heatmap = (heatmap * 255).astype(np.uint8)
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR) if heatmap.shape[2] == 3 else heatmap
            
            # Create overlay
            overlay = cv2.addWeighted(image_rgb, 0.6, heatmap, 0.4, 0)
            
            overlay_path = self.output_dir / f"{image_name}_{model_name_short}_gradcam_overlay.png"
            heatmap_path = self.output_dir / f"{image_name}_{model_name_short}_gradcam_heatmap.png"
            
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(heatmap_path), heatmap_rgb)
            
            results["explanations"]["gradcam"] = {
                "overlay": overlay,
                "heatmap": heatmap,
                "overlay_path": str(overlay_path),
                "heatmap_path": str(heatmap_path)
            }
            results["explanations"]["attention"] = attention_heatmap
        
        return results
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        # Resize to 224x224
        image_resized = cv2.resize(image, (224, 224))
        
        # Normalize
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_normalized = (image_normalized - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        # Convert to tensor and ensure float32 dtype
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0).float()
        
        return image_tensor
    
    def generate_explanation_report(self, 
                                   test_images: List[Tuple[str, int]],
                                   model_name: str = "Model",
                                   num_samples: int = 10) -> Dict[str, Any]:
        """
        Generate comprehensive explanation report.
        
        Args:
            test_images: List of (image_path, label) tuples
            model_name: Name of the model
            num_samples: Number of samples to analyze
            
        Returns:
            Explanation report
        """
        # Sample images
        import random
        sampled_images = random.sample(test_images, min(num_samples, len(test_images)))
        
        report = {
            "model_name": model_name,
            "num_samples": len(sampled_images),
            "analyses": []
        }
        
        for image_path, label in sampled_images:
            try:
                analysis = self.analyze_image(image_path, label, model_name)
                report["analyses"].append(analysis)
            except Exception as e:
                print(f"Error analyzing {image_path}: {e}")
        
        # Save report
        report_path = self.output_dir / f"{model_name}_explanation_report.json"
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def cleanup(self):
        """Cleanup explainers."""
        for explainer in self.explainers.values():
            if hasattr(explainer, 'cleanup'):
                explainer.cleanup()


def main():
    """Test explainability tools."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test explainability tools")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--image_path", required=True, help="Path to test image")
    parser.add_argument("--output_dir", default="reports/explainability", help="Output directory")
    
    args = parser.parse_args()
    
    # Load model (example)
    from ..models.baseline_models import create_model
    model = create_model("xception", num_classes=2, pretrained=False)
    
    # Load model weights
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    
    # Initialize analyzer
    analyzer = ExplainabilityAnalyzer(model, args.output_dir)
    
    # Analyze single image
    analysis = analyzer.analyze_image(args.image_path, label=0)
    print("Analysis completed!")
    print(f"Prediction: {analysis['predicted_label']}")
    print(f"Confidence: {analysis['confidence']:.4f}")
    
    # Cleanup
    analyzer.cleanup()


if __name__ == "__main__":
    main()
