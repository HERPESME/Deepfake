"""
Advanced models for deepfake detection.

Active Research Models (Used in Bridging the Gap Paper):
- Hybrid CNN-Transformer: Combining EfficientNet features with Transformer attention.
- Multi-Scale Feature Fusion: Aggregating features across multiple semantic granularities.

Experimental Models (Future Work - Not currently in paper):
- CLIP-based Detector: Leveraging OpenAI's CLIP for semantic visual processing.
- Contrastive Learning: Self-supervised learning to maximize real/fake feature separation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import numpy as np
from transformers import CLIPModel, CLIPProcessor
import timm
from einops import rearrange, repeat
import math


# =============================================================================
# ACTIVE RESEARCH MODELS (Used in Paper Results)
# =============================================================================

class HybridCNNTransformer(nn.Module):
    """
    Hybrid CNN-Transformer model combining CNN feature extraction with Transformer attention.
    Used in the "Bridging the Gap" paper to combine local and global reasoning.
    """
    
    def __init__(self,
                 cnn_backbone: str = "efficientnet_b0",
                 embed_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 num_classes: int = 2,
                 patch_size: int = 7,
                 pretrained: bool = True):
        """
        Initialize hybrid model.
        
        Args:
            cnn_backbone: CNN backbone architecture
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            num_classes: Number of output classes
            patch_size: Patch size for spatial attention
            pretrained: Whether to use pretrained weights
        """
        super(HybridCNNTransformer, self).__init__()
        
        # CNN backbone
        self.cnn_backbone = timm.create_model(
            cnn_backbone, 
            pretrained=pretrained, 
            features_only=True,
            out_indices=[-1]  # Get last feature map
        )
        
        # Get CNN output dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            cnn_features = self.cnn_backbone(dummy_input)[0]
            self.cnn_channels = cnn_features.shape[1]
            self.cnn_height = cnn_features.shape[2]
            self.cnn_width = cnn_features.shape[3]
        
        # Patch embedding
        self.patch_size = patch_size
        self.num_patches = (self.cnn_height // patch_size) * (self.cnn_width // patch_size)
        
        self.patch_embedding = nn.Conv2d(
            self.cnn_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
        # Positional embedding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim)
        )
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.patch_embedding.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size = x.shape[0]
        
        # CNN feature extraction
        cnn_features = self.cnn_backbone(x)[0]  # (B, C, H, W)
        
        # Patch embedding
        patches = self.patch_embedding(cnn_features)  # (B, embed_dim, H/patch_size, W/patch_size)
        patches = patches.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        patches = torch.cat((cls_tokens, patches), dim=1)
        
        # Add positional embedding
        patches = patches + self.pos_embedding
        
        # Transformer encoder
        transformer_output = self.transformer(patches)
        
        # Classification
        cls_output = self.norm(transformer_output[:, 0])
        output = self.classifier(cls_output)
        
        return output
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification."""
        batch_size = x.shape[0]
        
        # CNN feature extraction
        cnn_features = self.cnn_backbone(x)[0]
        
        # Patch embedding
        patches = self.patch_embedding(cnn_features)
        patches = patches.flatten(2).transpose(1, 2)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        patches = torch.cat((cls_tokens, patches), dim=1)
        
        # Add positional embedding
        patches = patches + self.pos_embedding
        
        # Transformer encoder
        transformer_output = self.transformer(patches)
        
        return transformer_output[:, 0]


class MultiScaleFeatureFusion(nn.Module):
    """
    Multi-scale feature fusion model for deepfake detection.
    Combines features from multiple scales and resolutions.
    Used in the "Bridging the Gap" paper for parameter-efficient artifact detection.
    """
    
    def __init__(self,
                 backbone: str = "efficientnet_b0",
                 num_classes: int = 2,
                 pretrained: bool = True):
        """
        Initialize multi-scale fusion model.
        
        Args:
            backbone: Backbone architecture
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
        """
        super(MultiScaleFeatureFusion, self).__init__()
        
        # Multi-scale backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=[1, 2, 3, 4]  # Multiple scales
        )
        
        # Get feature dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            self.feature_dims = [f.shape[1] for f in features]
        
        # Feature fusion layers
        self.fusion_layers = nn.ModuleList()
        for i, dim in enumerate(self.feature_dims):
            fusion_layer = nn.Sequential(
                nn.Conv2d(dim, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
            self.fusion_layers.append(fusion_layer)
        
        # Global feature aggregation
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256 * len(self.feature_dims), 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Extract multi-scale features
        features = self.backbone(x)
        
        # Fuse features
        fused_features = []
        for i, (feature, fusion_layer) in enumerate(zip(features, self.fusion_layers)):
            fused_feature = fusion_layer(feature)
            fused_feature = self.global_pool(fused_feature)
            fused_feature = fused_feature.flatten(1)
            fused_features.append(fused_feature)
        
        # Concatenate all features
        combined_features = torch.cat(fused_features, dim=1)
        
        # Classification
        output = self.classifier(combined_features)
        
        return output


# =============================================================================
# EXPERIMENTAL MODELS (Not currently in Paper)
# =============================================================================

class CLIPDeepfakeDetector(nn.Module):
    """
    CLIP-based deepfake detector using contrastive learning.
    Leverages CLIP's powerful visual representations for deepfake detection.
    (Currently experimental/future work).
    """
    
    def __init__(self,
                 model_name: str = "openai/clip-vit-base-patch32",
                 num_classes: int = 2,
                 freeze_clip: bool = False,
                 projection_dim: int = 512):
        """
        Initialize CLIP-based detector.
        
        Args:
            model_name: CLIP model name
            num_classes: Number of output classes
            freeze_clip: Whether to freeze CLIP weights
            projection_dim: Dimension of projection layer
        """
        super(CLIPDeepfakeDetector, self).__init__()
        
        # Load CLIP model
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(model_name)
        
        # Freeze CLIP if requested
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        # Get CLIP feature dimension
        clip_dim = self.clip_model.config.vision_config.hidden_size
        
        # Projection layers
        self.visual_projection = nn.Sequential(
            nn.Linear(clip_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim, projection_dim)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
        # Contrastive learning components
        self.temperature = nn.Parameter(torch.tensor(0.07))
        self.contrastive_projection = nn.Linear(projection_dim, 128)
    
    def forward(self, images: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            images: Input images
            return_features: Whether to return features
            
        Returns:
            Classification logits or (logits, features)
        """
        # Get CLIP visual features
        visual_features = self.clip_model.get_image_features(images)
        
        # Project features
        projected_features = self.visual_projection(visual_features)
        
        # Classification
        logits = self.classifier(projected_features)
        
        if return_features:
            return logits, projected_features
        return logits
    
    def contrastive_loss(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss for self-supervised learning.
        
        Args:
            features: Feature embeddings
            labels: Ground truth labels
            
        Returns:
            Contrastive loss
        """
        # Project to contrastive space
        contrastive_features = self.contrastive_projection(features)
        contrastive_features = F.normalize(contrastive_features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(contrastive_features, contrastive_features.T)
        similarity_matrix = similarity_matrix / self.temperature
        
        # Create positive/negative masks
        batch_size = labels.size(0)
        labels_expanded = labels.unsqueeze(1)
        positive_mask = (labels_expanded == labels_expanded.T).float()
        negative_mask = 1 - positive_mask
        
        # Remove diagonal (self-similarity)
        positive_mask.fill_diagonal_(0)
        
        # Compute contrastive loss
        exp_sim = torch.exp(similarity_matrix)
        positive_sim = exp_sim * positive_mask
        negative_sim = exp_sim * negative_mask
        
        # Avoid division by zero
        positive_sum = positive_sim.sum(dim=1, keepdim=True)
        negative_sum = negative_sim.sum(dim=1, keepdim=True)
        
        # Contrastive loss
        loss = -torch.log(positive_sum / (positive_sum + negative_sum + 1e-8))
        return loss.mean()


class ContrastiveLearningModel(nn.Module):
    """
    Self-supervised contrastive learning model for deepfake detection.
    (Currently experimental/future work).
    """
    
    def __init__(self,
                 backbone: str = "resnet50",
                 projection_dim: int = 128,
                 num_classes: int = 2,
                 pretrained: bool = True):
        """
        Initialize contrastive learning model.
        
        Args:
            backbone: Backbone architecture
            projection_dim: Projection dimension
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
        """
        super(ContrastiveLearningModel, self).__init__()
        
        # Backbone
        self.backbone = timm.create_model(
            backbone, 
            pretrained=pretrained, 
            num_classes=0  # Remove classifier
        )
        
        # Get backbone feature dimension
        backbone_dim = self.backbone.num_features
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
        # Temperature parameter
        self.temperature = nn.Parameter(torch.tensor(0.07))
    
    def forward(self, x: torch.Tensor, return_projection: bool = False) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images
            return_projection: Whether to return projection features
            
        Returns:
            Classification logits or (logits, projection_features)
        """
        # Extract features
        features = self.backbone(x)
        
        # Classification
        logits = self.classifier(features)
        
        if return_projection:
            projection = self.projection_head(features)
            return logits, projection
        return logits
    
    def contrastive_loss(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss between two augmented views.
        
        Args:
            x1: First augmented view
            x2: Second augmented view
            
        Returns:
            Contrastive loss
        """
        # Extract features
        features1 = self.backbone(x1)
        features2 = self.backbone(x2)
        
        # Project to contrastive space
        projection1 = self.projection_head(features1)
        projection2 = self.projection_head(features2)
        
        # Normalize projections
        projection1 = F.normalize(projection1, dim=1)
        projection2 = F.normalize(projection2, dim=1)
        
        # Compute similarity
        similarity = torch.matmul(projection1, projection2.T) / self.temperature
        
        # Create labels (diagonal should be positive pairs)
        batch_size = x1.size(0)
        labels = torch.arange(batch_size).to(x1.device)
        
        # Contrastive loss
        loss = F.cross_entropy(similarity, labels)
        return loss


# =============================================================================
# MODEL FACTORY
# =============================================================================

def create_advanced_model(model_name: str,
                         num_classes: int = 2,
                         pretrained: bool = True,
                         **kwargs) -> nn.Module:
    """
    Factory function to create advanced models.
    
    Args:
        model_name: Name of the model
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        **kwargs: Additional model-specific arguments
        
    Returns:
        Initialized model
    """
    model_name = model_name.lower()
    
    if model_name == "hybrid_cnn_transformer":
        return HybridCNNTransformer(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == "multiscale":
        return MultiScaleFeatureFusion(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == "clip":
        return CLIPDeepfakeDetector(num_classes=num_classes, **kwargs)
    elif model_name == "contrastive":
        return ContrastiveLearningModel(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == "frequency_aware":
        from src.models.frequency_models import FrequencyAwareCNN
        return FrequencyAwareCNN(num_classes=num_classes, pretrained=pretrained, **kwargs)
    else:
        raise ValueError(f"Unsupported advanced model: {model_name}")


def main():
    """Test advanced model creation and forward pass."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test advanced models")
    parser.add_argument("--model", default="hybrid_cnn_transformer", help="Model name")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    
    args = parser.parse_args()
    
    # Create model
    model = create_advanced_model(args.model, num_classes=2, pretrained=False)
    print(f"Created model: {args.model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(args.batch_size, 3, args.image_size, args.image_size)
    with torch.no_grad():
        output = model(x)
        print(f"Output shape: {output.shape}")
        
        # Test feature extraction
        if hasattr(model, 'get_features'):
            features = model.get_features(x)
            print(f"Features shape: {features.shape}")


if __name__ == "__main__":
    main()
