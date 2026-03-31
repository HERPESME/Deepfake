"""
Baseline models for deepfake detection.
Includes XceptionNet, EfficientNet, and Vision Transformer implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from transformers import ViTModel, ViTConfig
from typing import Optional, Tuple
import math


class XceptionNet(nn.Module):
    """
    XceptionNet model for deepfake detection.
    Based on the FaceForensics++ baseline implementation.
    """
    
    def __init__(self, 
                 num_classes: int = 2,
                 pretrained: bool = True,
                 dropout_rate: float = 0.5):
        """
        Initialize XceptionNet.
        
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate
        """
        super(XceptionNet, self).__init__()
        
        # Load pretrained Xception using timm (more reliable)
        import timm
        try:
            # Try xception65 first (closest to original)
            self.backbone = timm.create_model('xception65', pretrained=pretrained, num_classes=0)
        except Exception:
            try:
                # Fallback to efficientnet if xception not available
                self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=0)
            except Exception:
                # Ultimate fallback
                self.backbone = timm.create_model('resnet50', pretrained=pretrained, num_classes=0)
        
        # Get num_features by forward pass
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy)
            if isinstance(features, torch.Tensor):
                num_features = features.shape[1] if len(features.shape) > 1 else features.shape[0]
            else:
                num_features = 2048  # Default for xception
        
        # Create classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.backbone(x)
        # Handle different output shapes from timm
        if len(features.shape) > 2:
            features = features.mean(dim=[2, 3])  # Global average pooling
        elif len(features.shape) > 1:
            features = features.view(features.size(0), -1)
        return self.classifier(features)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification."""
        features = self.backbone(x)
        # Handle different output shapes
        if len(features.shape) > 2:
            features = features.mean(dim=[2, 3])  # Global average pooling
        elif len(features.shape) > 1:
            features = features.view(features.size(0), -1)
        return features


class EfficientNetModel(nn.Module):
    """
    EfficientNet model for deepfake detection.
    Supports various EfficientNet variants.
    """
    
    def __init__(self, 
                 model_name: str = "efficientnet_b0",
                 num_classes: int = 2,
                 pretrained: bool = True,
                 dropout_rate: float = 0.5):
        """
        Initialize EfficientNet.
        
        Args:
            model_name: EfficientNet variant name
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate
        """
        super(EfficientNetModel, self).__init__()
        
        # Load pretrained EfficientNet
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0  # Remove classifier
        )
        
        # Get number of features
        num_features = self.backbone.num_features
        
        # Add custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.backbone(x)
        return self.classifier(features)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification."""
        return self.backbone(x)


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) for deepfake detection.
    Implements a custom ViT architecture optimized for face images.
    """
    
    def __init__(self,
                 image_size: int = 224,
                 patch_size: int = 16,
                 num_classes: int = 2,
                 dim: int = 768,
                 depth: int = 12,
                 heads: int = 12,
                 mlp_dim: int = 3072,
                 dropout: float = 0.1,
                 pretrained: bool = True):
        """
        Initialize Vision Transformer.
        
        Args:
            image_size: Input image size
            patch_size: Patch size
            num_classes: Number of output classes
            dim: Embedding dimension
            depth: Number of transformer layers
            heads: Number of attention heads
            mlp_dim: MLP dimension
            dropout: Dropout rate
            pretrained: Whether to use pretrained weights
        """
        super(VisionTransformer, self).__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embedding = nn.Conv2d(
            3, dim, kernel_size=patch_size, stride=patch_size
        )
        
        # Positional embedding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, dim)
        )
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=depth
        )
        
        # Classification head
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
        # Load pretrained weights if available
        if pretrained:
            self._load_pretrained_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.patch_embedding.weight, std=0.02)
    
    def _load_pretrained_weights(self):
        """Load pretrained weights from Hugging Face."""
        try:
            config = ViTConfig(
                image_size=self.image_size,
                patch_size=self.patch_size,
                hidden_size=self.dim,
                num_hidden_layers=self.transformer.num_layers,
                num_attention_heads=self.transformer.layers[0].self_attn.num_heads,
                intermediate_size=self.transformer.layers[0].linear2.out_features,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                num_labels=self.classifier[-1].out_features
            )
            
            pretrained_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
            
            # Copy weights
            with torch.no_grad():
                self.patch_embedding.weight.copy_(pretrained_model.embeddings.patch_embeddings.weight)
                self.patch_embedding.bias.copy_(pretrained_model.embeddings.patch_embeddings.bias)
                self.pos_embedding.copy_(pretrained_model.embeddings.position_embeddings.weight)
                self.cls_token.copy_(pretrained_model.embeddings.cls_token)
                
                # Copy transformer weights
                for i, layer in enumerate(self.transformer.layers):
                    layer.load_state_dict(pretrained_model.encoder.layer[i].state_dict())
                
                self.norm.load_state_dict(pretrained_model.layernorm.state_dict())
            
            print("Successfully loaded pretrained ViT weights")
            
        except Exception as e:
            print(f"Could not load pretrained weights: {e}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embedding(x)  # (B, dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embedding
        
        # Transformer encoder
        x = self.transformer(x)
        
        # Classification
        x = self.norm(x)
        cls_output = x[:, 0]  # Use class token
        output = self.classifier(cls_output)
        
        return output
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification."""
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embedding
        
        # Transformer encoder
        x = self.transformer(x)
        
        # Return class token features
        return x[:, 0]
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention weights for visualization."""
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embedding
        
        # Store attention weights
        attention_weights = []
        
        for layer in self.transformer.layers:
            # Get attention weights from the layer
            attn_output, attn_weights = layer.self_attn(
                x, x, x, need_weights=True
            )
            attention_weights.append(attn_weights)
            x = layer.norm1(attn_output + x)
            x = layer.norm2(layer.linear2(layer.dropout(layer.activation(layer.linear1(x)))) + x)
        
        return attention_weights


class ResNetModel(nn.Module):
    """
    ResNet model for deepfake detection.
    Supports various ResNet variants.
    """
    
    def __init__(self,
                 model_name: str = "resnet50",
                 num_classes: int = 2,
                 pretrained: bool = True,
                 dropout_rate: float = 0.5):
        """
        Initialize ResNet.
        
        Args:
            model_name: ResNet variant name
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate
        """
        super(ResNetModel, self).__init__()
        
        # Load pretrained ResNet
        if model_name == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
        elif model_name == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
        elif model_name == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
        elif model_name == "resnet101":
            self.backbone = models.resnet101(pretrained=pretrained)
        elif model_name == "resnet152":
            self.backbone = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ResNet variant: {model_name}")
        
        # Replace classifier
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification."""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x


def create_model(model_name: str, 
                num_classes: int = 2,
                pretrained: bool = True,
                **kwargs) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_name: Name of the model
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        **kwargs: Additional model-specific arguments
        
    Returns:
        Initialized model
    """
    model_name = model_name.lower()
    
    if model_name == "xception":
        return XceptionNet(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == "efficientnet_b0":
        return EfficientNetModel("efficientnet_b0", num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == "efficientnet_b4":
        return EfficientNetModel("efficientnet_b4", num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == "vit":
        return VisionTransformer(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
        return ResNetModel(model_name, num_classes=num_classes, pretrained=pretrained, **kwargs)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def main():
    """Test model creation and forward pass."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test baseline models")
    parser.add_argument("--model", default="xception", help="Model name")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    
    args = parser.parse_args()
    
    # Create model
    model = create_model(args.model, num_classes=2, pretrained=False)
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
