"""
Frequency-domain models for deepfake detection.
Implements dual-branch architecture combining spatial CNN with FFT spectral analysis.

Hypothesis: GAN-generated deepfakes exhibit characteristic spectral artifacts (e.g.,
periodic patterns from upsampling, checkerboard artifacts) that are more consistent
across datasets than spatial-domain artifacts, potentially improving cross-dataset
generalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math


class FFTFeatureExtractor(nn.Module):
    """
    Extracts frequency-domain features from images using 2D FFT.
    Converts spatial-domain images to log-magnitude spectrum, which captures
    the energy distribution across spatial frequencies.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (B, C, H, W)
        Returns:
            Log-magnitude spectrum (B, C, H, W)
        """
        # Apply 2D FFT on spatial dimensions
        fft = torch.fft.fft2(x, dim=(-2, -1))
        # Shift zero-frequency to center
        fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
        # Log-magnitude spectrum (add epsilon for numerical stability)
        magnitude = torch.abs(fft_shifted)
        log_magnitude = torch.log1p(magnitude)
        
        return log_magnitude


class FrequencyBranch(nn.Module):
    """
    CNN branch that processes FFT spectral features.
    Uses a lightweight CNN architecture optimized for spectral pattern recognition.
    """
    
    def __init__(self, in_channels: int = 3, feature_dim: int = 256):
        super().__init__()
        
        self.fft_extractor = FFTFeatureExtractor()
        
        # Lightweight CNN for spectral features
        self.conv_layers = nn.Sequential(
            # Block 1: 224x224 -> 112x112
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Block 2: 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 3: 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Block 4: 28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Block 5: 14x14 -> 7x7
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, feature_dim)
        self.bn = nn.BatchNorm1d(feature_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Raw images (B, C, H, W)
        Returns:
            Frequency features (B, feature_dim)
        """
        # Convert to frequency domain
        freq = self.fft_extractor(x)
        # Extract features from spectrum
        feat = self.conv_layers(freq)
        feat = self.pool(feat).flatten(1)
        feat = self.bn(self.fc(feat))
        return feat


class SpatialBranch(nn.Module):
    """
    Spatial branch using pre-trained EfficientNet-B0.
    Extracts rich spatial features from the image domain.
    """
    
    def __init__(self, pretrained: bool = True, feature_dim: int = 256):
        super().__init__()
        
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,  # Remove classifier, get features only
            global_pool="avg",
        )
        
        backbone_dim = 1280  # EfficientNet-B0 feature dimension
        self.fc = nn.Linear(backbone_dim, feature_dim)
        self.bn = nn.BatchNorm1d(feature_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (B, C, H, W)
        Returns:
            Spatial features (B, feature_dim)
        """
        feat = self.backbone(x)
        feat = self.bn(self.fc(feat))
        return feat


class FrequencyAwareCNN(nn.Module):
    """
    Dual-branch deepfake detector combining spatial and frequency analysis.
    
    Architecture:
        Input Image -> [Spatial Branch (EfficientNet-B0)] -> spatial_features (256-d)
                    -> [FFT -> Frequency Branch (CNN)]     -> freq_features (256-d)
        
        Concatenate(spatial, freq) -> Fusion MLP -> Classification (real/fake)
    
    The key insight is that deepfake generation techniques (GANs, autoencoders)
    introduce spectral artifacts that may be more consistent across different
    datasets and manipulation methods than spatial-domain artifacts.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        feature_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.spatial_branch = SpatialBranch(pretrained=pretrained, feature_dim=feature_dim)
        self.frequency_branch = FrequencyBranch(in_channels=3, feature_dim=feature_dim)
        
        # Late fusion classifier
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (B, C, H, W)
        Returns:
            Logits (B, num_classes)
        """
        spatial_feat = self.spatial_branch(x)
        freq_feat = self.frequency_branch(x)
        
        # Concatenate spatial + frequency features
        combined = torch.cat([spatial_feat, freq_feat], dim=1)
        
        return self.fusion(combined)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get concatenated spatial + frequency features for analysis."""
        with torch.no_grad():
            spatial_feat = self.spatial_branch(x)
            freq_feat = self.frequency_branch(x)
            return torch.cat([spatial_feat, freq_feat], dim=1)
    
    def get_branch_features(self, x: torch.Tensor):
        """Get separate branch features for analysis."""
        with torch.no_grad():
            spatial_feat = self.spatial_branch(x)
            freq_feat = self.frequency_branch(x)
            return {"spatial": spatial_feat, "frequency": freq_feat}
