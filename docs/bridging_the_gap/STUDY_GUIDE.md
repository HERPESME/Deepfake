# Comprehensive Study Guide: Bridging the Generalization Gap in Deepfake Detection

This study guide provides an in-depth code-level walkthrough for the "Bridging the Gap" phase of the Deepfake Detection project. It breaks down the theory and the corresponding PyTorch implementations of the novel architectures, ensemble strategies, and adversarial robustness tests introduced in this research.

---

## 1. Introduction: The Generalization Gap

In the first phase of research, we observed that standard CNNs (EfficientNet, XceptionNet, ResNet) achieved near-perfect accuracy on the training dataset (Celeb-DF) but collapsed to near-random guessing (~50%) when evaluated on an unseen dataset (FaceForensics++). 

This phenomenon is known as the **generalization gap**. CNNs tend to overfit to dataset-specific manipulations (like blending boundaries or specific GAN artifacts) rather than learning universal characteristics of deepfakes. 

To bridge this gap, three novel architectures were designed to force the models to learn more robust, dataset-agnostic features:
1. **Hybrid CNN-Transformer**: Leveraging CNNs for local texture and Transformers for global dependencies.
2. **Multi-Scale Feature Fusion**: Extracting features at multiple spatial resolutions.
3. **Frequency-Aware Dual-Branch**: Analyzing spatial pixels and spectral (frequency) artifacts simultaneously.

---

## 2. Core Architectures: Theory & Code

### 2.1 Hybrid CNN-Transformer 
*(Located in `src/models/advanced_models.py`)*

#### Theory
CNNs excel at extracting local texture anomalies (e.g., blurred edges around a fake face), but they lack "global context" (e.g., ensuring the lighting on the left side of the face matches the right). Vision Transformers (ViT) capture global context perfectly via self-attention but are data-hungry. This architecture combines the best of both by using an `EfficientNet` to extract a rich 2D feature map, flattening it into "patches", and passing those patches through a `TransformerEncoder`.

#### Code Snippet
```python
class HybridCNNTransformer(nn.Module):
    def __init__(self, cnn_backbone="efficientnet_b0", embed_dim=512, num_heads=8, num_layers=6):
        super(HybridCNNTransformer, self).__init__()
        
        # 1. Feature Extractor: CNN backbone
        self.cnn_backbone = timm.create_model(
            cnn_backbone, pretrained=True, features_only=True, out_indices=[-1]
        )
        
        # 2. Patch Embedding Conv Layer
        self.patch_embedding = nn.Conv2d(self.cnn_channels, embed_dim, kernel_size=7, stride=7)
        
        # 3. Positional & Class Tokens
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # Extract CNN features (B, C, H, W)
        cnn_features = self.cnn_backbone(x)[0]
        
        # View as sequence of patches and embed
        patches = self.patch_embedding(cnn_features).flatten(2).transpose(1, 2)
        
        # Add class token and positional embeddings
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        patches = torch.cat((cls_tokens, patches), dim=1) + self.pos_embedding
        
        # Apply Self-Attention
        transformer_output = self.transformer(patches)
        
        # Classify based on the CLS token
        return self.classifier(self.norm(transformer_output[:, 0]))
```

### 2.2 Multi-Scale Feature Fusion
*(Located in `src/models/advanced_models.py`)*

#### Theory
Deepfake artifacts manifest at different scales. Pixel-level noise requires high-resolution analysis, while warped facial structures require low-resolution, high-level features. The `MultiScaleFeatureFusion` model taps into four intermediate stages of the CNN backbone simultaneously, projecting them all into a common 256-D space, pooling them, and combining the scales.

#### Code Snippet
```python
class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, backbone="efficientnet_b0", num_classes=2):
        super(MultiScaleFeatureFusion, self).__init__()
        
        # Multi-scale backbone extracting 4 distinct resolutions
        self.backbone = timm.create_model(
            backbone, pretrained=True, features_only=True, out_indices=[1, 2, 3, 4]
        )
        
        # Separate projection layers (1x1 Convs) for each scale
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ) for dim in self.feature_dims
        ])
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256 * len(self.feature_dims), 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x) # Returns a list of 4 tensors
        
        # Process and pool each scale independently
        fused_features = [
            self.global_pool(fusion_layer(feat)).flatten(1) 
            for feat, fusion_layer in zip(features, self.fusion_layers)
        ]
        
        # Concatenate scales and classify
        return self.classifier(torch.cat(fused_features, dim=1))
```

### 2.3 Frequency-Aware Dual-Branch
*(Located in `src/models/frequency_models.py`)*

#### Theory
GANs and upsampling algorithms leave distinct trace patterns in the frequency domain (checkerboard artifacts). This model uses two parallel branches: one processes the standard RGB image, while the other computes a 2D-FFT (Fast Fourier Transform) to analyze the frequency spectra. The features are fused before classification.

#### Code Snippet
```python
class FrequencyAwareCNN(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        # Parallel feature extractors
        self.spatial_branch = SpatialBranch(feature_dim=feature_dim)
        self.frequency_branch = FrequencyBranch(in_channels=3, feature_dim=feature_dim)
        
        # Late fusion via MLP
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, 2)
        )
    
    def forward(self, x):
        # Note: FrequencyBranch internally converts 'x' using torch.fft.fft2(x)
        spatial_features = self.spatial_branch(x)
        freq_features = self.frequency_branch(x) 
        
        combined = torch.cat([spatial_features, freq_features], dim=1)
        return self.fusion(combined)
```

---

## 3. Ensemble Strategies

To maximize resilience against unknown deepfake variants, multiple distinct models are combined. The `EnsembleVotingModel` provides different aggregation mechanisms. 

#### Soft Voting
Soft voting averages the probability outputs (confidence scores) of all models. This is highly effective because highly confident correct predictions outweigh unsure incorrect predictions.

```python
    def soft_vote(self, images: torch.Tensor, use_weights: bool = True):
        # Get raw logits from all models in the ensemble
        outputs = self._get_all_outputs(images) 
        
        weighted_probs = torch.zeros_like(outputs[list(self.models.keys())[0]][0])
        total_weight = 0.0
        
        # Compute weighted average of predicted probabilities
        for name in self.models:
            w = self.model_weights[name] if use_weights else 1.0
            
            # outputs[name][0] corresponds to the softmax probabilities
            weighted_probs += w * outputs[name][0] 
            total_weight += w
            
        return None, weighted_probs / total_weight
```

---

## 4. Adversarial Robustness Testing

Deepfake detectors must be secure against attackers who intentionally manipulate images to bypass detection. We use two White-Box attacks (where the attacker has full access to the model gradients) to evaluate models.

### 4.1 Fast Gradient Sign Method (FGSM)
*(Located in `src/evaluation/adversarial.py`)*

**Theory**: A single-step attack. It calculates the gradient of the loss with respect to the input image, takes the `sign()` of that gradient, and adds a small perturbation ($\epsilon$) to instantly maximize the model's error.

```python
def fgsm_attack(model, images, labels, epsilon, device="cpu"):
    images = images.clone().detach().to(device).requires_grad_(True)
    
    # 1. Forward pass and get loss
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    
    # 2. Compute gradients with respect to input pixels
    model.zero_grad()
    loss.backward()
    
    # 3. Create adversarial perturbation along gradient sign
    perturbed = images + epsilon * images.grad.sign()
    
    # 4. Clamp back to valid image domains [0,1]
    return torch.clamp(perturbed, 0, 1).detach()
```

### 4.2 Projected Gradient Descent (PGD)
**Theory**: The multi-step "iterative" version of FGSM. Instead of taking one big step $\epsilon$, it takes `num_steps` tiny steps ($\alpha$) in the gradient direction, clamping the total perturbation to never exceed $\epsilon$. It is a much stronger attack.

```python
def pgd_attack(model, images, labels, epsilon, alpha=None, num_steps=10, device="cpu"):
    if alpha is None:
        alpha = epsilon / 4.0
        
    images = images.clone().detach().to(device)
    
    # 1. Start with random noise within epsilon bounds
    perturbed = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
    perturbed = torch.clamp(perturbed, 0, 1).detach()
    
    # 2. Iteratively maximize the loss
    for _ in range(num_steps):
        perturbed.requires_grad_(True)
        outputs = model(perturbed)
        loss = F.cross_entropy(outputs, labels)
        
        model.zero_grad()
        loss.backward()
        
        # Take a small step alpha
        adv_images = perturbed + alpha * perturbed.grad.sign()
        
        # Project (clip) the perturbation back into the allowed epsilon ball
        eta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
        perturbed = torch.clamp(images + eta, min=0, max=1).detach()
        
    return perturbed
```
