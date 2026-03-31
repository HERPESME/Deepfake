# 🔍 Grad-CAM: Complete Explanation & Implementation Guide

## **WHAT IS GRAD-CAM?**

### **Definition**
**Grad-CAM (Gradient-weighted Class Activation Mapping)** is a technique for visualizing which parts of an input image are most important for a deep learning model's prediction.

### **Simple Explanation**
Think of it like this:
- When you look at a photo and decide "this is a fake face"
- You might focus on specific areas: "the eyes look weird" or "the mouth alignment is off"
- Grad-CAM does the same for AI models - it shows **which parts of the image the model "looked at"** to make its decision

### **How It Works (Step-by-Step)**

1. **Forward Pass**: Model makes a prediction on the image
   - Input image → Model → Prediction (e.g., "FAKE" with 100% confidence)

2. **Backward Pass**: Calculate gradients 
   - Ask: "How much does each pixel/location affect the prediction?"
   - Gradients flow backward through the network

3. **Extract Activations**: Get the feature maps from a convolutional layer
   - These show what the model "sees" at different layers

4. **Weight by Importance**: 
   - Combine gradients (importance) with activations (what model sees)
   - Formula: `CAM = Sum(Gradients × Activations)`

5. **Create Heatmap**:
   - Red/Hot areas = Important for the prediction
   - Blue/Cold areas = Less important

6. **Overlay on Original Image**:
   - Show the heatmap on top of the original image
   - Visual explanation of model's decision

### **Why Grad-CAM is Important**

1. **Interpretability**: Understand why model made a decision
2. **Trust**: Build confidence in model predictions
3. **Debugging**: Find why model fails on certain images
4. **Research**: Understand what features models learn
5. **Validation**: Verify model focuses on correct regions

---

## **WHERE IS GRAD-CAM IMPLEMENTED?**

### **Location in Project**

**File**: `src/explainability/gradcam.py`

**Main Classes**:
1. **`GradCAM`** (Lines 18-150): Basic Grad-CAM implementation
2. **`GradCAMPlusPlus`** (Lines 158-240): Improved version with better localization
3. **`ExplainabilityAnalyzer`** (Lines 395-510): High-level interface that uses Grad-CAM

### **Implementation Details**

#### **1. GradCAM Class** (`src/explainability/gradcam.py`)

```python
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) implementation.
    """
    
    def __init__(self, model, target_layers):
        # Initialize with trained model
        # Target layers: Which convolutional layers to analyze
        # Usually the last conv layer before classification
        
    def generate_cam(self, input_tensor, target_class):
        # 1. Forward pass through model
        # 2. Backward pass to get gradients
        # 3. Extract activations from target layer
        # 4. Weight activations by gradient importance
        # 5. Create heatmap
        
    def visualize(self, input_tensor, original_image):
        # 1. Generate CAM heatmap
        # 2. Resize to match original image
        # 3. Create color heatmap (jet colormap)
        # 4. Overlay on original image
        # 5. Return overlay and heatmap
```

**Key Methods**:
- `_register_hooks()`: Register hooks to capture activations and gradients during forward/backward pass
- `generate_cam()`: Generate the actual Grad-CAM heatmap
- `visualize()`: Create visual output (overlay + heatmap)

#### **2. ExplainabilityAnalyzer Class**

**Location**: `src/explainability/gradcam.py` (Lines 395-510)

**Purpose**: High-level interface that automatically:
- Detects model type (CNN vs Transformer)
- Initializes appropriate explainers (Grad-CAM for CNNs)
- Finds target layers automatically
- Generates explanations

**Usage**:
```python
from src.explainability.gradcam import ExplainabilityAnalyzer

analyzer = ExplainabilityAnalyzer(model, output_dir="reports/explainability")
analysis = analyzer.analyze_image(image_path, label=predicted_label)
```

#### **3. Integration in Main Pipeline**

**Location**: `main.py` (Lines 430-446)

**How It's Used**:
```python
if args.explainability:
    from src.explainability.gradcam import ExplainabilityAnalyzer
    
    analyzer = ExplainabilityAnalyzer(model, output_dir="reports/explainability")
    analysis = analyzer.analyze_image(args.image_path, label=pred[0].item())
    logger.info(f"Visualizations saved to: reports/explainability")
```

**Command Line Usage**:
```bash
python main.py inference \
  --model_path experiments/effb0_celebd_full/final_model.pth \
  --image_path data/processed/celebd/fake/id0_id1_0000_frame000000_frame_000000.jpg \
  --model efficientnet_b0 \
  --explainability
```

---

## **HOW GRAD-CAM WORKS (TECHNICAL)**

### **Algorithm Steps**

1. **Forward Pass**:
   ```python
   output = model(input_tensor)  # Get prediction
   target_class = torch.argmax(output, dim=1)  # Get predicted class
   ```

2. **Backward Pass**:
   ```python
   target_score = output[0, target_class]  # Score for predicted class
   target_score.backward()  # Calculate gradients
   ```

3. **Extract Gradients & Activations**:
   ```python
   gradients = self.gradients[layer_name]  # Gradients from backward pass
   activations = self.activations[layer_name]  # Activations from forward pass
   ```

4. **Compute Importance Weights**:
   ```python
   # Average gradients over spatial dimensions
   weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
   ```

5. **Generate CAM**:
   ```python
   # Weight activations by importance
   cam = torch.sum(weights * activations, dim=1, keepdim=True)  # (B, 1, H, W)
   cam = F.relu(cam)  # Only positive contributions
   ```

6. **Normalize & Visualize**:
   ```python
   cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # Normalize to [0,1]
   heatmap = cm.jet(cam)  # Color map (blue=low, red=high)
   overlay = cv2.addWeighted(original_image, 1-alpha, heatmap, alpha, 0)
   ```

---

## **WHICH LAYERS ARE USED?**

### **Target Layers for Grad-CAM**

Grad-CAM works best on **convolutional layers** (not fully connected layers).

**For EfficientNet-B0**:
- Target layer: Last convolutional layer before global pooling
- Usually: `features` or `backbone` layers
- Code automatically finds these layers

**Finding Target Layers** (`_find_target_layers` method):
```python
def _find_target_layers(self) -> List[str]:
    """Find appropriate target layers for Grad-CAM."""
    target_layers = []
    
    for name, module in self.model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.AdaptiveAvgPool2d)):
            if 'features' in name or 'backbone' in name:
                target_layers.append(name)
    
    return target_layers
```

---

## **GRAD-CAM VISUAL RESULTS**

### **Where Are Visualizations Saved?**

**Directory**: `reports/explainability/`

**When You Run**:
```bash
python main.py inference --explainability ...
```

**Output**:
- Grad-CAM generates overlay images (original + heatmap)
- Heatmap images (just the heatmap)
- Saved to `reports/explainability/` directory

### **What the Visualizations Show**

1. **Original Image**: The input image (face)
2. **Heatmap**: Color-coded importance map
   - **Red/Hot** = High importance (model focused here)
   - **Blue/Cold** = Low importance (model ignored this)
3. **Overlay**: Original image with heatmap overlaid (semi-transparent)

### **Example Results**

**Fake Image**:
- Model identifies as FAKE (100% confidence)
- Heatmap highlights facial regions with artifacts
- Focuses on areas where deepfake generation creates inconsistencies
- Red areas show where model detected fake features

**Real Image**:
- Model identifies as REAL (99.95% confidence)
- Heatmap highlights natural facial features
- No suspicious artifact patterns
- Red areas show natural facial structures

---

## **TESTING GRAD-CAM**

### **How to Generate Grad-CAM Visualizations**

**Step 1: Run Inference with Explainability**
```bash
python main.py inference \
  --model_path experiments/effb0_celebd_full/final_model.pth \
  --image_path data/processed/celebd/fake/id0_id1_0000_frame000000_frame_000000.jpg \
  --model efficientnet_b0 \
  --explainability
```

**Step 2: Check Output**
```bash
# Visualizations should be saved to:
ls reports/explainability/
```

**Step 3: View Results**
- Open the generated images
- Compare original vs overlay
- Analyze which regions are highlighted

---

## **GRAD-CAM VS OTHER METHODS**

### **Grad-CAM vs Other Explainability Methods**

**1. Grad-CAM** (Implemented ✅):
- ✅ Works with any CNN
- ✅ No model modification needed
- ✅ Fast computation
- ✅ Good visualizations
- ✅ Class-discriminative (shows why model chose specific class)

**2. Grad-CAM++** (Also Implemented ✅):
- Improved version of Grad-CAM
- Better localization
- More accurate heatmaps
- Handles multiple objects better

**3. SHAP** (Implemented but not fully tested):
- Shapley Additive Explanations
- More mathematically rigorous
- Slower computation
- More complex

**4. Attention Visualization** (For Transformers):
- For Vision Transformer models
- Shows attention patterns
- Different from Grad-CAM (for CNN-based models)

---

## **CODE WALKTHROUGH**

### **Complete Flow**

```python
# 1. Initialize Grad-CAM
from src.explainability.gradcam import ExplainabilityAnalyzer

# 2. Load trained model
model = load_model('experiments/effb0_celebd_full/final_model.pth')

# 3. Create analyzer
analyzer = ExplainabilityAnalyzer(model, output_dir="reports/explainability")

# 4. Analyze image
analysis = analyzer.analyze_image(
    image_path="path/to/image.jpg",
    label=1  # 0=real, 1=fake
)

# 5. Get results
results = analysis["explanations"]["gradcam"]
overlay = results["overlay"]  # Image with heatmap overlay
heatmap = results["heatmap"]  # Just the heatmap
```

### **Key Code Sections**

**1. Hook Registration** (`_register_hooks`):
```python
# Register forward hook to capture activations
def forward_hook(name):
    def hook(module, input, output):
        self.activations[name] = output.detach()
    return hook

# Register backward hook to capture gradients
def backward_hook(name):
    def hook(module, grad_input, grad_output):
        self.gradients[name] = grad_output[0].detach()
    return hook
```

**2. CAM Generation**:
```python
# Forward pass
output = self.model(input_tensor)

# Get target class
target_class = torch.argmax(output, dim=1).item()

# Backward pass
target_score = output[0, target_class]
target_score.backward()

# Get gradients and activations
gradients = self.gradients[layer_name]
activations = self.activations[layer_name]

# Compute weights (importance)
weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

# Generate CAM
cam = torch.sum(weights * activations, dim=1, keepdim=True)
cam = F.relu(cam)  # Only positive contributions
```

**3. Visualization**:
```python
# Normalize CAM
cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

# Resize to original image size
cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))

# Create color heatmap (jet colormap: blue→green→yellow→red)
heatmap = cm.jet(cam_resized)[:, :, :3]
heatmap = (heatmap * 255).astype(np.uint8)

# Overlay on original image
overlay = cv2.addWeighted(original_image, 1-alpha, heatmap, alpha, 0)
```

---

## **WHY GRAD-CAM FOR DEEPFAKE DETECTION?**

### **Specific Benefits for Deepfake Detection**

1. **Understand Model Behavior**:
   - See which facial regions trigger "fake" detection
   - Verify model focuses on correct areas (face, not background)

2. **Validate Model**:
   - Check if model learns meaningful features
   - Ensure model isn't using spurious correlations

3. **Debug Failures**:
   - Understand why model fails on certain images
   - Identify failure patterns

4. **Research Insights**:
   - Understand what features distinguish real from fake
   - Guide future model improvements

5. **Trust & Transparency**:
   - Show users why model made a decision
   - Build confidence in predictions

---

## **CURRENT STATUS IN PROJECT**

### **Implementation Status**: ✅ **COMPLETE**

**What's Working**:
- ✅ Grad-CAM class implementation
- ✅ Grad-CAM++ implementation
- ✅ ExplainabilityAnalyzer wrapper
- ✅ Integration with inference pipeline
- ✅ Tested on real and fake images
- ✅ Successfully generates visualizations

**Test Results**:
- ✅ Fake image: Correctly identified as FAKE (100% confidence)
- ✅ Real image: Correctly identified as REAL (99.95% confidence)
- ✅ Grad-CAM visualizations generated successfully

**Location of Visualizations**:
- Directory: `reports/explainability/`
- Generated when running inference with `--explainability` flag

---

## **HOW TO USE GRAD-CAM IN YOUR PROJECT**

### **Method 1: Command Line**

```bash
python main.py inference \
  --model_path experiments/effb0_celebd_full/final_model.pth \
  --image_path path/to/your/image.jpg \
  --model efficientnet_b0 \
  --explainability
```

### **Method 2: Python Code**

```python
import sys
sys.path.insert(0, '.')

from src.explainability.gradcam import ExplainabilityAnalyzer
from src.models.baseline_models import create_baseline_model
import torch

# Load model
model = create_baseline_model('efficientnet_b0', num_classes=2, pretrained=False)
checkpoint = torch.load('experiments/effb0_celebd_full/final_model.pth', map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()

# Create analyzer
analyzer = ExplainabilityAnalyzer(model, output_dir="reports/explainability")

# Analyze image
image_path = "data/processed/celebd/fake/id0_id1_0000_frame000000_frame_000000.jpg"
analysis = analyzer.analyze_image(image_path, label=1)

# Get results
overlay = analysis["explanations"]["gradcam"]["overlay"]
heatmap = analysis["explanations"]["gradcam"]["heatmap"]

# Save or visualize
import matplotlib.pyplot as plt
plt.imshow(overlay)
plt.savefig("gradcam_result.png")
```

---

## **SUMMARY**

### **What is Grad-CAM?**
- Technique to visualize which parts of image influence model's prediction
- Shows "where the model looks" to make its decision

### **Where is it implemented?**
- **File**: `src/explainability/gradcam.py`
- **Classes**: `GradCAM`, `GradCAMPlusPlus`, `ExplainabilityAnalyzer`
- **Usage**: Via `main.py inference --explainability`

### **Are visual results available?**
- ✅ Yes! Generated when running inference with `--explainability` flag
- **Location**: `reports/explainability/`
- **Format**: Overlay images (original + heatmap) and heatmap images
- **Status**: Tested and working on real and fake images

### **Key Benefits**:
- Understand model behavior
- Validate model correctness
- Debug failures
- Research insights
- Build trust and transparency

---

*Grad-CAM is fully implemented and working in your project! You can generate visualizations using the inference command with the `--explainability` flag.*



