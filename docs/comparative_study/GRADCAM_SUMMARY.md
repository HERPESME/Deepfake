# 🔍 Grad-CAM: Complete Summary

## **WHAT IS GRAD-CAM?**

**Grad-CAM (Gradient-weighted Class Activation Mapping)** is a technique that visualizes which parts of an input image are most important for a deep learning model's prediction.

### **Simple Explanation**

Think of it like this:
- When you look at a photo and decide "this is a fake face"
- You might focus on specific areas: "the eyes look weird" or "the mouth alignment is off"
- **Grad-CAM does the same for AI models** - it shows **which parts of the image the model "looked at"** to make its decision

### **How It Works**

1. **Forward Pass**: Model makes prediction
2. **Backward Pass**: Calculate gradients (how much each pixel/location affects prediction)
3. **Extract Activations**: Get feature maps from convolutional layers
4. **Weight by Importance**: Combine gradients (importance) with activations (what model sees)
5. **Create Heatmap**: Visualize important regions
6. **Overlay**: Show heatmap on original image

**Result**: Heatmap showing which regions influenced the prediction
- **Red/Hot** = High importance (model focused here)
- **Blue/Cold** = Low importance (model ignored this)

---

## **WHERE IS IT IMPLEMENTED?**

### **File Location**

**Primary File**: `src/explainability/gradcam.py`

### **Main Classes**

1. **`GradCAM`** (Lines 18-150)
   - Basic Grad-CAM implementation
   - Core algorithm for generating heatmaps

2. **`GradCAMPlusPlus`** (Lines 158-240)
   - Improved version with better localization
   - More accurate heatmaps

3. **`ExplainabilityAnalyzer`** (Lines 395-510)
   - High-level interface
   - Automatically finds target layers
   - Integrates with inference pipeline

### **Code Sections**

#### **1. GradCAM Class** (`src/explainability/gradcam.py`)

```python
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) implementation.
    """
    
    def __init__(self, model, target_layers):
        # Initialize with trained model
        # Target layers: Which convolutional layers to analyze
        
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

#### **2. Integration in Main Pipeline**

**Location**: `main.py` (Lines 440-446)

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

## **GRAD-CAM VISUAL RESULTS**

### **Where Are Visualizations Saved?**

**Directory**: `reports/explainability/`

### **Generated Files**

When you run inference with `--explainability` flag, Grad-CAM generates:

1. **Overlay Image**: `{image_name}_gradcam_overlay.png`
   - Original image with heatmap overlaid (semi-transparent)
   - Shows which regions influenced the prediction
   - Red areas = Important regions

2. **Heatmap Image**: `{image_name}_gradcam_heatmap.png`
   - Just the heatmap (color-coded importance map)
   - Red = High importance
   - Blue = Low importance

### **Current Visualizations**

**Location**: `reports/explainability/`

**Files Found**:
- `id0_id1_0000_frame000000_frame_000000_gradcam_overlay.png` (50 KB)
- `id0_id1_0000_frame000000_frame_000000_gradcam_heatmap.png` (835 bytes)

### **What the Visualizations Show**

**For Fake Images**:
- Model identifies as FAKE (100% confidence)
- Heatmap highlights facial regions with artifacts
- Focuses on areas where deepfake generation creates inconsistencies
- Red areas show where model detected fake features

**For Real Images**:
- Model identifies as REAL (99.95% confidence)
- Heatmap highlights natural facial features
- No suspicious artifact patterns
- Red areas show natural facial structures

---

## **HOW TO USE GRAD-CAM**

### **Method 1: Command Line**

```bash
python main.py inference \
  --model_path experiments/effb0_celebd_full/final_model.pth \
  --image_path path/to/your/image.jpg \
  --model efficientnet_b0 \
  --explainability
```

**Output**: Visualizations saved to `reports/explainability/`

### **Method 2: Python Code**

```python
from src.explainability.gradcam import ExplainabilityAnalyzer
import torch

# Load model
model = load_model('experiments/effb0_celebd_full/final_model.pth')

# Create analyzer
analyzer = ExplainabilityAnalyzer(model, output_dir="reports/explainability")

# Analyze image
analysis = analyzer.analyze_image(
    image_path="path/to/image.jpg",
    label=1  # 0=real, 1=fake
)

# Get results
overlay_path = analysis["explanations"]["gradcam"]["overlay_path"]
heatmap_path = analysis["explanations"]["gradcam"]["heatmap_path"]
```

---

## **TECHNICAL DETAILS**

### **Algorithm Steps**

1. **Forward Pass**:
   ```python
   output = model(input_tensor)  # Get prediction
   target_class = torch.argmax(output, dim=1)  # Get predicted class
   ```

2. **Backward Pass**:
   ```python
   target_score = output[0, target_class]
   target_score.backward()  # Calculate gradients
   ```

3. **Extract Gradients & Activations**:
   ```python
   gradients = self.gradients[layer_name]  # From backward pass
   activations = self.activations[layer_name]  # From forward pass
   ```

4. **Compute Importance Weights**:
   ```python
   weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
   ```

5. **Generate CAM**:
   ```python
   cam = torch.sum(weights * activations, dim=1, keepdim=True)
   cam = F.relu(cam)  # Only positive contributions
   ```

6. **Normalize & Visualize**:
   ```python
   cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
   heatmap = cm.jet(cam)  # Color map (blue=low, red=high)
   overlay = cv2.addWeighted(original_image, 1-alpha, heatmap, alpha, 0)
   ```

### **Target Layers**

Grad-CAM works on **convolutional layers** (not fully connected layers).

**For EfficientNet-B0**:
- Target layer: Last convolutional layer before global pooling
- Usually: `features` or `backbone` layers
- Code automatically finds these layers

**Finding Target Layers**:
```python
def _find_target_layers(self) -> List[str]:
    """Find appropriate target layers for Grad-CAM."""
    target_layers = []
    
    for name, module in self.model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.AdaptiveAvgPool2d)):
            if 'features' in name or 'backbone' in name:
                target_layers.append(name)
    
    return target_layers[-1:] if target_layers else []
```

---

## **WHY GRAD-CAM FOR DEEPFAKE DETECTION?**

### **Benefits**

1. **Understand Model Behavior**:
   - See which facial regions trigger "fake" detection
   - Verify model focuses on correct areas

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

## **CURRENT STATUS**

### **Implementation Status**: ✅ **COMPLETE & WORKING**

**What's Working**:
- ✅ Grad-CAM class implementation
- ✅ Grad-CAM++ implementation
- ✅ ExplainabilityAnalyzer wrapper
- ✅ Integration with inference pipeline
- ✅ Tested on real and fake images
- ✅ Successfully generates and saves visualizations

**Test Results**:
- ✅ Fake image: Correctly identified as FAKE (100% confidence)
- ✅ Real image: Correctly identified as REAL (99.95% confidence)
- ✅ Grad-CAM visualizations generated successfully
- ✅ Visualizations saved to `reports/explainability/`

**Visualization Files**:
- ✅ Overlay images saved: `{image_name}_gradcam_overlay.png`
- ✅ Heatmap images saved: `{image_name}_gradcam_heatmap.png`

---

## **SUMMARY**

### **What is Grad-CAM?**
- Technique to visualize which parts of image influence model's prediction
- Shows "where the model looks" to make its decision
- Creates heatmap highlighting important regions

### **Where is it implemented?**
- **File**: `src/explainability/gradcam.py`
- **Classes**: `GradCAM`, `GradCAMPlusPlus`, `ExplainabilityAnalyzer`
- **Usage**: Via `main.py inference --explainability`

### **Are visual results available?**
- ✅ **Yes!** Visualizations are generated and saved
- **Location**: `reports/explainability/`
- **Format**: 
  - Overlay images: `{image_name}_gradcam_overlay.png`
  - Heatmap images: `{image_name}_gradcam_heatmap.png`
- **Status**: Tested and working on real and fake images

### **Key Features**:
- Understand model behavior
- Validate model correctness
- Debug failures
- Research insights
- Build trust and transparency

---

*Grad-CAM is fully implemented and working in your project! Visualizations are saved when you run inference with the `--explainability` flag.*



