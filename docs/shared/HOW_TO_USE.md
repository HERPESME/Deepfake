# 🚀 How to Use the Deepfake Detection System

## **Quick Start Guide**

Your deepfake detection system is ready to use! Here's how to use it for different purposes.

---

## 📋 **TABLE OF CONTENTS**

1. [Using Pre-Trained Models (Inference)](#using-pre-trained-models)
2. [Training on New Data](#training-on-new-data)
3. [Evaluating Models](#evaluating-models)
4. [Common Use Cases](#common-use-cases)
5. [Troubleshooting](#troubleshooting)

---

## 🔍 **1. USING PRE-TRAINED MODELS (Inference)**

### **Check Available Models**

First, see what models you have trained:

```bash
ls -lh experiments/*/final_model.pth
```

You should see:
- `experiments/efficientnet_b0_sample/final_model.pth` (Best model - 18MB)
- `experiments/resnet50_sample/final_model.pth` (94MB)
- `experiments/xception_sample/final_model.pth` (149MB)
- `experiments/xception_5epochs/final_model.pth` (149MB)

### **Run Inference on a Single Image**

```bash
# Basic inference
python main.py inference \
    --model_path experiments/efficientnet_b0_sample/final_model.pth \
    --image_path path/to/your/image.jpg \
    --cpu

# With explainability (shows what the model is looking at)
python main.py inference \
    --model_path experiments/efficientnet_b0_sample/final_model.pth \
    --image_path path/to/your/image.jpg \
    --explainability \
    --cpu
```

**Output**: 
- Prediction: REAL or FAKE
- Confidence score
- (With --explainability) Visualization saved to `reports/explainability/`

### **Example: Test on Your Own Image**

```bash
# Replace "your_image.jpg" with actual image path
python main.py inference \
    --model_path experiments/efficientnet_b0_sample/final_model.pth \
    --image_path your_image.jpg \
    --explainability \
    --cpu
```

---

## 🎓 **2. TRAINING ON NEW DATA**

### **Option A: Quick Test with Sample Data** (Already Done)

```bash
# This creates 150 sample images automatically
python scripts/create_sample_data.py

# Train on sample data
python main.py train \
    --model efficientnet_b0 \
    --dataset sample \
    --data_root data/processed \
    --epochs 10 \
    --batch_size 4 \
    --cpu
```

### **Option B: Train on Real Datasets**

#### **Step 1: Download Dataset**

```bash
# Get instructions for downloading
python scripts/download_datasets.py --dataset faceforensics

# Or for Celeb-DF
python scripts/download_datasets.py --dataset celebd
```

Follow the instructions to download the dataset manually (they require registration/forms).

#### **Step 2: Preprocess the Dataset**

```bash
# For FaceForensics++
python src/data/preprocessing.py \
    --dataset faceforensics \
    --data_path data/raw/faceforensics \
    --output_path data/processed/faceforensics \
    --face_detector mtcnn

# For Celeb-DF
python src/data/preprocessing.py \
    --dataset celebd \
    --data_path data/raw/celebd \
    --output_path data/processed/celebd \
    --face_detector mtcnn
```

#### **Step 3: Train Model**

```bash
python main.py train \
    --model efficientnet_b0 \
    --dataset faceforensics \
    --data_root data/processed \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.0001
```

---

## 📊 **3. EVALUATING MODELS**

### **Evaluate a Trained Model**

```bash
python main.py evaluate \
    --model_path experiments/efficientnet_b0_sample/final_model.pth \
    --dataset sample \
    --data_root data/processed \
    --batch_size 4 \
    --cpu
```

**Output**: Accuracy, AUC, Precision, Recall, F1-Score

### **Cross-Dataset Evaluation** (Test Generalization)

```bash
python main.py cross-dataset \
    --model efficientnet_b0 \
    --train_dataset sample \
    --test_datasets sample \
    --cpu
```

---

## 🎯 **4. COMMON USE CASES**

### **Use Case 1: Detect Deepfake in a Single Image**

```bash
# Use your best model
python main.py inference \
    --model_path experiments/efficientnet_b0_sample/final_model.pth \
    --image_path suspicious_image.jpg \
    --explainability \
    --cpu
```

**Result**: 
- ✅ Prediction: REAL or FAKE
- ✅ Confidence percentage
- ✅ Visualization showing what the model detected

### **Use Case 2: Batch Processing Multiple Images**

Create a simple script:

```python
# batch_inference.py
import os
from pathlib import Path
import subprocess

model_path = "experiments/efficientnet_b0_sample/final_model.pth"
image_folder = "path/to/images"
results = []

for image_file in Path(image_folder).glob("*.jpg"):
    result = subprocess.run(
        ["python", "main.py", "inference",
         "--model_path", model_path,
         "--image_path", str(image_file),
         "--cpu"],
        capture_output=True,
        text=True
    )
    results.append(f"{image_file.name}: {result.stdout}")

with open("batch_results.txt", "w") as f:
    f.write("\n".join(results))
```

### **Use Case 3: Train Custom Model for Specific Dataset**

```bash
# 1. Prepare your data in this structure:
# data/processed/your_dataset/
#   ├── images/
#   │   ├── real_001.jpg
#   │   ├── fake_001.jpg
#   └── splits/
#       ├── train.txt
#       ├── val.txt
#       └── test.txt

# 2. Train
python main.py train \
    --model efficientnet_b0 \
    --dataset your_dataset \
    --data_root data/processed \
    --epochs 20 \
    --batch_size 16
```

### **Use Case 4: Compare Different Models**

```bash
# Train multiple models and compare results
for model in efficientnet_b0 resnet50 xception; do
    python main.py train \
        --model $model \
        --dataset sample \
        --data_root data/processed \
        --epochs 5 \
        --experiment_name "${model}_comparison" \
        --cpu
done

# Check results in experiments/ folder
```

---

## 🔧 **5. PRACTICAL EXAMPLES**

### **Example 1: Quick Test of System**

```bash
# 1. Create sample data (if not exists)
python scripts/create_sample_data.py

# 2. Train a quick model (2 minutes)
python main.py train \
    --model efficientnet_b0 \
    --dataset sample \
    --data_root data/processed \
    --epochs 1 \
    --batch_size 4 \
    --cpu

# 3. Test on a sample image
python main.py inference \
    --model_path experiments/efficientnet_b0_sample/final_model.pth \
    --image_path data/processed/sample/images/real_001.jpg \
    --cpu
```

### **Example 2: Production-Ready Training**

```bash
# Train EfficientNet for real production use
python main.py train \
    --model efficientnet_b0 \
    --dataset faceforensics \
    --data_root data/processed \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.0001 \
    --experiment_name "production_efficientnet"
```

### **Example 3: Generate Report for Presentation**

Reports are **automatically generated** after training, but you can view them:

```bash
# Open HTML report in browser
open reports/efficientnet_b0_sample_*/efficientnet_b0_sample_report.html

# Or view PDF
open reports/efficientnet_b0_sample_*/efficientnet_b0_sample_report.pdf
```

---

## 📱 **6. COMMAND REFERENCE**

### **Training Commands**

```bash
# Basic training
python main.py train --model <model> --dataset <dataset> --data_root data/processed --epochs <n>

# With custom parameters
python main.py train \
    --model efficientnet_b0 \
    --dataset sample \
    --data_root data/processed \
    --epochs 20 \
    --batch_size 16 \
    --lr 0.0001 \
    --cpu \
    --experiment_name "my_experiment"
```

**Available Models**: `xception`, `efficientnet_b0`, `resnet50`, `vit`

### **Inference Commands**

```bash
# Basic inference
python main.py inference --model_path <path> --image_path <image>

# With explainability
python main.py inference --model_path <path> --image_path <image> --explainability
```

### **Evaluation Commands**

```bash
# Single dataset evaluation
python main.py evaluate --model_path <path> --dataset <dataset> --data_root data/processed

# Cross-dataset evaluation
python main.py cross-dataset --model <model> --train_dataset <dataset1> --test_datasets <dataset2>
```

---

## 🎓 **7. FOR COURSE DEMONSTRATION**

### **Quick Demo Script**

```bash
#!/bin/bash
# demo.sh - Complete demonstration

echo "=== Deepfake Detection System Demo ==="

# 1. Show available models
echo "1. Available Models:"
ls -lh experiments/*/final_model.pth

# 2. Run inference
echo "2. Testing Inference:"
python main.py inference \
    --model_path experiments/efficientnet_b0_sample/final_model.pth \
    --image_path data/processed/sample/images/real_001.jpg \
    --explainability \
    --cpu

# 3. Show reports
echo "3. Generated Reports:"
ls reports/*/xception_sample_report.html

# 4. Show training results
echo "4. Training Results:"
cat experiments/efficientnet_b0_sample/final_results.json | python -m json.tool
```

Run: `chmod +x demo.sh && ./demo.sh`

---

## ❓ **8. TROUBLESHOOTING**

### **Problem: "No module named 'src'"**
```bash
# Solution: Make sure you're in project root
cd /Users/eeshansingh/Desktop/FullStackProjects/Deepfake
```

### **Problem: "Model file not found"**
```bash
# Solution: Check available models
ls experiments/*/final_model.pth

# Use correct path
python main.py inference --model_path experiments/efficientnet_b0_sample/final_model.pth ...
```

### **Problem: "Dataset not found"**
```bash
# Solution: Create sample data first
python scripts/create_sample_data.py

# Or check your data path
ls data/processed/
```

### **Problem: "CUDA out of memory" or errors**
```bash
# Solution: Use CPU flag and smaller batch size
python main.py train ... --cpu --batch_size 4
```

### **Problem: Image inference fails**
```bash
# Solution: Check image format (must be JPG/PNG)
file your_image.jpg

# Make sure path is correct
python main.py inference --model_path <model> --image_path /full/path/to/image.jpg
```

---

## 📚 **9. QUICK REFERENCE CARD**

### **Most Common Commands**

```bash
# INFERENCE (Use existing model)
python main.py inference --model_path experiments/efficientnet_b0_sample/final_model.pth --image_path image.jpg --explainability --cpu

# TRAIN (Create new model)
python main.py train --model efficientnet_b0 --dataset sample --data_root data/processed --epochs 10 --batch_size 4 --cpu

# EVALUATE (Test model)
python main.py evaluate --model_path experiments/efficientnet_b0_sample/final_model.pth --dataset sample --data_root data/processed --cpu

# VIEW REPORTS
open reports/*/efficientnet_b0_sample_report.html
```

---

## ✅ **10. CHECKLIST: Is Everything Working?**

Run this to verify:

```bash
# Check 1: Models exist
echo "Models:" && ls experiments/*/final_model.pth

# Check 2: Sample data exists
echo "Data:" && ls data/processed/sample/images/*.jpg | head -3

# Check 3: Reports exist
echo "Reports:" && ls reports/*/*.html | head -3

# Check 4: Test inference
python main.py inference \
    --model_path $(ls experiments/efficientnet_b0_sample/final_model.pth) \
    --image_path $(ls data/processed/sample/images/*.jpg | head -1) \
    --cpu
```

If all pass ✅, your system is ready!

---

## 🎯 **SUMMARY**

**To Use the System:**

1. **For Detection**: Use pre-trained models with `main.py inference`
2. **For Training**: Use `main.py train` with your data
3. **For Evaluation**: Use `main.py evaluate` to test models
4. **For Reports**: View auto-generated reports in `reports/` folder

**Best Model**: `experiments/efficientnet_b0_sample/final_model.pth` (18MB, perfect performance)

**Start Here**: 
```bash
python main.py inference \
    --model_path experiments/efficientnet_b0_sample/final_model.pth \
    --image_path your_image.jpg \
    --explainability \
    --cpu
```

---

**🎉 Your system is ready to use! Start with inference on your images!**

