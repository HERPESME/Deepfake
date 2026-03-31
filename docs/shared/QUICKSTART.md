# Quick Start Guide - Deepfake Detection Project

## 🚀 **3-Step Setup (5 minutes)**

### **Step 1: Install Dependencies**
```bash
# Make sure you're in the project directory
cd /Users/eeshansingh/Desktop/FullStackProjects/Deepfake

# Activate virtual environment (if using)
source deepfake_env/bin/activate  # or: deepfake_env\Scripts\activate on Windows

# Install requirements
pip install -r requirements.txt
```

### **Step 2: Create Test Data**
```bash
# Create sample dataset for testing (no real data needed!)
python scripts/create_sample_data.py
```

This creates:
- `data/processed/sample/` with fake face images
- Train/val/test splits
- Ready to use for testing

### **Step 3: Run Quick Test**
```bash
# Test training on sample data (5 epochs, ~2 minutes)
python main.py train \
    --model xception \
    --dataset sample \
    --data_root data/processed \
    --epochs 5 \
    --batch_size 4 \
    --cpu  # Use CPU if no GPU
```

If this completes successfully, **everything is working!** ✅

---

## 📋 **Common Commands**

### **Training**
```bash
# Train on sample data
python main.py train --model xception --dataset sample --data_root data/processed --epochs 10

# Train with custom parameters
python main.py train \
    --model vit \
    --dataset sample \
    --data_root data/processed \
    --epochs 20 \
    --batch_size 16 \
    --lr 0.0001
```

### **Evaluation**
```bash
# Evaluate trained model
python main.py evaluate \
    --model_path experiments/xception_sample/final_model.pth \
    --dataset sample \
    --data_root data/processed
```

### **Inference on Single Image**
```bash
# Test on a single image
python main.py inference \
    --model_path experiments/xception_sample/final_model.pth \
    --image_path path/to/image.jpg \
    --explainability  # Add explainability visualization
```

### **Check Dataset Structure**
```bash
python scripts/download_datasets.py --check_structure
```

---

## 🎯 **Next Steps After Quick Test**

### **Option A: Use Real Datasets (Recommended for Actual Research)**

1. **Download Datasets:**
   ```bash
   python scripts/download_datasets.py --dataset faceforensics
   # Follow instructions to manually download
   ```

2. **Preprocess Data:**
   ```bash
   # Preprocessing uses direct module call (has its own main function)
   python src/data/preprocessing.py \
       --dataset faceforensics \
       --data_path data/raw/faceforensics \
       --output_path data/processed/faceforensics
   ```

3. **Train on Real Data:**
   ```bash
   # Use main.py for unified interface (recommended)
   python main.py train \
       --model xception \
       --dataset faceforensics \
       --data_root data/processed \
       --epochs 50
   ```

### **Option B: Continue with Sample Data**

You can experiment with different models:
- `xception` - Fast, proven baseline
- `efficientnet_b0` - Efficient and accurate
- `vit` - Vision Transformer (requires more memory)
- `resnet50` - Classic architecture

### **Option C: Run Jupyter Notebook Demo**

```bash
# Install jupyter if not already installed
pip install jupyter notebook

# Start Jupyter
jupyter notebook

# Open: notebooks/deepfake_detection_demo.ipynb
```

---

## 🔧 **Troubleshooting**

### **Issue: "No module named 'src'"**
```bash
# Make sure you're in the project root directory
cd /Users/eeshansingh/Desktop/FullStackProjects/Deepfake
```

### **Issue: CUDA Out of Memory**
```bash
# Use smaller batch size or CPU
python main.py train ... --batch_size 4 --cpu
```

### **Issue: Dataset Not Found**
```bash
# Create sample data first
python scripts/create_sample_data.py
```

### **Issue: Import Errors**
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

---

## 📊 **Expected Output**

### **Successful Training Output:**
```
2024-01-XX 10:30:00 - INFO - Model: XceptionNet
2024-01-XX 10:30:00 - INFO - Dataset: sample
2024-01-XX 10:30:00 - INFO - Train samples: 200
2024-01-XX 10:30:00 - INFO - Starting training...
2024-01-XX 10:30:05 - INFO - Epoch 0: Train Loss: 0.6234, Train Acc: 0.6500
...
2024-01-XX 10:32:00 - INFO - Training completed!
2024-01-XX 10:32:00 - INFO - Best validation AUC: 0.8500
```

### **Output Files:**
```
experiments/
└── xception_sample_YYYYMMDD_HHMMSS/
    ├── final_model.pth          # Trained model
    ├── training_history.json    # Training metrics
    └── final_results.json       # Final evaluation

reports/
└── xception_sample_YYYYMMDD_HHMMSS/
    ├── xception_sample_report.pdf
    ├── xception_sample_report.html
    └── visualizations/
        ├── training_history.png
        └── confusion_matrix.png
```

---

## ✅ **Verification Checklist**

- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Sample data created (`python scripts/create_sample_data.py`)
- [ ] Quick test completed (`python main.py train ...`)
- [ ] Model saved to `experiments/` folder
- [ ] Reports generated in `reports/` folder

If all checked, you're ready to start serious work! 🎉

---

## 📚 **Learn More**

- Full documentation: `README.md`
- Configuration: `configs/training_config.yaml`
- Dataset info: `python scripts/download_datasets.py --dataset all`

---

**Ready to go! Start with Step 1 above.** 🚀