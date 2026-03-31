# Step-by-Step: Running the Deepfake Detection Project

## 🎯 **Complete Walkthrough**

This guide will walk you through running the entire project from start to finish.

---

## **STEP 1: Verify Python Environment** ✅

Check Python version and location:
```bash
cd /Users/eeshansingh/Desktop/FullStackProjects/Deepfake
/usr/bin/python3 --version
```

**Expected**: Python 3.9 or higher

---

## **STEP 2: Install Dependencies** 📦

Install all required packages (this may take 5-10 minutes):

```bash
# Make sure you're in the project directory
cd /Users/eeshansingh/Desktop/FullStackProjects/Deepfake

# Install all dependencies
/usr/bin/python3 -m pip install -r requirements.txt --user
```

**What this does**: Installs PyTorch, OpenCV, transformers, and all other required libraries.

**Troubleshooting**: If you get SSL errors, the packages might already be installed. Continue to next step.

---

## **STEP 3: Verify Installation** 🔍

Test that everything is set up correctly:

```bash
/usr/bin/python3 scripts/test_setup.py
```

**Expected output**: Should show ✅ for all checks (torch, numpy, opencv, etc.)

---

## **STEP 4: Create Sample Data** 📸

Create a small test dataset (no real data needed):

```bash
/usr/bin/python3 scripts/create_sample_data.py
```

**What this does**:
- Creates 150 synthetic face images (75 real, 75 fake)
- Saves to `data/processed/sample/`
- Creates train/val/test splits
- Takes ~30 seconds

**Expected output**:
```
✅ Created sample dataset
✅ 150 images created
✅ Splits created: train=120, val=15, test=15
```

---

## **STEP 5: Quick Training Test** 🚀

Train a model on the sample data (2 epochs, ~5 minutes):

```bash
/usr/bin/python3 main.py train \
    --model xception \
    --dataset sample \
    --data_root data/processed \
    --epochs 2 \
    --batch_size 4 \
    --cpu
```

**What this does**:
- Loads sample images
- Trains XceptionNet model
- Validates after each epoch
- Saves model checkpoints
- Generates reports

**Expected output**:
```
2025-XX-XX INFO - Model: XceptionNet
2025-XX-XX INFO - Dataset: sample
2025-XX-XX INFO - Starting training...
Epoch 0 - Training: 100%|██████| 25/25 [XX:XX<XX:XX]
Epoch 0 - Validation: 100%|██████| 8/8 [XX:XX<XX:XX]
...
2025-XX-XX INFO - Training completed!
2025-XX-XX INFO - Best validation AUC: 0.XXXX
2025-XX-XX INFO - Report saved to: reports/xception_sample_.../
```

**Output files created**:
- `experiments/xception_sample_YYYYMMDD_HHMMSS/final_model.pth` - Trained model
- `experiments/.../training_history.json` - Training metrics
- `reports/.../xception_sample_report.pdf` - PDF report
- `reports/.../xception_sample_report.html` - HTML report

---

## **STEP 6: View Training Results** 📊

Check what was created:

```bash
# List experiment folders
ls -lh experiments/

# List report folders
ls -lh reports/

# View training history (JSON)
cat experiments/xception_sample_*/training_history.json | python3 -m json.tool | head -50

# Or open the HTML report
open reports/xception_sample_*/xception_sample_report.html
```

---

## **STEP 7: Evaluate the Model** 🎯

Test the trained model on test set:

```bash
# Find the latest model path
MODEL_PATH=$(ls -t experiments/xception_sample_*/final_model.pth | head -1)

# Evaluate
/usr/bin/python3 main.py evaluate \
    --model_path "$MODEL_PATH" \
    --dataset sample \
    --data_root data/processed \
    --batch_size 4 \
    --cpu
```

**Expected output**:
```
2025-XX-XX INFO - Loading model from: experiments/...
2025-XX-XX INFO - Evaluating on test set...
2025-XX-XX INFO - Test Accuracy: X.XXXX
2025-XX-XX INFO - Test AUC: X.XXXX
```

---

## **STEP 8: Run Inference on Single Image** 🖼️

Test on a single image:

```bash
# Use an image from the test set
TEST_IMAGE=$(find data/processed/sample/images -name "*.jpg" | head -1)

# Find latest model
MODEL_PATH=$(ls -t experiments/xception_sample_*/final_model.pth | head -1)

# Run inference with explainability
/usr/bin/python3 main.py inference \
    --model_path "$MODEL_PATH" \
    --image_path "$TEST_IMAGE" \
    --explainability \
    --cpu
```

**Expected output**:
```
2025-XX-XX INFO - Prediction: REAL (confidence: 0.XX) or FAKE (confidence: 0.XX)
2025-XX-XX INFO - Visualizations saved to: reports/explainability
```

---

## **STEP 9: View Explainability Results** 🔍

See what the model is looking at:

```bash
# Open explainability visualizations
open reports/explainability/*.png

# Or list them
ls -lh reports/explainability/
```

---

## **NEXT STEPS: Train on Real Data** 🌟

Once you've verified everything works, you can use real datasets:

### **Option A: FaceForensics++**

1. **Download dataset** (requires Google Form):
   ```bash
   python scripts/download_datasets.py --dataset faceforensics
   # Follow instructions
   ```

2. **Preprocess**:
   ```bash
   python src/data/preprocessing.py \
       --dataset faceforensics \
       --data_path data/raw/faceforensics \
       --output_path data/processed/faceforensics \
       --face_detector mtcnn
   ```

3. **Train**:
   ```bash
   python main.py train \
       --model xception \
       --dataset faceforensics \
       --data_root data/processed \
       --epochs 50 \
       --batch_size 32
   ```

### **Option B: Different Models**

Try other models on sample data:
```bash
# EfficientNet
python main.py train --model efficientnet_b0 --dataset sample --data_root data/processed --epochs 2 --batch_size 4 --cpu

# Vision Transformer
python main.py train --model vit --dataset sample --data_root data/processed --epochs 2 --batch_size 2 --cpu

# ResNet50
python main.py train --model resnet50 --dataset sample --data_root data/processed --epochs 2 --batch_size 4 --cpu
```

---

## **TROUBLESHOOTING** 🔧

### **Issue: "No module named 'src'"**
```bash
# Make sure you're in the project root
cd /Users/eeshansingh/Desktop/FullStackProjects/Deepfake
```

### **Issue: "CUDA out of memory" or "MPS errors"**
```bash
# Use CPU flag
--cpu

# Or reduce batch size
--batch_size 2
```

### **Issue: "Dataset not found"**
```bash
# Create sample data first
python scripts/create_sample_data.py
```

### **Issue: "Import errors"**
```bash
# Reinstall requirements
/usr/bin/python3 -m pip install -r requirements.txt --user --force-reinstall
```

---

## **QUICK REFERENCE** ⚡

**Training**:
```bash
python main.py train --model <model> --dataset <dataset> --data_root data/processed --epochs <n>
```

**Evaluation**:
```bash
python main.py evaluate --model_path <path> --dataset <dataset> --data_root data/processed
```

**Inference**:
```bash
python main.py inference --model_path <path> --image_path <image> --explainability
```

**Cross-Dataset**:
```bash
python main.py cross-dataset --model <model> --train_dataset <dataset1> --test_datasets <dataset2> <dataset3>
```

---

## **SUCCESS CHECKLIST** ✅

After completing all steps, you should have:

- [x] Dependencies installed
- [x] Sample data created (`data/processed/sample/`)
- [x] Trained model (`experiments/xception_sample_*/final_model.pth`)
- [x] Training report (`reports/xception_sample_*/`)
- [x] Model evaluated successfully
- [x] Inference working with explainability

**If all checked, you're ready to scale up to real datasets!** 🎉

---

## **WHAT'S HAPPENING UNDER THE HOOD** 🔍

1. **Data Loading**: `dataloader.py` loads images with augmentations
2. **Model Forward**: Images pass through XceptionNet layers
3. **Loss Calculation**: Cross-entropy loss computed
4. **Backpropagation**: Gradients update weights
5. **Validation**: Model tested on unseen data
6. **Metrics**: Accuracy, AUC calculated
7. **Saving**: Best model checkpoint saved
8. **Reporting**: Automated PDF/HTML reports generated

---

**Ready to start? Begin with Step 1!** 🚀

