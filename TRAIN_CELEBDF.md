# Step-by-Step: Train on Celeb-DF Dataset

## 🚀 **Complete Pipeline**

Follow these steps in order to train on your Celeb-DF dataset and get results.

---

## **STEP 1: Preprocess Celeb-DF Dataset** ⏱️ Several hours (depends on video count)

This extracts faces from videos and saves them as images.

### **Option A: Fast Preprocessing (Recommended for 5000+ videos)**

```bash
cd /Users/eeshansingh/Desktop/FullStackProjects/Deepfake

python src/data/preprocessing.py \
  --dataset celebd \
  --data_path data/raw/celebd \
  --output_path data/processed/celebd \
  --face_detector opencv \
  --frame_stride 60 \
  --max_frames_per_video 3
```

**Settings:**
- `--face_detector opencv`: Fastest (vs mtcnn/dlib)
- `--frame_stride 60`: Extract every 60th frame (saves disk/time)
- `--max_frames_per_video 3`: Only 3 frames per video

**Expected time:** 2-4 hours for ~5000 videos

### **Option B: Balanced Preprocessing (Better quality)**

```bash
python src/data/preprocessing.py \
  --dataset celebd \
  --data_path data/raw/celebd \
  --output_path data/processed/celebd \
  --face_detector opencv \
  --frame_stride 30 \
  --max_frames_per_video 10
```

**Expected time:** 4-8 hours for ~5000 videos

### **Option C: High Quality (Best results, slowest)**

```bash
python src/data/preprocessing.py \
  --dataset celebd \
  --data_path data/raw/celebd \
  --output_path data/processed/celebd \
  --face_detector mtcnn \
  --frame_stride 10 \
  --max_frames_per_video 50
```

**Expected time:** 8-12+ hours for ~5000 videos

---

## **STEP 2: Verify Preprocessed Data** ✅

Check that preprocessing worked:

```bash
# Check processed images
ls data/processed/celebd/real/*.jpg | wc -l
ls data/processed/celebd/fake/*.jpg | wc -l

# Check splits were created
ls data/processed/celebd/splits/
cat data/processed/celebd/splits/train.txt | wc -l
```

**Expected:** Should see thousands of `.jpg` files and split files created.

---

## **STEP 3: Train Model** ⏱️ 2-6 hours (depends on epochs)

### **Quick Training (10 epochs)**

```bash
python main.py train \
  --model efficientnet_b0 \
  --dataset celebd \
  --data_root data/processed \
  --epochs 10 \
  --batch_size 16 \
  --cpu
```

**Expected time:** ~2-3 hours

### **Standard Training (50 epochs)**

```bash
python main.py train \
  --model efficientnet_b0 \
  --dataset celebd \
  --data_root data/processed \
  --epochs 50 \
  --batch_size 32 \
  --cpu
```

**Expected time:** ~6-8 hours

### **Full Training with Different Models**

```bash
# EfficientNet-B0 (Best balance)
python main.py train \
  --model efficientnet_b0 \
  --dataset celebd \
  --data_root data/processed \
  --epochs 50 \
  --batch_size 32 \
  --cpu \
  --experiment_name effb0_celebd

# ResNet50 (Alternative)
python main.py train \
  --model resnet50 \
  --dataset celebd \
  --data_root data/processed \
  --epochs 50 \
  --batch_size 32 \
  --cpu \
  --experiment_name resnet50_celebd

# XceptionNet (Baseline)
python main.py train \
  --model xception \
  --dataset celebd \
  --data_root data/processed \
  --epochs 50 \
  --batch_size 16 \
  --cpu \
  --experiment_name xception_celebd
```

---

## **STEP 4: Evaluate Model** ✅

Test your trained model:

```bash
# Find latest model
MODEL_PATH=$(ls -t experiments/effb0_celebd*/final_model.pth 2>/dev/null | head -1)

# Evaluate
python main.py evaluate \
  --model_path "$MODEL_PATH" \
  --dataset celebd \
  --data_root data/processed \
  --batch_size 16 \
  --cpu
```

**Output:** Accuracy, AUC, Precision, Recall, F1-Score

---

## **STEP 5: Run Inference** 🖼️

Test on a single image:

```bash
# Get an image from test set
TEST_IMAGE=$(find data/processed/celebd -name "*.jpg" | head -1)

# Run inference
python main.py inference \
  --model_path "$MODEL_PATH" \
  --image_path "$TEST_IMAGE" \
  --explainability \
  --cpu
```

**Output:** 
- Prediction: REAL or FAKE
- Confidence score
- Visualization (with --explainability)

---

## **STEP 6: View Results** 📊

### **Training Reports**

```bash
# Open HTML report
open reports/effb0_celebd_*/effb0_celebd_report.html

# Or view PDF
open reports/effb0_celebd_*/effb0_celebd_report.pdf
```

**Reports contain:**
- Training curves (loss, accuracy over epochs)
- Confusion matrix
- ROC curves
- Performance metrics
- Model comparison (if multiple models)

### **Model Files**

```bash
# Trained models
ls -lh experiments/effb0_celebd*/final_model.pth

# Training history
cat experiments/effb0_celebd*/training_history.json | python -m json.tool

# Final results
cat experiments/effb0_celebd*/final_results.json | python -m json.tool
```

---

## **STEP 7: Compare Models** 📈

If you trained multiple models:

```bash
# Compare all experiments
for exp in experiments/*_celebd*/; do
  echo "=== $(basename $exp) ==="
  cat "$exp/final_results.json" | python -m json.tool | grep -E "(accuracy|auc|f1)" | head -3
  echo ""
done
```

---

## **QUICK REFERENCE: All Commands**

```bash
# 1. Preprocess (fast mode)
python src/data/preprocessing.py \
  --dataset celebd \
  --data_path data/raw/celebd \
  --output_path data/processed/celebd \
  --face_detector opencv \
  --frame_stride 60 \
  --max_frames_per_video 3

# 2. Train
python main.py train \
  --model efficientnet_b0 \
  --dataset celebd \
  --data_root data/processed \
  --epochs 50 \
  --batch_size 32 \
  --cpu

# 3. Evaluate
python main.py evaluate \
  --model_path experiments/effb0_celebd*/final_model.pth \
  --dataset celebd \
  --data_root data/processed \
  --cpu

# 4. Inference
python main.py inference \
  --model_path experiments/effb0_celebd*/final_model.pth \
  --image_path <your_image.jpg> \
  --explainability \
  --cpu
```

---

## **TROUBLESHOOTING**

### **Issue: Preprocessing too slow**
- Use `--face_detector opencv` (fastest)
- Increase `--frame_stride` (e.g., 90 instead of 30)
- Decrease `--max_frames_per_video` (e.g., 2 instead of 10)

### **Issue: Out of memory during training**
- Reduce `--batch_size` (e.g., 8 or 4)
- Use `--cpu` flag
- Process fewer frames per video

### **Issue: Dataset not found**
- Check: `ls data/processed/celebd/real/` should show images
- Check: `ls data/processed/celebd/splits/` should show train.txt, val.txt, test.txt

---

## **EXPECTED RESULTS**

With Celeb-DF dataset:
- **Training Accuracy**: 85-95% (after 50 epochs)
- **Validation AUC**: 0.90-0.98
- **Test Accuracy**: 80-90%

These are realistic targets for Celeb-DF with EfficientNet-B0.

---

**🎉 Good luck! Start with Step 1 (Preprocessing)!**

