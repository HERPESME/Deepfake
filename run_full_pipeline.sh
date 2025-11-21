#!/bin/bash
# Automated Deepfake Detection Pipeline
# Runs preprocessing, training, evaluation, and generates reports

set -e

PROJECT_DIR="/Users/eeshansingh/Desktop/FullStackProjects/Deepfake"
cd "$PROJECT_DIR"

echo "═══════════════════════════════════════════════════════════"
echo "  🚀 AUTOMATED DEEPFAKE DETECTION PIPELINE"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "📊 Dataset: Celeb-DF"
echo "  Real videos: $(ls data/raw/celebd/Celeb-real/*.mp4 2>/dev/null | wc -l | xargs)"
echo "  Fake videos: $(ls data/raw/celebd/Celeb-synthesis/*.mp4 2>/dev/null | wc -l | xargs)"
echo ""

# STEP 1: Preprocessing
echo "═══════════════════════════════════════════════════════════"
echo "  STEP 1: PREPROCESSING (Extract faces from videos)"
echo "═══════════════════════════════════════════════════════════"
echo "⏱️  This will take several hours..."
echo ""

python3 src/data/preprocessing.py \
  --dataset celebd \
  --data_path data/raw/celebd \
  --output_path data/processed/celebd \
  --face_detector opencv \
  --frame_stride 60 \
  --max_frames_per_video 3

if [ $? -ne 0 ]; then
    echo "❌ Preprocessing failed!"
    exit 1
fi

echo ""
echo "✅ Preprocessing complete!"
echo "  Images created: $(find data/processed/celebd -name "*.jpg" | wc -l | xargs)"
echo ""

# STEP 2: Training
echo "═══════════════════════════════════════════════════════════"
echo "  STEP 2: TRAINING EfficientNet-B0"
echo "═══════════════════════════════════════════════════════════"
echo "⏱️  This will take ~4-6 hours (50 epochs)..."
echo ""

python3 main.py train \
  --model efficientnet_b0 \
  --dataset celebd \
  --data_root data/processed \
  --epochs 50 \
  --batch_size 32 \
  --cpu \
  --experiment_name effb0_celebd_auto

if [ $? -ne 0 ]; then
    echo "❌ Training failed!"
    exit 1
fi

echo ""
echo "✅ Training complete!"
echo ""

# STEP 3: Evaluation
echo "═══════════════════════════════════════════════════════════"
echo "  STEP 3: EVALUATION"
echo "═══════════════════════════════════════════════════════════"

MODEL_PATH=$(ls -t experiments/effb0_celebd_auto*/final_model.pth 2>/dev/null | head -1)

if [ -z "$MODEL_PATH" ]; then
    echo "❌ Model not found!"
    exit 1
fi

echo "Using model: $MODEL_PATH"
echo ""

python3 main.py evaluate \
  --model_path "$MODEL_PATH" \
  --dataset celebd \
  --data_root data/processed \
  --batch_size 16 \
  --cpu

if [ $? -ne 0 ]; then
    echo "❌ Evaluation failed!"
    exit 1
fi

echo ""
echo "✅ Evaluation complete!"
echo ""

# STEP 4: Inference Test
echo "═══════════════════════════════════════════════════════════"
echo "  STEP 4: INFERENCE TEST"
echo "═══════════════════════════════════════════════════════════"

TEST_IMAGE=$(find data/processed/celebd -name "*.jpg" 2>/dev/null | head -1)

if [ -n "$TEST_IMAGE" ]; then
    echo "Testing on: $TEST_IMAGE"
    echo ""
    
    python3 main.py inference \
      --model_path "$MODEL_PATH" \
      --image_path "$TEST_IMAGE" \
      --explainability \
      --cpu
    
    echo ""
    echo "✅ Inference complete!"
else
    echo "⚠️  No test images found"
fi

echo ""

# STEP 5: Summary
echo "═══════════════════════════════════════════════════════════"
echo "  ✅ PIPELINE COMPLETE!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "📁 Results saved to:"
echo "  Model: $MODEL_PATH"
echo "  Reports: reports/effb0_celebd_auto*/"
echo ""
echo "📊 View results:"
echo "  open reports/effb0_celebd_auto*/effb0_celebd_auto_report.html"
echo ""
echo "🎉 All steps completed successfully!"

