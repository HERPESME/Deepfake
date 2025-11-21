#!/bin/bash
# Continue pipeline after preprocessing completes
# Run this once preprocessing finishes

set -e

PROJECT_DIR="/Users/eeshansingh/Desktop/FullStackProjects/Deepfake"
cd "$PROJECT_DIR"

echo "═══════════════════════════════════════════════════════════"
echo "  🔄 CONTINUING PIPELINE (Post-Preprocessing)"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Check if preprocessing completed
if [ ! -d "data/processed/celebd/real" ] || [ ! -d "data/processed/celebd/fake" ]; then
    echo "❌ Preprocessing not complete yet!"
    echo "   Wait for preprocessing to finish first."
    exit 1
fi

echo "✅ Preprocessing complete!"
echo "  Real images: $(find data/processed/celebd/real -name "*.jpg" | wc -l | xargs)"
echo "  Fake images: $(find data/processed/celebd/fake -name "*.jpg" | wc -l | xargs)"
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

echo ""
echo "✅ Evaluation complete!"
echo ""

# STEP 4: Summary
echo "═══════════════════════════════════════════════════════════"
echo "  ✅ PIPELINE COMPLETE!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "📁 Results:"
echo "  Model: $MODEL_PATH"
echo "  Reports: reports/effb0_celebd_auto*/"
echo ""
echo "📊 View report:"
echo "  open reports/effb0_celebd_auto*/effb0_celebd_auto_report.html"

