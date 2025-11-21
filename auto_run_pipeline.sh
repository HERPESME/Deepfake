#!/bin/bash
# Fully automated pipeline - waits for preprocessing if needed
# Run this script once and it handles everything

set -e

PROJECT_DIR="/Users/eeshansingh/Desktop/FullStackProjects/Deepfake"
cd "$PROJECT_DIR"

echo "═══════════════════════════════════════════════════════════"
echo "  🤖 FULLY AUTOMATED DEEPFAKE PIPELINE"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Check if preprocessing is needed
NEEDS_PREPROCESSING=true

if [ -d "data/processed/celebd/real" ] && [ -d "data/processed/celebd/fake" ]; then
    REAL_COUNT=$(find data/processed/celebd/real -name "*.jpg" | wc -l | xargs)
    FAKE_COUNT=$(find data/processed/celebd/fake -name "*.jpg" | wc -l | xargs)
    
    if [ "$REAL_COUNT" -gt 100 ] && [ "$FAKE_COUNT" -gt 100 ]; then
        NEEDS_PREPROCESSING=false
        echo "✅ Preprocessing already complete!"
        echo "  Real images: $REAL_COUNT"
        echo "  Fake images: $FAKE_COUNT"
    fi
fi

# STEP 1: Preprocessing (if needed)
if [ "$NEEDS_PREPROCESSING" = true ]; then
    # Check if preprocessing is already running
    if ps aux | grep -v grep | grep -q "preprocessing.py"; then
        echo "⏳ Preprocessing already running, waiting for it to complete..."
        echo ""
        
        # Wait for preprocessing to finish (check every 60 seconds)
        while ps aux | grep -v grep | grep -q "preprocessing.py"; do
            PROGRESS=$(tail -5 /tmp/preprocessing.log 2>/dev/null | grep -oP '\d+%' | tail -1 || echo "0%")
            echo "  Progress: $PROGRESS (checking again in 60s...)"
            sleep 60
        done
        
        echo "✅ Preprocessing process finished!"
        sleep 5  # Wait a bit for cleanup
    else
        echo "═══════════════════════════════════════════════════════════"
        echo "  STEP 1: STARTING PREPROCESSING"
        echo "═══════════════════════════════════════════════════════════"
        echo "⏱️  This will take ~4-6 hours for 6,229 videos..."
        echo ""
        
        python3 src/data/preprocessing.py \
          --dataset celebd \
          --data_path data/raw/celebd \
          --output_path data/processed/celebd \
          --face_detector opencv \
          --frame_stride 60 \
          --max_frames_per_video 3 2>&1 | tee /tmp/preprocessing.log
        
        if [ $? -ne 0 ]; then
            echo "❌ Preprocessing failed!"
            exit 1
        fi
        
        echo ""
        echo "✅ Preprocessing complete!"
    fi
fi

# Verify preprocessing completed successfully
if [ ! -d "data/processed/celebd/real" ] || [ ! -d "data/processed/celebd/fake" ]; then
    echo "❌ Preprocessing incomplete! Check logs."
    exit 1
fi

REAL_COUNT=$(find data/processed/celebd/real -name "*.jpg" | wc -l | xargs)
FAKE_COUNT=$(find data/processed/celebd/fake -name "*.jpg" | wc -l | xargs)

echo ""
echo "📊 Preprocessed Dataset:"
echo "  Real images: $REAL_COUNT"
echo "  Fake images: $FAKE_COUNT"
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

# Final Summary
echo "═══════════════════════════════════════════════════════════"
echo "  🎉 FULL PIPELINE COMPLETE!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "📁 Results saved:"
echo "  Model: $MODEL_PATH"
echo "  Reports: reports/effb0_celebd_auto*/"
echo ""
echo "📊 View report:"
echo "  open reports/effb0_celebd_auto*/effb0_celebd_auto_report.html"
echo ""
echo "✅ All steps completed successfully!"

