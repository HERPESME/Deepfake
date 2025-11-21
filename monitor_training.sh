#!/bin/bash
# Monitor XceptionNet Training Progress

echo "═══════════════════════════════════════════════════════════"
echo "  📊 XCEPTIONNET TRAINING MONITOR"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Check if training is running
if ps aux | grep -q "main.py train.*xception" | grep -v grep; then
    echo "✅ Training Status: RUNNING"
    PID=$(ps aux | grep "main.py train.*xception" | grep -v grep | awk '{print $2}' | head -1)
    echo "   Process ID: $PID"
else
    echo "❌ Training Status: NOT RUNNING"
fi

echo ""
echo "📊 Latest Training Progress:"
echo "───────────────────────────────────────────────────────────"
tail -20 /tmp/xception_training.log 2>/dev/null | grep -E "(Epoch|INFO|Training|Validation|Loss|Acc|AUC)" | tail -10 || echo "Waiting for training to start..."

echo ""
echo "📁 Checkpoints:"
ls -lh experiments/xception_celebd/*.pth 2>/dev/null | tail -3 || echo "No checkpoints yet"

echo ""
echo "📝 Full Log:"
echo "   tail -f /tmp/xception_training.log"
echo ""

