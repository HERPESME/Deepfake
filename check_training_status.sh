#!/bin/bash
cd /Users/eeshansingh/Desktop/FullStackProjects/Deepfake

echo "═══════════════════════════════════════════════════════════"
echo "  📊 ViT TRAINING STATUS"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Check if process is running
if ps aux | grep -q "main.py train.*vit" | grep -v grep; then
    echo "✅ Status: RUNNING"
    ps aux | grep "main.py train.*vit" | grep -v grep | head -1 | awk '{print "   PID: " $2 "\n   CPU: " $3 "%\n   Memory: " $4 "%"}'
else
    echo "❌ Status: NOT RUNNING"
fi

echo ""
echo "📝 Latest Progress:"
tail -3 training_vit.log 2>/dev/null | strings | grep -E "Epoch|Training|Validation" | tail -1 || echo "   (No recent progress found)"

echo ""
echo "📁 Checkpoints:"
ls -lh experiments/vit_celebd_optimized/*.pth 2>/dev/null | wc -l | awk '{print "   " $1 " checkpoint(s) saved"}'

echo ""
echo "═══════════════════════════════════════════════════════════"
