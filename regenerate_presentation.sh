#!/bin/bash
# Script to regenerate presentation PDF with workflow diagram

cd "$(dirname "$0")"

echo "═══════════════════════════════════════════════════════════"
echo "  🔄 Regenerating Presentation PDF"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Check if workflow diagram exists
if [ ! -f "workflow_diagram.png" ]; then
    echo "❌ Error: workflow_diagram.png not found!"
    echo "   Please ensure the file exists in the project root."
    exit 1
fi

echo "✅ Found workflow_diagram.png"
echo ""

# Check for virtual environment
if [ -d "deepfake_env" ]; then
    echo "📦 Activating virtual environment..."
    source deepfake_env/bin/activate
fi

# Check if reportlab is installed
python3 -c "import reportlab" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  reportlab not found. Attempting to install..."
    pip3 install reportlab Pillow 2>&1 | grep -E "(Successfully|ERROR|Requirement)" | head -5
    
    # Check again
    python3 -c "import reportlab" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ Could not install reportlab automatically."
        echo ""
        echo "Please install manually:"
        echo "  pip3 install reportlab Pillow"
        echo ""
        echo "Or if using virtual environment:"
        echo "  source deepfake_env/bin/activate"
        echo "  pip install reportlab Pillow"
        exit 1
    fi
fi

echo "✅ Dependencies available"
echo ""
echo "🔄 Generating presentation PDF..."
echo ""

# Generate PDF
python3 generate_presentation.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Presentation PDF generated successfully!"
    echo "   File: Deepfake_Detection_Project_Presentation.pdf"
    echo ""
    echo "📊 The workflow diagram should now be included on Slide 7"
else
    echo ""
    echo "❌ Error generating PDF. Check the error messages above."
    exit 1
fi



