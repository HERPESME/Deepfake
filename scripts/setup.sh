#!/bin/bash

# Script to setup the Deepfake Detection project environment
# Run this after deleting the old virtual environment

set -e  # Exit on error

echo "=========================================="
echo "🚀 DEEPFAKE DETECTION - SETUP SCRIPT"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo -e "${RED}❌ Error: main.py not found${NC}"
    echo "Please run this script from the Deepfake project root directory"
    exit 1
fi

# Step 1: Remove old virtual environment
echo -e "${YELLOW}Step 1: Removing old virtual environment...${NC}"
if [ -d "deepfake_env" ]; then
    rm -rf deepfake_env
    echo -e "${GREEN}✅ Old environment removed${NC}"
else
    echo -e "${GREEN}✅ No old environment to remove${NC}"
fi
echo ""

# Step 2: Create new virtual environment
echo -e "${YELLOW}Step 2: Creating new virtual environment...${NC}"

# Try to find Python with SSL support
if command -v /opt/homebrew/bin/python3 &> /dev/null; then
    PYTHON_BIN="/opt/homebrew/bin/python3"
    echo "Using Homebrew Python: $PYTHON_BIN"
elif command -v /usr/local/bin/python3 &> /dev/null; then
    PYTHON_BIN="/usr/local/bin/python3"
    echo "Using local Python: $PYTHON_BIN"
elif command -v python3 &> /dev/null; then
    PYTHON_BIN="python3"
    echo "Using system Python: $PYTHON_BIN"
else
    echo -e "${RED}❌ Error: Python 3 not found${NC}"
    echo "Please install Python 3.8+ first"
    exit 1
fi

# Verify Python version
PYTHON_VERSION=$($PYTHON_BIN --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Create virtual environment
$PYTHON_BIN -m venv deepfake_env
echo -e "${GREEN}✅ Virtual environment created${NC}"
echo ""

# Step 3: Activate and verify
echo -e "${YELLOW}Step 3: Activating environment and verifying SSL...${NC}"
source deepfake_env/bin/activate

# Check SSL
if python -c "import ssl" 2>/dev/null; then
    echo -e "${GREEN}✅ SSL support verified${NC}"
else
    echo -e "${RED}❌ SSL not working - Python may need reinstallation${NC}"
    exit 1
fi
echo ""

# Step 4: Upgrade pip
echo -e "${YELLOW}Step 4: Upgrading pip...${NC}"
python -m pip install --upgrade pip
echo -e "${GREEN}✅ Pip upgraded${NC}"
echo ""

# Step 5: Install PyTorch
echo -e "${YELLOW}Step 5: Installing PyTorch...${NC}"
echo "This may take several minutes..."

# Detect system architecture
if [[ $(uname -m) == 'arm64' ]]; then
    # Apple Silicon Mac
    echo "Detected Apple Silicon Mac"
    pip install torch torchvision torchaudio
else
    # Intel Mac or Linux
    echo "Detected Intel/x86_64 system"
    echo "For CUDA support, modify this script to install CUDA version"
    pip install torch torchvision torchaudio
fi
echo -e "${GREEN}✅ PyTorch installed${NC}"
echo ""

# Step 6: Install other dependencies
echo -e "${YELLOW}Step 6: Installing other dependencies...${NC}"
echo "This may take several minutes..."
pip install -r requirements.txt
echo -e "${GREEN}✅ Dependencies installed${NC}"
echo ""

# Step 7: Verify installation
echo -e "${YELLOW}Step 7: Running setup verification...${NC}"
echo ""
python setup_check.py
echo ""

# Step 8: Create directories
echo -e "${YELLOW}Step 8: Creating project directories...${NC}"
mkdir -p data/raw data/processed data/splits experiments reports logs
echo -e "${GREEN}✅ Directories created${NC}"
echo ""

# Done
echo "=========================================="
echo -e "${GREEN}✅ SETUP COMPLETE!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate environment: source deepfake_env/bin/activate"
echo "  2. Create sample data: python scripts/create_sample_data.py"
echo "  3. Test training: python main.py train --model xception --dataset sample --epochs 5"
echo ""
echo "See QUICKSTART.md for detailed instructions"
