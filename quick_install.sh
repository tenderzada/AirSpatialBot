#!/bin/bash

# AirSpatialBot Hierarchical UAV System - Quick Installation Script
# 一键完成环境准备、依赖安装和LLaVA集成

set -e  # Exit on error

echo "=========================================="
echo "AirSpatialBot - Quick Installation"
echo "=========================================="
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Check Python version
echo "Step 1/5: Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
    echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION found"
else
    echo -e "${RED}✗${NC} Python 3.8+ required, found $PYTHON_VERSION"
    exit 1
fi

# Step 2: Check CUDA
echo ""
echo "Step 2/5: Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo -e "${GREEN}✓${NC} CUDA $CUDA_VERSION detected"

    # Determine PyTorch CUDA version
    if [[ "$CUDA_VERSION" == 12.* ]]; then
        TORCH_CUDA="cu121"
    else
        TORCH_CUDA="cu118"
    fi
else
    echo -e "${YELLOW}⚠${NC} No CUDA detected, will install CPU-only PyTorch (slow)"
    TORCH_CUDA="cpu"
fi

# Step 3: Install dependencies
echo ""
echo "Step 3/5: Installing Python dependencies..."
echo "  (This may take several minutes)"

if [ "$TORCH_CUDA" == "cpu" ]; then
    pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu
else
    pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/$TORCH_CUDA
fi

pip install -r requirements.txt -q

echo -e "${GREEN}✓${NC} Dependencies installed"

# Step 4: Clone and install LLaVA
echo ""
echo "Step 4/5: Setting up LLaVA..."

if [ -d "LLaVA" ]; then
    echo "  LLaVA directory exists, checking installation..."
    if python -c "from llava.model.builder import load_pretrained_model" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} LLaVA already installed"
    else
        echo "  Reinstalling LLaVA..."
        cd LLaVA
        pip install -e . -q
        cd ..
        echo -e "${GREEN}✓${NC} LLaVA installed"
    fi
else
    echo "  Cloning LLaVA repository..."
    git clone https://github.com/haotian-liu/LLaVA.git -q
    cd LLaVA
    pip install -e . -q
    cd ..
    echo -e "${GREEN}✓${NC} LLaVA cloned and installed"
fi

# Step 5: Verify installation
echo ""
echo "Step 5/5: Verifying installation..."

CHECKS_PASSED=0
CHECKS_TOTAL=5

# Check PyTorch
if python -c "import torch; print(torch.__version__)" &> /dev/null; then
    TORCH_VER=$(python -c "import torch; print(torch.__version__)")
    echo -e "  ${GREEN}✓${NC} PyTorch $TORCH_VER"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
else
    echo -e "  ${RED}✗${NC} PyTorch"
fi

# Check CUDA availability in PyTorch
if python -c "import torch; assert torch.cuda.is_available()" &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} CUDA available in PyTorch"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
else
    echo -e "  ${YELLOW}⚠${NC} CUDA not available in PyTorch (CPU mode)"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))  # Not critical
fi

# Check LLaVA
if python -c "from llava.model.builder import load_pretrained_model" &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} LLaVA"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
else
    echo -e "  ${RED}✗${NC} LLaVA"
fi

# Check MAC module
if python -c "from hierarchical_uav.mac_memory import NeuralMemory" &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} MAC Memory module"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
else
    echo -e "  ${RED}✗${NC} MAC Memory module"
fi

# Check Communication module
if python -c "from hierarchical_uav.communication import SelfMatchingGate" &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} Communication module"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
else
    echo -e "  ${RED}✗${NC} Communication module"
fi

echo ""
echo "Verification: $CHECKS_PASSED/$CHECKS_TOTAL checks passed"

if [ $CHECKS_PASSED -eq $CHECKS_TOTAL ]; then
    echo ""
    echo -e "${GREEN}=========================================="
    echo "✓ Installation Complete!"
    echo "==========================================${NC}"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Download data:"
    echo "   huggingface-cli download erenzhou/AirSpatial --repo-type dataset --local-dir ./airspatial_data"
    echo "   ./setup_data.sh"
    echo ""
    echo "2. Download models:"
    echo "   huggingface-cli download erenzhou/AirSpatialBot --local-dir ./models/AirSpatialBot"
    echo "   huggingface-cli download openai/clip-vit-large-patch14-336 --local-dir ./models/clip-vit-large-patch14-336"
    echo ""
    echo "3. Run quick test:"
    echo "   python hierarchical_uav/mac_memory/neural_memory.py"
    echo ""
    echo "4. See DEPLOYMENT_GUIDE.md for detailed instructions"
    echo ""
else
    echo ""
    echo -e "${RED}=========================================="
    echo "⚠ Installation Incomplete"
    echo "==========================================${NC}"
    echo ""
    echo "Some components failed to install."
    echo "Please check the errors above and:"
    echo "  1. Ensure you have Python 3.8+"
    echo "  2. Try: pip install -r requirements.txt"
    echo "  3. Run: ./install_llava.sh"
    echo ""
fi
