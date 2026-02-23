#!/bin/bash
# setup/setup_environment.sh
# Install Python dependencies

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "Python Environment Setup"
echo "=========================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

echo "Python version:"
python3 --version
echo ""

# Create virtual environment (optional)
echo "Creating virtual environment..."
cd "$PROJECT_ROOT"
python3 -m venv venv
source venv/bin/activate

echo "Installing Python dependencies..."
pip install --upgrade pip setuptools wheel

# Install requirements
if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
    pip install -r "$PROJECT_ROOT/requirements.txt"
else
    echo "Warning: requirements.txt not found"
    echo "Installing common OCR/HTR dependencies..."
    pip install numpy pandas matplotlib opencv-python pillow
    pip install pytesseract paddleocr easyocr
    pip install keras-ocr
    pip install transformers torch huggingface-hub
fi

echo ""
echo "=========================================="
echo "Environment Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
