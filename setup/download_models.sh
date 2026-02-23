#!/bin/bash
# setup/download_models.sh
# Download and prepare all OCR/HTR models

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_ROOT/models"

echo "=========================================="
echo "Model Download and Preparation Script"
echo "=========================================="

# Create model directories
mkdir -p "$MODELS_DIR/tesseract"
mkdir -p "$MODELS_DIR/paddleocr"
mkdir -p "$MODELS_DIR/easyocr"
mkdir -p "$MODELS_DIR/keras_ocr"
mkdir -p "$MODELS_DIR/trocr"

echo ""
echo "Model Download Instructions"
echo "=========================================="
echo ""

echo "1. Tesseract"
echo "----------------------------------------"
echo "Tesseract is typically installed via system package manager"
echo "Install with: apt-get install tesseract-ocr (Linux)"
echo "             brew install tesseract (macOS)"
echo "Or pre-download for Adroit to: $MODELS_DIR/tesseract/"
echo ""

echo "2. PaddleOCR"
echo "----------------------------------------"
echo "Models are auto-downloaded on first use"
echo "Location: ~/.paddleocr/ (will be cached)"
echo "Pre-download script will run on Adroit if needed"
echo ""

echo "3. EasyOCR"
echo "----------------------------------------"
echo "Models are auto-downloaded on first use"
echo "Location: ~/.EasyOCR/ (will be cached)"
echo "Pre-download script will run on Adroit if needed"
echo ""

echo "4. Keras-OCR"
echo "----------------------------------------"
echo "Models are auto-downloaded on first use"
echo "Location: ~/.keras/ (will be cached)"
echo "Pre-download script will run on Adroit if needed"
echo ""

echo "5. TrOCR (Transformer-based OCR)"
echo "----------------------------------------"
echo "Hugging Face model downloads: microsoft/trocr-*"
echo "Models are auto-downloaded on first use"
echo "Location: ~/.cache/huggingface/ (will be cached)"
echo "Pre-download script will run on Adroit if needed"
echo ""

echo "=========================================="
echo "For Adroit HPC:"
echo "=========================================="
echo "Models will be downloaded on first job submission"
echo "Ensure sufficient disk space in home directory"
echo "Typical requirements:"
echo "  - Tesseract: ~50 MB"
echo "  - PaddleOCR: ~200-500 MB"
echo "  - EasyOCR: ~500 MB - 1 GB"
echo "  - Keras-OCR: ~400 MB"
echo "  - TrOCR: ~500 MB - 1.5 GB"
echo ""
echo "Total estimated space: ~3-4 GB in home directory"
echo ""
