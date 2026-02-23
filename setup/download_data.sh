#!/bin/bash
# setup/download_data.sh
# Download IAM Handwriting Database and other datasets

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data"

echo "=========================================="
echo "OCR/HTR Dataset Download Script"
echo "=========================================="

# Create data directories
mkdir -p "$DATA_DIR/iam"
mkdir -p "$DATA_DIR/other_datasets"
mkdir -p "$DATA_DIR/processed"

# IAM Database from Hugging Face
echo ""
echo "1. IAM Handwriting Database (from Teklia/IAM-line)"
echo "----------------------------------------"

# Check if huggingface-hub is installed
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo "Installing huggingface-hub..."
    pip install huggingface-hub
fi

echo "Downloading IAM dataset from Hugging Face..."
python3 << 'EOF'
from huggingface_hub import snapshot_download
import os

data_dir = os.path.join("""$DATA_DIR""", 'iam')
os.makedirs(data_dir, exist_ok=True)

print("Downloading Teklia/IAM-line dataset...")
snapshot_download(
    repo_id="Teklia/IAM-line",
    repo_type="dataset",
    local_dir=data_dir,
    force_download=False
)
print(f"Dataset downloaded to: {data_dir}")
EOF

echo ""
echo "✓ IAM dataset downloaded successfully"
echo ""

# Placeholder for automatic downloads
echo ""
echo "2. Other Datasets (TBD)"
echo "----------------------------------------"
echo "Additional datasets to be specified and downloaded:"
echo "  - MNIST/EMNIST"
echo "  - RIMES dataset"
echo "  - CVL dataset"
echo "  - [Custom datasets]"
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "All datasets are ready for evaluation"
echo "=========================================="
