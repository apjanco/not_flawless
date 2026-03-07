"""
Download Qwen3-VL-8B-Instruct model and processor for offline use on compute nodes.
Run this script on a login node with internet access.
"""
from pathlib import Path
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
# Save to scratch directory for HPC
LOCAL_MODEL_PATH = "/scratch/network/aj7878/not_flawless/models/Qwen3-VL-8B-Instruct"

def download_model():
    """Download and save model and processor locally"""
    print(f"Downloading {MODEL_ID}...")
    
    # Create directory
    Path(LOCAL_MODEL_PATH).mkdir(parents=True, exist_ok=True)
    
    # Download and save processor
    print("Downloading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.save_pretrained(LOCAL_MODEL_PATH)
    print(f"Processor saved to {LOCAL_MODEL_PATH}")
    
    # Download and save model
    print("Downloading model (this may take a while)...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
    )
    model.save_pretrained(LOCAL_MODEL_PATH)
    print(f"Model saved to {LOCAL_MODEL_PATH}")
    
    print("Done! You can now run the evaluator on compute nodes.")

if __name__ == "__main__":
    download_model()
