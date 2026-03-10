"""
combine_results.py
Combines JSONL results from all evaluators into a unified dataset.

Loads the IAM dataset, merges predictions and metrics from each model,
and saves/pushes the combined dataset.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict, concatenate_datasets

def get_results_dir() -> Path:
    """Get the results directory path."""
    return Path(__file__).parent / "results"


def discover_result_files() -> Dict[str, Path]:
    """
    Discover all JSONL result files in the results directory.
    
    Returns:
        Dictionary mapping model_name -> jsonl_path
    """
    results_dir = get_results_dir()
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return {}
    
    result_files = {}
    for jsonl_path in results_dir.glob("*_results.jsonl"):
        # Extract model name from filename (e.g., "chatgpt-vision_results.jsonl" -> "chatgpt-vision")
        model_name = jsonl_path.stem.replace("_results", "")
        result_files[model_name] = jsonl_path
    
    return result_files


def load_jsonl_results(jsonl_path: Path) -> Dict[int, Dict[str, Any]]:
    """
    Load JSONL results from a file.
    
    Args:
        jsonl_path: Path to the JSONL file
    
    Returns:
        Dictionary mapping index -> result dict
    """
    results = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                idx = record.get("index")
                if idx is not None:
                    results[idx] = record
    
    return results


def sanitize_column_name(model_name: str) -> str:
    """Convert model name to valid column name."""
    return model_name.replace("-", "_").replace(".", "_")


def combine_results(
    dataset: Dataset,
    models: List[str] = None,
    include_logprobs: bool = False
) -> Dataset:
    """
    Combine evaluation results into the dataset.
    
    Results are stored as a list of model predictions per sample:
    {
        "image": ...,
        "text": "ground truth",
        "model_results": [
            {"model": "chatgpt_gpt4o", "predicted": "...", "cer": 0.05, ...},
            {"model": "gemini_direct", "predicted": "...", "cer": 0.03, ...}
        ]
    }
    
    Args:
        dataset: Original IAM dataset
        models: List of model names to include (default: all discovered)
        include_logprobs: Whether to include raw logprobs data (can be large)
    
    Returns:
        Dataset with model_results column containing all predictions
    """
    # Discover available result files
    available_files = discover_result_files()
    
    if not available_files:
        print("No result files found in results directory!")
        return dataset
    
    print(f"\nDiscovered {len(available_files)} result files:")
    for model_name, path in available_files.items():
        print(f"  - {model_name}: {path.name}")
    
    # Filter to requested models if specified
    if models:
        available_files = {k: v for k, v in available_files.items() if k in models}
        if not available_files:
            print(f"No matching result files for requested models: {models}")
            return dataset
    
    print(f"\nLoading results from {len(available_files)} models...")
    
    # Load all results
    all_results = {}
    for model_name, jsonl_path in available_files.items():
        results = load_jsonl_results(jsonl_path)
        if results:
            all_results[model_name] = results
            print(f"  [OK] Loaded {len(results)} results from {model_name}")
    
    if not all_results:
        print("No results loaded!")
        return dataset
    
    print(f"\nCombining results from {len(all_results)} models...")
    
    # Build model_results list for each sample
    model_results_column = []
    
    for idx in range(len(dataset)):
        sample_results = []
        for model_name, results in all_results.items():
            result = results.get(idx, {})
            
            model_result = {
                "model": model_name,
                "predicted": result.get("predicted_text"),
                "cer": result.get("cer"),
                "wer": result.get("wer"),
                "inference_time": result.get("inference_time"),
                "error": result.get("error"),
                # Semantic metrics
                "semantic_error": result.get("semantic_error"),
                "kl_divergence": result.get("kl_divergence"),
                "entropy": result.get("entropy"),
                "mean_logprob": result.get("mean_logprob"),
                "confidence": result.get("confidence"),
            }
            sample_results.append(model_result)
        
        model_results_column.append(sample_results)
    
    # Add the nested column
    dataset = dataset.add_column("model_results", model_results_column)
    
    print(f"\nAdded model_results column with {len(all_results)} models: {list(all_results.keys())}")
    
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Combine OCR evaluation results into a unified dataset"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="local",
        choices=["local", "hub"],
        help="Source of IAM dataset: 'local' (from disk) or 'hub' (from HuggingFace)"
    )
    parser.add_argument(
        "--local-path",
        type=str,
        default="/scratch/network/aj7878/not_flawless/data/iam",
        help="Path to local IAM dataset (if source=local)"
    )
    parser.add_argument(
        "--hub-dataset",
        type=str,
        default="Teklia/IAM-line",
        help="HuggingFace dataset name (if source=hub)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./combined_results",
        help="Output path for saving combined dataset"
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        help="Push to HuggingFace Hub with this repo name (e.g., 'username/dataset-name')"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="List of model names to include (default: all available)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split to use (e.g., 'test'). If None, uses all splits."
    )
    
    args = parser.parse_args()
    
    # Determine source: use hub if --hub-dataset provided, or if local path doesn't exist
    local_path = Path(args.local_path)
    use_hub = args.source == "hub" or not local_path.exists()
    
    # Load source dataset
    print("Loading source dataset...")
    if use_hub:
        print(f"  From HuggingFace Hub: {args.hub_dataset}")
        dataset = load_dataset(args.hub_dataset)
    else:
        print(f"  From local path: {args.local_path}")
        dataset = load_from_disk(args.local_path)
    
    # Handle DatasetDict vs Dataset
    if isinstance(dataset, DatasetDict):
        if args.split:
            print(f"  Using split: {args.split}")
            dataset = dataset[args.split]
        else:
            # Combine all splits
            splits = list(dataset.keys())
            print(f"  Available splits: {splits}")
            print(f"  Combining all splits...")
            datasets_to_combine = [dataset[split] for split in splits]
            dataset = concatenate_datasets(datasets_to_combine)
            print(f"  Combined {len(splits)} splits")
    
    print(f"  Loaded {len(dataset)} samples")
    
    # Combine results
    combined = combine_results(dataset, models=args.models)
    
    # Save to disk
    print(f"\nSaving combined dataset to: {args.output}")
    combined.save_to_disk(args.output)
    print(f"  [OK] Saved to disk: {args.output}")
    
    # Push to hub if requested
    if args.push_to_hub:
        print(f"\nPushing to HuggingFace Hub: {args.push_to_hub}")
        combined.push_to_hub(args.push_to_hub, private=True)
        print("  [OK] Pushed to Hub")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total samples: {len(combined)}")
    print(f"Columns: {combined.column_names}")
    
    # Show sample
    print("\nSample record (index 0):")
    sample = combined[0]
    print(f"  text: {sample.get('text', 'N/A')}")
    print(f"  image: {type(sample.get('image', 'N/A'))}")
    print(f"  model_results ({len(sample.get('model_results', []))} models):")
    for mr in sample.get('model_results', []):
        print(f"    - {mr.get('model')}: predicted='{mr.get('predicted', '')[:50]}...' cer={mr.get('cer')}")


if __name__ == "__main__":
    main()
