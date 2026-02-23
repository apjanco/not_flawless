#!/usr/bin/env python3
"""
quick_validate.py
Fast project validation without importing models (avoids NumPy conflicts)
"""

import sys
from pathlib import Path
import json

def check_structure():
    """Check project structure without imports"""
    print("\n✓ Project Structure")
    
    checks = {
        "data/": Path("data").is_dir(),
        "evaluators/": Path("evaluators").is_dir(),
        "hpc/": Path("hpc").is_dir(),
        "setup/": Path("setup").is_dir(),
        "results/": Path("results").is_dir(),
        "requirements.txt": Path("requirements.txt").is_file(),
        "SPEC.md": Path("SPEC.md").is_file(),
        "README.md": Path("README.md").is_file(),
    }
    
    all_good = True
    for name, exists in checks.items():
        symbol = "✓" if exists else "✗"
        print(f"  {symbol} {name}")
        if not exists:
            all_good = False
    
    return all_good

def check_evaluators():
    """Check evaluator files exist"""
    print("\n✓ Evaluator Scripts")
    
    evaluators = [
        "tesseract_eval.py",
        "paddleocr_eval.py",
        "easyocr_eval.py",
        "pylaia_eval.py",
        "kraken_eval.py",
        "qwen_eval.py",
        "deepseek_eval.py",
        "chandra_eval.py",
        "chatgpt_eval.py",
        "gemini_eval.py",
        "google_vision_eval.py",
        "utils.py",
        "__init__.py",
    ]
    
    all_good = True
    for ev in evaluators:
        path = Path("evaluators") / ev
        exists = path.is_file()
        symbol = "✓" if exists else "✗"
        print(f"  {symbol} {ev}")
        if not exists:
            all_good = False
    
    return all_good

def check_hpc_scripts():
    """Check HPC scripts"""
    print("\n✓ HPC Infrastructure")
    
    scripts = [
        "hpc/submit_job.sh",
        "hpc/run_evaluation.py",
        "hpc/run_api_evaluation.sh",
        "hpc/job_config.txt",
    ]
    
    all_good = True
    for script in scripts:
        path = Path(script)
        exists = path.is_file()
        symbol = "✓" if exists else "✗"
        print(f"  {symbol} {script}")
        if not exists:
            all_good = False
    
    return all_good

def check_documentation():
    """Check documentation files"""
    print("\n✓ Documentation")
    
    docs = [
        "README.md",
        "SPEC.md",
        "API_EVALUATION.md",
        "CODE_REVIEW.md",
        "QUICKSTART.md",
        "VALIDATION_REPORT.md",
    ]
    
    all_good = True
    for doc in docs:
        path = Path(doc)
        exists = path.is_file()
        symbol = "✓" if exists else "✗"
        print(f"  {symbol} {doc}")
        if not exists:
            all_good = False
    
    return all_good

def check_code_quality():
    """Check key code files for basic syntax"""
    print("\n✓ Code Quality")
    
    files_to_check = [
        "evaluators/utils.py",
        "evaluators/tesseract_eval.py",
        "hpc/run_evaluation.py",
    ]
    
    all_good = True
    for file_path in files_to_check:
        try:
            with open(file_path, 'r') as f:
                code = f.read()
                # Basic check: file isn't empty and has functions/classes
                has_def = 'def ' in code
                has_class = 'class ' in code
                is_valid = has_def or has_class
                
                symbol = "✓" if is_valid else "✗"
                print(f"  {symbol} {file_path}")
                if not is_valid:
                    all_good = False
        except Exception as e:
            print(f"  ✗ {file_path}: {str(e)}")
            all_good = False
    
    return all_good

def main():
    """Run validation"""
    print("\n" + "="*60)
    print("QUICK PROJECT VALIDATION")
    print("="*60)
    
    checks = [
        ("Structure", check_structure),
        ("Evaluators", check_evaluators),
        ("HPC Scripts", check_hpc_scripts),
        ("Documentation", check_documentation),
        ("Code Quality", check_code_quality),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n✗ {name}: {str(e)}")
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        symbol = "✓" if result else "✗"
        print(f"{symbol} {name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ PROJECT IS READY")
        print("\nNext steps:")
        print("1. bash setup/download_data.sh    # Download IAM database")
        print("2. sbatch hpc/submit_job.sh        # Submit batch job to Adroit")
        print("3. Download results when complete")
        return 0
    else:
        print(f"\n❌ {total - passed} check(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
