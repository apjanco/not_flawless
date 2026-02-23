#!/usr/bin/env python3
"""
test_setup.py
Validation script to test project setup and identify issues
"""

import sys
import os
from pathlib import Path

def check_project_structure():
    """Verify project directory structure"""
    print("\n" + "="*60)
    print("PROJECT STRUCTURE CHECK")
    print("="*60)
    
    project_root = Path.cwd()
    required_dirs = [
        "data",
        "evaluators",
        "hpc",
        "setup",
        "results"
    ]
    
    required_files = [
        "requirements.txt",
        "SPEC.md",
        "README.md",
        "API_EVALUATION.md",
        ".gitignore",
    ]
    
    all_good = True
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        status = "✓" if dir_path.exists() else "✗"
        print(f"{status} {dir_name}/")
        if not dir_path.exists():
            all_good = False
    
    for file_name in required_files:
        file_path = project_root / file_name
        status = "✓" if file_path.exists() else "✗"
        print(f"{status} {file_name}")
        if not file_path.exists():
            all_good = False
    
    return all_good

def check_evaluators():
    """Check if all evaluator scripts exist"""
    print("\n" + "="*60)
    print("EVALUATOR SCRIPTS CHECK")
    print("="*60)
    
    project_root = Path.cwd()
    evaluators_dir = project_root / "evaluators"
    
    required_evaluators = [
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
    for evaluator in required_evaluators:
        file_path = evaluators_dir / evaluator
        status = "✓" if file_path.exists() else "✗"
        print(f"{status} {evaluator}")
        if not file_path.exists():
            all_good = False
    
    return all_good

def check_python_imports():
    """Check if critical imports work"""
    print("\n" + "="*60)
    print("PYTHON IMPORTS CHECK")
    print("="*60)
    
    imports_to_test = [
        ("json", "json"),
        ("pathlib", "pathlib"),
        ("datetime", "datetime"),
        ("typing", "typing"),
        ("numpy", "numpy"),
    ]
    
    all_good = True
    for module_name, package_name in imports_to_test:
        try:
            __import__(package_name)
            print(f"✓ {module_name}")
        except ImportError as e:
            print(f"✗ {module_name}: {str(e)}")
            all_good = False
    
    return all_good

def check_hpc_scripts():
    """Check HPC-related scripts"""
    print("\n" + "="*60)
    print("HPC SCRIPTS CHECK")
    print("="*60)
    
    project_root = Path.cwd()
    hpc_dir = project_root / "hpc"
    
    required_scripts = [
        "submit_job.sh",
        "run_evaluation.py",
        "run_api_evaluation.sh",
        "job_config.txt",
    ]
    
    all_good = True
    for script in required_scripts:
        file_path = hpc_dir / script
        status = "✓" if file_path.exists() else "✗"
        print(f"{status} {script}")
        if not file_path.exists():
            all_good = False
    
    return all_good

def check_utils_functions():
    """Verify key utility functions exist"""
    print("\n" + "="*60)
    print("UTILITY FUNCTIONS CHECK")
    print("="*60)
    
    try:
        from evaluators.utils import (
            get_project_root,
            get_data_dir,
            get_results_dir,
            load_iam_data,
            character_error_rate,
            word_error_rate,
            save_metrics,
            save_results_jsonl,
            log_info,
            log_error,
            log_warning,
        )
        
        functions = [
            "get_project_root",
            "get_data_dir",
            "get_results_dir",
            "load_iam_data",
            "character_error_rate",
            "word_error_rate",
            "save_metrics",
            "save_results_jsonl",
            "log_info",
            "log_error",
            "log_warning",
        ]
        
        for func_name in functions:
            print(f"✓ {func_name}()")
        
        return True
    except Exception as e:
        print(f"✗ Error importing utilities: {str(e)}")
        return False

def check_data_loading():
    """Test data loading function"""
    print("\n" + "="*60)
    print("DATA LOADING CHECK")
    print("="*60)
    
    try:
        from evaluators.utils import load_iam_data, get_data_dir
        
        data_dir = get_data_dir()
        iam_dir = data_dir / "iam"
        
        if not iam_dir.exists():
            print(f"⚠ IAM data directory not found: {iam_dir}")
            print("  Run: bash setup/download_data.sh")
            return False
        
        print(f"✓ IAM data directory exists: {iam_dir}")
        
        # Check structure
        if (iam_dir / "splits").exists():
            print("✓ Found 'splits' subdirectory structure")
        elif (iam_dir / "train").exists() or (iam_dir / "test").exists():
            print("✓ Found alternative data split structure")
        else:
            print("⚠ Could not find standard splits structure")
            print("  Expected: data/iam/splits/[train|val|test]/ or data/iam/[train|val|test]/")
        
        return True
    except Exception as e:
        print(f"✗ Error checking data loading: {str(e)}")
        return False

def main():
    """Run all checks"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "OCR/HTR PROJECT VALIDATION".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    checks = [
        ("Project Structure", check_project_structure),
        ("Evaluator Scripts", check_evaluators),
        ("Python Imports", check_python_imports),
        ("HPC Scripts", check_hpc_scripts),
        ("Utility Functions", check_utils_functions),
        ("Data Loading", check_data_loading),
    ]
    
    results = {}
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"\n✗ Error in {check_name}: {str(e)}")
            results[check_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for check_name, result in results.items():
        status = "✓" if result else "✗"
        print(f"{status} {check_name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All checks passed! Project is ready.")
        return 0
    else:
        print(f"\n✗ {total - passed} check(s) failed. Please review above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
