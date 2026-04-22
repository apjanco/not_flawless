#!/usr/bin/env python3
"""
Test script to verify API key validation in gemini_eval.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_missing_portkey_key():
    """Test that evaluation fails without PORTKEY_API_KEY"""
    print("\n" + "="*60)
    print("TEST 1: Missing PORTKEY_API_KEY")
    print("="*60)
    
    # Clear API keys
    os.environ.pop('PORTKEY_API_KEY', None)
    os.environ.pop('GOOGLE_API_KEY', None)
    
    # Import after clearing env vars
    from evaluators import gemini_eval
    
    result = gemini_eval.evaluate(str(project_root))
    
    if result.get("error"):
        print(f"✅ PASSED: Evaluation correctly stopped")
        print(f"   Error message: {result['error']}")
        return True
    else:
        print(f"❌ FAILED: Evaluation should have stopped but got: {result}")
        return False


def test_missing_google_key():
    """Test that evaluation fails without GOOGLE_API_KEY"""
    print("\n" + "="*60)
    print("TEST 2: Missing GOOGLE_API_KEY (with PORTKEY_API_KEY set)")
    print("="*60)
    
    # Set only Portkey key
    os.environ['PORTKEY_API_KEY'] = 'test-portkey-key-12345'
    os.environ.pop('GOOGLE_API_KEY', None)
    
    # Reload module to pick up new env vars
    import importlib
    from evaluators import gemini_eval
    importlib.reload(gemini_eval)
    
    result = gemini_eval.evaluate(str(project_root))
    
    if result.get("error"):
        print(f"✅ PASSED: Evaluation correctly stopped")
        print(f"   Error message: {result['error']}")
        return True
    else:
        print(f"❌ FAILED: Evaluation should have stopped but got: {result}")
        return False


def test_both_keys_set():
    """Test that evaluation proceeds when both keys are set (will fail later due to no data)"""
    print("\n" + "="*60)
    print("TEST 3: Both API keys set (should proceed to data loading)")
    print("="*60)
    
    # Set both keys
    os.environ['PORTKEY_API_KEY'] = 'test-portkey-key-12345'
    os.environ['GOOGLE_API_KEY'] = 'test-google-key-67890'
    
    # Reload module
    import importlib
    from evaluators import gemini_eval
    importlib.reload(gemini_eval)
    
    result = gemini_eval.evaluate(str(project_root))
    
    # At this point, it might fail due to:
    # 1. Invalid API keys
    # 2. Missing data
    # 3. Network issues
    # But it should NOT fail due to missing env vars
    
    if result.get("error"):
        error = result["error"]
        # Check if error is about missing keys (which would be BAD)
        if "not set" in error or "not configured" in error:
            print(f"❌ FAILED: Still getting API key validation error: {error}")
            return False
        else:
            # Other errors are OK - we got past validation
            print(f"✅ PASSED: Got past API key validation")
            print(f"   (Later error is expected with test keys: {error})")
            return True
    else:
        print(f"✅ PASSED: No API key validation errors")
        return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("GEMINI EVALUATOR - API KEY VALIDATION TESTS")
    print("="*60)
    
    results = []
    
    # Run tests
    try:
        results.append(("Test 1: Missing PORTKEY_API_KEY", test_missing_portkey_key()))
    except Exception as e:
        print(f"❌ Test 1 ERROR: {e}")
        results.append(("Test 1: Missing PORTKEY_API_KEY", False))
    
    try:
        results.append(("Test 2: Missing GOOGLE_API_KEY", test_missing_google_key()))
    except Exception as e:
        print(f"❌ Test 2 ERROR: {e}")
        results.append(("Test 2: Missing GOOGLE_API_KEY", False))
    
    try:
        results.append(("Test 3: Both keys set", test_both_keys_set()))
    except Exception as e:
        print(f"❌ Test 3 ERROR: {e}")
        results.append(("Test 3: Both keys set", False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n✅ All validation tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)
