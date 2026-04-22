# API Key Validation for Gemini Evaluator

## Overview

The Gemini evaluator (`gemini_eval.py`) now enforces **strict validation** of both required API keys before any evaluation proceeds. If either key is missing, the script stops immediately with clear error messages.

## Required Environment Variables

### 1. **PORTKEY_API_KEY** (Required)
- Used for Portkey API calls to route requests to Gemini
- Required for all evaluations
- Get from: https://www.portkey.ai/

```bash
export PORTKEY_API_KEY='your-portkey-api-key'
```

### 2. **GOOGLE_API_KEY** (Required)
- Used for Google Gemini API tokenizer access
- Required for token-level semantic error analysis
- Get from: https://aistudio.google.com/apikey

```bash
export GOOGLE_API_KEY='your-google-api-key'
```

## Validation Flow

```
┌─────────────────────────────────────┐
│ evaluate() called                   │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ Check dependencies                  │
│ (requests, google-genai)            │
└────────────┬────────────────────────┘
             │
        ✗ STOP ─► ERROR: Dependencies not installed
             │
             ▼ ✓
┌─────────────────────────────────────┐
│ Check PORTKEY_API_KEY               │
└────────────┬────────────────────────┘
             │
        ✗ STOP ─► ERROR: PORTKEY_API_KEY not set
             │
             ▼ ✓
┌─────────────────────────────────────┐
│ Check GOOGLE_API_KEY                │
└────────────┬────────────────────────┘
             │
        ✗ STOP ─► ERROR: GOOGLE_API_KEY not set
             │
             ▼ ✓
┌─────────────────────────────────────┐
│ Initialize tokenizer                │
└────────────┬────────────────────────┘
             │
        ✗ STOP ─► ERROR: Tokenizer init failed
             │
             ▼ ✓
┌─────────────────────────────────────┐
│ Load test data                      │
│ Run evaluation                      │
│ Return metrics                      │
└─────────────────────────────────────┘
```

## Error Messages

If PORTKEY_API_KEY is missing:
```
ERROR: PORTKEY_API_KEY environment variable is REQUIRED but not set
ERROR: Please set: export PORTKEY_API_KEY='your-portkey-key'
ERROR: PORTKEY_API_KEY not configured - evaluation cannot proceed
```

If GOOGLE_API_KEY is missing:
```
ERROR: GOOGLE_API_KEY environment variable is REQUIRED but not set
ERROR: Please set: export GOOGLE_API_KEY='your-google-api-key'
ERROR: Note: GOOGLE_API_KEY is needed for accurate token-level semantic error analysis
ERROR: GOOGLE_API_KEY not configured - evaluation cannot proceed
```

If tokenizer initialization fails:
```
ERROR: Failed to initialize Gemini tokenizer
ERROR: Ensure GOOGLE_API_KEY is set correctly and the Gemini API is accessible
ERROR: Tokenizer initialization failed - evaluation cannot proceed
```

## Testing

Run the API key validation test:

```bash
python test_api_validation.py
```

This will test:
1. **Missing PORTKEY_API_KEY** → Evaluation stops ✓
2. **Missing GOOGLE_API_KEY** → Evaluation stops ✓
3. **Both keys set** → Evaluation proceeds past validation ✓

## Return Values

When validation fails, the `evaluate()` function returns an error dictionary:

```python
{
    "error": "PORTKEY_API_KEY not configured - evaluation cannot proceed"
}
```

Or for successful validation continuing to data issues:
```python
{
    "error": "No test data"  # Failed at a later stage
}
```

## Implementation Details

**File:** `evaluators/gemini_eval.py`

**Changes:**
- Line ~99-111: Added explicit checks for both PORTKEY_API_KEY and GOOGLE_API_KEY
- Line ~113-117: Added validation that tokenizer was successfully initialized
- Clear error messages guide users to fix configuration issues

**Key Functions:**
- `evaluate()`: Main entry point with validation logic
- `get_portkey_key()`: Returns PORTKEY_API_KEY or None
- `get_google_api_key()`: Returns GOOGLE_API_KEY or None
- `get_gemini_tokenizer()`: Initializes tokenizer with validated keys

## Usage Example

```python
from evaluators import gemini_eval
import os

# Set required environment variables
os.environ['PORTKEY_API_KEY'] = 'pk_...'
os.environ['GOOGLE_API_KEY'] = 'AIza...'

# Evaluation will now proceed if keys are valid
result = gemini_eval.evaluate(project_root='/path/to/project')

if result.get('error'):
    print(f"Evaluation failed: {result['error']}")
else:
    print(f"Evaluation succeeded!")
    print(f"Mean CER: {result['mean_cer']}")
    print(f"Mean WER: {result['mean_wer']}")
```

## Notes

- ✅ **Early validation:** Errors caught before any API calls
- ✅ **Clear messages:** Users know exactly what to fix
- ✅ **Fail-fast:** No wasted time if config is wrong
- ✅ **Graceful degradation:** Other parts of system unaffected
- ✅ **Backward compatible:** Existing error handling preserved
