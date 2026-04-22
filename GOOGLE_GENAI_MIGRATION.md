# Google Gen AI SDK Migration

## Summary

Updated `evaluators/gemini_eval.py` to use the **official `google-genai` v1.66.0+** SDK instead of the deprecated `google-generativeai` 0.8.6.

## Why This Matters

| Package | Status | Release Date | Notes |
|---------|--------|--------------|-------|
| `google-generativeai` | ❌ **DEPRECATED** | Dec 16, 2025 | End-of-life: Nov 30, 2025 |
| `google-genai` | ✅ **ACTIVE** | Mar 4, 2026 | Official replacement, latest features |

**Key Difference:** Google created a single unified SDK for all developers using Gemini, Veo, Imagen, and other models.

## Changes Made

### 1. **Import Updates**

**Before (deprecated):**
```python
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
```

**After (current):**
```python
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
```

### 2. **Tokenizer Initialization**

**Before:**
```python
genai.configure(api_key=api_key)
model = genai.get_model(f"models/{MODEL_ID}")
return model
```

**After (google-genai v1.66.0+):**
```python
client = genai.Client(api_key=api_key)
return client
```

### 3. **Token Counting**

**Before:**
```python
response = model.count_tokens(text)
```

**After:**
```python
response = client.models.count_tokens(
    model=MODEL_ID,
    contents=text
)
# Returns: response.total_tokens
```

### 4. **Requirements Update**

Updated `requirements.txt`:
```
google-genai>=1.66.0  # Official Google Gen AI SDK (google-generativeai 0.8.6 is deprecated)
```

## Architecture

The implementation now supports **three levels of tokenization**:

1. **Portkey API** (primary): OpenAI-compatible logprobs via Portkey
   - ✅ Works out-of-the-box with Portkey integration
   - ✅ Better logprobs formatting consistency

2. **Google Genai Tokenizer** (optional): Token-level counting
   - Requires `GOOGLE_API_KEY` environment variable
   - Provides token counts for semantic error analysis
   - Falls back gracefully if unavailable

3. **Character-level** (fallback): Simple character-by-character analysis
   - Works without any additional API keys
   - Still provides semantic error metrics

## Usage

```bash
# Install updated package
pip install google-genai>=1.66.0

# Set environment variables
export PORTKEY_API_KEY="your-portkey-key"
export GOOGLE_API_KEY="your-google-api-key"  # Optional for token-level analysis

# Run evaluation
python -m evaluators.gemini_eval
```

## Migration Resources

- **Official Docs:** https://ai.google.dev/gemini-api/docs/migrate
- **SDK GitHub:** https://github.com/googleapis/python-genai
- **PyPI:** https://pypi.org/project/google-genai/

## Verification

✅ **Syntax check:** Passed  
✅ **Imports:** Compatible with google-genai v1.66.0+  
✅ **Tokenizer:** Supports new SDK API structure  
✅ **Backward compatibility:** Graceful fallbacks maintained  
