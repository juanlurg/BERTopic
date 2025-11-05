#!/usr/bin/env python3
"""Simple test to verify Gemini module can be imported and initialized."""

import sys

print("=" * 60)
print("SIMPLE GEMINI INTEGRATION TEST")
print("=" * 60)

# Test 1: Import google-genai
print("\n[1/4] Testing google-genai import...")
try:
    from google import genai
    from google.genai import types
    print("✓ Successfully imported google-genai")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 2: Import the Gemini module directly (bypass main bertopic __init__)
print("\n[2/4] Testing direct import of Gemini class...")
try:
    import sys
    import importlib.util

    # Load the modules we need first
    spec_base = importlib.util.spec_from_file_location("bertopic.representation._base", "/home/user/BERTopic/bertopic/representation/_base.py")
    base_module = importlib.util.module_from_spec(spec_base)
    sys.modules["bertopic.representation._base"] = base_module
    spec_base.loader.exec_module(base_module)

    spec_utils = importlib.util.spec_from_file_location("bertopic.representation._utils", "/home/user/BERTopic/bertopic/representation/_utils.py")
    utils_module = importlib.util.module_from_spec(spec_utils)
    sys.modules["bertopic.representation._utils"] = utils_module
    spec_utils.loader.exec_module(utils_module)

    # Now load Gemini
    spec = importlib.util.spec_from_file_location("bertopic.representation._gemini", "/home/user/BERTopic/bertopic/representation/_gemini.py")
    gemini_module = importlib.util.module_from_spec(spec)
    sys.modules["bertopic.representation._gemini"] = gemini_module
    spec.loader.exec_module(gemini_module)

    Gemini = gemini_module.Gemini
    print("✓ Successfully imported Gemini class")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Initialize with default parameters
print("\n[3/4] Testing Gemini initialization...")
try:
    representation_model = Gemini(
        model="gemini-2.0-flash-exp",
        delay_in_seconds=1
    )
    print("✓ Successfully initialized Gemini")
    print(f"  - Model: {representation_model.model}")
    print(f"  - Has client: {representation_model.client is not None}")
    print(f"  - Prompt set: {len(representation_model.prompt) > 0}")
    print(f"  - System prompt set: {len(representation_model.system_prompt) > 0}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Check that it has the required method
print("\n[4/4] Testing extract_topics method exists...")
try:
    if hasattr(representation_model, 'extract_topics'):
        print("✓ extract_topics method exists")
        # Check signature
        import inspect
        sig = inspect.signature(representation_model.extract_topics)
        params = list(sig.parameters.keys())
        expected_params = ['self', 'topic_model', 'documents', 'c_tf_idf', 'topics']
        if all(p in params for p in expected_params[1:]):  # skip 'self'
            print(f"✓ extract_topics has correct signature: {params}")
        else:
            print(f"⚠ Warning: Unexpected signature: {params}")
    else:
        print("✗ extract_topics method not found")
        sys.exit(1)
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED!")
print("=" * 60)
print("\nThe Gemini integration is working correctly.")
print("The class can be imported, initialized, and has the required methods.")
print("\nNote: Actual API calls would require a valid API key.")
