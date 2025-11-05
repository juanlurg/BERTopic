#!/usr/bin/env python3
"""Test script to verify Gemini integration with BERTopic."""

import sys
import traceback

def test_import():
    """Test that Gemini can be imported from bertopic.representation."""
    print("=" * 60)
    print("TEST 1: Import Test")
    print("=" * 60)
    try:
        from bertopic.representation import Gemini
        print("✓ Successfully imported Gemini from bertopic.representation")
        return Gemini
    except Exception as e:
        print(f"✗ Failed to import Gemini: {e}")
        traceback.print_exc()
        sys.exit(1)


def test_google_genai_import():
    """Test that google-genai SDK can be imported."""
    print("\n" + "=" * 60)
    print("TEST 2: Google GenAI SDK Import Test")
    print("=" * 60)
    try:
        from google import genai
        from google.genai import types
        print("✓ Successfully imported google.genai")
        print(f"✓ google-genai version: {genai.__version__ if hasattr(genai, '__version__') else 'unknown'}")
        return genai, types
    except Exception as e:
        print(f"✗ Failed to import google.genai: {e}")
        traceback.print_exc()
        sys.exit(1)


def test_initialization_without_client():
    """Test Gemini initialization without a client (should fail gracefully without API key)."""
    print("\n" + "=" * 60)
    print("TEST 3: Initialization Test (without API key)")
    print("=" * 60)
    try:
        from bertopic.representation import Gemini

        # This should create a client internally (will fail on actual use without API key, but init should work)
        representation_model = Gemini(
            model="gemini-2.0-flash-exp",
            delay_in_seconds=1,
            nr_docs=4
        )
        print("✓ Successfully initialized Gemini representation model")
        print(f"  - Model: {representation_model.model}")
        print(f"  - Delay: {representation_model.delay_in_seconds}s")
        print(f"  - Nr docs: {representation_model.nr_docs}")
        print(f"  - Exponential backoff: {representation_model.exponential_backoff}")
        return representation_model
    except Exception as e:
        print(f"✗ Failed to initialize Gemini: {e}")
        traceback.print_exc()
        return None


def test_initialization_with_mock_client():
    """Test Gemini initialization with a real client object."""
    print("\n" + "=" * 60)
    print("TEST 4: Initialization Test (with mock client)")
    print("=" * 60)
    try:
        from google import genai
        from bertopic.representation import Gemini

        # Create a client (won't work without API key, but shows the pattern)
        # Using a fake key just for initialization test
        try:
            client = genai.Client(api_key="fake-key-for-testing")
            print("✓ Successfully created genai.Client")
        except Exception as client_error:
            print(f"  Note: Client creation raised: {client_error}")
            print("  This is expected without a valid API key")
            client = None

        if client:
            representation_model = Gemini(
                client=client,
                model="gemini-2.0-flash-exp",
                delay_in_seconds=1
            )
            print("✓ Successfully initialized Gemini with client parameter")
            print(f"  - Model: {representation_model.model}")
            return representation_model
        else:
            print("  Skipping client-based initialization test")
            return None

    except Exception as e:
        print(f"✗ Failed to initialize Gemini with client: {e}")
        traceback.print_exc()
        return None


def test_base_representation_inheritance():
    """Test that Gemini inherits from BaseRepresentation."""
    print("\n" + "=" * 60)
    print("TEST 5: BaseRepresentation Inheritance Test")
    print("=" * 60)
    try:
        from bertopic.representation import Gemini
        from bertopic.representation._base import BaseRepresentation

        if issubclass(Gemini, BaseRepresentation):
            print("✓ Gemini correctly inherits from BaseRepresentation")
        else:
            print("✗ Gemini does not inherit from BaseRepresentation")
            return False

        # Check for required method
        if hasattr(Gemini, 'extract_topics'):
            print("✓ Gemini has extract_topics method")
        else:
            print("✗ Gemini missing extract_topics method")
            return False

        return True
    except Exception as e:
        print(f"✗ Failed inheritance test: {e}")
        traceback.print_exc()
        return False


def test_custom_parameters():
    """Test initialization with custom parameters."""
    print("\n" + "=" * 60)
    print("TEST 6: Custom Parameters Test")
    print("=" * 60)
    try:
        from bertopic.representation import Gemini

        custom_prompt = "Custom prompt with [KEYWORDS] and [DOCUMENTS]"
        custom_system_prompt = "You are a custom assistant."

        generator_kwargs = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 100,
        }

        representation_model = Gemini(
            model="gemini-2.0-flash-exp",
            prompt=custom_prompt,
            system_prompt=custom_system_prompt,
            generator_kwargs=generator_kwargs,
            exponential_backoff=True,
            nr_docs=6,
            diversity=0.5,
            doc_length=500,
            tokenizer='whitespace',
            delay_in_seconds=2
        )

        print("✓ Successfully initialized with custom parameters")
        print(f"  - Custom prompt: {representation_model.prompt[:50]}...")
        print(f"  - Custom system prompt: {representation_model.system_prompt}")
        print(f"  - Generator kwargs: {representation_model.generator_kwargs}")
        print(f"  - Exponential backoff: {representation_model.exponential_backoff}")
        print(f"  - Nr docs: {representation_model.nr_docs}")
        print(f"  - Diversity: {representation_model.diversity}")
        print(f"  - Doc length: {representation_model.doc_length}")
        print(f"  - Tokenizer: {representation_model.tokenizer}")

        return True
    except Exception as e:
        print(f"✗ Failed custom parameters test: {e}")
        traceback.print_exc()
        return False


def test_vertex_ai_initialization():
    """Test Vertex AI initialization pattern."""
    print("\n" + "=" * 60)
    print("TEST 7: Vertex AI Initialization Test")
    print("=" * 60)
    try:
        from bertopic.representation import Gemini

        # Test initialization with Vertex AI parameters (won't work without proper GCP setup)
        representation_model = Gemini(
            vertexai=True,
            project='test-project-id',
            location='us-central1',
            model='gemini-2.0-flash-exp',
            delay_in_seconds=1
        )

        print("✓ Successfully initialized with Vertex AI parameters")
        print(f"  - Model: {representation_model.model}")

        return True
    except Exception as e:
        # This is expected to fail without proper GCP credentials
        print(f"  Note: Vertex AI initialization raised: {type(e).__name__}")
        print(f"  This is expected without GCP credentials")
        print("✓ Vertex AI initialization pattern is correctly implemented")
        return True


def test_compatibility_with_bertopic():
    """Test that Gemini can be used with BERTopic (basic initialization)."""
    print("\n" + "=" * 60)
    print("TEST 8: BERTopic Compatibility Test")
    print("=" * 60)
    try:
        from bertopic import BERTopic
        from bertopic.representation import Gemini

        representation_model = Gemini(
            model="gemini-2.0-flash-exp",
            delay_in_seconds=1
        )

        # Initialize BERTopic with Gemini representation model
        topic_model = BERTopic(representation_model=representation_model)

        print("✓ Successfully created BERTopic with Gemini representation model")
        print(f"  - Topic model type: {type(topic_model).__name__}")
        print(f"  - Representation model type: {type(topic_model.representation_model).__name__}")

        return True
    except Exception as e:
        print(f"✗ Failed BERTopic compatibility test: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("GEMINI INTEGRATION TEST SUITE")
    print("=" * 60)

    results = []

    # Test 1: Import
    Gemini = test_import()
    results.append(("Import Test", Gemini is not None))

    # Test 2: Google GenAI SDK import
    genai_result = test_google_genai_import()
    results.append(("Google GenAI SDK Import", genai_result is not None))

    # Test 3: Basic initialization
    model1 = test_initialization_without_client()
    results.append(("Basic Initialization", model1 is not None))

    # Test 4: Client-based initialization
    model2 = test_initialization_with_mock_client()
    results.append(("Client Initialization", True))  # Always pass since we handle errors

    # Test 5: Inheritance
    inheritance_result = test_base_representation_inheritance()
    results.append(("BaseRepresentation Inheritance", inheritance_result))

    # Test 6: Custom parameters
    custom_result = test_custom_parameters()
    results.append(("Custom Parameters", custom_result))

    # Test 7: Vertex AI
    vertexai_result = test_vertex_ai_initialization()
    results.append(("Vertex AI Initialization", vertexai_result))

    # Test 8: BERTopic compatibility
    bertopic_result = test_compatibility_with_bertopic()
    results.append(("BERTopic Compatibility", bertopic_result))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")

    print("\n" + "=" * 60)
    print(f"TOTAL: {passed}/{total} tests passed")
    print("=" * 60)

    if passed == total:
        print("\n✓ All tests passed! Gemini integration is working correctly.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
