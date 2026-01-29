#!/usr/bin/env python3
"""
Tests for OpenWebUI client.
Run with: python -m pytest tests/test_openwebui_client.py -v
Or directly: python tests/test_openwebui_client.py
"""

import os
import sys
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.openwebui_client import OpenWebUIClient


def get_api_key():
    """Get API key from environment or secrets.json"""
    api_key = os.getenv("OPEN_WEB_UI_API_KEY")
    if api_key:
        return api_key

    secrets_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "secrets.json")
    if os.path.exists(secrets_path):
        with open(secrets_path) as f:
            secrets = json.load(f)
            return secrets.get("OPEN_WEB_UI_API_KEY")

    return None


def run_quick_test():
    """Run a quick connectivity test without pytest."""
    print("=" * 60)
    print("OpenWebUI Client Quick Test")
    print("=" * 60)

    api_key = get_api_key()
    if not api_key:
        print("ERROR: No API key found!")
        print("Set OPEN_WEB_UI_API_KEY environment variable or add to secrets.json")
        return False

    print(f"API key found (length: {len(api_key)})")

    client = OpenWebUIClient(
        server_url="http://llm.xdddev.chtc.io",
        model_name="lexu14/porpoise1",
        api_key=api_key,
        temperature=0,
        top_p=0.95,
        max_tokens=10
    )

    print(f"\nServer URL: {client.server_url}")
    print(f"Model: {client.model_name}")

    # Test 1: Health check
    print("\n[Test 1] Checking server health...")
    if client.check_server_health():
        print("  PASS: Server is reachable")
    else:
        print("  FAIL: Server not reachable")
        return False

    # Test 2: Simple generation
    print("\n[Test 2] Testing simple generation...")
    result = client.generate("Say 'hello' and nothing else.")
    if "error" in result:
        print(f"  FAIL: {result.get('error_message', result['error'])}")
        return False
    print(f"  PASS: Got response: '{result['text_output'].strip()}'")

    # Test 3: Relevance classification format
    print("\n[Test 3] Testing relevance classification format...")
    prompt = """Abstract: This study shows inflammation markers are elevated in heart failure.
Hypothesis: Inhibition of inflammation would help treat heart failure.
Instructions: Classify this abstract as either 0 (Not Relevant) or 1 (Relevant).
Score: """

    result = client.generate(
        prompt,
        sampling_parameters={"temperature": 0, "top_p": 0.95, "max_tokens": 1}
    )
    if "error" in result:
        print(f"  FAIL: {result.get('error_message', result['error'])}")
        return False

    output = result["text_output"].strip()
    if output in ["0", "1"]:
        print(f"  PASS: Got valid classification: '{output}'")
    else:
        print(f"  WARN: Unexpected output: '{output}' (expected '0' or '1')")

    # Test 4: Batch generation
    print("\n[Test 4] Testing batch generation (3 prompts)...")
    prompts = [
        "What is 1+1? Just the number.",
        "What is 2+2? Just the number.",
        "What is 3+3? Just the number."
    ]
    results = client.generate_batch(prompts, max_workers=3, show_progress=False)

    if len(results) != 3:
        print(f"  FAIL: Expected 3 results, got {len(results)}")
        return False

    errors = sum(1 for r in results if "error" in r)
    if errors > 0:
        print(f"  FAIL: {errors}/3 requests failed")
        return False

    outputs = [r["text_output"].strip()[:20] for r in results]
    print(f"  PASS: Got {len(results)} responses: {outputs}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    return True


# Pytest tests (only loaded if pytest is available)
try:
    import pytest

    API_KEY = get_api_key()
    pytestmark = pytest.mark.skipif(
        not API_KEY,
        reason="OPEN_WEB_UI_API_KEY not found in environment or secrets.json"
    )

    class TestOpenWebUIClient:
        """Test suite for OpenWebUI client."""

        @pytest.fixture
        def client(self):
            """Create a client instance for testing."""
            return OpenWebUIClient(
                server_url="http://llm.xdddev.chtc.io",
                model_name="lexu14/porpoise1",
                api_key=API_KEY,
                temperature=0,
                top_p=0.95,
                max_tokens=10
            )

        def test_client_initialization(self, client):
            """Test that client initializes correctly."""
            assert client.server_url == "http://llm.xdddev.chtc.io"
            assert client.model_name == "lexu14/porpoise1"
            assert client.temperature == 0
            assert client.top_p == 0.95
            assert client.max_tokens == 10
            assert client.completions_url == "http://llm.xdddev.chtc.io/api/chat/completions"

        def test_server_health(self, client):
            """Test server health check."""
            is_healthy = client.check_server_health()
            assert is_healthy, "Server should be reachable"

        def test_simple_generation(self, client):
            """Test a simple generation request."""
            result = client.generate("What is 2+2? Answer with just the number.")
            assert "error" not in result, f"Generation failed: {result.get('error_message', result)}"
            assert "text_output" in result, "Response should contain text_output"
            assert len(result["text_output"]) > 0, "Output should not be empty"

        def test_relevance_classification_format(self, client):
            """Test the exact format used for relevance classification."""
            prompt = """Abstract: This study investigates the role of inflammation in heart failure patients.
Hypothesis: Inhibition of inflammation would be a novel and useful approach to treat heart failure.
Instructions: Classify this abstract as either 0 (Not Relevant) or 1 (Relevant) for evaluating the provided hypothesis.
Score: """
            result = client.generate(
                prompt,
                sampling_parameters={"temperature": 0, "top_p": 0.95, "max_tokens": 1}
            )
            assert "error" not in result, f"Generation failed: {result.get('error_message', result)}"
            assert "text_output" in result, "Response should contain text_output"
            output = result["text_output"].strip()
            assert output in ["0", "1"], f"Expected '0' or '1', got '{output}'"

        def test_batch_generation(self, client):
            """Test batch generation with multiple prompts."""
            prompts = [
                "What is 1+1? Answer with just the number.",
                "What is 2+2? Answer with just the number.",
                "What is 3+3? Answer with just the number."
            ]
            results = client.generate_batch(prompts, max_workers=2, show_progress=True)
            assert len(results) == 3, "Should return results for all prompts"
            for i, result in enumerate(results):
                assert "error" not in result, f"Generation {i} failed"
                assert "text_output" in result, f"Response {i} should contain text_output"

except ImportError:
    pass  # pytest not available, skip class definition


if __name__ == "__main__":
    success = run_quick_test()
    sys.exit(0 if success else 1)
