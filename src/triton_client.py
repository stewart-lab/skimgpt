#!/usr/bin/env python3
"""
Simple Triton Inference Server client for the 'porpoise' model.
Uses the generate endpoint for vLLM models with decoupled transaction policy.
Model: lexu14/porpoise1 (3.8B parameter text generation model)
Server: https://xdddev.chtc.io/triton
"""

import json
import requests
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed


class TritonClient:
    """Client for sending inference requests to Triton server."""

    def __init__(self, server_url: str, model_name: str):
        """
        Initialize the Triton client.
        Args:
            server_url: Base URL of the Triton server
            model_name: Name of the model to use for inference
        """
        self.server_url = server_url.rstrip('/')
        self.model_name = model_name
        self.generate_url = f"{self.server_url}/v2/models/{self.model_name}/generate"

    def check_server_health(self) -> bool:
        """Check if the Triton server is ready."""
        try:
            response = requests.get(f"{self.server_url}/v2/health/ready")
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"Error checking server health: {e}")
            return False

    def get_model_metadata(self) -> dict:
        """Get metadata about the model."""
        try:
            response = requests.get(f"{self.server_url}/v2/models/{self.model_name}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting model metadata: {e}")
            return {}

    def generate(self, text_input: str, stream: bool = False,
                 sampling_parameters: dict = None,
                 exclude_input_in_output: bool = True) -> dict:
        """
        Send a generation request to the Triton server using vLLM backend.
        Args:
            text_input: The input text/prompt for generation
            stream: Whether to stream the response
            sampling_parameters: Dict of sampling params (temperature, top_p, max_tokens, etc.)
            exclude_input_in_output: Whether to exclude the input prompt from output
        Returns:
            Dictionary containing the generation response
        """
        # Default sampling parameters
        if sampling_parameters is None:
            sampling_parameters = {
                "temperature": 0.7,
                "top_p": 0.95,
                "max_tokens": 100
            }

        # Construct the request payload using Triton's format
        payload = {
            "text_input": text_input,
            "stream": stream,
            "sampling_parameters": json.dumps(sampling_parameters),
            "exclude_input_in_output": exclude_input_in_output
        }

        try:
            response = requests.post(
                self.generate_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                stream=stream
            )
            response.raise_for_status()

            if stream:
                # Handle streaming response
                result_text = ""
                for line in response.iter_lines():
                    if line:
                        result_text += line.decode('utf-8') + "\n"
                return {"text_output": result_text}
            else:
                return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Error during generation: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            return {}

    def generate_batch(self, text_inputs: list[str],
                      sampling_parameters: dict = None,
                      exclude_input_in_output: bool = True,
                      max_workers: int = 10) -> list[dict]:
        """
        Send multiple generation requests concurrently to the Triton server.
        Args:
            text_inputs: List of input texts/prompts for generation
            sampling_parameters: Dict of sampling params (temperature, top_p, max_tokens, etc.)
            exclude_input_in_output: Whether to exclude the input prompt from output
            max_workers: Maximum number of concurrent requests
        Returns:
            List of dictionaries containing the generation responses (in same order as inputs)
        """
        # Default sampling parameters
        if sampling_parameters is None:
            sampling_parameters = {
                "temperature": 0.7,
                "top_p": 0.95,
                "max_tokens": 100
            }

        def _generate_single(idx_input):
            idx, text_input = idx_input
            result = self.generate(
                text_input=text_input,
                stream=False,
                sampling_parameters=sampling_parameters,
                exclude_input_in_output=exclude_input_in_output
            )
            return idx, result

        results = [None] * len(text_inputs)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all requests with their indices
            futures = {
                executor.submit(_generate_single, (idx, text)): idx
                for idx, text in enumerate(text_inputs)
            }

            # Collect results as they complete
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        return results


def main():
    """Main function to demonstrate the Triton client."""

    parser = argparse.ArgumentParser(
        description="Send inference requests to the Triton Porpoise model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default example prompt
  python triton_client.py
  # Custom prompt
  python triton_client.py --prompt "Write a short story about"
  # Adjust generation parameters
  python triton_client.py --prompt "Explain quantum computing" --max-tokens 200 --temperature 0.5
  # Check server status only
  python triton_client.py --health-check
        """
    )

    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default="I really enjoyed this",
        help="Input prompt for text generation (default: 'I really enjoyed this')"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate (default: 100)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling top-p (default: 0.95)"
    )
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Only check server health and exit"
    )
    parser.add_argument(
        "--show-metadata",
        action="store_true",
        help="Show model metadata"
    )

    args = parser.parse_args()

    # Configuration
    SERVER_URL = "https://xdddev.chtc.io/triton"
    MODEL_NAME = "porpoise"

    # Initialize client
    client = TritonClient(SERVER_URL, MODEL_NAME)

    # Check server health
    print("Checking server health...")
    if client.check_server_health():
        print("✓ Server is ready\n")
    else:
        print("✗ Server is not ready")
        return

    if args.health_check:
        return

    # Get model metadata if requested
    if args.show_metadata:
        print(f"Getting metadata for model '{MODEL_NAME}'...")
        metadata = client.get_model_metadata()
        if metadata:
            print("Model metadata:")
            print(json.dumps(metadata, indent=2))
            print()

    # Send generation request
    print(f"Prompt: {args.prompt}")
    print(f"Parameters: max_tokens={args.max_tokens}, temperature={args.temperature}, top_p={args.top_p}")
    print("\nGenerating...\n")

    result = client.generate(
        text_input=args.prompt,
        stream=False,
        sampling_parameters={
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens
        }
    )

    if result:
        output_text = result.get("text_output", "")
        print("=" * 60)
        print("Generated Text:")
        print("=" * 60)
        print(output_text)
        print("=" * 60)
    else:
        print("Failed to get generation result")


if __name__ == "__main__":
    main()

