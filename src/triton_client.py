#!/usr/bin/env python3
"""
Optimized Triton Inference Server client for the 'porpoise' model.
Uses the generate endpoint for vLLM models with decoupled transaction policy.
Model: lexu14/porpoise1 (3.8B parameter text generation model)
Server: https://xdddev.chtc.io/triton

Features:
- HTTP connection pooling for reduced latency
- Automatic retry with exponential backoff
- Configurable timeouts
- Progress tracking for batch operations
- Chunked batch processing for large datasets
"""

import json
import requests
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional, Dict, List
from tqdm import tqdm

# Use SKiM-GPT logger for consistency with the rest of the application
logger = logging.getLogger("SKiM-GPT")


class TritonClient:
    """Optimized client for sending inference requests to Triton server.
    
    Performance Features:
    - Connection pooling: Reuses TCP connections for better throughput
    - Automatic retry: Handles transient network failures with exponential backoff
    - Configurable concurrency: Tune max_workers based on server capacity
    - Progress tracking: Visual feedback for batch operations
    - Batch chunking: Memory-efficient processing of large datasets
    """
    
    # Default configuration values
    DEFAULT_SERVER_URL = "https://xdddev.chtc.io/triton"
    DEFAULT_MODEL_NAME = "porpoise"
    DEFAULT_MAX_WORKERS = 10
    DEFAULT_CONNECT_TIMEOUT = 30  # seconds
    DEFAULT_READ_TIMEOUT = 300  # seconds (5 minutes for long inference)
    DEFAULT_POOL_CONNECTIONS = 20
    DEFAULT_POOL_MAXSIZE = 20
    DEFAULT_MAX_RETRIES = 3

    def __init__(self, server_url: str = None, model_name: str = None,
                 max_retries: int = DEFAULT_MAX_RETRIES,
                 connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
                 read_timeout: float = DEFAULT_READ_TIMEOUT,
                 pool_connections: int = DEFAULT_POOL_CONNECTIONS,
                 pool_maxsize: int = DEFAULT_POOL_MAXSIZE,
                 temperature: float = None,
                 top_p: float = None,
                 max_tokens: int = None):
        """
        Initialize the Triton client with connection pooling and retry logic.
        
        Args:
            server_url: Base URL of the Triton server (defaults to DEFAULT_SERVER_URL)
            model_name: Name of the model to use for inference (defaults to DEFAULT_MODEL_NAME)
            max_retries: Maximum number of retry attempts for failed requests (default: 3)
            connect_timeout: Timeout for establishing connection in seconds (default: 30)
            read_timeout: Timeout for reading response in seconds (default: 300)
            pool_connections: Number of connection pools to cache (default: 20)
            pool_maxsize: Maximum size of the connection pool (default: 20)
            temperature: Sampling temperature for generation (required)
            top_p: Nucleus sampling parameter (required)
            max_tokens: Maximum tokens to generate (required)
        """
        self.server_url = (server_url or self.DEFAULT_SERVER_URL).rstrip('/')
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.generate_url = f"{self.server_url}/v2/models/{self.model_name}/generate"
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        
        # Store required sampling parameters (validation should happen in Config)
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        
        # Create a session with connection pooling
        self.session = requests.Session()
        
        # Configure retry strategy with exponential backoff
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,  # 1s, 2s, 4s, 8s...
            status_forcelist=[429, 502, 503, 504],  # Retry on these status codes
            allowed_methods=["GET", "POST"],
            raise_on_status=False
        )
        
        # Mount adapter with connection pooling and retry strategy
        adapter = HTTPAdapter(
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize,
            max_retries=retry_strategy
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        logger.debug(f"TritonClient initialized with connection pool (size={pool_maxsize}), "
                    f"retries={max_retries}, timeouts=({connect_timeout}s, {read_timeout}s)")

    def check_server_health(self) -> bool:
        """Check if the Triton server is ready."""
        try:
            response = self.session.get(
                f"{self.server_url}/v2/health/ready",
                timeout=(self.connect_timeout, self.read_timeout)
            )
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            logger.error(f"Error checking server health: {e}")
            return False

    def get_model_metadata(self) -> dict:
        """Get metadata about the model."""
        try:
            response = self.session.get(
                f"{self.server_url}/v2/models/{self.model_name}",
                timeout=(self.connect_timeout, self.read_timeout)
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting model metadata: {e}")
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
                               If None, uses the default parameters set during initialization
            exclude_input_in_output: Whether to exclude the input prompt from output
            
        Returns:
            Dictionary containing the generation response, or dict with 'error' key on failure
        """
        # Use instance defaults if no sampling parameters provided
        if sampling_parameters is None:
            sampling_parameters = {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens
            }

        # Construct the request payload using Triton's format
        payload = {
            "text_input": text_input,
            "stream": stream,
            "sampling_parameters": json.dumps(sampling_parameters),
            "exclude_input_in_output": exclude_input_in_output
        }

        try:
            response = self.session.post(
                self.generate_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                stream=stream,
                timeout=(self.connect_timeout, self.read_timeout)
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
                result = response.json()
                logger.debug(f"Triton raw response keys: {result.keys()}")
                logger.debug(f"Triton raw response (truncated): {str(result)[:500]}")
                return result

        except requests.exceptions.Timeout as e:
            error_msg = f"Request timeout after {self.connect_timeout}s (connect) / {self.read_timeout}s (read)"
            logger.error(f"{error_msg}: {e}")
            logger.debug(f"Failed prompt (first 200 chars): {text_input[:200]}...")
            return {"error": "timeout", "error_message": error_msg}
        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {type(e).__name__}"
            logger.error(f"{error_msg}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response text: {e.response.text[:500]}")
            logger.debug(f"Failed prompt (first 200 chars): {text_input[:200]}...")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return {"error": "request_failed", "error_message": error_msg}

    def generate_batch(self, text_inputs: List[str],
                      sampling_parameters: dict = None,
                      exclude_input_in_output: bool = True,
                      max_workers: int = None,
                      show_progress: bool = False,
                      batch_chunk_size: int = None) -> List[dict]:
        """
        Send multiple generation requests concurrently to the Triton server.
        
        Args:
            text_inputs: List of input texts/prompts for generation
            sampling_parameters: Dict of sampling params (temperature, top_p, max_tokens, etc.)
            exclude_input_in_output: Whether to exclude the input prompt from output
            max_workers: Maximum number of concurrent requests (default: 10)
                       Tuning guide:
                       - Start with 10 for typical servers
                       - Increase to 20-50 for high-capacity servers
                       - Decrease to 5 if seeing 429/503 errors
                       - Monitor server CPU/GPU utilization for optimal value
            show_progress: Show progress bar during batch processing (requires tqdm)
            batch_chunk_size: Process large batches in chunks (default: None = no chunking)
                            Recommended for batches > 1000 to manage memory and provide
                            better progress visibility
                            
        Returns:
            List of dictionaries containing the generation responses (in same order as inputs)
        """
        if not text_inputs:
            return []
        
        # Use instance defaults if no sampling parameters provided
        if sampling_parameters is None:
            sampling_parameters = {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens
            }
        
        # Use default max_workers if not specified
        if max_workers is None:
            max_workers = self.DEFAULT_MAX_WORKERS
        
        # If batch chunking is requested, process in chunks
        if batch_chunk_size and len(text_inputs) > batch_chunk_size:
            return self._generate_batch_chunked(
                text_inputs, sampling_parameters, exclude_input_in_output,
                max_workers, show_progress, batch_chunk_size
            )
        
        # Process entire batch at once
        return self._generate_batch_single(
            text_inputs, sampling_parameters, exclude_input_in_output,
            max_workers, show_progress
        )
    
    def _generate_batch_single(self, text_inputs: List[str],
                               sampling_parameters: dict,
                               exclude_input_in_output: bool,
                               max_workers: int,
                               show_progress: bool) -> List[dict]:
        """Process a single batch of requests concurrently."""
        start_time = time.time()
        total_requests = len(text_inputs)
        error_count = 0
        
        def _generate_single(idx_input):
            idx, text_input = idx_input
            result = self.generate(
                text_input=text_input,
                stream=False,
                sampling_parameters=sampling_parameters,
                exclude_input_in_output=exclude_input_in_output
            )
            # Track errors
            has_error = "error" in result
            return idx, result, has_error

        results = [None] * total_requests

        # Create progress bar if requested
        progress_bar = None
        if show_progress:
            progress_bar = tqdm(total=total_requests, desc="Generating", unit="req")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all requests with their indices
            futures = {
                executor.submit(_generate_single, (idx, text)): idx
                for idx, text in enumerate(text_inputs)
            }

            # Collect results as they complete
            for future in as_completed(futures):
                idx, result, has_error = future.result()
                results[idx] = result
                if has_error:
                    error_count += 1
                
                # Update progress bar
                if progress_bar:
                    progress_bar.update(1)
        
        if progress_bar:
            progress_bar.close()
        
        # Log batch statistics
        elapsed_time = time.time() - start_time
        avg_latency = elapsed_time / total_requests if total_requests > 0 else 0
        logger.info(f"Batch complete: {total_requests} requests in {elapsed_time:.2f}s "
                   f"(avg {avg_latency:.2f}s/req, {error_count} errors, {max_workers} workers)")
        
        if error_count > 0:
            error_pct = (error_count / total_requests) * 100
            logger.warning(f"Batch had {error_count}/{total_requests} ({error_pct:.1f}%) failed requests")

        return results
    
    def _generate_batch_chunked(self, text_inputs: List[str],
                                sampling_parameters: dict,
                                exclude_input_in_output: bool,
                                max_workers: int,
                                show_progress: bool,
                                chunk_size: int) -> List[dict]:
        """Process a large batch in chunks for better memory management."""
        total_requests = len(text_inputs)
        num_chunks = (total_requests + chunk_size - 1) // chunk_size
        
        logger.info(f"Processing large batch of {total_requests} requests in {num_chunks} chunks "
                   f"(chunk_size={chunk_size})")
        
        all_results = []
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, total_requests)
            chunk = text_inputs[start_idx:end_idx]
            
            logger.debug(f"Processing chunk {chunk_idx + 1}/{num_chunks} "
                        f"(requests {start_idx}-{end_idx})")
            
            # Process this chunk
            chunk_results = self._generate_batch_single(
                chunk, sampling_parameters, exclude_input_in_output,
                max_workers, show_progress
            )
            all_results.extend(chunk_results)
        
        logger.info(f"Chunked batch complete: {total_requests} total requests processed")
        return all_results
