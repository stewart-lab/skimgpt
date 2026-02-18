"""Relevance analysis using remote Triton inference server.

Primary entry point for relevance filtering.  Tries the Triton server first;
if **all** requests in the batch fail, falls back to the CHTC vLLM path by
invoking ``relevance_chtc.py`` as a subprocess (no local vllm/torch import).
"""
from __future__ import annotations

import subprocess
import sys
import time

from src.utils import Config, RaggedTensor
from src.triton_client import TritonClient
from src.relevance_helper import (
    PreprocessedData,
    preprocess_tsv,
    run_postprocessing,
)


# ---------------------------------------------------------------------------
# Triton-specific generation
# ---------------------------------------------------------------------------

def gen(
    prompts: RaggedTensor,
    model: TritonClient,
    logger=None,
    max_workers: int = None,
    show_progress: bool = False,
    batch_chunk_size: int = None,
) -> tuple[RaggedTensor, bool]:
    """Generate outputs using Triton client batch inference.

    Returns:
        A tuple of (outputs_ragged_tensor, all_failed) where *all_failed*
        is ``True`` when every single request in the batch returned an error.
    """
    if logger:
        logger.info(f"DEBUG gen(): Number of prompts: {len(prompts.data)}")
        if len(prompts.data) > 0:
            logger.info(f"DEBUG gen(): First prompt (truncated): {prompts.data[0][:300]}...")
            logger.info(
                f"DEBUG gen(): Model sampling params - temperature: {model.temperature}, "
                f"top_p: {model.top_p}, max_tokens: {model.max_tokens}"
            )
        if max_workers:
            logger.info(f"DEBUG gen(): Using max_workers={max_workers}")
        if batch_chunk_size:
            logger.info(f"DEBUG gen(): Using batch_chunk_size={batch_chunk_size}")

    batch_results = model.generate_batch(
        text_inputs=prompts.data,
        max_workers=max_workers,
        show_progress=show_progress,
        batch_chunk_size=batch_chunk_size,
    )

    if logger:
        logger.info(f"DEBUG gen(): Number of results: {len(batch_results)}")
        error_count = sum(1 for r in batch_results if "error" in r)
        if error_count > 0:
            logger.warning(f"DEBUG gen(): {error_count}/{len(batch_results)} requests failed")
        if len(batch_results) > 0:
            logger.info(f"DEBUG gen(): First result: {batch_results[0]}")

    # Detect total failure
    total = len(batch_results)
    error_count = sum(1 for r in batch_results if "error" in r)
    all_failed = total > 0 and error_count == total

    outputs = RaggedTensor(
        [result.get("text_output", "") for result in batch_results],
        prompts.break_point,
    )
    return outputs, all_failed


# ---------------------------------------------------------------------------
# CHTC fallback (subprocess — no vllm/torch import)
# ---------------------------------------------------------------------------

def _fallback_to_chtc(config: Config, km_output_path: str) -> None:
    """Run the full CHTC relevance pipeline as a subprocess.

    This avoids importing vllm/torch in the current process.  The CHTC
    script (``src.relevance_chtc``) handles the entire pipeline
    (preprocessing, vLLM inference, post-processing) independently.
    """
    logger = config.logger

    cmd = [
        sys.executable, "-m", "src.relevance_chtc",
        "--km_output", km_output_path,
        "--config", config.job_config_path,
        "--secrets", config.secrets_path,
    ]

    logger.info(f"Running CHTC fallback: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(
            f"CHTC fallback failed (exit code {result.returncode}). "
            f"Ensure vLLM and a GPU are available, or run relevance_chtc.py "
            f"manually on a CHTC cluster node."
        )

    logger.info("CHTC fallback completed successfully")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_relevance_analysis(config: Config, km_output_path: str) -> None:
    """Run relevance analysis on the provided TSV file.

    Pipeline:
      1. Shared preprocessing (TSV parsing, PubMed fetch, prompt building)
      2. Triton remote inference (with automatic CHTC subprocess fallback)
      3. Shared post-processing (filtering, dataframe processing, output)

    If the Triton server is unreachable or all requests fail, the entire
    pipeline is delegated to ``relevance_chtc.py`` via subprocess.
    """
    logger = config.logger
    start_time = time.time()

    # 1. Shared preprocessing
    data: PreprocessedData = preprocess_tsv(config, km_output_path)

    # 2. Model inference — try Triton first
    logger.info("Initializing Triton client for remote inference...")
    try:
        model = TritonClient(
            server_url=config.filter_config.get("SERVER_URL"),
            model_name=config.filter_config.get("MODEL_NAME"),
            temperature=config.filter_config["TEMPERATURE"],
            top_p=config.filter_config["TOP_P"],
            max_tokens=config.filter_config["MAX_COT_TOKENS"] if config.debug else 1,
        )

        if not model.check_server_health():
            logger.error(f"Triton server at {model.server_url} is not ready")
            raise RuntimeError(f"Triton server at {model.server_url} is not responding")

        logger.info(f"Successfully connected to Triton server at {model.server_url}")
        logger.info(f"Using model: {model.model_name}")
    except Exception as e:
        logger.error(f"Failed to initialize Triton client: {e}")
        logger.warning("Falling back to CHTC (vLLM subprocess)...")
        _fallback_to_chtc(config, km_output_path)
        _log_elapsed(logger, start_time)
        return

    # Optional Triton performance tuning from config
    max_workers = config.global_settings.get("TRITON_MAX_WORKERS", None)
    if max_workers:
        max_workers = int(max_workers)
        logger.info(f"Using configured max_workers={max_workers} for Triton inference")

    show_progress = config.global_settings.get("TRITON_SHOW_PROGRESS", False)
    if isinstance(show_progress, str):
        show_progress = show_progress.lower() in ("true", "1", "yes")

    batch_chunk_size = config.global_settings.get("TRITON_BATCH_CHUNK_SIZE", None)
    if batch_chunk_size:
        batch_chunk_size = int(batch_chunk_size)
        logger.info(f"Using batch_chunk_size={batch_chunk_size} for large batches")

    answers, all_failed = gen(
        data.prompts,
        model,
        logger,
        max_workers=max_workers,
        show_progress=show_progress,
        batch_chunk_size=batch_chunk_size,
    )

    # If every request failed, fall back to CHTC subprocess
    if all_failed:
        logger.warning(
            "All Triton requests failed. Falling back to CHTC (vLLM subprocess)..."
        )
        _fallback_to_chtc(config, km_output_path)
        _log_elapsed(logger, start_time)
        return

    # 3. Shared post-processing (only reached if Triton succeeded)
    run_postprocessing(config, data, answers)

    _log_elapsed(logger, start_time)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_elapsed(logger, start_time: float) -> None:
    elapsed = time.time() - start_time
    logger.info(f"Relevance analysis completed in {elapsed:.2f} seconds")
