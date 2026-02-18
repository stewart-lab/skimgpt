"""Relevance analysis using local vLLM on CHTC GPU clusters.

Can be run standalone (``python -m src.relevance_chtc --km_output ... --config ... --secrets ...``)
or used as a fallback from the Triton path via :func:`run_vllm_inference`.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time

import vllm
import torch

from src.utils import Config, RaggedTensor
from src.relevance_helper import (
    PreprocessedData,
    preprocess_tsv,
    run_postprocessing,
)


# ---------------------------------------------------------------------------
# GPU / vLLM environment helpers (CHTC-specific)
# ---------------------------------------------------------------------------

def convert_gpu_uuid_to_device_id(gpu_uuid):
    """Convert GPU UUID to numeric device ID for vLLM compatibility."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,index", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                uuid_part, index_part = parts[0], parts[1]
                if (
                    gpu_uuid in uuid_part
                    or uuid_part.endswith(gpu_uuid)
                    or gpu_uuid.startswith(uuid_part.split("-")[0])
                ):
                    return index_part

        if "-" in gpu_uuid:
            suffix = gpu_uuid.split("-")[-1]
            if suffix.isdigit():
                return suffix

    except (subprocess.SubprocessError, subprocess.CalledProcessError, FileNotFoundError):
        pass

    return gpu_uuid


def estimate_max_batched_tokens(
    seq_len: int = 4000,
    gpu_memory_util: float = 0.8,
    token_mem_mb: float = 0.5,
    weight_mem_gib: float = 8.0,
    config: Config = None,
) -> int:
    """Estimate ``max_num_batched_tokens`` for vLLM based on available GPU memory."""
    try:
        config.logger.info(f"device count: {torch.cuda.device_count()}")
        prop = torch.cuda.get_device_properties(0)
        total_gib = prop.total_memory / (1024**3)
        avail_gib = total_gib * gpu_memory_util - weight_mem_gib
        if avail_gib <= 0:
            return seq_len
        token_mem_gib = token_mem_mb / 1024
        tokens_budget = int(avail_gib / token_mem_gib)
        max_tokens = max(seq_len, (tokens_budget // seq_len) * seq_len)
        return max_tokens
    except Exception:
        return seq_len


def _configure_gpu_environment(config: Config) -> int:
    """Configure GPU / PyTorch / vLLM environment variables.

    Returns the estimated ``max_num_batched_tokens``.
    """
    logger = config.logger
    dynamic_max_tokens = 4000

    try:
        os.environ["VLLM_NO_USAGE_STATS"] = "1"
        os.environ["DO_NOT_TRACK"] = "1"

        logger.info("Configuring PyTorch environment for containerized execution...")
        os.environ["TORCH_COMPILE_DISABLE"] = "1"
        os.environ["TORCHINDUCTOR_DISABLE"] = "1"

        current_dir = os.getcwd()
        os.environ["TORCH_HOME"] = current_dir
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(current_dir, "torch_cache")
        os.environ["TRITON_CACHE_DIR"] = os.path.join(current_dir, "triton_cache")

        os.environ["PYTORCH_DISABLE_DISTRIBUTED_SAMPLING"] = "1"
        os.environ["NCCL_DISABLE_WARN"] = "1"

        os.environ["VLLM_DISABLE_CUSTOM_ALL_REDUCE"] = "1"
        os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "0"
        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

        dynamic_max_tokens = estimate_max_batched_tokens(
            seq_len=4000,
            gpu_memory_util=0.8,
            token_mem_mb=0.31,
            weight_mem_gib=7.2,
            config=config,
        )
        logger.info(f"Dynamic max_num_batched_tokens estimated: {dynamic_max_tokens}")
        logger.info("PyTorch and vLLM environment configured for containerized execution")

        if torch.cuda.is_available():
            cuda_devices = os.getenv("CUDA_VISIBLE_DEVICES", "")
            logger.info(f"Original CUDA_VISIBLE_DEVICES: {cuda_devices}")

            if cuda_devices and not cuda_devices.replace(",", "").isdigit():
                device_list = [d.strip() for d in cuda_devices.split(",") if d.strip()]
                converted_devices = []
                for device in device_list:
                    if device.startswith("GPU-"):
                        numeric_id = convert_gpu_uuid_to_device_id(device)
                        logger.info(f"Converted GPU UUID {device} to device ID {numeric_id}")
                        converted_devices.append(str(numeric_id))
                    else:
                        converted_devices.append(device)
                new_cuda_devices = ",".join(converted_devices)
                os.environ["CUDA_VISIBLE_DEVICES"] = new_cuda_devices
                logger.info(f"Updated CUDA_VISIBLE_DEVICES: {new_cuda_devices}")

    except Exception as e:
        logger.error(f"Error configuring GPU environment: {str(e)}")
        logger.warning("Proceeding with default GPU configuration")

    return dynamic_max_tokens


# ---------------------------------------------------------------------------
# vLLM generation
# ---------------------------------------------------------------------------

def gen(prompts: RaggedTensor, model, sampling_config) -> RaggedTensor:
    """Generate outputs using local vLLM model."""
    generated = model.generate(prompts.data, sampling_params=sampling_config)
    outputs = RaggedTensor(
        [output.outputs[0].text for output in generated], prompts.break_point
    )
    return outputs


def run_vllm_inference(prompts: RaggedTensor, config: Config) -> RaggedTensor:
    """Set up GPU environment, initialise vLLM, and run inference.

    Called by :func:`main` for standalone CHTC execution.  Also invoked
    when the Triton path delegates to this module via subprocess.

    Returns:
        RaggedTensor of model output texts.
    """
    logger = config.logger

    if "TRANSFORMERS_CACHE" in os.environ:
        os.environ.setdefault("HF_HOME", os.environ["TRANSFORMERS_CACHE"])
        os.environ.pop("TRANSFORMERS_CACHE", None)

    dynamic_max_tokens = _configure_gpu_environment(config)

    logger.info("Initializing vLLM model (fallback) …")
    try:
        model = vllm.LLM(
            model=config.filter_config["MODEL"],
            max_model_len=4000,
            max_num_batched_tokens=dynamic_max_tokens,
            gpu_memory_utilization=0.8,
            enforce_eager=True,
            disable_custom_all_reduce=True,
            trust_remote_code=False,
        )
        logger.info("Successfully initialized vLLM model")
    except Exception as e:
        logger.error(f"Failed to initialize vLLM model: {e}")
        raise

    sampling_config = vllm.SamplingParams(
        temperature=config.filter_config["TEMPERATURE"],
        top_k=config.filter_config["TOP_K"],
        top_p=config.filter_config["TOP_P"],
        max_tokens=config.filter_config["MAX_COT_TOKENS"] if config.debug else 1,
    )

    return gen(prompts, model, sampling_config)


# ---------------------------------------------------------------------------
# Standalone entry point (CHTC cluster execution)
# ---------------------------------------------------------------------------

def main():
    """Standalone entry point for running relevance analysis on CHTC clusters."""
    if "TRANSFORMERS_CACHE" in os.environ:
        os.environ.setdefault("HF_HOME", os.environ["TRANSFORMERS_CACHE"])
        os.environ.pop("TRANSFORMERS_CACHE", None)

    parser = argparse.ArgumentParser(
        description="kmGPT relevance analysis (vLLM fine-tuned phi-3 mini)"
    )
    parser.add_argument(
        "--km_output", type=str, required=True,
        help="Tsv file to run relevance filtering on.",
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Config file for kmGPT run."
    )
    parser.add_argument(
        "--secrets", type=str, required=True, help="Secrets file for kmGPT run."
    )
    args = parser.parse_args()

    config = Config(args.config)
    logger = config.logger
    start_time = time.time()

    # 1. Shared preprocessing
    data: PreprocessedData = preprocess_tsv(config, args.km_output)

    # 2. Local vLLM inference
    answers = run_vllm_inference(data.prompts, config)

    # 3. Shared post-processing
    run_postprocessing(config, data, answers)

    elapsed = time.time() - start_time
    logger.info(f"Relevance analysis completed in {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
