from __future__ import annotations
import os
import sys
import subprocess

def convert_gpu_uuid_to_device_id(gpu_uuid):
    """Convert GPU UUID to numeric device ID for vLLM compatibility"""
    try:
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=uuid,index',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)

        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 2:
                uuid_part, index_part = parts[0], parts[1]
                if gpu_uuid in uuid_part or uuid_part.endswith(gpu_uuid):
                    return index_part

        if '-' in gpu_uuid:
            suffix = gpu_uuid.split('-')[-1]
            if suffix.isdigit():
                return suffix

    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    return gpu_uuid

# ── IMMEDIATE GPU FIXUP ───────────────────────────────────────────────────
# We MUST fix CUDA_VISIBLE_DEVICES before ANY other imports (like pandas/numpy)
# touch the CUDA runtime.
cuda_devices = os.getenv('CUDA_VISIBLE_DEVICES', '')
if cuda_devices and not cuda_devices.replace(',', '').replace(' ', '').isdigit():
    converted_devices = []
    for dev in cuda_devices.split(','):
        dev = dev.strip()
        if dev.startswith('GPU-'):
            numeric_id = convert_gpu_uuid_to_device_id(dev)
            print(f"BOOTSTRAP: Converted GPU UUID {dev} -> device {numeric_id}", file=sys.stderr)
            converted_devices.append(str(numeric_id))
        else:
            converted_devices.append(dev)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(converted_devices)
    print(f"BOOTSTRAP: Updated CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}", file=sys.stderr)

for _key, _val in {
    'VLLM_NO_USAGE_STATS': '1',
    'DO_NOT_TRACK': '1',
    'TORCH_COMPILE_DISABLE': '1',
    'TORCHINDUCTOR_DISABLE': '1',
}.items():
    os.environ.setdefault(_key, _val)
# ──────────────────────────────────────────────────────────────────────────

import argparse
import logging
from skimgpt.utils import Config, RaggedTensor, configure_logging
from skimgpt.relevance_helper import run_relevance_pipeline

logger = logging.getLogger(__name__)

# vllm and torch are imported lazily inside main() so that GPU environment
# variables (especially CUDA_VISIBLE_DEVICES) can be configured before CUDA
# initializes.
vllm = None   # populated in main()
torch = None  # populated in main()

# vLLM model constants — keep in sync across estimate_max_batched_tokens() and LLM init
VLLM_MAX_MODEL_LEN = 4000
VLLM_GPU_MEMORY_UTIL = 0.8
VLLM_TOKEN_MEM_MB = 0.31       # per-token KV-cache memory for Phi-3-mini
VLLM_WEIGHT_MEM_GIB = 7.2      # model weight footprint for Phi-3-mini


def estimate_max_batched_tokens(seq_len: int = VLLM_MAX_MODEL_LEN,
                               gpu_memory_util: float = VLLM_GPU_MEMORY_UTIL,
                               token_mem_mb: float = VLLM_TOKEN_MEM_MB,
                               weight_mem_gib: float = VLLM_WEIGHT_MEM_GIB,
                               config: Config = None) -> int:
    """Estimate max_num_batched_tokens for vLLM based on available GPU memory.

    Args:
        seq_len: Target tokens per sequence (context length).
        gpu_memory_util: Fraction of total GPU memory vLLM is allowed to use.
        token_mem_mb: Per-token memory in MB (Phi-3-mini ~0.31 MB).
        weight_mem_gib: Estimated memory footprint of model weights (GiB).
            For Phi-3-mini the weight footprint is ~7.2 GiB.
    Returns:
        An integer suitable for vLLM's ``max_num_batched_tokens`` parameter.
    """
    try:
        logger.info(f"device count: {torch.cuda.device_count()}")
        prop = torch.cuda.get_device_properties(0)
        total_gib = prop.total_memory / (1024 ** 3)
        avail_gib = total_gib * gpu_memory_util - weight_mem_gib
        if avail_gib <= 0:
            return seq_len
        token_mem_gib = token_mem_mb / 1024
        tokens_budget = int(avail_gib / token_mem_gib)
        max_tokens = max(seq_len, (tokens_budget // seq_len) * seq_len)
        return max_tokens
    except Exception:
        return seq_len


def main():

    if 'TRANSFORMERS_CACHE' in os.environ:
        os.environ.setdefault('HF_HOME', os.environ['TRANSFORMERS_CACHE'])
        os.environ.pop('TRANSFORMERS_CACHE', None)

    parser = argparse.ArgumentParser(description="kmGPT relevance analysis (vLLM fine-tuned phi-3 mini)")
    parser.add_argument("--km_output", type=str, required=True, help="Tsv file to run relevance filtering on.")
    parser.add_argument("--config", type=str, required=True, help="Config file for kmGPT run.")
    args = parser.parse_args()

    configure_logging()
    config = Config(args.config)
    logger.debug(f"args.km_output: {args.km_output}")
    km_output_path = args.km_output
    if os.path.basename(km_output_path) == "files.txt":
        with open(km_output_path, "r") as f:
            tsv_filename = f.readline().strip()
        if tsv_filename:
            km_output_path = os.path.join(os.path.dirname(km_output_path), tsv_filename)
            logger.debug(f"Resolved files.txt -> {km_output_path}")

    # Load early for file-logging setup (pipeline will call again — idempotent)
    config.load_km_output(km_output_path)

    # ── Configure GPU environment before import ──────────────────────────────
    logger.info("Verifying GPU environment and importing torch/vllm...")
    current_dir = os.getcwd()
    for _key, _val in {
        'TORCH_HOME': current_dir,
        'TORCHINDUCTOR_CACHE_DIR': os.path.join(current_dir, 'torch_cache'),
        'TRITON_CACHE_DIR': os.path.join(current_dir, 'triton_cache'),
        'PYTORCH_DISABLE_DISTRIBUTED_SAMPLING': '1',
        'NCCL_DISABLE_WARN': '1',
        'VLLM_DISABLE_CUSTOM_ALL_REDUCE': '1',
        'VLLM_USE_TRITON_FLASH_ATTN': '0',
        'VLLM_ALLOW_LONG_MAX_MODEL_LEN': '1',
    }.items():
        os.environ[_key] = _val

    # ── Lazy-import torch and vllm AFTER env is clean ────────────────────
    global torch, vllm
    import torch as _torch   # noqa: E402
    import vllm as _vllm     # noqa: E402
    torch = _torch
    vllm = _vllm

    logger.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    logger.info(f"torch.cuda.device_count(): {torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available — cannot run vLLM without a GPU. "
            f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '')!r}"
        )

    dynamic_max_tokens = estimate_max_batched_tokens(config=config)
    logger.info(f"Dynamic max_num_batched_tokens estimated: {dynamic_max_tokens}")

    # ── Initialize vLLM model ────────────────────────────────────────────
    logger.info("Initializing vLLM model...")
    model = vllm.LLM(
        model=config.filter_config["MODEL"],
        max_model_len=VLLM_MAX_MODEL_LEN,
        max_num_batched_tokens=dynamic_max_tokens,
        gpu_memory_utilization=VLLM_GPU_MEMORY_UTIL,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        trust_remote_code=False,
    )

    sampling_config = vllm.SamplingParams(
        temperature=config.filter_config["TEMPERATURE"],
        top_k=config.filter_config["TOP_K"],
        top_p=config.filter_config["TOP_P"],
        max_tokens=config.filter_config["MAX_COT_TOKENS"] if config.debug else 1,
    )

    def vllm_infer(prompts):
        generated = model.generate(prompts.data, sampling_params=sampling_config)
        return RaggedTensor([o.outputs[0].text for o in generated], prompts.break_point)

    run_relevance_pipeline(config, km_output_path, infer=vllm_infer, output_base_dir="output")


if __name__ == "__main__":
    main()
