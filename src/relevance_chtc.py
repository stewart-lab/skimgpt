from __future__ import annotations
import os
import sys
import subprocess

def convert_gpu_uuid_to_device_id(gpu_uuid):
    """Convert GPU UUID to numeric device ID for vLLM compatibility"""
    try:
        # Get GPU information from nvidia-smi
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
                # Handle different UUID formats
                if gpu_uuid in uuid_part or uuid_part.endswith(gpu_uuid) or gpu_uuid.startswith(uuid_part.split('-')[0]):
                    return index_part
        
        # If not found, try to extract numeric suffix from UUID
        if '-' in gpu_uuid:
            suffix = gpu_uuid.split('-')[-1]
            if suffix.isdigit():
                return suffix
                
    except (subprocess.SubprocessError, subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    return gpu_uuid  # Return original if conversion fails

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

# Essential vLLM/PyTorch flags to set early
os.environ.setdefault('VLLM_NO_USAGE_STATS', '1')
os.environ.setdefault('DO_NOT_TRACK', '1')
os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')
os.environ.setdefault('TORCHINDUCTOR_DISABLE', '1')
# ──────────────────────────────────────────────────────────────────────────

import ast
import pandas as pd
import argparse
import socket
import time
from itertools import chain
from src.utils import Config, RaggedTensor
from src.pubmed_fetcher import PubMedFetcher
from src.relevance_helper import (
    getHypothesis, getPrompts,
    postProcess, process_dataframe, process_results,
)

# vllm and torch are imported lazily inside main() so that GPU environment
# variables (especially CUDA_VISIBLE_DEVICES) can be configured before CUDA
# initializes.  
vllm = None   # populated in main()
torch = None  # populated in main()


def gen(
    prompts: RaggedTensor, model: any, sampling_config: any
) -> RaggedTensor:
    generated = model.generate(prompts.data, sampling_params=sampling_config)
    outputs = RaggedTensor(
        [output.outputs[0].text for output in generated], prompts.break_point
    )
    return outputs


def estimate_max_batched_tokens(seq_len: int = 4000,
                               gpu_memory_util: float = 0.8,
                               token_mem_mb: float = 0.5,
                               weight_mem_gib: float = 8.0, config: Config = None) -> int:
    """Estimate max_num_batched_tokens for vLLM based on available GPU memory.

    Args:
        seq_len: Target tokens per sequence (context length).
        gpu_memory_util: Fraction of total GPU memory vLLM is allowed to use.
            Phi-3-mini a more accurate value is ~0.31 MB).
        weight_mem_gib: Estimated memory footprint of model weights (GiB).
            For Phi-3-mini the weight footprint is ~7.2 GiB.
    Returns:
        An integer suitable for vLLM's ``max_num_batched_tokens`` parameter.
    """
    try:
        config.logger.info(f"device count: {torch.cuda.device_count()}")
        prop = torch.cuda.get_device_properties(0)
        total_gib = prop.total_memory / (1024 ** 3)
        # Memory budget for KV cache under gpu_memory_utilisation
        avail_gib = total_gib * gpu_memory_util - weight_mem_gib
        if avail_gib <= 0:
            return seq_len
        # Convert per-token MB to GiB
        token_mem_gib = token_mem_mb / 1024
        tokens_budget = int(avail_gib / token_mem_gib)
        # Round down to nearest multiple of seq_len
        max_tokens = max(seq_len, (tokens_budget // seq_len) * seq_len)
        return max_tokens
    except Exception:
        # In case of any error fall back to seq_len
        return seq_len


def main():

    if 'TRANSFORMERS_CACHE' in os.environ:
        os.environ.setdefault('HF_HOME', os.environ['TRANSFORMERS_CACHE'])
        os.environ.pop('TRANSFORMERS_CACHE', None)
    
    # Parse arguments FIRST
    parser = argparse.ArgumentParser(description="kmGPT relevance analysis (vLLM fine-tuned phi-3 mini)")
    parser.add_argument(
        "--km_output",
        type=str,
        required=True,
        help="Tsv file to run relevance filtering on.",
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Config file for kmGPT run."
    )
    parser.add_argument(
        "--secrets", type=str, required=True, help="Secrets file for kmGPT run."
    )  
    args = parser.parse_args()
    
    # THEN create Config
    config = Config(args.config)  # Pass config path from arguments
    logger = config.logger
    logger.debug(f"config: {config}")
    logger.debug(f"args.km_output: {args.km_output}")
    km_output_path = args.km_output
    if os.path.basename(km_output_path) == "files.txt":
        with open(km_output_path, "r") as f:
            tsv_filename = f.readline().strip()
        if tsv_filename:
            km_output_path = os.path.join(os.path.dirname(km_output_path), tsv_filename)
            logger.debug(f"Resolved files.txt -> {km_output_path}")

    config.load_km_output(km_output_path)
    start_time = time.time()
    logger.info("Starting relevance analysis...")
    
    try:
        # DNS resolution test in Python
        host = "eutils.ncbi.nlm.nih.gov"
        ip_address = socket.gethostbyname(host)
        logger.debug(f"Python DNS resolution test: Successfully resolved '{host}' to '{ip_address}'")
    except socket.gaierror as e:
        logger.error(f"Python DNS resolution test: Failed to resolve '{host}'. Error: {e}")
        raise  # Re-raise the exception to stop script execution

    out_df = config.data.copy(deep=True)
    logger.debug(f"Working with dataframe of shape {out_df.shape}")

    # Initialize PubMedFetcher
    pubmed_fetcher = PubMedFetcher(
        config=config,
        email="jfreeman@morgridge.org",
        api_key=config.secrets["PUBMED_API_KEY"],
        max_retries=config.max_retries,
        backoff_factor=0.5
    )
    logger.info("Initialized PubMedFetcher")
    
    # Process each row individually (unified for DCH and non-DCH)
    ab_pmids = []
    ab_hypotheses = []

    for _, row in config.data.iterrows():
        a_term = row['a_term']
        logger.debug(f"Row b_term from dataframe: {row['b_term']}, type: {type(row['b_term'])}")
        b_term = row['b_term']

        # Convert string representation of list to actual list
        pmids = ast.literal_eval(row['ab_pmid_intersection'])
        ab_pmids.append(pmids)

        # Generate hypothesis for this specific pair
        # Preserve pipes here; do not strip before relevance
        hypothesis = getHypothesis(config=config, a_term=a_term, b_term=b_term)
        ab_hypotheses.append(hypothesis)

    # Convert to RaggedTensor format
    ab_pmids = RaggedTensor(ab_pmids)
    ab_hypotheses = RaggedTensor(ab_hypotheses)
    all_pmids = ab_pmids.flatten()
    all_hypotheses = ab_hypotheses.expand(ab_pmids.shape)

    if config.is_skim_with_gpt:
        # Process BC and AC terms row by row
        bc_pmids = []
        bc_hypotheses = []
        ac_pmids = []
        ac_hypotheses = []

        for _, row in config.data.iterrows():
            b_term = row['b_term']
            c_term = row['c_term']
            a_term = row['a_term']
            
            # Process BC terms
            bc_pmid_list = ast.literal_eval(row['bc_pmid_intersection'])
            bc_pmids.append(bc_pmid_list)
            bc_hypothesis = getHypothesis(config=config, c_term=c_term, b_term=b_term)
            bc_hypotheses.append(bc_hypothesis)
            
            # Process AC terms if available
            if config.has_ac and 'ac_pmid_intersection' in row:
                ac_pmid_list = ast.literal_eval(row['ac_pmid_intersection'])
                ac_pmids.append(ac_pmid_list)
                ac_hypothesis = getHypothesis(config=config, a_term=a_term, c_term=c_term)
                ac_hypotheses.append(ac_hypothesis)

        # Convert to RaggedTensor format and add to all_pmids/hypotheses
        bc_pmids = RaggedTensor(bc_pmids)
        bc_hypotheses = RaggedTensor(bc_hypotheses)
        all_pmids += bc_pmids.flatten()
        all_hypotheses += bc_hypotheses.expand(bc_pmids.shape)

        if config.has_ac and ac_pmids:
            ac_pmids = RaggedTensor(ac_pmids)
            ac_hypotheses = RaggedTensor(ac_hypotheses)
            all_pmids += ac_pmids.flatten()
            all_hypotheses += ac_hypotheses.expand(ac_pmids.shape)

    # Fetch abstracts
    abstract_map = pubmed_fetcher.fetch_abstracts(all_pmids)
    num_abstracts_fetched = len(abstract_map)
    abstracts = all_pmids.map(lambda pmid: abstract_map.get(str(pmid), ""))
    
    # Ensure abstracts remain flattened for prompt generation; DCH merging happens in process_results

    # ── Configure GPU environment before import ──────────────────────────────
    # Much of this is now done at module-level "bootstrap" to ensure
    # environment is locked before any other heavy deps are imported.
    logger.info("Verifying GPU environment and importing torch/vllm...")

    current_dir = os.getcwd()
    os.environ['TORCH_HOME'] = current_dir
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = os.path.join(current_dir, 'torch_cache')
    os.environ['TRITON_CACHE_DIR'] = os.path.join(current_dir, 'triton_cache')

    os.environ['PYTORCH_DISABLE_DISTRIBUTED_SAMPLING'] = '1'
    os.environ['NCCL_DISABLE_WARN'] = '1'

    os.environ['VLLM_DISABLE_CUSTOM_ALL_REDUCE'] = '1'
    os.environ['VLLM_USE_TRITON_FLASH_ATTN'] = '0'
    os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'

    logger.info("Now importing torch and vllm...")

    # ── Lazy-import torch and vllm AFTER env is clean ────────────────────
    global torch, vllm
    import torch as _torch   # noqa: E402
    import vllm as _vllm     # noqa: E402
    torch = _torch
    vllm = _vllm

    logger.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    logger.info(f"torch.cuda.device_count(): {torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        logger.error(
            "No CUDA device available after environment fixup. "
            f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '')!r}"
        )
        raise RuntimeError("CUDA is not available — cannot run vLLM without a GPU")

    # Dynamically estimate max_num_batched_tokens based on available memory
    dynamic_max_tokens = estimate_max_batched_tokens(
        seq_len=4000,
        gpu_memory_util=0.8,
        token_mem_mb=0.31,
        weight_mem_gib=7.2,
        config=config
    )
    logger.info(f"Dynamic max_num_batched_tokens estimated: {dynamic_max_tokens}")

    # ── Initialize vLLM model ────────────────────────────────────────────
    logger.info("Initializing vLLM model …")
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

    prompts = getPrompts(abstracts, all_hypotheses)
    answers = gen(prompts, model, sampling_config)

    defaults = 3 * [RaggedTensor([])]
    ab_outputs, bc_outputs, ac_outputs, *_ = chain(answers.split(), defaults)
    ab_abstracts, bc_abstracts, ac_abstracts, *_ = chain(abstracts.split(), defaults)
        
    postProcess(
        config, ab_outputs, ab_abstracts, ab_hypotheses, out_df, "ab", ab_pmids.shape
     )

    if config.is_skim_with_gpt:
        postProcess(
            config, bc_outputs, bc_abstracts, bc_hypotheses, out_df, "bc", bc_pmids.shape
        )
        if config.has_ac:
            postProcess(
                    config, ac_outputs, ac_abstracts, ac_hypotheses, out_df, "ac", ac_pmids.shape
                )
        
    out_df = process_dataframe(out_df, config, pubmed_fetcher)

    # Save the initial processed dataframe
    initial_output_file = config.debug_tsv_name if config.debug else config.filtered_tsv_name
    out_df.to_csv(initial_output_file, sep="\t")
    logger.info(f"Saved initial processed data to {initial_output_file}")
    
    # Check if we need to run iterations
    if config.iterations:
        # Determine number of iterations
        num_iterations = 1
        if isinstance(config.iterations, bool) and config.iterations:
            logger.warning("iterations is set to True but no number specified, defaulting to 1 iteration")
        elif isinstance(config.iterations, int) and config.iterations > 0:
            num_iterations = config.iterations
            logger.info(f"Will perform {num_iterations} iterations of analysis")
        else:
            logger.warning("Invalid iterations config, defaulting to 1 iteration")
        
        # Create base output directory for iterations
        logger.info(f"Setting up for {num_iterations} iterations")
        # Create iteration directories
        for i in range(1, num_iterations + 1):
            iteration_dir = os.path.join(config.km_output_dir, f"iteration_{i}")
            if not os.path.exists(iteration_dir):
                os.makedirs(iteration_dir)
                logger.info(f"Created output directory for iteration {i}: {iteration_dir}")
        
        # Use the same filtered data for all iterations
        filtered_df = out_df.copy(deep=True)
        
        # Process all iterations
        for iteration in range(1, num_iterations + 1):
            iteration_start_time = time.time()
            logger.info(f"Processing iteration {iteration}/{num_iterations}...")
            
            # Set current iteration in config to update output paths
            config.set_iteration(iteration)
            
            # Process results for this iteration (using same filtered data)
            process_results(filtered_df, config, num_abstracts_fetched, output_base_dir="output")
            
            iteration_end_time = time.time()
            iteration_elapsed_time = iteration_end_time - iteration_start_time
            logger.info(f"Iteration {iteration} completed in {iteration_elapsed_time:.2f} seconds")
        
        logger.info(f"All {num_iterations} iterations completed successfully")
    else:
        # No iterations, just process results once to the base output directory
        logger.info("No iterations requested, processing results once")
        # Reset current_iteration to 0 to ensure results go to the base directory
        config.current_iteration = 0
        process_results(out_df, config, num_abstracts_fetched, output_base_dir="output")

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Relevance analysis completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()