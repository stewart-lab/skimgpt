from __future__ import annotations
import pandas as pd
import argparse
import os
import socket
import subprocess
import time
import vllm
from itertools import chain
import random
import torch
from src.utils import Config, RaggedTensor, sanitize_term_for_filename, strip_pipe
from src.pubmed_fetcher import PubMedFetcher
from src.classifier import (
    calculate_relevance_ratios,
    process_single_row,
    write_to_json,
)


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


def getHypothesis(
    config: Config, a_term: str = None, b_term: str = None, c_term: str = None
) -> str:


    if config.is_km_with_gpt:
        assert a_term and b_term and not c_term
        hypothesis_template = config.km_hypothesis
        return hypothesis_template.format(a_term=a_term, b_term=b_term)
    elif config.is_skim_with_gpt:
        assert (
            (a_term and b_term and not c_term)
            or (b_term and c_term and not a_term)
            or (a_term and c_term and not b_term)
        )

        if a_term and b_term and not c_term:
            hypothesis_template = config.skim_hypotheses.get("AB")
            return hypothesis_template.format(a_term=a_term, b_term=b_term)
        elif b_term and c_term and not a_term:
            hypothesis_template = config.skim_hypotheses.get("BC")
            return hypothesis_template.format(b_term=b_term, c_term=c_term)
        elif a_term and c_term and not b_term:
            hypothesis_template = config.skim_hypotheses.get("rel_AC")
            return hypothesis_template.format(a_term=a_term, c_term=c_term)

    return f"No valid hypothesis for the provided {config.job_type}."


def prompt(abstract, hyp) -> str:
    return f"Abstract: {abstract}\nHypothesis: {hyp}\nInstructions: Classify this abstract as either 0 (Not Relevant) or 1 (Relevant) for evaluating the provided hypothesis.\nScore: "


def gen(
    prompts: RaggedTensor, model: any, sampling_config: any
) -> RaggedTensor:
    generated = model.generate(prompts.data, sampling_params=sampling_config)
    outputs = RaggedTensor(
        [output.outputs[0].text for output in generated], prompts.break_point
    )
    return outputs


def getPrompts(abstracts: RaggedTensor, hypotheses: RaggedTensor) -> RaggedTensor:
    assert not abstracts.is2D(), "abstracts should be flattened."
    assert not hypotheses.is2D(), "hypotheses should be flattened."
    return RaggedTensor(
        [prompt(abstracts[i], hypotheses[i]) for i in range(abstracts.shape)],
        hypotheses.break_point,
    )

def postProcess(
    config: Config,
    outputs: RaggedTensor,
    abstracts: RaggedTensor,
    hypotheses: RaggedTensor,
    out_df: pd.DataFrame,
    terms: str,
    shape: list,
):

    abstracts.reshape(shape)

    if not config.debug:
        answer_masks = outputs.map(eval)
        answer_masks.reshape(shape)
        abstracts.applyFilter(answer_masks)
    else:
        answer_masks = RaggedTensor([eval(answer[0]) for answer in outputs])
        answer_masks.reshape(shape)
        cot = RaggedTensor([answer[1:] for answer in outputs])
        cot.reshape(shape)

        if terms == "ac":
            out_df[f"{terms}_mask"] = answer_masks.data
            out_df[f"{terms}_cot"] = cot.data
            out_df[f"{terms}_hypothesis"] = hypotheses.data
        else:
            out_df[f"{terms}_mask"] = answer_masks.data
            out_df[f"{terms}_cot"] = cot.data
            out_df[f"{terms}_hypothesis"] = hypotheses.data

    out_df[f"{terms}_mask"] = answer_masks.data
    out_df[f"{terms}_pmid_intersection"] = abstracts.data


def process_dataframe(out_df: pd.DataFrame, config: Config, pubmed_fetcher: PubMedFetcher) -> pd.DataFrame:
    """Process dataframe with optimizations and filtering."""
    logger = config.logger
    columns_to_process = [col for col in [
        "ab_pmid_intersection",
        "bc_pmid_intersection",
        "ac_pmid_intersection"
    ] if col in out_df.columns]
    
    num_intersections = len(columns_to_process)
    logger.info(f"Processing {num_intersections} intersections")
    
    for column in columns_to_process:
        # Optimize text length with evenly distributed tokens
        out_df[column] = out_df[column].apply(
            lambda x: pubmed_fetcher.optimize_text_length(
                x, 
                max_tokens=110000000,  # Total tokens across all intersections
                num_intersections=num_intersections
            )
        )
        # Sort by year and limit to top N if configured
        if config.post_n > 0:
            out_df[column] = out_df[column].apply(
                lambda x: pubmed_fetcher.interleave_abstracts(x, config.post_n, config.top_n_articles_most_cited, config.top_n_articles_most_recent)
            )
    logger.debug(f"out_df in classifier process_dataframe: {out_df}")
    out_df = calculate_relevance_ratios(out_df, config)
    return out_df


def sample_consolidated_abstracts(v1, v2, config: Config):
    """Sample from two abstract collections; return consolidated text, sampled count, total deduped count.

    Args:
        v1: First collection of abstracts (list or single string or empty).
        v2: Second collection of abstracts (list or single string or empty).
        config: Global configuration providing sampling parameters.

    Returns:
        A tuple of (consolidated_abstracts: str, expected_count: int, total_relevant_abstracts: int)
    """
    logger = config.logger
    def normalize_entries(value):
        """Return a flat list of individual abstracts.

        Handles cases where input is a single concatenated string containing
        multiple abstracts separated by the '===END OF ABSTRACT===' sentinel.
        """
        segments = []
        # Normalize to list for iteration
        if isinstance(value, list):
            iterable = value
        elif isinstance(value, str):
            iterable = [value]
        else:
            iterable = []

        for item in iterable:
            if not isinstance(item, str):
                # Coerce non-strings defensively
                segments.append(str(item))
                continue

            text = item.strip()
            if not text:
                continue

            if '===END OF ABSTRACT===' in text:
                parts = [p.strip() for p in text.split('===END OF ABSTRACT===') if p.strip()]
                # Re-append sentinel to each piece to preserve downstream expectations
                segments.extend([f"{p}===END OF ABSTRACT===" for p in parts])
            else:
                # Fallback: treat as a single abstract entry
                segments.append(text)
        return segments

    list1 = normalize_entries(v1)
    list2 = normalize_entries(v2)

    # Deduplicate across rows: prefer PMID if present; else hash normalized text
    def make_key(text: str) -> str:
        try:
            import re as _re
            m = _re.search(r"PMID:\s*(\d+)", text)
            if m:
                return f"pmid:{m.group(1)}"
        except Exception:
            pass
        try:
            import hashlib as _hashlib
            norm = " ".join(text.split()).lower()
            return "hash:" + _hashlib.sha1(norm.encode("utf-8")).hexdigest()
        except Exception:
            return "hash:" + text[:64]

    seen_keys = set()
    dedup1 = []
    for t in list1:
        k = make_key(t)
        if k in seen_keys:
            continue
        seen_keys.add(k)
        dedup1.append(t)

    dedup2 = []
    for t in list2:
        k = make_key(t)
        if k in seen_keys:
            continue
        seen_keys.add(k)
        dedup2.append(t)

    list1 = dedup1
    list2 = dedup2

    total1 = len(list1)
    total2 = len(list2)
    logger.debug(f"entities_in_candidate1: {total1}")
    logger.debug(f"entities_in_candidate2: {total2}")
    total = total1 + total2
    logger.debug(f"entities_total: {total}")

    # Configurable parameters (cap removed, floor fixed at 0.06)
    min_floor = float(config.global_settings.get("DCH_MIN_SAMPLING_FRACTION", 0.06))
    target_total = int(config.global_settings.get("DCH_SAMPLE_SIZE", 50))

    n1 = 0
    n2 = 0
    if total > 0:
        s1 = (total1 / total) if total > 0 else 0.0
        s2 = (total2 / total) if total > 0 else 0.0

        # Apply minimum floor only to non-empty sets
        if total1 > 0:
            s1 = max(s1, min_floor)
        if total2 > 0:
            s2 = max(s2, min_floor)

        # Normalize shares
        if s1 == 0 and s2 > 0:
            s2 = 1.0
        elif s2 == 0 and s1 > 0:
            s1 = 1.0
        else:
            sum_s = s1 + s2
            s1 = s1 / sum_s if sum_s > 0 else 0.0
            s2 = s2 / sum_s if sum_s > 0 else 0.0

        # No max ratio cap; proceed directly to allocation

        # Initial allocation with rounding
        n1 = int(round(s1 * target_total)) if total1 > 0 else 0
        n2 = int(round(s2 * target_total)) if total2 > 0 else 0

        # Adjust to hit target_total
        diff = target_total - (n1 + n2)
        if diff != 0:
            if diff > 0:
                # Allocate remaining to the side with larger fractional share and capacity
                for _ in range(diff):
                    cap1 = total1 - n1
                    cap2 = total2 - n2
                    if (s1 >= s2 and cap1 > 0) or cap2 <= 0:
                        n1 += 1 if cap1 > 0 else 0
                    else:
                        n2 += 1 if cap2 > 0 else 0
            else:
                # Remove extras from the side with larger current allocation
                for _ in range(-diff):
                    if n1 >= n2 and n1 > 0:
                        n1 -= 1
                    elif n2 > 0:
                        n2 -= 1

        # Cap by availability
        n1 = min(n1, total1)
        n2 = min(n2, total2)

        # If still under target due to limited availability, top up from the other side
        remaining = target_total - (n1 + n2)
        if remaining > 0:
            add1 = min(remaining, total1 - n1)
            n1 += add1
            remaining -= add1
            if remaining > 0:
                add2 = min(remaining, total2 - n2)
                n2 += add2

    # Perform sampling
    sampled1 = random.sample(list1, n1) if n1 > 0 else []
    sampled2 = random.sample(list2, n2) if n2 > 0 else []
    logger.debug(f"sampled1: len {len(sampled1)} {sampled1}")
    logger.debug(f"sampled2: len {len(sampled2)} {sampled2}")
    sampled_abstracts = sampled1 + sampled2
    logger.debug(f"num_sampled_candidate1: {n1}, num_sampled_candidate2: {n2}, total_sampled: {len(sampled_abstracts)}")

    if sampled_abstracts:
        consolidated_abstracts = "\n\n".join(sampled_abstracts)
        expected_count = len(sampled_abstracts)
    else:
        consolidated_abstracts = ""
        expected_count = 0

    total_relevant_abstracts = total1 + total2

    return consolidated_abstracts, expected_count, total_relevant_abstracts


def process_results(out_df: pd.DataFrame, config: Config, num_abstracts_fetched: int) -> None:
    logger = config.logger
    """Process results and write to JSON files."""
    total_rows = len(out_df)
    logger.info(f"Processing {total_rows} results...")

    # Determine output directory based on iteration
    output_base_dir = "output"
    
    # If we're in an iteration, use the iteration subdirectory
    if config.iterations and hasattr(config, 'current_iteration') and config.current_iteration > 0:
        iteration_dir = f"iteration_{config.current_iteration}"
        output_base_dir = os.path.join(output_base_dir, iteration_dir)
        # Make sure the directory exists
        os.makedirs(output_base_dir, exist_ok=True)
        logger.info(f"Writing results to iteration directory: {output_base_dir}")
    else:
        logger.info(f"Writing results to base output directory: {output_base_dir}")

    if config.is_dch:
        # Build direct comparison hypotheses and provide consolidated text content
        # Preserve pipes in terms for relevance; canonicalization happens later
        hypotheses = [getHypothesis(config=config, a_term=a_term, b_term=b_term) for a_term, b_term in zip(out_df['a_term'], out_df['b_term'])]
        logger.debug(f"hypotheses: {hypotheses}")
        hyp1 = hypotheses[0]
        hyp2 = hypotheses[1]
        logger.debug(f"hyp1: {hyp1}")
        logger.debug(f"hyp2: {hyp2}")

        # Consolidate abstracts/text from both candidate rows (if present)
        v1 = out_df.iloc[0].get("ab_pmid_intersection", [])
        v2 = out_df.iloc[1].get("ab_pmid_intersection", [])

        # Use deduplicated pool sizes from sampling function (pre-trim, cross-row dedup): total1 + total2
        consolidated_abstracts, expected_count, total_relevant_abstracts = sample_consolidated_abstracts(v1, v2, config)

        # Store canonical (pipe-stripped) hypotheses in final JSON for DCH
        dch_row = {
            "hypothesis1": strip_pipe(hyp1) if isinstance(hyp1, str) else hyp1,
            "hypothesis2": strip_pipe(hyp2) if isinstance(hyp2, str) else hyp2,
            "ab_pmid_intersection": consolidated_abstracts,
            "expected_per_abstract_count": expected_count,
            "total_relevant_abstracts": total_relevant_abstracts,
        }
        out_df = pd.DataFrame([dch_row])

    for index, row in out_df.iterrows():
        result_dict = process_single_row(row, config)
        logger.debug(f" Result dict: {result_dict}")
        if result_dict:
            for ratio_type in ["ab", "bc", "ac"]:
                ratio_col = f"{ratio_type}_relevance_ratio"
                fraction_col = f"{ratio_type}_relevance_fraction"
                if ratio_col in out_df.columns and fraction_col in out_df.columns:
                    ratio = row[ratio_col]
                    fraction = row[fraction_col]
                    result_dict[f"{ratio_type}_relevance"] = f"{ratio:.2f} ({fraction})"

            result_dict["num_abstracts_fetched"] = num_abstracts_fetched
            # Append authoritative total relevant count from relevance filter for DCH outputs
            if config.is_dch:
                try:
                    result_dict["total_relevant_abstracts"] = int(row.get("total_relevant_abstracts", 0))
                except Exception:
                    result_dict["total_relevant_abstracts"] = 0
            if config.is_dch:
                logger.info(f"Processed row {index + 1}/{total_rows} (DCH)")
            else:
                logger.info(f"Processed row {index + 1}/{total_rows} ({row['b_term']})")

            raw_a = row.get("a_term", "")
            raw_b = row.get("b_term", "")
            raw_c = row.get("c_term", "")

            if config.is_dch:
                hyp1_name = sanitize_term_for_filename(row.get("hypothesis1", "hypothesis1"))
                hyp2_name = sanitize_term_for_filename(row.get("hypothesis2", "hypothesis2"))
                output_json = f"{hyp1_name}_vs_{hyp2_name}_km_with_gpt_direct_comp.json"
            elif config.is_skim_with_gpt:
                a_fname = sanitize_term_for_filename(raw_a)
                b_fname = sanitize_term_for_filename(raw_b)
                c_fname = sanitize_term_for_filename(raw_c)
                output_json = f"{a_fname}_{c_fname}_{b_fname}_skim_with_gpt.json"
            else:
                a_fname = sanitize_term_for_filename(raw_a)
                b_fname = sanitize_term_for_filename(raw_b)
                output_json = f"{a_fname}_{b_fname}_km_with_gpt.json"

            logger.debug(f" IN PROCESS RESULTS   Output json before writing: {output_json}")
            logger.debug(f" IN PROCESS RESULTS   Result dict: {result_dict}")
            write_to_json([result_dict], output_json, output_base_dir, config)


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
    config.load_km_output(args.km_output)   
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
        max_retries=config.global_settings.get("MAX_RETRIES", 3),
        backoff_factor=0.5
    )
    logger.info("Initialized PubMedFetcher")
    
    # Process each row individually (unified for DCH and non-DCH)
    ab_pmids = []
    ab_hypotheses = []

    for _, row in config.data.iterrows():
        a_term = row['a_term'].split("&")[0]  # Handle potential compound terms
        logger.debug(f"Row b_term from dataframe: {row['b_term']}, type: {type(row['b_term'])}")
        b_term = row['b_term']

        # Convert string representation of list to actual list
        pmids = eval(row['ab_pmid_intersection'])
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
            a_term = row['a_term'].split("&")[0]  # Handle potential compound terms
            
            # Process BC terms
            bc_pmid_list = eval(row['bc_pmid_intersection'])
            bc_pmids.append(bc_pmid_list)
            bc_hypothesis = getHypothesis(config=config, c_term=c_term, b_term=b_term)
            bc_hypotheses.append(bc_hypothesis)
            
            # Process AC terms if available
            if config.has_ac and 'ac_pmid_intersection' in row:
                ac_pmid_list = eval(row['ac_pmid_intersection'])
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

    dynamic_max_tokens = 4000

    # Configure GPU environment before vLLM model initialization
    try:
        os.environ['VLLM_NO_USAGE_STATS'] = '1'
        os.environ['DO_NOT_TRACK'] = '1'

        logger.info("Configuring PyTorch environment for containerized execution...")
        
        os.environ['TORCH_COMPILE_DISABLE'] = '1'
        os.environ['TORCHINDUCTOR_DISABLE'] = '1'
        
        # Set cache directories to current working directory (writable)
        current_dir = os.getcwd()
        os.environ['TORCH_HOME'] = current_dir
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = os.path.join(current_dir, 'torch_cache')
        os.environ['TRITON_CACHE_DIR'] = os.path.join(current_dir, 'triton_cache')
        
        # Disable other PyTorch features that might cause issues in containerized environments
        os.environ['PYTORCH_DISABLE_DISTRIBUTED_SAMPLING'] = '1'
        os.environ['NCCL_DISABLE_WARN'] = '1'
        
        # vLLM-specific environment configurations
        os.environ['VLLM_DISABLE_CUSTOM_ALL_REDUCE'] = '1'
        os.environ['VLLM_USE_TRITON_FLASH_ATTN'] = '0'
        os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
        
        # Dynamically estimate max_num_batched_tokens based on available memory
        dynamic_max_tokens = estimate_max_batched_tokens(
            seq_len=4000,
            gpu_memory_util=0.8,
            token_mem_mb=0.31,  # ~0.31 MB per token for Phi-3-mini KV-cache
            weight_mem_gib=7.2,  # ~7.2 GiB weights for Phi-3-mini
            config=config
        )
        logger.info(f"Dynamic max_num_batched_tokens estimated: {dynamic_max_tokens}")
        
        logger.info("PyTorch and vLLM environment configured for containerized execution")
        
        # Set environment variables to handle GPU device ID issues
        if torch.cuda.is_available():
            cuda_devices = os.getenv('CUDA_VISIBLE_DEVICES', '')
            logger.info(f"Original CUDA_VISIBLE_DEVICES: {cuda_devices}")
            
            if cuda_devices and not cuda_devices.replace(',', '').isdigit():
                # Contains UUIDs, need to convert
                device_list = [d.strip() for d in cuda_devices.split(',') if d.strip()]
                converted_devices = []
                
                for device in device_list:
                    if device.startswith('GPU-'):
                        # Convert UUID to numeric ID
                        numeric_id = convert_gpu_uuid_to_device_id(device)
                        logger.info(f"Converted GPU UUID {device} to device ID {numeric_id}")
                        converted_devices.append(str(numeric_id))
                    else:
                        converted_devices.append(device)
                
                # Set the converted device list
                new_cuda_devices = ','.join(converted_devices)
                os.environ['CUDA_VISIBLE_DEVICES'] = new_cuda_devices
                logger.info(f"Updated CUDA_VISIBLE_DEVICES: {new_cuda_devices}")
        
    except Exception as e:
        logger.error(f"Error configuring GPU environment: {str(e)}")
        logger.warning("Proceeding with default GPU configuration")

    # Model setup and inference (single, non-defensive initialisation)
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
        # Fail fast – we no longer attempt multiple fallbacks
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
            process_results(filtered_df, config, num_abstracts_fetched)
            
            iteration_end_time = time.time()
            iteration_elapsed_time = iteration_end_time - iteration_start_time
            logger.info(f"Iteration {iteration} completed in {iteration_elapsed_time:.2f} seconds")
        
        logger.info(f"All {num_iterations} iterations completed successfully")
    else:
        # No iterations, just process results once to the base output directory
        logger.info("No iterations requested, processing results once")
        # Reset current_iteration to 0 to ensure results go to the base directory
        config.current_iteration = 0
        process_results(out_df, config, num_abstracts_fetched)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Relevance analysis completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
