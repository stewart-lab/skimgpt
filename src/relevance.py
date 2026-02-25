from __future__ import annotations
import ast
import json
import pandas as pd
import os
import socket
import time
from itertools import chain
import random
from src.utils import (
    Config,
    RaggedTensor,
    sanitize_term_for_filename,
    strip_pipe,
    normalize_entries,
    write_to_json,
    PMID_PATTERN,
    extract_pmid,
)
from src.pubmed_fetcher import PubMedFetcher
from src.full_text_chunker import FullTextChunker
from src.classifier import (
    calculate_relevance_ratios,
    process_single_row,
    analyze_abstract_with_frontier_LLM,
)

from src.triton_client import TritonClient
from src.image_analyzer import ImageAnalyzer


def flatten_nested(data: list) -> list:
    """Flatten nested lists into a single list if the first element is itself a list."""
    if data and isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
        return [item for sublist in data for item in sublist]
    return data


def getHypothesis(
    config: Config, a_term: str = None, b_term: str = None, c_term: str = None
) -> str:
    # Sanitize terms for hypothesis generation
    if a_term:
        a_term = a_term.replace("&", " ")
    if b_term:
        b_term = b_term.replace("&", " ")
    if c_term:
        c_term = c_term.replace("&", " ")

    if config.is_km_with_gpt:
        assert a_term and b_term and not c_term
        return config.km_hypothesis.format(a_term=a_term, b_term=b_term)

    if config.is_skim_with_gpt:
        assert (
            (a_term and b_term and not c_term)
            or (b_term and c_term and not a_term)
            or (a_term and c_term and not b_term)
        )
        if a_term and b_term and not c_term:
            return config.skim_hypotheses["AB"].format(a_term=a_term, b_term=b_term)
        if b_term and c_term and not a_term:
            return config.skim_hypotheses["BC"].format(b_term=b_term, c_term=c_term)
        if a_term and c_term and not b_term:
            return config.skim_hypotheses["rel_AC"].format(a_term=a_term, c_term=c_term)

    return f"No valid hypothesis for the provided {config.job_type}."


def prompt(abstract, hyp) -> str:
    return f"Abstract: {abstract}\nHypothesis: {hyp}\nInstructions: Classify this abstract as either 0 (Not Relevant) or 1 (Relevant) for evaluating the provided hypothesis.\nScore: "


def safe_eval(
    text: str,
    idx: int = -1,
    abstract: str = "",
    hypothesis: str = "",
    default: int = 0,
    logger=None,
) -> int:
    """Safely evaluate model output, handling empty or invalid responses."""
    text = text.strip()
    if not text:
        if logger:
            logger.warning(
                f"Empty model output at index {idx}, using default value {default}"
            )
            logger.warning(
                f"  Abstract: {abstract[:200]}..."
                if len(abstract) > 200
                else f"  Abstract: {abstract}"
            )
            logger.warning(f"  Hypothesis: {hypothesis}")
        return default
    try:
        result = ast.literal_eval(text)
        if result not in [0, 1]:
            if logger:
                logger.warning(
                    f"Invalid model output '{text}' at index {idx} (expected 0 or 1), using default {default}"
                )
                logger.warning(
                    f"  Abstract: {abstract[:200]}..."
                    if len(abstract) > 200
                    else f"  Abstract: {abstract}"
                )
                logger.warning(f"  Hypothesis: {hypothesis}")
            return default
        return result
    except (SyntaxError, NameError, ValueError) as e:
        if logger:
            logger.warning(
                f"Failed to evaluate model output '{text}' at index {idx}: {e}, using default {default}"
            )
            logger.warning(
                f"  Abstract: {abstract[:200]}..."
                if len(abstract) > 200
                else f"  Abstract: {abstract}"
            )
            logger.warning(f"  Hypothesis: {hypothesis}")
        return default


def gen(
    prompts: RaggedTensor,
    model: TritonClient,
    logger=None,
    max_workers: int = None,
    show_progress: bool = False,
    batch_chunk_size: int = None,
) -> RaggedTensor:
    """Generate outputs using Triton client batch inference with configurable parameters.

    Args:
        prompts: RaggedTensor containing input prompts
        model: TritonClient instance for inference (with pre-configured sampling parameters)
        logger: Logger instance for debugging
        max_workers: Maximum concurrent requests (default: None = use client default)
        show_progress: Show progress bar during inference (default: False)
        batch_chunk_size: Process large batches in chunks (default: None = no chunking)

    Returns:
        RaggedTensor containing model outputs
    """
    # Debug: log first few prompts
    if logger:
        logger.info(f"DEBUG gen(): Number of prompts: {len(prompts.data)}")
        if len(prompts.data) > 0:
            logger.info(
                f"DEBUG gen(): First prompt (truncated): {prompts.data[0][:300]}..."
            )
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

    # Debug: log first few results and check for errors
    if logger:
        logger.info(f"DEBUG gen(): Number of results: {len(batch_results)}")
        error_count = sum(1 for r in batch_results if "error" in r)
        if error_count > 0:
            logger.warning(
                f"DEBUG gen(): {error_count}/{len(batch_results)} requests failed"
            )
        if len(batch_results) > 0:
            logger.info(f"DEBUG gen(): First result: {batch_results[0]}")

    outputs = RaggedTensor(
        [result.get("text_output", "") for result in batch_results], prompts.break_point
    )
    return outputs


def getPrompts(abstracts: RaggedTensor, hypotheses: RaggedTensor) -> RaggedTensor:
    assert not abstracts.is2D(), "abstracts should be flattened."
    assert not hypotheses.is2D(), "hypotheses should be flattened."
    return RaggedTensor(
        [prompt(abstracts[i], hypotheses[i]) for i in range(abstracts.shape)],
        hypotheses.break_point,
    )


def _truncate_for_log(text: str, max_len: int) -> str:
    """Truncate text for logging, appending ellipsis if needed."""
    if len(text) > max_len:
        return f"{text[:max_len]}..."
    return text


def postProcess(
    config: Config,
    outputs: RaggedTensor,
    abstracts: RaggedTensor,
    hypotheses: RaggedTensor,
    out_df: pd.DataFrame,
    terms: str,
    shape: list,
) -> None:
    flat_abstracts = abstracts.data.copy() if abstracts.data else []
    flat_hypotheses = hypotheses.data.copy() if hypotheses.data else []
    abstracts.reshape(shape)

    logger = config.logger
    logger.info(f"Processing {len(outputs.data)} abstracts for {terms} relationship")

    # Evaluate outputs
    evaluated_results = []
    for idx, answer in enumerate(outputs.data):
        abstract = flat_abstracts[idx] if idx < len(flat_abstracts) else ""
        hypothesis = flat_hypotheses[idx] if idx < len(flat_hypotheses) else ""

        # In debug mode, extract first char from chain-of-thought answer
        text_to_eval = answer[0] if config.debug and answer else answer
        result = safe_eval(text_to_eval, idx, abstract, hypothesis, 0, logger)
        evaluated_results.append(result)

        # Log evaluation
        relevance_status = "RELEVANT" if result == 1 else "NOT RELEVANT"
        log_fn = logger.info if config.debug else logger.debug
        log_fn(f"[{terms}] Abstract {idx}: {relevance_status}")

        if config.debug:
            logger.info(f"  First char: '{text_to_eval}'")
            logger.info(f"  Full answer: {_truncate_for_log(answer, 500)}")
        else:
            logger.debug(f"  Model output: '{answer.strip()}'")

        log_fn(f"  Hypothesis: {_truncate_for_log(hypothesis, 150)}")
        log_fn(f"  Abstract: {_truncate_for_log(abstract, 200)}")

    answer_masks = RaggedTensor(evaluated_results, outputs.break_point)
    answer_masks.reshape(shape)

    if config.debug:
        cot = RaggedTensor([answer[1:] for answer in outputs.data])
        cot.reshape(shape)
        out_df[f"{terms}_cot"] = cot.data
        out_df[f"{terms}_hypothesis"] = hypotheses.data

    abstracts.applyFilter(answer_masks)
    out_df[f"{terms}_mask"] = answer_masks.data
    out_df[f"{terms}_abstracts"] = abstracts.data

    # Log filtering statistics
    flat_masks = flatten_nested(answer_masks.data)
    total_count = len(flat_masks)
    relevant_count = sum(flat_masks)
    logger.info(
        f"[{terms}] Filtering summary: {relevant_count}/{total_count} abstracts marked RELEVANT, {total_count - relevant_count} filtered out"
    )

    if config.debug:
        excluded_indices = [idx for idx, mask in enumerate(flat_masks) if mask == 0]
        if excluded_indices:
            preview = excluded_indices[:20]
            suffix = "..." if len(excluded_indices) > 20 else ""
            logger.info(
                f"[{terms}] The following {len(excluded_indices)} abstract indices were marked NOT RELEVANT and should be excluded from sampling: {preview}{suffix}"
            )


def process_dataframe(
    out_df: pd.DataFrame, config: Config, pubmed_fetcher: PubMedFetcher
) -> pd.DataFrame:
    """Process dataframe with optimizations and filtering."""
    logger = config.logger
    columns_to_process = [
        col
        for col in ["ab_abstracts", "bc_abstracts", "ac_abstracts"]
        if col in out_df.columns
    ]

    num_intersections = len(columns_to_process)
    logger.info(f"Processing {num_intersections} intersections")

    for column in columns_to_process:
        # Optimize text length with evenly distributed tokens
        out_df[column] = out_df[column].apply(
            lambda x: pubmed_fetcher.optimize_text_length(
                x,
                max_tokens=110000000,  # Total tokens across all intersections
                num_intersections=num_intersections,
            )
        )
        # Sort by year and limit to top N if configured
        if config.post_n > 0:
            out_df[column] = out_df[column].apply(
                lambda x: pubmed_fetcher.interleave_abstracts(
                    x,
                    config.post_n,
                    config.top_n_articles_most_cited,
                    config.top_n_articles_most_recent,
                )
            )
    logger.debug(f"out_df in classifier process_dataframe: {out_df}")
    out_df = calculate_relevance_ratios(out_df, config)
    return out_df


def _deduplicate_by_pmid(entries: list, seen_pmids: set) -> list:
    """Deduplicate entries by PMID, updating seen_pmids in place."""
    result = []
    for text in entries:
        pmid = extract_pmid(text)
        if pmid:
            if pmid in seen_pmids:
                continue
            seen_pmids.add(pmid)
        result.append(text)
    return result


def _compute_sampling_allocation(
    total1: int, total2: int, target_total: int, min_floor: float
) -> tuple[int, int]:
    """Compute sample sizes for two pools with proportional allocation.

    Returns (n1, n2) sample counts respecting availability and floor constraints.
    """
    total = total1 + total2
    if total == 0:
        return 0, 0

    # Compute proportional shares with minimum floor
    s1 = max(total1 / total, min_floor) if total1 > 0 else 0.0
    s2 = max(total2 / total, min_floor) if total2 > 0 else 0.0

    # Normalize shares
    sum_s = s1 + s2
    if sum_s > 0:
        s1, s2 = s1 / sum_s, s2 / sum_s

    # Initial allocation with rounding
    n1 = int(round(s1 * target_total)) if total1 > 0 else 0
    n2 = int(round(s2 * target_total)) if total2 > 0 else 0

    # Adjust to hit target_total
    diff = target_total - (n1 + n2)
    while diff > 0:
        cap1, cap2 = total1 - n1, total2 - n2
        if (s1 >= s2 and cap1 > 0) or cap2 <= 0:
            if cap1 > 0:
                n1 += 1
        elif cap2 > 0:
            n2 += 1
        diff -= 1

    while diff < 0:
        if n1 >= n2 and n1 > 0:
            n1 -= 1
        elif n2 > 0:
            n2 -= 1
        diff += 1

    # Cap by availability
    n1 = min(n1, total1)
    n2 = min(n2, total2)

    # Top up from the other side if under target
    remaining = target_total - (n1 + n2)
    if remaining > 0:
        add1 = min(remaining, total1 - n1)
        n1 += add1
        remaining -= add1
        n2 += min(remaining, total2 - n2)

    return n1, n2


def sample_consolidated_abstracts(v1, v2, config: Config) -> tuple[str, int, int]:
    """Sample from two abstract collections; return consolidated text, sampled count, total deduped count.

    Args:
        v1: First collection of abstracts (list or single string or empty).
        v2: Second collection of abstracts (list or single string or empty).
        config: Global configuration providing sampling parameters.

    Returns:
        A tuple of (consolidated_abstracts: str, expected_count: int, total_relevant_abstracts: int)
    """
    logger = config.logger

    # Normalize and deduplicate
    seen_pmids: set = set()
    list1 = _deduplicate_by_pmid(normalize_entries(v1), seen_pmids)
    list2 = _deduplicate_by_pmid(normalize_entries(v2), seen_pmids)

    total1, total2 = len(list1), len(list2)

    # Log PMIDs in the sampling pool
    pool_pmids1 = [extract_pmid(abstract) for abstract in list1]
    pool_pmids2 = [extract_pmid(abstract) for abstract in list2]

    logger.info(f"Sampling pool: Candidate 1 has {total1} deduplicated abstracts")
    logger.info(
        f"  Candidate 1 PMIDs in pool: {pool_pmids1[:20]}{'...' if len(pool_pmids1) > 20 else ''}"
    )
    logger.info(f"Sampling pool: Candidate 2 has {total2} deduplicated abstracts")
    logger.info(
        f"  Candidate 2 PMIDs in pool: {pool_pmids2[:20]}{'...' if len(pool_pmids2) > 20 else ''}"
    )

    logger.debug(f"entities_in_candidate1: {total1}")
    logger.debug(f"entities_in_candidate2: {total2}")
    logger.debug(f"entities_total: {total1 + total2}")

    # Configurable parameters
    min_floor = float(config.global_settings.get("DCH_MIN_SAMPLING_FRACTION", 0.06))
    target_total = int(config.global_settings.get("DCH_SAMPLE_SIZE", 50))

    n1, n2 = _compute_sampling_allocation(total1, total2, target_total, min_floor)

    # Perform sampling
    sampled1 = random.sample(list1, n1) if n1 > 0 else []
    sampled2 = random.sample(list2, n2) if n2 > 0 else []

    # Log sampled PMIDs
    sampled_pmids1 = [extract_pmid(abstract) for abstract in sampled1]
    sampled_pmids2 = [extract_pmid(abstract) for abstract in sampled2]

    logger.info(f"Sampling: Selected {n1}/{total1} abstracts from candidate 1")
    logger.info(
        f"  Candidate 1 PMIDs sampled: {sampled_pmids1[:10]}{'...' if len(sampled_pmids1) > 10 else ''}"
    )
    logger.info(f"Sampling: Selected {n2}/{total2} abstracts from candidate 2")
    logger.info(
        f"  Candidate 2 PMIDs sampled: {sampled_pmids2[:10]}{'...' if len(sampled_pmids2) > 10 else ''}"
    )

    logger.debug(f"sampled1: len {len(sampled1)} {sampled1}")
    logger.debug(f"sampled2: len {len(sampled2)} {sampled2}")

    sampled_abstracts = sampled1 + sampled2
    logger.info(
        f"Total sampled: {len(sampled_abstracts)} abstracts ({n1} from candidate1 + {n2} from candidate2)"
    )
    logger.debug(
        f"num_sampled_candidate1: {n1}, num_sampled_candidate2: {n2}, total_sampled: {len(sampled_abstracts)}"
    )

    consolidated = "\n\n".join(sampled_abstracts) if sampled_abstracts else ""
    return consolidated, len(sampled_abstracts), total1 + total2


def _get_output_filename(row: pd.Series, config: Config) -> str:
    """Generate the output JSON filename based on config mode and row data."""
    if config.is_dch:
        hyp1_name = sanitize_term_for_filename(row.get("hypothesis1", "hypothesis1"))
        hyp2_name = sanitize_term_for_filename(row.get("hypothesis2", "hypothesis2"))
        return f"{hyp1_name}_vs_{hyp2_name}_km_with_gpt_direct_comp.json"

    a_fname = sanitize_term_for_filename(row.get("a_term", ""))
    b_fname = sanitize_term_for_filename(row.get("b_term", ""))

    if config.is_skim_with_gpt:
        c_fname = sanitize_term_for_filename(row.get("c_term", ""))
        return f"{a_fname}_{c_fname}_{b_fname}_skim_with_gpt.json"

    return f"{a_fname}_{b_fname}_km_with_gpt.json"


def process_results(
    out_df: pd.DataFrame, config: Config, num_abstracts_fetched: int
) -> None:
    """Process results and write to JSON files."""
    logger = config.logger
    total_rows = len(out_df)
    logger.info(f"Processing {total_rows} results...")

    # Determine output directory based on iteration
    output_base_dir = config.km_output_dir
    if config.iterations and config.current_iteration > 0:
        output_base_dir = os.path.join(
            output_base_dir, f"iteration_{config.current_iteration}"
        )
        os.makedirs(output_base_dir, exist_ok=True)
        logger.info(f"Writing results to iteration directory: {output_base_dir}")
    else:
        logger.info(f"Writing results to base output directory: {output_base_dir}")

    if config.is_dch:
        # Build direct comparison hypotheses
        a_terms_clean = [strip_pipe(a_term) for a_term in out_df["a_term"]]
        b_terms_clean = [strip_pipe(b_term) for b_term in out_df["b_term"]]
        hypotheses = [
            getHypothesis(config=config, a_term=a, b_term=b)
            for a, b in zip(a_terms_clean, b_terms_clean)
        ]
        logger.debug(f"hypotheses: {hypotheses}")
        logger.debug(f"hyp1: {hypotheses[0]}")
        logger.debug(f"hyp2: {hypotheses[1]}")

        # Consolidate abstracts from both candidate rows
        v1 = flatten_nested(out_df.iloc[0].get("ab_abstracts", []))
        v2 = flatten_nested(out_df.iloc[1].get("ab_abstracts", []))
        logger.info(f"DCH Sampling: Candidate 1 has {len(v1)} relevant abstracts")
        logger.info(f"DCH Sampling: Candidate 2 has {len(v2)} relevant abstracts")

        consolidated_abstracts, expected_count, total_relevant = (
            sample_consolidated_abstracts(v1, v2, config)
        )
        out_df = pd.DataFrame(
            [
                {
                    "hypothesis1": hypotheses[0],
                    "hypothesis2": hypotheses[1],
                    "ab_abstracts": consolidated_abstracts,
                    "expected_per_abstract_count": expected_count,
                    "total_relevant_abstracts": total_relevant,
                }
            ]
        )

    for index, row in out_df.iterrows():
        result_dict = process_single_row(row, config)
        logger.debug(f" Result dict: {result_dict}")
        if not result_dict:
            continue

        # Add hypothesis for standard KM outputs
        if config.is_km_with_gpt and not config.is_dch:
            try:
                hyp_str = getHypothesis(
                    config=config,
                    a_term=row.get("a_term", ""),
                    b_term=row.get("b_term", ""),
                )
                if "A_B_Relationship" in result_dict:
                    result_dict["A_B_Relationship"].setdefault("Hypothesis", hyp_str)
            except Exception:
                pass

        # Add relevance ratios
        for ratio_type in ["ab", "bc", "ac"]:
            ratio_col = f"{ratio_type}_relevance_ratio"
            fraction_col = f"{ratio_type}_relevance_fraction"
            if ratio_col in out_df.columns and fraction_col in out_df.columns:
                result_dict[f"{ratio_type}_relevance"] = (
                    f"{row[ratio_col]:.2f} ({row[fraction_col]})"
                )

        result_dict["num_abstracts_fetched"] = num_abstracts_fetched

        # DCH-specific processing
        if config.is_dch:
            try:
                result_dict["total_relevant_abstracts"] = int(
                    row.get("total_relevant_abstracts", 0)
                )
            except Exception:
                result_dict["total_relevant_abstracts"] = 0
            logger.info(f"Processed row {index + 1}/{total_rows} (DCH)")
        else:
            logger.info(f"Processed row {index + 1}/{total_rows} ({row['b_term']})")

        output_json = _get_output_filename(row, config)
        logger.debug(f" IN PROCESS RESULTS   Output json before writing: {output_json}")
        logger.debug(f" IN PROCESS RESULTS   Result dict: {result_dict}")
        write_to_json([result_dict], output_json, output_base_dir, config)


def run_relevance_analysis(config: Config, km_output_path: str) -> None:
    """Run relevance analysis on the provided TSV file.

    Args:
        config: Initialized Config object
        km_output_path: Path to the TSV file to process
    """

    logger = config.logger
    logger.debug(f"config: {config}")
    logger.debug(f"km_output_path: {km_output_path}")
    config.load_km_output(km_output_path)
    start_time = time.time()
    logger.info("Starting relevance analysis...")

    try:
        # DNS resolution test in Python
        host = "eutils.ncbi.nlm.nih.gov"
        ip_address = socket.gethostbyname(host)
        logger.debug(
            f"Python DNS resolution test: Successfully resolved '{host}' to '{ip_address}'"
        )
    except socket.gaierror as e:
        logger.error(
            f"Python DNS resolution test: Failed to resolve '{host}'. Error: {e}"
        )
        raise  # Re-raise the exception to stop script execution

    out_df = config.data.copy(deep=True)
    logger.debug(f"Working with dataframe of shape {out_df.shape}")

    # Initialize PubMedFetcher
    pubmed_fetcher = PubMedFetcher(
        config=config,
        email="jfreeman@morgridge.org",
        api_key=config.secrets["PUBMED_API_KEY"],
        max_retries=config.max_retries,
        backoff_factor=0.5,
    )
    logger.info("Initialized PubMedFetcher")

    # Process each row individually (unified for DCH and non-DCH)
    ab_pmids = []
    ab_hypotheses = []

    for _, row in config.data.iterrows():
        a_term = row["a_term"]
        logger.debug(
            f"Row b_term from dataframe: {row['b_term']}, type: {type(row['b_term'])}"
        )
        b_term = row["b_term"]

        # Convert string representation of list to actual list
        pmids = ast.literal_eval(row["ab_pmid_intersection"])
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
            b_term = row["b_term"]
            c_term = row["c_term"]
            a_term = row["a_term"]

            # Process BC terms
            bc_pmid_list = ast.literal_eval(row["bc_pmid_intersection"])
            bc_pmids.append(bc_pmid_list)
            bc_hypothesis = getHypothesis(config=config, c_term=c_term, b_term=b_term)
            bc_hypotheses.append(bc_hypothesis)

            # Process AC terms if available
            if config.has_ac and "ac_pmid_intersection" in row:
                ac_pmid_list = ast.literal_eval(row["ac_pmid_intersection"])
                ac_pmids.append(ac_pmid_list)
                ac_hypothesis = getHypothesis(
                    config=config, a_term=a_term, c_term=c_term
                )
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
    # PubMedFetcher already prefixes with PMID, so we don't need to add it here
    abstracts = all_pmids.map(lambda pmid: abstract_map.get(str(pmid), ""))

    # Ensure abstracts remain flattened for prompt generation; DCH merging happens in process_results

    # Initialize Triton client for inference
    logger.info("Initializing Triton client for remote inference...")
    try:
        # Initialize with configuration from relevance_filter section, including sampling parameters
        model = TritonClient(
            server_url=config.filter_config.get("SERVER_URL"),
            model_name=config.filter_config.get("MODEL_NAME"),
            temperature=config.filter_config["TEMPERATURE"],
            top_p=config.filter_config["TOP_P"],
            max_tokens=config.filter_config["MAX_COT_TOKENS"] if config.debug else 1,
        )

        # Check server health
        if not model.check_server_health():
            logger.error(f"Triton server at {model.server_url} is not ready")
            raise RuntimeError(f"Triton server at {model.server_url} is not responding")

        logger.info(f"Successfully connected to Triton server at {model.server_url}")
        logger.info(f"Using model: {model.model_name}")
    except Exception as e:
        logger.error(f"Failed to initialize Triton client: {e}")
        raise

    # Get optional performance tuning parameters from config
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

    prompts = getPrompts(abstracts, all_hypotheses)
    answers = gen(
        prompts,
        model,
        logger,
        max_workers=max_workers,
        show_progress=show_progress,
        batch_chunk_size=batch_chunk_size,
    )

    defaults = 3 * [RaggedTensor([])]
    ab_outputs, bc_outputs, ac_outputs, *_ = chain(answers.split(), defaults)
    ab_abstracts, bc_abstracts, ac_abstracts, *_ = chain(abstracts.split(), defaults)

    postProcess(
        config, ab_outputs, ab_abstracts, ab_hypotheses, out_df, "ab", ab_pmids.shape
    )

    if config.is_skim_with_gpt:
        postProcess(
            config,
            bc_outputs,
            bc_abstracts,
            bc_hypotheses,
            out_df,
            "bc",
            bc_pmids.shape,
        )
        if config.has_ac:
            postProcess(
                config,
                ac_outputs,
                ac_abstracts,
                ac_hypotheses,
                out_df,
                "ac",
                ac_pmids.shape,
            )

    # Skip process_dataframe for DCH mode - sampling handles context window sizing
    if not config.is_dch:
        out_df = process_dataframe(out_df, config, pubmed_fetcher)

    # --- ENRICHMENT STEP ---
    # Now that we have relevance scores and filtered abstracts, selectively fetch full text for relevant articles.
    # We need to act on the dataframe contents.

    if getattr(config, "full_text", False):
        logger.info("Full-text enrichment enabled. Identifying relevant PMIDs...")
        relevant_pmids = set()

        # Collect relevant PMIDs from out_df columns
        # The columns containing abstracts (e.g., 'ab_abstracts') might have been modified/sampled.
        # But we also have the RaggedTensor masks/data that were used.
        # However, it's easier to iterate through the dataframe which lines up with our outputs.
        # Wait, out_df['ab_abstracts'] etc contain lists of text.
        # We need to parse PMIDs from them if we want to enrich them.
        # Or better, we can use the 'ab_mask' etc columns if we have access to the original PMIDs.
        # The 'process_results' step does filtering. 'postProcess' updates 'out_df'.
        # 'all_pmids' was configured earlier.

        # Let's iterate through the rows and checking for relevant items.
        # But wait, out_df[col] usually holds a list of STRINGS (abstracts).
        # We need to update these specific strings in place or map them.

        pmids_to_enrich = []

        # Helper to extract PMIDs from relevant entries
        def collect_relevant(df, col_prefix):
            mask_col = f"{col_prefix}_mask"
            abs_col = f"{col_prefix}_abstracts"
            if mask_col in df.columns and abs_col in df.columns:
                for idx, row in df.iterrows():
                    masks = row[mask_col]
                    abstracts = row[abs_col]

                    if not isinstance(masks, list):
                        continue

                    # Handle case where pandas stored single abstract as string
                    if isinstance(abstracts, str):
                        abstracts = [abstracts]

                    if not isinstance(abstracts, list):
                        continue

                    for m, text in zip(masks, abstracts):
                        if m == 1:  # Relevant
                            pmid = extract_pmid(text)
                            if pmid:
                                pmids_to_enrich.append(pmid)

        collect_relevant(out_df, "ab")
        if config.is_skim_with_gpt:
            collect_relevant(out_df, "bc")
            if config.has_ac:
                collect_relevant(out_df, "ac")

        # Deduplicate
        pmids_to_enrich = list(set(pmids_to_enrich))
        logger.info(
            f"Identified {len(pmids_to_enrich)} distinctive relevant articles for enrichment."
        )

        if pmids_to_enrich:
            # Fetch raw data to allow for figure processing
            enriched_data_map = pubmed_fetcher.fetch_full_text_context(
                pmids_to_enrich, return_raw=True
            )

            # Access underlying data lists from RaggedTensors
            all_pmids_list = all_pmids.data
            all_hypotheses_list = all_hypotheses.data
            # Initialize ImageAnalyzer
            logger.info("Initializing ImageAnalyzer for figure transcription...")
            image_analyzer = None
            try:
                image_analyzer = ImageAnalyzer(
                    secrets=config.secrets,
                    model_name=config.full_text_model,
                    logger=logger,
                )
            except Exception as e:
                logger.error(f"Failed to initialize ImageAnalyzer: {e}")

            # Process figures and prepare final content map
            enriched_content_map = {}
            for pmid, data in enriched_data_map.items():
                pmid_str = str(pmid)
                figures = data.get("figures", [])
                pmcid = data.get("pmcid")

                # Full text as string for context in image analysis
                full_text_body = pubmed_fetcher._format_fulltext_complete(data)

                if figures and image_analyzer and pmcid:
                    logger.info(
                        f"Processing {len(figures)} figures for PMID {pmid_str} (PMCID: {pmcid})"
                    )
                    # Create temporary directory for figures
                    temp_fig_dir = os.path.join(
                        config.km_output_dir, "figures", pmid_str
                    )
                    os.makedirs(temp_fig_dir, exist_ok=True)

                    try:
                        # 1. Download figures
                        figures = pubmed_fetcher._download_figures_from_package(
                            pmcid, figures, temp_fig_dir
                        )

                        # 2. Analyze figures (Transcription only)
                        # Filter to only those that were actually downloaded
                        downloaded_figures = [f for f in figures if "local_path" in f]
                        if downloaded_figures:
                            if config.use_hypothesis_for_figures:
                                figure_hypothesis = (
                                    get_hypothesis_from_all_hypotheses(
                                        all_hypotheses_list, all_pmids_list, pmid_str
                                    )
                                    or ""
                                )
                            else:
                                figure_hypothesis = ""
                            analyzed_figures = (
                                image_analyzer.enhance_figure_descriptions(
                                    downloaded_figures,
                                    full_text_body,
                                    figure_hypothesis,
                                )
                            )

                            # Update original figures list with results
                            fig_map = {f["id"]: f for f in analyzed_figures}
                            for f in figures:
                                if f["id"] in fig_map:
                                    f.update(fig_map[f["id"]])

                        # 3. Reinject transcriptions into sections
                        # We need to replace [[FIGURE:id]] in the section text
                        sections = data.get("sections", {})
                        for sec_name, sec_text in sections.items():
                            for fig in figures:
                                fig_id = fig.get("id")
                                transcription = fig.get(
                                    "enhanced_content", fig.get("caption", "")
                                )
                                placeholder = f"[[FIGURE:{fig_id}]]"
                                if placeholder in sec_text:
                                    replacement = f"\n\n[FIGURE ANALYSIS {fig_id}]: {transcription}\n\n"
                                    sections[sec_name] = sec_text.replace(
                                        placeholder, replacement
                                    )
                                    logger.debug(
                                        f"Reinjected transcription for {fig_id} into section {sec_name}"
                                    )

                    except Exception as e:
                        logger.error(
                            f"Error processing figures for PMID {pmid_str}: {e}"
                        )

                # Re-format the (now potentially enriched) data
                final_text = pubmed_fetcher._format_fulltext_complete(data)
                enriched_content_map[pmid_str] = (
                    f"PMID: {pmid_str}\n[FULL-TEXT]\n{final_text}\n\n===END OF FULL TEXT===\n\n"
                )

            # 5. Chunking Agent: Extract Evidence
            logger.info("Running Chunking Agent on enriched texts...")
            # Initialize Chunker with Gemini model details from config
            chunker = FullTextChunker(
                secrets=config.secrets, model_name=config.full_text_model, logger=logger
            )

            # Access underlying data lists

            all_pmids_list = all_pmids.data
            all_hypotheses_list = all_hypotheses.data

            evidence_map = {}
            for pmid in pmids_to_enrich:
                # pmid in pmids_to_enrich is string (from extract_pmid)
                pmid_str = str(pmid)
                hypothesis = get_hypothesis_from_all_hypotheses(
                    all_hypotheses_list, all_pmids_list, pmid_str
                )
                # Find index in all_pmids_list to get hypothesis
                # all_pmids_list items might be int or str
                # idx = -1
                # if pmid_str in all_pmids_list:
                #     idx = all_pmids_list.index(pmid_str)
                # else:
                #     try:
                #         pmid_int = int(pmid_str)
                #         if pmid_int in all_pmids_list:
                #             idx = all_pmids_list.index(pmid_int)
                #     except ValueError:
                #         pass

                if hypothesis:
                    if pmid_str in enriched_content_map:
                        full_text = enriched_content_map[pmid_str]
                        logger.debug(
                            f"Chunking PMID {pmid_str} with hypothesis: {hypothesis[:50]}..."
                        )
                        try:
                            evidence = chunker.chunk_document(full_text, hypothesis)
                            evidence_map[pmid_str] = evidence
                        except Exception as e:
                            logger.error(f"Chunking failed for {pmid_str}: {e}")
                            evidence_map[pmid_str] = f"Error: {e}"
                    else:
                        evidence_map[pmid_str] = "Full text not available."
                else:
                    logger.warning(f"Could not find hypothesis for PMID {pmid_str}")

            # Save Evidence for potential downstream usage and debugging
            if evidence_map:
                sample_key = list(evidence_map.keys())[0]
                logger.info(
                    f"Sample Evidence for {sample_key}:\n{evidence_map[sample_key][:200]}..."
                )

                # Save artifacts: Raw Full Text and Chunked Evidence
                try:
                    debug_dir = os.path.dirname(
                        config.debug_tsv_name
                        if config.debug
                        else config.filtered_tsv_name
                    )
                    if not os.path.exists(debug_dir):
                        # Should ideally exist by now if output dir was created, but safety check
                        # config.filtered_tsv_name is usually in output_DIR/filename
                        debug_dir = os.path.dirname(config.filtered_tsv_name)

                    # Create a specialized debug folder inside the output dir if preferred, or just use `debug/` from project root?
                    # config.filtered_tsv_name path is usually /.../output_TIMESTAMP/filename.tsv.
                    # Let's verify where 'debug/' is relative to the output.
                    # Usually main.py creates iteration dirs.
                    # We will save adjacent to the TSV for now.

                    full_text_raw_path = os.path.join(debug_dir, "full_text_raw.json")
                    full_text_chunked_path = os.path.join(
                        debug_dir, "full_text_chunked.json"
                    )

                    with open(full_text_raw_path, "w") as f:
                        json.dump(enriched_content_map, f, indent=2)
                    logger.info(f"Saved raw full text artifact to {full_text_raw_path}")

                    with open(full_text_chunked_path, "w") as f:
                        json.dump(evidence_map, f, indent=2)
                    logger.info(
                        f"Saved chunked evidence artifact to {full_text_chunked_path}"
                    )

                except Exception as e:
                    logger.error(f"Failed to save full text artifacts: {e}")

            # Now update the dataframe content
            def update_content(df, col_prefix):
                abs_col = f"{col_prefix}_abstracts"
                if abs_col in df.columns:
                    updated_column = []
                    total_replaced = 0
                    total_abstracts = 0
                    for idx, row in df.iterrows():
                        abstracts_data = row[abs_col]

                        # Handle case where abstracts is a string (concatenated by process_dataframe)
                        is_string = isinstance(abstracts_data, str)
                        if is_string:
                            # Split back into individual entries for processing
                            abstracts_list = normalize_entries(abstracts_data)
                        elif isinstance(abstracts_data, list):
                            abstracts_list = abstracts_data
                        else:
                            updated_column.append(abstracts_data)
                            continue

                        new_abs_list = []
                        for text in abstracts_list:
                            total_abstracts += 1
                            pmid = extract_pmid(text)
                            # FIX: Use EVIDENCE (chunked) if available, otherwise fallback to raw -> abstract
                            # IMPORTANT: Preserve PMID prefix so downstream URL generation works
                            if pmid and str(pmid) in evidence_map:
                                # Preserve PMID prefix for citation tracking
                                evidence_with_pmid = f"PMID: {pmid}\n[ENRICHED EVIDENCE]\n{evidence_map[str(pmid)]}"
                                new_abs_list.append(evidence_with_pmid)
                                total_replaced += 1
                                logger.debug(
                                    f"Updating PMID {pmid} with chunked evidence"
                                )
                            elif pmid and str(pmid) in enriched_content_map:
                                # Fallback if chunking failed/wasn't done but we have full text
                                new_abs_list.append(enriched_content_map[str(pmid)])
                                total_replaced += 1
                                logger.debug(
                                    f"Updating PMID {pmid} with raw full text (fallback)"
                                )
                            else:
                                new_abs_list.append(text)

                        if is_string:
                            # Re-concatenate if it was originally a string
                            updated_column.append(
                                "\n\n===END OF ABSTRACT===\n\n".join(new_abs_list)
                                + "\n\n===END OF ABSTRACT===\n\n"
                            )
                        else:
                            updated_column.append(new_abs_list)

                    df[abs_col] = updated_column
                    if total_abstracts > 0:
                        logger.info(
                            f"[{col_prefix}] Replaced {total_replaced}/{total_abstracts} abstracts with full-text evidence"
                        )

            update_content(out_df, "ab")
            if config.is_skim_with_gpt:
                update_content(out_df, "bc")
                if config.has_ac:
                    update_content(out_df, "ac")

            logger.info(
                "Enrichment complete. Dataframe updated with full text content."
            )
    # --- END ENRICHMENT STEP ---

    # Save the initial processed dataframe
    initial_output_file = (
        config.debug_tsv_name if config.debug else config.filtered_tsv_name
    )
    out_df.to_csv(initial_output_file, sep="\t")
    logger.info(f"Saved initial processed data to {initial_output_file}")

    # Check if we need to run iterations
    if config.iterations:
        # Determine number of iterations
        num_iterations = 1
        if isinstance(config.iterations, bool) and config.iterations:
            logger.warning(
                "iterations is set to True but no number specified, defaulting to 1 iteration"
            )
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
                logger.info(
                    f"Created output directory for iteration {i}: {iteration_dir}"
                )

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
            logger.info(
                f"Iteration {iteration} completed in {iteration_elapsed_time:.2f} seconds"
            )

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


def get_hypothesis_from_all_hypotheses(
    all_hypotheses_list: list, all_pmids_list: list, pmid_str: str
) -> str:
    # Find index in all_pmids_list to get hypothesis
    # all_pmids_list items might be int or str
    idx = -1
    if pmid_str in all_pmids_list:
        idx = all_pmids_list.index(pmid_str)
    else:
        try:
            pmid_int = int(pmid_str)
            if pmid_int in all_pmids_list:
                idx = all_pmids_list.index(pmid_int)
        except ValueError:
            pass
    if idx != -1:
        hypothesis = str(all_hypotheses_list[idx])
    else:
        hypothesis = None
    return hypothesis
