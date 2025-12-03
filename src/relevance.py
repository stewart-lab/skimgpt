from __future__ import annotations
import ast
import pandas as pd
import os
import socket
import time
from itertools import chain
import random
from src.utils import Config, RaggedTensor, sanitize_term_for_filename, strip_pipe, normalize_entries, write_to_json, PMID_PATTERN, extract_pmid
from src.pubmed_fetcher import PubMedFetcher
from src.classifier import (
    calculate_relevance_ratios,
    process_single_row,
)
from src.triton_client import TritonClient


def getHypothesis(
    config: Config, a_term: str = None, b_term: str = None, c_term: str = None
) -> str:

    # Replace ampersands with spaces in all terms for hypothesis generation
    if a_term:
        a_term = a_term.replace("&", " ")
    if b_term:
        b_term = b_term.replace("&", " ")
    if c_term:
        c_term = c_term.replace("&", " ")

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
            hypothesis_template = config.skim_hypotheses["AB"]
            return hypothesis_template.format(a_term=a_term, b_term=b_term)
        elif b_term and c_term and not a_term:
            hypothesis_template = config.skim_hypotheses["BC"]
            return hypothesis_template.format(b_term=b_term, c_term=c_term)
        elif a_term and c_term and not b_term:
            hypothesis_template = config.skim_hypotheses["rel_AC"]
            return hypothesis_template.format(a_term=a_term, c_term=c_term)

    return f"No valid hypothesis for the provided {config.job_type}."


def prompt(abstract, hyp) -> str:
    return f"Abstract: {abstract}\nHypothesis: {hyp}\nInstructions: Classify this abstract as either 0 (Not Relevant) or 1 (Relevant) for evaluating the provided hypothesis.\nScore: "


def safe_eval(text: str, idx: int = -1, abstract: str = "", hypothesis: str = "", default: int = 0, logger = None) -> int:
    """Safely evaluate model output, handling empty or invalid responses."""
    text = text.strip()
    if not text:
        if logger:
            logger.warning(f"Empty model output at index {idx}, using default value {default}")
            logger.warning(f"  Abstract: {abstract[:200]}..." if len(abstract) > 200 else f"  Abstract: {abstract}")
            logger.warning(f"  Hypothesis: {hypothesis}")
        return default
    try:
        result = ast.literal_eval(text)
        if result not in [0, 1]:
            if logger:
                logger.warning(f"Invalid model output '{text}' at index {idx} (expected 0 or 1), using default {default}")
                logger.warning(f"  Abstract: {abstract[:200]}..." if len(abstract) > 200 else f"  Abstract: {abstract}")
                logger.warning(f"  Hypothesis: {hypothesis}")
            return default
        return result
    except (SyntaxError, NameError, ValueError) as e:
        if logger:
            logger.warning(f"Failed to evaluate model output '{text}' at index {idx}: {e}, using default {default}")
            logger.warning(f"  Abstract: {abstract[:200]}..." if len(abstract) > 200 else f"  Abstract: {abstract}")
            logger.warning(f"  Hypothesis: {hypothesis}")
        return default


def gen(
    prompts: RaggedTensor, model: TritonClient, logger=None,
    max_workers: int = None, show_progress: bool = False, batch_chunk_size: int = None
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
            logger.info(f"DEBUG gen(): First prompt (truncated): {prompts.data[0][:300]}...")
            logger.info(f"DEBUG gen(): Model sampling params - temperature: {model.temperature}, "
                       f"top_p: {model.top_p}, max_tokens: {model.max_tokens}")
        if max_workers:
            logger.info(f"DEBUG gen(): Using max_workers={max_workers}")
        if batch_chunk_size:
            logger.info(f"DEBUG gen(): Using batch_chunk_size={batch_chunk_size}")
    
    batch_results = model.generate_batch(
        text_inputs=prompts.data,
        max_workers=max_workers,
        show_progress=show_progress,
        batch_chunk_size=batch_chunk_size
    )
    
    # Debug: log first few results and check for errors
    if logger:
        logger.info(f"DEBUG gen(): Number of results: {len(batch_results)}")
        error_count = sum(1 for r in batch_results if "error" in r)
        if error_count > 0:
            logger.warning(f"DEBUG gen(): {error_count}/{len(batch_results)} requests failed")
        if len(batch_results) > 0:
            logger.info(f"DEBUG gen(): First result: {batch_results[0]}")
    
    outputs = RaggedTensor(
        [result.get("text_output", "") for result in batch_results], 
        prompts.break_point
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
    # Save flat references before reshaping for logging context
    flat_abstracts = abstracts.data.copy() if abstracts.data else []
    flat_hypotheses = hypotheses.data.copy() if hypotheses.data else []

    abstracts.reshape(shape)
    
    logger = config.logger
    logger.info(f"Processing {len(outputs.data)} abstracts for {terms} relationship")

    if not config.debug:
        # Non-debug mode: simple evaluation
        evaluated_results = []
        for idx, output in enumerate(outputs.data):
            abstract = flat_abstracts[idx] if idx < len(flat_abstracts) else ""
            hypothesis = flat_hypotheses[idx] if idx < len(flat_hypotheses) else ""
            result = safe_eval(output, idx, abstract, hypothesis, 0, logger)
            evaluated_results.append(result)
            
            # Log each evaluation
            relevance_status = "RELEVANT" if result == 1 else "NOT RELEVANT"
            logger.debug(f"[{terms}] Abstract {idx}: {relevance_status}")
            logger.debug(f"  Model output: '{output.strip()}'")
            logger.debug(f"  Hypothesis: {hypothesis[:150]}..." if len(hypothesis) > 150 else f"  Hypothesis: {hypothesis}")
            logger.debug(f"  Abstract: {abstract[:200]}..." if len(abstract) > 200 else f"  Abstract: {abstract}")
        
        answer_masks = RaggedTensor(evaluated_results, outputs.break_point)
        answer_masks.reshape(shape)
        abstracts.applyFilter(answer_masks)
    else:
        # Debug mode: evaluation with chain-of-thought
        evaluated_results = []
        for idx, answer in enumerate(outputs.data):
            abstract = flat_abstracts[idx] if idx < len(flat_abstracts) else ""
            hypothesis = flat_hypotheses[idx] if idx < len(flat_hypotheses) else ""
            first_char = answer[0] if answer else ""
            result = safe_eval(first_char, idx, abstract, hypothesis, 0, logger)
            evaluated_results.append(result)
            
            # Log each evaluation with full answer (CoT)
            relevance_status = "RELEVANT" if result == 1 else "NOT RELEVANT"
            logger.info(f"[{terms}] Abstract {idx}: {relevance_status}")
            logger.info(f"  First char: '{first_char}'")
            logger.info(f"  Full answer: {answer[:500]}..." if len(answer) > 500 else f"  Full answer: {answer}")
            logger.info(f"  Hypothesis: {hypothesis[:150]}..." if len(hypothesis) > 150 else f"  Hypothesis: {hypothesis}")
            logger.info(f"  Abstract: {abstract[:200]}..." if len(abstract) > 200 else f"  Abstract: {abstract}")
        
        answer_masks = RaggedTensor(evaluated_results, outputs.break_point)
        answer_masks.reshape(shape)
        cot = RaggedTensor([answer[1:] for answer in outputs.data])
        cot.reshape(shape)

        if terms == "ac":
            out_df[f"{terms}_mask"] = answer_masks.data
            out_df[f"{terms}_cot"] = cot.data
            out_df[f"{terms}_hypothesis"] = hypotheses.data
        else:
            out_df[f"{terms}_mask"] = answer_masks.data
            out_df[f"{terms}_cot"] = cot.data
            out_df[f"{terms}_hypothesis"] = hypotheses.data
        
        # In debug mode, still filter abstracts for downstream processing
        # The mask is saved separately for debugging purposes
        abstracts.applyFilter(answer_masks)

    out_df[f"{terms}_mask"] = answer_masks.data
    out_df[f"{terms}_abstracts"] = abstracts.data
    
    # Log filtering statistics - flatten the data if it's nested (after reshape)
    def flatten_if_needed(data):
        """Flatten nested lists into a single list"""
        if data and isinstance(data[0], list):
            return [item for sublist in data for item in sublist]
        return data
    
    flat_masks = flatten_if_needed(answer_masks.data)
    total_abstracts = len(flat_masks)
    relevant_count = sum(flat_masks)
    filtered_count = total_abstracts - relevant_count
    logger.info(f"[{terms}] Filtering summary: {relevant_count}/{total_abstracts} abstracts marked RELEVANT, {filtered_count} filtered out")
    
    # In debug mode, abstracts aren't filtered in-place, so we need to log which ones will be excluded
    if config.debug:
        excluded_indices = [idx for idx, mask in enumerate(flat_masks) if mask == 0]
        if excluded_indices:
            logger.info(f"[{terms}] The following {len(excluded_indices)} abstract indices were marked NOT RELEVANT and should be excluded from sampling: {excluded_indices[:20]}{'...' if len(excluded_indices) > 20 else ''}")


def process_dataframe(out_df: pd.DataFrame, config: Config, pubmed_fetcher: PubMedFetcher) -> pd.DataFrame:
    """Process dataframe with optimizations and filtering."""
    logger = config.logger
    columns_to_process = [col for col in [
        "ab_abstracts",
        "bc_abstracts",
        "ac_abstracts"
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

    list1 = normalize_entries(v1)
    list2 = normalize_entries(v2)

    # Deduplicate across rows using PMID
    seen_pmids = set()
    dedup1 = []
    for t in list1:
        pmid = extract_pmid(t)
        if pmid:
            if pmid in seen_pmids:
                continue
            seen_pmids.add(pmid)
        dedup1.append(t)

    dedup2 = []
    for t in list2:
        pmid = extract_pmid(t)
        if pmid:
            if pmid in seen_pmids:
                continue
            seen_pmids.add(pmid)
        dedup2.append(t)

    list1 = dedup1
    list2 = dedup2

    total1 = len(list1)
    total2 = len(list2)
    
    # Log PMIDs in the sampling pool
    pool_pmids1 = [extract_pmid(abstract) for abstract in list1]
    pool_pmids2 = [extract_pmid(abstract) for abstract in list2]
    
    logger.info(f"Sampling pool: Candidate 1 has {total1} deduplicated abstracts")
    logger.info(f"  Candidate 1 PMIDs in pool: {pool_pmids1[:20]}{'...' if len(pool_pmids1) > 20 else ''}")
    logger.info(f"Sampling pool: Candidate 2 has {total2} deduplicated abstracts")
    logger.info(f"  Candidate 2 PMIDs in pool: {pool_pmids2[:20]}{'...' if len(pool_pmids2) > 20 else ''}")
    
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
    
    # Extract PMIDs from sampled abstracts for logging
    sampled_pmids1 = [extract_pmid(abstract) for abstract in sampled1]
    sampled_pmids2 = [extract_pmid(abstract) for abstract in sampled2]
    
    logger.info(f"Sampling: Selected {n1}/{total1} abstracts from candidate 1")
    logger.info(f"  Candidate 1 PMIDs sampled: {sampled_pmids1[:10]}{'...' if len(sampled_pmids1) > 10 else ''}")
    logger.info(f"Sampling: Selected {n2}/{total2} abstracts from candidate 2")
    logger.info(f"  Candidate 2 PMIDs sampled: {sampled_pmids2[:10]}{'...' if len(sampled_pmids2) > 10 else ''}")
    
    logger.debug(f"sampled1: len {len(sampled1)} {sampled1}")
    logger.debug(f"sampled2: len {len(sampled2)} {sampled2}")
    sampled_abstracts = sampled1 + sampled2
    logger.info(f"Total sampled: {len(sampled_abstracts)} abstracts ({n1} from candidate1 + {n2} from candidate2)")
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
    output_base_dir = config.km_output_dir
    
    # If we're in an iteration, use the iteration subdirectory
    if config.iterations and config.current_iteration > 0:
        iteration_dir = f"iteration_{config.current_iteration}"
        output_base_dir = os.path.join(output_base_dir, iteration_dir)
        # Make sure the directory exists
        os.makedirs(output_base_dir, exist_ok=True)
        logger.info(f"Writing results to iteration directory: {output_base_dir}")
    else:
        logger.info(f"Writing results to base output directory: {output_base_dir}")

    if config.is_dch:
        # Build direct comparison hypotheses and provide consolidated text content
        # Strip pipes from individual terms BEFORE formatting hypotheses
        a_terms_clean = [strip_pipe(a_term) for a_term in out_df['a_term']]
        b_terms_clean = [strip_pipe(b_term) for b_term in out_df['b_term']]
        hypotheses = [getHypothesis(config=config, a_term=a_term, b_term=b_term) for a_term, b_term in zip(a_terms_clean, b_terms_clean)]
        logger.debug(f"hypotheses: {hypotheses}")
        hyp1 = hypotheses[0]
        hyp2 = hypotheses[1]
        logger.debug(f"hyp1: {hyp1}")
        logger.debug(f"hyp2: {hyp2}")

        # Consolidate abstracts/text from both candidate rows (if present)
        v1_all_raw = out_df.iloc[0].get("ab_abstracts", [])
        v2_all_raw = out_df.iloc[1].get("ab_abstracts", [])
        
        # Flatten if needed (abstracts might be nested after RaggedTensor reshape)
        def flatten_if_nested(data):
            if data and isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                return [item for sublist in data for item in sublist]
            return data
        
        v1_all = flatten_if_nested(v1_all_raw)
        v2_all = flatten_if_nested(v2_all_raw)
        
        # Abstracts are now already filtered in postProcess (even in debug mode)
        # so we can use them directly
        v1 = v1_all
        v2 = v2_all
        logger.info(f"DCH Sampling: Candidate 1 has {len(v1)} relevant abstracts")
        logger.info(f"DCH Sampling: Candidate 2 has {len(v2)} relevant abstracts")

        # Use deduplicated pool sizes from sampling function (pre-trim, cross-row dedup): total1 + total2
        consolidated_abstracts, expected_count, total_relevant_abstracts = sample_consolidated_abstracts(v1, v2, config)

        # Store hypotheses (already cleaned via strip_pipe on individual terms)
        dch_row = {
            "hypothesis1": hyp1,
            "hypothesis2": hyp2,
            "ab_abstracts": consolidated_abstracts,
            "expected_per_abstract_count": expected_count,
            "total_relevant_abstracts": total_relevant_abstracts,
        }
        out_df = pd.DataFrame([dch_row])

    for index, row in out_df.iterrows():
        result_dict = process_single_row(row, config)
        logger.debug(f" Result dict: {result_dict}")
        if result_dict:
            # Ensure Hypothesis is present for standard KM outputs (non-DCH)
            if config.is_km_with_gpt and not config.is_dch:
                try:
                    a_term_val = row.get("a_term", "")
                    b_term_val = row.get("b_term", "")
                    hyp_str = getHypothesis(config=config, a_term=a_term_val, b_term=b_term_val)
                    if "A_B_Relationship" in result_dict:
                        result_dict["A_B_Relationship"].setdefault("Hypothesis", hyp_str)
                except Exception:
                    pass
            for ratio_type in ["ab", "bc", "ac"]:
                ratio_col = f"{ratio_type}_relevance_ratio"
                fraction_col = f"{ratio_type}_relevance_fraction"
                if ratio_col in out_df.columns and fraction_col in out_df.columns:
                    ratio = row[ratio_col]
                    fraction = row[fraction_col]
                    result_dict[f"{ratio_type}_relevance"] = f"{ratio:.2f} ({fraction})"

            result_dict["num_abstracts_fetched"] = num_abstracts_fetched
            
            # DCH-specific processing
            if config.is_dch:
                # Append authoritative total relevant count from relevance filter
                try:
                    result_dict["total_relevant_abstracts"] = int(row.get("total_relevant_abstracts", 0))
                except Exception:
                    result_dict["total_relevant_abstracts"] = 0
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

    # Initialize Triton client for inference
    logger.info("Initializing Triton client for remote inference...")
    try:
        # Initialize with configuration from relevance_filter section, including sampling parameters
        model = TritonClient(
            server_url=config.filter_config.get("SERVER_URL"),
            model_name=config.filter_config.get("MODEL_NAME"),
            temperature=config.filter_config["TEMPERATURE"],
            top_p=config.filter_config["TOP_P"],
            max_tokens=config.filter_config["MAX_COT_TOKENS"] if config.debug else 1
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
    answers = gen(prompts, model, logger, 
                  max_workers=max_workers,
                  show_progress=show_progress,
                  batch_chunk_size=batch_chunk_size)

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
    
    # Skip process_dataframe for DCH mode - sampling handles context window sizing
    if not config.is_dch:
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
