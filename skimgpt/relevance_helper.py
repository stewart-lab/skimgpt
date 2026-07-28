from __future__ import annotations
import ast
import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import chain
from typing import Callable

import pandas as pd
from skimgpt.utils import (
    Config, RaggedTensor, sanitize_term_for_filename,
    strip_pipe, normalize_entries, write_to_json,
    extract_pmid, get_hypothesis,
)
from skimgpt.pubmed_fetcher import PubMedFetcher
from skimgpt.classifier import calculate_relevance_ratios, process_single_row
from skimgpt.triton_client import TritonBatchFailureError
from skimgpt.image_analyzer import ImageAnalyzer
from skimgpt.full_text_chunker import FullTextChunker

logger = logging.getLogger(__name__)

# Cap on iterations running concurrently. 5 fits comfortably under GPT-5.5
# tier-3 RPM/TPM budgets for DCH-sized payloads while still giving most of the
# wallclock benefit (5× speedup ≈ 4 min instead of 20 min at iterations=20).
ITERATION_MAX_PARALLELISM = 5


def _flatten_if_nested(data):
    """Flatten one level of nesting if the data is a list of lists."""
    if data and isinstance(data, list) and isinstance(data[0], list):
        return [item for sublist in data for item in sublist]
    return data


def _dedup_by_pmid(abstracts, seen_pmids):
    """Return abstracts with duplicate PMIDs removed, updating seen_pmids in place."""
    deduped = []
    for text in abstracts:
        pmid = extract_pmid(text)
        if pmid:
            if pmid in seen_pmids:
                continue
            seen_pmids.add(pmid)
        deduped.append(text)
    return deduped



def prompt(abstract, hyp) -> str:
    return f"Abstract: {abstract}\nHypothesis: {hyp}\nInstructions: Classify this abstract as either 0 (Not Relevant) or 1 (Relevant) for evaluating the provided hypothesis.\nScore: "


def safe_eval(text: str, idx: int = -1, abstract: str = "", hypothesis: str = "", default: int = 0) -> int:
    """Safely evaluate model output, handling empty or invalid responses."""
    text = text.strip()
    if not text:
        logger.warning(f"Empty model output at index {idx}, using default value {default}")
        logger.warning(f"  Abstract: {abstract[:200]}..." if len(abstract) > 200 else f"  Abstract: {abstract}")
        logger.warning(f"  Hypothesis: {hypothesis}")
        return default
    try:
        result = ast.literal_eval(text)
        if result not in [0, 1]:
            logger.warning(f"Invalid model output '{text}' at index {idx} (expected 0 or 1), using default {default}")
            logger.warning(f"  Abstract: {abstract[:200]}..." if len(abstract) > 200 else f"  Abstract: {abstract}")
            logger.warning(f"  Hypothesis: {hypothesis}")
            return default
        return result
    except (SyntaxError, NameError, ValueError) as e:
        logger.warning(f"Failed to evaluate model output '{text}' at index {idx}: {e}, using default {default}")
        logger.warning(f"  Abstract: {abstract[:200]}..." if len(abstract) > 200 else f"  Abstract: {abstract}")
        logger.warning(f"  Hypothesis: {hypothesis}")
        return default


def get_prompts(abstracts: RaggedTensor, hypotheses: RaggedTensor) -> RaggedTensor:
    assert not abstracts.is_2d(), "abstracts should be flattened."
    assert not hypotheses.is_2d(), "hypotheses should be flattened."
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

    logger.info(f"Processing {len(outputs.data)} abstracts for {terms} relationship")

    if not config.debug:
        evaluated_results = []
        for idx, output in enumerate(outputs.data):
            abstract = flat_abstracts[idx] if idx < len(flat_abstracts) else ""
            hypothesis = flat_hypotheses[idx] if idx < len(flat_hypotheses) else ""
            result = safe_eval(output, idx, abstract, hypothesis, 0)
            evaluated_results.append(result)

            relevance_status = "RELEVANT" if result == 1 else "NOT RELEVANT"
            logger.debug(f"[{terms}] Abstract {idx}: {relevance_status}")
            logger.debug(f"  Model output: '{output.strip()}'")
            logger.debug(f"  Hypothesis: {hypothesis[:150]}..." if len(hypothesis) > 150 else f"  Hypothesis: {hypothesis}")
            logger.debug(f"  Abstract: {abstract[:200]}..." if len(abstract) > 200 else f"  Abstract: {abstract}")

        answer_masks = RaggedTensor(evaluated_results, outputs.break_point)
        answer_masks.reshape(shape)
    else:
        evaluated_results = []
        for idx, answer in enumerate(outputs.data):
            abstract = flat_abstracts[idx] if idx < len(flat_abstracts) else ""
            hypothesis = flat_hypotheses[idx] if idx < len(flat_hypotheses) else ""
            first_char = answer[0] if answer else ""
            result = safe_eval(first_char, idx, abstract, hypothesis, 0)
            evaluated_results.append(result)

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

        out_df[f"{terms}_cot"] = cot.data
        out_df[f"{terms}_hypothesis"] = hypotheses.data

    # Filter abstracts using the computed masks (both debug and non-debug)
    abstracts.apply_filter(answer_masks)

    out_df[f"{terms}_mask"] = answer_masks.data
    out_df[f"{terms}_abstracts"] = abstracts.data

    flat_masks = _flatten_if_nested(answer_masks.data)
    total_abstracts = len(flat_masks)
    relevant_count = sum(flat_masks)
    filtered_count = total_abstracts - relevant_count
    logger.info(f"[{terms}] Filtering summary: {relevant_count}/{total_abstracts} abstracts marked RELEVANT, {filtered_count} filtered out")

    if config.debug:
        excluded_indices = [idx for idx, mask in enumerate(flat_masks) if mask == 0]
        if excluded_indices:
            logger.info(f"[{terms}] The following {len(excluded_indices)} abstract indices were marked NOT RELEVANT and should be excluded from sampling: {excluded_indices[:20]}{'...' if len(excluded_indices) > 20 else ''}")


def process_dataframe(out_df: pd.DataFrame, config: Config, pubmed_fetcher: PubMedFetcher) -> pd.DataFrame:
    """Process dataframe with optimizations and filtering."""
    columns_to_process = [col for col in [
        "ab_abstracts",
        "bc_abstracts",
        "ac_abstracts"
    ] if col in out_df.columns]

    num_intersections = len(columns_to_process)
    logger.info(f"Processing {num_intersections} intersections")

    for column in columns_to_process:
        out_df[column] = out_df[column].apply(
            lambda x: pubmed_fetcher.optimize_text_length(
                x,
                max_tokens=110000000,
                num_intersections=num_intersections
            )
        )
        if config.post_n > 0:
            out_df[column] = out_df[column].apply(
                lambda x: pubmed_fetcher.interleave_abstracts(x, config.post_n, config.top_n_articles_most_cited, config.top_n_articles_most_recent)
            )
    logger.debug(f"out_df in classifier process_dataframe: {out_df}")
    out_df = calculate_relevance_ratios(out_df, config)
    return out_df


def _normalize_fractions(
    total1: int, total2: int, min_floor: float, target_total: int,
) -> tuple[int, int]:
    """Compute sample counts for two pools, enforcing a minimum fraction and target total.

    Returns:
        ``(n1, n2)`` — the number of items to sample from each pool.
    """
    total = total1 + total2
    if total == 0:
        return 0, 0

    s1 = total1 / total
    s2 = total2 / total

    if total1 > 0:
        s1 = max(s1, min_floor)
    if total2 > 0:
        s2 = max(s2, min_floor)

    if s1 == 0 and s2 > 0:
        s2 = 1.0
    elif s2 == 0 and s1 > 0:
        s1 = 1.0
    else:
        sum_s = s1 + s2
        s1 = s1 / sum_s if sum_s > 0 else 0.0
        s2 = s2 / sum_s if sum_s > 0 else 0.0

    n1 = int(round(s1 * target_total)) if total1 > 0 else 0
    n2 = int(round(s2 * target_total)) if total2 > 0 else 0

    diff = target_total - (n1 + n2)
    if diff > 0:
        for _ in range(diff):
            cap1 = total1 - n1
            cap2 = total2 - n2
            if (s1 >= s2 and cap1 > 0) or cap2 <= 0:
                n1 += 1 if cap1 > 0 else 0
            else:
                n2 += 1 if cap2 > 0 else 0
    elif diff < 0:
        for _ in range(-diff):
            if n1 >= n2 and n1 > 0:
                n1 -= 1
            elif n2 > 0:
                n2 -= 1

    n1 = min(n1, total1)
    n2 = min(n2, total2)

    remaining = target_total - (n1 + n2)
    if remaining > 0:
        add1 = min(remaining, total1 - n1)
        n1 += add1
        remaining -= add1
        if remaining > 0:
            add2 = min(remaining, total2 - n2)
            n2 += add2

    return n1, n2


def _sample_entries(entries: list, count: int) -> list:
    """Randomly sample *count* items from *entries* (returns [] when count is 0)."""
    if count > 0:
        return random.sample(entries, count)
    return []


def sample_consolidated_abstracts(v1, v2, config: Config):
    """Sample from two abstract collections; return consolidated text, sampled count, total deduped count.

    Args:
        v1: First collection of abstracts (list or single string or empty).
        v2: Second collection of abstracts (list or single string or empty).
        config: Global configuration providing sampling parameters.

    Returns:
        A tuple of (consolidated_abstracts: str, expected_count: int, total_relevant_abstracts: int)
    """
    list1 = normalize_entries(v1)
    list2 = normalize_entries(v2)

    # Deduplicate across both lists using PMID
    seen_pmids = set()
    list1 = _dedup_by_pmid(list1, seen_pmids)
    list2 = _dedup_by_pmid(list2, seen_pmids)

    total1 = len(list1)
    total2 = len(list2)

    pool_pmids1 = [extract_pmid(abstract) for abstract in list1]
    pool_pmids2 = [extract_pmid(abstract) for abstract in list2]

    logger.info(f"Sampling pool: Candidate 1 has {total1} deduplicated abstracts")
    logger.info(f"  Candidate 1 PMIDs in pool: {pool_pmids1[:20]}{'...' if len(pool_pmids1) > 20 else ''}")
    logger.info(f"Sampling pool: Candidate 2 has {total2} deduplicated abstracts")
    logger.info(f"  Candidate 2 PMIDs in pool: {pool_pmids2[:20]}{'...' if len(pool_pmids2) > 20 else ''}")

    logger.debug(f"entities_in_candidate1: {total1}")
    logger.debug(f"entities_in_candidate2: {total2}")
    logger.debug(f"entities_total: {total1 + total2}")

    min_floor = float(config.global_settings.get("DCH_MIN_SAMPLING_FRACTION", 0.06))
    target_total = int(config.global_settings.get("DCH_SAMPLE_SIZE", 50))

    n1, n2 = _normalize_fractions(total1, total2, min_floor, target_total)

    sampled1 = _sample_entries(list1, n1)
    sampled2 = _sample_entries(list2, n2)

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

    consolidated_abstracts = "\n\n".join(sampled_abstracts) if sampled_abstracts else ""
    total_relevant_abstracts = total1 + total2

    return consolidated_abstracts, len(sampled_abstracts), total_relevant_abstracts


def process_results(
    out_df: pd.DataFrame,
    config: Config,
    num_abstracts_fetched: int,
    output_base_dir: str,
    iteration_number: int = 0,
) -> None:
    """Process results and write to JSON files.

    Args:
        out_df: DataFrame with analysis results.
        config: Global configuration.
        num_abstracts_fetched: Total abstracts fetched for metadata.
        output_base_dir: Base directory for JSON output.  Resolved by the
            caller (e.g. ``config.km_output_dir`` for Triton, ``"output"``
            for CHTC).
        iteration_number: 1-indexed iteration index; 0 means "not iterated".
            Threaded through as a parameter (not read from config) so multiple
            iterations can run concurrently without racing on shared state.
    """
    total_rows = len(out_df)
    logger.info(f"Processing {total_rows} results...")

    if config.iterations and iteration_number > 0:
        iteration_dir = f"iteration_{iteration_number}"
        output_base_dir = os.path.join(output_base_dir, iteration_dir)
        os.makedirs(output_base_dir, exist_ok=True)
        logger.info(f"Writing results to iteration directory: {output_base_dir}")
    else:
        logger.info(f"Writing results to base output directory: {output_base_dir}")

    if config.is_dch:
        a_terms_clean = [strip_pipe(a_term) for a_term in out_df['a_term']]
        b_terms_clean = [strip_pipe(b_term) for b_term in out_df['b_term']]
        hypotheses = [get_hypothesis(config=config, a_term=a_term, b_term=b_term) for a_term, b_term in zip(a_terms_clean, b_terms_clean)]
        logger.debug(f"hypotheses: {hypotheses}")
        hyp1 = hypotheses[0]
        hyp2 = hypotheses[1]
        logger.debug(f"hyp1: {hyp1}")
        logger.debug(f"hyp2: {hyp2}")

        v1_all_raw = out_df.iloc[0].get("ab_abstracts", [])
        v2_all_raw = out_df.iloc[1].get("ab_abstracts", [])

        v1 = _flatten_if_nested(v1_all_raw)
        v2 = _flatten_if_nested(v2_all_raw)
        logger.info(f"DCH Sampling: Candidate 1 has {len(v1)} relevant abstracts")
        logger.info(f"DCH Sampling: Candidate 2 has {len(v2)} relevant abstracts")

        consolidated_abstracts, expected_count, total_relevant_abstracts = sample_consolidated_abstracts(v1, v2, config)

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
            if config.is_km_with_gpt and not config.is_dch:
                a_term_val = row.get("a_term", "")
                b_term_val = row.get("b_term", "")
                hyp_str = get_hypothesis(config=config, a_term=a_term_val, b_term=b_term_val)
                if "A_B_Relationship" in result_dict:
                    result_dict["A_B_Relationship"].setdefault("Hypothesis", hyp_str)
            for ratio_type in ["ab", "bc", "ac"]:
                ratio_col = f"{ratio_type}_relevance_ratio"
                fraction_col = f"{ratio_type}_relevance_fraction"
                if ratio_col in out_df.columns and fraction_col in out_df.columns:
                    ratio = row[ratio_col]
                    fraction = row[fraction_col]
                    result_dict[f"{ratio_type}_relevance"] = f"{ratio:.2f} ({fraction})"

            result_dict["num_abstracts_fetched"] = num_abstracts_fetched

            if config.is_dch:
                try:
                    result_dict["total_relevant_abstracts"] = int(row.get("total_relevant_abstracts", 0))
                except (TypeError, ValueError):
                    logger.warning("Could not parse total_relevant_abstracts, defaulting to 0")
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
            write_to_json([result_dict], output_json, output_base_dir)


def collect_pmids_and_hypotheses(config: Config):
    """Collect PMIDs and hypotheses from config.data for all term-pair intersections.

    Both relevance_chtc.py and relevance_triton.py need identical PMID/hypothesis
    collection logic. This function centralises that work.

    Returns:
        A dict with keys:
            ab_pmids, ab_hypotheses: always present (RaggedTensor)
            bc_pmids, bc_hypotheses: present when is_skim_with_gpt (RaggedTensor)
            ac_pmids, ac_hypotheses: present when is_skim_with_gpt and has_ac (RaggedTensor)
            all_pmids, all_hypotheses: flattened/expanded union (RaggedTensor)
    """
    ab_pmids_raw = []
    ab_hypotheses_raw = []
    bc_pmids_raw = []
    bc_hypotheses_raw = []
    ac_pmids_raw = []
    ac_hypotheses_raw = []

    for _, row in config.data.iterrows():
        a_term = row['a_term']
        b_term = row['b_term']

        ab_pmids_raw.append(ast.literal_eval(row['ab_pmid_intersection']))
        ab_hypotheses_raw.append(get_hypothesis(config=config, a_term=a_term, b_term=b_term))

        if config.is_skim_with_gpt:
            c_term = row['c_term']
            bc_pmids_raw.append(ast.literal_eval(row['bc_pmid_intersection']))
            bc_hypotheses_raw.append(get_hypothesis(config=config, c_term=c_term, b_term=b_term))

            if config.has_ac and 'ac_pmid_intersection' in row:
                ac_pmids_raw.append(ast.literal_eval(row['ac_pmid_intersection']))
                ac_hypotheses_raw.append(get_hypothesis(config=config, a_term=a_term, c_term=c_term))

    ab_pmids = RaggedTensor(ab_pmids_raw)
    ab_hypotheses = RaggedTensor(ab_hypotheses_raw)
    all_pmids = ab_pmids.flatten()
    all_hypotheses = ab_hypotheses.expand(ab_pmids.shape)

    result = {
        "ab_pmids": ab_pmids,
        "ab_hypotheses": ab_hypotheses,
    }

    if config.is_skim_with_gpt:
        bc_pmids = RaggedTensor(bc_pmids_raw)
        bc_hypotheses = RaggedTensor(bc_hypotheses_raw)
        all_pmids += bc_pmids.flatten()
        all_hypotheses += bc_hypotheses.expand(bc_pmids.shape)
        result["bc_pmids"] = bc_pmids
        result["bc_hypotheses"] = bc_hypotheses

        if config.has_ac and ac_pmids_raw:
            ac_pmids = RaggedTensor(ac_pmids_raw)
            ac_hypotheses = RaggedTensor(ac_hypotheses_raw)
            all_pmids += ac_pmids.flatten()
            all_hypotheses += ac_hypotheses.expand(ac_pmids.shape)
            result["ac_pmids"] = ac_pmids
            result["ac_hypotheses"] = ac_hypotheses

    result["all_pmids"] = all_pmids
    result["all_hypotheses"] = all_hypotheses
    return result


def run_iterations(config: Config, out_df: pd.DataFrame, num_abstracts_fetched: int,
                   output_base_dir: str) -> None:
    """Handle iteration-based or single-pass result processing.

    Both relevance_chtc.py and relevance_triton.py share nearly identical
    iteration handling code.  This function centralises it.

    Args:
        config: Global configuration.
        out_df: Processed DataFrame.
        num_abstracts_fetched: Total abstracts fetched for metadata.
        output_base_dir: Base directory for JSON output, forwarded to
            process_results.  Resolved by the caller (e.g.
            ``config.km_output_dir`` for Triton, ``"output"`` for CHTC).
    """
    if config.iterations:
        num_iterations = 1
        if isinstance(config.iterations, bool) and config.iterations:
            logger.warning("iterations is set to True but no number specified, defaulting to 1 iteration")
        elif isinstance(config.iterations, int) and config.iterations > 0:
            num_iterations = config.iterations
            logger.info(f"Will perform {num_iterations} iterations of analysis")
        else:
            logger.warning("Invalid iterations config, defaulting to 1 iteration")

        for i in range(1, num_iterations + 1):
            os.makedirs(os.path.join(output_base_dir, f"iteration_{i}"), exist_ok=True)

        max_parallel = min(ITERATION_MAX_PARALLELISM, num_iterations)
        total_start = time.time()
        logger.info(f"Running {num_iterations} iterations with parallelism={max_parallel}")

        def _run_one(iteration: int) -> tuple[int, float]:
            t0 = time.time()
            process_results(out_df, config, num_abstracts_fetched,
                            output_base_dir=output_base_dir,
                            iteration_number=iteration)
            return iteration, time.time() - t0

        with ThreadPoolExecutor(max_workers=max_parallel) as ex:
            futures = {ex.submit(_run_one, i): i
                       for i in range(1, num_iterations + 1)}
            for fut in as_completed(futures):
                iteration, dt = fut.result()
                logger.info(f"Iteration {iteration}/{num_iterations} completed in {dt:.2f}s")

        logger.info(f"All {num_iterations} iterations completed in "
                    f"{time.time() - total_start:.2f}s total wallclock")
    else:
        logger.info("No iterations requested, processing results once")
        process_results(out_df, config, num_abstracts_fetched,
                        output_base_dir=output_base_dir, iteration_number=0)


InferenceFn = Callable[[RaggedTensor], RaggedTensor]
SinglePromptInferFn = Callable[[str], dict]


def _streaming_fetch_and_infer(
    pubmed_fetcher: PubMedFetcher,
    all_pmids: RaggedTensor,
    all_hypotheses: RaggedTensor,
    single_prompt_infer: SinglePromptInferFn,
    max_workers: int,
) -> tuple[RaggedTensor, RaggedTensor, int]:
    """Pipeline PubMed fetch with single-prompt Triton inference.

    For each PubMed batch that lands, build the relevance prompt for every
    position in ``all_pmids.data`` that the batch's PMIDs map to and submit
    them to ``single_prompt_infer`` via a shared ``ThreadPoolExecutor``. The
    executor stays busy while later PubMed batches are still in flight, so
    the fetch step's wallclock cost is mostly absorbed into the inference
    step's wallclock.

    Returns ``(abstracts, answers, num_fetched)`` shaped to mirror
    ``all_pmids`` so downstream split/postProcess code is unchanged.

    Raises:
        TritonBatchFailureError: when every submitted request fails
            (error-dict result, exception, or drain timeout), so the caller
            can fall back to CHTC instead of silently emitting all-empty
            answers (which downstream reads as "no evidence").
    """
    from collections import defaultdict

    n = len(all_pmids.data)
    flat_abstracts: list[str] = [""] * n
    flat_answers: list[str] = [""] * n

    pmid_to_positions: dict[str, list[int]] = defaultdict(list)
    for i, pmid in enumerate(all_pmids.data):
        pmid_to_positions[str(pmid)].append(i)

    abstract_map: dict[str, str] = {}
    pos_by_fut: dict[object, int] = {}

    # Total drain time is bounded so a wedged keep-alive connection can't
    # block the whole run indefinitely. 30 min covers ~4000 prompts at the
    # pessimistic 2 rps floor; real workloads are 5-15x faster.
    drain_timeout_s = 1800

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for batch_dict in pubmed_fetcher.fetch_abstracts_iter(list(all_pmids)):
            abstract_map.update(batch_dict)
            for pmid_str, abstract_text in batch_dict.items():
                for pos in pmid_to_positions.get(pmid_str, []):
                    flat_abstracts[pos] = abstract_text
                    p = prompt(abstract_text, all_hypotheses.data[pos])
                    pos_by_fut[ex.submit(single_prompt_infer, p)] = pos

        total = len(pos_by_fut)
        logger.info(f"Streaming pipeline: {total} prompts submitted; "
                    f"waiting for inference to drain (timeout {drain_timeout_s}s)...")

        done = 0
        failed = 0
        drain_start = time.time()
        try:
            for fut in as_completed(pos_by_fut, timeout=drain_timeout_s):
                pos = pos_by_fut[fut]
                try:
                    r = fut.result(timeout=0)
                    if isinstance(r, dict) and "error" not in r:
                        flat_answers[pos] = r.get("text_output", "")
                    else:
                        # TritonClient.generate returns {"error": ...} instead
                        # of raising; it already logged the details.
                        failed += 1
                        flat_answers[pos] = ""
                except Exception as e:
                    logger.warning(f"Streaming prompt at position {pos} failed: {e}")
                    failed += 1
                    flat_answers[pos] = ""
                done += 1
                if done % 250 == 0 or done == total:
                    rps = done / max(time.time() - drain_start, 1e-6)
                    logger.info(f"Streaming pipeline: {done}/{total} "
                                f"completed ({rps:.1f} rps, {failed} failed)")
        except TimeoutError:
            stuck = sum(1 for f in pos_by_fut if not f.done())
            logger.error(f"Streaming pipeline: drain timeout — {stuck}/{total} "
                         f"prompts unfinished, marking as empty answers")
            for fut, pos in pos_by_fut.items():
                if not fut.done():
                    fut.cancel()
                    failed += 1
                    flat_answers[pos] = ""

    if failed:
        logger.warning(f"Streaming pipeline: {failed}/{total} requests failed")
    if total > 0 and failed == total:
        # Mirror the old batch-path semantics: all-failed means the server is
        # down or unreachable, so raise instead of returning all-empty answers
        # (which downstream would score as "no evidence"). The caller catches
        # this and falls back to a CHTC GPU job.
        raise TritonBatchFailureError(
            f"All {total} streaming Triton requests failed — server appears down or unreachable"
        )

    num_fetched = len(abstract_map)
    logger.info(f"Streaming pipeline: fetched {num_fetched} abstracts, "
                f"completed {len(pos_by_fut)} inferences")

    abstracts_rt = RaggedTensor(flat_abstracts, list(all_pmids.break_point))
    answers_rt = RaggedTensor(flat_answers, list(all_pmids.break_point))
    return abstracts_rt, answers_rt, num_fetched


def get_hypothesis_from_all_hypotheses(
    all_hypotheses_list: list, all_pmids_list: list, pmid_str: str
) -> str | None:
    """Map a PMID back to its hypothesis via the flattened pmid/hypothesis lists."""
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
        return str(all_hypotheses_list[idx])
    return None


def enrich_with_full_text(
    config: Config,
    out_df: pd.DataFrame,
    all_pmids: RaggedTensor,
    all_hypotheses: RaggedTensor,
    pubmed_fetcher: PubMedFetcher,
) -> None:
    """Enrich relevant abstracts in ``out_df`` with PMC full text + figure analysis.

    Runs after relevance scoring/filtering: identifies the PMIDs marked relevant,
    fetches their PMC full text, transcribes figures (Gemini), chunks the text
    against the per-PMID hypothesis, and rewrites the ``{prefix}_abstracts``
    columns in place with the enriched evidence. No-op unless ``config.full_text``.
    """
    if not getattr(config, "full_text", False):
        return

    logger.info("Full-text enrichment enabled. Identifying relevant PMIDs...")

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

    if not pmids_to_enrich:
        return

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

    evidence_map = {}
    for pmid in pmids_to_enrich:
        # pmid in pmids_to_enrich is string (from extract_pmid)
        pmid_str = str(pmid)
        hypothesis = get_hypothesis_from_all_hypotheses(
            all_hypotheses_list, all_pmids_list, pmid_str
        )

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
                debug_dir = os.path.dirname(config.filtered_tsv_name)

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
                    # Use EVIDENCE (chunked) if available, otherwise fallback to raw -> abstract.
                    # Preserve PMID prefix so downstream URL generation works.
                    if pmid and str(pmid) in evidence_map:
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

    logger.info("Enrichment complete. Dataframe updated with full text content.")


def run_relevance_pipeline(
    config: Config,
    km_output_path: str,
    infer: InferenceFn | None = None,
    *,
    output_base_dir: str | None = None,
    streaming_single_prompt_infer: SinglePromptInferFn | None = None,
    streaming_max_workers: int = 10,
) -> None:
    """Run the shared relevance-analysis pipeline.

    This function consolidates the orchestration logic common to both
    ``relevance_triton.py`` and ``relevance_chtc.py``.  The only part that
    differs between backends is the *inference* step: exactly one of
    ``infer`` (two-phase fetch-then-infer, CHTC) or
    ``streaming_single_prompt_infer`` (pipelined fetch+infer, Triton)
    must be provided.

    Args:
        config: Initialized Config object.
        km_output_path: Path to the TSV file to process.
        infer: Callable that takes a RaggedTensor of prompts and returns
            a RaggedTensor of model outputs.  Used by the two-phase path
            (CHTC); leave None when streaming.
        output_base_dir: Base directory for JSON output.  When *None*,
            ``config.km_output_dir`` is used (Triton path).  CHTC callers
            pass ``"output"``.
        streaming_single_prompt_infer: When provided, opt into pipelined
            fetch+infer: PubMed batches feed Triton submissions as they
            land instead of blocking on the full fetch first. Triton path
            passes ``triton_client.generate``. CHTC leaves this None
            and uses the legacy two-phase path.
        streaming_max_workers: Concurrent inference workers when streaming.
            Match to the Triton server's ``max_num_seqs`` admission cap
            (currently 10).

    Raises:
        TritonBatchFailureError: from the streaming path when every Triton
            request fails — callers catch this to fall back to CHTC.
    """
    if (infer is None) == (streaming_single_prompt_infer is None):
        raise ValueError(
            "Provide exactly one of 'infer' (two-phase) or "
            "'streaming_single_prompt_infer' (streaming)."
        )
    config.load_km_output(km_output_path)
    if output_base_dir is None:
        output_base_dir = config.km_output_dir
    start_time = time.time()
    logger.info("Starting relevance analysis...")

    out_df = config.data.copy(deep=True)
    logger.debug(f"Working with dataframe of shape {out_df.shape}")

    pubmed_fetcher = PubMedFetcher(
        config=config,
        email="jfreeman@morgridge.org",
        api_key=config.secrets["PUBMED_API_KEY"],
        max_retries=config.max_retries,
        backoff_factor=0.5,
    )
    logger.info("Initialized PubMedFetcher")

    # Collect PMIDs and hypotheses for all term-pair intersections
    collected = collect_pmids_and_hypotheses(config)
    ab_pmids = collected["ab_pmids"]
    ab_hypotheses = collected["ab_hypotheses"]
    all_pmids = collected["all_pmids"]
    all_hypotheses = collected["all_hypotheses"]
    bc_pmids = collected.get("bc_pmids")
    bc_hypotheses = collected.get("bc_hypotheses")
    ac_pmids = collected.get("ac_pmids")
    ac_hypotheses = collected.get("ac_hypotheses")

    # Fetch abstracts + run inference, either streamed (Triton) or two-phase (CHTC)
    if streaming_single_prompt_infer is not None:
        abstracts, answers, num_abstracts_fetched = _streaming_fetch_and_infer(
            pubmed_fetcher, all_pmids, all_hypotheses,
            streaming_single_prompt_infer, streaming_max_workers,
        )
    else:
        abstract_map = pubmed_fetcher.fetch_abstracts(all_pmids)
        num_abstracts_fetched = len(abstract_map)
        abstracts = all_pmids.map(lambda pmid: abstract_map.get(str(pmid), ""))
        prompts = get_prompts(abstracts, all_hypotheses)
        answers = infer(prompts)

    # Split answers and abstracts by intersection type
    defaults = [RaggedTensor([]) for _ in range(3)]
    ab_outputs, bc_outputs, ac_outputs, *_ = chain(answers.split(), defaults)
    ab_abstracts, bc_abstracts, ac_abstracts, *_ = chain(abstracts.split(), defaults)

    postProcess(config, ab_outputs, ab_abstracts, ab_hypotheses, out_df, "ab", ab_pmids.shape)

    if config.is_skim_with_gpt:
        postProcess(config, bc_outputs, bc_abstracts, bc_hypotheses, out_df, "bc", bc_pmids.shape)
        if config.has_ac:
            postProcess(config, ac_outputs, ac_abstracts, ac_hypotheses, out_df, "ac", ac_pmids.shape)

    # Skip process_dataframe for DCH mode — sampling handles context window sizing
    if not config.is_dch:
        out_df = process_dataframe(out_df, config, pubmed_fetcher)

    # Optionally enrich relevant abstracts with PMC full text + figure analysis
    enrich_with_full_text(config, out_df, all_pmids, all_hypotheses, pubmed_fetcher)

    # Save processed dataframe
    initial_output_file = config.debug_tsv_name if config.debug else config.filtered_tsv_name
    out_df.to_csv(initial_output_file, sep="\t")
    logger.info(f"Saved initial processed data to {initial_output_file}")

    run_iterations(config, out_df, num_abstracts_fetched, output_base_dir=output_base_dir)

    logger.info(f"Relevance analysis completed in {time.time() - start_time:.2f} seconds")
