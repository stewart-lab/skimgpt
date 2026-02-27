from __future__ import annotations
import ast
import os
import random
import pandas as pd
from src.utils import (
    Config, RaggedTensor, sanitize_term_for_filename,
    strip_pipe, normalize_entries, write_to_json,
    PMID_PATTERN, extract_pmid,
)
from src.pubmed_fetcher import PubMedFetcher
from src.classifier import calculate_relevance_ratios, process_single_row


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
        evaluated_results = []
        for idx, output in enumerate(outputs.data):
            abstract = flat_abstracts[idx] if idx < len(flat_abstracts) else ""
            hypothesis = flat_hypotheses[idx] if idx < len(flat_hypotheses) else ""
            result = safe_eval(output, idx, abstract, hypothesis, 0, logger)
            evaluated_results.append(result)

            relevance_status = "RELEVANT" if result == 1 else "NOT RELEVANT"
            logger.debug(f"[{terms}] Abstract {idx}: {relevance_status}")
            logger.debug(f"  Model output: '{output.strip()}'")
            logger.debug(f"  Hypothesis: {hypothesis[:150]}..." if len(hypothesis) > 150 else f"  Hypothesis: {hypothesis}")
            logger.debug(f"  Abstract: {abstract[:200]}..." if len(abstract) > 200 else f"  Abstract: {abstract}")

        answer_masks = RaggedTensor(evaluated_results, outputs.break_point)
        answer_masks.reshape(shape)
        abstracts.applyFilter(answer_masks)
    else:
        evaluated_results = []
        for idx, answer in enumerate(outputs.data):
            abstract = flat_abstracts[idx] if idx < len(flat_abstracts) else ""
            hypothesis = flat_hypotheses[idx] if idx < len(flat_hypotheses) else ""
            first_char = answer[0] if answer else ""
            result = safe_eval(first_char, idx, abstract, hypothesis, 0, logger)
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

        if terms == "ac":
            out_df[f"{terms}_mask"] = answer_masks.data
            out_df[f"{terms}_cot"] = cot.data
            out_df[f"{terms}_hypothesis"] = hypotheses.data
        else:
            out_df[f"{terms}_mask"] = answer_masks.data
            out_df[f"{terms}_cot"] = cot.data
            out_df[f"{terms}_hypothesis"] = hypotheses.data

        # In debug mode, still filter abstracts for downstream processing
        abstracts.applyFilter(answer_masks)

    out_df[f"{terms}_mask"] = answer_masks.data
    out_df[f"{terms}_abstracts"] = abstracts.data

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

    min_floor = float(config.global_settings.get("DCH_MIN_SAMPLING_FRACTION", 0.06))
    target_total = int(config.global_settings.get("DCH_SAMPLE_SIZE", 50))

    n1 = 0
    n2 = 0
    if total > 0:
        s1 = (total1 / total) if total > 0 else 0.0
        s2 = (total2 / total) if total > 0 else 0.0

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
        if diff != 0:
            if diff > 0:
                for _ in range(diff):
                    cap1 = total1 - n1
                    cap2 = total2 - n2
                    if (s1 >= s2 and cap1 > 0) or cap2 <= 0:
                        n1 += 1 if cap1 > 0 else 0
                    else:
                        n2 += 1 if cap2 > 0 else 0
            else:
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

    sampled1 = random.sample(list1, n1) if n1 > 0 else []
    sampled2 = random.sample(list2, n2) if n2 > 0 else []

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


def process_results(
    out_df: pd.DataFrame,
    config: Config,
    num_abstracts_fetched: int,
    output_base_dir: str = None,
) -> None:
    """Process results and write to JSON files.

    Args:
        out_df: DataFrame with analysis results.
        config: Global configuration.
        num_abstracts_fetched: Total abstracts fetched for metadata.
        output_base_dir: Base directory for JSON output.  When *None*
            (default / Triton path), ``config.km_output_dir`` is used.
            CHTC callers pass ``"output"`` (relative to the Docker scratch dir).
    """
    logger = config.logger
    total_rows = len(out_df)
    logger.info(f"Processing {total_rows} results...")

    if output_base_dir is None:
        output_base_dir = config.km_output_dir

    if config.iterations and config.current_iteration > 0:
        iteration_dir = f"iteration_{config.current_iteration}"
        output_base_dir = os.path.join(output_base_dir, iteration_dir)
        os.makedirs(output_base_dir, exist_ok=True)
        logger.info(f"Writing results to iteration directory: {output_base_dir}")
    else:
        logger.info(f"Writing results to base output directory: {output_base_dir}")

    if config.is_dch:
        a_terms_clean = [strip_pipe(a_term) for a_term in out_df['a_term']]
        b_terms_clean = [strip_pipe(b_term) for b_term in out_df['b_term']]
        hypotheses = [getHypothesis(config=config, a_term=a_term, b_term=b_term) for a_term, b_term in zip(a_terms_clean, b_terms_clean)]
        logger.debug(f"hypotheses: {hypotheses}")
        hyp1 = hypotheses[0]
        hyp2 = hypotheses[1]
        logger.debug(f"hyp1: {hyp1}")
        logger.debug(f"hyp2: {hyp2}")

        v1_all_raw = out_df.iloc[0].get("ab_abstracts", [])
        v2_all_raw = out_df.iloc[1].get("ab_abstracts", [])

        def flatten_if_nested(data):
            if data and isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                return [item for sublist in data for item in sublist]
            return data

        v1 = flatten_if_nested(v1_all_raw)
        v2 = flatten_if_nested(v2_all_raw)
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

            if config.is_dch:
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
