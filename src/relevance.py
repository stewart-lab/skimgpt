from __future__ import annotations
import pandas as pd
import vllm
import argparse
from src.utils import Config, RaggedTensor, setup_logger
from itertools import chain
from typing import List, Dict, Any
import time
from src.pubmed_fetcher import PubMedFetcher
from src.classifier import process_single_row, write_to_json, calculate_relevance_ratios
import logging
import os
import json

# Initialize the centralized logger
logger = setup_logger()


def getHypothesis(
    config, a_term: str = None, b_term: str = None, c_term: str = None
) -> str:
    job_type = config.get("JOB_TYPE", "").lower()

    if job_type == "km_with_gpt":
        assert a_term and b_term and not c_term
        hypothesis_template = config.get("KM_hypothesis", "")
        return hypothesis_template.format(a_term=a_term, b_term=b_term)

    elif job_type == "skim_with_gpt":
        assert (
            (a_term and b_term and not c_term)
            or (b_term and c_term and not a_term)
            or (a_term and c_term and not b_term)
        )

        if a_term and b_term and not c_term:
            hypothesis_template = config.get("SKIM_hypotheses", "").get("AB")
            return hypothesis_template.format(a_term=a_term, b_term=b_term)
        elif b_term and c_term and not a_term:
            hypothesis_template = config.get("SKIM_hypotheses", "").get("BC")
            return hypothesis_template.format(b_term=b_term, c_term=c_term)
        elif a_term and c_term and not b_term:
            hypothesis_template = config.get("SKIM_hypotheses", "").get("rel_AC")
            return hypothesis_template.format(a_term=a_term, c_term=c_term)

    return "No valid hypothesis for the provided JOB_TYPE."


def prompt(abstract, hyp) -> str:
    return f"Abstract: {abstract}\nHypothesis: {hyp}\nInstructions: Classify this abstract as either 0 (Not Relevant) or 1 (Relevant) for evaluating the provided hypothesis.\nScore: "


def gen(
    prompts: RaggedTensor, model: any, sampling_config: vllm.SamplingParams
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
            out_df[f"{terms}_mask"] = answer_masks.data * len(out_df)
            out_df[f"{terms}_cot"] = cot.data * len(out_df)
            out_df[f"{terms}_hypothesis"] = hypotheses.data * len(out_df)
        else:
            out_df[f"{terms}_mask"] = answer_masks.data
            out_df[f"{terms}_cot"] = cot.data
            out_df[f"{terms}_hypothesis"] = hypotheses.data

    if terms == "ac":
        out_df[f"{terms}_pmid_intersection"] = abstracts.data * len(out_df)
        out_df[f"{terms}_mask"] = answer_masks.data * len(out_df)
    else:
        out_df[f"{terms}_mask"] = answer_masks.data
        out_df[f"{terms}_pmid_intersection"] = abstracts.data


def process_dataframe(out_df: pd.DataFrame, config: Config, pubmed_fetcher: PubMedFetcher) -> pd.DataFrame:
    """Process dataframe with optimizations and filtering."""
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
                max_tokens=110000,  # Total tokens across all intersections
                num_intersections=num_intersections
            )
        )
        # Sort by year and limit to top N if configured
        if config.post_n > 0:
            out_df[column] = out_df[column].apply(
                lambda x: pubmed_fetcher.sort_by_year(x, config.post_n)
            )
    
    out_df = calculate_relevance_ratios(out_df)
    return out_df


def process_results(out_df: pd.DataFrame, config: Config) -> None:
    """Process results and write to JSON files."""
    total_rows = len(out_df)
    logger.info(f"Processing {total_rows} results...")

    for index, row in out_df.iterrows():
        result_dict = process_single_row(row, config)
        if result_dict:
            for ratio_type in ["ab", "bc", "ac"]:
                ratio_col = f"{ratio_type}_relevance_ratio"
                fraction_col = f"{ratio_type}_relevance_fraction"
                if ratio_col in out_df.columns and fraction_col in out_df.columns:
                    ratio = row[ratio_col]
                    fraction = row[fraction_col]
                    result_dict[f"{ratio_type}_relevance"] = f"{ratio:.2f} ({fraction})"
            
            logger.info(f"Processed row {index + 1}/{total_rows} ({row['b_term']})")
            
            if config.is_skim_gpt:
                output_json = f"{row['a_term']}_{row['b_term']}_{row['c_term']}_skim_with_gpt.json"
            else:
                output_json = f"{row['a_term']}_{row['b_term']}_km_with_gpt.json"
            
            write_to_json([result_dict], output_json)
            logger.debug(f"Results saved to {output_json}")


def main():
    start_time = time.time()
    logger.info("Starting relevance analysis...")
    
    parser = argparse.ArgumentParser(description="Mistral7B Inference")
    parser.add_argument(
        "--km_output",
        type=str,
        required=True,
        help="Tsv file to run relevance filtering on.",
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Config file for kmGPT run."
    )
    args = parser.parse_args()
    
    config = Config(args)
    logger.info(f"Loaded configuration from {args.config}")
    
    out_df = config.data.copy(deep=True)
    logger.debug(f"Working with dataframe of shape {out_df.shape}")

    # Initialize PubMedFetcher
    pubmed_fetcher = PubMedFetcher(
        email="jfreeman@morgridge.org",
        api_key=config.job_config.get("PUBMED_API_KEY", ""),
        max_retries=config.job_config.get("GLOBAL_SETTINGS", {}).get("MAX_RETRIES", 3),
        backoff_factor=0.5
    )
    logger.info("Initialized PubMedFetcher")

    # Process terms and get hypotheses
    a_term = config.data.a_term.unique().tolist()[0].split("&")[0]
    b_terms = config.data.b_term.unique().tolist()

    ab_pmids = RaggedTensor([eval(lst) for lst in config.data.ab_pmid_intersection])
    ab_hypotheses = RaggedTensor(
        [getHypothesis(config.job_config, a_term=a_term, b_term=b_term) for b_term in b_terms]
    )

    all_pmids = ab_pmids.flatten()
    all_hypotheses = ab_hypotheses.expand(ab_pmids.shape)

    if config.is_skim_gpt:
        c_term = config.data.c_term.unique().tolist()[0]
        bc_pmids = RaggedTensor([eval(lst) for lst in config.data.bc_pmid_intersection])
        bc_hypotheses = RaggedTensor(
            [getHypothesis(config.job_config, c_term=c_term, b_term=b_term) for b_term in b_terms]
        )

        all_pmids += bc_pmids.flatten()
        all_hypotheses += bc_hypotheses.expand(bc_pmids.shape)

        if config.has_ac:
            ac_pmids = RaggedTensor(eval(config.data.ac_pmid_intersection[0]))
            ac_hypothesis = RaggedTensor(
                [getHypothesis(config.job_config, a_term=a_term, c_term=c_term)]
            )

            all_pmids += ac_pmids
            all_hypotheses += ac_hypothesis.expand([ac_pmids.shape])

    # Fetch abstracts
    abstract_map = pubmed_fetcher.fetch_abstracts(all_pmids)
    abstracts = all_pmids.map(lambda pmid: abstract_map.get(str(pmid), ""))

    # Model setup and inference
    model = vllm.LLM(model=config.filter_config["MODEL"], max_model_len=4000)
    sampling_config = vllm.SamplingParams(
        temperature=config.filter_config["TEMPERATURE"],
        top_k=config.filter_config["TOP_K"],
        top_p=config.filter_config["TOP_P"],
        max_tokens=config.filter_config["MAX_COT_TOKENS"] if config.debug else 1,
    )

    prompts = getPrompts(abstracts, all_hypotheses)
    answers = gen(prompts, model, sampling_config)

    # Post-processing
    defaults = 3 * [RaggedTensor([])]
    ab_outputs, bc_outputs, ac_outputs, *_ = chain(answers.split(), defaults)
    ab_abstracts, bc_abstracts, ac_abstracts, *_ = chain(abstracts.split(), defaults)

    postProcess(
        config, ab_outputs, ab_abstracts, ab_hypotheses, out_df, "ab", ab_pmids.shape
    )

    if config.is_skim_gpt:
        postProcess(
            config, bc_outputs, bc_abstracts, bc_hypotheses, out_df, "bc", bc_pmids.shape
        )
        if config.has_ac:
            postProcess(
                config, ac_outputs, ac_abstracts, ac_hypothesis, out_df, "ac", [ac_pmids.shape]
            )

    # Final processing and output
    out_df = process_dataframe(out_df, config, pubmed_fetcher)
    logger.info("Completed dataframe processing")

    if config.test_leakage:
        leakage_data = load_data("leakage.csv")
        out_df = update_ab_pmid_intersection(out_df, leakage_data, config.test_leakage_type)
        logger.info("Updated leakage intersection data")

    output_file = config.debug_tsv_name if config.debug else config.filtered_tsv_name
    out_df.to_csv(output_file, sep="\t")
    logger.info(f"Saved processed data to {output_file}")

    # Process results using the new function
    process_results(out_df, config)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Relevance analysis completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
