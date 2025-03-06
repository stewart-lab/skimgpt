from __future__ import annotations
import pandas as pd
import vllm
import argparse
from src.utils import Config, RaggedTensor
from itertools import chain
import time
from src.pubmed_fetcher import PubMedFetcher
from src.classifier import process_single_row, write_to_json, calculate_relevance_ratios
import socket
import tiktoken



def getHypothesis(
    config: Config, a_term: str = None, b_term: str = None, c_term: str = None
) -> str:
    logger = config.logger
    if config.is_km_with_gpt:
        assert a_term and b_term and not c_term
        hypothesis_template = config.km_hypothesis
        return hypothesis_template.format(a_term=a_term, b_term=b_term)
    elif config.is_km_with_gpt_direct_comp:
        logger.debug(f"config.is_km_with_gpt_direct_comp is: {config.is_km_with_gpt_direct_comp}")
        assert a_term and b_term and not c_term
        logger.debug(f"a_term: {a_term}, b_term: {b_term}")
        logger.debug(f"b_term is a list: {isinstance(b_term, list)}")
        if not isinstance(b_term, list):
            # Remove brackets and split by comma
            b_term_str = b_term.strip("[]")  # Remove brackets
            b_term_list = [item.strip() for item in b_term_str.split(',')] # Split by comma and strip whitespaceß
            # Filter out any empty strings that might result from the split
            b_term = [item for item in b_term_list if item]
            assert len(b_term) == 2
        logger.debug(f"b_term1: {b_term[0]}, b_term2: {b_term[1]}")
        logger.debug(f"b_term length: {len(b_term)}")
        logger.debug(f"b_term type: {type(b_term)}")
        hypothesis_template = config.km_direct_comp_hypothesis
        logger.debug(f"hypothesis_template: {hypothesis_template}")
        
        # Check if b_term is a list and extract b_term1 and b_term2
        if isinstance(b_term, list) and len(b_term) == 2:
            b_term1, b_term2 = b_term[0], b_term[1]
            return hypothesis_template.format(a_term=a_term, b_term1=b_term1, b_term2=b_term2)
        else:
            #   Handle the case where b_term is not a list or not of length 2 (optional error handling)
            logger.error("Error: b_term is not a list of exactly two terms for km_with_gpt_direct_comp")
            return "Error in hypothesis generation" # Or raise an exception
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
    logger = config.logger
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

    # Simply assign the data without multiplication
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
                max_tokens=110000,  # Total tokens across all intersections
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


def process_results(out_df: pd.DataFrame, config: Config, num_abstracts_fetched: int) -> None:
    logger = config.logger
    """Process results and write to JSON files."""
    total_rows = len(out_df)
    logger.info(f"Processing {total_rows} results...")

    for index, row in out_df.iterrows():
        result_dict = process_single_row(row, config)
        logger.debug(f" IN PROCESS RESULTS   Result dict: {result_dict}")
        if result_dict:
            for ratio_type in ["ab", "bc", "ac"]:
                ratio_col = f"{ratio_type}_relevance_ratio"
                fraction_col = f"{ratio_type}_relevance_fraction"
                if ratio_col in out_df.columns and fraction_col in out_df.columns:
                    ratio = row[ratio_col]
                    fraction = row[fraction_col]
                    result_dict[f"{ratio_type}_relevance"] = f"{ratio:.2f} ({fraction})"

            result_dict["num_abstracts_fetched"] = num_abstracts_fetched
            logger.info(f"Processed row {index + 1}/{total_rows} ({row['b_term']})")

            if config.is_skim_with_gpt:
                output_json = f"{row['a_term']}_{row['c_term']}_{row['b_term']}_skim_with_gpt.json"
            elif config.is_km_with_gpt_direct_comp:
                output_json = f"{row['a_term']}_{row['b_term']}_km_with_gpt_direct_comp.json"
            else:
                output_json = f"{row['a_term']}_{row['b_term']}_km_with_gpt.json"
            logger.debug(f" IN PROCESS RESULTS   Output json before writing: {output_json}")
            logger.debug(f" IN PROCESS RESULTS   Result dict: {result_dict}")
            write_to_json([result_dict], output_json, "output", config) 


def main():
    # Parse arguments FIRST
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
        logger.info(f"Python DNS resolution test: Successfully resolved '{host}' to '{ip_address}'")
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
    
    # Process each row individually
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

    if config.is_skim_with_gpt:
        postProcess(
            config, bc_outputs, bc_abstracts, bc_hypotheses, out_df, "bc", bc_pmids.shape
        )
        if config.has_ac:
            postProcess(
                config, ac_outputs, ac_abstracts, ac_hypotheses, out_df, "ac", ac_pmids.shape 
            )

    # Final processing and output
    out_df = process_dataframe(out_df, config, pubmed_fetcher)
    
    # Log token counts for abstracts in each row
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        
        logger.info("Token counts for abstracts in each row:")
        for idx, row in out_df.iterrows():
            row_token_counts = {}
            
            # Check and count tokens for each intersection type
            for col in ['ab_pmid_intersection', 'bc_pmid_intersection', 'ac_pmid_intersection']:
                if col in row and isinstance(row[col], str) and row[col]:
                    token_count = len(encoding.encode(row[col]))
                    row_token_counts[col] = token_count
            
            # Calculate total tokens for this row
            total_tokens = sum(row_token_counts.values())
            
            # Log the token counts
            term_info = f"Row {idx}: {row.get('a_term', '')} - {row.get('b_term', '')}"
            if 'c_term' in row:
                term_info += f" - {row['c_term']}"
            
            logger.info(f"{term_info} | Total tokens: {total_tokens} | Details: {row_token_counts}")
    except ImportError:
        logger.error("tiktoken not installed. Required for token counting.")
    except Exception as e:
        logger.error(f"Error counting tokens: {str(e)}")
    
    output_file = config.debug_tsv_name if config.debug else config.filtered_tsv_name
    out_df.to_csv(output_file, sep="\t")
    logger.info(f"Saved processed data to {output_file}")

    # Process results using the new function
    process_results(out_df, config, num_abstracts_fetched)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Relevance analysis completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
