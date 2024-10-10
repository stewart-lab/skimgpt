from __future__ import annotations
import pandas as pd
import json
from Bio import Entrez
import vllm
import argparse
from utils import Config, RaggedTensor
import tiktoken
from classifier import process_single_row, write_to_json, calculate_relevance_ratios
from itertools import chain
from leakage import load_data, update_ab_pmid_intersection, save_updated_data
import re
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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
            hypothesis_template = config.get("SKIM_hypotheses", "").get("AC")
            return hypothesis_template.format(a_term=a_term, c_term=c_term)

    else:
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


# Returns a dictionary for each PMID & Abstract Pair
# This method is needed since Entrez automatically removes duplicates in the pmid list


def getAbstractMap(config: json, pmids: list[str]) -> dict:
    returned_pmids = []
    returned_contents = []
    delimiter = "\n\n===END OF ABSTRACT===\n\n"
    Entrez.email = "your_email@example.com"  # Replace with your email
    Entrez.api_key = config["PUBMED_API_KEY"]
    Entrez.max_tries = config["GLOBAL_SETTINGS"]["MAX_RETRIES"]
    Entrez.sleep_between_tries = config["GLOBAL_SETTINGS"]["RETRY_DELAY"]

    try:
        efetch = Entrez.efetch(db="pubmed", id=pmids, retmode="xml", rettype="abstract")
        output = Entrez.read(efetch)
        efetch.close()
    except Exception as e:
        logging.error(f"Error fetching abstracts: {e}")
        return {}

    for paper in output["PubmedArticle"]:
        pmid = str(paper["MedlineCitation"]["PMID"])
        article = paper["MedlineCitation"]["Article"]

        # Extract the title
        title = article.get("ArticleTitle", "No title available")

        # Extract the abstract
        abstract_text = " ".join(
            article.get("Abstract", {}).get("AbstractText", ["No abstract available"])
        )
        # Check if the abstract has at least 50 words
        if len(abstract_text.split()) >= 50:
            returned_pmids.append(pmid)
            # Format the content with PMID, Title, and Abstract, separated by delimiter
            # Even though there's typically one abstract per PMID, the delimiter is included for consistency
            content = (
                f"PMID: {pmid}\nTitle: {title}\nAbstract: {abstract_text}{delimiter}"
            )
            returned_contents.append(content)

    if not returned_pmids:
        logging.warning("No valid abstracts found with at least 50 words.")
        return {}

    # Create a dictionary mapping PMIDs to their content with delimiters
    abstract_dict = dict(zip(returned_pmids, returned_contents))

    return abstract_dict


# Packages all the inputted data into the provided dataframes
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
        # If we're not debugging, the only output from the model will be a number from 0 to 1, so we can create answer masks
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

    # This is because we'll only ever have one AC relation in a tsv
    if terms == "ac":
        out_df[f"{terms}_pmid_intersection"] = abstracts.data * len(out_df)
        out_df[f"{terms}_mask"] = answer_masks.data * len(out_df)
    else:
        # Debug file doesn't have the filter applied.
        out_df[f"{terms}_mask"] = answer_masks.data
        out_df[f"{terms}_pmid_intersection"] = abstracts.data


def optimize_text_length(df, model="gpt-4"):
    enc = tiktoken.encoding_for_model(model)

    # Define maximum tokens
    max_tokens = 100000

    # Check for presence of columns
    has_bc = "bc_pmid_intersection" in df.columns
    has_ac = "ac_pmid_intersection" in df.columns

    # Calculate number of text fields to divide the tokens among
    num_fields = 1 + has_bc + has_ac

    # Tokens per field
    tokens_per_field = max_tokens // num_fields

    def truncate_text(text_list, max_tokens):
        if isinstance(text_list, list):
            text = " ".join(text_list)
        else:
            text = text_list

        sentences = text.split(". ")
        truncated_text = ""
        current_tokens = 0

        for sentence in sentences:
            sentence_with_period = sentence + ". "
            sentence_tokens = enc.encode(sentence_with_period)
            if current_tokens + len(sentence_tokens) > max_tokens:
                break
            truncated_text += sentence_with_period
            current_tokens += len(sentence_tokens)
        print(f"Truncated text token length: {current_tokens}")
        return truncated_text.strip()

    # Truncate ab_pmid_intersection
    df["ab_pmid_intersection"] = df["ab_pmid_intersection"].apply(
        lambda x: truncate_text(x, tokens_per_field)
    )

    # Truncate bc_pmid_intersection if present
    if has_bc:
        df["bc_pmid_intersection"] = df["bc_pmid_intersection"].apply(
            lambda x: truncate_text(x, tokens_per_field)
        )

    # Truncate ac_pmid_intersection if present
    if has_ac:
        df["ac_pmid_intersection"] = df["ac_pmid_intersection"].apply(
            lambda x: truncate_text(x, tokens_per_field)
        )

    return df


def interleave_and_get_top_n_pmids(text, n):
    if not isinstance(text, str) or text == "[]":
        return ""

    # Split the text by PMID
    pmid_entries = re.split(r"(?=PMID: \d+)", text)
    # Remove any empty entries
    pmid_entries = [entry.strip() for entry in pmid_entries if entry.strip()]

    # Extract PMIDs, skipping entries that don't match the expected format
    pmids = []
    for entry in pmid_entries:
        match = re.search(r"PMID: (\d+)", entry)
        if match:
            pmids.append(int(match.group(1)))

    if not pmids:
        return ""  # Return empty string if no valid PMIDs found

    # Create a sorted copy of PMIDs in descending order
    sorted_pmids = sorted(pmids, reverse=True)

    # Interleave the original and sorted PMIDs
    interleaved = []
    for original, sorted_pmid in zip(pmids, sorted_pmids):
        if original not in interleaved:
            interleaved.append(original)
        if sorted_pmid not in interleaved:
            interleaved.append(sorted_pmid)

    # Add any remaining PMIDs
    interleaved.extend(
        [pmid for pmid in pmids + sorted_pmids if pmid not in interleaved]
    )

    # Take the top n PMIDs
    top_n_pmids = interleaved[:n]

    # Map the top n PMIDs back to their original entries
    pmid_to_entry = {
        int(re.search(r"PMID: (\d+)", entry).group(1)): entry
        for entry in pmid_entries
        if re.search(r"PMID: (\d+)", entry)
    }
    top_n_entries = [
        pmid_to_entry[pmid] for pmid in top_n_pmids if pmid in pmid_to_entry
    ]

    # Join them back together
    return " ".join(top_n_entries)


def filter_top_n_articles(df, config):
    post_n = config.post_n
    if post_n <= 0:
        return df  # Return original dataframe if POST_N is not set or invalid

    columns_to_filter = [
        "ab_pmid_intersection",
        "bc_pmid_intersection",
        "ac_pmid_intersection",
    ]

    for column in columns_to_filter:
        if column in df.columns:
            df[column] = df[column].apply(
                lambda x: interleave_and_get_top_n_pmids(x, post_n)
            )

    return df


def process_dataframe(out_df, config):
    out_df = optimize_text_length(out_df)
    out_df = calculate_relevance_ratios(out_df)
    out_df = filter_top_n_articles(out_df, config)  # Add this line

    return out_df


def main():
    ###################### Argument Parsing ############################

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
    ###################### AB Data Loading & Processsing ############################
    config = Config(args)
    out_df = config.data.copy(deep=True)

    a_term = config.data.a_term.unique().tolist()[0].split("&")[0]
    b_terms = config.data.b_term.unique().tolist()

    ab_pmids = RaggedTensor([eval(lst) for lst in config.data.ab_pmid_intersection])
    ab_hypotheses = RaggedTensor(
        [
            getHypothesis(config.job_config, a_term=a_term, b_term=b_term)
            for b_term in b_terms
        ]
    )

    all_pmids = ab_pmids.flatten()
    all_hypotheses = ab_hypotheses.expand(ab_pmids.shape)

    ###################### BC Data Loading & Processsing ############################
    if config.is_skim_gpt:
        c_term = config.data.c_term.unique().tolist()[0]
        bc_pmids = RaggedTensor([eval(lst) for lst in config.data.bc_pmid_intersection])
        bc_hypotheses = RaggedTensor(
            [
                getHypothesis(config.job_config, c_term=c_term, b_term=b_term)
                for b_term in b_terms
            ]
        )

        all_pmids += bc_pmids.flatten()
        all_hypotheses += bc_hypotheses.expand(bc_pmids.shape)

        if config.has_ac:
            # For each atomic run there should only be one unique ac_pmid intersection
            ac_pmids = RaggedTensor(eval(config.data.ac_pmid_intersection[0]))
            ac_hypothesis = RaggedTensor(
                [getHypothesis(config.job_config, a_term=a_term, c_term=c_term)]
            )

            all_pmids += ac_pmids
            all_hypotheses += ac_hypothesis.expand([ac_pmids.shape])

    abstract_map = getAbstractMap(config.job_config, all_pmids)
    abstracts = all_pmids.map(lambda pmid: abstract_map.get(str(pmid), ""))

    ##################### Model Loading & Generation ############################
    model = vllm.LLM(model=config.filter_config["MODEL"], max_model_len=4000)

    sampling_config = vllm.SamplingParams(
        temperature=config.filter_config["TEMPERATURE"],
        top_k=config.filter_config["TOP_K"],
        top_p=config.filter_config["TOP_P"],
        max_tokens=config.filter_config["MAX_COT_TOKENS"] if config.debug else 1,
    )

    ##################### LLM Inference ############################
    prompts = getPrompts(abstracts, all_hypotheses)
    answers = gen(prompts, model, sampling_config)

    ##################### Post process answers ############################
    # Adding defaults for unraveling. In the case where there's no AC or BC, they will be filled with empty RaggedTensors
    defaults = 3 * [RaggedTensor([])]

    ab_outputs, bc_outputs, ac_outputs, *_ = chain(answers.split(), defaults)
    ab_abstracts, bc_abstracts, ac_abstracts, *_ = chain(abstracts.split(), defaults)

    postProcess(
        config,
        ab_outputs,
        ab_abstracts,
        ab_hypotheses,
        out_df,
        terms="ab",
        shape=ab_pmids.shape,
    )

    ##################### Post process BC answers ############################
    if config.is_skim_gpt:
        postProcess(
            config,
            bc_outputs,
            bc_abstracts,
            bc_hypotheses,
            out_df,
            terms="bc",
            shape=bc_pmids.shape,
        )
        if config.has_ac:
            postProcess(
                config,
                ac_outputs,
                ac_abstracts,
                ac_hypothesis,
                out_df,
                terms="ac",
                shape=[ac_pmids.shape],
            )

    out_df = process_dataframe(out_df, config)

    if config.test_leakage:
        leakage_data = load_data("leakage.csv")
        out_df = update_ab_pmid_intersection(
            out_df, leakage_data, config.test_leakage_type
        )
    out_df.to_csv(
        f"{config.debug_tsv_name if config.debug else config.filtered_tsv_name}",
        sep="\t",
    )

    ##################### Classify ############################
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
            print(f"Processed row {index + 1} ({row['b_term']}) of {len(out_df)}")
            if config.is_skim_gpt:
                output_json = f"{row['a_term']}_{row['b_term']}_{row['c_term']}_skim_with_gpt.json"
            else:
                output_json = f"{row['a_term']}_{row['b_term']}_km_with_gpt.json"
            write_to_json([result_dict], output_json)  # Wrap result_dict in a list
            print(f"Analysis results have been saved to {output_json}")


if __name__ == "__main__":
    main()
