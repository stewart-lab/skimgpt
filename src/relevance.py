from __future__ import annotations
import pandas as pd
from transformers import set_seed
import json
from Bio import Entrez
import vllm
import argparse
from abstract_comprehension import read_tsv_to_dataframe
from utils import Config, RaggedTensor
from tqdm import tqdm
import numpy as np
import os
import jinja2
import tiktoken
from classifier import process_single_row, write_to_json
from itertools import chain

# Returns either AB or BC hypotheses depending on the input. If A, B is passed in, getHypothesis will retrieve the AB hypothesis.
# Only two arguements should be specified at once


def getHypothesis(
    config, a_term: str = None, b_term: str = None, c_term: str = None
) -> str:
    job_type = config.get("JOB_TYPE", "").lower()

    if job_type == "km_with_gpt":
        assert a_term and b_term and not c_term
        hypothesis_template = config.get("KM_hypothesis", "")

        return hypothesis_template.format(a_term=a_term, b_term=b_term)

    elif job_type == "position_km_with_gpt":
        assert a_term and b_term and not c_term

        hypothesis_template = config.get("POSITION_KM_hypothesis", "")
        return hypothesis_template.format(a_term=a_term, b_term=b_term), None

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
    returned_abstracts = []
    Entrez.email = "leoxu27@gmail.com"
    Entrez.api_key = config["PUBMED_API_KEY"]
    Entrez.max_tries = config["GLOBAL_SETTINGS"]["MAX_RETRIES"]
    Entrez.sleep_between_tries = config["GLOBAL_SETTINGS"]["RETRY_DELAY"]

    # Hard coding PubMed parameters directly
    efetch = Entrez.efetch(db="pubmed", id=pmids, retmode="xml", rettype="abstract")

    output = Entrez.read(efetch)
    efetch.close()
    for paper in output["PubmedArticle"]:
        pmid = paper["MedlineCitation"]["PMID"]
        returned_pmids.append(str(pmid))

        abstract_text = " ".join(
            paper["MedlineCitation"]["Article"]
            .get("Abstract", {})
            .get("AbstractText", ["No abstract available"])
        )
        returned_abstracts.append(abstract_text)

    return dict(
        zip(
            returned_pmids,
            [
                f"PMID: {pmid} {abstract}"
                for pmid, abstract in zip(returned_pmids, returned_abstracts)
            ],
        )
    )


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

    out_df = optimize_text_length(out_df)
    out_df.to_csv(

        f"{config.debug_tsv_name if config.debug else config.filtered_tsv_name}",
        sep="\t",
    )

    ##################### Classify ############################
    for index, row in out_df.iterrows():
        results_list = []  # Initialize the results_list for each row
        result_dict = process_single_row(row, config)
        results_list.append(result_dict)
        print(f"Processed row {index + 1} ({row['b_term']}) of {len(out_df)}")
        assert results_list, "No results were processed"
        # Generate a unique output JSON file name for each a_term and c_term combination
        output_json = (
            f"{row['a_term']}_{row['b_term']}_{row['c_term']}_skim_with_gpt.json"
        )
        write_to_json(results_list, output_json)
        print(f"Analysis results have been saved to {output_json}")


if __name__ == "__main__":
    main()
