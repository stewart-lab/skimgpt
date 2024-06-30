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
from classifier import process_single_row, write_to_json, test_openai_connection
from itertools import chain

# Returns either AB or BC hypotheses depending on the input. If A, B is passed in, getHypothesis will retrieve the AB hypothesis.
# Only two arguements should be specified at once.


def getHypothesis(config, a_term: str = None, b_term: str = None, c_term: str = None) -> str:
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
        assert (a_term and b_term and not c_term) or (
            b_term and c_term and not a_term) or (a_term and c_term and not b_term)

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


def gen(prompts: RaggedTensor, model: any, sampling_config: vllm.SamplingParams) -> RaggedTensor:
    generated = model.generate(prompts.data, sampling_params=sampling_config)
    outputs = RaggedTensor(
        [output.outputs[0].text for output in generated], prompts.break_point)
    return outputs


def getPrompts(abstracts: RaggedTensor, hypotheses: RaggedTensor) -> RaggedTensor:
    assert not abstracts.is2D(), "abstracts should be flattened."
    assert not hypotheses.is2D(), "hypotheses should be flattened."
    return RaggedTensor([prompt(abstracts[i], hypotheses[i]) for i in range(abstracts.shape)], hypotheses.break_point)

# Returns a dictionary for each PMID & Abstract Pair
# This method is needed since Entrez automatically removes duplicates in the pmid list


def getAbstractMap(config: json, pmids: list[str]) -> dict:
    returned_pmids = []
    returned_abstracts = []
    global_config = config["GLOBAL_SETTINGS"]
    pmid_config = global_config["PUBMED_PARAMS"]

    Entrez.email = 'leoxu27@gmail.com'
    Entrez.api_key = "8bfe67116f93cedbee9e4f31a1e65b7e1d09"
    Entrez.max_tries = global_config["MAX_RETRIES"]
    Entrez.sleep_between_tries = global_config["RETRY_DELAY"]
    efetch = Entrez.efetch(
        db=pmid_config["db"], id=pmids, rettype=pmid_config["rettype"])

    output = Entrez.read(efetch)
    efetch.close()

    for paper in output["PubmedArticle"]:
        pmid = paper["MedlineCitation"]["PMID"]
        returned_pmids.append(str(pmid))
        try:
            abstract_text = " ".join(
                paper["MedlineCitation"]["Article"]["Abstract"]["AbstractText"])
        except:
            print(f"No abstract found for PMID {pmid}. Skipping...")
            continue
        returned_abstracts.append(f"PMID {pmid}: {abstract_text}")
    return dict(zip(returned_pmids, returned_abstracts))


# Packages all the inputted data into the provided dataframes
def postProcess(config: Config, outputs: RaggedTensor, abstracts: RaggedTensor, hypotheses: RaggedTensor, out_df: pd.DataFrame, terms: str, shape: list):
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


def main():
    ###################### Argument Parsing ############################
    parser = argparse.ArgumentParser(description='Porpoise Inference')

    parser.add_argument('--km_output', type=str, required=True,
                        help='Tsv file to run relevance filtering on.')
    parser.add_argument('--config', type=str, required=True,
                        help='Config file for kmGPT run.')
    args = parser.parse_args()
    ###################### AB Data Loading & Processsing ############################
    config = Config(args)
    out_df = config.data.copy(deep=True)

    a_term = config.data.a_term.unique().tolist()[0].split("&")[0]
    b_terms = config.data.b_term.unique().tolist()

    ab_pmids = RaggedTensor([eval(lst)
                            for lst in config.data.ab_pmid_intersection])
    ab_hypotheses = RaggedTensor([getHypothesis(
        config.job_config, a_term=a_term, b_term=b_term) for b_term in b_terms])

    all_pmids = ab_pmids.flatten()
    all_hypotheses = ab_hypotheses.expand(ab_pmids.shape)

    ###################### BC Data Loading & Processsing ############################
    if config.is_skim_gpt:
        c_term = config.data.c_term.unique().tolist()[0]
        bc_pmids = RaggedTensor([eval(lst)
                                for lst in config.data.bc_pmid_intersection])
        bc_hypotheses = RaggedTensor([getHypothesis(
            config.job_config, c_term=c_term, b_term=b_term) for b_term in b_terms])

        all_pmids += bc_pmids.flatten()
        all_hypotheses += bc_hypotheses.expand(bc_pmids.shape)

        if config.has_ac:
            # For each atomic run there should only be one unique ac_pmid intersection
            ac_pmids = RaggedTensor(eval(config.data.ac_pmid_intersection[0]))
            ac_hypothesis = RaggedTensor(
                [getHypothesis(config.job_config, a_term=a_term, c_term=c_term)])

            all_pmids += ac_pmids
            all_hypotheses += ac_hypothesis.expand([ac_pmids.shape])

    abstract_map = getAbstractMap(config.job_config, all_pmids)
    abstracts = all_pmids.map(lambda pmid: abstract_map.get(str(pmid), ""))

    ##################### Model Loading & Generation ############################
    model = vllm.LLM(model=config.filter_config["MODEL"], max_model_len=4000)

    sampling_config = vllm.SamplingParams(
        temperature=config.filter_config["TEMPERATURE"],
        top_k=config.filter_config["TOP_K"], top_p=config.filter_config["TOP_P"],
        max_tokens=config.filter_config["MAX_COT_TOKENS"] if config.debug else 1)

    ##################### LLM Inference ############################
    prompts = getPrompts(abstracts, all_hypotheses)
    answers = gen(prompts, model, sampling_config)

    ##################### Post process answers ############################
    # Adding defaults for unraveling. In the case where there's no AC or BC, they will be filled with empty RaggedTensors
    defaults = 3 * [RaggedTensor([])]

    ab_outputs, bc_outputs, ac_outputs, *_ = chain(answers.split(), defaults)
    ab_abstracts, bc_abstracts, ac_abstracts, * \
        _ = chain(abstracts.split(), defaults)

    postProcess(config, ab_outputs, ab_abstracts, ab_hypotheses,
                out_df, terms="ab", shape=ab_pmids.shape)

    ##################### Post process BC answers ############################
    if config.is_skim_gpt:
        postProcess(config, bc_outputs, bc_abstracts, bc_hypotheses,
                    out_df, terms="bc", shape=bc_pmids.shape)
        if config.has_ac:
            postProcess(config, ac_outputs, ac_abstracts, ac_hypothesis,
                        out_df, terms="ac", shape=[ac_pmids.shape])

    out_df.to_csv(
        f"{config.debug_tsv_name if config.debug else config.filtered_tsv_name}", sep="\t")

    ###################### Open AI Call #####################################
    results = {}

    # Test OpenAI connection if necessary
    test_openai_connection(config.job_config)  # Ensure this function exists in the classifier module

    # Process each row in the DataFrame
    for index, row in out_df.iterrows():
        term = row["b_term"]
        result_dict = process_single_row(row, config.job_config)
        if result_dict:  # Ensure that result_dict is not None
            if term not in results:
                results[term] = [result_dict]
            else:
                results[term].append(result_dict)
            print(f"Processed row {index + 1} ({row['b_term']}) of {len(out_df)}")
        else:
            print(f"Skipping row {index + 1} ({row['b_term']}) due to no results.")

    # Check if results were processed
    if not results:
        print("No results were processed.")
    else:
        # Save the results to a JSON file
        output_json_path = os.path.join(config.km_output_dir, config.job_config["OUTPUT_JSON"])
        write_to_json(results, output_json_path, config.km_output_dir)
        print(f"Analysis results have been saved to {config.job_config['OUTPUT_JSON']}")


if __name__ == '__main__':
    main()
