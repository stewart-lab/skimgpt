from __future__ import annotations
import pandas as pd
from transformers import set_seed
import json
from Bio import Entrez
import vllm
from lmformatenforcer import RegexParser
from lmformatenforcer.integrations.vllm import build_vllm_logits_processor, build_vllm_token_enforcer_tokenizer_data
import argparse
from abstract_comprehension import read_tsv_to_dataframe
from utils import Config, RaggedTensor
from tqdm import tqdm
import numpy as np
import os
from classifier import process_single_row, write_to_json, test_openai_connection
        
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
        assert (a_term and b_term and not c_term) or (b_term and c_term and not a_term)
        
        if a_term and b_term and not c_term:
            hypothesis_template = config.get("SKIM_hypotheses", "").get("AB")
            return hypothesis_template.format(a_term=a_term, b_term=b_term)
        
        elif b_term and c_term and not a_term:
            hypothesis_template = config.get("SKIM_hypotheses", "").get("BC")
            return hypothesis_template.format(b_term=b_term, c_term = c_term)

    else:
        return "No valid hypothesis for the provided JOB_TYPE."
    
    

def cot_prompt(sys_prompt: str, hyp: str, abstract: str) -> str:
    return f"""
    <|im_start|>system
    {sys_prompt}
    <|im_end|>
    <|im_start|>user
    Hypothesis: {hyp}
    Abstract: {abstract}

    Determine whether or not this abstract is relevant for scientifically evaluating the provided hypothesis. A relevant abstract must directly comment on the hypothesis and either support the given hypothesis or have evidence to refute the hypothesis.

    Analyze the abstract above, and throughly describe your thought process for evaluating the hypothesis. Pay attention to particular details in the abstract as it relates to the hypothesis. Make sure to stay focused on what the hypothesis is specifically saying. Let's work this out in a step by step way to be sure we have the right answer.
    <|im_end|>
    <|im_start|>assistant
    """

def answer_prompt(sys_prompt: str, hypothesis: str, abstract: str, chain_of_thought: str) -> str:
    return f"""
    <|im_start|>system
    {sys_prompt}
    <|im_end|>
    <|im_start|>user
    Hypothesis: {hypothesis}
    Abstract: {abstract}

    Determine whether or not this abstract is relevant for scientifically evaluating the provided hypothesis. A relevant abstract must directly comment on the hypothesis and either support the given hypothesis or have evidence to refute the hypothesis.

    Analyze the abstract above, and throughly describe your thought process for evaluating the hypothesis. Pay attention to particular details in the abstract as it relates to the hypothesis. Make sure to stay focused on what the hypothesis is specifically saying. Let's work this out in a step by step way to be sure we have the right answer.
    {chain_of_thought}

    Classify the given abstract as either 0 (Not relevant for scientifically assessing the hypothesis) or 1 (Relevant for scientifically assessing the hypothesis) based on the reasoning above and other useful pieces of information in the abstract and hypothesis.
    Answer: 
    <|im_end|>
    <|im_start|>assistant
    """
    
def gen(prompts: RaggedTensor, model: any, sampling_config: vllm.SamplingParams) -> RaggedTensor:
	generated = model.generate(prompts.data, sampling_params = sampling_config)
	outputs = RaggedTensor([output.outputs[0].text for output in generated], prompts.break_point)
	return outputs

def getCoTPrompts(abstracts: RaggedTensor, sys_prompt: str, hypotheses: RaggedTensor) -> RaggedTensor:
    assert not abstracts.is2D(), "abstracts should be flattened."
    assert not hypotheses.is2D(), "hypotheses should be flattened."
    return RaggedTensor([cot_prompt(sys_prompt, hypotheses[i], abstracts[i]) for i in range(abstracts.shape)], hypotheses.break_point)

def getAnswerPrompts(abstracts: RaggedTensor, sys_prompt: str, hypotheses: RaggedTensor, cot_outputs: RaggedTensor) -> RaggedTensor:
    assert not abstracts.is2D(), "abstracts should be flattened."
    assert not hypotheses.is2D(), "hypotheses should be flattened."
    assert not cot_outputs.is2D(), "cot outputs should be flattened"
    return RaggedTensor([answer_prompt(sys_prompt, hypotheses[i], abstracts[i], cot_outputs[i]) for i in range(abstracts.shape)], hypotheses.break_point)
    
# Returns a dictionary for each PMID & Abstract Pair
# This method is needed since Entrez automatically removes duplicates in the pmid list
def getAbstractMap(config: json, pmids: list[str]) -> dict:
    returned_pmids = []
    returned_abstracts = []
    global_config = config["GLOBAL_SETTINGS"]
    pmid_config = global_config["PUBMED_PARAMS"]
    
    Entrez.email = 'leoxu27@gmail.com'
    Entrez.api_key = config["PUBMED_API_KEY"]
    Entrez.max_tries = global_config["MAX_RETRIES"]
    Entrez.sleep_between_tries = global_config["RETRY_DELAY"]
    efetch = Entrez.efetch(db=pmid_config["db"], id=pmids, rettype=pmid_config["rettype"])
    
    output = Entrez.read(efetch)
    efetch.close()
    
    for paper in output["PubmedArticle"]:
        returned_pmids.append(str(paper["MedlineCitation"]["PMID"]))
        abstract_text = " ".join(paper["MedlineCitation"]["Article"]["Abstract"]["AbstractText"])
        returned_abstracts.append(abstract_text)
    return dict(zip(returned_pmids, returned_abstracts))

def main():
    ###################### Argument Parsing ############################ 
    parser = argparse.ArgumentParser(description='Mistral7B Inference')

    parser.add_argument('--km_output', type=str, required=True, help='Path to the TSV file holding a km run output.')
    parser.add_argument('--config', type=str, required=True, help='Config file for kmGPT run.')
    args = parser.parse_args()
    ###################### AB Data Loading & Processsing ############################ 
    config = Config(args)
    cot_df = config.data.copy(deep = True)
    filtered_df = config.data.copy(deep = True)

    a_term = config.data.a_term.unique().tolist()[0].split("&")[0]
    b_terms = config.data.b_term.unique().tolist()
    
    ab_pmids = RaggedTensor([eval(lst) for lst in config.data.ab_pmid_intersection])
    ab_hypotheses = RaggedTensor([getHypothesis(config.job_config, a_term = a_term, b_term = b_term) for b_term in b_terms])

    all_pmids = ab_pmids.flatten()
    all_hypotheses = ab_hypotheses.expand(ab_pmids.shape)

    
    ###################### BC Data Loading & Processsing ############################ 
    if config.is_skim_gpt:
        c_term = config.data.c_term.unique().tolist()[0]
        bc_pmids = RaggedTensor([eval(lst) for lst in config.data.bc_pmid_intersection])
        bc_hypotheses = RaggedTensor([getHypothesis(config.job_config, c_term = c_term, b_term = b_term) for b_term in b_terms])
    
        all_pmids += bc_pmids.flatten()
        all_hypotheses += bc_hypotheses.expand(bc_pmids.shape)
        
    abstract_map = getAbstractMap(config.job_config, all_pmids)
    abstracts = all_pmids.map(lambda pmid: abstract_map.get(str(pmid), ""))

    ##################### Model Loading & Generation ############################ 
    mistral = vllm.LLM(model=config.filter_config["MODEL"], max_model_len=16832)
    tokenizer_data = build_vllm_token_enforcer_tokenizer_data(mistral)
    logits_processor = build_vllm_logits_processor(tokenizer_data, RegexParser(r"0|1"))
    
    sampling_cot = vllm.SamplingParams(
            temperature = config.filter_config["TEMPERATURE"], 
            top_k = config.filter_config["TOP_K"], top_p = config.filter_config["TOP_P"], 
            repetition_penalty = config.filter_config["REPETITION_PENALTY"],
            max_tokens = 1024)
    
    sampling_answer = vllm.SamplingParams(
            temperature=config.filter_config["TEMPERATURE"], 
            top_k = config.filter_config["TOP_K"], top_p = config.filter_config["TOP_P"], 
            max_tokens = 1,
            repetition_penalty = config.filter_config["REPETITION_PENALTY"],
            logits_processors = [logits_processor])

    ##################### LLM Inference ############################
    cot_prompts = getCoTPrompts(abstracts, config.sys_prompt, all_hypotheses)
    cot_outputs = gen(cot_prompts, mistral, sampling_cot)
    
    answer_prompts = getAnswerPrompts(abstracts, config.sys_prompt, all_hypotheses, cot_outputs)
    answers = gen(answer_prompts, mistral, sampling_answer)
    answers = answers.map(lambda x: eval(x))

    ##################### Post process AB answers ############################ 
    ab_answers, bc_answers = answers.split()
    ab_abstracts, bc_abstracts = abstracts.split()
    ab_cot, bc_cot = cot_outputs.split()
    
    ab_answers.reshape(ab_pmids.shape)
    ab_abstracts.reshape(ab_pmids.shape)
    ab_abstracts.applyFilter(ab_answers)
    ab_cot.reshape(ab_pmids.shape)
    
    filtered_df["ab_pmid_intersection"] = ab_abstracts.data
    cot_df["ab_scores"] = ab_answers.data
    cot_df["ab_cot"] = ab_cot.data
    cot_df["ab_hypothesis"] = ab_hypotheses.data

    ##################### Post process BC answers ############################ 
    if config.is_skim_gpt:
        bc_answers.reshape(bc_pmids.shape)
        bc_abstracts.reshape(bc_pmids.shape)
        bc_abstracts.applyFilter(bc_answers)
        bc_cot.reshape(bc_pmids.shape)
    
        filtered_df["bc_pmid_intersection"] = bc_abstracts.data
        cot_df["bc_scores"] = bc_answers.data
        cot_df["bc_cot"] = bc_cot.data
        cot_df["bc_hypothesis"] = bc_hypotheses.data
    
    filtered_df.to_csv(f"{config.filtered_tsv_name}", sep="\t")
    cot_df.to_csv(f"{config.cot_tsv_name}", sep="\t")
    
    ###################### Open AI Call #####################################
    results = {}

    # Test OpenAI connection if necessary
    test_openai_connection(config.job_config)  # Ensure this function exists in the classifier module

    # Process each row in the DataFrame
    for index, row in filtered_df.iterrows():
        term = row["b_term"]
        result_dict = process_single_row(row, config.job_config)
        if result_dict:  # Ensure that result_dict is not None
            if term not in results:
                results[term] = [result_dict]
            else:
                results[term].append(result_dict)
            print(f"Processed row {index + 1} ({row['b_term']}) of {len(filtered_df)}")
        else:
            print(f"Skipping row {index + 1} ({row['b_term']}) due to no results.")

    # Check if results were processed
    if not results:
        print("No results were processed.")
    else:
        # Save the results to a JSON file
        output_json_path = os.path.join(config.km_output_dir, config.job_config["OUTPUT_JSON"])
        write_to_json(results, output_json_path, config.km_output_dir)
        print(f"Analysis results have been saved to {config.km_output_dir}")
    return

if __name__ == '__main__':
    main()
