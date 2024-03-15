import pandas as pd
import pandas as pd
from transformers import set_seed
import json
from Bio import Entrez
import vllm
from lmformatenforcer import RegexParser
from lmformatenforcer.integrations.vllm import build_vllm_logits_processor, build_vllm_token_enforcer_tokenizer_data
import argparse
from abstract_comprehension import read_tsv_to_dataframe
from tqdm import tqdm
import numpy as np

def getHypothesis(config, b_term: str, a_term: str) -> str:
    job_type = config.get("JOB_TYPE", "").lower()
    if job_type == "km_with_gpt":
        hypothesis_template = config.get("KM_hypothesis", "")
    elif job_type == "position_km_with_gpt":
        hypothesis_template = config.get("POSITION_KM_hypothesis", "")
    elif job_type == "skim_with_gpt":
        hypothesis_template = config.get("SKIM_hypothesis", "")
    else:
        return "No valid hypothesis for the provided JOB_TYPE."
    
    return hypothesis_template.format(a_term=a_term, b_term=b_term)

def cot_prompt(sys_prompt: str, hyp: str, abstract: str) -> str:
  return f"""
    <|im_start|>system
    {sys_prompt}
    <|im_end|>
    <|im_start|>user
    Hypothesis: {hyp}
    Abstract: {abstract}
    
    Determine whether or not this abstract is relevant for scientifically evaluating the provided hypothesis. A relevant abstract must directly comment on the hypothesis and either support the given hypothesis or have evidence to refute the hypothesis.

    Analyze the abstract above, and throughly describe your thought process for evaluating the hypothesis. Pay attention to particular details in the abstract as it relates to the hypothesis. Let's work this out in a step by step way to be sure we have the right answer.
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

    Analyze the abstract above, and throughly describe your thought process for evaluating the hypothesis. Pay attention to particular details in the abstract as it relates to the hypothesis. Let's work this out in a step by step way to be sure we have the right answer.
    {chain_of_thought}
    
    Classify the given abstract as either 0 (Not relevant) or 1 (Relevant) based on your reasoning above and any information in the abstract and hypothesis.
    Answer: 
    <|im_end|>
    <|im_start|>assistant
    """

def gen(prompts: list[str], model: any, sampling_config: vllm.SamplingParams) -> list[str]:
	generated = model.generate(prompts, sampling_params = sampling_config)
	outputs = [output.outputs[0].text for output in generated]
	return outputs

# Redefined reshape function to work with ragged string arrays
def reshape(inp: list, shape: list) -> list:
    assert(len(inp) == sum(shape))
    output = []
    running_length = 0;
    for length in shape:
        output.append(inp[running_length: running_length + length])
        running_length += length
        
    return output

def expand(inputs: list, shape_list: list) -> list:
    assert(len(inputs) == len(shape_list))
    expanded = []
    for idx, inp in enumerate(inputs):
        expanded.extend([inp] * shape_list[idx])
    return expanded

def getCoTPrompts(abstracts: list[str], sys_prompt: str, hypotheses: list[str]) -> list[str]:
	return [cot_prompt(sys_prompt, hypotheses[i], abstract) for i, abstract in enumerate(abstracts)]

def getAnswerPrompts(abstracts: list[str], sys_prompt: str, hypotheses: list[str], cot_outputs: list[str]) -> list[str]:
	return [answer_prompt(sys_prompt, hypotheses[i], abstract, cot_outputs[i]) for i, abstract in enumerate(abstracts)]

# Returns a dictionary for each PMID & Abstract Pair
# This method is needed since Entrez automatically removes duplicates in the pmid list
def getAbstractMap(config: json, pmids: list[str]) -> dict:
    returned_pmids = []
    returned_abstracts = []
    global_config = config["GLOBAL_SETTINGS"]
    pmid_config = global_config["PUBMED_PARAMS"]
    
    Entrez.email = 'leoxu27@gmail.com'
    Entrez.api_key = pmid_config["api_key"]
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

	parser.add_argument('--km_output', type = argparse.FileType('r'), required = True,
						help='Path to the TSV file holding a km run output.')
	parser.add_argument('--config', type = argparse.FileType('r'),
						help='Config file for kmGPT run.', required = True)
	parser.add_argument('--filtered_tsv_name', type = str, required = True, help='Name of TSV file holding filtered output.')
	parser.add_argument('--cot_tsv_name', type = str, required = True, help='Name of TSV file hold CoT outputs for testing.')
	args = parser.parse_args()

	###################### Data Loading & Processsing ############################ 
	km_output = read_tsv_to_dataframe(args.km_output)
	config = json.load(args.config)
	filtered_tsv_name = args.filtered_tsv_name
	cot_tsv_name = args.cot_tsv_name
 
	ab_intersection = []
	shape = []
	for intersection in km_output.ab_pmid_intersection:
		ab_intersection.extend(eval(intersection))
		shape.append(len(eval(intersection)))
  
	abstract_map = getAbstractMap(config, ab_intersection)
	abstracts = [f"PMID {PMID}: {abstract_map.get(str(pmid), '')}" for pmid in ab_intersection]

	# There should only be one a_term, so it's safe to grab the first index
	a_term = km_output.a_term.unique().tolist()[0].split("&")[0]
	b_terms = km_output.b_term.unique().tolist()

	filter_config = config["abstract_filter"]
	sys_prompt = filter_config['SYS_PROMPT']
	hypotheses = [getHypothesis(config, a_term, b_term) for b_term in b_terms]  # Done to make creating prompts easier
	expanded_hypotheses = expand(hypotheses, shape)

	##################### Model Loading & Generation ############################ 
	mistral = vllm.LLM(model=filter_config["MODEL"], max_model_len=16832)
	tokenizer_data = build_vllm_token_enforcer_tokenizer_data(mistral)
	logits_processor = build_vllm_logits_processor(tokenizer_data, RegexParser(r"0|1"))
 
	##################### Generate Chains of Thought ############################
	cot_prompts = getCoTPrompts(abstracts, sys_prompt, expanded_hypotheses)
	sampling_cot = vllm.SamplingParams(
				temperature=filter_config["TEMPERATURE"], 
				top_k = filter_config["TOP_K"], top_p=filter_config["TOP_P"], 
				repetition_penalty=filter_config["REPETITION_PENALTY"],
    			max_tokens = 1024)
	cot_outputs = gen(cot_prompts, mistral, sampling_cot)

	##################### Generate Scores from Chain of Thought ############################ 
	answer_prompts = getAnswerPrompts(abstracts, sys_prompt, expanded_hypotheses, cot_outputs)
	sampling_answer = vllm.SamplingParams(
				temperature=filter_config["TEMPERATURE"], 
				top_k = filter_config["TOP_K"], top_p=filter_config["TOP_P"], 
				max_tokens=1,
				repetition_penalty=filter_config["REPETITION_PENALTY"],
				logits_processors=[logits_processor])
	answers = gen(answer_prompts, mistral, sampling_answer)

	##################### Post process answers ############################ 
	cot_tsv = km_output.copy(deep = True)
	filtered_tsv = km_output.copy(deep = True)
 
	answers = reshape([eval(answer) for answer in answers], shape)
	cot_outputs = reshape(cot_outputs, shape)
	abstracts = reshape(abstracts, shape)

	cot_tsv["scores"] = answers
	cot_tsv["chain_of_thought"] = cot_outputs
	cot_tsv["hypothesis"] = hypotheses
	cot_tsv.to_csv(f"{cot_tsv_name}", sep='\t')

	# Filter out the abstracts according to the scores
	filtered_abstracts = []
	for i, abstract_list in tqdm(enumerate(abstracts), desc = "Post-processing abstracts..."):
		mask = np.array(answers[i]) == 1
		filtered = list(np.array(abstract_list)[mask])
		filtered_abstracts.append(filtered)

	filtered_tsv["ab_pmid_intersection"] = filtered_abstracts
	filtered_tsv.to_csv(f"{filtered_tsv_name}", sep="\t")
	return
    
if __name__ == '__main__':
    main()
