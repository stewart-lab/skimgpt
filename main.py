import pandas as pd
from transformers import set_seed
from get_pubmed_text import process_abstracts_data
import json
import vllm
from lmformatenforcer import RegexParser
from lmformatenforcer.integrations.vllm import build_vllm_logits_processor, build_vllm_token_enforcer_tokenizer_data
import argparse
from get_pubmed_text import process_abstracts_data
from abstract_comprehension import read_tsv_to_dataframe
from tqdm import tqdm

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

def gen(batches: list[str], model: any, sampling_config: vllm.SamplingParams) -> list[str]:
    outputs = []
    for batch in batches:
        generated = model.generate(batch, sampling_params = sampling_config)
        outputs.extend([output.outputs[0].text for output in generated])
    return outputs

def get_batch(inp: list, batch_size: int) -> list:
    return [inp[i * batch_size:(i + 1) * batch_size] for i in range((len(inp) + batch_size - 1) // batch_size )]

# Redefined reshape function to work with ragged string arrays
def reshape(inp: list, shape: list) -> list:
    assert(len(inp) == sum(shape))
    output = []
    running_length = 0;
    for length in shape:
        output.append(inp[running_length: running_length + length])
        running_length = length
        
    return output

def getCoTPrompts(abstracts: list[str], sys_prompt: str, hypotheses: list[str]) -> list[str]:
	return [cot_prompt(sys_prompt, hypotheses[i], abstract) for i, abstract_list in enumerate(abstracts) for abstract in abstract_list]

def getAnswerPrompts(abstracts: list[str], sys_prompt: str, hypotheses: list[str], cot_outputs: list[str]) -> list[str]:
	answer_prompts = []
	total_idx = 0
	for i, abstract_list in enumerate(abstracts):
		for j, abstract in enumerate(abstract_list):
			answer_prompts.append(answer_prompt(sys_prompt, hypotheses[i], abstract, cot_outputs[total_idx + j]))
		total_idx += len(abstract_list)
	return answer_prompts
  
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

	b_terms_pmids = km_output.ab_pmid_intersection.map(lambda pmid_list: pmid_list.strip('][').split(', '))
	# Grab only the abstract from each list of pmids in the TSV
	abstracts = [process_abstracts_data(config, pmid_list)[0] for pmid_list in tqdm(b_terms_pmids, desc = "Fetching abstracts from PubMed")] # Fetch abstracts from each b_term's PMID list

	# There should only be one a_term, so it's safe to grab the first index
	a_term = km_output.a_term.unique().tolist()[0].split("&")[0]
	b_terms = km_output.b_term.unique().tolist()

	filter_config = config["abstract_filter"]
	sys_prompt = filter_config['SYS_PROMPT']
	hypotheses = [getHypothesis(config, a_term, b_term) for b_term in b_terms]

	##################### Model Loading & Generation ############################ 
	mistral = vllm.LLM(model=filter_config["MODEL"], max_model_len=16832)
	tokenizer_data = build_vllm_token_enforcer_tokenizer_data(mistral)
	logits_processor = build_vllm_logits_processor(tokenizer_data, RegexParser(r"0|1"))
 
	##################### Generate Chains of Thought ############################
	cot_prompts = getCoTPrompts(abstracts, sys_prompt, hypotheses)
	cot_batches = get_batch(cot_prompts, filter_config["BATCH_SIZE"])
	sampling_cot = vllm.SamplingParams(
				temperature=filter_config["TEMPERATURE"], 
				top_k = filter_config["TOP_K"], top_p=filter_config["TOP_P"], 
				repetition_penalty=filter_config["REPETITION_PENALTY"],
    			max_tokens = 1024)
	cot_outputs = gen(cot_batches, mistral, sampling_cot)

	##################### Generate Scores from Chain of Thought ############################ 
	answer_prompts = getAnswerPrompts(abstracts, sys_prompt, hypotheses, cot_outputs)
	answer_batches = get_batch(answer_prompts, filter_config["BATCH_SIZE"])
	sampling_answer = vllm.SamplingParams(
				temperature=filter_config["TEMPERATURE"], 
				top_k = filter_config["TOP_K"], top_p=filter_config["TOP_P"], 
				max_tokens=1,
				repetition_penalty=filter_config["REPETITION_PENALTY"],
				logits_processors=[logits_processor])
	answers = gen(answer_batches, mistral, sampling_answer)

	##################### Post process answers ############################ 
	cot_tsv = km_output.copy(deep = True)
	filtered_tsv = km_output.copy(deep = True)
  
	answers = [eval(answer) for answer in answers] # Turn answers into list of ints
	shape = [len(abstract_list) for abstract_list in abstracts] # Get the shape of the abstracts
	answers = reshape(answers, shape)

	cot_tsv["scores"] = answers
	cot_tsv["chain_of_thought"] = reshape(cot_outputs, shape)
	cot_tsv["hypothesis"] = hypotheses
	cot_tsv.to_csv(f"{cot_tsv_name}", sep='\t')

	# Filter out the abstracts according to the scores
	filtered_abstracts = []
	for i, abstract_list in enumerate(abstracts):
		filtered = []
		for j, score in enumerate(answers[i]):
			if score == 1:
				filtered.append(abstract_list[j])
		filtered_abstracts.append(filtered)


	filtered_tsv["ab_pmid_intersection"] = filtered_abstracts
	filtered_tsv.to_csv(f"{filtered_tsv_name}", sep="\t")
	return
    
if __name__ == '__main__':
    main()
