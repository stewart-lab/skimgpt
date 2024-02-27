import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from typing import Callable
import argparse
from src.get_pubmed_text import process_abstracts_data
from src.abstract_comprehension import read_tsv_to_dataframe, process_single_row
import json
from guidance import models, gen, select, system, assistant, user


def retrieveZeroShotCoTPrompt(hyp: str, abstract: str) -> str:
  zero_shot_prompt = f"""
    Hypothesis: {hyp}
    Abstract: {abstract}

    Determine whether or not this abstract is relevant for scientifically evaluating the provided hypothesis.
    A relevant abstract should either support the given hypothesis or have evidence to refute the hypothesis.
    A relevant abstract must directly comment on the hypothesis.

    Let us think through this step by step.
  """
  return zero_shot_prompt


def generateOutput(model: any, config: any, hyp: str, abstract: str) -> json:
  sys_prompt = "You are an incredibly brilliant biomedical researcher who has spent their lifetime reading all the papers in PubMed. You are focused on assisting other researchers in evaluating suggested hypotheses given abstracts in PubMed."
  cot_prompt = retrieveZeroShotCoTPrompt(hyp, abstract)
  answer_prompt = "Give your answer as either 0 (Not Relevant) or 1 (Relevant) for the above abstract. Answer: "
  
  with system():
    lm = model + sys_prompt

  with user():
      lm += cot_prompt

  with assistant():
      lm += gen(max_tokens = 500, temperature = config["TEMPERATURE"], name = "chain_of_thought")

  with user():
      lm += answer_prompt + select([0, 1], name = "answer")
      
  output = {"chain_of_thought": lm["chain_of_thought"], "answer": lm["answer"]}
  return json.dumps(output)
  
  

def getHypothesis(a_term: str, b_term: str) -> str:
  return f"{b_term} may effectively alleviate or target key pathogenic mechanisms of {a_term}, potentially offering therapeutic benefits or slowing disease progression."

def main():
    ###################### Argument Parsing ############################ 
    parser = argparse.ArgumentParser(description='Mistral7B Inference')
    
    parser.add_argument('--km_output', type = argparse.FileType('r'), required = True,
                        help='Path to the TSV file holding a km run output.')
    parser.add_argument('--config', type = argparse.FileType('r'),
                        help='Config file for kmGPT run.', required = True)
    parser.add_argument('--output_file', type = argparse.FileType('w', encoding='latin-1'), required = True, help='Name of TSV file hold filtered output.')
    args = parser.parse_args()
    
    ###################### Data Loading & Processsing ############################ 
    km_output = read_tsv_to_dataframe(args.km_output)
    config = json.load(args.config)
    output_file = args.output_file
    
    # PMID intersection is a list represented as a string...
    b_terms_pmids = km_output.ab_pmid_intersection.map(lambda pmid_list: pmid_list.strip('][').split(', '))
    # Grab only the abstract from each list of pmids in the TSV
    abstracts = [process_abstracts_data(config, pmid_list)[0] for pmid_list in b_terms_pmids] # Fetch abstracts from each b_term's PMID list
    # There should only be one a_term, so it's safe to grab the first index
    a_term = km_output.a_term.unique().tolist()[0]
    b_terms = km_output.b_term.unique().tolist()
  
    hypotheses = [getHypothesis(a_term, b_term) for b_term in b_terms]
    ###################### Model Loading & Inference ############################
    inference_config = config["JOB_SPECIFIC_SETTINGS"]["abstract_filter"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=inference_config["BNB_CONFIG"]["LOAD_IN_4BIT"],
        bnb_4bit_quant_type=inference_config["BNB_CONFIG"]["BNB_4BIT_QUANT_TYPE"],
        bnb_4bit_use_double_quant=inference_config["BNB_CONFIG"]["BNB_4BIT_USE_DOUBLE_QUANT"],
    )
    model = models.TransformersChat(inference_config["MODEL"], quantization_config = bnb_config, torch_dtype = torch.bfloat16, device_map = inference_config["DEVICE"], trust_remote_code = True)
    
    output = generateOutput(model, inference_config, hypotheses[0], abstracts[0])
    print(output)
    return output
    
if __name__ == '__main__':
    main()
