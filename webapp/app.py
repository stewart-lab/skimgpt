import gradio as gr
from datetime import datetime
import json
import subprocess


def skim_gpt(a_terms, b_terms, c_terms):
	with open("../config.json", 'r') as config_file:
		base_config = json.load(config_file)
	 
	with open("a_terms.txt", "w") as file:
		file.write(a_terms)
	 
	with open("b_terms.txt", "w") as file:
		file.write(b_terms)
	
	with open("c_terms.txt", "w") as file:
		file.write(c_terms)
		
	base_config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]["A_TERMS_FILE"] = "../webapp/a_terms.txt"
	base_config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]["B_TERMS_FILE"] = "../webapp/b_terms.txt"
	base_config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]["C_TERMS_FILE"] = "../webapp/c_terms.txt"
	
	with open("run.json", "w") as outfile: 
		json.dump(base_config, outfile)
  
	output_path = subprocess.run(["python", "../src/main.py", "--config", "run.json"])
	import pdb; pdb.set_trace()
	return "Hi"

demo = gr.Interface(
	fn=skim_gpt,
	inputs=["text", "text", "text"],
	outputs=["json"],
	title="SkimGPT",
	description = "Runs a SkimGPT run with an A Term, B Term, and a C Term. Outputs GPT4 Json."
)

demo.launch()