import gradio as gr
from datetime import datetime
import json
import subprocess
import os

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
	
	current_time = datetime.now().strftime("%Y%m%d%H%M")
	os.chdir("../src")
	subprocess.run(["python", "main.py", "--config", "../webapp/run.json", "-output_dir_ext", current_time])
	print("Subprocess done.")
	os.chdir("../webapp")
 
	with open(f"../output/output_{current_time}/config.json", 'r') as config_file:
		run_config = json.load(config_file)
  
	output_json = run_config["OUTPUT_JSON"]
	output_path = f"../output/output_{current_time}/filtered/{output_json}"
	
	with open(output_path, "r") as gpt_output:
		return json.load(gpt_output)

demo = gr.Interface(
	fn=skim_gpt,
	inputs=["text", "text", "text"],
	outputs=["json"],
	title="SkimGPT",
	description = "Runs a SkimGPT run with an A Term, B Term, and a C Term. Outputs GPT4's Response."
)

demo.launch()