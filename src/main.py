import multiprocessing
from atomic_chtc import main_workflow
# add parent directory to path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
# import openai
import time
import itertools
import json
import shutil
import re
import importlib
import inspect
import copy
from functools import partial
from datetime import datetime
import skim_and_km_api as skim
import argparse
import sys
import get_pubmed_text as pubmed
import ssh_helper as ssh

class Singleton(type):
	def __init__(cls, name, bases, dict):
		super(Singleton, cls).__init__(name, bases, dict)
		cls.instance = None

class GlobalClass(object):
	__metaclass__ = Singleton
	config_file = 'y'
	def __init__():
		print("I am global and whenever attributes are added in one instance, any other instance will be affected as well.")

# Ron is using: "./configRMS_needSpecialTunnel.json"
def initialize_workflow():
	# Generate a timestamp string
	timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
	
	# Define the base output directory and ensure it exists
	base_output_dir = "../output"
	os.makedirs(base_output_dir, exist_ok=True)
	
	# Define the name of the timestamped output directory
	timestamp_dir_name = f"output_{timestamp}"
	
	# Create the timestamped output directory within 'output'
	output_directory = os.path.join(base_output_dir, timestamp_dir_name)
	os.makedirs(output_directory, exist_ok=True)
	
	# Set timestamp_output_path to just the name of the timestamped directory
	timestamp_output_path = timestamp_dir_name  # This holds just the directory name, not the full path
	
	# Copy the config file into the timestamped output directory
	shutil.copy(
		GlobalClass.config_file,  # Assuming GlobalClass.config_file is defined elsewhere
		os.path.join(output_directory, "config.json"),
	)
	
	# Assuming get_config is a function that reads and returns the configuration
	config = get_config(output_directory)  # Use the full path here for reading the config
	assert config, "Configuration is empty or invalid"
	
	# Return the configuration, the full path to the output directory, and the lowest level directory name
	return config, output_directory, timestamp_output_path


def get_output_json_filename(config, job_settings):
	a_term = config["GLOBAL_SETTINGS"]["A_TERM"]
	output_json_map = {
		"km_with_gpt": f"km_with_gpt_{a_term}_output.json",
		"post_km_analysis": f"{a_term}_drug_synergy_maxAbstracts{config['GLOBAL_SETTINGS'].get('MAX_ABSTRACTS', '')}.json",
		"drug_discovery_validation": f"{a_term}_censorYear{job_settings.get('skim', {}).get('censor_year', '')}_numCTerms{config['GLOBAL_SETTINGS'].get('NUM_C_TERMS', '')}.json",
		"position_km_with_gpt": "position_km_with_gpt.json",
		"skim_with_gpt": "skim_with_gpt.json",
	}

	output_json = output_json_map.get(config["JOB_TYPE"])
	if output_json is None:
		raise ValueError(f"Invalid job type: {config['JOB_TYPE']}")

	return output_json.replace(" ", "_").replace("'", "")

def get_config(output_directory):
	config_path = os.path.join(output_directory, "config.json")
	with open(config_path, "r") as f:
		config = json.load(f)

	job_settings = config["JOB_SPECIFIC_SETTINGS"].get(config["JOB_TYPE"], {})
	config["OUTPUT_JSON"] = get_output_json_filename(config, job_settings)

	api_key = os.getenv("OPENAI_API_KEY", "")
	if not api_key:
		raise ValueError("OPENAI_API_KEY environment variable not set.")

	config["API_KEY"] = api_key

	pubmed_api_key = os.getenv("PUBMED_API_KEY", "")
	if not pubmed_api_key:
		raise ValueError("PUBMED_API_KEY environment variable not set.")
	config["PUBMED_API_KEY"] = pubmed_api_key

	with open(os.path.join(output_directory, "config.json"), "w") as f:
		json.dump(config, f, indent=4)

	return config


def read_tsv_to_dataframe(file_path):
	return pd.read_csv(file_path, sep="\t")


def write_to_json(data, file_path, output_directory):
	full_path = os.path.join(output_directory, file_path)
	with open(full_path, "w") as outfile:
		json.dump(data, outfile, indent=4)



def create_corrected_file_path(original_path):
	# Split the original path into name and extension
	file_name, file_extension = os.path.splitext(original_path)
	# Create a new path with "corrected" appended
	new_path = f"{file_name}_corrected{file_extension}"
	return new_path

def main():
	parser = argparse.ArgumentParser("arg_parser")
	parser.add_argument("-config", "--config_file", dest='config_file', help="Config file. Default=config.json.", default="../config.json", type=str)
	args = parser.parse_args()
	GlobalClass.config_file = args.config_file

	# Assuming initialize_workflow loads the config file and sets up the output directory
	config, output_directory, timestamp_output_path = initialize_workflow()
	
	c_terms = skim.read_terms_from_file(
		config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]["C_TERMS_FILE"]
	)
	
	a_terms = [config["GLOBAL_SETTINGS"]["A_TERM"]]
	if config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]["A_TERM_LIST"]:
		a_terms = skim.read_terms_from_file(config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]["A_TERMS_FILE"])
  
	terms = itertools.product(a_terms, c_terms)
	workflow = partial(main_workflow, config, output_directory, timestamp_output_path)
 
	with multiprocessing.Pool() as p:
		p.map(workflow, terms)
  
if __name__ == "__main__":
    main()