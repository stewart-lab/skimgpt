import ssh_helper as ssh
import get_pubmed_text as pubmed
import argparse
import skim_and_km_api as skim
from datetime import datetime
from functools import partial
import copy
import inspect
import importlib
import re
import shutil
import json
import itertools
import time
import pandas as pd
import multiprocessing
from jobs import main_workflow
# add parent directory to path
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import openai


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
	# This holds just the directory name, not the full path
	timestamp_output_path = timestamp_dir_name

	# Copy the config file into the timestamped output directory
	shutil.copy(
		GlobalClass.config_file,  # Assuming GlobalClass.config_file is defined elsewhere
		os.path.join(output_directory, "config.json"),
	)

	# Assuming get_config is a function that reads and returns the configuration
	# Use the full path here for reading the config
	config = get_config(output_directory)
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
	parser.add_argument("-config", "--config_file", dest='config_file',
						help="Config file. Default=config.json.", default="../config.json", type=str)
	args = parser.parse_args()
	GlobalClass.config_file = args.config_file

	# Assuming initialize_workflow loads the config file and sets up the output directory
	config, output_directory, timestamp_output_path = initialize_workflow()

	c_terms = skim.read_terms_from_file(
		config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]["C_TERMS_FILE"]
	)

	a_terms = [config["GLOBAL_SETTINGS"]["A_TERM"]]
	if config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]["A_TERM_LIST"]:
		a_terms = skim.read_terms_from_file(
			config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]["A_TERMS_FILE"])

	terms = itertools.product(a_terms, c_terms)
	workflow = partial(main_workflow, config,
					   output_directory, timestamp_output_path)

	with multiprocessing.Pool() as p:
		generated_file_paths = p.map(workflow, terms)
 
	ssh_config = config.get("SSH", {})
	if ssh_config and generated_file_paths:
		# Create SSH client using the key for authentication
		ssh_client = ssh.create_ssh_client(
			ssh_config['server'], ssh_config['port'], ssh_config['user'], ssh_config.get('key_path'))

		# Create the subdirectory in the remote path
		remote_subdir_path = os.path.join(
			ssh_config['remote_path'], timestamp_output_path)
		remote_src_path = os.path.join(ssh_config['remote_path'], 'src')
		ssh.execute_remote_command(ssh_client, f"mkdir -p {remote_src_path}")
		ssh.execute_remote_command(
			ssh_client, f"mkdir -p {remote_subdir_path}")
		config_path = os.path.join(output_directory, "config.json")

		try:
			# Transfer generated files to the newly created subdirectory
			remote_file_paths = []
			dynamic_file_names = []
			for path_item in generated_file_paths:
				# Normalize handling for both individual path items and lists
				if not isinstance(path_item, list):
					# Make it a list for uniform processing
					path_item = [path_item]
				for file_path in path_item:
					# Get the absolute path of the file
					local_file = os.path.abspath(file_path)
					file_name = os.path.basename(
						file_path)  # Extract the file name
					# Make a safe file name
					safe_file_name = re.sub(
						r'[^a-zA-Z0-9_\-\.]', '_', file_name)
					base_name = safe_file_name.replace(
						"_output_filtered.tsv", "")
					json_file_name = f"{base_name}.json"
					config["OUTPUT_JSON"] = json_file_name
					with open(os.path.join(output_directory, "config.json"), "w") as f:
						json.dump(config, f, indent=4)
					# Construct the remote path with file name
					remote_file_path = os.path.join(
						remote_subdir_path, safe_file_name)

					remote_file_paths.append(remote_file_path.split("/")[-1])
					dynamic_file_names.append(f"filtered_{safe_file_name}")
					dynamic_file_names.append(f"cot_{safe_file_name}")
					dynamic_file_names.append(json_file_name)
					# Transfer the filex
					ssh.transfer_files(
						ssh_client, local_file, remote_file_path)
					ssh.transfer_files(
						ssh_client, ssh_config["src_path"], remote_src_path)
					# Transfer the config.json file from a local path specified in ssh_config to the remote subdirectory
					remote_config_path = os.path.join(
						remote_subdir_path, "config.json")
					ssh.transfer_files(
						ssh_client, config_path, remote_config_path)
	 
			with open("./files.txt", "w+") as f:
				for remote_file in remote_file_paths:
					f.write(f"{remote_file}\n")
	 
			ssh.transfer_files(
						ssh_client, "./files.txt", remote_subdir_path)
			ssh.execute_remote_command(
				ssh_client, f"cp {remote_src_path}/run.sub {remote_subdir_path}")
			ssh.execute_remote_command(
				ssh_client, f"cp {remote_src_path}/run.sh {remote_subdir_path}")
			ssh.execute_remote_command(
				ssh_client, f"cd {remote_subdir_path} && condor_submit -verbose -debug run.sub")
			# Informative print to know what files are being waited on
			print(f"Waiting for files: {dynamic_file_names}")
			# Wait for the dynamically specified files
			ssh.monitor_files_and_extensions(
				ssh_client, remote_subdir_path, f"{output_directory}/filtered", dynamic_file_names, ['.log', '.err', '.out', ])
			print("Files transferred successfully.")
			# cleanup
			ssh.execute_remote_command(ssh_client, f"rm -rf {remote_src_path}")
			ssh.execute_remote_command(
				ssh_client, f"rm -rf {remote_subdir_path}")
		finally:
			# Close the SSH connection
			ssh_client.close()
	else:
		print("SSH configuration not found or no files to transfer.")
		return


if __name__ == "__main__":
	main()
