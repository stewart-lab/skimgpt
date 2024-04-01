# add parent directory to path
import ssh_helper as ssh
import get_pubmed_text as pubmed
import argparse
import skim_and_km_api as skim
from datetime import datetime
import copy
import inspect
import importlib
import re
import shutil
import json
import time
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import openai


def km_with_gpt_workflow(config, output_directory):
	if file_paths is None:
		file_paths = []

	if config["JOB_SPECIFIC_SETTINGS"]["km_with_gpt"].get("A_TERM_LIST"):
		a_terms = skim.read_terms_from_file(
			config["JOB_SPECIFIC_SETTINGS"]["km_with_gpt"]["A_TERMS_FILE"]
		)
		for a_term in a_terms:
			local_config = copy.deepcopy(config)
			local_config["GLOBAL_SETTINGS"]["A_TERM"] = a_term

			# Recursive call for each term, passing the file_paths list to accumulate paths
			km_with_gpt_workflow(local_config, output_directory, file_paths)

		return file_paths

	km_file_path = skim.km_with_gpt_workflow(
		config=config, output_directory=output_directory
	)

	if km_file_path:
		file_paths.append(km_file_path)

	return file_paths


def position_km_with_gpt_workflow(config, output_directory):
	km_file_paths = []  # Initialize the list to collect file paths

	assert config["JOB_SPECIFIC_SETTINGS"]["position_km_with_gpt"]["A_TERMS_FILE"], "A_TERMS_FILE is not defined in the configuration"

	a_terms = skim.read_terms_from_file(
		config["JOB_SPECIFIC_SETTINGS"]["position_km_with_gpt"]["A_TERMS_FILE"]
	)
	b_terms = skim.read_terms_from_file(
		config["JOB_SPECIFIC_SETTINGS"]["position_km_with_gpt"]["B_TERMS_FILE"]
	)

	for i, a_term in enumerate(a_terms):
		local_config = copy.deepcopy(config)
		local_config["GLOBAL_SETTINGS"]["A_TERM"] = a_term
		b_term = b_terms[i]
		b_term_file = os.path.join(output_directory, f"b_term_{i}.txt")
		with open(b_term_file, "w") as f:
			f.write(b_term)
		local_config["JOB_SPECIFIC_SETTINGS"]["km_with_gpt"]["B_TERMS_FILE"] = b_term_file

		km_file_path = skim.km_with_gpt_workflow(
			local_config, output_directory)
		if km_file_path is not None:
			base, extension = os.path.splitext(km_file_path)
			new_file_name = f"{base}_{b_term}{extension}"
			os.rename(km_file_path, new_file_name)
			km_file_paths.append(new_file_name)

	return km_file_paths


def skim_with_gpt_workflow(config, output_directory, a_term, c_term):
	skim_file_paths = []  # Initialize the list to collect file paths

	local_config = copy.deepcopy(config)
	local_config["GLOBAL_SETTINGS"]["A_TERM"] = a_term
	c_term_file = os.path.join(output_directory, f"{c_term}.txt")
	with open(c_term_file, "w") as f:
		f.write(c_term)
	local_config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]["C_TERMS_FILE"] = c_term_file
	# set the [OUTPUT_JSON] key in the config to its current name with the a_term and c_term appended
	skim_file_path = skim.skim_with_gpt_workflow(local_config, output_directory)
	if skim_file_path:
		skim_file_paths.append(skim_file_path)
		config["OUTPUT_JSON"] = os.path.basename(skim_file_path)

	return skim_file_paths, config


def main_workflow(config, output_directory, timestamp_output_path, terms):
	job_type = config.get("JOB_TYPE", "")
	a_term = terms[0]
	c_term = terms[1]

	# Placeholder for the file paths that workflows are expected to generate
	generated_file_paths = []

	# Process based on job type
	if job_type == "km_with_gpt":
		generated_file_paths.extend(
			km_with_gpt_workflow(config, output_directory))
	elif job_type == "position_km_with_gpt":
		generated_file_paths.extend(
			position_km_with_gpt_workflow(config, output_directory))
	elif job_type == "skim_with_gpt":
		generated_file_paths, config = skim_with_gpt_workflow(
			config, output_directory, a_term, c_term)
	else:
		print("JOB_TYPE does not match known workflows.")
	return generated_file_paths
