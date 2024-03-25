# add parent directory to path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
# import openai
import time
import json
import shutil
import re
import importlib
import inspect
import copy
from datetime import datetime
import skim_and_km_api as skim
import argparse
import sys
import get_pubmed_text as pubmed
import ssh_helper as ssh


def km_with_gpt_workflow(config, output_directory, file_paths=None):
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

        km_file_path = skim.km_with_gpt_workflow(local_config, output_directory)
        if km_file_path is not None:
            base, extension = os.path.splitext(km_file_path)
            new_file_name = f"{base}_{b_term}{extension}"
            os.rename(km_file_path, new_file_name)
            km_file_paths.append(new_file_name)

    return km_file_paths

def skim_with_gpt_workflow(config, output_directory, a_term, c_term):
    skim_file_paths = []  # Initialize the list to collect file paths

    # c_terms = skim.read_terms_from_file(
    #     config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]["C_TERMS_FILE"]
    # )

    # if not c_terms:
    #     print("C terms are empty")
    #     return skim_file_paths

    # a_terms = [config["GLOBAL_SETTINGS"]["A_TERM"]]
    # if config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]["A_TERM_LIST"]:
    #     a_terms = skim.read_terms_from_file(
    #         config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]["A_TERMS_FILE"]
    #     )

    # for a_term in a_terms:
    #     for c_term in c_terms:
    local_config = copy.deepcopy(config)
    local_config["GLOBAL_SETTINGS"]["A_TERM"] = a_term
    c_term_file = os.path.join(output_directory, f"{c_term}.txt")
    with open(c_term_file, "w") as f:
        f.write(c_term)
    local_config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]["C_TERMS_FILE"] = c_term_file
    # set the [OUTPUT_JSON] key in the config to its current name with the a_term and c_term appended
    skim_file_path = skim.skim_run(local_config, output_directory)
    if skim_file_path:
        skim_file_paths.append(skim_file_path)
        config["OUTPUT_JSON"] = os.path.basename(skim_file_path)
    os.remove(c_term_file)

    return skim_file_paths, config


def main_workflow(config, output_directory, timestamp_output_path, terms):
    job_type = config.get("JOB_TYPE", "")
    a_term = terms[0]
    c_term = terms[1]
    
    # Placeholder for the file paths that workflows are expected to generate
    generated_file_paths = []

    # Process based on job type
    if job_type == "km_with_gpt":
        generated_file_paths.extend(km_with_gpt_workflow(config, output_directory))
    elif job_type == "position_km_with_gpt":
        generated_file_paths.extend(position_km_with_gpt_workflow(config, output_directory))
    elif job_type == "skim_with_gpt":
        generated_file_paths, config = skim_with_gpt_workflow(config, output_directory, a_term, c_term)
    else:
        print("JOB_TYPE does not match known workflows.")
        return
    # SSH configurations are assumed to be stored under the "SSH" key in the config
    ssh_config = config.get("SSH", {})
    if ssh_config and generated_file_paths:
        # Create SSH client using the key for authentication
        ssh_client = ssh.create_ssh_client(ssh_config['server'], ssh_config['port'], ssh_config['user'], ssh_config.get('key_path'))
        
        # Create the subdirectory in the remote path
        remote_subdir_path = os.path.join(ssh_config['remote_path'], timestamp_output_path)
        remote_src_path = os.path.join(ssh_config['remote_path'], 'src')
        ssh.execute_remote_command(ssh_client, f"mkdir -p {remote_src_path}")
        ssh.execute_remote_command(ssh_client, f"mkdir -p {remote_subdir_path}")
        config_path = os.path.join(output_directory, "config.json")

        try:
            # Transfer generated files to the newly created subdirectory
            for path_item in generated_file_paths:
                # Normalize handling for both individual path items and lists
                if not isinstance(path_item, list):
                    path_item = [path_item]  # Make it a list for uniform processing
                for file_path in path_item:
                    local_file = os.path.abspath(file_path)  # Get the absolute path of the file
                    file_name = os.path.basename(file_path)  # Extract the file name
                    # Make a safe file name
                    safe_file_name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', file_name)
                    base_name = safe_file_name.replace("_output_filtered.tsv", "")
                    json_file_name = f"{base_name}.json"
                    config["OUTPUT_JSON"] = json_file_name
                    with open(os.path.join(output_directory, "config.json"), "w") as f:
                        json.dump(config, f, indent=4)
                    remote_file_path = os.path.join(remote_subdir_path, safe_file_name)  # Construct the remote path with file name

                    # Transfer the file
                    ssh.transfer_files(ssh_client, local_file, remote_file_path)
                    ssh.transfer_files(ssh_client, ssh_config["src_path"], remote_src_path)
                    # Transfer the config.json file from a local path specified in ssh_config to the remote subdirectory
                    remote_config_path = os.path.join(remote_subdir_path, "config.json")
                    ssh.transfer_files(ssh_client, config_path, remote_config_path)
                    ssh.execute_remote_command(ssh_client, f"cp {remote_src_path}/run.sub {remote_subdir_path}")
                    ssh.execute_remote_command(ssh_client, f"cp {remote_src_path}/run.sh {remote_subdir_path}")
                    ssh.execute_remote_command(ssh_client, f"cd {remote_subdir_path} && condor_submit -verbose -debug run.sub")
                    dynamic_file_names = [
                        f"filtered_{safe_file_name}", 
                        f"cot_{safe_file_name}", 
                        json_file_name  # Use the dynamically generated name for the json file
                    ]
                    # Informative print to know what files are being waited on
                    print(f"Waiting for files: {dynamic_file_names}")
                    # Wait for the dynamically specified files
                    ssh.monitor_files_and_extensions(ssh_client, remote_subdir_path, output_directory, dynamic_file_names, ['.log', '.err', '.out', ])
            print("Files transferred successfully.")
            # cleanup
            ssh.execute_remote_command(ssh_client, f"rm -rf {remote_src_path}")
            ssh.execute_remote_command(ssh_client, f"rm -rf {remote_subdir_path}")
        finally:
            # Close the SSH connection
            ssh_client.close()
    else:
        print("SSH configuration not found or no files to transfer.")
        return
