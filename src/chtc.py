# add parent directory to path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import openai
import time
import json
import shutil
import re
import importlib
import inspect
import copy
from datetime import datetime
from src import skim_and_km_api as skim
import test.test_abstract_comprehension as test
import argparse
import sys
from src import get_pubmed_text as pubmed
from src import ssh_helper as ssh

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
    base_output_dir = "output"
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
        "km_with_gpt": f"{a_term}_km_with_gpt.json",
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

def skim_with_gpt_workflow(config, output_directory):
    skim_file_paths = []  # Initialize the list to collect file paths

    c_terms = skim.read_terms_from_file(
        config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]["C_TERMS_FILE"]
    )

    if not c_terms:
        print("C terms are empty")
        return skim_file_paths

    a_terms = [config["GLOBAL_SETTINGS"]["A_TERM"]]
    if config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]["A_TERM_LIST"]:
        a_terms = skim.read_terms_from_file(
            config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]["A_TERMS_FILE"]
        )

    for a_term in a_terms:
        for c_term in c_terms:
            local_config = copy.deepcopy(config)
            local_config["GLOBAL_SETTINGS"]["A_TERM"] = a_term
            c_term_file = os.path.join(output_directory, f"{c_term}.txt")
            with open(c_term_file, "w") as f:
                f.write(c_term)
            local_config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]["C_TERMS_FILE"] = c_term_file

            skim_file_path = skim.skim_run(local_config, output_directory)
            if skim_file_path:
                skim_file_paths.append(skim_file_path)

            os.remove(c_term_file)

    return skim_file_paths


def main_workflow():
    parser = argparse.ArgumentParser("arg_parser")
    parser.add_argument("-config", "--config_file", dest='config_file', help="Config file. Default=config.json.", default="config.json", type=str)
    args = parser.parse_args()
    GlobalClass.config_file = args.config_file

    # Assuming initialize_workflow loads the config file and sets up the output directory
    config, output_directory, timestamp_output_path = initialize_workflow()
    job_type = config.get("JOB_TYPE", "")
    
    # Placeholder for the file paths that workflows are expected to generate
    generated_file_paths = []

    # Process based on job type
    if job_type == "km_with_gpt":
        generated_file_paths.extend(km_with_gpt_workflow(config, output_directory))
    elif job_type == "position_km_with_gpt":
        generated_file_paths.extend(position_km_with_gpt_workflow(config, output_directory))
    elif job_type == "skim_with_gpt":
        generated_file_paths.extend(skim_with_gpt_workflow(config, output_directory))
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
                    file_name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', file_name)
                    remote_file_path = os.path.join(remote_subdir_path, file_name)  # Construct the remote path with file name
                    
                    # Transfer the file
                    ssh.transfer_files(ssh_client, local_file, remote_file_path)
                    remote_sub_file = os.path.join(ssh_config['remote_path'], "run.sub")
                    remote_executable_file = os.path.join(ssh_config['remote_path'], "run.sh")

                    # Paths for the files in the remote subdirectory
                    remote_subdir_sub_file = os.path.join(remote_subdir_path, "run.sub")
                    remote_subdir_executable_file = os.path.join(remote_subdir_path, "run.sh")

                    # Copy the .sub file to the remote subdirectory
                    ssh.execute_remote_command(ssh_client, f"cp {remote_sub_file} {remote_subdir_sub_file}")

                    # Copy the .sh file to the remote subdirectory
                    ssh.execute_remote_command(ssh_client, f"cp {remote_executable_file} {remote_subdir_executable_file}")

                    # Transfer the config.json file from a local path specified in ssh_config to the remote subdirectory
                    remote_config_path = os.path.join(remote_subdir_path, "config.json")
                    ssh.transfer_files(ssh_client, config_path , remote_config_path)
                    ssh.execute_remote_command(ssh_client, f"cd {remote_subdir_path} && condor_submit run.sub")
            
            # Execute a command if needed, e.g., to process files in the remote subdirectory
            # ssh.execute_remote_command(ssh_client, "your_command_here")
        finally:
            # Close the SSH connection
            ssh_client.close()




if __name__ == "__main__":
    main_workflow()
