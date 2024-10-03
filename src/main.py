from ssh_helper import SSHHelper
import argparse
import skim_and_km_api as skim
from datetime import datetime
from functools import partial
from eval_JSON_results import extract_and_write_scores
import shutil
import json
import itertools
import pandas as pd
import multiprocessing
from jobs import main_workflow
from glob import glob
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Singleton(type):
    def __init__(cls, name, bases, dict):
        super(Singleton, cls).__init__(name, bases, dict)
        cls.instance = None


class GlobalClass(object):
    __metaclass__ = Singleton
    config_file = "y"

    def __init__():
        print(
            "I am global and whenever attributes are added in one instance, any other instance will be affected as well."
        )


def initialize_workflow():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    base_output_dir = "../output"
    os.makedirs(base_output_dir, exist_ok=True)
    timestamp_dir_name = f"output_{timestamp}"
    output_directory = os.path.join(base_output_dir, timestamp_dir_name)
    os.makedirs(output_directory, exist_ok=True)
    timestamp_output_path = timestamp_dir_name
    shutil.copy(
        GlobalClass.config_file,
        os.path.join(output_directory, "config.json"),
    )
    config = get_config(output_directory)
    assert config, "Configuration is empty or invalid"
    return config, output_directory, timestamp_output_path


def initialize_eval_workflow(tsv_dir):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    base_output_dir = "../relevance_tests"
    os.makedirs(base_output_dir, exist_ok=True)
    timestamp_dir_name = f"eval_{timestamp}"
    output_directory = os.path.join(base_output_dir, timestamp_dir_name)
    os.makedirs(output_directory, exist_ok=True)
    timestamp_output_path = timestamp_dir_name
    shutil.copy(
        GlobalClass.config_file,
        os.path.join(output_directory, "config.json"),
    )
    generated_file_paths = []
    for test_file in glob(f"{tsv_dir}/*.tsv"):
        output_path = os.path.join(output_directory, test_file.split("/")[-1])
        shutil.copy(test_file, output_path)
        generated_file_paths.append(output_path)
    config = get_config(output_directory)
    assert config, "Configuration is empty or invalid"
    return config, output_directory, timestamp_output_path, generated_file_paths


def get_config(output_directory):
    config_path = os.path.join(output_directory, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    job_settings = config["JOB_SPECIFIC_SETTINGS"].get(config["JOB_TYPE"], {})
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
    file_name, file_extension = os.path.splitext(original_path)
    new_path = f"{file_name}_corrected{file_extension}"
    return new_path


def organize_output(directory):
    results_dir = os.path.join(directory, "results")
    debug_dir = os.path.join(directory, "debug")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".json") and file != "config.json":
                shutil.move(file_path, os.path.join(results_dir, file))
            elif file.endswith((".tsv", ".log", ".err", ".sub", ".out")):
                shutil.move(file_path, os.path.join(debug_dir, file))
            elif file != "config.json":
                os.remove(file_path)
    for root, dirs, files in os.walk(directory, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)
    filtered_dir = os.path.join(directory, "filtered")
    if os.path.exists(filtered_dir):
        shutil.rmtree(filtered_dir)


def main():
    parser = argparse.ArgumentParser("arg_parser")
    parser.add_argument(
        "-config",
        "--config_file",
        dest="config_file",
        help="Config file. Default=config.json.",
        default="../config.json",
        type=str,
    )
    parser.add_argument("-tsv_dir", default=None, type=str)
    args = parser.parse_args()

    GlobalClass.config_file = args.config_file
    if not args.tsv_dir:
        config, output_directory, timestamp_output_path = initialize_workflow()
        ssh_config = config.get("SSH", {})
        if ssh_config:
            ssh_helper = SSHHelper(ssh_config)
        c_terms = skim.read_terms_from_file(
            config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]["C_TERMS_FILE"]
        )
        b_terms = skim.read_terms_from_file(
            config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]["B_TERMS_FILE"]
        )
        a_terms = [config["GLOBAL_SETTINGS"]["A_TERM"]]
        if config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]["A_TERM_LIST"]:
            a_terms = skim.read_terms_from_file(
                config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]["A_TERMS_FILE"]
            )
        if config["JOB_TYPE"] == "skim_with_gpt":
            terms = list(itertools.product(a_terms, c_terms))
            if config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"].get("position", False):
                terms = list(zip(a_terms, b_terms, c_terms))
        else:
            a_terms = [config["GLOBAL_SETTINGS"]["A_TERM"]]
            if config["JOB_SPECIFIC_SETTINGS"]["km_with_gpt"]["A_TERM_LIST"]:
                a_terms = skim.read_terms_from_file(
                    config["JOB_SPECIFIC_SETTINGS"]["km_with_gpt"]["A_TERMS_FILE"]
                )
            terms = a_terms
        workflow = partial(
            main_workflow, config, output_directory, timestamp_output_path
        )
        with multiprocessing.Pool() as p:
            generated_file_paths = p.map(workflow, terms)
    else:
        config, output_directory, timestamp_output_path, generated_file_paths = (
            initialize_eval_workflow(args.tsv_dir)
        )

    ssh_config = config.get("SSH", {})
    if ssh_config and generated_file_paths:
        ssh_helper = SSHHelper(ssh_config)

        remote_subdir_path = os.path.join(
            ssh_config["remote_path"], timestamp_output_path
        )
        remote_src_path = os.path.join(ssh_config["remote_path"], "src")

        try:
            ssh_helper.prepare_remote_directories(remote_src_path, remote_subdir_path)

            remote_file_paths, dynamic_file_names = ssh_helper.transfer_files_to_remote(
                output_directory, remote_subdir_path, generated_file_paths
            )
            if not remote_file_paths:
                print(
                    "No files were transferred. Skipping job submission and monitoring."
                )
                return

            ssh_helper.setup_and_submit_job(remote_src_path, remote_subdir_path)

            print(f"Job submitted. Monitoring {len(dynamic_file_names)} files...")
            ssh_helper.monitor_files_and_extensions(
                remote_subdir_path,
                f"{output_directory}/filtered",
                dynamic_file_names,
                [".log", ".err", ".out"],
                len(generated_file_paths),
                interval=10,
            )

            print("Job completed. Cleaning up remote directories...")
            ssh_helper.cleanup_remote_directories(remote_src_path, remote_subdir_path)

            print("Organizing output and extracting scores...")
            organize_output(output_directory)
            extract_and_write_scores(output_directory)

            print(f"Analysis complete. Results are in {output_directory}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
        finally:
            # Don't close the connection here, as it should persist for future use
            pass
    else:
        print("SSH configuration not found or no files to transfer.")
        return


if __name__ == "__main__":
    main()
