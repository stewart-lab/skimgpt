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
import time
from utils import Config, setup_logger
from htcondor_helper import HTCondorHelper
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize logger at the module level
logger = setup_logger()

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
    base_output_dir = os.path.abspath("../output")  # Make path absolute
    os.makedirs(base_output_dir, exist_ok=True)
    timestamp_dir_name = f"output_{timestamp}"
    output_directory = os.path.join(base_output_dir, timestamp_dir_name)
    os.makedirs(output_directory, exist_ok=True)

    # Update logger with output directory
    logger = setup_logger(output_directory)
    logger.info(f"Initializing workflow in {output_directory}")

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
    base_output_dir = os.path.abspath("../relevance_tests")  # Make path absolute
    os.makedirs(base_output_dir, exist_ok=True)
    timestamp_dir_name = f"eval_{timestamp}"
    output_directory = os.path.join(base_output_dir, timestamp_dir_name)
    os.makedirs(output_directory, exist_ok=True)
    
    logger.info(f"Initializing eval workflow in {output_directory}")
    
    # Copy config and input files
    shutil.copy(
        GlobalClass.config_file,
        os.path.join(output_directory, "config.json"),
    )
    
    generated_file_paths = []
    for test_file in glob(f"{tsv_dir}/*.tsv"):
        output_path = os.path.join(output_directory, os.path.basename(test_file))
        shutil.copy2(test_file, output_path)
        generated_file_paths.append(output_path)
        
    config = get_config(output_directory)
    assert config, "Configuration is empty or invalid"
    return config, output_directory, timestamp_dir_name, generated_file_paths


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
    try:
        with open(full_path, "w", encoding='utf-8') as outfile:
            json.dump(data, outfile, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error writing JSON file {full_path}: {str(e)}")
        raise


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
            try:
                file_path = os.path.join(root, file)
                if file.endswith(".json") and file != "config.json":
                    shutil.move(file_path, os.path.join(results_dir, file))
                elif file == "no_results.txt":
                    shutil.move(file_path, os.path.join(results_dir, file))
                elif file.endswith((".tsv", ".log", ".err", ".sub", ".out")):
                    shutil.move(file_path, os.path.join(debug_dir, file))
                elif file != "config.json":
                    os.remove(file_path)
            except Exception as e:
                logger.error(f"Error processing file {file}: {str(e)}")
                continue

    # Clean up empty directories
    for root, dirs, files in os.walk(directory, topdown=False):
        for dir in dirs:
            try:
                dir_path = os.path.join(root, dir)
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)
            except Exception as e:
                logger.error(f"Error removing directory {dir}: {str(e)}")
                continue

    filtered_dir = os.path.join(directory, "filtered")
    if os.path.exists(filtered_dir):
        try:
            shutil.rmtree(filtered_dir)
        except Exception as e:
            logger.error(f"Error removing filtered directory: {str(e)}")


def main():
    global logger  # Move global declaration to the start of the function
    
    start_time = time.time()
    logger.info("Main workflow started.")
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
    logger = setup_logger()  # Initial setup without output directory
    start_time = time.time()
    logger.info("Main workflow started.")
    if not args.tsv_dir:
        config, output_directory, timestamp_output_path = initialize_workflow()
        logger = setup_logger(output_directory)  # Update logger with output directory
        # Comment out SSH configuration
        """
        ssh_config = config.get("SSH", {})
        if ssh_config:
            ssh_helper = SSHHelper(ssh_config)
        """
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
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Main workflow completed in {elapsed_time:.2f} seconds.")
    
    # Comment out SSH section
    """
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
                print("No files were transferred. Skipping job submission and monitoring.")
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
            pass
    else:
        print("SSH configuration not found or no files to transfer.")
        return
    """

    # Use HTCondor submission method
    if config.get("HTCONDOR"):
        htcondor_helper = HTCondorHelper(config["HTCONDOR"])

        try:
            # Create src directory in output directory
            output_src_dir = os.path.join(output_directory, "src")
            os.makedirs(output_src_dir, exist_ok=True)
            
            # Ensure we're working with absolute paths
            output_directory = os.path.abspath(output_directory)
            
            # Flatten and resolve file paths
            flattened_file_paths = []
            for item in generated_file_paths:
                if item:
                    if isinstance(item, list):
                        flattened_file_paths.extend([os.path.abspath(p) for p in item])
                    else:
                        flattened_file_paths.append(os.path.abspath(item))
            
            if not flattened_file_paths:
                print("No files to process")
                return

            # Create files.txt for HTCondor queue
            files_txt_path = os.path.join(output_directory, "files.txt")
            logger.info(f"Creating files.txt at {files_txt_path}")
            
            with open(files_txt_path, "w") as f:
                for file_path in flattened_file_paths:
                    filename = os.path.basename(file_path)
                    f.write(f"{filename}\n")
                    logger.debug(f"Added {filename} to files.txt")

            # Verify files.txt was created
            if not os.path.exists(files_txt_path):
                raise FileNotFoundError(f"Failed to create {files_txt_path}")

            # Copy necessary files to output directory
            for file in ["run.sh", "run.sub", "relevance.py"]:
                src_path = os.path.abspath(os.path.join(os.getcwd(), file))
                dst_path = os.path.join(output_directory, file)
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                    logger.debug(f"Copied {file} to {dst_path}")
            
            # Copy source files to src directory
            for src_file in glob("*.py"):
                src_path = os.path.join(os.getcwd(), src_file)
                dst_path = os.path.join(output_src_dir, src_file)
                if os.path.abspath(src_path) != os.path.abspath(dst_path):
                    shutil.copy2(src_path, dst_path)
            
            # Copy input files to output directory
            for src_path in flattened_file_paths:
                dst_path = os.path.abspath(os.path.join(output_directory, os.path.basename(src_path)))
                if src_path != dst_path and os.path.exists(src_path):
                    try:
                        shutil.copy2(src_path, dst_path)
                    except shutil.SameFileError:
                        logging.debug(f"Skipping copy of {src_path} as it's already in the destination")
                        continue
            
            # Submit jobs from the output directory
            original_dir = os.getcwd()
            os.chdir(output_directory)
            try:
                cluster_id = htcondor_helper.submit_jobs(files_txt_path, output_directory)
                print(f"Jobs submitted with cluster ID {cluster_id}")

                # Monitor jobs
                if htcondor_helper.monitor_jobs(cluster_id):
                    print("Jobs completed, retrieving output...")
                    htcondor_helper.retrieve_output(cluster_id)

                # Process results
                print("Processing results...")
                organize_output(output_directory)
                extract_and_write_scores(output_directory)
                
                # Cleanup
                htcondor_helper.cleanup(cluster_id)
            finally:
                os.chdir(original_dir)
            
            print(f"Analysis complete. Results are in {output_directory}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            logging.error(f"Job processing failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
