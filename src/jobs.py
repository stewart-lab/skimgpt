import src.skim_and_km_api as skim
import copy
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def km_with_gpt_workflow(config, output_directory, a_terms=None, file_paths=None):
    # Initialize file_paths if not provided
    if file_paths is None:
        file_paths = []
    local_config = copy.deepcopy(config)
    local_config["GLOBAL_SETTINGS"]["A_TERM"] = a_terms
    km_file_path = skim.km_with_gpt_workflow(local_config, output_directory)
    # Collect results
    if km_file_path:
        file_paths.append(km_file_path)
    return file_paths


def skim_with_gpt_workflow(config, output_directory, a_term, b_term, c_term):
    skim_file_paths = []  # Initialize the list to collect file paths

    local_config = copy.deepcopy(config)
    local_config["GLOBAL_SETTINGS"]["A_TERM"] = a_term
    if b_term:
        b_term_file = os.path.join(output_directory, f"{b_term}.txt")
        with open(b_term_file, "w") as f:
            f.write(b_term)
        local_config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"][
            "B_TERMS_FILE"
        ] = b_term_file

    c_term_file = os.path.join(output_directory, f"{c_term}.txt")
    with open(c_term_file, "w") as f:
        f.write(c_term)
    local_config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]["C_TERMS_FILE"] = c_term_file

    skim_file_path = skim.skim_with_gpt_workflow(local_config, output_directory)
    if skim_file_path:
        skim_file_paths.append(skim_file_path)
        config["OUTPUT_JSON"] = os.path.basename(skim_file_path)
    return skim_file_paths, config


def main_workflow(config, output_directory, timestamp_output_path, terms):
    job_type = config.get("JOB_TYPE", "")

    # Placeholder for the file paths that workflows are expected to generate
    generated_file_paths = []

    # Process based on job type
    if job_type == "km_with_gpt":
        a_terms = terms
        generated_file_paths.extend(
            km_with_gpt_workflow(config, output_directory, a_terms)
        )
    elif job_type == "skim_with_gpt":
        if config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]["position"]:
            a_term = terms[0]
            b_term = terms[1]
            c_term = terms[2]
        else:
            a_term = terms[0]
            c_term = terms[1]
            b_term = None
        generated_file_paths, config = skim_with_gpt_workflow(
            config, output_directory, a_term, b_term, c_term
        )
    else:
        print("JOB_TYPE does not match known workflows.")
    return generated_file_paths
