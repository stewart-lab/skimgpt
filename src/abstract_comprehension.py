import os
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
import skim_and_km_api as skim
import argparse
import sys
import get_pubmed_text as pubmed


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


# Ron is using: "./configRMS_needSpecialTunnel.json"
def initialize_workflow():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs("../output", exist_ok=True)
    output_directory = os.path.join("../output", f"output_{timestamp}")
    os.makedirs(output_directory, exist_ok=True)
    shutil.copy(
        GlobalClass.config_file,
        os.path.join(output_directory, "config.json"),
    )
    config = get_config(output_directory)
    assert config, "Configuration is empty or invalid"
    return config, output_directory


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


def analyze_abstract_with_gpt4(
    consolidated_abstracts, b_term, a_term, config, c_term=None
):
    if not b_term or not a_term:
        print("B term or A term is empty.")
        return []

    api_key = config.get("API_KEY", os.getenv("OPENAI_API_KEY", ""))
    if not api_key:
        raise ValueError("OpenAI API key is not set.")
    openai_client = openai.OpenAI(api_key=api_key)
    responses = []
    if not config["Evaluate_single_abstract"]:
        prompt = generate_prompt(
            b_term=b_term,
            a_term=a_term,
            content=consolidated_abstracts,
            config=config,
            c_term=c_term if c_term is not None else None,
        )
        response = call_openai(openai_client, prompt, config)
        if response:
            responses.append(response)
    elif config["Evaluate_single_abstract"]:
        for abstract in consolidated_abstracts:
            # Pass c_term if it is not None
            prompt = generate_prompt(
                b_term,
                a_term,
                abstract,
                config,
                c_term=c_term if c_term is not None else None,
            )
            response = call_openai(openai_client, prompt, config)
            if response:
                responses.append(response)
    else:
        raise ValueError("Please set True or False for evaluate_single_abstract.")

    return responses, prompt


def generate_prompt(b_term, a_term, content, config, c_term=None):
    job_type = config.get("JOB_TYPE", "").lower()
    b_term = b_term.replace("&", " ")
    # Define hypothesis templates directly based on job_type
    abc_hypothesis = config.get("SKIM_hypotheses")["ABC"].format(
        c_term=c_term, a_term=a_term, b_term=b_term
    )

    # Now, incorporate this into your hypothesis_templates dictionary
    hypothesis_templates = {
        "km_with_gpt": config.get("KM_hypothesis", "").format(
            b_term=b_term, a_term=a_term
        ),
        "position_km_with_gpt": config.get("POSITION_KM_hypothesis", "").format(
            b_term=b_term, a_term=a_term
        ),
        "skim_with_gpt": abc_hypothesis,  # Using the formatted ABC hypothesis directly
    }
    # Fetch the hypothesis template for the given job_type
    hypothesis_template = hypothesis_templates.get(job_type)
    if not hypothesis_template:
        return "No valid hypothesis for the provided JOB_TYPE."

    # Dynamically import the prompts module
    prompts_module = importlib.import_module("prompt_library")
    assert prompts_module, "Failed to import the prompts module."

    # Use job_type to fetch the corresponding prompt function
    prompt_function = getattr(prompts_module, job_type, None)
    if not prompt_function:
        raise ValueError(
            f"Prompt function for '{job_type}' not found in the prompts module."
        )
    prompt_args = (b_term, a_term, content, config)
    if "hypothesis_template" in inspect.signature(prompt_function).parameters:
        # If the prompt function expects a hypothesis_template, adjust the arguments
        prompt_args = (b_term, a_term, hypothesis_template, content)
        if c_term is not None:
            return prompt_function(*prompt_args, c_term=c_term)
        else:
            return prompt_function(*prompt_args)
    else:
        # Fallback for functions not expecting a hypothesis_template directly
        if (
            "c_term" in inspect.signature(prompt_function).parameters
            and c_term is not None
        ):
            return prompt_function(*prompt_args, c_term=c_term)
        else:
            return prompt_function(*prompt_args)


def perform_analysis(job_type, row, config, abstracts_data):
    b_term = row["b_term"]
    a_term = config["GLOBAL_SETTINGS"]["A_TERM"]
    c_term = None  # Initialize c_term with a default value of None

    # Set c_term if job_type is "skim_with_gpt"
    if job_type == "skim_with_gpt":
        c_term = row.get("c_term")  # Use .get to avoid KeyError if "c_term" is missing

    pmids = []
    if job_type in [
        "drug_discovery_validation",
        "km_with_gpt",
        "position_km_with_gpt",
    ]:
        pmids = pubmed.parse_pmids(row, "ab_pmid_intersection")
    elif job_type == "skim_with_gpt":
        if row["ac_pmid_intersection"] != "[]":
            print("Processing all three PMID lists for ABC")
            pmids = (
                pubmed.parse_pmids(row, "ab_pmid_intersection")
                + pubmed.parse_pmids(row, "bc_pmid_intersection")
                + pubmed.parse_pmids(row, "ac_pmid_intersection")
            )
        else:
            pmids = pubmed.parse_pmids(
                row, "ab_pmid_intersection"
            ) + pubmed.parse_pmids(row, "bc_pmid_intersection")
    print(f"Processing {len(pmids)} PMIDs for {b_term} and {a_term}")
    (
        consolidated_abstracts,
        paper_urls,
        publication_years,
    ) = pubmed.process_abstracts_data(config, pmids)

    # if all three lists are empty, then we have no data to process
    if not consolidated_abstracts and not paper_urls and not publication_years:
        return None, None, None, None, None

    # Pass c_term to the analyze function
    result, prompt = analyze_abstract_with_gpt4(
        consolidated_abstracts, b_term, a_term, config, c_term=c_term
    )

    return result, prompt, paper_urls, consolidated_abstracts, publication_years


def process_single_row(row, config):
    job_type = config.get("JOB_TYPE")

    if job_type not in [
        "drug_discovery_validation",
        "pathway_augmentation",
        "km_with_gpt",
        "position_km_with_gpt",
        "skim_with_gpt",
    ]:
        print("Invalid job type (caught in process_single_row)")
        return None

    (
        result,
        prompt,
        paper_urls,
        consolidated_abstracts,
        publication_years,
    ) = perform_analysis(job_type, row, config, {})

    # if everything is empty, then we have no data to process
    if (
        not result
        and not prompt
        and not paper_urls
        and not consolidated_abstracts
        and not publication_years
    ):
        return None
    return {
        "Relationship": f"{row['a_term']} - {row['b_term']}"
        + (f" - {row['c_term']}" if "c_term" in row else ""),
        "Result": result,
        "Prompt": prompt,  # Added prompt here
        "URLs": paper_urls,
        "Abstracts": consolidated_abstracts,
        "Years": publication_years,
    }


def test_openai_connection(config):
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    client = openai.OpenAI(api_key=openai.api_key)
    try:
        response = client.chat.completions.create(
            model=config["GLOBAL_SETTINGS"]["MODEL"],
            messages=[
                {"role": "system", "content": "You are a medical research analyst."},
                {"role": "user", "content": "Test connection to OpenAI."},
            ],
        )
        print("Successfully connected to OpenAI!")
    except Exception as e:
        print(f"Failed to connect to OpenAI. Error: {e}")


def call_openai(client, prompt, config):
    retry_delay = config["GLOBAL_SETTINGS"]["RETRY_DELAY"]
    max_retries = config["GLOBAL_SETTINGS"]["MAX_RETRIES"]
    model = config["GLOBAL_SETTINGS"]["MODEL"]
    max_tokens = config["GLOBAL_SETTINGS"]["MAX_TOKENS"]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a biomedical research analyst.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if content:
                return content
            else:
                print("Empty response received from OpenAI API.")
                time.sleep(retry_delay)

        except openai.RateLimitError as e:
            print("A 429 status code was received; we should back off a bit.")
            time.sleep(retry_delay)
            print(e.__cause__)
        except openai.APIStatusError as e:
            print("Another non-200-range status code was received")
            print(e.status_code)
            print(e.response)
            print(e.__cause__)
    return None


def save_to_json(data, config, output_directory):
    output_filename = os.path.join(
        output_directory, config["OUTPUT_JSON"] + "_filtered.json"
    )
    # Adding "_filtered" to the filename
    with open(output_filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Filtered results have been saved to {output_filename}")


def api_cost_estimator(df, config):
    job_type = config.get("JOB_TYPE", "")
    max_abstracts = config["GLOBAL_SETTINGS"]["MAX_ABSTRACTS"]
    estimated_cost = 0
    total_calls = 0

    def read_terms_length(file_path):
        return len(skim.read_terms_from_file(file_path))

    if job_type in ["drug_discovery_validation", "pathway_augmentation"]:
        estimated_cost = max_abstracts * len(df) * 0.006
    elif job_type == "km_with_gpt":
        if config["JOB_SPECIFIC_SETTINGS"]["km_with_gpt"]["A_TERM_LIST"]:
            term_length = read_terms_length(
                config["JOB_SPECIFIC_SETTINGS"]["km_with_gpt"]["A_TERMS_FILE"]
            )
        else:
            term_length = 1  # Default value if A_TERM_LIST is not set

        num_b_terms = config["JOB_SPECIFIC_SETTINGS"]["km_with_gpt"]["NUM_B_TERMS"]
        total_calls = max_abstracts * term_length * num_b_terms
        estimated_cost = total_calls * 0.006
    elif job_type == "post_km_analysis":
        robust_setting = (
            config["JOB_SPECIFIC_SETTINGS"]["post_km_analysis"]
            .get("robust", "False")
            .lower()
        )

        if robust_setting == "true":
            total_calls = sum(
                len(eval(row["panc & ggp & kras-mapk set"])) // (max_abstracts // 2)
                + len(eval(row["brd & ggp set"])) // (max_abstracts // 2)
                for _, row in df.iterrows()
            )
        else:
            total_calls = df.apply(
                lambda row: min(
                    len(eval(row["panc & ggp & kras-mapk set"]))
                    + len(eval(row["brd & ggp set"])),
                    max_abstracts,
                ),
                axis=1,
            ).sum()

        estimated_cost = total_calls * 0.06

    user_input = input(
        f"The following job consists of {total_calls} abstracts and will cost roughly ${estimated_cost:.2f} in GPT-4 API calls. Do you wish to proceed? [Y/n]: "
    )
    if user_input.lower() != "y":
        print("Exiting workflow.")
        return False
    return True


def apply_scores_to_df(df, scores_by_term):
    # Iterate over the DataFrame rows
    for index, row in df.iterrows():
        # Check if 'b_term' matches a term in the scores dictionary
        if row["b_term"] in scores_by_term:
            # Add the score to the 'ab_count' for the row
            df.at[index, "ab_count"] += scores_by_term[row["b_term"]]


def create_corrected_file_path(original_path):
    # Split the original path into name and extension
    file_name, file_extension = os.path.splitext(original_path)
    # Create a new path with "corrected" appended
    new_path = f"{file_name}_corrected{file_extension}"
    return new_path


def km_with_gpt_workflow(config, output_directory):
    a_terms = None
    a_terms_file_path = (
        config.get("JOB_SPECIFIC_SETTINGS", {})
        .get("km_with_gpt", {})
        .get("A_TERMS_FILE")
    )
    if a_terms_file_path and config.get("JOB_SPECIFIC_SETTINGS", {}).get("A_TERM_LIST"):
        try:
            a_terms = skim.read_terms_from_file(a_terms_file_path)
        except Exception as e:
            print(f"Failed to read terms from file {a_terms_file_path}: {e}")
            return

        if not a_terms:
            print("No terms found in the A_TERMS_FILE.")
            return

    for a_term in (
        a_terms if a_terms else [config.get("GLOBAL_SETTINGS", {}).get("A_TERM")]
    ):
        local_config = copy.deepcopy(config)
        local_config["GLOBAL_SETTINGS"]["A_TERM"] = a_term

        # Dynamically generate output_json name for each a_term
        output_json = f"{a_term}_km_with_gpt.json".replace(" ", "_").replace("'", "")
        local_config["OUTPUT_JSON"] = output_json

        # Recursive call for each term
        km_file_path = skim.km_with_gpt_workflow(local_config, output_directory)

        if not km_file_path:
            print("No km_file_path returned from km_with_gpt_workflow.")
            return

        df = read_tsv_to_dataframe(km_file_path)
        if df.empty:
            print("The dataframe is empty, skipping processing.")
            return

        df = df.iloc[: config["JOB_SPECIFIC_SETTINGS"]["km_with_gpt"]["NUM_B_TERMS"]]
        results = {}
        test_openai_connection(config)
        for index, row in df.iterrows():
            term = row["b_term"]
            result_dict = process_single_row(row, config)
            if term not in results:
                results[term] = [result_dict]
            else:
                results[term].append(result_dict)
            print(f"Processed row {index + 1} ({row['b_term']}) of {len(df)}")

        if not results:
            print("No results were processed")
            return

        write_to_json(results, config["OUTPUT_JSON"], output_directory)
        print(f"Analysis results have been saved to {config['OUTPUT_JSON']}")

        # if the prompt name ends in cc then we need to correct the file
        if config.get("PROMPT_NAME", "").endswith("cc"):
            print("Correcting the counts based off cc suffix...")
            json_file_path = os.path.join(output_directory, config["OUTPUT_JSON"])
            corrected_file_path = create_corrected_file_path(json_file_path)
            df.to_csv(corrected_file_path, sep="\t", index=False)
            print(f"Corrected file has been saved to {corrected_file_path}")


def position_km_with_gpt_workflow(config, output_directory):
    assert config["JOB_SPECIFIC_SETTINGS"]["position_km_with_gpt"][
        "A_TERMS_FILE"
    ], "A_TERMS_FILE is not defined in the configuration"
    a_terms = skim.read_terms_from_file(
        config["JOB_SPECIFIC_SETTINGS"]["position_km_with_gpt"]["A_TERMS_FILE"]
    )
    b_terms = skim.read_terms_from_file(
        config["JOB_SPECIFIC_SETTINGS"]["position_km_with_gpt"]["B_TERMS_FILE"]
    )
    # check if the a_terms and b_terms are the same length
    assert len(a_terms) == len(
        b_terms
    ), "The number of A terms and B terms do not match"
    results = {}
    config["OUTPUT_JSON"] = "position_km_with_gpt.json"
    for i, a_term in enumerate(a_terms):
        local_config = copy.deepcopy(config)
        local_config["GLOBAL_SETTINGS"]["A_TERM"] = a_term
        local_config["JOB_SPECIFIC_SETTINGS"]["km_with_gpt"]["NUM_B_TERMS"] = 1
        # save the b_term to a file and use that file in the km_with_gpt_workflow
        b_term = b_terms[i]
        b_term_file = os.path.join(output_directory, f"b_term_{i}.txt")
        with open(b_term_file, "w") as f:
            f.write(b_term)
        local_config["JOB_SPECIFIC_SETTINGS"]["km_with_gpt"][
            "B_TERMS_FILE"
        ] = b_term_file
        km_file_path = skim.km_with_gpt_workflow(local_config, output_directory)
        # if km_file_path is None, then the file was not created and we should skip to the next term
        if km_file_path is None:
            print(
                f"KM file not found for {a_term} and {b_term}. Please lower fet or check the spelling"
            )
            continue
        base, extension = os.path.splitext(km_file_path)
        new_file_name = f"{base}_{b_term}{extension}"
        os.rename(km_file_path, new_file_name)
        df = pd.read_csv(new_file_name, sep="\t")
        assert not df.empty, "The dataframe is empty"
        test_openai_connection(config)
        result_dict = process_single_row(df.iloc[0], local_config)
        if a_term in results:
            results[a_term].append(result_dict)
        else:
            results[a_term] = [result_dict]
        print(f"Processed row {i + 1} ({a_term}) of {len(a_terms)}")
        # remove the b_term file
        os.remove(b_term_file)
    assert results, "No results were processed"
    write_to_json(results, config["OUTPUT_JSON"], output_directory)
    print(f"Analysis results have been saved to {config['OUTPUT_JSON']}")


def skim_with_gpt_workflow(config, output_directory):
    # Read c_terms from file
    c_terms = skim.read_terms_from_file(
        config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]["C_TERMS_FILE"]
    )
    assert c_terms, "C terms are empty"
    b_terms = skim.read_terms_from_file(
        config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]["B_TERMS_FILE"]
    )
    if config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]["A_TERM_LIST"]:
        # Read a_terms from file
        a_terms = skim.read_terms_from_file(
            config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]["A_TERMS_FILE"]
        )
    else:
        a_terms = [config["GLOBAL_SETTINGS"]["A_TERM"]]
    for i, a_term in enumerate(a_terms):
        c_term = c_terms[i]
        b_term = b_terms[i]
        local_config = copy.deepcopy(config)
        local_config["GLOBAL_SETTINGS"]["A_TERM"] = a_term
        # local_config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]["skim"]["censor_year"] = determine_censor_year_exercise5(i)
        c_term_short = c_term[:10] if len(c_term) > 10 else c_term
        c_term_file = os.path.join(output_directory, f"{c_term_short}.txt")
        with open(c_term_file, "w") as f:
            f.write(c_term)
        local_config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"][
            "C_TERMS_FILE"
        ] = c_term_file

        b_term_short = b_term[:10] if len(b_term) > 10 else b_term
        b_term_file = os.path.join(output_directory, f"{b_term_short}.txt")
        with open(b_term_file, "w") as f:
            f.write(b_term)
        local_config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"][
            "B_TERMS_FILE"
        ] = b_term_file

        skim_file_path = skim.skim_with_gpt_workflow(local_config, output_directory)
        if skim_file_path is None:
            print(
                f"Skim file not found for {a_term}, {b_term} and {c_term}. Please lower fet or check the spelling"
            )
            continue
        df = read_tsv_to_dataframe(skim_file_path)
        # df = test_example_3(df, i, "/w5home/jfreeman/kmGPT/test/example3.csv")
        assert not df.empty, "The dataframe is empty"
        df = df.iloc[: config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]["NUM_B_TERMS"]]
        # Process the dataframe and gather results
        results_list = []
        test_openai_connection(config)
        for index, row in df.iterrows():
            result_dict = process_single_row(row, local_config)
            results_list.append(result_dict)
            print(f"Processed row {index + 1} ({row['b_term']}) of {len(df)}")
            assert results_list, "No results were processed"
            # Generate a unique output JSON file name for each a_term and c_term combination
            output_json = f"{a_term}_{b_term}_{c_term}_skim_with_gpt.json"
            write_to_json(results_list, output_json, output_directory)
            print(f"Analysis results have been saved to {output_json}")
            # delete the temporary file
            os.remove(c_term_file)
            os.remove(b_term_file)


def skim_with_gpt_workflow_old(config, output_directory):
    # Read c_terms from file
    c_terms = skim.read_terms_from_file(
        config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]["C_TERMS_FILE"]
    )
    assert c_terms, "C terms are empty"
    if config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]["A_TERM_LIST"]:
        # Read a_terms from file
        a_terms = skim.read_terms_from_file(
            config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]["A_TERMS_FILE"]
        )
    else:
        a_terms = [config["GLOBAL_SETTINGS"]["A_TERM"]]
    for a_term in a_terms:
        print(f"Processing {len(c_terms)} c_terms for {a_term}")
        for c_term in c_terms:
            local_config = copy.deepcopy(config)
            local_config["GLOBAL_SETTINGS"]["A_TERM"] = a_term
            # Create a temporary file for the current c_term
            c_term_file = os.path.join(output_directory, f"{c_term}.txt")
            with open(c_term_file, "w") as f:
                f.write(c_term)
            local_config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"][
                "C_TERMS_FILE"
            ] = c_term_file

            skim_file_path = skim.skim_run(local_config, output_directory)
            if skim_file_path is None:
                print(
                    f"Skim file not found for {a_term} and {c_term}. Please lower fet or check the spelling"
                )
                continue
            df = read_tsv_to_dataframe(skim_file_path)
            assert not df.empty, "The dataframe is empty"
            df = df.iloc[
                : config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]["NUM_B_TERMS"]
            ]
            # Process the dataframe and gather results
            results_list = []
            test_openai_connection(config)
            for index, row in df.iterrows():
                result_dict = process_single_row(row, local_config)
                results_list.append(result_dict)
                print(f"Processed row {index + 1} ({row['b_term']}) of {len(df)}")

            assert results_list, "No results were processed"

            # Generate a unique output JSON file name for each a_term and c_term combination
            output_json = f"{a_term}_{c_term}_skim_with_gpt.json"
            write_to_json(results_list, output_json, output_directory)
            print(f"Analysis results have been saved to {output_json}")
            # delete the temporary file
            os.remove(c_term_file)


# TODO add time


def main_workflow():
    parser = argparse.ArgumentParser("arg_parser")
    parser.add_argument(
        "-config",
        "--config_file",
        dest="config_file",
        help="Config file. Default=config.json.",
        default="../config.json",
        type=str,
    )
    args = parser.parse_args()
    GlobalClass.config_file = args.config_file
    config, output_directory = initialize_workflow()
    job_type = config.get("JOB_TYPE", "")

    if job_type == "km_with_gpt":
        if not api_cost_estimator([], config):
            return
        km_with_gpt_workflow(config, output_directory)
    elif job_type == "position_km_with_gpt":
        position_km_with_gpt_workflow(config, output_directory)
    elif job_type == "skim_with_gpt":
        skim_with_gpt_workflow(config, output_directory)
    else:
        print("JOB_TYPE does not match known workflows.")


if __name__ == "__main__":
    main_workflow()
