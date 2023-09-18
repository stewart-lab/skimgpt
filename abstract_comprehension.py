import os
import requests
import pandas as pd
import openai
import time
import json
import shutil
import csv
import re
from datetime import datetime
from xml.etree import ElementTree
import skim_no_km as skim
import prompt_and_scoring_library as prompts
import test.test_abstract_comprehension as test


# Step 1: Constants and Configuration
def get_config():
    with open("config.json", "r") as f:
        config = json.load(f)

    # Dynamically build the OUTPUT_JSON field
    a_term = config.get("A_TERM", "")
    censor_year = config.get("JOB_SETTINGS", {}).get("skim", {}).get("censor_year", "")
    num_c_terms = config.get("NUM_C_TERMS", "")

    output_json = f"{a_term}_censorYear{censor_year}_numCTerms{num_c_terms}.json"
    output_json = output_json.replace(" ", "_").replace("'", "")

    config["OUTPUT_JSON"] = output_json
    config["API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    if not config["API_KEY"]:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    return config


# Step 2: File Operations
def read_tsv_to_dataframe(file_path):
    return pd.read_csv(file_path, sep="\t")


def write_to_json(data, file_path, output_directory):
    full_path = os.path.join(output_directory, file_path)
    with open(full_path, "w") as outfile:
        json.dump(data, outfile, indent=4)


# Step 3: API Calls
def fetch_abstract_from_pubmed(config, pmid, base_url, params):
    for attempt in range(config["MAX_RETRIES"]):
        try:
            response = requests.get(base_url, params={**params, "id": pmid})
            tree = ElementTree.fromstring(response.content)
            break  # If successful, break out of the loop
        except (ElementTree.ParseError, requests.exceptions.RequestException) as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            if attempt < config["MAX_RETRIES"] - 1:
                print(f"Retrying in {config['RETRY_DELAY']} seconds...")
                time.sleep(config["RETRY_DELAY"])
            else:
                print("Max retries reached. Skipping this PMID.")
                return None, None

    abstract_text = ""
    for abstract in tree.findall(".//AbstractText"):
        abstract_text += abstract.text
    paper_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    return abstract_text, paper_url


def analyze_abstract_with_gpt4(consolidated_abstracts, b_term, a_term, api_key):
    openai.api_key = api_key
    responses = []
    for abstract in consolidated_abstracts:
        assert abstract and len(abstract) > 100, "Abstract is empty or too short"
        assert b_term, "B term is empty"
        assert a_term, "A term is empty"
        prompt = prompts.drug_process_relationship_classification_prompt(b_term, a_term)
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical research analyst.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=512,
        )
        assert response["choices"][0]["message"]["content"], "Response is empty"
        responses.append(response["choices"][0]["message"]["content"])
    return responses


# Step 4: Data Manipulation and Analysis
def process_abstracts(config, pmids, rate_limit, delay, base_url, params):
    assert pmids, "List of PMIDs is empty"
    pmid_batches = [pmids[i : i + rate_limit] for i in range(0, len(pmids), rate_limit)]
    consolidated_abstracts = []
    paper_urls = []
    for batch in pmid_batches:
        for pmid in batch:
            abstract, url = fetch_abstract_from_pubmed(config, pmid, base_url, params)
            if abstract and url:
                consolidated_abstracts.append(abstract)
                paper_urls.append(url)
        time.sleep(delay)
    return consolidated_abstracts, paper_urls


def process_single_row(row, config):
    pmids = eval(row["ab_pmid_intersection"])
    assert pmids, "No PMIDs found in the row"
    pmids = pmids[: config["MAX_ABSTRACTS"]]

    consolidated_abstracts, paper_urls = process_abstracts(
        config,
        pmids,
        config["RATE_LIMIT"],
        config["DELAY"],
        config["BASE_URL"],
        config["PUBMED_PARAMS"],
    )
    b_term = row["b_term"]
    result = analyze_abstract_with_gpt4(
        consolidated_abstracts, b_term, config["A_TERM"], config["API_KEY"]
    )
    return {
        "Term": b_term,
        "Result": result,
        "URLs": paper_urls,
        "Abstracts": consolidated_abstracts,
    }


def process_json(json_data, config):
    nested_dict = {}
    a_term = config["A_TERM"]
    nested_dict[a_term] = {}

    try:
        term_scores = {}
        term_counts = {}
        term_details = {}  # New dictionary to hold the details

        for entry in json_data:
            term = entry.get("Term", None)
            results = entry.get("Result", None)
            urls = entry.get("URLs", None)  # Get URLs

            if term is None or results is None or urls is None:
                print("Warning: Invalid entry found in JSON data. Skipping.")
                continue

            if term not in term_scores:
                term_scores[term] = 0
                term_counts[term] = 0
                term_details[term] = []  # Initialize list to hold details

            for i, result in enumerate(results):
                term_counts[term] += 1  # Increase the count for this term
                score = 0
                scoring_sentence = ""
                patterns = prompts.drug_process_relationship_scoring(term)

                # Search for the pattern in the result to find the scoring sentence and score
                for pattern, pattern_score in patterns:
                    if pattern in result:
                        score = pattern_score
                        scoring_sentence = re.search(
                            f"([^\.]*{pattern}[^\.]*\.)", result
                        ).group(1)
                        break

                term_scores[term] += score

                # Extract PMID from URL
                url = urls[i]
                pmid_match = re.search(r"(\d+)/?$", url)
                pmid = pmid_match.group(1) if pmid_match else "Unknown"

                # Record the details
                term_details[term].append(
                    {"PMID": pmid, "Scoring Sentence": scoring_sentence, "Score": score}
                )

        term_avg_scores = {
            term: (term_scores[term] / term_counts[term]) for term in term_scores
        }

        for term, score in term_scores.items():
            avg_score = term_avg_scores[term]
            nested_dict[a_term][term] = {
                "Total Score": score,
                "Average Score": avg_score,
                "Count": term_counts[term],
                "Details": term_details[term],  # Add the details
            }

        return nested_dict

    except Exception as e:
        print(f"An error occurred while processing the JSON data: {e}")
        return None  # Returning None to indicate failure


def save_to_json(data, config, output_directory):
    output_filename = os.path.join(
        output_directory, config["OUTPUT_JSON"] + "_filtered.json"
    )
    # Adding "_filtered" to the filename
    with open(output_filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Filtered results have been saved to {output_filename}")


# Step 5: Main Workflow
def main_workflow():
    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # Create a new directory with the timestamp
    output_directory = f"output_{timestamp}"
    os.makedirs(output_directory, exist_ok=True)
    shutil.copy("config.json", os.path.join(output_directory, "config.json"))
    config = get_config()

    assert config, "Configuration is empty or invalid"
    a_term = config["A_TERM"]
    assert a_term, "A_TERM is not defined in the configuration"

    try:
        df = read_tsv_to_dataframe(skim.main_workflow(a_term, config, output_directory))
        assert not df.empty, "The dataframe is empty"

        # Calculate the total number of abstracts
        x = df["ab_count"].sum()

        # Calculate and display the estimated cost
        estimated_cost = x * 0.006
        user_input = input(
            f"The following job consists of {x} abstracts and will cost roughly ${estimated_cost} in GPT-4 API calls. Do you wish to proceed? [Y/n]: "
        )

        if user_input.lower() != "y":
            print("Exiting workflow.")
            return

        results_list = []
        test.test_openai_connection(config["API_KEY"])
        for index, row in df.iterrows():
            result_dict = process_single_row(row, config)
            results_list.append(result_dict)
            print(f"Processed row {index + 1} ({row['b_term']}) of {len(df)}")

        assert results_list, "No results were processed"
        write_to_json(results_list, config["OUTPUT_JSON"], output_directory)
        print(f"Analysis results have been saved to {config['OUTPUT_JSON']}")
        json_file_path = os.path.join(
            output_directory, config["OUTPUT_JSON"]
        )  # Make sure to use the full path
        with open(json_file_path, "r") as f:  # Use the full path here
            json_data = json.load(f)
        result = process_json(json_data, config)
        if result:
            save_to_json(result, config, output_directory)

    except Exception as e:
        print(f"Error occurred during processing: {e}")


if __name__ == "__main__":
    main_workflow()
