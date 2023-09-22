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
import skim_and_km_api as skim
import prompt_and_scoring_library as prompts
import test.test_abstract_comprehension as test


# Step 1: Constants and Configuration
def get_config(output_directory):
    with open("config.json", "r") as f:
        config = json.load(f)

    # Dynamically build the OUTPUT_JSON field
    job_settings = config["JOB_SPECIFIC_SETTINGS"].get(config["JOB_TYPE"], {})
    a_term = job_settings.get("A_TERM", "")

    # Adjusting the way we access censor_year based on the job type
    if config["JOB_TYPE"] == "marker_list":
        output_json = (
            f"{a_term}_numCTerms{config['GLOBAL_SETTINGS'].get('NUM_C_TERMS', '')}.json"
        )
    else:
        censor_year = job_settings.get("skim", {}).get("censor_year", "")
        num_c_terms = config["GLOBAL_SETTINGS"].get("NUM_C_TERMS", "")
        output_json = f"{a_term}_censorYear{censor_year}_numCTerms{num_c_terms}.json"

    output_json = output_json.replace(" ", "_").replace("'", "")

    # Update the OUTPUT_JSON in the config dictionary
    config["OUTPUT_JSON"] = output_json

    # Do not write the API key back to the config file
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    # Write the modified configuration back to the output-specific config.json file
    with open(os.path.join(output_directory, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # Add the API key to the config dictionary in memory (not in the file)
    config["API_KEY"] = api_key

    return config


# Step 2: File Operations
def read_tsv_to_dataframe(file_path):
    return pd.read_csv(file_path, sep="\t")


def write_to_json(data, file_path, output_directory):
    full_path = os.path.join(output_directory, file_path)
    with open(full_path, "w") as outfile:
        json.dump(data, outfile, indent=4)


# Step 3: API Calls
def fetch_abstract_from_pubmed(config, pmid):
    global_settings = config["GLOBAL_SETTINGS"]
    for attempt in range(global_settings["MAX_RETRIES"]):
        try:
            response = requests.get(
                global_settings["BASE_URL"],
                params={**global_settings["PUBMED_PARAMS"], "id": pmid},
            )
            tree = ElementTree.fromstring(response.content)
            break  # If successful, break out of the loop
        except (ElementTree.ParseError, requests.exceptions.RequestException) as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            if attempt < config["MAX_RETRIES"] - 1:
                print(f"Retrying in {config['RETRY_DELAY']} seconds...")
                time.sleep(config["RETRY_DELAY"])
            else:
                print("Max retries reached. Skipping this PMID.")
                return None, None, None

    abstract_text = ""
    for abstract in tree.findall(".//AbstractText"):
        abstract_text += abstract.text

    # Extract the year
    year = None
    pub_date = tree.find(".//PubDate")
    if pub_date is not None:
        year_element = pub_date.find("Year")
        if year_element is not None:
            year = year_element.text

    paper_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    return abstract_text, paper_url, year


def analyze_abstract_with_gpt4(consolidated_abstracts, b_term, a_term, api_key, config):
    openai.api_key = api_key
    responses = []

    for abstract in consolidated_abstracts:
        assert abstract and len(abstract) > 100, "Abstract is empty or too short"
        assert b_term, "B term is empty"
        assert a_term, "A term is empty"

        # Accessing JOB_TYPE from the top level of the config
        if config["JOB_TYPE"] == "drug_discovery_validation":
            prompt = prompts.drug_process_relationship_classification_prompt(
                b_term, a_term, abstract
            )

        # Accessing MAX_RETRIES and RETRY_DELAY from the GLOBAL_SETTINGS
        for attempt in range(config["GLOBAL_SETTINGS"]["MAX_RETRIES"]):
            try:
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
                break  # Successful request, break the retry loop
            except openai.Error as e:
                if "502" in str(e):  # Bad Gateway
                    print(
                        f"Bad Gateway error, attempt {attempt + 1}. Retrying in {config['GLOBAL_SETTINGS']['RETRY_DELAY']} seconds..."
                    )
                    time.sleep(config["GLOBAL_SETTINGS"]["RETRY_DELAY"])
                else:
                    print(f"An unexpected error occurred: {e}")
                    break  # For other errors, you might want to break the retry loop
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                break  # For non-OpenAI errors, you might want to break the retry loop

    return responses


# Step 4: Data Manipulation and Analysis
def process_abstracts(config, pmids, rate_limit, delay):
    assert pmids, "List of PMIDs is empty"
    pmid_batches = [pmids[i : i + rate_limit] for i in range(0, len(pmids), rate_limit)]
    consolidated_abstracts = []
    paper_urls = []
    publication_years = []  # List to store the years of publication

    for batch in pmid_batches:
        for pmid in batch:
            abstract, url, year = fetch_abstract_from_pubmed(
                config, pmid
            )  # Updated call
            if abstract and url:
                consolidated_abstracts.append(abstract)
                paper_urls.append(url)
                publication_years.append(year)  # Append the year to the list
        time.sleep(delay)

    return (
        consolidated_abstracts,
        paper_urls,
        publication_years,
    )  # Return the years as well


def process_single_row(row, config):
    pmids = eval(row["ab_pmid_intersection"])
    assert pmids, "No PMIDs found in the row"

    # Accessing MAX_ABSTRACTS from the GLOBAL_SETTINGS
    pmids = pmids[: config["GLOBAL_SETTINGS"]["MAX_ABSTRACTS"]]

    # Capture the years as well
    consolidated_abstracts, paper_urls, publication_years = process_abstracts(
        config,
        pmids,
        config["GLOBAL_SETTINGS"]["RATE_LIMIT"],
        config["GLOBAL_SETTINGS"]["DELAY"],
    )

    b_term = row["b_term"]

    # Accessing A_TERM and API_KEY from the appropriate sections of the config
    result = analyze_abstract_with_gpt4(
        consolidated_abstracts,
        b_term,
        config["JOB_SPECIFIC_SETTINGS"]["drug_discovery_validation"]["A_TERM"],
        config["API_KEY"],
        config,
    )

    # Include the years in the returned dictionary
    return {
        "Term": b_term,
        "Result": result,
        "URLs": paper_urls,
        "Abstracts": consolidated_abstracts,
        "Years": publication_years,  # Add the years here
    }


def process_json(json_data, config):
    nested_dict = {}
    a_term = config["JOB_SPECIFIC_SETTINGS"][config["JOB_TYPE"]]["A_TERM"]
    nested_dict[a_term] = {}

    try:
        term_scores = {}
        term_counts = {}
        term_details = {}  # New dictionary to hold the details

        for entry in json_data:
            term = entry.get("Term", None)
            results = entry.get("Result", None)
            urls = entry.get("URLs", None)  # Get URLs
            years = entry.get("Years", None)  # Get Years

            if term is None or results is None or urls is None or years is None:
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
                if config["JOB_TYPE"] == "drug_discovery_validation":
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

                # Get the year for this PMID
                year = years[i] if i < len(years) else "Unknown"

                # Record the details
                term_details[term].append(
                    {
                        "PMID": pmid,
                        "Year": year,
                        "Scoring Sentence": scoring_sentence,
                        "Score": score,
                    }
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

def marker_list_filtration(km_file_paths, output_directory):
    try:
        print(f"KM file paths: {km_file_paths}")
        
        dfs = {}
        max_ratios = {}
        
        # Read all DataFrames and find the maximum ab_sort_ratio for each b_term
        for file_name in km_file_paths:
            file_path = os.path.join(output_directory, file_name)
            # Read the file into a DataFrame
            df = pd.read_csv(file_path, sep='\t')
            
            # Sort the DataFrame by the 'ab_sort_ratio' column in descending order
            df_sorted = df.sort_values(by='ab_sort_ratio', ascending=False)
            
            # Modify the file name to add "_sorted" before the file extension
            base, ext = os.path.splitext(file_path)
            sorted_file_path = f"{base}_sorted{ext}"
            
            # Store the sorted DataFrame in the dictionary
            dfs[sorted_file_path] = df_sorted
            
            # Update the maximum ab_sort_ratio for each b_term
            for index, row in df_sorted.iterrows():
                b_term = row['b_term']
                ab_sort_ratio = row['ab_sort_ratio']
                if b_term not in max_ratios or ab_sort_ratio > max_ratios[b_term]:
                    max_ratios[b_term] = ab_sort_ratio
        
        # Filter rows in each DataFrame and save the filtered DataFrames
        for sorted_file_path, df in dfs.items():
            df_filtered = df[df.apply(lambda row: row['ab_sort_ratio'] == max_ratios[row['b_term']], axis=1)]
            df_filtered.to_csv(sorted_file_path, sep='\t', index=False)

    except Exception as e:
        print(f"An error occurred: {e}")
# Step 5: Main Workflow
def main_workflow():
    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # Create a new directory with the timestamp
    output_directory = f"output_{timestamp}"
    os.makedirs(output_directory, exist_ok=True)
    shutil.copy("config.json", os.path.join(output_directory, "config.json"))
    config = get_config(output_directory)

    assert config, "Configuration is empty or invalid"
    job_type = config.get("JOB_TYPE", "")
    if job_type == "drug_discovery_validation":
        a_term = config["JOB_SPECIFIC_SETTINGS"]["drug_discovery_validation"]["A_TERM"]
        assert a_term, "A_TERM is not defined in the configuration"
        try:
            df = read_tsv_to_dataframe(
                skim.skim_no_km_workflow(config, output_directory)
            )

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
            json_file_path = os.path.join(output_directory, config["OUTPUT_JSON"])
            with open(json_file_path, "r") as f:
                json_data = json.load(f)
            result = process_json(json_data, config)
            if result:
                save_to_json(result, config, output_directory)

        except Exception as e:
            print(f"Error occurred during processing: {e}")
    elif job_type == "marker_list":
            km_file_paths = skim.marker_list_workflow(
                config=config, output_directory=output_directory
            )
            marker_list_filtration(km_file_paths = km_file_paths, output_directory = output_directory)
        
    else:
        print("JOB_TYPE does not match known workflows.")


if __name__ == "__main__":
    main_workflow()
