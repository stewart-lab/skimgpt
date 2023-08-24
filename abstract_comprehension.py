import os
import requests
import pandas as pd
import openai
import time
import json
import csv
from xml.etree import ElementTree
import skim_no_km as skim


# Step 1: Constants and Configuration
def get_config():
    # Define the generation logic for OUTPUT_JSON
    a_term = "Crohn's disease"
    censor_year = 1992  # This is taken directly from the skim settings below
    num_c_terms = 3  # This is also defined below

    # Format the values into the desired output string.
    output_json = f"{a_term}_censorYear{censor_year}_numCTerms{num_c_terms}.json"
    # Replace spaces and special characters with underscores to create a valid filename.
    output_json = output_json.replace(" ", "_").replace("'", "")

    # Return the configuration dictionary
    return {
        "PORT": "5081",
        "API_URL": f"http://localhost:5081/skim/api/jobs",
        "API_KEY": os.getenv("OPENAI_API_KEY"),
        "RATE_LIMIT": 3,  # This is the number of PMIDs to process per API call
        "DELAY": 10,  # This is the number of seconds to wait between API calls
        "BASE_URL": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        "PUBMED_PARAMS": {"db": "pubmed", "retmode": "xml", "rettype": "abstract"},
        "A_TERM": a_term,
        "C_TERMS_FILE": "FDA_approved_ProductsActiveIngredientOnly_DupsRemovedCleanedUp.txt",  # This file contains FDA-approved drugs
        "B_TERMS_FILE": "BIO_PROCESS_cleaned.txt",  # This file contains biological processes
        "OUTPUT_JSON": output_json,
        "MAX_ABSTRACTS": 5,  # This is the maximum number of abstracts to process per final KM query
        "SORT_COLUMN": "bc_sort_ratio",  # This is the column to sort by in the SKIM query
        "NUM_C_TERMS": num_c_terms,
        "JOB_SETTINGS": {
            "skim": {
                "ab_fet_threshold": 1e-5,  # This is the p-value threshold for the Fisher's Exact Test
                "bc_fet_threshold": 1e-5,
                "censor_year": censor_year,  # This is the year to censor the data at
            },
            "first_km": {"ab_fet_threshold": 0.05, "censor_year": censor_year},
            "final_km": {"ab_fet_threshold": 1.1, "censor_year": 2023},
        },
    }


# Step 2: File Operations
def read_tsv_to_dataframe(file_path):
    return pd.read_csv(file_path, sep="\t")


def write_to_json(data, file_path):
    with open(file_path, "w") as outfile:
        json.dump(data, outfile, indent=4)


# Step 3: API Calls
def fetch_abstract_from_pubmed(pmid, base_url, params):
    response = requests.get(base_url, params={**params, "id": pmid})
    try:
        tree = ElementTree.fromstring(response.content)
    except ElementTree.ParseError:
        print(
            f"Error parsing XML for PMID {pmid}. Response content:\n{response.content.decode('utf-8')}"
        )
        return None, None
    abstract_text = ""
    for abstract in tree.findall(".//AbstractText"):
        abstract_text += abstract.text
    paper_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    return abstract_text, paper_url


def test_openai_connection(api_key):
    openai.api_key = api_key
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a medical research analyst."},
                {"role": "user", "content": "Test connection to OpenAI."},
            ],
        )
        print("Successfully connected to OpenAI!")
    except Exception as e:
        print(f"Failed to connect to OpenAI. Error: {e}")


def analyze_abstract_with_gpt4(
    consolidated_abstracts, b_term, a_term, abstracts_separate, api_key
):
    openai.api_key = api_key
    if abstracts_separate:
        responses = []
        for abstract in consolidated_abstracts:
            assert abstract and len(abstract) > 100, "Abstract is empty or too short"
            assert b_term, "B term is empty"
            assert a_term, "A term is empty"
            prompt = (
                f"Read the following abstract and classify the discussed treatment: {b_term} "
                f"as useful, potentially useful, ineffective, potentially harmful, or harmful for "
                f"successfully treating {a_term}. Provide at least two sentences explaining your "
                f"classification. Your answer should be in the following format: 'Classification': "
                f"'Rationale': {abstract}"
            )
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
    else:
        prompt = (
            f"Read the following abstracts and classify the discussed treatment: {b_term} as "
            f"useful, potentially useful, ineffective, potentially harmful, or harmful for successfully "
            f"treating {a_term}. Provide at least two sentences explaining your classification. Your answer "
            f"should be in the following format: 'Classification': 'Rationale' : {consolidated_abstracts}"
        )
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a medical research analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=256,
        )
        return response["choices"][0]["message"]["content"]


# Step 4: Data Manipulation and Analysis
def process_abstracts(pmids, rate_limit, delay, base_url, params):
    assert pmids, "List of PMIDs is empty"
    pmid_batches = [pmids[i : i + rate_limit] for i in range(0, len(pmids), rate_limit)]
    consolidated_abstracts = []
    paper_urls = []
    for batch in pmid_batches:
        for pmid in batch:
            abstract, url = fetch_abstract_from_pubmed(pmid, base_url, params)
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
        pmids,
        config["RATE_LIMIT"],
        config["DELAY"],
        config["BASE_URL"],
        config["PUBMED_PARAMS"],
    )
    b_term = row["b_term"]
    result = analyze_abstract_with_gpt4(
        consolidated_abstracts, b_term, config["A_TERM"], True, config["API_KEY"]
    )
    return {
        "Term": b_term,
        "Result": result,
        "URLs": paper_urls,
        "Abstracts": consolidated_abstracts,
    }


# Step 5: Main Workflow
def main_workflow():
    config = get_config()

    assert config, "Configuration is empty or invalid"
    a_term = config["A_TERM"]
    assert a_term, "A_TERM is not defined in the configuration"

    try:
        df = read_tsv_to_dataframe(skim.main_workflow(a_term, config))
        assert not df.empty, "The dataframe is empty"

        results_list = []
        test_openai_connection(config["API_KEY"])
        for index, row in df.iterrows():
            result_dict = process_single_row(row, config)
            results_list.append(result_dict)
            print(f"Processed row {index + 1} ({row['b_term']}) of {len(df)}")

        assert results_list, "No results were processed"
        write_to_json(results_list, config["OUTPUT_JSON"])
        print(f"Analysis results have been saved to {config['OUTPUT_JSON']}")

    except Exception as e:
        print(f"Error occurred during processing: {e}")


if __name__ == "__main__":
    main_workflow()
