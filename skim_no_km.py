import os
import requests
import pandas as pd
import openai
import time
import json
import csv
from xml.etree import ElementTree

# Constants
RATE_LIMIT = 3
DELAY = 10
OUTPUT_JSON = "test_updated.json"
PORT = "5081"
API_URL = f"http://localhost:{PORT}/skim/api/jobs"
openai.api_key = os.getenv("OPENAI_API_KEY")
A_TERM = "Crohn's disease"


def read_terms_from_file(filename):
    with open(filename, "r") as f:
        terms = [line.strip() for line in f.readlines()]
    return terms


def configure_skim_job(a_term, filtered_terms):
    c_terms = read_terms_from_file(
        "FDA_approved_ProductsActiveIngredientOnly_DupsRemovedCleanedUp.txt"
    )
    b_terms = read_terms_from_file("BIO_PROCESS_cleaned.txt")

    # Remove terms from c_terms that are in filtered_terms
    c_terms = [term for term in c_terms if term not in filtered_terms]

    return {
        "a_terms": [a_term],
        "b_terms": b_terms,
        "c_terms": c_terms,
        "ab_fet_threshold": 1e-5,
        "bc_fet_threshold": 1e-5,
        "top_n_articles": 20,
        "return_pmids": True,
        "query_knowledge_graph": False,
        "censor_year": 1992,
    }


def configure_km_job(a_term):
    b_terms = read_terms_from_file(
        "FDA_approved_ProductsActiveIngredientOnly_DupsRemovedCleanedUp.txt"
    )

    return {
        "a_terms": [a_term],
        "b_terms": b_terms,
        "ab_fet_threshold": 0.05,  # FET threshold is 0.05
        "top_n_articles": 20,
        "return_pmids": True,
        "query_knowledge_graph": False,
        "censor_year": 1992,
    }


def configure_final_km(a_term, skim_file_path):
    # Read skim_file to extract unique c_terms
    df = pd.read_csv(skim_file_path, sep="\t")
    unique_c_terms = df["c_term"].unique().tolist()

    return {
        "a_terms": [a_term],
        "b_terms": unique_c_terms,
        "ab_fet_threshold": 0.05,
        "top_n_articles": 20,
        "return_pmids": True,
        "query_knowledge_graph": False,
        "censor_year": 2023,
    }


def run_skim_query(the_json: dict, url):
    response = requests.post(url, json=the_json, auth=("username", "password")).json()
    job_id = response["id"]
    response = requests.get(url + "?id=" + job_id, auth=("username", "password")).json()
    job_status = response["status"]

    while job_status == "queued" or job_status == "started":
        time.sleep(1)
        response = requests.get(
            url + "?id=" + job_id, auth=("username", "password")
        ).json()
        job_status = response["status"]

    if job_status == "finished":
        return response["result"]


def save_to_tsv(data, filename):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        keys = data[0].keys()
        writer.writerow(keys)
        for row in data:
            writer.writerow([row[k] for k in keys])


def read_file_path(file_path):
    return pd.read_csv(file_path, sep="\t")


def get_abstract_from_pubmed(pmid):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": pmid, "retmode": "xml", "rettype": "abstract"}

    response = requests.get(base_url, params=params)

    try:
        tree = ElementTree.fromstring(response.content)
    except ElementTree.ParseError:
        print(
            f"Error parsing XML for PMID {pmid}. Response content:\n{response.content.decode('utf-8')}"
        )
        return None, None

    # Extract abstract from the XML response
    abstract_text = ""
    for abstract in tree.findall(".//AbstractText"):
        abstract_text += abstract.text

    # Construct the URL for the paper
    paper_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

    return abstract_text, paper_url


def analyze_paper(consolidated_abstracts, b_term, abstracts_separate):
    if abstracts_separate:
        responses = []
        for abstract in consolidated_abstracts:
            prompt = f"Read the following abstract and classify the discussed treatment: {b_term} as useful, potentially useful, ineffective, potentially harmful, or harmful for successfully treating {A_TERM}. Provide at least two sentences explaining your classification. Your answer should be in the following format: 'Classification': 'Rationale': {abstract}"
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
                max_tokens=512,  # Increased max tokens to accommodate for the rationale
            )
            responses.append(response["choices"][0]["message"]["content"])
        return responses
    else:
        prompt = f"Read the following abstracts and classify the discussed treatment: {b_term} as useful, potentially useful, ineffective, potentially harmful, or harmful for successfully treating {A_TERM}. Provide at least two sentences explaining your classification. Your answer should be in the following format: 'Classification': 'Rationale' : {consolidated_abstracts}"
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


def process_row(row, abstracts_separate):
    # Extract PMIDs from the 'ab_pmid_intersection' column
    pmids = eval(row["ab_pmid_intersection"])

    # Split PMIDs into batches of 3
    pmid_batches = [pmids[i : i + RATE_LIMIT] for i in range(0, len(pmids), RATE_LIMIT)]

    # Retrieve abstracts and URLs for each PMID
    consolidated_abstracts = []
    paper_urls = []

    for batch in pmid_batches:
        for pmid in batch:
            abstract, url = get_abstract_from_pubmed(pmid)
            if abstract and url:
                consolidated_abstracts.append(abstract)
                paper_urls.append(url)
        time.sleep(DELAY)  # Introduce a delay between batches

    # If you don't want separate abstracts, join them into a single string
    if not abstracts_separate:
        consolidated_abstracts = " ".join(consolidated_abstracts)

    b_term = row["b_term"]
    result = analyze_paper(consolidated_abstracts, b_term, abstracts_separate)

    return result, paper_urls, consolidated_abstracts


def run_and_save_km_query(a_term):
    km_job_json = configure_km_job(a_term)
    result = run_skim_query(km_job_json, API_URL)
    km_b_terms = [entry["b_term"] for entry in result]

    return km_b_terms


def run_and_save_skim_query(a_term, filtered_terms):
    skim_job_json = configure_skim_job(a_term, filtered_terms)
    result = run_skim_query(skim_job_json, API_URL)
    file_path = f'gpt4_{skim_job_json["a_terms"][0]}_input.tsv'
    save_to_tsv(result, file_path)
    return file_path


def run_and_save_final_km_query(a_term, skim_file_path):
    # Configure the final KM job
    final_km_job = configure_final_km(a_term, skim_file_path)

    # Run the final KM query
    result = run_skim_query(final_km_job, API_URL)

    # Save the result to a TSV file
    file_path = f'gpt4_final_km_{final_km_job["a_terms"][0]}_input.tsv'
    save_to_tsv(result, file_path)

    return file_path


def main():
    # Step 1: Run initial KM Query and get b_terms
    km_b_terms = run_and_save_km_query(A_TERM)

    # Step 2: Run Skim Query and save to file
    skim_file_path = run_and_save_skim_query(A_TERM, km_b_terms)

    # Step 3: Run and save the final KM query
    final_km_file_path = run_and_save_final_km_query(A_TERM, skim_file_path)

    print(f"Final KM query results saved to {final_km_file_path}")

    df = read_file_path(final_km_file_path)

    results_list = []

    # Process each row in the dataframe
    for index, row in df.iterrows():
        result, paper_urls, abstracts = process_row(row, abstracts_separate=True)

        # Store the results in a dictionary
        result_dict = {
            "Term": row["b_term"],
            "Result": result,
            "URLs": paper_urls,
            "Abstracts": abstracts,
        }
        results_list.append(result_dict)

    # Write results to a JSON file
    with open(OUTPUT_JSON, "w") as outfile:
        json.dump(results_list, outfile, indent=4)


if __name__ == "__main__":
    main()
