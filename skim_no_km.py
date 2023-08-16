import os
import requests
import pandas as pd
import openai
import time
import json
from xml.etree import ElementTree

RATE_LIMIT = 3
DELAY = 10
OUTPUT_JSON = "results_old.json"

FILE_PATH = "./km_results_SaffSoyRanit_1992.txt"
openai.api_key = os.getenv("OPENAI_API_KEY")
A_TERM = "Crohn's disease"


def read_file_path():
    return pd.read_csv(FILE_PATH, sep="\t")


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
            prompt = f'Read the following abstract and categorize the discussed treatment {b_term} as useful, harmful, ineffective, or inconclusive for successfully treating {A_TERM}. Provide at least two sentences explaining your categorization: "{abstract}"'
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
        prompt = f"Read the following abstracts and categorize the discussed treatment {b_term} as useful, harmful, ineffective, or inconclusive for successfully treating {A_TERM}. Provide at least two sentences explaining your categorization: {consolidated_abstracts}"
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


if __name__ == "__main__":
    # Read the input file
    df = read_file_path()

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

        # Print the results
        print(f"\nTerm: {row['b_term']}")
        print(f"Result: {result}")
        print(f"URLs: {paper_urls}")
        print(f"Abstracts: {abstracts}")
        print("-" * 50)

    # Write results to a JSON file
    with open(OUTPUT_JSON, "w") as outfile:
        json.dump(results_list, outfile, indent=4)
