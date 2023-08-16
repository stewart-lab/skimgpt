import os
import requests
import pandas as pd
import openai
import time
import json

FILE_PATH = "./skim_crohn_bioprocess_drugs_but_not_km_crohn_colitis_drugs0.05.txt"
openai.api_key = os.getenv("OPENAI_API_KEY")
A_TERM = "Crohn's disease"


def read_file_path():
    return pd.read_csv(FILE_PATH, sep="\t")


def find_paper_given_pmid(pmid):
    base_url = "https://api.semanticscholar.org/v1/paper/"
    response = requests.get(base_url + pmid)
    if response.status_code != 200:
        print(
            f"Error: Unable to fetch data for PMID {pmid}. Status code: {response.status_code}"
        )
        return None
    data = response.json()
    return {
        "abstract": data.get("abstract", None),
        "year": data.get("year", None),
        "title": data.get("title", None),
        "url": data.get("url", None),
        "isOpenAccess": data.get("isOpenAccess", None),
    }


def find_papers_given_terms(c_term, year=1992, num_papers=10):
    papers = []
    offset = 0
    while len(papers) < num_papers:
        query = f"{A_TERM}+{c_term}"
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&offset={offset}&limit=10&fields=paperId,title,url,abstract,year"
        response = requests.get(url)
        if response.status_code == 429:
            print("Rate limit reached, sleeping for a moment...")
            time.sleep(60)
            continue
        if response.status_code == 200:
            data = response.json()
            fetched_papers = data["data"]
            filtered_papers = [
                paper
                for paper in fetched_papers
                if paper["year"] is not None
                and paper["year"] <= year
                and paper["abstract"] is not None
            ]
            papers.extend(filtered_papers)
            if len(papers) >= num_papers or len(fetched_papers) < 10:
                break
            offset += 10
        else:
            print(f"Request failed with status code {response.status_code}")
            return [], [], []
    return papers[:num_papers]


def analyze_paper(consolidated_abstracts, c_term, abstracts_separate):
    if abstracts_separate:
        responses = []
        for abstract in consolidated_abstracts:
            prompt = f'Read the following abstract and categorize the discussed treatment {c_term} as useful, harmful, ineffective, or inconclusive for successfully treating {A_TERM}. Return your categorization and nothing else.: "{abstract}"'
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
                max_tokens=256,
            )
            responses.append(response["choices"][0]["message"]["content"])
        return responses
    else:
        prompt = f"Read the following abstracts and categorize the discussed treatment {c_term} as useful, harmful, ineffective, or inconclusive for successfully treating {A_TERM}. For each abstract, return your categorization and nothing else.: {consolidated_abstracts}"
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


def process_row(c_term, abstracts_separate):
    papers = find_papers_given_terms(c_term)
    if not papers:
        print(f"Skipping {c_term} due to an error fetching papers")
        return None, [], []
    consolidated_abstracts = [] if abstracts_separate else ""
    paper_urls = []
    individual_abstracts = []
    for i, paper in enumerate(papers):
        abstract = paper.get("abstract", "")
        if abstract:
            if abstracts_separate:
                consolidated_abstracts.append(abstract)
            else:
                consolidated_abstracts += f'{i+1}. "{abstract}" - '
            paper_urls.append(paper.get("url", "N/A"))
            individual_abstracts.append(abstract)
        else:
            print(f"Skipping paper due to missing abstract for {c_term}")
    result = analyze_paper(consolidated_abstracts, c_term, abstracts_separate)
    return result, paper_urls, individual_abstracts


if __name__ == "__main__":
    df = read_file_path()
    results_dict = {}
    unique_c_terms = df["C_term"].unique()[:1]
    for c_term in unique_c_terms:
        result, urls, abstracts = process_row(c_term, abstracts_separate=False)
        results_dict[c_term] = {"result": result, "urls": urls, "abstracts": abstracts}
    with open("results2.json", "w") as json_file:
        json.dump(results_dict, json_file, indent=4)
