import os
import requests
import pandas as pd
import openai
import time

file_path = "/isiseqruns/jfreeman_tmp_home/GPT/skim_crohn_bioprocess_drugs_but_not_km_crohn_colitis_drugs0.05.txt"
df = pd.read_csv(file_path, sep="\t")

openai.api_key = os.getenv("OPENAI_API_KEY")


def find_papers(c_term, b_term):
    query = f"{c_term}+{b_term}"
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=10&fields=paperId,title,abstract"

    response = requests.get(url)

    if response.status_code == 429:
        print("Rate limit reached, sleeping for a moment...")
        time.sleep(10)
        response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        papers = data["data"]
        return papers
    else:
        print(f"Request failed with status code {response.status_code}")
        return None



def analyze_paper(consolidated_abstracts):
    prompt = f"Read the following abstracts and categorize the discussed treatment as new, ineffective, inconclusive, or known as of 1992: {consolidated_abstracts}"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=256,
    )

    return response["choices"][0]["message"]["content"]



def process_row(row):
    c_term = row["C_term"]
    b_term = row["B_term"]

    papers = find_papers(c_term, b_term)

    if papers is None:
        print(f"Skipping {c_term}, {b_term} due to an error fetching papers")
        return []

    # Concatenate the abstracts together, numbering them for readability
    consolidated_abstracts = ""
    for i, paper in enumerate(papers):
        abstract = paper.get("abstract", "")
        if abstract:
            consolidated_abstracts += f"{i+1}. \"{abstract}\" - "
        else:
            print(f"Skipping paper due to missing abstract for {c_term}, {b_term}")

    # Pass the consolidated abstracts to the analyze_paper function
    result = analyze_paper(consolidated_abstracts)

    return result




if __name__ == "__main__":
    # Taking only the first row of the DataFrame for debugging
    first_row = df.head(1)

    # Processing the row using your existing code
    results = process_row(first_row.iloc[0])

    # You can print the results here or do further debugging as needed
    print(results)

