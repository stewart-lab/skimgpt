import os
import requests
import pandas as pd
import openai
import time

file_path = "./skim_crohn_bioprocess_drugs_but_not_km_crohn_colitis_drugs0.05.txt"
df = pd.read_csv(file_path, sep="\t")

openai.api_key = os.getenv("OPENAI_API_KEY")


def find_papers(c_term, b_term, year=1992):
    papers = []
    offset = 0

    while len(papers) < 10:
        query = f"{c_term}+{b_term}"
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&offset={offset}&limit=10&fields=paperId,title,abstract,year"

        response = requests.get(url)

        if response.status_code == 429:
            print("Rate limit reached, sleeping for a moment...")
            time.sleep(10)
            response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            fetched_papers = data["data"]
            # Check that the 'year' field is not None before comparing it
            filtered_papers = [paper for paper in fetched_papers if paper['year'] is not None and paper['year'] <= year and paper['abstract'] is not None]

            papers.extend(filtered_papers)

            # If there are enough papers, break the loop
            if len(papers) >= 10:
                break

            # If no more papers are available to fetch, break the loop
            if len(fetched_papers) < 10:
                break

            # Increase the offset by 10 for the next iteration
            offset += 10
        else:
            print(f"Request failed with status code {response.status_code}")
            return None

    return papers[:10] # Return only the first 10 papers






def analyze_paper(consolidated_abstracts):
    prompt = f"Read the following abstracts and categorize the discussed treatment as valid, ineffective, or inconclusive. For each abstract, return your categorization and nothing else.: {consolidated_abstracts}"

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
    # Create a dictionary to hold the results
    results_dict = {}
    
    # Taking only the first 10 rows of the DataFrame
    first_10_rows = df.head(10)

    # Loop through the first 10 rows, processing each one
    for index, row in first_10_rows.iterrows():
        result = process_row(row)
        key = f"{row['B_term']} + {row['C_term']}" # Create the key by concatenating B_term and C_term
        results_dict[key] = result

    # Print the results in key-value pair format
    for key, value in results_dict.items():
        print(f"{key}: {value}")

