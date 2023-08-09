import os
import requests
import pandas as pd
import openai
import time

# Load your DataFrame here
file_path = "/isiseqruns/jfreeman_tmp_home/GPT/skim_crohn_bioprocess_drugs_but_not_km_crohn_colitis_drugs0.05.txt"
df = pd.read_csv(file_path, sep="\t")

openai.api_key = os.getenv("OPENAI_API_KEY")


def find_papers(c_term, b_term):
    # Construct the query
    query = f"{c_term}+{b_term}"

    # Construct the API endpoint
    url = (
        f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=100"
    )

    # Make the API call
    response = requests.get(url)

    if response.status_code == 429:
        print("Rate limit reached, sleeping for a moment...")
        time.sleep(10)  # Adjust the sleep time as needed
        response = requests.get(url)

    # Check the response status code to make sure the request was successful
    elif response.status_code == 200:
        # If the request was successful, parse the JSON response
        data = response.json()

        # You can access the 'data' field in the JSON response for the list of papers
        papers = data["data"]
        return papers
    else:
        print(f"Request failed with status code {response.status_code}")
        return None


def analyze_paper(abstract):
    # Construct the prompt
    prompt = f"Read the following abstract and categorize the discussed treatment as new, ineffective, inconclusive, or known as of 1992: {abstract}"

    # Make the API call
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        max_tokens=256,
    )

    # The response from the model will be in response['choices'][0]['message']['content']
    return response


# Create a list to hold the analyses
analyses = []

# Iterate over the rows of your DataFrame
for index, row in df.iterrows():
    c_term = row["C_term"]
    b_term = row["B_term"]

    papers = find_papers(c_term, b_term)

    if papers is None:
        print(f"Skipping {c_term}, {b_term} due to an error fetching papers")
        continue

    # Each dictionary contains information about a paper
    for paper in papers:
        abstract = paper.get("abstract", "")
        response = analyze_paper(abstract)

        # Assuming the desired information is directly in the content
        # You may need to further process it if needed
        analysis = response["choices"][0]["message"]["content"]

        # Append the analysis to the list
        analyses.append(analysis)

# Combine the analyses into a single string
combined_analysis = "\n".join(analyses)

# Write to a text file
output_file_path = "analysis_output.txt"
with open(output_file_path, "w") as file:
    file.write(combined_analysis)

print(f"Analysis written to {output_file_path}")
