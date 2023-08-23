import os
import pandas as pd
import requests
import time


# File Operations
def read_terms_from_file(filename):
    """Read terms from a given file and return them as a list."""
    with open(filename, "r") as f:
        terms = [line.strip() for line in f]
    return terms


def save_to_tsv(data, filename):
    """Save the data into a TSV (Tab Separated Values) file."""
    df = pd.DataFrame(data)
    df.to_csv(filename, sep="\t", index=False)


# API Calls
def post_api_request(url, payload, username, password):
    """Send a POST request to the API and return the JSON response."""
    response = requests.post(url, json=payload, auth=(username, password))
    response.raise_for_status()
    return response.json()


def get_api_request(url, job_id, username, password):
    """Send a GET request to the API and return the JSON response."""
    response = requests.get(f"{url}?id={job_id}", auth=(username, password))
    response.raise_for_status()
    return response.json()


def wait_for_job_completion(url, job_id, username, password):
    """Wait for an API job to complete and return the result."""
    while True:
        response_json = get_api_request(url, job_id, username, password)
        job_status = response_json["status"]
        if job_status in ["finished", "failed"]:
            break
        time.sleep(5)
    assert "result" in response_json, "Job did not complete successfully."
    return response_json.get("result", None)


def run_api_query(payload, url, username, password):
    """Initiate an API query and wait for its completion."""
    initial_response = post_api_request(url, payload, username, password)
    job_id = initial_response["id"]
    return wait_for_job_completion(url, job_id, username, password)


# Job Configuration
def configure_job(
    job_type, a_term, c_terms, b_terms=None, filtered_terms=None, config=None
):
    """Configure a job based on the provided type and terms."""
    assert config, "No configuration provided"

    common_settings = {
        "a_terms": [a_term],
        "return_pmids": True,
        "query_knowledge_graph": False,
    }
    job_specific_settings = config["JOB_SETTINGS"].get(job_type, {})

    if job_type == "skim":
        c_terms_filtered = list(set(c_terms) - set(filtered_terms))
        return {
            **common_settings,
            **job_specific_settings,
            "b_terms": b_terms,
            "c_terms": c_terms_filtered,
            "top_n_articles": 20,
        }
    elif job_type == "first_km":
        return {
            **common_settings,
            **job_specific_settings,
            "b_terms": c_terms,
            "top_n_articles": 20,
        }
    elif job_type == "final_km":
        return {
            **common_settings,
            **job_specific_settings,
            "b_terms": c_terms,
            "top_n_articles": 10,
        }
    else:
        raise ValueError(f"Invalid job type: {job_type}")


# Main Workflow
def run_and_save_query(
    job_type,
    a_term,
    c_terms,
    b_terms=None,
    filtered_terms=None,
    username="username",
    password="password",
    config=None,
):
    """Run a query, save the results, and return the saved file path."""
    job_config = configure_job(
        job_type, a_term, c_terms, b_terms, filtered_terms, config
    )
    result = run_api_query(job_config, config["API_URL"], username, password)
    file_path = f'{job_type}_{job_config["a_terms"][0]}_output.tsv'
    save_to_tsv(result, file_path)
    return file_path


def main_workflow(a_term, config=None):
    """Execute the main workflow for processing terms."""
    assert config, "No configuration provided"

    print("Reading terms from files...")
    c_terms = read_terms_from_file(config["C_TERMS_FILE"])
    b_terms = read_terms_from_file(config["B_TERMS_FILE"])

    print("Running and saving first KM query to subset...")
    km_file_path = run_and_save_query("first_km", a_term, c_terms, config=config)

    km_df = pd.read_csv(km_file_path, sep="\t")
    km_filtered_terms = km_df["b_term"].tolist()

    print("Running and saving SKIM query...")
    skim_file_path = run_and_save_query(
        "skim", a_term, c_terms, b_terms, km_filtered_terms, config=config
    )

    print("Processing SKIM results...")
    skim_df = pd.read_csv(skim_file_path, sep="\t")
    sort_column = config.get("SORT_COLUMN", "bc_sort_ratio")
    skim_df = skim_df.sort_values(by=sort_column, ascending=False)
    assert not skim_df.empty, "SKIM results are empty"

    unique_c_terms = skim_df["c_term"].unique().tolist()

    print("Running and saving final KM query...")
    final_km_file_path = run_and_save_query(
        "final_km", a_term, unique_c_terms, config=config
    )

    final_km_df = pd.read_csv(final_km_file_path, sep="\t")

    # Filter final_km_df to get rows where 'ab_pmid_intersection' is not empty
    # until number of rows equals NUM_C_TERMS
    valid_rows = final_km_df[final_km_df["ab_pmid_intersection"].notna()]
    final_km_df_filtered = valid_rows.head(config["NUM_C_TERMS"])

    # Save the filtered final_km_df to disk and return its path
    filtered_file_path = os.path.splitext(final_km_file_path)[0] + "_filtered.tsv"
    final_km_df_filtered.to_csv(filtered_file_path, sep="\t", index=False)

    print(f"Filtered final KM query results saved to {filtered_file_path}")
    return filtered_file_path
