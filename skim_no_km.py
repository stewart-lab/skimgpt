import os
import pandas as pd
import requests
import time


# Step 1: Constants and Configuration
def get_config():
    return {
        "PORT": "5081",
        "API_URL": f"http://localhost:5081/skim/api/jobs",
        "A_TERM": "Crohn's disease",
        "C_TERMS_FILE": "FDA_approved_ProductsActiveIngredientOnly_DupsRemovedCleanedUp.txt",
        "B_TERMS_FILE": "BIO_PROCESS_cleaned.txt",
    }


config = get_config()


# Step 2: File Operations
def read_terms_from_file(filename):
    with open(filename, "r") as f:
        terms = [line.strip() for line in f]
    return terms


def save_to_tsv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, sep="\t", index=False)


# Step 3: API Calls
def post_api_request(url, payload, username, password):
    response = requests.post(url, json=payload, auth=(username, password))
    return response.json()


def get_api_request(url, job_id, username, password):
    response = requests.get(f"{url}?id={job_id}", auth=(username, password))
    return response.json()


def wait_for_job_completion(url, job_id, username, password):
    while True:
        response_json = get_api_request(url, job_id, username, password)
        job_status = response_json["status"]
        if job_status in ["finished", "failed"]:
            break
        time.sleep(5)
    return response_json.get("result", None)


def run_api_query(payload, url, username, password):
    initial_response = post_api_request(url, payload, username, password)
    job_id = initial_response["id"]
    return wait_for_job_completion(url, job_id, username, password)


# Step 4: Job Configuration
def configure_job(
    job_type, a_term, c_terms, b_terms=None, filtered_terms=None, config=None
):
    if not config:
        config = get_config()  # If no config passed, use the default configuration

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
            **job_specific_settings,  # Unpack the job-specific settings here
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
        raise ValueError("Invalid job type")


# Step 5: Main Workflow
def run_and_save_query(
    job_type,
    a_term,
    c_terms,
    b_terms=None,
    filtered_terms=None,
    username="username",
    password="password",
    config=None,  # Add the config parameter here
):
    job_config = configure_job(
        job_type, a_term, c_terms, b_terms, filtered_terms, config
    )
    result = run_api_query(
        job_config, config["API_URL"], username, password
    )  # Ensure config is used here
    file_path = f'{job_type}_{job_config["a_terms"][0]}_output.tsv'
    save_to_tsv(result, file_path)
    return file_path


def main_workflow(a_term, config=None):
    c_terms = read_terms_from_file(config["C_TERMS_FILE"])
    b_terms = read_terms_from_file(config["B_TERMS_FILE"])

    km_file_path = run_and_save_query(
        "first_km", a_term, c_terms, config=config
    )  # Pass the config here

    km_df = pd.read_csv(km_file_path, sep="\t")
    km_filtered_terms = km_df["b_term"].tolist()

    skim_file_path = run_and_save_query(
        "skim",
        a_term,
        c_terms,
        b_terms,
        km_filtered_terms,
        config=config,  # Pass the config here
    )

    skim_df = pd.read_csv(skim_file_path, sep="\t")
    unique_c_terms = skim_df["c_term"].unique().tolist()

    final_km_file_path = run_and_save_query(
        "final_km", a_term, unique_c_terms, config=config
    )  # Pass the config here

    print(f"Final KM query results saved to {final_km_file_path}")
    return final_km_file_path


if __name__ == "__main__":
    main_workflow(config["A_TERM"])
