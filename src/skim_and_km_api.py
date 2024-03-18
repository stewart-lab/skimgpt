import os
import pandas as pd
import requests
import time


# File Operations
def read_terms_from_file(filename):
    """Read terms from a given file and return them as a list."""
    with open(filename, "r") as f:
        terms = [line.strip() for line in f]
        # Remove empty strings from the list
        terms = list(filter(None, terms))
    return terms


def save_to_tsv(data, filename, output_directory):
    """Save the data into a TSV (Tab Separated Values) file."""
    full_path = os.path.join(output_directory, filename)
    df = pd.DataFrame(data)
    df.to_csv(full_path, sep="\t", index=False)


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

    # Adjust how we retrieve the job-specific settings based on the job type
    if (
        job_type in ["skim", "first_km", "final_km"]
        and config["JOB_TYPE"] == "drug_discovery_validation"
    ):
        job_specific_settings = config["JOB_SPECIFIC_SETTINGS"][
            "drug_discovery_validation"
        ][job_type]
    elif job_type == "km_with_gpt":
        job_specific_settings = config["JOB_SPECIFIC_SETTINGS"]["km_with_gpt"][
            "km_with_gpt"
        ]
    elif job_type == "position_km_with_gpt":
        job_specific_settings = config["JOB_SPECIFIC_SETTINGS"]["position_km_with_gpt"][
            "position_km_with_gpt"
        ]
    elif job_type == "skim_with_gpt":
        job_specific_settings = config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]["skim"]
    else:
        job_specific_settings = config["JOB_SPECIFIC_SETTINGS"][job_type]

    if job_type == "skim" or job_type == "skim_with_gpt":
        c_terms_filtered = c_terms
        if config["JOB_TYPE"] == "km_skim_no_km":
            c_terms_filtered = list(set(c_terms) - set(filtered_terms))
        return {
            **common_settings,
            **job_specific_settings,
            "b_terms": b_terms,
            "c_terms": c_terms_filtered,
            "top_n_articles": 10,
        }
    elif job_type in [
        "first_km",
        "final_km",
        "km_with_gpt",
        "position_km_with_gpt",
    ]:
        return {
            **common_settings,
            **job_specific_settings,
            "b_terms": c_terms,
            "top_n_articles": 50,
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
    output_directory=None,
):
    """Run a query, save the results, and return the saved file path."""
    job_config = configure_job(
        job_type, a_term, c_terms, b_terms, filtered_terms, config
    )
    # print(job_config)
    # Accessing API_URL from the updated config structure
    api_url = config["GLOBAL_SETTINGS"].get("API_URL", "")
    assert api_url, "'API_URL' is not defined in the configuration"

    result = run_api_query(job_config, api_url, username, password)
    if not result or result is None:
        print("The result is empty")
        return None
    file_name = job_config["a_terms"][0]
    if job_type == "skim_with_gpt":
        file_name = job_config["a_terms"][0] + "_" + job_config["c_terms"][0]
    # print(result)
    file_path = f"{job_type}_{file_name}_output.tsv"
    save_to_tsv(result, file_path, output_directory)  # Pass output_directory
    return file_path


def km_skim_no_km_workflow(config=None, output_directory=None):
    """Execute the workflow for drug discovery validation without KM."""
    assert config, "No configuration provided"

    a_term = config["GLOBAL_SETTINGS"]["A_TERM"]
    assert a_term, "A_TERM is not defined in the configuration"

    print("Executing drug discovery validation workflow...")
    print("Reading terms from files...")
    c_terms_file = config["JOB_SPECIFIC_SETTINGS"]["drug_discovery_validation"][
        "C_TERMS_FILE"
    ]
    b_terms_file = config["JOB_SPECIFIC_SETTINGS"]["drug_discovery_validation"][
        "B_TERMS_FILE"
    ]

    c_terms = read_terms_from_file(c_terms_file)
    b_terms = read_terms_from_file(b_terms_file)

    print("Running and saving first KM query to subset...")
    km_file_path = run_and_save_query(
        "first_km",
        a_term,
        c_terms,
        config=config,
        output_directory=output_directory,
    )
    full_km_file_path = os.path.join(output_directory, km_file_path)
    km_df = pd.read_csv(full_km_file_path, sep="\t")
    km_filtered_terms = km_df["b_term"].tolist()

    print("Running and saving SKIM query...")
    skim_file_path = run_and_save_query(
        "skim",
        a_term,
        c_terms,
        b_terms,
        km_filtered_terms,
        config=config,
        output_directory=output_directory,
    )

    print("Processing SKIM results...")
    full_skim_file_path = os.path.join(output_directory, skim_file_path)
    skim_df = pd.read_csv(full_skim_file_path, sep="\t")
    sort_column = config["JOB_SPECIFIC_SETTINGS"]["drug_discovery_validation"].get(
        "SORT_COLUMN", "bc_sort_ratio"
    )
    skim_df = skim_df.sort_values(by=sort_column, ascending=False)
    # assert the sort_column is in the skim_df
    assert sort_column in skim_df.columns, f"{sort_column} is not in the skim_df"
    assert not skim_df.empty, "SKIM results are empty"

    unique_c_terms = skim_df["c_term"].unique().tolist()

    print("Running and saving final KM query...")
    final_km_file_path = run_and_save_query(
        "final_km",
        a_term,
        unique_c_terms,
        config=config,
        output_directory=output_directory,
    )
    full_final_km_file_path = os.path.join(output_directory, final_km_file_path)
    final_km_df = pd.read_csv(full_final_km_file_path, sep="\t")

    # Set the order of the b_term column based on unique_c_terms
    final_km_df["b_term"] = pd.Categorical(
        final_km_df["b_term"], categories=unique_c_terms, ordered=True
    )

    # Sort the DataFrame based on the custom order
    final_km_df = final_km_df.sort_values("b_term").reset_index(drop=True)

    valid_rows = final_km_df[
        final_km_df["ab_pmid_intersection"].apply(lambda x: x != "[]")
    ]

    final_km_df_filtered = valid_rows.head(
        config["JOB_SPECIFIC_SETTINGS"]["drug_discovery_validation"]["NUM_C_TERMS"]
    )
    # Save the filtered final_km_df to disk and return its path
    filtered_file_path = os.path.join(
        output_directory, os.path.splitext(final_km_file_path)[0] + "_filtered.tsv"
    )
    final_km_df_filtered.to_csv(filtered_file_path, sep="\t", index=False)

    print(f"Filtered final KM query results saved to {filtered_file_path}")
    return filtered_file_path


def km_with_gpt_workflow(config=None, output_directory=None):
    assert config, "No configuration provided"
    a_term = config["GLOBAL_SETTINGS"].get("A_TERM", "")
    assert a_term, "A_TERM is not defined in the configuration"
    if config["GLOBAL_SETTINGS"]["A_TERM_SUFFIX"]:
        a_term_suffix = config["GLOBAL_SETTINGS"]["A_TERM_SUFFIX"]
        a_term = str(a_term) + str(a_term_suffix)
    print("Executing KM workflow...")
    print("Reading terms from files...")
    c_terms = read_terms_from_file(
        config["JOB_SPECIFIC_SETTINGS"]["km_with_gpt"]["B_TERMS_FILE"]
    )
    assert c_terms, "B_TERM is not defined in the configuration"
    print(f"Running and saving KM query for a_term: {a_term}...")
    km_file_path = run_and_save_query(
        "km_with_gpt",
        a_term,
        c_terms,
        config=config,
        output_directory=output_directory,
    )

    full_km_file_path = os.path.join(output_directory, km_file_path)
    if os.path.getsize(full_km_file_path) <= 1:
        print("KM results are empty. Returning None to indicate no KM results.")
        return None
   
    km_df = pd.read_csv(full_km_file_path, sep="\t")

    sort_column = config["JOB_SPECIFIC_SETTINGS"]["km_with_gpt"].get(
        "SORT_COLUMN", "ab_sort_ratio"
    )
    # assert the sort_column is in the km_df
    assert sort_column in km_df.columns, f"{sort_column} is not in the km_df"
    km_df = km_df.sort_values(by=sort_column, ascending=False)
    assert not km_df.empty, "KM results are empty"

    # check if ab_pmid_intersection is [] and remove those rows
    valid_rows = km_df[
        km_df["ab_pmid_intersection"].apply(lambda x: x != "[]")
    ]
    #print ("valid rows:", valid_rows)
    #print("len valid rows:", len(valid_rows))
    # Save the filtered final_km_df to disk and return its path replacing any spaces with underscores
    filtered_file_path = os.path.join(
        output_directory,
        os.path.splitext(km_file_path)[0].replace(" ", "_") + "_filtered.tsv",
    )
    valid_rows.to_csv(filtered_file_path, sep="\t", index=False)

    print(f"Filtered KM query results saved to {filtered_file_path}")
    if (len(valid_rows) == 0): 
        print ("No KM results after filtering. Return None to indicate no KM results.")
        return None
    return filtered_file_path


def skim_run(config, output_directory):
    """Run the SKIM workflow."""
    print("Executing SKIM workflow...")

    skim_file_path = run_and_save_query(
        "skim_with_gpt",
        config["GLOBAL_SETTINGS"]["A_TERM"],
        read_terms_from_file(
            config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]["C_TERMS_FILE"]
        ),
        read_terms_from_file(
            config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]["B_TERMS_FILE"]
        ),
        config=config,
        output_directory=output_directory,
    )
    if skim_file_path is None:
        return None
    full_skim_file_path = os.path.join(output_directory, skim_file_path)
    skim_df = pd.read_csv(full_skim_file_path, sep="\t")
    sort_column = config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"].get(
        "SORT_COLUMN", "bc_sort_ratio"
    )
    skim_df = skim_df.sort_values(by=sort_column, ascending=False)
    valid_rows = skim_df[
        skim_df["bc_pmid_intersection"].apply(lambda x: x != "[]")
    ]
    skim_df = valid_rows

    # assert the sort_column is in the skim_df
    assert sort_column in skim_df.columns, f"{sort_column} is not in the skim_df"
    assert not skim_df.empty, "SKIM results are empty"

    skim_df.to_csv(full_skim_file_path, sep="\t", index=False)
    print(f"SKIM results saved to {full_skim_file_path}")
    return full_skim_file_path
