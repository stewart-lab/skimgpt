import os
import pandas as pd
import requests
import time
import ast


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
class APIClient:
    def __init__(self, username="username", password="password"):
        self.auth = (username, password)

    def post_api_request(self, url, payload):
        """Send a POST request to the API and return the JSON response."""
        response = requests.post(url, json=payload, auth=self.auth)
        response.raise_for_status()
        return response.json()

    def get_api_request(self, url, job_id):
        """Send a GET request to the API and return the JSON response."""
        response = requests.get(f"{url}?id={job_id}", auth=self.auth)
        response.raise_for_status()
        return response.json()

    def wait_for_job_completion(self, url, job_id, max_retries=100):
        """Wait for an API job to complete and return the result."""
        retries = 0
        while retries < max_retries:
            try:
                response_json = self.get_api_request(url, job_id)
                job_status = response_json["status"]
                if job_status in ["finished", "failed"]:
                    break
                time.sleep(5)
                retries += 1
            except Exception as e:
                print(f"Attempt {retries+1}/{max_retries} failed with error: {e}")
                retries += 1

        if "result" not in response_json:
            print(f"Job did not complete successfully after {max_retries} retries.")
            raise AssertionError("Job did not complete successfully.")

        print("Job completed successfully.")
        return response_json.get("result", None)

    def run_api_query(self, payload, url):
        """Initiate an API query and wait for its completion."""
        print(f"Initiating job with payload: {payload}")
        initial_response = self.post_api_request(url, payload)
        job_id = initial_response["id"]
        return self.wait_for_job_completion(url, job_id)


def run_and_save_query(
    job_type, a_term, c_terms, b_terms=None, config=None, output_directory=None
):
    job_config = configure_job(job_type, a_term, c_terms, b_terms, config)
    api_url = config["GLOBAL_SETTINGS"].get("API_URL", "")
    assert api_url, "'API_URL' is not defined in the configuration"

    api_client = APIClient()
    result = api_client.run_api_query(job_config, api_url)

    if not result:
        return None

    result_df = pd.DataFrame(result)

    if job_type == "skim_with_gpt":
        file_name = f"{job_config['a_terms'][0]}_{job_config['c_terms'][0]}"
    else:
        file_name = job_config["a_terms"][0]

    file_path = f"{job_type}_{file_name}_output.tsv"
    save_to_tsv(result_df, file_path, output_directory)
    return file_path


def filter_term_columns(df):
    for column in ["a_term", "b_term", "c_term"]:
        if column in df.columns:
            df[column] = df[column].apply(
                lambda x: x.split("|")[0] if "|" in str(x) else x
            )
    return df


# Job Configuration
def configure_job(job_type, a_term, c_terms, b_terms=None, config=None):
    """Configure a job based on the provided type and terms."""
    assert config, "No configuration provided"

    common_settings = {
        "a_terms": [a_term],
        "return_pmids": True,
        "query_knowledge_graph": False,
        "top_n_articles_most_cited": config["GLOBAL_SETTINGS"].get(
            "TOP_N_ARTICLES_MOST_CITED", 50
        ),
        "top_n_articles_most_recent": config["GLOBAL_SETTINGS"].get(
            "TOP_N_ARTICLES_MOST_RECENT", 50
        ),
    }

    if job_type == "km_with_gpt":
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
        return {
            **common_settings,
            **job_specific_settings,
            "b_terms": b_terms,
            "c_terms": c_terms,
        }
    elif job_type in ["km_with_gpt", "position_km_with_gpt"]:
        return {**common_settings, **job_specific_settings, "b_terms": c_terms}
    else:
        raise ValueError(f"Invalid job type: {job_type}")


def km_with_gpt_workflow(config=None, output_directory=None):
    assert config, "No configuration provided"
    a_term = config["GLOBAL_SETTINGS"].get("A_TERM", "")
    assert a_term, "A_TERM is not defined in the configuration"
    if config["GLOBAL_SETTINGS"]["A_TERM_SUFFIX"]:
        a_term_suffix = config["GLOBAL_SETTINGS"]["A_TERM_SUFFIX"]
        a_term = str(a_term) + str(a_term_suffix)
    print("Executing KM workflow...")
    print("Reading terms from files...")
    b_terms = read_terms_from_file(
        config["JOB_SPECIFIC_SETTINGS"]["km_with_gpt"]["B_TERMS_FILE"]
    )
    assert b_terms, "B_TERM is not defined in the configuration"
    print(f"Running and saving KM query for a_term: {a_term}...")
    km_file_path = run_and_save_query(
        "km_with_gpt", a_term, b_terms, config=config, output_directory=output_directory
    )

    full_km_file_path = os.path.join(output_directory, km_file_path)
    if os.path.getsize(full_km_file_path) <= 1:
        print("KM results are empty. Returning None to indicate no KM results.")
        return None

    km_df = pd.read_csv(full_km_file_path, sep="\t")

    sort_column = config["JOB_SPECIFIC_SETTINGS"]["km_with_gpt"].get(
        "SORT_COLUMN", "ab_sort_ratio"
    )
    assert sort_column in km_df.columns, f"{sort_column} is not in the km_df"
    km_df = km_df.sort_values(by=sort_column, ascending=False)
    assert not km_df.empty, "KM results are empty"
    valid_rows = km_df[km_df["ab_pmid_intersection"].apply(lambda x: len(x) > 0)]
    valid_rows = filter_term_columns(valid_rows)
    filtered_file_path = os.path.join(
        output_directory,
        os.path.splitext(km_file_path)[0].replace(" ", "_") + "_filtered.tsv",
    )
    valid_rows.to_csv(filtered_file_path, sep="\t", index=False)

    print(f"Filtered KM query results saved to {filtered_file_path}")
    if len(valid_rows) == 0:
        print(
            "No KM results after filtering. Returning None to indicate no KM results."
        )
        return None
    return filtered_file_path


def skim_with_gpt_workflow(config, output_directory):
    """Run the SKIM workflow."""
    print("Executing SKIM workflow...")

    b_terms_file = config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]["B_TERMS_FILE"]
    c_terms_file = config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]["C_TERMS_FILE"]

    skim_file_path = run_and_save_query(
        "skim_with_gpt",
        config["GLOBAL_SETTINGS"]["A_TERM"],
        read_terms_from_file(c_terms_file),
        read_terms_from_file(b_terms_file),
        config=config,
        output_directory=output_directory,
    )
    full_skim_file_path = os.path.join(output_directory, skim_file_path)
    skim_df = pd.read_csv(full_skim_file_path, sep="\t")
    sort_column = config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"].get(
        "SORT_COLUMN", "bc_sort_ratio"
    )
    skim_df = skim_df.sort_values(by=sort_column, ascending=False)
    valid_rows = skim_df[
        (skim_df["bc_pmid_intersection"].apply(len) > 0)
        | (skim_df["ab_pmid_intersection"].apply(len) > 0)
        | (skim_df["ac_pmid_intersection"].apply(len) > 0)
    ]
    if valid_rows.empty:
        print(
            "No SKIM results after filtering. All intersection columns are empty. Returning None."
        )
        return None

    skim_df = valid_rows
    skim_df = filter_term_columns(skim_df)
    full_skim_file_path = os.path.join(
        output_directory, os.path.splitext(skim_file_path)[0] + "_filtered.tsv"
    )
    assert sort_column in skim_df.columns, f"{sort_column} is not in the skim_df"
    skim_df.to_csv(full_skim_file_path, sep="\t", index=False)
    print(f"SKIM results saved to {full_skim_file_path}")
    return full_skim_file_path
