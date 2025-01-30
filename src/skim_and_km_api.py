import os
import pandas as pd
import requests
import time
import ast
import logging
from src.utils import setup_logger

# Get the centralized logger instance
logger = setup_logger()

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
    logger.info(f"Data saved to {full_path}")


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

    def wait_for_job_completion(self, url, job_id):
        """Wait for an API job to complete and return the result."""
        start_time = time.time()
        last_report_time = start_time

        while True:
            try:
                response_json = self.get_api_request(url, job_id)
                status = response_json["status"]

                current_time = time.time()
                if current_time - last_report_time >= 300:  # 5 minutes
                    elapsed = int((current_time - start_time) / 60)
                    logging.info(f"Job status after {elapsed} minutes: {status}")
                    last_report_time = current_time

                if status == "finished":
                    logging.info("Job completed successfully")
                    return response_json.get("result")
                elif status == "failed":
                    raise AssertionError(f"Job failed with status: {status}")
                elif status in ["queued", "started"]:
                    time.sleep(5)
                else:
                    raise ValueError(f"Unknown job status: {status}")

            except Exception as e:
                logging.error(f"API request failed: {e}")
                time.sleep(5)

    def run_api_query(self, payload, url):
        """Initiate an API query and wait for its completion."""
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
        logging.warning("API query returned no results.")
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
    """
    Filter the term columns by splitting on '|' and retaining the first element.

    Args:
        df (pd.DataFrame): DataFrame containing term columns.

    Returns:
        pd.DataFrame: Updated DataFrame with filtered term columns.
    """
    for column in ["a_term", "b_term", "c_term"]:
        if column in df.columns:
            # Use .loc to explicitly set the values
            df.loc[:, column] = df[column].apply(
                lambda x: x.split("|")[0] if "|" in str(x) else x
            )
    return df


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

    job_specific_settings = {
        "km_with_gpt": config["JOB_SPECIFIC_SETTINGS"]["km_with_gpt"]["km_with_gpt"],
        "skim_with_gpt": config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]["skim"],
    }.get(job_type)

    if not job_specific_settings:
        job_specific_settings = config["JOB_SPECIFIC_SETTINGS"].get(job_type)
        if not job_specific_settings:
            raise ValueError(f"Invalid or unsupported job type: {job_type}")

    if job_type in ["skim", "skim_with_gpt"]:
        return {
            **common_settings,
            **job_specific_settings,
            "b_terms": b_terms,
            "c_terms": c_terms,
        }
    elif job_type == "km_with_gpt":
        return {**common_settings, **job_specific_settings, "b_terms": c_terms}
    else:
        raise ValueError(f"Invalid job type: {job_type}")


def process_query_results(
    job_type: str,
    df: pd.DataFrame,
    config: dict,
    output_directory: str,
    sort_column_default: str,
    intersection_columns: list,
    filter_condition: callable,
) -> pd.DataFrame:
    """
    Processes query results by sorting, parsing intersection columns, and filtering valid rows.

    Args:
        job_type (str): Type of the job (e.g., 'km_with_gpt', 'skim_with_gpt').
        df (pd.DataFrame): DataFrame containing the query results.
        config (dict): Configuration dictionary.
        output_directory (str): Directory to save output files.
        sort_column_default (str): Default sort column if not specified in config.
        intersection_columns (list): List of intersection columns to parse.
        filter_condition (callable): Function to determine valid rows.

    Returns:
        pd.DataFrame: Filtered DataFrame with valid rows.
    """
    # Determine sort column from config or use default
    sort_column = config["JOB_SPECIFIC_SETTINGS"][job_type].get(
        "SORT_COLUMN", sort_column_default
    )
    if sort_column not in df.columns:
        raise KeyError(f"Sort column '{sort_column}' not found in {job_type} results.")

    # Sort the DataFrame
    df_sorted = df.sort_values(by=sort_column, ascending=False)
    if df_sorted.empty:
        raise ValueError(f"{job_type} results are empty after sorting.")

    # Parse intersection columns
    for col in intersection_columns:
        if col in df_sorted.columns:
            df_sorted[col] = df_sorted[col].apply(ast.literal_eval)
        else:
            raise KeyError(f"Expected column '{col}' not found in {job_type} results.")

    # Apply filter condition to get a boolean mask
    filter_mask = filter_condition(df_sorted)

    # Ensure the filter_condition returns a boolean Series
    if not isinstance(filter_mask, pd.Series) or filter_mask.dtype != bool:
        raise TypeError("filter_condition must return a boolean Series.")

    # Apply the boolean mask to filter the DataFrame
    valid_rows = df_sorted[filter_mask]

    # Apply additional term column filtering
    valid_rows = filter_term_columns(valid_rows)

    return valid_rows


def save_filtered_results(
    job_type: str,
    skim_file_path: str,
    valid_rows: pd.DataFrame,
    skim_df: pd.DataFrame,
    output_directory: str,
) -> str:
    """
    Save the filtered results and handle no_results.txt.

    Args:
        job_type (str): Type of the job ('km_with_gpt' or 'skim_with_gpt').
        skim_file_path (str): Path to the original skim file.
        valid_rows (pd.DataFrame): DataFrame with valid rows.
        skim_df (pd.DataFrame): Original skim DataFrame before filtering.
        output_directory (str): Directory to save output files.

    Returns:
        str: Path to the filtered results file.
    """
    # Define the path for the filtered results
    filtered_file_path = os.path.join(
        output_directory,
        f"{os.path.splitext(skim_file_path)[0].replace(' ', '_')}_filtered.tsv",
    )

    # Save the filtered results
    valid_rows.to_csv(filtered_file_path, sep="\t", index=False)
    logger.info(f"Filtered {job_type} query results saved to {filtered_file_path}")

    # Identify removed rows
    removed_rows = skim_df[~skim_df.index.isin(valid_rows.index)].copy()

    if not removed_rows.empty:
        # Extract relevant columns based on job type
        columns_to_extract = (
            ["a_term", "b_term", "c_term"]
            if job_type == "skim_with_gpt"
            else ["a_term", "b_term"]
        )
        no_results_df = removed_rows[columns_to_extract]

        # Define the path for the no_results.txt file
        no_results_file_path = os.path.join(output_directory, "no_results.txt")

        # Write the no_results to the file
        no_results_df.to_csv(no_results_file_path, sep="\t", index=False, header=True)

        logger.info(f"No-result entries saved to {no_results_file_path}")
    else:
        logger.info(
            f"All {job_type} queries returned results. No entries to write to no_results.txt."
        )

    return filtered_file_path


def read_terms_from_file_with_retry(filename, max_retries=3, delay=1):
    """
    Read terms from a file with retry mechanism.
    
    Args:
        filename (str): Path to the file
        max_retries (int): Maximum number of retry attempts
        delay (float): Delay between retries in seconds
    """
    for attempt in range(max_retries):
        try:
            if os.path.exists(filename):
                with open(filename, "r") as f:
                    terms = [line.strip() for line in f]
                    terms = list(filter(None, terms))
                    if terms:
                        logger.debug(f"Successfully read {len(terms)} terms from {filename}")
                        return terms
            
            if attempt < max_retries - 1:
                logger.warning(f"Attempt {attempt + 1}: File {filename} empty or not ready, retrying in {delay} seconds...")
                time.sleep(delay)
            
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Attempt {attempt + 1}: Error reading {filename}: {e}, retrying in {delay} seconds...")
                time.sleep(delay)
            
    raise ValueError(f"Failed to read terms from {filename} after {max_retries} attempts")


def km_with_gpt_workflow(config=None, output_directory=None):
    """
    Execute the KM workflow.
    """
    assert config, "No configuration provided"
    a_term = config["GLOBAL_SETTINGS"].get("A_TERM", "")
    assert a_term, "A_TERM is not defined in the configuration"

    if config["GLOBAL_SETTINGS"].get("A_TERM_SUFFIX"):
        a_term_suffix = config["GLOBAL_SETTINGS"]["A_TERM_SUFFIX"]
        a_term = f"{a_term}{a_term_suffix}"

    logger.info("Executing KM workflow...")
    logger.debug("Reading terms from files...")

    b_terms_file = config["JOB_SPECIFIC_SETTINGS"]["km_with_gpt"]["B_TERMS_FILE"]
    logger.debug(f"Reading B terms from file: {b_terms_file}")
    b_terms = read_terms_from_file_with_retry(b_terms_file)
    assert b_terms, "B_TERMS_FILE is empty or not defined in the configuration"

    logger.info(f"Running and saving KM query for a_term: {a_term}...")
    km_file_path = run_and_save_query(
        "km_with_gpt", a_term, b_terms, config=config, output_directory=output_directory
    )

    full_km_file_path = os.path.join(output_directory, km_file_path)

    if not os.path.exists(full_km_file_path) or os.path.getsize(full_km_file_path) <= 1:
        logger.warning(
            "KM results are empty. Returning None to indicate no KM results."
        )
        return None

    # Read the KM results
    km_df = pd.read_csv(full_km_file_path, sep="\t")

    # Process the DataFrame
    valid_rows = process_query_results(
        job_type="km_with_gpt",
        df=km_df,
        config=config,
        output_directory=output_directory,
        sort_column_default="ab_sort_ratio",
        intersection_columns=["ab_pmid_intersection"],
        filter_condition=lambda df: df["ab_pmid_intersection"].apply(len) > 0,
    )

    # Save filtered results and handle no_results.txt
    filtered_file_path = save_filtered_results(
        job_type="km_with_gpt",
        skim_file_path=km_file_path,
        valid_rows=valid_rows,
        skim_df=km_df,
        output_directory=output_directory,
    )

    # Check if there are valid results to return
    if valid_rows.empty:
        logger.warning(
            "No KM results after filtering. Returning None to indicate no KM results."
        )
        return None

    return filtered_file_path


def skim_with_gpt_workflow(config, output_directory):
    """
    Run the SKIM workflow.
    """
    assert config, "No configuration provided"
    a_term = config["GLOBAL_SETTINGS"].get("A_TERM", "")
    assert a_term, "A_TERM is not defined in the configuration"

    if config["GLOBAL_SETTINGS"].get("A_TERM_SUFFIX"):
        a_term_suffix = config["GLOBAL_SETTINGS"]["A_TERM_SUFFIX"]
        a_term = f"{a_term}{a_term_suffix}"

    logger.info("Executing SKIM workflow...")

    # Add error checking for B_TERMS_FILE configuration
    if "skim_with_gpt" not in config["JOB_SPECIFIC_SETTINGS"]:
        raise KeyError("'skim_with_gpt' section missing from JOB_SPECIFIC_SETTINGS")
    
    skim_config = config["JOB_SPECIFIC_SETTINGS"]["skim_with_gpt"]
    if "B_TERMS_FILE" not in skim_config:
        raise KeyError("B_TERMS_FILE not defined in skim_with_gpt configuration")
    
    b_terms_file = skim_config["B_TERMS_FILE"]
    c_terms_file = skim_config["C_TERMS_FILE"]

    logger.debug(f"Reading B terms from file: {b_terms_file}")
    logger.debug(f"Reading C terms from file: {c_terms_file}")
    
    b_terms = read_terms_from_file_with_retry(b_terms_file)
    c_terms = read_terms_from_file_with_retry(c_terms_file)

    if not b_terms:
        logger.error(f"B_TERMS_FILE '{b_terms_file}' is empty or could not be read")
        raise ValueError(f"B_TERMS_FILE '{b_terms_file}' is empty or could not be read")
    
    if not c_terms:
        logger.error(f"C_TERMS_FILE '{c_terms_file}' is empty or could not be read")
        raise ValueError(f"C_TERMS_FILE '{c_terms_file}' is empty or could not be read")

    logger.info(f"Running and saving SKIM query for a_term: {a_term}...")
    skim_file_path = run_and_save_query(
        "skim_with_gpt",
        a_term,
        c_terms,  # c_terms is passed as the third argument
        b_terms,  # b_terms is passed as the fourth argument
        config=config,
        output_directory=output_directory,
    )

    full_skim_file_path = os.path.join(output_directory, skim_file_path)

    if (
        not os.path.exists(full_skim_file_path)
        or os.path.getsize(full_skim_file_path) <= 1
    ):
        logger.warning(
            "SKIM results are empty. Returning None to indicate no SKIM results."
        )
        return None

    # Read the SKIM results
    skim_df = pd.read_csv(full_skim_file_path, sep="\t")

    # Process the DataFrame
    valid_rows = process_query_results(
        job_type="skim_with_gpt",
        df=skim_df,
        config=config,
        output_directory=output_directory,
        sort_column_default="bc_sort_ratio",
        intersection_columns=["ab_pmid_intersection", "bc_pmid_intersection"],
        filter_condition=lambda df: (
            (df["ab_pmid_intersection"].apply(len) > 0)
            | (df["bc_pmid_intersection"].apply(len) > 0)
        ),
    )

    # Save filtered results and handle no_results.txt
    filtered_file_path = save_filtered_results(
        job_type="skim_with_gpt",
        skim_file_path=skim_file_path,
        valid_rows=valid_rows,
        skim_df=skim_df,
        output_directory=output_directory,
    )

    # Check if there are valid results to return
    if valid_rows.empty:
        logger.warning(
            "No SKIM results after filtering. Returning None to indicate no SKIM results."
        )
        return None
    return filtered_file_path
