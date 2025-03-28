import os
import pandas as pd
import requests
import time
import ast
from src.utils import Config

def save_to_tsv(data, filename, output_directory, config: Config):
    """Save the data into a TSV (Tab Separated Values) file."""
    full_path = os.path.join(output_directory, filename)
    df = pd.DataFrame(data)
    df.to_csv(full_path, sep="\t", index=False)
    config.logger.debug(f"Data saved to {full_path}")


# API Calls
class APIClient:
    def __init__(self, config: Config, username="username", password="password"):
        self.auth = (username, password)
        self.config = config
        self.logger = config.logger

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
                    self.logger.info(f"Job status after {elapsed} minutes: {status}")
                    last_report_time = current_time

                if status == "finished":
                    self.logger.info("Job completed successfully")
                    return response_json.get("result")
                elif status == "failed":
                    raise AssertionError(f"Job failed with status: {status}")
                elif status in ["queued", "started"]:
                    time.sleep(5)
                else:
                    raise ValueError(f"Unknown job status: {status}")

            except Exception as e:
                self.logger.error(f"API request failed: {e}")
                time.sleep(5)

    def run_api_query(self, payload, url):
        """Initiate an API query and wait for its completion."""
        initial_response = self.post_api_request(url, payload)
        job_id = initial_response["id"]
        return self.wait_for_job_completion(url, job_id)


def run_and_save_query(
    config: Config,
    job_type: str,
    a_term: str,
    b_terms: list[str],
    c_terms: list[str],
    output_directory: str,
) -> str:
    job_config = configure_job(
        config,
        job_type,
        a_term,
        c_terms,
        b_terms
    )
    api_url = config.km_api_url 
    config.logger.debug(f"Running API query for {job_type} with a_term: {a_term}, b_terms: {b_terms}, c_terms: {c_terms}")
    config.logger.debug(f"Job config: {job_config}")
    api_client = APIClient(config=config)
    result = api_client.run_api_query(job_config, api_url)

    if not result:
        config.logger.warning("API query returned no results.")
        return None

    result_df = pd.DataFrame(result)

    if config.is_skim_with_gpt:
        file_name = f"{job_config['a_terms'][0]}_{job_config['c_terms'][0]}"
    else:
        file_name = job_config["a_terms"][0]

    file_path = f"{job_type}_{file_name}_output.tsv"
    save_to_tsv(result_df, file_path, output_directory, config)
    return file_path


def filter_term_columns(df, config: Config):
    logger = config.logger
    if config.is_km_with_gpt_direct_comp:
        for column in ["a_term", "b_term"]:
            if column == 'a_term':
                # Use .loc to explicitly set the values
                df.loc[:, column] = df[column].apply(
                    lambda x: x.split("|")[0] if "|" in str(x) else x
                )
            elif column == 'b_term':
                # keeping only first 2 b terms for km_with_gpt_direct_comp, return a list of 2 terms
                logger.debug(f"filter_term_columns: df[column] = {df[column]}")               
                df.loc[:, column] = df[column].apply(
                    lambda x: (x.split("|")[0:2]) if "|" in str(x) else [x, ""]
                )
                logger.debug(f"filter_term_columns AFTER: df[column] = {df[column]}")
    else:
        for column in ["a_term", "b_term", "c_term"]:
            if column in df.columns:
                # Use .loc to explicitly set the values
                df.loc[:, column] = df[column].apply(
                    lambda x: x.split("|")[0] if "|" in str(x) else x
                )
    return df


def configure_job(config: Config, job_type, a_term, c_terms, b_terms=None):
    # Fix: Ensure a_term is always a single string
    if isinstance(a_term, list):
        a_term = a_term[0]

    common_settings = {
        "a_terms": [a_term],  # Wrap single string in list
        "return_pmids": True,
        "query_knowledge_graph": False,
        "top_n_articles_most_cited": config.top_n_articles_most_cited,
        "top_n_articles_most_recent": config.top_n_articles_most_recent,
    }

    # Get job-specific settings dynamically based on job_type
    job_specific_settings = config.job_specific_settings.get(job_type)
    if not job_specific_settings:
        raise ValueError(f"Missing JOB_SPECIFIC_SETTINGS for {job_type} in config")

    if config.is_skim_with_gpt:
        return {
            **common_settings,
            **job_specific_settings,
            "b_terms": b_terms,
            "c_terms": c_terms,
        }
    elif not config.is_skim_with_gpt:
        return {**common_settings, **job_specific_settings, "b_terms": b_terms}
    else:  # right now, this else cannot get reached!
        raise ValueError(f"Invalid job type: {job_type}")


def process_query_results(
    job_type: str,
    df: pd.DataFrame,
    config: Config,
    sort_column_default: str,
    intersection_columns: list,
    filter_condition: callable,
) -> pd.DataFrame:

    # Access config properties through object attributes
    sort_column = config.job_specific_settings.get(
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
    valid_rows = filter_term_columns(valid_rows, config)
    
    return valid_rows


def save_filtered_results(
    job_type: str,
    skim_file_path: str,
    valid_rows: pd.DataFrame,
    skim_df: pd.DataFrame,
    output_directory: str,
    config: Config
) -> str:

    # Define the path for the filtered results
    filtered_file_path = os.path.join(
        output_directory,
        f"{os.path.splitext(skim_file_path)[0].replace(' ', '_')}_filtered.tsv",
    )

    # Save the filtered results
    valid_rows.to_csv(filtered_file_path, sep="\t", index=False)
    config.logger.debug(f"Filtered {job_type} query results saved to {filtered_file_path}")

    # Identify removed rows
    removed_rows = skim_df[~skim_df.index.isin(valid_rows.index)].copy()

    if not removed_rows.empty:
        # Extract relevant columns based on job type
        columns_to_extract = (
            ["a_term", "b_term", "c_term"]
            if config.is_skim_with_gpt
            else ["a_term", "b_term"]
        )
        no_results_df = removed_rows[columns_to_extract]

        # Define the path for the no_results.txt file
        no_results_file_path = os.path.join(output_directory, "no_results.txt")

        # Write the no_results to the file
        no_results_df.to_csv(no_results_file_path, sep="\t", index=False, header=True)

        config.logger.debug(f"No-result entries saved to {no_results_file_path}")
    else:
        config.logger.debug(
            f"All {job_type} queries returned results. No entries to write to no_results.txt."
        )

    return filtered_file_path

def km_with_gpt_direct_comp_workflow(term: dict, config: Config, output_directory: str):
    config.logger.debug(f"km_with_gpt_direct_comp_workflow: term = {term}, type(term['b_terms']) = {type(term['b_terms'])}")
    filtered_file_path = km_with_gpt_workflow(term, config, output_directory)
    return filtered_file_path

def km_with_gpt_workflow(term: dict, config: Config, output_directory: str):
    """Process one A term with ALL B terms in a single API call"""
    logger = config.logger
    a_term = term["a_term"]
    b_terms = term["b_terms"]  # Full list of B terms
    logger.debug(f"km_with_gpt_workflow: term = {term}, type(b_terms) = {type(b_terms)}, b_terms = {b_terms}")
    
    # Add suffix if configured
    if config.global_settings.get("A_TERM_SUFFIX"):
        a_term += config.global_settings["A_TERM_SUFFIX"]

    logger.info(f"Processing A term: {a_term} with {len(b_terms)} B terms")
    
    # Single API call with all B terms
    km_file_path = run_and_save_query(
        config=config,
        job_type=config.job_type,
        a_term=a_term,
        b_terms=b_terms,  # Pass all B terms
        c_terms=[],
        output_directory=output_directory
    )

    # Add null check before path manipulation
    if km_file_path is None:
        logger.error("KM query failed to generate valid file path")
        return None

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
        job_type=config.job_type,
        df=km_df,
        config=config,
        sort_column_default="ab_sort_ratio",
        intersection_columns=["ab_pmid_intersection"],
        filter_condition=lambda df: df["ab_pmid_intersection"].apply(len) > 0,
    )

    # Save filtered results and handle no_results.txt
    filtered_file_path = save_filtered_results(
        job_type=config.job_type,
        skim_file_path=km_file_path,
        valid_rows=valid_rows,
        skim_df=km_df,
        output_directory=output_directory,
        config=config
    )

    # Check if there are valid results to return
    if valid_rows.empty:
        config.logger.warning(
            "No KM results after filtering. Returning None to indicate no KM results."
        )
        return None

    return filtered_file_path


def skim_with_gpt_workflow(term: dict, config: Config, output_directory: str):
    logger = config.logger
    """Process Skim combination with proper B terms handling"""
    a_term = term["a_term"]
    c_term = term["c_term"]
    b_terms = term["b_terms"]  # Could be single or multiple based on position

    # Add suffix if configured
    if config.global_settings.get("A_TERM_SUFFIX"):
        a_term += config.global_settings["A_TERM_SUFFIX"]

    logger.info(f"Processing Skim combination: {a_term} with {len(b_terms)} B terms and {c_term}")

    # Single API call with all relevant B terms
    skim_file_path = run_and_save_query(
        config=config,
        job_type=config.job_type,
        a_term=a_term,
        b_terms=b_terms,
        c_terms=[c_term],
        output_directory=output_directory
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
        job_type=config.job_type,
        df=skim_df,
        config=config,
        sort_column_default="bc_sort_ratio",
        intersection_columns=["ab_pmid_intersection", "bc_pmid_intersection"],
        filter_condition=lambda df: (
            (df["ab_pmid_intersection"].apply(len) > 0)
            | (df["bc_pmid_intersection"].apply(len) > 0)
        ),
    )

    # Save filtered results and handle no_results.txt
    filtered_file_path = save_filtered_results(
        job_type=config.job_type,
        skim_file_path=skim_file_path,
        valid_rows=valid_rows,
        skim_df=skim_df,
        output_directory=output_directory,
        config=config
    )

    # Check if there are valid results to return
    if valid_rows.empty:
        logger.warning(
            "No SKIM results after filtering. Returning None to indicate no SKIM results."
        )
        return None
    return filtered_file_path
