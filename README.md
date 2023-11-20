# GPT4-based Abstract Analysis for SKIM-identified Relationships

This repository provides tools to SKIM through PubMed abstracts and analyze the relationship between a given term (A_TERM) and a SKIM/KM identified (C_TERM) using the KM API, the PubMed API and the GPT-4 model. The primary goal is to extract, consolidate, and categorize abstracts from scientific papers into use-case specfic classifications.

This pipeline consists of two modules:
- [`skim_no_km` Module Overview](#skim-no-km-overview)
- [`abstract_comprehension` Module Overview](#abstract-comprehension-overview)



## Directory Structure

```bash
├── abstract_comprehension.py
├── requirements.txt
├── skim_no_km.py
├── input_lists/(B and C terms)
├── config.json
└── test/
```


## Requirements

- Python 3.x
- Libraries specified in `requirements.txt`

## Getting Started

1. **Setup**:
   Clone the repository to your local machine.
   
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Install Dependencies**
   Install the required packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables**
   Before running the script, ensure you have set up your OpenAI API key in your environment. You can set it using:
  ```bash
    export OPENAI_API_KEY=your_api_key_here
```
4. **Configuring Parameters**
   #### Configuration Setup in `config.json`

The application's configuration can be customized using the `config.json` file. This file contains various settings that are essential for tailoring the application to specific use cases. Below is a detailed description of the key sections and parameters in the `config.json` file:

- **`JOB_TYPE`**: This specifies the type of job the application will run. For example: `"drug_discovery_validation"`.

- **`GLOBAL_SETTINGS`**: These settings apply globally across the application.
    - `PORT`: The port on which the application runs.
    - `A_TERM`: A specific term, such as `"Crohn's disease"`.
    - `API_URL`: The API endpoint for job submissions.
    - `RATE_LIMIT`: The maximum number of requests per unit time.
    - `DELAY`: The delay between requests, in seconds.
    - `BASE_URL`: The base URL for API requests.
    - `PUBMED_PARAMS`: Parameters for PubMed API requests.
    - `MAX_ABSTRACTS`: Maximum number of abstracts to retrieve.
    - `MAX_RETRIES`: Maximum number of retries for a request.
    - `RETRY_DELAY`: Delay between retries, in seconds.
    - `SORT_COLUMN`: Column used for sorting results.
    - `NUM_C_TERMS`: Number of terms for a specific category.

- **`JOB_SPECIFIC_SETTINGS`**: These settings are specific to different job types.
    - For `drug_discovery_validation`:
        - Paths to files containing terms and thresholds for various analyses (e.g., `C_TERMS_FILE`, `B_TERMS_FILE`).
        - Specific settings for different stages of the job (e.g., `skim`, `first_km`, `final_km`).
    - For `marker_list`:
        - File paths and parameters for marker list generation.
    - For `post_km_analysis`:
        - Specific file paths and settings for post-knowledge mining analysis.

This JSON file should be updated and maintained to reflect the current operational parameters of the application. Changes to this file will directly impact how the application functions.




   
5. **Running the script**
   ```bash
   python abstract_comprehension.py
   ```


----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



<a name="skim-no-km-overview"></a>
# `skim_no_km` Module Overview


This module provides a set of functions to interact with an external API, process data, and handle file operations. The main purpose is to subset a SKIM query via a KM query of the SKIM A and C terms. The relationship between these terms is well-defined and do not need to be included in downstream analysis. A final KM query is ran to extract PubMed IDs that will serve as the foundation to our GPT4-powered abstract comprehension module. Final output is: terms sorted by our sort_column parameter and respective PMIDs in TSV format.

## File Operations

### `read_terms_from_file(filename)`
- **Purpose**: Reads terms from a given file.
- **Input**: A filename string.
- **Output**: A list of terms.

### `save_to_tsv(data, filename)`
- **Purpose**: Saves data into a TSV (Tab Separated Values) file.
- **Input**: Data (typically a list of dictionaries) and a filename string.
- **Output**: None.

## API Calls

### `post_api_request(url, payload, username, password)`
- **Purpose**: Sends a POST request to an API.
- **Input**: API URL, payload, username, and password.
- **Output**: JSON response.

### `get_api_request(url, job_id, username, password)`
- **Purpose**: Sends a GET request to an API.
- **Input**: API URL, job ID, username, and password.
- **Output**: JSON response.

### `wait_for_job_completion(url, job_id, username, password)`
- **Purpose**: Waits for an API job to complete.
- **Input**: API URL, job ID, username, and password.
- **Output**: Job result.

### `run_api_query(payload, url, username, password)`
- **Purpose**: Initiates an API query and waits for its completion.
- **Input**: Payload, API URL, username, and password.
- **Output**: Job result.

## Job Configuration

### `configure_job(job_type, a_term, c_terms, b_terms=None, filtered_terms=None, config=None)`
- **Purpose**: Configures a job based on type and terms.
- **Input**: Job type, terms (a_term, c_terms, and optionally b_terms and filtered_terms), and a configuration.
- **Output**: Job configuration.

## Main Workflow

### `run_and_save_query(job_type, a_term, c_terms, b_terms=None, filtered_terms=None, username="username", password="password", config=None)`
- **Purpose**: Runs a query, saves the results, and returns the saved file path.
- **Input**: Job type, terms, credentials, and a configuration.
- **Output**: File path where results were saved.

### `main_workflow(a_term, config=None)`
- **Purpose**: Executes the main workflow for processing terms.
- **Input**: A term and a configuration.
- **Output**: File path where final filtered results were saved.

## Key Points

- The module reads terms from files, runs various types of queries using these terms, processes the results, and saves them in TSV format.
- It supports multiple types of jobs, namely "skim", "first_km", and "final_km".
- The main workflow function orchestrates the entire process from reading terms to saving the final filtered results.

<a name="abstract-comprehension-overview"></a>
# `abstract_comprehension` Module Overview

This module provides functions for extracting and analyzing abstracts from PubMed, and then classifying the discussed treatments using the OpenAI GPT-4 model.


## File Operations

### `read_tsv_to_dataframe(file_path)`
- **Purpose**: Reads a TSV file into a dataframe.
- **Input**: Path to the file.
- **Output**: Dataframe with the file's content.

### `write_to_json(data, file_path)`
- **Purpose**: Writes data to a file in JSON format.
- **Input**: Data and path to the file.
- **Output**: None.

## API Calls

### `fetch_abstract_from_pubmed(pmid, base_url, params)`
- **Purpose**: Fetches the abstract of a paper from PubMed using its PMID.
- **Input**: PMID, base URL of the PubMed API, and parameters.
- **Output**: Abstract text and URL of the paper.

### `test_openai_connection(api_key)`
- **Purpose**: Tests the connection to the OpenAI API.
- **Input**: OpenAI API key.
- **Output**: None. Prints success or error message.

### `analyze_abstract_with_gpt4(...)`
- **Purpose**: Analyzes abstract(s) using the GPT-4 model.
- **Input**: Consolidated abstracts, terms, API key, and some flags.
- **Output**: Classification and rationale from GPT-4.

## Data Manipulation and Analysis

### `process_abstracts(pmids, rate_limit, delay, base_url, params)`
- **Purpose**: Fetches and processes abstracts from PubMed in batches.
- **Input**: List of PMIDs, rate limit, delay time, base URL, and parameters.
- **Output**: List of consolidated abstracts and their URLs.

### `process_single_row(row, config)`
- **Purpose**: Processes a single row from the dataframe to fetch and analyze abstracts.
- **Input**: A row from the dataframe and configuration settings.
- **Output**: A dictionary containing the term, result from GPT-4, URLs, and abstracts.

## Main Workflow

### `main_workflow()`
- **Purpose**: Orchestrates the entire workflow from reading the TSV file to analyzing the abstracts and saving the results.
- **Input**: None.
- **Output**: None. Prints the progress and saves the results in a JSON file.

## Key Points

- The module fetches abstracts from PubMed, consolidates them, and then classifies the discussed treatments using OpenAI's GPT-4.
- It supports batch processing to handle rate limits and delays.
- The main workflow function manages the entire process, calling other functions as necessary.


## Contributions

Feel free to contribute to this repository by submitting a pull request or opening an issue for suggestions and bugs.


## Acknowledgments

Thanks to OpenAI for providing the GPT model which powers the analysis in this tool.



