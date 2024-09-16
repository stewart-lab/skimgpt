# GPT4-based Abstract Analysis for co-occurence-identified Relationships (Note: Must be on Mir-81)
This repository provides tools to SKIM through PubMed abstracts and analyze the relationship between a given A_TERM and a SKIM/KM identified (B_TERM and/or C_TERM) using the KM API, the PubMed API and the GPT-4 model. We also accept A_TERM lists to perfrom multiple queries agaisnt a B_TERM list. The primary goal is to extract, consolidate, and categorize abstracts from scientific papers into use-case specfic classifications.


 ## Requirements

 - Python 3.11
 - Libraries specified in `requirements.txt`
 - OpenAI API key
 - Pubmed API key
 - Mir-81 access 

 ## Getting Started

 1. **Setup**:
    Clone the repository to your machine and change to its top level directory. 

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

 2. **Install Dependencies (with conda)**
    Install the required packages using pip:
    ```bash
    conda create --name {myenv} python=3.11.3
    conda activate {myenv}
    pip install -r requirements.txt
    ```
3. **Environment Variables**
   Before running the script, ensure you have set up your OpenAI API key in your environment. We recommend setting in your shell profile. You can set it using (NOTE: you must source your shell profile after setting the API keys:
  ```bash
    export OPENAI_API_KEY=your_api_key_here
    export PUBMED_API_KEY=your_api_key_here
```
4. **Setting up SSH Key pair**
  ```bash
    ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
    chmod 600 ~/.ssh/<your_key>
    chmod 600 ~/.ssh/<your_key>.pub
    ssh-copy-id -i /path/to/.ssh/<your_key>.pub <chtc_username>@ap2002.chtc.wisc.edu
    
```
   

4. **Configuring Parameters**
The `config.json` file includes global parameters as well as several job types, each with unique paramenters. Please view the [`config` Module Overview](#config-overview) to help set up your job. Additionally, we need to set up your gateway to CHTC:
   ```bash
    "SSH": {
        "server": "ap2002.chtc.wisc.edu",
        "port": 22,
        "user": <your_chtc_username>",
        "key_path": <~/.ssh/<your_key>,
        "config_path": "../config.json",
        "src_path": "./",
        "remote_path": "<your_chtc_home_dir>/kmGPT/"
    },
   ```

5. **Running the script**

   ```bash
   
   source run_analysis.sh
 
   ```
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
<a name="skim-and-km-api-overview"></a>
# `skim_and_km_api` Module Overview
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
<a name="prompt-and-scoring-overview"></a>
# `prompt_and_scoring` Module Overview
This module contains functions that generate prompts and scoring mechanisms for various classifications and analyses. Below are descriptions of each function:
## `build_your_own_prompt`
- **Purpose**: Generates a prompt to determine the relationship between (`b_term`) and (`a_term`).
- **Parameters**:
  - `b_term`: Whatever terms were in the b_term list.
  - `a_term`: A string or a iterable list of the main term of interest.
   - `consolidated_abstracts`: A collection of biomedical abstracts provided for reference.
 - **Returns**: A f'string prompt that will be written out to the output_json and fed into gpt with the appropriate parameters.

 ## `hypothesis_confirmation_rms`
 - **Purpose**: Generates a prompt to confirm a DDI between (`b_term`) and (`a_term`).  Does NOT perform count correction yet.
 - **Parameters**:
   - `b_term`: Whatever terms were in the b_term list.
   - `a_term`: A string or a iterable list of the main term of interest.
   - `consolidated_abstracts`: A collection of biomedical abstracts provided for reference.
 - **Returns**: A f’string prompt that will be written out to the output_json and fed into gpt with the appropriate parameters.

 ## `drug_process_relationship_classification_prompt`
 - **Purpose**: Creates a prompt for classifying the relationship between a drug (`b_term`) and a process or disease (`a_term`).
 - **Parameters**:
  - `b_term`: The drug or compound in question.
  - `a_term`: The biological process or disease.
  - `abstract`: The abstract provided for reference.
- **Returns**: A string prompt asking for classification into specific categories with a rationale.
## `drug_process_relationship_scoring`
- **Purpose**: Defines the scoring criteria for the relationship classification between a drug and a process/disease.
- **Parameter**:
  - `term`: The term (drug or process/disease) used in the scoring criteria.
- **Returns**: A list of tuples containing relationship statements and their corresponding score.
## `drug_synergy_prompt`
- **Purpose**: Generates a prompt to evaluate the synergistic effect of inhibiting BRD4 and another gene (`b_term`) on a disease or process (`a_term`).
- **Parameters**:
  - `b_term`: The gene of interest.
  - `a_term`: The disease or biological process.
  - `consolidated_abstracts`: A collection of biomedical abstracts for reference.
- **Returns**: A string prompt asking for a score (0-10) and rationale based on the hypothesis's reasonability from the abstracts.
## `pathway_augmentation_prompt`
- **Purpose**: Creates a prompt similar to `build_your_own_prompt`, for determining if a gene (`b_term`) is in a pathway (`a_term`).
- **Parameters**:
  - `b_term`, `a_term`, `consolidated_abstracts`: Same as in `build_your_own_prompt`.
- **Returns**: A string prompt asking for a binary classification and rationale based on the given abstracts.
Each function in this module plays a vital role in facilitating specific biomedical analyses and classifications, utilizing the provided abstracts as the primary source of information.
<a name="config-overview"></a>
# `config` Module Overview

This configuration file contains various settings for different job types. Below are descriptions of each parameter:

## General Parameters

- `JOB_TYPE`: Specifies the type of job to be executed, e.g., `km_with_gpt` or `skim_with_gpt`.
- `KM_hypothesis`: Hypothesis template for KM analysis, using f-string format like `{a_term}` and `{b_term}` (e.g., `"Treatment with {b_term} will have no effect on {a_term} patient outcomes."`).
- `SKIM_hypotheses`: A dictionary of hypothesis templates for SKIM analysis (Must use f-string format).
  - `AB`: Relevance hyopthesis between `{a_term}` and `{b_term}` (e.g., `"There exists an interaction between the organ {a_term} and the gene {b_term}."`).
  - `BC`: Relevance hyopthesis between`{c_term}` and `{b_term}` (e.g., `"There exists an interaction between the disease {c_term} and the gene {b_term}."`).
  - `AC`: Relevance hyopthesis between `{c_term}` and `{a_term}` (e.g., `"There exists an interaction between the disease {c_term} and the organ {a_term}."`).
  - `ABC`: Evaluation hypothesis (e.g., `"The gene {b_term} links the organ {a_term} to the disease {c_term}."`).
- `Evaluate_single_abstract`: Boolean flag to evaluate a single abstract (e.g., `false`).

## Global Settings

- `A_TERM`: The primary term of interest, such as an organ (e.g., `"Thymus"`).
- `A_TERM_SUFFIX`: Optional suffix for the `A_TERM` (e.g., `""`).
- `TOP_N_ARTICLES_MOST_CITED`: Number of top-cited articles to consider (e.g., `300`).
- `TOP_N_ARTICLES_MOST_RECENT`: Number of most recent articles to consider (e.g., `0`).
- `POST_N`: Number of articles to process after relevance filtering (e.g., `20`).
- `MIN_WORD_COUNT`: Minimum word count for an abstract to be considered (e.g., `98`).
- `MODEL`: Machine learning model used for processing (e.g., `"gpt-4o-2024-08-06"`).
- `MAX_TOKENS`: Maximum number of tokens per API request (e.g., `1000`).
- `API_URL`: URL for the API endpoint (e.g., `"http://localhost:5099/skim/api/jobs"`).
- `PORT`: Port number for the API service (e.g., `"5081"`).
- `RATE_LIMIT`: Maximum number of requests allowed per time unit (e.g., `3`).
- `DELAY`: Time in seconds to wait before making a new request (e.g., `10`).
- `MAX_RETRIES`: Maximum number of retry attempts after a failed request (e.g., `10`).
- `RETRY_DELAY`: Delay in seconds before retrying a failed request (e.g., `5`).

## SSH Settings

- `server`: SSH server address (e.g., `"ap2002.chtc.wisc.edu"`).
- `port`: SSH port number (e.g., `22`).
- `user`: SSH username (e.g., `"jfreeman23"`).
- `key_path`: Path to the SSH key file (e.g., `"/w5home/jfreeman/.ssh/chtc2"`).
- `config_path`: Path to the configuration file (e.g., `"../config.json"`).
- `src_path`: Source path for files to be transferred (e.g., `"./"`).
- `remote_path`: Remote directory path on the SSH server (e.g., `"/home/jfreeman23/kmGPT/"`).

## Abstract Filter Settings

- `MODEL`: Model used for abstract filtering (e.g., `"lexu14/porpoise1"`).
- `TEMPERATURE`: Sampling temperature for model inference (e.g., `0`).
- `TOP_K`: Number of highest-probability vocabulary tokens to keep for top-k-filtering (e.g., `20`).
- `TOP_P`: Cumulative probability for nucleus sampling (e.g., `0.95`).
- `MAX_COT_TOKENS`: Maximum tokens for Chain-of-Thought reasoning (e.g., `500`).
- `DEBUG`: Boolean flag to enable debug mode (e.g., `false`).
- `TEST_LEAKAGE`: Boolean flag to test for data leakage (e.g., `false`).
- `TEST_LEAKAGE_TYPE`: Type of data leakage test (e.g., `"empty"`).

## Job-Specific Settings

### km_with_gpt

- `position`: Boolean flag to consider positional data (e.g., `false`).
- `A_TERM_LIST`: Boolean to indicate if a list of `A` terms is used (e.g., `false`).
- `A_TERMS_FILE`: File path for the `A` terms list (e.g., `"../input_lists/test/km_a.txt"`).
- `B_TERMS_FILE`: File path for the `B` terms list (e.g., `"../input_lists/leakage_b_terms.txt"`).
- `SORT_COLUMN`: Column used for sorting A-B relationships (e.g., `"ab_sort_ratio"`).
- `NUM_B_TERMS`: Number of `B` terms to consider after sorting (e.g., `25`).
- `km_with_gpt`:
  - `ab_fet_threshold`: Fisher Exact Test threshold for A-B relationships (e.g., `1`).
  - `censor_year`: Year for data censoring or time-slicing (e.g., `2024`).

### skim_with_gpt

- `position`: Boolean flag to consider positional data (e.g., `false`).
- `A_TERM_LIST`: Boolean to indicate if a list of `A` terms is used (e.g., `false`).
- `A_TERMS_FILE`: File path for the `A` terms list (e.g., `"../input_lists/exercise3/skim_a.txt"`).
- `B_TERMS_FILE`: File path for the `B` terms list (e.g., `"../input_lists/genes_no_syn.txt"`).
- `NUM_B_TERMS`: Number of `B` terms to consider (e.g., `20000`).
- `C_TERMS_FILE`: File path for the `C` terms list (e.g., `"../input_lists/down_syndrome.txt"`).
- `SORT_COLUMN`: Column used for sorting B-C relationships (e.g., `"bc_sort_ratio"`).
- `skim`:
  - `ab_fet_threshold`: Fisher Exact Test threshold for A-B relationships (e.g., `0.1`).
  - `bc_fet_threshold`: Fisher Exact Test threshold for B-C relationships (e.g., `0.5`).
  - `censor_year`: Year for data censoring or time-slicing (e.g., `2024`).
  - `top_n`: Number of top items to consider after AB linkage (e.g., `300`).

This configuration is critical for tailoring the behavior of the system to specific job types and requirements. Ensure all file paths and parameters are correctly set before execution to avoid runtime errors.
## Contributions
Feel free to contribute to this repository by submitting a pull request or opening an issue for suggestions and bugs.
## Acknowledgments
Thanks to OpenAI for providing the GPT model which powers the analysis in this tool.



