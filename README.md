# GPT-based Paper Analysis

This repository provides tools to SKIM! through scientific papers and analyze the effectiveness of treatments discussed within them using the OpenAI GPT model. The primary goal is to extract, consolidate, and categorize abstracts from scientific papers into categories such as useful, harmful, potentially useful, potentially harmful, or ineffective treatments for specific medical conditions.

This pipeline consists of two modules:
- [`skim_no_km` Module Overview](#skim-no-km-overview)
- [`abstract_comprehension` Module Overview](#abstract-comprehension-overview)



## Directory Structure

```bash
├── abstract_comprehension.py
├── requirements.txt
├── skim_no_km.py
├── BIO_PROCESS_cleaned.txt (SKIM B_TERMS)
├── FDA_approved_ProductsActiveIngrediantsOnly_DUPsRemovedCleanedUp.txt (SKIM C_TERMS and KM B_TERMS)
└── test_skim_no_km.py
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
   #### Configuration Setup in `abstract_comprehension.py`

In the `get_config` function within the `abstract_comprehension.py` file, you will find several settings that you can adjust to fit your specific use case:

#### Core Settings

- **`A_TERM`**: Specifies the term `A` for the queries. For example, it could be a disease name like `Crohn's disease`.

- **`C_TERMS_FILE`**: The name of the text file containing terms of type `C`. These usually are FDA-approved drugs. Make sure this file exists in the same directory or provide an absolute path.

- **`B_TERMS_FILE`**: The name of the text file containing terms of type `B`, which usually relate to biological processes. Like `C_TERMS_FILE`, ensure this file is in the correct directory.

- **`OUTPUT_JSON`**: The name of the output JSON file that will store the results of the queries.

- **`MAX_ABSTRACTS`**: The maximum number of abstracts to process per final KM query.

- **`SORT_COLUMN`**: The column name to sort the SKIM query results by. 

- **`NUM_C_TERMS`**: The number of C terms to consider for processing.

#### Job-Specific Settings

The `JOB_SETTINGS` dictionary contains configurations specific to each type of job: `skim`, `first_km`, and `final_km`.

- **`skim`**:
  - **`ab_fet_threshold`**: The p-value threshold for Fisher's Exact Test when considering the `A-B` term pair.
  - **`bc_fet_threshold`**: The p-value threshold for Fisher's Exact Test when considering the `B-C` term pair.
  - **`censor_year`**: The year to censor the data at.

- **`first_km`**:
  - **`ab_fet_threshold`**: The p-value threshold for Fisher's Exact Test for the `first_km` query.
  - **`censor_year`**: The year to censor the data at for this job type.

- **`final_km`**:
  - **`ab_fet_threshold`**: The p-value threshold for Fisher's Exact Test for the `final_km` query.
  - **`censor_year`**: The year to censor the data at for this job type.

You can adjust these settings to better suit your research or project requirements.




   
5. **Running the script**
   ```bash
   python abstract_comprehension.py
   ```

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



