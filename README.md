# Evaluating hypotheses using SKiM-GPT (Note: Must be on Mir-81)
This repository provides tools to SKIM through PubMed abstracts to evalaute hypotheses.


 ## Requirements

 - Python 3.9^
 - Libraries specified in `requirements.txt`
 - OpenAI API key
 - Pubmed API key
 - Gemini API key
 - Rstewart2 access

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
    conda create --name {myenv} python=3.9
    conda activate {myenv}
    pip install -r requirements.txt
    ```

   
3. **Environment Variables**
   Before running the script, ensure you have set up your environment variables. We recommend setting in your shell profile. You must source your shell profile after setting 
   the environment variables (Jack has our OpenAI, Pubmed, and Gemini keys in his .bashrc on the server FYI):
   
  ```bash
    export OPENAI_API_KEY=your_api_key_here
    export PUBMED_API_KEY=your_api_key_here
    export GEMINI_API_KEY=your_api_key_here
   ```

4. **Configuring Parameters**
The `config.json` file includes global parameters as well as several job types, each with unique paramenters. Please view the [`config` Module Overview](#config-overview) to help set up your job.

5. **Running the script**

   ```bash
   
   python main.py
 
   ```
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
<a name="config-overview"></a>
# `config` Module Overview

This configuration file contains various settings for different job types. Below are descriptions of each parameter:

## General Parameters

- `JOB_TYPE`: Specifies the type of job to be executed, e.g., `km_with_gpt` or `skim_with_gpt`.
- `KM_hypothesis`: Hypothesis template for KM analysis, using f-string format like `{a_term}` and `{b_term}` (e.g., `"Treatment with {b_term} will have no effect on {a_term} patient outcomes."`).
- `SKIM_hypotheses`: A dictionary of hypothesis templates for SKIM analysis (Must use f-string format).
  - `AB`: Relevance hyopthesis between `{a_term}` and `{b_term}` (e.g., `"There exists an interaction between the organ {a_term} and the gene {b_term}."`).
  - `BC`: Relevance hyopthesis between`{c_term}` and `{b_term}` (e.g., `"There exists an interaction between the disease {c_term} and the gene {b_term}."`).
  - `rel_AC`: Relevance hyopthesis between `{c_term}` and `{a_term}` (e.g., `"There exists an interaction between the disease {c_term} and the organ {a_term}."`).
  - `ABC`: Evaluation hypothesis (e.g., `"The gene {b_term} links the organ {a_term} to the disease {c_term}."`).
  - `AC`: Evaluation hypothesis (e.g., `"The gene {a_term} influences the disease {c_term}."`).

## Global Settings

- `A_TERM`: The primary term of interest, such as an organ (e.g., `"Thymus"`).
- `A_TERM_SUFFIX`: Optional suffix for the `A_TERM` (e.g., `""`).
- `TOP_N_ARTICLES_MOST_CITED`: Number of top-cited articles to consider (e.g., `300`).
- `TOP_N_ARTICLES_MOST_RECENT`: Number of most recent articles to consider (e.g., `0`).
- `POST_N`: Number of articles to process after relevance filtering (e.g., `20`).
- `MIN_WORD_COUNT`: Minimum word count for an abstract to be considered (e.g., `98`).
- `EVAL_MODEL`: Model used for evaluation/classification (e.g., `"o3"`).
- `FULL_TEXT_MODEL`: Model used for full-text analysis (e.g., `"gemini-3-flash-preview"`).
- `RATE_LIMIT`: Maximum number of requests allowed per time unit (e.g., `3`).
- `DELAY`: Time in seconds to wait before making a new request (e.g., `10`).
- `MAX_RETRIES`: Maximum number of retry attempts after a failed request (e.g., `10`).
- `RETRY_DELAY`: Delay in seconds before retrying a failed request (e.g., `5`).
- `LOG_LEVEL`: Logging verbosity level (e.g., `"INFO"`).
- `OUTDIR_SUFFIX`: Optional suffix for the output directory (e.g., `""`).
- `iterations`: Number of processing iterations (e.g., `3`).
- `DCH_MIN_SAMPLING_FRACTION`: Minimum sampling fraction for Direct Comparison Hypothesis (e.g., `0.06`).
- `DCH_SAMPLE_SIZE`: Sample size for Direct Comparison Hypothesis (e.g., `50`).
- `TRITON_MAX_WORKERS`: Max workers for Triton client (e.g., `10`).
- `TRITON_SHOW_PROGRESS`: Show progress bar for Triton requests (e.g., `true`).
- `TRITON_BATCH_CHUNK_SIZE`: Chunk size for Triton batches (e.g., `null`).

## Relevance Filter Settings

- `SERVER_URL`: URL for the Triton inference server (e.g., `"https://xdddev.chtc.io/triton"`).
- `MODEL_NAME`: Model name for abstract filtering (e.g., `"porpoise"`).
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
- `B_TERMS_FILE`: File path for the `B` terms list (e.g., `"../input_lists/hpv.txt"`).
- `is_dch`: Boolean flag for Direct Comparison Hypothesis mode (e.g., `false`).
- `SORT_COLUMN`: Column used for sorting A-B relationships (e.g., `"ab_sort_ratio"`).
- `ab_fet_threshold`: Fisher Exact Test threshold for A-B relationships (e.g., `1`).
- `censor_year_upper`: Upper bound year for censoring (e.g., `1980`).
- `censor_year_lower`: Lower bound year for censoring (e.g., `0`).

### skim_with_gpt

- `position`: Boolean flag to consider positional data (e.g., `false`).
- `A_TERM_LIST`: Boolean to indicate if a list of `A` terms is used (e.g., `false`).
- `A_TERMS_FILE`: File path for the `A` terms list (e.g., `"../input_lists/exercise3/skim_a.txt"`).
- `B_TERMS_FILE`: File path for the `B` terms list (e.g., `"../input_lists/exercise3/skim_b.txt"`).
- `C_TERMS_FILE`: File path for the `C` terms list (e.g., `"../input_lists/exercise3/skim_c.txt"`).
- `SORT_COLUMN`: Column used for sorting B-C relationships (e.g., `"bc_sort_ratio"`).
- `ab_fet_threshold`: Fisher Exact Test threshold for A-B relationships (e.g., `0.1`).
- `bc_fet_threshold`: Fisher Exact Test threshold for B-C relationships (e.g., `0.5`).
- `censor_year_upper`: Upper bound year for censoring (e.g., `2024`).
- `censor_year_lower`: Lower bound year for censoring (e.g., `0`).

### km_with_gpt_direct_comp

 - `position`: Boolean flag to consider positional data (e.g., `false`).
 - `A_TERM_LIST`: Boolean to indicate if a list of `A` terms is used (e.g., `false`).
 - `A_TERMS_FILE`: File path for the `A` terms list (e.g., `"../input_lists/test/km_a.txt"`).
 - `B_TERMS_FILE`: File path for the `B` terms list (e.g., `"../input_lists/scrapie_b_terms_ProteinvsVI_directlyForNewCode.txt"`).
 - `SORT_COLUMN`: Column used for sorting A-B relationships (e.g., `"ab_sort_ratio"`).
 - `NUM_B_TERMS`: Number of `B` terms to consider (e.g., `25`).
 - `km_with_gpt_direct_comp`:
 - `ab_fet_threshold`: Fisher Exact Test threshold for A-B relationships (e.g., `1`).
 - `censor_year_upper`: Upper bound year for censoring (e.g., `1990`).

This configuration is critical for tailoring the behavior of the system to specific job types and requirements. Ensure all file paths and parameters are correctly set before execution to avoid runtime errors.
## Contributions
Thanks to Leo Xu, Kevin Shine George, Aviral Bal, Beth Moore, Rob Millikin, Ian Ross and Ron Stewart 
