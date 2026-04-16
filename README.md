# Evaluating hypotheses using SKiM-GPT (Note: Must be on Mir-81)
This repository provides tools to SKIM through PubMed abstracts to evalaute hypotheses.


 ## Requirements

 - Python 3.10^
 - Libraries specified in `requirements.txt`
 - OpenAI API key
 - Pubmed API key
 - CHTC auth token
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
    conda create --name {myenv} python>=3.10
    conda activate {myenv}
    pip install -r requirements.txt
    pip install --no-build-isolation -e .
    ```
   
  3. **Environment Variables**
     Before running the script, ensure you have set up your environment variables. We recommend setting in your shell profile. You must source your shell profile after   setting 
     the environment variables (Jack has our OpenAI and Pubmed keys in his .bashrc on the server FYI):
     
    ```bash
      export OPENAI_API_KEY=your_api_key_here
      export PUBMED_API_KEY=your_api_key_here
     ```
 
  4. **Configuring Parameters**
  The `config.json` file includes global parameters as well as several job types, each with unique paramenters. Please view the [`config` Module Overview] (#config-overview)  to help set up your job.
  
  5. **Running the script**
  
     ```bash
     
     python skimgpt/main.py
   
     ```

     or if running multiple years, censor_year_range sets the upper and lower bounds of the years to run, censor_year_increment sets the increment between years, and censor_year_depth sets the depth of the censor year (1 means the lower bound is the same as the upper bound, 2 means the lower bound is one less than the upper bound, etc.)

     ```bash

     python main_wrapper.py -censor_year_range 2020-2025 -censor_year_increment 1 -censor_year_depth 1

     ```
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
<a name="config-overview"></a>
# `config` Module Overview

This configuration file contains various settings for different job types. Below are descriptions of each parameter:

## General Parameters

- `JOB_TYPE`: Specifies the type of job to be executed, e.g., `km_with_gpt` or `skim_with_gpt`.
- `KM_hypothesis`: Hypothesis template for KM analysis, using f-string format like `{a_term}` and `{b_term}` (e.g., `"Treatment with {b_term} will have no effect on {a_term} patient outcomes."`).
- `SKIM_hypotheses`: A dictionary of hypothesis templates for SKIM analysis (Must use f-string format).
  - `AB`: Relevance hypothesis between `{a_term}` and `{b_term}` (e.g., `"There exists an interaction between the organ {a_term} and the gene {b_term}."`).
  - `BC`: Relevance hypothesis between`{c_term}` and `{b_term}` (e.g., `"There exists an interaction between the disease {c_term} and the gene {b_term}."`).
  - `rel_AC`: Relevance hypothesis between `{c_term}` and `{a_term}` (e.g., `"There exists an interaction between the disease {c_term} and the organ {a_term}."`).
  - `ABC`: Evaluation hypothesis (e.g., `"The gene {b_term} links the organ {a_term} to the disease {c_term}."`).
  - `AC`: Evaluation hypothesis (e.g., `"The gene {a_term} influences the disease {c_term}."`).

## Global Settings

- `A_TERM`: The primary term of interest, such as an organ (e.g., `"Thymus"`).
- `A_TERM_SUFFIX`: Optional suffix for the `A_TERM` (e.g., `""`).
- `TOP_N_ARTICLES_MOST_CITED`: Number of top-cited articles to consider (e.g., `50`).
- `TOP_N_ARTICLES_MOST_RECENT`: Number of most recent articles to consider (e.g., `50`).
- `POST_N`: Number of articles to process after relevance filtering (e.g., `5`).
- `MIN_WORD_COUNT`: Minimum word count for an abstract to be considered (e.g., `98`).
- `MODEL`: Machine learning model used for processing (e.g., `"o3"`).
- `RATE_LIMIT`: Maximum number of requests allowed per time unit (e.g., `3`).
- `DELAY`: Time in seconds to wait before making a new request (e.g., `10`).
- `MAX_RETRIES`: Maximum number of retry attempts after a failed request (e.g., `10`).
- `RETRY_DELAY`: Delay in seconds before retrying a failed request (e.g., `5`).
- `LOG_LEVEL`: Logging level (e.g., `"INFO"`).
- `OUTDIR_SUFFIX`: Suffix for the output directory (e.g., `""`).
- `iterations`: Number of iterations for processing (e.g., `3`).
- `DCH_MIN_SAMPLING_FRACTION`: Minimum sampling fraction for DCH (e.g., `0.06`).
- `DCH_SAMPLE_SIZE`: Sample size for DCH (e.g., `50`).
- `TRITON_MAX_WORKERS`: Maximum number of workers for Triton (e.g., `10`).
- `TRITON_SHOW_PROGRESS`: Boolean to show progress for Triton (e.g., `true`).
- `TRITON_BATCH_CHUNK_SIZE`: Batch chunk size for Triton (e.g., `null`).

## Relevance Filter Settings

- `SERVER_URL`: URL for the Triton server (e.g., `"https://xdddev.chtc.io/triton"`).
- `MODEL_NAME`: Model name for relevance filtering (e.g., `"porpoise"`).
- `TEMPERATURE`: Sampling temperature for model inference (e.g., `0`).
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
- `is_dch`: Boolean flag for DCH mode (e.g., `false`).
- `SORT_COLUMN`: Column used for sorting A-B relationships (e.g., `"ab_sort_ratio"`).
- `ab_fet_threshold`: Fisher Exact Test threshold for A-B relationships (e.g., `1`).
- `censor_year_upper`: Upper bound year for data censoring (e.g., `1980`).
- `censor_year_lower`: Lower bound year for data censoring (e.g., `0`).

### skim_with_gpt

- `position`: Boolean flag to consider positional data (e.g., `false`).
- `A_TERM_LIST`: Boolean to indicate if a list of `A` terms is used (e.g., `true`).
- `A_TERMS_FILE`: File path for the `A` terms list (e.g., `"../input_lists/exercise3/skim_a.txt"`).
- `B_TERMS_FILE`: File path for the `B` terms list (e.g., `"../input_lists/exercise3/skim_b.txt"`).
- `C_TERMS_FILE`: File path for the `C` terms list (e.g., `"../input_lists/exercise3/skim_c.txt"`).
- `SORT_COLUMN`: Column used for sorting B-C relationships (e.g., `"bc_sort_ratio"`).
- `ab_fet_threshold`: Fisher Exact Test threshold for A-B relationships (e.g., `0.1`).
- `bc_fet_threshold`: Fisher Exact Test threshold for B-C relationships (e.g., `0.5`).
- `censor_year_upper`: Upper bound year for data censoring (e.g., `2024`).
- `censor_year_lower`: Lower bound year for data censoring (e.g., `0`).

This configuration is critical for tailoring the behavior of the system to specific job types and requirements. Ensure all file paths and parameters are correctly set before execution to avoid runtime errors.

## Visualizing KM-GPT runs over time

All visualization scripts are found in the skimgpt/visualization folder.

Need output from KM-GPT run (km_with_gpt_wrapper_results.tsv) where file has the following header:
censor_year	Hypothesis	B_term	support	refute	inconclusive	iter_number	o3_score<img width="521" height="17" alt="image" src="https://github.com/user-attachments/assets/f3d8e834-7cc7-4a44-86d2-fca07e32d91d" />

Arguments:

* projpath # path where input file is
* -datatype # km or skim
* -discover # discover date of hypothesis
* -accept # acceptance date of hypothesis
* -x_date # extra date line
* -labels # comma-separated term labels for legend
* -title # title of plot
* -labels2 # comma-separated labels for discovery and acceptance
* -move # move label positions (4 comma-sep numbers)
* -xinterval # interval for x-axis labels

Run plot_separate_runs.py
```
python plot_separate_runs.py <projpath> -datatype <km> -discover <date> -accept <date> -labels <Bterm1,Bterm2> -title <title> -labels2 <disocvery,acceptance> -xinterval <1>
```

## Visualizing a Bayesian credible interval for KM-GPT-DCH

Script uses a beta distribution to find the credible interval of a KM-GPT score.

Note: You must have more than one run of the  **same** direct hypothesis comparison for the **same** year (ideally at least 3) in order to calculate a credible interval.

Use KM-GPT-DCH output results.tsv file where the tab-delimited file has the following information:

  * censor_year:              year
  * iteration:                iteration number
  * A_term:                   Hyp A term
  * B1_term:                  H1 B term
  * B2_term:                  H2 B term        
  * score:                    from 0 to 100, with 0 indicating B2 hypothesis is more likely and 100 indicating B1 hypothesis is more likely
  * decision:                 final model decision
  * num_abstracts:            total abstracts
  * support_H1:               number of abstracts supporting H1
  * support_H2:               number of abstracts supporting H2
  * both:                     number of abstracts supporting both hypotheses
  * neither_or_inconclusive:  number of abstracts not supporting either hypothesis

### Running bayesian credible interval over multiple years 

bayesian_ci.py

Arguments:
  * <projpath> # directory where results.txt or results.tsv file (KM-GPT-DCH output) is located
  * <filename> # input CSV filename with the above headers
  * -discover <year> # optional: year of discovery
  * -accept <year> # optional: year of acceptance
  * -x_date <year> # extra date line
  * -title <title of graph> # optional: title for figure- default is "Aterm: aterm Co-occurrence terms: term1 vs. term2 Years: year1-lastyear km or skim data"
  * -labels <label1,label2> # optional: Comma-separated list of labels for discovery and acceptance (e.g., 'discover,accept')
  * -move <list of numbers> # optional: move discovery/acceptance labels. Comma separated list of 4 numbers required: x, y for discovery, x, y for acceptance. e.g. -m '0.1,0.1,0.1,0.1'
  * -xinterval <int> # interval for x-axis labels, default is 1


To Run:
```
python bayesian_ci.py <projpath> <filename> -discover <discovery date> -accept <acceptance date> -title <title> -labels <list of labels> -move <"0.1,0.1,0.1,0.1"> -xinterval 1
```

Output:

* ribbon plot of scores across time where shaded region is the credible interval

### Running bayesian credible interval for one year
bayes_ci_violinplot.py

Arguments:
  * <projpath> # directory where output directory for km-gpt-dch is that contains results.txt/results.tsv
    
To Run:
```
python bayes_ci_violinplot.py <projpath>
```

Output:

* violin plot showing posterior distribution with bars for credible interval

# Running KM co-occurrence only

The steps of KM are typically run from the command line. The first step involves querying a database to create co-occurrence numbers for statistical analysis.  We provide the output for this step for the three main historical hypotheses involving cervical cancer, scrapie, and peptic ulcer. All scripts, environmental variables, and example data are found in the skimgpt/visualization folder

## Running KM/Skim Direct Hypothesis Comparison (DCH) only

## Use web interface to run KM/Skim
Output:
* km_hyp.txt or skim_hyp.txt file
* km_hyp.txt:

   * Date
   * A Term
   * A Count: count of all abstracts with A term
   * B Term
   * B Count: count of all abstracts with B term
   * AB Count: count of intersection abstracts with A and B term
   * AB PMIDS: Pubmed IDs of abstracts with A and B term
   * AB Pred Score: ratio of AB counts/B counts * p.value
   * AB Pvalue: FET p-value for AB table
   * AB Sort Ratio: ratio of AB counts/B counts
   * Total_count: count of all abstracts

* skim_hyp.txt:

   * Date
   * A Term
   * A Count: count of all abstracts with A term
   * B Term
   * B Count: count of all abstracts with B term
   * AB Count: count of intersection abstracts with A and B term
   * AB PMIDS: Pubmed IDs of abstracts with A and B term
   * AB Pred Score: ratio of AB counts/B counts * p.value
   * AB Pvalue: FET p-value for AB table
   * AB Sort Ratio: ratio of AB counts/B counts
   * B_term: B term again
   * BC Count: count of intersection abstracts with B and C term
   * BC_PMIDS: Pubmed IDs of abstracts with B and C term
   * BC Pred Score: ratio of BC counts/C counts * p.value
   * BC Pvalue: FET p-value for BC table
   * BC sort ratio: ratio of BC counts/C counts
   * C term
   * C count: count of all abstracts with C term
   * Total_count: count of all abstracts
   * FET_BC cutoff: fet p-value cutoff for B-C relationships

## Run calculate stats on output of KM/Skim run
* use calcStatsHyp1vsHyp2.py to calculate additional stats.
* check calcStatsHyp1vsHyp2_commandLine.txt for arguments, flagged (-) arguments are optional.
```
python calcStatsHyp1vsHyp2.py <out dir> <input dir where km_hyp.txt or skim_hyp.txt is> "term1,term2" -out_dir_suf=<suffix for output dir>  -fetab=<FET cutoff for A-B> fetbc=<FET cutoff for B-C> -skip_skim
```
Example:
```
python calcStatsHyp1VsHyp2.py out_stats_hrt_1990-2005 out_hyp_HRT_12312024/2024_12_31_10_23_21_HRT_1990_2005/ "hormone&therapy,tonsillectomy" -out_dir_suf=CVD_HT_Tonsil_fet0.05 -fetab=1 -fetbc=0.05 -skip_skim
```
Output:

* km_hyp_stats.txt or skim_hyp_stats.txt

   * Year
   * StatType: type of test
   * Terms: the two terms compared
   * Statistic: statistic value (for example the z-score for the z-score test)
   * P-Value: p-value associated with test
   * Additional Info

   Stat Types used:
   * chi square test
   * binomial test (skim only)
   * z-test of proportions
   * permutation test (skim only)
   * confidence intervals of odds ratio
 
 * km_kept.txt or skim_kept.txt: 

  lines from km_hyp.txt or skim_hyp.txt kept based on p-value cutoff

## Visualize stats
* use plot_hyp_stats.py to show scatterplot of selected stat across years

  Arguments:

  * <projpath> # directory where either km_hyp_stats.txt or skim_hyp_stats.txt is located
  * -datatype <km or skim> # which type of comparison to visualize (km or skim)
  * -stattype <ratio_of_ratios_zprop> # this is from the StatType column in the *_hyp_stats.txt file. Which stat to visualize
  * -discover <year> # optional: year of discovery
  * -accept <year> # optional: year of acceptance
  * -x_date <year> # optional: extra date line
  * -labels <list of labels> # optional: Comma-separated list of labels to relabel samples in legend (e.g., 'microbiome,vaccines')
  * -title <title of graph> # optional: title for figure- default is "Aterm: aterm Co-occurrence terms: term1 vs. term2 Years: year1-lastyear km or skim data"
  * -labels2 <list of labels> # optional: Comma-separated list of labels for discovery and acceptance (e.g., 'discover,accept')
  * -move <list of numbers to move discovery & acceptance labels> # optional: move discovery/acceptance labels. Comma separated list of 4 numbers required: x, y for discovery, x, y for acceptance. e.g. -m '0.1,0.1,0.1,0.1'
  * -xinterval <integer> # optional: interval for x-axis labels

To run:
```
python plot_hyp_stats.py <projpath> -datatype <datatype> -stattype < stat type> -discover <year> -accept <year> -labels <label1,label2> -title <title> -labels2 <labelA,labelB> -m <"0.1,0.1,0.1,0.1">
```

Output (in visualize folder that is in the input stat folder):

* pdf file of figure
* sessionInfo.txt: R package used


## Contributions
Feel free to contribute to this repository by submitting a pull request or opening an issue for suggestions and bugs.
