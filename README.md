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
    Install the required packages using pip (within repository directory):
    ```bash
    conda create --name {myenv} python>=3.10
    conda activate {myenv}
    pip install -r requirements.txt
    pip install --no-build-isolation -e .
    ```
   
  3. **Environment Variables**
     Before running the script, ensure you have set up your environment variables. We recommend setting in your shell profile. You must source your shell profile after   setting 
     the environment variables:
     
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
- `DCH_MIN_SAMPLING_FRACTION`: Minimum share of `DCH_SAMPLE_SIZE` guaranteed to each
  candidate, so a small abstract pool still contributes evidence instead of being
  rounded out of the comparison (e.g., `0.06` — at least 3 of 50). Capped by what a
  pool actually holds: a pool of 1 contributes 1. Values above `0.5` cannot be
  honored for both candidates and fall back to an even split with a logged warning.
- `DCH_SAMPLE_SIZE`: Sample size for DCH (e.g., `50`).
- `DCH_SAMPLE_SEED`: Integer seed for DCH abstract sampling, or `null` for unseeded
  (default). When set, each iteration draws a reproducible sample — iterations still
  differ from one another, but re-running the same config redraws iteration 1..N
  identically. Useful for holding the sample fixed while varying something else
  (model, prompt, censor year). Note that reproducibility is conditional on the
  abstract pool being unchanged: if PubMed results or relevance-filter labels shift,
  the same seed yields a different sample. The `Pool fingerprint:` INFO log line
  identifies that case.
- `DCH_FIX_SAMPLE_ACROSS_ITERATIONS`: Boolean, default `false`. When `true`, one
  abstract sample is drawn before the iteration loop and every iteration scores that
  *same* sample, instead of each iteration drawing its own. Use this to isolate the
  LLM's own output variance from sampling variance — with the default `false`,
  differences across iterations mix both sources; with this `true`, the input is
  held constant so any difference across iterations comes from the model alone.
  This is the opposite intent of `DCH_SAMPLE_SEED`, which keeps iterations
  *different* from each other on purpose; if both are set, the one fixed draw is
  made reproducible via the seed, but it is still identical across iterations.
- `TRITON_MAX_WORKERS`: Concurrent streaming inference workers for Triton (e.g., `10`).

## HTCONDOR

- `collector_host`: URL for CHTC server (e.g., "cm.chtc.wisc.edu")
- `submit_host`: URL for CHTC submit node (e.g., "ap2002.chtc.wisc.edu")
- `docker_image`: Docker image for skimgpt ("docker://stewartlab/skimgpt:2.0.1")

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
- `is_dch`: Boolean flag for DCH mode (e.g., `false`). Compares exactly 2 B terms.
- `tournament`: Settings for tournament-style N-way DCH comparison (see [Running a Tournament-Style N-Way DCH Comparison](#running-a-tournament-style-n-way-dch-comparison)). Must leave `is_dch` set to `false` when `tournament.enabled` is `true`.
  - `enabled`: Boolean flag to turn on tournament mode (e.g., `false`).
  - `tie_threshold`: Half-width of the tie band around a score of 50 (e.g., `5` means scores 45-55 count as a tie).
  - `max_rounds`: Hard cap on tournament rounds (e.g., `null` to auto-compute from the number of B terms).
  - `max_consecutive_stalls`: Stop and report co-winners if this many rounds in a row fail to reduce the surviving term count (e.g., `2`).
  - `seed`: Optional integer seed for reproducible shuffling/pairing (e.g., `null` for non-reproducible).
  - `on_pair_error`: `"advance_both"` (default, tolerate a failed pairwise comparison) or `"abort_round"` (stop the whole tournament).
  - `pair_retries`: Number of automatic retries for a failed pairwise comparison before applying `on_pair_error` (e.g., `1`).
  - `max_parallel_pairs`: Concurrency cap for pairwise comparisons within a round (e.g., `null` to run every pair in the round at once).
- `ranking`: Settings for ranking all N B terms via a parallel bubble sort (see [Ranking All B Terms via a Parallel Bubble Sort](#ranking-all-b-terms-via-a-parallel-bubble-sort)). Must leave `is_dch` set to `false` when `ranking.enabled` is `true`.
  - `enabled`: Boolean flag to turn on ranking mode (e.g., `false`).
  - `tie_threshold`: Half-width of the tie band around a score of 50, same meaning as in `tournament` (e.g., `5`).
  - `max_passes`: Hard cap on sort passes (e.g., `null` to auto-compute as `len(B terms) + 2`).
  - `min_no_evidence_before_quarantine`: Number of times a term must show up with zero supporting abstracts *for itself* in a real (non-cached) comparison before it's pulled out of the active sort (e.g., `1` to quarantine immediately; raise it if quarantining looks too aggressive for noisy/sparse-evidence terms). This checks each term's own support independently, not just symmetric zero-vs-zero ties -- a term with no literature of its own gets flagged even if it lost decisively to a well-evidenced opponent.
  - `min_errors_before_quarantine`: Number of times a term must be involved in a comparison that failed at the pipeline level (not a scientific "no evidence" result -- a subprocess/infrastructure failure) before it's pulled out (e.g., `2`, giving it a couple of fresh retries first since a failure isn't a reproducible fact about the term the way zero support is).
  - `enable_escape_comparisons`: Boolean flag for the frozen-tie-chain escape/validation mechanism (e.g., `true`; see the note on it below).
  - `reseed_every_n_passes`: Every N passes, re-sort the active list by each term's (wins - losses) record so far (e.g., `2`; `0`/`null` to disable). Corrects positions that reflect an accident of the adjacent-sort process rather than a term's actual record -- see the note below.
  - `seed`: Optional integer seed for reproducible initial shuffling (e.g., `null` for non-reproducible).
  - `pair_retries`: Number of automatic retries for a failed pairwise comparison (e.g., `1`).
  - `max_parallel_pairs`: Concurrency cap for pairwise comparisons within a pass (e.g., `null` to run every comparison in the pass at once).
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

# Running KM co-occurrence only

The steps of KM are typically run from the command line. The first step involves querying a database to create co-occurrence numbers for statistical analysis.  We provide the output for this step for the three main historical hypotheses involving cervical cancer, scrapie, and peptic ulcer. All scripts, environmental variables, and example data are found in the skimgpt/visualization folder

## Running KM/Skim Direct Hypothesis Comparison only

Set `JOB_SPECIFIC_SETTINGS.km_with_gpt.is_dch` to `true` and point `B_TERMS_FILE` at a file with **exactly 2** B terms, then run `python skimgpt/main.py` as usual. The first line is scored as hypothesis 1 (score toward 100 favors it) and the second line as hypothesis 2 (score toward 0 favors it); a score near 50 is a tie.

### Running a Tournament-Style N-Way DCH Comparison

To compare **more than 2** B terms pairwise and pick an overall winner, use `tournament_wrapper.py` instead of running `is_dch` directly:

1. Point `B_TERMS_FILE` at a file with more than 2 terms (one per line, `|`-delimited synonym groups allowed as usual).
2. In `config.json`, leave `is_dch` set to `false` and set `JOB_SPECIFIC_SETTINGS.km_with_gpt.tournament.enabled` to `true`. Adjust `tie_threshold`, `seed`, etc. as needed (see [Job-Specific Settings](#job-specific-settings) above).
3. Run:

   ```bash
   python tournament_wrapper.py
   ```

Each round randomly pairs up the remaining B terms (one gets a bye if the count is odd) and runs an ordinary `is_dch` comparison per pair (in parallel). Winners are decided as follows:
- A clear score (outside the tie band) advances that term as usual.
- A tie (score within `tie_threshold` of 50) advances **both** terms, but only if each side actually has supporting literature (`support_H1`/`support_H2` > 0). If one side has zero supporting abstracts, the other side wins outright despite the tied score; if **both** sides have zero support, neither term advances (`eliminated_no_support`) — an evidence-free tie shouldn't let an unstudied gene coast into the next round.
- A term that gets a bye is guaranteed to be paired (not byed again) in the very next round, so no term can advance two rounds in a row without ever being compared.

This repeats until one term remains, every remaining term has been eliminated (zero survivors), or the bracket stalls (e.g. every pair ties) for `max_consecutive_stalls` rounds in a row, in which case the surviving terms are reported as co-winners.

Output (under `output/output_<timestamp>_tournament_<suffix>/`):
- `round_<n>/output/<pair_id>/` — a normal single-pair `is_dch` output directory (`results.tsv`, `results/`, `debug/`, `config.json`) for every pairwise comparison in round `n`.
- `bracket_history.json` — full round-by-round record (pairs, scores, decisions, winners, and each side's support flags).
- `bracket_summary.tsv` — flat table, one row per pairwise comparison across all rounds.
- `TOURNAMENT_WINNER.txt` — the final winner (or co-winners, an abort notice, or a "no term survived" notice).

### Summarizing the Tournament Winner

After a tournament run, use `summarize_winner.py` to collect all the evidence behind the winning term into a single JSON file:

```bash
python summarize_winner.py output/output_<timestamp>_tournament_<suffix>
```

This walks `bracket_history.json` for every round the winner won or tied through (skipping any round where its pairwise comparison errored out), reads that pair's per-iteration DCH result JSON files (`results/iteration_*/*_km_with_gpt_direct_comp.json`), and produces:
- `winner_summary.json` — `{"Winner": "<term>", "Result": {"per_abstract": [...], "tallies": {"support": N}, "overall_rationale": [...], "overall_score": <mean>, "scores": [...]}}`. Scores are transformed to be winner-relative (`100 - score` when the winner was the second B term in that pairing) and averaged in code (`statistics.mean`), not estimated. `per_abstract` is de-duplicated by PMID, since the same abstract is often independently re-fetched across rounds/iterations.
- `winner_round_rationales_raw.json` — the raw, un-summarized `score_rationale` sentences per round, kept as an audit trail behind `overall_rationale` (which should read as a genuine synthesis, not a raw dump — treat it as something to review/rewrite per round rather than trust blindly).

If the tournament ended in a tie (multiple entries in `final_terms`), pass `-winner <term>` to pick which co-winner to summarize:

```bash
python summarize_winner.py output/output_<timestamp>_tournament_<suffix> -winner GeneA
```

### Ranking All B Terms via a Parallel Bubble Sort

A single-elimination tournament only tells you the winner. To get a full ranking of every B term, use `rank_wrapper.py` instead — it sorts all N terms via **odd-even transposition sort** (the parallelizable version of bubble sort): each pass compares/swaps adjacent pairs in the current order, alternating "even" pairs `(0,1),(2,3),...` and "odd" pairs `(1,2),(3,4),...`, with every comparison in a pass run concurrently (same `is_dch`-per-pair machinery as the tournament).

1. Point `B_TERMS_FILE` at a file with 2+ terms.
2. In `config.json`, leave `is_dch` set to `false` and set `JOB_SPECIFIC_SETTINGS.km_with_gpt.ranking.enabled` to `true`. Adjust `tie_threshold`, `seed`, etc. as needed (see [Job-Specific Settings](#job-specific-settings) above).
3. Run:

   ```bash
   python rank_wrapper.py
   ```

Comparisons and swaps work similarly to the tournament's tie/support logic, with one difference: a ranking has to place every term somewhere, so terms can't just be eliminated -- instead, unresolvable ones are pulled out of the active sort and appended to the bottom.
- A clear score swaps the pair into the correct relative order (or leaves it, if already correct).
- A tie where both sides have real support leaves the pair as-is (no strict preference either way).
- **Zero-support quarantine**: each term's own supporting-abstract count is checked in *every* real comparison it's part of, not just symmetric ties -- a term with no literature of its own gets flagged even if it happened to lose decisively to a well-evidenced opponent. Once a term crosses `min_no_evidence_before_quarantine`, it's pulled from the active sort (`quarantine_reason: "no_evidence"`). This matters because such a term never swaps regardless of its neighbor, so left in place it becomes an immovable wall that can block correct ordering of the terms on either side of it.
- **Repeated-error quarantine**: a comparison can fail at the pipeline/infrastructure level (not a scientific finding -- e.g. a relevance-filtering worker crashing) and come back with no usable score at all. These failures are never cached, so the pair always gets a fresh attempt if it becomes adjacent again; but if a term racks up `min_errors_before_quarantine` real failures, it's pulled out too (`quarantine_reason: "repeated_error"`) rather than blocking the sort indefinitely.
- The same two terms are never actually re-compared twice across the whole sort for a real (successful) result — results are cached by unordered pair and reused if they become adjacent again in a later pass. Cached reuses don't count toward win/loss/tie/error tallies or quarantine thresholds, since they're the same underlying observation, not a new one.
- **Frozen-tie-chain boundary checks**: when several adjacent *genuine* ties chain together (e.g. `X ≈ Y ≈ Z`, each with real evidence on both sides), the whole chain freezes solid — nothing inside it can swap past either boundary, since that requires a decisive result there, and the boundary is tied too. A term buried in the middle of such a chain never gets independently compared to anything outside it — it just inherits the block's position. Every pass, at most one frozen chain gets one real boundary comparison (never both directions in the same pass, to avoid two swaps corrupting each other's position bookkeeping):
  - **Upward escape**: the chain's **deepest** untried member vs. its **upward** neighbor. A win relocates it above the whole chain, breaking it free. Tests whether a buried member is secretly *better* than what's above.
  - **Downward validation**: the chain's **shallowest** untried member vs. its **downward** neighbor. A loss relocates it below the whole chain. Tests whether a member is secretly *worse* than what's below — the tied block moves together, but normally only the outward-facing member's relationship to that neighbor is ever checked, so other members can silently inherit a position they never earned (see below).

  Either way, a tie/no-change result is cached and the next untried member gets a turn on a later pass. Controlled by `enable_escape_comparisons`. **Known v1 limitation**: each direction only tests one hop per chain per opportunity. If the immediate neighbor is itself an unresolvable tie, a buried/inherited term won't get tested against anyone further out in the same run.

- **Periodic re-seed by record**: every `reseed_every_n_passes` passes, the active list is re-sorted by each term's (wins − losses) tally so far (ties in record keep their current relative order — a stable sort, so no preference is invented where there isn't evidence for one). This catches positions that reflect an accident of the adjacent-sort process rather than a term's actual record — e.g. a tied pair that got positionally "teleported" upward because unrelated neighbors above it were quarantined away, without ever having to beat what it's now sitting above. Right after re-seeding, any adjacent pair the re-seed placed in an order that contradicts an *already-known direct comparison* between them is immediately swapped back (using only the existing cache, no new comparisons) — otherwise a term's aggregate record (which reflects different opponents) could repeatedly override a specific head-to-head result the sort had already correctly resolved, oscillating forever instead of converging.

The sort stops as soon as one full even+odd cycle produces no swaps, no new quarantines, and no re-seed changes (confirmed sorted), or `max_passes` is reached.

Output (under `output/output_<timestamp>_rank_<suffix>/`):
- `pass_<n>_<even|odd>/output/<pair_id>/` — a normal single-pair `is_dch` output directory for every *freshly run* comparison in pass `n` (passes made entirely of cache hits don't create any output directory).
- `pass_<n>_escape/output/escape<idx>_<pair_id>/` and `pass_<n>_validate/output/validate<idx>_<pair_id>/` — the one boundary comparison (if any) run during pass `n`.
- `ranking_history.json` — full pass-by-pass record (comparisons, scores, outcomes, swaps, per-term support flags, `boundary_kind`, `reseeded` flag) plus the `final_ranking` (rank, term, win/tie/loss/error counts, `avg_score`, `scores`, `insufficient_evidence` flag, `quarantine_reason`).
- `ranking_summary.tsv` — flat table, one row per comparison across all passes (plus a marker row for any pass where a re-seed happened), including a `Boundary_Kind` column (`escape`, `validate`, or blank for a normal comparison).
- `FINAL_RANKING.txt` — the numbered final ranking, with each term's `avg_score`, and insufficient-evidence/unresolved terms flagged.

Each term's `avg_score` (and the underlying `scores` list in the JSON) averages every real comparison it was part of, corrected to that term's own perspective: a comparison's score is always recorded from term_i/H1's point of view (100 favors term_i, 0 favors term_j), so when a term was term_j in a given comparison, `100 - score` is used instead before averaging. Cache-hit reuses and errored comparisons are excluded (same reasoning as the win/tie/loss tallies) so the same underlying observation is never counted twice.

## Use web interface to run KM/Skim

Check results for KM results, and download km_hyp.txt or skim_hyp.txt file	

Output:
* km_hyp.txt or skim_hyp.txt file
* km_hyp.txt headers:

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

* skim_hyp.txt headers:

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

## Run calculate stats on output of KM/Skim run (km_hyp.txt or skim_hyp.txt file)

* use hyp_stats.py to calculate additional stats.
* check hyp_stats_commandline.txt for arguments, flagged (-) arguments are optional.
```
python hyp_stats.py <out dir> <input dir where km_hyp.txt or skim_hyp.txt is> "term1,term2" -out_dir_suf=<suffix for output dir>  -fetab=<FET cutoff for A-B> fetbc=<FET cutoff for B-C> -skip_skim
```
Example:
```
python hyp_stats.py out_stats_hrt_1990-2005 out_hyp_HRT_12312024/2024_12_31_10_23_21_HRT_1990_2005/ "hormone&therapy,tonsillectomy" -out_dir_suf=CVD_HT_Tonsil_fet0.05 -fetab=1 -fetbc=0.05 -skip_skim
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

  Arguments: (also can check plot_hyp_stats_commandline.txt for arguments), (-) arguments are optional

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
