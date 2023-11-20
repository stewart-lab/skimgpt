import os
import requests
import pandas as pd
import openai
import time
import json
import shutil
import re
import ast
import logging
from datetime import datetime
from xml.etree import ElementTree
import skim_and_km_api as skim
import prompt_and_scoring_library as prompts
import test.test_abstract_comprehension as test


def get_config(output_directory):
    with open("/isiseqruns/jfreeman_tmp_home/skimGPT/config.json", "r") as f:
        config = json.load(f)
    job_settings = config["JOB_SPECIFIC_SETTINGS"].get(config["JOB_TYPE"], {})
    a_term = config["GLOBAL_SETTINGS"]["A_TERM"]
    if config["JOB_TYPE"] == "marker_list":
        output_json = f"marker_list_numCTerms{config['GLOBAL_SETTINGS'].get('NUM_C_TERMS', '')}.json"
    elif config["JOB_TYPE"] == "post_km_analysis":
        output_json = f"{a_term}_drug_synergy_maxAbstracts{config['GLOBAL_SETTINGS'].get('MAX_ABSTRACTS', '')}.json"
    elif config["JOB_TYPE"] == "drug_discovery_validation":
        censor_year = job_settings.get("skim", {}).get("censor_year", "")
        num_c_terms = config["GLOBAL_SETTINGS"].get("NUM_C_TERMS", "")
        output_json = f"{a_term}_censorYear{censor_year}_numCTerms{num_c_terms}.json"
    else:
        print("Invalid job type (caught in get_config)")

    output_json = output_json.replace(" ", "_").replace("'", "")
    config["OUTPUT_JSON"] = output_json
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    with open(os.path.join(output_directory, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    config["API_KEY"] = api_key
    return config


def read_tsv_to_dataframe(file_path):
    return pd.read_csv(file_path, sep="\t")


def write_to_json(data, file_path, output_directory):
    full_path = os.path.join(output_directory, file_path)
    with open(full_path, "w") as outfile:
        json.dump(data, outfile, indent=4)


def extract_text_from_xml(element):
    text = element.text or ""
    for child in element:
        text += extract_text_from_xml(child)
        if child.tail:
            text += child.tail
    return text


def fetch_abstract_from_pubmed(config, pmid):
    global_settings = config.get("GLOBAL_SETTINGS", {})
    max_retries = global_settings.get("MAX_RETRIES", 3)
    retry_delay = global_settings.get("RETRY_DELAY", 5)
    base_url = global_settings.get(
        "BASE_URL", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    )
    pubmed_params = global_settings.get("PUBMED_PARAMS", {})

    for attempt in range(max_retries):
        try:
            response = requests.get(base_url, params={**pubmed_params, "id": pmid})
            response.raise_for_status()  # Check if the request was successful
            tree = ElementTree.fromstring(response.content)
            break
        except (ElementTree.ParseError, requests.exceptions.RequestException) as e:
            logging.error(f"Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                logging.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logging.error("Max retries reached. Skipping this PMID.")
                raise

    abstract_texts = [
        extract_text_from_xml(abstract) for abstract in tree.findall(".//AbstractText")
    ]
    abstract_text = " ".join(abstract_texts)
    year = next((year.text for year in tree.findall(".//PubDate/Year")), None)
    paper_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

    return abstract_text, paper_url, year


def abstract_quality_control(config, pmids, rate_limit, delay, min_word_count=100):
    assert pmids, "List of PMIDs is empty"
    results = {}
    pmid_batches = [pmids[i : i + rate_limit] for i in range(0, len(pmids), rate_limit)]

    for batch in pmid_batches:
        for pmid in batch:
            try:
                abstract, url, year = fetch_abstract_from_pubmed(config, pmid)
                if not all([abstract, url, year]):
                    logging.warning(
                        f"PMID {pmid} has missing data and will be removed from pool."
                    )
                    continue

                if len(abstract.split()) < min_word_count:
                    logging.warning(
                        f"Abstract for PMID {pmid} is {len(abstract.split())} words. Removing from pool."
                    )
                    continue

                results[pmid] = (abstract, url, year)
            except Exception as e:  # Replace with specific exceptions
                logging.error(f"Error processing PMID {pmid}: {e}")
        time.sleep(delay)

    return results


def analyze_abstract_with_gpt4(consolidated_abstracts, b_term, a_term, config):
    # Assuming openai.api_key is set outside this function globally.
    responses = []

    if not b_term or not a_term:
        logging.error("B term or A term is empty.")
        return responses

    if config["JOB_TYPE"] == "post_km_analysis":
        prompt = prompts.drug_synergy_prompt(b_term, a_term, consolidated_abstracts)
        response = call_openai(prompt, config)
        if response:
            responses.append(response)
    else:
        for abstract in consolidated_abstracts:
            if config["JOB_TYPE"] == "drug_discovery_validation":
                prompt = prompts.drug_process_relationship_classification_prompt(
                    b_term, a_term, abstract
                )
            response = call_openai(prompt, config)
            if response:
                responses.append(response)

    return responses


def handle_rate_limit(e, retry_delay):
    # Extract the retry-after value from the error message if available
    retry_after = int(e.response.headers.get("Retry-After", retry_delay))
    logging.warning(f"Rate limit exceeded. Retrying after {retry_after} seconds...")
    time.sleep(retry_after)


def call_openai(prompt, config):
    retry_delay = config["GLOBAL_SETTINGS"]["RETRY_DELAY"]
    max_retries = config["GLOBAL_SETTINGS"]["MAX_RETRIES"]

    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-1106-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical research analyst.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=512,
            )
            # Check if the response is valid and has content
            content = response.get("choices", [{}])[0].get("message", {}).get("content")
            if content:
                return content
            else:
                logging.error("Received an empty response.")
                time.sleep(retry_delay)
        except openai.error.InvalidRequestError as e:
            if e.response.status_code == 429:  # Rate limit exceeded
                logging.warning("Rate limit exceeded, waiting before retrying...")
                time.sleep(retry_delay)
            else:
                logging.error(f"An unexpected OpenAI API error occurred: {e}")
                time.sleep(retry_delay)
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            time.sleep(retry_delay)

    logging.error("Max retries reached or no valid response received.")
    return None


def synergy_dfr_preprocessing(config):
    csv_path = config["JOB_SPECIFIC_SETTINGS"]["post_km_analysis"]["B_TERMS_FILE"]
    df = pd.read_csv(csv_path)
    desired_columns = ["term", "panc & ggp & kras-mapk set", "brd & ggp set"]
    filtered_df = df[desired_columns]

    new_row = {
        "term": "CDK9",
        "panc & ggp & kras-mapk set": "{26934555, 33879459, 35819261, 31311847}",
        "brd & ggp set": "{19103749, 28673542, 35856391, 16893449, 17942543, 18513937, 32331282, 27764245, 18483222, 23658523, 29415456, 33164842, 18039861, 29212213, 30971469, 28448849, 28077651, 32559187, 29490263, 32012890, 29563491, 28262505, 20201073, 15046258, 28930680, 18971272, 28062857, 29743242, 24335499, 32787081, 33776776, 31594641, 22084242, 34688663, 32203417, 34935961, 23027873, 33619107, 33446572, 18223296, 27322055, 19297489, 29491412, 30068949, 19828451, 36154607, 36690674, 31597822, 23596302, 36046113, 28630312, 29991720, 34045230, 30227759, 34253616, 32188727, 17535807, 16109376, 16109377, 31633227, 28481868, 17686863, 29156698, 26186095, 26083714, 21900162, 27793799, 35249548, 26504077, 29649811, 33298848, 27067814, 31399344, 35337136, 28215221, 22046134, 34062779, 25263550, 21149631, 34971588, 26627013, 26974661, 24518598, 33406420, 36631514, 28182006, 33781756, 24367103}",
    }

    new_row_df = pd.DataFrame([new_row])
    filtered_df = pd.concat([filtered_df, new_row_df], ignore_index=True)

    terms_to_retain = [
        "CDK9",
        "p AKT",
        "JAK",
        "HH",
        "NANOG",
        "CXCL1",
        "BAX",
        "ETS",
        "IKK",
    ]

    # Filter the DataFrame
    filtered_df = filtered_df[
        filtered_df["term"]
        .str.strip()
        .str.lower()
        .isin([term.lower() for term in terms_to_retain])
    ]
    filtered_df.loc[filtered_df["term"] == "p akt", "term"] = "AKT"
    filtered_df["term"] = filtered_df["term"].str.upper()
    return filtered_df


def process_single_row(row, config):
    b_term = row["term"]
    job_type = config.get("JOB_TYPE")
    robust_setting = config["JOB_SPECIFIC_SETTINGS"]["post_km_analysis"].get(
        "robust", False
    )

    if job_type == "post_km_analysis":
        pmids_panc = ast.literal_eval(row["panc & ggp & kras-mapk set"])
        pmids_brd = ast.literal_eval(row["brd & ggp set"])
        combined_pmids = list(
            set(pmids_panc) | set(pmids_brd)
        )  # Combine and remove duplicates

        # Get the abstracts and years for all combined PMIDs
        abstracts_data = abstract_quality_control(
            config,
            combined_pmids,
            config["GLOBAL_SETTINGS"]["RATE_LIMIT"],
            config["GLOBAL_SETTINGS"]["DELAY"],
        )

        # Filter and sort PMIDs based on success and publication years
        successful_pmids = [pmid for pmid in combined_pmids if pmid in abstracts_data]
        pmids_panc = [pmid for pmid in pmids_panc if pmid in successful_pmids]
        pmids_brd = [pmid for pmid in pmids_brd if pmid in successful_pmids]

        sorted_pmids_panc = sort_pmids_by_year(pmids_panc, abstracts_data)
        sorted_pmids_brd = sort_pmids_by_year(pmids_brd, abstracts_data)
        if robust_setting.lower() == "true":
            return perform_robust_analysis(
                sorted_pmids_panc,
                sorted_pmids_brd,
                [abstracts_data[pmid][0] for pmid in successful_pmids],
                successful_pmids,
                b_term,
                config,
            )
        else:
            # Non-robust analysis
            num_pmids_panc = min(
                len(sorted_pmids_panc), config["GLOBAL_SETTINGS"]["MAX_ABSTRACTS"] // 2
            )
            num_pmids_brd = min(
                len(sorted_pmids_brd),
                config["GLOBAL_SETTINGS"]["MAX_ABSTRACTS"] - num_pmids_panc,
            )

            # Select the top PMIDs from each set
            selected_pmids_panc = sorted_pmids_panc[:num_pmids_panc]
            selected_pmids_brd = sorted_pmids_brd[:num_pmids_brd]

            # Combine the selected PMIDs from both sets
            selected_pmids = selected_pmids_panc + selected_pmids_brd

            # Filter the abstracts, URLs, and years based on the selected PMIDs
            consolidated_abstracts = [
                abstracts_data[pmid][0] for pmid in selected_pmids
            ]
            paper_urls = [abstracts_data[pmid][1] for pmid in selected_pmids]
            publication_years = [abstracts_data[pmid][2] for pmid in selected_pmids]

            # Perform analysis on the selected abstracts
            result = analyze_abstract_with_gpt4(
                consolidated_abstracts,
                b_term,
                config["GLOBAL_SETTINGS"]["A_TERM"],
                config,
            )

        # Return the result in the desired format
        return {
            "Term": b_term,
            "Result": result,
            "URLs": paper_urls,
            "Abstracts": consolidated_abstracts,
            "Years": publication_years,
        }


def sort_pmids_by_year(pmids, abstracts_data):
    """Sort PMIDs based on publication year in descending order."""
    return sorted(
        pmids,
        key=lambda x: abstracts_data[x][2]
        if x in abstracts_data and abstracts_data[x][2]
        else "",
        reverse=True,
    )


def perform_robust_analysis(
    sorted_pmids_panc,
    sorted_pmids_brd,
    all_abstracts,
    combined_pmids,
    b_term,
    config,
):
    group_size_panc = config["GLOBAL_SETTINGS"]["MAX_ABSTRACTS"] // 2
    groups_panc = [
        sorted_pmids_panc[i : i + group_size_panc]
        for i in range(0, len(sorted_pmids_panc), group_size_panc)
    ]
    groups_brd = [
        sorted_pmids_brd[i : i + group_size_panc]
        for i in range(0, len(sorted_pmids_brd), group_size_panc)
    ]

    # Create a mapping from PMID to its index to improve efficiency
    pmid_to_index = {pmid: idx for idx, pmid in enumerate(combined_pmids)}

    results = []
    for group_panc in groups_panc:
        for group_brd in groups_brd:
            consolidated_group = group_panc + group_brd

            # Filter the abstracts, URLs, and years based on the consolidated group
            try:
                consolidated_abstracts = [
                    all_abstracts[pmid_to_index[pmid]] for pmid in consolidated_group
                ]
            except KeyError as e:
                # Handle the case where a PMID is not found in the index mapping
                logging.error(f"PMID not found in combined list: {e}")
                continue  # Skip this group

            # Analyze the consolidated abstracts
            result = analyze_abstract_with_gpt4(
                consolidated_abstracts,
                b_term,
                config["GLOBAL_SETTINGS"]["A_TERM"],
                config,
            )
            results.append(result)

    return results


def process_json(json_data, config):
    nested_dict = {}
    a_term = config["GLOBAL_SETTINGS"]["A_TERM"]
    nested_dict[a_term] = {}

    try:
        term_scores = {}
        term_counts = {}
        term_details = {}  # New dictionary to hold the details

        for entry in json_data:
            term = entry.get("Term", None)
            results = entry.get("Result", None)
            urls = entry.get("URLs", None)  # Get URLs
            years = entry.get("Years", None)  # Get Years

            if term is None or results is None or urls is None or years is None:
                print("Warning: Invalid entry found in JSON data. Skipping.")
                continue

            if term not in term_scores:
                term_scores[term] = 0
                term_counts[term] = 0
                term_details[term] = []  # Initialize list to hold details

            for i, result in enumerate(results):
                term_counts[term] += 1  # Increase the count for this term
                score = 0
                scoring_sentence = ""
                if config["JOB_TYPE"] == "drug_discovery_validation":
                    patterns = prompts.drug_process_relationship_scoring(term)

                # Search for the pattern in the result to find the scoring sentence and score
                for pattern, pattern_score in patterns:
                    if pattern in result:
                        score = pattern_score
                        scoring_sentence = re.search(
                            f"([^\.]*{pattern}[^\.]*\.)", result
                        ).group(1)
                        break

                term_scores[term] += score

                # Extract PMID from URL
                url = urls[i]
                pmid_match = re.search(r"(\d+)/?$", url)
                pmid = pmid_match.group(1) if pmid_match else "Unknown"

                # Get the year for this PMID
                year = years[i] if i < len(years) else "Unknown"

                # Record the details
                term_details[term].append(
                    {
                        "PMID": pmid,
                        "Year": year,
                        "Scoring Sentence": scoring_sentence,
                        "Score": score,
                    }
                )

        term_avg_scores = {
            term: (term_scores[term] / term_counts[term]) for term in term_scores
        }

        for term, score in term_scores.items():
            avg_score = term_avg_scores[term]
            nested_dict[a_term][term] = {
                "Total Score": score,
                "Average Score": avg_score,
                "Count": term_counts[term],
                "Details": term_details[term],  # Add the details
            }

        return nested_dict

    except Exception as e:
        print(f"An error occurred while processing the JSON data: {e}")
        return None  # Returning None to indicate failure


def save_to_json(data, config, output_directory):
    output_filename = os.path.join(
        output_directory, config["OUTPUT_JSON"] + "_filtered.json"
    )
    # Adding "_filtered" to the filename
    with open(output_filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Filtered results have been saved to {output_filename}")


def marker_list_filtration(km_file_paths, output_directory, config):
    try:
        dfs = {}
        max_ratios = {}
        pretty_file_paths = []

        filter_method = config["JOB_SPECIFIC_SETTINGS"]["marker_list"]["filter_method"]
        make_pretty = config["JOB_SPECIFIC_SETTINGS"]["marker_list"].get(
            "make_marker_list_pretty", False
        )

        for file_name in km_file_paths:
            file_path = os.path.join(output_directory, file_name)
            df = pd.read_csv(file_path, sep="\t")

            # Save the original DataFrame to a new path with _original suffix
            original_file_path = f"{os.path.splitext(file_path)[0]}_original{os.path.splitext(file_path)[1]}"
            df.to_csv(original_file_path, sep="\t", index=False)

            dfs[file_path] = df

        if filter_method == "highest_sort":
            for file_path, df in dfs.items():
                original_length = len(df)
                df_sorted = df.sort_values(by="ab_sort_ratio", ascending=False)

                for index, row in df_sorted.iterrows():
                    b_term = row["b_term"]
                    ab_sort_ratio = row["ab_sort_ratio"]
                    if b_term not in max_ratios or ab_sort_ratio > max_ratios[b_term]:
                        max_ratios[b_term] = ab_sort_ratio

                df_filtered = df[
                    df.apply(
                        lambda row: row["ab_sort_ratio"] == max_ratios[row["b_term"]],
                        axis=1,
                    )
                ]
                filtered_length = len(df_filtered)

                print(
                    f"File: {file_path}, Method: highest_sort, Original Rows: {original_length}, Filtered Rows: {filtered_length}"
                )
                df_filtered.to_csv(file_path, sep="\t", index=False)

        elif filter_method == "ratio_of_ratios":
            threshold = config["JOB_SPECIFIC_SETTINGS"]["marker_list"][
                "ratio_threshold"
            ]

            # Calculate the average ab_sort_ratio for each b_term across all files
            average_ratios = {}
            for file_path, df in dfs.items():
                for index, row in df.iterrows():
                    b_term = row["b_term"]
                    ab_sort_ratio = row["ab_sort_ratio"]
                    average_ratios.setdefault(b_term, []).append(ab_sort_ratio)

            average_ratios = {
                b_term: sum(ratios) / len(ratios)
                for b_term, ratios in average_ratios.items()
            }

            for file_path, df in dfs.items():
                original_length = len(df)
                df["ratio_of_ratios"] = df["ab_sort_ratio"].map(
                    lambda x: x / average_ratios.get(b_term, 1)
                )
                df_filtered = df[df["ratio_of_ratios"] > threshold]
                filtered_length = len(df_filtered)

                print(
                    f"File: {file_path}, Method: ratio_of_ratios, Original Rows: {original_length}, Filtered Rows: {filtered_length}, Threshold: {threshold}"
                )
                df_filtered.to_csv(file_path, sep="\t", index=False)

        elif filter_method == "diff_genes":
            occurrence_threshold = config["JOB_SPECIFIC_SETTINGS"]["marker_list"][
                "occurrence_threshold"
            ]

            gene_occurrence = {
                gene: 0 for df in dfs.values() for gene in df["b_term"].unique()
            }

            for df in dfs.values():
                for gene in df["b_term"].unique():
                    gene_occurrence[gene] += 1

            total_lists = len(km_file_paths)
            for file_path, df in dfs.items():
                original_length = len(df)
                df_filtered = df[
                    df["b_term"].apply(
                        lambda gene: (gene_occurrence[gene] / total_lists) * 100
                        <= occurrence_threshold
                    )
                ]
                filtered_length = len(df_filtered)

                print(
                    f"File: {file_path}, Method: diff_genes, Original Rows: {original_length}, Filtered Rows: {filtered_length}, Occurrence Threshold: {occurrence_threshold}"
                )
                df_filtered.to_csv(file_path, sep="\t", index=False)

        else:
            print("Invalid filtration method")

        if make_pretty:
            combined_file_path = os.path.join(output_directory, "marker_list.tsv")

            # Clear/create the combined file with headers
            with open(combined_file_path, "w") as f:
                f.write("a_term\tb_term\n")

            for file_path in dfs.keys():
                df_filtered = pd.read_csv(file_path, sep="\t")
                df_filtered["b_term"] = df_filtered["b_term"].apply(
                    lambda x: x.split("|")[0]
                )
                df_pretty = df_filtered[["a_term", "b_term"]]
                pretty_file_path = f"{os.path.splitext(file_path)[0]}_pretty{os.path.splitext(file_path)[1]}"
                df_pretty.to_csv(pretty_file_path, sep="\t", index=False, header=False)

                with open(combined_file_path, "a") as f:
                    df_pretty.to_csv(f, sep="\t", index=False, header=False)

                pretty_file_paths.append(pretty_file_path)

            return pretty_file_paths

    except Exception as e:
        print(f"An error occurred: {e}")


def api_cost_estimator(df, config):
    job_type = config.get("JOB_TYPE", "")
    estimated_cost = 0

    if job_type == "drug_discovery_validation":
        x = config["GLOBAL_SETTINGS"]["MAX_ABSTRACTS"] * len(df)
        estimated_cost = x * 0.006

    elif job_type == "post_km_analysis":
        robust_setting = config["JOB_SPECIFIC_SETTINGS"]["post_km_analysis"].get(
            "robust", "False"
        )

        if robust_setting.lower() == "true":  # If the robust method is being used
            total_calls = 0
            for _, row in df.iterrows():
                groups_panc = len(eval(row["panc & ggp & kras-mapk set"])) // (
                    config["GLOBAL_SETTINGS"]["MAX_ABSTRACTS"] // 2
                )
                groups_brd = len(eval(row["brd & ggp set"])) // (
                    config["GLOBAL_SETTINGS"]["MAX_ABSTRACTS"] // 2
                )
                total_calls += groups_panc + groups_brd
            x = total_calls
        else:
            x = sum(
                df.apply(
                    lambda row: min(
                        len(eval(row["panc & ggp & kras-mapk set"]))
                        + len(eval(row["brd & ggp set"])),
                        config["GLOBAL_SETTINGS"]["MAX_ABSTRACTS"],
                    ),
                    axis=1,
                )
            )

        estimated_cost = x * 0.06  # Using the provided cost for this method

    user_input = input(
        f"The following job consists of {x} abstracts and will cost roughly ${estimated_cost:.2f} in GPT-4 API calls. Do you wish to proceed? [Y/n]: "
    )
    if user_input.lower() != "y":
        print("Exiting workflow.")
        return False
    return True


def initialize_workflow():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs("output", exist_ok=True)
    output_directory = os.path.join("output", f"output_{timestamp}")
    os.makedirs(output_directory, exist_ok=True)
    shutil.copy(
        "/isiseqruns/jfreeman_tmp_home/skimGPT/config.json",
        os.path.join(output_directory, "config.json"),
    )
    config = get_config(output_directory)
    assert config, "Configuration is empty or invalid"
    return config, output_directory


def drug_discovery_validation_workflow(config, output_directory):
    a_term = config["GLOBAL_SETTINGS"].get("A_TERM", "")
    assert a_term, "A_TERM is not defined in the configuration"
    try:
        df = read_tsv_to_dataframe(skim.skim_no_km_workflow(config, output_directory))
        assert not df.empty, "The dataframe is empty"
        if not api_cost_estimator(df, config):
            return

        results_list = []
        test.test_openai_connection()
        for index, row in df.iterrows():
            result_dict = process_single_row(row, config)
            results_list.append(result_dict)
            print(f"Processed row {index + 1} ({row['b_term']}) of {len(df)}")

        assert results_list, "No results were processed"
        write_to_json(results_list, config["OUTPUT_JSON"], output_directory)
        print(f"Analysis results have been saved to {config['OUTPUT_JSON']}")
        json_file_path = os.path.join(output_directory, config["OUTPUT_JSON"])
        with open(json_file_path, "r") as f:
            json_data = json.load(f)
        result = process_json(json_data, config)
        if result:
            save_to_json(result, config, output_directory)
    except Exception as e:
        print(f"Error occurred during processing: {e}")


def marker_list_workflow_handler(config, output_directory):
    km_file_paths = skim.marker_list_workflow(
        config=config, output_directory=output_directory
    )
    marker_list_filtration(
        km_file_paths=km_file_paths,
        output_directory=output_directory,
        config=config,
    )


def post_km_analysis_workflow(config, output_directory):
    try:
        df = synergy_dfr_preprocessing(config)
        df.reset_index(drop=True, inplace=True)
        assert not df.empty, "The dataframe is empty"

        if not api_cost_estimator(df, config):
            return
        # take the last row for testing purposes
        df = df.iloc[[1]]
        results = {}
        test.test_openai_connection()
        for index, row in df.iterrows():
            term = row["term"]
            result_dict = process_single_row(row, config)
            if term not in results:
                results[term] = [result_dict]
            else:
                results[term].append(result_dict)
            print(f"Processed row {index + 1} ({row['term']}) of {len(df)}")
        assert results, "No results were processed"
        write_to_json(results, config["OUTPUT_JSON"], output_directory)
        print(f"Analysis results have been saved to {config['OUTPUT_JSON']}")
    except Exception as e:
        print(f"Error occurred during processing: {e}")


def main_workflow():
    logging.basicConfig(level=logging.INFO)
    config, output_directory = initialize_workflow()
    job_type = config.get("JOB_TYPE", "")

    if job_type == "drug_discovery_validation":
        drug_discovery_validation_workflow(config, output_directory)
    elif job_type == "marker_list":
        marker_list_workflow_handler(config, output_directory)
    elif job_type == "post_km_analysis":
        post_km_analysis_workflow(config, output_directory)
    else:
        print("JOB_TYPE does not match known workflows.")


if __name__ == "__main__":
    main_workflow()
