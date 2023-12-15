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
import importlib
import copy
from datetime import datetime
from xml.etree import ElementTree
import skim_and_km_api as skim
import test.test_abstract_comprehension as test

CONFIG_FILE = "./config.json"  # typically "./config.json"


# Ron is using: "./configRMS_needSpecialTunnel.json"
def initialize_workflow():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs("output", exist_ok=True)
    output_directory = os.path.join("output", f"output_{timestamp}")
    os.makedirs(output_directory, exist_ok=True)
    shutil.copy(
        CONFIG_FILE,
        os.path.join(output_directory, CONFIG_FILE),
    )
    config = get_config(output_directory)
    assert config, "Configuration is empty or invalid"
    return config, output_directory


def get_output_json_filename(config, job_settings):
    a_term = config["GLOBAL_SETTINGS"]["A_TERM"]
    output_json_map = {
        "km_with_gpt": f"{a_term}_km_with_gpt.json",
        "marker_list": f"marker_list_numCTerms{config['GLOBAL_SETTINGS'].get('NUM_C_TERMS', '')}.json",
        "post_km_analysis": f"{a_term}_drug_synergy_maxAbstracts{config['GLOBAL_SETTINGS'].get('MAX_ABSTRACTS', '')}.json",
        "drug_discovery_validation": f"{a_term}_censorYear{job_settings.get('skim', {}).get('censor_year', '')}_numCTerms{config['GLOBAL_SETTINGS'].get('NUM_C_TERMS', '')}.json",
        "pathway_augmentation": f"{a_term}_pathway_augmentation.json",
    }

    output_json = output_json_map.get(config["JOB_TYPE"])
    if output_json is None:
        raise ValueError(f"Invalid job type: {config['JOB_TYPE']}")

    return output_json.replace(" ", "_").replace("'", "")


def get_config(output_directory, config_path=CONFIG_FILE):
    with open(config_path, "r") as f:
        config = json.load(f)

    job_settings = config["JOB_SPECIFIC_SETTINGS"].get(config["JOB_TYPE"], {})
    config["OUTPUT_JSON"] = get_output_json_filename(config, job_settings)

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    config["API_KEY"] = api_key

    with open(os.path.join(output_directory, config_path), "w") as f:
        json.dump(config, f, indent=4)

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


def abstract_quality_control(config, pmids, rate_limit, delay):
    min_word_count = config["GLOBAL_SETTINGS"].get(
        "MIN_WORD_COUNT", 100
    )  # Default to 100 if not specified
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
    if not b_term or not a_term:
        logging.error("B term or A term is empty.")
        return []

    api_key = config.get("API_KEY", os.getenv("OPENAI_API_KEY", ""))
    if not api_key:
        raise ValueError("OpenAI API key is not set.")
    openai_client = openai.OpenAI(api_key=api_key)

    responses = []
    job_type = config["JOB_TYPE"]

    if job_type in ["post_km_analysis", "pathway_augmentation", "km_with_gpt"]:
        prompt = generate_prompt(b_term, a_term, consolidated_abstracts, config)
        response = call_openai(openai_client, prompt, config)
        if response:
            responses.append(response)
    elif job_type == "drug_discovery_validation":
        for abstract in consolidated_abstracts:
            prompt = generate_prompt(b_term, a_term, abstract, config)
            response = call_openai(openai_client, prompt, config)
            if response:
                responses.append(response)

    return responses, prompt


def generate_prompt(b_term, a_term, content, config):
    prompt_name = config.get("PROMPT_NAME")
    if not prompt_name:
        raise ValueError("PROMPT_NAME is not specified in the configuration.")

    # Dynamically import the prompts module
    prompts_module = importlib.import_module("prompt_library")

    # Retrieve the function based on the prompt name
    prompt_function = getattr(prompts_module, prompt_name, None)
    if not prompt_function:
        raise ValueError(
            f"Prompt function '{prompt_name}' not found in the prompts module."
        )

    # Call the prompt function
    return prompt_function(b_term, a_term, content)


def handle_rate_limit(e, retry_delay):
    # Extract the retry-after value from the error message if available0-
    retry_after = int(e.response.headers.get("Retry-After", retry_delay))
    logging.warning(f"Rate limit exceeded. Retrying after {retry_after} seconds...")
    time.sleep(retry_after)


def call_openai(client, prompt, config):
    retry_delay = config["GLOBAL_SETTINGS"]["RETRY_DELAY"]
    max_retries = config["GLOBAL_SETTINGS"]["MAX_RETRIES"]
    model = config["GLOBAL_SETTINGS"]["MODEL"]
    max_tokens = config["GLOBAL_SETTINGS"]["MAX_TOKENS"]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a biomedical research analyst.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if content:
                return content
            else:
                logging.error("Received an empty response.")
                time.sleep(retry_delay)

        except openai.RateLimitError as e:
            print("A 429 status code was received; we should back off a bit.")
            time.sleep(retry_delay)
            print(e.__cause__)
        except openai.APIStatusError as e:
            print("Another non-200-range status code was received")
            print(e.status_code)
            print(e.response)
    logging.error("Max retries reached or no valid response received.")
    return None


def synergy_dfr_preprocessing(config):
    csv_path = config["JOB_SPECIFIC_SETTINGS"]["post_km_analysis"]["B_TERMS_FILE"]
    df = pd.read_csv(csv_path)
    desired_columns = ["b_term", "panc & ggp & kras-mapk set", "brd & ggp set"]
    # change the column name "term" to "b_term"
    df.rename(columns={"term": "b_term"}, inplace=True)

    filtered_df = df[desired_columns]

    new_row = {
        "b_term": "CDK9",
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
        filtered_df["b_term"]
        .str.strip()
        .str.lower()
        .isin([term.lower() for term in terms_to_retain])
    ]
    filtered_df.loc[filtered_df["b_term"] == "p akt", "b_term"] = "AKT"
    filtered_df["b_term"] = filtered_df["b_term"].str.upper()
    return filtered_df


def parse_pmids(row, key):
    return ast.literal_eval(row[key])


def get_successful_pmids(pmids, abstracts_data):
    return [pmid for pmid in pmids if pmid in abstracts_data]


def process_abstracts_data(config, pmids):
    abstracts_data = abstract_quality_control(
        config,
        pmids,
        config["GLOBAL_SETTINGS"]["RATE_LIMIT"],
        config["GLOBAL_SETTINGS"]["DELAY"],
    )
    successful_pmids = get_successful_pmids(pmids, abstracts_data)
    consolidated_abstracts = [abstracts_data[pmid][0] for pmid in successful_pmids]
    paper_urls = [abstracts_data[pmid][1] for pmid in successful_pmids]
    publication_years = [abstracts_data[pmid][2] for pmid in successful_pmids]
    return consolidated_abstracts, paper_urls, publication_years


def perform_analysis(job_type, row, config, robust_setting, abstracts_data):
    max_abstracts = config["GLOBAL_SETTINGS"].get("MAX_ABSTRACTS", 10)
    b_term = row["b_term"]
    a_term = config["GLOBAL_SETTINGS"]["A_TERM"]

    pmids = []
    if job_type == "post_km_analysis":
        pmids_panc = parse_pmids(row, "panc & ggp & kras-mapk set")
        pmids_brd = parse_pmids(row, "brd & ggp set")
        pmids = list(set(pmids_panc).union(pmids_brd))[:max_abstracts]
    elif job_type in [
        "drug_discovery_validation",
        "pathway_augmentation",
        "km_with_gpt",
    ]:
        pmids = parse_pmids(row, "ab_pmid_intersection")[:max_abstracts]

    consolidated_abstracts, paper_urls, publication_years = process_abstracts_data(
        config, pmids
    )

    result = prompt = None
    if job_type == "post_km_analysis" and robust_setting.lower() == "true":
        result = perform_robust_analysis(consolidated_abstracts, b_term, config)
    else:
        result, prompt = analyze_abstract_with_gpt4(
            consolidated_abstracts, b_term, a_term, config
        )

    return result, prompt, paper_urls, consolidated_abstracts, publication_years


def process_single_row(row, config):
    job_type = config.get("JOB_TYPE")
    robust_setting = config["JOB_SPECIFIC_SETTINGS"]["post_km_analysis"].get(
        "robust", False
    )

    if job_type not in [
        "post_km_analysis",
        "drug_discovery_validation",
        "pathway_augmentation",
        "km_with_gpt",
    ]:
        print("Invalid job type (caught in process_single_row)")
        return None

    (
        result,
        prompt,
        paper_urls,
        consolidated_abstracts,
        publication_years,
    ) = perform_analysis(job_type, row, config, robust_setting, {})

    return {
        "Term": row["b_term"],
        "Result": result,
        "Prompt": prompt,  # Added prompt here
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


def process_drug_discovery_validation_json(json_data, config):
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
            urls = entry.get("URLs", None)
            years = entry.get("Years", None)

            if term is None or results is None or urls is None or years is None:
                print("Warning: Invalid entry found in JSON data. Skipping.")
                continue

            if term not in term_scores:
                term_scores[term] = 0
                term_counts[term] = 0
                term_details[term] = []  # Initialize list to hold details

            for i, result in enumerate(results):
                term_counts[term] += 1  # Increase the count for this term
                patterns = prompts.drug_process_relationship_scoring(term)
                score = 0
                scoring_sentence = ""
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
                "Details": term_details[term],
            }

        return nested_dict

    except Exception as e:
        print(f"An error occurred while processing the JSON data: {e}")
        return None


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
    max_abstracts = config["GLOBAL_SETTINGS"]["MAX_ABSTRACTS"]
    estimated_cost = 0
    total_calls = 0

    def read_terms_length(file_path):
        return len(skim.read_terms_from_file(file_path))

    if job_type in ["drug_discovery_validation", "pathway_augmentation"]:
        estimated_cost = max_abstracts * len(df) * 0.006
    elif job_type == "km_with_gpt":
        if config["JOB_SPECIFIC_SETTINGS"]["km_with_gpt"]["A_TERM_LIST"]:
            term_length = read_terms_length(
                config["JOB_SPECIFIC_SETTINGS"]["km_with_gpt"]["A_TERMS_FILE"]
            )
        else:
            term_length = 1  # Default value if A_TERM_LIST is not set

        num_b_terms = config["JOB_SPECIFIC_SETTINGS"]["km_with_gpt"]["NUM_B_TERMS"]
        total_calls = max_abstracts * term_length * num_b_terms
        estimated_cost = total_calls * 0.006
    elif job_type == "post_km_analysis":
        robust_setting = (
            config["JOB_SPECIFIC_SETTINGS"]["post_km_analysis"]
            .get("robust", "False")
            .lower()
        )

        if robust_setting == "true":
            total_calls = sum(
                len(eval(row["panc & ggp & kras-mapk set"])) // (max_abstracts // 2)
                + len(eval(row["brd & ggp set"])) // (max_abstracts // 2)
                for _, row in df.iterrows()
            )
        else:
            total_calls = df.apply(
                lambda row: min(
                    len(eval(row["panc & ggp & kras-mapk set"]))
                    + len(eval(row["brd & ggp set"])),
                    max_abstracts,
                ),
                axis=1,
            ).sum()

        estimated_cost = total_calls * 0.06

    user_input = input(
        f"The following job consists of {total_calls} abstracts and will cost roughly ${estimated_cost:.2f} in GPT-4 API calls. Do you wish to proceed? [Y/n]: "
    )
    if user_input.lower() != "y":
        print("Exiting workflow.")
        return False
    return True


def drug_discovery_validation_workflow(config, output_directory):
    a_term = config["GLOBAL_SETTINGS"].get("A_TERM", "")
    assert a_term, "A_TERM is not defined in the configuration"
    try:
        if (
            config["JOB_SPECIFIC_SETTINGS"]["drug_discovery_validation"].get("test")
            == "True"
        ):
            df = test.test_gpt4_leakage()
        else:
            df = read_tsv_to_dataframe(
                skim.skim_no_km_workflow(config, output_directory)
            )
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
        result = process_drug_discovery_validation_json(json_data, config)
        if result:
            save_to_json(result, config, output_directory)
    except Exception as e:
        print(f"Error occurred during processing: {e}")


def extract_term_and_scoring_sentence(json_data):
    extracted_data = {}

    for term_data in json_data.values():
        for entry in term_data:
            term = entry.get("Term")
            results = entry.get("Result", [])

            if term and results:
                # Joining the result strings and extracting the first significant sentence
                full_text = " ".join(results)
                first_sentence = full_text.split(".")[0] + "." if full_text else ""

                # Adding to the extracted data
                extracted_data[term] = first_sentence

    return extracted_data


def marker_list_workflow(config, output_directory):
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
        results = {}
        test.test_openai_connection()
        for index, row in df.iterrows():
            term = row["b_term"]
            result_dict = process_single_row(row, config)
            if term not in results:
                results[term] = [result_dict]
            else:
                results[term].append(result_dict)
            print(f"Processed row {index + 1} ({row['b_term']}) of {len(df)}")
        assert results, "No results were processed"
        write_to_json(results, config["OUTPUT_JSON"], output_directory)
        print(f"Analysis results have been saved to {config['OUTPUT_JSON']}")
    except Exception as e:
        print(f"Error occurred during processing: {e}")


def pathway_augmentation_workflow(config, output_directory):
    try:
        df = read_tsv_to_dataframe(
            config["JOB_SPECIFIC_SETTINGS"]["pathway_augmentation"]["B_TERMS_FILE"]
        )
        # take rows 19 through 24
        df = df.iloc[25:30]
        assert not df.empty, "The dataframe is empty"
        if not api_cost_estimator(df, config):
            return
        results = {}
        test.test_openai_connection()
        for index, row in df.iterrows():
            term = row["b_term"]
            result_dict = process_single_row(row, config)
            if term not in results:
                results[term] = [result_dict]
            else:
                results[term].append(result_dict)
            print(f"Processed row {index + 1} ({row['b_term']}) of {len(df)}")
        assert results, "No results were processed"
        write_to_json(results, config["OUTPUT_JSON"], output_directory)
        print(f"Analysis results have been saved to {config['OUTPUT_JSON']}")
        print("Processing JSON data...")
        json_file_path = os.path.join(output_directory, config["OUTPUT_JSON"])
        with open(json_file_path, "r") as f:
            json_data = json.load(f)
        result = extract_term_and_scoring_sentence(json_data)
        if result:
            save_to_json(result, config, output_directory)
    except Exception as e:
        print(f"Error occurred within pathway_augmentation_workflow: {e}")


def km_with_gpt_workflow(config, output_directory):
    km_file_path = skim.km_with_gpt_workflow(
        config=config, output_directory=output_directory
    )
    df = read_tsv_to_dataframe(km_file_path)
    df = df.iloc[: config["JOB_SPECIFIC_SETTINGS"]["km_with_gpt"]["NUM_B_TERMS"]]
    assert not df.empty, "The dataframe is empty"
    results = {}
    test.test_openai_connection()
    for index, row in df.iterrows():
        term = row["b_term"]
        result_dict = process_single_row(row, config)
        if term not in results:
            results[term] = [result_dict]
        else:
            results[term].append(result_dict)
        print(f"Processed row {index + 1} ({row['b_term']}) of {len(df)}")
    assert results, "No results were processed"
    write_to_json(results, config["OUTPUT_JSON"], output_directory)
    print(f"Analysis results have been saved to {config['OUTPUT_JSON']}")


def main_workflow():
    logging.basicConfig(level=logging.INFO)
    config, output_directory = initialize_workflow()
    job_type = config.get("JOB_TYPE", "")

    if job_type == "km_with_gpt":
        if not api_cost_estimator([], config):
            return
        if config["JOB_SPECIFIC_SETTINGS"]["km_with_gpt"].get("A_TERM_LIST"):
            a_terms = skim.read_terms_from_file(
                config["JOB_SPECIFIC_SETTINGS"]["km_with_gpt"]["A_TERMS_FILE"]
            )
            for a_term in a_terms:
                local_config = copy.deepcopy(config)
                local_config["GLOBAL_SETTINGS"]["A_TERM"] = a_term

                # Dynamically generate output_json name for each a_term
                output_json = f"{a_term}_km_with_gpt.json"
                output_json = output_json.replace(" ", "_").replace("'", "")
                local_config["OUTPUT_JSON"] = output_json

                km_with_gpt_workflow(local_config, output_directory)
        else:
            km_with_gpt_workflow(config, output_directory)
    elif job_type == "drug_discovery_validation":
        drug_discovery_validation_workflow(config, output_directory)
    elif job_type == "marker_list":
        marker_list_workflow(config, output_directory)
    elif job_type == "post_km_analysis":
        post_km_analysis_workflow(config, output_directory)
    elif job_type == "pathway_augmentation":
        pathway_augmentation_workflow(config, output_directory)
    else:
        print("JOB_TYPE does not match known workflows.")


if __name__ == "__main__":
    main_workflow()
