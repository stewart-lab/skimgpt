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


def post_km_analysis_workflow(config, output_directory):
    try:
        df = synergy_dfr_preprocessing(config)
        df.reset_index(drop=True, inplace=True)
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
    except Exception as e:
        print(f"Error occurred during processing: {e}")
