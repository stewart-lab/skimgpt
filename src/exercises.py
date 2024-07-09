def determine_censor_year_exercise5(index):
    if 0 <= index <= 8:
        return 1995
    elif index == 9:
        return 1985
    elif 10 <= index <= 13:
        return 1987
    elif 14 <= index <= 17:
        return 1989
    return 2024 


def rename_terms(df, index):
    if 0 <= index <= 8:
        a_term = "Alzheimer's Disease"
        b_terms = ["Normal gait", "Onset", "Depression", "Central nervous system disease", "Dermal atrophy"]
        c_terms = ["Indomethacin", "Estrogen"]
        cycle = index // len(b_terms)
        b_term = b_terms[index % len(b_terms)]
        c_term = c_terms[cycle] if cycle < len(c_terms) else "Unknown"
    elif index == 9:
        a_term = "Raynaud's Disease"
        b_term = "Blood viscosity"
        c_term = "Dietary fish oil"
    elif 10 <= index <= 13:
        a_term = "Migraine"
        b_terms = ["Seizures", "Depression", "Muscle spasms", "Tension"]
        b_term = b_terms[index - 10]
        c_term = "Magnesium"
    elif 14 <= index <= 17:
        a_term = "Somatomedin C"
        b_terms = ["Normal gait", "Paraganglioma", "Glucose intolerance", "Cognitive abnormality"]
        b_term = b_terms[index - 14]
        c_term = "Arginine"
    
    df.at[index, 'a_term'] = a_term
    df.at[index, 'b_term'] = b_term
    df.at[index, 'c_term'] = c_term
    df.at[index, 'censor_year'] = determine_censor_year_exercise5(index)
    

def test_example_3(df, i, csv_file_path):
    # Load the PMIDs from the CSV file with converters to ensure correct format
    pmids = pd.read_csv(csv_file_path, converters={"A-B PMIDs": lambda x: x, 
                                                   "B-C PMIDs": lambda x: x, 
                                                   "A-C PMIDs": lambda x: x})
    print(f"Processing row {i + 1} of {len(pmids)}")
    print(pmids.iloc[i]["A-B PMIDs"])  # Print the A-B PMIDs for this row
    print(pmids.iloc[i]["B-C PMIDs"])  # Print the B-C PMIDs for this row
    print(pmids.iloc[i]["A-C PMIDs"])  # Print the A-C PMIDs for this row
    print(pmids)
    print(pmids["A-B PMIDs"])
    print(pmids["B-C PMIDs"])
    print(pmids["A-C PMIDs"])

    # Function to convert a comma-separated string of numbers into a list of integers
    def convert_to_list(pmid_string):
        # Ensure pmid_string is treated as a string
        pmid_string = str(pmid_string)
        if pd.isna(pmid_string) or pmid_string.strip() == '':
            return []  # Return an empty list if the data is NaN or an empty string
        try:
            # Remove any leading/trailing whitespace and split the string by commas
            pmid_list = pmid_string.strip().split(',')
            # Map each split string to an integer, stripping any extra spaces around the numbers
            return [int(pmid.strip()) for pmid in pmid_list if pmid.strip()]
        except ValueError:
            print(f"Warning: Non-integer value encountered in the data: {pmid_string}")
            return []

    ab_pmids = convert_to_list(pmids.iloc[i]["A-B PMIDs"])
    bc_pmids = convert_to_list(pmids.iloc[i]["B-C PMIDs"])
    ac_pmids = convert_to_list(pmids.iloc[i]["A-C PMIDs"])

    # Correct way to assign values to avoid SettingWithCopyWarning
    df.at[0, "ab_pmid_intersection"] = ab_pmids
    df.at[0, "bc_pmid_intersection"] = bc_pmids
    df.at[0, "ac_pmid_intersection"] = ac_pmids if ac_pmids else []


    print(df[["b_term", "ab_pmid_intersection", "bc_pmid_intersection", "ac_pmid_intersection"]])
    return df

def get_rons_list(df, file1, file2, sort_column):
    # Read the first file and convert it into a list
    with open(file1, 'r') as f:
        ab_pmid_intersection = f.read().split()
    
    # Read the second file and convert it into a list
    with open(file2, 'r') as f:
        bc_pmid_intersection = f.read().split()
    
    # Replace the columns in the DataFrame with the new lists
    df["ab_pmid_intersection"] = [ab_pmid_intersection] * len(df)
    df["bc_pmid_intersection"] = [bc_pmid_intersection] * len(df)
    
    # Sort the DataFrame by a specified column and in descending order
    df = df.sort_values(by=sort_column, ascending=False)
    
    # Filter rows where both intersections are not empty lists
    valid_rows = df[
        (df["bc_pmid_intersection"].apply(lambda x: x != [])) &
        (df["ab_pmid_intersection"].apply(lambda x: x != []))
    ]
    
    return valid_rows