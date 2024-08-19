import pandas as pd
import random
import csv
import os


def save_drugs_to_files(csv_path, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read the CSV file
    with open(csv_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)

        # Iterate through each row
        for row in reader:
            drug = row["Drug"]

            # Create the output file path
            output_file = os.path.join(output_dir, "leakage_b_terms")

            # Append the drug name to the file
            with open(output_file, "a") as f:
                f.write(drug + "\n")

    print(f"Drugs have been saved to {output_file}")


# Usage


# Load the CSV file into a DataFrame
def load_data(filepath):
    return pd.read_csv(filepath, engine="python")


# Function to return a list of drugs
def get_drugs(data):
    return data["Drug"].tolist()


# Function to get neutral text associated with a specific drug
def get_neutral_text(data, drug):
    return data[data["Drug"] == drug]["Neutral text"].iloc[0]


# Function to get negative text associated with a specific drug
def get_negative_text(data, drug):
    return data[data["Drug"] == drug]["Negative text"].iloc[0]


# Write the list of drugs to a file
def write_drugs_to_file(drugs, filename):
    with open(filename, "w") as file:
        for drug in drugs:
            file.write(drug + "\n")


def load_filtered_data(filepath):
    return pd.read_csv(filepath, sep="\t")


def update_ab_pmid_intersection(filtered_data, leakage_data, text_type="neutral"):
    assert text_type.lower() in [
        "neutral",
        "negative",
        "empty",
    ], "text_type must be 'neutral', 'negative', or 'empty'"

    if text_type.lower() == "empty":
        # Set all ab_pmid_intersection values to an empty string
        filtered_data["ab_pmid_intersection"] = ""
        return filtered_data

    text_column = f"{text_type.capitalize()} text"

    for index, row in filtered_data.iterrows():
        b_term = row["b_term"].split("|")[0].strip()  # Take the part before the pipe
        matching_drugs = leakage_data["Drug"].str.lower() == b_term.lower()

        if matching_drugs.any():
            text = leakage_data[matching_drugs][text_column].iloc[0]
            filtered_data.at[index, "ab_pmid_intersection"] = text

    return filtered_data


def save_updated_data(data, output_filepath):
    data.to_csv(output_filepath, sep="\t", index=False)


# Example usage
if __name__ == "__main__":
    filepath = "/w5home/jfreeman/kmGPT/src/leakage.csv"
    data = load_data(filepath)

    drugs = get_drugs(data)
    print("List of Drugs (b_terms):", drugs)

    # Write drugs to a file
    output_file = "b_terms.txt"
    write_drugs_to_file(drugs, output_file)
    print(f"Drugs have been written to {output_file}.")

    # Example to get neutral and negative text for a specific drug
    drug = "abemaciclib"  # Replace with any drug from the list
    if drug in drugs:
        neutral_text = get_neutral_text(data, drug)
        negative_text = get_negative_text(data, drug)

        print(f"Neutral text for {drug}:", neutral_text)
        print(f"Negative text for {drug}:", negative_text)
    else:
        print(f"{drug} not found in the data.")
