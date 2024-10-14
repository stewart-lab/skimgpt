import os
import json
import re
import sys


def extract_and_write_scores(directory):
    results = []
    # Define the outer keys to look for
    outer_keys = ["A_B_C_Relationship", "A_C_Relationship", "A_B_Relationship"]

    # Walk through all files in the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json") and file != "config.json":
                # Construct the full file path
                file_path = os.path.join(root, file)
                # Open and load the JSON file
                with open(file_path, "r", encoding="utf-8") as json_file:
                    try:
                        data = json.load(json_file)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON in file {file_path}: {e}")
                        continue

                    # Iterate over each entry in the JSON array
                    for entry in data:
                        # Iterate over the specified outer keys
                        for outer_key in outer_keys:
                            relationship_data = entry.get(outer_key)
                            if relationship_data:
                                Relationship = relationship_data.get(
                                    "Relationship", "No Relationship Provided"
                                )
                                score_details = relationship_data.get("Result", [])
                                for detail in score_details:
                                    # Using regex to find the score pattern
                                    match = re.search(
                                        r"\**Score:\**\s*\**([-+]?\d+)\**", detail
                                    )
                                    if match:
                                        score = match.group(1)
                                        results.append(
                                            {
                                                "Relationship_Type": outer_key,
                                                "Relationship": Relationship,
                                                "Score": score,
                                            }
                                        )

    if not results:
        print("No scores found in the specified directory.")
        return

    # Writing results to results.txt file in the specified directory
    results_file_path = os.path.join(directory, "results.txt")
    with open(results_file_path, "w", encoding="utf-8") as file:
        # Write header
        file.write("Relationship_Type\tRelationship\tScore\n")
        # Write each result
        for result in results:
            file.write(
                f"{result['Relationship_Type']}\t{result['Relationship']}\t{result['Score']}\n"
            )

    print(f"Results written to {results_file_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract and write scores from JSON files."
    )
    parser.add_argument(
        "--directory",
        required=True,
        help="Path to the directory containing JSON result files.",
    )
    args = parser.parse_args()

    extract_and_write_scores(args.directory)


def main():
    if len(sys.argv) != 2:
        print("Usage: python eval_JSON_results.py <directory_path>")
        sys.exit(1)

    directory_path = sys.argv[1]
    extract_and_write_scores(directory_path)


if __name__ == "__main__":
    main()
