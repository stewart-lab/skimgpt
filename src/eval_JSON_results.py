import os
import sys
import json
import re


def extract_and_write_scores(directory):
    results = []
    # Walk through all files in the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json") and file != "config.json":
                # Construct the full file path
                file_path = os.path.join(root, file)
                # Open and load the JSON file
                with open(file_path, "r") as json_file:
                    data = json.load(json_file)
                    # Iterate over each entry in the JSON array
                    for entry in data:
                        Relationship = entry.get(
                            "Relationship", "No Relationship Provided"
                        )
                        score_details = entry.get("Result", [])
                        for detail in score_details:
                            # Using regex to find the score pattern
                            match = re.search(r"Score: ([-+]?\d+)", detail)
                            if match:
                                score = match.group(1)
                                results.append(
                                    {"Relationship": Relationship, "Score": score}
                                )

    # Writing results to results.txt file in the specified directory
    results_file_path = os.path.join(directory, "results.txt")
    with open(results_file_path, "w") as file:
        for result in results:
            file.write(
                f"Relationship: {result['Relationship']}, Score: {result['Score']}\n"
            )

    print(f"Results written to {results_file_path}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python eval_JSON_results.py <directory_path>")
        sys.exit(1)

    directory_path = sys.argv[1]
    extract_and_write_scores(directory_path)


if __name__ == "__main__":
    main()
