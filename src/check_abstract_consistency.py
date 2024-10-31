#!/usr/bin/env python3
import os
import sys
import json
import argparse
from collections import defaultdict


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Script to check for inconsistencies in the abstracts used across multiple output directories."
    )
    parser.add_argument(
        "output_dirs",
        metavar="OUTPUT_DIR",
        type=str,
        nargs="+",
        help="Paths to the output directories to be checked.",
    )
    return parser.parse_args()


def collect_urls(output_dirs):
    relationships_dict = defaultdict(lambda: defaultdict(list))

    for dir_path in output_dirs:
        results_path = os.path.join(dir_path, "results")
        if not os.path.isdir(results_path):
            print(
                f"Warning: 'results' subdirectory not found in '{dir_path}'. Skipping."
            )
            continue

        for json_file in [f for f in os.listdir(results_path) if f.endswith(".json")]:
            json_path = os.path.join(results_path, json_file)
            try:
                with open(json_path, "r", encoding="utf-8") as jf:
                    data = json.load(jf)

                items = data if isinstance(data, list) else [data]

                for item in items:
                    relationship = None
                    ab_urls = []
                    bc_urls = []
                    ac_urls = []

                    if "A_B_C_Relationship" in item:
                        abc_rel = item["A_B_C_Relationship"]
                        relationship = abc_rel["Relationship"]
                        ab_urls = abc_rel.get("URLS", {}).get("AB", [])
                        bc_urls = abc_rel.get("URLS", {}).get("BC", [])

                        if "A_C_Relationship" in item:
                            ac_urls = (
                                item["A_C_Relationship"].get("URLS", {}).get("AC", [])
                            )

                    elif "Relationship" in item:
                        relationship = item["Relationship"]
                        urls = item.get("URLS", {})
                        ab_urls = urls.get("AB", [])
                        bc_urls = urls.get("BC", [])
                        ac_urls = urls.get("AC", [])

                    if relationship:
                        relationships_dict[relationship]["AB"].append(
                            (tuple(sorted(ab_urls)), json_file)
                        )
                        relationships_dict[relationship]["BC"].append(
                            (tuple(sorted(bc_urls)), json_file)
                        )
                        relationships_dict[relationship]["AC"].append(
                            (tuple(sorted(ac_urls)), json_file)
                        )

            except Exception as e:
                print(f"Error processing {json_path}: {e}")
                continue

    return relationships_dict


def compare_urls(relationships_dict):
    for relationship, url_types in relationships_dict.items():
        print(f"\nChecking relationship: '{relationship}'")

        for url_type in ["AB", "BC", "AC"]:
            # Create sets of just URLs for comparison
            url_sets = set(urls for urls, _ in url_types[url_type])

            if len(url_sets) > 1:
                print(f"\nInconsistency found in {url_type} URLs:")

                # Create mapping of URLs to their source files
                url_sources = {}
                for urls, source in url_types[url_type]:
                    if urls not in url_sources:
                        url_sources[urls] = []
                    url_sources[urls].append(source)

                for idx, urls in enumerate(url_sets, 1):
                    print(f"  Variation {idx} (from {', '.join(url_sources[urls])}):")
                    # Get unique URLs across all variations
                    all_urls = set().union(*url_sets)

                    for url in sorted(all_urls):
                        if url in urls:
                            print(f"    {url}")
                        else:
                            print(f"    * {url}  (missing in this variation)")
            else:
                print(f"{url_type} URLs consistent")


def main():
    args = parse_arguments()
    valid_dirs = [d for d in args.output_dirs if os.path.isdir(d)]

    if not valid_dirs:
        print("Error: No valid output directories provided")
        sys.exit(1)

    relationships_dict = collect_urls(valid_dirs)

    if not relationships_dict:
        print("No relationships found")
        sys.exit(0)

    compare_urls(relationships_dict)


if __name__ == "__main__":
    main()
