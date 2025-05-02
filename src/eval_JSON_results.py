#!/usr/bin/env python3

import os
import json
import re
import sys


def extract_and_write_scores(directory):
    results = []
    # Define the outer keys to look for
    outer_keys = ["A_B_C_Relationship", "A_C_Relationship", "A_B_Relationship", "A_B1_B2_Relationship"]

    # 1) detect whether iterations were enabled
    cfg_path      = os.path.join(directory, "config.json")
    has_iterations = False
    if os.path.isfile(cfg_path):
        try:
            cfg        = json.load(open(cfg_path))
            iterations = cfg.get("GLOBAL_SETTINGS", {}).get("iterations", None)
            if isinstance(iterations, int) and iterations > 1:
                has_iterations = True
        except Exception:
            pass
    # fallback: look for iteration_N subdirectories
    if not has_iterations:
        for name in os.listdir(directory):
            if name.startswith("iteration_") and os.path.isdir(os.path.join(directory, name)):
                has_iterations = True
                break

    # 2) walk all JSON files and capture an 'Iteration' for each
    for root, dirs, files in os.walk(directory):
        for fname in files:
            if not fname.endswith(".json") or fname == "config.json":
                continue

            # determine iteration number from path if needed
            iter_number = ""
            if has_iterations:
                rel_root = os.path.relpath(root, directory)
                for part in rel_root.split(os.sep):
                    if part.startswith("iteration_"):
                        iter_number = part.split("_", 1)[1]
                        break

            with open(os.path.join(root, fname), encoding="utf-8") as json_file:
                try:
                    data = json.load(json_file)
                except json.JSONDecodeError:
                    continue

            for entry in data:
                for outer_key in outer_keys:
                    relationship_data = entry.get(outer_key)
                    if not relationship_data:
                        continue
                    Relationship   = relationship_data.get("Relationship", "").strip()
                    score_details  = relationship_data.get("Result", [])
                    for detail in score_details:
                        match = re.search(
                            r"\**Score\**:\**\s*\**([-+]?\d+|N/A)\**",
                            detail,
                            re.IGNORECASE,
                        )
                        if not match:
                            continue
                        score = match.group(1).strip()
                        # build the row
                        results.append({
                            "Relationship_Type": outer_key,
                            "Relationship":     Relationship,
                            "Score":            score,
                            "Iteration":        iter_number
                        })

    # 3) write results.txt, injecting the Iteration column only if needed
    out_path = os.path.join(directory, "results.txt")
    with open(out_path, "w", encoding="utf-8") as outf:
        if has_iterations:
            outf.write("Relationship_Type\tRelationship\tScore\tIteration\n")
            for r in results:
                outf.write(
                    f"{r['Relationship_Type']}\t"
                    f"{r['Relationship']}\t"
                    f"{r['Score']}\t"
                    f"{r['Iteration']}\n"
                )
        else:
            outf.write("Relationship_Type\tRelationship\tScore\n")
            for r in results:
                outf.write(
                    f"{r['Relationship_Type']}\t"
                    f"{r['Relationship']}\t"
                    f"{r['Score']}\n"
                )

    print(f"Results written to {out_path}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python eval_JSON_results.py <directory_path>")
        sys.exit(1)

    directory_path = sys.argv[1]

    if not os.path.isdir(directory_path):
        print(f"Error: The path '{directory_path}' is not a valid directory.")
        sys.exit(1)

    extract_and_write_scores(directory_path)


if __name__ == "__main__":
    main()
