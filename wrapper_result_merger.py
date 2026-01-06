import argparse
import csv
import json
import logging
import os
import re
import sys

def clean_model_name(s: str) -> str:
    return s.replace('-', '_')

def get_config_info(cfg_path: str):
    with open(cfg_path) as f:
        cfg = json.load(f)
    jt    = cfg.get("JOB_TYPE", "unknown").strip()
    model = cfg.get("GLOBAL_SETTINGS", {}).get("MODEL", "model").strip()
    return jt, clean_model_name(model)

def setup_logger(parent_dir: str, job_type: str):
    logger = logging.getLogger("SKiM-GPT-wrapper")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    fmt = "%(asctime)s - SKiM-GPT - %(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(parent_dir, f"{job_type}_wrapper.log"))
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def parse_results_file(fn: str, job_type: str):
    rows = []
    with open(fn) as f:
        rd = csv.DictReader(f, delimiter="\t")
        for r in rd:
            iter_number = r.get("Iteration", "").strip()
            if not iter_number:
                iter_number = "1"

            rel   = r.get("Relationship", "").strip()
            score = r.get("Score",        "").strip()
            parts = [p.strip().strip("'") for p in rel.split(" - ")]
            hyp = r.get("Hypothesis", "").strip()
            print(hyp)
            if job_type == "km_with_gpt":
                terms = [p.strip(".") for p in hyp.split(" ")]
                print(terms)
                row = {
                    "Hypothesis": hyp,
                    "support": r.get("support", "").strip(),
                    "refute": r.get("refute", "").strip(),
                    "inconclusive": r.get("inconclusive", "").strip(),
                    "Score": score,
                    "A_term": terms[0],
                    "B_term": terms[-1]
                }
                
            elif job_type == "skim_with_gpt":
                if len(parts) < 3: continue
                row = {
                    "A_term": parts[0],
                    "B_term": parts[1],
                    "C_term": parts[2],
                    "Score":   score
                }
            elif job_type == "km_with_gpt_direct_comp":
                if len(parts) < 3: continue
                row = {
                    "A_term":  parts[0],
                    "B1_term": parts[1],
                    "B2_term": parts[2],
                    "SOC": r.get("SOC", "").strip(),
                    "Abstracts Supporting Hypothesis 1": r.get("Abstracts Supporting Hypothesis 1", "").strip(),
                    "Abstracts Supporting Hypothesis 2": r.get("Abstracts Supporting Hypothesis 2", "").strip(),
                    "Abstracts Supporting Neither Hypothesis or are Inconclusive": r.get("Abstracts Supporting Neither Hypothesis or are Inconclusive", "").strip(),
                    "Score":   score
                }
            else:
                if len(parts) < 2: continue
                row = {
                    "A_term": parts[0],
                    "B_term": parts[1],
                    "Score":  score
                }

            row["iter_number"] = iter_number
            rows.append(row)
    return rows

def merge_results(parent_dir: str, logger: logging.Logger):
    cfg_path  = os.path.join(parent_dir, "config.json")
    job_type, model = get_config_info(cfg_path)

    if job_type == "skim_with_gpt":
        fieldnames = ["A_term","B_term","C_term","censor_year","iter_number",f"{model}_score"]
    elif job_type == "km_with_gpt":
        fieldnames = ["censor_year","Hypothesis","A_term","B_term","support","refute","inconclusive","iter_number",f"{model}_score"]
    elif job_type == "km_with_gpt_direct_comp":
        fieldnames = ["A_term","B1_term","B2_term","SOC","Abstracts Supporting Hypothesis 1","Abstracts Supporting Hypothesis 2","Abstracts Supporting Neither Hypothesis or are Inconclusive","censor_year","iter_number",f"{model}_score"]
    else:
        fieldnames = ["A_term","B_term","censor_year","iter_number",f"{model}_score"]
        

    output_root = os.path.join(parent_dir, "output")
    if not os.path.isdir(output_root):
        logger.error(f"No output/ folder found under {parent_dir}")
        sys.exit(1)

    merged = []
    for cy_dir in sorted(os.listdir(output_root)):
        cy_path = os.path.join(output_root, cy_dir)
        if not os.path.isdir(cy_path):
            continue

        m = re.search(r'cy(\d+)', cy_dir)
        cy = m.group(1) if m else ""

        results_txt = os.path.join(cy_path, "results.txt")
        if not os.path.isfile(results_txt):
            results_txt = os.path.join(cy_path, "results.tsv")
            if not os.path.isfile(results_txt):
                logger.error(f"Skipping {cy_dir}: no results.txt or results.tsv found")
                continue

        logger.info(f"Merging {results_txt}")
        for row in parse_results_file(results_txt, job_type):
            row["censor_year"]    = cy
            row[f"{model}_score"] = row.pop("Score")
            merged.append(row)

    out_path = os.path.join(parent_dir, f"{job_type}_wrapper_results.tsv")
    with open(out_path, "w", newline="") as wf:
        wr = csv.DictWriter(wf, fieldnames=fieldnames, delimiter="\t")
        wr.writeheader()
        for r in merged:
            wr.writerow(r)

    logger.info(f"Merged results written to {out_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-parent_dir", required=True,
                   help="Top‐level wrapper output directory")
    args = p.parse_args()

    parent_dir = args.parent_dir
    cfg_path   = os.path.join(parent_dir, "config.json")
    job_type, _ = get_config_info(cfg_path)
    logger = setup_logger(parent_dir, job_type)
    logger.info("Starting wrapper_result_merger")

    try:
        merge_results(parent_dir, logger)
    except Exception as e:
        logger.error(f"Merge failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
