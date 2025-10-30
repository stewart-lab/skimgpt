import os
import json
import re
import sys

def _extract_and_write_structured_output_results(directory):
    """Read structured results (DCH and KM) and emit a compact results.tsv.

    Columns written (some may be blank depending on job type):
    - Score, Decision, Iteration (optional)
    - For DCH: H1, H2, Neither, Both, Total Relevant Abstracts, Hypothesis1, Hypothesis2
    - For KM: Support, Refute, Inconclusive, Hypothesis, Relationship
    """
    # Discover whether iterations are present under results/
    results_root = os.path.join(directory, "results")
    has_iterations = False
    iteration_dirs = []
    if os.path.isdir(results_root):
        for name in os.listdir(results_root):
            sub = os.path.join(results_root, name)
            if name.startswith("iteration_") and os.path.isdir(sub):
                has_iterations = True
                iteration_dirs.append((name.split("_", 1)[1], sub))

    # Determine job type (KM vs DCH) from config.json
    km_mode = False
    try:
        cfg = json.load(open(os.path.join(directory, "config.json")))
        job_type = cfg.get("JOB_TYPE")
        jss = cfg.get("JOB_SPECIFIC_SETTINGS", {}) if isinstance(cfg, dict) else {}
        job_cfg = jss.get(job_type, {}) if isinstance(jss, dict) and job_type else {}
        is_dch = bool(job_cfg.get("is_dch") or cfg.get("is_dch") or cfg.get("GLOBAL_SETTINGS", {}).get("is_dch"))
        km_mode = (job_type == "km_with_gpt") and not is_dch
    except Exception:
        km_mode = False

    # Compute headers based on mode
    if km_mode:
        HEADERS = ["Hypothesis", "Score", "support", "refute", "inconclusive"]
        if has_iterations:
            HEADERS.append("Iteration")
    else:
        HEADERS = ["Score", "Decision"]
        if has_iterations:
            HEADERS.append("Iteration")
        # DCH-specific fields
        HEADERS.extend(["H1", "H2", "Neither", "Both", "Total Relevant Abstracts", "Hypothesis1", "Hypothesis2"])

    # Build list of (iteration_value, results_dir) to scan
    targets = []
    if has_iterations:
        for iter_num, iter_path in iteration_dirs:
            # iteration path is already results/iteration_X
            targets.append((iter_num, iter_path))
    else:
        if os.path.isdir(results_root):
            targets.append(("", results_root))

    out_path = os.path.join(directory, "results.tsv")
    if not targets:
        # Nothing to write except header
        with open(out_path, "w", encoding="utf-8") as outf:
            outf.write("\t".join(HEADERS) + "\n")
        return

    out_rows = []

    def to_int(value, default=0):
        try:
            return int(value)
        except Exception:
            return default
    for iter_value, results_dir in targets:
        json_files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith(".json")]
        for jf in json_files:
            try:
                candidate = json.load(open(jf, encoding="utf-8"))
            except Exception:
                continue
            records = candidate if isinstance(candidate, list) else [candidate]
            for rec in records:
                if not isinstance(rec, dict):
                    continue

                # DCH structured (Hypothesis_Comparison) - only when not KM mode
                if (not km_mode) and "Hypothesis_Comparison" in rec:
                    hc = rec["Hypothesis_Comparison"]
                    if not isinstance(hc, dict):
                        continue
                    hypothesis1 = hc.get("hypothesis1", "")
                    hypothesis2 = hc.get("hypothesis2", "")
                    results_list = hc.get("Result") or []
                    if not isinstance(results_list, list):
                        results_list = [results_list]
                    for result_entry in results_list:
                        if not isinstance(result_entry, dict):
                            continue
                        score = result_entry.get("score", "")
                        decision = result_entry.get("decision", "")
                        tallies = result_entry.get("tallies", {}) or {}
                        count_h1 = to_int(tallies.get("support_H1", 0))
                        count_h2 = to_int(tallies.get("support_H2", 0))
                        count_neither = to_int(tallies.get("neither_or_inconclusive", 0))
                        per_abs = result_entry.get("per_abstract", []) or []
                        if not isinstance(per_abs, list):
                            per_abs = []
                        if "both" in tallies and tallies.get("both") is not None:
                            count_both = to_int(tallies.get("both"), 0)
                        else:
                            try:
                                count_both = sum(1 for it in per_abs if isinstance(it, dict) and it.get("label") == "both")
                            except Exception:
                                count_both = 0
                        total_relevant = rec.get("total_relevant_abstracts")
                        if not isinstance(total_relevant, int):
                            total_relevant = len(per_abs)

                        row = [str(score), str(decision)]
                        if has_iterations:
                            row.append(str(iter_value))
                        # DCH fields
                        row.extend([
                            str(count_h1),
                            str(count_h2),
                            str(count_neither),
                            str(count_both),
                            str(total_relevant),
                            str(hypothesis1),
                            str(hypothesis2),
                        ])
                        # KM fields (blank)
                        row.extend(["", "", "", "", ""]) 
                        out_rows.append(row)
                    continue

                # KM structured (A_B_Relationship) - only when KM mode
                if km_mode:
                    abr = rec.get("A_B_Relationship")
                    if isinstance(abr, dict):
                        hypothesis = abr.get("Hypothesis", "")
                        results_list = abr.get("Result") or []
                        if not isinstance(results_list, list):
                            results_list = [results_list]
                        for result_entry in results_list:
                            if not isinstance(result_entry, dict):
                                continue
                            score = result_entry.get("score", "")
                            tallies = result_entry.get("tallies", {}) or {}
                            support = to_int(tallies.get("support", 0))
                            refute = to_int(tallies.get("refute", 0))
                            inconclusive = to_int(tallies.get("inconclusive", 0))

                            # KM row in requested order: Hypothesis, Score, support, refute, inconclusive, Iteration
                            row = [
                                str(hypothesis),
                                str(score),
                                str(support),
                                str(refute),
                                str(inconclusive),
                            ]
                            if has_iterations:
                                row.append(str(iter_value))
                            out_rows.append(row)

    with open(out_path, "w", encoding="utf-8") as outf:
        outf.write("\t".join(HEADERS) + "\n")
        for row in out_rows:
            outf.write("\t".join(row) + "\n")

def extract_and_write_scores(directory):
    results = []
    # Define the outer keys to look for
    outer_keys = ["A_B_C_Relationship", "A_C_Relationship", "A_B_Relationship", "A_B1_B2_Relationship"]

    # 1) detect whether iterations were enabled
    cfg_path      = os.path.join(directory, "config.json")
    has_iterations = False
    is_dch = False
    if os.path.isfile(cfg_path):
        try:
            cfg        = json.load(open(cfg_path))
            iterations = cfg.get("GLOBAL_SETTINGS", {}).get("iterations", None)
            if isinstance(iterations, int) and iterations > 1:
                has_iterations = True
            # detect DCH mode: prefer nested under JOB_SPECIFIC_SETTINGS for current JOB_TYPE
            job_type = cfg.get("JOB_TYPE")
            jss = cfg.get("JOB_SPECIFIC_SETTINGS", {}) if isinstance(cfg, dict) else {}
            job_cfg = jss.get(job_type, {}) if isinstance(jss, dict) and job_type else {}
            is_dch = bool(job_cfg.get("is_dch")
                          or cfg.get("is_dch")
                          or cfg.get("GLOBAL_SETTINGS", {}).get("is_dch"))
        except Exception:
            pass
    # fallback: look for iteration_N subdirectories
    if not has_iterations:
        for name in os.listdir(directory):
            if name.startswith("iteration_") and os.path.isdir(os.path.join(directory, name)):
                has_iterations = True
                break

    # Always prefer structured output writer (handles DCH and KM)
    _extract_and_write_structured_output_results(directory)
    return

    # Old results.txt writer removed in favor of structured results.tsv


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
