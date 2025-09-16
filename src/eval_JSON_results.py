import os
import json
import re
import sys

def _extract_and_write_dch_results(directory):
    """Read structured results produced by DCH runs and emit a compact results.tsv.

    Output columns:
    - score
    - decision
    - H1 (count)
    - H2 (count)
    - neither (count)
    - both (count)
    - total_relevant_abstracts
    - hypothesis1
    - hypothesis2
    """
    # Prefer a structured results.json at the directory root
    data = None
    results_json_path = os.path.join(directory, "results.json")
    if os.path.isfile(results_json_path):
        try:
            data = json.load(open(results_json_path, encoding="utf-8"))
        except Exception:
            data = None

    # If missing, look under results/ for a single structured output JSON
    if data is None:
        results_dir = os.path.join(directory, "results")
        if os.path.isdir(results_dir):
            json_files = [
                os.path.join(results_dir, f)
                for f in os.listdir(results_dir)
                if f.endswith(".json")
            ]
            for jf in json_files:
                try:
                    candidate = json.load(open(jf, encoding="utf-8"))
                except Exception:
                    continue
                # Accept if it looks like the expected structured output
                to_check = candidate if isinstance(candidate, list) else [candidate]
                ok = any(isinstance(it, dict) and (
                    "Hypothesis_Comparison" in it or isinstance(it.get("Hypothesis_Comparison"), dict)
                ) for it in to_check)
                if ok:
                    data = candidate
                    break

    if not data:
        # Nothing to write
        out_path = os.path.join(directory, "results.tsv")
        with open(out_path, "w", encoding="utf-8") as outf:
            outf.write("\t".join([
                "score", "decision", "H1", "H2", "neither", "both",
                "total_relevant_abstracts", "hypothesis1", "hypothesis2"
            ]) + "\n")
        return

    # Normalize to a list of records
    records = data if isinstance(data, list) else [data]

    out_rows = []
    for rec in records:
        # Handle shape where the object is wrapped inside "Hypothesis_Comparison"
        hc = rec.get("Hypothesis_Comparison") if isinstance(rec, dict) else None
        if hc is None and isinstance(rec, dict):
            hc = rec  # try treating the dict directly
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
            count_h1 = int(tallies.get("support_H1", 0) or 0)
            count_h2 = int(tallies.get("support_H2", 0) or 0)
            count_neither = int(tallies.get("neither_or_inconclusive", 0) or 0)
            per_abs = result_entry.get("per_abstract", []) or []
            if not isinstance(per_abs, list):
                per_abs = []
            # Prefer explicit tallies for 'both' if present, else compute from per_abstract
            if "both" in tallies and tallies.get("both") is not None:
                try:
                    count_both = int(tallies.get("both") or 0)
                except Exception:
                    count_both = 0
            else:
                try:
                    count_both = sum(1 for it in per_abs if isinstance(it, dict) and it.get("label") == "both")
                except Exception:
                    count_both = 0
            # Prefer top-level total_relevant_abstracts if provided
            total_relevant = rec.get("total_relevant_abstracts")
            if not isinstance(total_relevant, int):
                total_relevant = len(per_abs)

            out_rows.append([
                str(score),
                str(decision),
                str(count_h1),
                str(count_h2),
                str(count_neither),
                str(count_both),
                str(total_relevant),
                str(hypothesis1),
                str(hypothesis2),
            ])

    out_path = os.path.join(directory, "results.tsv")
    with open(out_path, "w", encoding="utf-8") as outf:
        outf.write("\t".join([
            "score", "decision", "H1", "H2", "neither", "both",
            "total_relevant_abstracts", "hypothesis1", "hypothesis2"
        ]) + "\n")
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

    # If DCH mode, use structured results and write a DCH-specific TSV, then return
    if is_dch:
        _extract_and_write_dch_results(directory)
        # Do not write results.txt in DCH mode
        return

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
                        # Extract score
                        score_match = re.search(
                            r"Score:\s*([-+]?\d+|N/A)",
                            detail,
                            re.IGNORECASE,
                        )
                        if not score_match:
                            continue
                        score = score_match.group(1).strip()
                        # Extract SOC and abstract counts if present
                        soc_match = re.search(r"SOC:\s*(\d+)", detail)
                        hyp1_match = re.search(r"#Abstracts supporting hypothesis 1:\s*(\d+)", detail)
                        hyp2_match = re.search(r"#Abstracts supporting hypothesis 2:\s*(\d+)", detail)
                        neu_match = re.search(r"#Abstracts supporting neither hypothesis or are inconclusive:\s*(\d+)", detail)
                        soc = soc_match.group(1) if soc_match else ""
                        hyp1 = hyp1_match.group(1) if hyp1_match else ""
                        hyp2 = hyp2_match.group(1) if hyp2_match else ""
                        neu = neu_match.group(1) if neu_match else ""
                        # build the row dict
                        row = {
                            "Relationship_Type": outer_key,
                            "Relationship":     Relationship,
                            "Score":            score,
                            "Iteration":        iter_number,
                            # include SOC and counts for direct-comp
                            "SOC":              soc,
                            "Abstracts Supporting Hypothesis 1":      hyp1,
                            "Abstracts Supporting Hypothesis 2":      hyp2,
                            "Abstracts Supporting Neither Hypothesis or are Inconclusive":          neu
                        }
                        results.append(row)

    # 3) write results.txt, injecting the Iteration column only if needed
    out_path = os.path.join(directory, "results.txt")
    # Determine if we have direct-comp entries (with SOC)
    has_direct = any(r.get('Relationship_Type') == 'A_B1_B2_Relationship' for r in results)
    with open(out_path, "w", encoding="utf-8") as outf:
        # build header
        headers = ["Relationship_Type", "Relationship", "Score"]
        if has_iterations:
            headers.append("Iteration")
        if has_direct:
            headers.extend(["SOC", "Abstracts Supporting Hypothesis 1", "Abstracts Supporting Hypothesis 2", "Abstracts Supporting Neither Hypothesis or are Inconclusive"])
        outf.write("\t".join(headers) + "\n")
        # write rows
        for r in results:
            cols = [r.get(h, "") for h in ["Relationship_Type", "Relationship", "Score"]]
            if has_iterations:
                cols.append(r.get("Iteration", ""))
            if has_direct:
                cols.extend([
                    r.get("SOC", ""),
                    r.get("Abstracts Supporting Hypothesis 1", ""),
                    r.get("Abstracts Supporting Hypothesis 2", ""),
                    r.get("Abstracts Supporting Neither Hypothesis or are Inconclusive", "")
                ])
            outf.write("\t".join(cols) + "\n")

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
