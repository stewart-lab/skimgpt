#!/usr/bin/env python3
"""
Parse kmGPT results.tsv files to summarize gene-cell type support per cluster.

For each cluster folder matching the pattern *_gamms1_clust<N>_*:
  1. Extract gene name and cell type from the Hypothesis column
  2. For each cell type, compute average Score (excluding N/A and 0)
  3. Identify genes with positive scores (supporting) and negative scores (refuting)
  4. Write a summary output file

Usage:
    python parse_kmgpt_results.py [--output_dir OUTPUT_DIR] [--pattern GLOB_PATTERN]
    python parse_kmgpt_results.py --combine   # combine all folders per cluster
"""

import os
import re
import csv
import json
import glob
import argparse
from collections import defaultdict


def extract_cluster_number(folder_name):
    """
    Extract cluster key from folder name, e.g. '...clust3...' -> 3.
    Falls back to the folder name minus the 'output_<timestamp>_' prefix,
    e.g. 'output_20260929134145_merkel_cell-FibroA_2015-2026_terra'
         -> 'merkel_cell-FibroA_2015-2026_terra'.
    """
    match = re.search(r'clust(\d+)', folder_name)
    if match:
        return int(match.group(1))
    label = re.sub(r'^output_\d{14}_?', '', folder_name)
    return label or None


def cluster_sort_key(key):
    """Sort numeric cluster keys first, then string labels."""
    return (isinstance(key, str), key)


def extract_timestamp(folder_name):
    """Extract timestamp from folder name for sorting, e.g. '20260406144251'."""
    match = re.search(r'output_(\d{14})', folder_name)
    if match:
        return match.group(1)
    return "0"


def parse_hypothesis(hypothesis):
    """
    Parse gene name and cell type from hypothesis string.
    Example: 'the gene E2F8 is a marker for the cell type Retinal progenitor cells.'
    Returns: ('E2F8', 'Retinal progenitor cells')
    """
    # Extract gene name: word(s) after "the gene"
    gene_match = re.search(r'The gene (\S+)', hypothesis)
    # Extract cell type: everything after "the cell type" up to the first period
    celltype_match = re.search(r'the cell type ([^.]+)', hypothesis)
    if not celltype_match:
        # Alternate wording: 'is a marker for the merkel cell type, ...'
        celltype_match = re.search(r'for the (.+?) cell type', hypothesis)

    gene = gene_match.group(1) if gene_match else None
    cell_type = celltype_match.group(1).strip() if celltype_match else None

    return gene, cell_type


def parse_results_file(filepath):
    """
    Parse a results.tsv file and return per-cell-type statistics.

    Returns a dict keyed by cell_type:
        {
            'scores': [list of numeric, non-NA, non-zero scores],
            'supporting_genes': set of genes with positive score,
            'refuting_genes': set of genes with negative score,
        }
    """
    celltype_data = defaultdict(lambda: {
        'scores': [],
        'supporting_genes': set(),
        'refuting_genes': set(),
    })

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            hypothesis = row.get('Hypothesis', '')
            score_str = row.get('Score', 'N/A').strip()
            gene, cell_type = parse_hypothesis(hypothesis)
            if gene is None or cell_type is None:
                continue

            # Parse score
            if score_str == 'N/A' or score_str == '':
                continue

            try:
                score = float(score_str)
            except ValueError:
                continue

            data = celltype_data[cell_type]

            # Collect non-zero scores for average calculation
            if score != 0:
                data['scores'].append(score)

            # Categorize genes
            if score > 0:
                data['supporting_genes'].add(gene)
            elif score < 0:
                data['refuting_genes'].add(gene)

    return celltype_data


def get_latest_folders(output_dir, pattern):
    """
    Find all matching folders and return only the latest one per cluster number.
    Returns dict: {clust_num: (folder_path, timestamp)}
    """
    folders = sorted(glob.glob(os.path.join(output_dir, pattern)))

    # Group by cluster number, keeping only the latest (by timestamp)
    cluster_folders = {}
    for folder in folders:
        folder_name = os.path.basename(folder)
        clust_num = extract_cluster_number(folder_name)
        if clust_num is None:
            continue
        results_path = os.path.join(folder, 'results.tsv')
        if not os.path.isfile(results_path):
            continue

        timestamp = extract_timestamp(folder_name)
        if clust_num not in cluster_folders or timestamp > cluster_folders[clust_num][1]:
            cluster_folders[clust_num] = (folder, timestamp)

    return cluster_folders


def get_all_folders(output_dir, pattern):
    """
    Find all matching folders and return ALL of them grouped by cluster number.
    Returns dict: {clust_num: [(folder_path, timestamp), ...]}
    """
    folders = sorted(glob.glob(os.path.join(output_dir, pattern)))

    cluster_folders = defaultdict(list)
    for folder in folders:
        folder_name = os.path.basename(folder)
        clust_num = extract_cluster_number(folder_name)
        if clust_num is None:
            continue
        results_path = os.path.join(folder, 'results.tsv')
        if not os.path.isfile(results_path):
            continue

        timestamp = extract_timestamp(folder_name)
        cluster_folders[clust_num].append((folder, timestamp))

    # Sort each cluster's folders by timestamp
    for clust_num in cluster_folders:
        cluster_folders[clust_num].sort(key=lambda x: x[1])

    return dict(cluster_folders)


def merge_celltype_data(all_data):
    """
    Merge multiple celltype_data dicts (from parse_results_file) into one.
    Combines scores lists and gene sets across all runs.
    """
    merged = defaultdict(lambda: {
        'scores': [],
        'supporting_genes': set(),
        'refuting_genes': set(),
    })

    for celltype_data in all_data:
        for cell_type, data in celltype_data.items():
            merged[cell_type]['scores'].extend(data['scores'])
            merged[cell_type]['supporting_genes'].update(data['supporting_genes'])
            merged[cell_type]['refuting_genes'].update(data['refuting_genes'])

    return merged


def load_gene_json(folder, a_term, b_term, iteration):
    """
    Load the km_with_gpt.json for one gene/iteration and pull out the
    score rationale and PMIDs labelled as supporting or refuting.
    Returns dict with 'rationale', 'supporting_pmids', 'refuting_pmids'
    (empty values if the file is missing).
    """
    info = {'rationale': '', 'supporting_pmids': [], 'refuting_pmids': []}
    json_path = os.path.join(folder, 'results', f'iteration_{iteration}',
                             f'{a_term}_{b_term}_km_with_gpt.json')
    if not os.path.isfile(json_path):
        return info

    with open(json_path, 'r') as f:
        entries = json.load(f)

    rationale = []
    for entry in entries:
        for result in entry.get('A_B_Relationship', {}).get('Result', []):
            rationale.extend(result.get('score_rationale') or [])
            for abstract in result.get('per_abstract', []):
                pmid = str(abstract.get('pmid', ''))
                if abstract.get('label') == 'supports':
                    info['supporting_pmids'].append(pmid)
                elif abstract.get('label') == 'refutes':
                    info['refuting_pmids'].append(pmid)

    info['rationale'] = ' | '.join(r.strip() for r in rationale)
    return info


def parse_gene_results(folder):
    """
    Parse per-gene, per-iteration records from a folder's results.tsv,
    joined with rationale and PMIDs from the matching km_with_gpt.json.
    Returns a list of dicts, one per (gene, iteration) row.
    """
    records = []
    with open(os.path.join(folder, 'results.tsv'), 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            a_term = row.get('Aterm', '').strip()
            b_term = row.get('Bterm', '').strip()
            iteration = row.get('Iteration', '').strip()
            score_str = row.get('Score', 'N/A').strip()

            try:
                score = float(score_str)
            except ValueError:
                score = None

            record = {
                'folder': os.path.basename(folder),
                'a_term': a_term,
                'gene': b_term,
                'iteration': iteration,
                'score': score,
                'support': int(row.get('support') or 0),
                'refute': int(row.get('refute') or 0),
                'inconclusive': int(row.get('inconclusive') or 0),
            }
            record.update(load_gene_json(folder, a_term, b_term, iteration))
            records.append(record)

    return records


def write_gene_tables(folders_by_cluster, summary_path, details_path):
    """
    Write two gene-level tables:
      summary_path: one row per (cluster, gene) with average score across
                    iterations (N/A excluded, 0 included), summed tallies,
                    and unique supporting/refuting PMIDs.
      details_path: one row per (cluster, gene, iteration) with score,
                    tallies, PMIDs, and the score rationale.
    """
    summary_rows = []
    detail_rows = []

    for clust_num in sorted(folders_by_cluster.keys(), key=cluster_sort_key):
        genes = defaultdict(list)
        for folder, _ in folders_by_cluster[clust_num]:
            for record in parse_gene_results(folder):
                genes[(record['a_term'], record['gene'])].append(record)

        for (a_term, gene) in sorted(genes.keys()):
            records = genes[(a_term, gene)]
            scores = [r['score'] for r in records if r['score'] is not None]
            supporting = set()
            refuting = set()

            for r in records:
                supporting.update(r['supporting_pmids'])
                refuting.update(r['refuting_pmids'])
                detail_rows.append({
                    'cluster': clust_num,
                    'folder': r['folder'],
                    'a_term': a_term,
                    'gene': gene,
                    'iteration': r['iteration'],
                    'score': 'N/A' if r['score'] is None else f"{r['score']:g}",
                    'support': r['support'],
                    'refute': r['refute'],
                    'inconclusive': r['inconclusive'],
                    'supporting_pmids': ', '.join(r['supporting_pmids']),
                    'refuting_pmids': ', '.join(r['refuting_pmids']),
                    'score_rationale': r['rationale'],
                })

            summary_rows.append({
                'cluster': clust_num,
                'a_term': a_term,
                'gene': gene,
                'avg_score': f"{sum(scores) / len(scores):.2f}" if scores else 'N/A',
                'n_scored_iterations': len(scores),
                'iteration_scores': ', '.join(
                    'N/A' if r['score'] is None else f"{r['score']:g}" for r in records),
                'total_support': sum(r['support'] for r in records),
                'total_refute': sum(r['refute'] for r in records),
                'total_inconclusive': sum(r['inconclusive'] for r in records),
                'supporting_pmids': ', '.join(sorted(supporting, key=lambda p: (len(p), p))),
                'refuting_pmids': ', '.join(sorted(refuting, key=lambda p: (len(p), p))),
            })

    for path, rows in [(summary_path, summary_rows), (details_path, detail_rows)]:
        if not rows:
            continue
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter='\t')
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nGene summary written to: {summary_path} ({len(summary_rows)} rows)")
    print(f"Gene details written to: {details_path} ({len(detail_rows)} rows)")


def main():
    parser = argparse.ArgumentParser(description='Parse kmGPT results into a summary table.')
    parser.add_argument('--output_dir', default='/w5home/bmoore/kmGPT/output',
                        help='Directory containing output folders (default: %(default)s)')
    parser.add_argument('--pattern', default='output_*gamms1_clust*',
                        help='Glob pattern for matching cluster folders (default: %(default)s)')
    parser.add_argument('--outfile', default='/w5home/bmoore/kmGPT/kmgpt_summary.tsv',
                        help='Output summary file path (default: %(default)s)')
    parser.add_argument('--combine', action='store_true',
                        help='Combine all matching folders per cluster instead of using only the latest')
    parser.add_argument('--gene_outfile', default=None,
                        help='Per-gene summary file (default: <outfile>_genes.tsv)')
    parser.add_argument('--gene_details_outfile', default=None,
                        help='Per-gene, per-iteration details with rationale/PMIDs '
                             '(default: <outfile>_gene_details.tsv)')
    args = parser.parse_args()

    if args.combine:
        all_cluster_folders = get_all_folders(args.output_dir, args.pattern)

        if not all_cluster_folders:
            print(f"No matching folders found in {args.output_dir} with pattern '{args.pattern}'")
            return

        print(f"Found {len(all_cluster_folders)} cluster(s) to process (--combine mode):")
        for clust_num in sorted(all_cluster_folders.keys(), key=cluster_sort_key):
            folder_list = all_cluster_folders[clust_num]
            print(f"  Cluster {clust_num}: {len(folder_list)} folder(s)")
            for folder, ts in folder_list:
                print(f"    - {os.path.basename(folder)}")

        # Collect all output rows
        output_rows = []

        for clust_num in sorted(all_cluster_folders.keys(), key=cluster_sort_key):
            folder_list = all_cluster_folders[clust_num]
            print(f"\nProcessing cluster {clust_num}: combining {len(folder_list)} folder(s)")

            all_data = []
            for folder, _ in folder_list:
                results_path = os.path.join(folder, 'results.tsv')
                celltype_data = parse_results_file(results_path)
                if celltype_data:
                    all_data.append(celltype_data)
                    print(f"  Parsed: {os.path.basename(folder)}")
                else:
                    print(f"  No valid data in: {os.path.basename(folder)}")

            if not all_data:
                print(f"  No valid data found for cluster {clust_num}")
                continue

            celltype_data = merge_celltype_data(all_data)

            for cell_type in sorted(celltype_data.keys()):
                data = celltype_data[cell_type]
                scores = data['scores']
                supporting = sorted(data['supporting_genes'])
                refuting = sorted(data['refuting_genes'])

                if scores:
                    avg_score = sum(scores) / len(scores)
                    avg_score_str = f"{avg_score:.2f}"
                else:
                    avg_score_str = "N/A"

                output_rows.append({
                    'cluster': clust_num,
                    'kmgpt_support': avg_score_str,
                    'supported_cell_type': cell_type,
                    'genes_supporting': ', '.join(supporting) if supporting else 'none',
                    'genes_refuting': ', '.join(refuting) if refuting else 'none',
                })

                print(f"  {cell_type}: avg_score={avg_score_str}, "
                      f"supporting=[{', '.join(supporting)}], "
                      f"refuting=[{', '.join(refuting)}]")

    else:
        cluster_folders = get_latest_folders(args.output_dir, args.pattern)

        if not cluster_folders:
            print(f"No matching folders found in {args.output_dir} with pattern '{args.pattern}'")
            return

        print(f"Found {len(cluster_folders)} cluster(s) to process:")
        for clust_num in sorted(cluster_folders.keys(), key=cluster_sort_key):
            print(f"  Cluster {clust_num}: {os.path.basename(cluster_folders[clust_num][0])}")

        # Collect all output rows
        output_rows = []

        for clust_num in sorted(cluster_folders.keys(), key=cluster_sort_key):
            folder, _ = cluster_folders[clust_num]
            results_path = os.path.join(folder, 'results.tsv')
            print(f"\nProcessing cluster {clust_num}: {os.path.basename(folder)}")

            celltype_data = parse_results_file(results_path)

            if not celltype_data:
                print(f"  No valid data found in {results_path}")
                continue

            for cell_type in sorted(celltype_data.keys()):
                data = celltype_data[cell_type]
                scores = data['scores']
                supporting = sorted(data['supporting_genes'])
                refuting = sorted(data['refuting_genes'])

                if scores:
                    avg_score = sum(scores) / len(scores)
                    avg_score_str = f"{avg_score:.2f}"
                else:
                    avg_score_str = "N/A"

                output_rows.append({
                    'cluster': clust_num,
                    'kmgpt_support': avg_score_str,
                    'supported_cell_type': cell_type,
                    'genes_supporting': ', '.join(supporting) if supporting else 'none',
                    'genes_refuting': ', '.join(refuting) if refuting else 'none',
                })

                print(f"  {cell_type}: avg_score={avg_score_str}, "
                      f"supporting=[{', '.join(supporting)}], "
                      f"refuting=[{', '.join(refuting)}]")

    # Write output file
    fieldnames = ['cluster', 'kmgpt_support', 'supported_cell_type',
                  'genes_supporting', 'genes_refuting']

    with open(args.outfile, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\nSummary written to: {args.outfile}")
    print(f"Total rows: {len(output_rows)}")

    # Gene-level tables (same folder selection as the cluster summary)
    if args.combine:
        folders_by_cluster = all_cluster_folders
    else:
        folders_by_cluster = {k: [v] for k, v in cluster_folders.items()}

    base, ext = os.path.splitext(args.outfile)
    write_gene_tables(folders_by_cluster,
                      args.gene_outfile or f"{base}_genes{ext}",
                      args.gene_details_outfile or f"{base}_gene_details{ext}")


if __name__ == '__main__':
    main()
