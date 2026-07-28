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
import glob
import argparse
from collections import defaultdict


def extract_cluster_number(folder_name):
    """Extract cluster number from folder name, e.g. '...clust3...' -> 3."""
    match = re.search(r'clust(\d+)', folder_name)
    if match:
        return int(match.group(1))
    return None


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
            print(hypothesis)
            gene, cell_type = parse_hypothesis(hypothesis)
            print(gene, cell_type)
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
    args = parser.parse_args()

    if args.combine:
        all_cluster_folders = get_all_folders(args.output_dir, args.pattern)

        if not all_cluster_folders:
            print(f"No matching folders found in {args.output_dir} with pattern '{args.pattern}'")
            return

        print(f"Found {len(all_cluster_folders)} cluster(s) to process (--combine mode):")
        for clust_num in sorted(all_cluster_folders.keys()):
            folder_list = all_cluster_folders[clust_num]
            print(f"  Cluster {clust_num}: {len(folder_list)} folder(s)")
            for folder, ts in folder_list:
                print(f"    - {os.path.basename(folder)}")

        # Collect all output rows
        output_rows = []

        for clust_num in sorted(all_cluster_folders.keys()):
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
        for clust_num in sorted(cluster_folders.keys()):
            print(f"  Cluster {clust_num}: {os.path.basename(cluster_folders[clust_num][0])}")

        # Collect all output rows
        output_rows = []

        for clust_num in sorted(cluster_folders.keys()):
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


if __name__ == '__main__':
    main()
