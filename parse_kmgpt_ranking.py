#!/usr/bin/env python3
"""
Expand a km-GPT ranking run (rank_wrapper.py output) into a per-comparison table.

Genes are listed in FINAL_RANKING order. Each gene gets one row per pairwise
comparison it took part in, with the opponent, the score from that gene's own
perspective, the outcome, the score rationale, and the PMIDs cited.

Only real comparisons are included (cache hits are skipped), matching the
W-T-L records and avg_score in FINAL_RANKING.txt.

Usage:
    python parse_kmgpt_ranking.py --rank_dir <output_..._rank_...> [--outfile FILE]
"""

import os
import csv
import json
import glob
import argparse
import statistics


def pair_dir_for(rank_dir, pass_record, comparison):
    """Locate the output folder for one comparison, mirroring rank_wrapper.py."""
    pid = comparison['pair_id']
    kind = comparison.get('boundary_kind')
    if kind == 'tie_break':
        pass_dir = 'tie_break_round_robin'
    elif kind in ('escape', 'validate'):
        pass_dir = f"pass_{pass_record['pass']:02d}_{kind}"
    else:
        pass_dir = f"pass_{pass_record['pass']:02d}_{pass_record['phase']}"

    path = os.path.join(rank_dir, pass_dir, 'output', pid)
    if os.path.isdir(path):
        return path
    # Fallback: pair_ids are unique across a run, so search all pass folders
    matches = glob.glob(os.path.join(rank_dir, '*', 'output', pid))
    return matches[0] if matches else None


def parse_pair_folder(pair_dir):
    """
    Read results.tsv and the per-iteration direct_comp JSONs for one pair.
    Scores are from H1 (term_i) perspective. Returns dict with per-iteration
    scores, H1/H2/both PMIDs, and rationale text keyed by iteration.
    """
    info = {'iteration_scores': [], 'h1_pmids': [], 'h2_pmids': [],
            'both_pmids': [], 'rationale': []}
    if pair_dir is None:
        return info

    results_tsv = os.path.join(pair_dir, 'results.tsv')
    if os.path.isfile(results_tsv):
        with open(results_tsv, 'r') as f:
            for row in csv.DictReader(f, delimiter='\t'):
                try:
                    info['iteration_scores'].append((row.get('Iteration', ''), float(row['Score'])))
                except (KeyError, TypeError, ValueError):
                    continue

    label_to_key = {'supports_H1': 'h1_pmids', 'supports_H2': 'h2_pmids', 'both': 'both_pmids'}
    iter_dirs = sorted(glob.glob(os.path.join(pair_dir, 'results', 'iteration_*')),
                       key=lambda p: int(p.rsplit('_', 1)[-1]))
    for iter_dir in iter_dirs:
        iteration = iter_dir.rsplit('_', 1)[-1]
        for json_path in glob.glob(os.path.join(iter_dir, '*_direct_comp.json')):
            with open(json_path, 'r') as f:
                entries = json.load(f)
            for entry in entries:
                for result in entry.get('Hypothesis_Comparison', {}).get('Result', []):
                    text = ' '.join(r.strip() for r in (result.get('score_rationale') or []))
                    if text:
                        info["rationale"].append(f"[iter {iteration}] {text}")
                    for abstract in result.get('per_abstract', []):
                        key = label_to_key.get(abstract.get('label'))
                        pmid = str(abstract.get('pmid', ''))
                        if key and pmid not in info[key]:
                            info[key].append(pmid)

    return info


def gene_outcome(tag, is_term_i):
    """Translate rank_wrapper's outcome tag into win/loss from the gene's view."""
    if tag == 'term_i':
        return 'win' if is_term_i else 'loss'
    if tag == 'term_j':
        return 'loss' if is_term_i else 'win'
    return tag  # tie, no_evidence, error


def fmt(score):
    return f"{score:.2f}" if score is not None else 'N/A'


def main():
    parser = argparse.ArgumentParser(description='Per-comparison table for a km-GPT ranking run.')
    parser.add_argument('--rank_dir', required=True,
                        help='Ranking output folder containing ranking_history.json')
    parser.add_argument('--outfile', default=None,
                        help='Output TSV (default: <rank_dir>/FINAL_RANKING_comparisons.tsv)')
    args = parser.parse_args()

    with open(os.path.join(args.rank_dir, 'ranking_history.json'), 'r') as f:
        history = json.load(f)

    final_ranking = history['final_ranking']
    rank_of = {entry['term']: entry['rank'] for entry in final_ranking}

    # Collect each real comparison once, then attach it to both genes
    comparisons_by_gene = {entry['term']: [] for entry in final_ranking}
    n_pairs = 0
    for pass_record in history['passes']:
        for c in pass_record['comparisons']:
            if c['from_cache']:
                continue
            n_pairs += 1
            pair_info = parse_pair_folder(pair_dir_for(args.rank_dir, pass_record, c))
            for gene, opponent, is_term_i in [(c['term_i'], c['term_j'], True),
                                              (c['term_j'], c['term_i'], False)]:
                comparisons_by_gene.setdefault(gene, []).append({
                    'pass': pass_record['pass'],
                    'phase': c.get('boundary_kind') or pass_record['phase'],
                    'pair_id': c['pair_id'],
                    'opponent': opponent,
                    'is_term_i': is_term_i,
                    'comparison': c,
                    'pair_info': pair_info,
                })

    rows = []
    for entry in final_ranking:
        gene = entry['term']
        record = f"W{entry['wins']}-T{entry['ties']}-L{entry['losses']}"
        # Order each gene's comparisons by the opponent's final rank
        gene_comps = sorted(comparisons_by_gene.get(gene, []),
                            key=lambda x: rank_of.get(x['opponent'], float('inf')))
        for comp in gene_comps:
            c, info, is_term_i = comp['comparison'], comp['pair_info'], comp['is_term_i']

            # Scores are recorded from term_i's view; flip for term_j
            mean_score = c['mean_score']
            if mean_score is not None and not is_term_i:
                mean_score = 100 - mean_score
            iter_scores = [s if is_term_i else 100 - s for _, s in info['iteration_scores']]

            gene_pmids, opp_pmids = ((info['h1_pmids'], info['h2_pmids']) if is_term_i
                                     else (info['h2_pmids'], info['h1_pmids']))

            rows.append({
                'rank': entry['rank'],
                'gene': gene,
                'record': record,
                'avg_score': fmt(entry['avg_score']),
                'insufficient_evidence': entry.get('quarantine_reason') or '',
                'compared_to': comp['opponent'],
                'compared_to_rank': rank_of.get(comp['opponent'], ''),
                'score': fmt(mean_score),
                'iteration_scores': ', '.join(f"{s:g}" for s in iter_scores),
                'outcome': gene_outcome(c['outcome'], is_term_i),
                'pass': f"{comp['pass']}_{comp['phase']}",
                'gene_supporting_pmids': ', '.join(gene_pmids),
                'opponent_supporting_pmids': ', '.join(opp_pmids),
                'both_supporting_pmids': ', '.join(info['both_pmids']),
                'score_rationale': ' || '.join(info['rationale']),
            })

        # Sanity check against FINAL_RANKING's own averages
        scores = [float(r['score']) for r in rows if r['gene'] == gene and r['score'] != 'N/A']
        if entry['avg_score'] is not None and scores and \
                abs(statistics.mean(scores) - entry['avg_score']) > 0.01:
            print(f"Warning: {gene} avg from comparisons ({statistics.mean(scores):.2f}) "
                  f"!= FINAL_RANKING avg ({entry['avg_score']:.2f})")

    outfile = args.outfile or os.path.join(args.rank_dir, 'FINAL_RANKING_comparisons.tsv')
    with open(outfile, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)

    print(f"Parsed {n_pairs} comparisons across {len(final_ranking)} genes")
    print(f"Comparison table written to: {outfile} ({len(rows)} rows)")


if __name__ == '__main__':
    main()
