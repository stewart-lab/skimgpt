#!/usr/bin/env python3
"""
Merge iteration results from an earlier km-gpt-dch run into a later run on
the same censor-year range, so all iterations can be analyzed together.

Renumbers the source run's iterations to continue after the target run's
existing iterations (e.g. source has iterations 1-2, target has 1-8 ->
source iterations become 9-10 in the target), for every censor-year
subfolder shared by both runs. Then regenerates the target's top-level
km_with_gpt_wrapper_results.tsv via wrapper_result_merger.py.

Usage:
    python merge_dch_iteration_runs.py --source SRC_RUN_DIR --target TGT_RUN_DIR [--dry-run]

Safe to re-run: already-copied iteration folders and already-present rows
are skipped. Modified TSVs are backed up as <file>.bak before the first edit.
"""
import argparse
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path

CY_RE = re.compile(r"_cy(\d{4})$")
ITER_RE = re.compile(r"iteration_(\d+)$")


def find_year_dirs(output_root):
    years = {}
    for p in sorted(output_root.iterdir()):
        if not p.is_dir():
            continue
        m = CY_RE.search(p.name)
        if m:
            years[m.group(1)] = p
    return years


def existing_iterations(year_dir):
    results_dir = year_dir / "results"
    nums = []
    if results_dir.exists():
        for p in results_dir.iterdir():
            m = ITER_RE.match(p.name)
            if m:
                nums.append(int(m.group(1)))
    return sorted(nums)


def read_tsv(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    return rows[0], rows[1:]


def write_tsv(path, header, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def backup_once(path):
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(path, bak)


def merge_year(year, src_dir, tgt_dir, dry_run):
    src_iters = existing_iterations(src_dir)
    tgt_iters = existing_iterations(tgt_dir)
    if not src_iters:
        print(f"[cy{year}] no iteration dirs in source, skipping")
        return

    start = (max(tgt_iters) if tgt_iters else 0) + 1
    mapping = {old: start + i for i, old in enumerate(src_iters)}

    for old, new in mapping.items():
        src_iter_dir = src_dir / "results" / f"iteration_{old}"
        dst_iter_dir = tgt_dir / "results" / f"iteration_{new}"
        if dst_iter_dir.exists():
            print(f"[cy{year}] {dst_iter_dir.name} already exists in target, skipping copy")
            continue
        print(f"[cy{year}] copy results/iteration_{old} -> results/iteration_{new}")
        if not dry_run:
            shutil.copytree(src_iter_dir, dst_iter_dir)

    src_tsv = src_dir / "results.tsv"
    tgt_tsv = tgt_dir / "results.tsv"
    if not (src_tsv.exists() and tgt_tsv.exists()):
        print(f"[cy{year}] WARNING: missing results.tsv on one side, skipping tsv merge")
        return mapping

    src_header, src_rows = read_tsv(src_tsv)
    tgt_header, tgt_rows = read_tsv(tgt_tsv)
    if src_header != tgt_header:
        print(f"[cy{year}] WARNING: results.tsv header mismatch, skipping tsv merge")
        return mapping

    iter_idx = src_header.index("Iteration")
    existing_vals = {r[iter_idx] for r in tgt_rows}
    new_rows = []
    for r in src_rows:
        old_it = int(r[iter_idx])
        new_it = mapping.get(old_it)
        if new_it is None:
            continue
        r = list(r)
        r[iter_idx] = str(new_it)
        if r[iter_idx] in existing_vals:
            print(f"[cy{year}] iteration {new_it} already in results.tsv, skipping row")
            continue
        new_rows.append(r)

    if new_rows:
        print(f"[cy{year}] appending {len(new_rows)} row(s) to results.tsv")
        if not dry_run:
            backup_once(tgt_tsv)
            write_tsv(tgt_tsv, tgt_header, tgt_rows + new_rows)

    return mapping


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="Earlier run dir with fewer iterations (copied FROM)")
    ap.add_argument("--target", required=True, help="Later run dir to merge INTO (becomes the combined run)")
    ap.add_argument("--dry-run", action="store_true", help="Show what would happen without changing anything")
    ap.add_argument("--skip-rebuild", action="store_true",
                     help="Don't regenerate the top-level wrapper_results.tsv afterward")
    ap.add_argument("--python", default="/w5home/bmoore/miniconda3/envs/kmgpt/bin/python3",
                     help="Python interpreter used to run wrapper_result_merger.py")
    args = ap.parse_args()

    src_root = Path(args.source).resolve()
    tgt_root = Path(args.target).resolve()

    src_years = find_year_dirs(src_root / "output")
    tgt_years = find_year_dirs(tgt_root / "output")

    missing = sorted(set(src_years) - set(tgt_years))
    if missing:
        sys.exit(f"Years present in source but missing in target: {missing}")

    for year in sorted(src_years):
        merge_year(year, src_years[year], tgt_years[year], args.dry_run)

    if args.skip_rebuild:
        print("Skipping top-level wrapper_results.tsv rebuild (--skip-rebuild)")
        return

    top_tsv = tgt_root / "km_with_gpt_wrapper_results.tsv"
    if top_tsv.exists():
        backup_once(top_tsv)
    merger_script = Path(__file__).resolve().parent / "wrapper_result_merger.py"
    cmd = [args.python, str(merger_script), "-parent_dir", str(tgt_root)]
    print(f"\nRegenerating top-level results: {' '.join(cmd)}")
    if not args.dry_run:
        subprocess.run(cmd, check=True)
    print("\nDone." + (" (dry run, no files were changed)" if args.dry_run else f"\nMerged run is in: {tgt_root}"))


if __name__ == "__main__":
    main()
