#!/usr/bin/env python3
"""One-shot Triton throughput sweep.

Reads PMIDs from a relevance-step input TSV (column ``ab_pmid_intersection``),
fetches abstracts from PubMed, caches them on disk, builds the production
relevance prompt for each abstract, then runs the same fixed prompt set through
``TritonClient.generate_batch`` at several ``max_workers`` settings and prints
a comparison table.

Usage:
    python tools/bench_triton.py \\
        --tsv output/output_.../debug/km_with_gpt_..._output_filtered.tsv \\
        [--n 500] [--workers 5,10,20,30,50]
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd
from Bio import Entrez

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from skimgpt.triton_client import TritonClient  # noqa: E402

CACHE_PATH = REPO_ROOT / "tools" / ".bench_abstracts_cache.json"
RESULTS_CSV = REPO_ROOT / "tools" / "bench_triton_results.csv"

PROMPT_TEMPLATE = (
    "Abstract: {abstract}\n"
    "Hypothesis: {hyp}\n"
    "Instructions: Classify this abstract as either 0 (Not Relevant) or 1 "
    "(Relevant) for evaluating the provided hypothesis.\n"
    "Score: "
)
BENCH_HYPOTHESIS = "The main cause of Schizophrenia is due to dopamine signaling."


def collect_pmids(tsv_path: Path) -> list[str]:
    df = pd.read_csv(tsv_path, sep="\t")
    if "ab_pmid_intersection" not in df.columns:
        raise SystemExit(f"TSV {tsv_path} missing ab_pmid_intersection column")
    pmids: list[str] = []
    for raw in df["ab_pmid_intersection"]:
        for p in ast.literal_eval(raw):
            pmids.append(str(p))
    seen, deduped = set(), []
    for p in pmids:
        if p.isdigit() and p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped


def fetch_abstracts(pmids: list[str], email: str, api_key: str) -> dict[str, str]:
    Entrez.email = email
    Entrez.api_key = api_key
    cache: dict[str, str] = {}
    if CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text())
    missing = [p for p in pmids if p not in cache]
    if not missing:
        return {p: cache[p] for p in pmids}
    print(f"Fetching {len(missing)} abstracts from PubMed "
          f"({len(pmids) - len(missing)} cached)...")
    BATCH = 200
    for i in range(0, len(missing), BATCH):
        batch = missing[i:i + BATCH]
        with Entrez.efetch(db="pubmed", id=batch, retmode="xml",
                           rettype="abstract") as h:
            recs = Entrez.read(h)
        for paper in recs.get("PubmedArticle", []):
            pmid = str(paper["MedlineCitation"]["PMID"])
            art = paper["MedlineCitation"]["Article"]
            text = " ".join(art.get("Abstract", {}).get("AbstractText", []))
            if text:
                cache[pmid] = text
        time.sleep(0.34)
        print(f"  fetched {min(i + BATCH, len(missing))}/{len(missing)}")
    CACHE_PATH.write_text(json.dumps(cache))
    return {p: cache[p] for p in pmids if p in cache}


def build_prompts(abstracts: dict[str, str], n: int, seed: int = 0) -> list[str]:
    pmids = sorted(abstracts.keys())
    random.Random(seed).shuffle(pmids)
    chosen = pmids[:n]
    return [PROMPT_TEMPLATE.format(abstract=abstracts[p], hyp=BENCH_HYPOTHESIS)
            for p in chosen]


KV_EXHAUSTED_SIGNATURES = ("NoFreeBlocksError", "Engine loop is not running")


def _classify_error(r: dict) -> str:
    if "error" not in r:
        return "ok"
    if r["error"] == "timeout":
        return "timeout"
    body = r.get("response_body", "") or ""
    if any(sig in body for sig in KV_EXHAUSTED_SIGNATURES):
        return "kv_exhausted"
    status = r.get("status_code")
    if status and 500 <= status < 600:
        return "http_5xx"
    if status:
        return f"http_{status}"
    return "other"


def make_client(cfg: dict, pool_size: int) -> TritonClient:
    rf = cfg["relevance_filter"]
    return TritonClient(
        server_url=rf.get("SERVER_URL"),
        model_name=rf.get("MODEL_NAME"),
        temperature=rf["TEMPERATURE"],
        top_p=rf["TOP_P"],
        max_tokens=1,
        pool_connections=pool_size,
        pool_maxsize=pool_size,
    )


def engine_probe(cfg: dict) -> tuple[bool, str]:
    """Single-prompt round-trip — confirms vLLM engine is serving, not just Triton."""
    client = make_client(cfg, pool_size=5)
    if not client.check_server_health():
        return False, "health/ready returned non-200"
    r = client.generate("Abstract: x. Hypothesis: y. Instructions: 0 or 1. Score: ")
    if "error" in r:
        klass = _classify_error(r)
        body = (r.get("response_body") or "")[:200]
        return False, f"probe failed ({klass}): {body}"
    return True, "ok"


def run_one_setting(prompts: list[str], workers: int, cfg: dict) -> dict:
    pool_size = max(20, workers + 5)
    client = make_client(cfg, pool_size)
    t0 = time.time()
    results = client.generate_batch(prompts, max_workers=workers,
                                    show_progress=True)
    elapsed = time.time() - t0
    n = len(prompts)
    classes: dict[str, int] = {}
    for r in results:
        k = _classify_error(r)
        classes[k] = classes.get(k, 0) + 1
    errors = n - classes.get("ok", 0)
    return {
        "workers": workers,
        "n": n,
        "elapsed_s": round(elapsed, 2),
        "throughput_rps": round(n / elapsed, 2) if elapsed > 0 else 0.0,
        "avg_latency_s": round(elapsed * workers / n, 3) if n else 0.0,
        "errors": errors,
        "timeouts": classes.get("timeout", 0),
        "kv_exhausted": classes.get("kv_exhausted", 0),
        "http_5xx": classes.get("http_5xx", 0),
        "other_errors": errors - classes.get("timeout", 0)
                        - classes.get("kv_exhausted", 0)
                        - classes.get("http_5xx", 0),
        "pool_size": pool_size,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True, type=Path)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--workers", default="5,10,15,20",
                    help="Comma-separated max_workers values. Default suits "
                         "server config gpu_memory_utilization=0.92 + "
                         "max_num_seqs=10. Pre-fix server, use '2,3,5,8'.")
    ap.add_argument("--cooldown", type=float, default=5.0,
                    help="Seconds between sweep settings.")
    ap.add_argument("--config", type=Path,
                    default=REPO_ROOT / "config.json")
    ap.add_argument("--secrets", type=Path,
                    default=REPO_ROOT / "secrets.json")
    ap.add_argument("--email", default="jfreeman@morgridge.org")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    secrets = json.loads(args.secrets.read_text())

    pmids = collect_pmids(args.tsv)
    print(f"PMIDs in TSV: {len(pmids)}")

    target_n = max(args.n * 2, 600)  # fetch extra to survive missing abstracts
    sampled = pmids[:target_n] if len(pmids) > target_n else pmids
    abstracts = fetch_abstracts(sampled, args.email, secrets["PUBMED_API_KEY"])
    print(f"Abstracts available: {len(abstracts)}")

    prompts = build_prompts(abstracts, args.n)
    if len(prompts) < args.n:
        print(f"Warning: only {len(prompts)} prompts available (requested {args.n})")
    print(f"Built {len(prompts)} prompts (hypothesis fixed)")

    ok, why = engine_probe(cfg)
    if not ok:
        raise SystemExit(f"Pre-flight engine probe failed: {why}\n"
                         f"Server appears unhealthy — aborting before sweep.")
    print("Pre-flight engine probe: ok")

    worker_values = [int(w) for w in args.workers.split(",")]
    rows = []
    aborted = False
    for i, w in enumerate(worker_values):
        if i > 0:
            print(f"Cooldown {args.cooldown}s...")
            time.sleep(args.cooldown)
            ok, why = engine_probe(cfg)
            if not ok:
                print(f"!! Engine probe failed before workers={w}: {why}")
                print("!! Aborting remainder of sweep.")
                aborted = True
                break
        print(f"\n=== max_workers={w} ===")
        row = run_one_setting(prompts, w, cfg)
        rows.append(row)
        if row["kv_exhausted"] > 0:
            pct = 100 * row["kv_exhausted"] / row["n"]
            print(f"!! KV exhaustion at workers={w}: "
                  f"{row['kv_exhausted']}/{row['n']} ({pct:.1f}%) requests "
                  f"hit NoFreeBlocksError. Engine likely wedged.")
            print("!! Aborting remainder of sweep — don't keep hammering it.")
            aborted = True
            break

    if not rows:
        raise SystemExit("No settings completed.")

    print("\n" + "=" * 100)
    hdr = f"{'workers':>8} {'n':>5} {'elapsed_s':>10} {'rps':>8} "
    hdr += f"{'avg_lat_s':>10} {'errors':>7} {'timeout':>8} "
    hdr += f"{'kv_exh':>7} {'5xx':>5} {'pool':>5}"
    print(hdr)
    print("-" * 100)
    for r in rows:
        print(f"{r['workers']:>8} {r['n']:>5} {r['elapsed_s']:>10} "
              f"{r['throughput_rps']:>8} {r['avg_latency_s']:>10} "
              f"{r['errors']:>7} {r['timeouts']:>8} "
              f"{r['kv_exhausted']:>7} {r['http_5xx']:>5} "
              f"{r['pool_size']:>5}")
    print("=" * 100)
    if aborted:
        print("Sweep ABORTED before completion — see warnings above.")

    with RESULTS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults written to {RESULTS_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
