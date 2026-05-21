#!/usr/bin/env python3
"""End-to-end integration smoke test for the relevance + iteration refactors.

Bypasses FastKM (which is heavy and orthogonal to what we changed) by feeding
``run_relevance_analysis`` an existing pre-relevance TSV. Runs with
``iterations=2`` to keep the GPT-5.4-nano cost negligible (~$0.20 total) and
to give the parallel-iterations path a non-trivial input to exercise.

Validates after the run:
  * Both iteration_1/ and iteration_2/ subdirs exist and contain JSON output.
  * The streaming pipeline log line fired (proves Refactor A is wired).
  * "All N iterations completed in X total wallclock" appears with X
    materially less than 2 × per-iteration time (proves Refactor B is wired).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from skimgpt.utils import Config, add_file_handler  # noqa: E402

INPUT_TSV = REPO_ROOT / "output" / "output_20260520112618_Schizophrenia_kmgptdch_2020-2026_o3" / "debug" / "km_with_gpt_Schizophrenia_output_filtered.tsv"
ITERATIONS_OVERRIDE = 2


def main() -> int:
    if not INPUT_TSV.exists():
        raise SystemExit(f"Input TSV not found: {INPUT_TSV}")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = REPO_ROOT / "output" / f"output_{timestamp}_integration_v2refactor"
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Run dir: {run_dir}")

    # Copy the pre-relevance TSV into the fresh output dir. run_relevance_pipeline
    # sets config.km_output_dir = dirname(tsv), so everything (iteration_N/, JSON
    # outputs, filtered TSV) lands here.
    copied_tsv = run_dir / INPUT_TSV.name
    shutil.copy2(INPUT_TSV, copied_tsv)

    # Use the live config.json; just override iterations in-memory so we don't
    # mutate the file on disk.
    config = Config(str(REPO_ROOT / "config.json"))
    config.iterations = ITERATIONS_OVERRIDE
    print(f"Overrode iterations -> {ITERATIONS_OVERRIDE} (was: {config.global_settings.get('iterations')})")

    # Config.load_km_output (called inside run_relevance_analysis) attaches a
    # SKiM-GPT.log file handler to dirname(tsv) = run_dir. We just need to
    # know the path so we can grep it during validate().
    log_path = run_dir / "SKiM-GPT.log"

    from skimgpt.relevance_triton import run_relevance_analysis

    t0 = time.time()
    run_relevance_analysis(config, str(copied_tsv))
    total_wallclock = time.time() - t0
    print(f"\nFull run wallclock: {total_wallclock:.2f}s")

    return validate(run_dir, total_wallclock)


def validate(run_dir: Path, total_wallclock: float) -> int:
    log_candidates = list(run_dir.glob("*.log"))
    log_text = log_candidates[0].read_text() if log_candidates else ""
    failures: list[str] = []
    successes: list[str] = []

    # Refactor A: streaming pipeline must have fired
    if "Streaming pipeline:" in log_text:
        successes.append("Refactor A: streaming pipeline log line found")
    else:
        failures.append("Refactor A: NO streaming pipeline log line — refactor did not wire through")

    # Refactor B: iteration subdirs
    iter_dirs = sorted(run_dir.glob("iteration_*"))
    iter_names = [p.name for p in iter_dirs]
    expected = {f"iteration_{i}" for i in range(1, ITERATIONS_OVERRIDE + 1)}
    if set(iter_names) >= expected:
        successes.append(f"Refactor B: all expected iteration dirs created: {iter_names}")
    else:
        failures.append(f"Refactor B: missing iteration dirs. Got {iter_names}, expected {sorted(expected)}")

    # Each iteration dir should contain at least one JSON file
    for d in iter_dirs:
        jsons = list(d.glob("*.json"))
        if jsons:
            successes.append(f"  {d.name}/: {len(jsons)} JSON file(s)")
        else:
            failures.append(f"  {d.name}/: NO JSON output")

    # Refactor B: parallelism — wallclock claim from the log
    m = re.search(r"All (\d+) iterations completed in ([\d.]+)s total wallclock", log_text)
    if m:
        total_iter_wallclock = float(m.group(2))
        # Per-iteration durations to compute the serial floor
        per_iter = [float(x) for x in re.findall(r"Iteration \d+/\d+ completed in ([\d.]+)s", log_text)]
        if per_iter and len(per_iter) >= 2:
            serial_floor = sum(per_iter)
            speedup = serial_floor / total_iter_wallclock if total_iter_wallclock > 0 else 1.0
            if speedup > 1.3:
                successes.append(
                    f"Refactor B: parallel iterations confirmed. "
                    f"Serial would be {serial_floor:.1f}s, ran in {total_iter_wallclock:.1f}s "
                    f"(speedup {speedup:.2f}×)"
                )
            else:
                failures.append(
                    f"Refactor B: iterations appear serial. "
                    f"Total {total_iter_wallclock:.1f}s vs serial-sum {serial_floor:.1f}s "
                    f"(speedup only {speedup:.2f}×)"
                )
    else:
        failures.append("Refactor B: NO 'All N iterations completed' log line found")

    print("\n=== Validation ===")
    for s in successes:
        print(f"  ✓ {s}")
    for f in failures:
        print(f"  ✗ {f}")
    print(f"\nTotal wallclock: {total_wallclock:.2f}s")
    print(f"Run dir: {run_dir}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
