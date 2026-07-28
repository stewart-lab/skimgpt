"""Refactor B: iteration parallelization.

These tests lock in two behaviours we just introduced:

1. ``process_results`` reads ``iteration_number`` from its argument, not from
   ``config.current_iteration``. (If a regression makes it read from config
   again, parallel iterations will race on the path.)
2. ``run_iterations`` executes iterations concurrently, so total wallclock is
   roughly max(per-iteration time), not sum(per-iteration time).
"""
from __future__ import annotations

import os
import time
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from skimgpt.relevance_helper import (
    ITERATION_MAX_PARALLELISM,
    process_results,
    run_iterations,
)


def _stub_config(iterations: int | bool = False) -> SimpleNamespace:
    """Minimal Config double — only the attributes process_results/run_iterations touch."""
    return SimpleNamespace(
        iterations=iterations,
        current_iteration=999,  # poisoned: a regression would surface as iteration_999/
        is_dch=False,
        is_km_with_gpt=False,
        is_skim_with_gpt=False,
    )


def test_process_results_uses_iteration_param_not_config(tmp_path, monkeypatch):
    """Path derivation must follow the iteration_number argument, not config.current_iteration.

    A regression that reads config.current_iteration would write everything
    into iteration_999/ (the poisoned value above) and parallel iterations
    would race on a single directory.
    """
    config = _stub_config(iterations=3)
    captured: list[str] = []

    def fake_process_single_row(*_args, **_kwargs):
        # Skip the LLM call path; we only care which directory got created.
        return None

    monkeypatch.setattr("skimgpt.relevance_helper.process_single_row",
                        fake_process_single_row)

    # Empty df → loop body skipped, but the iteration-subdir setup at the top
    # of process_results still runs. That's the bit we're testing.
    out_df = pd.DataFrame({"a_term": [], "b_term": []})
    process_results(out_df, config, num_abstracts_fetched=0,
                    output_base_dir=str(tmp_path), iteration_number=2)

    assert (tmp_path / "iteration_2").is_dir(), (
        "iteration_number=2 should create iteration_2/, not "
        f"iteration_{config.current_iteration}/"
    )
    assert not (tmp_path / f"iteration_{config.current_iteration}").exists(), (
        "process_results must not read config.current_iteration anymore"
    )


def test_process_results_iteration_zero_uses_base_dir(tmp_path, monkeypatch):
    config = _stub_config(iterations=False)
    monkeypatch.setattr("skimgpt.relevance_helper.process_single_row",
                        lambda *_a, **_kw: None)

    out_df = pd.DataFrame({"a_term": [], "b_term": []})
    process_results(out_df, config, num_abstracts_fetched=0,
                    output_base_dir=str(tmp_path), iteration_number=0)

    # No iteration_N/ subdir created when iteration_number=0
    subdirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert subdirs == [], f"Expected no iteration subdirs, got {subdirs}"


def test_run_iterations_executes_in_parallel(tmp_path, monkeypatch):
    """If iterations ran serially, 4 iterations × 0.5s sleep ≥ 2.0s wallclock.
    Running them in parallel with ITERATION_MAX_PARALLELISM=5 should finish
    in ~0.5s + overhead. We give a generous 1.5s upper bound so this isn't
    flaky on busy CI hosts.
    """
    num_iterations = 4
    sleep_s = 0.5
    config = _stub_config(iterations=num_iterations)
    called_iterations: list[int] = []

    def slow_stub(out_df, config, num_abstracts_fetched, *, output_base_dir,
                  iteration_number):
        called_iterations.append(iteration_number)
        time.sleep(sleep_s)

    monkeypatch.setattr("skimgpt.relevance_helper.process_results", slow_stub)

    t0 = time.time()
    run_iterations(config, pd.DataFrame(), num_abstracts_fetched=0,
                   output_base_dir=str(tmp_path))
    elapsed = time.time() - t0

    serial_floor = num_iterations * sleep_s
    parallel_ceiling = sleep_s * 1.5 + 0.5  # 0.5s headroom for thread setup

    assert elapsed < parallel_ceiling, (
        f"Expected parallel execution (<{parallel_ceiling:.2f}s), "
        f"got {elapsed:.2f}s. Serial would have taken ≥{serial_floor:.1f}s."
    )
    assert sorted(called_iterations) == list(range(1, num_iterations + 1)), (
        f"Each iteration index must run exactly once; got {called_iterations}"
    )


def test_run_iterations_respects_parallelism_cap(tmp_path, monkeypatch):
    """If iterations > ITERATION_MAX_PARALLELISM, the extras must wait — verify
    by checking peak concurrent invocations doesn't exceed the cap."""
    import threading

    num_iterations = ITERATION_MAX_PARALLELISM + 3
    config = _stub_config(iterations=num_iterations)
    in_flight = 0
    peak = 0
    lock = threading.Lock()

    def tracking_stub(*_args, **_kwargs):
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        time.sleep(0.2)
        with lock:
            in_flight -= 1

    monkeypatch.setattr("skimgpt.relevance_helper.process_results", tracking_stub)

    run_iterations(config, pd.DataFrame(), num_abstracts_fetched=0,
                   output_base_dir=str(tmp_path))

    assert peak <= ITERATION_MAX_PARALLELISM, (
        f"Peak concurrent iterations was {peak}, cap is "
        f"{ITERATION_MAX_PARALLELISM}"
    )
