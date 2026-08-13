"""DCH_FIX_SAMPLE_ACROSS_ITERATIONS — locks in the fixed-sample-across-iterations contract.

This is the complement to DCH_SAMPLE_SEED (see test_dch_sample_seed.py): that
flag deliberately keeps iterations *different* from each other (reproducible
per-run, varying per-iteration) so N iterations measure something. This flag
does the opposite on purpose — it holds the abstract sample identical across
every iteration of a run, so that any variation across iterations can only
come from the LLM itself, not from sampling a different set of abstracts each
time. Without this test, a refactor that moved the sampling call back inside
the per-iteration path would silently turn the LLM-variance measurement back
into a mix of LLM variance and sampling variance.
"""
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from skimgpt.relevance_helper import run_iterations

DELIMITER = "===END OF ABSTRACT==="


def _pool(prefix: str, count: int) -> list[str]:
    return [
        f"Title {prefix}{i}. Abstract body {prefix}{i}. PMID: {prefix}{i:04d}{DELIMITER}"
        for i in range(count)
    ]


def _dch_out_df(v1_count=200, v2_count=80) -> pd.DataFrame:
    return pd.DataFrame([
        {"a_term": "A1", "b_term": "B1", "ab_abstracts": _pool("1", v1_count)},
        {"a_term": "A2", "b_term": "B2", "ab_abstracts": _pool("2", v2_count)},
    ])


def _dch_config(iterations, fix_sample_across_iterations, seed=None,
                sample_size=10, min_fraction=0.06):
    return SimpleNamespace(
        iterations=iterations,
        current_iteration=999,
        is_dch=True,
        is_km_with_gpt=True,
        is_skim_with_gpt=False,
        dch_fix_sample_across_iterations=fix_sample_across_iterations,
        global_settings={
            "DCH_SAMPLE_SEED": seed,
            "DCH_SAMPLE_SIZE": sample_size,
            "DCH_MIN_SAMPLING_FRACTION": min_fraction,
        },
    )


def _run_and_capture_samples(config, out_df, tmp_path):
    """Run run_iterations with process_single_row stubbed; return each
    iteration's consolidated ab_abstracts string.

    Iterations run concurrently in a thread pool and each rebuilds its own
    single-row DataFrame indexed at 0, so captures are collected into a
    lock-protected list rather than keyed by row identity — keying by row
    identity would let later iterations silently overwrite earlier ones.
    """
    captured = []
    lock = threading.Lock()

    def fake_process_single_row(row, _config):
        with lock:
            captured.append(row["ab_abstracts"])
        return None

    with patch("skimgpt.relevance_helper.process_single_row", fake_process_single_row), \
         patch("skimgpt.relevance_helper.get_hypothesis", lambda **_k: "hyp"), \
         patch("skimgpt.relevance_helper.write_to_json", lambda *_a, **_k: None):
        run_iterations(config, out_df, num_abstracts_fetched=280,
                       output_base_dir=str(tmp_path))

    return captured


def test_fixed_sample_is_identical_across_all_iterations(tmp_path):
    config = _dch_config(iterations=5, fix_sample_across_iterations=True)
    out_df = _dch_out_df()

    samples = _run_and_capture_samples(config, out_df, tmp_path)

    assert len(samples) == 5
    distinct = set(samples)
    assert len(distinct) == 1, "every iteration must score the identical sample"


def test_unfixed_sampling_still_varies_across_iterations(tmp_path):
    """Default (flag off) behavior must be untouched: independent draws per iteration."""
    config = _dch_config(iterations=5, fix_sample_across_iterations=False)
    out_df = _dch_out_df()

    samples = _run_and_capture_samples(config, out_df, tmp_path)

    assert len(samples) == 5
    distinct = set(samples)
    assert len(distinct) > 1, "unfixed iterations drawing identical samples would be a regression"


def test_fixed_sample_reproducible_with_seed(tmp_path_factory):
    """Combining with DCH_SAMPLE_SEED makes the one fixed draw reproducible across runs."""
    out_df = _dch_out_df()

    config_a = _dch_config(iterations=3, fix_sample_across_iterations=True, seed=1234)
    samples_a = _run_and_capture_samples(config_a, out_df, tmp_path_factory.mktemp("run_a"))

    config_b = _dch_config(iterations=3, fix_sample_across_iterations=True, seed=1234)
    samples_b = _run_and_capture_samples(config_b, out_df, tmp_path_factory.mktemp("run_b"))

    assert set(samples_a) == set(samples_b)


def test_fixed_sample_without_seed_varies_across_reruns(tmp_path_factory):
    """Without a seed, the single fixed draw is still random per run (just constant within it)."""
    out_df = _dch_out_df()

    config_a = _dch_config(iterations=3, fix_sample_across_iterations=True)
    samples_a = _run_and_capture_samples(config_a, out_df, tmp_path_factory.mktemp("run_a"))

    config_b = _dch_config(iterations=3, fix_sample_across_iterations=True)
    samples_b = _run_and_capture_samples(config_b, out_df, tmp_path_factory.mktemp("run_b"))

    assert set(samples_a) != set(samples_b)


def test_flag_ignored_for_non_dch_jobs(tmp_path):
    """A stray DCH_FIX_SAMPLE_ACROSS_ITERATIONS=true on a non-DCH job must be a no-op,
    not an attempt to build DCH abstract pools from a DataFrame that doesn't have them."""
    config = SimpleNamespace(
        iterations=2,
        current_iteration=999,
        is_dch=False,
        is_km_with_gpt=True,
        is_skim_with_gpt=False,
        dch_fix_sample_across_iterations=True,
        global_settings={},
    )
    out_df = pd.DataFrame({"a_term": ["A1"], "b_term": ["B1"]})

    with patch("skimgpt.relevance_helper.process_single_row", lambda *_a, **_k: None), \
         patch("skimgpt.relevance_helper.write_to_json", lambda *_a, **_k: None):
        run_iterations(config, out_df, num_abstracts_fetched=0, output_base_dir=str(tmp_path))
