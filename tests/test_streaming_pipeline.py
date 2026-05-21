"""Refactor A: streaming PubMed-fetch / Triton-infer pipeline.

The streaming path bypasses the all-at-once ``infer`` callable and submits
each prompt to a single-prompt inference function as soon as its abstract
arrives. These tests pin down:

1. ``fetch_abstracts_iter`` yields per-batch dicts and applies censor-year
   filtering per batch (matches the legacy ``fetch_abstracts`` behaviour).
2. ``_streaming_fetch_and_infer`` writes answers into the correct flat
   positions even when PubMed returns PMIDs out of order — the downstream
   reshape relies on position-stable arrays.
3. PMIDs that PubMed doesn't return get empty answers (mask=0 downstream).
4. A single PMID present at multiple positions (e.g. shared between AB and
   AC intersections in SKiM) is classified for each position independently.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from skimgpt.pubmed_fetcher import PubMedFetcher
from skimgpt.relevance_helper import _streaming_fetch_and_infer
from skimgpt.utils import RaggedTensor


# ---- fetch_abstracts_iter ----------------------------------------------------

def _make_fetcher(censor_lower=2000, censor_upper=2030, batches=None,
                  pmid_years=None):
    """Build a PubMedFetcher without invoking __init__ (which does Entrez DNS).

    We stub out the only attributes ``fetch_abstracts_iter`` touches:
    ``_batch_pmids``, ``_fetch_batch``, ``config``, ``pmid_years``, ``validate_pmids``.
    """
    f = PubMedFetcher.__new__(PubMedFetcher)
    f.config = SimpleNamespace(
        censor_year_lower=censor_lower,
        censor_year_upper=censor_upper,
    )
    f.pmid_years = pmid_years or {}
    f._batch_pmids = lambda pmids, batch_size=200: [pmids]  # one batch
    f.validate_pmids = lambda pmids: [str(p) for p in pmids if str(p).isdigit()]
    f._fetch_batch = MagicMock(side_effect=batches or [])
    return f


def test_fetch_abstracts_iter_yields_batch_by_batch():
    pmids = ["1", "2", "3", "4"]
    f = _make_fetcher(
        pmid_years={"1": 2010, "2": 2011, "3": 2012, "4": 2013},
        batches=[
            {"pmids": ["1", "2"], "contents": ["a1", "a2"]},
            {"pmids": ["3", "4"], "contents": ["a3", "a4"]},
        ],
    )
    # Override _batch_pmids to produce two batches
    f._batch_pmids = lambda p, batch_size=200: [p[:2], p[2:]]

    yielded = list(f.fetch_abstracts_iter(pmids))

    assert len(yielded) == 2, "Should yield one dict per PubMed batch"
    assert yielded[0] == {"1": "a1", "2": "a2"}
    assert yielded[1] == {"3": "a3", "4": "a4"}


def test_fetch_abstracts_iter_filters_by_censor_years():
    pmids = ["1", "2", "3"]
    f = _make_fetcher(
        censor_lower=2005, censor_upper=2015,
        pmid_years={"1": 2000, "2": 2010, "3": 2020},  # 1 too old, 3 too new
        batches=[{"pmids": ["1", "2", "3"], "contents": ["old", "mid", "new"]}],
    )
    yielded = list(f.fetch_abstracts_iter(pmids))
    assert yielded == [{"2": "mid"}], (
        "PMIDs outside the censor window must be filtered per batch, "
        f"got {yielded}"
    )


def test_fetch_abstracts_iter_empty_batch_skipped():
    """A batch with all PMIDs out of year range should be skipped entirely
    (no empty dict yielded), so downstream consumers can iterate cleanly."""
    pmids = ["1", "2"]
    f = _make_fetcher(
        censor_lower=2020, censor_upper=2030,
        pmid_years={"1": 2000, "2": 2001},  # both too old
        batches=[{"pmids": ["1", "2"], "contents": ["a", "b"]}],
    )
    assert list(f.fetch_abstracts_iter(pmids)) == []


# ---- _streaming_fetch_and_infer ---------------------------------------------

def _streaming_fetcher(batches: list[dict[str, str]]) -> SimpleNamespace:
    """A fake fetcher whose ``fetch_abstracts_iter`` yields the given batches."""
    fetcher = SimpleNamespace()
    fetcher.fetch_abstracts_iter = lambda _pmids: iter(batches)
    return fetcher


def test_streaming_preserves_position_when_batches_arrive_out_of_order():
    """PMID positions in all_pmids are the contract for downstream reshape.
    Even when PubMed returns the later-positioned PMIDs first, the answers
    must land at the right indices."""
    all_pmids = RaggedTensor(["100", "200", "300", "400"])
    all_hyps = RaggedTensor(["h0", "h1", "h2", "h3"])

    # PubMed returns the later batch first, then the earlier one
    fetcher = _streaming_fetcher([
        {"300": "abs_300", "400": "abs_400"},
        {"100": "abs_100", "200": "abs_200"},
    ])

    def infer_echo(prompt: str) -> dict:
        # Returns the PMID embedded in the prompt — lets us verify positioning
        for pmid in ("100", "200", "300", "400"):
            if f"abs_{pmid}" in prompt:
                return {"text_output": f"ans_{pmid}"}
        return {"text_output": "miss"}

    abstracts, answers, num_fetched = _streaming_fetch_and_infer(
        fetcher, all_pmids, all_hyps, infer_echo, max_workers=4,
    )

    assert num_fetched == 4
    assert abstracts.data == ["abs_100", "abs_200", "abs_300", "abs_400"]
    assert answers.data == ["ans_100", "ans_200", "ans_300", "ans_400"]


def test_streaming_handles_pmids_not_returned_by_pubmed():
    """Missing PMIDs → empty abstract + empty answer at their position
    (downstream postProcess turns empty into mask=0)."""
    all_pmids = RaggedTensor(["100", "200", "300"])
    all_hyps = RaggedTensor(["h0", "h1", "h2"])
    fetcher = _streaming_fetcher([{"100": "abs_100", "300": "abs_300"}])  # 200 missing

    infer_one = lambda p: {"text_output": "1"}

    abstracts, answers, num_fetched = _streaming_fetch_and_infer(
        fetcher, all_pmids, all_hyps, infer_one, max_workers=2,
    )

    assert num_fetched == 2
    assert abstracts.data == ["abs_100", "", "abs_300"]
    assert answers.data == ["1", "", "1"], (
        "Missing PMIDs must occupy their position with empty answer, "
        "not be dropped (would shift downstream indexing)."
    )


def test_streaming_same_pmid_at_multiple_positions():
    """One PMID can appear in both AB and AC intersections (SKiM). Each
    position needs its own (abstract, hypothesis, answer) — they share the
    abstract text but get classified independently against different
    hypotheses."""
    all_pmids = RaggedTensor(["42", "99", "42"])  # pmid 42 at positions 0 and 2
    all_hyps = RaggedTensor(["hyp_at_0", "hyp_at_1", "hyp_at_2"])
    fetcher = _streaming_fetcher([{"42": "abs_42", "99": "abs_99"}])

    seen_prompts: list[str] = []
    def infer_capture(p: str) -> dict:
        seen_prompts.append(p)
        return {"text_output": str(len(seen_prompts))}

    abstracts, answers, num_fetched = _streaming_fetch_and_infer(
        fetcher, all_pmids, all_hyps, infer_capture, max_workers=2,
    )

    assert num_fetched == 2  # only 2 unique PMIDs fetched
    assert abstracts.data == ["abs_42", "abs_99", "abs_42"]
    # Three inferences fire — one per position, not one per unique PMID
    assert len(seen_prompts) == 3
    # Each position's prompt must use its own hypothesis
    assert "hyp_at_0" in seen_prompts[0] or "hyp_at_2" in seen_prompts[0]
    assert any("hyp_at_1" in p for p in seen_prompts)


def test_streaming_propagates_inference_errors_as_empty():
    """A single-prompt infer that raises must not abort the whole pipeline.
    The failed slot gets empty answer (logged warning); other slots succeed."""
    all_pmids = RaggedTensor(["1", "2"])
    all_hyps = RaggedTensor(["h", "h"])
    fetcher = _streaming_fetcher([{"1": "a1", "2": "a2"}])

    def flaky(p):
        if "a1" in p:
            raise RuntimeError("Triton hiccup")
        return {"text_output": "ok"}

    abstracts, answers, _ = _streaming_fetch_and_infer(
        fetcher, all_pmids, all_hyps, flaky, max_workers=2,
    )

    assert answers.data == ["", "ok"]
