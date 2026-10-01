"""DCH reference PMIDs: lookup from the packaged CSV and forced inclusion in sampling."""
import csv
import random
from importlib import resources
from types import SimpleNamespace

import pandas as pd
import pytest

from skimgpt.reference_pmids import REFERENCE_CSV, reference_pmids_for
from skimgpt.relevance_helper import fetch_dch_reference_abstracts, sample_consolidated_abstracts
from skimgpt.utils import ABSTRACT_DELIMITER, extract_pmids


def _csv_rows():
    path = resources.files("skimgpt").joinpath("data", REFERENCE_CSV)
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _abstract(pmid):
    return f"PMID: {pmid}\nTitle: t{pmid}\nAbstract: words{ABSTRACT_DELIMITER}"


def _config(sample_size=50, **globals_):
    return SimpleNamespace(global_settings={"DCH_SAMPLE_SIZE": sample_size, **globals_})


@pytest.mark.parametrize("row", _csv_rows(), ids=lambda r: f"{r['a term'][:20]}-{r['pubmedID']}")
def test_every_csv_pmid_is_returned_for_its_combo(row):
    assert row["pubmedID"] in reference_pmids_for(row["a term"], row["b1 term"], row["b2 term"])


def test_lookup_ignores_b_order_case_and_extra_synonyms():
    expected = reference_pmids_for("Alzheimer's", "amyloid plaque", "tau pathology")
    assert len(expected) == 10
    assert reference_pmids_for("alzheimer's|AD", "Tau Pathology", "amyloid plaque|Abeta") == expected


def test_lookup_strips_a_term_suffix():
    assert reference_pmids_for("Alzheimer's AND human", "amyloid plaque", "tau pathology",
                               a_term_suffix=" AND human")


def test_same_a_term_with_different_b_terms_stays_separate():
    smt = reference_pmids_for("cancer", "Somatic Mutation Theory|SMT", "Tissue Organization Field Theory")
    stem = reference_pmids_for("cancer", "stem cell", "Stochastic clonal evolution")
    assert smt and stem and not set(smt) & set(stem)


def test_unknown_combo_has_no_references():
    assert reference_pmids_for("Alzheimer's", "amyloid plaque", "microglia") == []


def test_references_always_sampled_and_never_duplicated():
    refs = [_abstract(p) for p in ("900", "901", "5")]  # "5" also sits in pool 1
    pool1 = [_abstract(i) for i in range(100)]
    pool2 = [_abstract(i) for i in range(100, 300)]
    for seed in range(20):
        random.seed(seed)
        text, count, total = sample_consolidated_abstracts(pool1, pool2, _config(50), refs)
        pmids = extract_pmids(text)
        assert pmids[:3] == ["900", "901", "5"]
        assert count == len(pmids) == 50
        assert len(set(pmids)) == 50
        assert total == 2 + 300  # "5" is counted once


def test_no_references_keeps_original_sampling():
    pool1 = [_abstract(i) for i in range(10)]
    random.seed(0)
    without = sample_consolidated_abstracts(pool1, [], _config(5))
    random.seed(0)
    empty = sample_consolidated_abstracts(pool1, [], _config(5), [])
    assert without == empty


class _FakeFetcher:
    def __init__(self, available):
        self.available = available
        self.requested = None

    def fetch_abstracts(self, pmids, min_word_count=None):
        self.requested = list(pmids)
        self.min_word_count = min_word_count
        return {p: _abstract(p) for p in pmids if p in self.available}


def _dch_config(a_term, b1, b2, is_dch=True):
    return SimpleNamespace(
        is_dch=is_dch,
        data=pd.DataFrame({"a_term": [a_term, a_term], "b_term": [b1, b2]}),
        global_settings={"A_TERM_SUFFIX": ""},
        censor_year_lower=2020,
        censor_year_upper=2026,
    )


def test_fetch_returns_only_fetchable_references_in_csv_order():
    config = _dch_config("REM sleep", "brain homeostasis", "memory consolidation")
    fetcher = _FakeFetcher(available={"40273975", "40074337"})
    abstracts = fetch_dch_reference_abstracts(config, fetcher)
    assert fetcher.requested == ["40074337", "40923478", "40273975"]
    assert [extract_pmids(a)[0] for a in abstracts] == ["40074337", "40273975"]
    assert fetcher.min_word_count == 0  # references bypass MIN_WORD_COUNT


def test_fetch_skips_non_dch_and_unlisted_combos():
    fetcher = _FakeFetcher(available=set())
    assert fetch_dch_reference_abstracts(
        _dch_config("REM sleep", "memory consolidation", "brain homeostasis", is_dch=False), fetcher) == []
    assert fetch_dch_reference_abstracts(_dch_config("x", "y", "z"), fetcher) == []
    assert fetcher.requested is None
