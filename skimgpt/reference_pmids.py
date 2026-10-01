"""Curated reference PMIDs that DCH abstract selection must always include.

``data/dch_reference_pmids.csv`` lists, per (A term, B1 term, B2 term)
combination, the papers a run on that combination is expected to read. When a
DCH run matches a combination, those PMIDs are fetched directly from PubMed —
bypassing the KM intersection and the relevance filter — and placed in every
iteration's sample ahead of the randomly drawn abstracts.

Terms are matched on their first pipe synonym, case-insensitively, with B1/B2
in either order, so a run with an extended synonym list still matches.
"""
from __future__ import annotations

import csv
import logging
from functools import lru_cache
from importlib import resources

from skimgpt.utils import strip_pipe

logger = logging.getLogger(__name__)

REFERENCE_CSV = "dch_reference_pmids.csv"

ComboKey = tuple[str, frozenset[str]]


def _norm(term: str) -> str:
    return strip_pipe(term).casefold()


def combo_key(a_term: str, b1_term: str, b2_term: str) -> ComboKey:
    return _norm(a_term), frozenset((_norm(b1_term), _norm(b2_term)))


@lru_cache(maxsize=1)
def load_reference_table() -> dict[ComboKey, tuple[str, ...]]:
    """Return {combo_key: PMIDs} from the packaged CSV, PMIDs in file order."""
    table: dict[ComboKey, list[str]] = {}
    with resources.files("skimgpt").joinpath("data", REFERENCE_CSV).open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pmid = (row.get("pubmedID") or "").strip()
            if not pmid:
                continue
            key = combo_key(row["a term"], row["b1 term"], row["b2 term"])
            pmids = table.setdefault(key, [])
            if pmid not in pmids:
                pmids.append(pmid)
    return {k: tuple(v) for k, v in table.items()}


def reference_pmids_for(a_term: str, b1_term: str, b2_term: str, a_term_suffix: str = "") -> list[str]:
    """PMIDs that must be selected for this DCH combination ([] if none listed).

    ``a_term_suffix`` is the configured ``A_TERM_SUFFIX``; the KM output's
    a_term carries it, so it is stripped before matching.
    """
    if a_term_suffix and a_term.endswith(a_term_suffix):
        a_term = a_term[: -len(a_term_suffix)]
    return list(load_reference_table().get(combo_key(a_term, b1_term, b2_term), ()))
