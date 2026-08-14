"""Relevance-filter prompt truncation — guards against vLLM's max_model_len crash.

The relevance-filter model (relevance_chtc.py) loads with max_model_len=4000
tokens and vLLM rejects any longer prompt outright rather than truncating it
itself. prompt() previously concatenated the raw abstract with no length
check at all, so a single unusually long abstract anywhere in a run's pool
(rare, but real -- see the cervical-cancer DCH run where 1 of 3104 prompts
hit this) would crash the whole batch. Without this test, a future edit to
prompt() could silently drop the truncation and reintroduce that crash.
"""
from skimgpt.relevance_helper import _RELEVANCE_FILTER_ABSTRACT_CHAR_LIMIT, prompt


def test_short_abstract_passes_through_unmodified():
    abstract = "A normal-length abstract about a gene and a disease."
    result = prompt(abstract, "some hypothesis")
    assert abstract in result


def test_long_abstract_is_truncated_to_the_char_limit():
    abstract = "a" * (_RELEVANCE_FILTER_ABSTRACT_CHAR_LIMIT + 5000)
    result = prompt(abstract, "some hypothesis")
    assert "a" * _RELEVANCE_FILTER_ABSTRACT_CHAR_LIMIT in result
    assert "a" * (_RELEVANCE_FILTER_ABSTRACT_CHAR_LIMIT + 1) not in result


def test_abstract_exactly_at_limit_is_untouched():
    abstract = "b" * _RELEVANCE_FILTER_ABSTRACT_CHAR_LIMIT
    result = prompt(abstract, "some hypothesis")
    assert abstract in result


def test_empty_abstract_does_not_crash():
    result = prompt("", "some hypothesis")
    assert "Hypothesis: some hypothesis" in result


def test_hypothesis_is_never_truncated():
    """Only the abstract is length-capped; a long hypothesis must pass through whole."""
    long_hypothesis = "h" * (_RELEVANCE_FILTER_ABSTRACT_CHAR_LIMIT + 100)
    result = prompt("short abstract", long_hypothesis)
    assert long_hypothesis in result
