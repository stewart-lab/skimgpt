"""expand_synonyms: the LLM is told that abstracts may name a term by another
name, and a fast_km server that ignores the setting is reported."""
from types import SimpleNamespace

import pytest

from skimgpt import prompt_library
from skimgpt.classifier import _system_instructions_for
from skimgpt.skim_and_km_api import warn_if_expansion_unsupported
from skimgpt.utils import Config


def _config(job_type="km_with_gpt", **settings):
    config = SimpleNamespace(
        job_type=job_type,
        is_dch=False,
        is_km_with_gpt=job_type == "km_with_gpt",
        is_skim_with_gpt=job_type == "skim_with_gpt",
        job_specific_settings=settings,
    )
    config.expand_synonyms = Config.expand_synonyms.fget(config)
    return config


@pytest.mark.parametrize("value, expected", [(True, True), (False, False), ("false", False), ("true", False), (None, False)])
def test_setting_is_on_only_when_true(value, expected):
    assert _config(expand_synonyms=value).expand_synonyms is expected


def test_setting_defaults_to_off():
    assert _config().expand_synonyms is False


@pytest.mark.parametrize("job_type, relationship_type", [
    ("km_with_gpt", None), ("skim_with_gpt", "A_C"), ("skim_with_gpt", "A_B_C")])
def test_llm_is_told_about_other_names_only_when_expanding(job_type, relationship_type):
    on = _system_instructions_for(_config(job_type, expand_synonyms=True), relationship_type)
    off = _system_instructions_for(_config(job_type), relationship_type)
    assert on == off + prompt_library.EXPANDED_TERMS_NOTE
    assert "p53" in on and "p53" not in off


def test_dch_is_told_too():
    config = _config(expand_synonyms=True)
    config.is_dch = True
    assert _system_instructions_for(config).endswith(prompt_library.EXPANDED_TERMS_NOTE)


def test_warns_when_server_ignores_the_setting(caplog):
    assert warn_if_expansion_unsupported(_config(expand_synonyms=True), {"status": "finished", "result": []})
    assert "doesn't support it" in caplog.text


@pytest.mark.parametrize("config, response", [
    (_config(expand_synonyms=True), {"status": "finished", "result": [], "expansions": {}}),
    (_config(expand_synonyms=True), None),
    (_config(), {"status": "finished", "result": []}),
])
def test_no_warning_otherwise(config, response, caplog):
    assert not warn_if_expansion_unsupported(config, response)
    assert caplog.text == ""
