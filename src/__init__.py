"""
skimgpt.src - Core modules for Scientific Knowledge Mining with GPT

This package contains the core functionality for biomedical literature analysis
and hypothesis evaluation using large language models.

Initialise *skimgpt.src* sub-package.

Keep it entirely lightweight so that a submission host without BioPython /
openai / vllm can still import modules that do **not** require those heavy
dependencies (e.g. ``src.eval_JSON_results``).

Heavier sub-modules are loaded lazily on first access.
"""

import os
from importlib import import_module
from types import ModuleType
from typing import Any

# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

_EXPOSED = {
    "utils",             # numpy/pandas only
    "classifier",        # needs openai – may be absent, but not critical for CLI
    "pubmed_fetcher",    # needs BioPython – container provides it
    "prompt_library",    # no heavy deps
    "scoring_guidelines",# no heavy deps
    "relevance",         # vLLM / torch – GPU container only
}

__all__ = list(_EXPOSED)


# ---------------------------------------------------------------------------
# Lazy importer
# ---------------------------------------------------------------------------

def __getattr__(name: str) -> Any:
    if name not in _EXPOSED:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    mod_path = f"{__name__}.{name}"

    try:
        module: ModuleType = import_module(mod_path)
    except ImportError as exc:
        # Provide a friendlier message for optional heavy deps
        raise AttributeError(
            f"Sub-module '{name}' could not be imported – optional dependency "
            f"missing? Original error: {exc}"
        ) from exc

    globals()[name] = module
    return module
