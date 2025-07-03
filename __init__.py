"""
skimgpt - Scientific Knowledge Mining with GPT

A biomedical literature analysis tool for knowledge discovery and hypothesis evaluation.
"""

__version__ = "0.1.9"
__author__ = "Jack Freeman"
__email__ = "jfreeman@morgridge.org"

"""Top-level package initialisation for *skimgpt*.

Only very lightweight, dependency-free modules should be imported eagerly here.
Heavier sub-modules (those that need BioPython, torch, vllm …) are imported
lazily the first time they are accessed.  This allows the submission wrapper
to run on a minimal Python environment while the GPU container provides the
full scientific stack at runtime.
"""

from importlib import import_module
from types import ModuleType
from typing import Any

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

# These names are exposed.  Everything except ``src`` will be loaded lazily.
_EXPOSED = {
    "src",            # the sub-package itself (lightweight)
    "utils",          # lightweight (numpy/pandas)
    "classifier",     # depends on openai but no GPU
    "pubmed_fetcher", # depends on BioPython → may be missing on submit host
    "relevance",      # heavy GPU / vLLM dependencies – container only
}

# What *import skimgpt as s; dir(s)* should show
__all__ = list(_EXPOSED)


# ---------------------------------------------------------------------------
# Lazy loader
# ---------------------------------------------------------------------------

def __getattr__(name: str) -> Any:  # noqa: D401 – single-line docstring ok
    """Dynamically import sub-modules on first access.

    This is executed only when an attribute that does not yet exist is
    requested from the *skimgpt* package.  It tries to import the matching
    module from *skimgpt.src* and caches it in ``globals()`` so subsequent
    look-ups are fast.
    """

    if name not in _EXPOSED:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    # Map attribute → real module path
    mod_path = "skimgpt" if name == "src" else f"skimgpt.src.{name}"

    try:
        module: ModuleType = import_module(mod_path)
    except ImportError as exc:
        # Provide a clearer message for optional heavy dependencies
        raise AttributeError(
            f"Sub-module '{name}' could not be imported – missing optional "
            f"dependency? Original error: {exc}"
        ) from exc

    globals()[name] = module  # cache for future access
    return module 