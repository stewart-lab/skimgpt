"""
skimgpt - Scientific Knowledge Mining with GPT

A biomedical literature analysis tool for knowledge discovery and hypothesis evaluation.
"""

__version__ = "0.1.6"
__author__ = "Jack Freeman"
__email__ = "jfreeman@morgridge.org"

# Import main modules for easy access
from . import src

# Expose core modules that don't have GPU dependencies
from .src import utils, classifier, pubmed_fetcher

# Conditionally expose relevance (has GPU/vLLM dependencies)
try:
    from .src import relevance
    __all__ = ["src", "relevance", "utils", "classifier", "pubmed_fetcher"]
except ImportError:
    # relevance module not available (likely due to vLLM/GPU dependencies)
    __all__ = ["src", "utils", "classifier", "pubmed_fetcher"] 