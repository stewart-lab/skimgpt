"""
skimgpt.src - Core modules for Scientific Knowledge Mining with GPT

This package contains the core functionality for biomedical literature analysis
and hypothesis evaluation using large language models.
"""

# Import main modules for easier access
from . import classifier
from . import relevance
from . import utils
from . import pubmed_fetcher
from . import prompt_library
from . import scoring_guidelines

__all__ = [
    "classifier",
    "relevance", 
    "utils",
    "pubmed_fetcher",
    "prompt_library",
    "scoring_guidelines",
]
