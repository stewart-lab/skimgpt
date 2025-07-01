"""
skimgpt.src - Core modules for Scientific Knowledge Mining with GPT

This package contains the core functionality for biomedical literature analysis
and hypothesis evaluation using large language models.
"""

import os

# Import main modules for easier access
from . import classifier
from . import utils
from . import pubmed_fetcher
from . import prompt_library
from . import scoring_guidelines

# Conditionally import relevance module to avoid vLLM import on CPU-only machines
try:
    # Check if we're explicitly in a GPU environment
    gpu_env = os.getenv('SKIMGPT_GPU_MODE', '').lower() in ('true', '1', 'yes')
    
    # Only import relevance if we're in GPU mode or if vLLM import succeeds
    if gpu_env:
        from . import relevance
    else:
        # Try to import, but don't fail if vLLM isn't available
        try:
            from . import relevance
        except ImportError as e:
            if 'vllm' in str(e).lower() or 'cuda' in str(e).lower():
                # vLLM/CUDA not available, skip relevance import
                relevance = None
            else:
                # Other import error, re-raise
                raise
except ImportError:
    relevance = None

__all__ = [
    "classifier",
    "utils",
    "pubmed_fetcher",
    "prompt_library",
    "scoring_guidelines",
]

# Only add relevance to __all__ if it was successfully imported
if 'relevance' in locals() and relevance is not None:
    __all__.append("relevance")
