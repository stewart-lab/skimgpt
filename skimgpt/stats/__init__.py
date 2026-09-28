"""skimgpt.stats - dependency-light statistical cores.

Modules here depend on numpy/scipy only: no pandas, matplotlib, cmdlogtime,
torch or openai. That keeps them importable from a plain web container
(SKiM_web installs skimgpt with ``--no-deps``) as well as from the plotting
CLIs in ``skimgpt.visualization``.

Keep it that way — a heavy import added here breaks the web deployment.
"""

from skimgpt.stats.bayes_ci import PosteriorCI, posterior_ci
from skimgpt.stats.bayes_posterior import (
    PosteriorSummary,
    call_from_result,
    posterior_summary,
)

__all__ = [
    "PosteriorCI",
    "posterior_ci",
    "PosteriorSummary",
    "call_from_result",
    "posterior_summary",
]
