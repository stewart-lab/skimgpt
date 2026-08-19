"""
skimgpt.src.visualization - Hypothesis comparison statistics and visualisation.

Ports of the Hyp1vsHyp2_paper analysis pipeline.  Heavy dependencies
(matplotlib, scipy, statsmodels, cmdlogtime) are optional; modules are
loaded lazily on first access.
"""

from importlib import import_module
from types import ModuleType
from typing import Any

_EXPOSED = {"hyp_stats", "plot_hyp_stats", "bayesian_ci", "plot_separate_runs"}

__all__ = list(_EXPOSED)


def __getattr__(name: str) -> Any:
    if name not in _EXPOSED:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    mod_path = f"{__name__}.{name}"

    try:
        module: ModuleType = import_module(mod_path)
    except ImportError as exc:
        raise AttributeError(
            f"Sub-module '{name}' could not be imported – optional dependency "
            f"missing? Original error: {exc}"
        ) from exc

    globals()[name] = module
    return module
