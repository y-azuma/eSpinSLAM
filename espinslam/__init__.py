"""Paper-faithful eSpinSLAM reproduction."""

from .config import ESpinSLAMConfig, load_config

__all__ = ["ESpinSLAM", "ESpinSLAMConfig", "ESpinSLAMResult", "load_config"]


def __getattr__(name: str):
    # Lazy import keeps light modules (config, geometry) importable on their own.
    if name in {"ESpinSLAM", "ESpinSLAMResult"}:
        from . import pipeline
        return getattr(pipeline, name)
    raise AttributeError(name)
