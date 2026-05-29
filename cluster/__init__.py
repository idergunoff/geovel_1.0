"""Cluster analysis package.

The package is split by responsibility while preserving the historical
``from cluster import *`` API used by the application.
"""
from __future__ import annotations

from importlib import import_module

_MODULE_NAMES = (
    "models",
    "well_dataset",
    "context",
    "auto_candidates",
    "auto_runner",
    "auto_cache",
    "profile_cache",
    "objects",
    "core",
    "well_visualization",
    "actions",
    "canonical_preparation",
)

_modules = [import_module(f"{__name__}.{module_name}") for module_name in _MODULE_NAMES]

# Recreate the former single-module namespace for cross-module references and
# for existing imports such as ``from cluster import _private_helper``.
for _module in _modules:
    for _name, _value in _module.__dict__.items():
        if _name.startswith("__"):
            continue
        globals()[_name] = _value

for _module in _modules:
    for _name, _value in globals().items():
        if _name.startswith("__"):
            continue
        _module.__dict__.setdefault(_name, _value)

__all__ = [
    _name
    for _name in globals()
    if not _name.startswith("__") and _name not in {"import_module", "_modules", "_module", "_name", "_value"}
]
