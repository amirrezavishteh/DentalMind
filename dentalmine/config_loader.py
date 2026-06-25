"""Lightweight YAML config loader with dot + dict access.

PROMPT.md accesses config both as attributes (``config.medclip.mmkd_weights``)
and occasionally as a flat value; this wrapper supports attribute access,
``[]`` access, and ``.get()`` with defaults.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from paths import DEFAULT_CONFIG, resolve


class DotDict(dict):
    """dict that also exposes keys as attributes, recursively."""

    def __getattr__(self, name: str) -> Any:
        try:
            val = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return _wrap(val)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def get(self, name: str, default: Any = None) -> Any:
        return _wrap(super().get(name, default))


def _wrap(val: Any) -> Any:
    if isinstance(val, dict) and not isinstance(val, DotDict):
        return DotDict(val)
    return val


def load_config(path: str | Path | None = None) -> DotDict:
    cfg_path = resolve(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return DotDict(raw)
