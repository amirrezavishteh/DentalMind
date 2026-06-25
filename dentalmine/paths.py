"""Project path resolution helpers.

`dentalmine/` is the project root and is placed on sys.path by `pip install -e .`,
so top-level packages (`schema`, `pipeline`, `models`, ...) import without a
`dentalmine.` prefix — matching the import style used throughout PROMPT.md.

Config and template YAML files are loaded by filesystem path (not as package
data) via the helpers below.
"""
from __future__ import annotations

from pathlib import Path

# This file lives at the project root (dentalmine/paths.py).
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = PROJECT_ROOT / "config"
TEMPLATES_DIR = PROJECT_ROOT / "templates"

DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"
DEFAULT_TEMPLATES = TEMPLATES_DIR / "treatment_templates.yaml"


def resolve(path_like) -> Path:
    """Resolve a path relative to the project root if it is not absolute."""
    p = Path(path_like)
    return p if p.is_absolute() else (PROJECT_ROOT / p)
