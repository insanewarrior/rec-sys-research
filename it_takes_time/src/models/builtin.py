"""Helpers for resolving a registry ``class`` field to something RecBole can train.

RecBole's ``run_recbole`` / ``Trainer`` accept either:
  - a string model name (resolved via ``recbole.utils.get_model``), or
  - a Python class subclassing one of the abstract recommender base classes.

We pass strings straight through and pass custom classes via the ``model_class`` kwarg
that ``recbole.quick_start`` accepts (we wrap that in ``runner.py``).
"""

from __future__ import annotations

from typing import Any


def is_builtin(class_field: Any) -> bool:
    return isinstance(class_field, str)


def resolve(class_field: Any) -> tuple[str | None, type | None]:
    """Return ``(model_name_str, model_class)``; exactly one will be non-None."""
    if isinstance(class_field, str):
        return class_field, None
    return class_field.__name__, class_field
