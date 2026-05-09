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
    """Return ``True`` if *class_field* refers to a RecBole built-in model by name.

    Parameters:
        class_field: The ``"class"`` value from a ``ModelSpec``; either a string
            (built-in) or a Python class (custom variant).

    Returns:
        ``True`` when *class_field* is a ``str``, ``False`` otherwise.
    """
    return isinstance(class_field, str)


def resolve(class_field: Any) -> tuple[str | None, type | None]:
    """Resolve a registry ``class`` field to a ``(name_str, model_cls)`` pair.

    Exactly one of the two returned values will be non-``None``:

    - Built-ins: returns ``(model_name_str, None)`` so RecBole resolves the class
      internally via ``get_model``.
    - Custom variants: returns ``(class.__name__, class)`` so the runner can pass
      the class directly to the trainer.

    Parameters:
        class_field: The ``"class"`` value from a ``ModelSpec``; either a string
            (built-in model name) or a Python class (custom variant).

    Returns:
        2-tuple ``(model_name_str, model_cls)`` where exactly one element is
        ``None``.
    """
    if isinstance(class_field, str):
        return class_field, None
    return class_field.__name__, class_field
