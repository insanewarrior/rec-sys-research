"""NumPy 2.x compatibility shim for RecBole.

RecBole's ``Config.compatibility_settings()`` (recbole/config/configurator.py)
unconditionally evaluates ``np.float_``, ``np.complex_``, ``np.object_``,
``np.unicode_``, ``np.str_`` to restore deprecated aliases. NumPy 2.0 removed
those right-hand-side names, so the line raises ``AttributeError`` before any
of our code gets to run.

We pre-populate the missing attributes on the ``numpy`` module. RecBole's
subsequent assignments (``np.float = np.float_`` etc.) then succeed and behave
identically to the old NumPy 1.x path.

Import this module **before** anything that imports ``recbole``.
"""

from __future__ import annotations

import numpy as np

_aliases = {
    "float_": np.float64,
    "complex_": np.complex128,
    "object_": object,
    "unicode_": np.str_,
    "bool_": np.bool_,
    "int_": np.int64,
}

for _name, _val in _aliases.items():
    if not hasattr(np, _name):
        setattr(np, _name, _val)
