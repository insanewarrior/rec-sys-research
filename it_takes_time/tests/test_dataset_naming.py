"""Guard against RecBole's hard-coded dataset-name overrides.

RecBole 1.2.0 ships a special case in ``recbole/config/configurator.py``::

    if self.dataset == "ml-100k":
        ...data_path = "<package>/dataset_example/ml-100k"

When our ``DATASETS`` registry uses the literal ``"ml-100k"``, RecBole
silently *replaces* our ``data_path`` with its bundled 3-column example
(`user_id, item_id, rating, timestamp` — no intensity column), and every
IA-SASRec variant collapses to vanilla SASRec because ``intensity_list`` is
never built. This test fails fast if a future contributor reverts to a
dataset name that collides with the override.
"""
from __future__ import annotations

import re
from pathlib import Path

import recbole.config.configurator as cfg_mod

from config import DATASETS


def _scrape_hardcoded_dataset_names() -> set[str]:
    """Parse RecBole's configurator source for literal dataset names that
    trigger a ``data_path`` override in ``_set_default_parameters``.

    We use a source-scrape (rather than hard-coding the known set) so the
    test stays accurate after RecBole upgrades. As of 1.2.0 there is one
    such name: ``"ml-100k"``.
    """
    src = Path(cfg_mod.__file__).read_text()
    # Find the body of _set_default_parameters.
    m = re.search(r"def _set_default_parameters\(self\):(.*?)(?=\n    def )",
                  src, re.DOTALL)
    if not m:
        # If the layout changed, surface that loudly — the guard has lost its
        # anchor and needs revisiting.
        raise AssertionError(
            "Could not locate _set_default_parameters in recbole configurator"
        )
    body = m.group(1)
    # Match: if self.dataset == "<name>":
    return set(re.findall(r'self\.dataset\s*==\s*"([^"]+)"', body))


def test_no_dataset_name_collides_with_recbole_override():
    overrides = _scrape_hardcoded_dataset_names()
    assert overrides, (
        "Scrape returned no overrides — anchor in RecBole source has moved; "
        "update _scrape_hardcoded_dataset_names."
    )
    collisions = set(DATASETS) & overrides
    assert not collisions, (
        f"DATASETS keys {collisions} collide with RecBole's hard-coded "
        f"data_path overrides {overrides}. Pick a different key (e.g. add "
        f"a suffix); RecBole will silently load the wrong .inter otherwise."
    )
