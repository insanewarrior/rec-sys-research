"""Regression test for the custom-variant Config construction path.

Background: RecBole's ``Config(model="SomeName", ...)`` calls
``recbole.utils.get_model("SomeName")``, which scans every model submodule
including ``exlib_recommender`` — the latter imports ``lightgbm``, which on
macOS requires ``libomp.dylib`` and crashes with ``OSError: Library not
loaded`` when it isn't installed. For custom variants whose class name is not
a built-in, this scan is gratuitous and dangerous. ``runner._build_config``
must therefore pass the class object directly (RecBole handles non-string
``model`` arguments without the submodule walk).
"""
from __future__ import annotations

import pytest

import recbole.utils as recbole_utils
from recbole.config import Config

import runner
from models.variants.ia_sasrec import IASASRecAdd


def test_build_config_custom_variant_skips_get_model(monkeypatch):
    """If we ever pass a name string for a custom variant, get_model would run.
    Patching it to raise ensures _build_config never takes that branch."""

    def _boom(*_a, **_kw):
        raise RuntimeError(
            "get_model should not be invoked for custom variants — would "
            "trigger lightgbm/libomp on macOS"
        )

    # Patch both the original symbol and any module that imported it by reference.
    monkeypatch.setattr(recbole_utils, "get_model", _boom)
    import recbole.config.configurator as cfg_mod
    monkeypatch.setattr(cfg_mod, "get_model", _boom)

    cfg, model_cls = runner._build_config(
        "ml-100k", "IA-SASRec-Add", overrides={}, epochs=1, saved=False
    )
    assert model_cls is IASASRecAdd
    assert isinstance(cfg, Config)
    # RecBole stores the resolved class on the Config instance.
    assert cfg.model_class is IASASRecAdd


def test_build_config_builtin_still_uses_name(monkeypatch):
    """Built-ins like SASRec must still resolve through get_model (we don't have
    the class in hand, only the registry name)."""
    calls = []
    real_get_model = recbole_utils.get_model

    def _spy(name):
        calls.append(name)
        return real_get_model(name)

    import recbole.config.configurator as cfg_mod
    monkeypatch.setattr(cfg_mod, "get_model", _spy)

    cfg, model_cls = runner._build_config(
        "ml-100k", "SASRec", overrides={}, epochs=1, saved=False
    )
    assert model_cls is None  # built-in path returns class=None
    assert "SASRec" in calls
    assert isinstance(cfg, Config)
