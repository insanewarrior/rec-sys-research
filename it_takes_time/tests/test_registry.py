"""Verify IA-SASRec variants are wired into the model registry."""
from __future__ import annotations

from unittest.mock import MagicMock

from models import MODEL_REGISTRY, get_spec, ia_sasrec_space, list_models
from models.variants.ia_sasrec import IASASRecAdd, IASASRecMul, IASASRecVal


def test_variants_registered():
    names = list_models()
    for n in ("IA-SASRec-Add", "IA-SASRec-Mul", "IA-SASRec-Val"):
        assert n in names, f"missing {n} from MODEL_REGISTRY"


def test_variant_classes_correct():
    assert get_spec("IA-SASRec-Add")["class"] is IASASRecAdd
    assert get_spec("IA-SASRec-Mul")["class"] is IASASRecMul
    assert get_spec("IA-SASRec-Val")["class"] is IASASRecVal


def test_ia_sasrec_space_includes_intensity_norm():
    trial = MagicMock()
    trial.suggest_int.return_value = 2
    trial.suggest_categorical.side_effect = lambda name, choices: choices[0]
    trial.suggest_float.return_value = 1e-3
    sampled = ia_sasrec_space(trial)
    assert "intensity_norm" in sampled
    assert "n_layers" in sampled  # inherited from sasrec_space
    assert sampled["intensity_norm"] in {"log1p_minmax", "minmax", "zscore"}


def test_ia_variants_use_ce_loss_static():
    for n in ("IA-SASRec-Add", "IA-SASRec-Mul", "IA-SASRec-Val"):
        spec = get_spec(n)
        assert spec["static"]["loss_type"] == "CE"
        assert spec["type"] == "sequential"


def test_ia_variants_carry_sasrec_yaml_defaults():
    """Custom variants don't get RecBole's SASRec.yaml internal config, so the
    static dict must inline the SASRec-specific fields that aren't HPO-tuned
    (hidden_act, layer_norm_eps, initializer_range). Missing them crashes
    SASRec.__init__ deep in FeedForward.get_hidden_act."""
    for n in ("IA-SASRec-Add", "IA-SASRec-Mul", "IA-SASRec-Val"):
        static = get_spec(n)["static"]
        assert static["hidden_act"] == "gelu"
        assert static["layer_norm_eps"] == 1e-12
        assert static["initializer_range"] == 0.02
