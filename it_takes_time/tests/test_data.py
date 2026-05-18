"""Smoke tests for the ``.inter`` writer across download paths."""
from __future__ import annotations

import json

import pandas as pd


def _patch_paths(monkeypatch, tmp_path):
    import config
    import data as data_mod

    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "RECBOLE_DATA_DIR", tmp_path / "rb")
    monkeypatch.setattr(data_mod, "DATA_DIR", tmp_path / "raw")
    monkeypatch.setattr(data_mod, "RECBOLE_DATA_DIR", tmp_path / "rb")
    return data_mod


def test_intensity_column_written_csv(tmp_path, monkeypatch):
    data_mod = _patch_paths(monkeypatch, tmp_path)
    raw = tmp_path / "raw" / "ml-100k"
    raw.mkdir(parents=True)
    (raw / "u.data").write_text(
        "1\t100\t5\t1\n"
        "1\t101\t3\t2\n"
        "1\t102\t4\t3\n"
        "2\t100\t2\t1\n"
        "2\t103\t5\t2\n"
        "2\t104\t1\t3\n"
        "3\t101\t4\t1\n"
        "3\t102\t3\t2\n"
    )

    out_dir = data_mod.prepare_recbole_dataset("ml-100k-iar")
    inter = out_dir / "ml-100k-iar.inter"
    assert inter.exists()
    text = inter.read_text().splitlines()
    assert text[0] == "user_id:token\titem_id:token\ttimestamp:float\tintensity:float"
    df = pd.read_csv(inter, sep="\t")
    df.columns = [c.split(":")[0] for c in df.columns]
    assert len(df) == 8
    assert set(df.columns) == {"user_id", "item_id", "timestamp", "intensity"}
    assert df["intensity"].min() >= 1.0 and df["intensity"].max() <= 5.0
    assert not df["intensity"].isna().any()


def test_jsonl_amazon_column_map(tmp_path, monkeypatch):
    data_mod = _patch_paths(monkeypatch, tmp_path)
    raw = tmp_path / "raw" / "amazon-digital-music"
    raw.mkdir(parents=True)
    records = [
        {"reviewerID": "A1", "asin": "I1", "overall": 5.0, "unixReviewTime": 1000},
        {"reviewerID": "A1", "asin": "I2", "overall": 3.0, "unixReviewTime": 1100},
        {"reviewerID": "A2", "asin": "I1", "overall": 4.0, "unixReviewTime": 1050},
    ]
    (raw / "reviews_Digital_Music_5.json").write_text(
        "\n".join(json.dumps(r) for r in records)
    )

    out_dir = data_mod.prepare_recbole_dataset("amazon-digital-music")
    inter = out_dir / "amazon-digital-music.inter"
    df = pd.read_csv(inter, sep="\t")
    df.columns = [c.split(":")[0] for c in df.columns]
    assert len(df) == 3
    assert set(df.columns) == {"user_id", "item_id", "timestamp", "intensity"}
    # column_map renamed reviewerID/asin/overall/unixReviewTime correctly
    assert set(df["user_id"]) == {"A1", "A2"}
    assert set(df["item_id"]) == {"I1", "I2"}
    assert df["intensity"].between(3.0, 5.0).all()
