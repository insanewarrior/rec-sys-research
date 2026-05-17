"""Smoke test for the extended ``.inter`` writer with intensity column."""
from __future__ import annotations

import pandas as pd


def test_intensity_column_written(tmp_path, monkeypatch):
    import config
    import data as data_mod

    raw = tmp_path / "raw" / "ml-1m"
    raw.mkdir(parents=True)
    # Three users, eight interactions, ratings 1..5.
    (raw / "ratings.dat").write_text(
        "1::100::5::1\n"
        "1::101::3::2\n"
        "1::102::4::3\n"
        "2::100::2::1\n"
        "2::103::5::2\n"
        "2::104::1::3\n"
        "3::101::4::1\n"
        "3::102::3::2\n"
    )
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "RECBOLE_DATA_DIR", tmp_path / "rb")
    monkeypatch.setattr(data_mod, "DATA_DIR", tmp_path / "raw")
    monkeypatch.setattr(data_mod, "RECBOLE_DATA_DIR", tmp_path / "rb")

    out_dir = data_mod.prepare_recbole_dataset("ml-1m")
    inter = out_dir / "ml-1m.inter"
    assert inter.exists()
    text = inter.read_text().splitlines()
    assert text[0] == "user_id:token\titem_id:token\ttimestamp:float\tintensity:float"
    df = pd.read_csv(inter, sep="\t")
    df.columns = [c.split(":")[0] for c in df.columns]
    assert len(df) == 8
    assert set(df.columns) == {"user_id", "item_id", "timestamp", "intensity"}
    assert df["intensity"].min() >= 1.0 and df["intensity"].max() <= 5.0
    assert not df["intensity"].isna().any()


def test_steam_synthesises_timestamps(tmp_path, monkeypatch):
    import config
    import data as data_mod

    raw = tmp_path / "raw" / "steam"
    raw.mkdir(parents=True)
    # Steam-200k CSV: user,item,behavior,hours,extra
    (raw / "steam-200k.csv").write_text(
        "1,Game A,play,5.5,0\n"
        "1,Game B,play,12.0,0\n"
        "1,Game A,purchase,1.0,0\n"   # should be filtered out
        "2,Game A,play,0.3,0\n"
        "2,Game C,play,40.0,0\n"
        "3,Game B,play,2.0,0\n"
        "3,Game C,play,15.0,0\n"
        "3,Game A,play,7.0,0\n"
    )
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "RECBOLE_DATA_DIR", tmp_path / "rb")
    monkeypatch.setattr(data_mod, "DATA_DIR", tmp_path / "raw")
    monkeypatch.setattr(data_mod, "RECBOLE_DATA_DIR", tmp_path / "rb")

    out_dir = data_mod.prepare_recbole_dataset("steam")
    inter = out_dir / "steam.inter"
    df = pd.read_csv(inter, sep="\t")
    df.columns = [c.split(":")[0] for c in df.columns]
    # 7 play rows after dropping the purchase row.
    assert len(df) == 7
    # Synthesised timestamps must be strictly increasing globally.
    assert (df["timestamp"].diff().dropna() >= 0).all()
    assert df["intensity"].min() > 0
