"""Download raw datasets and convert to RecBole atomic file format.

RecBole expects, under ``data_path/<dataset>/``, an ``<dataset>.inter`` file with a
typed header line such as::

    user_id:token\titem_id:token\ttimestamp:float

This module is idempotent: if the atomic file already exists, ``prepare_recbole_dataset``
is a no-op. That keeps notebook re-runs cheap.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from config import DATA_DIR, DATASETS, RECBOLE_DATA_DIR


def _download_and_unzip(url: str, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[data] Downloading {url} ...")
    with urllib.request.urlopen(url) as resp:
        zbytes = resp.read()
    with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
        zf.extractall(target_dir)
    print(f"[data] Extracted to {target_dir}")


def _ensure_raw(dataset_name: str) -> Path:
    spec = DATASETS[dataset_name]
    raw_dir = DATA_DIR / spec["raw_subdir"]
    if (raw_dir / spec["ratings_file"]).exists():
        return raw_dir
    _download_and_unzip(spec["url"], DATA_DIR)
    if not (raw_dir / spec["ratings_file"]).exists():
        raise FileNotFoundError(f"Expected {raw_dir / spec['ratings_file']} after download")
    return raw_dir


def prepare_recbole_dataset(dataset_name: str, force: bool = False) -> Path:
    """Materialize ``<RECBOLE_DATA_DIR>/<dataset_name>/<dataset_name>.inter``.

    Returns the directory path (which is what RecBole's ``data_path`` should point at,
    one level above).
    """
    if dataset_name not in DATASETS:
        raise KeyError(f"Unknown dataset: {dataset_name}. Known: {list(DATASETS)}")
    spec = DATASETS[dataset_name]
    out_dir = RECBOLE_DATA_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    inter_path = out_dir / f"{dataset_name}.inter"
    if inter_path.exists() and not force:
        return out_dir

    raw_dir = _ensure_raw(dataset_name)
    df = pd.read_csv(
        raw_dir / spec["ratings_file"],
        sep=spec["sep"],
        names=spec["columns"],
        engine="python",
        encoding="latin-1",
    )
    if spec.get("rating_threshold", 0) > 0:
        df = df[df["rating"] >= spec["rating_threshold"]]
    df = df[["user_id", "item_id", "timestamp"]].sort_values(["user_id", "timestamp"])

    header = "user_id:token\titem_id:token\ttimestamp:float\n"
    with inter_path.open("w") as f:
        f.write(header)
        df.to_csv(f, sep="\t", index=False, header=False)
    print(f"[data] Wrote {inter_path}  ({len(df):,} interactions, "
          f"{df['user_id'].nunique():,} users, {df['item_id'].nunique():,} items)")
    return out_dir


def dataset_stats(dataset_name: str) -> dict:
    out_dir = prepare_recbole_dataset(dataset_name)
    df = pd.read_csv(out_dir / f"{dataset_name}.inter", sep="\t")
    df.columns = [c.split(":")[0] for c in df.columns]
    return {
        "interactions": len(df),
        "users": df["user_id"].nunique(),
        "items": df["item_id"].nunique(),
        "density": len(df) / (df["user_id"].nunique() * df["item_id"].nunique()),
        "min_ts": int(df["timestamp"].min()),
        "max_ts": int(df["timestamp"].max()),
    }
