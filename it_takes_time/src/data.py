"""Download raw datasets and convert to RecBole atomic file format.

RecBole expects, under ``data_path/<dataset>/``, an ``<dataset>.inter`` file with a
typed header line such as::

    user_id:token\titem_id:token\ttimestamp:float\tintensity:float

The ``intensity:float`` column carries the per-interaction strength signal used by
IA-SASRec variants (ratings on MovieLens, hours-played on Steam). Baseline models
ignore the extra column.

This module is idempotent: if the atomic file already exists, ``prepare_recbole_dataset``
is a no-op. That keeps notebook re-runs cheap.
"""

from __future__ import annotations

import gzip
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from config import DATA_DIR, DATASETS, RECBOLE_DATA_DIR


def _download_and_unzip(url: str, target_dir: Path) -> None:
    """Download a ZIP archive from *url* and extract it into *target_dir*.

    Parameters:
        url: Full HTTP/HTTPS URL of the ZIP file to download.
        target_dir: Destination directory; created (with parents) if absent.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[data] Downloading {url} ...")
    with urllib.request.urlopen(url) as resp:
        zbytes = resp.read()
    with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
        zf.extractall(target_dir)
    print(f"[data] Extracted to {target_dir}")


def _download_and_gunzip(url: str, target_dir: Path, out_file: str) -> None:
    """Download a gzip-compressed file from *url* and write it decompressed to *target_dir/out_file*."""
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[data] Downloading {url} ...")
    with urllib.request.urlopen(url) as resp:
        gz_bytes = resp.read()
    out_path = target_dir / out_file
    with gzip.open(io.BytesIO(gz_bytes)) as gz:
        out_path.write_bytes(gz.read())
    print(f"[data] Wrote {out_path}")


def _download_keep_gz(url: str, target_dir: Path, out_file: str) -> None:
    """Download a gzip-compressed file as-is (no decompression) to *target_dir/out_file*.

    Used when the uncompressed payload is large enough that we want to stream-decode it
    later rather than materialize it to disk. The compressed file is streamed to disk
    in chunks so we never hold the full payload in memory.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / out_file
    print(f"[data] Downloading {url} ...")
    with urllib.request.urlopen(url) as resp, out_path.open("wb") as f:
        while True:
            chunk = resp.read(1 << 20)  # 1 MiB chunks
            if not chunk:
                break
            f.write(chunk)
    print(f"[data] Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB compressed)")


def _download_via_kagglehub(kaggle_dataset: str, target_dir: Path, ratings_file: str) -> None:
    """Download a Kaggle dataset via the ``kagglehub`` SDK and copy the wanted file.

    Parameters:
        kaggle_dataset: ``"owner/slug"`` Kaggle dataset identifier.
        target_dir: Local directory the file should end up in.
        ratings_file: Filename to copy out of the Kaggle cache.

    Requires ``~/.kaggle/kaggle.json`` or the env vars ``KAGGLE_USERNAME`` and
    ``KAGGLE_KEY`` to be configured.
    """
    import shutil
    try:
        import kagglehub
    except ImportError as e:
        raise ImportError(
            "kagglehub is required to download Kaggle-hosted datasets.\n"
            "  pip install kagglehub"
        ) from e
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[data] Downloading Kaggle dataset {kaggle_dataset} via kagglehub ...")
    cache_dir = Path(kagglehub.dataset_download(kaggle_dataset))
    src = cache_dir / ratings_file
    if not src.exists():
        # Fall back to first match by name (some Kaggle datasets unpack into subdirs).
        candidates = list(cache_dir.rglob(ratings_file))
        if not candidates:
            raise FileNotFoundError(
                f"{ratings_file} not present in Kaggle cache {cache_dir}"
            )
        src = candidates[0]
    shutil.copy2(src, target_dir / ratings_file)
    print(f"[data] Copied {src} -> {target_dir / ratings_file}")


def _parse_pylit_jsonl_gz(gz_path: Path, spec: dict) -> pd.DataFrame:
    """Stream-parse a gzipped Python-repr jsonl file (one ``{u'k': v, ...}`` per line).

    Used for the Steam reviews dump, which is ~1.3 GB compressed / ~7 GB uncompressed
    in ``repr()`` form rather than strict JSON, so ``json.loads`` fails. We
    ``ast.literal_eval`` each line, keep only the renamed fields, and cache the result
    as parquet next to the .gz so subsequent ``prepare_recbole_dataset`` calls skip
    the multi-minute parse.
    """
    import ast
    cache = gz_path.with_suffix("").with_suffix(".parquet")  # foo.json.gz → foo.parquet
    if cache.exists():
        return pd.read_parquet(cache)

    col_map = dict(spec.get("column_map") or {})
    src_cols = list(col_map.keys())
    dst_cols = [col_map[c] for c in src_cols]

    print(f"[data] Stream-parsing {gz_path.name} (Python-repr jsonl) — this is slow once …")
    rows: list[tuple] = []
    bad = 0
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = ast.literal_eval(line)
            except (ValueError, SyntaxError, MemoryError):
                bad += 1
                continue
            try:
                rows.append(tuple(d[c] for c in src_cols))
            except KeyError:
                bad += 1
                continue
    print(f"[data] Parsed {len(rows):,} records, skipped {bad:,}")
    df = pd.DataFrame(rows, columns=dst_cols)
    df.to_parquet(cache, index=False)
    print(f"[data] Cached parsed dataframe to {cache}")
    return df


def _ensure_raw(dataset_name: str) -> Path:
    """Return the raw data directory for *dataset_name*, downloading if necessary.

    Parameters:
        dataset_name: Key into ``DATASETS`` (e.g. ``"ml-1m"``).

    Returns:
        Path to the directory containing the raw ratings file.

    Raises:
        FileNotFoundError: If the ratings file is absent even after a successful download.
    """
    spec = DATASETS[dataset_name]
    raw_dir = DATA_DIR / spec["raw_subdir"]
    ratings_path = raw_dir / spec["ratings_file"]
    if ratings_path.exists():
        return raw_dir
    if "kaggle_dataset" in spec:
        _download_via_kagglehub(spec["kaggle_dataset"], raw_dir, spec["ratings_file"])
    elif "url" in spec:
        download_format = spec.get("download_format")
        if download_format == "gz":
            _download_and_gunzip(spec["url"], raw_dir, spec["ratings_file"])
        elif download_format == "gz_keep":
            _download_keep_gz(spec["url"], raw_dir, spec["ratings_file"])
        else:
            _download_and_unzip(spec["url"], DATA_DIR)
    else:
        raise FileNotFoundError(
            f"No download source configured for {dataset_name}; "
            f"drop {ratings_path} in place manually."
        )
    if not ratings_path.exists():
        raise FileNotFoundError(f"Expected {ratings_path} after download")
    return raw_dir


def prepare_recbole_dataset(dataset_name: str, force: bool = False) -> Path:
    """Materialize ``<RECBOLE_DATA_DIR>/<dataset_name>/<dataset_name>.inter``.

    Downloads the raw dataset if needed, applies the rating threshold filter, and
    writes the RecBole atomic interaction file. Idempotent unless *force* is set.

    Parameters:
        dataset_name: Key into ``DATASETS`` (e.g. ``"ml-1m"``).
        force: Re-build the ``.inter`` file even if it already exists.

    Returns:
        Path to the dataset output directory (the value to pass as RecBole's
        ``data_path`` is one level above this directory).

    Raises:
        KeyError: If *dataset_name* is not present in ``DATASETS``.
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
    fmt = spec.get("format")
    if fmt == "jsonl":
        df = pd.read_json(raw_dir / spec["ratings_file"], lines=True)
        if spec.get("column_map"):
            df = df.rename(columns=spec["column_map"])
    elif fmt == "pylit_jsonl_gz":
        df = _parse_pylit_jsonl_gz(raw_dir / spec["ratings_file"], spec)
    else:
        df = pd.read_csv(
            raw_dir / spec["ratings_file"],
            sep=spec["sep"],
            names=spec["columns"],
            header=None,
            engine="python",
            encoding="latin-1",
        )

    if spec.get("rating_threshold", 0) > 0:
        df = df[df["rating"] >= spec["rating_threshold"]]

    # Some sources (Steam) provide timestamps as date strings — convert to unix seconds.
    ts_format = spec.get("timestamp_format")
    if ts_format is not None and df["timestamp"].dtype == object:
        ts = pd.to_datetime(df["timestamp"], format=ts_format, errors="coerce")
        df = df.assign(timestamp=(ts.astype("int64") // 10**9))
        df = df[df["timestamp"] > 0]

    intensity_col = spec["intensity_col"]
    if intensity_col != "intensity":
        df = df.rename(columns={intensity_col: "intensity"})

    df = df[["user_id", "item_id", "timestamp", "intensity"]].dropna()

    # Dedupe repeated (user, item) interactions: SASRec-style sequential models
    # expect each (user, item) to appear once per history. The Steam dump in
    # particular contains exact-row duplicates (review re-scrapes) and legitimate
    # multi-review pairs at different timestamps. Keep the latest row — its
    # intensity reflects the most recent playtime, strictly more informative
    # than any earlier value.
    if spec.get("dedupe_user_item", True):
        n0 = len(df)
        # Sort by (user, timestamp) only — stable mergesort preserves the original
        # source-file order within tied timestamps. We deliberately do NOT include
        # item_id in the sort key: that would make same-day ties land in alphabetical
        # item_id order, which is arbitrary noise.
        df = (
            df.sort_values(["user_id", "timestamp"], kind="mergesort")
              .drop_duplicates(subset=["user_id", "item_id"], keep="last")
        )
        if len(df) < n0:
            print(f"[data] Deduped (user, item): {n0:,} → {len(df):,} rows "
                  f"(dropped {n0 - len(df):,})")

    # Optional deterministic subsample by user (used by Steam to hit ml-100k scale).
    # Sample from users who would survive the dataset's 5-core filter, so the
    # post-RecBole user count actually lands near sub_n. Without this, Steam's
    # heavy long tail (many users with 1–2 reviews) leaves us with far fewer
    # users than expected after RecBole applies user_inter_num_interval.
    sub_n = spec.get("subsample_users")
    if sub_n is not None:
        import numpy as np
        seed = int(spec.get("subsample_seed", 2020))
        min_inter = int(spec.get("min_user_inter", 1))
        counts = df.groupby("user_id").size()
        eligible = counts[counts >= min_inter].index.to_numpy()
        if len(eligible) > sub_n:
            rng = np.random.default_rng(seed)
            keep = rng.choice(eligible, size=sub_n, replace=False)
            df = df[df["user_id"].isin(keep)]
            print(f"[data] Subsampled to {sub_n:,} users with ≥{min_inter} "
                  f"interactions (from {len(eligible):,} eligible, seed={seed}) "
                  f"→ {len(df):,} interactions pre-RecBole-filter")
        else:
            df = df[df["user_id"].isin(eligible)]
            print(f"[data] Only {len(eligible):,} users meet ≥{min_inter} "
                  f"interactions; keeping all (requested {sub_n:,})")

    df = df.sort_values(["user_id", "timestamp"], kind="mergesort")

    header = "user_id:token\titem_id:token\ttimestamp:float\tintensity:float\n"
    with inter_path.open("w") as f:
        f.write(header)
        df.to_csv(f, sep="\t", index=False, header=False)
    print(f"[data] Wrote {inter_path}  ({len(df):,} interactions, "
          f"{df['user_id'].nunique():,} users, {df['item_id'].nunique():,} items, "
          f"intensity range [{df['intensity'].min():.2f}, {df['intensity'].max():.2f}])")
    return out_dir


def dataset_stats(dataset_name: str) -> dict[str, int | float]:
    """Return summary statistics for a prepared RecBole dataset.

    Parameters:
        dataset_name: Key into ``DATASETS`` (e.g. ``"ml-1m"``).

    Returns:
        Dictionary with keys ``interactions``, ``users``, ``items``, ``density``,
        ``min_ts``, and ``max_ts``.
    """
    out_dir = prepare_recbole_dataset(dataset_name)
    df = pd.read_csv(out_dir / f"{dataset_name}.inter", sep="\t")
    df.columns = [c.split(":")[0] for c in df.columns]
    stats = {
        "interactions": len(df),
        "users": df["user_id"].nunique(),
        "items": df["item_id"].nunique(),
        "density": len(df) / (df["user_id"].nunique() * df["item_id"].nunique()),
        "min_ts": int(df["timestamp"].min()),
        "max_ts": int(df["timestamp"].max()),
    }
    if "intensity" in df.columns:
        stats["intensity_min"] = float(df["intensity"].min())
        stats["intensity_max"] = float(df["intensity"].max())
        stats["intensity_mean"] = float(df["intensity"].mean())
    return stats
