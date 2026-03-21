"""Dataset loading utilities.

All datasets are stored under the project's ``data/`` directory (repo root),
each in its own subfolder::

    data/
    ├── lastfm_360k/
    │   └── usersha1-artmbid-artname-plays.tsv   ← auto-downloaded
    ├── steam/
    │   └── steam_recommendations.csv            ← manual
    └── taste_profile/
        └── train_triplets.txt                   ← manual

Last.fm 360K can be downloaded automatically; Steam and Taste Profile require
a manual download — see each function's docstring for instructions.
"""

import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd


def _resolve_data_dir(data_dir: "str | os.PathLike | None") -> Path:
    """Return the data directory as an absolute Path.

    Defaults to ``<repo_root>/data/``, where repo root is inferred as two
    levels above this file (``src/utils/datasets.py``).
    """
    if data_dir is not None:
        return Path(data_dir).resolve()
    return Path(__file__).resolve().parents[2] / "data"


# ---------------------------------------------------------------------------
# Last.fm 360K
# ---------------------------------------------------------------------------

_LASTFM_URL = (
    "http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-360K.tar.gz"
)
_LASTFM_REL = "lastfm_360k/usersha1-artmbid-artname-plays.tsv"


def download_lastfm_360k(data_dir=None) -> Path:
    """Download and extract the Last.fm 360K dataset into *data_dir*.

    Source: http://ocelma.net/MusicRecommendationDataset/lastfm-360K.html

    The play-count TSV is placed at ``data/lastfm_360k/``.
    Returns the path to the extracted TSV file.
    """
    root = _resolve_data_dir(data_dir)
    dest = root / _LASTFM_REL
    if dest.exists():
        print(f"Last.fm 360K already present at {dest}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "lastfm-dataset-360K.tar.gz"
        print(f"Downloading Last.fm 360K → {archive} …")
        urllib.request.urlretrieve(_LASTFM_URL, archive)

        print("Extracting …")
        with tarfile.open(archive) as tar:
            tar.extractall(tmp)

        src = (
            Path(tmp)
            / "lastfm-dataset-360K"
            / "usersha1-artmbid-artname-plays.tsv"
        )
        shutil.move(str(src), dest)

    print(f"Done. Dataset at {dest}")
    return dest


def load_lastfm_360k(data_dir=None, download: bool = True) -> pd.DataFrame:
    """Load the Last.fm 360K user–artist play-count dataset.

    Columns returned: ``user_id``, ``item_id``, ``target`` (play count).

    Args:
        data_dir: Path to the data directory. Defaults to
            ``<repo_root>/data/``.
        download: Download automatically if the file is missing (default:
            True).
    """
    root = _resolve_data_dir(data_dir)
    path = root / _LASTFM_REL

    if not path.exists():
        if download:
            path = download_lastfm_360k(data_dir)
        else:
            raise FileNotFoundError(
                f"{path} not found.\n"
                "Call download_lastfm_360k() or set download=True."
            )

    return (
        pd.read_csv(
            path,
            sep="\t",
            header=None,
            usecols=[0, 2, 3],
            names=["user_id", "item_id", "play_count"],
        )
        .loc[:, ["user_id", "item_id", "play_count"]]
        .dropna()
        .rename(columns={"play_count": "target"})
    )


# ---------------------------------------------------------------------------
# Steam
# ---------------------------------------------------------------------------

_STEAM_REL = "steam/steam_recommendations.csv"


def load_steam(data_dir=None) -> pd.DataFrame:
    """Load the Steam game-hours dataset.

    Manual download
    ~~~~~~~~~~~~~~~
    1. Go to https://www.kaggle.com/datasets/antonkozyriev/
       game-recommendations-on-steam
    2. Download ``recommendations.csv`` and save it as::

           data/steam/steam_recommendations.csv

    Columns returned: ``user_id``, ``item_id`` (app_id), ``target`` (hours).
    """
    root = _resolve_data_dir(data_dir)
    path = root / _STEAM_REL

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n\n"
            "Download instructions:\n"
            "  1. Visit https://www.kaggle.com/datasets/"
            "antonkozyriev/game-recommendations-on-steam\n"
            "  2. Download recommendations.csv\n"
            "  3. Save it as  data/steam/steam_recommendations.csv"
        )

    return (
        pd.read_csv(path, usecols=["user_id", "app_id", "hours"])
        .loc[:, ["user_id", "app_id", "hours"]]
        .drop_duplicates()
        .dropna()
        .rename(columns={"app_id": "item_id", "hours": "target"})
    )


# ---------------------------------------------------------------------------
# Million Song Dataset — Taste Profile
# ---------------------------------------------------------------------------

_TASTE_REL = "taste_profile/train_triplets.txt"


def load_taste_profile(data_dir=None) -> pd.DataFrame:
    """Load the Echo Nest Taste Profile Subset (Million Song Dataset).

    Manual download
    ~~~~~~~~~~~~~~~
    1. Visit http://millionsongdataset.com/tasteprofile/ and agree to the
       terms.
    2. Download ``train_triplets.txt`` and save it as::

           data/taste_profile/train_triplets.txt

    Columns returned: ``user_id``, ``item_id``, ``target`` (play count).
    """
    root = _resolve_data_dir(data_dir)
    path = root / _TASTE_REL

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n\n"
            "Download instructions:\n"
            "  1. Visit http://millionsongdataset.com/tasteprofile/\n"
            "  2. Agree to the terms and download train_triplets.txt\n"
            "  3. Save it as  data/taste_profile/train_triplets.txt"
        )

    return pd.read_table(
        path,
        sep="\t",
        header=None,
        usecols=[0, 1, 2],
        names=["user_id", "item_id", "target"],
    )
