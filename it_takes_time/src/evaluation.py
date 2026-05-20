"""Aggregate eval JSONs into a comparison table; produce top-K recommendations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from config import EVAL_DIR


def aggregate_results(
    dataset_name: str | None = None,
    include_std: bool = True,
) -> pd.DataFrame:
    """Load every saved (dataset, model) result and return a sorted DataFrame.

    Per-seed JSONs are grouped by (dataset, model) and aggregated via
    :func:`runner.aggregate_seed_records` so each model contributes a single
    row regardless of seed count.

    Parameters:
        dataset_name: If provided, only rows whose ``"dataset"`` field matches
            this value are included. Pass ``None`` to aggregate across all datasets.
        include_std: When ``True`` (default), add ``<metric>_std`` columns and
            an ``n_seeds`` column derived from the per-seed aggregation.

    Returns:
        DataFrame with one row per (dataset, model), sorted descending by
        ``ndcg@10`` (or the last metric column if that key is absent). Empty
        DataFrame if no JSON files exist under ``EVAL_DIR``.
    """
    # Lazy import to avoid circular dependency at module load time.
    from runner import load_result

    pairs: set[tuple[str, str]] = set()
    for p in sorted(EVAL_DIR.glob("*.json")):
        # Filenames are either "<ds>__<model>.json" (legacy) or
        # "<ds>__<model>__seed<N>.json" (per-seed). Strip the optional seed
        # suffix to recover the (dataset, model) pair.
        stem = p.stem
        head, _, tail = stem.rpartition("__")
        if head and tail.startswith("seed"):
            stem = head
        ds, _, model = stem.partition("__")
        if not ds or not model:
            continue
        if dataset_name and ds != dataset_name:
            continue
        pairs.add((ds, model))

    rows = []
    for ds, model in sorted(pairs):
        try:
            rec = load_result(ds, model)
        except FileNotFoundError:
            continue
        row = {
            "dataset": rec["dataset"],
            "model": rec["model"],
            "train_seconds": rec.get("train_seconds"),
            "best_valid_score": rec.get("best_valid_score"),
            "n_seeds": rec.get("n_seeds", 1),
        }
        for k, v in (rec.get("test_result") or {}).items():
            row[k] = v
        if include_std:
            for k, v in (rec.get("test_result_std") or {}).items():
                row[f"{k}_std"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        sort_col = "ndcg@10" if "ndcg@10" in df.columns else df.columns[-1]
        df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    return df


def top_k_recommend(
    model: Any,
    dataset: Any,
    test_data: Any,
    user_external_ids: Sequence[Any],
    k: int = 100,
    mask_seen: bool = True,
) -> pd.DataFrame:
    """Return top-K item recommendations for given external user IDs.

    Implements the "next-1 model -> top-K" path: score every item at the next
    position, mask the user's history (and padding), take ``argpartition`` top-K,
    sort that slice. This is the standard sequential-recommender serving path.

    Parameters:
        model: Trained RecBole model (must be in eval mode or will be set to eval).
        dataset: RecBole ``Dataset`` object used during training, providing
            ``token2id`` / ``id2token`` mappings and interaction history.
        test_data: RecBole test dataloader; its ``dataset.inter_feat`` is passed to
            ``full_sort_predict`` when the model supports it.
        user_external_ids: External (string/int) user IDs to generate recommendations
            for. Must be present in the dataset's user vocabulary.
        k: Number of top items to return per user.
        mask_seen: If ``True``, items already in the user's interaction history are
            masked to ``-inf`` before taking the top-K.

    Returns:
        DataFrame with columns ``user_id``, ``rank`` (1-based), ``item_id``, and
        ``score``, one row per (user, rank) pair.
    """
    model.eval()
    device = next(model.parameters()).device
    uid_field = dataset.uid_field
    iid_field = dataset.iid_field

    uid_series = dataset.token2id(uid_field, np.asarray([str(u) for u in user_external_ids]))
    uid_tensor = torch.as_tensor(uid_series, device=device, dtype=torch.long)

    with torch.no_grad():
        try:
            interaction = test_data.dataset.inter_feat.to(device)
            scores = model.full_sort_predict(interaction)
        except (NotImplementedError, AttributeError):
            n_items = dataset.item_num
            all_items = torch.arange(n_items, device=device).repeat(len(uid_tensor))
            users_rep = uid_tensor.repeat_interleave(n_items)
            scores = model.predict_in_batch_users_items(users_rep, all_items).view(len(uid_tensor), n_items)

    if scores.dim() == 1:
        scores = scores.view(len(uid_tensor), -1)
    scores[:, 0] = -float("inf")  # padding item

    if mask_seen:
        history_matrix, _, _ = dataset.history_item_matrix()
        history_matrix = history_matrix.to(device)
        for row, uid in enumerate(uid_tensor):
            seen = history_matrix[uid]
            scores[row, seen] = -float("inf")

    top_scores, top_idx = torch.topk(scores, k=k, dim=1)
    top_idx_np = top_idx.cpu().numpy()
    top_scores_np = top_scores.cpu().numpy()

    rows = []
    for u_ext, idx_row, score_row in zip(user_external_ids, top_idx_np, top_scores_np):
        item_externals = dataset.id2token(iid_field, idx_row)
        for rank, (iid, sc) in enumerate(zip(item_externals, score_row), start=1):
            rows.append({"user_id": u_ext, "rank": rank, "item_id": iid, "score": float(sc)})
    return pd.DataFrame(rows)
