"""Aggregate eval JSONs into a comparison table; produce top-K recommendations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from config import EVAL_DIR


def aggregate_results(dataset_name: str | None = None) -> pd.DataFrame:
    rows = []
    for p in sorted(EVAL_DIR.glob("*.json")):
        rec = json.loads(p.read_text())
        if dataset_name and rec.get("dataset") != dataset_name:
            continue
        row = {
            "dataset": rec["dataset"],
            "model": rec["model"],
            "train_seconds": rec.get("train_seconds"),
            "best_valid_score": rec.get("best_valid_score"),
        }
        for k, v in (rec.get("test_result") or {}).items():
            row[k] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        sort_col = "ndcg@10" if "ndcg@10" in df.columns else df.columns[-1]
        df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    return df


def top_k_recommend(
    model,
    dataset,
    test_data,
    user_external_ids: Sequence,
    k: int = 100,
    mask_seen: bool = True,
) -> pd.DataFrame:
    """Return top-K item recommendations for given external user IDs.

    Implements the "next-1 model -> top-K" path: score every item at the next
    position, mask the user's history (and padding), take ``argpartition`` top-K,
    sort that slice. This is the standard sequential-recommender serving path.
    """
    model.eval()
    device = next(model.parameters()).device
    uid_field = dataset.uid_field
    iid_field = dataset.iid_field

    uid_series = dataset.token2id(uid_field, np.asarray([str(u) for u in user_external_ids]))
    uid_tensor = torch.as_tensor(uid_series, device=device, dtype=torch.long)

    with torch.no_grad():
        try:
            interaction = test_data.dataset.inter_feat
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
