"""IA-SASRec: Intensity-Aware SASRec variants.

Three variants inject the per-interaction intensity weight ``w_j`` (e.g. rating,
hours played) into the self-attention mechanism of the base SASRec encoder
(Kang & McAuley, 2018). All three share a learnable scalar ``lambda`` per
transformer block (initialised to 1.0) that collapses the variant to vanilla
SASRec at ``lambda = 0``:

* ``IASASRecAdd``  additive logit bias:
      softmax(QK^T/sqrt(d) + lambda * w_k) V

* ``IASASRecMul``  multiplicative logit scaling:
      softmax((QK^T/sqrt(d)) * (1 + lambda * (w_k - 1))) V

* ``IASASRecVal``  post-softmax attention reweighting:
      (softmax(QK^T/sqrt(d)) * (1 + lambda * (w_k - 1))) V

The intensity vector reaches the model via RecBole's automatic sequence
expansion: the ``intensity:float`` column in ``.inter`` becomes
``interaction["intensity_list"]`` of shape ``[B, max_seq_len]``.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn

from recbole.model.sequential_recommender.sasrec import SASRec


# Floor used by the "minmax" normalisation so the least-intense real item maps
# to MINMAX_FLOOR (not 0). Without this, after `(w - wmin)/(wmax - wmin)` the
# lowest real item is indistinguishable from padding, which silently throws
# away one real interaction per sequence in Mul/Val variants.
MINMAX_FLOOR = 0.1


def normalise_intensity(w: torch.Tensor, mode: str, mask: torch.Tensor) -> torch.Tensor:
    """Normalise per-user intensity sequences to a numerically safe range.

    Parameters:
        w: Raw intensity tensor of shape ``[B, T]``.
        mode: One of ``"log1p_minmax"``, ``"minmax"``, ``"zscore"``, or ``"none"``.
        mask: Boolean tensor of shape ``[B, T]``; ``True`` where the position is
            a real item, ``False`` for left-padding. Padding positions are
            zeroed out after normalisation regardless of mode.

    Returns:
        Normalised tensor of the same shape, padding positions set to 0.
    """
    w = w.float()
    if mode == "log1p_minmax":
        w = torch.log1p(w.clamp(min=0))
        w = w * mask.float()
        wmax = w.amax(dim=1, keepdim=True).clamp(min=1e-8)
        w = w / wmax
    elif mode == "minmax":
        w = w * mask.float()
        wmin = w.masked_fill(~mask, float("inf")).amin(dim=1, keepdim=True)
        wmax = w.masked_fill(~mask, float("-inf")).amax(dim=1, keepdim=True)
        denom = (wmax - wmin).clamp(min=1e-8)
        w = MINMAX_FLOOR + (1.0 - MINMAX_FLOOR) * (w - wmin) / denom
    elif mode == "zscore":
        m = mask.float()
        cnt = m.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean = (w * m).sum(dim=1, keepdim=True) / cnt
        var = (((w - mean) * m) ** 2).sum(dim=1, keepdim=True) / cnt
        w = (w - mean) / (var.sqrt() + 1e-8)
    elif mode == "none":
        pass
    else:
        raise ValueError(f"Unknown intensity_norm mode: {mode}")
    return w * mask.float()


class IAMultiHeadAttention(nn.Module):
    """SASRec multi-head attention augmented with an intensity signal.

    Behaves identically to :class:`recbole.model.layers.MultiHeadAttention` when
    the intensity vector is absent; injects intensity according to
    :attr:`intensity_mode` when present.
    """

    def __init__(
        self,
        n_heads: int,
        hidden_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        layer_norm_eps: float,
        intensity_mode: str,
    ):
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                f"hidden_size {hidden_size} not divisible by n_heads {n_heads}"
            )
        self.num_attention_heads = n_heads
        self.attention_head_size = hidden_size // n_heads
        self.all_head_size = hidden_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)
        self.intensity_mode = intensity_mode

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

        # Every intensity mode is gated by a learnable scalar λ, initialised
        # to 1.0 so behaviour at init matches the original hard-wired variant.
        # At λ=0 every variant collapses to vanilla SASRec, giving the model
        # an escape hatch when intensity is uninformative.
        self.intensity_lambda = nn.Parameter(torch.ones(1))

        # Diagnostic cache for the last forward's post-softmax attention probs.
        # Off by default to avoid holding tensors during normal training; tests
        # flip `capture_probs = True` on the module to enable.
        self.capture_probs = False
        self.last_probs: torch.Tensor | None = None

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(*new_shape)

    def forward(
        self,
        input_tensor: torch.Tensor,
        attention_mask: torch.Tensor,
        intensity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute attention with optional intensity injection.

        Parameters:
            input_tensor: ``[B, T, H]`` token embeddings.
            attention_mask: ``[B, 1, 1, T]`` additive mask (``0`` keep, ``-inf`` drop).
            intensity: ``[B, T]`` per-position weights (already normalised) or ``None``.

        Returns:
            ``[B, T, H]`` output tensor.
        """
        q = self.transpose_for_scores(self.query(input_tensor)).permute(0, 2, 1, 3)
        k = self.transpose_for_scores(self.key(input_tensor)).permute(0, 2, 3, 1)
        v = self.transpose_for_scores(self.value(input_tensor)).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k) / self.sqrt_attention_head_size

        if intensity is not None and self.intensity_mode == "add":
            # w_k broadcast across heads (dim 1) and queries (dim 2):
            #   [B, 1, 1, T]  added to  [B, H, T, T]
            w_k = intensity[:, None, None, :]
            scores = scores + self.intensity_lambda * w_k
        elif intensity is not None and self.intensity_mode == "mul":
            # λ=0 → vanilla SASRec; λ=1 → scores * w_k (original Mul).
            w_k = intensity[:, None, None, :]
            scores = scores * (1.0 + self.intensity_lambda * (w_k - 1.0))

        # Padding/causal mask is applied AFTER intensity, so padded keys still
        # collapse to zero probability regardless of the variant.
        scores = scores + attention_mask
        probs = self.attn_dropout(self.softmax(scores))

        if intensity is not None and self.intensity_mode == "val":
            # Soft attention reweighting by key-position intensity, applied to
            # the post-softmax probs *before* the matmul with V. λ=0 → vanilla;
            # λ=1 → each key's attention probability scales by its intensity.
            # Equivalently A·(D_λ·V) since the factor depends only on k — hence
            # the name "Val". No renormalisation: the post-attention LayerNorm
            # in the residual block below normalises any scale inflation,
            # while the direction change in the output vector survives.
            w_k = intensity[:, None, None, :]
            probs = probs * (1.0 + self.intensity_lambda * (w_k - 1.0))

        if self.capture_probs:
            self.last_probs = probs.detach()

        ctx = torch.matmul(probs, v)

        ctx = ctx.permute(0, 2, 1, 3).contiguous()
        ctx = ctx.view(ctx.size(0), ctx.size(1), self.all_head_size)
        out = self.dense(ctx)
        out = self.out_dropout(out)
        return self.LayerNorm(out + input_tensor)


class IATransformerLayer(nn.Module):
    """One transformer block: IA-aware self-attention + standard FeedForward."""

    def __init__(
        self,
        n_heads,
        hidden_size,
        inner_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
        intensity_mode,
    ):
        super().__init__()
        from recbole.model.layers import FeedForward  # local import to avoid recbole side-effects at import time

        self.multi_head_attention = IAMultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob,
            layer_norm_eps, intensity_mode,
        )
        self.feed_forward = FeedForward(
            hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps,
        )

    def forward(self, hidden_states, attention_mask, intensity):
        att = self.multi_head_attention(hidden_states, attention_mask, intensity)
        return self.feed_forward(att)


class IATransformerEncoder(nn.Module):
    """Stack of :class:`IATransformerLayer` blocks."""

    def __init__(
        self,
        n_layers,
        n_heads,
        hidden_size,
        inner_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
        intensity_mode,
    ):
        super().__init__()
        layer = IATransformerLayer(
            n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob,
            hidden_act, layer_norm_eps, intensity_mode,
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask, intensity, output_all_encoded_layers=True):
        all_outs = []
        for blk in self.layer:
            hidden_states = blk(hidden_states, attention_mask, intensity)
            if output_all_encoded_layers:
                all_outs.append(hidden_states)
        if not output_all_encoded_layers:
            all_outs.append(hidden_states)
        return all_outs


class IASASRecBase(SASRec):
    """Base class for Intensity-Aware SASRec variants.

    Subclasses set ``intensity_mode`` to one of ``"add" | "mul" | "val"``.
    """

    intensity_mode: str = "add"  # overridden by subclasses
    # When True, raise if `intensity_list` is absent at runtime instead of
    # silently degrading to vanilla SASRec. Tests that hand-craft batches
    # without the field can flip this to False.
    strict_intensity: bool = True

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        # Replace the vanilla encoder with an intensity-aware stack.
        self.trm_encoder = IATransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            intensity_mode=self.intensity_mode,
        )
        self.intensity_field = config["INTENSITY_FIELD"] or "intensity"
        self.intensity_list_field = self.intensity_field + config["LIST_SUFFIX"]
        # RecBole's Config only supports __getitem__/__contains__, not .get().
        self.intensity_norm = (
            config["intensity_norm"] if "intensity_norm" in config else "log1p_minmax"
        )
        # Re-initialise the freshly created encoder weights.
        self.trm_encoder.apply(self._init_weights)

    def _resolve_intensity(self, interaction, item_seq):
        """Pull and normalise the intensity sequence from the interaction batch."""
        if self.intensity_list_field not in interaction.interaction:
            if self.strict_intensity:
                raise RuntimeError(
                    f"IA-SASRec batch is missing '{self.intensity_list_field}'. "
                    f"Present keys: {sorted(interaction.interaction.keys())}. "
                    f"This usually means RecBole loaded the wrong .inter file — "
                    f"e.g. the dataset name is literally 'ml-100k', which RecBole "
                    f"1.2.0 hard-overrides to its bundled 3-column example. "
                    f"Rename the dataset and check `cfg['data_path']`."
                )
            return None
        raw = interaction[self.intensity_list_field]
        mask = item_seq != 0
        return normalise_intensity(raw, self.intensity_norm, mask)

    def forward(self, item_seq, item_seq_len, intensity=None):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, intensity, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B, H]

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        intensity = self._resolve_intensity(interaction, item_seq)
        seq_output = self.forward(item_seq, item_seq_len, intensity=intensity)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_emb = self.item_embedding(pos_items)
            neg_emb = self.item_embedding(neg_items)
            pos_score = (seq_output * pos_emb).sum(dim=-1)
            neg_score = (seq_output * neg_emb).sum(dim=-1)
            return self.loss_fct(pos_score, neg_score)
        logits = seq_output @ self.item_embedding.weight.T
        return self.loss_fct(logits, pos_items)

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        intensity = self._resolve_intensity(interaction, item_seq)
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len, intensity=intensity)
        test_item_emb = self.item_embedding(test_item)
        return (seq_output * test_item_emb).sum(dim=1)

    def get_intensity_params(self) -> dict[str, float]:
        """Return the learned per-layer intensity λ values.

        Used by the runner to record λ alongside metrics, so the benchmark
        notebook can plot how strongly the model leans on intensity per dataset.
        """
        return {
            f"layer_{i}_lambda": float(
                blk.multi_head_attention.intensity_lambda.detach().cpu()
            )
            for i, blk in enumerate(self.trm_encoder.layer)
        }

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        intensity = self._resolve_intensity(interaction, item_seq)
        seq_output = self.forward(item_seq, item_seq_len, intensity=intensity)
        return seq_output @ self.item_embedding.weight.T


class IASASRecAdd(IASASRecBase):
    """Additive logit bias variant: ``softmax(QK^T/sqrt(d) + lambda * w) V``."""
    intensity_mode = "add"


class IASASRecMul(IASASRecBase):
    """Multiplicative scaling variant: ``softmax((QK^T/sqrt(d)) * w) V``."""
    intensity_mode = "mul"


class IASASRecVal(IASASRecBase):
    """Value modulation variant: ``softmax(QK^T/sqrt(d)) (V * w)``."""
    intensity_mode = "val"
