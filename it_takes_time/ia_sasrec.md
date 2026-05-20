# IA-SASRec: Intensity-Aware Self-Attentive Sequential Recommendation

A framework that injects per-interaction **intensity** signals (ratings, hours
played, dwell time, click counts) into the self-attention mechanism of SASRec
(Kang & McAuley, ICDM 2018). The name parallels TiSASRec (Time-Interval Aware
SASRec) — a familiar cadence for reviewers in this domain.

This document is both the theoretical write-up of the method and the
implementation reference for the code in this repository.

---

## 1. Motivation

Vanilla SASRec treats every item in a user's history as a binary presence
signal: either an item `i` appears in user `u`'s sequence or it doesn't.
Real implicit-feedback data carries far richer information — *how much* the
user engaged with each item:

- MovieLens: explicit rating in `{1, 2, 3, 4, 5}`
- Amazon reviews: explicit rating in `{1, 2, 3, 4, 5}` per review (with native unix timestamps)
- News / e-commerce: dwell time, click frequency, purchase value
- Hours-played / listen counts: implicit but unbounded in `[0, ∞)` (out of scope here — the public hours-played benchmarks we surveyed all lack timestamps)

A literature sweep confirmed that while there is extensive work on optimizing
SASRec hyperparameters and on handling implicit feedback at the loss / sampling
level (e.g. weighted matrix factorization, Hu et al. 2008), modifying the
**core self-attention logits** with a continuous intensity multiplier — as
opposed to binary sequence presence — remains an unfilled gap. The closest
neighbours are TiSASRec (time-interval *positions*, not interaction strength)
and weighted-CE losses that do not touch the attention layer.

IA-SASRec fills that gap with three minimal, additive modifications to the
attention computation.

---

## 2. Background: standard SASRec attention

A SASRec encoder maps a user's left-padded item sequence
$s = [i_1, i_2, \dots, i_T]$ to embeddings $E \in \mathbb{R}^{T \times d}$.
Each transformer block projects $E$ to query, key, and value matrices
$Q, K, V \in \mathbb{R}^{T \times d}$ and computes scaled dot-product
self-attention with a causal mask $M \in \{0, -\infty\}^{T \times T}$:

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + M\right) V \tag{1} $$

The softmax is row-wise: row $q$ of the result distributes attention from
query position $q$ across all key positions $1 \dots q$. The intensity of
interaction $i_j$ plays **no role** in this computation — only its identity
(through the item-embedding lookup) and its position (through the positional
embedding) matter.

To inject intensity, let $w \in \mathbb{R}^T$ be the per-position intensity
vector for the user's history (post-normalisation; see §4). Define the
intensity matrix $M_W \in \mathbb{R}^{T \times T}$ whose every row equals
$w$ — i.e. $M_W[q, k] = w_k$. $M_W$ carries intensity along the **key axis**,
which is the axis along which the softmax normalises.

---

## 3. The three variants

All three are realised by exactly one modification to equation (1), in
[src/models/variants/ia_sasrec.py](src/models/variants/ia_sasrec.py).

### 3.1 `IA-SASRec-Add` — additive logit bias (pre-softmax)

**Idea.** Treat intensity as an explicit bias added to the raw attention
scores before they are converted into probabilities.

$$ \text{Attention}_{add}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + \lambda M_W + M\right) V \tag{2} $$

$\lambda \in \mathbb{R}$ is a **learnable scalar** (initialised to $1.0$,
one per layer), giving the model freedom to discover how much intensity
should matter.

**Behaviour.** When a historical item $i_j$ has very high $w_j$,
equation (2) directly inflates the logit on that column, forcing the
softmax to assign it large probability **regardless of semantic similarity
to the query**. Concretely, as $\lambda \cdot w_j$ grows the column
dominates and the limit recovers "attend hardest where engagement was
highest."

**Edge cases.** At $\lambda = 0$ the variant collapses to vanilla SASRec
attention exactly — a useful sanity check (verified in
[tests/test_ia_attention.py](tests/test_ia_attention.py)::`test_add_lambda_zero_matches_vanilla`).
At large $\lambda$ the variant becomes a hard intensity-argmax — verified
in the same test module.

### 3.2 `IA-SASRec-Mul` — multiplicative attention scaling (pre-softmax)

**Idea.** Instead of adding a bias, **scale** the semantic similarity by
the intensity of the key, with a learnable strength so the model can decide
how aggressively to do so.

$$ \text{Attention}_{mul}(Q, K, V) = \text{softmax}\left(\left(\frac{QK^T}{\sqrt{d}}\right) \odot \left(1 + \lambda_{mul} (M_W - 1)\right) + M\right) V \tag{3} $$

$\odot$ is element-wise multiplication. The mask $M$ is still added
*after* the scaling so padded keys remain at $-\infty$.

$\lambda_{mul} \in \mathbb{R}$ is a **learnable scalar** (initialised to
$1.0$, one per layer). The $(w_k - 1)$ parameterisation makes $\lambda$
interpolate between two endpoints: at $\lambda_{mul} = 0$ the multiplier is
$1$ and the variant collapses to vanilla SASRec; at $\lambda_{mul} = 1$ the
multiplier is exactly $w_k$, recovering the original hard-gatekeeper
formulation. This gives the model an escape hatch when intensity is
uninformative — without it, a noisy intensity signal cannot be turned off.

**Behaviour.** With normalised $w \in [0, 1]$, $\lambda_{mul} = 1$ shrinks
logits at low-intensity keys toward zero — but this *flattens* their
post-softmax probability toward uniform, not toward zero. For aggressive
suppression, $\lambda_{mul}$ would need to grow large.

**Edge cases.** At $\lambda_{mul} = 0$, the variant equals vanilla SASRec
attention — verified in `test_mul_lambda_zero_matches_vanilla`. At
$\lambda_{mul} = 1$ with all-zero intensity over visible keys, all attention
logits become zero and the softmax outputs a uniform distribution —
verified in `test_mul_lambda_one_collapses_zero_intensity`.

### 3.3 `IA-SASRec-Val` — post-softmax key reweighting

**Idea.** Leave the attention probabilities to be computed by the standard
softmax, then **reweight each key's contribution by its intensity** before
aggregating values. Functionally this is "soft attention reweighting":
intensity acts as a per-key gain applied to the attention distribution.

$$ \text{Attention}_{val}(Q, K, V) = \left( \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + M\right) \odot \left(1 + \lambda_{val} (M_W - 1)\right) \right) V \tag{4} $$

$\lambda_{val}$ is a learnable scalar per layer (initialised to $1.0$),
analogous to $\lambda_{mul}$. We deliberately do **not** renormalise the
reweighted distribution — the dense output projection absorbs scale, which
is the standard treatment in masked / sparse attention layers.

**Behaviour.** Unlike Mul (which acts on raw logits and can be diluted by
the softmax normalisation), Val acts directly on the attention probabilities
the model uses to mix values. This makes it the most direct of the three
mechanisms for *suppressing the contribution of low-intensity items in the
final user vector*.

**Why this differs from the original Val.** A previous formulation of Val
multiplied the **post-attention context** by *query-position* intensity:

$$ \text{ctx} \leftarrow \text{ctx} \odot \mathbf{w}_{\text{query}} $$

This was broken for ranking. At prediction time only the last query
position is read out, so the entire user vector gets multiplied by a single
scalar — every candidate item's score is rescaled by the same constant, and
the ordering is identical to vanilla SASRec. The new formulation reweights
**key-position** intensity inside the attention sum, which actually
changes which items dominate the final representation. The change is
**breaking**: prior `IA-SASRec-Val` results are not directly comparable.

**Edge cases.** At $\lambda_{val} = 0$ the variant equals vanilla SASRec —
verified in `test_val_lambda_zero_matches_vanilla`. With non-trivial
intensity, the attention distribution actually used in the matmul differs
from the un-modified softmax — verified in
`test_val_attention_distribution_changes_with_intensity`. Padded positions
have softmax probability zero (mask is $-\infty$ before softmax), so they
contribute zero to the output regardless of the reweighting.

### 3.4 Summary table

| Variant | Where intensity enters | $\lambda$ at $0$ | $\lambda$ at $1$ |
|---------|------------------------|------------------|------------------|
| `Add`   | additive logit bias: $QK^T/\sqrt{d} + \lambda M_W$              | vanilla SASRec | logits offset by $w$ |
| `Mul`   | multiplicative logit scale: $(QK^T/\sqrt{d}) \odot (1 + \lambda (M_W - 1))$ | vanilla SASRec | original $(QK^T/\sqrt{d}) \odot M_W$ |
| `Val`   | post-softmax key reweighting: $(\text{softmax}(\cdot) \odot (1 + \lambda (M_W - 1)))\, V$ | vanilla SASRec | $\text{softmax}(\cdot)$ scaled by $w$ per key |

All three variants share a single design discipline: **a learnable
$\lambda$ that collapses the variant to vanilla SASRec at $\lambda = 0$**.
This was a deliberate response to a prior negative result, in which the
non-learnable Mul and Val variants underperformed SASRec on every dataset
in the benchmark. Adding learnable strength turns the formulation into a
**proper ablation** of intensity injection, and the learned $\lambda$ per
layer per dataset becomes an interpretable signal about how strongly the
model leans on intensity (logged into `results/eval/*.json` via
`IASASRecBase.get_intensity_params()`).

---

## 4. Normalisation

Raw intensities can be wildly heterogeneous (Steam hours: $0.1 \to 1000+$;
review ratings: a bounded $\{1, \dots, 5\}$ which is much milder but still
warrants per-user scaling). Feeding raw values into a softmax causes
catastrophic collapse: $\exp(1000)$ overflows to $+\infty$ and the
attention distribution becomes a one-hot on the single highest-intensity
item. Normalisation is **load-bearing**, not cosmetic — even on bounded
rating scales, per-user MinMax avoids batch-level rating bias dominating
the attention.

Per the original `sasrec_advances.md` analysis (§4 of that file), MinMax
scaling per user or a $\log(1+x)$ transformation tame the long tail. The
variant base class implements four modes selectable via
`config["intensity_norm"]`, applied inside the model's `forward()` so the
saved `.inter` file keeps the human-readable original:

| Mode | Operation on $w \in \mathbb{R}^T$ with mask $m \in \{0, 1\}^T$ |
|------|----------------------------------------------------------------|
| `log1p_minmax` (default) | $w \leftarrow \log(1 + \max(w, 0))$, then $w \leftarrow w \,/\, \max_k(w_k \cdot m_k)$ |
| `minmax` | $w \leftarrow f + (1 - f) \cdot (w - \min) / (\max - \min)$ over masked entries, with floor $f = 0.1$ |
| `zscore` | $w \leftarrow (w - \mu) / \sigma$ over masked entries |
| `none`   | identity (sanity-check ablation) |

After every mode, padding positions are forcibly zeroed regardless of the
math above. Implementation: `normalise_intensity` in
[src/models/variants/ia_sasrec.py](src/models/variants/ia_sasrec.py).

**Minmax floor.** Without the floor $f$, the least-intense real item maps
to exactly $0$ after $(w - w_{\min}) / (w_{\max} - w_{\min})$ — making it
indistinguishable from padding. In `Mul` this zeros the corresponding
attention logit; in `Val` it zeros the key's contribution to the output.
Either way, one real interaction per sequence is silently thrown away. The
constant `MINMAX_FLOOR = 0.1` (module-level in
[src/models/variants/ia_sasrec.py](src/models/variants/ia_sasrec.py)) maps
real items into $[0.1, 1.0]$, keeping the lowest-intensity item distinct
from padding while preserving the relative ordering of the rest.
`log1p_minmax` and `zscore` are unchanged — `log1p_minmax` does not
subtract a per-sequence minimum and so does not suffer the same collapse.

`intensity_norm` is part of the Optuna search space — it doubles as a clean
ablation axis for the paper.

---

## 5. Implementation

### 5.1 Source layout

| File | Role |
|------|------|
| [src/models/variants/ia_sasrec.py](src/models/variants/ia_sasrec.py) | `IASASRecBase`, `IASASRecAdd/Mul/Val`, `IAMultiHeadAttention`, `IATransformerLayer`, `IATransformerEncoder`, `normalise_intensity` |
| [src/models/__init__.py](src/models/__init__.py) | Registers the three variants and `ia_sasrec_space`; inlines SASRec yaml defaults that RecBole skips for non-built-in classes |
| [src/data.py](src/data.py) | Writes the `intensity:float` column into `.inter`; HTTP-zip, gzip-json, and Kaggle downloaders |
| [src/config.py](src/config.py) | Adds `INTENSITY_FIELD` to common config; declares ML-1M, ML-100K, Amazon Digital Music, Amazon Office Products dataset specs |
| [src/runner.py](src/runner.py) | Passes the class object (not name) to RecBole's `Config` for custom variants; `invalidate_cache(dataset, model=None)` helper |
| [tests/](tests/) | pytest unit + regression tests covering attention math, normalisation modes, λ gradients, registry, runner, and data pipeline |

### 5.2 Data format

RecBole's atomic interaction file now carries a fourth typed column:

```text
user_id:token   item_id:token   timestamp:float   intensity:float
1               3186            978300019         5
1               1270            978300055         3
2               260             978824291         4
```

`intensity:float` is the **raw** signal (ratings 1–5 for MovieLens and Amazon
reviews) — normalisation happens inside the model so the file remains
human-readable.

`token` tells RecBole to remap user / item IDs to contiguous integers starting
at 1, which is required by the embedding layer (0 is reserved for padding).

### 5.3 How the intensity reaches the model

RecBole's `SequentialDataset._aug_presets()` walks every field in
`load_col["inter"]` and creates a list-form companion field per
`LIST_SUFFIX` (`"_list"` by default). FLOAT fields become `FLOAT_SEQ` of
shape `[batch, max_seq_len]`.

Because [src/config.py](src/config.py) declares

```python
load_col = {"inter": ["user_id", "item_id", "timestamp", "intensity"]}
INTENSITY_FIELD = "intensity"
```

every batch the trainer hands to the model contains
`interaction["intensity_list"]`, aligned position-by-position with
`interaction["item_id_list"]`. Inside the model:

```python
# IASASRecBase.calculate_loss / .predict / .full_sort_predict
item_seq = interaction[self.ITEM_SEQ]             # [B, T]
intensity_raw = interaction["intensity_list"]     # [B, T]
mask = item_seq != 0
intensity = normalise_intensity(intensity_raw, self.intensity_norm, mask)
seq_output = self.forward(item_seq, item_seq_len, intensity=intensity)
```

Baseline models simply never read `intensity_list`, so the **same** `.inter`
serves every model — apples-to-apples comparison.

### 5.4 Two subtleties that the test suite locks in

1. **Pass the class object to `Config`, not its name.** When RecBole's
   `Config(model="IASASRecAdd", …)` is called, it invokes
   `recbole.utils.get_model(name)`, which scans every model submodule
   including `exlib_recommender` — the latter imports `lightgbm`, which on
   macOS requires `libomp.dylib` and crashes with
   `OSError: Library not loaded` when it isn't installed. For custom
   variants we therefore pass the class directly
   ([src/runner.py:_build_config](src/runner.py); regression test in
   [tests/test_runner_build_config.py](tests/test_runner_build_config.py)).

2. **Inline SASRec's yaml defaults for custom subclasses.** RecBole looks up
   `<classname>.yaml` for internal config defaults; `IASASRecAdd.yaml` does
   not exist, so fields like `hidden_act`, `layer_norm_eps`,
   `initializer_range` are left as `None` and `FeedForward.get_hidden_act`
   crashes with `KeyError: None`. The IA-SASRec registry entries inline
   those three fields via `_sasrec_yaml_defaults()` in
   [src/models/__init__.py](src/models/__init__.py).

3. **RecBole's `Config` is not a dict.** It exposes `__getitem__` and
   `__contains__` but no `.get()`; `__getattr__` is overridden to look up
   keys, which means missing keys raise `AttributeError`, not `KeyError`.
   Use `config["k"] if "k" in config else default`.

---

## 6. Datasets

| Dataset | Intensity signal | Source | Notes |
|---------|------------------|--------|-------|
| **ml-1m**                    | Explicit rating 1–5 | GroupLens HTTP zip                                | Native unix timestamps; downloads automatically |
| **ml-100k**                  | Explicit rating 1–5 | GroupLens HTTP zip                                | Native unix timestamps; ~10× smaller than ml-1m |
| **amazon-digital-music**     | Explicit rating 1–5 | snap.stanford.edu JSON-gz (McAuley 2014 5-core)   | ~5.5k users × ~3.6k items × ~64k reviews; native `unixReviewTime` |
| **amazon-office-products**   | Explicit rating 1–5 | snap.stanford.edu JSON-gz (McAuley 2014 5-core)   | ~4.9k users × ~2.4k items × ~53k reviews; native `unixReviewTime` |

All four datasets are small-to-medium after `min_user_inter ≥ 5`,
`min_item_inter ≥ 5` filtering — fast enough for full Optuna HPO sweeps on
a single GPU within a coffee break (and tolerable on CPU).

---

## 7. Running the benchmark

```bash
# One-time: invalidate cached baseline results since the .inter schema
# changed (added the intensity column). After this, baselines retrain on
# the new file so the comparison stays apples-to-apples.
python -c "from runner import invalidate_cache; invalidate_cache('ml-100k')"

# Then open the notebook — IA-SASRec variants are in the default registry,
# so they slot in alongside the baselines automatically.
jupyter lab notebooks/0_benchmark.ipynb
```

Programmatic equivalent:

```python
from data import prepare_recbole_dataset
from hpo import run_optuna
from runner import train_and_eval

for ds in ["ml-1m", "ml-100k", "amazon-digital-music", "amazon-office-products"]:
    prepare_recbole_dataset(ds)
    for name in ["SASRec", "IA-SASRec-Add", "IA-SASRec-Mul", "IA-SASRec-Val"]:
        hpo = run_optuna(ds, name)
        train_and_eval(ds, name, best_params=hpo["best_params"])
```

Results land in `results/eval/<dataset>__<model>.json`. The runner is
resumable: cached records are returned unchanged on re-run unless `force=True`
is passed or `invalidate_cache` removed the JSON first.

---

## 8. Tests

```bash
pytest -q
```

**27 tests**, covering:

- `normalise_intensity` ranges and NaN-safety for every mode
- `IAMultiHeadAttention` math:
  - `Add` at `λ = 0` matches vanilla SASRec attention exactly (bit-identical)
  - `Add` at large `λ` concentrates attention on the highest-intensity key
  - `Mul` with zero intensity collapses to uniform attention over visible keys
  - `Val` leaves the attention probability distribution identical to vanilla
  - Padding mask remains effective after intensity injection (all three modes)
- Full forward + backward for each variant, including learnable-`λ` gradient
  flow into the additive variant
- Registry: variants registered, SASRec yaml defaults inlined, HPO space
  exposes `intensity_norm`
- Runner: custom-variant `Config` construction bypasses RecBole's `get_model`
  scan; built-ins still use the name-based lookup
- Data: 4-column `.inter` written correctly on tiny synthetic fixtures for
  each download path (HTTP zip, gzip-JSON, Kaggle); no network or Kaggle auth
  required for the tests

---

## 9. Suggested paper structure

1. **Introduction.** Vanilla SASRec discards interaction strength; this is
   wasteful because the signal is logged anyway in any production
   recommender. IA-SASRec recovers it through three minimally invasive
   attention modifications.
2. **Related Work.** SASRec, TiSASRec, BERT4Rec, GRU4Rec, NARM, FPMC;
   implicit-feedback weighting at the loss level (Hu et al. 2008, Pan et al.
   2008); contrast: those weight the *objective*, IA-SASRec weights the
   *attention*.
3. **Method: Framework for Intensity-Aware Sequential Recommendation.**
   Present the three variants (Add / Mul / Val) as a unified framework —
   each corresponds to one of the three algebraic positions in the attention
   expression where `w` can be inserted. Derive equations (2)–(4) from
   equation (1).
4. **Experiments.** Four datasets, all with bounded explicit-rating intensity
   and native unix timestamps: ML-1M, ML-100K, Amazon Digital Music 5-core,
   Amazon Office Products 5-core. Two domains (movies, products) × two scales
   let us separate domain effects from data-volume effects. All 11 models × 4
   datasets. Optuna TPE HPO with NDCG@10 as the primary metric. Eval:
   full-vocabulary scoring → HR / NDCG / MRR / Recall / Precision @ {10, 20,
   50, 100}.
5. **Ablation: variants × normalisation.**
   `{Add, Mul, Val} × {log1p_minmax, minmax, zscore, none}` — a 3×4 table per
   dataset. Demonstrates that (a) normalisation is load-bearing even on
   bounded rating scales, and (b) the optimal variant depends on the dataset
   (hypothesis: `Add` wins on the bounded rating regimes here; `Mul` would
   favour noisy long-tail signals if added later via a hours-played-style
   dataset; `Val` is a safe default).
6. **Discussion.** Per-dataset winner analysis; when each variant helps and
   why; learned-$\lambda$ analysis (does the model shrink $\lambda$ toward
   $0$ on datasets where intensity is uninformative, and grow it elsewhere?);
   cost analysis (extra parameters: one scalar $\lambda$ per layer per
   variant — `Add`, `Mul`, `Val` all add `n_layers` scalars).
7. **Conclusion.** IA-SASRec is a drop-in upgrade for any SASRec deployment
   that already logs interaction strength — at the cost of one learnable
   scalar per layer per variant. The unified $(1 + \lambda(w - 1))$
   parameterisation lets the model gracefully fall back to vanilla SASRec on
   datasets where intensity is noise, while still capturing the signal when
   it's present.

---

## 10. Notes & caveats

- **Reproducibility.** Optuna studies persist to SQLite under `results/hpo/`;
  individual model results persist to JSON under `results/eval/`. Both layers
  are resumable: kill and re-run the notebook freely.
- **No hours-played-style benchmark.** The popular hours-played datasets
  (Steam-200k, HetRec LastFM-2K's aggregated listen counts) ship without
  per-event timestamps, so they can't drive a sequential model honestly.
  A future swap to McAuley's Steam Reviews (which has `unix_timestamp`
  and hours per review) could revisit hours-played-as-intensity properly.
- **`λ` per layer.** Each of `n_layers` IA-SASRec transformer blocks (for
  all three variants) has its own learnable `λ`, exposed at training end via
  `IASASRecBase.get_intensity_params()` and persisted in
  `results/eval/<dataset>__<model>.json` under the `intensity_params` key.
  If you want a single global `λ`, share the parameter across layers in
  `IATransformerEncoder.__init__`.
- **Generalisation.** The "intensity" framework is dataset-agnostic — anything
  you can express as a per-interaction `float` works (review helpfulness
  votes, listening counts, scroll depth, purchase value, etc.).
