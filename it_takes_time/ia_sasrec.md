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
- Steam: hours played in `[0, ∞)` (typical tail goes to 1000+)
- News / e-commerce: dwell time, click frequency, purchase value

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
the intensity of the key. Two items must be both relevant *and* engaged
with to attract attention.

$$ \text{Attention}_{mul}(Q, K, V) = \text{softmax}\left(\left(\frac{QK^T}{\sqrt{d}}\right) \odot M_W + M\right) V \tag{3} $$

$\odot$ is element-wise multiplication. Note the mask $M$ is still added
*after* the multiplication so padded keys remain at $-\infty$.

**Behaviour.** Acts as a strict **gatekeeper**: if the dot product
$QK^T / \sqrt{d}$ says "this key looks relevant" but $w_k \approx 0$ (an
accidental 1-second click), the multiplication crushes the score to $\sim 0$
before the softmax sees it. The opposite is also true — a high-intensity
key that is semantically irrelevant (small or negative dot product) gets
its score amplified in magnitude, not necessarily in the helpful direction.

**Edge cases.** With all-zero intensity over visible keys, all attention
logits become zero and the softmax outputs a uniform distribution over
those keys — verified in `test_mul_zero_intensity_collapses_logits`.

### 3.3 `IA-SASRec-Val` — value modulation (post-softmax)

**Idea.** Leave the attention probability distribution **completely alone**
and only amplify the actual value vectors that get aggregated.

$$ \text{Attention}_{val}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + M\right) (V \odot \mathbf{w}) \tag{4} $$

$\mathbf{w}$ is broadcast across the embedding dimension of $V$: each
value row $V_k$ is multiplied by the scalar $w_k$.

**Behaviour.** The same historical pattern is learned, but when the model
aggregates the sequence the representations of highly-interacted items
contribute with larger magnitude than weakly-interacted items. This is the
softest of the three modifications: it never changes *which* items the
model attends to, only the **magnitude of their contribution**.

**Edge cases.** The attention probability distribution is identical to
vanilla SASRec for any $\mathbf{w}$ — verified bit-for-bit in
`test_val_attention_distribution_unchanged`. Padded positions are zeroed
in $\mathbf{w}$ before this step, so they make zero contribution to the
output even though the softmax may assign them small probability mass.

### 3.4 Summary table

| Variant | Pre-softmax score | Post-softmax aggregation | What it controls |
|---------|------------------|--------------------------|------------------|
| `Add`   | $QK^T/\sqrt{d} + \lambda M_W$ | $\text{softmax}(\cdot)\, V$       | logits (learnable strength) |
| `Mul`   | $(QK^T/\sqrt{d}) \odot M_W$   | $\text{softmax}(\cdot)\, V$       | logits (hard gatekeeper)    |
| `Val`   | $QK^T/\sqrt{d}$               | $\text{softmax}(\cdot)\, (V \odot \mathbf{w})$ | aggregated magnitude only |

These are the three obvious "first-derivative" hooks in the attention
expression — one for each algebraic position where $\mathbf{w}$ can be
introduced without changing the rest of the architecture.

---

## 4. Normalisation

Raw intensities are wildly heterogeneous (Steam hours: $0.1 \to 1000+$).
Feeding raw values into a softmax causes catastrophic collapse:
$\exp(1000)$ overflows to $+\infty$ and the attention distribution becomes
a one-hot on the single highest-intensity item. Normalisation is
**load-bearing**, not cosmetic.

Per the original `sasrec_advances.md` analysis (§4 of that file), MinMax
scaling per user or a $\log(1+x)$ transformation tame the long tail. The
variant base class implements four modes selectable via
`config["intensity_norm"]`, applied inside the model's `forward()` so the
saved `.inter` file keeps the human-readable original:

| Mode | Operation on $w \in \mathbb{R}^T$ with mask $m \in \{0, 1\}^T$ |
|------|----------------------------------------------------------------|
| `log1p_minmax` (default) | $w \leftarrow \log(1 + \max(w, 0))$, then $w \leftarrow w \,/\, \max_k(w_k \cdot m_k)$ |
| `minmax` | $w \leftarrow (w - \min) / (\max - \min)$ over masked entries |
| `zscore` | $w \leftarrow (w - \mu) / \sigma$ over masked entries |
| `none`   | identity (sanity-check ablation) |

After every mode, padding positions are forcibly zeroed regardless of the
math above. Implementation: `normalise_intensity` in
[src/models/variants/ia_sasrec.py](src/models/variants/ia_sasrec.py).

`intensity_norm` is part of the Optuna search space — it doubles as a clean
ablation axis for the paper.

---

## 5. Implementation

### 5.1 Source layout

| File | Role |
|------|------|
| [src/models/variants/ia_sasrec.py](src/models/variants/ia_sasrec.py) | `IASASRecBase`, `IASASRecAdd/Mul/Val`, `IAMultiHeadAttention`, `IATransformerLayer`, `IATransformerEncoder`, `normalise_intensity` |
| [src/models/__init__.py](src/models/__init__.py) | Registers the three variants and `ia_sasrec_space`; inlines SASRec yaml defaults that RecBole skips for non-built-in classes |
| [src/data.py](src/data.py) | Writes the `intensity:float` column into `.inter`; Steam downloader via `kagglehub` |
| [src/config.py](src/config.py) | Adds `INTENSITY_FIELD` to common config; declares ML-1M and Steam dataset specs |
| [src/runner.py](src/runner.py) | Passes the class object (not name) to RecBole's `Config` for custom variants; `invalidate_cache(dataset, model=None)` helper |
| [tests/](tests/) | 27 pytest unit / regression tests |

### 5.2 Data format

RecBole's atomic interaction file now carries a fourth typed column:

```text
user_id:token   item_id:token   timestamp:float   intensity:float
1               3186            978300019         5
1               1270            978300055         3
2               260             978824291         4
```

`intensity:float` is the **raw** signal (ratings 1–5 for MovieLens, hours for
Steam) — normalisation happens inside the model so the file remains
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
| **ml-1m**  | Explicit rating 1–5  | GroupLens HTTP zip                                       | Downloads automatically |
| **steam**  | Hours played         | Kaggle `tamber/steam-video-games` via `kagglehub`        | Drops `behavior == purchase` rows; synthesises monotonic timestamps by stable-sorting on hours within each user; needs `~/.kaggle/kaggle.json` once |

Both are medium-sized after `min_user_inter ≥ 5`, `min_item_inter ≥ 5`
filtering — fast enough for full Optuna HPO sweeps on a single GPU within
a coffee break.

---

## 7. Running the benchmark

```bash
# One-time: invalidate cached baseline results since the .inter schema
# changed (added the intensity column). After this, baselines retrain on
# the new file so the comparison stays apples-to-apples.
python -c "from runner import invalidate_cache; invalidate_cache('ml-1m')"

# Then open the notebook — IA-SASRec variants are in the default registry,
# so they slot in alongside the baselines automatically.
jupyter lab notebooks/0_movielens_1m_benchmark.ipynb
```

Programmatic equivalent:

```python
from data import prepare_recbole_dataset
from hpo import run_optuna
from runner import train_and_eval

for ds in ["ml-1m", "steam"]:
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
- Data: 4-column `.inter` written correctly; Steam timestamp synthesis works
  on a tiny synthetic fixture (no Kaggle auth required for the test)

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
4. **Experiments.** ML-1M (rating intensity) and Steam (hours-played
   intensity) — covering both bounded-discrete and continuous-long-tail
   regimes. All 11 models × 2 datasets. Optuna TPE HPO with NDCG@10 as the
   primary metric. Eval: full-vocabulary scoring → HR / NDCG / MRR / Recall
   / Precision @ {10, 20, 50, 100}.
5. **Ablation: variants × normalisation.**
   `{Add, Mul, Val} × {log1p_minmax, minmax, zscore, none}` — a 3×4 table per
   dataset. Demonstrates that (a) normalisation is load-bearing
   (`none` → softmax collapse on Steam), and (b) the optimal variant depends
   on the dataset (hypothesis: `Mul` favours noisy long-tail signals like
   Steam hours; `Add` wins on bounded ratings; `Val` is a safe default).
6. **Discussion.** Per-dataset winner analysis; when each variant helps and
   why; cost analysis (extra parameters: 1 scalar `λ` per layer for `Add`,
   zero for `Mul`/`Val`).
7. **Conclusion.** IA-SASRec is a drop-in upgrade for any SASRec deployment
   that already logs interaction strength — no new parameters required for
   two of three variants, and a single learnable scalar for the third.

---

## 10. Notes & caveats

- **Reproducibility.** Optuna studies persist to SQLite under `results/hpo/`;
  individual model results persist to JSON under `results/eval/`. Both layers
  are resumable: kill and re-run the notebook freely.
- **Steam timestamps are synthetic.** Steam-200k has no real timestamp;
  we sort by hours-played within each user and assign monotonic synthetic
  timestamps so RecBole's sequential dataloader can order the sequences.
  Document this in the paper's "Datasets" section.
- **`λ` per layer.** Each of `n_layers` IA-SASRec-Add transformer blocks has
  its own learnable `λ`. If you want a single global `λ`, share the parameter
  across layers in `IATransformerEncoder.__init__`.
- **Generalisation.** The "intensity" framework is dataset-agnostic — anything
  you can express as a per-interaction `float` works (review helpfulness
  votes, listening counts, scroll depth, purchase value, etc.).
