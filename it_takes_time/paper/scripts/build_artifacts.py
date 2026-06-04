"""Generate every paper artifact (tables + figures) from results/.

Mirrors notebooks/1_significance.ipynb cells 3, 9, 11, 12, 14 plus a
learned-lambda table and a forest plot. No GPU, no checkpoint reloads —
reads results/eval/*.json and results/per_user/*.parquet only.

Outputs:
    paper/tables/main_results.tex   — seed-t + per-user bootstrap, Add/Mul/Val x NDCG/Recall/MRR @10
    paper/tables/lambda_table.tex   — mean +/- std learned lambda per (dataset, variant, layer)
    paper/tables/datasets.tex       — |U|, |I|, |interactions|, density, intensity source
    paper/figures/lambda_vs_delta.pdf
    paper/figures/forest_ndcg10.pdf

Run:  python paper/scripts/build_artifacts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import config as cfg  # noqa: E402
from evaluation import paired_significance  # noqa: E402

EVAL_DIR = Path(cfg.EVAL_DIR)
PER_USER_DIR = REPO / "results" / "per_user"
TABLES = REPO / "paper" / "tables"
FIGURES = REPO / "paper" / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

DATASETS = [
    "ml-100k-iar",
    "ml-1m",
    "amazon-digital-music",
    "amazon-office-products",
    "steam-3k",
    "steam-8k",
    "steam-15k",
]
DS_PRETTY = {
    "ml-100k-iar": "ML-100K",
    "ml-1m": "ML-1M",
    "amazon-digital-music": "Amazon-Music",
    "amazon-office-products": "Amazon-Office",
    "steam-3k": "Steam-3k",
    "steam-8k": "Steam-8k",
    "steam-15k": "Steam-15k",
}
BASELINE = "SASRec"
VARIANTS = ["IA-SASRec-Add", "IA-SASRec-Mul", "IA-SASRec-Val"]
VAR_SHORT = {"IA-SASRec-Add": "Add", "IA-SASRec-Mul": "Mul", "IA-SASRec-Val": "Val"}
METRICS = ["ndcg@10", "recall@10", "mrr@10"]
SUPP_K = [10, 20, 50, 100]
SUPP_METRICS = [f"{m}@{k}" for m in ("ndcg", "recall", "mrr") for k in SUPP_K]
N_BOOT = 2000


# ---------------------------------------------------------------------------
# 1. seed-level paired t-test (cell 3)
# ---------------------------------------------------------------------------
def seed_table(metrics: list[str] = METRICS) -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        df = paired_significance(ds, BASELINE, VARIANTS, metrics=metrics)
        if df.empty:
            continue
        df.insert(0, "dataset", ds)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# 2. per-user paired bootstrap (cell 9)
# ---------------------------------------------------------------------------
def _bootstrap_ci(diffs: np.ndarray, n_boot: int, rng: np.random.Generator):
    n = len(diffs)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)
    lo, hi = np.quantile(boot_means, [0.025, 0.975])
    n_le = int((boot_means <= 0).sum())
    n_ge = int((boot_means >= 0).sum())
    p = 2.0 * min(n_le, n_ge) / n_boot
    return float(lo), float(hi), max(p, 1.0 / n_boot)


def _cohens_d(diffs: np.ndarray) -> float:
    if len(diffs) < 2:
        return float("nan")
    s = diffs.std(ddof=1)
    return float(diffs.mean() / s) if s > 0 else float("nan")


def paired_user_bootstrap(dataset: str, variant: str, metric: str, n_boot: int = N_BOOT) -> dict | None:
    rng = np.random.default_rng(0)
    base_files = sorted(PER_USER_DIR.glob(f"{dataset}__{BASELINE}__seed*.parquet"))
    base_seeds = {int(p.stem.rsplit("seed", 1)[1]): p for p in base_files}
    var_files = sorted(PER_USER_DIR.glob(f"{dataset}__{variant}__seed*.parquet"))
    var_seeds = {int(p.stem.rsplit("seed", 1)[1]): p for p in var_files}
    shared = sorted(set(base_seeds) & set(var_seeds))
    if not shared:
        return None

    pooled_diffs = []
    pooled_base = []
    per_seed_signs = []
    for s in shared:
        b = pd.read_parquet(base_seeds[s], columns=["user_id", metric]).rename(columns={metric: "b"})
        c = pd.read_parquet(var_seeds[s], columns=["user_id", metric]).rename(columns={metric: "c"})
        m = b.merge(c, on="user_id", how="inner")
        d = (m["c"] - m["b"]).to_numpy()
        pooled_diffs.append(d)
        pooled_base.append(m["b"].mean())
        lo, hi, _ = _bootstrap_ci(d, n_boot, rng)
        per_seed_signs.append((lo > 0, hi < 0))

    diffs_pool = np.concatenate(pooled_diffs)
    base_mean = float(np.mean(pooled_base))
    lo_p, hi_p, p_pool = _bootstrap_ci(diffs_pool, n_boot, rng)
    return {
        "dataset": dataset,
        "model": variant,
        "metric": metric,
        "n_seeds": len(shared),
        "n_users_total": len(diffs_pool),
        "baseline_mean": base_mean,
        "mean_diff": float(diffs_pool.mean()),
        "rel_diff_user_%": (float(diffs_pool.mean()) / base_mean * 100.0) if base_mean else 0.0,
        "ci_lo_pool": lo_p,
        "ci_hi_pool": hi_p,
        "p_boot_pool": p_pool,
        "cohens_d_pool": _cohens_d(diffs_pool),
        "seeds_sig_pos": sum(1 for p, _ in per_seed_signs if p),
        "seeds_sig_neg": sum(1 for _, n in per_seed_signs if n),
    }


def boot_table(metrics: list[str] = METRICS) -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        for v in VARIANTS:
            for m in metrics:
                r = paired_user_bootstrap(ds, v, m)
                if r:
                    rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. combined view (cell 11) + LaTeX export
# ---------------------------------------------------------------------------
def _stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:
        return r"$^{**}$"
    if p < 0.05:
        return r"$^{*}$"
    return ""


def _fmt_cell(row: pd.Series) -> str:
    """Format one cell: rel% + paired-t stars; bold if BOTH t<.05 AND boot<.05 AND seed majority."""
    if pd.isna(row.get("rel_diff_seed_%")):
        return "--"
    pct = row["rel_diff_seed_%"]
    t_p = row.get("t_pvalue", float("nan"))
    boot_p = row.get("p_boot_pool", float("nan"))
    n_seeds = row.get("n_seeds")
    pos = row.get("seeds_sig_pos", 0) or 0
    neg = row.get("seeds_sig_neg", 0) or 0
    robust = (
        pd.notna(t_p) and pd.notna(boot_p)
        and t_p < 0.05 and boot_p < 0.05
        and pd.notna(n_seeds)
        and (pos > n_seeds / 2 or neg > n_seeds / 2)
    )
    s = f"{pct:+.1f}\\%"
    s += _stars(t_p)
    if robust:
        # negative robust -> red; positive robust -> green-ish via \best macro
        if pct > 0:
            s = r"\best{" + s + "}"
        else:
            s = r"\textbf{" + s + "}"  # red unsupported in monochrome; bold suffices
    return s


def write_main_results(combined: pd.DataFrame) -> None:
    # Wide layout: rows = dataset, columns = variant x metric.
    combined = combined.copy()
    combined["cell"] = combined.apply(_fmt_cell, axis=1)
    pivot = combined.pivot_table(
        index="dataset", columns=["model", "metric"], values="cell", aggfunc="first"
    )
    # enforce column order
    cols = [(v, m) for v in VARIANTS for m in METRICS]
    pivot = pivot.reindex(columns=cols)
    pivot.index = [DS_PRETTY.get(d, d) for d in pivot.index]
    pivot = pivot.reindex(index=[DS_PRETTY[d] for d in DATASETS])

    # build LaTeX manually for the multi-header
    lines = []
    lines.append(r"% AUTO-GENERATED by paper/scripts/build_artifacts.py — do not edit by hand.")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\scriptsize")
    lines.append(r"\caption{Relative change (\%) of IA-SASRec variants vs SASRec across 7 datasets and 5 seeds. Stars: paired $t$-test, $^{*}p<0.05$, $^{**}p<0.01$. Bold cells are robust: both the paired-$t$ at the seed level and the per-user paired bootstrap (B=2000) reject at $\alpha=0.05$, and the per-seed CI sign-majority stability check holds.}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\begin{tabular}{l" + "ccc" * len(VARIANTS) + "}")
    lines.append(r"\toprule")
    lines.append(
        " & "
        + " & ".join(
            r"\multicolumn{3}{c}{IA-SASRec-" + VAR_SHORT[v] + "}" for v in VARIANTS
        )
        + r" \\"
    )
    # cmidrule per group
    cmid = []
    start = 2
    for _ in VARIANTS:
        cmid.append(rf"\cmidrule(lr){{{start}-{start+2}}}")
        start += 3
    lines.append(" ".join(cmid))
    metric_short = {"ndcg@10": "NDCG@10", "recall@10": "Recall@10", "mrr@10": "MRR@10"}
    lines.append(
        "Dataset & "
        + " & ".join(metric_short[m] for v in VARIANTS for m in METRICS)
        + r" \\"
    )
    lines.append(r"\midrule")
    for ds_pretty in pivot.index:
        row_vals = [pivot.at[ds_pretty, (v, m)] for v in VARIANTS for m in METRICS]
        row_vals = ["--" if pd.isna(x) else x for x in row_vals]
        lines.append(ds_pretty + " & " + " & ".join(row_vals) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    out = TABLES / "main_results.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# 4. lambda table from intensity_params
# ---------------------------------------------------------------------------
def write_lambda_table() -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        for v in VARIANTS:
            files = sorted(EVAL_DIR.glob(f"{ds}__{v}__seed*.json"))
            if not files:
                continue
            layers: dict[int, list[float]] = {}
            ndcgs = []
            for f in files:
                d = json.loads(f.read_text())
                ip = d.get("intensity_params") or {}
                for k, val in ip.items():
                    if "lambda" in k:
                        # k like 'layer_0_lambda'
                        try:
                            li = int(k.split("_")[1])
                        except (IndexError, ValueError):
                            continue
                        layers.setdefault(li, []).append(float(val))
                ndcgs.append(d["test_result"]["ndcg@10"])
            if not layers:
                continue
            for li, vals in sorted(layers.items()):
                rows.append({
                    "dataset": DS_PRETTY[ds],
                    "variant": VAR_SHORT[v],
                    "layer": li,
                    "lambda_mean": float(np.mean(vals)),
                    "lambda_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "ndcg_mean": float(np.mean(ndcgs)),
                    "n_seeds": len(vals),
                })
    lam = pd.DataFrame(rows)

    # collapse layers: per (dataset, variant) show mean lambda across layers (mean over seeds first).
    g = (
        lam.groupby(["dataset", "variant"], as_index=False)
        .agg(lambda_mean=("lambda_mean", "mean"), lambda_std=("lambda_std", "mean"), n_layers=("layer", "nunique"))
    )

    # Wide pivot: row=dataset, col=variant
    pivot = g.pivot(index="dataset", columns="variant", values="lambda_mean")
    std_pivot = g.pivot(index="dataset", columns="variant", values="lambda_std")
    pivot = pivot.reindex(index=[DS_PRETTY[d] for d in DATASETS], columns=["Add", "Mul", "Val"])
    std_pivot = std_pivot.reindex(index=[DS_PRETTY[d] for d in DATASETS], columns=["Add", "Mul", "Val"])

    lines = [
        r"% AUTO-GENERATED by paper/scripts/build_artifacts.py — do not edit by hand.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Learned $\lambda$ per (dataset, variant), averaged over attention layers and 5 seeds (mean$\pm$std). Values are the per-layer learnable scalar gating the intensity signal; $\lambda{=}0$ collapses to SASRec.}",
        r"\label{tab:lambda}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Dataset & Add & Mul & Val \\",
        r"\midrule",
    ]
    for ds in pivot.index:
        cells = []
        for v in ["Add", "Mul", "Val"]:
            m = pivot.at[ds, v]
            s = std_pivot.at[ds, v]
            if pd.isna(m):
                cells.append("--")
            else:
                cells.append(f"{m:.2f}$\\pm${s:.2f}")
        lines.append(ds + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    out = TABLES / "lambda_table.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    return lam


# ---------------------------------------------------------------------------
# 5. lambda-vs-delta figure
# ---------------------------------------------------------------------------
def write_lambda_vs_delta() -> None:
    rows = []
    for ds in DATASETS:
        base_files = sorted(EVAL_DIR.glob(f"{ds}__SASRec__seed*.json"))
        if not base_files:
            continue
        base_ndcg = [json.loads(f.read_text())["test_result"]["ndcg@10"] for f in base_files]
        base_mean = float(np.mean(base_ndcg))
        for v in VARIANTS:
            files = sorted(EVAL_DIR.glob(f"{ds}__{v}__seed*.json"))
            if not files:
                continue
            ndcgs, lambdas = [], []
            for f in files:
                d = json.loads(f.read_text())
                ndcgs.append(d["test_result"]["ndcg@10"])
                ip = d.get("intensity_params") or {}
                ll = [val for k, val in ip.items() if "lambda" in k]
                if ll:
                    lambdas.append(float(np.mean(ll)))
            if not ndcgs or not lambdas:
                continue
            rows.append({
                "dataset": ds, "variant": v,
                "lambda_mean": float(np.mean(lambdas)),
                "rel_delta": (float(np.mean(ndcgs)) - base_mean) / base_mean * 100.0,
            })
    lam_df = pd.DataFrame(rows)

    markers = {"IA-SASRec-Add": "o", "IA-SASRec-Mul": "s", "IA-SASRec-Val": "^"}
    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:cyan"]
    colors = {ds: c for ds, c in zip(DATASETS, palette)}

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for _, r in lam_df.iterrows():
        ax.scatter(r["lambda_mean"], r["rel_delta"],
                   marker=markers[r["variant"]], color=colors[r["dataset"]],
                   s=80, edgecolor="black", linewidth=0.5, zorder=3)
    ax.axhline(0, color="grey", lw=0.7, ls="--")
    ax.axvline(0, color="grey", lw=0.7, ls="--")
    ax.set_xlabel(r"mean learned $\lambda$ (across layers, seeds)")
    ax.set_ylabel(r"NDCG@10 $\Delta$ vs SASRec (%)")
    ax.set_title(r"$\lambda$-vs-$\Delta$ calibration")

    ds_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[d], markeredgecolor="black", label=DS_PRETTY[d], markersize=8) for d in DATASETS]
    var_handles = [Line2D([0], [0], marker=markers[v], color="w", markerfacecolor="grey", markeredgecolor="black", label=VAR_SHORT[v], markersize=8) for v in VARIANTS]
    # Both legends below the axes (single row), so they never overlap data
    # and the full dataset names fit without clipping.
    fig.subplots_adjust(bottom=0.28)
    fig.legend(handles=ds_handles, loc="lower center",
               bbox_to_anchor=(0.32, 0.0), title="dataset",
               fontsize=8, title_fontsize=8, ncol=4, frameon=True)
    fig.legend(handles=var_handles, loc="lower center",
               bbox_to_anchor=(0.82, 0.0), title="variant",
               fontsize=8, title_fontsize=8, ncol=3, frameon=True)

    out = FIGURES / "lambda_vs_delta.pdf"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# 6. forest plot from per-user parquets
# ---------------------------------------------------------------------------
def write_forest_plot() -> None:
    """One row per (dataset, variant); centre = mean per-user NDCG@10 diff (rel %), whiskers = 95% bootstrap CI."""
    rng = np.random.default_rng(0)
    rows = []
    for ds in DATASETS:
        for v in VARIANTS:
            base_files = sorted(PER_USER_DIR.glob(f"{ds}__{BASELINE}__seed*.parquet"))
            base_seeds = {int(p.stem.rsplit("seed", 1)[1]): p for p in base_files}
            var_files = sorted(PER_USER_DIR.glob(f"{ds}__{v}__seed*.parquet"))
            var_seeds = {int(p.stem.rsplit("seed", 1)[1]): p for p in var_files}
            shared = sorted(set(base_seeds) & set(var_seeds))
            if not shared:
                continue
            pooled = []
            base_pool = []
            for s in shared:
                b = pd.read_parquet(base_seeds[s], columns=["user_id", "ndcg@10"]).rename(columns={"ndcg@10": "b"})
                c = pd.read_parquet(var_seeds[s], columns=["user_id", "ndcg@10"]).rename(columns={"ndcg@10": "c"})
                m = b.merge(c, on="user_id", how="inner")
                pooled.append((m["c"] - m["b"]).to_numpy())
                base_pool.append(m["b"].mean())
            d = np.concatenate(pooled)
            base_mean = float(np.mean(base_pool))
            lo, hi, _ = _bootstrap_ci(d, N_BOOT, rng)
            rows.append({
                "dataset": DS_PRETTY[ds], "variant": VAR_SHORT[v],
                "mean_pct": d.mean() / base_mean * 100.0,
                "lo_pct": lo / base_mean * 100.0,
                "hi_pct": hi / base_mean * 100.0,
            })
    fdf = pd.DataFrame(rows)
    # Page-wide layout: one panel per variant, datasets as rows sharing the x-axis.
    var_order = [VAR_SHORT[v] for v in VARIANTS]
    ds_order = [DS_PRETTY[ds] for ds in DATASETS]  # reading order, top-to-bottom
    y = np.arange(len(ds_order))[::-1]

    fig, axes = plt.subplots(1, len(var_order), sharex=True, sharey=True,
                             figsize=(13, 4))
    for ax, vshort in zip(np.atleast_1d(axes), var_order):
        sub = fdf[fdf["variant"] == vshort].set_index("dataset")
        for ds_pretty, yi in zip(ds_order, y):
            if ds_pretty not in sub.index:
                continue
            r = sub.loc[ds_pretty]
            color = "tab:red" if r["hi_pct"] < 0 else ("tab:green" if r["lo_pct"] > 0 else "grey")
            ax.errorbar(r["mean_pct"], yi,
                        xerr=[[r["mean_pct"] - r["lo_pct"]], [r["hi_pct"] - r["mean_pct"]]],
                        fmt="o", color=color, ecolor=color, capsize=3, markersize=5)
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.set_title(vshort)
        ax.set_yticks(y)
        ax.set_yticklabels(ds_order)
        ax.tick_params(axis="y", labelsize=8)
    fig.supxlabel(r"NDCG@10 $\Delta$ vs SASRec (%) — per-user mean (95% bootstrap CI)")
    plt.tight_layout()
    out = FIGURES / "forest_ndcg10.pdf"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# 7. dataset stats table
# ---------------------------------------------------------------------------
INTENSITY_SOURCE = {
    "ml-100k-iar": "rating 1--5",
    "ml-1m": "rating 1--5",
    "amazon-digital-music": "rating 1--5",
    "amazon-office-products": "rating 1--5",
    "steam-3k": "hours played",
    "steam-8k": "hours played",
    "steam-15k": "hours played",
}


def write_dataset_table() -> None:
    rows = []
    for ds in DATASETS:
        inter = REPO / "recbole_data" / ds / f"{ds}.inter"
        if not inter.exists():
            inter = REPO / "data" / ds / f"{ds}.inter"
        if not inter.exists():
            rows.append({"dataset": DS_PRETTY[ds], "U": "--", "I": "--", "N": "--", "density": "--",
                         "src": INTENSITY_SOURCE[ds]})
            continue
        df = pd.read_csv(inter, sep="\t")
        # RecBole header: 'user_id:token', 'item_id:token', 'timestamp:float', 'intensity:float'
        ucol = [c for c in df.columns if c.startswith("user_id")][0]
        icol = [c for c in df.columns if c.startswith("item_id")][0]
        u = df[ucol].nunique()
        i = df[icol].nunique()
        n = len(df)
        density = n / (u * i) * 100.0
        rows.append({"dataset": DS_PRETTY[ds], "U": f"{u:,}", "I": f"{i:,}",
                     "N": f"{n:,}", "density": f"{density:.3f}\\%",
                     "src": INTENSITY_SOURCE[ds]})
    lines = [
        r"% AUTO-GENERATED by paper/scripts/build_artifacts.py — do not edit by hand.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Datasets after 5-core filtering. Density is $|\mathcal{I}| / (|\mathcal{U}|\cdot|\mathcal{V}|)$. Intensity is the per-interaction signal threaded into IA-SASRec attention.}",
        r"\label{tab:datasets}",
        r"\begin{tabular}{lrrrrl}",
        r"\toprule",
        r"Dataset & $|\mathcal{U}|$ & $|\mathcal{V}|$ & $|\mathcal{I}|$ & density & intensity \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(f"{r['dataset']} & {r['U']} & {r['I']} & {r['N']} & {r['density']} & {r['src']} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    out = TABLES / "datasets.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Supplementary: full dual-criterion table at k in {10,20,50,100}
# ---------------------------------------------------------------------------
def write_supplementary_csv() -> None:
    print("== supplementary: dual-criterion at k in {10,20,50,100} ==")
    seed_df = seed_table(metrics=SUPP_METRICS)
    boot_df = boot_table(metrics=SUPP_METRICS)
    left = seed_df.rename(columns={"n": "n_seeds", "rel_diff_%": "rel_diff_seed_%"})[
        ["dataset", "model", "metric", "n_seeds", "rel_diff_seed_%", "t_pvalue", "wilcoxon_pvalue"]
    ]
    right = boot_df[
        ["dataset", "model", "metric", "n_users_total", "rel_diff_user_%",
         "ci_lo_pool", "ci_hi_pool", "p_boot_pool", "cohens_d_pool",
         "seeds_sig_pos", "seeds_sig_neg"]
    ]
    combined = left.merge(right, on=["dataset", "model", "metric"], how="outer")
    out = TABLES / "supplementary_significance.csv"
    combined.to_csv(out, index=False)
    print(f"wrote {out}  ({len(combined)} rows)")

    # Robustness summary: count cells that robustly reject *positively* at each k.
    # "Robust positive" = paired-t p<0.05 AND bootstrap CI strictly above 0 AND >=3/5 seeds sign-positive.
    robust_pos = (
        (combined["t_pvalue"] < 0.05)
        & (combined["ci_lo_pool"] > 0)
        & (combined["seeds_sig_pos"] >= 3)
    )
    combined["_k"] = combined["metric"].str.split("@").str[1].astype(int)
    by_k = combined.assign(robust_pos=robust_pos).groupby("_k")
    print("  robust-positive cells by k:")
    for k, grp in by_k:
        print(f"    k={k:>3}: {int(grp['robust_pos'].sum())}/{len(grp)}")
    deeper = combined[combined["_k"].isin([20, 50, 100])]
    n_deeper_pos = int((
        (deeper["t_pvalue"] < 0.05)
        & (deeper["ci_lo_pool"] > 0)
        & (deeper["seeds_sig_pos"] >= 3)
    ).sum())
    print(f"  supp: {n_deeper_pos}/{len(deeper)} robust-positive cells at k in {{20,50,100}}")


# ---------------------------------------------------------------------------
def main() -> None:
    print("== seed-level paired-t ==")
    seed_df = seed_table()
    print(f"  rows: {len(seed_df)}")

    print("== per-user bootstrap ==")
    boot_df = boot_table()
    print(f"  rows: {len(boot_df)}")

    left = seed_df.rename(columns={"n": "n_seeds", "rel_diff_%": "rel_diff_seed_%"})[
        ["dataset", "model", "metric", "n_seeds", "rel_diff_seed_%", "t_pvalue", "wilcoxon_pvalue"]
    ]
    right = boot_df[
        ["dataset", "model", "metric", "n_users_total", "rel_diff_user_%",
         "ci_lo_pool", "ci_hi_pool", "p_boot_pool", "cohens_d_pool",
         "seeds_sig_pos", "seeds_sig_neg"]
    ]
    combined = left.merge(right, on=["dataset", "model", "metric"], how="outer")
    write_main_results(combined)

    write_lambda_table()
    write_lambda_vs_delta()
    write_forest_plot()
    write_dataset_table()
    write_supplementary_csv()


if __name__ == "__main__":
    main()
