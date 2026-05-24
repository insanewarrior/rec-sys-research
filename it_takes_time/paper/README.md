# Paper: IA-SASRec — A Negative Result

Skeleton for the IA-SASRec paper. Section-by-section descriptions live
inside the `% DESCRIPTION` blocks of each `sections/*.tex` file — read
those first when you start drafting prose.

## Build

```bash
make pdf     # latexmk -> main.pdf
make docx    # pandoc  -> paper.docx
make clean
```

`make docx` produces a Word file via pandoc. The LaTeX skeleton is
intentionally written against generic `article.cls` (plus `amsmath`,
`booktabs`, `natbib`, `graphicx`, `hyperref`) so pandoc conversion is
one shot. Math, sections, citations, tables, and figures survive the
conversion; complex TikZ does not — render any TikZ figures to
`.pdf` / `.png` and `\includegraphics` them.

## Retargeting a venue

The skeleton is venue-agnostic. To target a specific venue, edit only
`main.tex`:

- **RecSys / SIGIR / KDD (ACM):** swap `\documentclass{article}` for
  `\documentclass[sigconf]{acmart}` and adjust the citation style.
- **IEEE conf:** swap to `\documentclass{IEEEtran}`.
- **arXiv preprint:** keep `article.cls`; no changes needed.

If you target a class with heavy custom macros (e.g. `acmart`), the
`make docx` pandoc path may need tweaks — keep a `main-generic.tex`
shim that includes the same `sections/*.tex` files for the docx
pipeline.

## Where the content comes from (lift-ready sources in this repo)

- **Method math + motivation:** `../ia_sasrec.md` §1–4.
- **IA-SASRec variant table:** `../README.md` ("IA-SASRec — what's
  new" section).
- **Results table + lambda figure:** `../notebooks/1_significance.ipynb`
  cells 11 (combined seed+bootstrap table), 12 (robust filter), 14
  (lambda-vs-Delta scatter). Export targets:
  - `tables/main_results.tex`
  - `figures/lambda_vs_delta.pdf` (source: `../results/lambda_vs_delta.png`)
- **Per-user forest plot:** built fresh from
  `../results/per_user/*.parquet`.
- **Table highlighting macros (`\best`, `\second`):**
  `../docs/protocol_neural_networks.tex` lines 33–34 — already copied
  into `main.tex`.

## Pre-submission TODO list (from the section description blocks)

1. **Run the λ ∈ {0, 1, learned} ablation** on at least Steam-15k and
   Amazon-Office-Products. Flagged in `06_analysis.tex` §6.3 as
   required before submission — directly tests the escape-hatch claim.
2. **Complete ML-1M seed2024** so all datasets have n=5 seeds. Flagged
   in `04_experimental_setup.tex` §4.2.
3. **Populate `refs.bib`** with full BibTeX records for the nine
   placeholder keys listed in the file header.
4. **Generate `tables/main_results.tex`** by adding an export cell to
   `1_significance.ipynb` that emits the combined seed+bootstrap view
   in LaTeX (`.to_latex(...)` with the `\best`/`\second` decorators).
5. **Decide target venue** and swap `\documentclass` accordingly.
