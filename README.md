# Search Ranking & Discoverability

Which pages should a reviewer look at first, when there are far more pages
than anyone can review?

This is my work from the FlyRank AI machine learning internship (Jul–Sep 2026).
The repository started from the programme's template; everything in `work/` and
the write-up below are mine.

---

## The problem

A content team can review a few dozen pages a week. The site has tens of
thousands. So the real question is not "will this page perform well" — it's
**"what order should the queue be in"**.

That reframing changes everything downstream:

- It's a **ranking** problem, not a classification problem
- The right metric is **Precision@K**, not accuracy — a reviewer only ever
  sees the top of the list, so what happens at rank 4,000 is irrelevant
- A model that is right 90% of the time overall but wrong about the top 50
  is worse than useless

---

## Data

- **Starter slice** — ~30k pages, anonymised, 44 columns (shipped in `data/raw/`)
- **Full release** — ~79M rows, queried directly with **DuckDB** over a dataset
  hosted on Hugging Face, no local download

No client names, domains, URLs, titles or keywords appear anywhere in this
repository. See `DATA_USE.md`.

---

## Approach

**1. A transparent baseline first.**
Before any model, a hand-written rule that scores "fix this first". If a model
can't beat a rule you can read out loud, the model isn't earning its place.

**2. Features, with a leakage check as its own step.**
This is where my first attempt went wrong — see below.

**3. Models.**
Logistic regression, decision tree and random forest, trained on a
**client-holdout split** rather than a random one.

**4. Evaluation as a ranked queue.**
The output is not a score column, it's an ordered list of pages to review,
plus charts and a written report.

---

## What broke, and what I changed

**Information leaked between training and test.** My first split was random,
which meant pages from the same client landed on both sides. The model looked
strong and wasn't — it was partly recognising clients rather than learning what
makes a page worth reviewing. I rebuilt the split as **client-holdout**: a
client appears in training or in test, never both. The scores dropped, and
became true.

That was the lesson of the internship for me: **validation fails quietly.**
A broken model is obvious. A broken evaluation looks like success.

**Accuracy was the wrong instrument from the start.** Most pages don't need a
refresh, so a model that says "no" to everything scores well and is worthless.
Precision@K measures the only thing a reviewer experiences.

---

## Repository structure
notebooks/ exploration — first look at the data, first readable model
scripts/ the pipeline: prepare → baseline → train → evaluate → report
work/ my assignment notebooks and capstone
data/raw/ anonymised starter dataset
outputs/ generated reports, ranked queue, charts
docs/ data dictionary (all 44 columns) and framework notes


### The pipeline

01_prepare_features.py clean, build the feature vector, define the label
02_baseline_score.py transparent hand-rule "fix this first" score
03_train_model.py logistic regression / decision tree / random forest,
client-holdout split
04_evaluate_and_export.py ranked queue, charts, Markdown report
05_build_pdf_report.py shareable PDF summary

## Status

Capstone write-up in progress — final model comparison and the SHAP-based
explanation of what drives a page up the queue.

---

*Part of the FlyRank AI ML internship. Code MIT (`LICENSE`); data terms in
`DATA_USE.md`. Results here are observed and directional decision-support —
not a claim about how any search engine ranks pages.*
